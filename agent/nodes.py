"""Graph node functions — the actual steps of the agentic loop.

Each node returns only the partial state it changes (LangGraph merges
these into the running state). Nodes that can fail in normal operation
(search, fetch, LLM calls) catch their own errors and record them in
`log`/`extracted[].error` instead of raising, so one bad link or a flaky
search call doesn't kill the whole run — the loop degrades gracefully and
keeps going with whatever it has.
"""

import re
from typing import List
from urllib.parse import urlparse

from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

from agent.config import (
    HIGH_COST_DOMAINS,
    MAX_LINKS_PER_ROUND,
    MAX_SEARCH_ROUNDS,
    MIN_SUCCESSFUL_EXTRACTIONS,
    get_extraction_llm,
    get_llm,
)
from agent.prompts import (
    CRITIC_SYSTEM_PROMPT,
    QUERY_REFINEMENT_SYSTEM_PROMPT,
    REPORT_WRITER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from agent.state import AgentState, ExtractedInfo
from tools.extract_tool import ExtractionError, FetchError, extract_fields, fetch_page
from tools.search_tool import SearchToolError, web_search


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "report"


# ---------------------------------------------------------------------------
# 1. Search
# ---------------------------------------------------------------------------

def search_node(state: AgentState) -> dict:
    query = state.get("search_query") or state["query"]
    round_num = state.get("round", 0)
    refinement_failure = None

    if round_num > 0:
        # Only spend an LLM call refining the query on a retry round —
        # the first pass uses the user's query directly (efficiency).
        try:
            llm = get_llm()
            tried = state.get("search_query", state["query"])
            refined = llm.invoke(
                [
                    {"role": "system", "content": QUERY_REFINEMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Original goal: {state['query']}\nPreviously tried query: {tried}",
                    },
                ]
            )
            query = refined.content.strip().strip('"')
        except Exception as exc:  # noqa: BLE001 - LLM call can fail; fall back to the original query
            query = state["query"]
            refinement_failure = str(exc)

    try:
        results = web_search(query)
        log = [f"[search] round {round_num}: '{query}' -> {len(results)} result(s)."]
        if refinement_failure:
            log.insert(0, f"[search] round {round_num}: query refinement failed ({refinement_failure}); reused original query.")
        return {"search_query": query, "search_results": results, "log": log}
    except SearchToolError as exc:
        return {
            "search_query": query,
            "search_results": [],
            "log": [f"[search] round {round_num}: FAILED ({exc}). Continuing with no new results."],
        }


# ---------------------------------------------------------------------------
# 2. Select links (heuristic, no LLM call needed -> saves tokens)
# ---------------------------------------------------------------------------

def select_links_node(state: AgentState) -> dict:
    already_visited = {e["url"] for e in state.get("extracted", [])}
    seen_domains = set()
    selected: List[str] = []

    for r in state.get("search_results", []):
        url = r["url"]
        if url in already_visited:
            continue
        domain = _domain(url)
        if domain in seen_domains:
            continue  # prefer domain diversity over multiple pages from one site
        seen_domains.add(domain)
        selected.append(url)
        if len(selected) >= MAX_LINKS_PER_ROUND:
            break

    log = [f"[select_links] chose {len(selected)} link(s) across distinct domains: {selected}"]
    return {"selected_links": selected, "log": log}


# ---------------------------------------------------------------------------
# 3. Human-in-the-loop: approve visiting high-cost / paywalled domains
# ---------------------------------------------------------------------------

def human_review_node(state: AgentState) -> dict:
    links = state.get("selected_links", [])
    high_cost = [u for u in links if _domain(u) in HIGH_COST_DOMAINS]

    if not high_cost:
        return {"log": ["[human_review] no high-cost domains in this batch; auto-approved."]}

    decision = interrupt(
        {
            "type": "high_cost_site_approval",
            "message": "The agent wants to browse the following high-cost/paywalled site(s). Approve?",
            "links": high_cost,
        }
    )
    approved = set(decision.get("approved_urls", [])) if isinstance(decision, dict) else set()
    kept = [u for u in links if u not in high_cost or u in approved]
    dropped = [u for u in high_cost if u not in approved]

    log = [f"[human_review] approved={sorted(approved)} dropped={dropped}"]
    return {"selected_links": kept, "log": log}


def dispatch_extraction(state: AgentState):
    """Conditional-edge fan-out: one parallel Send per link to visit."""
    links = state.get("selected_links", [])
    if not links:
        return "evaluate_sufficiency"
    return [Send("extract_link", {"url": url}) for url in links]


# ---------------------------------------------------------------------------
# 4. Extract (runs once per link, in parallel via Send)
# ---------------------------------------------------------------------------

def extract_link_node(state: dict) -> dict:
    url = state["url"]
    try:
        page_text = fetch_page(url)
    except FetchError as exc:
        return {
            "extracted": [
                ExtractedInfo(
                    url=url, product_name=None, key_ingredients=[], pricing=None,
                    usp=None, summary="", success=False, error=str(exc),
                )
            ],
            "log": [f"[extract] {url}: fetch failed - {exc}"],
        }

    try:
        llm = get_extraction_llm()
        fields = extract_fields(llm, url, page_text)
    except ExtractionError as exc:
        return {
            "extracted": [
                ExtractedInfo(
                    url=url, product_name=None, key_ingredients=[], pricing=None,
                    usp=None, summary="", success=False, error=str(exc),
                )
            ],
            "log": [f"[extract] {url}: LLM extraction failed - {exc}"],
        }

    info = ExtractedInfo(
        url=url,
        product_name=fields.product_name,
        key_ingredients=fields.key_ingredients,
        pricing=fields.pricing,
        usp=fields.usp,
        summary=fields.summary,
        success=True,
        error=None,
    )
    log = [f"[extract] {url}: found product='{fields.product_name}', ingredients={fields.key_ingredients}"]
    return {"extracted": [info], "log": log}


# ---------------------------------------------------------------------------
# 5. Decide whether to loop back for another search round or move on
# ---------------------------------------------------------------------------

def evaluate_sufficiency_node(state: AgentState) -> dict:
    successful = [e for e in state.get("extracted", []) if e["success"]]
    round_num = state.get("round", 0)

    if len(successful) >= MIN_SUCCESSFUL_EXTRACTIONS or round_num >= MAX_SEARCH_ROUNDS - 1:
        next_step = "synthesize"
        msg = (
            f"[evaluate] {len(successful)} successful extraction(s) after round {round_num}. "
            "Proceeding to synthesis."
        )
    else:
        next_step = "retry_search"
        msg = (
            f"[evaluate] only {len(successful)} successful extraction(s) after round {round_num} "
            f"(need {MIN_SUCCESSFUL_EXTRACTIONS}). Retrying search with a refined query."
        )

    return {"round": round_num + 1, "next_step": next_step, "log": [msg]}


def route_after_evaluation(state: AgentState) -> str:
    return "search" if state["next_step"] == "retry_search" else "synthesize"


# ---------------------------------------------------------------------------
# 6. Synthesize themes & gaps
# ---------------------------------------------------------------------------

class SynthesisSchema(BaseModel):
    themes: List[str] = Field(description="Recurring patterns, each citing supporting sources.")
    gaps: List[str] = Field(description="Inferred market white-space / under-served angles.")


def synthesize_node(state: AgentState) -> dict:
    successful = [e for e in state.get("extracted", []) if e["success"]]
    if not successful:
        return {
            "themes": [],
            "gaps": [],
            "log": ["[synthesize] no successful extractions to synthesize from; skipping."],
        }

    data_blob = "\n\n".join(
        f"Source: {e['url']}\nProduct: {e['product_name']}\nIngredients: {e['key_ingredients']}\n"
        f"Pricing: {e['pricing']}\nUSP: {e['usp']}\nSummary: {e['summary']}"
        for e in successful
    )
    try:
        llm = get_llm().with_structured_output(SynthesisSchema)
        result = llm.invoke(
            [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Research goal: {state['query']}\n\n{data_blob}"},
            ]
        )
    except Exception as exc:  # noqa: BLE001 - the model can fail to produce a valid tool call
        return {
            "themes": [],
            "gaps": [],
            "log": [f"[synthesize] LLM call failed ({exc}); proceeding with no themes/gaps."],
        }

    log = [f"[synthesize] identified {len(result.themes)} theme(s) and {len(result.gaps)} gap(s)."]
    return {"themes": result.themes, "gaps": result.gaps, "log": log}


# ---------------------------------------------------------------------------
# 7. Critic / fact-check pass (bonus: second agent verifying the first)
# ---------------------------------------------------------------------------

class CritiqueSchema(BaseModel):
    flagged_claims: List[str] = Field(default_factory=list, description="Unsupported themes/gaps, with reasons.")


def critic_node(state: AgentState) -> dict:
    if not state.get("themes") and not state.get("gaps"):
        return {"critique_notes": [], "log": ["[critic] nothing to review; skipped."]}

    successful = [e for e in state.get("extracted", []) if e["success"]]
    data_blob = "\n".join(f"- {e['url']}: {e['summary']}" for e in successful)
    claims_blob = "Themes:\n" + "\n".join(state.get("themes", [])) + "\n\nGaps:\n" + "\n".join(state.get("gaps", []))

    try:
        llm = get_llm().with_structured_output(CritiqueSchema)
        result = llm.invoke(
            [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": f"{claims_blob}\n\nSOURCE DATA:\n{data_blob}"},
            ]
        )
    except Exception as exc:  # noqa: BLE001 - the model can fail to produce a valid tool call
        return {
            "critique_notes": [],
            "log": [f"[critic] LLM call failed ({exc}); skipping fact-check for this run."],
        }

    log = [f"[critic] flagged {len(result.flagged_claims)} unsupported claim(s)."]
    return {"critique_notes": result.flagged_claims, "log": log}


# ---------------------------------------------------------------------------
# 8. Write the final Markdown report
# ---------------------------------------------------------------------------

def write_report_node(state: AgentState) -> dict:
    successful = [e for e in state.get("extracted", []) if e["success"]]
    failed = [e for e in state.get("extracted", []) if not e["success"]]

    data_blob = "\n\n".join(
        f"Source: {e['url']}\nProduct: {e['product_name']}\nIngredients: {e['key_ingredients']}\n"
        f"Pricing: {e['pricing']}\nUSP: {e['usp']}\nSummary: {e['summary']}"
        for e in successful
    )
    failed_blob = "\n".join(f"- {e['url']}: {e['error']}" for e in failed) or "None"
    critique_blob = "\n".join(state.get("critique_notes", [])) or "None"

    try:
        llm = get_llm()
        result = llm.invoke(
            [
                {"role": "system", "content": REPORT_WRITER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {state['query']}\n\nEXTRACTED DATA:\n{data_blob}\n\n"
                        f"THEMES:\n{chr(10).join(state.get('themes', [])) or 'None'}\n\n"
                        f"GAPS:\n{chr(10).join(state.get('gaps', [])) or 'None'}\n\n"
                        f"CRITIC-FLAGGED CLAIMS (soften or omit these):\n{critique_blob}\n\n"
                        f"FAILED SOURCES:\n{failed_blob}"
                    ),
                },
            ]
        )
        return {"report_markdown": result.content, "log": ["[write_report] draft generated."]}
    except Exception as exc:  # noqa: BLE001 - fall back to a templated report rather than losing the run's data
        report = _fallback_report(state, successful, failed)
        return {
            "report_markdown": report,
            "log": [f"[write_report] LLM call failed ({exc}); used a templated fallback report instead."],
        }


def _fallback_report(state: AgentState, successful: list, failed: list) -> str:
    lines = [f"# {state['query']}", "", "## Products Surveyed", ""]
    lines.append("| Product | Key Ingredients | Pricing | USP | Source URL |")
    lines.append("| --- | --- | --- | --- | --- |")
    for e in successful:
        lines.append(
            f"| {e['product_name'] or 'N/A'} | {', '.join(e['key_ingredients']) or 'N/A'} | "
            f"{e['pricing'] or 'N/A'} | {e['usp'] or 'N/A'} | {e['url']} |"
        )
    lines += ["", "## Themes", ""]
    lines += [f"- {t}" for t in state.get("themes", [])] or ["- None identified."]
    lines += ["", "## Gaps", ""]
    lines += [f"- {g}" for g in state.get("gaps", [])] or ["- None identified."]
    lines += ["", "## Sources", ""]
    lines += [f"1. {e['url']}" for e in successful]
    lines += [f"1. {e['url']} (failed: {e['error']})" for e in failed]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Human-in-the-loop: approve finalizing the report
# ---------------------------------------------------------------------------

def human_approve_report_node(state: AgentState) -> dict:
    decision = interrupt(
        {
            "type": "finalize_report_approval",
            "message": "Draft report ready. Approve saving it to disk?",
            "preview": state["report_markdown"][:1000],
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else False
    return {"next_step": "save" if approved else "cancel", "log": [f"[human_approve] approved={approved}"]}


def route_after_approval(state: AgentState) -> str:
    return state["next_step"]


# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------

def save_node(state: AgentState) -> dict:
    import os

    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", f"{_slugify(state['query'])}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(state["report_markdown"])
    return {"output_path": path, "log": [f"[save] report written to {path}"]}
