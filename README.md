# Brand Intelligence Agent

An autonomous research agent built for the "Brand Intelligence Agent" assignment.
Given a single query (e.g. *"Competitive landscape of premium skincare serums in
Southeast Asia"*), it searches the web, visits multiple product/news pages in
parallel, extracts structured product data, synthesizes themes and market gaps,
fact-checks its own synthesis, and writes a clean Markdown report — pausing for
human approval before browsing paywalled sites or finalizing the report.

## Stack

- **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) — an explicit state-graph agentic loop (chosen over CrewAI so every decision point, retry, and parallel branch is visible and controllable, not hidden behind a higher-level abstraction).
- **LLM:** Groq (`llama-3.1-8b-instant` by default) via `langchain-groq`. OpenAI, Anthropic, Gemini, and a local Ollama route are also wired up — see [Switching LLM provider](#switching-llm-provider).
- **Search tool:** [Tavily](https://tavily.com/) via `tavily-python`.
- **Page fetching:** `httpx` + `BeautifulSoup`.

> **Note on LLM choice:** the assignment brief lists OpenAI GPT-4o/4o-mini or
> Anthropic Claude 3.5 Sonnet as the intended models. This build defaults to
> Groq's `llama-3.1-8b-instant` per project instruction. The code is
> provider-agnostic (`agent/config.py`), so swapping in GPT-4o-mini or Claude
> 3.5 Sonnet is a one-line `.env` change if stronger extraction/reasoning
> quality is needed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env          # then fill in your API keys
```

Required in `.env`:
- `TAVILY_API_KEY` — search tool
- `GROQ_API_KEY` — LLM (default provider)

## Run

```bash
python main.py "Latest trends in organic hair oils in India 2024"

# or, with no argument, runs the assignment's required sample query:
python main.py
```

The agent prints its reasoning trail (Chain-of-Thought log) live to the
console as it works, pauses in the terminal for any human-in-the-loop
approvals, and writes the final report to `outputs/<slugified-query>.md`.

## Web UI (Gradio)

A browser frontend is available alongside the CLI:

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`. Enter a query and click **Run Agent** — the
reasoning log streams to a text panel, and when the agent hits a
human-in-the-loop gate (high-cost domain, or finalize report) an
approve/reject panel appears in place of the CLI's `input()` prompt. The
final report renders inline and is available as a `.md` download.

## Switching LLM provider

Edit `.env`:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

or

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

or

```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=...
```

## Architecture: the agentic loop

```
 search ──► select_links ──► human_review ──► extract_link (× N, parallel)
   ▲                                                    │
   │                                                    ▼
   └──────────────── retry ── evaluate_sufficiency ◄────┘
                             │
                             ▼ (enough data, or round limit hit)
                        synthesize ──► critic ──► write_report
                                                        │
                                                        ▼
                                            human_approve_report
                                             │                │
                                          approved          declined
                                             │                │
                                             ▼                ▼
                                            save              END
```

This is a LangGraph `StateGraph` — a directed graph of nodes that read and
write a shared `AgentState`, wired with explicit conditional edges rather
than an implicit "agent decides everything" loop. That's a deliberate
choice for the eval criteria in this assignment (tool-failure handling,
loop prevention, visible reasoning, efficiency):

1. **`search`** — calls Tavily with the user's query. On round 0 this uses
   the query as-is (no LLM call — efficiency). On retry rounds, an LLM call
   refines the query based on what was already tried. If Tavily fails, the
   node catches the error, logs it, and returns empty results instead of
   crashing — the graph continues to `select_links` with nothing to select,
   which naturally routes back through `evaluate_sufficiency` to retry.

2. **`select_links`** — a plain heuristic (dedupe by domain, cap at
   `MAX_LINKS_PER_ROUND`, skip already-visited URLs), not an LLM call.
   Link ranking from search results is already decent; spending a model
   call here would waste tokens for no reasoning benefit.

3. **`human_review`** — checks the selected links against a
   `HIGH_COST_DOMAINS` list (paywalled sites like WSJ, FT, Bloomberg). If
   any are present, the graph **pauses** via LangGraph's `interrupt()` and
   the CLI asks the user to approve/skip each one before browsing
   continues. This is the assignment's human-in-the-loop bonus point,
   applied to "before browsing a high-cost site."

4. **`extract_link`** — fans out via LangGraph's `Send` API: every approved
   link is visited **in parallel**, each in its own node invocation, each
   with its own try/except around (a) the HTTP fetch/parse and (b) the LLM
   structured-extraction call. A failure in one link (timeout, 404,
   unparseable page, malformed LLM output) is recorded with its error and
   does not block or fail the other parallel branches — this is the
   graceful tool-failure handling the eval criteria ask about.

5. **`evaluate_sufficiency`** — the loop-control node. It counts successful
   extractions; if fewer than `MIN_SUCCESSFUL_EXTRACTIONS` (default 3) and
   the round cap (`MAX_SEARCH_ROUNDS`, default 2) hasn't been hit, it routes
   back to `search` with `round += 1`. The hard round cap is what prevents
   the "infinite looping" failure mode the assignment explicitly warns
   about — the agent always terminates in a bounded number of search
   rounds regardless of data quality.

6. **`synthesize`** — one LLM call, given only the structured extractions
   (not raw page text), asked to name recurring themes (each citing its
   supporting sources) and market gaps. Grounding the prompt in structured
   data rather than free text reduces hallucination.

7. **`critic`** *(bonus: a second agent fact-checking the first)* — a
   separate LLM pass whose only job is to try to refute each theme/gap
   against the same source data and flag anything unsupported. Flagged
   claims are passed to the writer to soften or drop, not silently kept.

8. **`write_report`** — composes the final Markdown (executive summary,
   product table, themes, gaps, notes/limitations, sources) strictly from
   the structured extractions, synthesis, and critic notes — no new facts
   introduced at this stage.

9. **`human_approve_report`** — a second `interrupt()`: the CLI shows a
   preview of the draft and asks for approval before anything is written
   to disk. Declining ends the run with nothing saved. This is the
   assignment's other human-in-the-loop bonus point ("before ... finalizing
   the report").

10. **`save`** — writes the approved Markdown to `outputs/`.

Every node also appends short, human-readable lines to a shared `log` list
in the state (e.g. `[extract] https://...: found product='X Rosemary Oil'`) —
this is the visible "I found X, now I need to check Y" reasoning trail the
assignment's evaluation criteria ask for, and it's what streams to the
console while the agent runs.

### Bonus points implemented

- ✅ **Human-in-the-loop**: approval gates before visiting high-cost domains
  and before finalizing the report (`agent/nodes.py`:`human_review_node`,
  `human_approve_report_node`).
- ✅ **Multi-agent-style critique**: a dedicated `critic` LLM pass
  fact-checks the `synthesize` pass before the report is written.
- ✅ **Local LLM via Ollama**: set `OLLAMA_EXTRACTION=true` in `.env` to
  route the per-page extraction sub-task (`extract_link`) through a local
  Ollama model instead of the cloud provider (`agent/config.py`:
  `get_extraction_llm`). Requires `ollama serve` running locally with the
  model pulled (default `llama3.1`).
- ✅ **Parallel tool use**: link extraction fans out via LangGraph's `Send`
  API so 3-5 pages are fetched/extracted concurrently rather than
  sequentially.

## System prompts used

All prompts live in `agent/prompts.py`:

| Prompt | Used by | Purpose |
|---|---|---|
| `QUERY_REFINEMENT_SYSTEM_PROMPT` | `search_node` (retry rounds only) | Proposes a more specific search query when the first round didn't yield enough usable pages |
| `EXTRACTION_SYSTEM_PROMPT` | `extract_link_node` | Pulls product name / ingredients / pricing / USP / summary strictly from a single page's text, with an explicit no-fabrication rule |
| `SYNTHESIS_SYSTEM_PROMPT` | `synthesize_node` | Derives cross-source themes (with citations) and market gaps from the structured extractions |
| `CRITIC_SYSTEM_PROMPT` | `critic_node` | Adversarially checks each theme/gap against the source data and flags unsupported claims |
| `REPORT_WRITER_SYSTEM_PROMPT` | `write_report_node` | Renders the final Markdown report structure, instructed to use only the supplied facts |

## Project structure

```
agent/
  config.py    # env + LLM provider factory (groq / openai / anthropic / ollama)
  state.py     # AgentState TypedDict (LangGraph shared state)
  prompts.py   # all system prompts
  nodes.py     # the 10 graph node functions
  graph.py     # StateGraph wiring
tools/
  search_tool.py   # Tavily wrapper with retry
  extract_tool.py  # page fetch + structured LLM extraction
main.py        # CLI entry point + human-in-the-loop terminal UI
app.py         # Gradio web frontend (same agent, browser-based HITL UI)
outputs/       # generated reports land here
```

## Known limitations

- `llama-3.1-8b-instant` is a small model; extraction/synthesis quality on
  ambiguous or JS-heavy pages will be noticeably weaker than GPT-4o-mini or
  Claude 3.5 Sonnet. Switch providers (see above) for higher-fidelity runs.
- Page fetching is static HTML only (no headless browser), so JavaScript-
  rendered product pages may yield little or no extractable text; this
  surfaces as a `fetch failed` / empty-summary entry rather than a crash.
