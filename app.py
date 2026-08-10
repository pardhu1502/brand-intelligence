"""Gradio web frontend for the Brand Intelligence Agent.

Wraps the same LangGraph agent used by main.py. The two human-in-the-loop
interrupts (approve visiting a high-cost domain, approve finalizing the
report) are rendered as an approve/reject panel in the UI instead of a
terminal prompt.

Run with: python app.py
"""

import uuid

import gradio as gr
from langgraph.types import Command

from agent.graph import build_graph
from agent.state import AgentState

SAMPLE_QUERY = "Competitive landscape of premium skincare serums in Southeast Asia"

# The compiled graph is stateless/reentrant, so one instance is shared across
# sessions; per-session progress lives in Gradio's gr.State (thread_id, how
# much of the log has already been streamed, and what kind of interrupt is
# currently pending).
_graph = build_graph()


def _initial_state(query: str) -> AgentState:
    return {
        "query": query,
        "search_query": query,
        "round": 0,
        "next_step": "",
        "search_results": [],
        "selected_links": [],
        "extracted": [],
        "themes": [],
        "gaps": [],
        "critique_notes": [],
        "report_markdown": "",
        "output_path": "",
        "log": [],
    }


def _render_interrupt(payload: dict):
    """Return (message_md, checkbox_update, preview_md) for the pending interrupt."""
    kind = payload.get("type")
    if kind == "high_cost_site_approval":
        links = payload.get("links", [])
        msg = f"### Human approval needed\n{payload.get('message', '')}"
        return msg, gr.update(choices=links, value=[], visible=True), ""
    if kind == "finalize_report_approval":
        msg = f"### Human approval needed\n{payload.get('message', '')}"
        return msg, gr.update(choices=[], value=[], visible=False), payload.get("preview", "")
    return "### Human approval needed", gr.update(choices=[], value=[], visible=False), ""


def _no_interrupt_result(result: dict, log_text: str, session: dict):
    session["pending_type"] = None
    return (
        log_text,
        gr.update(visible=False),
        "",
        gr.update(choices=[], value=[], visible=False),
        "",
        result.get("report_markdown", "") or "_No report was generated (see log above)._",
        result.get("output_path") or None,
        session,
    )


def _interrupt_result(payload: dict, log_text: str, session: dict):
    session["pending_type"] = payload.get("type")
    msg, checkbox_update, preview = _render_interrupt(payload)
    return (
        log_text,
        gr.update(visible=True),
        msg,
        checkbox_update,
        preview,
        "",
        None,
        session,
    )


def start_run(query: str, session: dict):
    if not query or not query.strip():
        query = SAMPLE_QUERY

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    session = {"config": config, "pending_type": None}

    result = _graph.invoke(_initial_state(query), config=config)
    log_text = "\n".join(result.get("log", []))

    if "__interrupt__" in result:
        return _interrupt_result(result["__interrupt__"][0].value, log_text, session)
    return _no_interrupt_result(result, log_text, session)


def resolve_approval(decision: str, checkbox_selection: list, session: dict):
    if not session or not session.get("config"):
        return (
            "No active run to resolve.", gr.update(visible=False), "",
            gr.update(choices=[], value=[], visible=False), "", "", None, session or {},
        )

    config = session["config"]
    pending_type = session.get("pending_type")

    if pending_type == "high_cost_site_approval":
        answer = {"approved_urls": checkbox_selection if decision == "approve" else []}
    elif pending_type == "finalize_report_approval":
        answer = {"approved": decision == "approve"}
    else:
        answer = {}

    result = _graph.invoke(Command(resume=answer), config=config)
    log_text = "\n".join(result.get("log", []))

    if "__interrupt__" in result:
        return _interrupt_result(result["__interrupt__"][0].value, log_text, session)
    return _no_interrupt_result(result, log_text, session)


with gr.Blocks(title="Brand Intelligence Agent") as demo:
    gr.Markdown(
        "# Brand Intelligence Agent\n"
        "Autonomous competitor research agent (LangGraph + Groq + Tavily). "
        "Enter a category/query and run — the agent searches the web, visits "
        "several sources in parallel, synthesizes themes and gaps, fact-checks "
        "itself, and drafts a Markdown report. It will pause below for approval "
        "before browsing paywalled sites or finalizing the report."
    )

    session_state = gr.State({})

    with gr.Row():
        query_box = gr.Textbox(label="Research query", value=SAMPLE_QUERY, scale=4)
        run_btn = gr.Button("Run Agent", variant="primary", scale=1)

    log_box = gr.Textbox(label="Agent reasoning log", lines=16, interactive=False)

    with gr.Group(visible=False) as approval_group:
        approval_message = gr.Markdown()
        approval_preview = gr.Markdown()
        approval_checkboxes = gr.CheckboxGroup(label="High-cost links", choices=[], visible=False)
        with gr.Row():
            approve_btn = gr.Button("Approve", variant="primary")
            reject_btn = gr.Button("Reject / Skip")

    report_box = gr.Markdown(label="Final report")
    download_box = gr.File(label="Download report (.md)")

    outputs = [log_box, approval_group, approval_message, approval_checkboxes, approval_preview, report_box, download_box, session_state]

    run_btn.click(start_run, inputs=[query_box, session_state], outputs=outputs)
    approve_btn.click(
        lambda checkboxes, session: resolve_approval("approve", checkboxes, session),
        inputs=[approval_checkboxes, session_state],
        outputs=outputs,
    )
    reject_btn.click(
        lambda checkboxes, session: resolve_approval("reject", checkboxes, session),
        inputs=[approval_checkboxes, session_state],
        outputs=outputs,
    )


if __name__ == "__main__":
    demo.launch()
