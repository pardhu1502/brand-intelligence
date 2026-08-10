"""CLI entry point for the Brand Intelligence Agent.

Usage:
    python main.py "Latest trends in organic hair oils in India 2024"
    python main.py            # uses the assignment's required sample query
"""

import argparse
import uuid

from langgraph.types import Command

from agent.graph import build_graph
from agent.state import AgentState

SAMPLE_QUERY = "Competitive landscape of premium skincare serums in Southeast Asia"


def handle_interrupt_cli(payload: dict) -> dict:
    """Render a human-in-the-loop prompt in the terminal and collect the decision."""
    print("\n--- HUMAN APPROVAL NEEDED ---")
    print(payload.get("message", ""))

    if payload.get("type") == "high_cost_site_approval":
        approved = []
        for link in payload.get("links", []):
            answer = input(f"  Approve visiting {link}? [y/N]: ").strip().lstrip("﻿").lower()
            if answer == "y":
                approved.append(link)
        return {"approved_urls": approved}

    if payload.get("type") == "finalize_report_approval":
        print(payload.get("preview", ""))
        answer = input("\nApprove finalizing & saving this report? [y/N]: ").strip().lstrip("﻿").lower()
        return {"approved": answer == "y"}

    return {}


def _flush_log(result: dict, printed: int) -> int:
    log = result.get("log", [])
    for line in log[printed:]:
        print(line)
    return len(log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brand Intelligence Agent (LangGraph)")
    parser.add_argument("query", nargs="?", default=SAMPLE_QUERY, help="Research query")
    args = parser.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state: AgentState = {
        "query": args.query,
        "search_query": args.query,
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

    print(f"Researching: {args.query}\n")
    result = graph.invoke(initial_state, config=config)
    printed = _flush_log(result, 0)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        answer = handle_interrupt_cli(payload)
        result = graph.invoke(Command(resume=answer), config=config)
        printed = _flush_log(result, printed)

    if result.get("output_path"):
        print(f"\nDone. Report saved to: {result['output_path']}")
    else:
        print("\nRun ended without saving a report (declined at approval, or no usable data found).")


if __name__ == "__main__":
    main()
