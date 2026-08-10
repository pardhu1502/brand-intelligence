"""Wires the nodes into the LangGraph agentic loop.

    search -> select_links -> human_review --(parallel Send fan-out)--> extract_link (xN)
                                                                              |
                                                                    evaluate_sufficiency
                                                                     /               \\
                                                          (insufficient)          (enough data)
                                                                /                       \\
                                                          back to search            synthesize -> critic -> write_report
                                                                                                                  |
                                                                                                      human_approve_report
                                                                                                       /               \\
                                                                                                 (approved)        (rejected)
                                                                                                     |                  |
                                                                                                    save               END

A MemorySaver checkpointer backs the two `interrupt()` calls (human_review,
human_approve_report) so the graph can pause mid-run and resume once the
CLI relays the human's decision.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.nodes import (
    critic_node,
    dispatch_extraction,
    evaluate_sufficiency_node,
    extract_link_node,
    human_approve_report_node,
    human_review_node,
    route_after_approval,
    route_after_evaluation,
    save_node,
    search_node,
    select_links_node,
    synthesize_node,
    write_report_node,
)
from agent.state import AgentState


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("search", search_node)
    builder.add_node("select_links", select_links_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("extract_link", extract_link_node)
    builder.add_node("evaluate_sufficiency", evaluate_sufficiency_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("critic", critic_node)
    builder.add_node("write_report", write_report_node)
    builder.add_node("human_approve_report", human_approve_report_node)
    builder.add_node("save", save_node)

    builder.set_entry_point("search")
    builder.add_edge("search", "select_links")
    builder.add_edge("select_links", "human_review")

    builder.add_conditional_edges(
        "human_review", dispatch_extraction, ["extract_link", "evaluate_sufficiency"]
    )
    builder.add_edge("extract_link", "evaluate_sufficiency")

    builder.add_conditional_edges(
        "evaluate_sufficiency", route_after_evaluation, {"search": "search", "synthesize": "synthesize"}
    )

    builder.add_edge("synthesize", "critic")
    builder.add_edge("critic", "write_report")
    builder.add_edge("write_report", "human_approve_report")

    builder.add_conditional_edges(
        "human_approve_report", route_after_approval, {"save": "save", "cancel": END}
    )
    builder.add_edge("save", END)

    return builder.compile(checkpointer=MemorySaver())
