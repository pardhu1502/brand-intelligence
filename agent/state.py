"""Shared state schema for the LangGraph agentic loop."""

import operator
from typing import Annotated, List, Optional, TypedDict


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


class ExtractedInfo(TypedDict):
    url: str
    product_name: Optional[str]
    key_ingredients: List[str]
    pricing: Optional[str]
    usp: Optional[str]
    summary: str
    success: bool
    error: Optional[str]


class AgentState(TypedDict):
    # Input
    query: str

    # Search loop
    search_query: str
    round: int
    next_step: str
    search_results: List[SearchResult]
    selected_links: List[str]

    # Extraction (parallel fan-out; operator.add merges branch results into one list)
    extracted: Annotated[List[ExtractedInfo], operator.add]

    # Synthesis
    themes: List[str]
    gaps: List[str]

    # Critic (bonus: fact-check pass)
    critique_notes: List[str]

    # Output
    report_markdown: str
    output_path: str

    # Chain-of-thought / reasoning trail shown to the user
    log: Annotated[List[str], operator.add]
