"""Web search tool (Tavily) with retry and graceful failure handling."""

import time

from agent.config import TAVILY_API_KEY
from agent.state import SearchResult


class SearchToolError(Exception):
    """Raised when the search tool fails after all retries."""


def web_search(query: str, max_results: int = 8, retries: int = 1) -> list[SearchResult]:
    """Run a Tavily web search. Retries once on transient failure, then
    raises SearchToolError so the caller can log it and degrade gracefully
    instead of crashing the whole agent run.
    """
    if not TAVILY_API_KEY:
        raise SearchToolError("TAVILY_API_KEY is not set.")

    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )
            results = response.get("results", [])
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                )
                for r in results
                if r.get("url")
            ]
        except Exception as exc:  # noqa: BLE001 - genuinely want to catch any client/network error
            last_error = exc
            if attempt < retries:
                time.sleep(1.5)

    raise SearchToolError(f"Tavily search failed after {retries + 1} attempt(s): {last_error}")
