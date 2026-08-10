"""Page-fetch + structured-extraction tool.

Two failure points are handled separately so the agent can tell them apart
in its reasoning log: network/parsing failures (fetch_page) vs. the LLM
extraction call itself (extract_fields).
"""

from typing import List, Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from agent.prompts import EXTRACTION_SYSTEM_PROMPT

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BrandIntelligenceAgent/1.0 "
    "(+research assignment; contact: user)"
)
MAX_PAGE_CHARS = 6000


class FetchError(Exception):
    """Raised when a page cannot be fetched/parsed."""


class ExtractionError(Exception):
    """Raised when the LLM extraction call fails."""


class ExtractionSchema(BaseModel):
    product_name: Optional[str] = Field(default=None, description="Specific product/brand name, if any.")
    key_ingredients: List[str] = Field(default_factory=list, description="Named ingredients/actives.")
    pricing: Optional[str] = Field(default=None, description="Price or price range as written on the page.")
    usp: Optional[str] = Field(default=None, description="Unique selling proposition / main product claim.")
    summary: str = Field(description="1-2 sentence neutral summary of the page.")


def fetch_page(url: str, timeout: float = 12.0) -> str:
    """Fetch a URL and return its visible text, truncated to a token-friendly length."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"Could not fetch {url}: {exc}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    if not text:
        raise FetchError(f"No readable text extracted from {url}")

    return text[:MAX_PAGE_CHARS]


def extract_fields(llm, url: str, page_text: str) -> ExtractionSchema:
    """Run structured extraction over fetched page text using the given LLM."""
    structured_llm = llm.with_structured_output(ExtractionSchema)
    try:
        return structured_llm.invoke(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"URL: {url}\n\nPAGE TEXT:\n{page_text}"},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"LLM extraction failed for {url}: {exc}") from exc
