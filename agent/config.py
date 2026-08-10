"""Environment/config loading and LLM client factory.

Supports groq (default), openai, anthropic, gemini, and ollama — switching
between them is a one-line env change (see .env / README). Ollama can also
be routed to just the extraction sub-task independently of the main
provider (bonus: local LLM for one sub-task).
"""

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

OLLAMA_EXTRACTION = os.getenv("OLLAMA_EXTRACTION", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "2"))
MIN_SUCCESSFUL_EXTRACTIONS = int(os.getenv("MIN_SUCCESSFUL_EXTRACTIONS", "3"))
MAX_LINKS_PER_ROUND = int(os.getenv("MAX_LINKS_PER_ROUND", "5"))

# Domains treated as "high-cost" (paywalled / rate-limited) — visiting one
# triggers the human-in-the-loop approval step instead of auto-browsing.
HIGH_COST_DOMAINS = {
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "economist.com",
    "nytimes.com",
    "businessinsider.com",
}


def _build_chat_model(provider: str, model: str, temperature: float = 0.0):
    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (required for LLM_PROVIDER=groq).")
        return ChatGroq(model=model, temperature=temperature, api_key=api_key)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for LLM_PROVIDER=openai).")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (required for LLM_PROVIDER=anthropic).")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (required for LLM_PROVIDER=gemini).")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature, base_url=OLLAMA_BASE_URL)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_llm(temperature: float = 0.0):
    """Main reasoning LLM: planning, synthesis, critique, report writing."""
    return _build_chat_model(LLM_PROVIDER, LLM_MODEL, temperature)


def get_extraction_llm(temperature: float = 0.0):
    """LLM used for the per-page structured extraction sub-task.

    Routed to a local Ollama model when OLLAMA_EXTRACTION=true (bonus:
    "Using a local LLM via Ollama for one of the sub-tasks"). Falls back to
    the main provider otherwise.
    """
    if OLLAMA_EXTRACTION:
        return _build_chat_model("ollama", OLLAMA_MODEL, temperature)
    return _build_chat_model(LLM_PROVIDER, LLM_MODEL, temperature)
