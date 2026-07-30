"""LLM + embeddings client factories.

LLM calls use **Groq** (via the OpenAI-compatible ``ChatOpenAI`` wrapper).
Embeddings still use **NVIDIA NIM** since Groq does not offer embeddings.

Usage::

    from llm import get_router_llm, get_answer_llm, get_embeddings

    router = get_router_llm()       # fast — classification only
    answer = get_answer_llm()       # stronger, streaming — answer generation
    embed = get_embeddings()        # vectorisation for ingestion / retrieval
"""

import sys
from typing import NoReturn

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
)

# ── Lazy-initialised module-level singletons ──

_router_llm: ChatOpenAI | None = None
_answer_llm: ChatOpenAI | None = None
_embeddings_client: NVIDIAEmbeddings | None = None


# ── Helpers ──


def _abort_missing_key(name: str, source: str) -> NoReturn:
    """Print a clear message and exit when an API key is missing."""
    print(
        f"ERROR: {name} is not set.\n"
        f"  Add it to backend/.env.\n"
        f"  Get a key at {source}",
        file=sys.stderr,
    )
    sys.exit(1)


def _check_groq_key() -> None:
    """Exit if ``GROQ_API_KEY`` is missing."""
    if not GROQ_API_KEY or GROQ_API_KEY.strip() == "":
        _abort_missing_key("GROQ_API_KEY", "https://console.groq.com/keys")


def _check_nvidia_key() -> None:
    """Exit if ``NVIDIA_API_KEY`` is missing (needed for embeddings)."""
    if not NVIDIA_API_KEY or NVIDIA_API_KEY.strip() == "":
        _abort_missing_key("NVIDIA_API_KEY", "https://build.nvidia.com/")


# ── Factory getters ──


def get_router_llm() -> ChatOpenAI:
    """Fast model for intent classification — via Groq.

    Low temperature for deterministic output, small max_tokens since the
    output is a structured intent label, not prose.
    Timeout: 45s total.
    """
    global _router_llm
    if _router_llm is None:
        _check_groq_key()
        _router_llm = ChatOpenAI(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            temperature=0.0,
            max_tokens=256,
            timeout=45,
        )
    return _router_llm


def get_answer_llm() -> ChatOpenAI:
    """Stronger model for answer generation (streaming) — via Groq.

    *streaming=True* by default in LangChain — enables token-by-token
    SSE delivery to the frontend.
    Timeout: 60s — Groq is fast, but RAG-heavy prompts can take time.
    """
    global _answer_llm
    if _answer_llm is None:
        _check_groq_key()
        _answer_llm = ChatOpenAI(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            temperature=0.2,
            max_tokens=4096,
            timeout=60,
        )
    return _answer_llm


def get_embeddings() -> NVIDIAEmbeddings:
    """Embeddings client — via NVIDIA NIM (Groq doesn't support embeddings).

    Used for chunk vectorisation during ingestion and for embedding the
    user query at retrieval time.
    Timeout: 60s.
    """
    global _embeddings_client
    if _embeddings_client is None:
        _check_nvidia_key()
        _embeddings_client = NVIDIAEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            timeout=60,
        )
    return _embeddings_client
