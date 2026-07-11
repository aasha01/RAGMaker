"""Adapters to use local LLMs (Ollama, Anthropic) as ragas judge LLMs.

ragas defaults to OpenAI's API for grading (needs OPENAI_API_KEY). This module
provides adapters so a learner can use Ollama (free, local) or Anthropic (Claude)
instead — same quality metrics, different backend.

Each adapter wraps our LLM providers to conform to ragas's expected interface.
All are lazy-imported: if ragas isn't installed, importing this module costs
nothing.
"""

from __future__ import annotations

import os
from typing import Optional


def get_ragas_judge_llm():
    """Factory: return a ragas-compatible judge LLM based on env vars.

    Priority (first match wins):
    1. RAGAS_LLM_PROVIDER env var: "ollama" (default Ollama), "anthropic"
       (requires ANTHROPIC_API_KEY)
    2. OLLAMA_HOST present → use Ollama
    3. ANTHROPIC_API_KEY present → use Anthropic
    4. None (ragas will default to OpenAI, needs OPENAI_API_KEY)

    Raises RuntimeError if the chosen provider can't be initialized (e.g.
    Ollama not running, Anthropic key missing).
    """
    provider = os.getenv("RAGAS_LLM_PROVIDER", "").lower()

    # Explicit provider selection
    if provider == "ollama":
        return _make_ollama_llm()
    if provider == "anthropic":
        return _make_anthropic_llm()

    # Auto-detect based on env vars
    if os.getenv("OLLAMA_HOST"):
        return _make_ollama_llm()
    if os.getenv("ANTHROPIC_API_KEY"):
        return _make_anthropic_llm()

    # No local LLM configured; ragas will try OpenAI
    return None


def _make_ollama_llm():
    """Create a ragas-compatible wrapper for Ollama."""
    try:
        from langchain_community.llms import Ollama
    except ImportError as e:
        raise RuntimeError(
            "Ollama adapter requires langchain-community. "
            "Install it with: pip install langchain-community"
        ) from e

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("RAGAS_OLLAMA_MODEL", "llama3.1:8b")

    try:
        return Ollama(
            base_url=host.rstrip("/"),
            model=model,
            temperature=0.0,  # Deterministic scoring
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Ollama judge LLM at {host}. "
            f"Is the Ollama server running? "
            f"Start it with 'ollama serve' and ensure '{model}' is pulled. "
            f"Original error: {e}"
        ) from e


def _make_anthropic_llm():
    """Create a ragas-compatible wrapper for Anthropic Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Anthropic adapter requires ANTHROPIC_API_KEY env var to be set."
        )

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise RuntimeError(
            "Anthropic adapter requires langchain-anthropic. "
            "Install it with: pip install langchain-anthropic"
        ) from e

    model = os.getenv("RAGAS_ANTHROPIC_MODEL", "claude-opus-4-8")

    try:
        return ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=0.0,  # Deterministic scoring
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Anthropic judge LLM (model={model}). "
            f"Check that ANTHROPIC_API_KEY is valid. Original error: {e}"
        ) from e
