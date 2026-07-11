"""LLM provider stage — interface and data contract.

Generation is treated as a *variable* in comparisons, not a fixed constant, so
providers sit behind the same swappable-Strategy interface as every other
stage. Each `generate` call reports not just the text but the latency, token
counts, and cost, so the Query & Compare grid can rank providers honestly
(ARCHITECTURE.md sections 6-7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """The output of one generation call.

    Attributes:
        text: The generated answer.
        latency_ms: Wall-clock time for the call, in milliseconds.
        input_tokens: Prompt tokens consumed (0 if the backend can't report it).
        output_tokens: Completion tokens produced.
        cost_usd: Estimated dollar cost, or None for local/free providers where
            cost is not meaningful (e.g. Ollama).
    """

    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class BaseLLMProvider(ABC):
    """Abstract LLM provider. Every concrete provider is a swappable Strategy.

    Optional/heavy SDKs (openai, anthropic, ...) must be imported lazily inside
    the concrete method so the app still runs when a provider's package or API
    key is absent; a missing provider surfaces a friendly message, never a
    crash at import time (CLAUDE.md Style).
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate an answer for `prompt` and report timing/token/cost stats."""
        raise NotImplementedError
