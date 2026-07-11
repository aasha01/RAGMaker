"""Hosted generation via the OpenAI API."""

from __future__ import annotations

import os
import time

from .base import BaseLLMProvider, GenerationResult


class OpenAIProvider(BaseLLMProvider):
    """Generate answers with an OpenAI chat model over their hosted API.

    What it does (mechanically): sends the prompt as a single user message to
    OpenAI's Chat Completions endpoint and reads back the reply. Token usage
    (``prompt_tokens`` / ``completion_tokens``) is reported by the API itself;
    latency is wall-clock timed here, and the dollar cost is computed from a small
    built-in price table for the model you chose.

    Tradeoff vs. the alternatives: OpenAI's models are strong and fast and need no
    local GPU, but every call costs money, needs an ``OPENAI_API_KEY``, and sends
    your prompt to OpenAI. The Ollama provider is free and private but runs on your
    own hardware; the Anthropic provider is the equivalent hosted option for Claude
    models — having all three side by side is the whole point of the compare grid.

    When a learner would prefer it: when you want to see how a strong, widely-used
    hosted model answers the *same* retrieved context as your local model — and
    exactly what that costs per query — so the quality-vs-cost tradeoff is a number
    you can point at, not a guess.

    Cost note: ``cost_usd`` is computed from ``PRICE_PER_1M`` below. If you pick a
    model that isn't in that table, cost is reported as ``None`` (unknown) rather
    than a fabricated number — the tokens are still real, only the dollar estimate
    is withheld.

    Parameters (recorded in config.json):
        model: the OpenAI chat model id (default ``gpt-4o-mini``).
        max_tokens: cap on the generated answer length.
        api_key: optional; falls back to the ``OPENAI_API_KEY`` environment
            variable when not given (the usual, safer path — no key in config).

    The ``openai`` SDK is imported lazily inside ``generate`` so the tool still runs
    (and this module still imports, e.g. to show its description in the UI) when the
    package isn't installed — a missing package surfaces a friendly install message.
    """

    name = "OpenAI (API)"
    description = (
        "Generates with an OpenAI chat model (default gpt-4o-mini) over their "
        "hosted API — strong and fast, no local GPU, but costs money per call and "
        "needs an OPENAI_API_KEY. Per-query dollar cost is computed and shown so "
        "you can compare it directly against the free local Ollama provider."
    )

    #: USD per 1M tokens, (input, output). Update as OpenAI pricing changes; an
    #: unlisted model yields cost_usd=None rather than a made-up figure.
    PRICE_PER_1M: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from e

        api_key = kwargs.get("api_key", self.api_key) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key found. Set the OPENAI_API_KEY environment "
                "variable, or pass api_key when configuring the provider."
            )

        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        client = OpenAI(api_key=api_key)
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._cost(model, input_tokens, output_tokens),
        )

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.PRICE_PER_1M.get(model)
        if price is None:
            return None  # unknown model — withhold rather than fabricate a cost
        in_rate, out_rate = price
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0
