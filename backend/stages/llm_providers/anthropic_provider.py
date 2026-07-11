"""Hosted generation via the Anthropic (Claude) API."""

from __future__ import annotations

import os
import time

from .base import BaseLLMProvider, GenerationResult


class AnthropicProvider(BaseLLMProvider):
    """Generate answers with an Anthropic Claude model over their hosted API.

    What it does (mechanically): sends the prompt as a single user message to the
    Claude Messages API and reads back the reply. Token usage
    (``input_tokens`` / ``output_tokens``) is reported by the API itself; latency
    is wall-clock timed here, and the dollar cost is computed from a small built-in
    price table for the model you chose. This is a plain single-turn completion
    (no extended thinking) so the latency, tokens, and cost line up fairly against
    the OpenAI and Ollama providers in the compare grid.

    Tradeoff vs. the alternatives: Claude models are strong at grounded,
    instruction-following answers over retrieved context, but every call costs
    money, needs an ``ANTHROPIC_API_KEY``, and sends your prompt to Anthropic.
    Ollama is the free, private, local option; OpenAI is the other hosted option —
    running all three on the *same* retrieved chunks is exactly what makes the
    comparison meaningful.

    When a learner would prefer it: when you want to compare how Claude answers the
    same RAG context as the other providers, and see the real per-query cost, so
    "which model is best for my documents" becomes a measured result rather than a
    vibe.

    Cost note: ``cost_usd`` is computed from ``PRICE_PER_1M`` below. A model not in
    that table yields ``None`` (unknown) rather than a fabricated number.

    Parameters (recorded in config.json):
        model: the Claude model id (default ``claude-opus-4-8``, the latest Opus).
        max_tokens: cap on the generated answer length (required by the API).
        api_key: optional; falls back to the ``ANTHROPIC_API_KEY`` environment
            variable when not given (the usual, safer path — no key in config).

    The ``anthropic`` SDK is imported lazily inside ``generate`` so the tool still
    runs (and this module still imports, e.g. to show its description in the UI)
    when the package isn't installed — a missing package surfaces a friendly
    install message.
    """

    name = "Anthropic Claude (API)"
    description = (
        "Generates with an Anthropic Claude model (default claude-opus-4-8) over "
        "their hosted API — strong at grounded answers over retrieved context, but "
        "costs money per call and needs an ANTHROPIC_API_KEY. Per-query dollar cost "
        "is computed and shown so you can compare it against the other providers."
    )

    #: USD per 1M tokens, (input, output). Update as Anthropic pricing changes; an
    #: unlisted model yields cost_usd=None rather than a made-up figure.
    PRICE_PER_1M: dict[str, tuple[float, float]] = {
        "claude-opus-4-8": (5.00, 25.00),
        "claude-opus-4-7": (5.00, 25.00),
        "claude-opus-4-6": (5.00, 25.00),
        "claude-sonnet-5": (3.00, 15.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-fable-5": (10.00, 50.00),
    }

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            ) from e

        api_key = kwargs.get("api_key", self.api_key) or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment "
                "variable, or pass api_key when configuring the provider."
            )

        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        client = Anthropic(api_key=api_key)
        start = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        # The response content is a list of blocks; concatenate the text blocks.
        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

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
