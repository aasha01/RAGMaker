"""Local generation via an Ollama server — the no-API-key, zero-cost default."""

from __future__ import annotations

import time

from .base import BaseLLMProvider, GenerationResult


class OllamaProvider(BaseLLMProvider):
    """Generate answers with a model running locally under Ollama.

    What it does (mechanically): sends the prompt to an Ollama server on your own
    machine (``http://localhost:11434`` by default) over plain HTTP and reads back
    the completion. Ollama hosts open-weight models (Llama, Mistral, Qwen, ...) as
    a local service, so the text never leaves your computer and there is no
    metered API bill. Token counts come straight from Ollama's own response
    (``prompt_eval_count`` / ``eval_count``); latency is wall-clock timed here.

    Tradeoff vs. the alternatives: the OpenAI and Anthropic providers reach
    larger, generally stronger hosted models, but they cost money per call, need
    an API key, and send your prompt to a third party. Ollama is free and private,
    at the price of running the model on your own CPU/GPU (slower, and quality
    depends on which local model you pulled).

    When a learner would prefer it: as the default first comparison — you can see
    a full RAG answer generated end-to-end with zero accounts or spend, then add a
    hosted provider alongside it to feel the quality/latency/cost difference in the
    Query & Compare grid for yourself.

    Cost note: ``cost_usd`` is reported as ``0.0`` (not ``None``) on purpose — local
    inference genuinely costs no API dollars, and a concrete ``$0.00`` in the
    comparison grid reads more clearly next to the paid providers' real costs than
    a blank would.

    Parameters (recorded in config.json):
        model: the Ollama model tag to run (e.g. ``llama3.2``, ``mistral``). You
            must have pulled it first with ``ollama pull <model>``.
        host: base URL of the Ollama server. Default ``http://localhost:11434``.
        options: optional dict of Ollama generation options (e.g.
            ``{"temperature": 0.2, "num_predict": 512}``) passed through verbatim.

    ``httpx`` (already a core dependency of the backend) is imported lazily inside
    ``generate`` so merely importing this module — e.g. to read its description in
    the UI — costs nothing.
    """

    name = "Ollama (local)"
    description = (
        "Runs an open-weight model locally via an Ollama server — free, private, "
        "no API key, zero cost. You choose which model to pull (llama3.2, mistral, "
        "...). The out-of-the-box default; the price is your own CPU/GPU time and "
        "local-model quality vs. the hosted providers."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        options: dict | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = options or {}

    def _get_available_models(self) -> list[str]:
        """Fetch the list of models currently downloaded on the local Ollama server.

        Returns an empty list if the server is unreachable or doesn't respond.
        """
        try:
            import httpx
        except ImportError:
            return []

        try:
            response = httpx.get(
                f"{self.host}/api/tags",
                timeout=5.0,
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models if m.get("name")]
        except Exception:
            pass
        return []

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - httpx is a core dependency
            raise ImportError(
                "The 'httpx' package is required for the Ollama provider. "
                "Install it with: pip install httpx"
            ) from e

        # Per-call overrides win over the instance defaults, so the UI can tweak
        # a single generation without rebuilding the provider.
        model = kwargs.get("model", self.model)
        options = {**self.options, **kwargs.get("options", {})}

        payload = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options

        url = f"{self.host}/api/generate"
        start = time.perf_counter()
        try:
            response = httpx.post(url, json=payload, timeout=kwargs.get("timeout", 120.0))
        except httpx.HTTPError as e:
            # Fail loudly and helpfully — no silent fallback to another provider.
            raise RuntimeError(
                f"Could not reach the Ollama server at {self.host}. Is it running? "
                f"Start it with 'ollama serve' and pull the model with "
                f"'ollama pull {model}'. Original error: {e}"
            ) from e
        latency_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code != 200:
            # Try to fetch the list of available models to give a better error message
            available_models = self._get_available_models()
            available_list = (
                "\n  Available models: " + ", ".join(available_models)
                if available_models
                else "\n  No models found. Download one with: ollama pull <model>"
            )
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} for model '{model}'. "
                f"Model may not be downloaded locally.{available_list}"
            )

        data = response.json()
        text = data.get("response", "")
        if not text and data.get("error"):
            raise RuntimeError(f"Ollama error for model '{model}': {data['error']}")

        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            cost_usd=0.0,  # local inference is genuinely free — see class docstring
        )
