"""Local embedder backed by an Ollama server — no API key, zero cost."""

from __future__ import annotations

import numpy as np

from .base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Turn text into vectors locally via an Ollama server's embeddings endpoint.

    What it does (mechanically): sends each chunk to an Ollama server on your own
    machine (``http://localhost:11434`` by default) over plain HTTP, one request
    per text against ``/api/embeddings``, and reads back the vector. The model
    (e.g. ``nomic-embed-text:latest``) must already be pulled with
    ``ollama pull <model>``. The output dimension is whatever that model reports
    the first time this embedder is constructed — it is discovered by embedding a
    short probe string, never hard-coded, so the class works with any Ollama
    embedding model.

    Tradeoff vs. the alternatives: like the sentence-transformers embedder it's
    free, private, and needs no account, but the model runs as its own server
    process rather than in-process, and each chunk is one HTTP round-trip — slower
    for large chunk sets than a batched in-process encode. Embedding-focused
    Ollama models (e.g. nomic-embed-text) are often trained specifically for
    retrieval and can out-perform a small general sentence-transformers model,
    at the cost of needing Ollama installed and running.

    When a learner would prefer it: when you already run Ollama for local
    generation and want the whole pipeline (embedding + LLM) on one local
    service, or want to compare a retrieval-tuned local model (nomic-embed-text)
    against sentence-transformers' all-MiniLM-L6-v2 side by side.

    Parameters (recorded in config.json):
        model_name: the Ollama embedding model tag (default
            ``nomic-embed-text:latest``). Must be pulled first.
        host: base URL of the Ollama server. Default ``http://localhost:11434``.
        normalize: if True, vectors are unit-length (cosine == dot product in the
            vector store).
        truncate_dim: optional int < the model's native dimension; cuts each
            vector to its first ``truncate_dim`` values (then re-normalises if
            ``normalize``). None = use the model's full dimension.

    ``httpx`` (already a core dependency of the backend) is imported lazily so
    merely importing this module — e.g. to read its description in the UI —
    costs nothing. The dimension-discovery probe call happens in ``__init__``
    (mirroring how the sentence-transformers embedder loads its model at
    construction time), so a learner picking this embedder gets an immediate,
    clear connection error rather than one deferred to the first real embed call.
    """

    name = "Ollama (local)"
    description = (
        "Embeds text locally via an Ollama server — free, private, no API key. "
        "Default model nomic-embed-text:latest, a model trained specifically for "
        "retrieval. One HTTP call per chunk (slower than a batched in-process "
        "encode). Requires Ollama running with the model already pulled."
    )

    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
        host: str = "http://localhost:11434",
        normalize: bool = True,
        truncate_dim: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.normalize = normalize

        # Discover the model's native dimension with a one-off probe call, so
        # `dimension` is known (and load failures surface) before any chunk is
        # ever embedded — no silent fallback.
        probe = self._embed_one("probe")
        self.default_dimension = len(probe)

        if truncate_dim is not None:
            if not (0 < truncate_dim <= self.default_dimension):
                raise ValueError(
                    f"truncate_dim must be between 1 and the model's native "
                    f"dimension ({self.default_dimension}); got {truncate_dim}."
                )
        self.truncate_dim = truncate_dim
        self.dimension = truncate_dim if truncate_dim is not None else self.default_dimension

    def _embed_one(self, text: str) -> list[float]:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - httpx is a core dependency
            raise ImportError(
                "The 'httpx' package is required for the Ollama embedder. "
                "Install it with: pip install httpx"
            ) from e

        url = f"{self.host}/api/embeddings"
        try:
            response = httpx.post(
                url, json={"model": self.model_name, "prompt": text}, timeout=30.0
            )
        except httpx.HTTPError as e:
            # Fail loudly and helpfully — no silent fallback to another embedder.
            raise RuntimeError(
                f"Could not reach the Ollama server at {self.host}. Is it running? "
                f"Start it with 'ollama serve' and pull the model with "
                f"'ollama pull {self.model_name}'. Original error: {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} for embedding model "
                f"'{self.model_name}'. It may not be downloaded locally — pull it "
                f"with 'ollama pull {self.model_name}'. Response: {response.text[:300]}"
            )

        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(
                f"Ollama returned no embedding for model '{self.model_name}': {data}"
            )
        return embedding

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(
            [self._embed_one(text) for text in texts], dtype=np.float32
        )

        if vectors.ndim != 2 or vectors.shape[1] != self.default_dimension:
            # Fail loudly rather than silently reshaping/padding (Non-Negotiable).
            raise RuntimeError(
                f"Embedder returned shape {vectors.shape}, expected "
                f"(n, {self.default_dimension}) for model '{self.model_name}'."
            )

        if self.truncate_dim is not None:
            vectors = np.ascontiguousarray(vectors[:, : self.truncate_dim])

        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = (vectors / norms).astype(np.float32)

        return vectors

    def model_info(self) -> dict:
        """User-facing details about the connected Ollama embedding model."""
        notes = None
        if self.truncate_dim is not None:
            notes = (
                f"Output truncated from {self.default_dimension} to "
                f"{self.truncate_dim} dims via truncate_dim."
            )
        return {
            "model_name": self.model_name,
            "backend": f"ollama ({self.host})",
            "default_dimension": self.default_dimension,
            "output_dimension": self.dimension,
            "dimension_customizable": True,
            "max_seq_length_tokens": None,
            "param_count": None,
            "approx_size_mb": None,
            "normalize": self.normalize,
            "truncate_dim": self.truncate_dim,
            "notes": notes,
        }
