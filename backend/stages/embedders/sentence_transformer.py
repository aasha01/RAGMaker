"""Local embedder backed by the sentence-transformers library."""

from __future__ import annotations

import numpy as np

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Turn text into vectors locally with a sentence-transformers model.

    What it does (mechanically): loads a small transformer model onto your own
    machine and runs each chunk through it to produce a fixed-length vector
    (384 numbers for the default ``all-MiniLM-L6-v2``). Similar meanings land at
    nearby points in that 384-dimensional space, which is what lets the vector
    store find relevant chunks later. Nothing leaves your computer.

    Tradeoff vs. the alternatives: the OpenAI and Cohere embedders often produce
    higher-quality vectors, especially for longer or more specialised text — but
    they cost money per call, need an API key, and send your text to a third
    party. This local model is free, private, and needs no account; the price is
    somewhat lower retrieval quality and using your own CPU/GPU time.

    When a learner would prefer it: as the default first recipe — you can build
    an end-to-end RAG pipeline with zero external accounts and see the whole
    thing work, then swap in a hosted embedder later to feel the quality
    difference for yourself.

    Two model facts worth understanding, both surfaced by ``model_info``:
      * Context window (``max_seq_length``): the model only reads this many
        tokens of each chunk (256 for all-MiniLM-L6-v2). Anything past that is
        silently ignored by the model — so a chunk far bigger than the context
        window is partly wasted. This is why chunk size and the model's context
        window need to be considered together.
      * Output dimension: fixed at 384 for MiniLM, but you *can* force a smaller
        output with ``truncate_dim`` (keep only the first N numbers, then
        re-normalise). Smaller vectors are cheaper to store/search. WARNING:
        MiniLM is not a "Matryoshka" model, so truncating it genuinely loses
        quality — unlike models trained for it (e.g. OpenAI text-embedding-3),
        where shrinking dimensions is nearly free. It is exposed here so you can
        measure that quality drop yourself.

    Parameters (recorded in config.json):
        model_name: the sentence-transformers model id (default all-MiniLM-L6-v2).
        normalize: if True, vectors are unit-length, which makes cosine
            similarity and dot-product equivalent in the vector store.
        truncate_dim: optional int < the model's native dimension. If set, each
            vector is cut to its first ``truncate_dim`` values (then re-normalised
            if ``normalize``). None = use the model's full dimension.

    The heavy library is imported lazily in ``__init__`` so importing this
    module (e.g. to read its description in the UI) doesn't drag in torch.
    """

    name = "Sentence-Transformers (local)"
    description = (
        "Runs a small transformer model on your own machine to embed text — "
        "free, private, no API key. Default model all-MiniLM-L6-v2 (384 dims, "
        "256-token context). Output dimension is customizable via truncate_dim "
        "(lossy for non-Matryoshka models like MiniLM). Best as your first recipe."
    )

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        truncate_dim: int | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'sentence-transformers' package is required for the local "
                "embedder. Install it with: pip install sentence-transformers"
            ) from e

        self.model_name = model_name
        self.normalize = normalize
        self._model = SentenceTransformer(model_name)

        # Native (full) output dimension of the model. The accessor was renamed
        # across library versions; support both (API-compat shim, not a silent
        # strategy fallback).
        if hasattr(self._model, "get_embedding_dimension"):
            self.default_dimension = int(self._model.get_embedding_dimension())
        else:
            self.default_dimension = int(self._model.get_sentence_embedding_dimension())

        # Optional dimension truncation — validated loudly, no silent clamping.
        if truncate_dim is not None:
            if not (0 < truncate_dim <= self.default_dimension):
                raise ValueError(
                    f"truncate_dim must be between 1 and the model's native "
                    f"dimension ({self.default_dimension}); got {truncate_dim}."
                )
        self.truncate_dim = truncate_dim
        #: The dimension actually produced (== truncate_dim when set).
        self.dimension = truncate_dim if truncate_dim is not None else self.default_dimension

        # Context window and model size — for the UI's model-details view.
        self.max_seq_length = getattr(self._model, "max_seq_length", None)
        try:
            self.param_count = int(sum(p.numel() for p in self._model.parameters()))
        except Exception:  # pragma: no cover - defensive
            self.param_count = None
        # Rough size estimate assuming float32 weights; clearly labelled "approx".
        self.approx_size_mb = (
            round(self.param_count * 4 / (1024 ** 2), 1) if self.param_count else None
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        # Encode without library normalization so truncation happens on the raw
        # vector and we re-normalise afterwards — the transparent order for
        # Matryoshka-style truncation.
        vectors = self._model.encode(
            texts,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

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
            norms[norms == 0] = 1.0  # avoid divide-by-zero on empty text
            vectors = (vectors / norms).astype(np.float32)

        return vectors

    def model_info(self) -> dict:
        """Rich, user-facing details about the loaded model (see base method)."""
        notes = None
        if self.truncate_dim is not None:
            notes = (
                f"Output truncated from {self.default_dimension} to "
                f"{self.truncate_dim} dims. MiniLM is not a Matryoshka model, so "
                f"this loses quality — exposed for learning/comparison."
            )
        return {
            "model_name": self.model_name,
            "backend": "sentence-transformers",
            "default_dimension": self.default_dimension,
            "output_dimension": self.dimension,
            "dimension_customizable": True,
            "max_seq_length_tokens": self.max_seq_length,
            "param_count": self.param_count,
            "approx_size_mb": self.approx_size_mb,
            "normalize": self.normalize,
            "truncate_dim": self.truncate_dim,
            "notes": notes,
        }
