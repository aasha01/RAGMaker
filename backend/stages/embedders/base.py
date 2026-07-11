"""Embedding stage — interface and data contract.

An embedder maps a list of texts to a matrix of vectors of shape (n, dimension).
The `dimension` and the model identity are first-class: they are recorded next
to the vectors (`04_embeddings_meta.json`) and re-checked at query time so a
store built with one model can never be searched with a vector from another
(CLAUDE.md Non-Negotiables: "Never mix embeddings from two different models").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract embedder. Every concrete embedder is a swappable Strategy.

    Concrete embedders are configured at construction (model name, normalize
    flag, ...) so that `dimension` and `model_name` are known and can be
    validated before any vector is produced.
    """

    name: str = ""
    description: str = ""
    #: Output vector length. Concrete embedders set this from their model.
    dimension: int = 0
    #: The exact model identifier, recorded in meta and checked at query time.
    model_name: str = ""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed `texts`; return a float array of shape (len(texts), dimension).

        Implementations must guarantee the returned array's second axis equals
        `self.dimension`, and must fail loudly (not silently truncate/pad) if a
        backend returns an unexpected shape.
        """
        raise NotImplementedError

    def model_info(self) -> dict:
        """User-facing details about the underlying model, for the UI.

        Concrete embedders override this to surface the model's default
        dimension, context window (max tokens), parameter count, size, and
        which parameters are being sent — teaching content shown next to the
        embedding stage. The base version returns only what every embedder
        already knows; keys absent for a given backend should simply be omitted
        or None rather than faked.
        """
        return {
            "model_name": self.model_name,
            "output_dimension": self.dimension,
            "normalize": getattr(self, "normalize", None),
        }
