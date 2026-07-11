"""Embedder strategy registry.

Maps the config key -> concrete `BaseEmbedder` subclass. Populated in a later
build step — empty here so Step 1 is interfaces only.
"""

from __future__ import annotations

from .base import BaseEmbedder
from .sentence_transformer import SentenceTransformerEmbedder

REGISTRY: dict[str, type[BaseEmbedder]] = {
    "sentence_transformers": SentenceTransformerEmbedder,
}

__all__ = ["BaseEmbedder", "REGISTRY"]
