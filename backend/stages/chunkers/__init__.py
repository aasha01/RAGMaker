"""Chunker strategy registry.

Maps the config key -> concrete `BaseChunker` subclass. Populated in a later
build step — empty here so Step 1 is interfaces only.
"""

from __future__ import annotations

from .base import BaseChunker, Chunk
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sentence import SentenceChunker
from .structure_aware import StructureAwareChunker

REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "sentence": SentenceChunker,
    "structure_aware": StructureAwareChunker,
}

__all__ = ["BaseChunker", "Chunk", "REGISTRY"]
