"""Vector store strategy registry.

Maps the config key -> concrete `BaseVectorStore` subclass. Populated in a
later build step — empty here so Step 1 is interfaces only.
"""

from __future__ import annotations

from .base import BaseVectorStore, SearchResult
from .faiss_store import FAISSStore

REGISTRY: dict[str, type[BaseVectorStore]] = {
    "faiss": FAISSStore,
}

__all__ = ["BaseVectorStore", "SearchResult", "REGISTRY"]
