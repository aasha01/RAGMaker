"""Vector store stage — interface and data contract.

A vector store ingests the embedding matrix + the chunks it corresponds to,
persists itself to disk, reloads, and answers nearest-neighbour queries. It is
the one stage that is written once at build time and reopened read-only at
query time, so `save`/`load` are part of the contract, not an afterthought.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from backend.stages.chunkers.base import Chunk


@dataclass
class SearchResult:
    """One hit from a similarity search.

    Attributes:
        chunk: The retrieved chunk (full object, so the UI can show it in the
            context of the original document).
        score: Similarity/score for this hit. Higher = more similar; each store
            documents its metric so scores are interpretable.
    """

    chunk: Chunk
    score: float


class BaseVectorStore(ABC):
    """Abstract vector store. Every concrete store is a swappable Strategy.

    Build-time parameters (index type, distance metric, ...) are passed to
    `build` and recorded in `config.json`. A store must never be searched with
    an embedding produced by a different model than the one used to build it;
    concrete stores enforce this using the model/dimension recorded alongside
    the vectors.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def build(self, embeddings: np.ndarray, chunks: list[Chunk], **params) -> None:
        """Index `embeddings` (row i corresponds to `chunks[i]`).

        `embeddings.shape[0]` must equal `len(chunks)`; implementations should
        assert this rather than silently indexing a mismatched subset.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index (and its chunk mapping) under `path`."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Reload a previously saved index from `path` in place."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Return the `top_k` nearest chunks to `query_embedding`, best first."""
        raise NotImplementedError

    def all_chunks(self) -> list[Chunk]:
        """Return every chunk in the store, in index order.

        Vector search only returns chunks a query already matches; a lexical
        (BM25) or hybrid retriever needs the *whole* corpus to find matches the
        vector side missed, so this exposes it uniformly. The default reads the
        `chunks` list concrete stores keep after `build`/`load`; a store that
        holds its corpus differently should override this.
        """
        return list(getattr(self, "chunks", []))
