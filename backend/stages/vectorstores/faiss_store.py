"""FAISS vector store — local similarity index, no server required."""

from __future__ import annotations

import json
import os

import numpy as np

from .base import BaseVectorStore, SearchResult
from backend.stages.chunkers.base import Chunk


class FAISSStore(BaseVectorStore):
    """Store and search vectors with FAISS, entirely on local disk.

    What it does (mechanically): builds an in-memory FAISS index over the chunk
    vectors and writes it to a file, so a query vector can be compared against
    every stored vector to find the nearest chunks. With ``index_type='flat'``
    it compares against every vector exactly (an exhaustive, exact search);
    with ``'hnsw'`` it builds a navigable graph that finds *approximate*
    neighbours much faster on large collections.

    Tradeoff vs. the alternatives: FAISS is a library, not a server — there is
    nothing to run or connect to, which makes it the friction-free default.
    Chroma and Qdrant are fuller "databases" that add metadata filtering,
    collections, and (for Qdrant) a running service; handy at scale but more
    moving parts. FAISS keeps the whole store as plain files you can inspect.

    When a learner would prefer it: for local experiments and every first
    recipe — you get exact search (with 'flat') and zero setup. Reach for
    'hnsw' when the collection grows large enough that exact search feels slow,
    to trade a little accuracy for speed.

    Parameters (recorded in config.json):
        index_type: 'flat' (exact) or 'hnsw' (approximate, faster at scale).
        metric: 'cosine', 'dot', or 'l2'.

    Score interpretation (kept transparent, not normalised away): for 'cosine'
    and 'dot' the score is an inner product where **higher = more similar**; for
    'l2' the score is a squared distance where **lower = more similar**.
    """

    name = "FAISS (local)"
    description = (
        "Local FAISS index saved as plain files — no server to run. 'flat' does "
        "exact search; 'hnsw' is approximate but fast at scale. Metric can be "
        "cosine, dot, or l2. The zero-setup default for local experiments."
    )

    _HNSW_M = 32  # graph connectivity for HNSW; recorded in saved meta

    def __init__(self) -> None:
        self.index = None
        self.chunks: list[Chunk] = []
        self.dimension: int = 0
        self.index_type: str = "flat"
        self.metric: str = "cosine"
        # Embedding identity, carried so the query side can enforce the
        # "never search with a different model's vectors" rule.
        self.model_name: str = ""
        self.normalize: bool = False

    def build(
        self,
        embeddings: np.ndarray,
        chunks: list[Chunk],
        index_type: str = "flat",
        metric: str = "cosine",
        model_name: str = "",
        normalize: bool = False,
        **_ignored,
    ) -> None:
        import faiss

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings/chunks length mismatch: {embeddings.shape[0]} "
                f"vectors vs {len(chunks)} chunks — refusing to index a "
                f"mismatched subset."
            )
        if index_type not in ("flat", "hnsw"):
            raise ValueError(f"index_type must be 'flat' or 'hnsw', got '{index_type}'")
        if metric not in ("cosine", "dot", "l2"):
            raise ValueError(f"metric must be 'cosine', 'dot', or 'l2', got '{metric}'")

        vectors = np.ascontiguousarray(embeddings.astype(np.float32))
        self.dimension = int(vectors.shape[1])
        self.chunks = list(chunks)
        self.index_type = index_type
        self.metric = metric
        self.model_name = model_name
        self.normalize = normalize

        # Cosine similarity = inner product on L2-normalised vectors.
        if metric == "cosine":
            vectors = self._normalized(vectors)

        faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        if index_type == "flat":
            if metric == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
        else:  # hnsw
            self.index = faiss.IndexHNSWFlat(self.dimension, self._HNSW_M, faiss_metric)

        self.index.add(vectors)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        import faiss

        if self.index is None:
            raise RuntimeError("Vector store is empty — build or load it first.")

        query = np.ascontiguousarray(query_embedding.astype(np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dimension:
            # Dimension mismatch is a hard error, never a silent re-embed.
            raise ValueError(
                f"Query vector has dimension {query.shape[1]} but this store "
                f"was built with dimension {self.dimension} (model "
                f"'{self.model_name}'). Refusing to search across dimensions."
            )
        if self.metric == "cosine":
            query = self._normalized(query)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self.chunks[int(idx)], score=float(score)))
        return results

    def save(self, path: str) -> None:
        import faiss

        if self.index is None:
            raise RuntimeError("Nothing to save — build the store first.")
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, "store_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "index_type": self.index_type,
                    "metric": self.metric,
                    "dimension": self.dimension,
                    "model_name": self.model_name,
                    "normalize": self.normalize,
                    "hnsw_M": self._HNSW_M if self.index_type == "hnsw" else None,
                    "count": len(self.chunks),
                },
                f,
                indent=2,
            )

    def load(self, path: str) -> None:
        import faiss

        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.json"), "r", encoding="utf-8") as f:
            self.chunks = [Chunk.from_dict(d) for d in json.load(f)]
        with open(os.path.join(path, "store_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.index_type = meta["index_type"]
        self.metric = meta["metric"]
        self.dimension = meta["dimension"]
        self.model_name = meta.get("model_name", "")
        self.normalize = meta.get("normalize", False)

    @staticmethod
    def _normalized(vectors: np.ndarray) -> np.ndarray:
        import faiss

        out = vectors.copy()
        faiss.normalize_L2(out)
        return out
