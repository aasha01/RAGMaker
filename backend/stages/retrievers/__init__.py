"""Retriever strategy registry (query-time stage, SPEC.md §6).

Maps the per-query key -> concrete `BaseRetriever` subclass. Adding a retriever
is a new file in this package plus one line here — nothing else changes.

Retrieval is chosen *per query*, not stored in the recipe, so one immutable
recipe can be probed with every strategy below. `naive_topk` is the key-free
default; the others lazy-import their optional dependency (rank-bm25, a
cross-encoder, an LLM SDK) inside `retrieve`, so importing this registry (e.g.
for the discovery endpoint) drags in none of them.
"""

from __future__ import annotations

from .base import BaseRetriever, SearchResult
from .naive_topk import NaiveTopKRetriever
from .mmr import MMRRetriever
from .hybrid import HybridRetriever
from .rerank import RerankRetriever
from .hyde import HyDERetriever

REGISTRY: dict[str, type[BaseRetriever]] = {
    "naive_topk": NaiveTopKRetriever,  # plain vector top-k — the key-free default
    "mmr": MMRRetriever,               # diversity-aware
    "hybrid": HybridRetriever,         # BM25 + vector, fused
    "rerank": RerankRetriever,         # vector top-N -> cross-encoder -> top-k
    "hyde": HyDERetriever,             # LLM hypothetical answer -> embed -> search
}

__all__ = ["BaseRetriever", "SearchResult", "REGISTRY"]
