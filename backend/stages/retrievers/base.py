"""Retrieval stage — interface and data contract (query-time, not build-time).

Retrieval is the one stage recorded **per query, not per recipe** (SPEC.md §6):
the same immutable recipe (its chunks, embeddings, and vector index) can be
queried with several retrieval strategies without rebuilding anything. That is
what makes retrieval the highest-leverage knob a learner can turn and *feel* the
effect of — swap `naive_topk` for `mmr` or `hybrid` on the same store and watch
the retrieved chunks change.

A retriever therefore does not own any data. It is handed, at call time:
  * the already-built `store` (a `BaseVectorStore`, searchable + its full corpus
    via `store.all_chunks()` for lexical strategies), and
  * the recipe's `embedder` (the SAME model the store was built with — the caller
    guarantees this match before calling, upholding the "never search a store
    with another model's vectors" Non-Negotiable),
and returns a ranked `list[SearchResult]`, best first — the exact same type the
store's own `search` returns, so everything downstream (prompt assembly, the
compare grid) is unchanged regardless of which retriever ran.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.stages.embedders.base import BaseEmbedder
from backend.stages.vectorstores.base import BaseVectorStore, SearchResult


class BaseRetriever(ABC):
    """Abstract retriever. Every concrete retriever is a swappable Strategy.

    All tunable parameters (candidate pool size, diversity weight, fusion method,
    a HyDE generation provider, ...) are passed to `retrieve` as keyword
    arguments so they can be read straight out of the per-query request and are
    visible/overridable in the UI — never hidden defaults (CLAUDE.md
    Non-Negotiables).

    `no_api_key` is a class-level hint for the UI: True means the strategy runs
    on the local default stack with no external account (naive_topk, mmr). It is
    documentation only — a False value never changes behaviour, it just lets the
    UI flag that a strategy may need a running LLM/model.
    """

    name: str = ""
    description: str = ""
    #: True if the strategy needs no API key / external service on the default
    #: stack. Surfaced in the UI so a learner knows which options are free.
    no_api_key: bool = False

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int,
        **params,
    ) -> list[SearchResult]:
        """Return the `top_k` best chunks for `query`, best first.

        `store` is already built and its `embedder` is the model it was built
        with (the caller has validated this). Implementations embed the query
        (or a transformed query) with `embedder` and/or use lexical signals over
        `store.all_chunks()`, then rank. Optional heavy/optional dependencies
        (rank-bm25, a cross-encoder, an LLM SDK) MUST be imported lazily inside
        this method and fail loudly with a friendly message if absent — never a
        silent fallback to a different strategy.
        """
        raise NotImplementedError


__all__ = ["BaseRetriever", "SearchResult"]
