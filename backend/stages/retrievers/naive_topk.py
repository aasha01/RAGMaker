"""Naive top-k retrieval — plain vector similarity, the baseline."""

from __future__ import annotations

from .base import BaseRetriever
from backend.stages.embedders.base import BaseEmbedder
from backend.stages.vectorstores.base import BaseVectorStore, SearchResult


class NaiveTopKRetriever(BaseRetriever):
    """Embed the question, return the `top_k` nearest chunks — nothing else.

    What it does (mechanically): turns the question into a vector with the same
    model the store was built with, then asks the vector store for the `top_k`
    chunks whose vectors are closest to it. This is exactly the plain search that
    was previously baked straight into the store — pulled out here so it sits
    beside the fancier strategies as the honest baseline to compare them against.

    Tradeoff vs. the alternatives: it is the simplest and fastest, needs no extra
    model, no API key, and no second index — but it has no notion of *diversity*
    (the top-k can be near-duplicates of one passage), no *lexical* matching (an
    exact keyword or rare code the embedder glosses over can be missed), and no
    second-pass *re-ranking*. Every other retriever here is an answer to one of
    those gaps.

    When a learner would prefer it: as the default and the control. Always run
    your first query with naive top-k, then switch strategies on the *same*
    recipe and watch what changes — that difference is the whole lesson.

    Parameters: none beyond `top_k`. No hidden knobs.
    """

    name = "Naive top-k (vector similarity)"
    description = (
        "Plain vector search: embed the question and return the top_k nearest "
        "chunks. The simplest, fastest baseline — no diversity, no keywords, no "
        "re-ranking. Needs no API key. Use it as the control to compare the "
        "other strategies against."
    )
    no_api_key = True

    def retrieve(
        self,
        query: str,
        *,
        store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int,
        **_ignored,
    ) -> list[SearchResult]:
        query_vec = embedder.embed([query])
        return store.search(query_vec, top_k=top_k)
