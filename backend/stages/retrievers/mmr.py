"""Maximal Marginal Relevance (MMR) retrieval — diversity-aware selection."""

from __future__ import annotations

import numpy as np

from .base import BaseRetriever
from backend.stages.embedders.base import BaseEmbedder
from backend.stages.vectorstores.base import BaseVectorStore, SearchResult


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return `matrix` with each row scaled to unit length (so a dot product is a
    cosine similarity). Zero rows are left as-is to avoid divide-by-zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class MMRRetriever(BaseRetriever):
    """Pick chunks that are relevant to the question *and* different from each
    other, instead of the top-k that may all say the same thing.

    What it does (mechanically): first pulls a larger pool of `fetch_k`
    candidates by plain vector similarity, then builds the final `top_k` one at a
    time. At each step it scores every remaining candidate as
    ``lambda_mult * (similarity to the question) - (1 - lambda_mult) * (highest
    similarity to anything already picked)`` and takes the best. So a passage
    that is relevant but nearly identical to one already chosen gets penalised,
    which spreads the results across the different things the document says.

    Tradeoff vs. the alternatives: naive top-k can return five paraphrases of the
    single best passage — great precision, terrible coverage. MMR trades a little
    raw relevance for breadth, which matters when the answer needs several
    distinct facts. The cost is a tunable knob (`lambda_mult`) that can hurt if
    set wrong, and re-embedding the candidate pool to measure their pairwise
    similarity.

    When a learner would prefer it: for broad or multi-part questions ("summarise
    the treatment plan", "what are the risks and the mitigations"), or whenever
    the naive results look repetitive. Slide `lambda_mult` toward 0 to feel
    diversity take over, toward 1 to collapse back to naive top-k.

    Parameters (recorded per query):
        fetch_k: size of the candidate pool to diversify within (default 20).
            Bigger = more room for diversity, more re-embedding work.
        lambda_mult: 0.0-1.0 relevance/diversity trade (default 0.5). 1.0 is
            pure relevance (== naive top-k); 0.0 is pure diversity.

    Score reported: each returned chunk's score is its cosine similarity to the
    question (the relevance term), so scores stay comparable to naive top-k —
    the *ordering*, not the number, is what MMR changes.

    No API key required: reuses the recipe's own embedder, nothing external.
    """

    name = "MMR (diversity-aware)"
    description = (
        "Maximal Marginal Relevance: fetch a candidate pool, then greedily pick "
        "chunks that are relevant to the question but different from each other, "
        "so the results aren't near-duplicates. Tune lambda_mult (1.0 = naive "
        "top-k, 0.0 = maximum diversity). No API key."
    )
    no_api_key = True

    def retrieve(
        self,
        query: str,
        *,
        store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **_ignored,
    ) -> list[SearchResult]:
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(f"lambda_mult must be between 0 and 1, got {lambda_mult}.")
        if fetch_k < 1:
            raise ValueError(f"fetch_k must be >= 1, got {fetch_k}.")

        # A pool at least as big as top_k; capped by the store's own size inside
        # search(). Diversifying needs candidates beyond the final top_k.
        pool_size = max(fetch_k, top_k)
        candidates = store.search(embedder.embed([query]), top_k=pool_size)
        if not candidates:
            return []

        # Re-embed the candidate texts with the SAME model so we can measure
        # candidate-to-candidate similarity, not just candidate-to-query. This is
        # the transparent, store-agnostic way to get the vectors back (the store
        # only hands out chunks + scores). It's redundant work, accepted here for
        # clarity over speed — a teaching-tool tradeoff.
        cand_vecs = _normalize_rows(embedder.embed([c.chunk.text for c in candidates]))
        query_vec = _normalize_rows(embedder.embed([query]))[0]
        sim_to_query = cand_vecs @ query_vec  # cosine similarity, shape (n,)

        n = len(candidates)
        k = min(top_k, n)
        selected: list[int] = []
        remaining = list(range(n))

        while len(selected) < k and remaining:
            if not selected:
                # Seed with the most relevant candidate.
                best = max(remaining, key=lambda i: sim_to_query[i])
            else:
                selected_vecs = cand_vecs[selected]  # (len(selected), dim)

                def mmr_score(i: int) -> float:
                    redundancy = float(np.max(cand_vecs[i] @ selected_vecs.T))
                    return lambda_mult * float(sim_to_query[i]) - (1 - lambda_mult) * redundancy

                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)

        # Report each pick's relevance-to-query as its score (comparable to naive).
        return [
            SearchResult(chunk=candidates[i].chunk, score=float(sim_to_query[i]))
            for i in selected
        ]
