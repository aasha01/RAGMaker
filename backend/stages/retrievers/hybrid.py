"""Hybrid retrieval — combine BM25 (lexical) with vector (semantic) search."""

from __future__ import annotations

import re

from .base import BaseRetriever
from backend.stages.embedders.base import BaseEmbedder
from backend.stages.vectorstores.base import BaseVectorStore, SearchResult

_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens — the shared tokenizer for BM25's corpus and query
    so a learner can see exactly what the lexical side matches on."""
    return _WORD.findall(text.lower())


class HybridRetriever(BaseRetriever):
    """Blend keyword search (BM25) with vector search so a chunk can win by
    matching the exact words *or* by matching the meaning.

    What it does (mechanically): runs two searches over the same corpus — a BM25
    lexical ranking (classic keyword relevance: rewards rare query words that
    appear often in a chunk) and the usual vector similarity ranking — then fuses
    the two rankings into one. Two fusion methods are offered:
      * ``rrf`` (Reciprocal Rank Fusion, the default): score a chunk by
        ``sum over both lists of 1 / (rrf_k + rank)``. It uses only each chunk's
        *rank* in each list, so the two very different score scales never need to
        be reconciled — robust and the reason RRF is the common default.
      * ``weighted``: min-max normalise each side's scores to 0-1 and take
        ``alpha * vector + (1 - alpha) * bm25``. More directly tunable, but
        sensitive to score outliers.

    Tradeoff vs. the alternatives: pure vector search misses exact tokens it
    wasn't trained to care about — product codes, names, acronyms, negations —
    while pure BM25 misses paraphrases. Hybrid catches both, which is often the
    single biggest quality jump in a real system. The cost is a second index and
    a fusion knob to reason about.

    When a learner would prefer it: whenever queries contain specific terms that
    must match literally (an error code, a drug name, a policy number) alongside
    natural-language meaning — try a query with a rare exact keyword under naive
    top-k, then under hybrid, and watch the keyword-bearing chunk climb.

    Parameters (recorded per query):
        fetch_k: candidates to pull from each side before fusing (default 20).
        fusion: 'rrf' (default) or 'weighted'.
        rrf_k: RRF dampening constant (default 60); higher flattens rank
            influence. Only used when fusion='rrf'.
        alpha: weight on the vector side, 0-1 (default 0.5). Only used when
            fusion='weighted'.

    Uses the small `rank-bm25` package, imported lazily below.
    """

    name = "Hybrid (BM25 + vector)"
    description = (
        "Runs BM25 keyword search and vector search over the same chunks, then "
        "fuses the two rankings (Reciprocal Rank Fusion by default, or a weighted "
        "blend). Catches exact-term matches vector search misses and paraphrases "
        "BM25 misses. Needs the rank-bm25 package (no API key)."
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
        fusion: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        **_ignored,
    ) -> list[SearchResult]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "The 'rank-bm25' package is required for the hybrid retriever. "
                "Install it with: pip install rank-bm25"
            ) from e

        if fusion not in ("rrf", "weighted"):
            raise ValueError(f"fusion must be 'rrf' or 'weighted', got '{fusion}'.")
        if fusion == "weighted" and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
        if fetch_k < 1:
            raise ValueError(f"fetch_k must be >= 1, got {fetch_k}.")

        corpus = store.all_chunks()
        if not corpus:
            return []
        by_id = {c.chunk_id: c for c in corpus}

        # --- lexical side: BM25 over the full corpus ------------------------
        bm25 = BM25Okapi([_tokenize(c.text) for c in corpus])
        bm25_scores = bm25.get_scores(_tokenize(query))  # one score per corpus chunk
        bm25_ranked = sorted(
            range(len(corpus)), key=lambda i: bm25_scores[i], reverse=True
        )[:fetch_k]
        bm25_ids = [corpus[i].chunk_id for i in bm25_ranked]
        bm25_score_by_id = {corpus[i].chunk_id: float(bm25_scores[i]) for i in bm25_ranked}

        # --- semantic side: vector search -----------------------------------
        vector_hits = store.search(embedder.embed([query]), top_k=fetch_k)
        vector_ids = [h.chunk.chunk_id for h in vector_hits]
        vector_score_by_id = {h.chunk.chunk_id: float(h.score) for h in vector_hits}

        candidate_ids = list(dict.fromkeys(vector_ids + bm25_ids))  # union, stable order

        if fusion == "rrf":
            fused = {cid: self._rrf(cid, vector_ids, bm25_ids, rrf_k) for cid in candidate_ids}
        else:
            fused = self._weighted(
                candidate_ids, vector_score_by_id, bm25_score_by_id, alpha
            )

        ranked = sorted(candidate_ids, key=lambda cid: fused[cid], reverse=True)[:top_k]
        return [SearchResult(chunk=by_id[cid], score=float(fused[cid])) for cid in ranked]

    @staticmethod
    def _rrf(cid: str, vector_ids: list[str], bm25_ids: list[str], rrf_k: int) -> float:
        """Reciprocal Rank Fusion contribution of chunk `cid`: sum of
        1/(rrf_k + rank) over each list it appears in (rank is 1-based)."""
        score = 0.0
        for ids in (vector_ids, bm25_ids):
            if cid in ids:
                score += 1.0 / (rrf_k + ids.index(cid) + 1)
        return score

    @staticmethod
    def _weighted(
        candidate_ids: list[str],
        vector_score_by_id: dict[str, float],
        bm25_score_by_id: dict[str, float],
        alpha: float,
    ) -> dict[str, float]:
        """alpha * min-max(vector) + (1 - alpha) * min-max(bm25). A chunk absent
        from one side contributes 0 on that side (after normalisation)."""

        def minmax(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            lo, hi = min(scores.values()), max(scores.values())
            if hi == lo:  # all equal (or a single item) -> treat as fully present
                return {cid: 1.0 for cid in scores}
            return {cid: (s - lo) / (hi - lo) for cid, s in scores.items()}

        vec_norm = minmax(vector_score_by_id)
        bm25_norm = minmax(bm25_score_by_id)
        return {
            cid: alpha * vec_norm.get(cid, 0.0) + (1 - alpha) * bm25_norm.get(cid, 0.0)
            for cid in candidate_ids
        }
