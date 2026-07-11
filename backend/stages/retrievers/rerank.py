"""Cross-encoder re-ranking — vector top-N, then a heavier model re-scores."""

from __future__ import annotations

from .base import BaseRetriever
from backend.stages.embedders.base import BaseEmbedder
from backend.stages.vectorstores.base import BaseVectorStore, SearchResult

# Cross-encoder models are heavy to load, so cache them by name across queries
# (like the embedder cache). Keyed by model_name; module-level so instances made
# per call still share one loaded model.
_MODEL_CACHE: dict[str, object] = {}


class RerankRetriever(BaseRetriever):
    """Take a wide net of vector candidates, then re-score each one with a slower
    but sharper cross-encoder and keep the best `top_k`.

    What it does (mechanically): first pulls `fetch_n` candidates by ordinary
    vector similarity (fast, approximate). Then a *cross-encoder* reads the
    question and each candidate chunk **together** as one input and outputs a
    direct relevance score for that pair. The embedder used everywhere else is a
    *bi-*encoder — it turns the query and each chunk into vectors *separately* and
    compares them, which is cheap but blurry. A cross-encoder never compresses
    them into vectors, so it can judge relevance far more precisely; the price is
    that it must run once per candidate, so you only apply it to a shortlist.

    Tradeoff vs. the alternatives: this is usually the biggest single quality win
    for the final ordering — the top result is much more often the truly best
    passage. The cost is a second model to download and real compute per query
    (proportional to `fetch_n`), and it only reorders what vector search already
    surfaced: if the right chunk isn't in the top `fetch_n`, re-ranking can't
    rescue it (that's what hybrid is for).

    When a learner would prefer it: when the *order* of results matters and the
    naive top-1 is often not quite the best of the top-5. Raise `fetch_n` to give
    the re-ranker more to work with; watch the ordering tighten.

    Parameters (recorded per query):
        fetch_n: how many vector candidates to re-rank (default 20). Higher =
            better recall into the re-ranker, more compute per query.
        model_name: the cross-encoder model
            (default 'cross-encoder/ms-marco-MiniLM-L-6-v2', small and fast).

    Score reported: the cross-encoder's own relevance score for each kept pair
    (higher = more relevant); note this is a *different scale* from the vector
    metric, so compare rerank scores to each other, not to naive top-k's.

    No API key required, but it downloads a cross-encoder model on first use and
    uses the (already-installed) sentence-transformers library, imported lazily.
    """

    name = "Cross-encoder re-rank"
    description = (
        "Vector search fetches a shortlist (fetch_n), then a cross-encoder reads "
        "the question and each candidate together and re-scores them for a much "
        "sharper final ordering. Only reorders what vector search found. No API "
        "key, but downloads a small cross-encoder model on first use."
    )
    no_api_key = True

    def retrieve(
        self,
        query: str,
        *,
        store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int,
        fetch_n: int = 20,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        **_ignored,
    ) -> list[SearchResult]:
        if fetch_n < 1:
            raise ValueError(f"fetch_n must be >= 1, got {fetch_n}.")

        candidates = store.search(embedder.embed([query]), top_k=max(fetch_n, top_k))
        if not candidates:
            return []

        cross_encoder = self._get_model(model_name)
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = cross_encoder.predict(pairs)  # one relevance score per pair

        reranked = sorted(
            zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True
        )[:top_k]
        return [
            SearchResult(chunk=c.chunk, score=float(score)) for c, score in reranked
        ]

    @staticmethod
    def _get_model(model_name: str):
        if model_name not in _MODEL_CACHE:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "The 'sentence-transformers' package is required for the "
                    "cross-encoder re-ranker. Install it with: "
                    "pip install sentence-transformers"
                ) from e
            _MODEL_CACHE[model_name] = CrossEncoder(model_name)
        return _MODEL_CACHE[model_name]
