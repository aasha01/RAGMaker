"""Scoring endpoints — optional RAGAS-style quality scores for the compare grid.

Two routes:
* `GET  /score/status` — can scoring run right now? (i.e. is `ragas` installed).
  The frontend calls this to decide whether to offer the "Score" button or show
  the friendly install note.
* `POST /score` — score a batch of grid cells and return per-cell scores plus a
  ranking summary sorted by a chosen metric.

Scoring is **entirely optional and feature-gated**. If `ragas` isn't installed
the POST returns `available=False` + a friendly note (HTTP 200) rather than
failing — the grid simply renders without scores. A real scoring failure (judge
LLM/API key not configured) surfaces loudly as 502 with the message, never a
silent fallback to a fake number.

The scorer is provided via a FastAPI dependency (`get_scorer`) so tests can
inject a mock through `app.dependency_overrides` and exercise the whole path
without ragas or an API key.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.core.evaluator import (
    SCORE_METRICS,
    BaseScorer,
    RagasNotInstalled,
    ScoreSample,
    create_ragas_scorer,
    rank_cells,
    with_mean,
)
from ..schemas import (
    RankRow,
    ScoredCell,
    ScoreRequest,
    ScoreResponse,
    ScoreStatus,
)

router = APIRouter(prefix="/score", tags=["scoring"])

#: Metrics the ranking summary may be sorted by: the four quality metrics plus
#: their derived average.
_SORTABLE = (*SCORE_METRICS, "mean")


def get_scorer() -> BaseScorer:
    """The scorer the endpoints use. Overridden in tests with a mock so scoring
    can be exercised without ragas or a judge LLM.

    Uses create_ragas_scorer() to auto-detect a judge LLM (Ollama or Anthropic).
    If judge LLM init fails during scoring, it will be caught and reported as 502.
    """
    return create_ragas_scorer()


@router.get("/status", response_model=ScoreStatus)
def score_status(scorer: BaseScorer = Depends(get_scorer)) -> ScoreStatus:
    """Report whether optional scoring can run (ragas importable). Reads the
    availability flag only — never imports ragas or calls a judge LLM."""
    available = scorer.available()
    return ScoreStatus(
        available=available,
        message=None if available else scorer.unavailable_message(),
        metrics=list(SCORE_METRICS),
    )


@router.post("", response_model=ScoreResponse)
def score(
    req: ScoreRequest, scorer: BaseScorer = Depends(get_scorer)
) -> ScoreResponse:
    """Score every grid cell and return per-cell scores + a ranking summary.

    Errors are surfaced honestly: an empty request → 422; an unknown `sort_by` →
    400; ragas absent → 200 with `available=False` + the friendly note; any real
    scoring failure (judge LLM/API key) → 502 with the message.
    """
    if not req.cells:
        raise HTTPException(status_code=422, detail="No cells to score.")
    if req.sort_by not in _SORTABLE:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sort_by '{req.sort_by}'. Available: {list(_SORTABLE)}",
        )

    if not scorer.available():
        # Feature-gated off: not an error, just no scores. Friendly note travels
        # back so the UI can tell the learner exactly how to enable it.
        return ScoreResponse(
            available=False,
            message=scorer.unavailable_message(),
            metrics=list(SCORE_METRICS),
            sort_by=req.sort_by,
            cells=[],
            ranking=[],
        )

    samples = [
        ScoreSample(
            question=c.question,
            answer=c.answer,
            contexts=list(c.contexts),
            ground_truth=c.ground_truth,
        )
        for c in req.cells
    ]

    try:
        raw = scorer.score(samples)
    except RagasNotInstalled as exc:  # became unavailable between check and run
        return ScoreResponse(
            available=False,
            message=str(exc),
            metrics=list(SCORE_METRICS),
            sort_by=req.sort_by,
            cells=[],
            ranking=[],
        )
    except Exception as exc:  # judge LLM/API key/metric failure — fail loudly
        raise HTTPException(status_code=502, detail=f"Scoring failed: {exc}") from exc

    scored = [
        {
            "recipe_id": cell.recipe_id,
            "provider": cell.provider,
            "scores": with_mean(row),
        }
        for cell, row in zip(req.cells, raw)
    ]
    ranking = rank_cells(scored, req.sort_by)

    return ScoreResponse(
        available=True,
        message=None,
        metrics=list(SCORE_METRICS),
        sort_by=req.sort_by,
        cells=[ScoredCell(**c) for c in scored],
        ranking=[RankRow(**r) for r in ranking],
    )
