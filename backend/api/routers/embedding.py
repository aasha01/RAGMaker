"""Embedding endpoint (stage 3): POST /embed.

Turns the chunks from stage 2 into vectors. Stateless like the earlier stages
(the frontend passes the chunks in and carries the vectors back out), but with
one important optimisation: embedder instances are cached in-process so the
(heavy) model is loaded once per (model, normalize) combination rather than on
every request.

The response carries the full vectors plus the `EmbeddingMeta` identity
(model + dimension) that must travel with the vectors and be re-checked before
any query — the "never mix two models' embeddings" Non-Negotiable.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.stages.embedders import REGISTRY as EMBEDDERS
from ..schemas import EmbeddingMeta, EmbedRequest, EmbedResponse, ModelInfo
from ..services import get_embedder

router = APIRouter(tags=["pipeline"])


@router.get("/embedders/{key}/model_info", response_model=ModelInfo)
def embedder_model_info(
    key: str,
    model_name: str = Query("all-MiniLM-L6-v2"),
    normalize: bool = Query(True),
    truncate_dim: int | None = Query(None, description="Custom output dimension"),
) -> ModelInfo:
    """Model details (dimension, context window, params, size) for the UI.

    Loads the model (cached) to report accurate facts. Bad params — e.g. a
    truncate_dim larger than the model's native dimension — fail loudly (422).
    """
    params: dict = {"model_name": model_name, "normalize": normalize}
    if truncate_dim is not None:
        params["truncate_dim"] = truncate_dim
    embedder = get_embedder(key, params)  # raises 400 unknown / 422 load-fail
    return ModelInfo(**embedder.model_info())


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if req.embedder not in EMBEDDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedder '{req.embedder}'. Available: {sorted(EMBEDDERS)}",
        )
    if not req.chunks:
        raise HTTPException(status_code=422, detail="No chunks to embed.")

    embedder = get_embedder(req.embedder, req.params)
    texts = [c.text for c in req.chunks]
    chunk_ids = [c.chunk_id for c in req.chunks]

    t0 = time.perf_counter()
    try:
        vectors = embedder.embed(texts)
    except Exception as exc:  # no silent fallback — report the real failure
        raise HTTPException(
            status_code=422,
            detail=f"Embedding failed with '{req.embedder}': {exc}",
        ) from exc
    elapsed = time.perf_counter() - t0

    norms = np.linalg.norm(vectors, axis=1)
    meta = EmbeddingMeta(
        model_name=embedder.model_name,
        dimension=embedder.dimension,
        normalize=embedder.normalize,
        chunk_id_order=chunk_ids,
    )
    return EmbedResponse(
        embedder=req.embedder,
        model_name=embedder.model_name,
        dimension=embedder.dimension,
        normalize=embedder.normalize,
        count=len(texts),
        embed_time_sec=round(elapsed, 3),
        cost_usd=0.0,  # local model — free; paid embedders will populate this
        vectors=vectors.tolist(),
        value_preview=[round(float(v), 5) for v in vectors[0][:12]],
        norms_preview=[round(float(n), 4) for n in norms[:8]],
        model_info=ModelInfo(**embedder.model_info()),
        meta=meta,
    )
