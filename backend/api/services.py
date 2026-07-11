"""Shared, stateful API services (kept out of the routers so several routers can
share one instance). Currently: the in-process embedder cache.

The cache is used by ``/embed`` (stage 3) and by the vector store's query-time
re-embedding (stage 4) — both must load a given model once, and stage 4 must
re-embed queries with the *exact same* embedder config that built the store.
"""

from __future__ import annotations

from fastapi import HTTPException

from backend.stages.embedders import REGISTRY as EMBEDDERS
from backend.stages.embedders.base import BaseEmbedder

# Keyed by every parameter that changes the produced vectors.
_EMBEDDER_CACHE: dict[tuple, BaseEmbedder] = {}


def get_embedder(key: str, params: dict) -> BaseEmbedder:
    """Return a cached embedder for (key, params), loading it once.

    Raises HTTP 400 for an unknown key and HTTP 422 if the model fails to load
    (missing package, bad model name, invalid truncate_dim) — never a silent
    fallback to a different model.
    """
    if key not in EMBEDDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedder '{key}'. Available: {sorted(EMBEDDERS)}",
        )
    model_name = params.get("model_name", "all-MiniLM-L6-v2")
    normalize = bool(params.get("normalize", True))
    truncate_dim = params.get("truncate_dim")
    cache_key = (key, model_name, normalize, truncate_dim)
    if cache_key not in _EMBEDDER_CACHE:
        try:
            _EMBEDDER_CACHE[cache_key] = EMBEDDERS[key](
                model_name=model_name, normalize=normalize, truncate_dim=truncate_dim
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to load embedder '{key}' ({model_name}): {exc}",
            ) from exc
    return _EMBEDDER_CACHE[cache_key]
