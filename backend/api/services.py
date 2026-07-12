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

    `params` is passed straight through as constructor kwargs (minus any None
    values), so each embedder's own `__init__` defaults apply when a param is
    omitted — this must stay generic across embedders (e.g. sentence-transformers'
    `model_name` default differs from Ollama's), not hard-code one backend's
    defaults here (Strategy Pattern: nothing outside `stages/embedders/` should
    need editing to add a new embedder).

    Raises HTTP 400 for an unknown key and HTTP 422 if the model fails to load
    (missing package, bad model name, invalid truncate_dim, unreachable server)
    — never a silent fallback to a different model.
    """
    if key not in EMBEDDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedder '{key}'. Available: {sorted(EMBEDDERS)}",
        )
    ctor_kwargs = {k: v for k, v in params.items() if v is not None}
    cache_key = (key, tuple(sorted(ctor_kwargs.items())))
    if cache_key not in _EMBEDDER_CACHE:
        try:
            _EMBEDDER_CACHE[cache_key] = EMBEDDERS[key](**ctor_kwargs)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to load embedder '{key}' ({params.get('model_name', 'default model')}): {exc}",
            ) from exc
    return _EMBEDDER_CACHE[cache_key]
