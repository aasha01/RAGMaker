"""Stage discovery endpoints — what techniques are available per stage.

The frontend calls these to populate its option pickers, using each strategy's
class-level `name`/`description` (the teaching text). Crucially, this reads the
class attributes *without instantiating* the strategy, so listing e.g.
embedders never triggers a model download.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.stages.parsers import REGISTRY as PARSERS
from backend.stages.chunkers import REGISTRY as CHUNKERS
from backend.stages.embedders import REGISTRY as EMBEDDERS
from backend.stages.vectorstores import REGISTRY as VECTORSTORES
from backend.stages.retrievers import REGISTRY as RETRIEVERS
from backend.stages.llm_providers import REGISTRY as LLM_PROVIDERS
from ..schemas import StrategyInfo

router = APIRouter(prefix="/stages", tags=["stages"])


def _list(registry: dict) -> list[StrategyInfo]:
    return [
        StrategyInfo(
            key=key,
            name=getattr(cls, "name", "") or key,
            description=getattr(cls, "description", ""),
        )
        for key, cls in registry.items()
    ]


@router.get("/parsers", response_model=list[StrategyInfo])
def list_parsers() -> list[StrategyInfo]:
    """Available parsing strategies (stage 1)."""
    return _list(PARSERS)


@router.get("/chunkers", response_model=list[StrategyInfo])
def list_chunkers() -> list[StrategyInfo]:
    """Available chunking strategies (stage 2)."""
    return _list(CHUNKERS)


@router.get("/embedders", response_model=list[StrategyInfo])
def list_embedders() -> list[StrategyInfo]:
    """Available embedding strategies (stage 3). Reads class attributes only —
    does not instantiate, so listing never triggers a model download."""
    return _list(EMBEDDERS)


@router.get("/vectorstores", response_model=list[StrategyInfo])
def list_vectorstores() -> list[StrategyInfo]:
    """Available vector store strategies (stage 4)."""
    return _list(VECTORSTORES)


@router.get("/retrievers", response_model=list[StrategyInfo])
def list_retrievers() -> list[StrategyInfo]:
    """Available query-time retrieval strategies (SPEC.md §6). Chosen per query,
    not stored in the recipe. Reads class attributes only — does not instantiate,
    so listing never imports rank-bm25, a cross-encoder, or an LLM SDK."""
    return _list(RETRIEVERS)


@router.get("/llm_providers", response_model=list[StrategyInfo])
def list_llm_providers() -> list[StrategyInfo]:
    """Available LLM providers for generation (Query & Compare). Reads class
    attributes only — does not instantiate, so listing never imports an SDK or
    needs an API key."""
    return _list(LLM_PROVIDERS)
