"""Recipe endpoints: build & persist a recipe, list them, and query a saved one.

A recipe is built **server-side from its config** (parse -> chunk -> embed ->
store), so `config.json` alone reproduces it. Saved recipes are durable (survive
restarts), which is what makes the Query & Compare features possible later.

Loaded stores are cached per recipe_id; a saved recipe is queried by re-embedding
the question with the SAME embedder config recorded in the recipe — enforcing the
"never search a store with another model's vectors" Non-Negotiable.
"""

from __future__ import annotations

import base64
import os
import tempfile

from fastapi import APIRouter, HTTPException

from backend.core.generation import generate_answer
from backend.core.recipe import (
    RECIPES_ROOT,
    build_recipe,
    list_recipes,
    load_recipe,
    open_recipe_store,
)
from backend.stages.retrievers import REGISTRY as RETRIEVERS
from ..schemas import (
    GenerateRequest,
    GenerateResponse,
    RecipeBuildRequest,
    RecipeDetail,
    RecipeSearchRequest,
    RecipeSummary,
    SearchHit,
    SearchResponse,
)
from ..services import get_embedder

router = APIRouter(prefix="/recipes", tags=["recipes"])

# recipe_id -> {"store", "embedder", "embed_params"} (lazily loaded from disk)
_RECIPE_STORES: dict[str, dict] = {}


def _retrieve(
    recipe_id: str,
    query: str,
    top_k: int,
    retriever: str = "naive_topk",
    retriever_params: dict | None = None,
):
    """Load (once, cached) the recipe's store and return its top-k hits for
    `query`, applying the chosen query-time retrieval strategy over the store —
    always re-embedding with the SAME embedder config the recipe was built with
    (the "never search a store with another model's vectors" guard).

    The retrieval strategy is chosen per query (SPEC.md §6), so the same recipe
    can be probed with several. The strategy runs over the store + the recipe's
    embedder; the model-match guard is enforced here, before the strategy, so a
    retriever never has to worry about it.

    Returns `(store, hits)`. Raises HTTPException — 404 unknown recipe, 400
    unknown retriever, 422 model mismatch / bad params, 502 a missing optional
    dependency or a failed HyDE generation — never a silent fallback.
    """
    if retriever not in RETRIEVERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown retriever '{retriever}'. Available: {sorted(RETRIEVERS)}",
        )

    if recipe_id not in _RECIPE_STORES:
        try:
            store, embedder_key, embed_params = open_recipe_store(recipe_id, RECIPES_ROOT)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _RECIPE_STORES[recipe_id] = {
            "store": store,
            "embedder": embedder_key,
            "embed_params": embed_params,
        }
    entry = _RECIPE_STORES[recipe_id]

    embedder = get_embedder(entry["embedder"], entry["embed_params"])
    store = entry["store"]
    if embedder.model_name != store.model_name:
        raise HTTPException(
            status_code=422,
            detail=f"Query embedder model '{embedder.model_name}' does not match "
            f"the recipe's store model '{store.model_name}'.",
        )

    try:
        hits = RETRIEVERS[retriever]().retrieve(
            query,
            store=store,
            embedder=embedder,
            top_k=top_k,
            **(retriever_params or {}),
        )
    except ValueError as exc:  # bad params / dimension guard
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # missing dep (rank-bm25/cross-encoder), HyDE LLM, ...
        raise HTTPException(
            status_code=502, detail=f"Retrieval failed with '{retriever}': {exc}"
        ) from exc
    return store, hits


@router.post("", response_model=RecipeSummary)
def create_recipe(req: RecipeBuildRequest) -> RecipeSummary:
    try:
        raw = base64.b64decode(req.source_b64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"source_b64 is not valid base64: {exc}")

    ext = os.path.splitext(req.source_filename)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
        tf.write(raw)
        tmp_path = tf.name
    try:
        row = build_recipe(
            tmp_path,
            req.source_filename,
            req.config,
            description=req.description,
            recipes_root=RECIPES_ROOT,
        )
    except Exception as exc:  # unknown strategy / stage failure — surface it
        raise HTTPException(status_code=422, detail=f"Recipe build failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return RecipeSummary(**row)


@router.get("", response_model=list[RecipeSummary])
def get_recipes() -> list[RecipeSummary]:
    return [RecipeSummary(**row) for row in list_recipes(RECIPES_ROOT)]


@router.get("/{recipe_id}", response_model=RecipeDetail)
def get_recipe(recipe_id: str) -> RecipeDetail:
    try:
        info = load_recipe(recipe_id, RECIPES_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RecipeDetail(config=info["config"], metadata=info["metadata"])


def _to_hits(hits) -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id=h.chunk.chunk_id,
            text=h.chunk.text,
            source=h.chunk.source,
            position=h.chunk.position,
            score=h.score,
        )
        for h in hits
    ]


@router.post("/{recipe_id}/search", response_model=SearchResponse)
def search_recipe(recipe_id: str, req: RecipeSearchRequest) -> SearchResponse:
    store, hits = _retrieve(
        recipe_id, req.query, req.top_k, req.retriever, req.retriever_params
    )
    return SearchResponse(
        store_id=recipe_id,
        query=req.query,
        metric=store.metric,
        model_name=store.model_name,
        retriever=req.retriever,
        hits=_to_hits(hits),
    )


@router.post("/{recipe_id}/generate", response_model=GenerateResponse)
def generate_recipe(recipe_id: str, req: GenerateRequest) -> GenerateResponse:
    """Retrieve top-k chunks, assemble a RAG prompt, and generate an answer with
    the chosen LLM provider — returning the answer, the exact prompt sent, the
    retrieved chunks, and the latency/token/cost stats (ARCHITECTURE.md §8).

    Errors surface loudly: unknown recipe → 404, unknown retriever/provider →
    400, model mismatch / bad retriever params → 422, and any retrieval or
    provider failure (missing package/API key, unreachable server) → 502 with the
    real message — never a silent fallback to a different strategy or provider.
    """
    store, hits = _retrieve(
        recipe_id, req.question, req.top_k, req.retriever, req.retriever_params
    )

    try:
        prompt, result, model = generate_answer(
            req.question,
            hits,
            req.provider,
            provider_params=req.provider_params,
            gen_params=req.gen_params,
        )
    except ValueError as exc:  # unknown provider key — no silent fallback
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # provider failed (no package/key, unreachable, ...)
        raise HTTPException(
            status_code=502, detail=f"Generation failed: {exc}"
        ) from exc

    return GenerateResponse(
        recipe_id=recipe_id,
        question=req.question,
        provider=req.provider,
        retriever=req.retriever,
        model=model,
        answer=result.text,
        prompt=prompt,
        retrieved_chunks=_to_hits(hits),
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        metric=store.metric,
        embedding_model=store.model_name,
    )
