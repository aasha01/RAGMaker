"""Vector store endpoints (stage 4): build a FAISS index and search it.

Stateful by necessity — a FAISS index can't round-trip through JSON, so the
built store lives in an in-process registry keyed by a ``store_id`` (these are
in-memory and reset on server restart; durable file-based recipes come with
core/recipe.py). Each stored entry remembers the exact embedder config that
built it, so query-time re-embedding uses the SAME model — enforcing the
"never search a store with another model's vectors" Non-Negotiable. The store's
own dimension guard is the second line of defence.
"""

from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.stages.vectorstores import REGISTRY as VECTORSTORES
from backend.stages.retrievers import REGISTRY as RETRIEVERS
from backend.stages.chunkers.base import Chunk
from ..schemas import (
    SearchHit,
    SearchRequest,
    SearchResponse,
    StoreMetaRow,
    VectorStoreBuildRequest,
    VectorStoreBuildResponse,
)
from ..services import get_embedder

router = APIRouter(tags=["pipeline"])

# store_id -> {"store", "embedder", "embed_params", "model_name", "dimension"}
_STORES: dict[str, dict] = {}


@router.post("/vectorstore/build", response_model=VectorStoreBuildResponse)
def build(req: VectorStoreBuildRequest) -> VectorStoreBuildResponse:
    if req.vectorstore not in VECTORSTORES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vector store '{req.vectorstore}'. Available: {sorted(VECTORSTORES)}",
        )

    vectors = np.array(req.vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise HTTPException(status_code=422, detail="`vectors` must be a 2D array (n, dim).")
    if vectors.shape[0] != len(req.chunks):
        raise HTTPException(
            status_code=422,
            detail=f"vectors/chunks length mismatch: {vectors.shape[0]} vectors "
            f"vs {len(req.chunks)} chunks.",
        )
    meta_dim = req.meta.get("dimension")
    if meta_dim is not None and vectors.shape[1] != meta_dim:
        raise HTTPException(
            status_code=422,
            detail=f"vector dimension {vectors.shape[1]} != embedding meta dimension {meta_dim}.",
        )

    chunks = [Chunk.from_dict(c.model_dump()) for c in req.chunks]
    store = VECTORSTORES[req.vectorstore]()
    try:
        store.build(
            vectors,
            chunks,
            model_name=req.meta.get("model_name", ""),
            normalize=req.meta.get("normalize", False),
            **req.params,
        )
    except Exception as exc:  # no silent fallback — report the real failure
        raise HTTPException(
            status_code=422,
            detail=f"Vector store build failed with '{req.vectorstore}': {exc}",
        ) from exc

    store_id = uuid.uuid4().hex[:12]
    _STORES[store_id] = {
        "store": store,
        "embedder": req.embedder,
        "embed_params": req.embed_params,
        "model_name": req.meta.get("model_name", ""),
        "dimension": int(vectors.shape[1]),
    }
    sample = [
        StoreMetaRow(chunk_id=c.chunk_id, source=c.source, position=c.position, char_len=c.char_len)
        for c in chunks[:50]
    ]
    return VectorStoreBuildResponse(
        store_id=store_id,
        vectorstore=req.vectorstore,
        index_type=store.index_type,
        metric=store.metric,
        count=len(chunks),
        dimension=int(vectors.shape[1]),
        model_name=req.meta.get("model_name", ""),
        metadata_sample=sample,
    )


@router.post("/vectorstore/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    entry = _STORES.get(req.store_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No such store_id '{req.store_id}'. Build a store first "
            f"(stores are in-memory and reset on server restart).",
        )
    if req.retriever not in RETRIEVERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown retriever '{req.retriever}'. Available: {sorted(RETRIEVERS)}",
        )

    # Embed the query with the SAME embedder config that built the store.
    embedder = get_embedder(entry["embedder"], entry["embed_params"])
    if embedder.model_name != entry["model_name"]:
        raise HTTPException(
            status_code=422,
            detail=f"Query embedder model '{embedder.model_name}' does not match "
            f"the store's model '{entry['model_name']}'. Refusing to mix models.",
        )

    store = entry["store"]
    # Apply the chosen query-time retrieval strategy over the store (SPEC.md §6),
    # chosen per query. naive_topk is the plain search; the others add diversity /
    # lexical fusion / re-ranking / HyDE.
    try:
        hits = RETRIEVERS[req.retriever]().retrieve(
            req.query,
            store=store,
            embedder=embedder,
            top_k=req.top_k,
            **req.retriever_params,
        )
    except ValueError as exc:  # store's own dimension guard / bad params
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # missing dep (rank-bm25/cross-encoder), HyDE LLM, ...
        raise HTTPException(
            status_code=502, detail=f"Retrieval failed with '{req.retriever}': {exc}"
        ) from exc

    return SearchResponse(
        store_id=req.store_id,
        query=req.query,
        metric=store.metric,
        model_name=entry["model_name"],
        retriever=req.retriever,
        hits=[
            SearchHit(
                chunk_id=h.chunk.chunk_id,
                text=h.chunk.text,
                source=h.chunk.source,
                position=h.chunk.position,
                score=h.score,
            )
            for h in hits
        ],
    )
