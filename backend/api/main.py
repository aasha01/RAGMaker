"""FastAPI application entrypoint.

Run locally with:
    uvicorn backend.api.main:app --reload
Interactive API docs are then at http://localhost:8000/docs

Only stages 1-2 (parse, chunk) are wired so far; embedding, vector store,
recipe persistence and the query grid are added in later build steps.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import embedding, pipeline, recipes, scoring, stages, vectorstore

app = FastAPI(
    title="RAG Lab API",
    version="0.1.0",
    description=(
        "Backend for the RAG Lab teaching tool. Exposes the swappable pipeline "
        "stages over HTTP. Wired so far: stage discovery, /parse, /chunk, /embed, "
        "/vectorstore/build, /vectorstore/search, per-query retrieval strategies "
        "(naive_topk/mmr/hybrid/rerank/hyde), recipe build/list/search/generate, "
        "and optional RAGAS-style answer scoring (/score)."
    ),
)

# The Streamlit frontend runs on a different port, so allow cross-origin calls.
# Local teaching tool: permissive is fine and keeps setup frictionless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness check the frontend uses to confirm the backend is reachable."""
    return {"status": "ok", "service": "rag-lab-api", "version": app.version}


app.include_router(stages.router)
app.include_router(pipeline.router)
app.include_router(embedding.router)
app.include_router(vectorstore.router)
app.include_router(recipes.router)
app.include_router(scoring.router)
