"""Pipeline execution endpoints for stages 1-2: /parse and /chunk.

These are deliberately stateless: /parse takes raw file bytes and returns the
parsed document; /chunk takes that document back and returns chunks. The
frontend holds the intermediate result between calls. (Persisting a full
immutable Recipe is Step 3, in core/recipe.py.)

Both endpoints surface the *real* stage error on failure (HTTP 422) rather than
substituting a different technique — the "no silent fallback" Non-Negotiable,
enforced at the API boundary.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Body, HTTPException, Query

from backend.stages.parsers import REGISTRY as PARSERS
from backend.stages.parsers.base import ParsedDocument
from backend.stages.chunkers import REGISTRY as CHUNKERS
from ..services import get_embedder
from ..schemas import (
    ChunkModel,
    ChunkRequest,
    ChunkResponse,
    ChunkSummary,
    ParsedDocumentModel,
    ParseResponse,
    ParseSummary,
)

router = APIRouter(tags=["pipeline"])


@router.post("/parse", response_model=ParseResponse)
def parse(
    filename: str = Query(..., description="Original filename; its extension selects the format"),
    parser: str = Query("manual", description="Parser registry key"),
    content: bytes = Body(
        ...,
        media_type="application/octet-stream",
        description="Raw file bytes (send the file as the request body).",
    ),
) -> ParseResponse:
    if parser not in PARSERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown parser '{parser}'. Available: {sorted(PARSERS)}",
        )

    # Write to a temp file preserving the extension so the parser can dispatch
    # on it, then clean up regardless of success.
    ext = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
        tf.write(content)
        tmp_path = tf.name
    try:
        doc = PARSERS[parser]().parse(tmp_path)
    except Exception as exc:  # no silent fallback — report the real failure
        raise HTTPException(
            status_code=422,
            detail=f"Parsing failed with '{parser}': {exc}",
        ) from exc
    finally:
        os.unlink(tmp_path)

    # The parser saw a temp filename; restore the real source name for the UI.
    doc.source = filename

    summary = ParseSummary(
        char_count=len(doc.text),
        word_count=len(doc.text.split()),
        pages=doc.metadata.get("pages"),
        format=doc.metadata.get("format") or ext.lstrip("."),
    )
    return ParseResponse(
        parser=parser,
        document=ParsedDocumentModel(**doc.to_dict()),
        summary=summary,
    )


@router.post("/chunk", response_model=ChunkResponse)
def chunk(req: ChunkRequest) -> ChunkResponse:
    if req.chunker not in CHUNKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chunker '{req.chunker}'. Available: {sorted(CHUNKERS)}",
        )

    doc = ParsedDocument.from_dict(req.document.model_dump())
    chunk_kwargs = dict(req.params)
    if req.embedder:
        # Only chunkers that declare an `embedder` param (e.g. semantic) use
        # this; others accept and ignore it via **_ignored (Strategy Pattern —
        # this router stays generic across chunkers).
        chunk_kwargs["embedder"] = get_embedder(req.embedder, req.embedder_params)
    try:
        chunks = CHUNKERS[req.chunker]().chunk(doc, **chunk_kwargs)
    except Exception as exc:  # e.g. invalid size/overlap — report it, don't guess
        raise HTTPException(
            status_code=422,
            detail=f"Chunking failed with '{req.chunker}': {exc}",
        ) from exc

    char_lens = [c.char_len for c in chunks] or [0]
    token_lens = [c.token_len for c in chunks] or [0]
    summary = ChunkSummary(
        chunk_count=len(chunks),
        char_len_min=min(char_lens),
        char_len_mean=sum(char_lens) // len(char_lens),
        char_len_max=max(char_lens),
        token_len_mean=sum(token_lens) // len(token_lens),
        total_overlap_chars=sum(c.overlap_with_prev for c in chunks),
    )
    return ChunkResponse(
        chunker=req.chunker,
        params=req.params,
        chunks=[ChunkModel(**c.to_dict()) for c in chunks],
        summary=summary,
    )
