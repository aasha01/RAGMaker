"""Step 2 demo: run one document through all five stages, no UI, no Recipe yet.

This is a throwaway harness to prove the default no-API-key path works
end-to-end and to produce the exact `stage_outputs/` artifacts from
ARCHITECTURE.md section 4. Step 3 (core/recipe.py) will replace this with a
proper, saved Recipe. Everything here talks only to the stage REGISTRY +
base interfaces — it never imports a concrete strategy directly.

Usage:
    python run_pipeline.py [path/to/file] [--out demo_output]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

from backend.stages.parsers import REGISTRY as PARSERS
from backend.stages.chunkers import REGISTRY as CHUNKERS
from backend.stages.embedders import REGISTRY as EMBEDDERS
from backend.stages.vectorstores import REGISTRY as VECTORSTORES

DEFAULT_FILE = os.path.join("sample_data", "discharge_summary_detailed.txt")

# The default no-API-key recipe, expressed purely as registry keys + params.
CONFIG = {
    "parser": {"key": "manual"},
    "chunker": {"key": "recursive", "size": 512, "overlap": 50},
    "embedder": {"key": "sentence_transformers", "model_name": "all-MiniLM-L6-v2", "normalize": True},
    "vectorstore": {"key": "faiss", "index_type": "flat", "metric": "cosine"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one file through all five RAG stages.")
    ap.add_argument("file", nargs="?", default=DEFAULT_FILE, help="source document")
    ap.add_argument("--out", default="demo_output", help="output directory")
    ap.add_argument("--query", default="How was the pneumonia treated in hospital?",
                    help="a sanity-check question to retrieve against")
    args = ap.parse_args()

    src = args.file
    if not os.path.isfile(src):
        raise SystemExit(f"File not found: {src}")

    out = args.out
    stage_dir = os.path.join(out, "stage_outputs")
    os.makedirs(stage_dir, exist_ok=True)

    # --- Stage 1: Ingestion --------------------------------------------------
    raw_dir = os.path.join(stage_dir, "01_raw")
    os.makedirs(raw_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(raw_dir, os.path.basename(src)))
    size_bytes = os.path.getsize(src)
    print(f"[1/5] Ingestion : {os.path.basename(src)}  ({size_bytes:,} bytes)  -> 01_raw/")

    # --- Stage 2: Parsing ----------------------------------------------------
    t0 = time.perf_counter()
    parser = PARSERS[CONFIG["parser"]["key"]]()
    doc = parser.parse(src)
    parse_sec = time.perf_counter() - t0
    with open(os.path.join(stage_dir, "02_parsed.json"), "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
    words = len(doc.text.split())
    print(f"[2/5] Parsing   : parser='{parser.name}'  {len(doc.text):,} chars, "
          f"{words:,} words, {doc.metadata.get('pages','?')} page(s)  -> 02_parsed.json")

    # --- Stage 3: Chunking ---------------------------------------------------
    t0 = time.perf_counter()
    ck = CHUNKERS[CONFIG["chunker"]["key"]]()
    params = {k: v for k, v in CONFIG["chunker"].items() if k != "key"}
    chunks = ck.chunk(doc, **params)
    chunk_sec = time.perf_counter() - t0
    with open(os.path.join(stage_dir, "03_chunks.json"), "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
    char_lens = [c.char_len for c in chunks]
    overlaps = [c.overlap_with_prev for c in chunks if c.overlap_with_prev]
    print(f"[3/5] Chunking  : chunker='{ck.name}' size={params['size']} overlap={params['overlap']}")
    print(f"                  {len(chunks)} chunks, char_len min/mean/max = "
          f"{min(char_lens)}/{sum(char_lens)//len(char_lens)}/{max(char_lens)}, "
          f"avg overlap={ (sum(overlaps)//len(overlaps)) if overlaps else 0 } chars  -> 03_chunks.json")

    # --- Stage 4: Embedding --------------------------------------------------
    print(f"[4/5] Embedding : loading model '{CONFIG['embedder']['model_name']}' (first run downloads it)...")
    t0 = time.perf_counter()
    emb_params = {k: v for k, v in CONFIG["embedder"].items() if k != "key"}
    embedder = EMBEDDERS[CONFIG["embedder"]["key"]](**emb_params)
    vectors = embedder.embed([c.text for c in chunks])
    embed_sec = time.perf_counter() - t0
    np.save(os.path.join(stage_dir, "04_embeddings.npy"), vectors)
    meta = {
        "model_name": embedder.model_name,
        "dimension": embedder.dimension,
        "normalize": emb_params.get("normalize", True),
        "chunk_id_order": [c.chunk_id for c in chunks],
    }
    with open(os.path.join(stage_dir, "04_embeddings_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"                  embedder='{embedder.name}'  vectors shape={vectors.shape}, "
          f"dim={embedder.dimension}, {embed_sec:.1f}s  -> 04_embeddings.npy (+ meta)")

    # --- Stage 5: Vector store ----------------------------------------------
    t0 = time.perf_counter()
    store = VECTORSTORES[CONFIG["vectorstore"]["key"]]()
    vs_params = {k: v for k, v in CONFIG["vectorstore"].items() if k != "key"}
    store.build(vectors, chunks, model_name=embedder.model_name,
                normalize=emb_params.get("normalize", True), **vs_params)
    vs_dir = os.path.join(out, "vectorstore")
    store.save(vs_dir)
    store_sec = time.perf_counter() - t0
    print(f"[5/5] Vectorstore: store='{store.name}' index={vs_params['index_type']} "
          f"metric={vs_params['metric']}  {len(chunks)} vectors indexed  -> vectorstore/")

    # --- Sanity check: reload from disk and run one query --------------------
    print("\n--- Sanity query (reloaded store) ---")
    reloaded = VECTORSTORES[CONFIG["vectorstore"]["key"]]()
    reloaded.load(vs_dir)
    if reloaded.model_name != embedder.model_name:
        raise SystemExit("Model mismatch between store and embedder — aborting (Non-Negotiable).")
    q_vec = embedder.embed([args.query])
    hits = reloaded.search(q_vec, top_k=3)
    print(f"Q: {args.query!r}")
    for rank, hit in enumerate(hits, 1):
        preview = " ".join(hit.chunk.text.split())[:160]
        print(f"  #{rank}  score={hit.score:.4f}  [{hit.chunk.chunk_id}]  {preview}...")

    # Prove the dimension guard fires on a mismatched query vector.
    try:
        reloaded.search(np.zeros((1, embedder.dimension + 1), dtype=np.float32), top_k=1)
        print("\n[guard] FAIL: mismatched-dimension query did NOT raise")
    except ValueError:
        print("\n[guard] OK: mismatched-dimension query correctly rejected")

    total = parse_sec + chunk_sec + embed_sec + store_sec
    print(f"\nDone. Stage timings (s): parse={parse_sec:.2f} chunk={chunk_sec:.2f} "
          f"embed={embed_sec:.2f} store={store_sec:.2f} total={total:.2f}")
    print(f"Artifacts written under: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
