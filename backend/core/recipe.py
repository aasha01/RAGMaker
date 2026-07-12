"""Recipe orchestration: build a full pipeline from a config and persist it.

A **Recipe** is the unit of comparison and it is immutable once built. Given a
source file and a config (which strategy + params for each stage), `build_recipe`
runs parse -> chunk -> embed -> store and writes everything to disk exactly as
ARCHITECTURE.md section 4 describes:

    recipes/recipe_<id>_<slug>/
        config.json              # SPEC.md section 8 schema — reproduces the recipe
        metadata.json            # timings, tokens, cost, chunk count
        stage_outputs/
            01_raw/<filename>
            02_parsed.json
            03_chunks.json
            04_embeddings.npy
            04_embeddings_meta.json
        vectorstore/             # the FAISS index files

The on-disk files are the source of truth. A small SQLite index
(`recipes/index.db`) is kept alongside purely for fast listing/filtering and is
fully rebuildable from the `config.json`/`metadata.json` files.

This module only ever talks to the stage REGISTRIES, never to a concrete
strategy — so a recipe is described entirely by its config.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np

from backend.stages.parsers import REGISTRY as PARSERS
from backend.stages.chunkers import REGISTRY as CHUNKERS
from backend.stages.embedders import REGISTRY as EMBEDDERS
from backend.stages.vectorstores import REGISTRY as VECTORSTORES

RECIPES_ROOT = os.environ.get("RAG_LAB_RECIPES_DIR", "recipes")
_INDEX_DB = "index.db"

_INDEX_COLUMNS = [
    "recipe_id", "name", "created_at", "source_filename", "source_type", "parser",
    "chunker", "embedder", "model_name", "dimension", "vectorstore",
    "index_type", "metric", "chunk_count", "build_time_sec", "cost_usd",
    "description", "path",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(filename: str, limit: int = 24) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug[:limit] or "recipe"


def _next_recipe_id(recipes_root: str) -> str:
    """Sequential, human-friendly ids (recipe_001, recipe_002, ...)."""
    highest = 0
    if os.path.isdir(recipes_root):
        for name in os.listdir(recipes_root):
            m = re.match(r"recipe_(\d+)_", name)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"recipe_{highest + 1:03d}"


def _recipe_dir(recipe_id: str, recipes_root: str) -> str | None:
    if not os.path.isdir(recipes_root):
        return None
    for name in os.listdir(recipes_root):
        if name.startswith(recipe_id + "_"):
            return os.path.join(recipes_root, name)
    return None


# --------------------------------------------------------------------------- #
# SQLite index (rebuildable convenience layer over the files)
# --------------------------------------------------------------------------- #

def _index_path(recipes_root: str) -> str:
    return os.path.join(recipes_root, _INDEX_DB)


def _connect(recipes_root: str) -> sqlite3.Connection:
    os.makedirs(recipes_root, exist_ok=True)
    conn = sqlite3.connect(_index_path(recipes_root))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS recipes (
            recipe_id TEXT PRIMARY KEY, name TEXT, created_at TEXT, source_filename TEXT,
            source_type TEXT, parser TEXT, chunker TEXT, embedder TEXT,
            model_name TEXT, dimension INTEGER, vectorstore TEXT,
            index_type TEXT, metric TEXT, chunk_count INTEGER,
            build_time_sec REAL, cost_usd REAL, description TEXT, path TEXT
        )"""
    )
    # existing DBs created before the `name` column existed
    cols = {row[1] for row in conn.execute("PRAGMA table_info(recipes)")}
    if "name" not in cols:
        conn.execute("ALTER TABLE recipes ADD COLUMN name TEXT")
    return conn


def _index_upsert(recipes_root: str, row: dict) -> None:
    conn = _connect(recipes_root)
    placeholders = ",".join("?" * len(_INDEX_COLUMNS))
    conn.execute(
        f"INSERT OR REPLACE INTO recipes ({','.join(_INDEX_COLUMNS)}) VALUES ({placeholders})",
        [row.get(c) for c in _INDEX_COLUMNS],
    )
    conn.commit()
    conn.close()


def _row_from_config(config: dict, metadata: dict, path: str, description: str | None) -> dict:
    return {
        "recipe_id": config["recipe_id"],
        "name": metadata.get("name") or description or config["recipe_id"],
        "created_at": config["created_at"],
        "source_filename": config["source"]["filename"],
        "source_type": config["source"].get("type"),
        "parser": config["parser"]["key"],
        "chunker": config["chunker"]["key"],
        "embedder": config["embedder"]["key"],
        "model_name": config["embedder"].get("model_name"),
        "dimension": config["embedder"].get("dimension"),
        "vectorstore": config["vectorstore"]["key"],
        "index_type": config["vectorstore"].get("index_type"),
        "metric": config["vectorstore"].get("metric"),
        "chunk_count": config["stats"]["chunk_count"],
        "build_time_sec": config["stats"]["build_time_sec"],
        "cost_usd": config["stats"].get("embedding_cost_usd", 0.0),
        "description": description,
        "path": path,
    }


def rebuild_index(recipes_root: str = RECIPES_ROOT) -> int:
    """Rebuild the SQLite index from the on-disk recipe folders (files win)."""
    conn = _connect(recipes_root)
    conn.execute("DELETE FROM recipes")
    conn.commit()
    conn.close()
    count = 0
    if os.path.isdir(recipes_root):
        for name in sorted(os.listdir(recipes_root)):
            cdir = os.path.join(recipes_root, name)
            cfg_path = os.path.join(cdir, "config.json")
            if not os.path.isfile(cfg_path):
                continue
            with open(cfg_path, encoding="utf-8") as f:
                config = json.load(f)
            metadata = {}
            meta_path = os.path.join(cdir, "metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            _index_upsert(recipes_root, _row_from_config(config, metadata, cdir, metadata.get("description")))
            count += 1
    return count


def list_recipes(recipes_root: str = RECIPES_ROOT) -> list[dict]:
    if not os.path.exists(_index_path(recipes_root)):
        rebuild_index(recipes_root)
    conn = _connect(recipes_root)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM recipes ORDER BY recipe_id DESC")]
    conn.close()
    return rows


# --------------------------------------------------------------------------- #
# build / load
# --------------------------------------------------------------------------- #

def build_recipe(
    source_path: str,
    source_filename: str,
    config: dict,
    name: str | None = None,
    description: str | None = None,
    recipes_root: str = RECIPES_ROOT,
) -> dict:
    """Run the full pipeline described by `config` and persist it as a Recipe.

    `config` shape (params beyond `key` are stage-specific and recorded verbatim):
        {"parser": {"key": ...},
         "chunker": {"key": ..., "size": ..., "overlap": ...},
         "embedder": {"key": ..., "model_name": ..., "normalize": ..., "truncate_dim": ...?},
         "vectorstore": {"key": ..., "index_type": ..., "metric": ...}}

    `name` is the human-friendly label the learner picks to tell recipes apart
    at a glance. If left blank it falls back to `description`, and if that's
    also blank, to the auto-generated `recipe_id` — so every recipe always has
    a usable display name without forcing the user to type one.

    Raises ValueError on any unknown strategy or stage failure (no silent
    fallback). Returns the index row dict for the new recipe.
    """
    parser_key = config["parser"]["key"]
    chunker_cfg = dict(config["chunker"]); chunker_key = chunker_cfg.pop("key")
    embedder_cfg = dict(config["embedder"]); embedder_key = embedder_cfg.pop("key")
    vs_cfg = dict(config["vectorstore"]); vs_key = vs_cfg.pop("key")

    for reg, key, label in (
        (PARSERS, parser_key, "parser"),
        (CHUNKERS, chunker_key, "chunker"),
        (EMBEDDERS, embedder_key, "embedder"),
        (VECTORSTORES, vs_key, "vectorstore"),
    ):
        if key not in reg:
            raise ValueError(f"Unknown {label} '{key}'. Available: {sorted(reg)}")

    timings: dict[str, float] = {}

    t = time.perf_counter()
    doc = PARSERS[parser_key]().parse(source_path)
    doc.source = source_filename
    timings["parse"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    # embedder_cfg carries only constructor params now (key was popped). Built
    # before chunking (not just before embedding) because some chunkers — e.g.
    # 'semantic' — need a live embedder instance to run at all.
    embedder = EMBEDDERS[embedder_key](**embedder_cfg)
    timings["embedder_load"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    chunks = CHUNKERS[chunker_key]().chunk(doc, embedder=embedder, **chunker_cfg)
    timings["chunk"] = round(time.perf_counter() - t, 3)
    if not chunks:
        raise ValueError("Chunking produced 0 chunks — check the chunker params.")

    t = time.perf_counter()
    vectors = embedder.embed([c.text for c in chunks])
    timings["embed"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    store = VECTORSTORES[vs_key]()
    store.build(vectors, chunks, model_name=embedder.model_name, normalize=embedder.normalize, **vs_cfg)
    timings["store"] = round(time.perf_counter() - t, 3)

    # --- persist ---------------------------------------------------------
    recipe_id = _next_recipe_id(recipes_root)
    rdir = os.path.join(recipes_root, f"{recipe_id}_{_slug(source_filename)}")
    stage_dir = os.path.join(rdir, "stage_outputs")
    raw_dir = os.path.join(stage_dir, "01_raw")
    os.makedirs(raw_dir, exist_ok=True)

    shutil.copy2(source_path, os.path.join(raw_dir, os.path.basename(source_filename)))
    with open(os.path.join(stage_dir, "02_parsed.json"), "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(stage_dir, "03_chunks.json"), "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
    np.save(os.path.join(stage_dir, "04_embeddings.npy"), vectors)
    with open(os.path.join(stage_dir, "04_embeddings_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": embedder.model_name,
                "dimension": embedder.dimension,
                "normalize": embedder.normalize,
                "chunk_id_order": [c.chunk_id for c in chunks],
            },
            f,
            indent=2,
        )
    store.save(os.path.join(rdir, "vectorstore"))

    build_time = round(sum(timings.values()), 3)
    total_tokens = int(sum(c.token_len for c in chunks))
    ext = os.path.splitext(source_filename)[1].lstrip(".").lower()

    embedder_out = {
        "key": embedder_key,
        "model_name": embedder.model_name,
        "dimension": embedder.dimension,
        "normalize": embedder.normalize,
    }
    if embedder_cfg.get("truncate_dim"):
        embedder_out["truncate_dim"] = embedder_cfg["truncate_dim"]

    config_out = {
        "recipe_id": recipe_id,
        "created_at": _now_iso(),
        "source": {"filename": os.path.basename(source_filename), "type": ext},
        "parser": {"key": parser_key},
        "chunker": {"key": chunker_key, **chunker_cfg},
        "embedder": embedder_out,
        "vectorstore": {"key": vs_key, **vs_cfg},
        "stats": {
            "chunk_count": len(chunks),
            "build_time_sec": build_time,
            "embedding_cost_usd": 0.0,
        },
    }
    with open(os.path.join(rdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    resolved_name = name or description or recipe_id

    metadata = {
        "created_at": config_out["created_at"],
        "timings_sec": timings,
        "build_time_sec": build_time,
        "total_tokens": total_tokens,
        "embedding_cost_usd": 0.0,
        "chunk_count": len(chunks),
        "dimension": embedder.dimension,
        "name": resolved_name,
        "description": description,
    }
    with open(os.path.join(rdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    row = _row_from_config(config_out, metadata, rdir, description)
    _index_upsert(recipes_root, row)
    return row


def load_recipe(recipe_id: str, recipes_root: str = RECIPES_ROOT) -> dict:
    """Return {config, metadata, dir} for a saved recipe. Raises if missing."""
    rdir = _recipe_dir(recipe_id, recipes_root)
    if rdir is None:
        raise FileNotFoundError(f"No such recipe '{recipe_id}' under {recipes_root}.")
    with open(os.path.join(rdir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    metadata = {}
    meta_path = os.path.join(rdir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    return {"config": config, "metadata": metadata, "dir": rdir}


def open_recipe_store(recipe_id: str, recipes_root: str = RECIPES_ROOT):
    """Load a recipe's vector store from disk; return (store, embedder_key,
    embedder_params) so the caller can re-embed queries with the SAME model."""
    info = load_recipe(recipe_id, recipes_root)
    cfg = info["config"]
    vs_key = cfg["vectorstore"]["key"]
    store = VECTORSTORES[vs_key]()
    store.load(os.path.join(info["dir"], "vectorstore"))

    emb = cfg["embedder"]
    embedder_params = {"model_name": emb["model_name"], "normalize": emb["normalize"]}
    if emb.get("truncate_dim"):
        embedder_params["truncate_dim"] = emb["truncate_dim"]
    return store, emb["key"], embedder_params
