"""Contract tests for core/recipe.py — build/persist/load/list a Recipe.

Uses a temporary recipes dir so it never touches the real recipes/ folder.
Requires sentence-transformers (embedding runs during build); skips cleanly if
absent.

Run with:  pytest -q tests/test_recipe.py   (or)   python tests/test_recipe.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_TXT = os.path.join("sample_data", "discharge_summary_detailed.txt")

CONFIG = {
    "parser": {"key": "manual"},
    "chunker": {"key": "recursive", "size": 512, "overlap": 50},
    "embedder": {"key": "sentence_transformers", "model_name": "all-MiniLM-L6-v2", "normalize": True},
    "vectorstore": {"key": "faiss", "index_type": "flat", "metric": "cosine"},
}


def _have_st() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def test_build_persist_load_list_search():
    if not _have_st():
        print("SKIP test_build_persist_load_list_search (no sentence-transformers)")
        return
    from backend.core import recipe as R

    root = tempfile.mkdtemp()
    row = R.build_recipe(SAMPLE_TXT, "discharge_summary_detailed.txt", CONFIG,
                         description="unit test", recipes_root=root)
    rid = row["recipe_id"]
    assert rid == "recipe_001"
    assert row["chunk_count"] > 1

    # All ARCHITECTURE §4 artifacts exist on disk.
    rdir = row["path"]
    for rel in [
        "config.json", "metadata.json",
        "stage_outputs/01_raw/discharge_summary_detailed.txt",
        "stage_outputs/02_parsed.json", "stage_outputs/03_chunks.json",
        "stage_outputs/04_embeddings.npy", "stage_outputs/04_embeddings_meta.json",
        "vectorstore/index.faiss", "vectorstore/store_meta.json",
    ]:
        assert os.path.exists(os.path.join(rdir, rel)), rel

    # config.json matches the SPEC §8 shape.
    with open(os.path.join(rdir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["recipe_id"] == rid
    assert cfg["source"]["filename"] == "discharge_summary_detailed.txt"
    assert cfg["parser"]["key"] == "manual"
    assert cfg["chunker"] == {"key": "recursive", "size": 512, "overlap": 50}
    assert cfg["embedder"]["dimension"] == 384
    assert cfg["vectorstore"] == {"key": "faiss", "index_type": "flat", "metric": "cosine"}
    assert cfg["stats"]["chunk_count"] == row["chunk_count"]

    # load + list + index round-trip.
    info = R.load_recipe(rid, recipes_root=root)
    assert info["config"]["recipe_id"] == rid
    listed = R.list_recipes(recipes_root=root)
    assert any(r["recipe_id"] == rid for r in listed)

    # second recipe increments the id.
    row2 = R.build_recipe(SAMPLE_TXT, "discharge_summary_detailed.txt", CONFIG, recipes_root=root)
    assert row2["recipe_id"] == "recipe_002"

    # open the store and search it (re-embed the query with the recipe's model).
    store, ekey, eparams = R.open_recipe_store(rid, recipes_root=root)
    from backend.stages.embedders import REGISTRY as EMBEDDERS
    embedder = EMBEDDERS[ekey](**eparams)
    hits = store.search(embedder.embed(["pneumonia treatment"]), top_k=3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)  # cosine: best first


def test_build_rejects_unknown_strategy():
    from backend.core import recipe as R

    root = tempfile.mkdtemp()
    bad = dict(CONFIG)
    bad = {**CONFIG, "chunker": {"key": "does_not_exist"}}
    raised = False
    try:
        R.build_recipe(SAMPLE_TXT, "x.txt", bad, recipes_root=root)
    except ValueError:
        raised = True  # no silent fallback
    assert raised


def test_rebuild_index_from_files():
    if not _have_st():
        print("SKIP test_rebuild_index_from_files (no sentence-transformers)")
        return
    from backend.core import recipe as R

    root = tempfile.mkdtemp()
    R.build_recipe(SAMPLE_TXT, "discharge_summary_detailed.txt", CONFIG, recipes_root=root)
    # delete the index; it must rebuild from the on-disk config files.
    os.remove(os.path.join(root, "index.db"))
    n = R.rebuild_index(root)
    assert n == 1
    assert len(R.list_recipes(recipes_root=root)) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
