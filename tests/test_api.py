"""Contract tests for the backend API and the frontend's HTTP client.

Covers stages 1-2 over HTTP: stage discovery, /parse, /chunk, the error paths
(no silent fallback), and the frontend APIClient driven against the FastAPI app
in-process (httpx ASGITransport) so the FE<->BE contract is exercised without a
running server.

Run with:  pytest -q tests/test_api.py   (or)   python tests/test_api.py
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile

# Isolate recipe persistence to a temp dir BEFORE importing the app (recipe.py
# reads RAG_LAB_RECIPES_DIR at import time). Keeps the real recipes/ untouched.
os.environ.setdefault("RAG_LAB_RECIPES_DIR", tempfile.mkdtemp(prefix="raglab_recipes_"))

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.main import app
from frontend.api_client import APIClient, APIError

client = TestClient(app)
SAMPLE_TXT = os.path.join("sample_data", "discharge_summary_detailed.txt")


def _sample_bytes() -> bytes:
    with open(SAMPLE_TXT, "rb") as f:
        return f.read()


def _sample_b64() -> str:
    return base64.b64encode(_sample_bytes()).decode()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_parsers_contains_manual():
    r = client.get("/stages/parsers")
    assert r.status_code == 200
    items = {o["key"]: o for o in r.json()}
    assert "manual" in items
    assert items["manual"]["name"]
    assert items["manual"]["description"]  # teaching text is surfaced


def test_list_chunkers_contains_recursive():
    r = client.get("/stages/chunkers")
    keys = {o["key"] for o in r.json()}
    assert "recursive" in keys


def test_parse_txt():
    r = client.post(
        "/parse",
        params={"filename": "discharge_summary_detailed.txt", "parser": "manual"},
        content=_sample_bytes(),
    )
    assert r.status_code == 200
    body = r.json()
    assert "DISCHARGE SUMMARY" in body["document"]["text"]
    assert body["document"]["source"] == "discharge_summary_detailed.txt"
    assert body["summary"]["char_count"] == len(body["document"]["text"])
    assert body["summary"]["word_count"] > 0


def test_parse_unknown_parser_400():
    r = client.post(
        "/parse", params={"filename": "x.txt", "parser": "nope"}, content=b"hi"
    )
    assert r.status_code == 400


def test_parse_unsupported_extension_422():
    # Manual parser raises ValueError for unknown extensions -> 422, no fallback.
    r = client.post(
        "/parse", params={"filename": "data.xyz", "parser": "manual"}, content=b"hi"
    )
    assert r.status_code == 422
    assert "does not support" in r.json()["detail"]


def test_chunk_roundtrip_and_overlap():
    parsed = client.post(
        "/parse",
        params={"filename": "discharge_summary_detailed.txt", "parser": "manual"},
        content=_sample_bytes(),
    ).json()
    r = client.post(
        "/chunk",
        json={
            "document": parsed["document"],
            "chunker": "recursive",
            "params": {"size": 512, "overlap": 50},
        },
    )
    assert r.status_code == 200
    body = r.json()
    chunks = body["chunks"]
    assert body["summary"]["chunk_count"] == len(chunks) > 1
    # First chunk shares nothing; overlaps are truthful against the prev tail.
    assert chunks[0]["overlap_with_prev"] == 0
    for prev, cur in zip(chunks, chunks[1:]):
        n = cur["overlap_with_prev"]
        if n:
            assert cur["text"][:n] == prev["text"][-n:]


def test_chunk_unknown_chunker_400():
    r = client.post(
        "/chunk",
        json={"document": {"text": "hi", "source": "s", "metadata": {}}, "chunker": "nope"},
    )
    assert r.status_code == 400


def test_chunk_bad_overlap_422():
    r = client.post(
        "/chunk",
        json={
            "document": {"text": "x" * 100, "source": "s", "metadata": {}},
            "chunker": "recursive",
            "params": {"size": 100, "overlap": 100},
        },
    )
    assert r.status_code == 422


def test_list_embedders_contains_sentence_transformers():
    r = client.get("/stages/embedders")
    assert r.status_code == 200
    keys = {o["key"] for o in r.json()}
    assert "sentence_transformers" in keys


def test_list_embedders_contains_ollama():
    r = client.get("/stages/embedders")
    keys = {o["key"] for o in r.json()}
    assert "ollama" in keys


def test_chunk_semantic_without_embedder_422():
    # Reproduces the original bug report: picking 'semantic' with no embedder
    # must fail loudly with a clear 422, not a 500.
    r = client.post(
        "/chunk",
        json={
            "document": {"text": "Sentence one. Sentence two.", "source": "s", "metadata": {}},
            "chunker": "semantic",
            "params": {"similarity_threshold": 0.3},
        },
    )
    assert r.status_code == 422
    assert "embedder" in r.json()["detail"].lower()


def test_chunk_semantic_with_embedder_end_to_end():
    # The actual fix: passing `embedder` + `embedder_params` on /chunk lets the
    # semantic chunker run, using an embedder constructed server-side (here the
    # Ollama one, with httpx mocked so no real server is needed).
    import httpx

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"embedding": [1.0, 0.0, 0.0]}

    def fake_post(url, json=None, timeout=None):
        return FakeResp()

    original = httpx.post
    httpx.post = fake_post
    try:
        r = client.post(
            "/chunk",
            json={
                "document": {
                    "text": "Sentence one is here. Sentence two follows after.",
                    "source": "s",
                    "metadata": {},
                },
                "chunker": "semantic",
                "params": {"similarity_threshold": 0.3},
                "embedder": "ollama",
                "embedder_params": {"model_name": "nomic-embed-text:test-e2e"},
            },
        )
    finally:
        httpx.post = original

    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["chunk_count"] >= 1


def test_embed_unknown_embedder_400():
    r = client.post(
        "/embed",
        json={"chunks": [{"chunk_id": "c0", "text": "hi", "source": "s",
                          "position": 0, "char_len": 2, "token_len": 1,
                          "overlap_with_prev": 0}], "embedder": "nope"},
    )
    assert r.status_code == 400


def test_embed_empty_chunks_422():
    r = client.post("/embed", json={"chunks": [], "embedder": "sentence_transformers"})
    assert r.status_code == 422


def test_embed_roundtrip():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_embed_roundtrip (sentence-transformers not installed)")
        return
    parsed = client.post(
        "/parse",
        params={"filename": "discharge_summary_detailed.txt", "parser": "manual"},
        content=_sample_bytes(),
    ).json()
    chunked = client.post(
        "/chunk",
        json={"document": parsed["document"], "chunker": "recursive",
              "params": {"size": 512, "overlap": 50}},
    ).json()
    r = client.post(
        "/embed",
        json={"chunks": chunked["chunks"], "embedder": "sentence_transformers",
              "params": {"model_name": "all-MiniLM-L6-v2", "normalize": True}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dimension"] == 384
    assert body["count"] == chunked["summary"]["chunk_count"]
    assert len(body["vectors"]) == body["count"]
    assert all(len(v) == 384 for v in body["vectors"])
    # normalize=True -> unit vectors
    assert all(abs(n - 1.0) < 1e-3 for n in body["norms_preview"])
    # meta identity travels with the vectors
    assert body["meta"]["model_name"] == "all-MiniLM-L6-v2"
    assert body["meta"]["chunk_id_order"] == [c["chunk_id"] for c in chunked["chunks"]]
    # model details are included in the embed response
    assert body["model_info"]["default_dimension"] == 384
    assert body["model_info"]["max_seq_length_tokens"] > 0


def test_embedder_model_info_unknown_400():
    r = client.get("/embedders/nope/model_info")
    assert r.status_code == 400


def test_embedder_model_info():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_embedder_model_info (no sentence-transformers)")
        return
    r = client.get(
        "/embedders/sentence_transformers/model_info",
        params={"model_name": "all-MiniLM-L6-v2"},
    )
    assert r.status_code == 200
    info = r.json()
    assert info["default_dimension"] == 384
    assert info["output_dimension"] == 384
    assert info["max_seq_length_tokens"] and info["max_seq_length_tokens"] > 0  # context window
    assert info["param_count"] and info["param_count"] > 0
    assert info["approx_size_mb"] and info["approx_size_mb"] > 0
    assert info["dimension_customizable"] is True
    assert info["backend"] == "sentence-transformers"


def test_embedder_model_info_truncated():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_embedder_model_info_truncated (no sentence-transformers)")
        return
    r = client.get(
        "/embedders/sentence_transformers/model_info",
        params={"model_name": "all-MiniLM-L6-v2", "truncate_dim": 128},
    )
    assert r.status_code == 200
    info = r.json()
    assert info["output_dimension"] == 128
    assert info["truncate_dim"] == 128
    assert info["notes"]  # lossy-truncation caveat present


def test_embedder_model_info_bad_truncate_422():
    r = client.get(
        "/embedders/sentence_transformers/model_info",
        params={"model_name": "all-MiniLM-L6-v2", "truncate_dim": 99999},
    )
    assert r.status_code == 422


def test_embed_truncated_dim():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_embed_truncated_dim (no sentence-transformers)")
        return
    parsed = client.post(
        "/parse",
        params={"filename": "discharge_summary_detailed.txt", "parser": "manual"},
        content=_sample_bytes(),
    ).json()
    chunked = client.post(
        "/chunk",
        json={"document": parsed["document"], "chunker": "recursive",
              "params": {"size": 512, "overlap": 50}},
    ).json()
    r = client.post(
        "/embed",
        json={"chunks": chunked["chunks"], "embedder": "sentence_transformers",
              "params": {"model_name": "all-MiniLM-L6-v2", "normalize": True, "truncate_dim": 128}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dimension"] == 128
    assert all(len(v) == 128 for v in body["vectors"])
    assert all(abs(n - 1.0) < 1e-3 for n in body["norms_preview"])
    assert body["model_info"]["output_dimension"] == 128
    assert body["model_info"]["default_dimension"] == 384
    assert body["meta"]["dimension"] == 128


def test_list_vectorstores_contains_faiss():
    r = client.get("/stages/vectorstores")
    assert r.status_code == 200
    keys = {o["key"] for o in r.json()}
    assert "faiss" in keys


def test_list_llm_providers():
    r = client.get("/stages/llm_providers")
    assert r.status_code == 200
    items = {o["key"]: o for o in r.json()}
    assert {"ollama", "openai", "anthropic"} <= set(items)
    # teaching text surfaced; discovery never instantiates or imports an SDK
    assert items["ollama"]["name"] and items["ollama"]["description"]


def _fake_chunks(n: int) -> list[dict]:
    return [
        {"chunk_id": f"chunk_{i:04d}", "text": f"text {i}", "source": "s",
         "position": i, "char_len": 6, "token_len": 2, "overlap_with_prev": 0}
        for i in range(n)
    ]


def test_vectorstore_build_unknown_400():
    r = client.post(
        "/vectorstore/build",
        json={"vectors": [[0.1, 0.2]], "chunks": _fake_chunks(1), "vectorstore": "nope"},
    )
    assert r.status_code == 400


def test_vectorstore_build_length_mismatch_422():
    r = client.post(
        "/vectorstore/build",
        json={"vectors": [[0.1, 0.2], [0.3, 0.4]], "chunks": _fake_chunks(3),
              "vectorstore": "faiss"},
    )
    assert r.status_code == 422
    assert "mismatch" in r.json()["detail"]


def test_search_unknown_store_404():
    r = client.post("/vectorstore/search", json={"store_id": "deadbeef", "query": "hi"})
    assert r.status_code == 404


def test_vectorstore_build_and_search():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_vectorstore_build_and_search (no sentence-transformers)")
        return
    parsed = client.post(
        "/parse",
        params={"filename": "discharge_summary_detailed.txt", "parser": "manual"},
        content=_sample_bytes(),
    ).json()
    chunked = client.post(
        "/chunk",
        json={"document": parsed["document"], "chunker": "recursive",
              "params": {"size": 512, "overlap": 50}},
    ).json()
    embedded = client.post(
        "/embed",
        json={"chunks": chunked["chunks"], "embedder": "sentence_transformers",
              "params": {"model_name": "all-MiniLM-L6-v2", "normalize": True}},
    ).json()

    built = client.post(
        "/vectorstore/build",
        json={
            "vectors": embedded["vectors"],
            "chunks": chunked["chunks"],
            "vectorstore": "faiss",
            "params": {"index_type": "flat", "metric": "cosine"},
            "embedder": embedded["embedder"],
            "embed_params": {"model_name": "all-MiniLM-L6-v2", "normalize": True},
            "meta": embedded["meta"],
        },
    )
    assert built.status_code == 200
    store = built.json()
    assert store["count"] == chunked["summary"]["chunk_count"]
    assert store["dimension"] == 384
    assert store["metric"] == "cosine"
    store_id = store["store_id"]

    r = client.post(
        "/vectorstore/search",
        json={"store_id": store_id, "query": "How was the pneumonia treated?", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["hits"]) == 3
    # cosine: scores in descending order (most similar first)
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert body["model_name"] == "all-MiniLM-L6-v2"


RECIPE_CONFIG = {
    "parser": {"key": "manual"},
    "chunker": {"key": "recursive", "size": 512, "overlap": 50},
    "embedder": {"key": "sentence_transformers", "model_name": "all-MiniLM-L6-v2", "normalize": True},
    "vectorstore": {"key": "faiss", "index_type": "flat", "metric": "cosine"},
}


def test_recipe_build_unknown_strategy_422():
    bad = {**RECIPE_CONFIG, "chunker": {"key": "does_not_exist"}}
    r = client.post(
        "/recipes",
        json={"source_filename": "d.txt", "source_b64": _sample_b64(), "config": bad},
    )
    assert r.status_code == 422


def test_recipe_get_unknown_404():
    assert client.get("/recipes/recipe_999").status_code == 404


def test_recipe_search_unknown_404():
    r = client.post("/recipes/recipe_999/search", json={"query": "x"})
    assert r.status_code == 404


def test_recipe_build_list_get_search():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_recipe_build_list_get_search (no sentence-transformers)")
        return
    built = client.post(
        "/recipes",
        json={"source_filename": "discharge_summary_detailed.txt",
              "source_b64": _sample_b64(), "config": RECIPE_CONFIG,
              "name": "my named recipe", "description": "api test"},
    )
    assert built.status_code == 200
    row = built.json()
    rid = row["recipe_id"]
    assert row["chunk_count"] > 1
    assert row["model_name"] == "all-MiniLM-L6-v2"
    assert row["name"] == "my named recipe"

    # appears in the list
    listed = client.get("/recipes").json()
    assert any(r["recipe_id"] == rid for r in listed)

    # no name given -> falls back to description, then to recipe_id
    fallback = client.post(
        "/recipes",
        json={"source_filename": "discharge_summary_detailed.txt",
              "source_b64": _sample_b64(), "config": RECIPE_CONFIG,
              "description": "fallback description"},
    ).json()
    assert fallback["name"] == "fallback description"

    no_label = client.post(
        "/recipes",
        json={"source_filename": "discharge_summary_detailed.txt",
              "source_b64": _sample_b64(), "config": RECIPE_CONFIG},
    ).json()
    assert no_label["name"] == no_label["recipe_id"]

    # detail carries the reproducible config
    detail = client.get(f"/recipes/{rid}").json()
    assert detail["config"]["vectorstore"] == {"key": "faiss", "index_type": "flat", "metric": "cosine"}
    assert detail["metadata"]["chunk_count"] == row["chunk_count"]

    # query the saved recipe
    s = client.post(f"/recipes/{rid}/search", json={"query": "pneumonia treatment", "top_k": 3})
    assert s.status_code == 200
    body = s.json()
    assert len(body["hits"]) == 3
    assert body["model_name"] == "all-MiniLM-L6-v2"
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)


def test_recipe_generate_unknown_recipe_404():
    r = client.post(
        "/recipes/recipe_999/generate", json={"question": "x", "provider": "ollama"}
    )
    assert r.status_code == 404


def test_recipe_generate_with_mock_provider():
    """Full generate flow with a mock provider registered in the REGISTRY — no
    network/SDK/key needed. Skips (like its sibling recipe tests) when
    sentence-transformers isn't installed, since retrieval needs the embedder."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_recipe_generate_with_mock_provider (no sentence-transformers)")
        return
    from backend.stages.llm_providers import REGISTRY as LLM_PROVIDERS
    from backend.stages.llm_providers.base import BaseLLMProvider, GenerationResult

    class MockProvider(BaseLLMProvider):
        name = "Mock"
        description = "test double"

        def __init__(self, model="mock-1"):
            self.model = model

        def generate(self, prompt, **kwargs):
            # Echo evidence the retrieved context reached the prompt.
            grounded = "pneumonia" if "pneumonia" in prompt.lower() else "unknown"
            return GenerationResult(
                text=f"MOCK[{grounded}]", latency_ms=2.0,
                input_tokens=len(prompt.split()), output_tokens=3, cost_usd=0.0,
            )

    built = client.post(
        "/recipes",
        json={"source_filename": "discharge_summary_detailed.txt",
              "source_b64": _sample_b64(), "config": RECIPE_CONFIG},
    ).json()
    rid = built["recipe_id"]

    LLM_PROVIDERS["_mock"] = MockProvider
    try:
        r = client.post(
            f"/recipes/{rid}/generate",
            json={"question": "How was the pneumonia treated?", "provider": "_mock",
                  "top_k": 3, "provider_params": {"model": "mock-2"}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "MOCK[pneumonia]"    # context was actually used
        assert body["model"] == "mock-2"              # provider_params honoured
        assert len(body["retrieved_chunks"]) == 3
        assert "pneumonia" in body["prompt"].lower()  # exact prompt is surfaced
        assert body["question"] in body["prompt"]
        assert body["input_tokens"] > 0 and body["output_tokens"] == 3
        assert body["cost_usd"] == 0.0
        assert body["embedding_model"] == "all-MiniLM-L6-v2"
        assert body["latency_ms"] >= 0.0

        # Recipe exists, so we get past retrieval: unknown provider -> 400.
        bad = client.post(
            f"/recipes/{rid}/generate", json={"question": "q", "provider": "nope"}
        )
        assert bad.status_code == 400
    finally:
        LLM_PROVIDERS.pop("_mock", None)


def test_list_retrievers():
    r = client.get("/stages/retrievers")
    assert r.status_code == 200
    items = {o["key"]: o for o in r.json()}
    assert {"naive_topk", "mmr", "hybrid", "rerank", "hyde"} <= set(items)
    # teaching text surfaced; discovery never imports rank-bm25/torch/an SDK
    assert items["naive_topk"]["name"] and items["naive_topk"]["description"]


def test_recipe_search_unknown_retriever_400():
    # The retriever key is validated before the recipe is even looked up, so an
    # unknown retriever is a 400 regardless of whether the recipe exists.
    r = client.post(
        "/recipes/recipe_999/search", json={"query": "x", "retriever": "nope"}
    )
    assert r.status_code == 400
    assert "Unknown retriever" in r.json()["detail"]


def test_recipe_search_with_mmr_retriever():
    """Build a recipe, then retrieve with a non-default strategy (mmr) over it —
    proving retrieval is chosen per query, not baked into the recipe. Skips when
    sentence-transformers isn't installed (retrieval needs the real embedder)."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_recipe_search_with_mmr_retriever (no sentence-transformers)")
        return
    built = client.post(
        "/recipes",
        json={"source_filename": "discharge_summary_detailed.txt",
              "source_b64": _sample_b64(), "config": RECIPE_CONFIG},
    ).json()
    rid = built["recipe_id"]

    r = client.post(
        f"/recipes/{rid}/search",
        json={"query": "pneumonia treatment", "top_k": 3, "retriever": "mmr",
              "retriever_params": {"lambda_mult": 0.3, "fetch_k": 10}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retriever"] == "mmr"          # echoed back for inspection
    assert len(body["hits"]) == 3
    assert len({h["chunk_id"] for h in body["hits"]}) == 3  # no duplicates

    # And the same recipe still answers under plain naive top-k.
    n = client.post(
        f"/recipes/{rid}/search",
        json={"query": "pneumonia treatment", "top_k": 3, "retriever": "naive_topk"},
    ).json()
    assert n["retriever"] == "naive_topk" and len(n["hits"]) == 3


def test_recipe_search_hybrid_retriever():
    """Hybrid retrieval over a saved recipe. Needs sentence-transformers (embedder)
    and rank-bm25 (lexical); skips if either is absent."""
    try:
        import sentence_transformers  # noqa: F401
        import rank_bm25  # noqa: F401
    except ImportError:
        print("SKIP test_recipe_search_hybrid_retriever (needs sentence-transformers + rank-bm25)")
        return
    built = client.post(
        "/recipes",
        json={"source_filename": "discharge_summary_detailed.txt",
              "source_b64": _sample_b64(), "config": RECIPE_CONFIG},
    ).json()
    rid = built["recipe_id"]
    r = client.post(
        f"/recipes/{rid}/search",
        json={"query": "pneumonia treatment", "top_k": 3, "retriever": "hybrid",
              "retriever_params": {"fusion": "rrf"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retriever"] == "hybrid"
    assert 1 <= len(body["hits"]) <= 3


def test_recipe_generate_uses_chosen_retriever():
    """Generation retrieves via the chosen strategy and echoes it back. Uses a
    mock LLM provider + mmr retrieval; skips without sentence-transformers."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_recipe_generate_uses_chosen_retriever (no sentence-transformers)")
        return
    from backend.stages.llm_providers import REGISTRY as LLM_PROVIDERS
    from backend.stages.llm_providers.base import BaseLLMProvider, GenerationResult

    class MockProvider(BaseLLMProvider):
        name = "Mock"
        description = "test double"

        def __init__(self, model="mock-1"):
            self.model = model

        def generate(self, prompt, **kwargs):
            return GenerationResult(text="ok", latency_ms=1.0, input_tokens=1,
                                    output_tokens=1, cost_usd=0.0)

    built = client.post(
        "/recipes",
        json={"source_filename": "discharge_summary_detailed.txt",
              "source_b64": _sample_b64(), "config": RECIPE_CONFIG},
    ).json()
    rid = built["recipe_id"]

    LLM_PROVIDERS["_mock"] = MockProvider
    try:
        r = client.post(
            f"/recipes/{rid}/generate",
            json={"question": "How was the pneumonia treated?", "provider": "_mock",
                  "top_k": 3, "retriever": "mmr", "retriever_params": {"lambda_mult": 0.4}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["retriever"] == "mmr"           # retrieval strategy echoed
        assert body["answer"] == "ok"
        assert len(body["retrieved_chunks"]) == 3

        # Unknown retriever -> 400 (validated before generation).
        bad = client.post(
            f"/recipes/{rid}/generate",
            json={"question": "q", "provider": "_mock", "retriever": "nope"},
        )
        assert bad.status_code == 400
    finally:
        LLM_PROVIDERS.pop("_mock", None)


def test_frontend_client_in_process():
    """Drive the real frontend APIClient against the app with no live server.

    TestClient is itself an httpx.Client (with a sync<->ASGI bridge), so we can
    inject it as the frontend's HTTP client and exercise the exact FE<->BE code
    path in-process.
    """
    api = APIClient(base_url="http://testserver", client=client)

    assert api.health()["status"] == "ok"
    assert any(p["key"] == "manual" for p in api.list_parsers())
    assert any(c["key"] == "recursive" for c in api.list_chunkers())

    parsed = api.parse("discharge_summary_detailed.txt", _sample_bytes(), parser="manual")
    assert "DISCHARGE SUMMARY" in parsed["document"]["text"]

    chunked = api.chunk(parsed["document"], chunker="recursive", params={"size": 512, "overlap": 50})
    assert chunked["summary"]["chunk_count"] > 1

    assert any(e["key"] == "sentence_transformers" for e in api.list_embedders())
    assert any(v["key"] == "faiss" for v in api.list_vectorstores())
    assert any(rt["key"] == "naive_topk" for rt in api.list_retrievers())
    try:
        import sentence_transformers  # noqa: F401
        embedded = api.embed(chunked["chunks"], embedder="sentence_transformers",
                             params={"model_name": "all-MiniLM-L6-v2", "normalize": True})
        assert embedded["dimension"] == 384
        assert embedded["count"] == chunked["summary"]["chunk_count"]

        store = api.build_vectorstore(
            vectors=embedded["vectors"], chunks=chunked["chunks"], vectorstore="faiss",
            params={"index_type": "flat", "metric": "cosine"},
            embedder=embedded["embedder"],
            embed_params={"model_name": "all-MiniLM-L6-v2", "normalize": True},
            meta=embedded["meta"],
        )
        assert store["count"] == embedded["count"]
        hits = api.search(store["store_id"], "pneumonia treatment", top_k=2)
        assert len(hits["hits"]) == 2
        # The client threads a non-default retriever through to the backend.
        mmr_hits = api.search(store["store_id"], "pneumonia treatment", top_k=2,
                              retriever="mmr", retriever_params={"lambda_mult": 0.5})
        assert mmr_hits["retriever"] == "mmr" and len(mmr_hits["hits"]) == 2
    except ImportError:
        pass

    # Error surfaces as APIError with the backend's own message.
    raised = False
    try:
        api.chunk(parsed["document"], chunker="does_not_exist")
    except APIError as e:
        raised = "Unknown chunker" in str(e)
    assert raised


# --- Scoring endpoints (optional RAGAS-style metrics) ------------------------
# The scorer is injected via a FastAPI dependency, so a mock replaces it through
# app.dependency_overrides — no ragas or judge LLM needed to drive the endpoint.

from backend.api.routers.scoring import get_scorer
from backend.core.evaluator import BaseScorer  # noqa: E402


class _ApiMockScorer(BaseScorer):
    """Deterministic scorer double: 'good' answers score high; context_recall is
    None unless a ground truth was supplied (mirrors the real skip behaviour)."""

    def __init__(self, available=True, message="install ragas please"):
        self._available = available
        self._message = message

    def available(self):
        return self._available

    def unavailable_message(self):
        return self._message

    def score(self, samples):
        rows = []
        for s in samples:
            base = 0.9 if "good" in s.answer.lower() else 0.4
            rows.append({
                "faithfulness": base,
                "answer_relevancy": base,
                "context_precision": base,
                "context_recall": base if (s.ground_truth or "").strip() else None,
            })
        return rows


def _cell(recipe_id, provider, answer, ground_truth=None):
    return {
        "recipe_id": recipe_id, "provider": provider,
        "question": "How was the pneumonia treated?", "answer": answer,
        "contexts": ["treated with intravenous antibiotics"],
        "ground_truth": ground_truth,
    }


def test_score_status():
    r = client.get("/score/status")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["available"], bool)  # reflects whether ragas is installed
    assert body["metrics"] == [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    ]
    if not body["available"]:
        assert "ragas" in (body["message"] or "")  # friendly note when off


def test_score_empty_cells_422():
    r = client.post("/score", json={"cells": [], "sort_by": "faithfulness"})
    assert r.status_code == 422


def test_score_unknown_sort_by_400():
    # Validated before the availability gate, so it 400s regardless of ragas.
    r = client.post(
        "/score",
        json={"cells": [_cell("r1", "ollama", "good")], "sort_by": "nope"},
    )
    assert r.status_code == 400
    assert "Unknown sort_by" in r.json()["detail"]


def test_score_with_mock_scorer_ranks_cells():
    app.dependency_overrides[get_scorer] = lambda: _ApiMockScorer()
    try:
        r = client.post(
            "/score",
            json={
                "cells": [
                    _cell("recipe_001", "ollama", "a good grounded answer", "ref"),
                    _cell("recipe_002", "ollama", "a bad answer"),
                ],
                "sort_by": "faithfulness",
            },
        )
    finally:
        app.dependency_overrides.pop(get_scorer, None)

    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    # Per-cell scores carry the four metrics + a derived mean.
    first = body["cells"][0]["scores"]
    assert set(first) >= {*("faithfulness", "answer_relevancy",
                            "context_precision", "context_recall"), "mean"}
    # recipe_002 had no ground truth -> context_recall not computed.
    by_id = {c["recipe_id"]: c["scores"] for c in body["cells"]}
    assert by_id["recipe_002"]["context_recall"] is None
    assert by_id["recipe_001"]["context_recall"] == 0.9
    # Ranking: the 'good' answer wins, ranks are 1-based and ordered.
    ranking = body["ranking"]
    assert ranking[0]["recipe_id"] == "recipe_001" and ranking[0]["rank"] == 1
    assert ranking[1]["recipe_id"] == "recipe_002" and ranking[1]["rank"] == 2


def test_score_unavailable_returns_friendly_note():
    app.dependency_overrides[get_scorer] = lambda: _ApiMockScorer(
        available=False, message="install ragas please"
    )
    try:
        r = client.post(
            "/score",
            json={"cells": [_cell("r1", "ollama", "good")], "sort_by": "faithfulness"},
        )
    finally:
        app.dependency_overrides.pop(get_scorer, None)
    assert r.status_code == 200  # not an error — scoring is simply off
    body = r.json()
    assert body["available"] is False
    assert "ragas" in body["message"]
    assert body["cells"] == [] and body["ranking"] == []


def test_frontend_client_scoring():
    api = APIClient(base_url="http://testserver", client=client)
    status = api.score_status()
    assert isinstance(status["available"], bool)
    assert any(p["key"] == "ollama" for p in api.list_llm_providers())

    app.dependency_overrides[get_scorer] = lambda: _ApiMockScorer()
    try:
        resp = api.score_cells(
            [_cell("recipe_001", "ollama", "good", "ref"),
             _cell("recipe_002", "ollama", "bad")],
            sort_by="mean",
        )
    finally:
        app.dependency_overrides.pop(get_scorer, None)
    assert resp["available"] is True
    assert resp["ranking"][0]["recipe_id"] == "recipe_001"  # sorted by mean, best first


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
