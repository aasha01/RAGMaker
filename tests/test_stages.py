"""Fast contract tests for the default no-API-key strategies.

Not exhaustive correctness tests — their job (per CLAUDE.md "Testing") is to
give a learner editing a strategy immediate feedback that they didn't break
the interface: chunk bookkeeping stays truthful, vector dims line up, the
store round-trips, and the safety guards fire.

Run with:  pytest -q         (or)   python tests/test_stages.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.stages.parsers import REGISTRY as PARSERS
from backend.stages.parsers.base import ParsedDocument
from backend.stages.chunkers import REGISTRY as CHUNKERS
from backend.stages.vectorstores import REGISTRY as VECTORSTORES
from backend.stages.llm_providers import REGISTRY as LLM_PROVIDERS
from backend.stages.llm_providers.base import GenerationResult

SAMPLE_TXT = os.path.join("sample_data", "discharge_summary_detailed.txt")


def test_manual_parser_reads_txt():
    doc = PARSERS["manual"]().parse(SAMPLE_TXT)
    assert isinstance(doc, ParsedDocument)
    assert "DISCHARGE SUMMARY" in doc.text
    assert doc.metadata["char_count"] == len(doc.text)


def test_manual_parser_rejects_unknown_extension():
    parser = PARSERS["manual"]()
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tf:
        tf.write(b"hello")
        path = tf.name
    try:
        raised = False
        try:
            parser.parse(path)
        except ValueError:
            raised = True  # no silent fallback to another format
        assert raised
    finally:
        os.unlink(path)


def test_langchain_parser_reads_txt():
    if "langchain" not in PARSERS:
        print("SKIP test_langchain_parser_reads_txt (langchain not registered)")
        return
    try:
        doc = PARSERS["langchain"]().parse(SAMPLE_TXT)
        assert isinstance(doc, ParsedDocument)
        assert "DISCHARGE SUMMARY" in doc.text
        assert doc.metadata["char_count"] == len(doc.text)
        assert doc.metadata["engine"] == "plain"
    except ImportError as e:
        print(f"SKIP test_langchain_parser_reads_txt (langchain not installed: {e})")


def test_langchain_parser_rejects_unknown_extension():
    if "langchain" not in PARSERS:
        print("SKIP test_langchain_parser_rejects_unknown_extension (langchain not registered)")
        return
    parser = PARSERS["langchain"]()
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tf:
        tf.write(b"hello")
        path = tf.name
    try:
        raised = False
        try:
            parser.parse(path)
        except ValueError:
            raised = True  # no silent fallback to another format
        assert raised
    finally:
        os.unlink(path)


def test_llamaindex_parser_reads_txt():
    if "llamaindex" not in PARSERS:
        print("SKIP test_llamaindex_parser_reads_txt (llamaindex not registered)")
        return
    try:
        doc = PARSERS["llamaindex"]().parse(SAMPLE_TXT)
        assert isinstance(doc, ParsedDocument)
        assert "DISCHARGE SUMMARY" in doc.text
        assert doc.metadata["char_count"] == len(doc.text)
        assert doc.metadata["engine"] == "plain"
    except ImportError as e:
        print(f"SKIP test_llamaindex_parser_reads_txt (llamaindex not installed: {e})")


def test_llamaindex_parser_rejects_unknown_extension():
    if "llamaindex" not in PARSERS:
        print("SKIP test_llamaindex_parser_rejects_unknown_extension (llamaindex not registered)")
        return
    parser = PARSERS["llamaindex"]()
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tf:
        tf.write(b"hello")
        path = tf.name
    try:
        raised = False
        try:
            parser.parse(path)
        except ValueError:
            raised = True  # no silent fallback to another format
        assert raised
    finally:
        os.unlink(path)


def test_recursive_chunker_bookkeeping():
    text = "\n\n".join(f"Paragraph {i}. " + ("word " * 60) for i in range(6))
    doc = ParsedDocument(text=text, source="synthetic.txt", metadata={})
    size, overlap = 300, 40
    chunks = CHUNKERS["recursive"]().chunk(doc, size=size, overlap=overlap)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.position == i
        assert c.char_len == len(c.text)
        assert c.chunk_id == f"chunk_{i:04d}"
    # First chunk shares nothing; later chunks report a real, bounded overlap.
    assert chunks[0].overlap_with_prev == 0
    for c in chunks[1:]:
        assert 0 <= c.overlap_with_prev <= overlap
        # the claimed overlap is genuinely shared with the previous chunk's tail
    for prev, cur in zip(chunks, chunks[1:]):
        n = cur.overlap_with_prev
        if n:
            assert cur.text[:n] == prev.text[-n:]


def test_recursive_chunker_rejects_bad_overlap():
    doc = ParsedDocument(text="x" * 100, source="s", metadata={})
    raised = False
    try:
        CHUNKERS["recursive"]().chunk(doc, size=100, overlap=100)
    except ValueError:
        raised = True
    assert raised


def test_faiss_store_roundtrip_and_guards(tmp_path=None):
    from backend.stages.chunkers.base import Chunk

    rng = np.random.default_rng(0)
    dim = 16
    vecs = rng.standard_normal((5, dim)).astype(np.float32)
    chunks = [Chunk(f"chunk_{i:04d}", f"text {i}", "s", i, 6, 2, 0) for i in range(5)]

    store = VECTORSTORES["faiss"]()
    store.build(vecs, chunks, index_type="flat", metric="cosine", model_name="test-model")

    # A vector identical to row 2 should retrieve chunk 2 as the top hit.
    hits = store.search(vecs[2], top_k=1)
    assert hits[0].chunk.chunk_id == "chunk_0002"

    # Dimension guard.
    raised = False
    try:
        store.search(np.zeros((1, dim + 1), dtype=np.float32), top_k=1)
    except ValueError:
        raised = True
    assert raised

    # Length-mismatch guard at build time.
    raised = False
    try:
        store.build(vecs, chunks[:3], index_type="flat", metric="cosine")
    except ValueError:
        raised = True
    assert raised

    # Save/load round-trip preserves results and identity.
    out = tempfile.mkdtemp()
    store.save(out)
    reloaded = VECTORSTORES["faiss"]()
    reloaded.load(out)
    assert reloaded.model_name == "test-model"
    assert reloaded.search(vecs[2], top_k=1)[0].chunk.chunk_id == "chunk_0002"


def test_embedder_dimension():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_embedder_dimension (sentence-transformers not installed)")
        return
    from backend.stages.embedders import REGISTRY as EMBEDDERS

    emb = EMBEDDERS["sentence_transformers"](model_name="all-MiniLM-L6-v2", normalize=True)
    vecs = emb.embed(["hello world", "a second sentence"])
    assert vecs.shape == (2, emb.dimension)
    assert emb.dimension == 384


def test_embedder_truncate_dim_and_model_info():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_embedder_truncate_dim_and_model_info (no sentence-transformers)")
        return
    from backend.stages.embedders import REGISTRY as EMBEDDERS

    emb = EMBEDDERS["sentence_transformers"](
        model_name="all-MiniLM-L6-v2", normalize=True, truncate_dim=128
    )
    assert emb.default_dimension == 384
    assert emb.dimension == 128

    vecs = emb.embed(["hello world", "second"])
    assert vecs.shape == (2, 128)
    # Re-normalized after truncation -> still unit vectors.
    assert all(abs(float(np.linalg.norm(row)) - 1.0) < 1e-3 for row in vecs)

    info = emb.model_info()
    assert info["default_dimension"] == 384
    assert info["output_dimension"] == 128
    assert isinstance(info["max_seq_length_tokens"], int) and info["max_seq_length_tokens"] > 0
    assert info["param_count"] and info["param_count"] > 0
    assert info["approx_size_mb"] and info["approx_size_mb"] > 0
    assert info["dimension_customizable"] is True
    assert info["notes"]  # lossy-truncation warning surfaced, not hidden


def test_embedder_rejects_bad_truncate_dim():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("SKIP test_embedder_rejects_bad_truncate_dim (no sentence-transformers)")
        return
    from backend.stages.embedders import REGISTRY as EMBEDDERS

    raised = False
    try:
        EMBEDDERS["sentence_transformers"](model_name="all-MiniLM-L6-v2", truncate_dim=99999)
    except ValueError:
        raised = True  # no silent clamping to the native dimension
    assert raised


# --- LLM providers -----------------------------------------------------------
# Network is always mocked: Ollama via a patched httpx.post, OpenAI/Anthropic via
# fake SDK modules injected into sys.modules. These tests need no packages
# installed and never hit a real endpoint or need an API key.


def _install_fake_module(name: str, module):
    saved = sys.modules.get(name)
    sys.modules[name] = module
    return saved


def _restore_module(name: str, saved) -> None:
    if saved is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved


def test_llm_provider_registry_and_teaching_text():
    # Adding a provider = registry key + teaching text, read without instantiating.
    for key in ("ollama", "openai", "anthropic"):
        assert key in LLM_PROVIDERS
        cls = LLM_PROVIDERS[key]
        assert cls.name and cls.description  # user-facing teaching content
    assert LLM_PROVIDERS["ollama"].name.startswith("Ollama")  # the default


def test_ollama_provider_generate():
    import httpx
    from backend.stages.llm_providers.ollama_provider import OllamaProvider

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"response": "Pneumonia was treated with IV antibiotics.",
                    "prompt_eval_count": 42, "eval_count": 13}

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp()

    original = httpx.post
    httpx.post = fake_post
    try:
        res = OllamaProvider(model="llama3.2").generate("How was the pneumonia treated?")
    finally:
        httpx.post = original

    assert isinstance(res, GenerationResult)
    assert res.text.startswith("Pneumonia")
    assert res.input_tokens == 42 and res.output_tokens == 13
    assert res.cost_usd == 0.0  # local inference is genuinely free, not None
    assert res.latency_ms >= 0.0
    assert captured["url"].endswith("/api/generate")
    assert captured["payload"]["model"] == "llama3.2"
    assert captured["payload"]["stream"] is False


def test_ollama_provider_unreachable_raises_friendly():
    import httpx
    from backend.stages.llm_providers.ollama_provider import OllamaProvider

    def fake_post(url, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    original = httpx.post
    httpx.post = fake_post
    raised = False
    try:
        OllamaProvider().generate("hi")
    except RuntimeError as e:
        raised = "Ollama" in str(e)  # learner-readable, no silent fallback
    finally:
        httpx.post = original
    assert raised


def test_ollama_provider_model_not_found_lists_available():
    """When a model is not downloaded (404), error message lists available models."""
    import httpx
    from backend.stages.llm_providers.ollama_provider import OllamaProvider

    class FakeResp404:
        status_code = 404
        text = '{"error":"model not found"}'

    class FakeRespTags:
        status_code = 200
        text = ""

        def json(self):
            return {
                "models": [
                    {"name": "llama3.2:latest"},
                    {"name": "mistral:latest"},
                    {"name": "neural-chat:latest"},
                ]
            }

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["post_url"] = url
        return FakeResp404()

    def fake_get(url, timeout=None):
        captured["get_url"] = url
        return FakeRespTags()

    original_post = httpx.post
    original_get = httpx.get
    httpx.post = fake_post
    httpx.get = fake_get
    raised = False
    error_msg = ""
    try:
        OllamaProvider(model="llama3.1").generate("Hi")
    except RuntimeError as e:
        raised = True
        error_msg = str(e)
    finally:
        httpx.post = original_post
        httpx.get = original_get

    assert raised
    # Error message should mention the missing model and list available ones.
    assert "llama3.1" in error_msg
    assert "llama3.2:latest" in error_msg
    assert "mistral:latest" in error_msg
    assert "not found" in error_msg or "locally" in error_msg.lower()


def test_ollama_provider_model_not_found_no_models_available():
    """When no models are available locally, error explains how to download one."""
    import httpx
    from backend.stages.llm_providers.ollama_provider import OllamaProvider

    class FakeResp404:
        status_code = 404
        text = '{"error":"model not found"}'

    class FakeRespEmptyTags:
        status_code = 200

        def json(self):
            return {"models": []}

    def fake_post(url, json=None, timeout=None):
        return FakeResp404()

    def fake_get(url, timeout=None):
        return FakeRespEmptyTags()

    original_post = httpx.post
    original_get = httpx.get
    httpx.post = fake_post
    httpx.get = fake_get
    raised = False
    error_msg = ""
    try:
        OllamaProvider().generate("Hi")
    except RuntimeError as e:
        raised = True
        error_msg = str(e)
    finally:
        httpx.post = original_post
        httpx.get = original_get

    assert raised
    # Error message should guide the learner on how to download a model.
    assert "ollama pull" in error_msg.lower()


def test_openai_provider_generate_and_cost():
    from backend.stages.llm_providers.openai_provider import OpenAIProvider

    fake = types.ModuleType("openai")

    class FakeCompletions:
        def create(self, model, messages, max_tokens):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="Antibiotics."))],
                usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            )

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake.OpenAI = FakeOpenAI
    saved = _install_fake_module("openai", fake)
    try:
        res = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test").generate("q?")
    finally:
        _restore_module("openai", saved)

    assert res.text == "Antibiotics."
    assert res.input_tokens == 100 and res.output_tokens == 20
    expected = (100 * 0.15 + 20 * 0.60) / 1_000_000.0
    assert abs(res.cost_usd - expected) < 1e-12


def test_openai_provider_unknown_model_cost_none():
    from backend.stages.llm_providers.openai_provider import OpenAIProvider

    fake = types.ModuleType("openai")

    class FakeCompletions:
        def create(self, model, messages, max_tokens):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="x"))],
                usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=5),
            )

    fake.OpenAI = lambda api_key=None: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions()))
    saved = _install_fake_module("openai", fake)
    try:
        res = OpenAIProvider(model="gpt-future-unlisted", api_key="sk-test").generate("q?")
    finally:
        _restore_module("openai", saved)
    assert res.cost_usd is None  # withheld, never fabricated for an unlisted model


def test_openai_provider_missing_key_raises():
    from backend.stages.llm_providers.openai_provider import OpenAIProvider

    fake = types.ModuleType("openai")  # present, so the key check (not import) fires
    fake.OpenAI = lambda api_key=None: None
    saved = _install_fake_module("openai", fake)
    saved_env = os.environ.pop("OPENAI_API_KEY", None)
    raised = False
    try:
        OpenAIProvider(api_key=None).generate("hi")
    except RuntimeError as e:
        raised = "OPENAI_API_KEY" in str(e)
    finally:
        _restore_module("openai", saved)
        if saved_env is not None:
            os.environ["OPENAI_API_KEY"] = saved_env
    assert raised


def test_anthropic_provider_generate_and_cost():
    from backend.stages.llm_providers.anthropic_provider import AnthropicProvider

    fake = types.ModuleType("anthropic")

    class FakeMessages:
        def create(self, model, max_tokens, messages):
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="Antibiotics and rest.")],
                usage=types.SimpleNamespace(input_tokens=200, output_tokens=30),
            )

    class FakeAnthropic:
        def __init__(self, api_key=None):
            self.messages = FakeMessages()

    fake.Anthropic = FakeAnthropic
    saved = _install_fake_module("anthropic", fake)
    try:
        prov = AnthropicProvider(api_key="sk-ant-test")
        assert prov.model == "claude-opus-4-8"  # defaults to the latest Opus
        res = prov.generate("How was the pneumonia treated?")
    finally:
        _restore_module("anthropic", saved)

    assert res.text == "Antibiotics and rest."
    assert res.input_tokens == 200 and res.output_tokens == 30
    expected = (200 * 5.00 + 30 * 25.00) / 1_000_000.0  # claude-opus-4-8 pricing
    assert abs(res.cost_usd - expected) < 1e-12


def test_anthropic_provider_missing_key_raises():
    from backend.stages.llm_providers.anthropic_provider import AnthropicProvider

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda api_key=None: None
    saved = _install_fake_module("anthropic", fake)
    saved_env = os.environ.pop("ANTHROPIC_API_KEY", None)
    raised = False
    try:
        AnthropicProvider(api_key=None).generate("hi")
    except RuntimeError as e:
        raised = "ANTHROPIC_API_KEY" in str(e)
    finally:
        _restore_module("anthropic", saved)
        if saved_env is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_env
    assert raised


# --- RAG prompt assembly + generation orchestration --------------------------


def _fake_hits(n: int):
    from backend.stages.chunkers.base import Chunk
    from backend.stages.vectorstores.base import SearchResult

    return [
        SearchResult(
            chunk=Chunk(f"chunk_{i:04d}", f"passage text {i}", "doc.txt", i, 14, 3, 0),
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


def test_build_rag_prompt_is_explicit_and_ordered():
    from backend.core.generation import RAG_INSTRUCTION, build_rag_prompt

    hits = _fake_hits(3)
    prompt = build_rag_prompt("How was the pneumonia treated?", hits)

    # Instruction, every passage, and the question are all visible in the prompt.
    assert RAG_INSTRUCTION in prompt
    for i in range(3):
        assert f"passage text {i}" in prompt
        assert f"chunk_{i:04d}" in prompt  # provenance is shown, not hidden
    assert "How was the pneumonia treated?" in prompt
    # Passages appear in retrieval order and before the question.
    assert prompt.index("passage text 0") < prompt.index("passage text 1") < prompt.index("passage text 2")
    assert prompt.rindex("passage text 2") < prompt.index("Question:")


def test_build_rag_prompt_no_context():
    from backend.core.generation import build_rag_prompt

    prompt = build_rag_prompt("q?", [])
    assert "(no context was retrieved)" in prompt  # honest, not a crash


def test_generate_answer_uses_registered_provider_and_prompt():
    """generate_answer builds the prompt, dispatches to the chosen provider, and
    returns (prompt, result, model). Uses a mock provider registered in the
    REGISTRY — no network, no SDK, no API key."""
    from backend.core.generation import generate_answer
    from backend.stages.llm_providers.base import BaseLLMProvider

    seen = {}

    class MockProvider(BaseLLMProvider):
        name = "Mock"
        description = "test double"

        def __init__(self, model="mock-1"):
            self.model = model

        def generate(self, prompt, **kwargs):
            seen["prompt"] = prompt
            seen["kwargs"] = kwargs
            return GenerationResult(
                text="mock answer", latency_ms=1.5,
                input_tokens=11, output_tokens=7, cost_usd=0.0,
            )

    LLM_PROVIDERS["_mock"] = MockProvider
    try:
        prompt, result, model = generate_answer(
            "How was the pneumonia treated?", _fake_hits(2), "_mock",
            provider_params={"model": "mock-2"}, gen_params={"temperature": 0.1},
        )
    finally:
        LLM_PROVIDERS.pop("_mock", None)

    assert result.text == "mock answer"
    assert result.input_tokens == 11 and result.output_tokens == 7
    assert model == "mock-2"                     # constructor param honoured
    assert seen["kwargs"] == {"temperature": 0.1}  # gen_params passed through
    assert "passage text 0" in seen["prompt"] and "pneumonia" in seen["prompt"]


def test_generate_answer_unknown_provider_raises():
    from backend.core.generation import generate_answer

    raised = False
    try:
        generate_answer("q?", _fake_hits(1), "does_not_exist")
    except ValueError as e:
        raised = "Unknown LLM provider" in str(e)  # no silent fallback
    assert raised


# --- Retrievers (query-time stage, SPEC.md §6) -------------------------------
# These use a tiny deterministic bag-of-words FakeEmbedder + a real FAISS store,
# so they need no torch/sentence-transformers and no network. The cross-encoder
# and the HyDE LLM are mocked. Only the hybrid test needs rank-bm25 (skip-guarded).

from backend.stages.retrievers import REGISTRY as RETRIEVERS
from backend.stages.retrievers.base import BaseRetriever

# A fixed vocabulary. Note "ceftriaxone" is deliberately NOT here, so the embedder
# is blind to it — that is what lets the hybrid/BM25 test prove its lexical value.
_VOCAB = [
    "pneumonia", "antibiotics", "treated", "intravenous", "patient", "received",
    "rest", "recovery", "blood", "pressure", "medication", "hypertension",
    "physical", "therapy", "knee", "injury", "followup", "appointment",
    "review", "dosage", "treatment", "infection", "care",
]
_VOCAB_INDEX = {w: i for i, w in enumerate(_VOCAB)}


class _FakeEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocabulary — no torch."""

    model_name = "fake-bow"
    normalize = True

    def __init__(self):
        self.dimension = len(_VOCAB)

    def embed(self, texts):
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for r, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                j = _VOCAB_INDEX.get(tok)
                if j is not None:
                    rows[r, j] += 1.0
            norm = float(np.linalg.norm(rows[r]))
            if norm > 0:
                rows[r] /= norm
        return rows


_TOY_TEXTS = [
    "Pneumonia was treated with intravenous antibiotics ceftriaxone.",   # c0
    "Pneumonia was treated with intravenous antibiotics ceftriaxone.",   # c1 (identical to c0)
    "The patient received rest and recovery.",                           # c2
    "Blood pressure medication for hypertension.",                       # c3
    "Physical therapy for the knee injury.",                             # c4
]


def _toy_store_and_embedder():
    """Build a real FAISS cosine store over the toy corpus with the FakeEmbedder."""
    from backend.stages.chunkers.base import Chunk

    embedder = _FakeEmbedder()
    chunks = [
        Chunk(f"chunk_{i:04d}", t, "toy.txt", i, len(t), max(1, len(t.split())), 0)
        for i, t in enumerate(_TOY_TEXTS)
    ]
    vecs = embedder.embed([c.text for c in chunks])
    store = VECTORSTORES["faiss"]()
    store.build(vecs, chunks, index_type="flat", metric="cosine",
                model_name=embedder.model_name, normalize=True)
    return store, embedder, chunks


def test_retriever_registry_and_teaching_text():
    for key in ("naive_topk", "mmr", "hybrid", "rerank", "hyde"):
        assert key in RETRIEVERS
        cls = RETRIEVERS[key]
        assert issubclass(cls, BaseRetriever)
        assert cls.name and cls.description  # user-facing teaching content
    # The key-free options are flagged so; HyDE (needs an LLM) is not.
    assert RETRIEVERS["naive_topk"].no_api_key is True
    assert RETRIEVERS["mmr"].no_api_key is True
    assert RETRIEVERS["hyde"].no_api_key is False


def test_naive_topk_matches_plain_search():
    store, embedder, _ = _toy_store_and_embedder()
    hits = RETRIEVERS["naive_topk"]().retrieve(
        "antibiotics treated pneumonia", store=store, embedder=embedder, top_k=2
    )
    assert len(hits) == 2
    # The pneumonia/antibiotics chunk (c0 or its duplicate c1) is the top hit.
    assert hits[0].chunk.chunk_id in ("chunk_0000", "chunk_0001")
    # Same result the store would give directly (this IS the plain top-k).
    direct = store.search(embedder.embed(["antibiotics treated pneumonia"]), top_k=2)
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in direct]


def test_mmr_diversifies_away_from_duplicate():
    store, embedder, _ = _toy_store_and_embedder()
    q = "antibiotics treated pneumonia"

    # Naive top-2 returns BOTH identical pneumonia chunks (c0 and c1).
    naive = RETRIEVERS["naive_topk"]().retrieve(q, store=store, embedder=embedder, top_k=2)
    assert {h.chunk.chunk_id for h in naive} == {"chunk_0000", "chunk_0001"}

    # MMR with strong diversity keeps one of them and swaps the duplicate for a
    # different chunk — proving the redundancy penalty fired.
    mmr = RETRIEVERS["mmr"]().retrieve(
        q, store=store, embedder=embedder, top_k=2, lambda_mult=0.2
    )
    ids = [h.chunk.chunk_id for h in mmr]
    assert len(ids) == 2 and len(set(ids)) == 2
    # Seed is one of the (tied, identical) most-relevant chunks; FAISS breaks the
    # tie arbitrarily so either is valid.
    assert ids[0] in ("chunk_0000", "chunk_0001")
    assert not ({"chunk_0000", "chunk_0001"} <= set(ids))  # the duplicate was dropped

    # lambda_mult=1.0 is pure relevance == naive top-k ordering.
    pure = RETRIEVERS["mmr"]().retrieve(q, store=store, embedder=embedder, top_k=2, lambda_mult=1.0)
    assert {h.chunk.chunk_id for h in pure} == {"chunk_0000", "chunk_0001"}


def test_mmr_rejects_bad_lambda():
    store, embedder, _ = _toy_store_and_embedder()
    raised = False
    try:
        RETRIEVERS["mmr"]().retrieve("q", store=store, embedder=embedder, top_k=2, lambda_mult=2.0)
    except ValueError:
        raised = True  # no silent clamping
    assert raised


def test_hybrid_bm25_recovers_term_the_embedder_is_blind_to():
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        print("SKIP test_hybrid_bm25... (rank-bm25 not installed)")
        return
    store, embedder, _ = _toy_store_and_embedder()

    # 'ceftriaxone' is outside the embedder's vocabulary, so pure vector search
    # cannot find it — but BM25 matches the literal token in c0/c1.
    q = "ceftriaxone dosage"
    hybrid = RETRIEVERS["hybrid"]().retrieve(
        q, store=store, embedder=embedder, top_k=3, fusion="rrf"
    )
    ids = {h.chunk.chunk_id for h in hybrid}
    assert ids & {"chunk_0000", "chunk_0001"}  # the ceftriaxone chunk surfaced

    # Weighted fusion also works and returns results.
    weighted = RETRIEVERS["hybrid"]().retrieve(
        q, store=store, embedder=embedder, top_k=3, fusion="weighted", alpha=0.5
    )
    assert weighted
    # Bad fusion mode fails loudly.
    raised = False
    try:
        RETRIEVERS["hybrid"]().retrieve(q, store=store, embedder=embedder, top_k=3, fusion="nope")
    except ValueError:
        raised = True
    assert raised


def test_rerank_reorders_by_cross_encoder():
    from backend.stages.retrievers import rerank as rerank_mod

    store, embedder, _ = _toy_store_and_embedder()

    class FakeCrossEncoder:
        # Promotes whichever candidate mentions "therapy" (c4), regardless of the
        # vector ranking — proving the re-ranker overrides vector order.
        def predict(self, pairs):
            return [1.0 if "therapy" in cand.lower() else 0.0 for _q, cand in pairs]

    rerank_mod._MODEL_CACHE["fake-ce"] = FakeCrossEncoder()
    try:
        hits = RETRIEVERS["rerank"]().retrieve(
            "treatment options", store=store, embedder=embedder, top_k=2,
            model_name="fake-ce",
        )
    finally:
        rerank_mod._MODEL_CACHE.pop("fake-ce", None)

    assert hits[0].chunk.chunk_id == "chunk_0004"   # the 'therapy' chunk won
    assert hits[0].score == 1.0                     # cross-encoder's own score


def test_hyde_embeds_hypothetical_answer():
    from backend.stages.llm_providers.base import BaseLLMProvider, GenerationResult

    store, embedder, _ = _toy_store_and_embedder()
    seen = {}

    class MockHydeProvider(BaseLLMProvider):
        name = "MockHyde"
        description = "test double"

        def __init__(self, model="mock"):
            self.model = model

        def generate(self, prompt, **kwargs):
            seen["prompt"] = prompt
            # A hypothetical answer full of the answer's vocabulary.
            return GenerationResult(
                text="The patient was treated with intravenous antibiotics for pneumonia.",
                latency_ms=1.0, input_tokens=5, output_tokens=9, cost_usd=0.0,
            )

    LLM_PROVIDERS["_mock_hyde"] = MockHydeProvider
    try:
        hits = RETRIEVERS["hyde"]().retrieve(
            "infection care", store=store, embedder=embedder, top_k=2,
            provider="_mock_hyde",
        )
    finally:
        LLM_PROVIDERS.pop("_mock_hyde", None)

    # The original question ("infection care") shares no answer vocabulary, but the
    # hypothetical answer does — so retrieval lands on the pneumonia chunk.
    assert hits[0].chunk.chunk_id in ("chunk_0000", "chunk_0001")
    assert "infection care" in seen["prompt"]  # the question framed the hypothesis


def test_hyde_unknown_provider_raises():
    store, embedder, _ = _toy_store_and_embedder()
    raised = False
    try:
        RETRIEVERS["hyde"]().retrieve(
            "q", store=store, embedder=embedder, top_k=2, provider="does_not_exist"
        )
    except ValueError as e:
        raised = "Unknown LLM provider" in str(e)  # no silent fallback
    assert raised


# --- Scoring / evaluator (optional RAGAS-style metrics, ARCHITECTURE §6) ------
# Feature-gated behind a lazy `ragas` import. These tests use a mock scorer and
# never import ragas or call a judge LLM, so they run with nothing extra
# installed. The real RagasScorer's lazy-import guard is checked in isolation.

from backend.core.evaluator import (
    RAGAS_INSTALL_HINT,
    SCORE_METRICS,
    BaseScorer,
    RagasNotInstalled,
    RagasScorer,
    ScoreSample,
    mean_score,
    rank_cells,
    ragas_available,
    score_samples,
    with_mean,
)


class _MockScorer(BaseScorer):
    """Test double: returns preset per-sample score rows, no ragas involved."""

    def __init__(self, rows, available=True, message="unavailable"):
        self._rows = rows
        self._available = available
        self._message = message

    def available(self):
        return self._available

    def unavailable_message(self):
        return self._message

    def score(self, samples):
        return [dict(self._rows[i]) for i in range(len(samples))]


def test_score_metrics_and_ragas_gate():
    assert SCORE_METRICS == (
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    )
    assert isinstance(ragas_available(), bool)  # answered without importing ragas
    assert "pip install ragas" in RAGAS_INSTALL_HINT  # the friendly note
    # The exception carries the friendly note as its message.
    assert "ragas" in str(RagasNotInstalled())


def test_mean_score_and_with_mean():
    assert mean_score({"faithfulness": 1.0, "answer_relevancy": 0.0,
                       "context_precision": None, "context_recall": None}) == 0.5
    assert mean_score({m: None for m in SCORE_METRICS}) is None  # nothing to average
    enriched = with_mean({"faithfulness": 0.8, "answer_relevancy": 0.6,
                          "context_precision": 0.4, "context_recall": None})
    assert enriched["mean"] == round((0.8 + 0.6 + 0.4) / 3, 4)
    assert enriched["faithfulness"] == 0.8  # original values preserved


def test_rank_cells_orders_best_first_and_missing_last():
    cells = [
        {"recipe_id": "r1", "provider": "p", "scores": {"faithfulness": 0.4}},
        {"recipe_id": "r2", "provider": "p", "scores": {"faithfulness": 0.9}},
        {"recipe_id": "r3", "provider": "p", "scores": {"faithfulness": None}},
    ]
    ranked = rank_cells(cells, "faithfulness")
    assert [c["recipe_id"] for c in ranked] == ["r2", "r1", "r3"]  # best first, None last
    assert [c["rank"] for c in ranked] == [1, 2, 3]  # 1-based rank stamped


def test_score_samples_with_mock_scorer():
    rows = [
        {"faithfulness": 0.9, "answer_relevancy": 0.8,
         "context_precision": 0.7, "context_recall": 0.6},
        {"faithfulness": 0.5, "answer_relevancy": 0.5,
         "context_precision": 0.5, "context_recall": None},
    ]
    samples = [
        ScoreSample("q1", "a1", ["ctx"], ground_truth="ref"),
        ScoreSample("q2", "a2", ["ctx"]),
    ]
    out = score_samples(samples, scorer=_MockScorer(rows))
    assert out == rows  # aligned by index, passed through untouched


def test_score_samples_unavailable_raises_friendly():
    scorer = _MockScorer([], available=False, message="no ragas here")
    raised = False
    try:
        score_samples([ScoreSample("q", "a", [])], scorer=scorer)
    except RagasNotInstalled as e:
        raised = "no ragas here" in str(e)  # friendly, not a crash
    assert raised


def test_ragas_scorer_rejects_unknown_metric():
    raised = False
    try:
        RagasScorer(metrics=("faithfulness", "not_a_metric"))
    except ValueError:
        raised = True  # validated up front, no silent drop
    assert raised


def test_ragas_scorer_lazy_import_guard():
    # Constructing never imports ragas; scoring does. When ragas is absent, the
    # attempt raises the friendly RagasNotInstalled — no crash at import time.
    scorer = RagasScorer()
    assert "ragas" not in sys.modules  # not imported by construction
    if ragas_available():
        print("SKIP test_ragas_scorer_lazy_import_guard (ragas IS installed)")
        return
    assert scorer.available() is False
    raised = False
    try:
        scorer.score([ScoreSample("q", "a", ["c"])])
    except RagasNotInstalled:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
