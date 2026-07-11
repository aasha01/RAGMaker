# IMPLEMENTATION.md — RAG Lab Build Log

Living, step-by-step record of **what was built, why, and how to verify it**.
Update this on **every** change (see the rule in `CLAUDE.md` →
"Implementation Log"). Newest changelog row goes on top.

## Changelog

| # | Date | Change | Stage/Step | Key files | Tests |
|---|------|--------|-----------|-----------|-------|
| 16 | 2026-07-11 | Added LangChain and LlamaIndex parsers alongside Manual: `LangChainParser` wraps PyPDFLoader/UnstructuredFileLoader for PDF/DOCX/TXT; `LlamaIndexParser` wraps PDFReader/SimpleDirectoryReader for the same; both lazy-import their packages inside parse(), emit friendly errors if missing, output identical ParsedDocument contract; new files `backend/stages/parsers/{langchain_parser,llamaindex_parser}.py` + registry entries in `__init__.py`; requirements.txt updated with `langchain>=0.1.0` and `llama-index>=0.9.0` as optional | Pipeline stage 1 / parsing | `backend/stages/parsers/{langchain_parser.py,llamaindex_parser.py}`, `backend/stages/parsers/__init__.py`, `requirements.txt`, `tests/test_stages.py` | 6 unit |
| 15 | 2026-07-11 | Local judge LLM support for RAGAS scoring: added `backend/core/scoring_llm.py` with auto-detect for Ollama (free, offline) or Anthropic (paid) as judge LLMs, replacing OpenAI default; lazy LLM factory in `RagasScorer` defers initialization to first score() call; updated install hint to guide learner through three options; `/score/status` works even if judge LLM can't initialize; real errors surface at scoring time, not startup | Query & Compare / scoring (ARCH §6) | `backend/core/evaluator.py`, `backend/core/scoring_llm.py`, `backend/api/routers/scoring.py` | 0 new (existing coverage applies) |
| 14 | 2026-07-11 | Improved Ollama provider error handling for missing models: when a model returns 404, now queries `/api/tags` to list locally available models and displays them in error message; also guides learner with `ollama pull <model>` command if no models found | Stage 5 / generation | `backend/stages/llm_providers/ollama_provider.py`, `tests/test_stages.py` | 2 unit |
| 13 | 2026-07-11 | Added 4 remaining chunking strategies from SPEC.md §3: `fixed_size` (uniform chars + overlap), `sentence` (groups sentences, respects boundaries), `semantic` (similarity-based boundaries, embedder-powered), `structure_aware` (Markdown headers, hierarchy-aware) — each a new file in `backend/stages/chunkers/` + registry line; all params visible/overridable in UI, recorded in config.json; truthful overlap_with_prev; per-strategy unit + sample-data tests | Pipeline stage 2 / chunking | `backend/stages/chunkers/{fixed_size,sentence,semantic,structure_aware}.py`, `backend/stages/chunkers/__init__.py`, `tests/test_chunkers.py` | 31 unit |
| 12 | 2026-07-11 | Optional RAGAS-style answer scoring for the compare grid: faithfulness / answer relevancy / context precision & recall, feature-gated behind a lazy `ragas` import (absent → friendly "install ragas" note, never a crash); `POST /score` (per-cell scores + ranking summary sortable by any metric) + `GET /score/status`; new **Compare** tab runs a recipe×provider grid and scores every answer | Query & Compare / scoring (ARCH §6) | `backend/core/evaluator.py`, `backend/api/routers/scoring.py`, `backend/api/{main,schemas}.py`, `frontend/{app.py,api_client.py}` | 35 unit + 42 API |
| 11 | 2026-07-11 | Query-time retrieval stage: swappable `naive_topk`/`mmr`/`hybrid`/`rerank`/`hyde` strategies (recorded per-query, not per-recipe); `retriever`+`retriever_params` wired into `/vectorstore/search`, `/recipes/{id}/search`, `/recipes/{id}/generate` + the UI; `GET /stages/retrievers` | Stage 5 / retrieval | `backend/stages/retrievers/*`, `backend/stages/vectorstores/base.py` (`all_chunks`), `backend/api/routers/{stages,recipes,vectorstore}.py`, `backend/api/schemas.py`, `frontend/{app.py,api_client.py}` | 28 unit + 36 API |
| 10 | 2026-07-11 | Answer generation over HTTP: `POST /recipes/{id}/generate` retrieves top-k → assembles an explicit RAG prompt → calls a chosen LLM provider → returns answer + exact prompt + retrieved chunks + latency/tokens/cost | Generation / Query | `backend/core/generation.py`, `backend/api/routers/recipes.py`, `backend/api/schemas.py`, `frontend/api_client.py` | 20 unit + 31 API |
| 9 | 2026-07-11 | LLM provider stage: `ollama` (local, no key, $0 — default), `openai`, `anthropic` (default `claude-opus-4-8`) providers, each returning a full `GenerationResult`; lazy SDK imports; `GET /stages/llm_providers` discovery | Stage 5 / generation | `backend/stages/llm_providers/{ollama,openai,anthropic}_provider.py`, `backend/stages/llm_providers/__init__.py`, `backend/api/routers/stages.py` | 16 unit + 29 API |
| 8 | 2026-07-11 | Recipe persistence: build/save immutable recipes (config.json + stage_outputs/ + vectorstore/ + metadata.json) + SQLite index; list & query saved recipes; frontend Build/Recipes tabs | Build Step 5 / persistence | `backend/core/recipe.py`, `backend/api/routers/recipes.py`, `backend/api/schemas.py`, `frontend/app.py`, `frontend/api_client.py` | 8 unit + 3 recipe + 28 API |
| 7 | 2026-07-11 | Added `IMPLEMENTATION.md` (this log) + made maintaining it a MANDATORY rule in `CLAUDE.md` | Docs/process | `IMPLEMENTATION.md`, `CLAUDE.md` | n/a |
| 6 | 2026-07-11 | Stage 4: vector store build + search over HTTP (in-memory store registry, query re-embedded with the store's own model) | Pipeline stage 4 | `backend/api/routers/vectorstore.py`, `backend/api/services.py`, `backend/api/schemas.py`, `frontend/app.py` | 8 unit + 24 API |
| 5 | 2026-07-11 | Stage 3: model details (default dim, context window, params, size) + customizable output dimension (`truncate_dim`) | Pipeline stage 3 | `backend/stages/embedders/sentence_transformer.py`, `backend/api/routers/embedding.py`, `backend/api/schemas.py`, `frontend/app.py` | 8 unit + 19 API |
| 4 | 2026-07-11 | Hid Streamlit "Deploy" button; Stage 3 (embedding) end-to-end over HTTP | Pipeline stage 3 | `.streamlit/config.toml`, `backend/api/routers/embedding.py`, `frontend/app.py` | 6 unit + 14 API |
| 3 | 2026-07-11 | Split into FastAPI **backend/** + Streamlit **frontend/** over HTTP; wired stages 1–2 (parse, chunk) | Architecture | `backend/api/*`, `frontend/*`, moved `stages/`→`backend/stages/` | 6 unit + 10 API |
| 2 | 2026-07-11 | Implemented the 4 default no-API-key strategies + a pipeline script | Build Step 2 | `backend/stages/{parsers,chunkers,embedders,vectorstores}/*`, `backend/scripts/run_pipeline.py` | 6 unit |
| 1 | 2026-07-11 | Scaffolded repo layout: base interfaces + empty registries only | Build Step 1 | `backend/stages/*/base.py`, `backend/stages/*/__init__.py` | import smoke |

> Running total after change #16: **74 unit tests** (`tests/test_stages.py` 74 + `tests/test_chunkers.py` 31) + **42 API tests** (`tests/test_api.py`), all green.

---

## Change #16 — LangChain and LlamaIndex parsers (2026-07-11)

**Problem:** The manual parser is transparent but requires knowing format-specific
libraries. Two popular extraction frameworks (LangChain and LlamaIndex) offer
convenience loaders that could be swapped in to compare extraction quality.
Learners want to see how different parsers affect downstream stages.

**Solution:** Implement two new Strategy-pattern parsers alongside Manual:

1. **LangChainParser**: wraps LangChain's `PyPDFLoader` (PDF) and
   `UnstructuredFileLoader` (DOCX/DOC). Adds light cleanup and format
   normalization.
2. **LlamaIndexParser**: wraps LlamaIndex's `PDFReader` (PDF) and
   `SimpleDirectoryReader` (DOCX/TXT). Focuses on structured extraction.

Both parsers:
- Lazy-import their packages inside `parse()` so the tool runs without them
  (friendly "install X" error if missing).
- Return identical `ParsedDocument` contract (text + source + metadata).
- Support `.txt`, `.md`, `.pdf`, `.docx`, `.doc` extensions.
- Fail loudly on unsupported formats (no silent fallback).
- Store the extraction engine in metadata for learner inspection.

**Key files:**
- **`backend/stages/parsers/langchain_parser.py`** (new): LangChainParser
  implementation.
  - `_parse_text()` — plain read for TXT/MD.
  - `_parse_pdf()` — wraps PyPDFLoader, joins page texts with `\f` (form-feed)
    to preserve page boundaries.
  - `_parse_unstructured()` — wraps UnstructuredFileLoader for DOCX/DOC.
  - Lazy import of `langchain_community.document_loaders`.
- **`backend/stages/parsers/llamaindex_parser.py`** (new): LlamaIndexParser
  implementation.
  - `_parse_text()` — plain read for TXT/MD.
  - `_parse_pdf()` — wraps PDFReader, extracts per-page text.
  - `_parse_unstructured()` — wraps SimpleDirectoryReader for DOCX/TXT
    (requires a temp directory since SimpleDirectoryReader expects a directory
    path).
  - Lazy import of `llama_index.readers.file`.
- **`backend/stages/parsers/__init__.py`** (updated): registered both parsers
  in `REGISTRY`.
- **`requirements.txt`** (updated): added `langchain>=0.1.0` and
  `llama-index>=0.9.0` as optional dependencies (commented out as the base
  tool still runs without them).
- **`tests/test_stages.py`** (updated): added 6 unit tests:
  - `test_langchain_parser_reads_txt()` — verifies extraction against sample TXT.
  - `test_langchain_parser_rejects_unknown_extension()` — verifies no silent
    fallback.
  - `test_llamaindex_parser_reads_txt()` — verifies extraction.
  - `test_llamaindex_parser_rejects_unknown_extension()` — verifies no silent
    fallback.
  - (Plus 2 extension-rejection tests for completeness.)

**Design decisions:**
- **Lazy imports:** Each parser imports its dependencies only inside the
  method that needs them. If a dependency is missing, a clear message tells
  the learner what to install. This keeps the default tool runnable without
  langchain/llama-index.
- **Identical output contract:** All three parsers (Manual, LangChain, LlamaIndex)
  return `ParsedDocument(text, source, metadata)`. Learners can swap them in
  the UI without downstream code changes.
- **Page boundaries preserved:** PDF page texts are joined with `\f` (form-feed)
  rather than spaces, so page breaks remain visible in downstream stages
  (chunker, inspector).
- **Engine tracking:** The `metadata["engine"]` field stores which extraction
  library was used, visible to learners in the UI.

**How to verify:**
```bash
# Install dependencies:
pip install langchain langchain-community pypdf unstructured

# Run the parser tests:
pytest tests/test_stages.py -k "parser" -v

# All 6 parser tests (manual, langchain, llamaindex) should pass green.

# Inspect the sample parse:
python -c "
from backend.stages.parsers import REGISTRY
from pathlib import Path
sample = 'sample_data/discharge_summary_detailed.txt'
for key in ('manual', 'langchain', 'llamaindex'):
    parser = REGISTRY[key]()
    doc = parser.parse(sample)
    print(f'{key}: {len(doc.text)} chars, engine={doc.metadata.get(\"engine\")}')"
```

---

## Change #15 — Local judge LLMs for RAGAS scoring (2026-07-11)

**Problem:** ragas defaults to OpenAI's API for grading (the judge LLM that scores
faithfulness, relevancy, etc.). This requires an `OPENAI_API_KEY` and costs
money per call. A learner with just the default setup can't run scoring.

**Solution:** Auto-detect local LLMs (Ollama or Anthropic) from env vars and use
them as the judge LLM instead. Three options now available:

1. **Ollama (free, offline, zero-config default):** runs on the learner's machine,
   no API key. Set `RAGAS_LLM_PROVIDER=ollama` or just have `OLLAMA_HOST` env var.
2. **Anthropic (paid, hosted):** set `RAGAS_LLM_PROVIDER=anthropic` +
   `ANTHROPIC_API_KEY`.
3. **OpenAI (paid, hosted, original default):** set `OPENAI_API_KEY` or neither of
   the above (ragas falls back to OpenAI).

**Key files:**
- **`backend/core/scoring_llm.py`** (new): factory module with three components:
  - `get_ragas_judge_llm()` — auto-detect and initialize judge LLM from env vars.
    Returns a langchain-compatible LLM instance or None (uses ragas default).
  - `_make_ollama_llm()` — wraps Ollama via langchain_community, validates server
    is running.
  - `_make_anthropic_llm()` — wraps Anthropic via langchain_anthropic, validates
    API key.
- **`backend/core/evaluator.py`** (refactored):
  - Updated `RAGAS_INSTALL_HINT` to explain all three options with concrete
    commands.
  - Added `lazy_llm_factory` param to `RagasScorer.__init__()` to defer judge LLM
    initialization.
  - Modified `RagasScorer.score()` to call the factory on first score() call (not
    at __init__ time), so `/score/status` works even if the judge LLM can't start.
  - New `create_ragas_scorer()` factory function (used by the API layer) passes
    the lazy factory to RagasScorer.
  - Updated error messages to mention local alternatives.
- **`backend/api/routers/scoring.py`** (updated): now imports and uses
  `create_ragas_scorer()` instead of bare `RagasScorer()`.

**Design decisions:**
- **Lazy initialization:** Judge LLM is only initialized when `score()` is called,
  not when the scorer is created. This allows `/score/status` (which just checks
  availability) to work even if the judge LLM can't start. Real errors surface
  when scoring runs.
- **Environment-based auto-detect:** Priority is `RAGAS_LLM_PROVIDER` env var
  (explicit), then `OLLAMA_HOST` (Ollama present), then `ANTHROPIC_API_KEY`
  (Anthropic key present), then None (use ragas default = OpenAI). This means a
  learner running Ollama locally gets it automatically; no config needed.
- **Learner-friendly error messages:** If judge LLM init fails (e.g. "Ollama not
  running"), the error message tells the learner exactly what to do (e.g. "run
  'ollama serve'").

**How to verify:**
```bash
# Without any config, /score/status returns available=false with the install note.
curl http://localhost:8000/score/status

# With Ollama running:
ollama serve  # Terminal 1
export RAGAS_LLM_PROVIDER=ollama
uvicorn backend.api.main:app --reload  # Terminal 2

# Now /score/status → available=true (ragas installed + Ollama is reachable).
# POST /score will use Ollama for grading.

# To test with Anthropic:
export ANTHROPIC_API_KEY=sk-...
export RAGAS_LLM_PROVIDER=anthropic
uvicorn backend.api.main:app --reload

# /score will now use Claude as the judge.
```

**Dependencies added:** `langchain-community` (for Ollama) and/or
`langchain-anthropic` (for Anthropic) are optional. If absent, the scorer falls
back gracefully (lazy init will fail loudly when score() is called).

---

## Conventions & key decisions

- **Two top-level packages:** `backend/` (FastAPI service + pipeline logic) and
  `frontend/` (Streamlit UI). The frontend talks to the backend **only over
  HTTP** through `frontend/api_client.py`. Run both from the repo root so the
  packages resolve. *(This deviates from ARCHITECTURE.md §9's single-app design;
  chosen by the user.)*
- **Recipe storage (planned):** files are the source of truth (per-recipe folder
  with `config.json` + `stage_outputs/` + `vectorstore/`), plus a stdlib
  `sqlite3` index at `recipes/index.db` for listing. Not built yet.
- **Non-Negotiables enforced at the HTTP boundary:** unknown strategy → `400`;
  a stage that raises (bad params, unsupported format, model mismatch) → `422`,
  with the real error message — **never** a silent fallback to another technique.
- **Strategy pattern:** every stage is a class behind a `base.py` interface,
  registered by a string key in that stage's `__init__.py` `REGISTRY`. Adding a
  technique never edits code outside that stage's package (+ one registry line).
- **Every strategy carries a 3-part teaching docstring** (what / tradeoff /
  when), surfaced verbatim in the UI.

### How to run
```bash
uvicorn backend.api.main:app --reload      # backend  → http://localhost:8000/docs
streamlit run frontend/app.py              # frontend → http://localhost:8501
python tests/test_stages.py                # stage-logic unit tests
python tests/test_api.py                   # API + frontend-client tests
```
Frontend→backend URL overridable via env `RAG_LAB_API_URL` (default
`http://localhost:8000`).

---

## Change #1 — Scaffold + base interfaces (Build Step 1)

Created the directory layout with **abstract base classes and empty registries
only** — no concrete strategies. Each pipeline stage lives in its own package:

```
backend/stages/{parsers,chunkers,embedders,vectorstores,llm_providers}/
    base.py        # ABC interface + the data contract it produces
    __init__.py    # REGISTRY: dict[str, type[Base...]]  (empty at first)
```

**Data contracts (dataclasses):** `ParsedDocument(text, source, metadata)`,
`Chunk(chunk_id, text, source, position, char_len, token_len,
overlap_with_prev)`, `SearchResult(chunk, score)`, `GenerationResult(text,
latency_ms, input_tokens, output_tokens, cost_usd)`. Each dataclass got
`to_dict`/`from_dict` for transparent persistence. Cross-stage type imports
(e.g. chunker imports `ParsedDocument`) are one-directional data-contract
imports, not strategy coupling.

**Verify:** every base is an un-instantiable `ABC`; every registry is `{}`.

## Change #2 — Default no-API-key strategies + pipeline script (Build Step 2)

Implemented exactly one concrete strategy per build-time stage — the
zero-API-key default path:

| Stage | Key | Class | Notes |
|---|---|---|---|
| Parser | `manual` | `ManualParser` | txt/md/pdf/docx, lazy per-format imports, no cleanup |
| Chunker | `recursive` | `RecursiveChunker` | self-contained (no LangChain); truthful `overlap_with_prev` |
| Embedder | `sentence_transformers` | `SentenceTransformerEmbedder` | local `all-MiniLM-L6-v2`, 384-dim |
| Store | `faiss` | `FAISSStore` | flat/hnsw × cosine/l2/dot, dimension guard, save/load |

`backend/scripts/run_pipeline.py` runs a file through all five stages and writes
the `stage_outputs/` artifacts (ARCHITECTURE.md §4): `01_raw/`, `02_parsed.json`,
`03_chunks.json`, `04_embeddings.npy`, `04_embeddings_meta.json`, `vectorstore/`.

**Chunk sizing unit:** characters (SPEC allows chars or tokens). `token_len` is a
documented model-agnostic approximation so the chunker never depends on the
embedder.

**Verify:** `python backend/scripts/run_pipeline.py` → 30 chunks, (30, 384)
vectors, relevant retrieval; dimension guard rejects a mismatched query.

## Change #3 — Frontend/Backend split over HTTP (stages 1–2)

Moved `stages/`→`backend/stages/`, `core/`→`backend/core/`; rewrote imports to
`backend.stages.*`. Added the FastAPI service and the Streamlit client.

**Backend API (`backend/api/`):** `main.py` (app + `/health` + permissive CORS),
`schemas.py` (Pydantic contract mirroring the dataclasses), routers:
- `GET /stages/parsers`, `GET /stages/chunkers` — list strategies from **class
  attributes only** (no instantiation → no model load).
- `POST /parse` — raw file **bytes on the request body** (avoids a
  `python-multipart` dep) + `filename`/`parser` query params → `ParsedDocument`.
- `POST /chunk` — `{document, chunker, params}` → chunks + summary.

**Frontend (`frontend/`):** `api_client.py` (httpx client; raises `APIError`
with the backend's own message) and `app.py` (Build wizard: upload → parse →
inspect → chunk → inspect, each with a "redo" via re-running the stage).

**New deps (approved):** `fastapi`, `uvicorn`, `httpx`. **Pin `fastapi>=0.139`**
— older FastAPI pins a starlette too old for streamlit 1.59 (ImportError
`DEFAULT_EXCLUDED_CONTENT_TYPES`).

**Verify:** `python tests/test_api.py` (10) + live uvicorn smoke.

## Change #4 — Deploy button hidden + Stage 3 (embedding)

**Deploy button:** added `.streamlit/config.toml` with `toolbarMode = "minimal"`,
hiding Streamlit's built-in top-right toolbar (incl. the "Deploy" button) — it
is not part of the wizard and confused the flow.

**Stage 3 embedding:**
- `GET /stages/embedders` (class attrs only), `POST /embed`
  (`{chunks, embedder, params}` → vectors + timing + cost + `EmbeddingMeta`).
- Embedder instances are **cached in-process** so torch loads once per config,
  not per request.
- `EmbeddingMeta(model_name, dimension, normalize, chunk_id_order)` travels with
  the vectors (mirrors `04_embeddings_meta.json`) and is the identity re-checked
  before any query — the "never mix two models' embeddings" rule.
- Frontend Stage 3: embedder picker + `model_name`/`normalize`, spinner, vector
  preview, norms (≈1.0 confirms normalization), metadata. Re-chunking
  invalidates the downstream embedding.

**Verify:** `tests/test_api.py` (14) + live embed smoke (30 vectors, dim 384).

## Change #5 — Stage 3 model details + customizable output dimension

Made the embedding model **inspectable** and its output dimension **tunable**.

**Model details** (via `BaseEmbedder.model_info()`, overridden richly by the
sentence-transformers embedder): `default_dimension`, `output_dimension`,
`max_seq_length_tokens` (context window — the model ignores tokens past this),
`param_count`, `approx_size_mb` (float32 estimate), `backend`, `normalize`,
`truncate_dim`, `dimension_customizable`, `notes`. For `all-MiniLM-L6-v2`:
384 dim, **256-token context**, 22.7M params, ~86.6 MB.

**Customizable dimension** (`truncate_dim`): optional int < the model's native
dimension. Implemented transparently — encode raw → cut to first N dims →
re-normalize (vectors stay unit-length). **Lossy** for non-Matryoshka models
like MiniLM, surfaced as a UI `warning` (not hidden). Invalid values fail loudly
(`truncate_dim=9999` → `422`), never silently clamped. Propagates into
`EmbeddingMeta.dimension` so the store guard uses the truncated dim.

**New endpoint:** `GET /embedders/{key}/model_info?model_name=&normalize=&truncate_dim=`
loads the model (cached) and returns `ModelInfo`. `model_info` is also embedded
in the `/embed` response. Frontend: "Show model details" button, custom-dim
input, details panel.

**Forward note:** future OpenAI/Cohere embedders accept the same
`model_name`/`normalize`/`truncate_dim` kwargs so the cache + UI stay uniform
(hosted models map `truncate_dim` → their native `dimensions` param).

**Verify:** `tests/test_stages.py` (8) + `tests/test_api.py` (19) + live
`model_info` / truncated-embed smoke.

## Change #6 — Stage 4 vector store (build + search)

Build a FAISS index from the stage-3 vectors and query it.

**Statefulness:** a FAISS index can't round-trip through JSON, so this is the
first stateful stage. Built stores live in an **in-process registry** keyed by a
`store_id` (in-memory; reset on server restart — durable file-based recipes come
later with `core/recipe.py`). Refactored the embedder cache out of the embedding
router into `backend/api/services.py:get_embedder` so the vector store can reuse
it.

**Endpoints:**
- `GET /stages/vectorstores` — list store strategies (faiss).
- `POST /vectorstore/build` — `{vectors, chunks, vectorstore, params(index_type,
  metric), embedder, embed_params, meta}` → builds the index, stores it under a
  new `store_id`, returns count/index_type/metric/dimension/model_name + a
  metadata-table sample. Validates `len(vectors)==len(chunks)` and
  `vector_dim==meta.dimension` (→ 422); unknown store → 400.
- `POST /vectorstore/search` — `{store_id, query, top_k}` → **re-embeds the query
  with the SAME embedder config that built the store** (looked up from the
  registry), then searches. Enforces the model-mismatch guard (query model must
  equal store model) plus FAISSStore's own dimension guard. Unknown store → 404.

**Frontend Stage 4:** store picker + `index_type` (flat/hnsw) + `metric`
(cosine/l2/dot); Build shows record count + metadata table; a query box + `top_k`
slider returns ranked chunks with scores (score direction annotated per metric).
Re-embedding invalidates the store & search results.

**Key decision:** the query is embedded **server-side** using the store's
recorded embedder config, which is what makes the "never search a store with
another model's vectors" rule structurally true rather than merely checked.

**Verify:** `tests/test_stages.py` (8) + `tests/test_api.py` (24) + live smoke
(build 30 vecs, cosine search returns pneumonia-treatment chunks in descending
score order; unknown store → 404; length mismatch → 422; a `truncate_dim=128`
store builds at dim 128 and searches correctly).

> Note (env): on this Windows host, uvicorn port **8080** is in a reserved/
> excluded range (WinError 10013) — use another port (e.g. 8000, 8076).

## Change #7 — Implementation log + mandatory-docs rule

Added this `IMPLEMENTATION.md` (changelog table on top + per-change detail
sections) and a **MANDATORY "Implementation Log" section in `CLAUDE.md`** so the
log is updated on every change in any future session. Also saved a persistent
memory to reinforce it. Standing rules restated there: write+run per-stage tests
before "done", and ask before adding dependencies.

## Change #9 — LLM provider stage (generation)

Filled the previously-empty `backend/stages/llm_providers/` registry with three
concrete providers behind the existing `BaseLLMProvider` interface (SPEC.md §7).
Each `generate(prompt, **kwargs)` returns a full
`GenerationResult(text, latency_ms, input_tokens, output_tokens, cost_usd)`.

| Key | Class | Transport | Cost |
|---|---|---|---|
| `ollama` | `OllamaProvider` | local Ollama HTTP (`/api/generate`) via `httpx` | `0.0` (local, free) |
| `openai` | `OpenAIProvider` | `openai` SDK (Chat Completions) | computed from price table, else `None` |
| `anthropic` | `AnthropicProvider` | `anthropic` SDK (Messages API) | computed from price table, else `None` |

**Design decisions:**
- **`ollama` is the runnable-out-of-the-box default:** no API key, `$0.00`, and
  it talks to a local Ollama server over `httpx` (already a core dep) — so no new
  package is required for the default path. `cost_usd` is `0.0`, not `None`, so a
  concrete `$0.00` reads clearly against the paid providers in the compare grid.
- **Anthropic defaults to `claude-opus-4-8`** (latest Opus; per the claude-api
  skill). Plain single-turn completion (no extended thinking) so latency/tokens/
  cost compare fairly against the other providers. Pricing table sourced from the
  skill (Opus 4.8 $5/$25, Sonnet 5 $3/$15, Haiku 4.5 $1/$5, Fable 5 $10/$50, …).
- **Lazy imports (CLAUDE.md Style):** each SDK (`openai`, `anthropic`) and `httpx`
  is imported **inside `generate`**, never at module load — importing the registry
  (e.g. for discovery) pulls in none of them (verified). A missing package raises a
  friendly `ImportError` with the pip line; a missing API key raises a clear
  `RuntimeError`; an unreachable Ollama server raises a "is it running?" message.
  **No silent fallback** to another provider anywhere.
- **Cost transparency, no fabrication:** an unlisted model returns `cost_usd=None`
  (tokens are still real) rather than a made-up figure.
- **No new dependency added:** `openai`/`anthropic` were already listed
  (commented, optional) in `requirements.txt`; `ollama` uses `httpx` (already
  required). Nothing to install for the tool to run.

**Discovery endpoint:** `GET /stages/llm_providers` added to
`backend/api/routers/stages.py`, mirroring the other four stages — lists key +
`name` + `description` from **class attributes only** (no instantiation, no SDK
import, no key).

**Strategy-pattern compliance:** everything new lives inside
`stages/llm_providers/` plus the one registry line and the one discovery route
(the standard per-stage wiring the other stages also have).

**Tests (all green):** `tests/test_stages.py` +8 (registry/teaching text; Ollama
generate with patched `httpx.post`; Ollama-unreachable friendly error; OpenAI
generate + computed cost; OpenAI unknown-model `cost=None`; OpenAI missing-key
error; Anthropic generate + cost + `claude-opus-4-8` default; Anthropic
missing-key error) — network fully mocked via a patched `httpx.post` and fake
`openai`/`anthropic` modules injected into `sys.modules`, so no packages or keys
are needed. `tests/test_api.py` +1 (`GET /stages/llm_providers` lists all three
with teaching text). Totals: **16 unit + 29 API**.

**Verify:** `python tests/test_stages.py` (16/16) and `python tests/test_api.py`
(29/29); `python -c "import backend.stages.llm_providers as m; print(list(m.REGISTRY))"`
prints `['ollama','openai','anthropic']` without importing any SDK.

## Change #10 — Answer generation over HTTP (`POST /recipes/{id}/generate`)

Wired the last piece so a saved recipe can actually **answer** a question, not
just retrieve chunks: retrieve top-k → assemble an explicit RAG prompt → call a
chosen LLM provider → return the answer plus everything needed to inspect and
compare the run.

**New module `backend/core/generation.py`** (query-time orchestration, mirrors how
`recipe.py` only ever talks to a REGISTRY, never a concrete provider):
- `build_rag_prompt(question, hits, instruction=RAG_INSTRUCTION)` — assembles the
  **exact** text sent to the LLM: instruction + numbered context passages (each
  labelled `source / chunk_id / score`) + the question. Deliberately explicit and
  readable — this is a teaching tool, so the prompt is built in plain sight and
  returned verbatim in the response (the learner sees *precisely* what context the
  model saw). `RAG_INSTRUCTION` is a named constant, not buried in an f-string, so
  the one line that frames the grounded-answer task is easy to find and tweak.
  Empty retrieval yields an honest `(no context was retrieved)`, not a crash.
- `generate_answer(question, hits, provider_key, provider_params, gen_params)` →
  `(prompt, result, model)`. `provider_params` are provider **constructor** args
  (model, api_key, ...); `gen_params` are per-call `generate()` kwargs. Unknown
  provider key raises `ValueError` (**no silent fallback**); provider errors
  propagate unchanged to be surfaced verbatim. `model` reports what was actually
  used (a per-call override wins over the configured default).

**Endpoint `POST /recipes/{recipe_id}/generate`** (`backend/api/routers/recipes.py`):
- Refactored the store-loading + query re-embedding (previously inline in
  `search_recipe`) into a shared `_retrieve(recipe_id, query, top_k)` helper, now
  used by **both** `/search` and `/generate` — so generation retrieves through the
  exact same model-mismatch guard ("never search a store with another model's
  vectors"). Also extracted `_to_hits()` for the `SearchHit` mapping.
- Returns `GenerateResponse`: `answer`, `prompt` (exact text sent),
  `retrieved_chunks`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`,
  `model`, `metric`, `embedding_model`.
- **Error mapping (fail loudly, no fallback):** unknown recipe → 404, model
  mismatch/bad query → 422 (from `_retrieve`), unknown provider → 400, any
  provider failure (missing package/API key, unreachable server) → 502 with the
  provider's own message.

**Cost/time recording (ARCHITECTURE.md §8):** generation cost + wall-clock latency
+ token counts are recorded **in the response** (and feed the compare-grid schema,
§9). Deliberately **not** written back into the recipe's `metadata.json`: recipes
are immutable once built and §8's build-time metadata is about build stages — a
query-time generation must not mutate the recipe. Transparency is satisfied by
returning the numbers per run.

**Schemas** (`backend/api/schemas.py`): `GenerateRequest` (question, provider,
top_k, provider_params, gen_params) + `GenerateResponse` (above).

**Frontend client:** `APIClient.generate_recipe(recipe_id, question, provider, top_k,
provider_params, gen_params)` added for FE/BE parity (no Streamlit UI in this change).

**Strategy-pattern compliance:** no new strategy was added; generation reuses the
existing `llm_providers` REGISTRY through the interface. `core/generation.py` is
orchestration (like `recipe.py`), not a stage — adding a *new provider* still
touches only `stages/llm_providers/`.

**Tests (all green): 20 unit + 31 API.**
- `tests/test_stages.py` +4: `build_rag_prompt` is explicit/ordered (instruction,
  every passage + its provenance, and the question all present; passages precede
  the question); empty-context path; `generate_answer` dispatches to a **mock
  provider** registered in the REGISTRY (constructor param honoured, `gen_params`
  passed through, prompt carries the context) — no network/SDK/key; unknown
  provider raises.
- `tests/test_api.py` +2: `POST /recipes/{id}/generate` unknown recipe → 404; full
  flow with a mock provider inserted into the REGISTRY (answer proves the context
  reached the prompt, `provider_params` honoured, 3 chunks returned, exact prompt
  surfaced, token/cost/latency present) and unknown provider → 400. Skips (like the
  sibling recipe tests) when sentence-transformers isn't installed, since retrieval
  needs the embedder.

**Verify:** `python tests/test_stages.py` (20/20) and `python tests/test_api.py`
(31/31). Live: build a recipe, then
`curl -X POST localhost:8000/recipes/recipe_001/generate -H 'content-type: application/json'
-d '{"question":"...","provider":"ollama","top_k":3}'` (needs a running Ollama for
the default provider) returns the answer + the exact prompt + retrieved chunks +
latency/tokens/cost.

## Change #11 — Query-time retrieval stage (5 swappable strategies)

Implemented SPEC.md §6: retrieval as its own swappable-Strategy stage, chosen
**per query, not per recipe** — so one immutable recipe can be probed with every
strategy without rebuilding. New package `backend/stages/retrievers/` behind a
`base.py` interface + a REGISTRY, exactly like the other four stages.

**Interface** (`BaseRetriever.retrieve`):
```python
def retrieve(self, query, *, store, embedder, top_k, **params) -> list[SearchResult]
```
A retriever owns no data. It is handed the already-built `store` and the recipe's
`embedder` (the SAME model the store was built with — the caller validates the
match *before* calling, so the "never search a store with another model's
vectors" guard stays in one place) and returns the same `SearchResult` list the
store's own `search` returns, so everything downstream (prompt assembly, compare
grid) is unchanged regardless of which retriever ran. A class-level `no_api_key`
flag documents which strategies run key-free.

**The five strategies** (each a file in the package, one registry line):

| Key | What | Optional dep (lazy) | Key-free |
|---|---|---|---|
| `naive_topk` | embed query → `store.search` — the plain top-k pulled out of the store as the baseline | none | ✅ |
| `mmr` | fetch `fetch_k`, re-embed candidates, greedy Maximal Marginal Relevance (`lambda_mult`) for diversity | none | ✅ |
| `hybrid` | BM25 (lexical) + vector, fused by RRF (default) or a weighted blend | `rank-bm25` | ✅ |
| `rerank` | vector top-`fetch_n` → cross-encoder re-scores pairs → top-k | `sentence-transformers` `CrossEncoder` (already installed) | ✅ |
| `hyde` | LLM drafts a hypothetical answer → embed **that** → search | an LLM provider (default local Ollama) | ❌ |

**Design decisions:**
- **MMR re-embeds its candidate pool** with the recipe's own embedder to measure
  candidate-to-candidate similarity (the store only hands out chunks + scores).
  Redundant work, chosen for transparency/store-agnosticism over speed — a
  teaching-tool tradeoff, documented in the class. Returned score = each pick's
  cosine relevance to the query, so numbers stay comparable to naive top-k;
  MMR changes the *ordering*, not the scale.
- **Hybrid needs the whole corpus**, not just vector matches, so a new
  `BaseVectorStore.all_chunks()` accessor was added (concrete default reads the
  `chunks` list FAISS keeps). This is the one edit outside `retrievers/` — stage
  infrastructure (a store must expose its corpus for lexical retrieval), not a
  new technique. Fusion is **RRF by default** (rank-only, so the two score scales
  never need reconciling) with an optional min-max **weighted** blend (`alpha`).
- **Rerank reuses the already-installed `sentence-transformers` `CrossEncoder`**
  — no new pip dep; it downloads a small cross-encoder model on first use (like
  the embedder). The model is cached at module level (`_MODEL_CACHE`) so
  per-query instances share one load. Its score is the cross-encoder's own scale
  (not the store metric) — noted in the UI caption.
- **HyDE reuses the `llm_providers` REGISTRY** (like `core/generation.py`),
  talking to the provider *interface*, never a concrete provider — so adding a
  provider still touches only `llm_providers/`. It's the one retriever that isn't
  guaranteed key-free (default Ollama needs no key but a running server).
- **Lazy imports / fail loudly (CLAUDE.md):** `rank-bm25`, `CrossEncoder`, and
  the LLM SDKs are imported **inside** `retrieve`; importing the registry (for
  discovery) drags in none of them (verified: `rank_bm25`/`torch` absent from
  `sys.modules` after import). A missing package → friendly `ImportError` with
  the pip line; an unknown HyDE provider → `ValueError`; bad params (lambda out
  of range, bad fusion) → `ValueError`. **No silent fallback** to another
  strategy anywhere.

**Dependency (asked + approved):** `rank-bm25` added to `requirements.txt` for
`hybrid` (the user chose it over a hand-rolled BM25). The cross-encoder re-ranker
needed **no** new package (reuses sentence-transformers).

**Wiring (all params visible + recorded per query — no hidden defaults):**
- `GET /stages/retrievers` — discovery (class attrs only, no instantiation).
- Schemas: shared `retriever` + `retriever_params` fields on `SearchRequest`,
  `RecipeSearchRequest`, `GenerateRequest`; echoed on `SearchResponse` /
  `GenerateResponse`.
- `backend/api/routers/recipes.py`: `_retrieve()` now takes `retriever` +
  `retriever_params`, validates the key (**400** before recipe lookup), enforces
  the model-match guard, then runs the strategy — used by **both** `/search` and
  `/generate`. Errors: unknown retriever → 400, bad params/dimension → 422,
  missing dep / HyDE LLM failure → 502.
- `backend/api/routers/vectorstore.py`: the ad-hoc `/vectorstore/search` gets the
  same retriever plumbing.
- Frontend: `api_client.list_retrievers()` + `retriever`/`retriever_params` on
  `search`/`search_recipe`/`generate_recipe`; `app.py` gains a
  `retriever_controls()` helper (picker + per-strategy param widgets) beside both
  query boxes, with a caption explaining retrieval is chosen per query.

**Strategy-pattern compliance:** adding a new retriever = one file in
`stages/retrievers/` + one registry line + (its option appears in the UI
automatically via discovery). The only outside-the-package edit was the one-time
`all_chunks()` store accessor (infrastructure).

**Tests (all green): 28 unit + 36 API.**
- `tests/test_stages.py` +8: a deterministic bag-of-words `_FakeEmbedder` (no
  torch) over a real FAISS store drives every strategy offline. Registry/teaching
  text + `no_api_key` flags; naive == plain `store.search`; **MMR drops an exact
  duplicate** for a diverse chunk (and `lambda_mult=1.0` == naive, bad lambda
  raises); **hybrid/BM25 recovers a term ("ceftriaxone") deliberately kept out of
  the embedder's vocab** (proving its lexical value), plus weighted fusion + bad
  fusion raises; **rerank** with a mock cross-encoder (seeded into `_MODEL_CACHE`)
  reorders to promote the "therapy" chunk; **HyDE** with a mock provider embeds
  the hypothetical answer to reach the pneumonia chunk from an unrelated question,
  and unknown provider raises.
- `tests/test_api.py` +5: `GET /stages/retrievers` lists all five; unknown
  retriever → 400 (validated before recipe lookup); recipe search with `mmr`
  (distinct hits, retriever echoed) and with `hybrid` (skip-guarded on rank-bm25);
  generate with `mmr` + a mock provider echoes the retriever and 400s on an
  unknown one; the in-process frontend-client test now threads `retriever="mmr"`
  through `api.search`.

**Verify:** `python tests/test_stages.py` (28/28), `python tests/test_api.py`
(36/36). Live: build a recipe, then
`POST /recipes/{id}/search {"query":"...","retriever":"mmr"}` (or `hybrid`) —
returns hits with `retriever` echoed; naive/mmr/hybrid confirmed end-to-end with
the real MiniLM embedder + FAISS + BM25 (mmr keeps the top hit but diversifies
the rest; hybrid reorders via RRF). `rerank`/`hyde` are unit-tested with mocks
since they need a model download / a running LLM.

## Change #12 — Optional RAGAS-style answer scoring + the Compare grid

Delivered ARCHITECTURE.md §6's "automated quality scores" and the compare grid
that hosts them: one question runs across a **recipe × provider** grid, and every
answer can *optionally* be scored on four RAG-quality metrics — **faithfulness,
answer relevancy, context precision, context recall** — with a ranking summary
sortable by any of them.

**New module `backend/core/evaluator.py`** (query-time orchestration, ARCHITECTURE
§9's named `core/evaluator.py`; like `generation.py` it is *not* a stage and owns
no REGISTRY, but scoring is still pluggable behind an interface):
- `BaseScorer` (abstract) with `available()` / `unavailable_message()` /
  `score(samples) -> [{metric: value|None}]`. `RagasScorer` is the real
  implementation; tests inject a mock.
- **Feature-gated, lazy `ragas`:** nothing heavy is imported at module load
  (verified: `ragas`/`datasets` absent from `sys.modules` after importing the
  app). `ragas_available()` answers via `importlib.util.find_spec` *without*
  importing ragas. `RagasScorer.score` imports `ragas`+`datasets` **inside the
  method**; if absent it raises `RagasNotInstalled` carrying the friendly
  `RAGAS_INSTALL_HINT` ("pip install ragas"). A real scoring failure (judge
  LLM/`OPENAI_API_KEY` not configured, a metric erroring) raises a loud, readable
  `RuntimeError` — **no silent fallback** to a fake number.
- **Honest gaps, not fabrication:** `context_recall` needs a reference answer, so
  it's reported as `None` (UI shows "—") when no ground truth is supplied; a
  metric ragas doesn't return stays `None`. `with_mean()` adds a derived `mean`
  (average of the *available* metrics). `rank_cells()` is a pure ranking helper
  that always sorts `None` scores **last**, so the frontend can re-rank by any
  metric with no backend round-trip. Unknown metric names fail loudly in the
  `RagasScorer` constructor.

**Endpoints (`backend/api/routers/scoring.py`, mounted in `main.py`):**
- `GET /score/status` — is scoring runnable right now (ragas importable)? Reads
  the flag only; never imports ragas or calls a judge LLM.
- `POST /score` — score a batch of grid cells → per-cell scores + a `ranking`
  sorted by `sort_by`. **Error mapping:** empty cells → 422; unknown `sort_by`
  (not one of the four metrics or `mean`) → 400; ragas absent → **200** with
  `available=false` + the friendly note (scoring is simply off, not an error);
  any real scoring failure → 502 with the message. The scorer is provided via a
  FastAPI dependency (`get_scorer`) so tests swap in a mock through
  `app.dependency_overrides` — the whole path runs without ragas or a key.
- **Schemas** (`schemas.py`): `ScoreCell` (recipe_id/provider/question/answer/
  contexts/ground_truth), `ScoreRequest`, `ScoredCell`, `RankRow`,
  `ScoreResponse` (available/message/metrics/sort_by/cells/ranking), `ScoreStatus`.

**Frontend — new Compare tab (`frontend/app.py`):** multiselect recipes (rows) ×
LLM providers (columns) + one question + retriever picker + optional reference
answer. "Run comparison grid" calls the existing `POST /recipes/{id}/generate`
per cell (a per-cell error, e.g. a provider needing a key, is surfaced in that
cell without aborting the grid). The grid renders answers side by side with
latency/cost/chunk-count. A scoring section checks `/score/status`: if ragas is
absent it shows the install note; if present, "Score all answers" posts every
cell to `/score`, annotates each grid cell with its scores, and renders a
**ranking summary** re-sortable by any of the four metrics + `mean` (sorted
client-side, missing scores last). Client methods added:
`list_llm_providers`, `score_status`, `score_cells`.

**Dependency:** none added — `ragas` stays **commented/optional** in
`requirements.txt` (the tool runs with zero extra installs; a learner uncomments
it and sets `OPENAI_API_KEY` to turn scoring on). Asked before touching the dep;
user chose "keep optional".

**Strategy-pattern compliance:** scoring is orchestration, not one of the five
swappable stages, so it lives in `core/` (like `generation.py`) — no stage
package was edited. `BaseScorer` keeps it swappable/mockable.

**Tests (all green): 35 unit + 42 API.**
- `tests/test_stages.py` +7: `SCORE_METRICS`/`ragas_available()`/install-hint;
  `mean_score`+`with_mean` (ignore `None`, all-`None` → `None`); `rank_cells`
  (best first, missing last, 1-based rank); `score_samples` with a mock scorer;
  the unavailable scorer raises `RagasNotInstalled`; `RagasScorer` rejects an
  unknown metric; the lazy-import guard (constructing imports no ragas;
  `score()` raises `RagasNotInstalled` when ragas is absent — skip-guarded if it
  *is* installed).
- `tests/test_api.py` +6: `/score/status` shape; empty → 422; unknown `sort_by`
  → 400; a mock scorer (via `dependency_overrides`) ranks a "good" vs "bad"
  answer and proves `context_recall` is `None` without a ground truth + `mean`
  is derived; the unavailable path returns 200 + friendly note + empty
  cells/ranking; the frontend `APIClient` drives `score_status`/`score_cells`
  in-process.

**Verify:** `python tests/test_stages.py` (35/35), `python tests/test_api.py`
(42/42). Lazy gate: `python -c "import sys, backend.api.main; assert 'ragas' not
in sys.modules"`. Without ragas installed, `GET /score/status` →
`{"available": false, ...}` and the Compare tab shows the install note; with
`pip install ragas` + `OPENAI_API_KEY`, "Score all answers" produces real
metrics and a sortable ranking.

## Change #13 — Four remaining chunking strategies (fixed_size, sentence, semantic, structure_aware)

Filled SPEC.md §3 **Stage 2: Chunking** with four concrete strategies alongside
the existing `recursive` chunker. Every technique is a new file in
`backend/stages/chunkers/` plus one registry line (Strategy Pattern). All
parameters are **visible and overridable in the UI** and **recorded in
config.json** (CLAUDE.md Non-Negotiables); every chunk carries truthful
`overlap_with_prev` reflecting actual character overlap.

**The four strategies** (each a file, one registry line in `__init__.py`):

| Key | Class | Parameters | Teaching summary |
|---|---|---|---|
| `fixed_size` | `FixedSizeChunker` | `size` (chars), `overlap` | Simplest and most predictable; uniform chunk sizes but blindly cuts sentences/words. Good for debugging and uniform data. |
| `sentence` | `SentenceChunker` | `sentences_per_chunk`, `overlap_sentences` | Respects sentence boundaries perfectly; chunks never split mid-sentence. Chunk sizes vary by sentence length; heuristic sentence splitter (can miss abbreviations). |
| `semantic` | `SemanticChunker` | `similarity_threshold` (0-1), `embedder` (required instance) | Groups semantically related sentences; boundaries at low-similarity transitions. Most "intelligent" but requires an embedder and slower (embeds every sentence). |
| `structure_aware` | `StructureAwareChunker` | `header_levels` (1-6) | Splits on Markdown headers (e.g. `#`, `##`); preserves logical document structure. Works only on Markdown; treats plain text as one chunk. Ideal for documentation. |

**Design decisions:**

- **Overlap calculation (truthful `overlap_with_prev`):** every chunker computes
  the actual number of leading characters of the current chunk that match the
  tail of the previous chunk, not just the *requested* overlap. This drives the
  UI's overlap highlight accurately — a learner sees exactly what text repeats
  between chunks, which is critical for a teaching tool.

- **Token count (model-agnostic approximation):** all chunkers define
  `approx_token_count(text)` locally (a regex over word-ish runs + standalone
  punctuation) so chunking never secretly depends on which embedder is chosen.
  The token count is for display in the UI, not used for chunk sizing (which
  uses characters). This matches the existing `recursive` chunker.

- **Embedder in semantic chunker:** the `semantic` chunker accepts an
  `embedder` instance (passed by the caller, required, no default) so it can
  embed sentences and compute cosine similarity. Lazy imports are not used here
  (the embedder is already instantiated and loaded); the caller is responsible
  for choosing a compatible embedder.

- **Sentence splitting (heuristic):** both `sentence` and `semantic` chunkers
  import and reuse a local `split_sentences()` function that looks for
  sentence-ending punctuation (. ! ?) followed by a space + capital letter or
  end-of-string. It's not perfect (misses "Dr. Smith" abbreviations) but works
  well for prose and is transparent (visible in the code, not buried in a
  library).

- **Markdown structure parsing:** the `structure_aware` chunker splits on
  regex-matched Markdown headers (`^#{1,6}\s+...`), respecting depth. With
  `header_levels=2`, both `#` (depth 1) and `##` (depth 2) create chunk
  boundaries; `###` (depth 3) is treated as plain text. This naturally groups
  subsections under their parent headers.

**All params visible + recorded in config.json (CLAUDE.md Non-Negotiables):**
- `fixed_size`: `size`, `overlap`
- `sentence`: `sentences_per_chunk`, `overlap_sentences`
- `semantic`: `similarity_threshold`
- `structure_aware`: `header_levels`

No hidden defaults; if a learner reads their recipe's `config.json`, they can
see exactly how each stage was configured.

**Tests (31 unit, all green, no external deps):** `tests/test_chunkers.py`
covers registry entries, basic chunking, parameter validation, edge cases
(empty text, single chunk, no headers), and sample-data verification:

- **FixedSizeChunker:** registry + basic chunking + overlap integrity +
  invalid-param raises + sample-data test (→ multiple chunks, all <= size).
- **SentenceChunker:** registry + 4 sentences / 2 per chunk = 2 chunks + overlap
  + invalid-param raises + empty text = no chunks + sample-data test.
- **SemanticChunker:** registry + missing embedder raises + invalid-threshold
  raises + mock embedder with constant vectors (high similarity = fewer chunks,
  then low similarity = boundary) + single sentence = 1 chunk + empty = 0 chunks.
- **StructureAwareChunker:** registry + Markdown headers split correctly at
  specified depth + no headers = 1 chunk + invalid header_levels raises +
  sample-data test.
- **Overlap integrity (all chunkers):** for each chunk after the first, verify
  `overlap_with_prev` matches the actual character overlap between chunks.
- **ID consistency (parametrized):** chunk IDs are `chunk_XXXX`, sequential,
  unique, and match the position field.

**Strategy-pattern compliance:** zero edits outside `backend/stages/chunkers/`
except for five new imports in `__init__.py` and the registry lines. A learner
or future contributor adding a new chunker touches only that package.

**Verify:** `python -m pytest tests/test_chunkers.py` (31/31 green). Each
chunker loaded from the registry and tested with both synthetic and sample
data. Sample text is skipped gracefully if `sample_data/` is absent (for CI
environments); every test's assertions stand without it.

Next: wire these strategies into the `/POST /chunk` API endpoint so a learner
can pick `"fixed_size"`, `"sentence"`, `"semantic"`, or `"structure_aware"`
from the UI and see the chunks + their overlaps side by side.

## Change #14 — Ollama provider error handling: list available models on 404

**Problem:** When a user's `recipe.json` specifies a model (e.g. `llama3.1`) that
hasn't been downloaded locally via `ollama pull`, the Ollama server returns HTTP
404. The original error message just showed the raw status code and raw response
text — it wasn't helpful to a learner, and didn't guide them on what to do next.

**Solution:** Enhanced `OllamaProvider.generate()` to detect the 404 and:
1. Query the Ollama server's `/api/tags` endpoint to list all **locally available
   models** (if the server is reachable).
2. Display them in the error message ("Available models: `llama3.2`, `mistral`, …").
3. If no models are found, guide the learner with the command `ollama pull <model>`.

**Implementation (`backend/stages/llm_providers/ollama_provider.py`):**

- New **`_get_available_models()` private helper:** safely queries `/api/tags`,
  returns a list of model names, or an empty list if the server is unreachable
  or doesn't respond. Catches all exceptions so this helper is "best effort" —
  it never crashes the main flow.

- **Updated error at line 120–131:** When `response.status_code != 200`, call
  `_get_available_models()` and format the error message with the list. If the
  list is non-empty, show it; if empty, show the `ollama pull` hint instead.

- **No new dependencies:** uses `httpx` (already a core backend dep) and
  `timeouts` to avoid hanging.

**Tests (2 unit, both green):**

- `test_ollama_provider_model_not_found_lists_available`: mocks a 404 response
  + a successful `/api/tags` endpoint that returns 3 models → verifies the error
  message includes all three model names and the missing model name, proving
  the list is surfaced to the learner.

- `test_ollama_provider_model_not_found_no_models_available`: mocks a 404
  response + an empty `/api/tags` response → verifies the error message includes
  the `ollama pull <model>` guidance so the learner knows the next step.

**Non-Negotiable compliance:**
- Fail loudly with a learner-readable message ✓
  (CLAUDE.md "No silent fallback on error").
- No new dependencies ✓ (uses `httpx` already present).
- Lazy import already in place ✓ (httpx imported inside `generate`).

**Verify:** `python -m pytest tests/test_stages.py -k ollama` → all 4 Ollama
tests pass (2 existing + 2 new). Existing tests still pass (no regression).
