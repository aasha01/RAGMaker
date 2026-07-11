# ARCHITECTURE.md — RAG Lab

## 1. Purpose

A hands-on tool for learning RAG architecture by **building multiple RAG
pipelines with different configurations ("Recipes"), inspecting every
intermediate stage, and comparing final answer quality side by side.**

## 2. Two Parts

### Part A — RAG Builder (wizard)
Walks the user through 5 configurable stages, one at a time. Each stage:
1. Presents options (with plain-language tradeoff explanations)
2. Runs the chosen technique
3. **Shows the actual output of that stage** (see §5, Stage Inspection)
4. Lets the user redo the stage with a different option before continuing
5. Persists output to disk

At the end, the full configuration + all intermediate artifacts + the final
vector store are saved together as a **Recipe**.

### Part B — Query & Compare
1. User asks one question.
2. The question runs against every selected saved Recipe.
3. Optionally, against multiple LLM providers per recipe (full grid: recipe
   × provider).
4. Results shown side by side: retrieved chunks, generated answer, scores,
   latency, cost.

## 3. Pipeline Stages (Part A)

| # | Stage | Example options |
|---|-------|------------------|
| 1 | Ingestion | txt, PDF, DOCX, CSV/JSON, URL, folder |
| 2 | Parsing | Manual (pypdf/pdfplumber/python-docx), LangChain loaders, LlamaIndex readers |
| 3 | Chunking | Fixed-size, Recursive, Sentence-based, Semantic, Structure-aware (Markdown headers), Sliding window |
| 4 | Embedding | Sentence-Transformers (local), OpenAI, Cohere, BGE/Instructor |
| 5 | Vector Store | FAISS (Flat/HNSW), Chroma, Qdrant |

Each stage is a swappable **Strategy** behind a common interface (see
`SPEC.md` for exact method signatures). The orchestrator (`core/recipe.py`)
never knows which concrete strategy it's using.

## 4. Recipe = Config + Artifacts

A Recipe is the unit of comparison. Nothing is mutated after creation.

```
/recipes/recipe_<id>_<short-desc>/
    config.json              # every choice made, human-readable, reproducible
    stage_outputs/
        01_raw/                    original uploaded file(s), untouched
        02_parsed.json             extracted text + per-page/doc metadata
        03_chunks.json             [{chunk_id, text, char_len, token_len, source, position}]
        04_embeddings.npy          vectors (or sampled subset if very large)
        04_embeddings_meta.json    chunk_id -> row index mapping, model name, dims
    vectorstore/               the actual DB files (FAISS index / Chroma dir)
    metadata.json              build timestamp, total tokens, cost, timing per stage
```

`config.json` alone must be enough to describe how to rebuild the recipe —
this is what makes recipes comparable and the project reproducible.

## 5. Stage Inspection (learning core requirement)

Every stage must be **visually inspectable**, both live during build and
later when reopening a saved recipe. Minimum per stage:

- **Ingestion**: filename, size, type, raw preview
- **Parsing**: raw vs. extracted text side-by-side; char/word count delta
- **Chunking**: scrollable chunk list, chunk length histogram, overlap
  highlighting between consecutive chunks, "chunks in context" view against
  the original document
- **Embedding**: vector dimension, raw value preview, similarity heatmap
  across a chunk sample, embedding time/cost
- **Vector Store**: record count, metadata table, 2D projection (PCA/UMAP)
  scatter plot colored by source document
- **Query**: retrieved chunks with similarity scores, highlighted in the
  context of the original parsed document

Each stage screen includes an explicit **"redo this stage"** control — a
learner should be able to see a bad result (e.g. a chunk split mid-sentence)
and immediately try another strategy without restarting the whole wizard.

## 6. Query & Compare (Part B) Design

- Comparison grid: rows = Recipes, columns = LLM Providers (or vice versa).
  Full grid mode (recipe × provider) is the default.
- Per cell: generated answer, retrieved chunk list + scores, latency, token
  cost, and (optional) automated quality scores (faithfulness, answer
  relevancy, context precision/recall — RAGAS-style).
- A summary table ranks all cells by a chosen metric, so the "which
  approach is best" question has a concrete answer, not just a vibe.

## 7. Pluggable LLM Providers

Generation is itself a variable in comparisons, not a fixed constant. All
providers implement one interface (`stages/llm_providers/base.py`):
`generate(prompt, **kwargs) -> str`. Concrete providers: OpenAI, Anthropic,
Ollama (local). New providers require no changes outside
`stages/llm_providers/`.

## 8. Non-Functional Requirements

- **No silent fallback**: if a chosen strategy fails, surface the error —
  never substitute a different technique automatically.
- **No cross-contamination**: never query a vector store with an embedding
  from a different model than the one used to build it. Validate at query
  time (model name + dimension stored in `04_embeddings_meta.json`).
- **Cost/time transparency**: every API-based stage records tokens used and
  wall-clock time in `metadata.json`.
- **Runs out of the box**: default stack (manual parser, recursive
  chunker, local sentence-transformers embedder, FAISS) requires no API
  keys, so a learner can build their first recipe with zero external
  accounts.

## 9. Suggested Project Layout

```
rag_lab/
├── app.py                       # Streamlit entrypoint (Build tab, Compare tab)
├── stages/
│   ├── parsers/        {base.py, manual.py, langchain_parser.py, llamaindex_parser.py, __init__.py (registry)}
│   ├── chunkers/        {base.py, fixed_size.py, recursive.py, sentence.py, semantic.py, structure_aware.py, __init__.py}
│   ├── embedders/        {base.py, sentence_transformer.py, openai_embedder.py, cohere_embedder.py, __init__.py}
│   ├── vectorstores/        {base.py, faiss_store.py, chroma_store.py, __init__.py}
│   └── llm_providers/        {base.py, openai_provider.py, anthropic_provider.py, ollama_provider.py, __init__.py}
├── core/
│   ├── recipe.py                # build/save/load orchestration
│   ├── inspector.py              # per-stage visualization helpers
│   └── evaluator.py              # query-time comparison + scoring
├── sample_data/                  # small sample docs to build a first recipe with
├── recipes/                      # persisted recipes (gitignored, generated at runtime)
├── requirements.txt
├── CLAUDE.md
├── ARCHITECTURE.md
└── SPEC.md
```
