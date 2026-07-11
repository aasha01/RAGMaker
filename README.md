# RAG Maker Lab

A hands-on learning tool for **building and comparing multiple RAG (Retrieval-Augmented Generation) pipelines** with different configurations. Inspect every intermediate stage, understand the tradeoffs of different techniques, and measure which approach works best for your data.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

No API keys required for the default stack (local embeddings + FAISS vector store). Optional features like OpenAI, Anthropic, or Cohere require their respective API keys in environment variables.

### 2. Run the Application

Backend:

```bash
uvicorn backend.api.main:app --reload
```

Frontend:

```bash
streamlit run frontend/app.py
```

The app opens at `http://localhost:8501/`. You'll see two tabs:

- **Build**: Walk through a RAG pipeline step-by-step, choosing strategies and inspecting output at each stage.
- **Compare**: Query multiple saved recipes side-by-side to compare answer quality, retrieval performance, and cost.

### 3. Build Your First Recipe

1. Go to the **Build** tab.
2. Upload a sample document (txt, PDF, DOCX supported).
3. Choose strategies for each stage:
   - **Parser**: How to extract text from the document.
   - **Chunker**: How to split text into chunks.
   - **Embedder**: How to convert chunks to vectors.
   - **Vector Store**: Where to index the vectors.
4. Inspect the output at each stage (chunks, embeddings, etc.).
5. Save the recipe — it captures all choices + intermediate artifacts for later comparison.

### 4. Query & Compare

Once you have 2+ recipes:

1. Go to the **Compare** tab.
2. Select recipes and LLM providers.
3. Ask a question — see answers side-by-side with retrieval scores, latency, and cost.

## What's a Recipe?

A **Recipe** is a complete RAG configuration saved to disk:

```
recipes/recipe_<id>_<description>/
├── config.json                  # Every choice made, reproducible
├── stage_outputs/
│   ├── 01_raw/                  # Original uploaded files
│   ├── 02_parsed.json           # Extracted text
│   ├── 03_chunks.json           # Text chunks
│   ├── 04_embeddings.npy        # Vector embeddings
│   └── 04_embeddings_meta.json  # Embedding metadata
├── vectorstore/                 # Vector database (FAISS/Chroma)
└── metadata.json                # Timing, cost, stats
```

Config alone is enough to reproduce the recipe — this is what makes comparison fair and trustworthy.

## Project Structure

```
rag_lab/
├── app.py                       # Streamlit UI (Build & Compare tabs)
├── stages/
│   ├── parsers/                 # Text extraction (PDF, DOCX, txt)
│   ├── chunkers/                # Text splitting strategies
│   ├── embedders/               # Vector embedding models
│   ├── vectorstores/            # Vector database backends
│   └── llm_providers/           # Generation providers (OpenAI, Anthropic, Ollama)
├── core/
│   ├── recipe.py                # Build/save/load orchestration
│   ├── inspector.py             # Stage visualization helpers
│   └── evaluator.py             # Query & comparison logic
├── sample_data/                 # Example documents to build with
├── recipes/                     # Saved recipes (generated at runtime)
├── requirements.txt
└── docs/
    ├── ARCHITECTURE.md          # System design & data flow
    ├── SPEC.md                  # Detailed stage interfaces
    └── CLAUDE.md                # Development guidelines
```

## Pipeline Stages

Each recipe passes through 5 configurable stages:

| Stage            | What It Does                      | Example Strategies                                |
| ---------------- | --------------------------------- | ------------------------------------------------- |
| **Parser**       | Extract text from uploaded files  | Manual (pypdf, pdfplumber), LangChain, LlamaIndex |
| **Chunker**      | Split text into searchable pieces | Fixed-size, Recursive, Sentence-based, Semantic   |
| **Embedder**     | Convert chunks to vectors         | Sentence-Transformers (local), OpenAI, Cohere     |
| **Vector Store** | Index & store vectors             | FAISS, Chroma, Qdrant                             |
| **Retrieval**    | Fetch relevant chunks for a query | Top-K, MMR, Hybrid, Re-ranking                    |

Each stage is swappable — no code changes needed to try a different strategy. The UI shows the output of each stage so you can debug and understand what's happening.

## Default Stack (Requires No API Keys)

- **Parser**: Manual (pypdf for PDF, python-docx for DOCX)
- **Chunker**: Recursive (splits on sentence/word/char boundaries)
- **Embedder**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (Flat index, cosine similarity)
- **LLM**: Ollama (local, requires separate installation)

This stack works offline. For production-grade embedding or generation, add API keys.

## Adding API Keys (Optional)

Set environment variables to enable optional providers:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
```

Then restart the app — new providers will appear in the dropdowns.

## Files to Read Next

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System design, data flow, stage inspection requirements.
- **[SPEC.md](SPEC.md)** — Detailed stage interfaces, parameters, config schema.
- **[CLAUDE.md](CLAUDE.md)** — Development guidelines (for contributors).

## Common Tasks

### View a saved recipe

Recipes are stored in `recipes/`. Open `config.json` to see all choices, or reload in the **Build** tab to inspect stages.

### Compare two recipes

1. Save both as different recipes (e.g., "recursive_chunks" vs "semantic_chunks").
2. Go to **Compare** tab → select both → ask a question.
3. See which performs better for your use case.

### Run tests

```bash
pytest tests/
```

Each stage has unit tests verifying strategy correctness against sample data.

## Troubleshooting

**"Module not found" errors**: Missing optional dependencies. Install with:

```bash
pip install langchain llamaindex anthropic openai cohere
```

**Embedding dimension mismatch**: Cannot query with embeddings from a different model than the one used to build the recipe. Check `04_embeddings_meta.json` in the recipe.

**Ollama not available**: Install [Ollama](https://ollama.ai) and run `ollama pull mistral` (or another model), then restart the app.

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines. Every new stage strategy goes in `stages/<stage>/` with no changes outside that package.
