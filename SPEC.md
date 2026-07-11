# SPEC.md — Stage Interfaces, Options & Config Schema

## 1. Common Interface Pattern

Every stage module exposes:
- `base.py` — abstract class with the method(s) all strategies must implement
- one file per concrete strategy
- `__init__.py` — a registry dict mapping a string key (used in `config.json`
  and the UI dropdown) to the concrete class

```python
# stages/<stage>/__init__.py pattern
from .strategy_a import StrategyA
from .strategy_b import StrategyB

REGISTRY = {
    "strategy_a": StrategyA,
    "strategy_b": StrategyB,
}
```

## 2. Stage 1 — Parsing

```python
class BaseParser(ABC):
    name: str
    description: str
    def parse(self, file_path: str) -> ParsedDocument: ...

@dataclass
class ParsedDocument:
    text: str
    source: str
    metadata: dict  # e.g. {"pages": int, "page_texts": [...]}
```

| Key | Class | Notes |
|---|---|---|
| `manual` | ManualParser | pypdf/pdfplumber for PDF, python-docx for DOCX, plain read for txt. No hidden cleanup. |
| `langchain` | LangChainParser | Wraps `PyPDFLoader` / `UnstructuredFileLoader` etc. |
| `llamaindex` | LlamaIndexParser | Wraps `SimpleDirectoryReader` / `PDFReader`. Lazy import. |

**Recorded in config:** `{"parser": "manual"}`
**Stage output artifact:** `02_parsed.json` = `{text, source, metadata}`

## 3. Stage 2 — Chunking

```python
class BaseChunker(ABC):
    name: str
    description: str
    def chunk(self, doc: ParsedDocument, **params) -> list[Chunk]: ...

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    position: int          # order within source doc
    char_len: int
    token_len: int
    overlap_with_prev: int  # characters shared with previous chunk, 0 if none
```

| Key | Class | Parameters |
|---|---|---|
| `fixed_size` | FixedSizeChunker | `size` (chars or tokens), `overlap` |
| `recursive` | RecursiveChunker | `size`, `overlap`, separator list |
| `sentence` | SentenceChunker | `sentences_per_chunk`, `overlap_sentences` |
| `semantic` | SemanticChunker | `similarity_threshold`, requires an embedder instance |
| `structure_aware` | StructureAwareChunker | `header_levels` (e.g. split on `#`, `##`) |

**Recorded in config:** `{"chunker": "recursive", "size": 512, "overlap": 50}`
**Stage output artifact:** `03_chunks.json` = list of `Chunk`

**UI must show:** chunk count, length histogram, overlap highlight between
consecutive chunks.

## 4. Stage 3 — Embedding

```python
class BaseEmbedder(ABC):
    name: str
    description: str
    dimension: int
    def embed(self, texts: list[str]) -> np.ndarray: ...  # shape (n, dim)
```

| Key | Class | Parameters |
|---|---|---|
| `sentence_transformers` | SentenceTransformerEmbedder | `model_name` (default `all-MiniLM-L6-v2`), `normalize` |
| `openai` | OpenAIEmbedder | `model_name` (e.g. `text-embedding-3-small`/`-large`), `dimensions` (truncation), `normalize` |
| `cohere` | CohereEmbedder | `model_name` (`embed-v3`), `input_type` |

**Recorded in config:** `{"embedder": "sentence_transformers", "model_name": "all-MiniLM-L6-v2", "dimension": 384, "normalize": true}`
**Stage output artifacts:** `04_embeddings.npy`, `04_embeddings_meta.json` =
`{"model_name", "dimension", "normalize", "chunk_id_order": [...]}`

**Validation rule:** any downstream query MUST check the query embedding's
`model_name`+`dimension` match the store's `04_embeddings_meta.json` before
searching. Mismatch = hard error, not silent re-embed.

## 5. Stage 4 — Vector Store

```python
class BaseVectorStore(ABC):
    name: str
    description: str
    def build(self, embeddings: np.ndarray, chunks: list[Chunk], **params) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]: ...

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
```

| Key | Class | Parameters |
|---|---|---|
| `faiss` | FAISSStore | `index_type` (`flat`/`hnsw`), `metric` (`cosine`/`l2`/`dot`) |
| `chroma` | ChromaStore | `metric`, persistent directory |
| `qdrant` | QdrantStore | `metric`, collection config |

**Recorded in config:** `{"vectorstore": "faiss", "index_type": "flat", "metric": "cosine"}`
**Stage output artifact:** `vectorstore/` (raw DB files)

## 6. Stage 5 — Retrieval (query-time, not build-time)

Kept as its own configurable step even though it runs at query time, since
retrieval strategy is often the highest-leverage lever in a RAG system.

| Key | Notes |
|---|---|
| `naive_topk` | plain vector similarity, top-k |
| `mmr` | Maximal Marginal Relevance — diversity-aware |
| `hybrid` | BM25 + vector, combined/re-ranked |
| `rerank` | vector top-N → cross-encoder re-rank → top-k |
| `hyde` | LLM generates a hypothetical answer first, embeds that for search |

**Recorded per-query, not per-recipe** (so the same recipe can be tested
with different retrieval strategies without rebuilding).

## 7. LLM Providers (generation)

```python
class BaseLLMProvider(ABC):
    name: str
    def generate(self, prompt: str, **kwargs) -> GenerationResult: ...

@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
```

| Key | Class |
|---|---|
| `openai` | OpenAIProvider |
| `anthropic` | AnthropicProvider |
| `ollama` | OllamaProvider (local, no cost) |

## 8. Full `config.json` Schema (per recipe)

```jsonc
{
  "recipe_id": "recipe_001",
  "created_at": "2026-07-11T10:00:00Z",
  "source": {"filename": "handbook.pdf", "type": "pdf"},
  "parser": {"key": "manual"},
  "chunker": {"key": "recursive", "size": 512, "overlap": 50},
  "embedder": {"key": "sentence_transformers", "model_name": "all-MiniLM-L6-v2", "dimension": 384, "normalize": true},
  "vectorstore": {"key": "faiss", "index_type": "flat", "metric": "cosine"},
  "stats": {"chunk_count": 214, "build_time_sec": 12.4, "embedding_cost_usd": 0.0}
}
```

## 9. Comparison Grid Output Schema (query-time)

```jsonc
{
  "question": "What is the refund policy?",
  "runs": [
    {
      "recipe_id": "recipe_001",
      "provider": "openai",
      "retrieved_chunks": [{"chunk_id": "...", "score": 0.83, "text": "..."}],
      "answer": "...",
      "latency_ms": 812,
      "cost_usd": 0.0021,
      "scores": {"faithfulness": 0.91, "answer_relevancy": 0.87}
    }
  ]
}
```

## 10. Dependencies (indicative — finalize in `requirements.txt` at build time)

Core (no API key required): `streamlit`, `sentence-transformers`, `faiss-cpu`,
`pypdf`, `pdfplumber`, `python-docx`, `numpy`, `scikit-learn` (PCA), `plotly`.

Optional (feature-gated, lazy import): `langchain`, `llama-index`, `openai`,
`anthropic`, `cohere`, `chromadb`, `qdrant-client`, `ragas`, `umap-learn`.
