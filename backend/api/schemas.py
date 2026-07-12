"""Pydantic request/response models — the HTTP contract for the API.

These mirror the stage dataclasses (`ParsedDocument`, `Chunk`) at the API
boundary so the OpenAPI docs (/docs) describe exactly what crosses the wire,
and so the frontend has a typed shape to rely on. The API layer converts
between these models and the internal dataclasses; the pipeline stages
themselves never import anything from here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StrategyInfo(BaseModel):
    """One selectable technique for a stage, as shown in the UI dropdown."""

    key: str = Field(..., description="Registry key recorded in config.json")
    name: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="What / tradeoff / when — teaching text")


# --- Parsing ---------------------------------------------------------------

class ParsedDocumentModel(BaseModel):
    text: str
    source: str
    metadata: dict = Field(default_factory=dict)


class ParseSummary(BaseModel):
    """At-a-glance numbers for the parsing inspection view."""

    char_count: int
    word_count: int
    pages: int | None = None
    format: str | None = None


class ParseResponse(BaseModel):
    parser: str
    document: ParsedDocumentModel
    summary: ParseSummary


# --- Chunking --------------------------------------------------------------

class ChunkModel(BaseModel):
    chunk_id: str
    text: str
    source: str
    position: int
    char_len: int
    token_len: int
    overlap_with_prev: int


class ChunkRequest(BaseModel):
    document: ParsedDocumentModel
    chunker: str = "recursive"
    params: dict = Field(
        default_factory=dict,
        description="Stage parameters (e.g. {'size': 512, 'overlap': 50}); "
        "recorded verbatim in config.json.",
    )
    embedder: str | None = Field(
        None,
        description="Embedder registry key. Required only by chunkers that need "
        "an embedder instance to run (currently: 'semantic'); ignored by chunkers "
        "that don't take one.",
    )
    embedder_params: dict = Field(
        default_factory=dict,
        description="Embedder constructor params (e.g. {'model_name': "
        "'nomic-embed-text:latest', 'normalize': true}), used together with "
        "`embedder`.",
    )


class ChunkSummary(BaseModel):
    """At-a-glance numbers for the chunking inspection view."""

    chunk_count: int
    char_len_min: int
    char_len_mean: int
    char_len_max: int
    token_len_mean: int
    total_overlap_chars: int


class ChunkResponse(BaseModel):
    chunker: str
    params: dict
    chunks: list[ChunkModel]
    summary: ChunkSummary


# --- Embedding -------------------------------------------------------------

class ModelInfo(BaseModel):
    """Details about an embedding model, for the stage-3 model-details view.

    Fields beyond model_name/output_dimension are optional because different
    backends expose different facts (a hosted API may not report a param count,
    a local model has no per-call cost, etc.). Nothing is faked — a backend that
    can't report a value leaves it None.
    """

    model_name: str
    backend: str | None = None
    default_dimension: int | None = Field(
        None, description="The model's native output dimension before any truncation."
    )
    output_dimension: int = Field(..., description="Dimension actually produced.")
    dimension_customizable: bool = Field(
        False, description="Whether output_dimension can be changed (truncate_dim)."
    )
    max_seq_length_tokens: int | None = Field(
        None, description="Context window: max input tokens the model reads per chunk."
    )
    param_count: int | None = Field(None, description="Number of model parameters.")
    approx_size_mb: float | None = Field(
        None, description="Rough model size in MB (float32 weight estimate)."
    )
    normalize: bool | None = None
    truncate_dim: int | None = None
    notes: str | None = Field(None, description="Caveats, e.g. lossy truncation.")


class EmbeddingMeta(BaseModel):
    """Mirrors stage_outputs/04_embeddings_meta.json — the identity carried with
    the vectors and re-checked before any query (never mix models)."""

    model_name: str
    dimension: int
    normalize: bool
    chunk_id_order: list[str]


class EmbedRequest(BaseModel):
    chunks: list[ChunkModel]
    embedder: str = "sentence_transformers"
    params: dict = Field(
        default_factory=dict,
        description="Embedder parameters, e.g. {'model_name': 'all-MiniLM-L6-v2', "
        "'normalize': true}; recorded verbatim in config.json.",
    )


class EmbedResponse(BaseModel):
    embedder: str
    model_name: str
    dimension: int
    normalize: bool
    count: int
    embed_time_sec: float
    cost_usd: float | None = Field(
        None, description="0.0 for local models; populated for paid API embedders."
    )
    # Full vectors are returned so the frontend can carry them into the vector
    # store stage and (later) render the similarity heatmap / 2D projection.
    vectors: list[list[float]]
    value_preview: list[float] = Field(
        ..., description="First few dimensions of the first vector, for a raw peek."
    )
    norms_preview: list[float] = Field(
        ..., description="L2 norms of the first few vectors (≈1.0 when normalized)."
    )
    model_info: ModelInfo
    meta: EmbeddingMeta


# --- Vector store ----------------------------------------------------------

class VectorStoreBuildRequest(BaseModel):
    vectors: list[list[float]] = Field(..., description="Row i corresponds to chunks[i].")
    chunks: list[ChunkModel]
    vectorstore: str = "faiss"
    params: dict = Field(
        default_factory=dict, description="e.g. {'index_type': 'flat', 'metric': 'cosine'}"
    )
    embedder: str = Field(
        "sentence_transformers",
        description="Embedder key used to build; reused to embed queries at search time.",
    )
    embed_params: dict = Field(
        default_factory=dict,
        description="Embedder params (model_name/normalize/truncate_dim) for query re-embedding.",
    )
    meta: dict = Field(
        default_factory=dict,
        description="EmbeddingMeta (model_name, dimension, normalize) carried with the vectors.",
    )


class StoreMetaRow(BaseModel):
    chunk_id: str
    source: str
    position: int
    char_len: int


class VectorStoreBuildResponse(BaseModel):
    store_id: str
    vectorstore: str
    index_type: str
    metric: str
    count: int
    dimension: int
    model_name: str
    metadata_sample: list[StoreMetaRow]


# Query-time retrieval strategy, chosen per request (SPEC.md §6) — never stored
# in the recipe, so one recipe can be probed with several. Shared by the ad-hoc
# store search, recipe search, and recipe generate.
RETRIEVER_FIELD = Field(
    "naive_topk",
    description="Retrieval strategy key (naive_topk/mmr/hybrid/rerank/hyde). "
    "Recorded per query, not in the recipe.",
)
RETRIEVER_PARAMS_FIELD = Field(
    default_factory=dict,
    description="Strategy params, e.g. {'lambda_mult': 0.5} (mmr), "
    "{'fusion': 'rrf'} (hybrid), {'fetch_n': 20} (rerank), "
    "{'provider': 'ollama'} (hyde).",
)


class SearchRequest(BaseModel):
    store_id: str
    query: str
    top_k: int = Field(4, ge=1, le=50)
    retriever: str = RETRIEVER_FIELD
    retriever_params: dict = RETRIEVER_PARAMS_FIELD


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    source: str
    position: int
    score: float


class SearchResponse(BaseModel):
    store_id: str
    query: str
    metric: str
    model_name: str
    retriever: str = Field("naive_topk", description="Retrieval strategy that produced these hits.")
    # higher = more similar for cosine/dot; lower = more similar for l2. Note:
    # rerank/hybrid report their own score scale (cross-encoder / fused), not the
    # store metric — compare those to each other, not across retrievers.
    hits: list[SearchHit]


# --- Recipes (persistence) -------------------------------------------------

class RecipeBuildRequest(BaseModel):
    source_filename: str
    source_b64: str = Field(..., description="Base64-encoded source file bytes.")
    config: dict = Field(
        ...,
        description="Per-stage {key + params} for parser/chunker/embedder/vectorstore.",
    )
    name: str | None = Field(
        None,
        description="Human-friendly label. Falls back to `description`, then the "
        "auto-generated recipe_id, if left blank.",
    )
    description: str | None = None


class RecipeSummary(BaseModel):
    """One row of the recipe index — what the Recipes list shows."""

    recipe_id: str
    name: str | None = None
    created_at: str | None = None
    source_filename: str | None = None
    source_type: str | None = None
    parser: str | None = None
    chunker: str | None = None
    embedder: str | None = None
    model_name: str | None = None
    dimension: int | None = None
    vectorstore: str | None = None
    index_type: str | None = None
    metric: str | None = None
    chunk_count: int | None = None
    build_time_sec: float | None = None
    cost_usd: float | None = None
    description: str | None = None
    path: str | None = None


class RecipeDetail(BaseModel):
    config: dict
    metadata: dict


class RecipeSearchRequest(BaseModel):
    query: str
    top_k: int = Field(4, ge=1, le=50)
    retriever: str = RETRIEVER_FIELD
    retriever_params: dict = RETRIEVER_PARAMS_FIELD


# --- Generation (answer over retrieved context) ----------------------------

class GenerateRequest(BaseModel):
    """Ask a saved recipe a question and generate an answer over its top-k
    chunks with a chosen LLM provider."""

    question: str
    provider: str = Field("ollama", description="LLM provider key (ollama/openai/anthropic).")
    top_k: int = Field(4, ge=1, le=50, description="How many chunks to retrieve as context.")
    retriever: str = RETRIEVER_FIELD
    retriever_params: dict = RETRIEVER_PARAMS_FIELD
    provider_params: dict = Field(
        default_factory=dict,
        description="Provider constructor args, e.g. {'model': 'llama3.2'} or "
        "{'model': 'claude-opus-4-8', 'max_tokens': 1024}. Recorded so the run "
        "is reproducible.",
    )
    gen_params: dict = Field(
        default_factory=dict,
        description="Per-call generate() kwargs (e.g. {'temperature': 0.2}).",
    )


class GenerateResponse(BaseModel):
    recipe_id: str
    question: str
    provider: str
    retriever: str = Field("naive_topk", description="Retrieval strategy used to gather the context.")
    model: str | None = Field(None, description="The LLM model actually used, if reported.")
    answer: str
    # The exact text sent to the LLM. Returned on purpose: this is a teaching
    # tool, so the learner sees precisely what context + question the model saw.
    prompt: str
    retrieved_chunks: list[SearchHit]
    latency_ms: float = Field(..., description="Generation wall-clock time (ARCHITECTURE.md §8).")
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = Field(
        None, description="Dollar cost of the generation; 0.0 for local, None if unknown."
    )
    metric: str = Field(..., description="The store's distance metric (score interpretation).")
    embedding_model: str = Field(..., description="Embedding model used for retrieval.")


# --- Scoring (optional RAGAS-style quality metrics for the compare grid) ----

class ScoreCell(BaseModel):
    """One grid cell to score: a generated answer + the context it was grounded
    in, tagged with which recipe/provider produced it so scores map back to the
    grid. `contexts` are the retrieved chunk texts (what the LLM actually saw)."""

    recipe_id: str
    provider: str
    question: str
    answer: str
    contexts: list[str] = Field(
        default_factory=list, description="Retrieved chunk texts the answer used."
    )
    ground_truth: str | None = Field(
        None,
        description="Optional reference answer. Required for context_recall; "
        "that metric is reported as null when it's absent.",
    )


class ScoreRequest(BaseModel):
    cells: list[ScoreCell]
    sort_by: str = Field(
        "faithfulness",
        description="Metric the ranking summary is sorted by "
        "(faithfulness/answer_relevancy/context_precision/context_recall/mean).",
    )


class ScoredCell(BaseModel):
    recipe_id: str
    provider: str
    # {metric: value|null} over the four metrics + a derived 'mean'. null = the
    # metric was not computed (e.g. context_recall with no ground truth) — an
    # honest gap, never a fabricated number.
    scores: dict[str, float | None]


class RankRow(BaseModel):
    rank: int
    recipe_id: str
    provider: str
    scores: dict[str, float | None]


class ScoreResponse(BaseModel):
    """Per-cell scores plus a ranking summary sorted by the chosen metric. When
    ragas is absent, `available` is False and `message` carries the friendly
    install note instead of scores — the tool never fails, it just opts out."""

    available: bool
    message: str | None = Field(
        None, description="Friendly note shown when scoring is unavailable."
    )
    metrics: list[str] = Field(..., description="Metric keys reported per cell.")
    sort_by: str
    cells: list[ScoredCell] = Field(default_factory=list)
    ranking: list[RankRow] = Field(default_factory=list)


class ScoreStatus(BaseModel):
    """Whether optional scoring can run right now (i.e. ragas is importable)."""

    available: bool
    message: str | None = None
    metrics: list[str]
