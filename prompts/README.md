# RAG Lab — Remaining Work Prompts

One prompt per outstanding item vs. `ARCHITECTURE.md` / `SPEC.md`. Each file
holds a single copy-paste-ready prompt. Work them roughly top to bottom.

**You do not need to restate the repo rules.** `CLAUDE.md` is auto-loaded at the
start of every session, so its **Definition of Done** checklist (Strategy
pattern, 3-part teaching docstrings, per-stage tests green, `IMPLEMENTATION.md`
updated, ask before adding deps, no hidden defaults, no silent fallback) is
applied automatically to whatever prompt you paste.

## Priority 1 — Unlock the app's core value (Part B: Query & Compare)

| # | File | Item |
|---|------|------|
| 01 | [01-llm-providers.md](01-llm-providers.md) | LLM providers (ollama / openai / anthropic) |
| 02 | [02-answer-generation-endpoint.md](02-answer-generation-endpoint.md) | Answer generation endpoint |
| 03 | [03-retrieval-strategies.md](03-retrieval-strategies.md) | Query-time retrieval strategies |
| 04 | [04-query-compare-grid.md](04-query-compare-grid.md) | Query & Compare tab + comparison grid |
| 05 | [05-ragas-scoring.md](05-ragas-scoring.md) | RAGAS-style scoring |

## Priority 2 — Make each build stage comparable (Part A alternatives)

| # | File | Item |
|---|------|------|
| 06 | [06-chunkers.md](06-chunkers.md) | Chunkers (fixed_size / sentence / semantic / structure_aware) |
| 07 | [07-embedders.md](07-embedders.md) | Embedders (openai / cohere / BGE) |
| 08 | [08-parsers.md](08-parsers.md) | Parsers (langchain / llamaindex) |
| 09 | [09-vector-stores.md](09-vector-stores.md) | Vector stores (chroma / qdrant) |
| 10 | [10-ingestion-sources.md](10-ingestion-sources.md) | Ingestion sources (CSV/JSON / URL / folder) |

## Priority 3 — Inspection visualizations (the "learning core", §5)

| # | File | Item |
|---|------|------|
| 11 | [11-inspection-visualizations.md](11-inspection-visualizations.md) | Inspection visualizations + core/inspector.py |
