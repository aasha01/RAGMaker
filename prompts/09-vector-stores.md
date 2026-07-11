# 09 — Vector Store Strategies

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Add `chroma` and `qdrant` vector stores from SPEC.md §5 alongside `faiss`, behind
the existing BaseVectorStore interface (build/save/load/search). Support the metric
param; persist to the recipe's vectorstore/ dir. Keep the dimension + model guards
that prevent cross-model contamination (ARCHITECTURE.md §8). Lazy-import chromadb /
qdrant-client; fail loudly if absent. New files + registry lines only. Ask before
adding deps. Tests: build + round-trip save/load + search on sample_data.
```
