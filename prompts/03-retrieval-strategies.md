# 03 — Query-Time Retrieval Strategies

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Implement the query-time retrieval stage from SPEC.md §6 as swappable strategies
(recorded per-query, NOT per-recipe, so one recipe can be tested with several).
Create backend/stages/retrievers/ with base.py + a registry, and implement:
`naive_topk` (extract the plain top-k currently baked into the store),
`mmr` (diversity-aware), `hybrid` (BM25 + vector), `rerank` (vector top-N →
cross-encoder → top-k), and `hyde` (LLM writes a hypothetical answer, embed that).
naive_topk must require no API key. Lazy-import optional deps; fail loudly on
missing ones. Wire a `retriever` param into the search/generate endpoints and the
UI. Tests per strategy; ask before adding deps (BM25, cross-encoder, etc.).
```
