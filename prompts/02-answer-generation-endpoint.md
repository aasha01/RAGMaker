# 02 — Answer Generation Endpoint

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Add answer generation over HTTP so a recipe can actually produce an answer, not
just retrieve chunks. Add POST /recipes/{recipe_id}/generate that: retrieves
top-k chunks from the recipe's vector store, assembles a RAG prompt (retrieved
context + question), calls a chosen LLM provider from stages/llm_providers, and
returns the answer plus retrieved chunks, latency, token counts, and cost. Keep
prompt assembly explicit and readable (this is a teaching tool — the learner
should see exactly what context was sent). Record generation cost/time per
ARCHITECTURE.md §8. Add tests with a mock provider.
```
