# 04 — Query & Compare Tab + Comparison Grid

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Build Part B (ARCHITECTURE.md §2, §6). Add a "Query & Compare" tab to
frontend/app.py and the backend to support it (core/evaluator.py, referenced in
ARCHITECTURE.md §9, does not exist yet — create it). The user asks ONE question;
it runs against every selected saved recipe × selected LLM provider (full grid is
the default). Per cell show: generated answer, retrieved chunks + scores, latency,
token cost. Add a summary table that ranks all cells by a chosen metric. Follow
the comparison-grid output schema in SPEC.md §9. Add tests.
```
