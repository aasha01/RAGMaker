# 05 — RAGAS-Style Scoring

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Add optional automated quality scoring to the comparison grid (ARCHITECTURE.md §6):
faithfulness, answer relevancy, context precision/recall. Feature-gate it behind a
lazy import (ragas) so the tool runs without it; if the package is absent, show a
friendly "install ragas to enable scoring" note rather than failing. Surface scores
per cell and let the ranking summary sort by any of them. Ask before adding ragas
as a dependency. Tests with mocked scorers.
```
