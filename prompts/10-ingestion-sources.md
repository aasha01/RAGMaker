# 10 — Ingestion Sources (Stage 1)

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Broaden ingestion (ARCHITECTURE.md §3, Stage 1) beyond single-file upload to add:
CSV/JSON, URL, and folder ingestion. Keep it explicit and inspectable (filename,
size, type, raw preview per §5). Decide whether this is a distinct configurable
stage or an extension of the parser input and document the choice in
IMPLEMENTATION.md. Lazy-import any network/parse deps; fail loudly. Ask before
adding deps. Tests for each source type.
```
