# 06 — Chunking Strategies

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Add the remaining chunking strategies from SPEC.md §3 alongside the existing
`recursive`: `fixed_size` (size, overlap), `sentence` (sentences_per_chunk,
overlap_sentences), `semantic` (similarity_threshold, takes an embedder instance),
and `structure_aware` (Markdown header_levels). Each is a new file in
backend/stages/chunkers/ + one registry line — no edits outside that package. All
params must be visible/overridable in the UI and recorded in config.json (no hidden
defaults). Each must produce truthful overlap_with_prev. Add per-chunker tests
against sample_data with a chunk-count assertion.
```
