# 11 — Inspection Visualizations + core/inspector.py

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Implement the stage-inspection visualizations from ARCHITECTURE.md §5 that are
still missing, and create core/inspector.py (referenced in §9 but absent) to hold
the reusable per-stage visualization helpers instead of inlining everything in
frontend/app.py. Add:
- Chunking: chunk-length histogram.
- Embedding: similarity heatmap across a chunk sample.
- Vector Store: 2D PCA (and optional UMAP) projection scatter, colored by source doc.
- Query: retrieved chunks highlighted in the context of the original parsed document.
Use plotly/scikit-learn already in requirements; ask before adding umap-learn.
Keep helpers pure/testable. These render live during build AND when reopening a
saved recipe. Add tests for the data-prep functions.
```
