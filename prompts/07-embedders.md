# 07 — Embedding Strategies

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Add the remaining embedding strategies from SPEC.md §4 alongside
`sentence_transformers`: `openai` (text-embedding-3-small/-large, dimensions
truncation, normalize) and `cohere` (embed-v3, input_type); optionally BGE/Instructor
via sentence-transformers. Reuse the same model_name/normalize/truncate_dim kwargs
so the in-process cache and UI stay uniform (hosted models map truncate_dim → their
native `dimensions` param, per the forward note in IMPLEMENTATION.md Change #5).
Lazy-import SDKs; fail loudly. Preserve the model+dimension metadata that the store
guard checks. Ask before adding deps. Tests (mock hosted calls).
```
