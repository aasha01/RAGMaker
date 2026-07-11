# PROMPT.md — Build Prompt

Paste this into Claude Code (or any coding agent) in the project root, after
placing `CLAUDE.md`, `ARCHITECTURE.md`, and `SPEC.md` there.

---

```
Read CLAUDE.md, ARCHITECTURE.md, and SPEC.md fully before writing any code.
This is a learning tool for RAG architecture — transparency and per-stage
inspectability matter more than performance or feature count.

Build this project in the exact incremental order described in CLAUDE.md
section "Build Order". Do not jump ahead to later steps before the current
one runs end-to-end. After each step, tell me what you built and how to
manually verify it before moving to the next step.

Step 1: Scaffold the repo layout from ARCHITECTURE.md section 9, with empty
__init__.py registries and abstract base classes only (no concrete
strategies yet). Show me the base interfaces for my review before
implementing any concrete strategy.

Step 2: Implement exactly one concrete strategy per stage — the ones marked
as the "no API key required" default path in SPEC.md (manual parser,
recursive chunker, sentence-transformers embedder, FAISS store). Write a
small script (not the UI yet) that runs sample_data through all five stages
and prints/saves the stage_outputs/ artifacts described in ARCHITECTURE.md
section 4. Confirm this works on a real sample file before continuing.

Step 3: Build core/recipe.py to orchestrate step 2 into a saved Recipe
folder (config.json + stage_outputs/ + vectorstore/), matching the schema
in SPEC.md section 8.

Step 4: Build a minimal Streamlit app.py with just the Build tab, single
default path, showing the stage inspection views listed in ARCHITECTURE.md
section 5 (start with the simplest version of each — text previews and
counts are fine before adding histograms/heatmaps/projections).

Step 5: Add the remaining strategies for each stage (per SPEC.md tables),
wire them into the UI as dropdowns/radio options with their docstring
descriptions shown inline, and add the "redo this stage" control.

Step 6: Add the richer inspection visuals — chunk length histogram, overlap
highlighting, embedding similarity heatmap, 2D vector projection scatter
plot.

Step 7: Build core/evaluator.py and the Query & Compare tab: run one
question against multiple saved recipes and LLM providers (full grid mode
per ARCHITECTURE.md section 6), showing retrieved chunks, answers, latency,
cost, and the summary ranking table.

Step 8: Add the pluggable LLM providers (OpenAI, Anthropic, Ollama) behind
the interface in SPEC.md section 7, gated so the app still runs without
API keys configured (Ollama or a clear "no provider configured" message).

At every step: follow the Non-Negotiables in CLAUDE.md (no silent fallback,
no hidden defaults, embedding/model mismatch validation, config.json must
fully describe the recipe). Ask me before adding any dependency not listed
in SPEC.md section 10.
```

---

## Notes for you (not part of the pasted prompt)

- Feed the agent one sample file first (put something small in
  `sample_data/`, e.g. a 2-page PDF or a markdown doc) so Step 2 has
  something concrete to run against.
- If using Claude Code, `CLAUDE.md` in the project root is picked up
  automatically as project instructions — you generally won't need to
  re-paste its contents into the prompt, just make sure the file exists at
  the repo root before you start.
- Review the base interfaces (end of Step 1) before letting the agent
  proceed — this is the one place where getting it wrong cascades into
  every later step.
