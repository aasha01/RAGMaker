# CLAUDE.md — Operating Instructions for This Repo

This file tells an AI coding agent (Claude Code or similar) how to work in
this repository. Read this before touching any code.

## Project Purpose

This is a **teaching tool for RAG (Retrieval-Augmented Generation) architecture**.
The primary user is a learner, not a production engineer. Every design decision
should optimize for **transparency and comparability** over performance or
cleverness. If a shortcut would hide what's actually happening at a stage,
don't take it.

Two concrete consequences:

1. Prefer explicit, readable code over abstraction-heavy "smart" code.
2. Every stage must persist its intermediate output to disk (see
   ARCHITECTURE.md → "Stage Inspection"). Never treat a stage's output as a
   throwaway in-memory value only.

## Core Design Pattern: Strategy Pattern, Everywhere

Every pipeline stage (parser, chunker, embedder, vector store, LLM provider)
is implemented as a class fulfilling an abstract base interface defined in
`stages/<stage>/base.py`. The orchestration code (`core/recipe.py`) only ever
talks to the interface, never to a concrete implementation directly.

**Rule: adding a new technique for an existing stage must never require
editing any file outside `stages/<that_stage>/`, plus one line to register it
in that stage's `__init__.py` registry.**

Do not add cross-stage shortcuts (e.g. a chunker peeking at which embedder
will be used) — that breaks the swappability the whole project depends on.

## Documentation-as-Code

Every strategy class must have a docstring answering three things:

1. What it does (mechanically)
2. What tradeoff it represents vs. the other options in that stage
3. One concrete situation where a learner would prefer it

This docstring is surfaced directly in the UI next to the option — it is not
just internal documentation, it's user-facing teaching content. Treat it as
such: plain language, no jargon left unexplained.

## Files to Read Before Building

Read in this order:

1. `ARCHITECTURE.md` — overall system design, data flow, persistence layout
2. `SPEC.md` — per-stage options, parameters, and config schema
3. This file

## Implementation Log (living documentation) — MANDATORY

`IMPLEMENTATION.md` is the running record of every change. **Every time you
finish a change** (a build step, a pipeline stage, a bug fix, a new endpoint,
a dependency, a refactor), you MUST, before reporting the change as done:

1. Add a new row to the **changelog table at the top** of `IMPLEMENTATION.md`
   (newest on top): number, date, one-line change, stage/step, key files,
   tests. Also update the "running total" line beneath the table.
2. Add or extend a **detailed section** below documenting what was built, the
   endpoints/files touched, the design decisions, and how to verify it.

This is not optional and not deferrable to "later" — an undocumented change is
an incomplete change. Read `IMPLEMENTATION.md` at the start of any session to
recover context. Keep it accurate: if a later change supersedes an earlier
one, note it rather than silently leaving stale text.

Also standing: **write per-stage tests and run them (green) before declaring a
stage/change done**, and **ask before adding any dependency** not already
present.

## Definition of Done — apply to EVERY task/prompt (auto-loaded)

This file is loaded automatically at the start of every session. **Any task
prompt you are given inherits the rules below without needing to restate them.**
Before you report *any* change as done, every box must be true:

- [ ] Read `ARCHITECTURE.md`, `SPEC.md`, and this file before writing code
      (see "Files to Read Before Building").
- [ ] New technique = a new file inside `stages/<that_stage>/` **plus one
      registry line** in its `__init__.py` — nothing edited outside that
      package (Strategy Pattern rule).
- [ ] Every strategy class carries the 3-part teaching docstring (what /
      tradeoff / when) — it is user-facing UI content ("Documentation-as-Code").
- [ ] Every parameter is visible + overridable in the UI and recorded in
      `config.json` — no hidden defaults (Non-Negotiables).
- [ ] No silent fallback on error — fail loudly with a learner-readable message.
- [ ] Optional/heavy deps are lazy-imported inside the method, with a friendly
      error if absent — never imported at module load (Style).
- [ ] Per-stage test written **and run green** against `sample_data/` before
      declaring done (Testing).
- [ ] **Asked before adding any dependency** not already present.
- [ ] `IMPLEMENTATION.md` updated (changelog row + running total + detail
      section) — an undocumented change is an incomplete change.

If a prompt conflicts with any box above, this file wins — flag the conflict
rather than silently dropping a rule.

## Build Order (recommended)

Build and manually test one vertical slice before generalizing:

1. `stages/parsers/base.py` + one concrete parser (manual, txt/pdf only)
2. `stages/chunkers/base.py` + one concrete chunker (recursive)
3. `stages/embedders/base.py` + one concrete embedder (local
   sentence-transformers — no API key required, keeps the tool runnable
   out of the box)
4. `stages/vectorstores/base.py` + one concrete store (FAISS, local, no
   server required)
5. `core/recipe.py` — orchestrates the above into one buildable recipe,
   persists `config.json` + `stage_outputs/` + `vectorstore/`
6. Minimal Streamlit `app.py` — Build tab only, single-path, no options yet
7. Only once that end-to-end path works, add the remaining strategies per
   stage and wire up the option pickers
8. Add the Query & Compare tab last, once ≥2 recipes exist to compare

## Non-Negotiables

- No hidden defaults: if a stage has a parameter (chunk size, distance
  metric, temperature), it must be visible and overridable in the UI, and
  recorded in `config.json`. A recipe's `config.json` alone must be
  sufficient to describe how to reproduce it.
- No stage may silently fall back to a different technique on error. Fail
  loudly with a clear message the learner can understand.
- Every recipe is immutable once built. "Editing" a recipe means creating a
  new one with a new ID. This is what makes comparison trustworthy.
- Never mix embeddings from two different models in the same vector store.
  Validate model + dimension match at store-write time and at query time.

## Style

- Python 3.11+, type hints on all public functions.
- Prefer `dataclasses` for config/data objects over raw dicts.
- Keep functions short enough that a learner reading the source (not just
  the UI) can understand a stage in under a minute.
- Avoid dependency-heavy magic imports at module load time for optional
  strategies (e.g. LlamaIndex, Cohere) — import them lazily inside the
  class method so the tool still runs if that optional package isn't
  installed, and surface a friendly error in the UI instead of a crash.

## Testing

Each stage implementation should have a small test using a fixed sample
document (`sample_data/`) with an assertion on chunk count / vector
dimension / etc. — not because correctness is subtle, but so a learner
modifying a strategy gets fast feedback that they didn't break the
interface contract.
