# 01 — LLM Providers

> Standing rules are auto-applied from `CLAUDE.md` → **Definition of Done**
> (loaded every session). Just paste the prompt below.


## Prompt

```
Implement the LLM provider stage. The registry at backend/stages/llm_providers/__init__.py
is empty. Add three concrete providers per SPEC.md §7 behind the existing
BaseLLMProvider interface: `ollama` (local, no API key, zero cost — the default,
runnable out of the box), `openai`, and `anthropic`. Each generate() must return
a full GenerationResult (text, latency_ms, input_tokens, output_tokens, cost_usd).
Lazy-import each SDK inside the method so the tool still runs if that package is
absent, surfacing a friendly error, never a crash or silent fallback. For
Anthropic, default to the latest Claude models (load the claude-api skill for
correct model IDs/pricing). Add a GET /stages/llm_providers discovery endpoint
mirroring the other stages. Write and run tests (mock the network calls); ask
before adding any SDK dependency.
```
