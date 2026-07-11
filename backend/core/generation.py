"""RAG prompt assembly + answer generation (query-time orchestration).

This is deliberately **explicit and readable** because RAG Lab is a teaching
tool: the exact text handed to the LLM is built here in plain sight and returned
to the UI, rather than hidden behind a framework's opaque prompt template. A
learner should be able to read `build_rag_prompt` and know precisely what
context the model saw.

Generation is a *query-time* step, not part of the immutable recipe build — so
nothing here mutates a saved recipe. The cost/time it reports (ARCHITECTURE.md
§8 transparency) travels back in the response and into the compare grid
(ARCHITECTURE.md §9), never back into the recipe's `metadata.json`.

Like `recipe.py`, this module only ever talks to the LLM provider REGISTRY,
never to a concrete provider — so which model answers is fully described by the
request.
"""

from __future__ import annotations

from backend.stages.llm_providers import REGISTRY as LLM_PROVIDERS
from backend.stages.llm_providers.base import GenerationResult
from backend.stages.vectorstores.base import SearchResult

#: The instruction prepended to every RAG prompt. Kept as a named constant (not
#: buried in an f-string) so a learner can see and tweak the one line that turns
#: retrieved passages into a grounded-answer task.
RAG_INSTRUCTION = (
    "You are a helpful assistant answering a question using only the context "
    "passages provided below. Use only information found in the context. If the "
    "context does not contain the answer, say you don't know — do not rely on "
    "outside knowledge or guess."
)


def build_rag_prompt(
    question: str,
    hits: list[SearchResult],
    instruction: str = RAG_INSTRUCTION,
) -> str:
    """Assemble the exact prompt sent to the LLM: instruction + numbered context
    passages + the question.

    Each retrieved chunk is labelled with its source, chunk id, and similarity
    score, so the learner (and, usefully, the model) can see *where* each piece
    of context came from and how strongly it matched. Nothing is summarised or
    hidden — this string is returned verbatim in the API response.
    """
    if hits:
        blocks = []
        for i, hit in enumerate(hits, start=1):
            c = hit.chunk
            header = (
                f"[{i}] (source: {c.source}, chunk: {c.chunk_id}, "
                f"score: {hit.score:.4f})"
            )
            blocks.append(f"{header}\n{c.text}")
        context = "\n\n".join(blocks)
    else:
        context = "(no context was retrieved)"

    return (
        f"{instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_answer(
    question: str,
    hits: list[SearchResult],
    provider_key: str,
    provider_params: dict | None = None,
    gen_params: dict | None = None,
) -> tuple[str, GenerationResult, str | None]:
    """Build the RAG prompt, run the chosen provider, and return
    `(prompt, result, model)`.

    `provider_params` are the provider **constructor** args (e.g. model, api_key);
    `gen_params` are per-call `generate()` kwargs (e.g. temperature). The prompt
    is returned alongside the result so the caller can surface exactly what was
    sent — the whole point of a teaching tool.

    Raises `ValueError` for an unknown provider key (no silent fallback to a
    different provider); any error the provider raises (missing package/API key,
    unreachable server) propagates unchanged so it can be surfaced verbatim.
    """
    if provider_key not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider_key}'. Available: {sorted(LLM_PROVIDERS)}"
        )

    prompt = build_rag_prompt(question, hits)
    provider = LLM_PROVIDERS[provider_key](**(provider_params or {}))
    gen_kwargs = gen_params or {}
    result = provider.generate(prompt, **gen_kwargs)

    # Report the model actually used: a per-call override wins over the
    # provider's configured default. `model` is a convention on the concrete
    # providers, not part of the base interface, so read it defensively.
    model = gen_kwargs.get("model") or getattr(provider, "model", None)
    return prompt, result, model
