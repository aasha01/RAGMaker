"""HyDE retrieval — embed a hypothetical *answer* instead of the question."""

from __future__ import annotations

from .base import BaseRetriever
from backend.stages.embedders.base import BaseEmbedder
from backend.stages.llm_providers import REGISTRY as LLM_PROVIDERS
from backend.stages.vectorstores.base import BaseVectorStore, SearchResult

#: The instruction handed to the LLM to draft the hypothetical answer. Kept as a
#: named constant (not buried in an f-string) so a learner can see and tweak the
#: one line that defines HyDE's whole idea.
HYDE_INSTRUCTION = (
    "Write a short, factual passage that could plausibly answer the following "
    "question, as if it were an excerpt from a relevant document. Do not say you "
    "are unsure or lack context — just write the passage. Question: {question}"
)


class HyDERetriever(BaseRetriever):
    """Ask an LLM to *guess* an answer first, then search with that guess instead
    of the raw question (Hypothetical Document Embeddings).

    What it does (mechanically): a short question and the passages that answer it
    often look quite different ("How is pneumonia treated?" vs. "The patient
    received IV ceftriaxone for 5 days..."). So HyDE first asks an LLM to write a
    *hypothetical* answer passage, embeds **that** passage, and uses its vector to
    search the store. The hypothesis doesn't need to be correct — it just needs to
    look like the kind of document that would contain the real answer, which pulls
    the query vector into the right neighbourhood.

    Tradeoff vs. the alternatives: it can markedly improve recall on short or
    vague questions where the query and the answer share little vocabulary. The
    costs are real: it needs an LLM call *before* retrieval even starts (latency,
    and a running Ollama or an API key), and a confidently wrong hypothesis can
    drag the search toward the wrong region — so it's the least predictable option
    here.

    When a learner would prefer it: for terse or abstract questions that naive
    top-k answers poorly because the wording doesn't match the source text. Run
    the same question with naive top-k and with HyDE and compare which chunks come
    back.

    Parameters (recorded per query):
        provider: LLM provider key for the hypothesis (default 'ollama' — local,
            no API key, but needs a running Ollama server).
        provider_params: constructor args for that provider (e.g. {'model': ...}).
        gen_params: per-call generate() kwargs (e.g. {'temperature': 0.0}).

    Requires an LLM to generate the hypothesis, so — unlike naive_topk/mmr — this
    strategy is not guaranteed API-key-free; it depends on the chosen provider.
    """

    name = "HyDE (hypothetical answer)"
    description = (
        "Asks an LLM to draft a hypothetical answer passage, embeds that passage "
        "instead of the raw question, and searches with it — helping when short "
        "questions don't lexically resemble the answers. Needs an LLM (default "
        "local Ollama; no key but a running server), so it's the one retriever "
        "that isn't guaranteed key-free."
    )
    no_api_key = False  # depends on the chosen generation provider

    def retrieve(
        self,
        query: str,
        *,
        store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int,
        provider: str = "ollama",
        provider_params: dict | None = None,
        gen_params: dict | None = None,
        **_ignored,
    ) -> list[SearchResult]:
        if provider not in LLM_PROVIDERS:
            # No silent fallback to a different provider (or to naive retrieval).
            raise ValueError(
                f"Unknown LLM provider '{provider}' for HyDE. "
                f"Available: {sorted(LLM_PROVIDERS)}"
            )

        llm = LLM_PROVIDERS[provider](**(provider_params or {}))
        result = llm.generate(HYDE_INSTRUCTION.format(question=query), **(gen_params or {}))
        hypothetical = result.text

        # Embed the hypothetical answer (not the question) and search with it.
        query_vec = embedder.embed([hypothetical])
        return store.search(query_vec, top_k=top_k)
