"""Query-time answer scoring — the "automated quality scores" of the compare
grid (ARCHITECTURE.md §6, §9's `core/evaluator.py`).

This module gives each cell of the Query & Compare grid an *optional* set of
RAGAS-style quality numbers — **faithfulness, answer relevancy, context
precision, context recall** — so the "which recipe/provider is best?" question
has a concrete, sortable answer instead of a vibe.

**It is entirely optional and feature-gated.** The heavy `ragas` package (and
the judge LLM it drives) is imported *lazily*, only inside the scorer that needs
it. If `ragas` isn't installed the tool keeps working exactly as before — the UI
just shows a friendly "install ragas to enable scoring" note instead of scores.
Nothing here is imported at module load beyond the standard library, so merely
importing this file never drags in ragas.

Like `generation.py`, this is *orchestration*, not a pipeline stage: it doesn't
own a swappable-strategy REGISTRY. But scoring is still pluggable behind
`BaseScorer` so (a) an alternative scorer can be dropped in and (b) tests can
inject a mock and never touch ragas or a judge LLM.

The four metrics, in plain language (this text is teaching content):

* **faithfulness** — is every claim in the answer actually supported by the
  retrieved context? Catches hallucination.
* **answer_relevancy** — does the answer actually address the question asked,
  rather than wandering off?
* **context_precision** — of the chunks we retrieved, how many were actually
  relevant? Low precision = the retriever pulled in noise.
* **context_recall** — did retrieval find *all* the context needed to answer?
  Needs a reference ("ground truth") answer to know what "all" was, so it is the
  one metric that is skipped (returns None) when no reference is supplied.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

#: The four quality metrics, in the order shown in the UI. `mean` (their
#: average) is derived on top of these, not a metric ragas computes.
SCORE_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

#: Metrics that cannot be computed without a reference / ground-truth answer.
_NEEDS_GROUND_TRUTH = frozenset({"context_recall"})

#: Shown verbatim in the UI when ragas is absent — a friendly note, not an error.
RAGAS_INSTALL_HINT = (
    "Automated quality scoring is turned off because the optional 'ragas' "
    "package isn't installed. Install it to enable faithfulness / answer "
    "relevancy / context precision & recall scoring:\n\n    pip install ragas\n\n"
    "ragas needs a judge LLM for scoring. Three options:\n\n"
    "  • Local (free): Ollama — run 'ollama serve' then set RAGAS_LLM_PROVIDER=ollama\n"
    "  • Hosted (paid): Anthropic — set ANTHROPIC_API_KEY and RAGAS_LLM_PROVIDER=anthropic\n"
    "  • Hosted (paid): OpenAI — set OPENAI_API_KEY (ragas default if no others configured)\n\n"
    "Ollama is the zero-cost, zero-key option: download a model with 'ollama pull llama3.1:8b', "
    "then scoring runs fully offline."
)


@dataclass
class ScoreSample:
    """One thing to score: a question, the answer produced for it, the context
    that answer was grounded in, and (optionally) a reference answer.

    `contexts` are the *texts* of the retrieved chunks — exactly what the LLM
    saw — so faithfulness/precision judge against the real evidence. `ground_truth`
    is an optional human reference answer; without it, `context_recall` can't be
    computed (there's nothing to measure completeness against) and is reported as
    None rather than guessed.
    """

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str | None = None


class RagasNotInstalled(RuntimeError):
    """Raised when scoring is attempted but `ragas` isn't importable.

    Carries the learner-readable install hint as its message so the API layer can
    turn it straight into the friendly UI note (never a bare stack trace).
    """

    def __init__(self, message: str = RAGAS_INSTALL_HINT) -> None:
        super().__init__(message)


def ragas_available() -> bool:
    """True iff the optional `ragas` package can be imported.

    Uses `find_spec` so we can answer the question *without* importing ragas (and
    without paying its heavy import cost) — the UI calls this to decide whether to
    offer scoring or show the install note.
    """
    return importlib.util.find_spec("ragas") is not None


class BaseScorer(ABC):
    """A pluggable answer scorer. The real one wraps ragas; tests inject a mock.

    Kept deliberately small: `available()` gates whether scoring can run at all
    (so the grid can show the install note before doing any work), and `score()`
    returns, for each input sample, a dict of `{metric: value|None}` over
    `SCORE_METRICS`. A value of None means "not computed" (e.g. context_recall
    with no ground truth) — an honest gap, never a fabricated number.
    """

    metrics: tuple[str, ...] = SCORE_METRICS

    def available(self) -> bool:
        """Whether this scorer can run. Override to gate on an optional dep."""
        return True

    def unavailable_message(self) -> str:
        """Friendly, learner-readable reason shown when `available()` is False."""
        return ""

    @abstractmethod
    def score(self, samples: list[ScoreSample]) -> list[dict[str, float | None]]:
        """Score every sample; return one `{metric: value|None}` dict per sample,
        aligned by index with `samples`."""
        raise NotImplementedError


class RagasScorer(BaseScorer):
    """The real scorer: computes the four metrics with the `ragas` library.

    Tradeoff vs. the mock/other options: these are *model-graded* metrics — ragas
    prompts a judge LLM (and an embedding model for answer relevancy) to grade
    each answer. That makes them powerful (they measure hallucination and
    retrieval quality directly) but also **slow and dependent on an optional
    package + an LLM**. The LLM can be free (Ollama, local) or metered (OpenAI,
    Anthropic). That's exactly why scoring is off by default and lazy: a learner
    turns it on deliberately.

    When a learner would prefer it: after building two or three recipes, to get a
    concrete ranking ("recipe_002 hallucinates less") instead of eyeballing
    answers side by side.

    `ragas` and the judge/embeddings are imported and constructed **lazily inside
    `score()`** — importing this module never pulls in ragas. Any failure (package
    missing, judge LLM/API key not configured, a metric erroring) surfaces loudly
    with a readable message; there is no silent fallback to a different metric or
    a made-up score.
    """

    name = "RAGAS"
    description = (
        "Model-graded RAG quality metrics (faithfulness, answer relevancy, "
        "context precision & recall) computed by the ragas library. Needs the "
        "optional 'ragas' package and a judge LLM (Ollama local, Anthropic, or OpenAI)."
    )

    def __init__(
        self,
        metrics: tuple[str, ...] = SCORE_METRICS,
        llm=None,
        embeddings=None,
        lazy_llm_factory=None,
    ) -> None:
        # Validate the requested metric names up front (fail loudly, not later).
        unknown = [m for m in metrics if m not in SCORE_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown score metric(s) {unknown}. Available: {list(SCORE_METRICS)}"
            )
        self.metrics = tuple(metrics)
        # Optional ragas-compatible judge LLM / embeddings. If left None, ragas
        # uses its own default (OpenAI), which needs OPENAI_API_KEY.
        self._llm = llm
        self._embeddings = embeddings
        # Optional factory to initialize LLM lazily on first score() call
        self._lazy_llm_factory = lazy_llm_factory

    def available(self) -> bool:
        return ragas_available()

    def unavailable_message(self) -> str:
        return RAGAS_INSTALL_HINT

    def score(self, samples: list[ScoreSample]) -> list[dict[str, float | None]]:
        if not samples:
            return []

        # --- lazy imports: nothing heavy is imported until scoring is asked for.
        try:
            from datasets import Dataset  # ragas builds evaluation over a Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:  # ragas (or its datasets dep) not installed
            raise RagasNotInstalled() from exc

        # Initialize judge LLM lazily on first score() call (if a factory is provided)
        if self._lazy_llm_factory and self._llm is None:
            try:
                self._llm = self._lazy_llm_factory()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize judge LLM for ragas: {exc}"
                ) from exc

        metric_objs = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

        # context_recall needs a reference for *every* row; if any sample lacks a
        # ground truth we skip the ground-truth metrics rather than feed ragas a
        # blank reference (which would produce a meaningless score). Skipped
        # metrics are reported as None for all rows — an explicit gap.
        all_have_gt = all((s.ground_truth or "").strip() for s in samples)
        run_metrics = [
            m
            for m in self.metrics
            if not (m in _NEEDS_GROUND_TRUTH and not all_have_gt)
        ]
        skipped = [m for m in self.metrics if m not in run_metrics]

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [list(s.contexts) for s in samples],
            # ragas expects a ground_truth column when a reference metric runs.
            "ground_truth": [(s.ground_truth or "") for s in samples],
        }
        dataset = Dataset.from_dict(data)

        evaluate_kwargs: dict = {"metrics": [metric_objs[m] for m in run_metrics]}
        if self._llm is not None:
            evaluate_kwargs["llm"] = self._llm
        if self._embeddings is not None:
            evaluate_kwargs["embeddings"] = self._embeddings

        try:
            result = evaluate(dataset, **evaluate_kwargs)
            frame = result.to_pandas()
        except Exception as exc:  # judge LLM/API key/metric failure — surface it
            raise RuntimeError(
                "ragas scoring failed. Configure a judge LLM:\n"
                "  • Ollama (free): RAGAS_LLM_PROVIDER=ollama (run 'ollama serve')\n"
                "  • Anthropic: RAGAS_LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY\n"
                "  • OpenAI: OPENAI_API_KEY (ragas default)\n"
                f"Original error: {exc}"
            ) from exc

        # ragas has renamed some output columns across versions; accept known
        # aliases so a version bump doesn't silently drop a metric.
        column_aliases = {
            "answer_relevancy": ("answer_relevancy", "response_relevancy"),
            "context_precision": (
                "context_precision",
                "llm_context_precision_with_reference",
            ),
            "context_recall": ("context_recall", "llm_context_recall"),
            "faithfulness": ("faithfulness",),
        }

        out: list[dict[str, float | None]] = []
        for i in range(len(samples)):
            row: dict[str, float | None] = {m: None for m in SCORE_METRICS}
            for metric in run_metrics:
                col = next(
                    (c for c in column_aliases.get(metric, (metric,)) if c in frame.columns),
                    None,
                )
                if col is not None:
                    val = frame[col].iloc[i]
                    row[metric] = None if val is None else float(val)
            out.append(row)
        # `skipped` metrics stay None in every row (already initialised so).
        _ = skipped
        return out


def mean_score(scores: dict[str, float | None]) -> float | None:
    """Average of the available metric values (None ones ignored); None if none
    are available. A convenience aggregate so the grid can rank by an overall
    number, not just a single metric."""
    vals = [scores[m] for m in SCORE_METRICS if scores.get(m) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def with_mean(scores: dict[str, float | None]) -> dict[str, float | None]:
    """Return a copy of `scores` with a derived `mean` key added."""
    enriched = dict(scores)
    enriched["mean"] = mean_score(scores)
    return enriched


def rank_cells(
    cells: list[dict], sort_by: str, descending: bool = True
) -> list[dict]:
    """Rank scored grid cells by one metric, best first, and stamp a 1-based
    `rank` on each.

    `cells` is a list of dicts each carrying a `scores` sub-dict. Cells whose
    chosen metric is None (not computed) always sort **last**, regardless of
    direction — a missing score should never masquerade as the best or worst.
    This is a pure function so the frontend can re-rank instantly by any metric
    without another backend round-trip.
    """

    def sort_key(cell: dict):
        value = cell.get("scores", {}).get(sort_by)
        missing = value is None
        numeric = 0.0 if value is None else float(value)
        # (missing sorts False<True → present first) then by value in the
        # requested direction.
        return (missing, -numeric if descending else numeric)

    ranked = sorted(cells, key=sort_key)
    return [{**cell, "rank": i + 1} for i, cell in enumerate(ranked)]


def create_ragas_scorer() -> RagasScorer:
    """Factory: create a RagasScorer with an optional local judge LLM.

    Tries to auto-detect or load a judge LLM (Ollama or Anthropic) from env vars.
    If no local LLM is configured, ragas will use OpenAI (needs OPENAI_API_KEY).

    Judge LLM initialization is deferred until score() is called via lazy_llm_factory,
    so dependency injection never raises (allowing /score/status to work even if
    the judge LLM can't be initialized). Judge LLM init errors are surfaced when
    scoring runs.

    This is the single entry point for creating scorers in the API layer. Tests
    can create a RagasScorer directly with a mock LLM/embeddings.
    """
    lazy_llm_factory = None

    # Only set up the factory if ragas is available AND a local LLM is configured
    if ragas_available():
        try:
            from backend.core.scoring_llm import get_ragas_judge_llm

            # Capture the factory function; it will be called lazily inside score()
            lazy_llm_factory = get_ragas_judge_llm
        except ImportError:
            # scoring_llm module import failed; just use ragas defaults
            pass

    return RagasScorer(lazy_llm_factory=lazy_llm_factory)


def score_samples(
    samples: list[ScoreSample], scorer: BaseScorer | None = None
) -> list[dict[str, float | None]]:
    """Score `samples` with `scorer` (a fresh `RagasScorer` if none given).

    Raises `RagasNotInstalled` if the scorer can't run (so the caller can show
    the friendly note). This is the single entry point the API layer uses; tests
    call it with a mock scorer to exercise everything without ragas or a key.
    """
    scorer = scorer or create_ragas_scorer()
    if not scorer.available():
        raise RagasNotInstalled(scorer.unavailable_message() or RAGAS_INSTALL_HINT)
    return scorer.score(samples)
