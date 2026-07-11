"""Semantic chunker — groups sentences using similarity-based boundaries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .base import BaseChunker, Chunk
from backend.stages.parsers.base import ParsedDocument

if TYPE_CHECKING:
    from backend.stages.embedders.base import BaseEmbedder


def approx_token_count(text: str) -> int:
    """Rough token count: word-ish runs plus standalone punctuation.

    This is deliberately model-agnostic (no tiktoken / model tokenizer) so the
    chunker never secretly depends on which embedder you picked. It is an
    approximation for display, not an exact match to any model's tokenizer.
    """
    return len(re.findall(r"\w+|[^\w\s]", text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving sentence boundaries.

    A heuristic: looks for sentence-ending punctuation (. ! ?) followed by a
    space and a capital letter, or end-of-string. Not perfect for "Dr. Smith"
    but works well for prose. Sentences are returned *with* their closing
    punctuation attached.
    """
    pattern = r"([.!?]+)(?:\s+(?=[A-Z])|$)"
    sentences = []
    last_end = 0

    for match in re.finditer(pattern, text):
        end = match.end(1)
        sentence = text[last_end : end].strip()
        if sentence:
            sentences.append(sentence)
        last_end = end

    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append(remaining)

    return sentences


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticChunker(BaseChunker):
    """Group sentences by semantic similarity, creating boundaries at low-similarity transitions.

    What it does (mechanically): splits text into sentences, embeds each one
    using the provided embedder, then computes cosine similarity between
    consecutive sentences. A new chunk boundary is created whenever the
    similarity drops below ``similarity_threshold``. This naturally groups
    semantically related sentences together.

    Tradeoff vs. the alternatives: respects semantic relatedness, so chunks
    naturally align with topic shifts rather than arbitrary boundaries. The
    cost is that it requires an embedder model and is slower (must embed every
    sentence at build time). Chunk sizes depend on the document's actual
    semantic structure, not on a configured parameter. It is the most
    "intelligent" option and the slowest.

    When a learner would prefer it: when you want chunks to respect the
    document's natural topic boundaries and you have the compute budget to
    embed every sentence. Works well for long-form content with clear topic
    shifts (reports, research papers, narrative text).

    Parameters (all recorded in config.json, none hidden):
        similarity_threshold: sentences below this cosine similarity create a
            chunk boundary (default 0.3, range [0, 1]). Higher = fewer/larger
            chunks; lower = more/smaller chunks.
        embedder: a BaseEmbedder instance. Required (no default). Passed by
            the caller, not instantiated here.
    """

    name = "Semantic (similarity-based)"
    description = (
        "Groups sentences by semantic similarity, creating chunk boundaries "
        "when similarity drops below a threshold. Respects topic shifts and "
        "naturally aligns chunks with semantic structure. Slower (embeds every "
        "sentence at build time). Good for long-form documents with clear "
        "topic boundaries."
    )

    def chunk(
        self,
        doc: ParsedDocument,
        similarity_threshold: float = 0.3,
        embedder: BaseEmbedder | None = None,
        **_ignored,
    ) -> list[Chunk]:
        if embedder is None:
            raise ValueError(
                "semantic chunker requires an embedder instance; "
                "none was passed (embedder=None)"
            )
        if not (0 <= similarity_threshold <= 1):
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )

        sentences = split_sentences(doc.text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                Chunk(
                    chunk_id="chunk_0000",
                    text=sentences[0],
                    source=doc.source,
                    position=0,
                    char_len=len(sentences[0]),
                    token_len=approx_token_count(sentences[0]),
                    overlap_with_prev=0,
                )
            ]

        # Embed all sentences.
        embeddings = embedder.embed(sentences)

        chunks: list[Chunk] = []
        position = 0
        buffer: list[str] = []
        prev_embedding = None

        for sent_idx, sentence in enumerate(sentences):
            buffer.append(sentence)
            # Handle both numpy arrays and lists from embedder.
            emb = embeddings[sent_idx]
            curr_embedding = emb.tolist() if hasattr(emb, 'tolist') else emb

            # Check if we should create a boundary (except before the first sentence).
            if sent_idx > 0 and prev_embedding is not None:
                similarity = cosine_similarity(prev_embedding, curr_embedding)
                if similarity < similarity_threshold:
                    # Create a chunk boundary.
                    text = " ".join(buffer[:-1])  # Exclude current sentence.
                    if text.strip():  # Only if non-empty.
                        chunks.append(
                            Chunk(
                                chunk_id=f"chunk_{position:04d}",
                                text=text,
                                source=doc.source,
                                position=position,
                                char_len=len(text),
                                token_len=approx_token_count(text),
                                overlap_with_prev=0,  # Will recalculate below.
                            )
                        )
                        position += 1
                    buffer = [sentence]

            prev_embedding = curr_embedding

        # Flush any remaining sentences.
        if buffer:
            text = " ".join(buffer)
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{position:04d}",
                    text=text,
                    source=doc.source,
                    position=position,
                    char_len=len(text),
                    token_len=approx_token_count(text),
                    overlap_with_prev=0,  # Will recalculate below.
                )
            )

        # Recalculate actual overlap_with_prev (sentences don't naturally overlap,
        # so this will be 0 for most cases unless the chunk boundary cutting left
        # a trailing sentence that should have been in the previous chunk).
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr_text = chunks[i].text

            # Count the longest prefix of curr that is a suffix of prev.
            match_count = 0
            for length in range(1, min(len(prev_text), len(curr_text)) + 1):
                if prev_text[-length:] == curr_text[:length]:
                    match_count = length

            chunks[i].overlap_with_prev = match_count

        return chunks
