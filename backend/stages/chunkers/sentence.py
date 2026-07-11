"""Sentence-based chunker — groups sentences into chunks."""

from __future__ import annotations

import re

from .base import BaseChunker, Chunk
from backend.stages.parsers.base import ParsedDocument


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
    # Match sentence-ending punctuation followed by space + capital or end.
    pattern = r"([.!?]+)(?:\s+(?=[A-Z])|$)"
    sentences = []
    last_end = 0

    for match in re.finditer(pattern, text):
        end = match.end(1)  # Position right after the punctuation.
        sentence = text[last_end : end].strip()
        if sentence:
            sentences.append(sentence)
        last_end = end

    # Catch any remaining text (text without sentence-ending punctuation).
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append(remaining)

    return sentences


class SentenceChunker(BaseChunker):
    """Group sentences into chunks.

    What it does (mechanically): splits the document into sentences (on . ! ?),
    then groups them greedily into chunks up to ``sentences_per_chunk``. If
    ``overlap_sentences > 0``, the last ``overlap_sentences`` sentences from
    each chunk are repeated at the start of the next.

    Tradeoff vs. the alternatives: respects sentence boundaries perfectly, so
    chunks always begin and end at sentence breaks — never mid-sentence or
    mid-word. The tradeoff is that chunk sizes vary based on sentence length,
    and the sentence splitter is a heuristic (can fail on abbreviations and
    unusual punctuation). It is more reliable than recursive chunking for data
    where sentence coherence matters.

    When a learner would prefer it: for articles, essays, and technical
    documentation where sentence-level granularity is important and you want
    to guarantee chunks never split a sentence. Less useful for code or
    pre-formatted content.

    Parameters (all recorded in config.json, none hidden):
        sentences_per_chunk: number of sentences per chunk (default 3).
        overlap_sentences: number of sentences from the previous chunk to
            repeat at the start of the next (default 0). Set 0 for no overlap.
    """

    name = "Sentence-based"
    description = (
        "Groups sentences into chunks, respecting sentence boundaries always. "
        "Guarantees no mid-sentence splits. Chunk sizes vary based on sentence "
        "length. Good for prose where sentence coherence is critical; the "
        "sentence splitter is a heuristic (can miss abbreviations)."
    )

    def chunk(
        self,
        doc: ParsedDocument,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 0,
        **_ignored,
    ) -> list[Chunk]:
        if sentences_per_chunk <= 0:
            raise ValueError(f"sentences_per_chunk must be positive, got {sentences_per_chunk}")
        if overlap_sentences < 0 or overlap_sentences >= sentences_per_chunk:
            raise ValueError(
                f"overlap_sentences must be >= 0 and < sentences_per_chunk; "
                f"got overlap_sentences={overlap_sentences}, sentences_per_chunk={sentences_per_chunk}"
            )

        sentences = split_sentences(doc.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        position = 0
        buffer: list[str] = []
        overlap_buffer: list[str] = []

        for sent_idx, sentence in enumerate(sentences):
            buffer.append(sentence)

            if len(buffer) >= sentences_per_chunk or sent_idx == len(sentences) - 1:
                # Flush the buffer into a chunk.
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

                # Prepare overlap for the next chunk.
                overlap_buffer = buffer[-overlap_sentences:] if overlap_sentences > 0 else []
                buffer = list(overlap_buffer)  # Start next buffer with overlap.
                position += 1

        # Recalculate actual overlap_with_prev.
        if overlap_sentences > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_text = chunks[i - 1].text
                curr_text = chunks[i].text

                # Count leading characters of curr_text that match tail of prev_text.
                match_count = 0
                for j in range(min(len(prev_text), len(curr_text))):
                    if prev_text[-(j + 1)] == curr_text[j]:
                        continue
                    else:
                        break
                else:
                    match_count = min(len(prev_text), len(curr_text))

                # Simpler: count the longest prefix of curr that is a suffix of prev.
                match_count = 0
                for length in range(1, min(len(prev_text), len(curr_text)) + 1):
                    if prev_text[-length:] == curr_text[:length]:
                        match_count = length

                chunks[i].overlap_with_prev = match_count

        return chunks
