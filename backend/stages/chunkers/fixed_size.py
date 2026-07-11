"""Fixed-size character chunker — simple uniform chunk boundaries."""

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


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size chunks of exactly ``size`` characters.

    What it does (mechanically): divides the document into ``size``-character
    pieces. If ``overlap > 0``, the last ``overlap`` characters of each chunk
    are repeated at the start of the next one. Chunks are produced in order.

    Tradeoff vs. the alternatives: the simplest chunker. It's predictable
    (uniform chunk sizes, no variability) but blind to natural boundaries —
    it will split sentences and words mid-way without hesitation. For
    structured, uniform content (logs, code, tables) the simplicity is a win;
    for prose it's usually worse than recursive chunking.

    When a learner would prefer it: when you want predictable, identical chunk
    sizes and don't care about sentence/paragraph boundaries. Useful for
    debugging (easy to calculate offsets) and for data types where tokens are
    the unit of meaning (code, structured logs).

    Parameters (all recorded in config.json, none hidden):
        size: target chunk size in characters (default 512).
        overlap: characters of the previous chunk repeated at the start of the
            next (default 50). Set 0 for no overlap.
    """

    name = "Fixed-size (character)"
    description = (
        "Divides text into uniform chunks of exactly ``size`` characters, with "
        "optional overlap. Simplest and most predictable, but will split "
        "sentences and words without hesitation. Good for debugging and uniform "
        "data; use recursive chunking for prose."
    )

    def chunk(
        self,
        doc: ParsedDocument,
        size: int = 512,
        overlap: int = 50,
        **_ignored,
    ) -> list[Chunk]:
        if size <= 0:
            raise ValueError(f"chunk size must be positive, got {size}")
        if overlap < 0 or overlap >= size:
            raise ValueError(
                f"overlap must be >= 0 and < size; got overlap={overlap}, size={size}"
            )

        chunks: list[Chunk] = []
        position = 0

        for chunk_idx in range(0, len(doc.text), size):
            start = chunk_idx
            end = min(chunk_idx + size, len(doc.text))
            text = doc.text[start:end]

            # Calculate overlap with previous chunk (actual shared chars).
            overlap_chars = 0
            if position > 0 and overlap > 0:
                # The previous chunk ended at (start), and carried overlap chars forward.
                # The actual overlap is the min of requested overlap and what's available.
                overlap_chars = min(overlap, start)

            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{position:04d}",
                    text=text,
                    source=doc.source,
                    position=position,
                    char_len=len(text),
                    token_len=approx_token_count(text),
                    overlap_with_prev=overlap_chars,
                )
            )
            position += 1

        # Recalculate overlaps to match actual carried text.
        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                # Previous chunk: text[max(0, len(text)-overlap):]
                prev_text = chunks[i - 1].text
                tail = prev_text[-overlap:] if len(prev_text) >= overlap else prev_text
                curr_text = chunks[i].text

                # Count how many leading chars of curr match tail.
                match_count = 0
                for j, char in enumerate(tail):
                    if j < len(curr_text) and curr_text[j] == char:
                        match_count += 1
                    else:
                        break
                chunks[i].overlap_with_prev = match_count

        return chunks
