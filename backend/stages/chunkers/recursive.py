"""Recursive character chunker — split on a priority list of separators."""

from __future__ import annotations

import re

from .base import BaseChunker, Chunk
from backend.stages.parsers.base import ParsedDocument

#: Default separators, tried in order from "most natural boundary" to "last
#: resort". The empty string means "give up and cut mid-word by character".
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def approx_token_count(text: str) -> int:
    """Rough token count: word-ish runs plus standalone punctuation.

    This is deliberately model-agnostic (no tiktoken / model tokenizer) so the
    chunker never secretly depends on which embedder you picked. It is an
    approximation for display, not an exact match to any model's tokenizer.
    """
    return len(re.findall(r"\w+|[^\w\s]", text))


class RecursiveChunker(BaseChunker):
    """Split text by trying a list of separators from coarse to fine.

    What it does (mechanically): it first tries to split the document on the
    biggest natural boundary (blank lines between paragraphs). Any piece still
    larger than ``size`` is split again on the next separator (single newlines,
    then sentence breaks, then spaces, then — only if nothing else works — a
    hard character cut). The resulting small pieces are then greedily packed
    back together into chunks up to ``size`` characters, and each new chunk
    keeps the last ``overlap`` characters of the previous one so context isn't
    lost at the seam.

    Tradeoff vs. the alternatives: a fixed-size chunker is simpler but blindly
    cuts every ``size`` characters, frequently slicing a word or sentence in
    half. This recursive approach respects natural boundaries *when it can* and
    only falls back to a blind cut as a last resort — better-formed chunks at
    the cost of slightly variable chunk sizes and a bit more logic.

    When a learner would prefer it: for general prose and documents with
    paragraph structure (like reports, articles, or this discharge summary),
    where you want chunks that mostly begin and end at sentence/paragraph
    boundaries. It is the sensible default before reaching for sentence- or
    semantic-aware chunking.

    Parameters (all recorded in config.json, none hidden):
        size: target maximum characters per chunk (default 512).
        overlap: characters of the previous chunk repeated at the start of the
            next (default 50). Set 0 for no overlap.
        separators: the priority list to split on (default DEFAULT_SEPARATORS).
    """

    name = "Recursive (character)"
    description = (
        "Splits on a priority list of separators (paragraph, line, sentence, "
        "word) and only cuts mid-word as a last resort, then packs pieces into "
        "chunks of ~size chars with overlap. Good general-purpose default for "
        "prose with paragraph structure; chunk sizes vary a little as a result."
    )

    def chunk(
        self,
        doc: ParsedDocument,
        size: int = 512,
        overlap: int = 50,
        separators: list[str] | None = None,
        **_ignored,
    ) -> list[Chunk]:
        if size <= 0:
            raise ValueError(f"chunk size must be positive, got {size}")
        if overlap < 0 or overlap >= size:
            raise ValueError(
                f"overlap must be >= 0 and < size; got overlap={overlap}, size={size}"
            )
        seps = separators if separators is not None else list(DEFAULT_SEPARATORS)

        pieces = self._recursive_split(doc.text, seps, size)
        merged = self._merge(pieces, size, overlap)

        chunks: list[Chunk] = []
        for position, (text, overlap_chars) in enumerate(merged):
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
        return chunks

    def _recursive_split(self, text: str, separators: list[str], size: int) -> list[str]:
        """Break text into pieces each <= size (where the text allows it).

        Separators are re-attached to the pieces so that concatenating all the
        pieces reproduces the original text — keeping the split fully auditable.
        """
        if len(text) <= size:
            return [text] if text else []

        # Choose the first separator that actually occurs in this text.
        chosen = ""
        remaining: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                chosen = ""
                remaining = []
                break
            if sep in text:
                chosen = sep
                remaining = separators[i + 1 :]
                break

        if chosen == "":
            # No usable separator left: hard-cut by character.
            return [text[i : i + size] for i in range(0, len(text), size)]

        parts = text.split(chosen)
        pieces: list[str] = []
        for idx, part in enumerate(parts):
            # Re-attach the separator to every part except the last.
            piece = part + (chosen if idx < len(parts) - 1 else "")
            if not piece:
                continue
            if len(piece) <= size:
                pieces.append(piece)
            else:
                pieces.extend(self._recursive_split(piece, remaining, size))
        return pieces

    def _merge(self, pieces: list[str], size: int, overlap: int) -> list[tuple[str, int]]:
        """Greedily pack pieces into <=size chunks, carrying an overlap tail.

        Returns (chunk_text, overlap_with_prev) pairs. ``overlap_with_prev`` is
        the *actual* number of leading characters this chunk shares with the
        previous one, so the UI's overlap highlight reflects reality (not just
        the requested overlap value).
        """
        chunks: list[tuple[str, int]] = []
        buffer = ""
        carried = 0  # chars carried over from the previous chunk into buffer

        for piece in pieces:
            if buffer and len(buffer) + len(piece) > size:
                chunks.append((buffer, carried))
                tail = buffer[-overlap:] if overlap > 0 else ""
                buffer = tail + piece
                carried = len(tail)
            else:
                buffer += piece

        if buffer:
            chunks.append((buffer, carried))
        return chunks
