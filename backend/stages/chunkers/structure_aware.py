"""Structure-aware chunker — splits on Markdown headers."""

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


class StructureAwareChunker(BaseChunker):
    """Split text on Markdown headers, grouping by header hierarchy.

    What it does (mechanically): recognizes Markdown headers (`#`, `##`, etc.)
    and creates chunk boundaries at headers up to ``header_levels`` deep. For
    example, with ``header_levels=2``, it splits on `#` and `##` headers but
    treats `###` and deeper headers as regular text within their parent section.
    Each chunk includes the header and all text until the next header at that
    level (or shallower).

    Tradeoff vs. the alternatives: ideal for Markdown-formatted documents
    (documentation, wikis, structured reports). It preserves the logical
    document structure perfectly. The limitation is that it only works on
    Markdown; for other formats or plain prose it falls back to treating the
    whole document as one chunk (no splitting).

    When a learner would prefer it: for documentation, READMEs, and any
    Markdown-formatted source where the header hierarchy carries semantic
    meaning. It guarantees chunks align with sections, so a retrieval system
    can return "the setup section" intact, not scattered.

    Parameters (all recorded in config.json, none hidden):
        header_levels: maximum header depth to split on (default 2, range
            [1, 6]). `header_levels=1` splits only on `#`; `header_levels=2`
            splits on `#` and `##`; etc.
    """

    name = "Structure-aware (Markdown headers)"
    description = (
        "Splits Markdown text on headers up to a specified depth, preserving "
        "the document's logical structure. Each chunk contains a header and all "
        "text until the next header at the same or shallower level. Works only "
        "on Markdown; for plain text or other formats, treats the whole "
        "document as one chunk. Ideal for documentation."
    )

    def chunk(
        self,
        doc: ParsedDocument,
        header_levels: int = 2,
        **_ignored,
    ) -> list[Chunk]:
        if not (1 <= header_levels <= 6):
            raise ValueError(f"header_levels must be in [1, 6], got {header_levels}")

        # Regex to find Markdown headers: one or more `#` at start of line.
        header_pattern = r"^(#{1,6})\s+(.+)$"
        lines = doc.text.split("\n")

        # Identify header lines and their depths.
        headers = []
        for line_idx, line in enumerate(lines):
            match = re.match(header_pattern, line)
            if match:
                depth = len(match.group(1))
                if depth <= header_levels:
                    headers.append((line_idx, depth, match.group(2)))

        if not headers:
            # No headers found; treat entire document as one chunk.
            return [
                Chunk(
                    chunk_id="chunk_0000",
                    text=doc.text,
                    source=doc.source,
                    position=0,
                    char_len=len(doc.text),
                    token_len=approx_token_count(doc.text),
                    overlap_with_prev=0,
                )
            ]

        # Build chunks: each chunk starts at a header and includes all text
        # until the next header at the same or shallower depth.
        chunks: list[Chunk] = []
        position = 0

        for h_idx, (header_line_idx, header_depth, header_text) in enumerate(headers):
            # Find the end of this chunk: the next header at same or shallower depth,
            # or end of document.
            end_line_idx = len(lines)
            for next_h_idx in range(h_idx + 1, len(headers)):
                next_line_idx, next_depth, _ = headers[next_h_idx]
                if next_depth <= header_depth:
                    end_line_idx = next_line_idx
                    break

            # Extract chunk text: from header line to end line (exclusive).
            chunk_lines = lines[header_line_idx:end_line_idx]
            text = "\n".join(chunk_lines).strip()

            if text:
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

        # Recalculate actual overlap_with_prev.
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
