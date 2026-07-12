"""PDF parsing → section detection → section-aware chunking.

Pure functions with no server or network dependencies (DESIGN.md §6):
``chunk_pdf(pdf_path) -> list[Chunk]`` plus ``pdf_page_count``. Fully
unit-testable against a tiny fixture PDF.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import fitz  # PyMuPDF

from app.config import get_settings
from app.errors import InvalidPDF
from app.models.schemas import Chunk

# A line is a heading if its font is clearly larger than body text AND it is
# either bold or looks like a numbered heading ("2 Background", "3.1 Foo").
_HEADING_SIZE_RATIO = 1.15
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\s+\w")
_REFERENCES_RE = re.compile(r"^(references|bibliography)\b", re.IGNORECASE)
# PyMuPDF span flag bit 4 (value 16) marks bold/synthetic-bold text.
_BOLD_FLAG = 1 << 4
# A line ending in "<letter>-" is a word wrapped across the line break.
_HYPHEN_BREAK = re.compile(r"[A-Za-z]-$")


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines into a paragraph, repairing end-of-line hyphenation.

    A line ending in ``<letter>-`` followed by a line starting with a lowercase
    letter is one word split across the break (``engi-`` + ``neering`` →
    ``engineering``): merge with neither hyphen nor space. Everything else is
    joined with a space, so mid-line compounds like ``state-of-the-art`` (whose
    hyphens are not at a break) keep their hyphen.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif _HYPHEN_BREAK.search(out) and line[:1].islower():
            out = out[:-1] + line  # drop the trailing hyphen, join with no space
        else:
            out = f"{out} {line}"
    return out


class _Line:
    __slots__ = ("text", "size", "bold", "page")

    def __init__(self, text: str, size: float, bold: bool, page: int) -> None:
        self.text = text
        self.size = size
        self.bold = bold
        self.page = page


def pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


_ARXIV_WATERMARK = re.compile(r"^\s*arxiv:", re.IGNORECASE)
_BAD_TITLE_PREFIXES = ("microsoft word", "untitled")


def _looks_like_title(text: str) -> bool:
    text = text.strip()
    if len(text) < 6:
        return False
    low = text.lower()
    if low.startswith(_BAD_TITLE_PREFIXES):
        return False
    if low.endswith((".doc", ".docx", ".pdf")):
        return False
    return True


def _first_page_text_lines(page: fitz.Page) -> list[tuple[str, float]]:
    """(text, font_size) for page-1 text lines, skipping the arXiv watermark."""
    lines: list[tuple[str, float]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:  # skip images
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text or _ARXIV_WATERMARK.match(text):
                continue
            size = max((s.get("size", 0.0) for s in spans), default=0.0)
            lines.append((text, size))
    return lines


def extract_title(pdf_path: Path) -> str | None:
    """Best-effort paper title for uploads with no supplied metadata.

    Prefers the PDF's embedded title; falls back to the largest-font text at
    the top of page 1 (skipping the arXiv margin watermark). Returns None when
    nothing usable is found, so the caller leaves the title unset rather than
    storing junk.
    """
    with fitz.open(pdf_path) as doc:
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        if _looks_like_title(meta_title):
            return meta_title
        if doc.page_count == 0:
            return None
        lines = _first_page_text_lines(doc[0])
    if not lines:
        return None
    max_size = max(size for _, size in lines)
    parts: list[str] = []
    for text, size in lines:
        if size >= max_size * 0.97:
            parts.append(text)
        elif parts:  # the title block ends once font drops below the top size
            break
    title = " ".join(parts).strip()
    return title if _looks_like_title(title) else None


def _extract_lines(doc: fitz.Document) -> list[_Line]:
    lines: list[_Line] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # skip images
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                # Line font size = max span size; bold if any span is bold.
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                bold = any(s.get("flags", 0) & _BOLD_FLAG for s in spans)
                lines.append(_Line(text, size, bold, page_index + 1))
    return lines


def _is_heading(line: _Line, body_median: float) -> bool:
    if len(line.text) > 120:  # real headings are short
        return False
    larger = line.size > _HEADING_SIZE_RATIO * body_median
    if _REFERENCES_RE.match(line.text):
        return True
    if not larger:
        return False
    return line.bold or bool(_NUMBERED_HEADING.match(line.text))


def _normalize_section(title: str) -> str:
    if _REFERENCES_RE.match(title):
        return "References"
    return title.strip()


def _pack_section(
    section: str,
    paragraphs: list[tuple[str, int]],
    start_index: int,
    chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], int]:
    """Greedily pack paragraphs into ~chunk_size chunks with char overlap."""
    chunks: list[Chunk] = []
    idx = start_index
    buf = ""
    buf_page_start: int | None = None
    buf_page_end: int | None = None

    def flush() -> str:
        nonlocal idx, buf, buf_page_start, buf_page_end
        if not buf.strip():
            return ""
        chunks.append(
            Chunk(
                text=buf.strip(),
                section=section,
                page_start=buf_page_start or 1,
                page_end=buf_page_end or buf_page_start or 1,
                chunk_index=idx,
            )
        )
        idx += 1
        tail = buf[-overlap:] if overlap > 0 else ""
        return tail

    for para, page in paragraphs:
        if buf_page_start is None:
            buf_page_start = page
        buf_page_end = page
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= chunk_size or not buf:
            buf = candidate
        else:
            tail = flush()
            buf = f"{tail}\n\n{para}" if tail else para
            buf_page_start = page
            buf_page_end = page
        # A single oversized paragraph: emit as its own chunk.
        while len(buf) > chunk_size * 1.5:
            head, buf = buf[:chunk_size], buf[chunk_size - overlap :]
            chunks.append(
                Chunk(
                    text=head.strip(),
                    section=section,
                    page_start=buf_page_start or page,
                    page_end=page,
                    chunk_index=idx,
                )
            )
            idx += 1
            buf_page_start = page
    flush()
    return chunks, idx


def chunk_pdf(
    pdf_path: Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Parse a PDF and return section-aware chunks.

    Raises ``InvalidPDF`` if the document has no extractable text layer
    (scanned/image-only PDFs are out of scope, DESIGN.md §9).
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    with fitz.open(pdf_path) as doc:
        lines = _extract_lines(doc)

    if not lines:
        raise InvalidPDF("no extractable text (scanned/image-only PDF?)")

    body_median = statistics.median(ln.size for ln in lines)

    # Walk lines, splitting into sections at detected headings. Accumulate the
    # body text of each section together with the page each paragraph sits on.
    sections: list[tuple[str, list[tuple[str, int]]]] = []
    current_title = "Body"
    current_text: list[tuple[str, int]] = []  # (line_text, page)

    def close_section() -> None:
        if not current_text:
            return
        joined_by_page: list[tuple[str, int]] = []
        # Rebuild paragraph-ish blocks per page from accumulated lines.
        buf: list[str] = []
        page_of_buf = current_text[0][1]
        for txt, pg in current_text:
            if pg != page_of_buf and buf:
                joined_by_page.append((_join_lines(buf), page_of_buf))
                buf = []
                page_of_buf = pg
            buf.append(txt)
        if buf:
            joined_by_page.append((_join_lines(buf), page_of_buf))
        sections.append((current_title, joined_by_page))

    for line in lines:
        if _is_heading(line, body_median):
            close_section()
            current_title = _normalize_section(line.text)
            current_text = []
        else:
            current_text.append((line.text, line.page))
    close_section()

    if not sections:
        sections = [("Body", [(ln.text, ln.page) for ln in lines])]

    chunks: list[Chunk] = []
    next_index = 0
    for title, paragraphs in sections:
        section_chunks, next_index = _pack_section(
            title, paragraphs, next_index, chunk_size, chunk_overlap
        )
        chunks.extend(section_chunks)
    return chunks
