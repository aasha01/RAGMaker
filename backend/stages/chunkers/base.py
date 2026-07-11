"""Chunking stage — interface and data contract.

A chunker splits one `ParsedDocument` into an ordered list of `Chunk`s. Each
chunk carries enough bookkeeping (position, lengths, overlap with the previous
chunk) for the UI to show a learner *exactly* how the text was cut — including
mid-sentence splits and how much consecutive chunks overlap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.stages.parsers.base import ParsedDocument


@dataclass
class Chunk:
    """One piece of the document, plus inspection bookkeeping.

    Attributes:
        chunk_id: Stable id, unique within the recipe (e.g. "chunk_0007").
        text: The chunk's text.
        source: The source document this chunk came from.
        position: 0-based order of this chunk within its source document.
        char_len: len(text) in characters.
        token_len: Approximate token count (how the strategy counts tokens is
            its own business, but it must record the number here).
        overlap_with_prev: Number of characters this chunk shares with the
            previous chunk, or 0 if none. Drives the UI overlap highlighting.
    """

    chunk_id: str
    text: str
    source: str
    position: int
    char_len: int
    token_len: int
    overlap_with_prev: int

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for `stage_outputs/03_chunks.json`."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "position": self.position,
            "char_len": self.char_len,
            "token_len": self.token_len,
            "overlap_with_prev": self.overlap_with_prev,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Rebuild from a persisted `03_chunks.json` entry."""
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source=data["source"],
            position=data["position"],
            char_len=data["char_len"],
            token_len=data["token_len"],
            overlap_with_prev=data["overlap_with_prev"],
        )


class BaseChunker(ABC):
    """Abstract chunker. Every concrete chunker is a swappable Strategy.

    Tunable parameters (size, overlap, ...) are passed to `chunk` as keyword
    arguments so they can be read straight out of `config.json` and so they
    stay visible/overridable in the UI — never baked in as hidden defaults
    (CLAUDE.md Non-Negotiables).
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def chunk(self, doc: ParsedDocument, **params) -> list[Chunk]:
        """Split `doc` into an ordered list of `Chunk`s.

        Every parameter that affects the result must be an explicit keyword in
        `params` and recorded in `config.json`; there must be no hidden knob
        that changes behaviour without appearing in the recipe config.
        """
        raise NotImplementedError
