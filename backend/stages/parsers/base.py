"""Parsing stage — interface and data contract.

A parser turns a single source file on disk into a `ParsedDocument`: the
extracted text plus whatever structural metadata the format made available
(page count, per-page text, ...). It performs *no* cleanup that would hide
what the raw extraction actually produced — the whole point of this stage in a
teaching tool is to let a learner see exactly what text the downstream stages
receive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """The output of the parsing stage.

    Attributes:
        text: The full extracted text of the document.
        source: The original file path / name this text came from.
        metadata: Format-specific extras, e.g. {"pages": 2, "page_texts": [...]}.
            Kept as an open dict so each parser can record what its format
            exposes without changing the interface.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for `stage_outputs/02_parsed.json`."""
        return {"text": self.text, "source": self.source, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "ParsedDocument":
        """Rebuild from a persisted `02_parsed.json` payload."""
        return cls(
            text=data["text"],
            source=data["source"],
            metadata=data.get("metadata", {}),
        )


class BaseParser(ABC):
    """Abstract parser. Every concrete parser is a swappable Strategy.

    Concrete subclasses set `name`/`description` (the description is shown
    verbatim in the UI next to the option — see CLAUDE.md "Documentation-as-
    Code") and implement `parse`.
    """

    #: Short human key/title for the UI. Set by each concrete subclass.
    name: str = ""
    #: User-facing explanation (what it does / tradeoff / when to prefer it).
    description: str = ""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Read `file_path` and return the extracted text + metadata.

        Must raise a clear, learner-readable error on failure rather than
        returning partial/empty results or silently falling back to another
        technique (CLAUDE.md Non-Negotiables).
        """
        raise NotImplementedError
