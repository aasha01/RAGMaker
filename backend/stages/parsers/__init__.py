"""Parser strategy registry.

Maps the string key recorded in `config.json` (and shown in the UI dropdown)
to the concrete `BaseParser` subclass that implements it.
"""

from __future__ import annotations

from .base import BaseParser, ParsedDocument
from .manual import ManualParser
from .langchain_parser import LangChainParser
from .llamaindex_parser import LlamaIndexParser

REGISTRY: dict[str, type[BaseParser]] = {
    "manual": ManualParser,
    "langchain": LangChainParser,
    "llamaindex": LlamaIndexParser,
}

__all__ = ["BaseParser", "ParsedDocument", "REGISTRY"]
