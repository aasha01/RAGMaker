"""LlamaIndex-based parser — wraps SimpleDirectoryReader and PDFReader."""

from __future__ import annotations

import os
import tempfile

from .base import BaseParser, ParsedDocument


class LlamaIndexParser(BaseParser):
    """Extract text using LlamaIndex's document readers and lazy unification.

    What it does (mechanically): dispatches on file extension and uses
    LlamaIndex's readers to load the file. For PDF, uses ``PDFReader``.
    For other formats (DOCX, TXT), uses ``SimpleDirectoryReader`` which
    auto-detects the format. Returns the text plus standard metadata.
    LlamaIndex readers normalize the output into Document objects, which are
    then concatenated.

    Tradeoff vs. the alternatives: LlamaIndex's readers have a different API
    and cleanup philosophy than LangChain. Both are more opinionated than the
    manual parser but offer broader format support without writing code for
    each format. LlamaIndex tends to be more focused on structured data and
    nested documents, so it may do more aggressive cleanup.

    When a learner would prefer it: when you want LlamaIndex's ecosystem
    integration later in the pipeline (chunking, embedding, retrieval all via
    LlamaIndex), or when you prefer LlamaIndex's extraction philosophy over
    LangChain's. Good for learning how LlamaIndex integrates into a RAG stack.

    Per-format readers are imported lazily inside ``parse`` so that the tool
    runs without optional LlamaIndex extras.
    """

    name = "LlamaIndex (PDFReader/SimpleDirectoryReader)"
    description = (
        "Uses LlamaIndex's document readers: PDFReader for PDF, "
        "SimpleDirectoryReader for DOCX/TXT. "
        "Offers tight integration with LlamaIndex ecosystem. "
        "Good when you plan to use other LlamaIndex components downstream."
    )

    SUPPORTED = (".txt", ".md", ".pdf", ".docx", ".doc")

    def parse(self, file_path: str) -> ParsedDocument:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file to parse: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".txt", ".md"):
            return self._parse_text(file_path, ext)
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        if ext in (".docx", ".doc"):
            return self._parse_unstructured(file_path, ext)

        raise ValueError(
            f"LlamaIndexParser does not support '{ext}' files. "
            f"Supported: {', '.join(self.SUPPORTED)}. "
            f"(No silent fallback — pick a parser that handles this format.)"
        )

    def _parse_text(self, file_path: str, ext: str) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return ParsedDocument(
            text=text,
            source=os.path.basename(file_path),
            metadata={
                "format": ext.lstrip("."),
                "pages": 1,
                "page_texts": [text],
                "char_count": len(text),
                "engine": "plain",
            },
        )

    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        try:
            from llama_index.readers.file import PDFReader
        except ImportError as e:
            raise ImportError(
                "Parsing PDF with LlamaIndexParser needs the 'llama-index-readers-file' "
                "package (PDFReader lives there, not in core 'llama-index') and the "
                "'pypdf' dependency. "
                "Install with: pip install llama-index-readers-file pypdf"
            ) from e

        from pathlib import Path

        reader = PDFReader()
        docs = reader.load_data(Path(file_path))

        if not docs:
            raise ValueError(
                f"LlamaIndexParser's PDFReader extracted no pages from {file_path}. "
                f"Verify the PDF is not corrupted or empty."
            )

        # LlamaIndex documents have .text attribute; extract text from each.
        page_texts = [doc.text for doc in docs]
        text = "\f".join(page_texts)

        return ParsedDocument(
            text=text,
            source=os.path.basename(file_path),
            metadata={
                "format": "pdf",
                "pages": len(page_texts),
                "page_texts": page_texts,
                "char_count": len(text),
                "engine": "llamaindex_pdf",
            },
        )

    def _parse_unstructured(self, file_path: str, ext: str) -> ParsedDocument:
        try:
            from llama_index.core import SimpleDirectoryReader
        except ImportError as e:
            raise ImportError(
                "Parsing DOCX/TXT with LlamaIndexParser needs the 'llama-index' package. "
                "Install with: pip install llama-index"
            ) from e

        # SimpleDirectoryReader expects a directory, so we create a temp one
        # and copy the file there.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the file into the temp directory so SimpleDirectoryReader can find it.
            import shutil
            temp_path = os.path.join(tmpdir, os.path.basename(file_path))
            shutil.copy2(file_path, temp_path)

            reader = SimpleDirectoryReader(tmpdir)
            docs = reader.load_data()

        if not docs:
            raise ValueError(
                f"LlamaIndexParser's SimpleDirectoryReader extracted no content "
                f"from {file_path}. Verify the file is not corrupted or empty."
            )

        # Join all documents (usually one, but handle multiple for safety).
        text = "\n\n".join(doc.text for doc in docs)

        return ParsedDocument(
            text=text,
            source=os.path.basename(file_path),
            metadata={
                "format": ext.lstrip("."),
                "pages": 1,  # SimpleDirectoryReader doesn't track pages like PDF
                "page_texts": [text],
                "char_count": len(text),
                "engine": "llamaindex_directory_reader",
            },
        )
