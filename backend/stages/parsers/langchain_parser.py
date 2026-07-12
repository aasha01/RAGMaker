"""LangChain-based parser — wraps PyPDFLoader and UnstructuredFileLoader."""

from __future__ import annotations

import os

from .base import BaseParser, ParsedDocument


class LangChainParser(BaseParser):
    """Extract text using LangChain's document loaders and lazy unification.

    What it does (mechanically): dispatches on file extension and uses
    LangChain's loaders to read the file. For PDF, uses ``PyPDFLoader``.
    For DOCX/DOC, uses ``UnstructuredFileLoader``. For TXT/MD, reads plain.
    Returns the text plus standard metadata. LangChain handles format detection
    and returns a list of Document objects, which are then concatenated.

    Tradeoff vs. the alternatives: LangChain's loaders add some convenience
    (auto fallback strategies, lighter cleanup) over the manual approach, but
    still give you a sense of what's happening. The llamaindex loaders are
    similar but have a different API and slightly different cleanup strategies.
    Both are heavier than the manual parser but lighter than frameworks that
    hide extraction entirely.

    When a learner would prefer it: when you want good coverage of formats
    (PDF, DOCX, etc.) without writing format-specific code, and you trust
    LangChain's extraction quality. Also good for learning how LangChain
    integrates into a RAG pipeline.

    Per-format loaders are imported lazily inside ``parse`` so that, for
    example, parsing a ``.txt`` file needs no PDF or document packages
    installed beyond langchain itself.
    """

    name = "LangChain (PyPDFLoader/Unstructured)"
    description = (
        "Uses LangChain's document loaders: PyPDFLoader for PDF, "
        "UnstructuredFileLoader for DOCX, plain read for TXT. "
        "Adds light cleanup and format standardization. "
        "Good when you want broader format coverage without manual dispatch code."
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
            f"LangChainParser does not support '{ext}' files. "
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
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as e:
            raise ImportError(
                "Parsing PDF with LangChainParser needs the 'langchain-community' "
                "package (PyPDFLoader lives there, not in core 'langchain') and "
                "the 'pypdf' dependency. "
                "Install with: pip install langchain-community pypdf"
            ) from e

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            raise ValueError(
                f"LangChainParser's PyPDFLoader extracted no pages from {file_path}. "
                f"Verify the PDF is not corrupted or empty."
            )

        page_texts = [doc.page_content for doc in docs]
        text = "\f".join(page_texts)

        return ParsedDocument(
            text=text,
            source=os.path.basename(file_path),
            metadata={
                "format": "pdf",
                "pages": len(page_texts),
                "page_texts": page_texts,
                "char_count": len(text),
                "engine": "langchain_pypdf",
            },
        )

    def _parse_unstructured(self, file_path: str, ext: str) -> ParsedDocument:
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader
        except ImportError as e:
            raise ImportError(
                "Parsing DOCX/DOC with LangChainParser needs the 'langchain-community' "
                "package (UnstructuredFileLoader lives there, not in core 'langchain') "
                "and the 'unstructured' dependency. "
                "Install with: pip install langchain-community unstructured pillow pptx "
                "(and pdf2image for PDFs, if not already present)"
            ) from e

        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()

        if not docs:
            raise ValueError(
                f"LangChainParser's UnstructuredFileLoader extracted no content "
                f"from {file_path}. Verify the file is not corrupted or empty."
            )

        # UnstructuredFileLoader typically returns one doc with all content.
        # Join them if there are multiple (edge case).
        text = "\n\n".join(doc.page_content for doc in docs)

        return ParsedDocument(
            text=text,
            source=os.path.basename(file_path),
            metadata={
                "format": ext.lstrip("."),
                "pages": 1,  # Unstructured doesn't track pages like PDF
                "page_texts": [text],
                "char_count": len(text),
                "engine": "langchain_unstructured",
            },
        )
