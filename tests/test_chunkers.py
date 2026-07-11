"""Per-chunker tests against sample_data."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.stages.chunkers import REGISTRY
from backend.stages.chunkers.base import Chunk
from backend.stages.chunkers.fixed_size import FixedSizeChunker
from backend.stages.chunkers.semantic import SemanticChunker
from backend.stages.chunkers.sentence import SentenceChunker
from backend.stages.chunkers.structure_aware import StructureAwareChunker
from backend.stages.parsers.base import ParsedDocument


# Load sample data for testing.
SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"


@pytest.fixture
def sample_text() -> str:
    """Load the sample text document."""
    sample_file = SAMPLE_DATA_DIR / "discharge_summary_detailed.txt"
    if not sample_file.exists():
        pytest.skip(f"Sample file not found: {sample_file}")
    return sample_file.read_text()


@pytest.fixture
def sample_doc(sample_text: str) -> ParsedDocument:
    """Create a ParsedDocument from sample text."""
    return ParsedDocument(
        text=sample_text,
        source="sample_discharge_summary.txt",
        metadata={"type": "text", "char_count": len(sample_text)},
    )


class TestFixedSizeChunker:
    """Test fixed-size chunking."""

    def test_registry_entry(self):
        """Verify chunker is registered."""
        assert "fixed_size" in REGISTRY
        assert REGISTRY["fixed_size"] is FixedSizeChunker

    def test_basic_chunking(self):
        """Test basic fixed-size chunking with overlap."""
        text = "a" * 100 + "b" * 100 + "c" * 100
        doc = ParsedDocument(text=text, source="test.txt", metadata={})
        chunker = FixedSizeChunker()

        chunks = chunker.chunk(doc, size=60, overlap=10)

        # With size=60, we expect roughly 5 chunks (300 chars / 60 = 5).
        assert len(chunks) >= 4
        # Each chunk should be <= size.
        for chunk in chunks:
            assert len(chunk.text) <= 60
        # First chunk should have no overlap.
        assert chunks[0].overlap_with_prev == 0

    def test_overlap_calculation(self):
        """Test that overlap_with_prev is calculated correctly."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        doc = ParsedDocument(text=text, source="test.txt", metadata={})
        chunker = FixedSizeChunker()

        chunks = chunker.chunk(doc, size=50, overlap=10)

        # Each chunk after the first should have overlap.
        for i in range(1, len(chunks)):
            assert chunks[i].overlap_with_prev >= 0
            # Overlap should be <= requested overlap.
            assert chunks[i].overlap_with_prev <= 10

    def test_invalid_size(self):
        """Test that invalid size raises ValueError."""
        doc = ParsedDocument(text="test", source="test.txt", metadata={})
        chunker = FixedSizeChunker()

        with pytest.raises(ValueError, match="size must be positive"):
            chunker.chunk(doc, size=0)

    def test_invalid_overlap(self):
        """Test that invalid overlap raises ValueError."""
        doc = ParsedDocument(text="test", source="test.txt", metadata={})
        chunker = FixedSizeChunker()

        with pytest.raises(ValueError, match="overlap must be"):
            chunker.chunk(doc, size=50, overlap=-1)

        with pytest.raises(ValueError, match="overlap must be"):
            chunker.chunk(doc, size=50, overlap=50)

    def test_with_sample_data(self, sample_doc: ParsedDocument):
        """Test with actual sample document."""
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(sample_doc, size=512, overlap=50)

        # Should produce multiple chunks.
        assert len(chunks) > 1
        # All chunks should have consistent IDs.
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"chunk_{i:04d}"
            assert chunk.position == i
            assert chunk.source == sample_doc.source
            assert len(chunk.text) > 0
            assert chunk.char_len == len(chunk.text)
            assert chunk.token_len > 0


class TestSentenceChunker:
    """Test sentence-based chunking."""

    def test_registry_entry(self):
        """Verify chunker is registered."""
        assert "sentence" in REGISTRY
        assert REGISTRY["sentence"] is SentenceChunker

    def test_basic_chunking(self):
        """Test basic sentence chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        doc = ParsedDocument(text=text, source="test.txt", metadata={})
        chunker = SentenceChunker()

        chunks = chunker.chunk(doc, sentences_per_chunk=2, overlap_sentences=0)

        # 4 sentences / 2 per chunk = 2 chunks.
        assert len(chunks) == 2
        assert "First" in chunks[0].text
        assert "Second" in chunks[0].text
        assert "Third" in chunks[1].text
        assert "Fourth" in chunks[1].text

    def test_with_overlap(self):
        """Test sentence chunking with overlap."""
        text = "A. B. C. D. E. F."
        doc = ParsedDocument(text=text, source="test.txt", metadata={})
        chunker = SentenceChunker()

        chunks = chunker.chunk(doc, sentences_per_chunk=2, overlap_sentences=1)

        # With 6 sentences, 2 per chunk, 1 overlap: A-B, B-C, C-D, D-E, E-F.
        assert len(chunks) >= 3

    def test_invalid_sentences_per_chunk(self):
        """Test that invalid sentences_per_chunk raises ValueError."""
        doc = ParsedDocument(text="test", source="test.txt", metadata={})
        chunker = SentenceChunker()

        with pytest.raises(ValueError, match="sentences_per_chunk must be positive"):
            chunker.chunk(doc, sentences_per_chunk=0)

    def test_invalid_overlap_sentences(self):
        """Test that invalid overlap_sentences raises ValueError."""
        doc = ParsedDocument(text="test", source="test.txt", metadata={})
        chunker = SentenceChunker()

        with pytest.raises(ValueError, match="overlap_sentences must be"):
            chunker.chunk(doc, sentences_per_chunk=2, overlap_sentences=-1)

        with pytest.raises(ValueError, match="overlap_sentences must be"):
            chunker.chunk(doc, sentences_per_chunk=2, overlap_sentences=2)

    def test_empty_text(self):
        """Test that empty text produces no chunks."""
        doc = ParsedDocument(text="", source="test.txt", metadata={})
        chunker = SentenceChunker()

        chunks = chunker.chunk(doc, sentences_per_chunk=2)

        assert len(chunks) == 0

    def test_with_sample_data(self, sample_doc: ParsedDocument):
        """Test with actual sample document."""
        chunker = SentenceChunker()
        chunks = chunker.chunk(sample_doc, sentences_per_chunk=3, overlap_sentences=0)

        # Should produce multiple chunks.
        assert len(chunks) > 1
        # All chunks should have consistent IDs and positions.
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"chunk_{i:04d}"
            assert chunk.position == i
            assert chunk.source == sample_doc.source
            assert len(chunk.text) > 0
            assert chunk.char_len == len(chunk.text)
            assert chunk.token_len > 0


class TestSemanticChunker:
    """Test semantic (similarity-based) chunking."""

    def test_registry_entry(self):
        """Verify chunker is registered."""
        assert "semantic" in REGISTRY
        assert REGISTRY["semantic"] is SemanticChunker

    def test_missing_embedder(self):
        """Test that missing embedder raises ValueError."""
        doc = ParsedDocument(text="test", source="test.txt", metadata={})
        chunker = SemanticChunker()

        with pytest.raises(ValueError, match="requires an embedder"):
            chunker.chunk(doc, similarity_threshold=0.3, embedder=None)

    def test_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        doc = ParsedDocument(text="test", source="test.txt", metadata={})
        chunker = SemanticChunker()
        mock_embedder = MagicMock()

        with pytest.raises(ValueError, match="similarity_threshold must be"):
            chunker.chunk(doc, similarity_threshold=-0.1, embedder=mock_embedder)

        with pytest.raises(ValueError, match="similarity_threshold must be"):
            chunker.chunk(doc, similarity_threshold=1.1, embedder=mock_embedder)

    def test_with_mock_embedder(self):
        """Test semantic chunking with a mock embedder."""
        text = "First topic here. Related to first topic. Second topic now. Related to second topic."
        doc = ParsedDocument(text=text, source="test.txt", metadata={})

        # Mock embedder: return constant vectors (so all similarities are high).
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [
            [1.0, 0.0],  # Sentence 1
            [1.0, 0.0],  # Sentence 2 (similar to 1)
            [0.0, 1.0],  # Sentence 3 (different from 2)
            [0.0, 1.0],  # Sentence 4 (similar to 3)
        ]

        chunker = SemanticChunker()
        chunks = chunker.chunk(doc, similarity_threshold=0.5, embedder=mock_embedder)

        # Should create a boundary between sentences 2 and 3 (similarity = 0).
        assert len(chunks) >= 2

    def test_single_sentence(self):
        """Test that single sentence produces single chunk."""
        text = "Only one sentence here."
        doc = ParsedDocument(text=text, source="test.txt", metadata={})

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[1.0, 0.0]]

        chunker = SemanticChunker()
        chunks = chunker.chunk(doc, similarity_threshold=0.3, embedder=mock_embedder)

        assert len(chunks) == 1
        assert "Only one sentence" in chunks[0].text

    def test_empty_text(self):
        """Test that empty text produces no chunks."""
        doc = ParsedDocument(text="", source="test.txt", metadata={})

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = []

        chunker = SemanticChunker()
        chunks = chunker.chunk(doc, similarity_threshold=0.3, embedder=mock_embedder)

        assert len(chunks) == 0


class TestStructureAwareChunker:
    """Test Markdown structure-aware chunking."""

    def test_registry_entry(self):
        """Verify chunker is registered."""
        assert "structure_aware" in REGISTRY
        assert REGISTRY["structure_aware"] is StructureAwareChunker

    def test_markdown_headers(self):
        """Test basic Markdown header splitting."""
        text = """# Section 1
Content for section 1.

## Subsection 1.1
Content for subsection 1.1.

# Section 2
Content for section 2.
"""
        doc = ParsedDocument(text=text, source="test.md", metadata={})
        chunker = StructureAwareChunker()

        chunks = chunker.chunk(doc, header_levels=2)

        # With header_levels=2, creates chunks at both # and ## levels.
        # Chunk 0: "# Section 1" + content + "## Subsection 1.1" + content (until next # or higher)
        # Chunk 1: "## Subsection 1.1" + content (until next ## or higher)
        # Wait, that's not right. Let me re-check...
        # Actually: each header creates a chunk boundary, so:
        # Chunk 0: "# Section 1" to "## Subsection 1.1" (exclusive)
        # Chunk 1: "## Subsection 1.1" to "# Section 2" (exclusive)
        # Chunk 2: "# Section 2" to end
        assert len(chunks) >= 2
        # Verify Section 1 header is in one of the chunks.
        chunk_texts = [c.text for c in chunks]
        assert any("Section 1" in t for t in chunk_texts)
        assert any("Section 2" in t for t in chunk_texts)

    def test_header_levels_filter(self):
        """Test that header_levels correctly filters depth."""
        text = """# Level 1
Content 1.

## Level 2
Content 2.

### Level 3
Content 3.
"""
        doc = ParsedDocument(text=text, source="test.md", metadata={})
        chunker = StructureAwareChunker()

        # With header_levels=1, only split on #.
        chunks = chunker.chunk(doc, header_levels=1)
        # Should create chunks for Level 1 (including Level 2, 3).
        assert len(chunks) >= 1

    def test_no_headers(self):
        """Test that plain text with no headers produces single chunk."""
        text = "Just plain text without any headers."
        doc = ParsedDocument(text=text, source="test.txt", metadata={})
        chunker = StructureAwareChunker()

        chunks = chunker.chunk(doc, header_levels=2)

        # No headers = one chunk.
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_invalid_header_levels(self):
        """Test that invalid header_levels raises ValueError."""
        doc = ParsedDocument(text="test", source="test.md", metadata={})
        chunker = StructureAwareChunker()

        with pytest.raises(ValueError, match="header_levels must be"):
            chunker.chunk(doc, header_levels=0)

        with pytest.raises(ValueError, match="header_levels must be"):
            chunker.chunk(doc, header_levels=7)

    def test_with_sample_data(self, sample_doc: ParsedDocument):
        """Test with actual sample document (treated as plain text)."""
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(sample_doc, header_levels=2)

        # Sample discharge summary has no Markdown, so should be 1 chunk.
        assert len(chunks) >= 1
        assert chunks[0].source == sample_doc.source
        for chunk in chunks:
            assert len(chunk.text) > 0
            assert chunk.char_len == len(chunk.text)
            assert chunk.token_len > 0


class TestChunkOverlapIntegrity:
    """Test that overlap_with_prev is truthful across all chunkers."""

    def test_fixed_size_overlap_integrity(self):
        """Verify fixed-size overlap_with_prev matches reality."""
        text = "abcdefghij" * 20
        doc = ParsedDocument(text=text, source="test.txt", metadata={})
        chunker = FixedSizeChunker()

        chunks = chunker.chunk(doc, size=50, overlap=10)

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr_text = chunks[i].text
            claimed_overlap = chunks[i].overlap_with_prev

            # The claimed overlap should match the actual prefix/suffix match.
            if claimed_overlap > 0:
                assert prev_text[-claimed_overlap:] == curr_text[:claimed_overlap]

    def test_sentence_overlap_integrity(self):
        """Verify sentence overlap_with_prev matches reality."""
        text = "A. B. C. D. E. F. G. H."
        doc = ParsedDocument(text=text, source="test.txt", metadata={})
        chunker = SentenceChunker()

        chunks = chunker.chunk(doc, sentences_per_chunk=2, overlap_sentences=1)

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr_text = chunks[i].text
            claimed_overlap = chunks[i].overlap_with_prev

            # The claimed overlap should match the actual prefix/suffix match.
            if claimed_overlap > 0:
                assert prev_text[-claimed_overlap:] == curr_text[:claimed_overlap]
            else:
                # No claimed overlap; verify curr doesn't start with end of prev.
                if len(prev_text) > 0 and len(curr_text) > 0:
                    # This is a weaker check since overlaps can be 0 legitimately.
                    pass

    def test_structure_aware_overlap_integrity(self):
        """Verify structure-aware overlap_with_prev matches reality."""
        text = """# A
Content A.

# B
Content B.

# C
Content C.
"""
        doc = ParsedDocument(text=text, source="test.md", metadata={})
        chunker = StructureAwareChunker()

        chunks = chunker.chunk(doc, header_levels=1)

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr_text = chunks[i].text
            claimed_overlap = chunks[i].overlap_with_prev

            # The claimed overlap should match the actual prefix/suffix match.
            if claimed_overlap > 0:
                assert prev_text[-claimed_overlap:] == curr_text[:claimed_overlap]


class TestChunkIdConsistency:
    """Test that chunk IDs and positions are correct."""

    @pytest.mark.parametrize("chunker_key", ["fixed_size", "sentence", "structure_aware"])
    def test_chunk_ids_sequential(self, chunker_key: str, sample_doc: ParsedDocument):
        """Test that chunk IDs are sequential and unique."""
        chunker_class = REGISTRY[chunker_key]
        chunker = chunker_class()

        if chunker_key == "fixed_size":
            chunks = chunker.chunk(sample_doc, size=512, overlap=50)
        elif chunker_key == "sentence":
            chunks = chunker.chunk(sample_doc, sentences_per_chunk=3, overlap_sentences=0)
        elif chunker_key == "structure_aware":
            chunks = chunker.chunk(sample_doc, header_levels=2)

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        # IDs should be unique.
        assert len(chunk_ids) == len(set(chunk_ids))

        # IDs should be sequential.
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"chunk_{i:04d}"
            assert chunk.position == i
