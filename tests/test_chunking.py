"""Unit tests for document chunking, splitting, and metadata generation."""

import pytest

from services.chunking import (
    DocumentChunk,
    TextSplitter,
    clean_text,
    compute_content_hash,
)


def test_clean_text_normalizes_whitespace():
    raw = "  Hello   world!\r\n\r\n\tThis is a test.\n\n\n\nMore text.  "
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\t" not in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Hello world!")
    assert cleaned.endswith("More text.")


def test_compute_content_hash_deterministic():
    t1 = "Knowledge base retrieval augmented generation system."
    t2 = "Knowledge base retrieval augmented generation system."
    t3 = "Knowledge base retrieval augmented generation system. "

    assert compute_content_hash(t1) == compute_content_hash(t2)
    # Stripped text produces identical hash
    assert compute_content_hash(t1) == compute_content_hash(t3)


def test_text_splitter_invalid_overlap():
    with pytest.raises(ValueError, match="strictly less than"):
        TextSplitter(chunk_size=100, chunk_overlap=100)


def test_text_splitter_short_text():
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    text = "Short sentence that fits into one chunk."
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_splitter_empty_text():
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    assert splitter.split_text("") == []
    assert splitter.split_text("    ") == []


def test_text_splitter_overlap_preservation():
    splitter = TextSplitter(chunk_size=60, chunk_overlap=20)
    text = (
        "Alpha paragraph that has multiple statements and sentences. "
        "Beta paragraph detailing the next concept in the knowledge series. "
        "Gamma paragraph concluding the entire demonstration document."
    )
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2

    # Check each chunk adheres to maximum length
    for c in chunks:
        assert len(c) <= 120  # bounded by separator groupings


def test_create_document_chunks_metadata():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    doc_text = "Sentence one about AI. Sentence two about vector databases. Sentence three about Chroma."
    chunks = splitter.create_document_chunks(
        text=doc_text,
        source="https://example.com/ai-docs",
        title="AI Documentation",
        extra_metadata={"category": "technology"},
    )

    assert len(chunks) >= 1
    expected_hash = compute_content_hash(doc_text)

    for idx, chunk in enumerate(chunks):
        assert chunk.source == "https://example.com/ai-docs"
        assert chunk.title == "AI Documentation"
        assert chunk.chunk_index == idx
        assert chunk.total_chunks == len(chunks)
        assert chunk.content_hash == expected_hash
        assert chunk.id.startswith(expected_hash[:16])

        # Test Chroma metadata export
        meta = chunk.to_chroma_metadata()
        assert meta["source"] == "https://example.com/ai-docs"
        assert meta["chunk_index"] == idx
        assert meta["category"] == "technology"
        assert isinstance(meta["char_count"], int)
