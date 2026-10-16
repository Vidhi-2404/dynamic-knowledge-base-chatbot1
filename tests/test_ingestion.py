"""Unit tests for document ingestion service and parsers."""

from unittest.mock import patch

import pymupdf as fitz
import pytest

from services.ingestion import (
    DocumentIngestionService,
    extract_text_from_html,
    extract_text_from_pdf,
    extract_text_from_plain_text,
)


def create_sample_pdf_bytes(text: str = "Test document content in PDF format.") -> bytes:
    """Helper to generate a minimal valid PDF in memory using pymupdf."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_text_from_html():
    raw_html = b"""
    <!DOCTYPE html>
    <html>
      <head><title>Vector Search Guide</title><script>var x = 1;</script></head>
      <body>
        <nav><a href="/">Home</a></nav>
        <h1>Main Heading</h1>
        <p>This is the first substantive paragraph.</p>
        <p>This is the second paragraph with details.</p>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    title, text = extract_text_from_html(raw_html, "https://example.com/guide")
    assert title == "Vector Search Guide"
    assert "Main Heading" in text
    assert "first substantive paragraph" in text
    assert "var x = 1" not in text
    assert "Copyright 2026" not in text
    assert "Home" not in text


def test_extract_text_from_pdf():
    pdf_bytes = create_sample_pdf_bytes("ChromaDB vector persistence test on page one.")
    title, text, page_count = extract_text_from_pdf(pdf_bytes, "manual.pdf")
    assert page_count == 1
    assert "ChromaDB vector persistence" in text
    assert "Page 1" in text


def test_extract_text_from_plain_text():
    sample = "Title: AI Knowledge Base\nContent line 1\nContent line 2".encode("utf-8")
    title, text = extract_text_from_plain_text(sample, "notes.txt")
    assert title == "notes.txt"
    assert "Content line 1" in text


@patch("services.ingestion.secure_fetch_url")
def test_ingest_url_success(mock_fetch):
    mock_fetch.return_value = (
        b"<html><head><title>RAG Overview</title></head><body><p>Retrieval Augmented Generation combines search and generation.</p></body></html>",
        "text/html",
        "https://example.com/rag",
    )
    svc = DocumentIngestionService()
    doc = svc.ingest_url("https://example.com/rag")
    assert doc.success is True
    assert doc.title == "RAG Overview"
    assert "Retrieval Augmented Generation" in doc.text


@patch("services.ingestion.secure_fetch_url")
def test_ingest_url_failed_download_safe_handling(mock_fetch):
    mock_fetch.side_effect = RuntimeError("Connection timeout (504 Gateway)")
    svc = DocumentIngestionService()
    doc = svc.ingest_url("https://example.com/unavailable")
    assert doc.success is False
    assert doc.text == ""
    assert "Failed to ingest URL" in doc.error_message


def test_ingest_file_unsupported_extension():
    svc = DocumentIngestionService()
    doc = svc.ingest_file(b"some binary content", "program.exe")
    assert doc.success is False
    assert "Unsupported file format" in doc.error_message


def test_ingest_file_empty_content():
    svc = DocumentIngestionService()
    doc = svc.ingest_file(b"", "empty.txt")
    assert doc.success is False
    assert "empty" in doc.error_message.lower()


def test_ingest_file_valid_pdf():
    svc = DocumentIngestionService()
    pdf_bytes = create_sample_pdf_bytes("Embedded systems in artificial intelligence.")
    doc = svc.ingest_file(pdf_bytes, "embedded_ai.pdf")
    assert doc.success is True
    assert "Embedded systems" in doc.text
    assert doc.metadata.get("page_count") == 1
