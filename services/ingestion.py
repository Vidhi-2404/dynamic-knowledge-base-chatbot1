"""Document Ingestion Service.

Extracts and cleans text from:
- Public web URLs (via SSRF-protected secure_fetch_url and BeautifulSoup)
- PDF files (via PyMuPDF / fitz with page extraction)
- Plain text (.txt), Markdown (.md), and CSV (.csv) files

Provides fail-safe results to ensure corrupt or failed sources do not
overwrite or destroy previously indexed valid content.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import bs4
from bs4 import BeautifulSoup

from services.ssrf_protection import (
    InvalidContentTypeError,
    OversizedContentError,
    SSRFValidationError,
    secure_fetch_url,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestedDocument:
    """Standard representation of an extracted document."""

    source: str
    title: str
    text: str
    content_type: str
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


def extract_text_from_html(html_bytes: bytes, source_url: str) -> Tuple[str, str]:
    """Parse HTML, strip navigational/script noise, and return (title, cleaned_text)."""
    # Try utf-8 decoding, fallback to bs4's auto-detection
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html_str = html_bytes.decode("latin-1", errors="replace")

    soup = BeautifulSoup(html_str, "html.parser")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1 and h1.get_text():
            title = h1.get_text().strip()
    if not title:
        # Fallback to URL path
        title = source_url.split("/")[-1] or source_url

    # Remove script, style, nav, footer, ads, header tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"]):
        tag.decompose()

    # Extract meaningful text from body or article
    main_content = soup.find("article") or soup.find("main") or soup.body or soup

    # Extract text with line breaks between blocks
    lines = []
    for element in main_content.stripped_strings:
        lines.append(element)

    cleaned_text = "\n".join(lines)
    return title, cleaned_text


def extract_text_from_pdf(pdf_bytes: bytes, source_name: str) -> Tuple[str, str, int]:
    """Extract text from PDF using PyMuPDF (fitz).

    Returns (title, text, page_count).
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as e:
            raise RuntimeError("PyMuPDF / fitz is not installed.") from e

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    title = ""
    if doc.metadata:
        title = doc.metadata.get("title", "")
    if not title:
        title = Path(source_name).name

    page_texts: List[str] = []
    page_count = len(doc)

    for page_idx in range(page_count):
        page = doc[page_idx]
        t = page.get_text("text").strip()
        if t:
            page_texts.append(f"--- Page {page_idx + 1} ---\n{t}")

    doc.close()
    full_text = "\n\n".join(page_texts)
    return title, full_text, page_count


def extract_text_from_plain_text(raw_bytes: bytes, filename: str) -> Tuple[str, str]:
    """Decode plain text or markdown files."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = raw_bytes.decode(enc)
            title = Path(filename).name
            return title, text
        except UnicodeDecodeError:
            continue
    return Path(filename).name, raw_bytes.decode("utf-8", errors="replace")


class DocumentIngestionService:
    """Unified ingestion service for web URLs and uploaded files."""

    def __init__(self, max_fetch_bytes: int = 5 * 1024 * 1024, timeout_seconds: int = 10):
        self.max_fetch_bytes = max_fetch_bytes
        self.timeout_seconds = timeout_seconds

    def ingest_url(self, url: str) -> IngestedDocument:
        """Fetch and extract text from an approved public URL with SSRF checks."""
        try:
            content_bytes, content_type, final_url = secure_fetch_url(
                url,
                max_bytes=self.max_fetch_bytes,
                timeout_seconds=self.timeout_seconds,
            )

            # Determine parser based on content-type or URL extension
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                title, text, page_count = extract_text_from_pdf(content_bytes, final_url)
                meta = {"page_count": page_count, "final_url": final_url, "content_type": "pdf"}
            else:
                title, text = extract_text_from_html(content_bytes, final_url)
                meta = {"final_url": final_url, "content_type": "html"}

            if not text.strip():
                return IngestedDocument(
                    source=url,
                    title=title or url,
                    text="",
                    content_type=content_type,
                    success=False,
                    error_message="Webpage contained no extractable textual content.",
                )

            return IngestedDocument(
                source=url,
                title=title,
                text=text,
                content_type=content_type,
                metadata=meta,
                success=True,
            )

        except (SSRFValidationError, OversizedContentError, InvalidContentTypeError) as e:
            logger.warning(f"Security validation rejected URL '{url}': {e}")
            return IngestedDocument(
                source=url,
                title=url,
                text="",
                content_type="unknown",
                success=False,
                error_message=f"Security rejection: {e}",
            )
        except Exception as e:
            logger.error(f"Error fetching URL '{url}': {e}", exc_info=True)
            return IngestedDocument(
                source=url,
                title=url,
                text="",
                content_type="unknown",
                success=False,
                error_message=f"Failed to ingest URL: {e}",
            )

    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> IngestedDocument:
        """Extract text from an uploaded file or local file."""
        if not file_bytes:
            return IngestedDocument(
                source=filename,
                title=filename,
                text="",
                content_type="empty",
                success=False,
                error_message="Uploaded file is empty.",
            )

        if len(file_bytes) > self.max_fetch_bytes:
            return IngestedDocument(
                source=filename,
                title=filename,
                text="",
                content_type="oversized",
                success=False,
                error_message=f"File exceeds maximum size limit of {self.max_fetch_bytes // (1024*1024)} MB.",
            )

        ext = Path(filename).suffix.lower()

        try:
            if ext == ".pdf":
                title, text, page_count = extract_text_from_pdf(file_bytes, filename)
                meta = {"page_count": page_count, "filename": filename, "type": "pdf"}
            elif ext in (".txt", ".md", ".csv", ".json", ".log"):
                title, text = extract_text_from_plain_text(file_bytes, filename)
                meta = {"filename": filename, "type": ext.lstrip(".")}
            else:
                return IngestedDocument(
                    source=filename,
                    title=filename,
                    text="",
                    content_type=ext,
                    success=False,
                    error_message=f"Unsupported file format '{ext}'. Allowed: .pdf, .txt, .md, .csv",
                )

            if not text.strip():
                return IngestedDocument(
                    source=filename,
                    title=title,
                    text="",
                    content_type=ext,
                    success=False,
                    error_message="Document contains no readable text.",
                )

            return IngestedDocument(
                source=filename,
                title=title,
                text=text,
                content_type=ext,
                metadata=meta,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error processing file '{filename}': {e}", exc_info=True)
            return IngestedDocument(
                source=filename,
                title=filename,
                text="",
                content_type=ext,
                success=False,
                error_message=f"Failed to parse document: {e}",
            )

    def ingest_local_path(self, file_path: Union[str, Path]) -> IngestedDocument:
        """Ingest a file from local disk."""
        path = Path(file_path)
        if not path.is_file():
            return IngestedDocument(
                source=str(path),
                title=path.name,
                text="",
                content_type="missing",
                success=False,
                error_message=f"File not found: {path}",
            )
        try:
            data = path.read_bytes()
            return self.ingest_file(data, path.name)
        except Exception as e:
            return IngestedDocument(
                source=str(path),
                title=path.name,
                text="",
                content_type="unreadable",
                success=False,
                error_message=f"Failed reading file from disk: {e}",
            )
