"""Document text chunking and metadata preparation service.

Provides recursive boundary-aware splitting, sliding overlap preservation,
deterministic chunk IDs, and rich metadata attachments.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    """Represents a single text chunk with rich metadata."""

    id: str
    text: str
    source: str
    title: str
    chunk_index: int
    total_chunks: int
    content_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Return flat metadata compatible with ChromaDB constraints."""
        base = {
            "source": str(self.source),
            "title": str(self.title),
            "chunk_index": int(self.chunk_index),
            "total_chunks": int(self.total_chunks),
            "content_hash": str(self.content_hash),
            "timestamp": str(self.timestamp),
            "char_count": int(len(self.text)),
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                base[k] = v
            else:
                base[k] = str(v)
        return base


def compute_content_hash(text: str) -> str:
    """Generate SHA-256 hash of normalized text for deduplication."""
    normalized = text.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-printable characters."""
    if not text:
        return ""
    # Normalize carriage returns and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Collapse multiple horizontal spaces
    text = re.sub(r"[ ]{2,}", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TextSplitter:
    """Recursive boundary-aware text splitter with overlap retention."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_into_atoms(self, text: str, seps: List[str]) -> List[str]:
        """Recursively break text down until all individual pieces are <= chunk_size."""
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        # Choose the first separator that actually occurs in text
        chosen_sep = ""
        chosen_index = len(seps) - 1
        for i, s in enumerate(seps):
            if s == "":
                chosen_sep = ""
                chosen_index = i
                break
            if s in text:
                chosen_sep = s
                chosen_index = i
                break

        remaining_seps = seps[chosen_index + 1 :] if chosen_index + 1 < len(seps) else [""]

        if chosen_sep == "":
            # Hard character cut if no separator matches
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        parts = text.split(chosen_sep)
        atoms: List[str] = []
        for part in parts:
            p_strip = part.strip()
            if not p_strip:
                continue
            if len(p_strip) <= self.chunk_size:
                atoms.append(p_strip)
            else:
                atoms.extend(self._split_into_atoms(p_strip, remaining_seps))

        return atoms

    def split_text(self, text: str) -> List[str]:
        cleaned = clean_text(text)
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        atoms = self._split_into_atoms(cleaned, self.separators)
        if not atoms:
            return []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for atom in atoms:
            # Separator space between merged atoms
            added_len = len(atom) + (1 if current_chunk else 0)

            if current_length + added_len <= self.chunk_size:
                current_chunk.append(atom)
                current_length += added_len
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Build overlap from the tail of current_chunk
                    overlap_atoms: List[str] = []
                    overlap_len = 0
                    for prev_atom in reversed(current_chunk):
                        if overlap_len + len(prev_atom) <= self.chunk_overlap:
                            overlap_atoms.insert(0, prev_atom)
                            overlap_len += len(prev_atom) + 1
                        else:
                            break

                    current_chunk = overlap_atoms
                    current_length = sum(len(a) for a in current_chunk) + max(0, len(current_chunk) - 1)

                current_chunk.append(atom)
                current_length += len(atom) + (1 if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def create_document_chunks(
        self,
        text: str,
        source: str,
        title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split document text and package into DocumentChunk objects."""
        cleaned = clean_text(text)
        if not cleaned:
            return []

        doc_hash = compute_content_hash(cleaned)
        doc_title = title or source.split("/")[-1] or "Untitled Document"
        iso_now = datetime.now(timezone.utc).isoformat()

        text_pieces = self.split_text(cleaned)
        total_chunks = len(text_pieces)

        chunks: List[DocumentChunk] = []
        for idx, piece in enumerate(text_pieces):
            chunk_id = f"{doc_hash[:16]}_{idx:04d}"
            chunk = DocumentChunk(
                id=chunk_id,
                text=piece,
                source=source,
                title=doc_title,
                chunk_index=idx,
                total_chunks=total_chunks,
                content_hash=doc_hash,
                timestamp=iso_now,
                metadata=extra_metadata.copy() if extra_metadata else {},
            )
            chunks.append(chunk)

        return chunks
