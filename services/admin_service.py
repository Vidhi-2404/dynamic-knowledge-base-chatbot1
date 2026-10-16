"""Knowledge Base Administration and Management Service.

Coordinates document ingestion, chunking, incremental vector updates,
activity logging, statistics aggregation, and administrative secret authentication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import AppConfig, config
from services.chunking import TextSplitter
from services.ingestion import DocumentIngestionService, IngestedDocument
from services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class ActivityLogEntry:
    """Audit log entry for knowledge base operations."""

    timestamp: str
    action: str
    source: str
    status: str  # "success", "unchanged", "warning", "error"
    message: str


class AdminService:
    """Administrative orchestrator for the Knowledge Base."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        app_config: Optional[AppConfig] = None,
    ):
        self.vector_store = vector_store
        self.cfg = app_config or config
        self.ingestion_service = DocumentIngestionService(
            max_fetch_bytes=self.cfg.max_fetch_bytes,
            timeout_seconds=self.cfg.fetch_timeout_seconds,
        )
        self.splitter = TextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        self.activity_logs: List[ActivityLogEntry] = []

    def _log_activity(self, action: str, source: str, status: str, message: str) -> None:
        entry = ActivityLogEntry(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            action=action,
            source=source,
            status=status,
            message=message,
        )
        self.activity_logs.insert(0, entry)
        # Keep latest 100 log entries
        if len(self.activity_logs) > 100:
            self.activity_logs = self.activity_logs[:100]

    def verify_admin_secret(self, secret: Optional[str]) -> bool:
        """Validate admin passphrase."""
        return self.cfg.validate_admin_token(secret)

    def get_system_stats(self) -> Dict[str, Any]:
        """Aggregate high-level knowledge base statistics."""
        sources = self.vector_store.get_all_indexed_sources()
        total_chunks = 0
        try:
            total_chunks = self.vector_store.get_collection().count()
        except Exception:
            pass

        return {
            "total_sources": len(sources),
            "total_chunks": total_chunks,
            "adapter_type": self.cfg.chroma_adapter_type,
            "collection_name": self.cfg.chroma_collection_name,
            "persist_dir": self.cfg.chroma_persist_dir,
            "embedding_provider": self.cfg.embedding_provider,
            "gemini_enabled": self.cfg.is_gemini_configured,
            "gemini_model": self.cfg.gemini_model,
            "sources": sources,
        }

    def ingest_and_index_url(
        self,
        url: str,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """Fetch, extract, chunk, and index a public webpage URL safely."""
        doc: IngestedDocument = self.ingestion_service.ingest_url(url)
        if not doc.success:
            self._log_activity(
                action="ingest_url",
                source=url,
                status="error",
                message=doc.error_message or "Ingestion failed.",
            )
            return {
                "success": False,
                "status": "error",
                "source": url,
                "message": doc.error_message,
            }

        # Create chunks
        chunks = self.splitter.create_document_chunks(
            text=doc.text,
            source=doc.source,
            title=doc.title,
            extra_metadata=doc.metadata,
        )

        if not chunks:
            msg = "Document text was empty after processing."
            self._log_activity("ingest_url", url, "error", msg)
            return {"success": False, "status": "error", "source": url, "message": msg}

        # Vector Store add or update
        result = self.vector_store.add_or_update_document(chunks, force_reindex=force_reindex)
        self._log_activity(
            action="ingest_url",
            source=url,
            status="success" if result["status"] in ("created", "updated") else "unchanged",
            message=result["message"],
        )

        return {
            "success": True,
            "status": result["status"],
            "source": url,
            "title": doc.title,
            "chunks_count": result.get("chunks_count", len(chunks)),
            "message": result["message"],
        }

    def ingest_and_index_file(
        self,
        file_bytes: bytes,
        filename: str,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """Parse and index uploaded file bytes (PDF, TXT, MD, CSV)."""
        doc = self.ingestion_service.ingest_file(file_bytes, filename)
        if not doc.success:
            self._log_activity("ingest_file", filename, "error", doc.error_message or "Failed to parse.")
            return {
                "success": False,
                "status": "error",
                "source": filename,
                "message": doc.error_message,
            }

        chunks = self.splitter.create_document_chunks(
            text=doc.text,
            source=filename,
            title=doc.title,
            extra_metadata=doc.metadata,
        )

        if not chunks:
            msg = "Document yielded no readable chunks."
            self._log_activity("ingest_file", filename, "error", msg)
            return {"success": False, "status": "error", "source": filename, "message": msg}

        result = self.vector_store.add_or_update_document(chunks, force_reindex=force_reindex)
        self._log_activity(
            action="ingest_file",
            source=filename,
            status="success" if result["status"] in ("created", "updated") else "unchanged",
            message=result["message"],
        )

        return {
            "success": True,
            "status": result["status"],
            "source": filename,
            "title": doc.title,
            "chunks_count": result.get("chunks_count", len(chunks)),
            "message": result["message"],
        }

    def delete_source(self, source: str) -> Dict[str, Any]:
        """Delete an indexed source and all its vectors."""
        res = self.vector_store.delete_source(source)
        if res.get("status") == "deleted":
            self._log_activity(
                "delete_source",
                source,
                "success",
                f"Deleted source and removed {res.get('deleted_chunks', 0)} chunks.",
            )
            return {"success": True, "message": f"Successfully removed '{source}'."}
        else:
            self._log_activity("delete_source", source, "warning", "Source not found in index.")
            return {"success": False, "message": f"Source '{source}' was not found in index."}

    def purge_database(self) -> int:
        """Purge all indexed knowledge."""
        count = self.vector_store.purge_all()
        self._log_activity(
            "purge_database",
            "ALL",
            "warning",
            f"Purged entire knowledge base ({count} chunks removed).",
        )
        return count
