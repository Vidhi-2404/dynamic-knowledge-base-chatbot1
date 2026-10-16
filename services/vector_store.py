"""Vector Store service built on ChromaDB with configurable storage adapters.

Supports:
- Local persistent storage (PersistentClient)
- Hosted / Client-Server ChromaDB (HttpClient)
- Ephemeral in-memory storage (EphemeralClient for unit testing)
- Content deduplication (hash checking)
- Safe incremental atomic updates (delete old chunks only if new content verified)
- Semantic search with cosine similarity scoring
- Source registry and administrative management
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from config import AppConfig, config
from services.chunking import DocumentChunk
from services.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector operations, adapter connections, deduplication, and updates."""

    def __init__(self, app_config: Optional[AppConfig] = None, client: Optional[ClientAPI] = None):
        self.cfg = app_config or config
        self._client: ClientAPI = client or self._init_client()
        self._embedding_function = get_embedding_function(
            provider=self.cfg.embedding_provider,
            model_name=self.cfg.embedding_model_name,
        )
        self._collection: Optional[Collection] = None

    def _init_client(self) -> ClientAPI:
        """Initialize ChromaDB client based on configured adapter type."""
        adapter = self.cfg.chroma_adapter_type.lower()

        if adapter == "http":
            logger.info(
                f"Connecting to hosted ChromaDB server at {self.cfg.chroma_host}:{self.cfg.chroma_port} (SSL: {self.cfg.chroma_ssl})"
            )
            headers = {}
            if self.cfg.chroma_auth_token:
                headers["Authorization"] = f"Bearer {self.cfg.chroma_auth_token}"

            return chromadb.HttpClient(
                host=self.cfg.chroma_host,
                port=self.cfg.chroma_port,
                ssl=self.cfg.chroma_ssl,
                headers=headers if headers else None,
            )

        elif adapter == "memory":
            logger.info("Initializing in-memory ChromaDB client (ephemeral mode).")
            return chromadb.EphemeralClient()

        else:
            # Default: local persistent directory
            persist_dir = os.path.abspath(self.cfg.chroma_persist_dir)
            os.makedirs(persist_dir, exist_ok=True)
            logger.info(f"Initializing local persistent ChromaDB at '{persist_dir}'.")
            return chromadb.PersistentClient(path=persist_dir)

    def get_collection(self) -> Collection:
        """Retrieve or create the knowledge base collection with cosine distance metric."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.cfg.chroma_collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def get_source_info(self, source: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata and chunk count for an existing source."""
        col = self.get_collection()
        try:
            results = col.get(
                where={"source": source},
                include=["metadatas"],
            )
            if not results or not results["ids"]:
                return None

            metadatas = results["metadatas"] or []
            first_meta = metadatas[0] if metadatas else {}
            return {
                "source": source,
                "title": first_meta.get("title", source),
                "content_hash": first_meta.get("content_hash", ""),
                "chunk_count": len(results["ids"]),
                "last_updated": first_meta.get("timestamp", ""),
            }
        except Exception as e:
            logger.error(f"Error querying source '{source}': {e}")
            return None

    def get_all_indexed_sources(self) -> List[Dict[str, Any]]:
        """List all unique sources indexed in the vector database with stats."""
        col = self.get_collection()
        try:
            total_count = col.count()
            if total_count == 0:
                return []

            # Retrieve all metadata to aggregate sources
            all_data = col.get(include=["metadatas"])
            metadatas = all_data.get("metadatas") or []

            sources_map: Dict[str, Dict[str, Any]] = {}
            for m in metadatas:
                src = m.get("source", "Unknown")
                if src not in sources_map:
                    sources_map[src] = {
                        "source": src,
                        "title": m.get("title", src),
                        "content_hash": m.get("content_hash", ""),
                        "chunk_count": 1,
                        "last_updated": m.get("timestamp", ""),
                    }
                else:
                    sources_map[src]["chunk_count"] += 1
                    # Keep the latest timestamp
                    ts = m.get("timestamp", "")
                    if ts > sources_map[src]["last_updated"]:
                        sources_map[src]["last_updated"] = ts

            return list(sources_map.values())
        except Exception as e:
            logger.error(f"Failed to list indexed sources: {e}")
            return []

    def add_or_update_document(
        self,
        chunks: List[DocumentChunk],
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """Safely index document chunks with deduplication and incremental updating.

        - If source exists and hash is identical -> skip (unchanged)
        - If source exists and hash changed -> delete old chunks and add new chunks
        - If new -> add chunks
        """
        if not chunks:
            return {
                "status": "error",
                "message": "No chunks provided to index.",
                "source": "",
                "chunks_count": 0,
            }

        source = chunks[0].source
        new_hash = chunks[0].content_hash
        col = self.get_collection()

        existing = self.get_source_info(source)

        if existing and not force_reindex:
            if existing.get("content_hash") == new_hash:
                logger.info(f"Source '{source}' is unchanged (hash: {new_hash[:8]}). Skipping re-index.")
                return {
                    "status": "unchanged",
                    "source": source,
                    "title": chunks[0].title,
                    "content_hash": new_hash,
                    "chunks_count": existing.get("chunk_count", 0),
                    "message": "Content is identical to existing indexed version; no re-index needed.",
                }

        # If source already exists with older content or force_reindex, safely purge old chunks first
        if existing:
            logger.info(f"Purging outdated chunks for source '{source}' prior to incremental update.")
            try:
                col.delete(where={"source": source})
            except Exception as e:
                logger.warning(f"Notice during old chunk deletion for '{source}': {e}")

        # Batch insert new chunks
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]

        col.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        status_type = "updated" if existing else "created"
        logger.info(f"Successfully {status_type} source '{source}' with {len(chunks)} chunks.")

        return {
            "status": status_type,
            "source": source,
            "title": chunks[0].title,
            "content_hash": new_hash,
            "chunks_count": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks.",
        }

    def delete_source(self, source: str) -> Dict[str, Any]:
        """Remove all chunks associated with a specific source."""
        col = self.get_collection()
        existing = self.get_source_info(source)
        if not existing:
            return {"status": "not_found", "source": source, "deleted_chunks": 0}

        col.delete(where={"source": source})
        return {
            "status": "deleted",
            "source": source,
            "deleted_chunks": existing["chunk_count"],
        }

    def purge_all(self) -> int:
        """Purge all data from the collection."""
        col = self.get_collection()
        count = col.count()
        if count > 0:
            # Recreate collection to reset everything cleanly
            self._client.delete_collection(self.cfg.chroma_collection_name)
            self._collection = None
            self.get_collection()
        return count

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a search query.

        Calculates similarity from cosine distance: similarity = 1 - distance
        Filters out matches below min_similarity.
        """
        k = top_k or self.cfg.top_k
        min_sim = min_similarity if min_similarity is not None else self.cfg.min_similarity_score

        col = self.get_collection()
        total_chunks = col.count()
        if total_chunks == 0 or not query.strip():
            return []

        # Chroma limit cannot exceed total count
        n_results = min(k, total_chunks)

        results = col.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for i in range(len(docs)):
            dist = distances[i] if i < len(distances) else 1.0
            # For cosine distance, distance ranges [0, 2], similarity = 1.0 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))

            if similarity < min_sim:
                continue

            meta = metas[i] if i < len(metas) else {}
            hits.append({
                "id": ids[i] if i < len(ids) else f"chunk_{i}",
                "text": docs[i],
                "source": meta.get("source", "Unknown"),
                "title": meta.get("title", meta.get("source", "Untitled")),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "similarity": round(similarity, 4),
                "distance": round(dist, 4),
                "metadata": meta,
            })

        # Sort descending by similarity
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits
