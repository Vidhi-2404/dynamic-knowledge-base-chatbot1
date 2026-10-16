"""Unit tests for ChromaDB Vector Store Manager, deduplication, updates, and retrieval."""

from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig
from services.chunking import TextSplitter
from services.vector_store import VectorStoreManager


def test_deduplication_identical_content(vector_store: VectorStoreManager, splitter: TextSplitter):
    text = "Machine learning models require training datasets and optimization algorithms."
    source = "https://example.com/ml-basics"

    chunks1 = splitter.create_document_chunks(text, source=source, title="ML Basics")
    res1 = vector_store.add_or_update_document(chunks1)
    assert res1["status"] == "created"
    assert res1["chunks_count"] == len(chunks1)
    initial_count = vector_store.get_collection().count()

    # Re-index exact same content
    chunks2 = splitter.create_document_chunks(text, source=source, title="ML Basics")
    res2 = vector_store.add_or_update_document(chunks2)
    assert res2["status"] == "unchanged"
    assert "identical" in res2["message"].lower()

    # Verify no duplicate chunks were created
    assert vector_store.get_collection().count() == initial_count


def test_incremental_update_changed_content(vector_store: VectorStoreManager, splitter: TextSplitter):
    source = "https://example.com/versioned-doc"

    # Version 1
    text_v1 = "Version 1.0 of the software release notes."
    chunks_v1 = splitter.create_document_chunks(text_v1, source=source, title="Release Notes")
    res1 = vector_store.add_or_update_document(chunks_v1)
    assert res1["status"] == "created"
    v1_count = len(chunks_v1)
    assert vector_store.get_collection().count() == v1_count

    # Version 2 (substantially modified content)
    text_v2 = (
        "Version 2.0 release notes with complete architectural changes, "
        "new database integrations, enhanced security protocols, and performance upgrades."
    )
    chunks_v2 = splitter.create_document_chunks(text_v2, source=source, title="Release Notes v2")
    res2 = vector_store.add_or_update_document(chunks_v2)
    assert res2["status"] == "updated"
    assert res2["content_hash"] != res1["content_hash"]

    # Verify old chunks were purged and replaced by new chunks
    assert vector_store.get_collection().count() == len(chunks_v2)


def test_safe_failure_preserves_existing_content(vector_store: VectorStoreManager, splitter: TextSplitter):
    source = "https://example.com/stable-doc"
    text = "Important persistent corporate knowledge that must never be deleted on failure."

    chunks = splitter.create_document_chunks(text, source=source, title="Stable Doc")
    res = vector_store.add_or_update_document(chunks)
    assert res["status"] == "created"
    initial_count = vector_store.get_collection().count()

    # Empty chunks (simulating parser/fetch failure) should error without touching database
    fail_res = vector_store.add_or_update_document([])
    assert fail_res["status"] == "error"

    # Verify previous content is untouched
    assert vector_store.get_collection().count() == initial_count
    info = vector_store.get_source_info(source)
    assert info is not None
    assert info["source"] == source


def test_delete_source(vector_store: VectorStoreManager, splitter: TextSplitter):
    src1 = "doc1.txt"
    src2 = "doc2.txt"

    c1 = splitter.create_document_chunks("Data for document number one.", source=src1)
    c2 = splitter.create_document_chunks("Data for document number two.", source=src2)

    vector_store.add_or_update_document(c1)
    vector_store.add_or_update_document(c2)

    assert len(vector_store.get_all_indexed_sources()) == 2

    # Delete doc1
    del_res = vector_store.delete_source(src1)
    assert del_res["status"] == "deleted"

    remaining = vector_store.get_all_indexed_sources()
    assert len(remaining) == 1
    assert remaining[0]["source"] == src2


def test_semantic_search_retrieval(vector_store: VectorStoreManager, splitter: TextSplitter):
    doc_python = "Python is a dynamic high-level programming language widely used in AI."
    doc_biology = "Mitochondria are the powerhouse of eukaryotic biological cells."

    vector_store.add_or_update_document(splitter.create_document_chunks(doc_python, source="python.txt"))
    vector_store.add_or_update_document(splitter.create_document_chunks(doc_biology, source="biology.txt"))

    hits = vector_store.search("Which programming language is used in AI?", top_k=2)
    assert len(hits) >= 1
    top_hit = hits[0]
    assert top_hit["source"] == "python.txt"
    assert "Python" in top_hit["text"]
    assert top_hit["similarity"] > 0.0


def test_hosted_adapter_configuration():
    remote_cfg = AppConfig(
        chroma_adapter_type="http",
        chroma_host="remote.chromadb.io",
        chroma_port=8000,
        chroma_auth_token="secret-token-xyz",
        chroma_ssl=True,
    )
    with patch("chromadb.HttpClient") as mock_http_client:
        VectorStoreManager(app_config=remote_cfg)
        mock_http_client.assert_called_once_with(
            host="remote.chromadb.io",
            port=8000,
            ssl=True,
            headers={"Authorization": "Bearer secret-token-xyz"},
        )
