"""Unit tests for Question Answering, Grounded Retrieval, Citations, and Free Retrieval Mode."""

from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig
from services.chunking import TextSplitter
from services.qa_service import QAService, is_web_url
from services.vector_store import VectorStoreManager


def test_is_web_url_detection():
    assert is_web_url("https://example.com/guide") is True
    assert is_web_url("http://example.org/page") is True
    assert is_web_url("manual.pdf") is False
    assert is_web_url("./docs/readme.txt") is False


def test_empty_knowledge_base_no_hallucination(qa_service: QAService):
    resp = qa_service.answer_query("What is quantum computing?")
    assert resp.mode == "no_match"
    assert resp.is_grounded is False
    assert len(resp.citations) == 0
    assert "could not find any relevant information" in resp.answer.lower()


def test_free_retrieval_mode_tfidf_synthesis(vector_store: VectorStoreManager, splitter: TextSplitter):
    # Vector store populated, but GEMINI_API_KEY is empty (API-free mode)
    cfg = AppConfig(gemini_api_key="", top_k=2)
    qa = QAService(vector_store=vector_store, app_config=cfg)

    doc_text = (
        "Retrieval Augmented Generation (RAG) is an architectural pattern in AI. "
        "It combines an external information retrieval system with a generative model. "
        "This drastically reduces hallucinations and grounds responses in factual documents."
    )
    vector_store.add_or_update_document(
        splitter.create_document_chunks(doc_text, source="https://ai.org/rag", title="RAG Guide")
    )

    resp = qa.answer_query("What is Retrieval Augmented Generation?")
    assert resp.mode == "retrieval_mode"
    assert resp.is_grounded is True
    assert len(resp.citations) >= 1
    assert "Retrieval Augmented Generation" in resp.answer
    assert resp.citations[0].source == "https://ai.org/rag"
    assert resp.citations[0].is_url is True
    assert "Key Findings from Retrieved Documents" in resp.answer or "Retrieved Knowledge" in resp.answer


def test_gemini_generation_when_configured(vector_store: VectorStoreManager, splitter: TextSplitter):
    cfg = AppConfig(gemini_api_key="mock-valid-api-key", top_k=2)
    qa = QAService(vector_store=vector_store, app_config=cfg)

    vector_store.add_or_update_document(
        splitter.create_document_chunks(
            "ChromaDB is an open-source embedding database.",
            source="chroma.pdf",
            title="ChromaDB Manual",
        )
    )

    with patch("services.qa_service.QAService._generate_gemini_answer") as mock_gemini:
        mock_gemini.return_value = "ChromaDB is an open-source embedding database [Source: ChromaDB Manual]."
        resp = qa.answer_query("What is ChromaDB?", force_mode="gemini")

        assert resp.mode == "gemini"
        assert resp.is_grounded is True
        assert "ChromaDB is an open-source" in resp.answer
        assert len(resp.citations) >= 1
        assert resp.citations[0].is_url is False
        assert resp.citations[0].source == "chroma.pdf"


def test_force_retrieval_mode_even_when_gemini_configured(vector_store: VectorStoreManager, splitter: TextSplitter):
    cfg = AppConfig(gemini_api_key="mock-valid-api-key", top_k=2)
    qa = QAService(vector_store=vector_store, app_config=cfg)

    vector_store.add_or_update_document(
        splitter.create_document_chunks(
            "Vector databases store embeddings for nearest neighbor search.",
            source="vectors.txt",
            title="Vector DBs",
        )
    )

    # Force retrieval mode
    resp = qa.answer_query("What do vector databases store?", force_mode="retrieval")
    assert resp.mode == "retrieval_mode"
    assert resp.is_grounded is True
    assert "vector databases store" in resp.answer.lower() or "Vector DBs" in resp.answer
