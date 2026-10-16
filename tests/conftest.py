"""Pytest configuration and shared fixtures for Dynamic Knowledge Base Chatbot tests."""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables before importing app modules
os.environ["CHROMA_ADAPTER_TYPE"] = "memory"
os.environ["CHROMA_COLLECTION_NAME"] = "test_kb"
os.environ["ADMIN_SECRET"] = "testadminsecret"
os.environ["GEMINI_API_KEY"] = ""  # Default to demo mode in tests
os.environ["EMBEDDING_PROVIDER"] = "sentence-transformers"

from config import AppConfig
from services.admin_service import AdminService
from services.chunking import TextSplitter
from services.qa_service import QAService
from services.vector_store import VectorStoreManager


@pytest.fixture
def test_config() -> AppConfig:
    """Return an isolated test configuration."""
    return AppConfig(
        chroma_adapter_type="memory",
        chroma_collection_name="test_kb_fixture",
        admin_secret="testadminsecret",
        gemini_api_key="",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        min_similarity_score=0.0,
    )


@pytest.fixture
def vector_store(test_config: AppConfig) -> Generator[VectorStoreManager, None, None]:
    """Return an in-memory VectorStoreManager fixture."""
    vsm = VectorStoreManager(app_config=test_config)
    yield vsm
    try:
        vsm.purge_all()
    except Exception:
        pass


@pytest.fixture
def splitter() -> TextSplitter:
    """Return a configured TextSplitter fixture."""
    return TextSplitter(chunk_size=200, chunk_overlap=40)


@pytest.fixture
def admin_service(vector_store: VectorStoreManager, test_config: AppConfig) -> AdminService:
    """Return an AdminService wired to the ephemeral VectorStoreManager."""
    return AdminService(vector_store=vector_store, app_config=test_config)


@pytest.fixture
def qa_service(vector_store: VectorStoreManager, test_config: AppConfig) -> QAService:
    """Return a QAService wired to the ephemeral VectorStoreManager."""
    return QAService(vector_store=vector_store, app_config=test_config)
