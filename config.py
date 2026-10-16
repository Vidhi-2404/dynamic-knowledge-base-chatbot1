"""Configuration module for Dynamic Knowledge Base Chatbot.

Loads settings from Streamlit secrets (if available), environment variables,
or sensible defaults. Works seamlessly across CLI scripts, tests, and Streamlit Cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load local .env if present
load_dotenv()


def _get_setting(key: str, default: str = "") -> str:
    """Retrieve setting from Streamlit secrets or OS environment."""
    # Check Streamlit secrets first if running within Streamlit runtime
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.getenv(key, default)


def _get_bool_setting(key: str, default: bool = False) -> bool:
    val = _get_setting(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


def _get_int_setting(key: str, default: int) -> int:
    val = _get_setting(key, str(default)).strip()
    try:
        return int(val)
    except ValueError:
        return default


def _get_float_setting(key: str, default: float) -> float:
    val = _get_setting(key, str(default)).strip()
    try:
        return float(val)
    except ValueError:
        return default


@dataclass
class AppConfig:
    """Application configuration container."""

    # Gemini API settings
    gemini_api_key: str = field(default_factory=lambda: _get_setting("GEMINI_API_KEY", ""))
    gemini_model: str = field(
        default_factory=lambda: _get_setting("GEMINI_MODEL", "gemini-1.5-flash")
    )

    # Admin secret for write operations
    admin_secret: str = field(
        default_factory=lambda: _get_setting("ADMIN_SECRET", "admin123")
    )

    # ChromaDB Vector Store settings
    chroma_adapter_type: str = field(
        default_factory=lambda: _get_setting("CHROMA_ADAPTER_TYPE", "local").lower()
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: _get_setting("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )
    chroma_collection_name: str = field(
        default_factory=lambda: _get_setting("CHROMA_COLLECTION_NAME", "dynamic_kb")
    )
    chroma_host: str = field(
        default_factory=lambda: _get_setting("CHROMA_HOST", "localhost")
    )
    chroma_port: int = field(
        default_factory=lambda: _get_int_setting("CHROMA_PORT", 8000)
    )
    chroma_ssl: bool = field(
        default_factory=lambda: _get_bool_setting("CHROMA_SSL", False)
    )
    chroma_auth_token: str = field(
        default_factory=lambda: _get_setting("CHROMA_AUTH_TOKEN", "")
    )

    # Embedding settings
    embedding_provider: str = field(
        default_factory=lambda: _get_setting("EMBEDDING_PROVIDER", "sentence-transformers").lower()
    )
    embedding_model_name: str = field(
        default_factory=lambda: _get_setting("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )

    # Document Chunking settings
    chunk_size: int = field(
        default_factory=lambda: _get_int_setting("CHUNK_SIZE", 600)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _get_int_setting("CHUNK_OVERLAP", 120)
    )

    # SSRF & Network Security
    max_fetch_size_mb: int = field(
        default_factory=lambda: _get_int_setting("MAX_FETCH_SIZE_MB", 5)
    )
    fetch_timeout_seconds: int = field(
        default_factory=lambda: _get_int_setting("FETCH_TIMEOUT_SECONDS", 10)
    )
    allowed_schemes: List[str] = field(
        default_factory=lambda: [
            s.strip().lower()
            for s in _get_setting("ALLOWED_URL_SCHEMES", "http,https").split(",")
            if s.strip()
        ]
    )

    # Retrieval settings
    top_k: int = field(
        default_factory=lambda: _get_int_setting("TOP_K_RETRIEVAL", 4)
    )
    min_similarity_score: float = field(
        default_factory=lambda: _get_float_setting("MIN_SIMILARITY_SCORE", 0.0)
    )

    @property
    def max_fetch_bytes(self) -> int:
        return self.max_fetch_size_mb * 1024 * 1024

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip() and self.gemini_api_key != "your_gemini_api_key_here")

    def validate_admin_token(self, provided_token: Optional[str]) -> bool:
        """Secure constant-time-like comparison for admin password."""
        if not provided_token:
            return False
        import hmac

        return hmac.compare_digest(provided_token.strip(), self.admin_secret.strip())


# Global singleton instance
config = AppConfig()
