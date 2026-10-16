"""Embedding provider service.

Implements Chroma-compatible embedding functions with support for:
- sentence-transformers (local, high performance, maintained)
- chromadb default (ONNX/MiniLM fallback)
- google-genai (text-embedding-004 via Google GenAI)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import chromadb.utils.embedding_functions as ef
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config import config

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbedding(EmbeddingFunction[Documents]):
    """Wrapper around SentenceTransformer with lazy model loading and caching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @staticmethod
    def name() -> str:
        return "sentence_transformer_local"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(model_name=config.get("model_name", "all-MiniLM-L6-v2"))

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer ({self.model_name}): {e}")
                raise
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        model = self._get_model()
        # Ensure input is list of strings
        texts = [str(t) for t in input]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


class GoogleGenAIEmbedding(EmbeddingFunction[Documents]):
    """Google Gemini text-embedding-004 provider."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name

    @staticmethod
    def name() -> str:
        return "google_genai_embedding"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(api_key="", model_name=config.get("model_name", "text-embedding-004"))

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            result = client.models.embed_content(
                model=self.model_name,
                contents=list(input),
            )
            # result.embeddings is a list of ContentEmbedding objects with .values
            return [e.values for e in result.embeddings]
        except Exception as e:
            logger.warning(f"Google GenAI embedding failed ({e}), falling back...")
            raise


def get_embedding_function(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> EmbeddingFunction[Documents]:
    """Factory to get the configured embedding function with graceful fallbacks."""
    provider = (provider or config.embedding_provider).lower()
    model_name = model_name or config.embedding_model_name

    if provider in ("sentence-transformers", "sentence_transformers"):
        try:
            return LocalSentenceTransformerEmbedding(model_name=model_name)
        except Exception as e:
            logger.warning(
                f"SentenceTransformer initialization failed: {e}. Falling back to Chroma Default."
            )

    elif provider == "gemini" and config.is_gemini_configured:
        try:
            return GoogleGenAIEmbedding(api_key=config.gemini_api_key, model_name="text-embedding-004")
        except Exception as e:
            logger.warning(f"Gemini embedding init failed: {e}. Falling back to default.")

    # Default robust fallback: Chroma's built-in DefaultEmbeddingFunction
    try:
        return ef.DefaultEmbeddingFunction()
    except Exception as e:
        logger.error(f"Chroma DefaultEmbeddingFunction failed: {e}")
        # Last resort: use sentence transformer
        return LocalSentenceTransformerEmbedding("all-MiniLM-L6-v2")
