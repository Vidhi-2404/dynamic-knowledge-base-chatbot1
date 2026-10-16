"""Question Answering and Grounded Generation Service.

Provides:
1. Free, API-Key-Free Retrieval Mode using TF-IDF and keyword matching over candidate chunks.
2. Optional Google Gemini LLM generation mode when GEMINI_API_KEY is configured.
3. Strict non-hallucinatory grounding: admits lack of knowledge when retrieval fails.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import AppConfig, config
from services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
}


@dataclass
class RetrievedSourceCitation:
    """Represents a cited source in the answer."""

    source: str
    title: str
    is_url: bool
    similarity_score: float
    excerpt: str
    chunk_index: int


@dataclass
class QAResponse:
    """Container for the complete QA response."""

    query: str
    answer: str
    citations: List[RetrievedSourceCitation]
    mode: str  # "retrieval_mode", "gemini", or "no_match"
    is_grounded: bool
    warning_message: Optional[str] = None
    raw_chunks_count: int = 0


def is_web_url(source: str) -> bool:
    """Determine if a source identifier is a public HTTP/HTTPS URL."""
    return source.startswith("http://") or source.startswith("https://")


def tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric words."""
    return [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())]


class QAService:
    """Handles semantic retrieval and grounded answer generation."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        app_config: Optional[AppConfig] = None,
    ):
        self.vector_store = vector_store
        self.cfg = app_config or config

    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        force_mode: Optional[str] = None,
    ) -> QAResponse:
        """Execute semantic search and produce grounded answer.

        If force_mode is 'retrieval' or Gemini is not configured, runs in
        Free Retrieval Mode (zero API keys required).
        """
        query_str = query.strip()
        if not query_str:
            return QAResponse(
                query=query,
                answer="Please enter a valid question or search query.",
                citations=[],
                mode="empty_query",
                is_grounded=False,
            )

        # 1. Retrieve candidate chunks from vector store
        hits = self.vector_store.search(
            query=query_str,
            top_k=top_k or self.cfg.top_k,
            min_similarity=min_similarity,
        )

        # 2. Check if retrieval failed or yielded no relevant chunks
        if not hits:
            return QAResponse(
                query=query_str,
                answer=(
                    "I could not find any relevant information in the indexed knowledge base to answer your question. "
                    "Please verify that the pertinent documents or URLs are indexed, or try rephrasing your question."
                ),
                citations=[],
                mode="no_match",
                is_grounded=False,
                warning_message="No indexed documents met the relevance criteria.",
                raw_chunks_count=0,
            )

        # Build citations
        citations: List[RetrievedSourceCitation] = []
        for hit in hits:
            src = hit["source"]
            citations.append(
                RetrievedSourceCitation(
                    source=src,
                    title=hit.get("title", src),
                    is_url=is_web_url(src),
                    similarity_score=hit["similarity"],
                    excerpt=hit["text"][:280] + ("..." if len(hit["text"]) > 280 else ""),
                    chunk_index=hit.get("chunk_index", 0),
                )
            )

        # 3. Determine Execution Mode
        use_gemini = (
            force_mode != "retrieval"
            and self.cfg.is_gemini_configured
        )

        if use_gemini:
            try:
                answer = self._generate_gemini_answer(query_str, hits)
                return QAResponse(
                    query=query_str,
                    answer=answer,
                    citations=citations,
                    mode="gemini",
                    is_grounded=True,
                    raw_chunks_count=len(hits),
                )
            except Exception as e:
                logger.error(f"Gemini generation error: {e}. Falling back to Free Retrieval Mode.", exc_info=True)
                retrieval_ans = self._generate_retrieval_answer(query_str, hits)
                return QAResponse(
                    query=query_str,
                    answer=retrieval_ans,
                    citations=citations,
                    mode="retrieval_mode",
                    is_grounded=True,
                    warning_message=f"Gemini API unavailable ({e}). Automatically using Free Retrieval Mode.",
                    raw_chunks_count=len(hits),
                )
        else:
            # Free API-Key-Free Retrieval Mode (Default)
            retrieval_ans = self._generate_retrieval_answer(query_str, hits)
            return QAResponse(
                query=query_str,
                answer=retrieval_ans,
                citations=citations,
                mode="retrieval_mode",
                is_grounded=True,
                warning_message=None,
                raw_chunks_count=len(hits),
            )

    def _generate_gemini_answer(self, query: str, hits: List[Dict[str, Any]]) -> str:
        """Call Google GenAI SDK with grounded prompt."""
        context_blocks = []
        for idx, hit in enumerate(hits, 1):
            src = hit["source"]
            title = hit.get("title", src)
            text = hit["text"]
            context_blocks.append(
                f"[Document {idx} | Source: {title} ({src})]\n{text}"
            )
        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are a factual, strictly grounded knowledge assistant. "
            "Your task is to answer the user's question using ONLY the provided context passages. "
            "Strict Grounding Rules:\n"
            "1. Answer strictly based on facts stated in the context.\n"
            "2. If the context does not contain enough information to answer the question, state: "
            "'Based on the indexed sources, the provided documents do not contain sufficient information to answer this question.'\n"
            "3. Do NOT extrapolate, hallucinate, or rely on outside knowledge.\n"
            "4. Cite sources using [Source: Title or URL] whenever stating facts from a document."
        )

        prompt = f"""CONTEXT DOCUMENTS:
{context_str}

USER QUESTION:
{query}

GROUNDED ANSWER:"""

        # Try google.genai (new SDK) first, fallback to google.generativeai
        try:
            from google import genai

            client = genai.Client(api_key=self.cfg.gemini_api_key)
            model_name = self.cfg.gemini_model or "gemini-1.5-flash"
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"system_instruction": system_instruction, "temperature": 0.2},
            )
            if response and response.text:
                return response.text.strip()
            return "Unable to generate answer from model response."
        except Exception as e_new:
            logger.info(f"google.genai call attempt: {e_new}; trying legacy google.generativeai interface...")
            import google.generativeai as genai_legacy

            genai_legacy.configure(api_key=self.cfg.gemini_api_key)
            model = genai_legacy.GenerativeModel(
                model_name=self.cfg.gemini_model or "gemini-1.5-flash",
                system_instruction=system_instruction,
            )
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2},
            )
            if response and response.text:
                return response.text.strip()
            return "Unable to generate answer from model response."

    def _generate_retrieval_answer(self, query: str, hits: List[Dict[str, Any]]) -> str:
        """Free, API-key-free answer synthesizer using TF-IDF sentence scoring.

        Extracts key sentences matching the query terms using term-frequency inverse-document-frequency
        combined with vector similarity, ensuring answers are 100% grounded and factual.
        """
        raw_query_tokens = tokenize(query)
        query_terms = [t for t in raw_query_tokens if t not in STOPWORDS]
        if not query_terms:
            query_terms = raw_query_tokens  # Fallback to all tokens if all were stopwords

        query_term_set = set(query_terms)

        # 1. Break retrieved chunks into sentence units
        candidate_sentences: List[Dict[str, Any]] = []
        for hit_idx, hit in enumerate(hits):
            text = hit.get("text", "")
            source = hit.get("source", "Unknown")
            title = hit.get("title", source)
            sim = hit.get("similarity", 0.0)

            # Split on sentence boundaries
            raw_sents = re.split(r"(?<=[.!?])\s+", text)
            for s in raw_sents:
                s_clean = s.strip()
                if len(s_clean) >= 25:
                    tokens = tokenize(s_clean)
                    candidate_sentences.append({
                        "text": s_clean,
                        "source": source,
                        "title": title,
                        "similarity": sim,
                        "tokens": tokens,
                        "hit_idx": hit_idx,
                    })

        if not candidate_sentences:
            # Fall back to returning top chunk text
            top_hit = hits[0]
            return (
                f"**Retrieved Knowledge Summary**:\n\n"
                f"{top_hit['text'][:450]}...\n\n"
                f"*Source: **{top_hit.get('title', top_hit['source'])}***"
            )

        # 2. Compute Document Frequencies for TF-IDF across candidate sentences
        N = len(candidate_sentences)
        doc_freq: Counter[str] = Counter()
        for sent in candidate_sentences:
            unique_tokens = set(sent["tokens"])
            for t in unique_tokens:
                doc_freq[t] += 1

        # 3. Score each sentence using TF-IDF + Vector Relevance
        scored_sentences: List[Tuple[float, Dict[str, Any]]] = []
        for sent in candidate_sentences:
            tokens = sent["tokens"]
            if not tokens:
                continue

            term_counts = Counter(tokens)
            total_tokens = len(tokens)

            tfidf_score = 0.0
            query_match_count = 0

            for qt in query_terms:
                if qt in term_counts:
                    tf = term_counts[qt] / total_tokens
                    df = doc_freq[qt]
                    # Smoothed inverse document frequency
                    idf = math.log((N + 1) / (df + 1)) + 1.0
                    tfidf_score += (tf * idf)
                    query_match_count += 1

            if query_match_count > 0:
                # Weight by query coverage, TF-IDF score, and chunk vector similarity
                coverage_boost = query_match_count / len(query_terms)
                final_score = (tfidf_score * 2.0 + coverage_boost) * (1.0 + sent["similarity"])
                scored_sentences.append((final_score, sent))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # 4. Assemble the top unique extracted facts
        extracted_facts: List[str] = []
        seen_texts = set()

        for score, sent in scored_sentences[:4]:
            norm = sent["text"].lower()
            if norm not in seen_texts:
                seen_texts.add(norm)
                extracted_facts.append(
                    f"• {sent['text']} *[Source: {sent['title']}]*"
                )

        if not extracted_facts:
            # No keyword overlap, return lead passage from top semantic match
            top = hits[0]
            return (
                f"**Relevant Knowledge Excerpt**:\n\n"
                f"> \"{top['text'][:380]}...\"\n\n"
                f"*Source: [{top.get('title', top['source'])}]({top['source']}) (Relevance: `{top['similarity']:.2f}`)*"
            )

        facts_block = "\n\n".join(extracted_facts)
        lead_chunk = hits[0]["text"][:280] + "..."

        return (
            f"**Key Findings from Retrieved Documents**:\n\n"
            f"{facts_block}\n\n"
            f"**Context Overview**:\n"
            f"> \"{lead_chunk}\"\n\n"
            f"*(Generated via local TF-IDF & semantic retrieval from {len(hits)} indexed documents. No external API key required.)*"
        )
