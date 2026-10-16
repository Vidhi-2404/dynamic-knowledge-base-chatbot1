"""Dynamic Knowledge Base Chatbot - Streamlit Application.

A modern, production-grade knowledge assistant featuring:
- Semantic retrieval grounded strictly in indexed documents
- Google Gemini LLM generation with graceful offline/demo extractive mode
- Ingestion of public web URLs (with strict SSRF protection) and PDF/TXT/MD files
- Incremental updates & deduplication in ChromaDB
- Protected Knowledge Base Administration panel
"""

import os
import sys
from pathlib import Path

# Ensure project root directory is always on sys.path for Streamlit Community Cloud
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

from config import config
from services.admin_service import AdminService
from services.qa_service import QAService
from services.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Page configuration
st.set_page_config(
    page_title="Dynamic Knowledge Base Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern glassmorphism aesthetic, typography, and card accents
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    .source-badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 6px;
        margin-bottom: 4px;
        text-decoration: none;
    }

    .source-badge:hover {
        background: #334155;
        color: #7dd3fc;
    }

    .demo-banner {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 18px;
        color: #f59e0b;
        font-size: 0.92rem;
    }

    .metric-container {
        border-radius: 10px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }

    .citation-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #6366f1;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStoreManager:
    """Initialize or retrieve singleton VectorStoreManager instance."""
    return VectorStoreManager()


@st.cache_resource(show_spinner=False)
def get_services() -> tuple[AdminService, QAService]:
    """Initialize singleton AdminService and QAService."""
    store = get_vector_store()
    admin = AdminService(vector_store=store)
    qa = QAService(vector_store=store)
    return admin, qa


# Initialize session state for chat and admin auth
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **Dynamic Knowledge Base Chatbot**.\n\n"
                "I provide verified, fact-grounded answers strictly based on documents and public web sources "
                "indexed in my knowledge base. Ask me any question, or visit the **Knowledge Base Admin** "
                "tab to index new documents and URLs."
            ),
            "citations": [],
            "mode": "system",
        }
    ]

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

admin_service, qa_service = get_services()


# --- SIDEBAR: Navigation & System Telemetry ---
with st.sidebar:
    st.markdown("### 🧠 Dynamic Knowledge Base")
    st.markdown("---")

    selected_view = st.radio(
        "Navigation",
        ["💬 Chatbot Assistant", "🛠️ Knowledge Base Admin"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ System Status")

    stats = admin_service.get_system_stats()
    total_sources = stats.get("total_sources", 0)
    total_chunks = stats.get("total_chunks", 0)

    st.markdown(f"**Indexed Sources:** `{total_sources}`")
    st.markdown(f"**Knowledge Chunks:** `{total_chunks}`")
    st.markdown(f"**Vector DB:** `{config.chroma_adapter_type.upper()}`")
    st.markdown(f"**Embeddings:** `{config.embedding_model_name}`")

    if config.is_gemini_configured:
        selected_engine = st.selectbox(
            "Answering Engine",
            ["✨ Google Gemini AI", "🔍 Free Retrieval Mode (TF-IDF)"],
            index=0,
            help="Choose between Google Gemini generative answering or free local retrieval.",
        )
        st.caption(f"✨ Model: `{config.gemini_model}`")
    else:
        selected_engine = "🔍 Free Retrieval Mode (TF-IDF)"
        st.markdown(
            """
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px; font-size: 0.85rem; color: #38bdf8;">
                🟢 <strong>Free Retrieval Active</strong><br/>
                100% local & API-key-free.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation history reset. How can I assist you with your knowledge base?",
                "citations": [],
                "mode": "system",
            }
        ]
        st.rerun()

    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 30px;">
        Dynamic KB Chatbot v1.0<br/>
        Protected with SSRF guards & SHA-256 deduplication.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- VIEW 1: CHATBOT ASSISTANT ---
if selected_view == "💬 Chatbot Assistant":
    st.markdown('<div class="main-header">Dynamic Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions grounded strictly in indexed documents. No hallucinations or unverified claims.</div>',
        unsafe_allow_html=True,
    )

    # Active Engine indicator
    if "Gemini" not in selected_engine:
        st.markdown(
            """
            <div class="demo-banner" style="color: #38bdf8; border-color: rgba(56, 189, 248, 0.3); background: rgba(56, 189, 248, 0.08);">
                <strong>🔍 Free Retrieval Mode:</strong> Answers are retrieved and extracted directly from your indexed documents using local TF-IDF & vector relevance. Zero external API keys required.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Empty knowledge base warning
    if total_chunks == 0:
        st.info(
            "💡 **The knowledge base is currently empty.** Head to the **🛠️ Knowledge Base Admin** tab "
            "to index web URLs (e.g. Wikipedia pages) or upload PDF/text documents before asking questions."
        )

    # Example question chips
    st.markdown("##### 💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    sample_q = None

    with col1:
        if st.button("📌 What topics are currently indexed?", use_container_width=True):
            sample_q = "What topics and documents are currently indexed in this knowledge base?"
    with col2:
        if st.button("🔍 Explain retrieval augmented generation", use_container_width=True):
            sample_q = "Explain retrieval augmented generation and how it prevents hallucinations."
    with col3:
        if st.button("📑 Summarize the key facts from sources", use_container_width=True):
            sample_q = "Summarize the key facts and takeaways from the indexed documents."

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("mode") == "retrieval_mode":
                st.markdown(
                    '<span class="source-badge" style="background:#0f172a; color:#38bdf8; border-color:#0284c7;">🔍 Retrieval-Based Answer (API-Free)</span>',
                    unsafe_allow_html=True,
                )
            elif msg.get("mode") == "gemini":
                st.markdown(
                    '<span class="source-badge" style="background:#0f172a; color:#c084fc; border-color:#9333ea;">✨ Gemini AI Generated</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(msg["content"])

            # Render citations if present
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 Verified Sources & Evidence ({len(citations)})", expanded=False):
                    for idx, c in enumerate(citations, 1):
                        source_label = c.get("source", "Unknown")
                        title = c.get("title", source_label)
                        score = c.get("similarity_score", 0.0)
                        excerpt = c.get("excerpt", "")
                        is_url = c.get("is_url", False)

                        st.markdown(f"**[{idx}] {title}** *(Relevance Score: `{score:.2f}`)*")
                        if is_url:
                            st.markdown(f"🔗 [Open Original Webpage]({source_label})")
                        else:
                            st.markdown(f"📄 Local Document: `{source_label}`")

                        st.markdown(
                            f'<div class="citation-box">"{excerpt}"</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")

    # Chat input box
    user_prompt = st.chat_input("Ask a question about the indexed knowledge...")
    prompt_to_process = sample_q or user_prompt

    if prompt_to_process:
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Generate response
        with st.chat_message("assistant"):
            engine_mode = "gemini" if "Gemini" in selected_engine else "retrieval"
            with st.spinner("Searching knowledge base and verifying sources..."):
                resp = qa_service.answer_query(prompt_to_process, force_mode=engine_mode)

            if resp.mode == "retrieval_mode":
                st.markdown(
                    '<span class="source-badge" style="background:#0f172a; color:#38bdf8; border-color:#0284c7;">🔍 Retrieval-Based Answer (API-Free)</span>',
                    unsafe_allow_html=True,
                )
            elif resp.mode == "gemini":
                st.markdown(
                    '<span class="source-badge" style="background:#0f172a; color:#c084fc; border-color:#9333ea;">✨ Gemini AI Generated</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(resp.answer)

            # Display citations
            if resp.citations:
                with st.expander(f"📚 Verified Sources & Evidence ({len(resp.citations)})", expanded=True):
                    for idx, c in enumerate(resp.citations, 1):
                        st.markdown(f"**[{idx}] {c.title}** *(Relevance Score: `{c.similarity_score:.2f}`)*")
                        if c.is_url:
                            st.markdown(f"🔗 [Open Original Webpage]({c.source})")
                        else:
                            st.markdown(f"📄 Local Document: `{c.source}`")

                        st.markdown(
                            f'<div class="citation-box">"{c.excerpt}"</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")

            if resp.warning_message:
                st.caption(f"ℹ️ {resp.warning_message}")

        # Store in session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": resp.answer,
                "citations": [
                    {
                        "source": c.source,
                        "title": c.title,
                        "is_url": c.is_url,
                        "similarity_score": c.similarity_score,
                        "excerpt": c.excerpt,
                    }
                    for c in resp.citations
                ],
                "mode": resp.mode,
            }
        )


# --- VIEW 2: KNOWLEDGE BASE ADMIN ---
elif selected_view == "🛠️ Knowledge Base Admin":
    st.markdown('<div class="main-header">Knowledge Base Administration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Manage indexed sources, trigger real-time SSRF-safe ingestion, inspect document stats, and view audit logs.</div>',
        unsafe_allow_html=True,
    )

    # Admin Authentication Gate
    col_auth, _ = st.columns([2, 1])
    with col_auth:
        admin_input = st.text_input(
            "🔑 Admin Secret Passphrase",
            type="password",
            help="Required to execute write actions (indexing, re-indexing, deletion).",
        )
        is_auth = admin_service.verify_admin_secret(admin_input)

        if admin_input:
            if is_auth:
                st.success("✅ Admin Access Granted")
            else:
                st.error("❌ Invalid Admin Secret Passphrase")

    st.markdown("---")

    # Metrics Overview
    stats = admin_service.get_system_stats()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Indexed Sources", stats.get("total_sources", 0))
    with m2:
        st.metric("Total Vector Chunks", stats.get("total_chunks", 0))
    with m3:
        st.metric("Storage Adapter", stats.get("adapter_type", "local").upper())
    with m4:
        st.metric("Collection Name", stats.get("collection_name", "dynamic_kb"))

    st.markdown("### 📥 Ingest New Knowledge")
    tab_url, tab_file = st.tabs(["🌐 Ingest Public Webpage URL", "📄 Upload Document Files"])

    with tab_url:
        st.markdown(
            "Ingest content from an approved public URL. Protected against SSRF, internal IP access, and oversized payloads."
        )
        url_input = st.text_input(
            "Target Webpage URL",
            placeholder="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        )
        force_url_reindex = st.checkbox("Force re-index (overwrite even if content hash matches)", key="force_url")

        if st.button("🚀 Fetch & Index URL", type="primary"):
            if not is_auth:
                st.error("Authentication required: Please provide a valid Admin Secret above.")
            elif not url_input or not url_input.strip():
                st.warning("Please enter a valid URL.")
            else:
                with st.spinner(f"Validating SSRF guards and ingesting '{url_input}'..."):
                    result = admin_service.ingest_and_index_url(
                        url_input.strip(),
                        force_reindex=force_url_reindex,
                    )
                if result["success"]:
                    if result.get("status") == "unchanged":
                        st.info(f"ℹ️ {result['message']}")
                    else:
                        st.success(f"✅ {result['message']} (Indexed {result.get('chunks_count', 0)} chunks)")
                    st.rerun()
                else:
                    st.error(f"❌ Ingestion Failed: {result.get('message', 'Unknown error')}")

    with tab_file:
        st.markdown("Upload local documents to index into the knowledge base. Supported: **PDF, TXT, MD, CSV**.")
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "txt", "md", "csv"],
            accept_multiple_files=True,
        )
        force_file_reindex = st.checkbox("Force re-index (overwrite even if content hash matches)", key="force_file")

        if st.button("🚀 Process & Index Documents", type="primary"):
            if not is_auth:
                st.error("Authentication required: Please provide a valid Admin Secret above.")
            elif not uploaded_files:
                st.warning("Please select at least one document to upload.")
            else:
                successes = 0
                for uf in uploaded_files:
                    with st.spinner(f"Extracting and chunking {uf.name}..."):
                        file_bytes = uf.read()
                        res = admin_service.ingest_and_index_file(
                            file_bytes=file_bytes,
                            filename=uf.name,
                            force_reindex=force_file_reindex,
                        )
                        if res["success"]:
                            successes += 1
                            st.write(f"✅ **{uf.name}**: {res['message']}")
                        else:
                            st.write(f"❌ **{uf.name}**: {res.get('message')}")
                if successes > 0:
                    st.success(f"Successfully processed {successes} document(s).")
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Indexed Sources Catalog")

    sources = stats.get("sources", [])
    if not sources:
        st.info("No sources have been indexed yet.")
    else:
        for s in sources:
            with st.container():
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"**{s.get('title', s['source'])}**")
                    st.caption(
                        f"Source: `{s['source']}` | Chunks: `{s['chunk_count']}` | "
                        f"Hash: `{s.get('content_hash', '')[:12]}...` | Updated: `{s.get('last_updated', 'N/A')}`"
                    )
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_{s['source']}"):
                        if not is_auth:
                            st.error("Admin secret required to delete sources.")
                        else:
                            del_res = admin_service.delete_source(s["source"])
                            if del_res["success"]:
                                st.success(del_res["message"])
                                st.rerun()
                            else:
                                st.error(del_res["message"])
                st.divider()

    st.markdown("### 📝 Audit & Activity Logs")
    if admin_service.activity_logs:
        log_rows = [
            {
                "Timestamp (UTC)": log.timestamp,
                "Action": log.action,
                "Source": log.source,
                "Status": log.status.upper(),
                "Details": log.message,
            }
            for log in admin_service.activity_logs
        ]
        st.dataframe(log_rows, use_container_width=True)
    else:
        st.caption("No administrative actions recorded in this session yet.")

    # Danger Zone
    st.markdown("---")
    with st.expander("⚠️ Danger Zone: Purge All Knowledge Base Data", expanded=False):
        st.warning("This operation will permanently delete all vector embeddings and indexed documents!")
        confirm_purge = st.checkbox("I understand that this action is irreversible.")
        if st.button("🔥 Purge Entire Knowledge Base", type="secondary"):
            if not is_auth:
                st.error("Admin secret required.")
            elif not confirm_purge:
                st.warning("Please check the confirmation box before purging.")
            else:
                count = admin_service.purge_database()
                st.success(f"Purged {count} chunks. Knowledge base is now empty.")
                st.rerun()
