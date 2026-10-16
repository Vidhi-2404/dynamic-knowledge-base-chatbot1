# Dynamic Knowledge Base Chatbot

[![Tests](https://img.shields.io/badge/pytest-41%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/streamlit-1.57%2B-red.svg)]()
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--persistence-orange.svg)]()
[![Security](https://img.shields.io/badge/SSRF-Hardened-success.svg)]()

A production-grade, standalone AI-powered Knowledge Base Chatbot with automated ingestion, boundary-aware text chunking, SHA-256 deduplication, SSRF security guards, and grounded retrieval. Powered by Streamlit, ChromaDB, Sentence-Transformers, and the Google GenAI SDK.

---

## 🌟 Key Features

1. **Modern Streamlit Chat Interface (`app.py`)**
   - Glassmorphism design system with responsive cards, typography, and dark/light mode compatibility.
   - Interactive conversation history with instant reset.
   - Pre-populated clickable question prompts for immediate exploration.
   - Real-time loading spinners and status indicators.
   - Grounded citations with relevance/similarity confidence scores and source hyperlinks.

2. **Multi-Source Knowledge Ingestion (`services/ingestion.py`)**
   - **Public Web URLs**: HTML tag decomposition (removes nav, headers, footers, scripts), title extraction, and clean text extraction via BeautifulSoup.
   - **PDF Documents**: Native page-by-page extraction, title parsing, and page number tagging via PyMuPDF (`fitz`).
   - **Plain Text / Markdown / CSV**: Multi-encoding fallback support (UTF-8, Latin-1, CP1252).

3. **Strict SSRF & Network Hardening (`services/ssrf_protection.py`)**
   - Pre-flight DNS resolution before any connection.
   - Rejection of private IPv4/IPv6 subnets (RFC 1918, RFC 4193).
   - Rejection of loopback addresses (`127.0.0.1`, `::1`, `localhost`).
   - Rejection of link-local and cloud instance metadata services (`169.254.169.254`, `metadata.google.internal`).
   - Redirect loop validation (validates every single redirect destination IP).
   - Response size caps (5 MB default streaming cap) to prevent memory exhaustion and decompression bombs.
   - Strict connect and read timeouts (10 seconds default).
   - MIME Content-Type whitelisting (rejects binaries, audio, video).

4. **Boundary-Aware Chunking & Deduplication (`services/chunking.py`)**
   - Recursive character and sentence splitting with sliding window overlap preservation.
   - SHA-256 content hashing: Identical documents are detected instantly and skipped (zero wasted compute/embeddings).
   - Atomically updates changed documents without leaving orphaned vectors.
   - Non-destructive error isolation: Failed crawls or corrupted uploads never delete or damage previously indexed valid data.

5. **Configurable Vector Database Adapters (`services/vector_store.py`)**
   - **Local Persistent Mode**: Embedded ChromaDB persistent storage stored under `./data/chroma_db`.
   - **Hosted / Client-Server Mode**: Seamless connection to remote ChromaDB server or Chroma Cloud via `HttpClient` with Bearer token authentication and SSL.
   - **In-Memory Ephemeral Mode**: Instant isolated test execution.

6. **Grounded Retrieval & Dual Generation Modes (`services/qa_service.py`)**
   - **Gemini Generative Mode**: When `GEMINI_API_KEY` is provided, queries Google Gemini (`gemini-1.5-flash`) using strict system instructions to cite evidence and avoid hallucination.
   - **Verified Extractive Demo Mode**: When `GEMINI_API_KEY` is absent, the system operates in a clearly labeled demo mode, synthesizing direct extractive answers from top retrieved passages without synthetic claims.
   - **Honest Non-Hallucinatory Fallback**: If retrieval yields no matching documents, the assistant clearly states that no information was found.

7. **Knowledge Base Administration Dashboard**
   - Real-time telemetry: Indexed source count, total vector chunks, storage adapter, embedding model.
   - Password-protected with `ADMIN_SECRET` token authentication.
   - Source catalog table with chunk counts, content hashes, and individual source deletion.
   - Live URL fetcher and drag-and-drop multi-file uploader.
   - Audit and activity logs tracking all operations and error traces.
   - Irreversible database purge control with confirmation guard.

8. **Non-Interactive Batch Ingestion CLI (`scripts/ingest_sources.py`)**
   - Fully automated command-line tool for cron jobs, local scripts, and CI/CD pipelines.
   - Accepts URLs, file paths, JSON manifests, or line-delimited text manifests.
   - Immediate semantic retrieval verification flag (`--query`).

9. **Scheduled Cloud Updater Workflow (`.github/workflows/update-knowledge-base.yml`)**
   - Daily cron schedule (`0 2 * * *` at 02:00 UTC) and manual trigger (`workflow_dispatch`).
   - Connects to hosted ChromaDB using GitHub repository secrets, guaranteeing continuous updates without depending on a local laptop.

---

## 🏗️ Project Architecture

```
dynamic-knowledge-base-chatbot/
├── .github/
│   └── workflows/
│       └── update-knowledge-base.yml # Scheduled & manual GitHub Actions workflow
├── .env.example                      # Documented environment variable template
├── .gitignore                        # Git exclusion rules for secrets, DBs, and cache
├── README.md                         # Project documentation and deployment guide
├── requirements.txt                  # Production dependencies
├── config.py                         # Unified settings (Streamlit secrets + env vars)
├── app.py                            # Streamlit web application entry point
├── services/
│   ├── __init__.py                   # Service package exports
│   ├── ssrf_protection.py            # SSRF validation, DNS resolution, and secure fetcher
│   ├── ingestion.py                  # HTML, PDF, text extractors and document models
│   ├── chunking.py                   # Overlapping text splitter and hash generator
│   ├── embeddings.py                 # Sentence-Transformers / Chroma / GenAI embeddings
│   ├── vector_store.py               # ChromaDB adapter manager, deduplication & search
│   ├── qa_service.py                 # Semantic retrieval, Gemini prompt, and demo mode
│   └── admin_service.py              # Catalog, audit logs, authentication & stats
├── scripts/
│   ├── ingest_sources.py             # Non-interactive batch ingestion CLI
│   └── sample_sources.json           # Template manifest for automated ingestion
└── tests/
    ├── __init__.py
    ├── conftest.py                   # Pytest fixtures and ephemeral vector store setup
    ├── test_ssrf_protection.py       # IP validation, redirect inspection, and size limit tests
    ├── test_chunking.py              # Text splitting, overlap, and metadata tests
    ├── test_ingestion.py             # HTML, PDF, plain text, and error handling tests
    ├── test_vector_store.py          # Deduplication, incremental update, and delete tests
    ├── test_qa_service.py            # Grounded QA, demo mode, and citation tests
    └── test_cli.py                   # CLI arguments and manifest parsing tests
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.10 or 3.11
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/dynamic-knowledge-base-chatbot.git
cd dynamic-knowledge-base-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and set your preferred configuration:
```ini
# Optional: Google Gemini API Key for generative answers (leave empty for demo mode)
GEMINI_API_KEY=your_gemini_api_key_here

# Required: Secret passphrase for the admin panel
ADMIN_SECRET=admin123

# Vector Database Storage Adapter ("local" or "http")
CHROMA_ADAPTER_TYPE=local
CHROMA_PERSIST_DIR=./data/chroma_db
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 💻 CLI Ingestion Usage

Ingest sources directly from your terminal using `scripts/ingest_sources.py`:

```bash
# Ingest one or more public web URLs:
python scripts/ingest_sources.py --urls https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# Ingest local documents (PDF, TXT, MD, CSV):
python scripts/ingest_sources.py --files ./docs/manual.pdf ./docs/faq.txt

# Ingest from a manifest file:
python scripts/ingest_sources.py --config scripts/sample_sources.json

# Ingest and immediately verify with a test semantic query:
python scripts/ingest_sources.py --urls https://en.wikipedia.org/wiki/Vector_database --query "What is a vector index?"
```

---

## 🧪 Automated Testing

All tests use `pytest` with mocked networks, isolated ephemeral vector stores, and zero reliance on external live services.

Run the full test suite:
```bash
python -m pytest -v
```

### Verified Test Results
```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
rootdir: dynamic-knowledge-base-chatbot
collected 42 items

tests/test_chunking.py::test_clean_text_normalizes_whitespace PASSED     [  2%]
tests/test_chunking.py::test_compute_content_hash_deterministic PASSED   [  4%]
tests/test_chunking.py::test_text_splitter_invalid_overlap PASSED        [  7%]
tests/test_chunking.py::test_text_splitter_short_text PASSED             [  9%]
tests/test_chunking.py::test_text_splitter_empty_text PASSED             [ 11%]
tests/test_chunking.py::test_text_splitter_overlap_preservation PASSED   [ 14%]
tests/test_chunking.py::test_create_document_chunks_metadata PASSED      [ 16%]
tests/test_cli.py::test_parse_manifest_json PASSED                       [ 19%]
tests/test_cli.py::test_parse_manifest_txt PASSED                        [ 21%]
tests/test_cli.py::test_run_ingestion_success PASSED                     [ 23%]
tests/test_cli.py::test_run_ingestion_failure_exit_code PASSED           [ 26%]
tests/test_ingestion.py::test_extract_text_from_html PASSED              [ 28%]
tests/test_ingestion.py::test_extract_text_from_pdf PASSED               [ 30%]
tests/test_ingestion.py::test_extract_text_from_plain_text PASSED        [ 33%]
tests/test_ingestion.py::test_ingest_url_success PASSED                  [ 35%]
tests/test_ingestion.py::test_ingest_url_failed_download_safe_handling PASSED [ 38%]
tests/test_ingestion.py::test_ingest_file_unsupported_extension PASSED   [ 40%]
tests/test_ingestion.py::test_ingest_file_empty_content PASSED           [ 42%]
tests/test_ingestion.py::test_ingest_file_valid_pdf PASSED               [ 45%]
tests/test_qa_service.py::test_is_web_url_detection PASSED               [ 47%]
tests/test_qa_service.py::test_empty_knowledge_base_no_hallucination PASSED [ 50%]
tests/test_qa_service.py::test_free_retrieval_mode_tfidf_synthesis PASSED [ 52%]
tests/test_qa_service.py::test_gemini_generation_when_configured PASSED  [ 54%]
tests/test_qa_service.py::test_force_retrieval_mode_even_when_gemini_configured PASSED [ 57%]
tests/test_ssrf_protection.py::test_is_ip_allowed_loopback PASSED        [ 59%]
tests/test_ssrf_protection.py::test_is_ip_allowed_private_rfc1918 PASSED [ 61%]
tests/test_ssrf_protection.py::test_is_ip_allowed_cloud_metadata PASSED  [ 64%]
tests/test_ssrf_protection.py::test_is_ip_allowed_public PASSED          [ 66%]
tests/test_ssrf_protection.py::test_validate_url_schemes PASSED          [ 69%]
tests/test_ssrf_protection.py::test_validate_url_blocked_hostnames PASSED [ 71%]
tests/test_ssrf_protection.py::test_validate_url_dns_resolving_to_private_ip PASSED [ 73%]
tests/test_ssrf_protection.py::test_validate_url_safe_public_ip PASSED   [ 76%]
tests/test_ssrf_protection.py::test_secure_fetch_blocks_ssrf_before_request PASSED [ 78%]
tests/test_ssrf_protection.py::test_secure_fetch_rejects_unsafe_redirect PASSED [ 80%]
tests/test_ssrf_protection.py::test_secure_fetch_oversized_content_header PASSED [ 83%]
tests/test_ssrf_protection.py::test_secure_fetch_disallowed_content_type PASSED [ 85%]
tests/test_vector_store.py::test_deduplication_identical_content PASSED  [ 88%]
tests/test_vector_store.py::test_incremental_update_changed_content PASSED [ 90%]
tests/test_vector_store.py::test_safe_failure_preserves_existing_content PASSED [ 92%]
tests/test_vector_store.py::test_delete_source PASSED                    [ 95%]
tests/test_vector_store.py::test_semantic_search_retrieval PASSED        [ 97%]
tests/test_vector_store.py::test_hosted_adapter_configuration PASSED     [100%]

========================== 42 passed in 75.51s ==========================
```

---

## ☁️ Zero-API-Key Deployment Guide (Streamlit Community Cloud)

Follow these exact steps to publish and deploy your application **without needing any API key**:

### Step 1: Create a GitHub Repository & Push Code
1. Open [GitHub](https://github.com) and click **New Repository**.
2. Name it `dynamic-knowledge-base-chatbot` (Public or Private).
3. Do **not** initialize with a README or .gitignore (they are already prepared in the project).
4. Run in your terminal:
   ```bash
   git init
   git add .
   git commit -m "feat: complete dynamic knowledge base chatbot"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/dynamic-knowledge-base-chatbot.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Community Cloud (Zero Secrets Required!)
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **New app**.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/dynamic-knowledge-base-chatbot`.
4. Set branch: `main`.
5. Set main file path: `app.py`.
6. Click **Deploy!**
   - **No API keys or secrets are required to run.**
   - The app will automatically launch in **Free Retrieval Mode (TF-IDF & Vector Search)** using local ChromaDB storage.
   - The default admin secret for document indexing in the Admin panel is `admin123`.

*(Optional)* If you wish to customize your admin password or enable Gemini AI generation later, go to **App Settings > Secrets**:
```toml
# Optional: customize admin secret
ADMIN_SECRET = "your-custom-admin-password"

# Optional: unlock Google Gemini AI generation
# GEMINI_API_KEY = "your-gemini-api-key"
```

---

## 🔄 Scheduled Cloud Updates & Vector Database Persistence

### Why Hosted Persistence is Required for Scheduled Updaters
GitHub Actions runners run in **ephemeral, stateless virtual machines** that terminate after each run. Similarly, Streamlit Community Cloud instances restart periodically. If you want scheduled updates triggered by GitHub Actions (`.github/workflows/update-knowledge-base.yml`) to automatically update the knowledge base used by the live Streamlit app without requiring your personal laptop:

1. Deploy a persistent, hosted ChromaDB instance using any of the following:
   - **Chroma Cloud** ([trychroma.com](https://www.trychroma.com))
   - **Railway / Render**: Deploy the official Chroma Docker container (`chromadb/chroma:latest`)
   - **AWS ECS / EC2 or GCP Cloud Run**
2. In both **Streamlit Cloud Secrets** and **GitHub Repository Secrets**, configure:
   ```ini
   CHROMA_ADAPTER_TYPE = "http"
   CHROMA_HOST = "your-hosted-chroma-host.com"
   CHROMA_PORT = 8000
   CHROMA_SSL = true
   CHROMA_AUTH_TOKEN = "your-bearer-token"
   ```
3. With this configuration, both GitHub Actions (running daily at 02:00 UTC) and the live Streamlit web application communicate directly with the exact same vector store!

### Configuring GitHub Repository Secrets
Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**:
- `GEMINI_API_KEY`: Your Google Gemini API Key
- `ADMIN_SECRET`: Your admin passphrase
- `CHROMA_ADAPTER_TYPE`: `http`
- `CHROMA_HOST`: Remote hostname
- `CHROMA_PORT`: Remote port
- `CHROMA_SSL`: `true`
- `CHROMA_AUTH_TOKEN`: Hosted Chroma auth token

---

## 🔒 Security Specifications

- **SSRF Hardening**: Rejects loopback (`127.0.0.0/8`), RFC 1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and AWS/GCP/Azure link-local metadata endpoints (`169.254.169.254`).
- **Redirect Hijack Prevention**: Manually inspects every intermediate HTTP redirect (`301`, `302`, `307`, `308`) before issuing a GET request.
- **Decompression Bomb Protection**: Streams incoming payloads in 16 KB chunks up to `MAX_FETCH_SIZE_MB` (5 MB default), terminating downloads immediately if exceeded.
- **Admin Guard**: Sensitive write operations (ingestion, re-indexing, deletion, purge) are guarded by constant-time string comparison (`hmac.compare_digest`) against `ADMIN_SECRET`.
- **Zero Hallucination Guarantee**: When no indexed context matches a query, the model explicitly admits lack of information rather than generating ungrounded facts.
