"""Non-interactive batch ingestion script for Dynamic Knowledge Base Chatbot.

Ingests public URLs and local document files (PDF, TXT, MD, CSV) into ChromaDB
with SSRF protection, deduplication, and incremental updating.

Usage:
    python scripts/ingest_sources.py --urls https://example.com/docs https://example.com/guide
    python scripts/ingest_sources.py --files ./docs/manual.pdf ./docs/faq.txt
    python scripts/ingest_sources.py --config scripts/sample_sources.json
    python scripts/ingest_sources.py --query "What is retrieval augmented generation?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from services.admin_service import AdminService
from services.qa_service import QAService
from services.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_sources")


def parse_manifest(config_path: str) -> tuple[List[str], List[str]]:
    """Parse JSON or newline-delimited text file containing URLs and file paths."""
    p = Path(config_path)
    if not p.is_file():
        logger.error(f"Config/manifest file not found: {config_path}")
        return [], []

    urls: List[str] = []
    files: List[str] = []

    if p.suffix.lower() == ".json":
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                urls = data.get("urls", [])
                files = data.get("files", [])
        except Exception as e:
            logger.error(f"Error parsing JSON manifest: {e}")
    else:
        # Line-delimited text file
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("http://") or line.startswith("https://"):
                    urls.append(line)
                else:
                    files.append(line)

    return urls, files


def run_ingestion(
    urls: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    config_file: Optional[str] = None,
    force_reindex: bool = False,
    query: Optional[str] = None,
) -> int:
    """Execute ingestion pipeline and return exit code (0 = success, 1 = failure)."""
    target_urls: List[str] = list(urls or [])
    target_files: List[str] = list(files or [])

    if config_file:
        cfg_urls, cfg_files = parse_manifest(config_file)
        target_urls.extend(cfg_urls)
        target_files.extend(cfg_files)

    # Deduplicate while preserving order
    target_urls = list(dict.fromkeys(target_urls))
    target_files = list(dict.fromkeys(target_files))

    logger.info("=" * 60)
    logger.info("Dynamic Knowledge Base - Non-Interactive Ingestion")
    logger.info(f"Target URLs: {len(target_urls)} | Target Files: {len(target_files)}")
    logger.info(f"Storage Adapter: {config.chroma_adapter_type} | Persist Dir: {config.chroma_persist_dir}")
    logger.info("=" * 60)

    store = VectorStoreManager()
    admin = AdminService(vector_store=store)

    success_count = 0
    failure_count = 0
    unchanged_count = 0

    # 1. Process URLs
    for url in target_urls:
        logger.info(f"--> Ingesting URL: {url}")
        res = admin.ingest_and_index_url(url, force_reindex=force_reindex)
        if res["success"]:
            if res.get("status") == "unchanged":
                unchanged_count += 1
                logger.info(f"    [UNCHANGED] {res['message']}")
            else:
                success_count += 1
                logger.info(f"    [SUCCESS] {res['message']} (Title: {res.get('title')})")
        else:
            failure_count += 1
            logger.warning(f"    [FAILED] {res['message']}")

    # 2. Process Files
    for file_path_str in target_files:
        fpath = Path(file_path_str)
        logger.info(f"--> Ingesting File: {fpath}")
        if not fpath.is_file():
            logger.error(f"    [ERROR] File does not exist: {fpath}")
            failure_count += 1
            continue

        try:
            file_bytes = fpath.read_bytes()
            res = admin.ingest_and_index_file(file_bytes, fpath.name, force_reindex=force_reindex)
            if res["success"]:
                if res.get("status") == "unchanged":
                    unchanged_count += 1
                    logger.info(f"    [UNCHANGED] {res['message']}")
                else:
                    success_count += 1
                    logger.info(f"    [SUCCESS] {res['message']} (Title: {res.get('title')})")
            else:
                failure_count += 1
                logger.warning(f"    [FAILED] {res['message']}")
        except Exception as e:
            failure_count += 1
            logger.error(f"    [ERROR] Exception processing {fpath.name}: {e}")

    logger.info("-" * 60)
    logger.info(f"Ingestion Finished: {success_count} indexed/updated, {unchanged_count} unchanged, {failure_count} failed.")

    # 3. Optional direct semantic search query verification
    if query:
        logger.info("=" * 60)
        logger.info(f"Verifying Retrieval with Query: '{query}'")
        qa = QAService(vector_store=store)
        response = qa.answer_query(query)
        logger.info(f"Mode: {response.mode} | Grounded: {response.is_grounded}")
        logger.info(f"Answer:\n{response.answer}")
        if response.citations:
            logger.info("Citations:")
            for c in response.citations:
                logger.info(f"  - [{c.similarity_score:.2f}] {c.title} ({c.source})")

    # If targets were specified and all failed, return error code
    total_targets = len(target_urls) + len(target_files)
    if total_targets > 0 and (success_count + unchanged_count) == 0:
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Ingest web URLs and local documents into the Dynamic Knowledge Base."
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="One or more approved public URLs to ingest.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="One or more local document paths (PDF, TXT, MD, CSV) to ingest.",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to JSON or TXT manifest containing sources to ingest.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-chunking and re-embedding even if content hash is identical.",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Run an immediate grounded semantic query after ingestion.",
    )

    args = parser.parse_args()

    # If no arguments provided, display help
    if not (args.urls or args.files or args.config or args.query):
        parser.print_help()
        sys.exit(0)

    exit_code = run_ingestion(
        urls=args.urls,
        files=args.files,
        config_file=args.config,
        force_reindex=args.force_reindex,
        query=args.query,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
