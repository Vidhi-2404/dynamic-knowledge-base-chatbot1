"""Unit tests for the CLI ingestion script."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.ingest_sources import parse_manifest, run_ingestion


def test_parse_manifest_json(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "urls": ["https://example.com/one", "https://example.com/two"],
        "files": ["docs/faq.txt"],
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    urls, files = parse_manifest(str(manifest_file))
    assert urls == ["https://example.com/one", "https://example.com/two"]
    assert files == ["docs/faq.txt"]


def test_parse_manifest_txt(tmp_path: Path):
    manifest_file = tmp_path / "manifest.txt"
    manifest_file.write_text(
        "# Comment line\nhttps://example.com/alpha\n./local_file.pdf\n",
        encoding="utf-8",
    )

    urls, files = parse_manifest(str(manifest_file))
    assert urls == ["https://example.com/alpha"]
    assert files == ["./local_file.pdf"]


@patch("scripts.ingest_sources.AdminService")
@patch("scripts.ingest_sources.VectorStoreManager")
def test_run_ingestion_success(mock_vsm, mock_admin_cls):
    mock_admin = MagicMock()
    mock_admin.ingest_and_index_url.return_value = {
        "success": True,
        "status": "created",
        "message": "Successfully indexed",
        "title": "Example Page",
        "chunks_count": 5,
    }
    mock_admin_cls.return_value = mock_admin

    exit_code = run_ingestion(urls=["https://example.com/test"])
    assert exit_code == 0
    mock_admin.ingest_and_index_url.assert_called_once_with(
        "https://example.com/test",
        force_reindex=False,
    )


@patch("scripts.ingest_sources.AdminService")
@patch("scripts.ingest_sources.VectorStoreManager")
def test_run_ingestion_failure_exit_code(mock_vsm, mock_admin_cls):
    mock_admin = MagicMock()
    mock_admin.ingest_and_index_url.return_value = {
        "success": False,
        "status": "error",
        "message": "SSRF rejected",
    }
    mock_admin_cls.return_value = mock_admin

    exit_code = run_ingestion(urls=["http://127.0.0.1/admin"])
    assert exit_code == 1
