"""Unit tests for SSRF Protection and Network Security."""

from unittest.mock import MagicMock, patch

import pytest

from services.ssrf_protection import (
    InvalidContentTypeError,
    OversizedContentError,
    SSRFValidationError,
    is_ip_allowed,
    secure_fetch_url,
    validate_url,
)


def test_is_ip_allowed_loopback():
    assert is_ip_allowed("127.0.0.1")[0] is False
    assert is_ip_allowed("127.0.0.2")[0] is False
    assert is_ip_allowed("::1")[0] is False


def test_is_ip_allowed_private_rfc1918():
    assert is_ip_allowed("10.0.0.1")[0] is False
    assert is_ip_allowed("172.16.0.1")[0] is False
    assert is_ip_allowed("192.168.1.1")[0] is False


def test_is_ip_allowed_cloud_metadata():
    assert is_ip_allowed("169.254.169.254")[0] is False
    assert is_ip_allowed("169.254.1.1")[0] is False


def test_is_ip_allowed_public():
    assert is_ip_allowed("8.8.8.8")[0] is True
    assert is_ip_allowed("1.1.1.1")[0] is True
    assert is_ip_allowed("93.184.216.34")[0] is True


def test_validate_url_schemes():
    assert validate_url("file:///etc/passwd")[0] is False
    assert validate_url("ftp://ftp.example.com/file.txt")[0] is False
    assert validate_url("gopher://evil.com/")[0] is False
    assert validate_url("javascript:alert(1)")[0] is False


def test_validate_url_blocked_hostnames():
    assert validate_url("http://localhost:8000")[0] is False
    assert validate_url("http://metadata.google.internal/computeMetadata/v1")[0] is False
    assert validate_url("http://service.local")[0] is False
    assert validate_url("http://database.internal")[0] is False


@patch("socket.getaddrinfo")
def test_validate_url_dns_resolving_to_private_ip(mock_dns):
    mock_dns.return_value = [(2, 1, 6, "", ("192.168.1.50", 80))]
    valid, reason = validate_url("http://evil-spoof.com/data")
    assert valid is False
    assert "Private/Internal IP rejected" in reason or "disallowed IP" in reason


@patch("socket.getaddrinfo")
def test_validate_url_safe_public_ip(mock_dns):
    mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
    valid, reason = validate_url("https://example.com/docs")
    assert valid is True
    assert "safe" in reason.lower()


@patch("services.ssrf_protection.validate_url")
def test_secure_fetch_blocks_ssrf_before_request(mock_val):
    mock_val.return_value = (False, "Blocked internal destination")
    with pytest.raises(SSRFValidationError, match="SSRF check failed"):
        secure_fetch_url("http://internal.service/secrets")


@patch("services.ssrf_protection.validate_url")
@patch("requests.Session.get")
def test_secure_fetch_rejects_unsafe_redirect(mock_get, mock_val):
    # First URL is allowed
    # Second URL (redirect destination) fails SSRF
    mock_val.side_effect = [
        (True, "Safe"),
        (False, "Redirect destination resolved to private IP"),
    ]

    redirect_resp = MagicMock()
    redirect_resp.is_redirect = True
    redirect_resp.status_code = 302
    redirect_resp.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
    mock_get.return_value = redirect_resp

    with pytest.raises(SSRFValidationError, match="SSRF check failed for URL"):
        secure_fetch_url("https://public-proxy.com/jump")


@patch("services.ssrf_protection.validate_url")
@patch("requests.Session.get")
def test_secure_fetch_oversized_content_header(mock_get, mock_val):
    mock_val.return_value = (True, "Safe")

    resp = MagicMock()
    resp.is_redirect = False
    resp.status_code = 200
    resp.headers = {
        "Content-Type": "text/html",
        "Content-Length": str(10 * 1024 * 1024),  # 10 MB > 1 MB limit
    }
    mock_get.return_value = resp

    with pytest.raises(OversizedContentError, match="exceeds limit"):
        secure_fetch_url("https://example.com/huge.iso", max_bytes=1024 * 1024)


@patch("services.ssrf_protection.validate_url")
@patch("requests.Session.get")
def test_secure_fetch_disallowed_content_type(mock_get, mock_val):
    mock_val.return_value = (True, "Safe")

    resp = MagicMock()
    resp.is_redirect = False
    resp.status_code = 200
    resp.headers = {"Content-Type": "audio/mpeg"}
    mock_get.return_value = resp

    with pytest.raises(InvalidContentTypeError, match="Unsupported content type"):
        secure_fetch_url("https://example.com/song.mp3")
