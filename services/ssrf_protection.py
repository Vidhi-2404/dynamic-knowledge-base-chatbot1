"""SSRF Protection and Secure Network Fetching Service.

Protects against:
- Private IP ranges (RFC 1918)
- Loopback / Localhost (127.0.0.1, ::1)
- Link-Local and Cloud Metadata endpoints (169.254.169.254, metadata.google.internal)
- Internal domain patterns (.local, .internal, localhost)
- Unsafe protocols (file://, ftp://, gopher://, etc.)
- Open redirect SSRF bypasses (validates every redirect target)
- Oversized responses / Denial of Service (Content-Length and stream limits)
- Slowloris / Hung connections (strict connect/read timeouts)
- Invalid or binary Content-Types
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# Cloud metadata endpoints and hostnames to block explicitly
BLOCKED_HOSTNAMES: Set[str] = {
    "localhost",
    "metadata.google.internal",
    "metadata.internal",
    "instance-data",
    "169.254.169.254",
    "fd00::",
}

# Accepted MIME content-type prefixes or exact types
ALLOWED_CONTENT_TYPES: Set[str] = {
    "text/html",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/pdf",
    "application/json",
    "application/xml",
    "text/xml",
}


class SSRFValidationError(Exception):
    """Raised when a URL fails SSRF security checks."""


class OversizedContentError(Exception):
    """Raised when remote content exceeds allowed download size."""


class InvalidContentTypeError(Exception):
    """Raised when remote content-type is disallowed."""


def is_ip_allowed(ip_str: str) -> Tuple[bool, str]:
    """Check if an IP address string is safe (public, non-loopback, non-private)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False, f"Invalid IP address representation: {ip_str}"

    if ip.is_loopback:
        return False, f"Loopback address rejected: {ip_str}"
    if ip.is_private:
        return False, f"Private/Internal IP rejected: {ip_str}"
    if ip.is_link_local:
        return False, f"Link-local/Cloud metadata IP rejected: {ip_str}"
    if ip.is_multicast:
        return False, f"Multicast address rejected: {ip_str}"
    if ip.is_reserved:
        return False, f"Reserved IP address rejected: {ip_str}"
    if ip.is_unspecified:
        return False, f"Unspecified IP address rejected: {ip_str}"

    # Explicit check for AWS/GCP/Azure 169.254.x.x
    if str(ip).startswith("169.254."):
        return False, f"Cloud instance metadata IP rejected: {ip_str}"

    return True, "Safe"


def validate_url(url: str, allowed_schemes: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """Validate a URL against SSRF vulnerabilities before any connection attempt.

    Returns:
        (is_valid: bool, reason: str)
    """
    if allowed_schemes is None:
        allowed_schemes = {"http", "https"}

    if not url or not isinstance(url, str):
        return False, "Empty or non-string URL."

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL syntax: {e}"

    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        return False, f"Disallowed URL scheme '{scheme}'. Only {allowed_schemes} are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL lacks a valid hostname."

    hostname_lower = hostname.lower()

    # Block well-known internal hostnames
    if hostname_lower in BLOCKED_HOSTNAMES:
        return False, f"Forbidden internal hostname: {hostname}"

    if hostname_lower.endswith(".local") or hostname_lower.endswith(".internal") or hostname_lower.endswith(".localhost"):
        return False, f"Forbidden internal top-level domain in: {hostname}"

    # Resolve hostname to all associated IP addresses
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        return False, f"DNS resolution failed for hostname '{hostname}': {e}"
    except Exception as e:
        return False, f"Failed resolving host '{hostname}': {e}"

    if not addr_info:
        return False, f"No IP address found for host '{hostname}'."

    # Validate every resolved IP
    for info in addr_info:
        ip = info[4][0]
        safe, reason = is_ip_allowed(ip)
        if not safe:
            return False, f"Host '{hostname}' resolves to disallowed IP '{ip}': {reason}"

    return True, "URL is safe"


def secure_fetch_url(
    url: str,
    max_bytes: int = 5 * 1024 * 1024,
    timeout_seconds: int = 10,
    max_redirects: int = 4,
) -> Tuple[bytes, str, str]:
    """Fetch remote content safely with SSRF protection, size limits, and redirect validation.

    Returns:
        (content_bytes, content_type, final_url)
    """
    current_url = url
    redirects_followed = 0

    headers = {
        "User-Agent": (
            "DynamicKB-Bot/1.0 (KnowledgeBase Ingestion; +https://github.com/standalone-kb)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain,application/pdf,*/*;q=0.8",
    }

    session = requests.Session()

    while True:
        # Pre-flight SSRF check on every URL hop
        safe, reason = validate_url(current_url)
        if not safe:
            raise SSRFValidationError(f"SSRF check failed for URL '{current_url}': {reason}")

        try:
            resp = session.get(
                current_url,
                headers=headers,
                timeout=(timeout_seconds, timeout_seconds),
                stream=True,
                allow_redirects=False,
            )
        except RequestException as e:
            raise RuntimeError(f"Network error while fetching '{current_url}': {e}") from e

        # Handle Redirects manually to enforce SSRF validation on the destination
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            redirects_followed += 1
            if redirects_followed > max_redirects:
                resp.close()
                raise SSRFValidationError(f"Exceeded max redirects ({max_redirects}).")

            redirect_target = resp.headers.get("Location")
            if not redirect_target:
                resp.close()
                raise SSRFValidationError("Redirect response missing Location header.")

            # Resolve relative redirects
            from urllib.parse import urljoin

            current_url = urljoin(current_url, redirect_target)
            resp.close()
            continue

        # Check HTTP status
        if resp.status_code != 200:
            resp.close()
            raise RuntimeError(
                f"HTTP fetch failed for '{current_url}' with status {resp.status_code}: {resp.reason}"
            )

        # Content-Type verification
        content_type_header = resp.headers.get("Content-Type", "").lower()
        content_type = content_type_header.split(";")[0].strip()

        # Check if MIME type is recognized/allowed
        is_allowed = any(
            content_type.startswith(allowed) or allowed in content_type
            for allowed in ALLOWED_CONTENT_TYPES
        )
        if not is_allowed and content_type:
            resp.close()
            raise InvalidContentTypeError(
                f"Unsupported content type '{content_type}' for URL '{current_url}'. Allowed types: {ALLOWED_CONTENT_TYPES}"
            )

        # Content-Length header pre-check
        content_length = resp.headers.get("Content-Length")
        if content_length:
            try:
                length_int = int(content_length)
                if length_int > max_bytes:
                    resp.close()
                    raise OversizedContentError(
                        f"Resource size ({length_int} bytes) exceeds limit of {max_bytes} bytes."
                    )
            except ValueError:
                pass

        # Stream reading with hard byte cap (protects against gzip/deflate bombs & missing Content-Length)
        downloaded = bytearray()
        try:
            for chunk in resp.iter_content(chunk_size=16384):
                if chunk:
                    downloaded.extend(chunk)
                    if len(downloaded) > max_bytes:
                        raise OversizedContentError(
                            f"Downloaded content exceeded {max_bytes} bytes limit."
                        )
        finally:
            resp.close()

        return bytes(downloaded), content_type, current_url
