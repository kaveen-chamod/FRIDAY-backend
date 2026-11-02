"""Low-exposure HTTPS egress with DNS/IP validation and destination pinning."""
from __future__ import annotations

import http.client
import ipaddress
import logging
import re
import socket
import ssl
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.defense.asset_registry import AssetRegistryError


_LOG = logging.getLogger("friday.defense.egress")
_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 10.0
_MAX_RESPONSE_BYTES = 1_048_576
_USER_AGENT = "Friday-Defensive-Connector/1.0"
_SAFE_HEADERS = {"accept", "content-type", "if-none-match", "if-modified-since"}
_SECRET_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "key",
    "password",
    "secret",
    "token",
}
_SECRET_VALUE = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^&\r\n]+"
    r"|(token|password|secret|api[_-]?key)\s*[:=]\s*[^&\s]+"
)


class SecureAccessError(ValueError):
    """An outbound request was rejected or failed safe validation."""


@dataclass(frozen=True)
class ValidatedTarget:
    url: str
    hostname: str
    addresses: tuple[str, ...]
    request_target: str


def redact_outbound_log(value: str) -> str:
    text = _SECRET_VALUE.sub(
        lambda match: (
            f"{match.group(1)}[REDACTED]"
            if match.group(1)
            else f"{match.group(2)}=[REDACTED]"
        ),
        str(value),
    )
    try:
        parts = urlsplit(text)
        if parts.query:
            query = [
                (key, "[REDACTED]" if key.casefold() in _SECRET_QUERY_KEYS else val)
                for key, val in parse_qsl(parts.query, keep_blank_values=True)
            ]
            text = urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    parts.path,
                    urlencode(query, safe="[]"),
                    parts.fragment,
                )
            )
    except ValueError:
        return "[REDACTED_URL]"
    return text[:2_048]


def _is_forbidden_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError as exc:
        raise SecureAccessError("DNS returned an invalid IP address.") from exc
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        not ip.is_global
        or ip.is_multicast
        or ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_external_url(
    url: str,
    *,
    resolver=socket.getaddrinfo,
) -> ValidatedTarget:
    if not isinstance(url, str) or not 1 <= len(url) <= 2_048:
        raise SecureAccessError("URL must contain 1 to 2048 characters.")
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in url):
        raise SecureAccessError("URL cannot contain whitespace or control characters.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise SecureAccessError("URL is malformed.") from exc
    if (
        parts.scheme.lower() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or port not in (None, 443)
    ):
        raise SecureAccessError("Only credential-free HTTPS URLs on port 443 are supported.")
    hostname = parts.hostname.rstrip(".").lower()
    if not hostname or len(hostname) > 253 or "%" in hostname:
        raise SecureAccessError("URL hostname is invalid.")
    try:
        literal = ipaddress.ip_address(hostname)
        resolved = [str(literal)]
    except ValueError:
        try:
            results = resolver(hostname, 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            _LOG.warning("HTTPS destination DNS resolution failed for %s.", hostname)
            raise SecureAccessError("Destination could not be resolved.") from exc
        resolved = sorted({
            str(item[4][0])
            for item in results
            if item and len(item) > 4 and isinstance(item[4], tuple) and item[4]
        })
    if not resolved:
        raise SecureAccessError("Destination resolved to no addresses.")
    forbidden = [address for address in resolved if _is_forbidden_address(address)]
    if forbidden:
        _LOG.warning("Rejected non-public HTTPS destination %s.", hostname)
        raise SecureAccessError("Destination resolves to a private, local, or reserved address.")

    path = parts.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    request_target = path + (f"?{parts.query}" if parts.query else "")
    url_hostname = f"[{hostname}]" if ":" in hostname else hostname
    safe_url = urlunsplit(("https", url_hostname, parts.path or "/", parts.query, ""))
    return ValidatedTarget(
        url=safe_url,
        hostname=hostname,
        addresses=tuple(resolved),
        request_target=request_target,
    )


def sanitize_outbound_headers(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    result = {"User-Agent": _USER_AGENT}
    for key, value in (headers or {}).items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or key.casefold() not in _SAFE_HEADERS
            or "\r" in key
            or "\n" in key
            or "\r" in value
            or "\n" in value
            or len(value) > 2_048
        ):
            continue
        canonical = "-".join(part.capitalize() for part in key.split("-"))
        result[canonical] = value
    return result


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, address: str, context: ssl.SSLContext) -> None:
        super().__init__(hostname, 443, timeout=_CONNECT_TIMEOUT_SECONDS, context=context)
        self._pinned_address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_address, 443), timeout=_CONNECT_TIMEOUT_SECONDS
        )
        try:
            self.sock = self._context.wrap_socket(
                raw_socket, server_hostname=self.host
            )
            self.sock.settimeout(_READ_TIMEOUT_SECONDS)
        except Exception:
            raw_socket.close()
            raise


class SecureEgressClient:
    """Small GET-only HTTPS client; redirects and unpinned DNS are disallowed."""

    def __init__(self, *, resolver=socket.getaddrinfo) -> None:
        self._resolver = resolver

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
    ) -> tuple[int, bytes, dict[str, str]]:
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or not 1 <= max_response_bytes <= _MAX_RESPONSE_BYTES
        ):
            raise SecureAccessError("Response size limit is invalid.")
        target = validate_external_url(url, resolver=self._resolver)
        safe_headers = sanitize_outbound_headers(headers)
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        connection: _PinnedHTTPSConnection | None = None
        last_error: OSError | ssl.SSLError | None = None
        for address in target.addresses:
            connection = _PinnedHTTPSConnection(target.hostname, address, context)
            try:
                connection.request("GET", target.request_target, headers=safe_headers)
                response = connection.getresponse()
                if 300 <= response.status < 400:
                    raise SecureAccessError("HTTPS redirects are disabled.")
                payload = response.read(max_response_bytes + 1)
                if len(payload) > max_response_bytes:
                    raise SecureAccessError("HTTPS response exceeds the configured size limit.")
                response_headers = {
                    key.lower(): value[:2_048]
                    for key, value in response.getheaders()
                    if key.lower() in {"content-type", "etag", "last-modified"}
                }
                _LOG.info(
                    "HTTPS GET completed for %s with status %d.",
                    redact_outbound_log(target.url),
                    response.status,
                )
                return response.status, payload, response_headers
            except SecureAccessError:
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
        _LOG.warning("HTTPS request failed for %s.", redact_outbound_log(target.url))
        raise SecureAccessError("HTTPS request failed after validated destination checks.") from last_error


def require_registered_asset(registry, asset_id: str) -> dict[str, Any]:
    """Fail closed when a control or telemetry event names an unknown asset."""
    try:
        return registry.require(asset_id)
    except AssetRegistryError as exc:
        raise SecureAccessError("Only registered, ownership-verified assets are permitted.") from exc
