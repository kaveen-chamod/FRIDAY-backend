from __future__ import annotations

import http.client
from html.parser import HTMLParser
import ipaddress
import json
import socket
import ssl
from urllib.parse import urlsplit


_MAX_RESPONSE_BYTES = 128 * 1024
_SECURITY_HEADERS = {
    "strict-transport-security": "Enforces HTTPS for future visits.",
    "content-security-policy": "Restricts the sources a browser may load.",
    "x-frame-options": "Helps prevent clickjacking in legacy browsers.",
    "x-content-type-options": "Prevents MIME-type sniffing.",
    "referrer-policy": "Limits referrer information sent to other sites.",
    "permissions-policy": "Restricts access to browser capabilities.",
}
_CSRF_NAMES = {"csrf", "csrf_token", "_csrf", "authenticity_token", "xsrf-token"}


class _FormInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[dict[str, object]] = []
        self._current: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if tag.lower() == "form":
            self._current = {
                "method": attributes.get("method", "get").strip().lower(),
                "has_token_field": False,
            }
            self.forms.append(self._current)
        elif tag.lower() == "input" and self._current is not None:
            name = attributes.get("name", "").strip().lower()
            if any(token in name for token in _CSRF_NAMES):
                self._current["has_token_field"] = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "form":
            self._current = None


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, address: str, port: int, timeout: float):
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._resolved_address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._resolved_address, self.port), self.timeout, self.source_address
        )
        self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)


def _validated_target(raw_url: str) -> tuple[str, str, int, str]:
    if len(raw_url) > 2048:
        raise ValueError("Website URL must not exceed 2048 characters.")
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Provide a public HTTPS URL; HTTP and non-web schemes are not scanned.")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Credentials and URL fragments are not accepted.")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port.") from exc
    if port != 443:
        raise ValueError("Only standard HTTPS port 443 is supported.")
    hostname = parsed.hostname.rstrip(".")
    try:
        address = ipaddress.ip_address(hostname)
        addresses = [str(address)]
    except ValueError:
        try:
            addresses = sorted({
                result[4][0]
                for result in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            })
        except OSError as exc:
            raise ValueError(f"Could not resolve the public website host: {exc}") from exc
    if not addresses:
        raise ValueError("The host did not resolve to an address.")
    for address_text in addresses:
        address = ipaddress.ip_address(address_text)
        if not address.is_global:
            raise ValueError("Private, local, and reserved network addresses are not scanned.")
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 2048 or any(char in path for char in "\r\n\x00"):
        raise ValueError("Website path is invalid or too long.")
    return hostname, addresses[0], port, path


def _scan_site(url: str) -> dict:
    hostname, address, port, path = _validated_target(url)
    connection = _PinnedHTTPSConnection(hostname, address, port, timeout=10)
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Host": hostname,
                "User-Agent": "Friday-Passive-Security-Review/1.0",
                "Accept": "text/html,application/xhtml+xml",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        headers = {key.lower(): value.strip() for key, value in response.getheaders()}
        body = response.read(_MAX_RESPONSE_BYTES)
        content_type = headers.get("content-type", "").lower()
        findings = [
            {
                "check": name,
                "present": header in headers,
                "detail": headers.get(header, "Header not present"),
                "recommendation": recommendation,
            }
            for header, recommendation in _SECURITY_HEADERS.items()
            for name in (header,)
        ]
    finally:
        connection.close()

    disclosures = {
        key: headers[key]
        for key in ("server", "x-powered-by", "x-aspnet-version")
        if key in headers
    }
    form_inspector = _FormInspector()
    if "html" in content_type:
        form_inspector.feed(body.decode("utf-8", errors="replace"))
    csrf_observations = [
        "A POST form has no recognizable CSRF token field; server-side validation was not tested."
        for form in form_inspector.forms
        if form["method"] == "post" and not form["has_token_field"]
    ]
    return {
        "target": f"https://{hostname}{path}",
        "status_code": response.status,
        "security_headers": findings,
        "information_disclosure_headers": disclosures,
        "csrf_form_observations": csrf_observations,
        "scope": (
            "One HTTPS GET to the requested path; redirects are not followed. "
            "Form inspection is passive. No injection payloads, form submissions, "
            "or state-changing requests were sent."
        ),
        "injection_tests": "Not performed; this is a passive review.",
        "response_sample_bytes": min(len(body), _MAX_RESPONSE_BYTES),
    }


def scan_website(parameters: dict, **_context) -> str:
    """Passively inspect security headers on a public HTTPS website."""
    url = str(parameters.get("url") or "").strip()
    if not url:
        return "A public HTTPS website URL is required."
    try:
        return json.dumps(_scan_site(url), ensure_ascii=False, indent=2)
    except (ValueError, OSError, ssl.SSLError, http.client.HTTPException) as exc:
        return f"Website review could not be completed: {exc}"


TOOL = {
    "name": "security_scanner",
    "description": (
        "With explicit confirmation, run a passive review of security headers "
        "and server disclosure on a public HTTPS site. It does not submit forms "
        "or run injection tests."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "url": {
                "type": "STRING",
                "description": "Public HTTPS URL for a site you are authorized to review.",
            },
        },
        "required": ["url"],
    },
    "handler": scan_website,
}
