from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.defense.asset_registry import (
    AssetRegistryError,
    AuthorizedAssetRegistry,
)
from app.defense.secure_access import (
    SecureAccessError,
    redact_outbound_log,
    sanitize_outbound_headers,
    validate_external_url,
)


class SecureAccessTests(unittest.TestCase):
    def setUp(self):
        self.public_resolver = Mock(
            return_value=[
                (2, 1, 6, "", ("93.184.216.34", 443)),
            ]
        )

    def test_ssrf_rejects_loopback_private_and_cloud_metadata_addresses(self):
        cases = {
            "https://127.0.0.1/data": "127.0.0.1",
            "https://service.example/data": "10.1.2.3",
            "https://metadata.example/latest": "169.254.169.254",
            "https://service.example/v6": "fe80::1",
            "https://service.example/multicast": "224.0.0.1",
        }
        for url, address in cases.items():
            with self.subTest(url=url):
                resolver = Mock(
                    return_value=[(2, 1, 6, "", (address, 443))]
                )
                with self.assertRaises(SecureAccessError):
                    validate_external_url(url, resolver=resolver)

    def test_ssrf_rejects_mixed_public_and_private_dns_answers(self):
        resolver = Mock(
            return_value=[
                (2, 1, 6, "", ("93.184.216.34", 443)),
                (2, 1, 6, "", ("192.168.1.7", 443)),
            ]
        )
        with self.assertRaises(SecureAccessError):
            validate_external_url("https://mixed.example/", resolver=resolver)

    def test_public_dns_validation_returns_pinned_address_and_query(self):
        target = validate_external_url(
            "https://example.com/api?key=sensitive",
            resolver=self.public_resolver,
        )
        self.assertEqual(target.hostname, "example.com")
        self.assertEqual(target.addresses, ("93.184.216.34",))
        self.assertEqual(target.request_target, "/api?key=sensitive")
        self.public_resolver.assert_called_once_with(
            "example.com", 443, type=__import__("socket").SOCK_STREAM
        )

    def test_https_get_uses_tls_13_and_sanitized_headers(self):
        from app.defense.secure_access import SecureEgressClient

        connection = Mock()
        response = Mock()
        response.status = 200
        response.read.return_value = b"ok"
        response.getheaders.return_value = [("Content-Type", "application/json")]
        connection.getresponse.return_value = response
        context = Mock()
        client = SecureEgressClient(resolver=self.public_resolver)
        with (
            patch("app.defense.secure_access.ssl.create_default_context", return_value=context),
            patch(
                "app.defense.secure_access._PinnedHTTPSConnection",
                return_value=connection,
            ) as connection_factory,
        ):
            status, body, headers = client.get(
                "https://example.com/data",
                headers={"Authorization": "Bearer secret", "Accept": "application/json"},
            )
        self.assertEqual((status, body, headers), (200, b"ok", {"content-type": "application/json"}))
        self.assertEqual(
            context.minimum_version,
            __import__("ssl").TLSVersion.TLSv1_3,
        )
        self.assertEqual(
            context.maximum_version,
            __import__("ssl").TLSVersion.TLSv1_3,
        )
        connection_factory.assert_called_once_with(
            "example.com", "93.184.216.34", context
        )
        connection.request.assert_called_once_with(
            "GET",
            "/data",
            headers={
                "User-Agent": "Friday-Defensive-Connector/1.0",
                "Accept": "application/json",
            },
        )
        connection.close.assert_called_once()

    def test_url_scheme_credentials_ports_and_redirect_fragments_are_rejected(self):
        urls = [
            "http://example.com/",
            "https://user:pass@example.com/",
            "https://example.com:8443/",
            "https://example.com/#fragment",
            "https://example.com/path with space",
        ]
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(SecureAccessError):
                    validate_external_url(url, resolver=self.public_resolver)

    def test_headers_are_allowlisted_and_user_agent_is_standardized(self):
        sanitized = sanitize_outbound_headers({
            "Authorization": "Bearer sensitive",
            "X-Internal-Debug": "trace",
            "User-Agent": "framework/9.9",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self.assertEqual(
            sanitized,
            {
                "User-Agent": "Friday-Defensive-Connector/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    def test_logs_mask_secret_query_parameters_and_credentials(self):
        safe = redact_outbound_log(
            "https://example.com/path?token=secret-value&mode=read"
        )
        self.assertNotIn("secret-value", safe)
        self.assertIn("mode=read", safe)
        self.assertIn("[REDACTED]", safe)
        self.assertNotIn("Bearer verysecret", redact_outbound_log("Authorization: Bearer verysecret"))

    def test_authorization_log_redaction_removes_bearer_secret(self):
        authorization = redact_outbound_log("Authorization: Bearer verysecret")
        self.assertNotIn("verysecret", authorization)
        self.assertIn("[REDACTED]", authorization)

    def test_asset_registry_rejects_unrestricted_telemetry_networks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = AuthorizedAssetRegistry(Path(temp_dir) / "assets.json")
            asset = {
                "asset_id": "valid-asset",
                "asset_type": "server",
                "allowed_telemetry_sources": [],
                "permitted_defense_actions": [],
                "criticality": "LOW",
            }
            for unrestricted in ("0.0.0.0/0", "::/0"):
                with self.subTest(source=unrestricted):
                    with self.assertRaises(AssetRegistryError):
                        registry.register(
                            {**asset, "allowed_telemetry_sources": [unrestricted]},
                            ownership_verified=True,
                        )

    def test_asset_registry_requires_verified_ownership_and_validates_capabilities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = AuthorizedAssetRegistry(Path(temp_dir) / "assets.json")
            asset = {
                "asset_id": "web-api-prod",
                "asset_type": "application",
                "allowed_telemetry_sources": ["127.0.0.1", "Telemetry.Example.com"],
                "permitted_defense_actions": ["RATE_LIMIT_IP"],
                "criticality": "HIGH",
            }
            with self.assertRaises(AssetRegistryError):
                registry.register(asset, ownership_verified=False)
            saved = registry.register(asset, ownership_verified=True)
            self.assertTrue(saved["ownership_verified"])
            self.assertEqual(saved["allowed_telemetry_sources"], ["127.0.0.1", "telemetry.example.com"])
            self.assertTrue(registry.telemetry_source_allowed("web-api-prod", "127.0.0.1"))
            self.assertFalse(registry.telemetry_source_allowed("web-api-prod", "127.0.0.2"))
            self.assertEqual(
                AuthorizedAssetRegistry(Path(temp_dir) / "assets.json").require(
                    "web-api-prod"
                )["criticality"],
                "HIGH",
            )
            with self.assertRaises(AssetRegistryError):
                registry.require("unknown-asset")
            with self.assertRaises(AssetRegistryError):
                registry.register(
                    {**asset, "permitted_defense_actions": ["DELETE_DATABASE"]},
                    ownership_verified=True,
                )

    def test_registry_rejects_invalid_sources_ids_and_duplicate_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = AuthorizedAssetRegistry(Path(temp_dir) / "assets.json")
            base = {
                "asset_id": "valid-asset",
                "asset_type": "server",
                "allowed_telemetry_sources": [],
                "permitted_defense_actions": [],
                "criticality": "LOW",
            }
            with self.assertRaises(AssetRegistryError):
                registry.register({**base, "asset_id": "../../outside"}, ownership_verified=True)
            with self.assertRaises(AssetRegistryError):
                registry.register({**base, "allowed_telemetry_sources": ["bad host/path"]}, ownership_verified=True)
            registry.register(base, ownership_verified=True)
            with self.assertRaises(AssetRegistryError):
                registry.register(base, ownership_verified=True)


if __name__ == "__main__":
    unittest.main()
