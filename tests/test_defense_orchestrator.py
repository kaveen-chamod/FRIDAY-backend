from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app import access_control
from app.main import _DEFENSE_SUBMISSIONS, app
from app.defense.asset_registry import (
    AssetRegistryError,
    AuthorizedAssetRegistry,
)
from app.defense.forensics import ForensicsError, IncidentLedger
from app.defense.orchestrator import (
    DefenseActionError,
    DefenseOrchestrator,
)
from app.defense.threat_matrix import ThreatMatrix


class DefenseOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.registry = AuthorizedAssetRegistry(root / "assets.json")
        self.registry.register(
            {
                "asset_id": "app-prod",
                "asset_type": "application",
                "allowed_telemetry_sources": ["192.0.2.10"],
                "permitted_defense_actions": [
                    "RATE_LIMIT_IP",
                    "BLOCK_INDICATOR_TEMP",
                    "REVOKE_SUSPECT_SESSION",
                    "DISPATCH_SECURITY_ALERT",
                    "ELEVATE_AUDIT_LOGGING",
                ],
                "criticality": "MISSION_CRITICAL",
            },
            ownership_verified=True,
        )
        self.ledger = IncidentLedger(root / "incidents.jsonl")
        self.alerts: list[str] = []
        self.orchestrator = DefenseOrchestrator(
            self.registry,
            ThreatMatrix(self.registry),
            self.ledger,
            alert_callback=self.alerts.append,
        )

    async def asyncTearDown(self):
        await self.orchestrator.stop()
        self.temp_dir.cleanup()

    def test_rate_limit_and_temporary_block_apply_at_application_boundary(self):
        limited = self.orchestrator.apply_safe_action(
            "app-prod",
            "RATE_LIMIT_IP",
            "8.8.8.8",
            ttl_seconds=900,
        )
        self.assertEqual(limited["rollback_status"], "scheduled")
        self.assertEqual(self.orchestrator.check_request("8.8.8.8", now=100.0), None)
        status = None
        for _ in range(31):
            status = self.orchestrator.check_request("8.8.8.8", now=100.0)
        self.assertEqual(status, "rate_limited")
        self.assertEqual(
            self.orchestrator.apply_safe_action(
                "app-prod", "BLOCK_INDICATOR_TEMP", "1.1.1.1", ttl_seconds=300
            )["rollback_status"],
            "scheduled",
        )
        self.assertEqual(self.orchestrator.check_request("1.1.1.1"), "blocked")

    def test_session_revocation_hashes_token_and_is_reversible(self):
        token = "signed-speaker-token-secret"
        applied = self.orchestrator.apply_safe_action(
            "app-prod",
            "REVOKE_SUSPECT_SESSION",
            token,
            ttl_seconds=300,
        )
        self.assertTrue(self.orchestrator.is_indicator_revoked(token))
        self.assertNotIn(token, json.dumps(applied))
        rolled_back = self.orchestrator.rollback(applied["action_id"])
        self.assertEqual(rolled_back["rollback_status"], "rolled_back")
        self.assertFalse(self.orchestrator.is_indicator_revoked(token))

    async def test_ttl_daemon_rolls_back_expired_mitigations(self):
        applied = self.orchestrator.apply_safe_action(
            "app-prod", "BLOCK_INDICATOR_TEMP", "8.8.4.4", ttl_seconds=300
        )
        record = self.orchestrator._actions[applied["action_id"]]
        record["expires_monotonic"] = 0.0
        await self.orchestrator.start(poll_seconds=0.1)
        try:
            await asyncio.sleep(0.25)
            self.assertIsNone(self.orchestrator.check_request("8.8.4.4"))
            self.assertEqual(
                self.orchestrator._actions[applied["action_id"]]["rollback_status"],
                "rolled_back",
            )
        finally:
            await self.orchestrator.stop()

    def test_autonomous_high_impact_actions_are_never_executed(self):
        for action in (
            "ISOLATE_SERVER",
            "NETWORK_FIREWALL_CHANGE",
            "DELETE_DATABASE",
            "MODIFY_SECURITY_POLICY",
            "UNSEAL_SECRET",
        ):
            with self.subTest(action=action):
                with self.assertRaises(DefenseActionError):
                    self.orchestrator.apply_safe_action(
                        "app-prod", action, "8.8.8.8", autonomous=True
                    )
                result = self.orchestrator.manual_mitigate(
                    "app-prod",
                    action,
                    "8.8.8.8",
                    ttl_seconds=None,
                    master_authorized=False,
                )
                self.assertEqual(result["status"], "AWAITING_MASTER_AUTH")
                authorized = self.orchestrator.manual_mitigate(
                    "app-prod",
                    action,
                    "8.8.8.8",
                    ttl_seconds=None,
                    master_authorized=True,
                )
                self.assertEqual(authorized["status"], "AWAITING_MASTER_AUTH")
                self.assertIn("no change was made", authorized["message"])

    def test_asset_action_policy_and_indicators_are_enforced(self):
        with self.assertRaises(DefenseActionError):
            self.orchestrator.apply_safe_action(
                "app-prod", "RATE_LIMIT_IP", "not-an-ip"
            )
        with self.assertRaises(DefenseActionError):
            self.orchestrator.apply_safe_action(
                "app-prod", "BLOCK_INDICATOR_TEMP", "224.0.0.1"
            )
        with self.assertRaises(AssetRegistryError):
            self.orchestrator.apply_safe_action(
                "unknown", "RATE_LIMIT_IP", "8.8.8.8"
            )
        with self.assertRaises(DefenseActionError):
            self.orchestrator.apply_safe_action(
                "app-prod", "RATE_LIMIT_IP", "8.8.8.8", ttl_seconds=60
            )

    def test_audit_logging_elevation_is_scoped_and_reversible(self):
        import logging

        logger = logging.getLogger("friday.defense")
        original = logger.level
        applied = self.orchestrator.apply_safe_action(
            "app-prod",
            "ELEVATE_AUDIT_LOGGING",
            "incident-1",
            ttl_seconds=60,
        )
        self.assertEqual(logger.level, logging.DEBUG)
        self.orchestrator.rollback(applied["action_id"])
        self.assertEqual(logger.level, original)

    def test_threat_evaluation_executes_only_asset_permitted_safe_actions(self):
        events = []
        for index in range(5):
            result = self.orchestrator.evaluate_event({
                "target_asset": "app-prod",
                "category": "BRUTE_FORCE",
                "severity": "CRITICAL",
                "source": "192.0.2.10",
                "signal_id": f"login-fail-{index}",
                "indicator": "8.8.8.8",
                "evidence": ["password=do-not-store"],
            })
            events.append(result)
        result = events[-1]
        self.assertEqual(result["classification"], "CRITICAL_INCIDENT")
        self.assertTrue(
            all(item["action"] != "NETWORK_FIREWALL_CHANGE" or item["status"] == "AWAITING_MASTER_AUTH"
                for item in result["actions_taken"])
        )
        self.assertTrue(any("DISPATCH_SECURITY_ALERT" in item["action"] for item in result["actions_taken"]))
        self.assertTrue(self.alerts)
        incidents = self.ledger.list_incidents(limit=100)
        self.assertEqual(incidents["total"], 5)
        encoded = json.dumps(incidents)
        self.assertNotIn("do-not-store", encoded)
        self.assertTrue(self.ledger.verify())

    def test_corrupt_incident_ledger_prevents_any_mitigation(self):
        self.ledger.path.write_text("corrupt\n", encoding="utf-8")
        with self.assertRaises(ForensicsError):
            self.orchestrator.evaluate_event({
                "target_asset": "app-prod",
                "category": "BRUTE_FORCE",
                "severity": "CRITICAL",
                "source": "192.0.2.10",
                "signal_id": "corrupt-test",
                "indicator": "8.8.8.8",
            })
        self.assertIsNone(self.orchestrator.check_request("8.8.8.8"))

    def test_tampering_with_a_valid_ledger_record_is_detected(self):
        self.ledger.append({
            "incident_id": "tamper-test",
            "category": "AUTH_ANOMALY",
            "threat_score": 0.5,
            "target_asset": "app-prod",
            "evidence": ["login failure"],
            "actions_taken": [],
        })
        record = json.loads(self.ledger.path.read_text(encoding="utf-8"))
        record["category"] = "BRUTE_FORCE"
        self.ledger.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaises(ForensicsError):
            self.ledger.verify()

    def test_forensic_append_failure_rolls_back_new_autonomous_mitigations(self):
        for index in range(4):
            self.orchestrator.evaluate_event({
                "target_asset": "app-prod",
                "category": "BRUTE_FORCE",
                "severity": "LOW",
                "source": "192.0.2.10",
                "signal_id": f"forensic-{index}",
                "indicator": "8.8.8.8",
            })
        with patch.object(self.ledger, "append", side_effect=ForensicsError("ledger full")):
            with self.assertRaises(ForensicsError):
                self.orchestrator.evaluate_event({
                    "target_asset": "app-prod",
                    "category": "BRUTE_FORCE",
                    "severity": "CRITICAL",
                    "source": "192.0.2.10",
                    "signal_id": "forensic-final",
                    "indicator": "8.8.8.8",
                })
        self.assertIsNone(self.orchestrator.check_request("8.8.8.8"))
        self.assertEqual(self.orchestrator.active_status()["temporary_actions"], [])
        self.assertFalse(self.alerts)

    def test_active_metrics_redact_ip_indicators(self):
        self.orchestrator.apply_safe_action(
            "app-prod", "BLOCK_INDICATOR_TEMP", "8.8.8.8", ttl_seconds=30
        )
        active = self.orchestrator.active_status()
        self.assertEqual(len(active["active_indicators"]["blocked_ip_hashes"]), 1)
        self.assertNotIn("8.8.8.8", json.dumps(active))


class _MasterAgent:
    def _verified_speaker(self, token: str | None):
        if token != "master-token":
            return None
        return {
            "profile_id": "master-profile",
            "role": "MASTER",
            "enrolled_role": "MASTER",
        }


class DefenseApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_state = app.state._state.copy()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.registry = AuthorizedAssetRegistry(root / "assets.json")
        self.ledger = IncidentLedger(root / "incidents.jsonl")
        self.orchestrator = DefenseOrchestrator(
            self.registry, ThreatMatrix(self.registry), self.ledger
        )
        app.state.defense_assets = self.registry
        app.state.defense_incidents = self.ledger
        app.state.defense_orchestrator = self.orchestrator
        app.state.agent = _MasterAgent()
        app.state.cognitive_io_slots = asyncio.Semaphore(2)
        app.state._state.pop("succession", None)
        _DEFENSE_SUBMISSIONS.clear()
        self.api_token = "defense-api-test-" + "x" * 32
        self.token_patch = patch("app.main._API_TOKEN", self.api_token)
        def verify_passphrase(passphrase):
            if passphrase != "test-master-passphrase":
                raise access_control.MasterAuthError("Valid Master Passphrase verification is required.")

        self.master_patch = patch(
            "app.main.access_control.require_master_passphrase",
            side_effect=verify_passphrase,
        )
        self.token_patch.start()
        self.master_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        self.master_headers = {
            "Authorization": f"Bearer {self.api_token}",
            "X-Friday-Speaker-Token": "master-token",
            "X-Friday-Master-Passphrase": "test-master-passphrase",
        }

    async def asyncTearDown(self):
        await self.client.aclose()
        self.master_patch.stop()
        self.token_patch.stop()
        app.state._state.clear()
        app.state._state.update(self.previous_state)
        self.temp_dir.cleanup()

    async def test_master_authenticated_asset_management_and_admin_gate(self):
        denied = await self.client.post(
            "/api/defense/assets/register",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "X-Friday-Speaker-Token": "not-master",
            },
            json={
                "asset_id": "app-api",
                "asset_type": "application",
                "master_passphrase": "value",
            },
        )
        self.assertEqual(denied.status_code, 403)

        registered = await self.client.post(
            "/api/defense/assets/register",
            headers=self.master_headers,
            json={
                "asset_id": "app-api",
                "asset_type": "application",
                "allowed_telemetry_sources": ["127.0.0.1/32"],
                "permitted_defense_actions": ["RATE_LIMIT_IP"],
                "criticality": "HIGH",
                "master_passphrase": "test-master-passphrase",
                "ownership_verified": False,
            },
        )
        self.assertEqual(registered.status_code, 200)
        self.assertTrue(registered.json()["ownership_verified"])

        listed = await self.client.get(
            "/api/defense/assets", headers=self.master_headers
        )
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["assets"]), 1)

        unauthorized = await self.client.get(
            "/api/defense/assets",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "X-Friday-Speaker-Token": "master-token",
            },
        )
        self.assertEqual(unauthorized.status_code, 403)

    async def test_threat_ingestion_uses_actual_peer_and_master_metrics_routes(self):
        self.registry.register(
            {
                "asset_id": "app-api",
                "asset_type": "application",
                "allowed_telemetry_sources": ["127.0.0.1/32"],
                "permitted_defense_actions": ["RATE_LIMIT_IP"],
                "criticality": "HIGH",
            },
            ownership_verified=True,
        )
        headers = {"Authorization": f"Bearer {self.api_token}"}
        request_payload = {
            "target_asset": "app-api",
            "category": "AUTH_ANOMALY",
            "severity": "CRITICAL",
            "signal_id": "auth-1",
            "indicator": "8.8.8.8",
            "evidence": ["password=not-persisted"],
        }
        response = await self.client.post(
            "/api/defense/threats/evaluate", headers=headers, json=request_payload
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["classification"], "INFO")

        invalid_source = await self.client.post(
            "/api/defense/threats/evaluate",
            headers=headers,
            json={**request_payload, "target_asset": "not-registered"},
        )
        self.assertEqual(invalid_source.status_code, 403)

        active = await self.client.get(
            "/api/defense/threats/active", headers=self.master_headers
        )
        self.assertEqual(active.status_code, 200)
        incidents = await self.client.get(
            "/api/defense/incidents", headers=self.master_headers
        )
        self.assertEqual(incidents.status_code, 200)
        self.assertEqual(incidents.json()["total"], 1)
        self.assertNotIn("not-persisted", json.dumps(incidents.json()))

    async def test_high_impact_manual_operation_is_never_executed(self):
        self.registry.register(
            {
                "asset_id": "app-api",
                "asset_type": "application",
                "allowed_telemetry_sources": [],
                "permitted_defense_actions": [],
                "criticality": "HIGH",
            },
            ownership_verified=True,
        )
        response = await self.client.post(
            "/api/defense/mitigate",
            headers=self.master_headers,
            json={
                "target_asset": "app-api",
                "action": "NETWORK_FIREWALL_CHANGE",
                "indicator": "all",
                "master_passphrase": "test-master-passphrase",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "AWAITING_MASTER_AUTH")
        self.assertIn("no change was made", response.json()["message"])


if __name__ == "__main__":
    unittest.main()
