from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.defense.asset_registry import AuthorizedAssetRegistry
from app.defense.threat_matrix import (
    THREAT_CATEGORIES,
    ThreatEvaluationError,
    ThreatMatrix,
)


class ThreatMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = AuthorizedAssetRegistry(Path(self.temp_dir.name) / "assets.json")
        self.registry.register(
            {
                "asset_id": "web-api-prod",
                "asset_type": "application",
                "allowed_telemetry_sources": ["192.0.2.10/32"],
                "permitted_defense_actions": [
                    "RATE_LIMIT_IP",
                    "BLOCK_INDICATOR_TEMP",
                    "DISPATCH_SECURITY_ALERT",
                ],
                "criticality": "HIGH",
            },
            ownership_verified=True,
        )
        self.matrix = ThreatMatrix(self.registry)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _event(self, category: str, signal_id: str = "signal-1", **kwargs):
        return {
            "target_asset": "web-api-prod",
            "category": category,
            "severity": "HIGH",
            "source": "192.0.2.10",
            "signal_id": signal_id,
            **kwargs,
        }

    def test_all_six_categories_are_scored_for_authorized_assets(self):
        self.assertEqual(len(THREAT_CATEGORIES), 6)
        for category in sorted(THREAT_CATEGORIES):
            with self.subTest(category=category):
                matrix = ThreatMatrix(self.registry)
                result = matrix.evaluate(self._event(category))
                self.assertEqual(result["category"], category)
                self.assertEqual(result["classification"], "INFO")
                self.assertAlmostEqual(result["threat_score"], 0.8 * 0.24 * 1.0)
                self.assertGreaterEqual(result["confidence"], 0.0)
                self.assertLessEqual(result["confidence"], 1.0)

    def test_correlation_increases_confidence_score_and_severity_classification(self):
        result = None
        for index in range(5):
            result = self.matrix.evaluate(
                self._event(
                    "BRUTE_FORCE",
                    signal_id=f"signal-{index}",
                )
            )
        self.assertEqual(result["correlated_event_count"], 5)
        self.assertEqual(result["confidence"], 0.72)
        self.assertEqual(result["classification"], "GUARDED")
        self.assertEqual(result["safe_auto_actions"], ["RATE_LIMIT_IP"])

    def test_mission_critical_asset_escalates_to_critical_incident(self):
        self.registry.register(
            {
                "asset_id": "control-plane",
                "asset_type": "server",
                "allowed_telemetry_sources": ["192.0.2.10"],
                "permitted_defense_actions": [
                    "RATE_LIMIT_IP",
                    "BLOCK_INDICATOR_TEMP",
                    "DISPATCH_SECURITY_ALERT",
                ],
                "criticality": "MISSION_CRITICAL",
            },
            ownership_verified=True,
        )
        result = None
        for index in range(5):
            result = self.matrix.evaluate({
                **self._event(
                    "PRIVILEGE_ESCALATION",
                    signal_id=f"priv-{index}",
                ),
                "target_asset": "control-plane",
                "severity": "CRITICAL",
            })
        self.assertEqual(result["classification"], "CRITICAL_INCIDENT")
        self.assertEqual(result["threat_score"], 0.9)
        self.assertTrue(result["master_alert_required"])
        self.assertTrue(result["master_action_required"])
        self.assertEqual(
            result["safe_auto_actions"],
            ["RATE_LIMIT_IP", "BLOCK_INDICATOR_TEMP", "DISPATCH_SECURITY_ALERT"],
        )

    def test_false_positive_filter_suppresses_automatic_mitigation(self):
        result = self.matrix.evaluate(
            self._event("API_ABUSE", false_positive=True)
        )
        self.assertEqual(result["classification"], "INFO")
        self.assertEqual(result["safe_auto_actions"], [])
        self.assertTrue(result["false_positive_filtered"])

    def test_unauthorized_assets_and_telemetry_sources_fail_closed(self):
        with self.assertRaises(ThreatEvaluationError):
            self.matrix.evaluate({
                **self._event("AUTH_ANOMALY"),
                "target_asset": "not-registered",
            })
        with self.assertRaises(ThreatEvaluationError):
            self.matrix.evaluate({
                **self._event("AUTH_ANOMALY"),
                "source": "192.0.2.11",
            })

    def test_invalid_category_severity_and_source_are_rejected(self):
        for change in (
            {"category": "UNSUPPORTED"},
            {"severity": "EXTREME"},
            {"source": ""},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ThreatEvaluationError):
                    self.matrix.evaluate({**self._event("AUTH_ANOMALY"), **change})


if __name__ == "__main__":
    unittest.main()
