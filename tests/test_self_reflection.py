from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from app.main import _REFLECTION_SUBMISSIONS, app
from app.predictive_engine import SimulationJobManager
from app.self_reflection import (
    ReflectionInputError,
    ReflectionStorageError,
    SelfReflectionLedger,
)


def _prediction(prediction_id: str, probability: float = 0.6) -> dict:
    return {
        "prediction_id": prediction_id,
        "action": "example action",
        "profile": "STANDARD",
        "seed": 12,
        "sample_size": 1_000,
        "model": {"source": "unit test", "version": "1"},
        "result": {
            "success_probability": probability,
            "failure_probability": 1.0 - probability,
            "catastrophic_risk_score": 0.01,
        },
    }


class SelfReflectionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "reflection_ledger.jsonl"
        self.ledger = SelfReflectionLedger(
            self.ledger_path,
            outcome_verifier=lambda _prediction_id, _actual, reference: (
                reference.startswith("independent:")
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_simulation_is_logged_as_hypothetical_and_sensitive_text_is_redacted(self):
        event = _prediction("simulation-1")
        event["action"] = "send token sk-abcdefghijklmnopqrstuvwxyz012345"
        calibration = self.ledger.record_simulation_prediction(event)
        self.assertEqual(calibration["version"], 1)
        self.assertTrue(calibration["does_not_change_recommendation"])
        record = json.loads(self.ledger_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["source"], "hypothetical_simulation")
        self.assertEqual(record["verification_status"], "not_real_world_evidence")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz012345", record["action"])
        self.assertEqual(record["predicted_probabilities"]["success"], 0.6)

    def test_unverified_feedback_and_sensor_observations_do_not_calibrate(self):
        self.ledger.record_simulation_prediction(_prediction("simulation-1"))
        feedback = self.ledger.record_user_feedback("simulation-1", True)
        sensor = self.ledger.record_sensor_observation(
            "simulation-1", False, "device-reading-1"
        )
        report = self.ledger.report()
        self.assertEqual(feedback["source"], "user_feedback")
        self.assertEqual(sensor["source"], "sensor_observation")
        self.assertEqual(report["outcome_counts_by_source"]["user_feedback"], 1)
        self.assertEqual(report["outcome_counts_by_source"]["sensor_observation"], 1)
        self.assertEqual(report["independently_verified_count"], 0)
        self.assertIsNone(report["brier_score"])
        self.assertEqual(report["calibration"]["version"], 1)

    def test_brier_score_and_calibration_drift_use_verified_outcomes(self):
        prediction_id = "brier-example"
        self.ledger.record_simulation_prediction(
            _prediction(prediction_id, probability=0.6)
        )
        self.ledger._record_verified_outcome(
            prediction_id, True, "independent:brier-evidence"
        )
        report = self.ledger.report()
        self.assertAlmostEqual(report["brier_score"], 0.16)
        self.assertAlmostEqual(report["calibration_drift"], 0.4)

    def test_verified_outcomes_produce_brier_score_and_holdout_validated_calibration(self):
        activated = []
        for index in range(80):
            prediction_id = f"verified-{index}"
            self.ledger.record_simulation_prediction(_prediction(prediction_id))
            result = self.ledger._record_verified_outcome(
                prediction_id, True, f"independent:external-verification-{index}"
            )
            if result["calibration_update"] and result["calibration_update"]["status"] == "activated":
                activated.append(result["calibration_update"])

        report = self.ledger.report()
        self.assertTrue(activated)
        self.assertGreater(report["calibration"]["version"], 1)
        self.assertGreater(report["calibration"]["success_probability_bias"], 0.0)
        self.assertLessEqual(
            report["calibration"]["success_probability_bias"], 0.20
        )
        history = report["calibration"]["history"]
        for previous, current in zip(history, history[1:]):
            self.assertLessEqual(
                abs(current["bias"] - previous["bias"]), 0.05 + 1e-12
            )
        self.assertEqual(report["independently_verified_count"], 80)
        self.assertIsNotNone(report["brier_score"])
        self.assertEqual(
            report["calibration"]["applies_to"],
            "supplemental probability estimate only",
        )
        self.assertFalse(report["calibration"]["changes_authorization_or_safety_limits"])
        future = self.ledger.record_simulation_prediction(_prediction("future-estimate"))
        self.assertGreater(future["calibrated_success_probability"], 0.6)
        self.assertTrue(future["does_not_change_recommendation"])

    def test_holdout_regression_rejects_calibration_candidate(self):
        ids = [f"holdout-{index}" for index in range(80)]
        holdout_ids = {
            prediction_id
            for prediction_id in ids
            if int(hashlib.sha256(prediction_id.encode()).hexdigest()[:8], 16) % 5 == 0
        }
        rejected = []
        for prediction_id in ids:
            self.ledger.record_simulation_prediction(
                _prediction(prediction_id, probability=0.5)
            )
            result = self.ledger._record_verified_outcome(
                prediction_id,
                prediction_id not in holdout_ids,
                f"independent:{prediction_id}",
            )
            update = result["calibration_update"]
            if update and update["status"] == "rejected":
                rejected.append(update)
        self.assertTrue(rejected)
        self.assertEqual(self.ledger.report()["calibration"]["version"], 1)
        self.assertTrue(
            all(
                item["candidate_holdout_brier"] > item["current_holdout_brier"]
                for item in rejected
            )
        )

    def test_insufficient_verified_data_does_not_change_calibration(self):
        for index in range(8):
            prediction_id = f"few-{index}"
            self.ledger.record_simulation_prediction(_prediction(prediction_id))
            self.ledger._record_verified_outcome(
                prediction_id, True, f"independent:verified-{index}"
            )
        report = self.ledger.report()
        self.assertEqual(report["calibration"]["version"], 1)
        self.assertEqual(report["calibration"]["success_probability_bias"], 0.0)

    def test_calibration_rollback_is_versioned_and_survives_reload(self):
        for index in range(80):
            prediction_id = f"rollback-{index}"
            self.ledger.record_simulation_prediction(_prediction(prediction_id))
            self.ledger._record_verified_outcome(
                prediction_id, True, f"independent:verified-{index}"
            )
        before = self.ledger.report()["calibration"]
        self.assertGreater(before["version"], 1)
        rolled_back = self.ledger._rollback_calibration(1)
        self.assertEqual(rolled_back["bias"], 0.0)
        self.assertGreater(rolled_back["version"], before["version"])
        reloaded = SelfReflectionLedger(self.ledger_path).report()["calibration"]
        self.assertEqual(reloaded["version"], rolled_back["version"])
        self.assertEqual(reloaded["success_probability_bias"], 0.0)

    def test_verified_outcomes_require_a_configured_trusted_verifier(self):
        prediction_id = "unverified-internal"
        ledger = SelfReflectionLedger(Path(self.temp_dir.name) / "no-verifier.jsonl")
        ledger.record_simulation_prediction(_prediction(prediction_id))
        with self.assertRaisesRegex(ReflectionInputError, "No trusted"):
            ledger._record_verified_outcome(
                prediction_id, True, "independent:but-unverified"
            )

    def test_duplicate_prediction_is_idempotent_but_conflicting_reuse_is_rejected(self):
        self.ledger.record_simulation_prediction(_prediction("same-id"))
        self.ledger.record_simulation_prediction(_prediction("same-id"))
        with self.assertRaises(ReflectionInputError):
            self.ledger.record_simulation_prediction(
                _prediction("same-id", probability=0.8)
            )
        self.assertEqual(self.ledger.report()["prediction_count"], 1)

    def test_user_feedback_cannot_claim_independent_verification(self):
        self.ledger.record_simulation_prediction(_prediction("feedback-only"))
        result = self.ledger.record_user_feedback("feedback-only", True)
        self.assertEqual(result["verification_status"], "unverified")
        self.assertNotIn("record_verified_outcome", dir(self.ledger))

    def test_corrupt_or_over_limit_ledger_fails_explicitly(self):
        self.ledger_path.write_text("{bad json}\n", encoding="utf-8")
        with self.assertRaises(ReflectionStorageError):
            self.ledger.report()
        oversized = Path(self.temp_dir.name) / "oversized.jsonl"
        oversized.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
        with self.assertRaises(ReflectionStorageError):
            SelfReflectionLedger(oversized).report()


class ReflectionManagerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_simulations_are_recorded_and_calibrated_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = SelfReflectionLedger(Path(temp_dir) / "ledger.jsonl")
            service = SimulationJobManager(
                max_concurrent_jobs=1,
                max_queued_jobs=1,
                max_iterations=1_000,
                executor_factory=ThreadPoolExecutor,
                on_completed=ledger.record_simulation_prediction,
            )
            await service.start()
            try:
                submitted = await service.submit(
                    {
                        "action": "integration simulation",
                        "success_probability": 1.0,
                        "catastrophic_probability": 0.0,
                        "epistemic_uncertainty": 0.01,
                        "model_source": "test",
                        "model_version": "1",
                    },
                    profile="QUICK",
                    seed=17,
                )
                result = None
                for _ in range(100):
                    result = await service.get_job(submitted["job_id"])
                    if result["status"] == "completed":
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(result["status"], "completed")
                self.assertIn("reflection_calibration", result["result"])
                self.assertTrue(
                    result["result"]["reflection_calibration"][
                        "does_not_change_recommendation"
                    ]
                )
                report = await asyncio.to_thread(ledger.report)
                self.assertEqual(report["prediction_count"], 1)
                self.assertEqual(report["independently_verified_count"], 0)
            finally:
                await service.shutdown()

    async def test_reflection_callback_failure_does_not_fail_simulation(self):
        def fail(_event):
            raise OSError("test ledger failure")

        service = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=1,
            max_iterations=1_000,
            executor_factory=ThreadPoolExecutor,
            on_completed=fail,
        )
        await service.start()
        try:
            with self.assertLogs("friday.predictive", level="ERROR") as captured:
                submitted = await service.submit(
                    {
                        "action": "callback failure",
                        "success_probability": 1.0,
                        "catastrophic_probability": 0.0,
                        "epistemic_uncertainty": 0.01,
                        "model_source": "test",
                        "model_version": "1",
                    },
                    profile="QUICK",
                    seed=1,
                )
                result = None
                for _ in range(100):
                    result = await service.get_job(submitted["job_id"])
                    if result["status"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.01)
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("reflection_calibration", result["result"])
            self.assertTrue(
                any("Could not record completed simulation" in item for item in captured.output)
            )
        finally:
            await service.shutdown()


class ReflectionApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_state = app.state._state.copy()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger = SelfReflectionLedger(
            Path(self.temp_dir.name) / "reflection_ledger.jsonl"
        )
        app.state.reflection_ledger = self.ledger
        app.state.cognitive_io_slots = asyncio.Semaphore(2)
        app.state._state.pop("succession", None)
        _REFLECTION_SUBMISSIONS.clear()
        self.api_token = "reflection-test-token-" + "x" * 32
        self.token_patch = patch("app.main._API_TOKEN", self.api_token)
        self.token_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.token_patch.stop()
        app.state._state.clear()
        app.state._state.update(self.previous_state)
        self.temp_dir.cleanup()

    async def test_feedback_api_records_only_unverified_user_feedback(self):
        unauthenticated = await self.client.get("/api/reflection/report")
        self.assertEqual(unauthenticated.status_code, 401)
        missing_prediction = await self.client.post(
            "/api/reflection/outcomes",
            headers=self.headers,
            json={"prediction_id": "unknown", "actual_success": True},
        )
        self.assertEqual(missing_prediction.status_code, 422)

        self.ledger.record_simulation_prediction(_prediction("api-prediction"))
        feedback = await self.client.post(
            "/api/reflection/outcomes",
            headers=self.headers,
            json={
                "prediction_id": "api-prediction",
                "actual_success": True,
                "source": "independently_verified",
                "verification_status": "verified",
            },
        )
        self.assertEqual(feedback.status_code, 200)
        self.assertEqual(feedback.json()["source"], "user_feedback")
        report = await self.client.get("/api/reflection/report", headers=self.headers)
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["independently_verified_count"], 0)
        self.assertIsNone(report.json()["brier_score"])

    async def test_report_work_is_off_event_loop(self):
        original_report = self.ledger.report

        def slow_report():
            time.sleep(0.12)
            return original_report()

        with patch.object(self.ledger, "report", side_effect=slow_report):
            ticks = 0

            async def ticker():
                nonlocal ticks
                for _ in range(5):
                    await asyncio.sleep(0.015)
                    ticks += 1

            response, _ = await asyncio.gather(
                self.client.get("/api/reflection/report", headers=self.headers),
                ticker(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ticks, 5)

    async def test_reflection_api_is_rate_limited(self):
        responses = []
        for _ in range(31):
            responses.append(
                await self.client.get("/api/reflection/report", headers=self.headers)
            )
        self.assertTrue(all(response.status_code == 200 for response in responses[:30]))
        self.assertEqual(responses[30].status_code, 429)


if __name__ == "__main__":
    unittest.main()
