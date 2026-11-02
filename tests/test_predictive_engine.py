from __future__ import annotations

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from app.predictive_engine import (
    SimulationCapacityError,
    SimulationInputError,
    SimulationJobManager,
    normalize_action_spec,
    run_prediction,
    wilson_interval,
)


def _action_spec(**overrides):
    spec = {
        "action": "sample action",
        "input_distributions": {
            "battery_voltage": {"type": "uniform", "low": 3.8, "high": 4.2},
        },
        "success_conditions": [
            {"variable": "battery_voltage", "operator": ">=", "value": 3.3},
        ],
        "success_probability": 0.99,
        "catastrophic_probability": 0.0,
        "epistemic_uncertainty": 0.05,
        "model_source": "test fixture",
        "model_version": "1",
    }
    spec.update(overrides)
    return spec


class PredictiveEngineTests(unittest.TestCase):
    def test_seed_makes_estimates_reproducible(self):
        first = run_prediction(_action_spec(), profile="QUICK", seed=17)
        second = run_prediction(_action_spec(), profile="QUICK", seed=17)
        for key in (
            "sample_size",
            "observed_failure_count",
            "observed_catastrophic_count",
            "success_probability",
            "failure_probability",
            "catastrophic_risk_score",
            "recommendation",
        ):
            self.assertEqual(first[key], second[key])

    def test_quick_profile_starts_at_default_sample_budget(self):
        result = run_prediction(_action_spec(), profile="QUICK", seed=9)
        self.assertEqual(result["sample_size"], 1_000)
        self.assertGreaterEqual(result["success_probability"], 0.85)
        self.assertLessEqual(result["failure_probability"], 1.0)
        self.assertIn("confidence_intervals_95", result)
        self.assertEqual(
            result["probability_basis"],
            "Monte Carlo estimate under caller-supplied distributions; not a measured real-world probability.",
        )

    def test_initial_budget_uses_four_bounded_batches_of_250(self):
        from app import predictive_engine

        original = predictive_engine._simulate_batch
        batch_sizes = []

        def capture_batch(spec, seed, sample_count):
            batch_sizes.append(sample_count)
            return original(spec, seed, sample_count)

        with patch(
            "app.predictive_engine._simulate_batch",
            side_effect=capture_batch,
        ):
            run_prediction(_action_spec(), profile="QUICK", seed=5)
        self.assertEqual(batch_sizes, [250, 250, 250, 250])

    def test_adaptive_budget_reports_exhaustion_instead_of_claiming_convergence(self):
        result = run_prediction(
            _action_spec(
                success_probability=0.5,
                epistemic_uncertainty=0.5,
            ),
            profile="STANDARD",
            seed=2,
            max_iterations=1_000,
        )
        self.assertEqual(result["sample_size"], 1_000)
        self.assertFalse(result["sampling_converged"])
        self.assertTrue(any("budget was exhausted" in warning for warning in result["warnings"]))

    def test_recommendation_matrix_uses_model_confidence_and_risk(self):
        recommended = run_prediction(
            _action_spec(
                success_probability=1.0,
                epistemic_uncertainty=0.01,
            ),
            profile="QUICK",
            seed=4,
        )
        discouraged = run_prediction(
            _action_spec(
                success_probability=0.4,
                epistemic_uncertainty=0.01,
            ),
            profile="QUICK",
            seed=4,
        )
        blocked = run_prediction(
            _action_spec(
                success_probability=0.99,
                catastrophic_probability=1.0,
            ),
            profile="QUICK",
            seed=4,
        )
        unknown_model = run_prediction(
            _action_spec(epistemic_uncertainty=None),
            profile="QUICK",
            seed=4,
        )
        self.assertEqual(recommended["recommendation"], "RECOMMENDED")
        self.assertEqual(discouraged["recommendation"], "STRONGLY_DISCOURAGED")
        self.assertEqual(blocked["recommendation"], "BLOCKED")
        self.assertEqual(unknown_model["recommendation"], "PROCEED_WITH_CAUTION")

    def test_failure_modes_are_ranked_by_observed_frequency(self):
        result = run_prediction(
            _action_spec(
                success_probability=1.0,
                failure_modes=[
                    {"name": "rare", "probability": 0.05},
                    {"name": "common", "probability": 0.4},
                ],
            ),
            profile="QUICK",
            seed=21,
        )
        self.assertEqual(result["failure_modes"][0]["name"], "common")
        self.assertGreater(
            result["failure_modes"][0]["observed_count"],
            result["failure_modes"][1]["observed_count"],
        )
        self.assertIn(
            "confidence_interval_95",
            result["failure_modes"][0],
        )

    def test_distribution_sampling_and_threshold_conditions(self):
        result = run_prediction(
            _action_spec(
                input_distributions={
                    "obstacle_distance": {
                        "type": "uniform",
                        "low": 0.0,
                        "high": 2.0,
                    },
                },
                success_conditions=[
                    {"variable": "obstacle_distance", "operator": ">", "value": 1.0},
                ],
                success_probability=1.0,
            ),
            profile="QUICK",
            seed=12,
        )
        self.assertGreater(result["failure_probability"], 0.35)
        self.assertLess(result["failure_probability"], 0.65)

    def test_wilson_interval_handles_zero_and_all_observed_events(self):
        low, high = wilson_interval(0, 1_000)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = wilson_interval(1_000, 1_000)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_rejects_invalid_probability_distribution_and_sample_cap(self):
        with self.assertRaises(SimulationInputError):
            normalize_action_spec(_action_spec(success_probability=1.1))
        with self.assertRaises(SimulationInputError):
            normalize_action_spec(
                _action_spec(
                    input_distributions={"x": {"type": "uniform", "low": 2, "high": 1}}
                )
            )
        with self.assertRaises(SimulationInputError):
            normalize_action_spec(
                _action_spec(
                    success_conditions=[
                        {"variable": "missing", "operator": ">", "value": 0}
                    ]
                )
            )
        with self.assertRaises(SimulationInputError):
            run_prediction(_action_spec(), max_iterations=10_001)
        with self.assertRaises(SimulationInputError):
            run_prediction(_action_spec(), max_iterations=999)

    def test_zero_probability_failure_mode_requires_a_trigger(self):
        with self.assertRaises(SimulationInputError):
            normalize_action_spec(
                _action_spec(failure_modes=[{"name": "impossible", "probability": 0.0}])
            )

    def test_failure_condition_without_matches_does_not_fail_every_sample(self):
        result = run_prediction(
            _action_spec(
                input_distributions={"voltage": 5.0},
                success_conditions=[],
                success_probability=1.0,
                failure_modes=[{
                    "name": "undervoltage",
                    "probability": 0.0,
                    "conditions": [
                        {"variable": "voltage", "operator": "<", "value": 3.0}
                    ],
                }],
            ),
            profile="QUICK",
            seed=10,
        )
        self.assertEqual(result["failure_probability"], 0.0)
        self.assertEqual(result["failure_modes"], [])


class SimulationJobManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_is_bounded_and_shutdown_cancels_waiting_jobs(self):
        def delayed_batch(spec, seed, sample_count):
            import time
            assert spec["action"] and seed >= 0
            time.sleep(0.2)
            return {
                "samples": sample_count,
                "successes": sample_count,
                "catastrophes": 0,
                "failure_modes": {},
                "cpu_seconds": 0.0,
                "rss_bytes": None,
            }

        manager = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=1,
            max_iterations=1_000,
            executor_factory=ThreadPoolExecutor,
        )
        with patch("app.predictive_engine._simulate_batch", side_effect=delayed_batch):
            await manager.start()
            first = await manager.submit(_action_spec(), profile="QUICK", seed=1)
            await asyncio.sleep(0.02)
            second = await manager.submit(_action_spec(), profile="QUICK", seed=2)
            with self.assertRaises(SimulationCapacityError):
                await manager.submit(_action_spec(), profile="QUICK", seed=3)
            await manager.shutdown()
        self.assertEqual((await manager.get_job(first["job_id"]))["status"], "cancelled")
        self.assertEqual((await manager.get_job(second["job_id"]))["status"], "cancelled")

    async def test_idempotency_key_reuses_job_and_rejects_different_request(self):
        manager = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=2,
            max_iterations=1_000,
        )
        await manager.start()
        try:
            first = await manager.submit(
                _action_spec(), profile="QUICK", idempotency_key="request-1"
            )
            repeated = await manager.submit(
                _action_spec(), profile="QUICK", idempotency_key="request-1"
            )
            self.assertEqual(first["job_id"], repeated["job_id"])
            with self.assertRaises(SimulationInputError):
                await manager.submit(
                    _action_spec(success_probability=0.1),
                    profile="QUICK",
                    idempotency_key="request-1",
                )
        finally:
            await manager.shutdown()

    async def test_process_pool_job_returns_result_and_resource_metrics(self):
        manager = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=2,
            timeout_seconds=10,
            max_iterations=1_000,
        )
        await manager.start()
        try:
            submitted = await manager.submit(
                _action_spec(), profile="QUICK", seed=31
            )
            for _ in range(200):
                job = await manager.get_job(submitted["job_id"])
                if job["status"] in {"completed", "failed", "timed_out"}:
                    break
                await asyncio.sleep(0.025)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["result"]["sample_size"], 1_000)
            snapshot = await manager.resource_snapshot()
            self.assertIn("process_rss_bytes", snapshot)
            self.assertEqual(snapshot["completed_samples"], 1_000)
        finally:
            await manager.shutdown()

    async def test_timeout_returns_bounded_partial_result(self):
        def delayed_batch(spec, seed, sample_count):
            import time
            assert spec["action"] and seed >= 0
            time.sleep(0.2)
            return {
                "samples": sample_count,
                "successes": sample_count,
                "catastrophes": 0,
                "failure_modes": {},
                "cpu_seconds": 0.0,
                "rss_bytes": None,
            }

        manager = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=1,
            timeout_seconds=0.1,
            max_iterations=1_000,
            executor_factory=ThreadPoolExecutor,
        )
        with patch("app.predictive_engine._simulate_batch", side_effect=delayed_batch):
            await manager.start()
            try:
                submitted = await manager.submit(
                    _action_spec(), profile="QUICK", seed=6
                )
                for _ in range(100):
                    job = await manager.get_job(submitted["job_id"])
                    if job["status"] == "timed_out":
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(job["status"], "timed_out")
                self.assertTrue(job["result"]["execution"]["timed_out"])
                self.assertEqual(job["result"]["sample_size"], 0)
            finally:
                await manager.shutdown()

    async def test_worker_memory_limit_fails_with_partial_diagnostics(self):
        manager = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=1,
            max_iterations=1_000,
            max_memory_bytes=256 * 1024 * 1024,
            executor_factory=ThreadPoolExecutor,
        )

        def over_limit_batch(spec, seed, sample_count):
            assert spec["action"] and seed >= 0
            return {
                "samples": sample_count,
                "successes": sample_count,
                "catastrophes": 0,
                "failure_modes": {},
                "cpu_seconds": 0.01,
                "rss_bytes": 512 * 1024 * 1024,
            }

        with patch(
            "app.predictive_engine._simulate_batch",
            side_effect=over_limit_batch,
        ):
            await manager.start()
            try:
                submitted = await manager.submit(
                    _action_spec(), profile="QUICK", seed=77
                )
                for _ in range(100):
                    job = await manager.get_job(submitted["job_id"])
                    if job["status"] in {"failed", "completed"}:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(job["status"], "failed")
                self.assertEqual(job["samples_completed"], 250)
                self.assertEqual(job["result"]["sample_size"], 250)
                self.assertTrue(any("memory limit" in warning for warning in job["result"]["warnings"]))
            finally:
                await manager.shutdown()

    async def test_running_job_cancellation_finishes_at_batch_boundary(self):
        manager = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=1,
            max_iterations=1_000,
            executor_factory=ThreadPoolExecutor,
        )
        def delayed_batch(spec, seed, sample_count):
            import time
            assert spec["action"] and seed >= 0
            time.sleep(0.1)
            return {
                "samples": sample_count,
                "successes": sample_count,
                "catastrophes": 0,
                "failure_modes": {},
                "cpu_seconds": 0.0,
                "rss_bytes": None,
            }

        with patch("app.predictive_engine._simulate_batch", side_effect=delayed_batch):
            await manager.start()
            try:
                submitted = await manager.submit(
                    _action_spec(), profile="QUICK", seed=44
                )
                await asyncio.sleep(0.02)
                await manager.cancel(submitted["job_id"])
                for _ in range(200):
                    job = await manager.get_job(submitted["job_id"])
                    if job["status"] == "cancelled":
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(job["status"], "cancelled")
            finally:
                await manager.shutdown()


if __name__ == "__main__":
    unittest.main()
