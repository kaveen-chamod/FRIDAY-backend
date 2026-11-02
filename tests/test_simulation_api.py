from __future__ import annotations

import asyncio
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from app.digital_twin import DigitalTwinEngine
from app.main import app
from app.predictive_engine import SimulationJobManager


_ACTION_SPEC = {
    "action": "API test simulation",
    "success_probability": 1.0,
    "catastrophic_probability": 0.0,
    "epistemic_uncertainty": 0.01,
    "model_source": "API test",
    "model_version": "1",
}


class SimulationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_state = app.state._state.copy()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=2,
            max_iterations=1_000,
            executor_factory=ThreadPoolExecutor,
        )
        await self.service.start()
        app.state.simulation_service = self.service
        app.state.digital_twin = DigitalTwinEngine(
            state_path=Path(self.temp_dir.name) / "twins.json"
        )
        app.state._state.pop("succession", None)
        from app.main import _SIMULATION_SUBMISSIONS

        _SIMULATION_SUBMISSIONS.clear()
        self.token_patch = patch("app.main._API_TOKEN", "test-token-" + "x" * 40)
        self.token_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.service.shutdown()
        self.token_patch.stop()
        app.state._state.clear()
        app.state._state.update(self.previous_state)
        self.temp_dir.cleanup()

    async def test_authenticated_predict_returns_job_then_result(self):
        response = await self.client.post(
            "/api/simulation/predict",
            headers={"Authorization": "Bearer " + "test-token-" + "x" * 40},
            json={"action_spec": _ACTION_SPEC, "profile": "QUICK", "seed": 17},
        )
        self.assertEqual(response.status_code, 202)
        submission = response.json()
        self.assertEqual(submission["status"], "queued")
        self.assertEqual(len(submission["job_id"]), 32)

        result_response = None
        for _ in range(100):
            result_response = await self.client.get(
                f"/api/simulation/jobs/{submission['job_id']}",
                headers={"Authorization": "Bearer " + "test-token-" + "x" * 40},
            )
            if result_response.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        self.assertIsNotNone(result_response)
        result = result_response.json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["sample_size"], 1_000)

    async def test_predict_requires_api_authentication(self):
        response = await self.client.post(
            "/api/simulation/predict",
            json={"action_spec": _ACTION_SPEC},
        )
        self.assertEqual(response.status_code, 401)

    async def test_simulation_request_body_has_a_hard_size_limit(self):
        response = await self.client.post(
            "/api/simulation/predict",
            headers={"Authorization": "Bearer " + "test-token-" + "x" * 40},
            json={
                "action_spec": {
                    "action": "oversized",
                    "padding": "x" * (70 * 1024),
                }
            },
        )
        self.assertEqual(response.status_code, 413)

    async def test_streamed_simulation_request_body_has_a_hard_size_limit(self):
        async def oversized_body():
            yield b'{"action_spec":{"action":"oversized","padding":"'
            yield b"x" * (70 * 1024)
            yield b'"}}'

        response = await self.client.post(
            "/api/simulation/predict",
            headers={
                "Authorization": "Bearer " + "test-token-" + "x" * 40,
                "Content-Type": "application/json",
            },
            content=oversized_body(),
        )
        self.assertEqual(response.status_code, 413)

    async def test_twin_endpoint_reports_unsupported_without_fabricated_result(self):
        response = await self.client.post(
            "/api/twin/simulate",
            headers={"Authorization": "Bearer " + "test-token-" + "x" * 40},
            json={
                "device_id": "unconfigured",
                "command": {"action": "move", "direction": "FORWARD"},
            },
        )
        self.assertEqual(response.status_code, 202)
        result = response.json()
        self.assertEqual(result["status"], "NOT_SUPPORTED")
        self.assertFalse(result["dispatch_allowed"])
        self.assertNotIn("result", result)


if __name__ == "__main__":
    unittest.main()
