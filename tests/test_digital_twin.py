from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from actions.iot_controller import control_iot
from app.digital_twin import DigitalTwinEngine, DigitalTwinError
from app.predictive_engine import SimulationJobManager


def _action_spec() -> dict:
    return {
        "action": "validated test model",
        "success_probability": 1.0,
        "catastrophic_probability": 0.0,
        "epistemic_uncertainty": 0.01,
        "model_source": "unit-test adapter",
        "model_version": "1",
    }


class _TestAdapter:
    domain = "test_device"
    model_version = "1"
    validated = True

    def validate_device(self, specification, state):
        return (
            specification.get("device_type") == "test"
            and isinstance(state.get("value"), int)
        )

    def build_action_spec(self, specification, state, command):
        return _action_spec()

    def allow_dispatch(self, result):
        return result.get("recommendation") == "RECOMMENDED"

    def build_text_request(self, text):
        return None


def _iso_timestamp(age_seconds: float = 0.0) -> str:
    value = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return value.isoformat()


class DigitalTwinTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temp_dir.name) / "twins.json"
        self.adapter = _TestAdapter()
        self.engine = DigitalTwinEngine(
            state_path=self.state_path,
            adapters=(self.adapter,),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _register_device(self, *, age_seconds: float = 0.0):
        self.engine.register_device(
            "test-1",
            "test_device",
            {"device_type": "test"},
            {"value": 1},
            telemetry_at=_iso_timestamp(age_seconds),
            telemetry_source="authenticated test telemetry",
        )

    async def test_status_is_empty_without_adapters_or_configured_devices(self):
        engine = DigitalTwinEngine(state_path=self.state_path)
        status = engine.status()
        self.assertEqual(status["devices"], [])
        self.assertEqual(status["supported_domains"], [])

    async def test_unknown_device_returns_not_supported_without_fabricating_result(self):
        result = await self.engine.simulate_command("unknown", {"action": "move"})
        self.assertEqual(result["status"], "NOT_SUPPORTED")
        self.assertFalse(result["dispatch_allowed"])
        self.assertNotIn("result", result)

    async def test_rejects_stale_telemetry(self):
        self._register_device(age_seconds=60)
        result = await self.engine.simulate_command("test-1", {"action": "check"})
        self.assertEqual(result["status"], "STALE_TELEMETRY")
        self.assertFalse(result["dispatch_allowed"])

    async def test_rejects_unvalidated_adapter_and_invalid_state(self):
        with self.assertRaises(DigitalTwinError):
            self.engine.register_adapter(type(
                "UnvalidatedAdapter",
                (),
                {
                    "domain": "unsafe",
                    "model_version": "1",
                    "validated": False,
                },
            )())
        with self.assertRaises(DigitalTwinError):
            self.engine.register_device(
                "test-1",
                "test_device",
                {"device_type": "test"},
                {"value": "invalid"},
                telemetry_at=_iso_timestamp(),
                telemetry_source="authenticated test telemetry",
            )

    async def test_authenticated_fresh_telemetry_synchronizes_state(self):
        self._register_device()
        with self.assertRaises(DigitalTwinError):
            self.engine.synchronize_telemetry(
                "test-1",
                {"value": 2},
                telemetry_at=_iso_timestamp(),
                source_authenticated=False,
            )
        self.engine.synchronize_telemetry(
            "test-1",
            {"value": 2},
            telemetry_at=_iso_timestamp(),
            source_authenticated=True,
        )
        status = self.engine.status()
        self.assertEqual(status["devices"][0]["state"]["value"], 2)

    def test_device_configuration_survives_reload_and_returns_a_copy(self):
        self._register_device()
        reloaded = DigitalTwinEngine(
            state_path=self.state_path,
            adapters=(self.adapter,),
        )
        configuration = reloaded.device_configuration("test-1")
        self.assertEqual(configuration["state"], {"value": 1})
        configuration["state"]["value"] = 99
        self.assertEqual(
            reloaded.device_configuration("test-1")["state"],
            {"value": 1},
        )
        with self.assertRaises(DigitalTwinError):
            reloaded.device_configuration("unknown")

    async def test_simulation_result_is_only_a_preflight_and_never_dispatches(self):
        service = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=1,
            max_iterations=1_000,
            executor_factory=ThreadPoolExecutor,
        )
        await service.start()
        self.engine.predictive_service = service
        self._register_device()
        try:
            result = await self.engine.assess_dispatch(
                "test-1", {"action": "safe test"}
            )
        finally:
            await service.shutdown()

        self.assertEqual(result["status"], "SIMULATED")
        self.assertTrue(result["dispatch_allowed"])
        self.assertIn("existing security and confirmation", result["detail"])
        self.assertEqual(result["simulation"]["sample_size"], 1_000)

    async def test_mqtt_non_stop_fails_closed_when_no_twin_is_available(self):
        with (
            patch("app.digital_twin.get_default_engine", return_value=None),
            patch("actions.iot_controller._publish") as publish,
        ):
            result = control_iot({
                "action": "move",
                "device_id": "test-1",
                "direction": "FORWARD",
            })
        self.assertIn("NOT_SUPPORTED", result)
        publish.assert_not_called()

    async def test_mqtt_non_stop_fails_closed_when_preflight_rejects(self):
        class RejectingTwin:
            def preflight_mqtt_command(self, _device_id, _command):
                return {"status": "STALE_TELEMETRY", "dispatch_allowed": False}

        with (
            patch("app.digital_twin.get_default_engine", return_value=RejectingTwin()),
            patch("actions.iot_controller._publish") as publish,
        ):
            result = control_iot({
                "action": "switch",
                "device_id": "test-1",
                "state": "ON",
            })
        self.assertIn("rejected", result.lower())
        publish.assert_not_called()

    async def test_robotics_drone_cannot_use_rc_car_mqtt_actuation(self):
        class DroneTwin:
            def device_configuration(self, _device_id):
                return {
                    "domain": "robotics",
                    "specification": {"vehicle_type": "drone"},
                }

            def preflight_mqtt_command(self, _device_id, _command):
                raise AssertionError("Drone must be rejected before simulation.")

        with (
            patch("app.digital_twin.get_default_engine", return_value=DroneTwin()),
            patch("actions.iot_controller._publish") as publish,
        ):
            result = control_iot({
                "action": "move",
                "device_id": "drone-1",
                "direction": "FORWARD",
            })
        self.assertIn("not an enabled RC car", result)
        publish.assert_not_called()

    async def test_stop_bypasses_twin_preflight_and_remains_immediate(self):
        with (
            patch(
                "app.digital_twin.get_default_engine",
                side_effect=AssertionError("STOP must not consult the twin."),
            ),
            patch(
                "actions.iot_controller._publish",
                return_value="stop published",
            ) as publish,
        ):
            result = control_iot({
                "action": "move",
                "device_id": "test-1",
                "direction": "STOP",
            })
        self.assertEqual(result, "stop published")
        publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
