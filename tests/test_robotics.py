from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from app.robotics.collision_guard import CollisionGuard, CollisionGuardError
from app.robotics.drone import DroneVehicle
from app.robotics.rc_car import RCCarVehicle
from app.robotics.spatial_map import SpatialMap, SpatialMapError, WaypointRegistry
from app.robotics.tracking_engine import ArucoTracker, TrackingError
from app.robotics.twin_adapter import RoboticsTwinAdapter


_SPEC = {
    "vehicle_type": "rc_car",
    "minimum_distance_cm": 15.0,
    "maximum_current_a": 4.0,
    "distance_uncertainty_cm": 0.1,
    "current_uncertainty_a": 0.01,
    "action_success_probability": 1.0,
    "action_catastrophic_probability": 0.0,
    "epistemic_uncertainty": 0.01,
}
_STATE = {
    "distance_cm": 100.0,
    "current_a": 1.0,
    "x": 0,
    "y": 0,
    "z": 0,
    "path_clear": True,
}


class SpatialMapTests(unittest.TestCase):
    def test_astar_returns_shortest_obstacle_avoiding_path(self):
        spatial_map = SpatialMap(5, 5, obstacles=[(1, 0), (1, 1), (1, 2)])
        path = spatial_map.find_path((0, 0), (4, 0))
        self.assertEqual(path[0], (0, 0, 0))
        self.assertEqual(path[-1], (4, 0, 0))
        self.assertEqual(len(path), 11)
        self.assertFalse(any(spatial_map.is_blocked(point) for point in path))

    def test_astar_supports_three_dimensions_and_no_path(self):
        spatial_map = SpatialMap(2, 2, depth=2)
        self.assertEqual(
            spatial_map.find_path((0, 0, 0), (0, 0, 1)),
            [(0, 0, 0), (0, 0, 1)],
        )
        blocked = SpatialMap(3, 1, obstacles=[(1, 0)])
        with self.assertRaises(SpatialMapError):
            blocked.find_path((0, 0), (2, 0))

    def test_map_rejects_unbounded_dimensions_and_bad_coordinates(self):
        with self.assertRaises(SpatialMapError):
            SpatialMap(1_000_000, 1_000_000)
        with self.assertRaises(SpatialMapError):
            SpatialMap(2, 2, obstacles=[(2, 0)])
        with self.assertRaises(SpatialMapError):
            SpatialMap(2, 2).find_path((False, 0), (1, 1))

    def test_waypoints_are_bounded_by_grid_and_obstacles(self):
        spatial_map = SpatialMap(4, 4, obstacles=[(1, 1)])
        waypoints = WaypointRegistry(spatial_map)
        self.assertEqual(waypoints.add("home", (0, 0))["coordinate"], (0, 0, 0))
        with self.assertRaises(SpatialMapError):
            waypoints.add("obstacle", (1, 1))
        with self.assertRaises(SpatialMapError):
            waypoints.add("outside", (4, 0))
        self.assertEqual(waypoints.get("home"), (0, 0, 0))


class CollisionGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_distance_under_15_immediately_stops_without_twin(self):
        stopped = []
        guard = CollisionGuard(emergency_stop=lambda vehicle_id: stopped.append(vehicle_id) or "stop sent")
        decision = guard.observe_telemetry("car-1", distance_cm=14.99)
        self.assertTrue(decision.emergency_stop_triggered)
        self.assertFalse(decision.allowed)
        self.assertEqual(stopped, ["car-1"])

    async def test_distance_of_15_cm_does_not_trigger_emergency_stop(self):
        guard = CollisionGuard(emergency_stop=Mock(return_value="stop sent"))
        decision = guard.observe_telemetry("car-1", distance_cm=15.0)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.emergency_stop_triggered)

    async def test_missing_or_failed_stop_handler_is_reported(self):
        with self.assertRaises(CollisionGuardError):
            CollisionGuard().observe_telemetry("car-1", distance_cm=10)
        guard = CollisionGuard(emergency_stop=Mock(return_value="Emergency stop not delivered"))
        with self.assertRaises(CollisionGuardError):
            guard.observe_telemetry("car-1", distance_cm=10)

    async def test_unsafe_current_triggers_immediate_hard_stop(self):
        stop = Mock(return_value="stop sent")
        guard = CollisionGuard(emergency_stop=stop)
        decision = guard.observe_telemetry(
            "car-1",
            distance_cm=100,
            current_a=4.1,
            maximum_current_a=4.0,
        )
        self.assertTrue(decision.emergency_stop_triggered)
        stop.assert_called_once_with("car-1")

    async def test_rejects_invalid_sensor_values(self):
        guard = CollisionGuard(emergency_stop=Mock(return_value="stop sent"))
        for value in (float("nan"), float("inf"), -1, True):
            with self.subTest(value=value):
                with self.assertRaises(CollisionGuardError):
                    guard.observe_telemetry("car-1", distance_cm=value)

    async def test_preflight_checks_path_then_requires_twin_approval(self):
        spatial_map = SpatialMap(4, 2, obstacles=[(1, 0)])
        twin = Mock()
        twin.assess_dispatch = AsyncMock(return_value={
            "status": "SIMULATED",
            "dispatch_allowed": True,
        })
        guard = CollisionGuard()
        unsafe_path = await guard.preflight(
            twin,
            "car-1",
            {"action": "move"},
            distance_cm=100,
            current_a=1,
            maximum_current_a=4,
            spatial_map=spatial_map,
            path=[(0, 0), (1, 0)],
        )
        self.assertEqual(unsafe_path["status"], "PATH_REJECTED")
        twin.assess_dispatch.assert_not_awaited()
        safe = await guard.preflight(
            twin,
            "car-1",
            {"action": "move"},
            distance_cm=100,
            current_a=1,
            maximum_current_a=4,
            spatial_map=spatial_map,
            path=[(0, 0), (0, 1), (1, 1)],
        )
        self.assertTrue(safe["dispatch_allowed"])
        twin.assess_dispatch.assert_awaited_once()

    async def test_preflight_fails_closed_when_twin_missing_or_rejects(self):
        guard = CollisionGuard()
        missing = await guard.preflight(
            None,
            "car-1",
            {"action": "move"},
            distance_cm=100,
            current_a=1,
            maximum_current_a=4,
        )
        self.assertFalse(missing["dispatch_allowed"])
        twin = Mock()
        twin.assess_dispatch = AsyncMock(return_value={
            "status": "REJECTED",
            "dispatch_allowed": False,
        })
        denied = await guard.preflight(
            twin,
            "car-1",
            {"action": "move"},
            distance_cm=100,
            current_a=1,
            maximum_current_a=4,
        )
        self.assertFalse(denied["dispatch_allowed"])


class VehicleTests(unittest.TestCase):
    def test_rc_car_normalizes_discrete_actions_and_stop_uses_existing_action(self):
        car = RCCarVehicle("rc-car-1")
        self.assertEqual(
            car.build_action({"direction": "forward"}),
            {"device_id": "rc-car-1", "action": "move", "direction": "FORWARD"},
        )
        with patch("actions.iot_controller.control_iot", return_value="stop sent") as stop:
            self.assertEqual(car.emergency_stop(), "stop sent")
        stop.assert_called_once_with({
            "action": "move",
            "device_id": "rc-car-1",
            "direction": "STOP",
        })
        self.assertFalse(RCCarVehicle.valid_device_id("../bad"))

    def test_drone_is_explicitly_disabled_without_protocol(self):
        drone = DroneVehicle("drone-1")
        self.assertIn("not delivered", drone.emergency_stop().lower())
        with self.assertRaisesRegex(ValueError, "disabled"):
            drone.build_action({"direction": "forward"})


class RoboticsTwinAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = RoboticsTwinAdapter()

    def test_validates_bounded_device_spec_and_state(self):
        self.assertTrue(self.adapter.validate_device(_SPEC, _STATE))
        self.assertFalse(
            self.adapter.validate_device(_SPEC, {**_STATE, "distance_cm": float("nan")})
        )
        self.assertFalse(
            self.adapter.validate_device(_SPEC, {**_STATE, "path_clear": "yes"})
        )

    def test_action_model_contains_collision_and_overcurrent_constraints(self):
        spec = self.adapter.build_action_spec(
            _SPEC,
            _STATE,
            {"device_id": "car-1", "action": "move", "direction": "FORWARD"},
        )
        self.assertEqual(len(spec["success_conditions"]), 2)
        self.assertEqual(spec["success_conditions"][0]["value"], 15.0)
        self.assertEqual(spec["success_conditions"][1]["value"], 4.0)
        blocked = self.adapter.build_action_spec(
            _SPEC,
            {**_STATE, "path_clear": False},
            {"device_id": "car-1", "action": "move", "direction": "FORWARD"},
        )
        self.assertEqual(blocked["catastrophic_probability"], 1.0)

    def test_adapter_only_allows_high_confidence_simulator_recommendations(self):
        self.assertTrue(self.adapter.allow_dispatch({
            "recommendation": "RECOMMENDED",
            "observed_catastrophic_count": 0,
            "success_probability": 0.99,
            "uncertainty_score": 0.01,
            "confidence_intervals_95": {
                "catastrophic_risk_score": {"upper": 0.01}
            },
        }))
        self.assertFalse(self.adapter.allow_dispatch({
            "recommendation": "PROCEED_WITH_CAUTION",
            "observed_catastrophic_count": 0,
            "success_probability": 0.99,
            "uncertainty_score": 0.01,
            "confidence_intervals_95": {
                "catastrophic_risk_score": {"upper": 0.01}
            },
        }))
        self.assertFalse(self.adapter.allow_dispatch({
            "recommendation": "RECOMMENDED",
            "observed_catastrophic_count": 0,
            "success_probability": 0.99,
            "uncertainty_score": 0.01,
            "confidence_intervals_95": {
                "catastrophic_risk_score": {"upper": 0.20}
            },
        }))


class TrackingTests(unittest.TestCase):
    def test_tracking_never_opens_camera_and_reports_optional_dependency(self):
        tracker = ArucoTracker()
        with patch.object(
            tracker, "_cv2", side_effect=TrackingError("optional dependency missing")
        ):
            with self.assertRaisesRegex(TrackingError, "optional dependency"):
                tracker.detect(object())

    def test_pose_calibration_bounds_are_checked(self):
        tracker = ArucoTracker()
        with self.assertRaises(TrackingError):
            tracker.estimate_pose(
                object(),
                marker_size_m=float("inf"),
                camera_matrix=[],
                distortion_coefficients=[],
            )


class RoboticsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx

        from app import access_control
        from app.digital_twin import DigitalTwinEngine
        from app.main import app

        self.app = app
        self.access_control = access_control
        self.previous_state = app.state._state.copy()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.token = "robotics-test-api-token-" + "x" * 24
        self.token_patch = patch("app.main._API_TOKEN", self.token)
        self.master_audit_patch = patch("app.main._audit_master_event")
        self.master_patch = patch(
            "app.main.access_control.require_master_passphrase",
            side_effect=lambda phrase: (
                None
                if phrase == "test-master-passphrase"
                else (_ for _ in ()).throw(
                    access_control.MasterAuthError("Master verification failed.")
                )
            ),
        )
        self.token_patch.start()
        self.master_audit_patch.start()
        self.master_patch.start()
        self.engine = DigitalTwinEngine(
            state_path=Path(self.temp_dir.name) / "twins.json",
            adapters=(RoboticsTwinAdapter(),),
        )
        self.vehicles = {}
        self.specs = {}
        self.engine.bind_loop(asyncio.get_running_loop())
        app.state.digital_twin = self.engine
        app.state.robotics_vehicles = self.vehicles
        app.state.robotics_specs = self.specs
        app.state.robotics_map = None
        app.state.robotics_waypoints = None
        app.state.robotics_collision_guard = CollisionGuard(
            emergency_stop=lambda vehicle_id: self.vehicles[vehicle_id].emergency_stop()
        )
        app.state.cognitive_io_slots = asyncio.Semaphore(2)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.master_headers = {
            **self.headers,
            "X-Friday-Speaker-Token": "master-token",
            "X-Friday-Master-Passphrase": "test-master-passphrase",
        }
        app.state.agent = type(
            "MasterAgent",
            (),
            {
                "_verified_speaker": lambda _self, token: (
                    {"profile_id": "master", "role": "MASTER", "enrolled_role": "MASTER"}
                    if token == "master-token"
                    else None
                ),
            },
        )()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.master_patch.stop()
        self.master_audit_patch.stop()
        self.token_patch.stop()
        self.app.state._state.clear()
        self.app.state._state.update(self.previous_state)
        self.temp_dir.cleanup()

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    async def _register_car(self):
        response = await self.client.post(
            "/api/robotics/devices/register",
            headers=self.master_headers,
            json={
                "device_id": "robot-car-1",
                "vehicle_type": "rc_car",
                "specification": _SPEC,
                "telemetry_state": _STATE,
                "master_passphrase": "test-master-passphrase",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

    async def test_robotics_routes_require_api_authentication(self):
        response = await self.client.get("/api/robotics/status")
        self.assertEqual(response.status_code, 401)

    async def test_map_waypoint_and_planning_endpoints_never_actuate(self):
        map_response = await self.client.put(
            "/api/robotics/map",
            headers=self.headers,
            json={"width": 4, "height": 3, "obstacles": [[1, 0]]},
        )
        self.assertEqual(map_response.status_code, 200)
        waypoint = await self.client.post(
            "/api/robotics/waypoints",
            headers=self.headers,
            json={"name": "home", "coordinate": [0, 0]},
        )
        self.assertEqual(waypoint.status_code, 200)
        planned = await self.client.post(
            "/api/robotics/plan",
            headers=self.headers,
            json={"start": [0, 0], "goal": [3, 0]},
        )
        self.assertEqual(planned.status_code, 200)
        self.assertEqual(planned.json()["physical_action"], "not_dispatched")
        self.assertEqual(planned.json()["path"][0], [0, 0, 0])
        self.assertEqual(
            (await self.client.get("/api/robotics/waypoints", headers=self.headers)).json()[
                "waypoints"
            ][0]["name"],
            "home",
        )

    async def test_device_registration_requires_master_and_preflight_never_dispatches(self):
        body = {
            "device_id": "robot-car-1",
            "vehicle_type": "rc_car",
            "specification": _SPEC,
            "telemetry_state": _STATE,
            "master_passphrase": "test-master-passphrase",
        }
        denied = await self.client.post(
            "/api/robotics/devices/register",
            headers=self.headers,
            json=body,
        )
        self.assertEqual(denied.status_code, 403)
        await self._register_car()
        await self.client.put(
            "/api/robotics/map",
            headers=self.headers,
            json={"width": 4, "height": 2, "obstacles": [[1, 0]]},
        )
        preflight = await self.client.post(
            "/api/robotics/preflight",
            headers=self.headers,
            json={
                "device_id": "robot-car-1",
                "action": {"direction": "FORWARD"},
                "path": [[0, 0], [1, 0]],
            },
        )
        self.assertEqual(preflight.status_code, 200, preflight.text)
        self.assertEqual(preflight.json()["status"], "PATH_REJECTED")

    async def test_telemetry_under_15_cm_uses_immediate_existing_stop_path(self):
        await self._register_car()
        timestamp = datetime.now(timezone.utc).isoformat()
        state = {**_STATE, "distance_cm": 14.0}
        with patch("actions.iot_controller._publish", return_value="stop sent") as publish:
            response = await self.client.post(
                "/api/robotics/telemetry",
                headers=self.headers,
                json={
                    "device_id": "robot-car-1",
                    "telemetry_at": timestamp,
                    "distance_cm": 14.0,
                    "state": state,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["collision_guard"]["emergency_stop_triggered"])
        publish.assert_called_once()
        sent = publish.call_args.args[0]
        self.assertEqual(sent["action"], "move")
        self.assertEqual(sent["direction"], "STOP")

    async def test_robotics_adapter_preflight_runs_simulation_before_allowing(self):
        from concurrent.futures import ThreadPoolExecutor

        from app.predictive_engine import SimulationJobManager

        service = SimulationJobManager(
            max_concurrent_jobs=1,
            max_queued_jobs=1,
            timeout_seconds=5,
            max_iterations=10_000,
            executor_factory=ThreadPoolExecutor,
        )
        await service.start()
        try:
            engine = self.engine
            engine.predictive_service = service
            await self._register_car()
            result = await engine.assess_dispatch(
                "robot-car-1",
                {
                    "device_id": "robot-car-1",
                    "action": "move",
                    "direction": "FORWARD",
                },
            )
        finally:
            await service.shutdown()
        self.assertEqual(result["status"], "SIMULATED")
        self.assertTrue(result["dispatch_allowed"])

    async def test_drone_registration_is_rejected_without_a_protocol(self):
        response = await self.client.post(
            "/api/robotics/devices/register",
            headers=self.master_headers,
            json={
                "device_id": "drone-1",
                "vehicle_type": "drone",
                "specification": {**_SPEC, "vehicle_type": "drone"},
                "telemetry_state": _STATE,
                "master_passphrase": "test-master-passphrase",
            },
        )
        self.assertEqual(response.status_code, 409, response.text)


if __name__ == "__main__":
    unittest.main()
