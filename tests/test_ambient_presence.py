from __future__ import annotations

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from app.presence import PresenceEngine


class PresenceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.engine = PresenceEngine(clock=lambda: self.now)

    def test_motion_clear_starts_fifteen_minute_vacancy_countdown(self) -> None:
        active = self.engine.ingest_sensor_telemetry(True, 0)
        self.assertEqual(active["state"], "ACTIVE_PRESENCE")
        self.assertTrue(active["mic_active"])

        vacant = self.engine.ingest_sensor_telemetry(False, 0)
        self.assertEqual(vacant["state"], "VACANT_COUNTDOWN")
        self.assertTrue(vacant["mic_active"])
        self.assertEqual(vacant["remaining_seconds"], 900)

        self.now += 899.2
        self.assertEqual(
            self.engine.get_presence_state()["remaining_seconds"], 1
        )
        self.now += 0.8
        asleep = self.engine.get_presence_state()
        self.assertEqual(asleep["state"], "SLEEP_STANDBY")
        self.assertFalse(asleep["mic_active"])
        self.assertEqual(asleep["remaining_seconds"], 0)

    def test_person_detection_or_pir_wakes_the_satellite(self) -> None:
        visual_presence = self.engine.ingest_sensor_telemetry(False, 1, 25.0)
        self.assertEqual(visual_presence["state"], "ACTIVE_PRESENCE")
        self.assertEqual(visual_presence["person_count"], 1)

        with patch(
            "app.access_control.verify_master_passphrase", return_value=False
        ):
            self.assertFalse(
                self.engine.force_state_override(
                    "SLEEP_STANDBY", "not-the-master"
                )
            )
        with patch(
            "app.access_control.verify_master_passphrase", return_value=True
        ):
            self.assertTrue(
                self.engine.force_state_override(
                    "SLEEP_STANDBY", "verified-master-passphrase"
                )
            )
        self.assertEqual(
            self.engine.get_presence_state()["state"], "SLEEP_STANDBY"
        )
        self.assertEqual(
            self.engine.ingest_sensor_telemetry(True, 0)["state"],
            "ACTIVE_PRESENCE",
        )

    def test_wake_word_is_immediate_and_rejects_unknown_sources(self) -> None:
        awake = self.engine.trigger_wake_word("energy_threshold")
        self.assertEqual(awake["state"], "ACTIVE_PRESENCE")
        self.assertTrue(awake["mic_active"])
        self.assertEqual(awake["last_wake_source"], "energy_threshold")
        with self.assertRaises(ValueError):
            self.engine.trigger_wake_word("untrusted_client")

    def test_force_override_requires_master_passphrase(self) -> None:
        with patch(
            "app.access_control.verify_master_passphrase", return_value=False
        ):
            self.assertFalse(
                self.engine.force_state_override(
                    "ACTIVE_PRESENCE", "incorrect-passphrase"
                )
            )
        self.assertEqual(
            self.engine.get_presence_state()["state"], "SLEEP_STANDBY"
        )


class PresenceApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_presence_endpoints_use_application_engine(self) -> None:
        from app import main

        engine = PresenceEngine()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            presence_engine=engine,
        )))
        telemetry = main.PresenceTelemetryRequest(
            pir_active=True, person_count=0, light_lux=10.5
        )
        result = await main.ingest_presence_telemetry(telemetry, request)
        self.assertEqual(result["state"], "ACTIVE_PRESENCE")

        wake = await main.trigger_presence_wake(
            main.PresenceWakeRequest(source="home_mic"), request
        )
        self.assertTrue(wake["mic_active"])
        state = await main.presence_state(request)
        self.assertEqual(state["state"], "ACTIVE_PRESENCE")


class PresenceWebSocketGateTests(unittest.IsolatedAsyncioTestCase):
    class FakeSession:
        def __init__(self) -> None:
            self.inputs: list[dict] = []

        async def receive(self):
            await asyncio.Event().wait()
            if False:
                yield None

        async def send_realtime_input(self, **kwargs) -> None:
            self.inputs.append(kwargs)

    class FakeAgent:
        def __init__(self) -> None:
            self.session = PresenceWebSocketGateTests.FakeSession()

        @asynccontextmanager
        async def live_session(self, *, speaker_token, master_passphrase):
            yield self.session, None

    class FakeWebSocket:
        headers = {"authorization": "test-token"}
        query_params = {}

        def __init__(
            self,
            client: str,
            presence_engine: PresenceEngine,
            later_messages: list[dict] | None = None,
        ) -> None:
            self.app = SimpleNamespace(state=SimpleNamespace(
                succession=None,
                agent=PresenceWebSocketGateTests.FakeAgent(),
                presence_engine=presence_engine,
            ))
            self.incoming = iter([
                {
                    "type": "websocket.receive",
                    "text": json.dumps({
                        "client": client,
                        "sample_rate": 16000,
                    }),
                },
                *(later_messages or [
                    {"type": "websocket.receive", "bytes": b"\x00\x00"},
                    {"type": "websocket.disconnect", "code": 1000},
                ]),
            ])
            self.sent_json: list[dict] = []

        async def accept(self) -> None:
            pass

        async def receive(self) -> dict:
            return next(self.incoming)

        async def send_json(self, value: dict) -> None:
            self.sent_json.append(value)

        async def send_bytes(self, _value: bytes) -> None:
            pass

        async def close(self, **_kwargs) -> None:
            pass

    async def _run_live_socket(
        self,
        client: str,
        engine: PresenceEngine,
        later_messages: list[dict] | None = None,
    ) -> "PresenceWebSocketGateTests.FakeWebSocket":
        from app import main

        websocket = self.FakeWebSocket(client, engine, later_messages)
        route = next(route for route in main.app.routes if route.path == "/ws/live")
        with patch("app.main._token_matches", return_value=True):
            await route.endpoint(websocket)
        return websocket

    async def test_sleeping_satellite_audio_is_discarded_before_gemini(self) -> None:
        engine = PresenceEngine()
        websocket = await self._run_live_socket("esp32_satellite", engine)

        self.assertEqual(websocket.sent_json[0]["type"], "ready")
        self.assertFalse(websocket.sent_json[0]["mic_active"])
        self.assertTrue(any(
            event["type"] == "audio_discarded"
            for event in websocket.sent_json
        ))
        self.assertEqual(websocket.app.state.agent.session.inputs, [])

    async def test_sleep_gate_does_not_suspend_web_live_audio(self) -> None:
        engine = PresenceEngine()
        websocket = await self._run_live_socket("web", engine)

        self.assertEqual(
            websocket.app.state.agent.session.inputs[0]["audio"].mime_type,
            "audio/pcm;rate=16000",
        )
        self.assertNotIn(
            "audio_discarded",
            [event["type"] for event in websocket.sent_json],
        )

    async def test_satellite_wake_control_enables_following_pcm_frames(self) -> None:
        engine = PresenceEngine()
        websocket = await self._run_live_socket(
            "esp32_satellite",
            engine,
            [
                {
                    "type": "websocket.receive",
                    "text": json.dumps({
                        "type": "wake_word",
                        "source": "energy_threshold",
                    }),
                },
                {"type": "websocket.receive", "bytes": b"\x00\x00"},
                {"type": "websocket.disconnect", "code": 1000},
            ],
        )

        self.assertEqual(len(websocket.app.state.agent.session.inputs), 1)
        self.assertEqual(
            websocket.app.state.agent.session.inputs[0]["audio"].mime_type,
            "audio/pcm;rate=16000",
        )
        states = [
            event["state"]
            for event in websocket.sent_json
            if event["type"] == "presence_state"
        ]
        self.assertEqual(states, ["ACTIVE_PRESENCE"])

    async def test_sleep_scope_leaves_non_satellite_routes_registered(self) -> None:
        from app import main

        route_paths = {getattr(route, "path", "") for route in main.app.routes}
        self.assertTrue({
            "/health",
            "/api/chat-text",
            "/api/defense/threats/active",
            "/api/telegram-webhook",
            "/ws/telemetry",
        }.issubset(route_paths))


if __name__ == "__main__":
    unittest.main()
