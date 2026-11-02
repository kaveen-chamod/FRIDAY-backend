from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from app.ambient_observer import (
    AmbientObserver,
    ObservationConsentRequired,
    ObservationRateLimited,
)
from app.main import app


class AmbientObserverUnitTests(unittest.TestCase):
    def setUp(self):
        self.observer = AmbientObserver()

    def test_observation_is_disabled_by_default(self):
        with self.assertRaises(ObservationConsentRequired):
            self.observer.observe_code(
                "profile-a", "event-a", "python_code_buffer", "x = ("
            )

    def test_consent_is_not_persisted_between_observer_instances(self):
        self.observer.set_consent("profile-a", True)
        restarted = AmbientObserver()
        self.assertFalse(restarted.consent_status("profile-a"))

    def test_explicit_code_event_returns_evidence_based_advisory_only(self):
        self.observer.set_consent("profile-a", True)
        result = self.observer.observe_code(
            "profile-a", "event-a", "python_code_buffer", "value = ("
        )
        self.assertEqual(result.status, "alert")
        self.assertGreaterEqual(result.alert["confidence"], 0.85)
        self.assertTrue(result.alert["advisory_only"])
        self.assertIn("AST parser diagnostic", result.alert["evidence"])
        self.assertNotIn("value = (", result.alert["message"])

    def test_valid_python_is_silent_and_transcript_source_is_rejected(self):
        self.observer.set_consent("profile-a", True)
        result = self.observer.observe_code(
            "profile-a", "event-a", "python_code_buffer", "value = 1"
        )
        self.assertEqual(result.status, "no_alert")
        with self.assertRaisesRegex(ValueError, "Only explicitly submitted"):
            self.observer.observe_code(
                "profile-a", "event-b", "conversation_transcript", "hello"
            )

    def test_event_deduplication_does_not_store_source_text(self):
        self.observer.set_consent("profile-a", True)
        first = self.observer.observe_code(
            "profile-a", "event-a", "python_code_buffer", "value = 1"
        )
        duplicate = self.observer.observe_code(
            "profile-a", "event-a", "python_code_buffer", "value = 1"
        )
        self.assertEqual(first.status, "no_alert")
        self.assertEqual(duplicate.status, "duplicate")
        self.assertNotIn("value = 1", repr(self.observer._seen_events))
        with self.assertRaisesRegex(ValueError, "different content"):
            self.observer.observe_code(
                "profile-a", "event-a", "python_code_buffer", "value = 2"
            )

    def test_alert_rate_limit_and_opt_out_clear_consent_state(self):
        self.observer.set_consent("profile-a", True)
        first = self.observer.observe_code(
            "profile-a", "event-a", "python_code_buffer", "value = (", now=10.0
        )
        limited = self.observer.observe_code(
            "profile-a", "event-b", "python_code_buffer", "value = (", now=20.0
        )
        self.assertEqual(first.status, "alert")
        self.assertEqual(limited.status, "rate_limited")
        self.observer.set_consent("profile-a", False)
        self.assertFalse(self.observer.consent_status("profile-a"))
        with self.assertRaises(ObservationConsentRequired):
            self.observer.observe_code(
                "profile-a", "event-c", "python_code_buffer", "value = 1"
            )

    def test_request_rate_limit_and_input_validation(self):
        self.observer.set_consent("profile-a", True)
        for index in range(30):
            result = self.observer.observe_code(
                "profile-a",
                f"valid-{index}",
                "python_code_buffer",
                "value = 1",
                now=100.0,
            )
            self.assertEqual(result.status, "no_alert")
        with self.assertRaises(ObservationRateLimited):
            self.observer.observe_code(
                "profile-a", "valid-30", "python_code_buffer", "value = 1", now=100.0
            )
        with self.assertRaises(ValueError):
            self.observer.observe_code(
                "profile-a",
                "too-long",
                "python_code_buffer",
                "x" * (16_384 + 1),
                now=161.0,
            )


class _VerifiedAgent:
    def _verified_speaker(self, token: str | None):
        if token != "enrolled-token":
            return None
        return {
            "profile_id": "profile-api",
            "role": "MASTER",
            "enrolled_role": "MASTER",
        }


class AmbientObserverApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_state = app.state._state.copy()
        self.observer = AmbientObserver()
        app.state.ambient_observer = self.observer
        app.state.agent = _VerifiedAgent()
        app.state.cognitive_io_slots = asyncio.Semaphore(2)
        app.state._state.pop("succession", None)
        self.api_token = "ambient-test-token-" + "x" * 32
        self.token_patch = patch("app.main._API_TOKEN", self.api_token)
        self.token_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "X-Friday-Speaker-Token": "enrolled-token",
        }

    async def asyncTearDown(self):
        await self.client.aclose()
        self.token_patch.stop()
        app.state._state.clear()
        app.state._state.update(self.previous_state)

    async def test_authenticated_token_and_per_submission_consent_are_required(self):
        denied = await self.client.post(
            "/api/ambient/observe",
            headers=self.headers,
            json={
                "event_id": "event-a",
                "source": "python_code_buffer",
                "content": "value = 1",
            },
        )
        self.assertEqual(denied.status_code, 403)

        invalid_speaker = await self.client.post(
            "/api/ambient/consent",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "X-Friday-Speaker-Token": "invalid",
            },
            json={"enabled": True},
        )
        self.assertEqual(invalid_speaker.status_code, 403)
        unauthenticated = await self.client.get("/api/ambient/consent")
        self.assertEqual(unauthenticated.status_code, 401)

        opted_in = await self.client.post(
            "/api/ambient/consent", headers=self.headers, json={"enabled": True}
        )
        self.assertEqual(opted_in.status_code, 200)
        self.assertTrue(opted_in.json()["enabled"])
        observed = await self.client.post(
            "/api/ambient/observe",
            headers=self.headers,
            json={
                "event_id": "event-a",
                "source": "python_code_buffer",
                "content": "value = 1",
            },
        )
        self.assertEqual(observed.status_code, 200)
        self.assertEqual(observed.json()["status"], "no_alert")

    async def test_conversation_source_is_explicitly_unsupported_and_opt_out_works(self):
        await self.client.post(
            "/api/ambient/consent", headers=self.headers, json={"enabled": True}
        )
        transcript = await self.client.post(
            "/api/ambient/observe",
            headers=self.headers,
            json={
                "event_id": "transcript-a",
                "source": "conversation_transcript",
                "content": "A family member said something.",
            },
        )
        self.assertEqual(transcript.status_code, 422)
        self.assertIn("NOT_SUPPORTED", transcript.json()["detail"])

        opted_out = await self.client.post(
            "/api/ambient/consent", headers=self.headers, json={"enabled": False}
        )
        self.assertFalse(opted_out.json()["enabled"])
        status = await self.client.get("/api/ambient/consent", headers=self.headers)
        self.assertFalse(status.json()["enabled"])

    async def test_observation_input_size_is_limited(self):
        await self.client.post(
            "/api/ambient/consent", headers=self.headers, json={"enabled": True}
        )
        oversized = await self.client.post(
            "/api/ambient/observe",
            headers=self.headers,
            json={
                "event_id": "oversized",
                "source": "python_code_buffer",
                "content": "x" * 16_385,
            },
        )
        self.assertEqual(oversized.status_code, 422)

    async def test_code_analysis_does_not_block_the_event_loop(self):
        await self.client.post(
            "/api/ambient/consent", headers=self.headers, json={"enabled": True}
        )
        original_observe = self.observer.observe_code

        def slow_observe(*args, **kwargs):
            time.sleep(0.12)
            return original_observe(*args, **kwargs)

        with patch.object(self.observer, "observe_code", side_effect=slow_observe):
            ticks = 0

            async def ticker():
                nonlocal ticks
                for _ in range(5):
                    await asyncio.sleep(0.015)
                    ticks += 1

            response, _ = await asyncio.gather(
                self.client.post(
                    "/api/ambient/observe",
                    headers=self.headers,
                    json={
                        "event_id": "responsive",
                        "source": "python_code_buffer",
                        "content": "value = 1",
                    },
                ),
                ticker(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ticks, 5)


if __name__ == "__main__":
    unittest.main()
