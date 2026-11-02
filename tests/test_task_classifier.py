from __future__ import annotations

import unittest
from unittest.mock import patch

from app.agent import FridayAgent
from app.task_classifier import TaskRisk, classify_task, has_explicit_simulation_intent


class TaskClassifierTests(unittest.TestCase):
    def test_explicit_english_and_singlish_requests_are_detected(self):
        self.assertTrue(has_explicit_simulation_intent("Run a simulation for this plan"))
        self.assertTrue(has_explicit_simulation_intent("Please simulate this command"))
        self.assertTrue(has_explicit_simulation_intent("meeka simulate karanna"))
        self.assertTrue(has_explicit_simulation_intent("අනුකරණය කරන්න"))

    def test_explanatory_questions_and_ordinary_messages_are_not_explicit(self):
        self.assertFalse(has_explicit_simulation_intent("How do simulations work?"))
        self.assertFalse(has_explicit_simulation_intent("What is a digital twin?"))
        self.assertFalse(has_explicit_simulation_intent("Please check the weather"))

    def test_emergency_stop_remains_normal_but_other_iot_actions_are_high_risk(self):
        stop = classify_task(
            tool_name="iot_control",
            arguments={"action": "move", "direction": "STOP"},
        )
        movement = classify_task(
            tool_name="iot_control",
            arguments={"action": "move", "direction": "FORWARD"},
        )
        self.assertEqual(stop.risk, TaskRisk.NORMAL)
        self.assertEqual(movement.risk, TaskRisk.HIGH_RISK)


class ExplicitSimulationAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_chat_does_not_invoke_twin(self):
        agent = object.__new__(FridayAgent)
        agent._session_locks = {}
        agent._verified_speaker = lambda _token: None
        agent._chat_sync = lambda _session, _text: "A normal response."

        with patch("app.digital_twin.get_default_engine") as get_engine:
            response = await agent.chat_text("How do simulations work?")

        self.assertEqual(response, "A normal response.")
        get_engine.assert_not_called()

    async def test_explicit_request_uses_twin_without_calling_gemini_chat(self):
        class FakeTwin:
            async def simulate_explicit_request(self, _text):
                return {
                    "status": "NOT_SUPPORTED",
                    "dispatch_allowed": False,
                    "detail": "No validated adapter is configured.",
                }

        agent = object.__new__(FridayAgent)
        agent._session_locks = {}
        agent._verified_speaker = lambda _token: None
        agent._chat_sync = lambda _session, _text: self.fail(
            "Explicit simulation must not enter the ordinary tool/chat loop."
        )

        with patch("app.digital_twin.get_default_engine", return_value=FakeTwin()):
            response = await agent.chat_text("Please run a simulation for this plan")

        self.assertIn("not supported", response.lower())
        self.assertIn("No validated adapter", response)

    async def test_explicit_voice_request_uses_twin_after_transcription(self):
        from types import SimpleNamespace

        class FakeTwin:
            async def simulate_explicit_request(self, _text):
                return {
                    "status": "NOT_SUPPORTED",
                    "dispatch_allowed": False,
                    "detail": "No validated adapter is configured.",
                }

        agent = object.__new__(FridayAgent)
        agent._session_locks = {}
        agent.voice_profiles = SimpleNamespace(
            identify=lambda _audio: {
                "profile_id": None,
                "name": "Guest",
                "role": "GUEST",
            }
        )
        agent._transcribe_sync = lambda _audio, _mime: (
            "Please run a simulation for this plan"
        )
        agent._synthesize_sync = lambda response: (response.encode(), "audio/test")
        agent._chat_sync = lambda _session, _text: self.fail(
            "Explicit voice simulation must not enter the ordinary chat loop."
        )

        with (
            patch("app.voice.decode_to_pcm_wav", return_value=b"normalized"),
            patch("app.digital_twin.get_default_engine", return_value=FakeTwin()),
        ):
            result = await agent.chat_audio(b"audio", "audio/wav")

        self.assertIn("not supported", result["response"].lower())
        self.assertEqual(result["audio"], result["response"].encode())


if __name__ == "__main__":
    unittest.main()
