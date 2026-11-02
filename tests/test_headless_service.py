import unittest
from unittest.mock import patch
from types import SimpleNamespace

from google.genai import types
from actions.iot_controller import control_iot
from actions.security_scanner import _validated_target, scan_website
from app.agent import FridayAgent
from app.main import app
from app.modules.base import BaseModule
from app.tool_maker import ToolMaker
from app.telegram_bot import TelegramBotService
from app.web_bridge import WebPlayerBridge
from core.security import TIER_DANGEROUS, tier_for


_VALID_TOOL = '''from app.modules.base import BaseModule

class ExampleTool(BaseModule):
    TOOL = {
        "name": "example_tool",
        "description": "A test proposal",
        "parameters": {"type": "OBJECT", "properties": {}},
    }

    def run(self, parameters):
        return "ok"
'''


class HeadlessServiceTests(unittest.TestCase):
    def test_api_exposes_required_rest_and_websocket_routes(self):
        paths = {route.path for route in app.routes}
        self.assertTrue({
            "/api/chat-text",
            "/api/chat-voice",
            "/api/confirmation",
            "/api/reload-modules",
            "/ws/audio",
            "/ws/telemetry",
        }.issubset(paths))

    def test_bridge_has_the_legacy_player_surface(self):
        bridge = WebPlayerBridge()
        bridge.write_log("test")
        bridge.show_content("title", "body")
        bridge.show_weather({})
        bridge.show_flight_cards([], "A", "B", "tomorrow")
        bridge.show_news_cards([])
        bridge.show_images([])
        bridge.show_document("report.pdf")
        bridge.add_reminder_card("reminder", None)
        bridge.start_timer(1)
        bridge.set_audio_level(0.5)
        self.assertIsNone(bridge.current_file)
        bridge.current_file = "report.pdf"
        self.assertEqual(bridge.current_file, "report.pdf")
        bridge.show_confirmation("Approve", "Confirm one action")
        self.assertEqual(bridge.pending_confirmation["title"], "Approve")
        bridge.hide_confirmation()
        self.assertIsNone(bridge.pending_confirmation)

    def test_new_external_actions_are_confirmation_gated(self):
        self.assertEqual(tier_for("iot_control", {}), TIER_DANGEROUS)
        self.assertEqual(tier_for("security_scanner", {}), TIER_DANGEROUS)
        self.assertEqual(
            tier_for("iot_control", {"action": "move", "direction": "STOP"}),
            "SAFE",
        )

    def test_scanner_rejects_non_https_and_private_hosts(self):
        with self.assertRaises(ValueError):
            _validated_target("http://example.com")
        with self.assertRaises(ValueError):
            _validated_target("https://127.0.0.1")
        self.assertIn("public HTTPS URL", scan_website({"url": "http://example.com"}))

    def test_mqtt_command_requires_a_supported_single_action(self):
        self.assertIn(
            "Movement must be",
            control_iot({
                "action": "move", "device_id": "car-1", "direction": "FAST",
            }),
        )
        self.assertIn(
            "valid device_id",
            control_iot({"action": "switch", "device_id": ""}),
        )

    def test_telegram_allows_only_configured_private_users(self):
        with patch.dict(
            "os.environ",
            {"FRIDAY_TELEGRAM_ALLOWED_USER_IDS": "12345, 67890"},
        ):
            service = TelegramBotService(agent=None)

        private_chat = SimpleNamespace(type="private")
        group_chat = SimpleNamespace(type="group")
        self.assertTrue(
            service._is_authorized(
                SimpleNamespace(
                    effective_user=SimpleNamespace(id=12345),
                    effective_chat=private_chat,
                )
            )
        )
        self.assertFalse(
            service._is_authorized(
                SimpleNamespace(
                    effective_user=SimpleNamespace(id=99999),
                    effective_chat=private_chat,
                )
            )
        )
        self.assertFalse(
            service._is_authorized(
                SimpleNamespace(
                    effective_user=SimpleNamespace(id=12345),
                    effective_chat=group_chat,
                )
            )
        )

    def test_telegram_denies_all_users_without_an_allowlist(self):
        with patch.dict("os.environ", {"FRIDAY_TELEGRAM_ALLOWED_USER_IDS": ""}):
            service = TelegramBotService(agent=None)
        self.assertFalse(
            service._is_authorized(
                SimpleNamespace(
                    effective_user=SimpleNamespace(id=12345),
                    effective_chat=SimpleNamespace(type="private"),
                )
            )
        )

    def test_telegram_allowlist_rejects_malformed_ids(self):
        with patch.dict("os.environ", {"FRIDAY_TELEGRAM_ALLOWED_USER_IDS": "123,all"}):
            with self.assertRaisesRegex(ValueError, "positive Telegram user IDs"):
                TelegramBotService(agent=None)

    def test_tool_proposal_validation_is_static_and_requires_base_module(self):
        ToolMaker._validate(_VALID_TOOL)
        with self.assertRaises(ValueError):
            ToolMaker._validate(_VALID_TOOL.replace('return "ok"', 'return open("secret")'))
        self.assertTrue(issubclass(BaseModule, object))

    def test_agent_completes_a_gemini_function_call_round_trip(self):
        function_call = types.Part(
            function_call={"name": "probe", "args": {}}
        )
        responses = [
            SimpleNamespace(
                text="",
                candidates=[SimpleNamespace(content=types.Content(
                    role="model", parts=[function_call],
                ))],
            ),
            SimpleNamespace(
                text="Confirmed, Sir.",
                candidates=[SimpleNamespace(content=types.Content(
                    role="model", parts=[types.Part(text="Confirmed, Sir.")],
                ))],
            ),
        ]

        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return responses.pop(0)

        agent = object.__new__(FridayAgent)
        agent.bridge = WebPlayerBridge()
        agent._history = {}
        agent.model = "test-model"
        agent.client = SimpleNamespace(models=FakeModels())
        agent._tool_declarations = lambda: []
        agent._system_instruction = lambda: "test instruction"
        agent._dispatch = lambda name, arguments: "tool result"

        self.assertEqual(agent._chat_sync("test-session", "run the tool"), "Confirmed, Sir.")
        self.assertEqual(len(agent.client.models.calls), 2)
        response_part = agent.client.models.calls[1]["contents"][-1].parts[0]
        self.assertEqual(response_part.function_response.response, {"result": "tool result"})


if __name__ == "__main__":
    unittest.main()
