from __future__ import annotations

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from google.genai.errors import ClientError


class SynthesisFallbackTests(unittest.TestCase):
    def setUp(self):
        from app.agent import FridayAgent

        self.agent = object.__new__(FridayAgent)
        self.agent.tts_model = "test-tts-model"
        self.agent.client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(
                    side_effect=ClientError(
                        429,
                        {"error": {
                            "status": "RESOURCE_EXHAUSTED",
                            "message": "quota exhausted",
                        }},
                    )
                )
            )
        )
        self.agent._response_content = Mock(return_value=SimpleNamespace(parts=[]))

    def test_gemini_429_falls_back_to_edge_tts(self):
        with patch.object(
            self.agent,
            "_synthesize_edge_sync",
            return_value=(b"edge-audio", "audio/mpeg"),
        ) as edge_tts:
            result = self.agent._synthesize_sync("Hello there")

        self.assertEqual(result, (b"edge-audio", "audio/mpeg"))
        edge_tts.assert_called_once_with("Hello there")

    def test_tts_returns_empty_audio_when_both_providers_fail(self):
        with patch.object(
            self.agent,
            "_synthesize_edge_sync",
            side_effect=OSError("Edge-TTS unavailable"),
        ):
            self.assertEqual(self.agent._synthesize_sync("Hello"), (b"", ""))

    def test_live_session_configures_native_audio_and_aoede(self):
        from app.agent import FridayAgent

        class FakeLiveSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        agent = object.__new__(FridayAgent)
        agent.live_model = "test-live"
        agent._session_locks = {}
        agent._verified_speaker = Mock(return_value=None)
        agent._system_instruction = Mock(return_value="Test voice instructions")
        agent._tool_declarations = Mock(return_value=[])
        connect = Mock(return_value=FakeLiveSession())
        agent.client = SimpleNamespace(
            aio=SimpleNamespace(live=SimpleNamespace(connect=connect))
        )

        async def open_session():
            async with agent.live_session():
                return

        asyncio.run(open_session())
        config = connect.call_args.kwargs["config"]
        self.assertEqual(config.response_modalities, ["AUDIO"])
        self.assertEqual(
            config.speech_config.voice_config.prebuilt_voice_config.voice_name,
            "Aoede",
        )


class TelegramVoiceFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_speech_audio_delivers_the_full_text_response(self):
        from app.telegram_bot import TelegramBotService

        message = SimpleNamespace(
            voice=SimpleNamespace(
                file_size=100,
                get_file=AsyncMock(
                    return_value=SimpleNamespace(
                        download_as_bytearray=AsyncMock(return_value=bytearray(b"ogg"))
                    )
                ),
            ),
            reply_text=AsyncMock(),
            reply_voice=AsyncMock(),
        )
        update = SimpleNamespace(
            effective_message=message,
            effective_user=SimpleNamespace(id=7),
            effective_chat=SimpleNamespace(id=42, type="private"),
        )
        agent = SimpleNamespace(
            chat_audio=AsyncMock(return_value={
                "response": "The requested result is ready.",
                "audio": b"",
                "audio_mime_type": "",
            })
        )
        service = TelegramBotService(agent)
        service._allowed_user_ids = frozenset({7})

        await service._handle_voice_update(update)

        message.reply_text.assert_awaited_once_with(
            "Sir, here is the response:\n\nThe requested result is ready."
        )
        message.reply_voice.assert_not_awaited()

    async def test_voice_handler_contains_agent_exceptions(self):
        from app.telegram_bot import TelegramBotService

        message = SimpleNamespace(
            voice=SimpleNamespace(file_size=100, get_file=AsyncMock()),
            reply_text=AsyncMock(),
            reply_voice=AsyncMock(),
        )
        update = SimpleNamespace(
            effective_message=message,
            effective_user=SimpleNamespace(id=7),
            effective_chat=SimpleNamespace(id=42, type="private"),
        )
        agent = SimpleNamespace(
            chat_audio=AsyncMock(side_effect=RuntimeError("provider unavailable"))
        )
        service = TelegramBotService(agent)
        service._allowed_user_ids = frozenset({7})

        await service._handle_voice_update(update)

        message.reply_text.assert_awaited_once()
        self.assertIn("could not process", message.reply_text.await_args.args[0])


class LiveWebSocketAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_unauthenticated_connection_is_rejected_before_accept(self):
        from app import main

        class FakeWebSocket:
            headers = {}
            query_params = {}
            app = SimpleNamespace(state=SimpleNamespace(succession=None))

            def __init__(self):
                self.closed = None

            async def close(self, *, code, reason):
                self.closed = (code, reason)

        websocket = FakeWebSocket()
        route = next(route for route in main.app.routes if route.path == "/ws/live")
        with patch("app.main._token_matches", return_value=False):
            await route.endpoint(websocket)
        self.assertEqual(websocket.closed[0], 4401)

    async def test_authenticated_satellite_stream_uses_pcm_protocol(self):
        from app import main

        class FakeSession:
            def __init__(self):
                self.inputs = []

            async def receive(self):
                await asyncio.Event().wait()
                if False:
                    yield None

            async def send_realtime_input(self, **kwargs):
                self.inputs.append(kwargs)

        class FakeAgent:
            def __init__(self):
                self.session = FakeSession()

            @asynccontextmanager
            async def live_session(self, *, speaker_token, master_passphrase):
                self.credentials = (speaker_token, master_passphrase)
                yield self.session, None

        class FakeWebSocket:
            headers = {"authorization": "Bearer test"}
            query_params = {}

            def __init__(self):
                self.app = SimpleNamespace(state=SimpleNamespace(
                    succession=None,
                    agent=FakeAgent(),
                ))
                self.received = iter([
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({
                            "client": "esp32_satellite",
                            "sample_rate": 16000,
                        }),
                    },
                    {"type": "websocket.receive", "bytes": b"\x00\x00"},
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "audio_end"}),
                    },
                    {"type": "websocket.disconnect", "code": 1000},
                ])
                self.sent_json = []

            async def accept(self):
                self.accepted = True

            async def receive(self):
                return next(self.received)

            async def send_json(self, value):
                self.sent_json.append(value)

            async def send_bytes(self, _value):
                pass

            async def close(self, **_kwargs):
                pass

        websocket = FakeWebSocket()
        route = next(route for route in main.app.routes if route.path == "/ws/live")
        with patch("app.main._token_matches", return_value=True):
            await route.endpoint(websocket)

        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.sent_json[0]["type"], "ready")
        self.assertEqual(websocket.sent_json[0]["output_audio"], "audio/pcm;rate=24000")
        self.assertEqual(
            websocket.app.state.agent.session.inputs[0]["audio"].mime_type,
            "audio/pcm;rate=16000",
        )
        self.assertEqual(
            websocket.app.state.agent.session.inputs[1],
            {"audio_stream_end": True},
        )

    async def test_live_function_call_is_returned_through_agent_dispatch(self):
        from app import main

        class FakeSession:
            def __init__(self):
                self.tool_responses = []

            async def receive(self):
                yield SimpleNamespace(
                    server_content=None,
                    tool_call=SimpleNamespace(function_calls=[
                        SimpleNamespace(
                            name="weather_report",
                            args={"location": "Colombo"},
                            id="call-1",
                        )
                    ]),
                )

            async def send_tool_response(self, *, function_responses):
                self.tool_responses.extend(function_responses)

        class FakeAgent:
            def __init__(self):
                self.session = FakeSession()
                self.dispatch_live_tool = AsyncMock(return_value="Sunny.")

            @asynccontextmanager
            async def live_session(self, *, speaker_token, master_passphrase):
                yield self.session, None

        class FakeWebSocket:
            headers = {"authorization": "Bearer test"}
            query_params = {}

            def __init__(self):
                self.app = SimpleNamespace(state=SimpleNamespace(
                    succession=None,
                    agent=FakeAgent(),
                ))
                self.received = iter([{
                    "type": "websocket.receive",
                    "text": json.dumps({"client": "web", "sample_rate": 16000}),
                }])
                self.sent_json = []
                self.initial_message = True

            async def accept(self):
                pass

            async def receive(self):
                if self.initial_message:
                    self.initial_message = False
                    return next(self.received)
                await asyncio.Future()

            async def send_json(self, value):
                self.sent_json.append(value)

            async def close(self, **_kwargs):
                pass

        websocket = FakeWebSocket()
        route = next(route for route in main.app.routes if route.path == "/ws/live")
        with patch("app.main._token_matches", return_value=True):
            await route.endpoint(websocket)

        websocket.app.state.agent.dispatch_live_tool.assert_awaited_once_with(
            "weather_report",
            {"location": "Colombo"},
            speaker_token=None,
            master_passphrase=None,
        )
        response = websocket.app.state.agent.session.tool_responses[0]
        self.assertEqual(response.response, {"result": "Sunny."})

    async def test_live_audio_output_is_forwarded_as_24khz_pcm_frames(self):
        from app import main

        pcm_chunk = b"\x01\x00\x02\x00"

        class FakeSession:
            async def receive(self):
                yield SimpleNamespace(
                    server_content=SimpleNamespace(
                        input_transcription=SimpleNamespace(text="Hello"),
                        output_transcription=SimpleNamespace(text="Hi"),
                        model_turn=SimpleNamespace(parts=[
                            SimpleNamespace(
                                inline_data=SimpleNamespace(data=pcm_chunk),
                                text=None,
                            )
                        ]),
                        interrupted=False,
                        turn_complete=True,
                    ),
                    tool_call=None,
                )

        class FakeAgent:
            @asynccontextmanager
            async def live_session(self, *, speaker_token, master_passphrase):
                yield FakeSession(), None

        class FakeWebSocket:
            headers = {"authorization": "Bearer test"}
            query_params = {}

            def __init__(self):
                self.app = SimpleNamespace(state=SimpleNamespace(
                    succession=None,
                    agent=FakeAgent(),
                ))
                self.initial_message = True
                self.sent_json = []
                self.sent_bytes = []

            async def accept(self):
                pass

            async def receive(self):
                if self.initial_message:
                    self.initial_message = False
                    return {
                        "type": "websocket.receive",
                        "text": json.dumps({"client": "web", "sample_rate": 16000}),
                    }
                await asyncio.Future()

            async def send_json(self, value):
                self.sent_json.append(value)

            async def send_bytes(self, value):
                self.sent_bytes.append(value)

            async def close(self, **_kwargs):
                pass

        websocket = FakeWebSocket()
        route = next(route for route in main.app.routes if route.path == "/ws/live")
        with patch("app.main._token_matches", return_value=True):
            await route.endpoint(websocket)

        audio_event = next(
            item for item in websocket.sent_json
            if item.get("type") == "audio_chunk"
        )
        self.assertEqual(audio_event["mime_type"], "audio/pcm;rate=24000")
        self.assertEqual(websocket.sent_bytes, [pcm_chunk])
        self.assertIn("turn_complete", [item["type"] for item in websocket.sent_json])


if __name__ == "__main__":
    unittest.main()
