from __future__ import annotations

import asyncio
import importlib
import io
import logging
import os
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx


class AudioStartupResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_and_health_start_with_ffmpeg_provisioning_broken(
        self,
    ) -> None:
        from app import main
        import static_ffmpeg
        from fastapi import FastAPI

        class FakeSimulation:
            def __init__(self, *, on_completed) -> None:
                self.on_completed = on_completed

            async def start(self) -> None:
                return None

            async def shutdown(self) -> None:
                return None

        class FakeReflection:
            def record_simulation_prediction(self, _event) -> None:
                return None

        class FakeTwin:
            def __init__(self, *, predictive_service) -> None:
                self.predictive_service = predictive_service

            def register_adapter(self, _adapter) -> None:
                return None

            def bind_loop(self, _loop) -> None:
                return None

            def status(self) -> dict:
                return {"devices": []}

        class FakeAdapter:
            domain = "robotics"

        class FakeDefense:
            def __init__(self, *_args, **_kwargs) -> None:
                return None

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        class FakeTelegram:
            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        fake_agent = SimpleNamespace(succession=None)
        application = FastAPI()
        with (
            patch.object(main, "_API_TOKEN", "test-token-that-is-at-least-32-chars"),
            patch.object(main, "SelfReflectionLedger", return_value=FakeReflection()),
            patch.object(main, "SimulationJobManager", FakeSimulation),
            patch.object(main, "DigitalTwinEngine", FakeTwin),
            patch.object(main, "RoboticsTwinAdapter", FakeAdapter),
            patch.object(main, "bind_default_engine"),
            patch.object(main, "FridayAgent", return_value=fake_agent),
            patch.object(main, "AuthorizedAssetRegistry"),
            patch.object(main, "IncidentLedger"),
            patch.object(main, "ThreatMatrix"),
            patch.object(main, "DefenseOrchestrator", FakeDefense),
            patch.object(main, "bind_default_orchestrator"),
            patch.object(main.security, "bind"),
            patch.object(main.confirm_gate, "bind"),
            patch(
                "app.telegram_bot.TelegramBotService",
                return_value=FakeTelegram(),
            ),
            patch.object(
                static_ffmpeg,
                "add_paths",
                side_effect=OSError("simulated offline asset host"),
            ) as provision,
        ):
            async with main.lifespan(application):
                transport = httpx.ASGITransport(app=main.app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    response = await client.get("/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "ok")
                self.assertIsNotNone(application.state.presence_engine)

        provision.assert_not_called()

    async def test_main_import_and_health_do_not_provision_ffmpeg(self) -> None:
        from app import main, voice
        import static_ffmpeg

        with patch.object(
            static_ffmpeg,
            "add_paths",
            side_effect=OSError("simulated offline asset host"),
        ) as provision:
            importlib.reload(voice)
            importlib.reload(main)
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        provision.assert_not_called()

    async def test_lazy_static_ffmpeg_failure_logs_and_returns_audio_error(self) -> None:
        from app import voice
        import static_ffmpeg

        previous_paths = voice._AUDIO_TOOL_PATHS
        previous_failure = voice._STATIC_FFMPEG_PROVISION_FAILED
        voice._AUDIO_TOOL_PATHS = None
        voice._STATIC_FFMPEG_PROVISION_FAILED = False
        try:
            with (
                patch("app.voice.shutil.which", return_value=None),
                patch.object(
                    static_ffmpeg,
                    "add_paths",
                    side_effect=OSError("offline TLS/asset download failure"),
                ) as provision,
                self.assertLogs("friday.voice", logging.WARNING) as captured,
            ):
                with self.assertRaisesRegex(RuntimeError, "FFmpeg is unavailable"):
                    voice.ffmpeg_executable()
            provision.assert_called_once_with()
            self.assertIn("Automated static FFmpeg provisioning failed", captured.output[0])
            self.assertIn("Server proceeding with runtime fallbacks", captured.output[0])
        finally:
            voice._AUDIO_TOOL_PATHS = previous_paths
            voice._STATIC_FFMPEG_PROVISION_FAILED = previous_failure

    async def test_native_pcm_fallback_decodes_without_ffmpeg(self) -> None:
        from app.voice import decode_to_pcm_wav

        samples = b"\x01\x00" * 800
        with patch("app.voice.ffmpeg_executable", side_effect=AssertionError):
            output = decode_to_pcm_wav(
                samples, mime_type="audio/pcm;rate=16000"
            )
        with wave.open(io.BytesIO(output), "rb") as decoded:
            self.assertEqual(decoded.getnchannels(), 1)
            self.assertEqual(decoded.getsampwidth(), 2)
            self.assertEqual(decoded.getframerate(), 16000)
            self.assertEqual(decoded.readframes(decoded.getnframes()), samples)

    async def test_native_pcm_wav_passes_through_without_ffmpeg(self) -> None:
        from app.voice import decode_to_pcm_wav

        output = io.BytesIO()
        samples = b"\x02\x00" * 160
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(samples)
        encoded = output.getvalue()
        with patch("app.voice.ffmpeg_executable", side_effect=AssertionError):
            self.assertEqual(
                decode_to_pcm_wav(encoded, mime_type="audio/wav"), encoded
            )


class StartupRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_failure_is_logged_and_retried_without_startup_error(
        self,
    ) -> None:
        from app import main

        service = SimpleNamespace(
            start=AsyncMock(
                side_effect=[RuntimeError("webhook endpoint unavailable"), None]
            )
        )
        with (
            patch("app.main.asyncio.sleep", new=AsyncMock()),
            self.assertLogs("friday.server", logging.WARNING) as captured,
        ):
            retry_task = await main._start_telegram_or_schedule_retry(service)
            self.assertIsNotNone(retry_task)
            await retry_task

        self.assertEqual(service.start.await_count, 2)
        self.assertIn("API will remain available", captured.output[0])

    async def test_cloud_port_forces_public_bind_even_if_local_host_is_set(
        self,
    ) -> None:
        from app import main

        with (
            patch.dict(
                os.environ,
                {"PORT": "8123", "FRIDAY_HOST": "127.0.0.1"},
            ),
            patch("uvicorn.run") as run_server,
        ):
            main.run()

        self.assertEqual(run_server.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(run_server.call_args.kwargs["port"], 8123)


if __name__ == "__main__":
    unittest.main()
