from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.telegram_bot import TelegramBotService, resolve_telegram_mode


_WEBHOOK_SECRET = "a-valid-webhook-secret"
_WEBHOOK_BASE = "https://friday.example.com/"


def _telegram_environment(**overrides):
    values = {
        "FRIDAY_TELEGRAM_BOT_TOKEN": "test-token",
        "FRIDAY_TELEGRAM_ALLOWED_USER_IDS": "12345",
        "FRIDAY_TELEGRAM_MODE": "polling",
        "FRIDAY_TELEGRAM_WEBHOOK_URL": "",
        "FRIDAY_TELEGRAM_WEBHOOK_SECRET": "",
    }
    values.update(overrides)
    return patch.dict(os.environ, values)


def _authorized_update(update_id: int = 25):
    return SimpleNamespace(
        update_id=update_id,
        effective_user=SimpleNamespace(id=12345),
        effective_chat=SimpleNamespace(id=12345, type="private"),
    )


def _mock_application():
    events: list[str] = []
    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=lambda **kwargs: events.append("poll_start")),
        stop=AsyncMock(side_effect=lambda: events.append("poll_stop")),
    )
    bot = SimpleNamespace(
        set_webhook=AsyncMock(
            side_effect=lambda **kwargs: events.append("set_webhook") or True
        ),
        delete_webhook=AsyncMock(
            side_effect=lambda **kwargs: events.append("delete_webhook") or True
        ),
    )
    application = SimpleNamespace(
        bot=bot,
        updater=updater,
        initialize=AsyncMock(side_effect=lambda: events.append("initialize")),
        start=AsyncMock(side_effect=lambda: events.append("application_start")),
        stop=AsyncMock(side_effect=lambda: events.append("application_stop")),
        shutdown=AsyncMock(side_effect=lambda: events.append("shutdown")),
        add_handler=Mock(),
        add_error_handler=Mock(),
        process_update=AsyncMock(),
    )
    builder = Mock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = application
    return application, builder, events


class TelegramModeTests(unittest.TestCase):
    def test_auto_selects_polling_without_webhook_url(self):
        self.assertEqual(resolve_telegram_mode("auto", ""), ("polling", None))

    def test_auto_and_webhook_modes_normalize_public_https_urls(self):
        self.assertEqual(
            resolve_telegram_mode("auto", _WEBHOOK_BASE),
            ("webhook", "https://friday.example.com/api/telegram-webhook"),
        )
        self.assertEqual(
            resolve_telegram_mode(
                "webhook", "https://friday.example.com/api/telegram-webhook"
            ),
            ("webhook", "https://friday.example.com/api/telegram-webhook"),
        )

    def test_webhook_mode_requires_valid_public_https_url(self):
        for value in ("", "http://friday.example.com", "https://localhost", "https://127.0.0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_telegram_mode("webhook", value)

    def test_invalid_mode_and_malformed_webhook_url_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_telegram_mode("longpoll", "")
        with self.assertRaises(ValueError):
            resolve_telegram_mode("auto", "https://friday.example.com/path?token=x")


class TelegramLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        from app import telegram_bot

        service = telegram_bot._ACTIVE_SERVICE
        if service is not None:
            await service.stop()

    def _patch_builder(self, builder):
        return patch("telegram.ext.ApplicationBuilder", return_value=builder)

    async def test_local_polling_starts_without_webhook_and_stops_cleanly(self):
        application, builder, events = _mock_application()
        with (
            _telegram_environment(FRIDAY_TELEGRAM_MODE="polling"),
            self._patch_builder(builder),
        ):
            service = TelegramBotService(agent=object())
            await service.start()
            self.assertEqual(service.mode, "polling")
            self.assertEqual(
                events,
                ["initialize", "delete_webhook", "poll_start", "application_start"],
            )
            application.bot.set_webhook.assert_not_awaited()
            application.updater.start_polling.assert_awaited_once()
            await service.stop()
            self.assertEqual(
                events[-3:], ["poll_stop", "application_stop", "shutdown"]
            )

    async def test_auto_mode_uses_polling_when_webhook_url_is_empty(self):
        application, builder, events = _mock_application()
        with (
            _telegram_environment(FRIDAY_TELEGRAM_MODE="auto"),
            self._patch_builder(builder),
        ):
            service = TelegramBotService(agent=object())
            await service.start()
            self.assertEqual(service.mode, "polling")
            self.assertIn("poll_start", events)
            await service.stop()

    async def test_cloud_webhook_registers_url_and_never_starts_polling(self):
        application, builder, events = _mock_application()
        with (
            _telegram_environment(
                FRIDAY_TELEGRAM_MODE="webhook",
                FRIDAY_TELEGRAM_WEBHOOK_URL=_WEBHOOK_BASE,
                FRIDAY_TELEGRAM_WEBHOOK_SECRET=_WEBHOOK_SECRET,
            ),
            self._patch_builder(builder),
        ):
            service = TelegramBotService(agent=object())
            await service.start()
            self.assertEqual(service.mode, "webhook")
            application.bot.set_webhook.assert_awaited_once_with(
                url="https://friday.example.com/api/telegram-webhook",
                secret_token=_WEBHOOK_SECRET,
                allowed_updates=["message"],
                drop_pending_updates=False,
            )
            application.updater.start_polling.assert_not_awaited()
            self.assertNotIn("poll_start", events)
            await service.stop()

    async def test_auto_mode_uses_webhook_when_public_url_is_configured(self):
        application, builder, events = _mock_application()
        with (
            _telegram_environment(
                FRIDAY_TELEGRAM_MODE="auto",
                FRIDAY_TELEGRAM_WEBHOOK_URL=_WEBHOOK_BASE,
                FRIDAY_TELEGRAM_WEBHOOK_SECRET=_WEBHOOK_SECRET,
            ),
            self._patch_builder(builder),
        ):
            service = TelegramBotService(agent=object())
            await service.start()
            self.assertEqual(service.mode, "webhook")
            application.updater.start_polling.assert_not_awaited()
            await service.stop()

    async def test_webhook_mode_requires_a_valid_secret(self):
        for secret, error in (
            ("", "SECRET is required"),
            ("contains spaces", "URL-safe"),
        ):
            _, builder, _ = _mock_application()
            with (
                self.subTest(secret=secret),
                _telegram_environment(
                    FRIDAY_TELEGRAM_MODE="webhook",
                    FRIDAY_TELEGRAM_WEBHOOK_URL=_WEBHOOK_BASE,
                    FRIDAY_TELEGRAM_WEBHOOK_SECRET=secret,
                ),
                self._patch_builder(builder),
            ):
                service = TelegramBotService(agent=object())
                with self.assertRaisesRegex(ValueError, error):
                    await service.start()
                builder.build.assert_not_called()

    async def test_webhook_registration_failure_does_not_fall_back_to_polling(self):
        application, builder, _ = _mock_application()
        application.bot.set_webhook.side_effect = RuntimeError("registration rejected")
        with (
            _telegram_environment(
                FRIDAY_TELEGRAM_MODE="webhook",
                FRIDAY_TELEGRAM_WEBHOOK_URL=_WEBHOOK_BASE,
                FRIDAY_TELEGRAM_WEBHOOK_SECRET=_WEBHOOK_SECRET,
            ),
            self._patch_builder(builder),
        ):
            service = TelegramBotService(agent=object())
            with self.assertRaisesRegex(RuntimeError, "registration rejected"):
                await service.start()
            application.updater.start_polling.assert_not_awaited()
            application.shutdown.assert_awaited_once()

    async def test_service_is_idempotent_and_rejects_a_second_instance(self):
        application, builder, _ = _mock_application()
        with (
            _telegram_environment(FRIDAY_TELEGRAM_MODE="polling"),
            self._patch_builder(builder),
        ):
            service = TelegramBotService(agent=object())
            await service.start()
            await service.start()
            self.assertEqual(builder.build.call_count, 1)
            duplicate = TelegramBotService(agent=object())
            with self.assertRaisesRegex(RuntimeError, "already active"):
                await duplicate.start()
            await service.stop()


class TelegramWebhookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = _telegram_environment()
        self.env.start()
        self.addCleanup(self.env.stop)
        self.service = TelegramBotService(agent=object())
        self.service._running = True
        self.service._mode = "webhook"
        self.service._webhook_secret = _WEBHOOK_SECRET
        self.application = SimpleNamespace(
            bot=object(),
            process_update=AsyncMock(),
        )
        self.service.application = self.application
        self.payload = json.dumps({"update_id": 25, "message": {}}).encode()

    async def test_valid_secret_processes_update_once_and_deduplicates_retries(self):
        update = _authorized_update()
        with patch("telegram.Update.de_json", return_value=update):
            self.assertTrue(await self.service.process_webhook(
                self.payload, _WEBHOOK_SECRET
            ))
            self.assertFalse(await self.service.process_webhook(
                self.payload, _WEBHOOK_SECRET
            ))
        self.application.process_update.assert_awaited_once_with(update)

    async def test_invalid_secret_and_malformed_payload_are_rejected(self):
        with self.assertRaises(PermissionError):
            await self.service.process_webhook(self.payload, "wrong")
        with self.assertRaises(ValueError):
            await self.service.process_webhook(b"{bad", _WEBHOOK_SECRET)
        with self.assertRaises(ValueError):
            await self.service.process_webhook(b"{}", _WEBHOOK_SECRET)

    async def test_unauthorized_webhook_update_is_acknowledgeable_and_never_runs(self):
        unauthorized = SimpleNamespace(
            update_id=25,
            effective_user=SimpleNamespace(id=99999),
            effective_chat=SimpleNamespace(id=99999, type="private"),
        )
        with patch("telegram.Update.de_json", return_value=unauthorized):
            accepted = await self.service.process_webhook(
                self.payload, _WEBHOOK_SECRET
            )
        self.assertFalse(accepted)
        self.application.process_update.assert_not_awaited()
        self.assertIn(25, self.service._processed_updates)

    async def test_fastapi_acknowledges_unauthorized_updates_with_http_200(self):
        import httpx

        from app.main import app

        unauthorized = SimpleNamespace(
            update_id=25,
            effective_user=SimpleNamespace(id=99999),
            effective_chat=SimpleNamespace(id=99999, type="private"),
        )
        previous = getattr(app.state, "telegram", None)
        app.state.telegram = self.service
        try:
            with patch("telegram.Update.de_json", return_value=unauthorized):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/api/telegram-webhook",
                        content=self.payload,
                        headers={
                            "X-Telegram-Bot-Api-Secret-Token": _WEBHOOK_SECRET,
                        },
                    )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ignored_or_duplicate"})
        finally:
            if previous is None:
                delattr(app.state, "telegram")
            else:
                app.state.telegram = previous

    async def test_processing_error_is_not_acknowledged_and_can_be_retried(self):
        update = _authorized_update()
        self.application.process_update.side_effect = [
            RuntimeError("temporary handler failure"), None,
        ]
        with patch("telegram.Update.de_json", return_value=update):
            with self.assertRaises(RuntimeError):
                await self.service.process_webhook(self.payload, _WEBHOOK_SECRET)
            self.assertTrue(await self.service.process_webhook(
                self.payload, _WEBHOOK_SECRET
            ))
        self.assertIn(25, self.service._processed_updates)

    async def test_only_allowlisted_private_telegram_users_are_authorized(self):
        def update(user_id, chat_type="private"):
            return SimpleNamespace(
                effective_user=SimpleNamespace(id=user_id),
                effective_chat=SimpleNamespace(type=chat_type),
            )

        self.assertTrue(self.service._is_authorized(update(12345)))
        self.assertFalse(self.service._is_authorized(update(99999)))
        self.assertFalse(self.service._is_authorized(update(12345, "group")))

    async def test_webhook_processing_is_disabled_in_polling_mode(self):
        self.service._mode = "polling"
        with self.assertRaisesRegex(RuntimeError, "disabled in polling mode"):
            await self.service.process_webhook(self.payload, _WEBHOOK_SECRET)


if __name__ == "__main__":
    unittest.main()
