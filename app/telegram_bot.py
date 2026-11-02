from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_LOG = logging.getLogger("friday.telegram")
_ACTIVE_SERVICE: "TelegramBotService | None" = None
_SERVICE_LOCK = threading.Lock()
_MAX_WEBHOOK_BYTES = 1_048_576
_MAX_SEEN_UPDATES = 4096
_WEBHOOK_PATH = "/api/telegram-webhook"
_WEBHOOK_SECRET_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,256}\Z")


def _webhook_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("FRIDAY_TELEGRAM_WEBHOOK_URL is not a valid URL.") from exc
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "FRIDAY_TELEGRAM_WEBHOOK_URL must be a public HTTPS URL without credentials, query, or fragment."
        )
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        if (
            "." not in hostname
            or not re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", hostname
            )
            or ".." in hostname
            or hostname.lower() == "localhost"
            or hostname.lower().endswith((".localhost", ".local", ".internal"))
        ):
            raise ValueError("Telegram webhook URL must use a public hostname.")
    else:
        if not address.is_global:
            raise ValueError("Telegram webhook URL must use a public hostname.")
    path = parsed.path.rstrip("/")
    if not path.endswith(_WEBHOOK_PATH):
        path += _WEBHOOK_PATH
    return urlunsplit(("https", parsed.netloc, path, "", ""))


def resolve_telegram_mode(
    mode: str | None = None, webhook_url: str | None = None
) -> tuple[str, str | None]:
    """Resolve and validate Telegram's polling or webhook startup mode."""
    selected = (mode or os.environ.get("FRIDAY_TELEGRAM_MODE", "auto")).strip().lower()
    if selected not in {"auto", "polling", "webhook"}:
        raise ValueError("FRIDAY_TELEGRAM_MODE must be auto, polling, or webhook.")
    configured_url = (
        os.environ.get("FRIDAY_TELEGRAM_WEBHOOK_URL", "")
        if webhook_url is None else webhook_url
    ).strip()
    endpoint = _webhook_endpoint(configured_url) if configured_url else None
    if selected == "webhook":
        if endpoint is None:
            raise ValueError(
                "FRIDAY_TELEGRAM_WEBHOOK_URL is required in webhook mode."
            )
        return "webhook", endpoint
    if selected == "polling":
        return "polling", None
    return ("webhook", endpoint) if endpoint else ("polling", None)


def _parse_allowed_user_ids(value: str) -> frozenset[int]:
    entries = [item.strip() for item in value.split(",") if item.strip()]
    if any(not re.fullmatch(r"[1-9]\d*", item) for item in entries):
        raise ValueError(
            "FRIDAY_TELEGRAM_ALLOWED_USER_IDS must contain comma-separated positive Telegram user IDs."
        )
    return frozenset(int(item) for item in entries)


class TelegramBotService:
    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.application = None
        self._running = False
        self._mode: str | None = None
        self._application_initialized = False
        self._application_started = False
        self._polling_started = False
        self._allowed_user_ids = _parse_allowed_user_ids(
            os.environ.get("FRIDAY_TELEGRAM_ALLOWED_USER_IDS", "")
        )
        self._alert_chat_ids = [
            item.strip()
            for item in os.environ.get("TELEGRAM_ALERT_CHAT_IDS", "").split(",")
            if item.strip()
        ]
        self._webhook_secret = ""
        self._processed_updates: OrderedDict[int, None] = OrderedDict()
        self._update_locks: dict[int, asyncio.Lock] = {}
        self._updates_lock = asyncio.Lock()

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def running(self) -> bool:
        return self._running

    def _is_authorized(self, update: Any) -> bool:
        user = getattr(update, "effective_user", None)
        chat = getattr(update, "effective_chat", None)
        return bool(
            user
            and chat
            and getattr(chat, "type", None) == "private"
            and user.id in self._allowed_user_ids
        )

    async def _handle_voice_update(self, update: Any) -> None:
        message = None
        try:
            message = getattr(update, "effective_message", None)
            if message is None or message.voice is None:
                return
            if not self._is_authorized(update):
                await message.reply_text(
                    "This bot is not authorized to accept messages from this account."
                )
                return
            if message.voice.file_size and message.voice.file_size > 20 * 1024 * 1024:
                await message.reply_text("That voice note is too large to process.")
                return

            session = f"telegram-{update.effective_chat.id}"
            telegram_file = await message.voice.get_file()
            audio = bytes(await telegram_file.download_as_bytearray())
            result = await self.agent.chat_audio(audio, "audio/ogg", session)
            reply_text = str(result.get("response") or "")
            spoken_audio = result.get("audio")
            spoken_mime_type = str(result.get("audio_mime_type") or "")
            if not isinstance(spoken_audio, bytes) or not spoken_audio:
                if reply_text:
                    await message.reply_text(
                        f"Sir, here is the response:\n\n{reply_text}"
                    )
                else:
                    await message.reply_text(
                        "F.R.I.D.A.Y. could not generate a voice or text response."
                    )
                return

            from telegram import InputFile
            from app.voice import decode_to_pcm_wav, encode_ogg_opus

            try:
                if spoken_mime_type.lower().startswith("audio/mpeg"):
                    spoken_audio = await asyncio.to_thread(
                        decode_to_pcm_wav,
                        spoken_audio,
                        mime_type=spoken_mime_type,
                    )
                    spoken_mime_type = "audio/wav"
                ogg = await asyncio.to_thread(
                    encode_ogg_opus,
                    spoken_audio,
                    spoken_mime_type,
                )
            except Exception:
                _LOG.exception("Could not encode Telegram voice response.")
                if reply_text:
                    await message.reply_text(
                        f"Sir, here is the response:\n\n{reply_text}"
                    )
                else:
                    await message.reply_text(
                        "F.R.I.D.A.Y. could not deliver a voice response."
                    )
                return
            if reply_text:
                await message.reply_text(reply_text)
            await message.reply_voice(
                voice=InputFile(ogg, filename="friday.ogg")
            )
        except Exception:
            _LOG.exception("Telegram voice request failed.")
            if message is not None:
                try:
                    await message.reply_text(
                        "F.R.I.D.A.Y. could not process that voice note. "
                        "Please try again."
                    )
                except Exception:
                    _LOG.exception(
                        "Telegram could not deliver the voice-processing error message."
                    )

    async def start(self) -> None:
        global _ACTIVE_SERVICE
        mode, webhook_url = resolve_telegram_mode()
        webhook_secret = os.environ.get(
            "FRIDAY_TELEGRAM_WEBHOOK_SECRET", ""
        ).strip()
        if webhook_secret and not _WEBHOOK_SECRET_PATTERN.fullmatch(webhook_secret):
            raise ValueError(
                "FRIDAY_TELEGRAM_WEBHOOK_SECRET must be 1-256 URL-safe characters."
            )
        if mode == "webhook" and not webhook_secret:
            raise ValueError(
                "FRIDAY_TELEGRAM_WEBHOOK_SECRET is required in webhook mode."
            )
        token = (
            os.environ.get("FRIDAY_TELEGRAM_BOT_TOKEN")
            or os.environ.get("TELEGRAM_BOT_TOKEN")
            or ""
        ).strip()
        if not token:
            _LOG.info("Telegram bot is disabled; no bot token is configured.")
            return
        if not self._allowed_user_ids:
            _LOG.warning(
                "Telegram inbound messages are denied until "
                "FRIDAY_TELEGRAM_ALLOWED_USER_IDS is configured."
            )

        with _SERVICE_LOCK:
            if _ACTIVE_SERVICE is self and self._running:
                return
            if _ACTIVE_SERVICE is not None:
                raise RuntimeError("Another Telegram background service is already active.")
            _ACTIVE_SERVICE = self

        try:
            from telegram import Update
            from telegram.error import TelegramError
            from telegram.ext import (
                ApplicationBuilder,
                ContextTypes,
                MessageHandler,
                filters,
            )
            from telegram.request import HTTPXRequest
        except ImportError as exc:
            with _SERVICE_LOCK:
                if _ACTIVE_SERVICE is self:
                    _ACTIVE_SERVICE = None
            raise RuntimeError(
                "Telegram bot token is configured, but python-telegram-bot is not installed."
            ) from exc

        def make_request() -> HTTPXRequest:
            return HTTPXRequest(
                connection_pool_size=16,
                connect_timeout=10,
                read_timeout=60,
                write_timeout=20,
                pool_timeout=10,
                http_version="1.1",
            )

        application = None

        async def handle_text(
            update: Update, _context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            message = update.effective_message
            if message is None or not message.text:
                return
            if not self._is_authorized(update):
                await message.reply_text(
                    "This bot is not authorized to accept messages from this account."
                )
                return
            session = f"telegram-{update.effective_chat.id}"
            try:
                response = await self.agent.chat_text(message.text, session)
                await message.reply_text(response)
            except Exception:
                _LOG.exception("Telegram text request failed.")
                await message.reply_text(
                    "F.R.I.D.A.Y. could not complete that request. Please try again."
                )

        async def handle_voice(
            update: Update, _context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            await self._handle_voice_update(update)

        try:
            builder = (
                ApplicationBuilder()
                .token(token)
                .request(make_request())
                .get_updates_request(make_request())
            )
            application = builder.build()
            self.application = application
            self._webhook_secret = webhook_secret
            application.add_handler(MessageHandler(filters.VOICE, handle_voice))
            application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            )

            async def handle_telegram_error(_update, context) -> None:
                _LOG.error(
                    "Telegram update handler failed.",
                    exc_info=context.error,
                )

            application.add_error_handler(handle_telegram_error)
            await application.initialize()
            self._application_initialized = True

            if mode == "webhook":
                webhook_set = await application.bot.set_webhook(
                    url=webhook_url,
                    secret_token=webhook_secret,
                    allowed_updates=["message"],
                    drop_pending_updates=False,
                )
                if webhook_set is False:
                    raise RuntimeError("Telegram rejected the webhook registration.")
            else:
                if application.updater is None:
                    raise RuntimeError(
                        "Telegram polling is unavailable: the Application has no Updater."
                    )
                webhook_deleted = await application.bot.delete_webhook(
                    drop_pending_updates=False
                )
                if webhook_deleted is False:
                    raise RuntimeError(
                        "Telegram rejected removal of the webhook before polling."
                    )
                await application.updater.start_polling(
                    poll_interval=1.0,
                    timeout=30,
                    bootstrap_retries=2,
                    allowed_updates=["message"],
                    drop_pending_updates=False,
                    error_callback=lambda error: _LOG.warning(
                        "Telegram long-poll request failed: %s", error
                    ),
                )
                self._polling_started = True

            await application.start()
            self._application_started = True
            self._mode = mode
            self._running = True
            _LOG.info("Telegram service started in %s mode.", mode)
        except Exception as exc:
            _LOG.exception("Telegram %s startup failed.", mode)
            try:
                await self._cleanup_application()
            except Exception:
                _LOG.exception("Telegram startup cleanup was incomplete.")
            finally:
                with _SERVICE_LOCK:
                    if _ACTIVE_SERVICE is self:
                        _ACTIVE_SERVICE = None
            if isinstance(exc, (TelegramError, TimeoutError, OSError)):
                raise RuntimeError(
                    f"Telegram {mode} startup failed because Telegram is unreachable or rejected the request."
                ) from exc
            raise

    async def stop(self) -> None:
        global _ACTIVE_SERVICE
        if self.application is None:
            return
        try:
            await self._cleanup_application()
        finally:
            self._running = False
            self._mode = None
            with _SERVICE_LOCK:
                if _ACTIVE_SERVICE is self:
                    _ACTIVE_SERVICE = None

    async def _cleanup_application(self) -> None:
        application = self.application
        if application is None:
            return
        first_error: Exception | None = None
        if self._polling_started and application.updater is not None:
            try:
                await application.updater.stop()
            except Exception as exc:
                first_error = exc
                _LOG.exception("Could not stop Telegram long polling.")
            finally:
                self._polling_started = False
        if self._application_started:
            try:
                await application.stop()
            except Exception as exc:
                first_error = first_error or exc
                _LOG.exception("Could not stop Telegram application tasks.")
            finally:
                self._application_started = False
        if self._application_initialized:
            try:
                await application.shutdown()
            except Exception as exc:
                first_error = first_error or exc
                _LOG.exception("Could not shut down Telegram application.")
            finally:
                self._application_initialized = False
        self.application = None
        self._running = False
        if first_error is not None:
            raise RuntimeError("Telegram service shutdown was incomplete.") from first_error

    async def process_webhook(self, body: bytes, secret_token: str | None) -> bool:
        if not self._running or self.application is None:
            raise RuntimeError("Telegram webhook service is unavailable.")
        if self._mode != "webhook":
            raise RuntimeError("Telegram webhook updates are disabled in polling mode.")
        if not hmac.compare_digest(self._webhook_secret, secret_token or ""):
            raise PermissionError("Telegram webhook authentication failed.")
        if not body or len(body) > _MAX_WEBHOOK_BYTES:
            raise ValueError("Telegram webhook payload is empty or too large.")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Telegram webhook payload must be valid JSON.") from exc
        if (
            not isinstance(payload, dict)
            or type(payload.get("update_id")) is not int
            or payload["update_id"] < 0
        ):
            raise ValueError("Telegram webhook payload has no valid update_id.")
        update_id = payload["update_id"]
        async with self._updates_lock:
            if update_id in self._processed_updates:
                return False
            update_lock = self._update_locks.setdefault(update_id, asyncio.Lock())

        async with update_lock:
            try:
                async with self._updates_lock:
                    if update_id in self._processed_updates:
                        return False
                from telegram import Update

                try:
                    update = Update.de_json(payload, self.application.bot)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "Telegram webhook payload contains an invalid update."
                    ) from exc
                if not self._is_authorized(update):
                    _LOG.warning(
                        "Discarding unauthorized Telegram webhook update %s.",
                        update_id,
                    )
                    async with self._updates_lock:
                        self._remember_update(update_id)
                    return False
                await self.application.process_update(update)
                async with self._updates_lock:
                    self._remember_update(update_id)
                return True
            except Exception:
                _LOG.exception("Telegram webhook update processing failed.")
                raise
            finally:
                async with self._updates_lock:
                    if self._update_locks.get(update_id) is update_lock:
                        self._update_locks.pop(update_id, None)

    def _remember_update(self, update_id: int) -> None:
        self._processed_updates[update_id] = None
        self._processed_updates.move_to_end(update_id)
        while len(self._processed_updates) > _MAX_SEEN_UPDATES:
            self._processed_updates.popitem(last=False)

    async def send_alert(self, message: str) -> None:
        if not self._running or self.application is None:
            raise RuntimeError("Telegram alerts are unavailable because the bot is not running.")
        if not self._alert_chat_ids:
            raise RuntimeError("Set TELEGRAM_ALERT_CHAT_IDS to receive Telegram alerts.")
        for chat_id in self._alert_chat_ids:
            await self.application.bot.send_message(chat_id=chat_id, text=message)


async def send_telegram_alert(message: str) -> None:
    if _ACTIVE_SERVICE is None:
        raise RuntimeError("Telegram alerts are unavailable because the bot is not running.")
    await _ACTIVE_SERVICE.send_alert(message)
