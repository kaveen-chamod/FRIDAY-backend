from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOG = logging.getLogger("friday.bridge")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class WebPlayerBridge:
    """UI-compatible event bridge for actions and plugins running headlessly."""

    def __init__(self) -> None:
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_file: str | None = None
        self._confirmation_lock = threading.Lock()
        self._pending_confirmation: dict[str, str] | None = None

    @property
    def current_file(self) -> str | None:
        return self._current_file

    @current_file.setter
    def current_file(self, value: str | None) -> None:
        self._current_file = value

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: Any) -> None:
        self._clients.discard(websocket)

    @property
    def pending_confirmation(self) -> dict[str, str] | None:
        with self._confirmation_lock:
            return dict(self._pending_confirmation) if self._pending_confirmation else None

    async def replay_confirmation(self, websocket: Any) -> None:
        confirmation = self.pending_confirmation
        if confirmation:
            await websocket.send_text(self._event_json(
                "confirmation_required", confirmation
            ))

    @staticmethod
    def _event_json(event_type: str, payload: Any) -> str:
        return json.dumps({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": _json_value(payload),
        }, ensure_ascii=False)

    async def broadcast(self, event_type: str, payload: Any) -> None:
        message = self._event_json(event_type, payload)
        for websocket in tuple(self._clients):
            try:
                await websocket.send_text(message)
            except Exception as exc:
                self._clients.discard(websocket)
                _LOG.warning("Removing disconnected WebSocket client: %s", exc)

    def _emit(self, event_type: str, payload: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.broadcast(event_type, payload))
            )
        except RuntimeError as exc:
            _LOG.warning("Could not publish event during shutdown: %s", exc)

    def write_log(self, text: str) -> None:
        message = str(text)
        print(message, flush=True)
        self._emit("log", {"text": message})

    def show_content(self, title: str, text: str) -> None:
        self._emit("content", {"title": title, "text": text})

    def show_weather(self, report: dict) -> None:
        self._emit("weather", report)

    def show_flight_cards(self, items, origin: str, destination: str, date: str) -> None:
        self._emit("flight_cards", {
            "items": items, "origin": origin, "destination": destination, "date": date,
        })

    def show_news_cards(self, items, title: str = "HEADLINES") -> None:
        self._emit("news_cards", {"items": items, "title": title})

    def show_images(self, images, query: str = "IMAGE RESULTS") -> None:
        self._emit("images", {"images": images, "query": query})

    def show_document(self, path: str, summary: str = "") -> None:
        self._emit("document", {"path": path, "summary": summary})

    def add_reminder_card(self, message: str, target) -> None:
        self._emit("reminder", {"message": message, "target": target})

    def start_timer(self, minutes: int, label: str = "COUNTDOWN") -> None:
        self._emit("timer", {"minutes": minutes, "label": label})

    def set_audio_level(self, level: float) -> None:
        self._emit("audio_level", {"level": max(0.0, min(1.0, float(level)))})

    def set_state(self, state: str) -> None:
        self._emit("state", {"state": str(state)})

    def show_confirmation(self, title: str, detail: str) -> None:
        confirmation = {"title": title, "detail": detail}
        with self._confirmation_lock:
            self._pending_confirmation = confirmation
        self._emit("confirmation_required", confirmation)

    def hide_confirmation(self) -> None:
        with self._confirmation_lock:
            self._pending_confirmation = None
        self._emit("confirmation_resolved", {})
