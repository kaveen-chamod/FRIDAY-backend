"""Thread-safe presence state machine for the HOME_EDGE_AUDIO satellite."""
from __future__ import annotations

import math
import threading
import time
from enum import Enum
from typing import Any, Callable


class PresenceState(str, Enum):
    ACTIVE_PRESENCE = "ACTIVE_PRESENCE"
    VACANT_COUNTDOWN = "VACANT_COUNTDOWN"
    SLEEP_STANDBY = "SLEEP_STANDBY"
    WAKE_OVERRIDE = "WAKE_OVERRIDE"


class PresenceEngine:
    """Track home presence and fail closed by muting the satellite at startup."""

    VACANCY_TIMEOUT_SECONDS = 900
    _WAKE_SOURCES = frozenset(
        {"home_mic", "esp32_satellite", "pir", "master", "energy_threshold"}
    )

    def __init__(
        self,
        *,
        vacancy_timeout_seconds: float = VACANCY_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(vacancy_timeout_seconds, bool)
            or not isinstance(vacancy_timeout_seconds, (int, float))
            or not math.isfinite(vacancy_timeout_seconds)
            or vacancy_timeout_seconds <= 0
        ):
            raise ValueError("vacancy_timeout_seconds must be a positive finite number.")
        self._lock = threading.RLock()
        self._clock = clock
        self._vacancy_timeout = float(vacancy_timeout_seconds)
        self._state = PresenceState.SLEEP_STANDBY
        self._deadline: float | None = None
        self._pir_active = False
        self._person_count = 0
        self._light_lux = 0.0
        self._last_transition = self._clock()
        self._transition_reason = "privacy_safe_startup"
        self._last_wake_source: str | None = None

    def ingest_sensor_telemetry(
        self, pir_active: bool, person_count: int, light_lux: float = 0.0
    ) -> dict[str, Any]:
        if type(pir_active) is not bool:
            raise ValueError("pir_active must be a boolean.")
        if (
            isinstance(person_count, bool)
            or not isinstance(person_count, int)
            or person_count < 0
        ):
            raise ValueError("person_count must be a non-negative integer.")
        if (
            isinstance(light_lux, bool)
            or not isinstance(light_lux, (int, float))
            or not math.isfinite(light_lux)
            or light_lux < 0
        ):
            raise ValueError("light_lux must be a finite non-negative number.")

        with self._lock:
            now = self._clock()
            self._refresh_locked(now)
            self._pir_active = pir_active
            self._person_count = person_count
            self._light_lux = float(light_lux)

            if pir_active or person_count >= 1:
                self._transition_locked(
                    PresenceState.ACTIVE_PRESENCE, now, "presence_detected"
                )
                self._deadline = None
            elif self._state is PresenceState.ACTIVE_PRESENCE:
                self._transition_locked(
                    PresenceState.VACANT_COUNTDOWN, now, "presence_cleared"
                )
                self._deadline = now + self._vacancy_timeout
            elif self._state is PresenceState.VACANT_COUNTDOWN:
                self._refresh_locked(now)
            return self._snapshot_locked(now)

    def trigger_wake_word(self, source: str = "home_mic") -> dict[str, Any]:
        if not isinstance(source, str) or source not in self._WAKE_SOURCES:
            raise ValueError("Unsupported wake source.")
        with self._lock:
            now = self._clock()
            self._last_wake_source = source
            self._transition_locked(
                PresenceState.ACTIVE_PRESENCE, now, f"wake_override:{source}"
            )
            self._deadline = None
            return self._snapshot_locked(now)

    def get_presence_state(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            self._refresh_locked(now)
            return self._snapshot_locked(now)

    def is_mic_active(self) -> bool:
        with self._lock:
            self._refresh_locked(self._clock())
            return self._state in {
                PresenceState.ACTIVE_PRESENCE,
                PresenceState.VACANT_COUNTDOWN,
            }

    def force_state_override(
        self, new_state: str, master_passphrase: str | None = None
    ) -> bool:
        try:
            state = PresenceState(new_state)
        except (TypeError, ValueError):
            return False

        from app.access_control import verify_master_passphrase

        if not verify_master_passphrase(master_passphrase):
            return False

        with self._lock:
            now = self._clock()
            self._transition_locked(state, now, "master_override")
            if state is PresenceState.ACTIVE_PRESENCE:
                self._deadline = None
            elif state is PresenceState.VACANT_COUNTDOWN:
                self._deadline = now + self._vacancy_timeout
            elif state is PresenceState.SLEEP_STANDBY:
                self._deadline = None
            elif state is PresenceState.WAKE_OVERRIDE:
                self._last_wake_source = "master"
                self._transition_locked(
                    PresenceState.ACTIVE_PRESENCE, now, "wake_override:master"
                )
                self._deadline = None
            return True

    def _refresh_locked(self, now: float) -> None:
        if (
            self._state is PresenceState.VACANT_COUNTDOWN
            and self._deadline is not None
            and now >= self._deadline
        ):
            self._transition_locked(
                PresenceState.SLEEP_STANDBY, now, "vacancy_timeout_expired"
            )
            self._deadline = None

    def _transition_locked(
        self, state: PresenceState, now: float, reason: str
    ) -> None:
        if state is not self._state:
            self._state = state
            self._last_transition = now
            self._transition_reason = reason

    def _snapshot_locked(self, now: float) -> dict[str, Any]:
        remaining = (
            max(0, math.ceil(self._deadline - now))
            if self._state is PresenceState.VACANT_COUNTDOWN
            and self._deadline is not None
            else 0
        )
        return {
            "state": self._state.value,
            "remaining_seconds": remaining,
            "mic_active": self._state in {
                PresenceState.ACTIVE_PRESENCE,
                PresenceState.VACANT_COUNTDOWN,
            },
            "vacancy_timeout_seconds": self._vacancy_timeout,
            "pir_active": self._pir_active,
            "person_count": self._person_count,
            "light_lux": self._light_lux,
            "last_transition_monotonic": self._last_transition,
            "transition_reason": self._transition_reason,
            "last_wake_source": self._last_wake_source,
        }
