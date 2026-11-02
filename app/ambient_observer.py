"""Explicit, privacy-bounded observation of submitted Python buffer events."""
from __future__ import annotations

import ast
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from memory.memory_manager import redact_sensitive_content


_LOG = logging.getLogger("friday.ambient")
MAX_CODE_CHARS = 16_384
MAX_EVENT_ID_CHARS = 128
MAX_TRACKED_EVENTS = 2_048
MAX_REQUESTS_PER_MINUTE = 30
ALERT_INTERVAL_SECONDS = 60.0
_INTERVENTION_THRESHOLD = 0.50


class ObservationConsentRequired(PermissionError):
    """Raised when an observation is submitted without current opt-in."""


class ObservationRateLimited(RuntimeError):
    """Raised when a profile exceeds the bounded observation request rate."""


@dataclass(frozen=True)
class ObservationResult:
    status: str
    alert: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "intervention": self.alert}


class AmbientObserver:
    """Analyze only explicitly submitted events; never starts capture or polling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consented_profiles: set[str] = set()
        self._requests: dict[str, deque[float]] = {}
        self._last_alert: dict[str, float] = {}
        self._seen_events: OrderedDict[tuple[str, str], str] = OrderedDict()

    def set_consent(self, profile_id: str, enabled: bool) -> bool:
        profile_id = self._validate_profile(profile_id)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")
        with self._lock:
            if enabled:
                self._consented_profiles.add(profile_id)
            else:
                self._consented_profiles.discard(profile_id)
                self._requests.pop(profile_id, None)
                self._last_alert.pop(profile_id, None)
                for key in [key for key in self._seen_events if key[0] == profile_id]:
                    del self._seen_events[key]
        return enabled

    def consent_status(self, profile_id: str) -> bool:
        profile_id = self._validate_profile(profile_id)
        with self._lock:
            return profile_id in self._consented_profiles

    def observe_code(
        self,
        profile_id: str,
        event_id: str,
        source: str,
        content: str,
        *,
        now: float | None = None,
    ) -> ObservationResult:
        profile_id = self._validate_profile(profile_id)
        with self._lock:
            if profile_id not in self._consented_profiles:
                raise ObservationConsentRequired("Explicit observer consent is required.")
        if source != "python_code_buffer":
            raise ValueError("Only explicitly submitted Python code-buffer events are supported.")
        if not isinstance(event_id, str) or not event_id.strip() or len(event_id) > MAX_EVENT_ID_CHARS:
            raise ValueError("event_id must contain 1 to 128 characters.")
        if not isinstance(content, str) or not content.strip() or len(content) > MAX_CODE_CHARS:
            raise ValueError(f"content must contain 1 to {MAX_CODE_CHARS} characters.")

        current = time.monotonic() if now is None else now
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._lock:
            if profile_id not in self._consented_profiles:
                raise ObservationConsentRequired("Explicit observer consent is required.")
            requests = self._requests.setdefault(profile_id, deque())
            while requests and current - requests[0] >= 60.0:
                requests.popleft()
            if len(requests) >= MAX_REQUESTS_PER_MINUTE:
                raise ObservationRateLimited("Observation request rate limit exceeded.")
            requests.append(current)

            key = (profile_id, event_id)
            prior_fingerprint = self._seen_events.get(key)
            if prior_fingerprint is not None:
                if prior_fingerprint != fingerprint:
                    raise ValueError("event_id has already been used for different content.")
                self._seen_events.move_to_end(key)
                return ObservationResult("duplicate")
            self._seen_events[key] = fingerprint
            while len(self._seen_events) > MAX_TRACKED_EVENTS:
                self._seen_events.popitem(last=False)

        try:
            ast.parse(content, filename="<submitted-buffer>", mode="exec")
        except SyntaxError as exc:
            line = max(1, int(exc.lineno or 1))
            safe_message = redact_sensitive_content(str(exc.msg or "Invalid syntax"))[:160]
            confidence = 0.99
            severity = 0.9
            intrusiveness_penalty = 0.25
            score = severity * confidence * (1.0 - intrusiveness_penalty)
            if confidence < 0.85 or score < _INTERVENTION_THRESHOLD:
                with self._lock:
                    if profile_id not in self._consented_profiles:
                        raise ObservationConsentRequired(
                            "Explicit observer consent is required."
                        )
                return ObservationResult("no_alert")
            with self._lock:
                if profile_id not in self._consented_profiles:
                    raise ObservationConsentRequired("Explicit observer consent is required.")
                previous = self._last_alert.get(profile_id)
                if previous is not None and current - previous < ALERT_INTERVAL_SECONDS:
                    return ObservationResult("rate_limited")
                self._last_alert[profile_id] = current
            _LOG.info("Explicit Python buffer observation found syntax issue at line %d.", line)
            return ObservationResult(
                "alert",
                {
                    "category": "python_syntax",
                    "message": f"Potential Python syntax issue on line {line}: {safe_message}.",
                    "confidence": confidence,
                    "severity": severity,
                    "intervention_score": score,
                    "evidence": "Python AST parser diagnostic; source text was not retained.",
                    "advisory_only": True,
                },
            )
        except RecursionError:
            _LOG.warning("Explicit Python buffer exceeded parser nesting limits.")
            with self._lock:
                if profile_id not in self._consented_profiles:
                    raise ObservationConsentRequired(
                        "Explicit observer consent is required."
                    )
            return ObservationResult("analysis_unavailable")
        with self._lock:
            if profile_id not in self._consented_profiles:
                raise ObservationConsentRequired("Explicit observer consent is required.")
        return ObservationResult("no_alert")

    @staticmethod
    def _validate_profile(profile_id: str) -> str:
        if not isinstance(profile_id, str) or not profile_id.strip() or len(profile_id) > 128:
            raise ValueError("A valid enrolled speaker profile is required.")
        return profile_id
