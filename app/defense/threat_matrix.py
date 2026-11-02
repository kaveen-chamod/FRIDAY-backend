"""Bounded multi-signal threat scoring for authorized assets only."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Mapping

from app.defense.asset_registry import AssetRegistryError, AuthorizedAssetRegistry


THREAT_CATEGORIES = {
    "AUTH_ANOMALY",
    "BRUTE_FORCE",
    "PRIVILEGE_ESCALATION",
    "API_ABUSE",
    "PROMPT_INJECTION",
    "INFRASTRUCTURE_PROBE",
}
SEVERITY_VALUES = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8, "CRITICAL": 1.0}
CRITICALITY_MULTIPLIERS = {
    "LOW": 0.5,
    "MEDIUM": 0.75,
    "HIGH": 1.0,
    "MISSION_CRITICAL": 1.25,
}
_CORRELATION_WINDOW_SECONDS = 300.0
_MAX_CORRELATION_KEYS = 4_096
_MAX_SIGNALS_PER_KEY = 100


class ThreatEvaluationError(ValueError):
    """Threat input cannot be safely evaluated."""


class ThreatMatrix:
    def __init__(self, registry: AuthorizedAssetRegistry) -> None:
        self.registry = registry
        self._lock = threading.Lock()
        self._signals: dict[tuple[str, str], deque[tuple[float, str, str]]] = defaultdict(deque)

    def evaluate(self, event: Mapping[str, Any], *, now: float | None = None) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise ThreatEvaluationError("Threat event must be an object.")
        category = event.get("category")
        severity = event.get("severity")
        asset_id = event.get("target_asset")
        source = event.get("source")
        signal_id = event.get("signal_id")
        if not isinstance(category, str) or category not in THREAT_CATEGORIES:
            raise ThreatEvaluationError("Threat category is not supported.")
        if not isinstance(severity, str) or severity.upper() not in SEVERITY_VALUES:
            raise ThreatEvaluationError("Threat severity is invalid.")
        if not all(isinstance(value, str) and value.strip() for value in (asset_id, source, signal_id)):
            raise ThreatEvaluationError("Asset, source, and signal identifiers are required.")
        if len(source) > 253 or len(signal_id) > 128:
            raise ThreatEvaluationError("Threat source or signal identifier exceeds its limit.")
        try:
            asset = self.registry.require(asset_id)
        except AssetRegistryError as exc:
            raise ThreatEvaluationError(
                "The target is not a registered, ownership-verified asset."
            ) from exc
        if not self.registry.telemetry_source_allowed(asset_id, source):
            raise ThreatEvaluationError("Threat telemetry source is not authorized for this asset.")
        false_positive = event.get("false_positive", False)
        if not isinstance(false_positive, bool):
            raise ThreatEvaluationError("false_positive must be a boolean.")

        observed_at = time.monotonic() if now is None else now
        correlation_key = (asset_id, category)
        with self._lock:
            if correlation_key not in self._signals and len(self._signals) >= _MAX_CORRELATION_KEYS:
                oldest = min(
                    self._signals,
                    key=lambda key: self._signals[key][0][0] if self._signals[key] else float("inf"),
                )
                del self._signals[oldest]
            signals = self._signals[correlation_key]
            while signals and observed_at - signals[0][0] > _CORRELATION_WINDOW_SECONDS:
                signals.popleft()
            if not false_positive and not any(
                item[1] == signal_id and item[2] == source for item in signals
            ):
                signals.append((observed_at, signal_id, source))
                while len(signals) > _MAX_SIGNALS_PER_KEY:
                    signals.popleft()
            event_count = len(signals)
            distinct_sources = len({item[2] for item in signals})

        confidence = min(
            1.0,
            max(
                0.0,
                0.12
                + min(event_count, 5) * 0.12
                + max(0, distinct_sources - 1) * 0.10
                - (0.50 if false_positive else 0.0),
            ),
        )
        multiplier = CRITICALITY_MULTIPLIERS[asset["criticality"]]
        score = round(
            min(1.0, SEVERITY_VALUES[severity.upper()] * confidence * multiplier),
            12,
        )
        if false_positive or confidence < 0.20 or score < 0.40:
            classification = "INFO"
            auto_actions: list[str] = []
        elif score < 0.70:
            classification = "GUARDED"
            auto_actions = ["RATE_LIMIT_IP"]
        elif score < 0.90:
            classification = "HIGH_THREAT"
            auto_actions = ["RATE_LIMIT_IP", "DISPATCH_SECURITY_ALERT"]
        else:
            classification = "CRITICAL_INCIDENT"
            auto_actions = ["RATE_LIMIT_IP", "BLOCK_INDICATOR_TEMP", "DISPATCH_SECURITY_ALERT"]

        permitted = set(asset["permitted_defense_actions"])
        auto_actions = [action for action in auto_actions if action in permitted]
        return {
            "category": category,
            "severity": severity.upper(),
            "target_asset": asset_id,
            "threat_score": score,
            "confidence": confidence,
            "correlated_event_count": event_count,
            "distinct_source_count": distinct_sources,
            "asset_criticality_multiplier": multiplier,
            "classification": classification,
            "safe_auto_actions": auto_actions,
            "master_alert_required": classification in {
                "HIGH_THREAT",
                "CRITICAL_INCIDENT",
            },
            "master_action_required": classification == "CRITICAL_INCIDENT",
            "false_positive_filtered": false_positive,
        }

    def active_correlation_summary(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            result = []
            for (asset_id, category), signals in list(self._signals.items()):
                while signals and now - signals[0][0] > _CORRELATION_WINDOW_SECONDS:
                    signals.popleft()
                if signals:
                    result.append({
                        "target_asset": asset_id,
                        "category": category,
                        "correlated_event_count": len(signals),
                    })
            return result
