"""Low-cost deterministic risk classification; it never grants permissions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from core import security


class TaskRisk(str, Enum):
    NORMAL = "NORMAL"
    MODERATE = "MODERATE"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


_EXPLICIT_SIMULATION_PATTERNS = (
    re.compile(
        r"^\s*(?:(?:please|can you|could you|would you)\s+)?"
        r"(?:run|perform|start)\s+(?:a\s+)?(?:digital[- ]twin\s+)?"
        r"(?:simulation|prediction|risk assessment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:(?:please|can you|could you|would you)\s+)?"
        r"(?:simulate|predict)\s+(?:this|the|my|our|proposed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:simulate karanna|simulate karala balanna|"
        r"risk eka predict karanna)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:අනුකරණය කරන්න|පුරෝකථනය කරන්න)"),
)


@dataclass(frozen=True)
class TaskClassification:
    risk: TaskRisk
    explicit_simulation_requested: bool
    reason: str


def has_explicit_simulation_intent(message: str) -> bool:
    if not isinstance(message, str) or not message.strip() or len(message) > 16_000:
        return False
    return any(pattern.search(message) for pattern in _EXPLICIT_SIMULATION_PATTERNS)


def classify_task(
    *,
    message: str = "",
    tool_name: str | None = None,
    arguments: Mapping[str, Any] | None = None,
    security_tier: str | None = None,
) -> TaskClassification:
    """Classify intent/risk without running a model or changing security policy."""
    explicit = has_explicit_simulation_intent(message)
    if security_tier == security.TIER_DANGEROUS:
        return TaskClassification(
            TaskRisk.CRITICAL,
            explicit,
            "The existing security tier marks this operation DANGEROUS.",
        )
    if tool_name == "iot_control":
        args = arguments or {}
        action = str(args.get("action") or "").strip().lower()
        direction = str(args.get("direction") or "").strip().upper()
        if action == "move" and direction == "STOP":
            return TaskClassification(
                TaskRisk.NORMAL,
                explicit,
                "Emergency STOP remains an immediate safety control.",
            )
        return TaskClassification(
            TaskRisk.HIGH_RISK,
            explicit,
            "Physical IoT actuation requires an applicable validated twin model.",
        )
    if explicit:
        return TaskClassification(
            TaskRisk.MODERATE,
            True,
            "The user explicitly requested a simulation or prediction.",
        )
    return TaskClassification(
        TaskRisk.NORMAL,
        False,
        "No simulation intent or high-risk operation was identified.",
    )
