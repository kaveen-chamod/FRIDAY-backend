"""Validated data-only adapter joining robot state to the existing simulator."""
from __future__ import annotations

import math
from typing import Any, Mapping


_MODEL_VERSION = "1.0"
_VEHICLE_TYPES = {"rc_car", "drone"}
_DIRECTIONS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
_REQUIRED_NUMBERS = {
    "minimum_distance_cm": (0.0, 100_000.0),
    "maximum_current_a": (0.001, 10_000.0),
    "distance_uncertainty_cm": (0.0, 100_000.0),
    "current_uncertainty_a": (0.0, 10_000.0),
    "action_success_probability": (0.0, 1.0),
    "action_catastrophic_probability": (0.0, 1.0),
    "epistemic_uncertainty": (0.0, 1.0),
}


def _valid_number(value: object, minimum: float, maximum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


class RoboticsTwinAdapter:
    """Uses the bounded Monte Carlo engine; it never sends vehicle commands."""

    domain = "robotics"
    model_version = _MODEL_VERSION
    validated = True

    def validate_device(
        self, specification: Mapping[str, Any], state: Mapping[str, Any]
    ) -> bool:
        if (
            not isinstance(specification, Mapping)
            or not isinstance(state, Mapping)
            or specification.get("vehicle_type") not in _VEHICLE_TYPES
            or not isinstance(state.get("path_clear"), bool)
        ):
            return False
        if any(
            not _valid_number(specification.get(name), *limits)
            for name, limits in _REQUIRED_NUMBERS.items()
        ):
            return False
        if (
            not _valid_number(state.get("distance_cm"), 0.0, 100_000.0)
            or not _valid_number(state.get("current_a"), 0.0, 10_000.0)
            or any(
                not _valid_number(state.get(axis), -1_000_000.0, 1_000_000.0)
                for axis in ("x", "y", "z")
            )
        ):
            return False
        return True

    def build_action_spec(
        self,
        specification: Mapping[str, Any],
        state: Mapping[str, Any],
        command: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self.validate_device(specification, state):
            raise ValueError("Robot physics specification or telemetry is invalid.")
        if not isinstance(command, Mapping):
            raise ValueError("Robot command must be an object.")
        if (
            command.get("device_id") is None
            or command.get("action") != "move"
            or command.get("direction") not in _DIRECTIONS
        ):
            raise ValueError("Twin supports only normalized discrete movement actions.")
        distance = float(state["distance_cm"])
        current = float(state["current_a"])
        distance_sigma = float(specification["distance_uncertainty_cm"])
        current_sigma = float(specification["current_uncertainty_a"])
        max_distance = min(100_000.0, max(distance + 6 * distance_sigma, 100.0))
        max_current = float(specification["maximum_current_a"])
        predicted_current = max(
            current,
            0.0 if command["direction"] == "STOP" else current,
        )
        catastrophe_probability = float(
            specification["action_catastrophic_probability"]
        )
        if state["path_clear"] is not True:
            catastrophe_probability = 1.0
        return {
            "action": f"robotics {specification['vehicle_type']} {command['direction']}",
            "input_distributions": {
                "distance_cm": {
                    "type": "normal",
                    "mean": distance,
                    "stddev": distance_sigma,
                    "minimum": 0.0,
                    "maximum": max_distance,
                },
                "current_a": {
                    "type": "normal",
                    "mean": predicted_current,
                    "stddev": current_sigma,
                    "minimum": 0.0,
                    "maximum": min(10_000.0, max(max_current * 3, predicted_current + current_sigma * 6, 1.0)),
                },
            },
            "success_conditions": [
                {
                    "variable": "distance_cm",
                    "operator": ">=",
                    "value": float(specification["minimum_distance_cm"]),
                },
                {
                    "variable": "current_a",
                    "operator": "<=",
                    "value": max_current,
                },
            ],
            "catastrophic_probability": catastrophe_probability,
            "success_probability": float(specification["action_success_probability"]),
            "epistemic_uncertainty": float(specification["epistemic_uncertainty"]),
            "model_source": "Master-configured robotics limits and authenticated telemetry",
            "model_version": _MODEL_VERSION,
        }

    def allow_dispatch(self, result: Mapping[str, Any]) -> bool:
        return (
            isinstance(result, Mapping)
            and result.get("recommendation") == "RECOMMENDED"
            and result.get("observed_catastrophic_count") == 0
            and result.get("success_probability", 0.0) >= 0.85
            and result.get("uncertainty_score", 1.0) <= 0.10
            and isinstance(result.get("confidence_intervals_95"), Mapping)
            and isinstance(
                result["confidence_intervals_95"].get("catastrophic_risk_score"),
                Mapping,
            )
            and result["confidence_intervals_95"]["catastrophic_risk_score"].get(
                "upper", 1.0
            ) <= 0.05
        )

    def build_text_request(self, text: str) -> Mapping[str, Any] | None:
        return None
