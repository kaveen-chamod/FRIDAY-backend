"""Local collision/current limits and immediate distance-triggered braking."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from app.robotics.spatial_map import Point3D, SpatialMap, SpatialMapError, _point


_LOG = logging.getLogger("friday.robotics.collision")
EMERGENCY_DISTANCE_CM = 15.0


class CollisionGuardError(ValueError):
    """Telemetry or a proposed action violated a robotics safety bound."""


@dataclass(frozen=True)
class CollisionDecision:
    allowed: bool
    emergency_stop_triggered: bool
    reason: str
    emergency_stop_result: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "emergency_stop_triggered": self.emergency_stop_triggered,
            "reason": self.reason,
            "emergency_stop_result": self.emergency_stop_result,
        }


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollisionGuardError(f"{label} must be a finite non-negative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise CollisionGuardError(f"{label} must be a finite non-negative number.")
    return number


class CollisionGuard:
    """Fail-closed local checks; hard stop is synchronous and model-independent."""

    def __init__(
        self,
        *,
        emergency_stop: Callable[[str], str] | None = None,
        minimum_distance_cm: float = EMERGENCY_DISTANCE_CM,
    ) -> None:
        threshold = _finite_nonnegative(minimum_distance_cm, "minimum_distance_cm")
        if threshold != EMERGENCY_DISTANCE_CM:
            raise CollisionGuardError("The emergency distance threshold is fixed at 15 cm.")
        self._emergency_stop = emergency_stop

    def observe_telemetry(
        self,
        vehicle_id: str,
        *,
        distance_cm: object,
        current_a: object | None = None,
        maximum_current_a: object | None = None,
    ) -> CollisionDecision:
        distance = _finite_nonnegative(distance_cm, "distance_cm")
        unsafe_current = False
        if current_a is not None and maximum_current_a is not None:
            current = _finite_nonnegative(current_a, "current_a")
            maximum = _finite_nonnegative(maximum_current_a, "maximum_current_a")
            unsafe_current = current > maximum
        distance_breach = distance < EMERGENCY_DISTANCE_CM
        if distance_breach or unsafe_current:
            if self._emergency_stop is None:
                raise CollisionGuardError(
                    "Emergency safety threshold crossed, but no hard-stop handler is configured."
                )
            try:
                result = self._emergency_stop(vehicle_id)
            except Exception:
                _LOG.exception("Immediate robotics emergency stop failed for %s.", vehicle_id)
                raise
            if not isinstance(result, str) or "not delivered" in result.lower():
                raise CollisionGuardError(
                    "Emergency stop was requested but the vehicle did not confirm delivery."
                )
            _LOG.critical(
                "Emergency stop dispatched for %s (distance=%.2f cm, overcurrent=%s).",
                vehicle_id, distance, unsafe_current,
            )
            return CollisionDecision(
                allowed=False,
                emergency_stop_triggered=True,
                reason=(
                    "Distance is below the fixed 15 cm emergency threshold."
                    if distance_breach
                    else "Measured current exceeds the configured maximum."
                ),
                emergency_stop_result=result,
            )
        return CollisionDecision(True, False, "Telemetry is within configured limits.")

    def check_path(
        self,
        spatial_map: SpatialMap,
        path: Sequence[object],
    ) -> CollisionDecision:
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or not path:
            raise CollisionGuardError("A non-empty planned path is required.")
        try:
            points: list[Point3D] = [_point(item, "path coordinate") for item in path]
            for point in points:
                if spatial_map.is_blocked(point):
                    return CollisionDecision(
                        False, False, "Planned path intersects a known obstacle."
                    )
        except SpatialMapError as exc:
            raise CollisionGuardError(str(exc)) from exc
        if any(
            sum(abs(a - b) for a, b in zip(first, second)) != 1
            for first, second in zip(points, points[1:])
        ):
            raise CollisionGuardError("Path must use adjacent grid cells without jumps.")
        return CollisionDecision(True, False, "Planned path avoids known grid obstacles.")

    async def preflight(
        self,
        twin: Any,
        vehicle_id: str,
        command: Mapping[str, Any],
        *,
        distance_cm: object,
        current_a: object,
        maximum_current_a: object,
        spatial_map: SpatialMap | None = None,
        path: Sequence[object] | None = None,
    ) -> dict[str, Any]:
        local = self.observe_telemetry(
            vehicle_id,
            distance_cm=distance_cm,
            current_a=current_a,
            maximum_current_a=maximum_current_a,
        )
        if not local.allowed:
            return {
                "status": "LOCAL_SAFETY_REJECTED",
                "dispatch_allowed": False,
                "collision_guard": local.as_dict(),
            }
        if (spatial_map is None) != (path is None):
            raise CollisionGuardError("Both spatial_map and path must be supplied together.")
        if spatial_map is not None and path is not None:
            path_result = self.check_path(spatial_map, path)
            if not path_result.allowed:
                return {
                    "status": "PATH_REJECTED",
                    "dispatch_allowed": False,
                    "collision_guard": path_result.as_dict(),
                }
        if twin is None or not callable(getattr(twin, "assess_dispatch", None)):
            return {
                "status": "UNAVAILABLE",
                "dispatch_allowed": False,
                "detail": "No digital-twin simulator is available.",
                "collision_guard": local.as_dict(),
            }
        simulated = await twin.assess_dispatch(vehicle_id, command)
        if not isinstance(simulated, Mapping):
            return {
                "status": "FAILED",
                "dispatch_allowed": False,
                "detail": "Digital-twin simulator returned an invalid result.",
                "collision_guard": local.as_dict(),
            }
        return {
            **dict(simulated),
            "dispatch_allowed": simulated.get("dispatch_allowed") is True,
            "collision_guard": local.as_dict(),
        }
