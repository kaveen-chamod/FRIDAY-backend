"""Hardware-agnostic vehicle contract; normal actuation is intentionally absent."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class VehicleError(ValueError):
    """A vehicle identifier, action, or safety result is invalid."""


class BaseVehicle(ABC):
    """Common identity, validation, and Digital Twin preflight surface."""

    def __init__(self, vehicle_id: str) -> None:
        if (
            not isinstance(vehicle_id, str)
            or not vehicle_id.strip()
            or len(vehicle_id) > 80
            or any(ord(char) < 33 for char in vehicle_id)
        ):
            raise VehicleError("vehicle_id must be a non-empty printable identifier.")
        self.vehicle_id = vehicle_id

    @property
    @abstractmethod
    def vehicle_type(self) -> str:
        """Return the registered vehicle kind."""

    @abstractmethod
    def build_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize a data-only vehicle action."""

    @abstractmethod
    def emergency_stop(self) -> str:
        """Request the vehicle's immediate safe-stop behavior."""

    async def preflight(
        self,
        twin: Any,
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run the digital-twin simulation; this method never dispatches."""
        command = self.build_action(action)
        if twin is None or not callable(getattr(twin, "assess_dispatch", None)):
            return {
                "status": "UNAVAILABLE",
                "dispatch_allowed": False,
                "detail": "No digital-twin simulator is available.",
            }
        result = await twin.assess_dispatch(self.vehicle_id, command)
        if not isinstance(result, dict):
            return {
                "status": "FAILED",
                "dispatch_allowed": False,
                "detail": "Digital-twin simulator returned an invalid result.",
            }
        return result
