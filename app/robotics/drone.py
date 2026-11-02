"""Drone HAL placeholder that fails closed until an approved protocol is configured."""
from __future__ import annotations

from typing import Any, Mapping

from app.robotics.base_vehicle import BaseVehicle, VehicleError


class DroneVehicle(BaseVehicle):
    """Protocol-neutral Drone identity; no flight packets are emitted."""

    @property
    def vehicle_type(self) -> str:
        return "drone"

    def build_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(action, Mapping):
            raise VehicleError("Drone action must be an object.")
        raise VehicleError(
            "Drone actuation is disabled until a specific protocol and safety adapter are configured."
        )

    def emergency_stop(self) -> str:
        return (
            "Emergency stop not delivered: no validated drone protocol "
            "is configured. Use the aircraft's independent hardware failsafe."
        )
