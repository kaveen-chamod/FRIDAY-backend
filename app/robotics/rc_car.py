"""RC car HAL backed by the existing confirmed TLS-MQTT action."""
from __future__ import annotations

from typing import Any, Mapping

from app.robotics.base_vehicle import BaseVehicle, VehicleError


_DIRECTIONS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
_DEVICE_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class RCCarVehicle(BaseVehicle):
    @property
    def vehicle_type(self) -> str:
        return "rc_car"

    def build_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(action, Mapping):
            raise VehicleError("RC car action must be an object.")
        direction = action.get("direction")
        if not isinstance(direction, str) or direction.upper() not in _DIRECTIONS:
            raise VehicleError("RC car direction is unsupported.")
        return {
            "device_id": self.vehicle_id,
            "action": "move",
            "direction": direction.upper(),
        }

    def emergency_stop(self) -> str:
        from actions.iot_controller import control_iot

        return control_iot({
            "action": "move",
            "device_id": self.vehicle_id,
            "direction": "STOP",
        })

    @classmethod
    def valid_device_id(cls, device_id: str) -> bool:
        return (
            isinstance(device_id, str)
            and 1 <= len(device_id) <= 80
            and all(char in _DEVICE_ID_CHARS for char in device_id)
        )
