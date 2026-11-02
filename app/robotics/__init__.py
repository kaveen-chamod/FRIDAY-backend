"""Headless robotics abstractions and fail-closed navigation helpers."""

from app.robotics.base_vehicle import BaseVehicle, VehicleError
from app.robotics.collision_guard import (
    CollisionDecision,
    CollisionGuard,
    CollisionGuardError,
)
from app.robotics.drone import DroneVehicle
from app.robotics.rc_car import RCCarVehicle
from app.robotics.spatial_map import SpatialMap, SpatialMapError, WaypointRegistry
from app.robotics.tracking_engine import ArucoTracker, TrackingError
from app.robotics.twin_adapter import RoboticsTwinAdapter

__all__ = [
    "ArucoTracker",
    "BaseVehicle",
    "CollisionDecision",
    "CollisionGuard",
    "CollisionGuardError",
    "DroneVehicle",
    "RCCarVehicle",
    "RoboticsTwinAdapter",
    "SpatialMap",
    "SpatialMapError",
    "TrackingError",
    "VehicleError",
    "WaypointRegistry",
]
