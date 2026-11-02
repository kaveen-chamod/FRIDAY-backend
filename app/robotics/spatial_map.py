"""Bounded integer-grid mapping, waypoint storage, and deterministic A*."""
from __future__ import annotations

import heapq
import itertools
from typing import Iterable


Point3D = tuple[int, int, int]
_MAX_GRID_CELLS = 250_000
_MAX_OBSTACLES = 100_000
_MAX_WAYPOINTS = 2_000
_MAX_PATH_EXPANSIONS = 100_000


class SpatialMapError(ValueError):
    """Map dimensions, coordinates, waypoints, or path requests are invalid."""


def _point(value: object, label: str) -> Point3D:
    if not isinstance(value, (tuple, list)) or len(value) not in (2, 3):
        raise SpatialMapError(f"{label} must contain two or three integer coordinates.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise SpatialMapError(f"{label} coordinates must be integers.")
    x, y = value[0], value[1]
    z = value[2] if len(value) == 3 else 0
    return x, y, z


class SpatialMap:
    """A finite, axis-aligned 2D/3D occupancy grid with 6-connected A*."""

    def __init__(
        self,
        width: int,
        height: int,
        depth: int = 1,
        obstacles: Iterable[object] = (),
    ) -> None:
        dimensions = (width, height, depth)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions
        ):
            raise SpatialMapError("Grid width, height, and depth must be positive integers.")
        if width * height * depth > _MAX_GRID_CELLS:
            raise SpatialMapError(f"Grid cannot exceed {_MAX_GRID_CELLS} cells.")
        obstacle_points: set[Point3D] = set()
        for obstacle in obstacles:
            obstacle_points.add(_point(obstacle, "obstacle"))
            if len(obstacle_points) > _MAX_OBSTACLES:
                raise SpatialMapError("Obstacle count exceeds its configured limit.")
        self.width = width
        self.height = height
        self.depth = depth
        self._obstacles: set[Point3D] = set()
        for obstacle in obstacle_points:
            self._assert_in_bounds(obstacle, "obstacle")
            self._obstacles.add(obstacle)

    def _assert_in_bounds(self, point: Point3D, label: str) -> None:
        x, y, z = point
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            raise SpatialMapError(f"{label} is outside the configured grid.")

    def is_blocked(self, value: object) -> bool:
        point = _point(value, "coordinate")
        self._assert_in_bounds(point, "coordinate")
        return point in self._obstacles

    def set_obstacle(self, value: object, blocked: bool = True) -> None:
        point = _point(value, "obstacle")
        self._assert_in_bounds(point, "obstacle")
        if not isinstance(blocked, bool):
            raise SpatialMapError("blocked must be a boolean.")
        if blocked:
            self._obstacles.add(point)
        else:
            self._obstacles.discard(point)

    def find_path(
        self,
        start: object,
        goal: object,
        *,
        max_expansions: int = _MAX_PATH_EXPANSIONS,
    ) -> list[Point3D]:
        origin = _point(start, "start")
        destination = _point(goal, "goal")
        self._assert_in_bounds(origin, "start")
        self._assert_in_bounds(destination, "goal")
        if origin in self._obstacles or destination in self._obstacles:
            raise SpatialMapError("Start and goal must be unoccupied.")
        if (
            isinstance(max_expansions, bool)
            or not isinstance(max_expansions, int)
            or not 1 <= max_expansions <= _MAX_PATH_EXPANSIONS
        ):
            raise SpatialMapError("max_expansions must be between 1 and 100000.")
        if origin == destination:
            return [origin]

        def distance(first: Point3D, second: Point3D) -> int:
            return sum(abs(a - b) for a, b in zip(first, second))

        serial = itertools.count()
        frontier: list[tuple[int, int, Point3D]] = [
            (distance(origin, destination), next(serial), origin)
        ]
        came_from: dict[Point3D, Point3D] = {}
        costs = {origin: 0}
        expansions = 0
        while frontier and expansions < max_expansions:
            _, _, current = heapq.heappop(frontier)
            expansions += 1
            if current == destination:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            x, y, z = current
            neighbors = (
                (x + 1, y, z), (x - 1, y, z),
                (x, y + 1, z), (x, y - 1, z),
                (x, y, z + 1), (x, y, z - 1),
            )
            for neighbor in neighbors:
                if (
                    not 0 <= neighbor[0] < self.width
                    or not 0 <= neighbor[1] < self.height
                    or not 0 <= neighbor[2] < self.depth
                    or neighbor in self._obstacles
                ):
                    continue
                new_cost = costs[current] + 1
                if new_cost >= costs.get(neighbor, _MAX_GRID_CELLS + 1):
                    continue
                came_from[neighbor] = current
                costs[neighbor] = new_cost
                priority = new_cost + distance(neighbor, destination)
                heapq.heappush(frontier, (priority, next(serial), neighbor))
        if frontier:
            raise SpatialMapError("Path search exceeded its expansion budget.")
        raise SpatialMapError("No obstacle-free path exists.")

    def status(self) -> dict[str, object]:
        return {
            "dimensions": {"width": self.width, "height": self.height, "depth": self.depth},
            "obstacle_count": len(self._obstacles),
            "coordinates": "zero-based integer grid cells",
        }


class WaypointRegistry:
    """Bounded in-memory named coordinates; names never execute code."""

    def __init__(self, spatial_map: SpatialMap) -> None:
        self.spatial_map = spatial_map
        self._waypoints: dict[str, Point3D] = {}

    def add(self, name: str, coordinate: object) -> dict[str, object]:
        if (
            not isinstance(name, str)
            or not 1 <= len(name.strip()) <= 64
            or any(ord(char) < 32 for char in name)
        ):
            raise SpatialMapError("Waypoint name must contain 1 to 64 printable characters.")
        key = name.strip()
        point = _point(coordinate, "waypoint")
        self.spatial_map._assert_in_bounds(point, "waypoint")
        if self.spatial_map.is_blocked(point):
            raise SpatialMapError("Waypoints cannot occupy an obstacle cell.")
        if key not in self._waypoints and len(self._waypoints) >= _MAX_WAYPOINTS:
            raise SpatialMapError("Waypoint capacity is full.")
        self._waypoints[key] = point
        return {"name": key, "coordinate": point}

    def get(self, name: str) -> Point3D:
        if not isinstance(name, str) or name not in self._waypoints:
            raise SpatialMapError("Waypoint was not found.")
        return self._waypoints[name]

    def list(self) -> list[dict[str, object]]:
        return [
            {"name": name, "coordinate": self._waypoints[name]}
            for name in sorted(self._waypoints)
        ]
