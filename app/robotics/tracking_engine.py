"""Optional, headless ArUco detection over caller-supplied image frames."""
from __future__ import annotations

import math
from typing import Any


class TrackingError(RuntimeError):
    """Vision dependency, frame, or calibration data is unavailable or invalid."""


class ArucoTracker:
    """Detect markers without opening a camera or creating any GUI windows."""

    def __init__(self, dictionary_name: str = "DICT_4X4_50") -> None:
        if (
            not isinstance(dictionary_name, str)
            or not dictionary_name.startswith("DICT_")
            or len(dictionary_name) > 64
        ):
            raise TrackingError("An OpenCV ArUco dictionary name is required.")
        self.dictionary_name = dictionary_name

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:
            raise TrackingError(
                "ArUco tracking is optional; install a headless OpenCV build to enable it."
            ) from exc
        if not hasattr(cv2, "aruco"):
            raise TrackingError(
                "The installed OpenCV build has no ArUco module; install opencv-contrib-python-headless."
            )
        return cv2

    def _detect(self, frame: Any):
        cv2 = self._cv2()
        try:
            import numpy as np
        except ImportError as exc:
            raise TrackingError("NumPy is required for frame validation.") from exc
        if not isinstance(frame, np.ndarray):
            raise TrackingError("frame must be a NumPy image array.")
        if (
            frame.ndim not in (2, 3)
            or frame.shape[0] < 1
            or frame.shape[1] < 1
            or frame.shape[0] * frame.shape[1] > 16_000_000
            or (frame.ndim == 3 and frame.shape[2] not in (1, 3, 4))
        ):
            raise TrackingError("Frame dimensions or channels are invalid.")
        aruco = cv2.aruco
        dictionary_id = getattr(aruco, self.dictionary_name, None)
        if dictionary_id is None:
            raise TrackingError("Requested ArUco dictionary is not supported.")
        dictionary = aruco.getPredefinedDictionary(dictionary_id)
        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
            )
        parameters = aruco.DetectorParameters()
        if hasattr(aruco, "ArucoDetector"):
            corners, ids, _ = aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray, dictionary, parameters=parameters
            )
        return cv2, corners, ids

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Return marker IDs and pixel-center coordinates; no image is persisted."""
        _, corners, ids = self._detect(frame)
        if ids is None:
            return []
        results = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            points = marker_corners.reshape(4, 2)
            center_x = float(points[:, 0].mean())
            center_y = float(points[:, 1].mean())
            if not math.isfinite(center_x) or not math.isfinite(center_y):
                raise TrackingError("OpenCV returned a non-finite marker coordinate.")
            results.append({
                "marker_id": int(marker_id),
                "pixel_x": center_x,
                "pixel_y": center_y,
            })
        return results

    def estimate_pose(
        self,
        frame: Any,
        *,
        marker_size_m: float,
        camera_matrix: Any,
        distortion_coefficients: Any,
    ) -> list[dict[str, Any]]:
        """Estimate marker-relative 3D poses using caller-supplied calibration."""
        if (
            isinstance(marker_size_m, bool)
            or not isinstance(marker_size_m, (int, float))
            or not math.isfinite(float(marker_size_m))
            or marker_size_m <= 0
            or marker_size_m > 10
        ):
            raise TrackingError("marker_size_m must be in the range (0, 10].")
        cv2, corners, ids = self._detect(frame)
        if ids is None:
            return []
        try:
            import numpy as np

            matrix = np.asarray(camera_matrix, dtype=float)
            distortion = np.asarray(distortion_coefficients, dtype=float)
        except (ImportError, TypeError, ValueError) as exc:
            raise TrackingError("Camera calibration must contain numeric arrays.") from exc
        if (
            matrix.shape != (3, 3)
            or distortion.ndim not in (1, 2)
            or not np.isfinite(matrix).all()
            or not np.isfinite(distortion).all()
            or distortion.size > 14
        ):
            raise TrackingError("Camera calibration dimensions or values are invalid.")
        aruco = cv2.aruco
        if not callable(getattr(aruco, "estimatePoseSingleMarkers", None)):
            raise TrackingError("This OpenCV ArUco build does not provide pose estimation.")
        rotation, translation, _ = aruco.estimatePoseSingleMarkers(
            corners, float(marker_size_m), matrix, distortion
        )
        results = []
        for index, marker_id in enumerate(ids.flatten()):
            translation_values = [
                float(value) for value in translation[index].reshape(3)
            ]
            rotation_values = [
                float(value) for value in rotation[index].reshape(3)
            ]
            if not all(
                math.isfinite(value)
                for value in translation_values + rotation_values
            ):
                raise TrackingError("OpenCV returned a non-finite marker pose.")
            results.append({
                "marker_id": int(marker_id),
                "translation_m": translation_values,
                "rotation_vector_rad": rotation_values,
            })
        return results
