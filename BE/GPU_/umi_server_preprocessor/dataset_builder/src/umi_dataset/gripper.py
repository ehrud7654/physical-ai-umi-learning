from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .intrinsics import scaled_camera_matrix
from .recording import longest_false_run


def _largest(corners, ids, marker_id):
    matches = [corner[0] for corner, value in zip(corners, ids.flatten()) if value == marker_id]
    return max(matches, key=cv2.contourArea) if matches else None


def _marker_translation(corners, camera_matrix, distortion, marker_size_m):
    half = marker_size_m / 2
    object_points = np.asarray([
        [-half, half, 0], [half, half, 0],
        [half, -half, 0], [-half, -half, 0],
    ], dtype=np.float32)
    ok, _, translation = cv2.solvePnP(
        object_points, corners.astype(np.float32), camera_matrix, distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return translation.reshape(3) if ok and translation[2] > 0 else None


def _rolling_median(values, radius=2):
    return np.asarray([np.median(values[max(0, i - radius):i + radius + 1])
                       for i in range(len(values))])


def extract_gripper(session: Path, camera_imu: dict, gripper: dict,
                    csv_output: Path, report_output: Path) -> dict:
    if gripper.get("status") != "validated_with_user_measurements":
        raise ValueError("gripper calibration has not been validated")
    capture = cv2.VideoCapture(str(session / "video.mp4"))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError("video cannot be decoded")
    matrix, distortion = scaled_camera_matrix(camera_imu["camera"], width, height)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
    distances = {}
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        corners, ids, _ = detector.detectMarkers(frame)
        left = _largest(corners, ids, 0) if ids is not None else None
        right = _largest(corners, ids, 1) if ids is not None else None
        if left is not None and right is not None:
            translations = [_marker_translation(marker, matrix, distortion,
                                                gripper["marker_side_mm"] / 1000)
                            for marker in (left, right)]
            if all(value is not None for value in translations):
                distances[decoded] = float(np.linalg.norm(translations[0] - translations[1]) * 1000)
        decoded += 1
    capture.release()
    if len(distances) < 10:
        raise ValueError(f"only {len(distances)} frames contain usable ArUco IDs 0 and 1")

    with (session / "encoded.csv").open(newline="", encoding="utf-8") as stream:
        encoded = list(csv.DictReader(stream))
    if decoded != len(encoded):
        raise ValueError(f"decoded/encoded frame mismatch {decoded} != {len(encoded)}")
    indices = np.asarray(sorted(distances), dtype=np.int64)
    raw = np.asarray([distances[index] for index in indices])
    interpolated = np.interp(np.arange(decoded), indices, raw)
    widths_mm = _rolling_median(np.maximum(0, interpolated - gripper["raw_closed_marker_center_mm"]))
    detected = [index in distances for index in range(decoded)]
    first_pts = int(encoded[0]["pts_us"])
    fieldnames = ("frame_index", "timestamp_s", "marker_detected", "marker_center_distance_mm",
                  "marker_center_interpolated_mm", "gripper_width_mm")
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(decoded):
            writer.writerow({
                "frame_index": index,
                "timestamp_s": (int(encoded[index]["pts_us"]) - first_pts) / 1e6,
                "marker_detected": detected[index],
                "marker_center_distance_mm": distances.get(index, ""),
                "marker_center_interpolated_mm": float(interpolated[index]),
                "gripper_width_mm": float(widths_mm[index]),
            })
    report = {
        "status": "pass",
        "method": "3D marker-center distance minus calibrated closed marker-center reference",
        "decoded_frames": decoded,
        "frames_with_both_markers": len(distances),
        "detection_rate": len(distances) / decoded,
        "maximum_missing_run_frames": longest_false_run(detected),
        "width_median_mm": float(np.median(widths_mm)),
        "width_min_mm": float(widths_mm.min()),
        "width_max_mm": float(widths_mm.max()),
        "raw_closed_marker_center_mm": gripper["raw_closed_marker_center_mm"],
        "marker_side_mm": gripper["marker_side_mm"],
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
