#!/usr/bin/env python3
"""Estimate the fixed-marker frame from an ORB-SLAM3 camera trajectory.

The trajectory rows are interpreted as ``T_world_camera``.  OpenCV solvePnP
provides ``T_camera_marker``; their product is therefore ``T_world_marker``.
Metric output remains provisional unless the physical black-square side length
is explicitly marked verified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def _summary(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _centre(transforms: np.ndarray) -> tuple[np.ndarray, Rotation]:
    translation = np.median(transforms[:, :3, 3], axis=0)
    rotation = Rotation.from_matrix(transforms[:, :3, :3]).mean()
    return translation, rotation


def _residuals(transforms: np.ndarray, translation: np.ndarray,
               rotation: Rotation) -> tuple[np.ndarray, np.ndarray]:
    translation_error = np.linalg.norm(transforms[:, :3, 3] - translation, axis=1)
    rotation_error = np.rad2deg(
        (rotation.inv() * Rotation.from_matrix(transforms[:, :3, :3])).magnitude()
    )
    return translation_error, rotation_error


def estimate_alignment(args: argparse.Namespace) -> dict:
    with args.trajectory.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if len(rows) != int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise ValueError("trajectory/video frame count mismatch")

    calibration_width, calibration_height = args.calibration_resolution
    sx, sy = width / calibration_width, height / calibration_height
    fx, fy, cx, cy = args.intrinsics
    camera_matrix = np.asarray([
        [fx * sx, 0.0, cx * sx],
        [0.0, fy * sy, cy * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    distortion = np.asarray(args.distortion, dtype=np.float64)
    half = args.marker_size_m / 2.0
    # Corner order follows cv::aruco detection and SOLVEPNP_IPPE_SQUARE.
    marker_points = np.asarray([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float64)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )

    transforms: list[np.ndarray] = []
    reprojection: list[float] = []
    frame_indices: list[int] = []
    marker_frame_indices: list[int] = []
    for index, row in enumerate(rows):
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"video ended at frame {index}")
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue
        matches = np.flatnonzero(ids.reshape(-1) == args.marker_id)
        if matches.size == 0:
            continue
        marker_frame_indices.append(index)
        pose_valid = row.get("state") == "2" and row.get("is_lost", "true").lower() == "false"
        if not pose_valid:
            continue
        image_points = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
        solved, rvec, tvec = cv2.solvePnP(
            marker_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not solved:
            continue
        transform_camera_marker = np.eye(4)
        transform_camera_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform_camera_marker[:3, 3] = tvec.reshape(3)
        transform_world_camera = np.eye(4)
        transform_world_camera[:3, :3] = Rotation.from_quat([
            float(row["q_x"]), float(row["q_y"]),
            float(row["q_z"]), float(row["q_w"]),
        ]).as_matrix()
        transform_world_camera[:3, 3] = [
            float(row["x"]), float(row["y"]), float(row["z"]),
        ]
        transforms.append(transform_world_camera @ transform_camera_marker)
        projected, _ = cv2.projectPoints(
            marker_points, rvec, tvec, camera_matrix, distortion
        )
        reprojection.append(float(np.linalg.norm(
            projected.reshape(4, 2) - image_points, axis=1
        ).mean()))
        frame_indices.append(index)
    capture.release()

    errors: list[str] = []
    warnings: list[str] = []
    array = np.asarray(transforms)
    reprojection_array = np.asarray(reprojection)
    if len(array) < args.minimum_observations:
        errors.append(
            f"only {len(array)} joint pose/marker observations; need {args.minimum_observations}"
        )
        inliers = np.zeros(len(array), dtype=bool)
    else:
        preliminary_translation, preliminary_rotation = _centre(array)
        translation_error, rotation_error = _residuals(
            array, preliminary_translation, preliminary_rotation
        )
        inliers = (
            (translation_error <= args.inlier_translation_m)
            & (rotation_error <= args.inlier_rotation_deg)
            & (reprojection_array <= args.maximum_reprojection_px)
        )
        if int(inliers.sum()) < args.minimum_observations:
            errors.append(
                f"only {int(inliers.sum())} robust observations; need {args.minimum_observations}"
            )

    if np.any(inliers):
        translation, rotation = _centre(array[inliers])
        translation_error, rotation_error = _residuals(array[inliers], translation, rotation)
        world_marker = np.eye(4)
        world_marker[:3, :3] = rotation.as_matrix()
        world_marker[:3, 3] = translation
        marker_world = np.linalg.inv(world_marker)
        reprojection_inliers = reprojection_array[inliers]
        if np.percentile(translation_error, 95) > args.maximum_translation_p95_m:
            errors.append("translation residual p95 exceeds the alignment gate")
        if np.percentile(rotation_error, 95) > args.maximum_rotation_p95_deg:
            errors.append("rotation residual p95 exceeds the alignment gate")
    else:
        world_marker = np.full((4, 4), np.nan)
        marker_world = np.full((4, 4), np.nan)
        translation_error = np.asarray([])
        rotation_error = np.asarray([])
        reprojection_inliers = np.asarray([])

    if not args.marker_size_verified:
        warnings.append("fixed marker size is provisional; metric alignment is not publishable")
    if errors:
        status = "REJECT_ARUCO_ALIGNMENT"
    elif not args.marker_size_verified:
        status = "PROVISIONAL_ARUCO_ALIGNMENT"
    else:
        status = "PASS_ARUCO_ALIGNMENT"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "video": str(args.video),
            "trajectory": str(args.trajectory),
            "trajectory_interpretation": "T_world_camera",
            "marker_id": args.marker_id,
            "marker_black_square_size_m": args.marker_size_m,
            "marker_size_verified": args.marker_size_verified,
            "video_resolution": [width, height],
            "calibration_resolution": [calibration_width, calibration_height],
        },
        "observations": {
            "marker_detected_frames": len(marker_frame_indices),
            "marker_first_last_frame": (
                [marker_frame_indices[0], marker_frame_indices[-1]]
                if marker_frame_indices else None
            ),
            "joint_pose_marker": len(array),
            "robust_inliers": int(inliers.sum()),
            "first_last_frame": ([frame_indices[0], frame_indices[-1]] if frame_indices else None),
        },
        "T_world_marker": world_marker.tolist(),
        "T_marker_world": marker_world.tolist(),
        "translation_residual_m": _summary(translation_error),
        "rotation_residual_deg": _summary(rotation_error),
        "reprojection_error_px": _summary(reprojection_inliers),
        "thresholds": {
            "minimum_observations": args.minimum_observations,
            "inlier_translation_m": args.inlier_translation_m,
            "inlier_rotation_deg": args.inlier_rotation_deg,
            "maximum_reprojection_px": args.maximum_reprojection_px,
            "maximum_translation_p95_m": args.maximum_translation_p95_m,
            "maximum_rotation_p95_deg": args.maximum_rotation_p95_deg,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument("--marker-size-m", type=float, required=True)
    parser.add_argument("--marker-size-verified", action="store_true")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                        default=[386.783698207, 388.105698794, 473.576277752, 272.084056994])
    parser.add_argument("--distortion", type=float, nargs=4, metavar=("K1", "K2", "P1", "P2"),
                        default=[0.0125279049159, -0.00376646144958,
                                 -0.000443263076207, 0.000485273184501])
    parser.add_argument("--calibration-resolution", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        default=[960.0, 540.0])
    parser.add_argument("--minimum-observations", type=int, default=20)
    parser.add_argument("--inlier-translation-m", type=float, default=0.04)
    parser.add_argument("--inlier-rotation-deg", type=float, default=5.0)
    parser.add_argument("--maximum-reprojection-px", type=float, default=3.0)
    parser.add_argument("--maximum-translation-p95-m", type=float, default=0.03)
    parser.add_argument("--maximum-rotation-p95-deg", type=float, default=2.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.marker_size_m <= 0:
        parser.error("--marker-size-m must be positive")
    report = estimate_alignment(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if report["status"].startswith("REJECT") else 0


if __name__ == "__main__":
    raise SystemExit(main())
