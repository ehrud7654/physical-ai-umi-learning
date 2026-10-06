"""World frame from the table marker: T_slam_tag estimated on the mapping session.

Mirrors Stanford UMI scripts/calibrate_slam_tag.py: every tracked mapping frame that sees the
tag gives one T_slam_tag sample (camera pose from SLAM times tag pose from solvePnP); outliers
are removed around the median position and the sample nearest the inlier mean is kept, so the
result is an exact rigid transform observed in the data rather than an averaged rotation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .intrinsics import scaled_camera_matrix


def load_slam_tag(path: Path) -> np.ndarray:
    matrix = np.asarray(json.loads(Path(path).read_text(encoding="utf-8"))["T_slam_tag"], dtype=np.float64)
    rotation = matrix[:3, :3]
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]) \
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
        raise ValueError(f"{path}: T_slam_tag is not a rigid transform")
    return matrix


def tag_pose(corners: np.ndarray, matrix: np.ndarray, distortion: np.ndarray, size_m: float):
    """T_cam_tag from the four ArUco corners (TL, TR, BR, BL); tag z points out of the print."""
    half = size_m / 2
    object_points = np.asarray([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
                               dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners.astype(np.float32), matrix, distortion,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or tvec[2] <= 0:
        return None
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rvec.reshape(3)).as_matrix()
    pose[:3, 3] = tvec.reshape(3)
    return pose


def _tracked_camera_poses(trajectory_csv: Path) -> dict[int, np.ndarray]:
    poses = {}
    with trajectory_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["state"] != "2" or row["is_lost"].lower() != "false":
                continue
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat([float(row[k]) for k in ("q_x", "q_y", "q_z", "q_w")]).as_matrix()
            pose[:3, 3] = [float(row[k]) for k in ("x", "y", "z")]
            poses[int(row["frame_idx"])] = pose
    return poses


def estimate_slam_tag(session: Path, trajectory_csv: Path, camera_imu: dict, tag: dict,
                      output: Path) -> dict:
    poses = _tracked_camera_poses(trajectory_csv)
    if not poses:
        raise ValueError("mapping trajectory has no tracked frames")
    capture = cv2.VideoCapture(str(session / "video.mp4"))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError("video cannot be decoded")
    matrix, distortion = scaled_camera_matrix(camera_imu["camera"], width, height)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tag["dictionary"])))
    samples, frames_with_tag, index = [], 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in poses:
            corners, ids, _ = detector.detectMarkers(frame)
            matches = [c[0] for c, i in zip(corners, ids.flatten()) if i == tag["tag_id"]] if ids is not None else []
            if matches:
                frames_with_tag += 1
                corner = max(matches, key=cv2.contourArea)
                cam_tag = tag_pose(corner, matrix, distortion, tag["tag_size_m"])
                offset = np.linalg.norm(corner.mean(axis=0) - [width / 2, height / 2]) / (height / 2)
                if cam_tag is not None and np.linalg.norm(cam_tag[:3, 3]) <= tag["maximum_distance_m"] \
                        and offset <= tag["maximum_center_offset_ratio"]:
                    samples.append(poses[index] @ cam_tag)
        index += 1
    capture.release()
    if len(samples) < tag["minimum_detections"]:
        raise ValueError(f"only {len(samples)} usable tag observations, need {tag['minimum_detections']}")
    samples = np.asarray(samples)
    positions = samples[:, :3, 3]
    distances = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
    inliers = distances <= tag["inlier_radius_m"]
    if inliers.sum() < tag["minimum_detections"]:
        raise ValueError(f"only {int(inliers.sum())} tag observations within {tag['inlier_radius_m']} m of the median")
    mean = positions[inliers].mean(axis=0)
    slam_tag = samples[inliers][np.argmin(np.linalg.norm(positions[inliers] - mean, axis=1))]
    report = {
        "schema_version": 1,
        "status": "pass",
        "method": "per-frame T_slam_cam @ T_cam_tag; median-radius inliers; sample nearest the inlier mean",
        "mapping_session": session.name,
        "trajectory": str(trajectory_csv.resolve()),
        "tag_id": tag["tag_id"],
        "tag_size_m": tag["tag_size_m"],
        "tracked_frames": len(poses),
        "frames_with_tag": frames_with_tag,
        "samples_used": int(inliers.sum()),
        "position_spread_m": {"median_abs_dev": float(np.median(distances[inliers])),
                              "max": float(distances[inliers].max())},
        "convention": "T_A_B maps B coordinates into A; metres; tag z points out of the print (up when flat)",
        "T_slam_tag": slam_tag.tolist(),
        "T_tag_slam": np.linalg.inv(slam_tag).tolist(),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
