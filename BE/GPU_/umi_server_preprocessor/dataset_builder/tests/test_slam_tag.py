"""Synthetic mapping video with a rendered 16 cm tag at a known pose -> T_slam_tag recovers it."""
import csv
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umi_dataset.intrinsics import scaled_camera_matrix  # noqa: E402
from umi_dataset.slam_tag import estimate_slam_tag, load_slam_tag  # noqa: E402


def render_tag(matrix, distortion, cam_tag, size_m, width, height, dictionary, tag_id):
    half = size_m / 2
    object_points = np.asarray([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float32)
    rvec = Rotation.from_matrix(cam_tag[:3, :3]).as_rotvec()
    projected, _ = cv2.projectPoints(object_points, rvec, cam_tag[:3, 3], matrix, distortion)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 400)
    canvas = np.full((520, 520), 255, np.uint8)     # white quiet zone around the black border
    canvas[60:460, 60:460] = marker
    source = np.asarray([[60, 60], [460, 60], [460, 460], [60, 460]], np.float32)
    warp = cv2.getPerspectiveTransform(source, projected.reshape(4, 2).astype(np.float32))
    frame = cv2.warpPerspective(canvas, warp, (width, height), borderValue=200)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def main() -> None:
    camera_imu = json.loads((ROOT / "configs/s22_camera_imu.json").read_text(encoding="utf-8"))
    tag = json.loads((ROOT / "configs/s22_slam_tag.json").read_text(encoding="utf-8"))
    width, height, frames = 960, 540, 40
    matrix, distortion = scaled_camera_matrix(camera_imu["camera"], width, height)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tag["dictionary"]))
    cam_tag = np.eye(4)
    cam_tag[:3, :3] = Rotation.from_euler("xyz", [200, 10, 5], degrees=True).as_matrix()  # tag lies below, facing camera
    cam_tag[:3, 3] = [0.05, 0.15, 0.9]
    slam_cam = np.eye(4)
    slam_cam[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    slam_cam[:3, 3] = [0.4, -0.2, 0.7]
    expected = slam_cam @ cam_tag
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        session = base / "rec_mapping"
        session.mkdir()
        writer = cv2.VideoWriter(str(session / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        assert writer.isOpened()
        for _ in range(frames):
            writer.write(render_tag(matrix, distortion, cam_tag, tag["tag_size_m"], width, height,
                                    dictionary, tag["tag_id"]))
        writer.release()
        trajectory = base / "camera_trajectory.csv"
        quaternion = Rotation.from_matrix(slam_cam[:3, :3]).as_quat()
        with trajectory.open("w", newline="", encoding="utf-8") as stream:
            out = csv.DictWriter(stream, fieldnames=["frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
                                                     "x", "y", "z", "q_x", "q_y", "q_z", "q_w"])
            out.writeheader()
            for index in range(frames):
                lost = index < 5  # initialisation lag must be ignored
                out.writerow({"frame_idx": index, "timestamp": index / 30, "state": 1 if lost else 2,
                              "is_lost": str(lost).lower(), "is_keyframe": "false",
                              "x": slam_cam[0, 3], "y": slam_cam[1, 3], "z": slam_cam[2, 3],
                              "q_x": quaternion[0], "q_y": quaternion[1], "q_z": quaternion[2], "q_w": quaternion[3]})
        output = base / "tx_slam_tag.json"
        report = estimate_slam_tag(session, trajectory, camera_imu, tag, output)
        assert report["frames_with_tag"] == frames - 5 and report["samples_used"] >= tag["minimum_detections"]
        estimated = load_slam_tag(output)
        position_error_mm = np.linalg.norm(estimated[:3, 3] - expected[:3, 3]) * 1000
        rotation_error_deg = np.degrees(Rotation.from_matrix(expected[:3, :3].T @ estimated[:3, :3]).magnitude())
        # mp4v compression blurs the rendered edges: ~1% depth error at 0.9 m is expected. The value
        # is one constant offset shared by every episode of the session, so relative actions are unaffected.
        assert position_error_mm < 15, position_error_mm
        assert rotation_error_deg < 1, rotation_error_deg
        print(f"SLAM_TAG_SELF_TEST_OK position_error_mm={position_error_mm:.2f} rotation_error_deg={rotation_error_deg:.3f}")


if __name__ == "__main__":
    main()
