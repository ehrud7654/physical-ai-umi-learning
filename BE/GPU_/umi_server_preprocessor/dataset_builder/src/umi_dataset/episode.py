from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .camera_tcp import camera_pose_to_tcp
from .recording import longest_false_run
from .trajectory import load_tracked_trajectory


def build_episode(recording: dict, trajectory_path: Path, gripper_path: Path,
                  camera_tcp: np.ndarray, dataset_config: dict,
                  world_slam: np.ndarray | None = None):
    selected, camera_position, camera_quaternion, trajectory_report = load_tracked_trajectory(
        trajectory_path,
        recording["times_s"],
        dataset_config["minimum_episode_frames"],
        dataset_config["maximum_pose_timestamp_error_ms"],
        dataset_config.get("maximum_lost_frames_after_initialization"),
    )
    # The first seconds are SLAM warm-up (small motion with the table marker in view), not the task.
    trim_s = float(dataset_config.get("episode_start_trim_s", 0.0))
    keep = recording["times_s"][selected] >= trim_s
    trimmed_frames = int((~keep).sum())
    selected, camera_position, camera_quaternion = selected[keep], camera_position[keep], camera_quaternion[keep]
    if len(selected) < dataset_config["minimum_episode_frames"]:
        raise ValueError(f"only {len(selected)} tracked frames remain after trimming the first {trim_s} s")
    with gripper_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != recording["frame_count"]:
        raise ValueError(f"gripper/frame count mismatch {len(rows)} != {recording['frame_count']}")
    if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("gripper frame indices are not contiguous")
    detected = [row["marker_detected"].lower() == "true" for row in rows]
    missing_run = longest_false_run(detected)
    if missing_run > dataset_config["maximum_marker_missing_run_frames"]:
        raise ValueError(f"marker missing run {missing_run} exceeds limit")
    width_key = "gripper_width_mm" if "gripper_width_mm" in rows[0] else "width_mm"
    widths = np.asarray([float(rows[index][width_key]) / 1000 for index in selected], dtype=np.float32)
    if not np.isfinite(widths).all() or widths.min() < 0 or widths.max() > dataset_config["maximum_gripper_width_m"]:
        raise ValueError(f"gripper width outside 0..{dataset_config['maximum_gripper_width_m']} m")
    tcp_position, tcp_rotvec = camera_pose_to_tcp(camera_position, camera_quaternion, camera_tcp, world_slam)
    pose = np.concatenate((tcp_position, tcp_rotvec), axis=1).astype(np.float32)
    data = {
        "robot0_eef_pos": tcp_position,
        "robot0_eef_rot_axis_angle": tcp_rotvec,
        "robot0_gripper_width": widths[:, None],
        "robot0_demo_start_pose": np.repeat(pose[:1], len(pose), axis=0),
        "robot0_demo_end_pose": np.repeat(pose[-1:], len(pose), axis=0),
    }
    return data, selected, {
        **trajectory_report,
        "episode_start_trim_s": trim_s,
        "trimmed_frames": trimmed_frames,
        "episode_start_time_s": float(recording["times_s"][selected[0]]),
        "world_frame": "slam_origin_of_this_recording" if world_slam is None else "table_tag",
        "native_sample_rate_hz": float(1 / np.median(np.diff(recording["times_s"][selected]))),
        "marker_detection_rate": sum(detected) / len(detected),
        "maximum_marker_missing_run_frames": missing_run,
        "gripper_width_min_m": float(widths.min()),
        "gripper_width_max_m": float(widths.max()),
    }
