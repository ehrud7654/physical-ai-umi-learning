"""Quality gates for ORB-SLAM3 UMI episode intake.

The ORB runner emits one trajectory row per decoded video frame.  Lost rows
may repeat an older timestamp, so only tracked pose timestamps are trusted;
the per-frame tracking sidecar is checked against the original camera clock.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, math.floor(fraction * (len(ordered) - 1)))
    return ordered[index]


def _loss_episodes(valid: list[bool], start: int) -> int:
    episodes = 0
    in_loss = False
    for tracked in valid[start:]:
        if not tracked and not in_loss:
            episodes += 1
            in_loss = True
        elif tracked:
            in_loss = False
    return episodes


def validate_orbslam_episode(
    raw_episode: Path,
    trajectory_csv: Path,
    tracking_csv: Path,
    gripper_report_json: Path,
    *,
    fixed_marker_id: int,
    fixed_marker_size_m: float,
    fixed_marker_size_verified: bool,
    jaw_marker_size_verified: bool | None = None,
    maximum_timestamp_error_ms: float = 2.0,
    maximum_first_pose_time_s: float = 3.0,
    minimum_pose_coverage: float = 0.70,
    minimum_pose_frames: int = 24,
    maximum_gripper_missing_run_frames: int = 7,
) -> dict[str, Any]:
    """Validate one raw episode plus ORB and gripper products.

    The result is deliberately strict: a valid trajectory alone is not enough
    for an episode to enter the metric-scale training dataset.
    """
    errors: list[str] = []
    warnings: list[str] = []

    manifest = json.loads((raw_episode / "manifest.json").read_text(encoding="utf-8"))
    frame_rows = _read_csv(raw_episode / "frames.csv")
    trajectory_rows = _read_csv(trajectory_csv)
    tracking_rows = _read_csv(tracking_csv)
    gripper = json.loads(gripper_report_json.read_text(encoding="utf-8"))

    frame_count = len(frame_rows)
    if manifest.get("outcome") != "success":
        errors.append(f"outcome is {manifest.get('outcome')!r}, not 'success'")
    if frame_count == 0:
        errors.append("frames.csv is empty")
        frame_times_s: list[float] = []
    else:
        sensor_ns = [int(row["sensor_timestamp_ns"]) for row in frame_rows]
        if any(right <= left for left, right in zip(sensor_ns, sensor_ns[1:])):
            errors.append("camera sensor timestamps are not strictly increasing")
        origin = sensor_ns[0]
        frame_times_s = [(value - origin) / 1e9 for value in sensor_ns]

    if len(trajectory_rows) != frame_count:
        errors.append(
            f"trajectory/frame count mismatch {len(trajectory_rows)} != {frame_count}"
        )
    if len(tracking_rows) != frame_count:
        errors.append(f"tracking/frame count mismatch {len(tracking_rows)} != {frame_count}")

    expected_indices = list(range(len(trajectory_rows)))
    try:
        actual_indices = [int(row["frame_idx"]) for row in trajectory_rows]
    except (KeyError, ValueError):
        actual_indices = []
        errors.append("trajectory has invalid or missing frame_idx")
    if actual_indices and actual_indices != expected_indices:
        errors.append("trajectory frame indices are not contiguous from zero")

    comparable = min(frame_count, len(trajectory_rows), len(tracking_rows))
    pose_valid: list[bool] = []
    tracking_valid: list[bool] = []
    imu_initialized: list[bool] = []
    pose_time_errors_ms: list[float] = []
    tracking_time_errors_ms: list[float] = []
    quaternion_errors: list[float] = []
    non_finite_pose = False

    for index in range(comparable):
        pose_row = trajectory_rows[index]
        track_row = tracking_rows[index]
        valid = pose_row.get("state") == "2" and not _truth(pose_row.get("is_lost", "true"))
        tracked = _truth(track_row.get("tracking_ok", "false"))
        pose_valid.append(valid)
        tracking_valid.append(tracked)
        imu_initialized.append(_truth(track_row.get("imu_initialized", "false")))
        try:
            tracking_time_errors_ms.append(
                abs(float(track_row["timestamp"]) - frame_times_s[index]) * 1000
            )
        except (KeyError, ValueError, IndexError):
            errors.append(f"invalid tracking timestamp at frame {index}")
            tracking_time_errors_ms.append(float("inf"))
        if not valid:
            continue
        if not tracked:
            errors.append(f"pose marked valid while tracking sidecar is false at frame {index}")
        try:
            pose_time_errors_ms.append(
                abs(float(pose_row["timestamp"]) - frame_times_s[index]) * 1000
            )
            values = [float(pose_row[key]) for key in ("x", "y", "z", "q_x", "q_y", "q_z", "q_w")]
            if not all(math.isfinite(value) for value in values):
                non_finite_pose = True
            quaternion_errors.append(abs(math.sqrt(sum(value * value for value in values[3:])) - 1.0))
        except (KeyError, ValueError, IndexError):
            non_finite_pose = True

    if non_finite_pose:
        errors.append("tracked trajectory contains invalid or non-finite pose values")
    if quaternion_errors and max(quaternion_errors) > 1e-3:
        errors.append(f"tracked quaternion norm error exceeds 1e-3 ({max(quaternion_errors):.6g})")

    max_tracking_time_error = max(tracking_time_errors_ms, default=float("inf"))
    max_pose_time_error = max(pose_time_errors_ms, default=float("inf"))
    if max_tracking_time_error > maximum_timestamp_error_ms:
        errors.append(
            f"tracking/camera timestamp error {max_tracking_time_error:.3f} ms exceeds "
            f"{maximum_timestamp_error_ms:.3f} ms"
        )
    if max_pose_time_error > maximum_timestamp_error_ms:
        errors.append(
            f"pose/camera timestamp error {max_pose_time_error:.3f} ms exceeds "
            f"{maximum_timestamp_error_ms:.3f} ms"
        )

    valid_indices = [index for index, valid in enumerate(pose_valid) if valid]
    first_valid = valid_indices[0] if valid_indices else None
    pose_count = len(valid_indices)
    pose_coverage = pose_count / frame_count if frame_count else 0.0
    first_pose_time = frame_times_s[first_valid] if first_valid is not None else None
    loss_episodes = _loss_episodes(pose_valid, first_valid) if first_valid is not None else 0
    if pose_count < minimum_pose_frames:
        errors.append(f"only {pose_count} valid poses; need at least {minimum_pose_frames}")
    if pose_coverage < minimum_pose_coverage:
        errors.append(
            f"pose coverage {pose_coverage:.3f} is below {minimum_pose_coverage:.3f}"
        )
    if first_pose_time is None or first_pose_time > maximum_first_pose_time_s:
        errors.append(
            f"first valid pose time {first_pose_time!r}s exceeds {maximum_first_pose_time_s:.3f}s"
        )
    if first_valid is not None and loss_episodes:
        errors.append(f"tracking has {loss_episodes} loss episode(s) after first valid pose")
    if pose_valid and not pose_valid[-1]:
        errors.append("trajectory is not tracked at the final frame")
    if not any(imu_initialized):
        errors.append("visual-inertial map never reports IMU initialized")
    elif not imu_initialized[-1]:
        errors.append("IMU is not initialized at the final frame")

    missing_run = int(gripper.get("maximum_missing_run_frames", frame_count + 1))
    if gripper.get("status") != "pass":
        errors.append(f"gripper extractor status is {gripper.get('status')!r}")
    if missing_run > maximum_gripper_missing_run_frames:
        errors.append(
            f"gripper marker missing run {missing_run} frames exceeds "
            f"{maximum_gripper_missing_run_frames}"
        )
    jaw_scale_verified = (
        bool(manifest.get("marker_black_square_size_verified", False))
        if jaw_marker_size_verified is None else jaw_marker_size_verified
    )
    if not jaw_scale_verified:
        errors.append("jaw marker metric scale is not verified")
    if fixed_marker_id < 0:
        errors.append("fixed ArUco marker id is not specified")
    if not math.isfinite(fixed_marker_size_m) or fixed_marker_size_m <= 0:
        errors.append("fixed ArUco marker size is invalid")
    if not fixed_marker_size_verified:
        errors.append("fixed ArUco marker metric size is provisional, not verified")

    if trajectory_rows and any(
        float(row.get("timestamp", "nan")) < 0 for row in trajectory_rows
        if row.get("timestamp") not in (None, "")
    ):
        warnings.append("trajectory contains a negative timestamp")

    return {
        "status": "PASS_ORB_INTAKE" if not errors else "REJECT_ORB_INTAKE",
        "errors": errors,
        "warnings": warnings,
        "episode": raw_episode.name,
        "outcome": manifest.get("outcome"),
        "fixed_marker": {
            "id": fixed_marker_id,
            "black_square_size_m": fixed_marker_size_m,
            "size_verified": fixed_marker_size_verified,
        },
        "camera": {
            "frames": frame_count,
            "duration_s": frame_times_s[-1] if frame_times_s else 0.0,
            "tracking_timestamp_error_ms_max": max_tracking_time_error,
            "tracking_timestamp_error_ms_p95": _percentile(tracking_time_errors_ms, 0.95),
        },
        "pose": {
            "valid_frames": pose_count,
            "coverage": pose_coverage,
            "first_valid_frame": first_valid,
            "first_valid_time_s": first_pose_time,
            "loss_episodes_after_first_valid": loss_episodes,
            "timestamp_error_ms_max": max_pose_time_error,
            "timestamp_error_ms_p95": _percentile(pose_time_errors_ms, 0.95),
            "imu_initialized_frames": sum(imu_initialized),
        },
        "gripper": {
            "status": gripper.get("status"),
            "frames_with_both_markers": gripper.get("frames_with_both_markers"),
            "detection_rate": gripper.get("detection_rate"),
            "maximum_missing_run_frames": missing_run,
            "jaw_marker_size_verified": jaw_scale_verified,
            "jaw_marker_size_verified_in_manifest": bool(
                manifest.get("marker_black_square_size_verified", False)
            ),
        },
        "thresholds": {
            "maximum_timestamp_error_ms": maximum_timestamp_error_ms,
            "maximum_first_pose_time_s": maximum_first_pose_time_s,
            "minimum_pose_coverage": minimum_pose_coverage,
            "minimum_pose_frames": minimum_pose_frames,
            "maximum_gripper_missing_run_frames": maximum_gripper_missing_run_frames,
        },
    }
