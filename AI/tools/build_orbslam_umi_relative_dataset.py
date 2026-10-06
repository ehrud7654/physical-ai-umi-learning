"""Build a v10-style relative-chunk dataset from reviewed ORB-SLAM episodes.

The selection JSON is the provenance boundary.  Each row names one raw app
session, one tracked ORB trajectory, and the fixed-marker alignment belonging
to the Atlas that produced that trajectory.  Different Atlas candidates are
allowed only because every trajectory is transformed into the same ArUco
table-marker frame before the camera-to-TCP transform is applied.

This tool does not turn provisional calibration into measured calibration.
The output index records the camera-to-TCP and gripper-scale status and marks
the dataset non-deployable whenever either is provisional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from umi.relative_dataset import SCHEMA, relative_vector, write_episode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def nearest_grid_indices(timestamps: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Select source rows nearest a fixed-rate grid without stretching time."""
    period = 1.0 / rate_hz
    grid = timestamps[0] + np.arange(
        int(np.floor((timestamps[-1] - timestamps[0]) / period)) + 1) * period
    indices = np.asarray([int(np.argmin(np.abs(timestamps - value))) for value in grid])
    keep = np.r_[True, np.diff(indices) > 0]
    indices, grid = indices[keep], grid[keep]
    if len(indices) < 2 or np.max(np.abs(timestamps[indices] - grid)) > 0.55 * period:
        raise ValueError("source frames do not cover the requested real-time grid")
    return indices, grid


def _rigid(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < 0.999):
        raise ValueError(f"{name} is not rigid")
    return matrix


def _camera_tcp(path: Path) -> tuple[np.ndarray, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    transform = _rigid(config["T_camera_tcp"], "T_camera_tcp")
    return transform, config


def _alignment(path: Path) -> tuple[np.ndarray, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_ARUCO_ALIGNMENT":
        raise ValueError(f"alignment is not publishable: {report.get('status')!r}")
    inputs = report.get("inputs", {})
    if inputs.get("marker_id") != 13 or not np.isclose(
            float(inputs.get("marker_black_square_size_m", np.nan)), 0.16):
        raise ValueError("alignment must use DICT_4X4_50 ID 13 at 0.16 m")
    return _rigid(report["T_marker_world"], "T_marker_world"), report


def _pose_6d(transform: np.ndarray, *, source_frame_index: int,
             timestamp_s: float, definition: str) -> dict:
    """Serialize one absolute TCP pose without hiding its frame or row semantics."""
    return {
        "frame": "aruco_id13_table",
        "definition": definition,
        "source_frame_index": int(source_frame_index),
        "timestamp_s": float(timestamp_s),
        "xyz_m": [float(value) for value in transform[:3, 3]],
        "rotation_vector_rad": [
            float(value) for value in Rotation.from_matrix(transform[:3, :3]).as_rotvec()
        ],
        "quaternion_xyzw": [
            float(value) for value in Rotation.from_matrix(transform[:3, :3]).as_quat()
        ],
    }


def _load_episode(selection: dict, camera_tcp: np.ndarray) -> dict:
    raw = Path(selection["raw_episode"])
    trajectory = Path(selection["trajectory"])
    gripper = Path(selection["gripper_csv"])
    alignment_path = Path(selection["alignment"])
    validation_path = Path(selection["trajectory_validation"])
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "pass":
        raise ValueError(f"trajectory validation is {validation.get('status')!r}")
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("outcome") != "success":
        raise ValueError(f"outcome is {manifest.get('outcome')!r}")
    if manifest.get("rotation_baked_into_pixels") is not True:
        raise ValueError("expected upright_v1 pixels with baked rotation")

    frame_rows = _rows(raw / "frames.csv")
    pose_rows = _rows(trajectory)
    grip_rows = _rows(gripper)
    if not frame_rows or len(frame_rows) != len(pose_rows) or len(frame_rows) != len(grip_rows):
        raise ValueError("frame/pose/gripper row count mismatch")
    if [int(row["frame_idx"]) for row in pose_rows] != list(range(len(pose_rows))):
        raise ValueError("trajectory frame_idx is not contiguous")
    if [int(row["frame_index"]) for row in grip_rows] != list(range(len(grip_rows))):
        raise ValueError("gripper frame_index is not contiguous")

    sensor_ns = np.asarray(
        [int(row["sensor_timestamp_ns"]) for row in frame_rows], dtype=np.int64)
    if np.any(np.diff(sensor_ns) <= 0):
        raise ValueError("camera timestamps are not strictly increasing")
    times = (sensor_ns - sensor_ns[0]).astype(np.float64) / 1e9
    marker_world, alignment = _alignment(alignment_path)

    transforms: list[np.ndarray] = []
    widths: list[float] = []
    valid: list[bool] = []
    for index, (pose, grip) in enumerate(zip(pose_rows, grip_rows)):
        tracked = pose.get("state") == "2" and pose.get("is_lost", "true").lower() == "false"
        width_key = "gripper_width_mm" if "gripper_width_mm" in grip else "width_mm"
        try:
            width = float(grip[width_key]) / 1000.0
        except (KeyError, TypeError, ValueError):
            width = float("nan")
        world_camera = np.eye(4)
        try:
            world_camera[:3, :3] = Rotation.from_quat([
                float(pose[key]) for key in ("q_x", "q_y", "q_z", "q_w")
            ]).as_matrix()
            world_camera[:3, 3] = [float(pose[key]) for key in ("x", "y", "z")]
        except (KeyError, ValueError):
            tracked = False
        marker_tcp = marker_world @ world_camera @ camera_tcp
        # The marker alignment is a robust average and may be a few ulps away
        # from SO(3).  Project it back to SO(3) before strict v10 validation.
        marker_tcp[:3, :3] = Rotation.from_matrix(marker_tcp[:3, :3]).as_matrix()
        marker_tcp[3] = [0.0, 0.0, 0.0, 1.0]
        transforms.append(marker_tcp)
        widths.append(width)
        valid.append(bool(tracked and np.isfinite(width) and 0 <= width <= 0.09))

    runs: list[list[int]] = []
    for index, okay in enumerate(valid):
        if not okay:
            continue
        if not runs or index != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(index)
    chosen = max(runs, key=len, default=[])
    if len(chosen) < 10:
        raise ValueError("no sufficiently long tracked pose+gap run")
    return {
        "episode_id": raw.name,
        "raw": raw,
        "video": raw / "video.mp4",
        "times": times[np.asarray(chosen)],
        "source_indices": np.asarray(chosen, dtype=np.int64),
        "poses": np.stack([transforms[index] for index in chosen]),
        "widths": np.asarray([widths[index] for index in chosen], dtype=np.float64),
        "manifest": manifest,
        "trajectory": trajectory,
        "trajectory_validation": validation_path,
        "alignment": alignment_path,
        "alignment_report": alignment,
        "atlas_sha256": selection["atlas_sha256"],
    }


def _decode_frames(video: Path, required: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    decoded: dict[int, np.ndarray] = {}
    index = 0
    try:
        while required - decoded.keys():
            ok, frame = capture.read()
            if not ok:
                break
            if index in required:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                decoded[index] = np.asarray(rgb, dtype=np.uint8)
            index += 1
    finally:
        capture.release()
    missing = sorted(required - decoded.keys())
    if missing:
        raise ValueError(f"video is missing required frames: {missing[:5]}")
    return decoded


def _build_one(episode: dict, output: Path, *, rate_hz: float,
               obs_horizon: int, action_horizon: int,
               camera_tcp_path: Path, camera_tcp_config: dict,
               gap_scale_verified: bool) -> dict:
    sampled, _ = nearest_grid_indices(episode["times"], rate_hz)
    poses = episode["poses"][sampled]
    widths = episode["widths"][sampled]
    source = episode["source_indices"][sampled]
    times = episode["times"][sampled]
    first = obs_horizon - 1
    last = len(sampled) - action_horizon
    if last <= first:
        raise ValueError("tracked run is too short for requested horizons")
    anchors = np.arange(first, last, dtype=np.int64)
    required: set[int] = set()
    for current in anchors:
        required.update(int(source[index]) for index in range(
            current - obs_horizon + 1, current + 1))
    decoded = _decode_frames(episode["video"], required)

    images, proprio, actions = [], [], []
    observation_times, action_times = [], []
    for current in anchors:
        history = list(range(current - obs_horizon + 1, current + 1))
        future = list(range(current + 1, current + action_horizon + 1))
        images.append(np.stack([
            decoded[int(source[index])].transpose(2, 0, 1) for index in history
        ]))
        proprio.append(np.stack([
            relative_vector(poses[current], poses[index], widths[index])
            for index in history
        ]))
        actions.append(np.stack([
            relative_vector(poses[current], poses[index], widths[index])
            for index in future
        ]))
        observation_times.append(times[history])
        action_times.append(times[future])

    arrays = {
        "image": np.stack(images).astype(np.uint8),
        "proprio": np.stack(proprio).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "observation_timestamp": np.stack(observation_times).astype(np.float64),
        "action_timestamp": np.stack(action_times).astype(np.float64),
        "source_row": source[anchors].astype(np.int64),
    }
    camera_status = str(camera_tcp_config.get("status", "unknown"))
    physical_ready = bool(
        gap_scale_verified
        and camera_tcp_config.get("physical_deployment_allowed", False)
    )
    meta = {
        "schema": SCHEMA,
        "episode_id": episode["episode_id"],
        "source": "S22 app v2 + ORB-SLAM3 shared Atlas + ArUco table frame",
        "source_video": str(episode["video"]),
        "source_video_sha256": _sha256(episode["video"]),
        "source_trajectory": str(episode["trajectory"]),
        "source_trajectory_sha256": _sha256(episode["trajectory"]),
        "trajectory_validation": str(episode["trajectory_validation"]),
        "alignment": str(episode["alignment"]),
        "alignment_sha256": _sha256(episode["alignment"]),
        "atlas_sha256": episode["atlas_sha256"],
        "fixed_marker": {"dictionary": "DICT_4X4_50", "id": 13,
                         "black_square_size_m": 0.16},
        "camera_tcp": str(camera_tcp_path),
        "camera_tcp_sha256": _sha256(camera_tcp_path),
        "camera_tcp_status": camera_status,
        "gripper_scale_verified": bool(gap_scale_verified),
        "physical_deployment_ready": physical_ready,
        "rate_hz": rate_hz,
        "observation_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "action_semantics": "all future TCP poses relative to the same current TCP pose; gap absolute metres",
        "action_columns": ["x_m", "y_m", "z_m", "r0x", "r0y", "r0z",
                           "r1x", "r1y", "r1z", "gap_m"],
        "rotation_6d": "first two rows; row-major; Gram-Schmidt decode",
        "source_time_preserved": True,
        "time_scale": 1.0,
        "source_duration_s": float(times[-1] - times[0]),
        "start_tcp_pose_6d": _pose_6d(
            poses[0], source_frame_index=int(source[0]), timestamp_s=float(times[0]),
            definition="first row of selected contiguous tracked-pose+valid-gap segment",
        ),
        "first_training_anchor_tcp_pose_6d": _pose_6d(
            poses[first], source_frame_index=int(source[first]),
            timestamp_s=float(times[first]),
            definition="current TCP at the first H=2 training anchor",
        ),
        "training_rows": int(len(anchors)),
        "success_label": "success",
        "robot_base_alignment_applied": False,
        "robot_ik_applied": False,
    }
    write_episode(output / f"{episode['episode_id']}.npz", arrays, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--camera-tcp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--gap-scale-verified", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    camera_tcp, camera_config = _camera_tcp(args.camera_tcp)
    args.out.mkdir(parents=True)
    selection_copy = args.out / "atlas_selection.json"
    camera_tcp_copy = args.out / "camera_tcp.json"
    shutil.copyfile(args.selection, selection_copy)
    shutil.copyfile(args.camera_tcp, camera_tcp_copy)
    accepted, rejected = [], {}
    items = selection.get("episodes", [])
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        items = items[:args.limit]
    for item in items:
        episode_id = Path(item.get("raw_episode", "unknown")).name
        try:
            episode = _load_episode(item, camera_tcp)
            meta = _build_one(
                episode, args.out, rate_hz=args.rate_hz,
                obs_horizon=args.obs_horizon,
                action_horizon=args.action_horizon,
                camera_tcp_path=camera_tcp_copy,
                camera_tcp_config=camera_config,
                gap_scale_verified=args.gap_scale_verified,
            )
            accepted.append(meta)
            print(f"{episode_id}: {meta['training_rows']} rows", flush=True)
        except Exception as error:
            rejected[episode_id] = str(error).splitlines()[0]
            print(f"{episode_id}: rejected - {rejected[episode_id]}", flush=True)
    index = {
        "schema": SCHEMA,
        "status": (
            "READY" if accepted and all(item["physical_deployment_ready"] for item in accepted)
            else "PROVISIONAL_CALIBRATION_NOT_FOR_PHYSICAL_DEPLOYMENT"
        ),
        "selection": selection_copy.name,
        "selection_source": str(args.selection),
        "selection_sha256": _sha256(selection_copy),
        "camera_tcp": camera_tcp_copy.name,
        "camera_tcp_source": str(args.camera_tcp),
        "camera_tcp_sha256": _sha256(camera_tcp_copy),
        "camera_tcp_status": camera_config.get("status"),
        "gripper_scale_verified": bool(args.gap_scale_verified),
        "episodes": [item["episode_id"] for item in accepted],
        "n_episodes": len(accepted),
        "n_rows": sum(int(item["training_rows"]) for item in accepted),
        "rejected": rejected,
        "rate_hz": args.rate_hz,
        "observation_horizon": args.obs_horizon,
        "action_horizon": args.action_horizon,
        "robot_base_alignment_applied": False,
        "robot_ik_applied": False,
        "physical_deployment_ready": bool(
            accepted and all(item["physical_deployment_ready"] for item in accepted)
        ),
    }
    (args.out / "dataset.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: index[key] for key in (
        "status", "n_episodes", "n_rows", "rejected")}, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
