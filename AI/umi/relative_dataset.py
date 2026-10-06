"""Provisional camera-centric UMI sequence contract.

This module deliberately does not modify ``contract/episode.py``.  The existing
0.3 contract stores SO-101 joint next-state labels, while this pilot stores
hardware-independent relative pinch trajectories for an action-chunk policy.
Track A/B agreement is required before this becomes the shared contract.
"""

from __future__ import annotations

import json
import csv
import io
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from contract.episode import GRIPPER_MAX_GAP_M
from umi.action_decode import rotation_6d_rows_to_matrix
from umi.camera_frames import rigid
from umi.ik import quat_to_matrix


SCHEMA = "umi_relative_chunk/0.2.0-provisional"
ACTION_DIM = 10
MAX_GRIPPER_WIDTH_M = GRIPPER_MAX_GAP_M


def load_longest_relative_run(
    bundle: Path, quality: dict[str, Any], t_camera_pinch: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read the longest contiguous measured run without simulator dependencies.

    Returned poses are relative to the first valid pinch pose. ``source_rows``
    are row positions in poses.csv; callers must use the pose ``index`` column
    when joining another stream.
    """
    camera_pinch = rigid(t_camera_pinch, name="t_camera_pinch")
    with zipfile.ZipFile(bundle) as archive:
        poses = list(csv.DictReader(io.StringIO(
            archive.read("poses.csv").decode("utf-8-sig"))))
        grips = list(csv.DictReader(io.StringIO(
            archive.read("gripper.csv").decode("utf-8-sig"))))
    if not poses or len(poses) != len(grips):
        raise ValueError(f"pose/gap row mismatch: {bundle.name}")
    allowed = np.zeros(len(poses), dtype=bool)
    segment_ids = np.full(len(poses), -1, dtype=int)
    for segment_id, segment in enumerate(quality.get("usable_segments", [])):
        if len(segment) != 2:
            raise ValueError("usable segment must be [start, end)")
        start, end = map(int, segment)
        if not (0 <= start < end <= len(poses)):
            raise ValueError(f"invalid usable segment: {segment}")
        allowed[start:end] = True
        segment_ids[start:end] = segment_id

    transforms: list[np.ndarray] = []
    gaps: list[float] = []
    valid: list[bool] = []
    for row_position, (pose, grip) in enumerate(zip(poses, grips)):
        if int(pose["index"]) != int(grip["frame_index"]):
            raise ValueError(f"pose/gap index mismatch at row {row_position}")
        world_camera = np.eye(4)
        world_camera[:3, :3] = quat_to_matrix(np.asarray(
            [float(pose[key]) for key in ("qw", "qx", "qy", "qz")]))
        world_camera[:3, 3] = [float(pose[key]) for key in ("x", "y", "z")]
        transforms.append(world_camera @ camera_pinch)
        gap = float(grip["gap_m"]) if grip["gap_m"].strip() else np.nan
        gaps.append(gap)
        valid.append(bool(
            allowed[row_position]
            and pose["tracking"] == "TRACKING"
            and grip["status"] in ("D", "M")
            and np.isfinite(gap)
            and 0 <= gap <= MAX_GRIPPER_WIDTH_M
        ))

    runs: list[list[int]] = []
    for row_position, ok in enumerate(valid):
        if not ok:
            continue
        if (not runs or row_position != runs[-1][-1] + 1
                or segment_ids[row_position] != segment_ids[runs[-1][-1]]):
            runs.append([])
        runs[-1].append(row_position)
    chosen = max(runs, key=len, default=[])
    if len(chosen) < 2:
        raise ValueError(f"no contiguous measured run: {bundle.name}")
    origin_inverse = np.linalg.inv(transforms[chosen[0]])
    relative = np.stack([origin_inverse @ transforms[index] for index in chosen])
    return relative, np.asarray([gaps[index] for index in chosen]), chosen


def transform_to_vector(transform: np.ndarray, gripper_width_m: float) -> np.ndarray:
    """Encode one relative SE(3) pose as xyz + first two rotation rows + gap."""
    value = rigid(transform, name="relative_transform")
    gap = float(gripper_width_m)
    if not np.isfinite(gap) or not 0 <= gap <= MAX_GRIPPER_WIDTH_M:
        raise ValueError(f"gripper width must be within [0, {MAX_GRIPPER_WIDTH_M}] m")
    return np.r_[value[:3, 3], value[:2, :3].reshape(-1), gap]


def vector_to_transform(vector: np.ndarray) -> np.ndarray:
    """Decode the geometric portion of a 10-D relative action vector."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError("relative action must be a finite (10,) vector")
    result = np.eye(4)
    result[:3, :3] = rotation_6d_rows_to_matrix(value[3:9])
    result[:3, 3] = value[:3]
    return result


def relative_vector(reference: np.ndarray, target: np.ndarray, gap_m: float) -> np.ndarray:
    """Express target in the reference pinch frame, with absolute gripper gap."""
    ref = rigid(reference, name="reference")
    dst = rigid(target, name="target")
    return transform_to_vector(np.linalg.inv(ref) @ dst, gap_m)


def validate_arrays(arrays: dict[str, np.ndarray], *, obs_horizon: int,
                    action_horizon: int) -> list[str]:
    problems: list[str] = []
    required = {
        "image": (None, obs_horizon, 3, 224, 224),
        "proprio": (None, obs_horizon, ACTION_DIM),
        "action": (None, action_horizon, ACTION_DIM),
        "observation_timestamp": (None, obs_horizon),
        "action_timestamp": (None, action_horizon),
        "source_row": (None,),
    }
    missing = sorted(set(required) - set(arrays))
    if missing:
        return [f"missing arrays: {missing}"]
    n = int(arrays["action"].shape[0]) if arrays["action"].ndim else -1
    for key, shape in required.items():
        expected = (n, *shape[1:])
        if arrays[key].shape != expected:
            problems.append(f"{key}.shape={arrays[key].shape}, expected {expected}")
    if arrays["image"].dtype != np.uint8:
        problems.append(f"image dtype must be uint8, got {arrays['image'].dtype}")
    for key in ("proprio", "action"):
        if arrays[key].dtype != np.float32:
            problems.append(f"{key} dtype must be float32, got {arrays[key].dtype}")
        if not np.isfinite(arrays[key]).all():
            problems.append(f"{key} contains NaN or inf")
        elif np.any((arrays[key][..., 9] < 0)
                    | (arrays[key][..., 9] > MAX_GRIPPER_WIDTH_M)):
            problems.append(f"{key} gripper width exceeds [0, {MAX_GRIPPER_WIDTH_M}] m")
    for key in ("observation_timestamp", "action_timestamp"):
        if arrays[key].dtype != np.float64:
            problems.append(f"{key} dtype must be float64, got {arrays[key].dtype}")
        if not np.isfinite(arrays[key]).all():
            problems.append(f"{key} contains NaN or inf")
    observation_timestamp = arrays["observation_timestamp"]
    if observation_timestamp.shape == (n, obs_horizon):
        if n and not np.all(np.diff(observation_timestamp, axis=1) > 0):
            problems.append("timestamps within each observation history must increase")
        if n > 1 and not np.all(np.diff(observation_timestamp[:, -1]) > 0):
            problems.append("current observation timestamps must increase between rows")
        if n and np.any(
            arrays["action_timestamp"] <= observation_timestamp[:, -1, None]
        ):
            problems.append("every action target must be in the future")
    # The last proprio pose is the current pose relative to itself.
    if n:
        identity = np.eye(4)
        for value in arrays["proprio"][:, -1]:
            if not np.allclose(vector_to_transform(value), identity, atol=1e-5, rtol=0):
                problems.append("last proprio pose must be identity")
                break
    return problems


def write_episode(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    obs_horizon = int(meta["observation_horizon"])
    action_horizon = int(meta["action_horizon"])
    problems = validate_arrays(arrays, obs_horizon=obs_horizon,
                               action_horizon=action_horizon)
    if problems:
        raise ValueError("invalid relative episode: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
