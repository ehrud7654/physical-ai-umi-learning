"""Fail-closed offline bridge from one timed real observation to joint targets.

The prediction is supplied by the caller; this module performs no learned
inference, ROS publish, continuous collision check, or motor command.  Its
output is diagnostic until the real sensor/IK/calibration/safety contracts
are independently approved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from contract.episode import GRIPPER_MAX_GAP_M
from umi.action_decode import decode_relative_actions
from umi.arm_preflight import prepare_arm_commands
from umi.chunk_runtime import Snapshot
from umi.joint_chunk_handoff import (
    JointChunkHandoff, JointExecutionPrefix, prepare_joint_chunk_handoff,
    select_future_target_prefix,
)
from umi.policy_preflight import extract_action_pred
from umi.ik import matrix_to_quat, roll_residual_deg
from umi.real_fk import JointStateSnapshot, SO101UrdfFK
from umi.real_installation import arm_ranges
from umi.relative_dataset import relative_vector


@dataclass(frozen=True)
class OfflineRealHandoff:
    """Checked values only; never a ROS goal or hardware approval."""

    status: str
    chunk: JointChunkHandoff
    execution_prefix: JointExecutionPrefix
    camera_timestamp_s: tuple[float, float]
    state_timestamp_s: float
    camera_state_skew_s: float


@dataclass(frozen=True)
class RealObservation:
    """A validated H=2 v10-shaped input; camera calibration is still external."""

    image: np.ndarray  # (2, 3, 224, 224), uint8, canonical camera orientation
    proprio: np.ndarray  # (2, 10), current-TCP-relative SE(3) + absolute gap
    camera_timestamp_s: tuple[float, float]
    current_state_snapshot: Snapshot
    calibration_id: str


class _SuppliedPrediction:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action.copy()

    def predict_action(self, _observation: Any) -> dict[str, np.ndarray]:
        return {"action_pred": self.action.copy()}


def _checked_state(snapshot: Snapshot, fk: SO101UrdfFK):
    if not isinstance(snapshot, Snapshot) or not isinstance(
        snapshot.payload, JointStateSnapshot
    ):
        raise ValueError("state_snapshot must contain a converted JointStateSnapshot")
    state = snapshot.payload
    arm = np.asarray(state.arm_rad, dtype=float).copy()
    pose = np.asarray(state.t_base_tcp, dtype=float).copy()
    gap = float(state.gripper_width_m)
    if (arm.shape != (5,) or pose.shape != (4, 4)
            or not np.isfinite(arm).all() or not np.isfinite(pose).all()
            or not math.isfinite(gap) or not 0.0 <= gap <= GRIPPER_MAX_GAP_M):
        raise ValueError("invalid joint/TCP snapshot")
    measured_fk = fk.pose(arm)
    if not np.allclose(measured_fk, pose, rtol=0, atol=1e-6):
        raise ValueError("snapshot TCP does not match the released FK")
    return arm, pose, gap


def _frozen_state_copy(snapshot: Snapshot) -> Snapshot:
    """Keep a later ROS buffer mutation from changing an accepted observation."""
    state = snapshot.payload
    arm = np.array(state.arm_rad, dtype=float, copy=True)
    pose = np.array(state.t_base_tcp, dtype=float, copy=True)
    arm.setflags(write=False)
    pose.setflags(write=False)
    return Snapshot(float(snapshot.timestamp), JointStateSnapshot(
        ros_time_s=float(state.ros_time_s), arm_rad=arm,
        gripper_width_m=float(state.gripper_width_m), t_base_tcp=pose,
    ))


def _verify_solver_with_fk(commands, targets, fk: SO101UrdfFK, *,
                           max_position_error_m: float,
                           max_axis_error_deg: float,
                           max_roll_error_deg: float) -> None:
    """Independently check the injected solver's claimed residuals."""
    for index, (command, target) in enumerate(zip(commands, targets.poses)):
        actual = fk.pose(command.arm_positions_rad)
        translation_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        axis_cosine = float(np.clip(np.dot(actual[:3, 2], target[:3, 2]), -1.0, 1.0))
        axis_error = float(np.degrees(np.arccos(axis_cosine)))
        roll_error = roll_residual_deg(
            matrix_to_quat(target[:3, :3]),
            actual[:3, 2], actual[:3, 0],
        )
        if (not np.isfinite([translation_error, axis_error, roll_error]).all()
                or translation_error > max_position_error_m + 1e-9
                or axis_error > max_axis_error_deg + 1e-7
                or roll_error > max_roll_error_deg + 1e-7):
            raise ValueError(
                f"step {index}: URDF FK contradicts injected IK residual "
                f"(position={translation_error:.6g}m, axis={axis_error:.6g}deg, "
                f"roll={roll_error:.6g}deg)"
            )


def build_v10_real_observation(
    image_history: Any, *, camera_timestamp_s: Any,
    state_history: Any, now_monotonic_s: float,
    fk: SO101UrdfFK, calibration_id: str,
    max_age_s: float, max_sync_error_s: float,
    max_history_period_error_s: float,
) -> RealObservation:
    """Build the v10 H=2 policy input from two preprocessed RGB/state pairs.

    The caller must independently establish JPEG orientation, S22 optical
    calibration and that both histories use the same TCP definition. This
    function never fetches images, predicts actions or sends commands.
    """
    if not isinstance(fk, SO101UrdfFK):
        raise ValueError("fk must be the released SO-101 base-to-TCP model")
    images = np.asarray(image_history)
    times = np.asarray(camera_timestamp_s, dtype=float)
    states = tuple(state_history)
    now = float(now_monotonic_s)
    age_limit = float(max_age_s)
    sync_limit = float(max_sync_error_s)
    period_error = float(max_history_period_error_s)
    if (images.shape != (2, 3, 224, 224) or images.dtype != np.uint8
            or times.shape != (2,) or len(states) != 2
            or not isinstance(calibration_id, str) or not calibration_id.strip()
            or not np.isfinite(times).all()
            or not np.isfinite([now, age_limit, sync_limit, period_error]).all()
            or age_limit <= 0 or sync_limit <= 0 or period_error < 0
            or times[0] >= times[1] or times[1] > now
            or now - times[1] > age_limit
            or abs(times[1] - times[0] - 0.1) > period_error):
        raise ValueError("invalid v10 RGB history, calibration or timestamps")
    poses, gaps = [], []
    for camera_time, snapshot in zip(times, states):
        if not isinstance(snapshot, Snapshot):
            raise ValueError("state history must contain converted snapshots")
        state_time = float(snapshot.timestamp)
        if (not math.isfinite(state_time) or state_time > now
                or abs(camera_time - state_time) > sync_limit):
            raise ValueError("RGB and joint-state history are not synchronized")
        _, pose, gap = _checked_state(snapshot, fk)
        poses.append(pose)
        gaps.append(gap)
    if states[0].timestamp >= states[1].timestamp or now - states[1].timestamp > age_limit:
        raise ValueError("state history is stale or not strictly increasing")
    proprio = np.stack([
        relative_vector(poses[1], pose, gap)
        for pose, gap in zip(poses, gaps)
    ]).astype(np.float32)
    images = images.copy()
    images.setflags(write=False)
    proprio.setflags(write=False)
    return RealObservation(
        image=images, proprio=proprio,
        camera_timestamp_s=(float(times[0]), float(times[1])),
        current_state_snapshot=_frozen_state_copy(states[1]),
        calibration_id=calibration_id,
    )


def prepare_offline_real_handoff(
    action_pred: Any, *, camera_timestamp_s: Any,
    prediction_observation_timestamp_s: float,
    state_snapshot: Snapshot, now_monotonic_s: float,
    fk: SO101UrdfFK, solver: Any, real_config: dict[str, Any],
    execute_steps: int, max_age_s: float, max_sync_error_s: float,
    max_history_period_error_s: float,
    max_time_scale: float, max_position_error_m: float,
    max_axis_error_deg: float, max_roll_error_deg: float,
) -> OfflineRealHandoff:
    """Check one H=2 observation, EEF chunk, IK and joint schedule as a unit.

    All timestamps here are in the *same monotonic seconds domain*. The ROS
    JointState must first pass ``joint_state_to_runtime_snapshot``.  The caller
    must prove that the supplied prediction used these exact two RGB frames;
    numeric timestamps alone cannot establish image content or camera/TCP
    calibration.  The injected five-axis solver must share the FK's base/TCP.
    """
    if not isinstance(state_snapshot, Snapshot) or not isinstance(
        state_snapshot.payload, JointStateSnapshot
    ):
        raise ValueError("state_snapshot must contain a converted JointStateSnapshot")
    if not isinstance(fk, SO101UrdfFK):
        raise ValueError("fk must be the released SO-101 base-to-TCP model")
    times = np.asarray(camera_timestamp_s, dtype=float)
    state_time = float(state_snapshot.timestamp)
    now = float(now_monotonic_s)
    predicted_at = float(prediction_observation_timestamp_s)
    age_limit = float(max_age_s)
    sync_limit = float(max_sync_error_s)
    period_error = float(max_history_period_error_s)
    if (times.shape != (2,) or not np.isfinite(times).all()
            or not np.isfinite([state_time, now, predicted_at,
                                age_limit, sync_limit, period_error]).all()
            or times[0] >= times[1] or age_limit <= 0 or sync_limit <= 0
            or period_error < 0
            or abs(times[1] - times[0] - 0.1) > period_error
            or times[1] > now or state_time > now
            or now - times[1] > age_limit or now - state_time > age_limit
            or abs(times[1] - state_time) > sync_limit
            or abs(times[1] - predicted_at) > sync_limit):
        raise ValueError("stale, unsynchronised or invalid observation timestamps")
    arm, pose, gap = _checked_state(state_snapshot, fk)
    ranges = arm_ranges(real_config)
    maximum_gap = float(real_config["max_gap_m"])
    if not np.isfinite(maximum_gap) or maximum_gap <= 0:
        raise ValueError("invalid configured gap")
    action = extract_action_pred(action_pred)
    if action.shape != (8, 10):
        raise ValueError("relative policy output must have eight future targets")
    commands = prepare_arm_commands(
        _SuppliedPrediction(action), None, t_current=pose,
        arm_current_rad=arm, gripper_current_m=gap, solver=solver,
        ranges_rad=ranges, gap_range_m=[0.0, maximum_gap],
        max_step_rad=ranges[:, 1] - ranges[:, 0],
        max_gap_step_m=maximum_gap,
        max_position_error_m=max_position_error_m,
        max_axis_error_deg=max_axis_error_deg,
        max_roll_error_deg=max_roll_error_deg,
    )
    targets = decode_relative_actions(action, pose)
    _verify_solver_with_fk(
        commands, targets, fk,
        max_position_error_m=max_position_error_m,
        max_axis_error_deg=max_axis_error_deg,
        max_roll_error_deg=max_roll_error_deg,
    )
    chunk = prepare_joint_chunk_handoff(
        commands, current_arm_rad=arm, current_gap_m=gap,
        ranges_rad=ranges,
        installed_wrist_roll_upper_rad=float(ranges[4, 1]),
        max_gap_m=maximum_gap, source_period_s=0.1,
        max_arm_speed=float(real_config["max_speed_rad_s"]),
        max_arm_accel=float(real_config["max_accel_rad_s2"]),
        max_gap_speed=float(real_config["max_gap_speed_m_s"]),
        max_gap_accel=float(real_config["max_gap_accel_m_s2"]),
        max_time_scale=max_time_scale,
    )
    prefix = select_future_target_prefix(chunk, execute_steps=execute_steps)
    return OfflineRealHandoff(
        status="OFFLINE_DIAGNOSTIC_ONLY", chunk=chunk,
        execution_prefix=prefix,
        camera_timestamp_s=(float(times[0]), float(times[1])),
        state_timestamp_s=state_time,
        camera_state_skew_s=abs(float(times[1]) - state_time),
    )
