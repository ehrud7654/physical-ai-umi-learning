"""Offline, all-or-nothing handoff from an EEF policy to five-joint waypoints.

This module prepares values only. It does not create ROS messages, check
continuous swept collision, or authorize any physical motor command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from contract.episode import GRIPPER_MAX_GAP_M
from umi.arm_preflight import ArmCommand
from umi.real_fk import ARM_JOINTS
from umi.timing import UniformTimeSchedule, uniform_time_schedule


@dataclass(frozen=True)
class JointChunkHandoff:
    """Eight future targets: index 0 is the first target after observation."""

    joint_names: tuple[str, ...]
    arm_positions_rad: np.ndarray  # (8, 5), ordered as joint_names
    gripper_width_m: np.ndarray  # (8,), absolute contact-surface gap
    time_from_start_s: np.ndarray  # (8,), relative to the current-state snapshot
    timing: UniformTimeSchedule


@dataclass(frozen=True)
class JointExecutionPrefix:
    """Offline prefix for one observe-execute-reobserve cycle; never a ROS goal."""

    future_target_indices: tuple[int, ...]
    arm_positions_rad: np.ndarray
    gripper_width_m: np.ndarray
    time_from_start_s: np.ndarray


def select_future_target_prefix(
    chunk: JointChunkHandoff, *, execute_steps: int,
) -> JointExecutionPrefix:
    """Select future indices [0, execute_steps), with no current-state row.

    Track B's official-UMI ``absolute[1:1+action_steps]`` has different
    indexing and must not be applied to this v10-style future-only chunk.
    Times are the validated, possibly retimed schedule; four targets do not
    necessarily mean 0.4 seconds of physical execution.
    """
    if (not isinstance(execute_steps, (int, np.integer))
            or isinstance(execute_steps, (bool, np.bool_))
            or not 1 <= execute_steps <= 8):
        raise ValueError("execute_steps must be an integer in [1, 8]")
    arm = np.array(chunk.arm_positions_rad[:execute_steps], copy=True)
    gap = np.array(chunk.gripper_width_m[:execute_steps], copy=True)
    times = np.array(chunk.time_from_start_s[:execute_steps], copy=True)
    if (arm.shape != (execute_steps, 5)
            or gap.shape != (execute_steps,)
            or times.shape != (execute_steps,)):
        raise ValueError("joint chunk has invalid future-target shapes")
    for value in (arm, gap, times):
        value.setflags(write=False)
    return JointExecutionPrefix(
        future_target_indices=tuple(range(execute_steps)),
        arm_positions_rad=arm,
        gripper_width_m=gap,
        time_from_start_s=times,
    )


def prepare_joint_chunk_handoff(
    commands: Sequence[ArmCommand], *, current_arm_rad: np.ndarray,
    current_gap_m: float, ranges_rad: np.ndarray,
    installed_wrist_roll_upper_rad: float, max_gap_m: float,
    source_period_s: float, max_arm_speed: float, max_arm_accel: float,
    max_gap_speed: float, max_gap_accel: float, max_time_scale: float,
) -> JointChunkHandoff:
    """Validate and retime exactly eight preflighted waypoints as one unit.

    ``installed_wrist_roll_upper_rad`` is the installation-specific offline
    cap and is intersected with ``ranges_rad`` even if those are raw URDF
    limits. A caller must independently check the
    synchronized TCP/state snapshot, IK residuals and swept collisions.
    The time scale is diagnostic until a robot-time policy is agreed.
    """
    if len(commands) != 8:
        raise ValueError("relative chunk must contain exactly eight waypoints")
    current = np.asarray(current_arm_rad, dtype=float)
    ranges = np.asarray(ranges_rad, dtype=float)
    gap0 = float(current_gap_m)
    maximum = float(max_gap_m)
    wrist_cap = float(installed_wrist_roll_upper_rad)
    scale_limit = float(max_time_scale)
    if (current.shape != (5,) or ranges.shape != (5, 2)
            or not np.isfinite(current).all() or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] >= ranges[:, 1])
            or not np.isfinite([gap0, maximum, wrist_cap, scale_limit]).all()
            or not np.isclose(maximum, GRIPPER_MAX_GAP_M, atol=1e-12, rtol=0)
            or not ranges[4, 0] < wrist_cap or scale_limit < 1
            or wrist_cap > ranges[4, 1]
            or np.any(current < ranges[:, 0])
            or not 0 <= gap0 <= maximum):
        raise ValueError("invalid current state or installed robot limits")
    ranges = ranges.copy()
    ranges[4, 1] = wrist_cap
    if np.any(current > ranges[:, 1]):
        raise ValueError("current state exceeds installed wrist-roll limit")

    arm = np.asarray([command.arm_positions_rad for command in commands], dtype=float)
    gap = np.asarray([command.gripper_width_m for command in commands], dtype=float)
    if (arm.shape != (8, 5) or gap.shape != (8,)
            or not np.isfinite(arm).all() or not np.isfinite(gap).all()
            or np.any(arm < ranges[:, 0]) or np.any(arm > ranges[:, 1])
            or np.any(gap < 0) or np.any(gap > maximum)):
        raise ValueError("joint chunk exceeds installed arm or gap limits")

    timing = uniform_time_schedule(
        np.vstack((current, arm)), np.r_[gap0, gap],
        source_period_s=source_period_s,
        max_arm_speed=max_arm_speed, max_arm_accel=max_arm_accel,
        max_gap_speed=max_gap_speed, max_gap_accel=max_gap_accel,
    )
    if timing.time_scale > scale_limit + 1e-9:
        raise ValueError("joint chunk requires excessive time expansion")
    times = np.asarray(timing.waypoint_time_s[1:], dtype=float)
    if times.shape != (8,) or np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise RuntimeError("invalid scheduled waypoint times")
    arm, gap, times = arm.copy(), gap.copy(), times.copy()
    for value in (arm, gap, times):
        value.setflags(write=False)
    return JointChunkHandoff(
        joint_names=ARM_JOINTS,
        arm_positions_rad=arm,
        gripper_width_m=gap,
        time_from_start_s=times,
        timing=timing,
    )
