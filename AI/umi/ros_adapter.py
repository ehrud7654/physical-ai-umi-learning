"""ROS message boundary for UMI, importable without a ROS installation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from contract.episode import GRIPPER_MAX_GAP_M
from umi.arm_preflight import ArmCommand
from umi.chunk_runtime import Snapshot
from umi.real_fk import ARM_JOINTS, SO101UrdfFK, snapshot_from_joint_state


ARM_COMMAND_TOPIC = "/arm_controller/joint_trajectory"
GRIPPER_COMMAND_TOPIC = "/gripper_controller/commands"
JOINT_STATE_TOPIC = "/joint_states"


@dataclass(frozen=True)
class JointTrajectoryFields:
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    time_from_start_sec: int
    time_from_start_nanosec: int


def _finite_nonnegative(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def joint_state_to_runtime_snapshot(message, fk: SO101UrdfFK, *,
                                    received_ros_time_s: float,
                                    received_monotonic_s: float,
                                    max_transport_age_s: float) -> Snapshot:
    """Validate JointState and translate its ROS timestamp to monotonic time."""
    received_ros = _finite_nonnegative(received_ros_time_s, "received_ros_time_s")
    received_mono = _finite_nonnegative(received_monotonic_s, "received_monotonic_s")
    max_age = _finite_nonnegative(max_transport_age_s, "max_transport_age_s")
    if max_age == 0:
        raise ValueError("max_transport_age_s must be positive")
    try:
        stamp = message.header.stamp
        state = snapshot_from_joint_state(
            message.name, message.position, stamp.sec, stamp.nanosec, fk
        )
    except AttributeError as error:
        raise ValueError("invalid sensor_msgs/JointState shape") from error
    age = received_ros - state.ros_time_s
    if not math.isfinite(age) or age < 0 or age > max_age:
        raise ValueError("future or stale JointState timestamp")
    sampled_monotonic = received_mono - age
    if sampled_monotonic < 0:
        raise ValueError("JointState predates monotonic clock origin")
    return Snapshot(timestamp=sampled_monotonic, payload=state)


def arm_trajectory_fields(command: ArmCommand, *, duration_s: float) -> JointTrajectoryFields:
    """Return the exact fields for one JointTrajectory point."""
    duration = float(duration_s)
    joints = np.asarray(command.arm_positions_rad, dtype=float)
    if (joints.shape != (5,) or not np.isfinite(joints).all()
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("invalid arm command or duration")
    sec = math.floor(duration)
    nanosec = round((duration - sec) * 1_000_000_000)
    if nanosec == 1_000_000_000:
        sec, nanosec = sec + 1, 0
    if (sec == 0 and nanosec == 0) or sec > 2**31 - 1:
        raise ValueError('duration outside positive ROS duration range')
    return JointTrajectoryFields(
        joint_names=ARM_JOINTS,
        positions=tuple(float(value) for value in joints),
        time_from_start_sec=sec,
        time_from_start_nanosec=nanosec,
    )


def gripper_command_data(
    command: ArmCommand, *, max_gap_m: float = GRIPPER_MAX_GAP_M
) -> tuple[float]:
    """Return Float64MultiArray.data for the full contact-surface width."""
    gap = float(command.gripper_width_m)
    maximum = float(max_gap_m)
    if not math.isclose(maximum, GRIPPER_MAX_GAP_M, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("robot max_gap_m does not match the episode contract")
    if (not math.isfinite(gap) or not math.isfinite(maximum) or maximum <= 0
            or not 0.0 <= gap <= maximum):
        raise ValueError("gripper width must be full gap in 0..max_gap_m")
    return (gap,)


def fill_ros_command_messages(command: ArmCommand, trajectory_message,
                              trajectory_point, gripper_message, *, duration_s: float,
                              max_gap_m: float = GRIPPER_MAX_GAP_M):
    """Populate real ROS message instances without importing ROS packages here."""
    fields = arm_trajectory_fields(command, duration_s=duration_s)
    gap_data = list(gripper_command_data(command, max_gap_m=max_gap_m))
    if hasattr(trajectory_message, 'header'):
        trajectory_message.header.stamp.sec = 0
        trajectory_message.header.stamp.nanosec = 0
    trajectory_message.joint_names = list(fields.joint_names)
    trajectory_point.positions = list(fields.positions)
    trajectory_point.velocities = []
    trajectory_point.accelerations = []
    trajectory_point.effort = []
    trajectory_point.time_from_start.sec = fields.time_from_start_sec
    trajectory_point.time_from_start.nanosec = fields.time_from_start_nanosec
    trajectory_message.points = [trajectory_point]
    gripper_message.data = gap_data
    return trajectory_message, gripper_message
