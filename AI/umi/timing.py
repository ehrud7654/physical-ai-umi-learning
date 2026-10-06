"""Robot-time scheduling helpers for relative UMI action chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicPeaks:
    """Peak command-space derivatives for one waypoint trajectory."""

    arm_speed_rad_s: float
    arm_accel_rad_s2: float
    gap_speed_m_s: float
    gap_accel_m_s2: float


@dataclass(frozen=True)
class UniformTimeSchedule:
    """Original waypoints with robot-time timestamps stretched uniformly."""

    waypoint_time_s: tuple[float, ...]
    time_scale: float
    source: DynamicPeaks
    scheduled: DynamicPeaks


def required_time_scale(*, arm_speed: float, arm_accel: float,
                        gap_speed: float, gap_accel: float,
                        max_arm_speed: float, max_arm_accel: float,
                        max_gap_speed: float, max_gap_accel: float) -> float:
    """Return the uniform duration multiplier required by all dynamic limits.

    Uniformly stretching time by ``s`` divides velocity by ``s`` and
    acceleration by ``s**2``. Source timestamps and training labels remain
    untouched; this value belongs to the robot-time command scheduler.
    """
    values = np.asarray([
        arm_speed, arm_accel, gap_speed, gap_accel,
        max_arm_speed, max_arm_accel, max_gap_speed, max_gap_accel,
    ], dtype=float)
    if (not np.isfinite(values).all() or np.any(values[:4] < 0)
            or np.any(values[4:] <= 0)):
        raise ValueError("dynamic peaks must be non-negative and limits positive")
    return float(max(
        1.0,
        arm_speed / max_arm_speed,
        np.sqrt(arm_accel / max_arm_accel),
        gap_speed / max_gap_speed,
        np.sqrt(gap_accel / max_gap_accel),
    ))


def trajectory_dynamic_peaks(arm_positions_rad: np.ndarray,
                             gripper_width_m: np.ndarray,
                             waypoint_time_s: np.ndarray) -> DynamicPeaks:
    """Measure discrete velocity and acceleration using actual waypoint times."""
    arm = np.asarray(arm_positions_rad, dtype=float)
    gap = np.asarray(gripper_width_m, dtype=float)
    time = np.asarray(waypoint_time_s, dtype=float)
    if (arm.ndim != 2 or arm.shape[1] != 5 or gap.shape != (len(arm),)
            or time.shape != (len(arm),) or len(arm) < 2
            or not np.isfinite(arm).all() or not np.isfinite(gap).all()
            or not np.isfinite(time).all()):
        raise ValueError("trajectory must contain finite arm[T,5], gap[T], time[T]")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("waypoint times must increase strictly")
    arm_velocity = np.diff(arm, axis=0) / dt[:, None]
    gap_velocity = np.diff(gap) / dt
    if len(arm_velocity) > 1:
        accel_dt = (dt[:-1] + dt[1:]) / 2.0
        arm_acceleration = np.diff(arm_velocity, axis=0) / accel_dt[:, None]
        gap_acceleration = np.diff(gap_velocity) / accel_dt
    else:
        arm_acceleration = np.empty((0, 5), dtype=float)
        gap_acceleration = np.empty(0, dtype=float)
    return DynamicPeaks(
        arm_speed_rad_s=float(np.abs(arm_velocity).max(initial=0.0)),
        arm_accel_rad_s2=float(np.abs(arm_acceleration).max(initial=0.0)),
        gap_speed_m_s=float(np.abs(gap_velocity).max(initial=0.0)),
        gap_accel_m_s2=float(np.abs(gap_acceleration).max(initial=0.0)),
    )


def uniform_time_schedule(arm_positions_rad: np.ndarray,
                          gripper_width_m: np.ndarray, *,
                          source_period_s: float,
                          max_arm_speed: float, max_arm_accel: float,
                          max_gap_speed: float,
                          max_gap_accel: float) -> UniformTimeSchedule:
    """Preserve waypoint values and stretch only their robot execution times.

    The first row is the current robot state at time zero. Remaining rows are
    future commands. This function does not interpolate, clip, or otherwise
    alter the learned path. A trajectory driver must honour the returned
    cumulative timestamps.
    """
    period = float(source_period_s)
    limits = np.asarray([
        max_arm_speed, max_arm_accel, max_gap_speed, max_gap_accel,
    ], dtype=float)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("source_period_s must be finite and positive")
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("dynamic limits must be finite and positive")
    arm = np.asarray(arm_positions_rad, dtype=float)
    source_time = np.arange(len(arm), dtype=float) * period
    source = trajectory_dynamic_peaks(
        arm, np.asarray(gripper_width_m, dtype=float), source_time)
    scale = required_time_scale(
        arm_speed=source.arm_speed_rad_s,
        arm_accel=source.arm_accel_rad_s2,
        gap_speed=source.gap_speed_m_s,
        gap_accel=source.gap_accel_m_s2,
        max_arm_speed=float(max_arm_speed),
        max_arm_accel=float(max_arm_accel),
        max_gap_speed=float(max_gap_speed),
        max_gap_accel=float(max_gap_accel),
    )
    scheduled_time = source_time * scale
    scheduled = trajectory_dynamic_peaks(
        arm, np.asarray(gripper_width_m, dtype=float), scheduled_time)
    tolerance = 1e-9
    if (scheduled.arm_speed_rad_s > max_arm_speed + tolerance
            or scheduled.arm_accel_rad_s2 > max_arm_accel + tolerance
            or scheduled.gap_speed_m_s > max_gap_speed + tolerance
            or scheduled.gap_accel_m_s2 > max_gap_accel + tolerance):
        raise RuntimeError("uniform time schedule did not satisfy dynamic limits")
    return UniformTimeSchedule(
        waypoint_time_s=tuple(float(value) for value in scheduled_time),
        time_scale=scale,
        source=source,
        scheduled=scheduled,
    )
