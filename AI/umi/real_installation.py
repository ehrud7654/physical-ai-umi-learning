"""ROS/MuJoCo-free offline interpretation of the installed arm limits."""

from __future__ import annotations

from typing import Any

import numpy as np

from umi.real_fk import ARM_JOINTS


def arm_ranges(real: dict[str, Any]) -> np.ndarray:
    """Intersect URDF ranges with the provisional installed wrist-roll cap.

    This screens offline IK candidates only. It does not certify swept
    collision clearance or authorise commands on the physical robot.
    """
    ranges = np.asarray([real["limits_rad"][name] for name in ARM_JOINTS],
                        dtype=float)
    if ranges.shape != (5, 2) or not np.isfinite(ranges).all():
        raise ValueError("real limits_rad must contain five finite [lo, hi] rows")
    if np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("real limits_rad must have lo < hi")
    installation = real.get("wrist_roll_installation")
    if not isinstance(installation, dict):
        raise ValueError("wrist_roll installation constraint is required")
    try:
        cap = float(installation["diagnostic_preflight_upper_rad"])
        reported_stop = float(installation["reported_stop_upper_rad"])
        first_intersection = float(installation["first_bvh_intersection_sample_deg"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid wrist_roll installation constraint") from error
    if (not np.isfinite([cap, reported_stop, first_intersection]).all()
            or not ranges[4, 0] < cap < min(
                ranges[4, 1], reported_stop, np.deg2rad(first_intersection))):
        raise ValueError(
            "wrist_roll diagnostic cap must precede the reported stop and mesh intersection")
    ranges[4, 1] = cap
    return ranges
