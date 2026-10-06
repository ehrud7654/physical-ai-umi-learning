"""Offline installed wrist-roll cap regression; never commands hardware."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.arm_preflight import ArmSolution, prepare_arm_commands
from umi.relative_robot_preflight import arm_ranges, real_limits


CONFIG = Path(__file__).resolve().parents[1] / "configs/real/so101_ver1.json"


class _Policy:
    def predict_action(self, observation):
        del observation
        return {"action_pred": np.array(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.045]])}


class _Solver:
    def __init__(self, wrist_roll: float):
        self.wrist_roll = wrist_roll

    def solve(self, target_pose, *, seed_rad, gripper_width_m):
        del target_pose, seed_rad, gripper_width_m
        return ArmSolution(np.array([0.0, 0.0, 0.0, 0.0, self.wrist_roll]),
                           0.0, 0.0, 0.0, True)


class Tests(unittest.TestCase):
    def test_urdf_limit_remains_raw_and_offline_cap_is_stricter(self):
        real = real_limits(CONFIG)
        ranges = arm_ranges(real)
        self.assertAlmostEqual(real["limits_rad"]["wrist_roll"][1], 2.841206309382605)
        self.assertAlmostEqual(ranges[4, 1], np.deg2rad(60.0))
        self.assertLess(ranges[4, 1], np.deg2rad(65.0))
        self.assertLess(ranges[4, 1], real["wrist_roll_installation"]["reported_stop_upper_rad"])
        installation = real["wrist_roll_installation"]
        computed_stop = np.deg2rad(360 * (
            installation["reported_positive_stop_tick"]
            - real["zero_tick"]["wrist_roll"]
        ) / installation["reported_ticks_per_turn"])
        self.assertAlmostEqual(installation["reported_stop_upper_rad"], computed_stop)
        self.assertFalse(real["wrist_roll_installation"]["swept_clearance_verified"])

    def test_above_cap_is_rejected_as_complete_chunk(self):
        ranges = arm_ranges(real_limits(CONFIG))
        with self.assertRaisesRegex(ValueError, "arm range"):
            prepare_arm_commands(
                _Policy(), {}, t_current=np.eye(4), arm_current_rad=np.zeros(5),
                gripper_current_m=0.045, solver=_Solver(ranges[4, 1] + 0.01),
                ranges_rad=ranges, gap_range_m=[0.0, 0.09],
                max_step_rad=np.full(5, 2.0), max_gap_step_m=0.09,
                max_position_error_m=0.005, max_axis_error_deg=5.0,
                max_roll_error_deg=5.0)

    def test_missing_or_bad_installation_constraint_fails_closed(self):
        real = real_limits(CONFIG)
        for key, value in (("diagnostic_preflight_upper_rad", np.deg2rad(65.0)),
                           ("reported_stop_upper_rad", float("nan"))):
            invalid = deepcopy(real)
            invalid["wrist_roll_installation"][key] = value
            with self.assertRaises(ValueError):
                arm_ranges(invalid)
        invalid = deepcopy(real)
        del invalid["wrist_roll_installation"]
        with self.assertRaises(ValueError):
            arm_ranges(invalid)

    def test_cap_is_configurable_without_code_change(self):
        real = deepcopy(real_limits(CONFIG))
        real["wrist_roll_installation"]["diagnostic_preflight_upper_rad"] = 0.9
        self.assertAlmostEqual(arm_ranges(real)[4, 1], 0.9)


if __name__ == "__main__":
    unittest.main()
