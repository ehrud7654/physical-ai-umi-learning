"""Offline eight-waypoint handoff checks; no ROS or motor connection."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.arm_preflight import ArmCommand
from umi.joint_chunk_handoff import (
    prepare_joint_chunk_handoff, select_future_target_prefix,
)
from umi.real_fk import ARM_JOINTS


class JointChunkHandoffTests(unittest.TestCase):
    def setUp(self):
        self.ranges = np.asarray([[-2.0, 2.0]] * 4 + [[-2.7, 2.8412]])
        self.arguments = dict(
            current_arm_rad=np.zeros(5), current_gap_m=0.045,
            ranges_rad=self.ranges,
            installed_wrist_roll_upper_rad=np.deg2rad(60), max_gap_m=0.09,
            source_period_s=0.1, max_arm_speed=1.0, max_arm_accel=3.0,
            max_gap_speed=0.08, max_gap_accel=0.3, max_time_scale=2.0,
        )

    def commands(self, arm=None, gap=0.045):
        target = np.zeros(5) if arm is None else np.asarray(arm, dtype=float)
        return [ArmCommand(target.copy(), gap) for _ in range(8)]

    def test_exact_eight_waypoint_units_and_order(self):
        output = prepare_joint_chunk_handoff(self.commands(), **self.arguments)
        self.assertEqual(output.joint_names, ARM_JOINTS)
        self.assertEqual(output.arm_positions_rad.shape, (8, 5))
        self.assertEqual(output.gripper_width_m.shape, (8,))
        np.testing.assert_allclose(output.time_from_start_s, np.arange(1, 9) * 0.1)
        self.assertFalse(output.arm_positions_rad.flags.writeable)
        self.assertFalse(output.gripper_width_m.flags.writeable)

    def test_short_chunk_and_installed_wrist_limit_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            prepare_joint_chunk_handoff(self.commands()[:7], **self.arguments)
        over_cap = [0, 0, 0, 0, np.deg2rad(61)]
        with self.assertRaisesRegex(ValueError, "installed arm"):
            prepare_joint_chunk_handoff(self.commands(over_cap), **self.arguments)
        arguments = {**self.arguments, "current_arm_rad": np.asarray(over_cap)}
        with self.assertRaisesRegex(ValueError, "current state"):
            prepare_joint_chunk_handoff(self.commands(), **arguments)
        wrong_gap_contract = {**self.arguments, "max_gap_m": 0.08}
        with self.assertRaisesRegex(ValueError, "installed robot limits"):
            prepare_joint_chunk_handoff(self.commands(), **wrong_gap_contract)

    def test_gap_and_time_expansion_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "gap limits"):
            prepare_joint_chunk_handoff(self.commands(gap=0.091), **self.arguments)
        # A 0.2-rad first step in 0.1 s requires more than a 2x scale.
        with self.assertRaisesRegex(ValueError, "excessive time"):
            prepare_joint_chunk_handoff(
                self.commands([0.2, 0, 0, 0, 0]), **self.arguments)
        arguments = {**self.arguments, "max_time_scale": 8.0}
        output = prepare_joint_chunk_handoff(
            self.commands([0.2, 0, 0, 0, 0]), **arguments)
        self.assertGreater(output.timing.time_scale, 2.0)
        self.assertLessEqual(output.timing.scheduled.arm_speed_rad_s, 1.0)
        self.assertLessEqual(output.timing.scheduled.arm_accel_rad_s2, 3.0)
        self.assertTrue(np.all(np.diff(output.time_from_start_s) > 0))
        prefix = select_future_target_prefix(output, execute_steps=4)
        self.assertGreater(prefix.time_from_start_s[-1], 0.4)

    def test_four_executed_targets_start_at_first_future_target(self):
        commands = [ArmCommand(
            np.array([0.001 * (index + 1), 0, 0, 0, 0]), 0.045,
        ) for index in range(8)]
        chunk = prepare_joint_chunk_handoff(commands, **self.arguments)
        prefix = select_future_target_prefix(chunk, execute_steps=4)
        self.assertEqual(prefix.future_target_indices, (0, 1, 2, 3))
        np.testing.assert_allclose(
            prefix.arm_positions_rad[:, 0], [0.001, 0.002, 0.003, 0.004])
        np.testing.assert_allclose(prefix.time_from_start_s, [0.1, 0.2, 0.3, 0.4])
        self.assertFalse(prefix.arm_positions_rad.flags.writeable)
        with self.assertRaisesRegex(ValueError, "execute_steps"):
            select_future_target_prefix(chunk, execute_steps=0)
        with self.assertRaisesRegex(ValueError, "execute_steps"):
            select_future_target_prefix(chunk, execute_steps=9)


if __name__ == "__main__":
    unittest.main()
