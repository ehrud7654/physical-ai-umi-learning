"""Hardware-free regression tests for the provisional relative-chunk contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from umi.action_decode import decode_relative_actions
from umi.relative_dataset import (
    MAX_GRIPPER_WIDTH_M,
    relative_vector,
    validate_arrays,
    vector_to_transform,
)
from umi.timing import required_time_scale, uniform_time_schedule


def pose(xyz=(0, 0, 0), yaw_deg=0.0):
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = xyz
    return result


class RelativeDatasetTests(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        current = pose((0.2, -0.1, 0.3), 30)
        target = current @ pose((0.03, 0.01, -0.02), -20)
        encoded = relative_vector(current, target, 0.039)[None]
        decoded = decode_relative_actions(encoded, current)
        np.testing.assert_allclose(decoded.poses[0], target, atol=1e-9)
        np.testing.assert_allclose(decoded.gripper_width_m, [0.039])

    def test_global_alignment_cancels(self):
        current, target = pose((0.1, 0.2, 0.3), 15), pose((0.4, -0.2, 0.5), -40)
        world_change = pose((-2, 3, 1), 127)
        direct = relative_vector(current, target, 0.04)
        moved = relative_vector(world_change @ current, world_change @ target, 0.04)
        np.testing.assert_allclose(direct, moved, atol=1e-9)

    def test_current_proprio_is_identity(self):
        current = pose((1, 2, 3), 72)
        encoded = relative_vector(current, current, 0.05)
        np.testing.assert_allclose(vector_to_transform(encoded), np.eye(4), atol=1e-9)

    def test_first_future_target_advances_to_next_observation_anchor(self):
        current = pose((0.2, -0.1, 0.3), 30)
        following = current @ pose((0.03, 0.01, -0.02), -20)
        first_future = vector_to_transform(
            relative_vector(current, following, 0.05))
        next_history = vector_to_transform(
            relative_vector(following, current, 0.05))
        np.testing.assert_allclose(first_future @ next_history, np.eye(4), atol=1e-9)

    def test_gap_contract_rejects_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "gripper width"):
            relative_vector(np.eye(4), np.eye(4), MAX_GRIPPER_WIDTH_M + 1e-3)

    def test_observation_timestamps_follow_history_axis(self):
        identity = relative_vector(np.eye(4), np.eye(4), 0.04).astype(np.float32)
        arrays = {
            "image": np.zeros((2, 2, 3, 224, 224), dtype=np.uint8),
            "proprio": np.tile(identity, (2, 2, 1)),
            "action": np.tile(identity, (2, 8, 1)),
            "observation_timestamp": np.array([[0.0, 0.1], [0.1, 0.2]]),
            "action_timestamp": np.array([
                np.arange(0.2, 1.0, 0.1), np.arange(0.3, 1.1, 0.1)
            ]),
            "source_row": np.array([1, 2], dtype=np.int64),
        }
        self.assertEqual(validate_arrays(arrays, obs_horizon=2, action_horizon=8), [])
        arrays["observation_timestamp"] = arrays["observation_timestamp"][:, -1]
        self.assertTrue(any(
            "observation_timestamp.shape" in problem
            for problem in validate_arrays(arrays, obs_horizon=2, action_horizon=8)
        ))

    def test_uniform_time_scale_respects_velocity_and_acceleration(self):
        scale = required_time_scale(
            arm_speed=2.0, arm_accel=12.0,
            gap_speed=0.04, gap_accel=0.075,
            max_arm_speed=1.0, max_arm_accel=3.0,
            max_gap_speed=0.08, max_gap_accel=0.3,
        )
        self.assertAlmostEqual(scale, 2.0)
        self.assertLessEqual(2.0 / scale, 1.0)
        self.assertLessEqual(12.0 / scale ** 2, 3.0)

    def test_uniform_time_scale_never_speeds_up(self):
        scale = required_time_scale(
            arm_speed=0.2, arm_accel=0.4,
            gap_speed=0.01, gap_accel=0.02,
            max_arm_speed=1.0, max_arm_accel=3.0,
            max_gap_speed=0.08, max_gap_accel=0.3,
        )
        self.assertEqual(scale, 1.0)

    def test_uniform_schedule_preserves_waypoints_and_meets_limits(self):
        arm = np.array([
            [0, 0, 0, 0, 0],
            [.2, 0, 0, 0, 0],
            [.2, .3, 0, 0, 0],
        ], dtype=float)
        gap = np.array([.07, .06, .04])
        original_arm = arm.copy()
        original_gap = gap.copy()
        schedule = uniform_time_schedule(
            arm, gap, source_period_s=.1,
            max_arm_speed=1.0, max_arm_accel=3.0,
            max_gap_speed=.08, max_gap_accel=.3,
        )
        self.assertGreater(schedule.time_scale, 1.0)
        self.assertEqual(schedule.waypoint_time_s[0], 0.0)
        self.assertAlmostEqual(
            schedule.waypoint_time_s[1], .1 * schedule.time_scale)
        self.assertLessEqual(schedule.scheduled.arm_speed_rad_s, 1.0 + 1e-9)
        self.assertLessEqual(schedule.scheduled.arm_accel_rad_s2, 3.0 + 1e-9)
        self.assertLessEqual(schedule.scheduled.gap_speed_m_s, .08 + 1e-9)
        self.assertLessEqual(schedule.scheduled.gap_accel_m_s2, .3 + 1e-9)
        np.testing.assert_array_equal(arm, original_arm)
        np.testing.assert_array_equal(gap, original_gap)

    def test_uniform_schedule_rejects_invalid_timestamps_or_shapes(self):
        with self.assertRaisesRegex(ValueError, "source_period_s"):
            uniform_time_schedule(
                np.zeros((2, 5)), np.zeros(2), source_period_s=0,
                max_arm_speed=1, max_arm_accel=1,
                max_gap_speed=1, max_gap_accel=1)
        with self.assertRaisesRegex(ValueError, "trajectory"):
            uniform_time_schedule(
                np.zeros((2, 4)), np.zeros(2), source_period_s=.1,
                max_arm_speed=1, max_arm_accel=1,
                max_gap_speed=1, max_gap_accel=1)


if __name__ == "__main__":
    unittest.main()
