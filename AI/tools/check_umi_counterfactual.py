"""Unit checks that displaced-object targets do not accumulate per cycle."""
from __future__ import annotations

import unittest

import numpy as np

from tools.eval_counterfactual_oracle_suite import _record
from umi.counterfactual import (
    dense_executed_future_rows,
    reference_anchors,
    shifted_chunk_from_reference,
)
from umi.relative_dataset import transform_to_vector


class CounterfactualTests(unittest.TestCase):
    @staticmethod
    def _chunks() -> np.ndarray:
        action = np.zeros((3, 8, 10), dtype=np.float32)
        for row in range(3):
            for future in range(8):
                target = np.eye(4)
                target[0, 3] = 0.01 * (future + 1)
                action[row, future] = transform_to_vector(target, 0.04)
        return action

    def test_reference_anchors_follow_only_first_future_step(self) -> None:
        anchors = reference_anchors(self._chunks(), start_row=0,
                                    start_pose=np.eye(4))
        np.testing.assert_allclose(anchors[:, 0, 3], [0, 0.01, 0.02])
        np.testing.assert_allclose(
            anchors[:, 1:, 3], np.tile([0.0, 0.0, 1.0], (3, 1)))

    def test_object_shift_is_applied_once_across_cycles(self) -> None:
        action = self._chunks()
        anchors = reference_anchors(action, start_row=0,
                                    start_pose=np.eye(4))
        shift = np.array([0.01, 0.0, 0.0])
        first = shifted_chunk_from_reference(
            action[0], reference_pose=anchors[0], actual_pose=anchors[0],
            delta_world_xyz=shift)
        self.assertAlmostEqual(float(first[0, 0]), 0.02)

        # At the next control cycle the robot already carries the 10mm shift.
        actual = anchors[1].copy()
        actual[:3, 3] += shift
        second = shifted_chunk_from_reference(
            action[1], reference_pose=anchors[1], actual_pose=actual,
            delta_world_xyz=shift)
        self.assertAlmostEqual(float(second[0, 0]), 0.01)
        np.testing.assert_allclose(second[:, 3:], action[1, :, 3:], atol=1e-6)

        # A 1mm lag is corrected; the full 10mm is not added again.
        actual[0, 3] -= 0.001
        lagged = shifted_chunk_from_reference(
            action[1], reference_pose=anchors[1], actual_pose=actual,
            delta_world_xyz=shift)
        self.assertAlmostEqual(float(lagged[0, 0]), 0.011)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            reference_anchors(self._chunks(), start_row=3,
                              start_pose=np.eye(4))
        with self.assertRaises(ValueError):
            shifted_chunk_from_reference(
                self._chunks()[0], reference_pose=np.eye(4),
                actual_pose=np.eye(4), delta_world_xyz=np.array([np.nan, 0, 0]))

    def test_physical_lift_does_not_hide_later_ik_failure(self) -> None:
        report = {
            "representative_episode": "fixture",
            "diagnostic_object_offset_xy_m": [-0.01, 0.0],
            "report_path": "fixture.json",
            "executed_commands": 2,
            "geometry_error": "step 5: joint limit",
            "bilateral_contact_ticks": 24,
            "bilateral_jaw_contact_final": True,
            "lift_height_m": 0.06,
            "max_pre_lift_object_xy_displacement_m": 0.003,
            "max_object_tilt_deg": 4.0,
            "stable_side_grasp_success": True,
        }
        result = _record(report)
        self.assertTrue(result["stable_side_grasp_success"])
        self.assertFalse(result["full_trajectory_geometry_valid"])

    def test_dense_future_rows_use_only_measured_uniform_samples(self) -> None:
        n = 12
        images = np.zeros((n, 3, 224, 224), dtype=np.uint8)
        images[:, 0, 0, 0] = np.arange(n)
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, 0, 3] = np.arange(n) * 0.001
        gaps = np.linspace(0.04, 0.05, n)
        times = np.arange(n) * 0.1
        source_rows = np.arange(n) // 4
        arrays = dense_executed_future_rows(
            images, poses, gaps, times, source_rows)
        self.assertEqual(arrays["image"].shape, (3, 2, 3, 224, 224))
        self.assertEqual(arrays["action"].shape, (3, 8, 10))
        np.testing.assert_allclose(arrays["action"][:, 0, 0], 0.001)
        np.testing.assert_allclose(
            arrays["action_timestamp"][:, 0]
            - arrays["observation_timestamp"][:, -1], 0.1)
        self.assertEqual(int(arrays["image"][0, 0, 0, 0, 0]), 0)
        self.assertEqual(int(arrays["image"][0, 1, 0, 0, 0]), 1)
        with self.assertRaisesRegex(ValueError, "uniformly timed"):
            bad_times = times.copy()
            bad_times[5] += 0.01
            dense_executed_future_rows(
                images, poses, gaps, bad_times, source_rows)


if __name__ == "__main__":
    unittest.main()
