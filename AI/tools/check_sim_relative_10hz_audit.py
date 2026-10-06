"""Small deterministic tests for the policy-free 10Hz candidate audit."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from tools.audit_sim_relative_10hz_candidates import pair_metrics, shortcut_metrics
from umi.relative_dataset import transform_to_vector


class CandidateAuditTests(unittest.TestCase):
    def test_constant_velocity_exact_linear_motion(self) -> None:
        previous = np.eye(4)
        previous[0, 3] = -0.01
        current = np.eye(4)
        history = np.stack([
            transform_to_vector(previous, 0.04),
            transform_to_vector(current, 0.04),
        ]).astype(np.float32)[None]
        targets = []
        for step in range(1, 9):
            future = np.eye(4)
            future[0, 3] = step * 0.01
            targets.append(transform_to_vector(future, 0.04))
        action = np.stack(targets).astype(np.float32)[None]
        metrics = shortcut_metrics(history, action)
        self.assertGreater(metrics["hold_translation_l2_mean_m"], 0.04)
        self.assertLess(metrics["constant_velocity_translation_l2_mean_m"], 1e-7)
        self.assertLess(metrics["constant_velocity_rotation6d_mae"], 1e-7)
        self.assertLess(metrics["constant_velocity_gap_mae_m"], 1e-7)

    def test_fixed_start_pair_reports_object_and_target_change(self) -> None:
        def candidate(x0: int, offset: float) -> dict:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            rgb[50:180, x0:x0 + 50] = (255, 180, 0)
            image = np.transpose(rgb, (2, 0, 1))
            action = np.zeros((1, 8, 10), dtype=np.float32)
            action[0, 0, 2] = offset
            return {
                "path": Path(f"candidate_{x0}.npz"),
                "meta": {
                    "source_episode": "episode_a",
                    "robot_start_is_fixed": True,
                    "object_offset_world_xy_m": [offset, 0.0],
                },
                "arrays": {
                    "image": image[None, None],
                    "action": action,
                    "observation_timestamp": np.asarray([[0.4, 0.5]]),
                    "source_row": np.asarray([7]),
                },
            }
        result = pair_metrics(candidate(70, -0.01), candidate(80, 0.01))
        self.assertEqual(result["object_offset_difference_world_xy_m"], [0.02, 0.0])
        self.assertEqual(result["object_center_shift_px"], [10.0, 0.0])
        self.assertGreater(result["initial_rgb_abs_difference_mean"], 0)
        self.assertAlmostEqual(
            result["first_future_tcp_translation_difference_m"][2], 0.02)


if __name__ == "__main__":
    unittest.main()
