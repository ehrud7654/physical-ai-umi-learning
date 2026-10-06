"""Deterministic unit checks for paired visual-position probe calculations."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.probe_sim_relative_visual_position import common_source_row, translation_response


class VisualPositionProbeTests(unittest.TestCase):
    def test_earliest_shared_source_row_not_repeated_robot_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"candidate_{index}.npz" for index in range(3)]
            for path, rows in zip(paths, ([4, 4, 5], [3, 4, 5], [4, 5, 5])):
                np.savez_compressed(path, source_row=np.asarray(rows))
            self.assertEqual(common_source_row(paths), (4, [0, 1, 0]))

    def test_alignment_gain_and_error_for_exact_target_change(self) -> None:
        target = np.zeros((8, 10), dtype=np.float32)
        target[:, 0] = 0.01
        same = translation_response(target, target)
        self.assertAlmostEqual(same["alignment_gain"], 1.0)
        self.assertAlmostEqual(same["cosine"], 1.0)
        self.assertAlmostEqual(same["delta_error_l2_mean_m"], 0.0)
        wrong = translation_response(-target, target)
        self.assertAlmostEqual(wrong["alignment_gain"], -1.0)
        self.assertAlmostEqual(wrong["cosine"], -1.0)


if __name__ == "__main__":
    unittest.main()
