"""Relative-motion audit must be invariant to the unknown world/base transform."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from umi.camera_frames import s22_arcore_to_pinch


class Tests(unittest.TestCase):
    def test_relative_pinch_motion_is_world_alignment_invariant(self):
        first = np.eye(4)
        second = np.eye(4)
        angle = np.deg2rad(12)
        second[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]]
        second[:3, 3] = [.02, -.01, .03]
        align = np.eye(4)
        align[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        align[:3, 3] = [1, 2, 3]
        pinch = s22_arcore_to_pinch()
        relative_a = np.linalg.inv(first @ pinch) @ (second @ pinch)
        relative_b = np.linalg.inv(align @ first @ pinch) @ (align @ second @ pinch)
        np.testing.assert_allclose(relative_a, relative_b, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
