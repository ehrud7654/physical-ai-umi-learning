"""Small deterministic checks for the visual-alignment geometry helpers."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.fit_s22_mujoco_visual_alignment import (
    _bbox_norm,
    _camera_rotation,
    _project_bbox,
    _rotation_xyz,
)


class VisualAlignmentGeometryTest(unittest.TestCase):
    def test_camera_rotation_is_right_handed(self):
        rotation = _camera_rotation([1, 0, 0, 0, 1, 0])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_projection_is_centred_and_scales_with_fov(self):
        corners = np.array([
            [x, y, z]
            for x in (-0.2, 0.2)
            for y in (-0.4, 0.4)
            for z in (-2.1, -1.9)
        ])
        narrow = _project_bbox(corners, np.zeros(3), np.eye(3), np.zeros(3),
                               np.eye(3), 40.0)
        wide = _project_bbox(corners, np.zeros(3), np.eye(3), np.zeros(3),
                             np.eye(3), 80.0)
        assert narrow is not None and wide is not None
        np.testing.assert_allclose(narrow[:2], [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(wide[:2], [0.5, 0.5], atol=1e-12)
        self.assertGreater(narrow[2], wide[2])
        self.assertGreater(narrow[3], wide[3])

    def test_bbox_normalisation(self):
        actual = _bbox_norm((25, 10, 74, 89), 100, 100)
        np.testing.assert_allclose(actual, [0.5, 0.5, 0.5, 0.8])

    def test_xyz_rotation_stays_orthonormal(self):
        rotation = _rotation_xyz(np.deg2rad([11.0, -23.0, 47.0]))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)


if __name__ == "__main__":
    unittest.main()
