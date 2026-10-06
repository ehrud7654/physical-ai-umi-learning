"""Hardware-free decoder regression tests: run from AI with Python + numpy."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from umi.action_decode import decode_relative_actions, rotation_6d_rows_to_matrix


def actions(n=1):
    result = np.zeros((n, 10))
    result[:, 3:9] = [1, 0, 0, 0, 1, 0]
    result[:, 9] = 0.04
    return result


class DecodeTests(unittest.TestCase):
    def test_physical_units_and_no_input_mutation(self):
        a = actions()
        a[0, :3] = [0.01, -0.02, 0.03]
        original = a.copy()
        decoded = decode_relative_actions(a, np.eye(4))
        np.testing.assert_allclose(decoded.poses[0, :3, 3], a[0, :3])
        np.testing.assert_array_equal(decoded.gripper_width_m, [0.04])
        decoded.gripper_width_m[0] = 0
        np.testing.assert_array_equal(a, original)

    def test_row_convention_z90(self):
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(rotation_6d_rows_to_matrix([0, -1, 0, 1, 0, 0]), expected)

    def test_local_translation_and_shared_horizon_anchor(self):
        current = np.eye(4)
        current[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        current[:3, 3] = [1, 2, 3]
        a = actions(2)
        a[:, 0] = [0.01, 0.02]
        result = decode_relative_actions(a, current)
        np.testing.assert_allclose(result.poses[:, :3, 3], [[1, 2.01, 3], [1, 2.02, 3]])

    def test_rotation_composition_order(self):
        current = np.eye(4)
        current[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        a = actions()
        a[0, 3:9] = [1, 0, 0, 0, 0, -1]  # local X +90 degrees
        expected = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        np.testing.assert_allclose(decode_relative_actions(a, current).poses[0, :3, :3], expected)

    def test_orthogonalization(self):
        r = rotation_6d_rows_to_matrix([2, 0, 0, 1, 3, 0])
        np.testing.assert_allclose(r, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_invalid_rotations(self):
        for value in ([0]*6, [1, 0, 0, 2, 0, 0], [float('nan')]*6, [1]*5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rotation_6d_rows_to_matrix(value)

    def test_invalid_actions(self):
        bad = [np.zeros((0, 10)), np.zeros(10), np.zeros((2, 9))]
        for column, value in ((0, np.inf), (9, -0.01)):
            a = actions()
            a[0, column] = value
            bad.append(a)
        for a in bad:
            with self.subTest(shape=a.shape), self.assertRaises(ValueError):
                decode_relative_actions(a, np.eye(4))

    def test_invalid_reference_transform(self):
        bad = [np.eye(3)]
        for index, value in (((3, 0), 1), ((0, 0), -1), ((0, 0), 2), ((0, 3), np.nan)):
            t = np.eye(4)
            t[index] = value
            bad.append(t)
        for t in bad:
            with self.subTest(t=t), self.assertRaises(ValueError):
                decode_relative_actions(actions(), t)


if __name__ == '__main__':
    unittest.main()
