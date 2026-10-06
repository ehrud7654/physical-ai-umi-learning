"""Failure/seed semantics tests independent of MuJoCo."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from umi.action_ik import actions_to_joints
from umi.ik import IKSolution


class Stub:
    def __init__(self, jump=0.01, fail=False):
        self.jump, self.fail, self.seeds = jump, fail, []

    def solve(self, pos, quat, gap, q_init=None):
        self.seeds.append(q_init.copy())
        q = q_init.copy()
        q[0] += self.jump
        q[5] = 0  # Deliberately wrong: the adapter must set gripper separately.
        return IKSolution(q, 0, 0, 0, True, not self.fail)


class Tests(unittest.TestCase):
    def run_adapter(self, solver, gap=0.05):
        a = np.zeros((3, 10))
        a[:, 3:9] = [1, 0, 0, 0, 1, 0]
        a[:, 9] = gap
        q = np.zeros(6)
        q[5] = 0.5
        return actions_to_joints(a, np.eye(4), q, solver,
                                 np.tile([-1, 1], (6, 1)), [[0, 0], [1, 10]],
                                 max_step_rad=0.1)

    def test_seed_and_gripper(self):
        stub = Stub()
        result = self.run_adapter(stub)
        np.testing.assert_allclose(result[:, 0], [0.01, 0.02, 0.03])
        np.testing.assert_allclose(result[:, 5], 0.5)
        np.testing.assert_allclose(stub.seeds[1][:5], result[0, :5])

    def test_reject_jump(self):
        with self.assertRaisesRegex(ValueError, 'jump'):
            self.run_adapter(Stub(jump=0.2))

    def test_reject_nonconvergence(self):
        stub = Stub(fail=True)
        with self.assertRaisesRegex(ValueError, 'IK rejected'):
            self.run_adapter(stub)
        self.assertEqual(len(stub.seeds), 1)

    def test_reject_width_outside_calibration(self):
        with self.assertRaises(ValueError):
            self.run_adapter(Stub(), gap=0.2)

    def test_reject_joint_limit(self):
        with self.assertRaisesRegex(ValueError, 'joint limit'):
            self.run_adapter(Stub(jump=2))


if __name__ == '__main__':
    unittest.main()
