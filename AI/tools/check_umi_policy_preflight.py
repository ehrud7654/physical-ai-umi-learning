"""Checkpoint-interface tests that need neither torch nor robot hardware."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from umi.ik import IKSolution
from umi.policy_preflight import extract_action_pred, run_policy_preflight


def valid_action(t=2):
    action = np.zeros((t, 10))
    action[:, 3:9] = [1, 0, 0, 0, 1, 0]
    action[:, 9] = 0.05
    return action


class Policy:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def predict_action(self, observation):
        self.calls += 1
        if observation != {"camera": "prepared-by-owner"}:
            raise AssertionError("observation was changed")
        return self.output


class Solver:
    def solve(self, pos, quat, gap, q_init=None):
        return IKSolution(np.array(q_init, copy=True), 0, 0, 0, True, True)


class TensorLike:
    def __init__(self, value):
        self.value = value

    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class Tests(unittest.TestCase):
    def test_extract_numpy_mapping_and_single_batch(self):
        expected = valid_action()
        for output in (expected, {"action_pred": expected},
                       TensorLike(expected[None])):
            with self.subTest(type=type(output).__name__):
                got = extract_action_pred(output)
                np.testing.assert_array_equal(got, expected)
                self.assertIsNot(got, expected)

    def test_reject_ambiguous_or_invalid_output(self):
        invalid = [{"action": valid_action()}, np.zeros((0, 10)),
                   np.zeros((2, 9)), np.zeros((2, 2, 10))]
        bad_nan = valid_action()
        bad_nan[0, 0] = np.nan
        invalid.append(bad_nan)
        for output in invalid:
            with self.subTest(output=type(output).__name__), self.assertRaises(ValueError):
                extract_action_pred(output)

    def test_preflight_calls_policy_once_and_returns_t6(self):
        policy = Policy({"action_pred": valid_action()})
        ticks = iter([1.0, 1.012, 1.015])
        result = run_policy_preflight(
            policy, {"camera": "prepared-by-owner"}, t_current=np.eye(4),
            q_current=np.zeros(6), solver=Solver(),
            ranges_rad=np.tile([-1, 1], (6, 1)),
            gap_curve=[[0, 0], [1, 10]], max_step_rad=0.6,
            clock=lambda: next(ticks),
        )
        self.assertEqual(policy.calls, 1)
        self.assertEqual(result.action_pred.shape, (2, 10))
        self.assertEqual(result.joint_targets_rad.shape, (2, 6))
        self.assertAlmostEqual(result.predict_ms, 12)
        self.assertAlmostEqual(result.convert_ms, 3)

    def test_requires_predict_action(self):
        with self.assertRaises(TypeError):
            run_policy_preflight(object(), None, t_current=np.eye(4),
                q_current=np.zeros(6), solver=Solver(),
                ranges_rad=np.tile([-1, 1], (6, 1)),
                gap_curve=[[0, 0], [1, 10]], max_step_rad=0.1)


if __name__ == "__main__":
    unittest.main()
