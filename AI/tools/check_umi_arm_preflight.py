"""Offline command boundary regression tests. No hardware access."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from umi.arm_preflight import ArmSolution, prepare_arm_commands


class Policy:
    def __init__(self):
        self.actions = np.tile([0.01, 0, 0, 1, 0, 0, 0, 1, 0, 0.045], (2, 1))
        self.callback = lambda: None

    def predict_action(self, obs):
        self.callback()
        obs['image'][0] = 99
        return {'action_pred': self.actions}


class Solver:
    def __init__(self):
        self.poses, self.seeds = [], []
        self.error = 0
        self.within_limits = True

    def solve(self, target_pose, *, seed_rad, gripper_width_m):
        self.poses.append(target_pose.copy())
        self.seeds.append(seed_rad.copy())
        return ArmSolution(
            seed_rad + 0.01, self.error, 0, 0, True, self.within_limits)


class FakeDriver:
    def __init__(self): self.received = []
    def command(self, arm_positions_rad, gripper_width_m):
        self.received.append((arm_positions_rad.copy(), gripper_width_m))


class Tests(unittest.TestCase):
    def setUp(self):
        self.policy, self.solver = Policy(), Solver()
        self.obs = {'image': np.zeros(3)}
        self.args = dict(t_current=np.eye(4), arm_current_rad=np.zeros(5),
            gripper_current_m=0.045, solver=self.solver,
            ranges_rad=np.tile([-1, 1], (5, 1)), gap_range_m=[0, 0.09],
            max_step_rad=0.1, max_gap_step_m=0.01,
            max_position_error_m=0.005, max_axis_error_deg=5, max_roll_error_deg=5)

    def run_chunk(self):
        return prepare_arm_commands(self.policy, self.obs, **self.args)

    def test_snapshot_and_shared_anchor(self):
        def mutate():
            self.args['t_current'][0, 3] = 10
            self.args['arm_current_rad'][:] = 0.8
        self.policy.callback = mutate
        commands = self.run_chunk()
        np.testing.assert_array_equal(self.obs['image'], np.zeros(3))
        for pose in self.solver.poses:
            self.assertAlmostEqual(pose[0, 3], 0.01)
        np.testing.assert_allclose(commands[0].arm_positions_rad, 0.01)
        np.testing.assert_allclose(self.solver.seeds[1], 0.01)

    def test_fake_driver_receives_five_radians_and_original_metres(self):
        driver = FakeDriver()
        for cmd in self.run_chunk():
            driver.command(cmd.arm_positions_rad, cmd.gripper_width_m)
        self.assertEqual(len(driver.received), 2)
        self.assertEqual(driver.received[0][0].shape, (5,))
        self.assertEqual(driver.received[0][1], 0.045)

    def test_invalid_later_target_returns_no_chunk(self):
        driver = FakeDriver()
        self.policy.actions[1, 9] = 0.2
        with self.assertRaisesRegex(ValueError, 'gripper'):
            for cmd in self.run_chunk():
                driver.command(cmd.arm_positions_rad, cmd.gripper_width_m)
        self.assertEqual(driver.received, [])

    def test_gap_jump_rejected(self):
        self.policy.actions[0, 9] = 0.07
        with self.assertRaisesRegex(ValueError, 'gripper'): self.run_chunk()

    def test_position_residual_rejected(self):
        self.solver.error = 0.006
        with self.assertRaisesRegex(ValueError, 'IK'): self.run_chunk()

    def test_arm_jump_rejected(self):
        self.args['max_step_rad'] = 0.005
        with self.assertRaisesRegex(ValueError, 'arm range'): self.run_chunk()

    def test_solver_limit_hit_is_reported_separately(self):
        self.solver.within_limits = False
        with self.assertRaisesRegex(ValueError, 'within_limits=False'):
            self.run_chunk()

    def test_invalid_initial_state_rejected_before_inference(self):
        self.args['arm_current_rad'][0] = np.nan
        self.policy.callback = lambda: self.fail('policy must not run')
        with self.assertRaises(ValueError): self.run_chunk()


if __name__ == '__main__': unittest.main()
