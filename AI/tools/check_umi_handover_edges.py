"""Regression for handover defects, using explicit synthetic trajectory budgets."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tools.run_umi_regression import is_shared_gpu_server
from tools.check_umi_convert import RANGES, GAP_CURVE, RATE, StubIK, make_raw
from umi.convert import convert, ConversionError, ConversionPolicy
from umi.trajectory import TrajectoryLimits
from umi.raw import validate_raw
from umi.ik import IKSolution
from umi.real_fk import ARM_JOINTS, SO101UrdfFK, snapshot_from_joint_state
from umi.arm_preflight import ArmCommand
from umi.action_decode import decode_relative_actions
from umi.ros_adapter import fill_ros_command_messages, arm_trajectory_fields

ROOT = Path(__file__).resolve().parents[1]


class HandoverEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fk = SO101UrdfFK(ROOT / 'configs/real/so101_ver1.urdf')

    def converted(self, raw, solver):
        return convert(raw, solver, RANGES, GAP_CURVE,
                       control_rate_hz=RATE, collected_by='handover-edge-test',
                       policy=ConversionPolicy(trajectory_limits=TrajectoryLimits(
                           (.5,)*5+(.02,), (2.,)*5+(.1,), (20.,)*5+(1.,))))

    def test_nonzero_joint_name_permutation(self):
        q = [.15, -.2, .3, -.4, .5]
        pairs = list(zip((*ARM_JOINTS, 'gripper'), (*q, .04)))
        pairs = [pairs[i] for i in [5, 3, 1, 4, 0, 2]]
        snap = snapshot_from_joint_state(*zip(*pairs), 10, 0, self.fk)
        np.testing.assert_array_equal(snap.arm_rad, q)
        np.testing.assert_allclose(snap.t_base_tcp, self.fk.pose(q))

    def test_relative_translation_uses_rotated_local_axes(self):
        anchor = np.eye(4)
        anchor[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        anchor[:3, 3] = [1, 2, 3]
        actions = np.array([[.1, 0, 0, 1, 0, 0, 0, 1, 0, .04],
                            [.2, 0, 0, 1, 0, 0, 0, 1, 0, .03]])
        decoded = decode_relative_actions(actions, anchor)
        np.testing.assert_allclose(decoded.poses[:, :3, 3], [[1, 2.1, 3], [1, 2.2, 3]])

    def test_released_tcp_near_roll_axis(self):
        # Model sensitivity audit, not a physical calibration or independent FK oracle.
        rng = np.random.default_rng(20260910)
        position_errors, axis_errors = [], []
        for _ in range(20):
            q = rng.uniform(-.4, .4, 5)
            q[4] = 0
            reference = self.fk.pose(q)
            for roll in np.linspace(-1, 1, 9):
                q[4] = roll
                pose = self.fk.pose(q)
                position_errors.append(np.linalg.norm(pose[:3, 3] - reference[:3, 3]))
                cosine = np.dot(pose[:3, 2], reference[:3, 2])
                axis_errors.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        self.assertLess(max(position_errors), 1e-6)  # 1 micrometre
        self.assertLess(max(axis_errors), .001)  # degrees

    def test_missing_calibrations_cannot_default_to_identity(self):
        from track_a.convert.arcore import to_raw
        with self.assertRaises(TypeError):
            to_raw(bundle=Path('synthetic-does-not-exist.zip'), calibration_id='fixture')

    def test_tx_missing_gap_splits_instead_of_bridging(self):
        raw = make_raw(70, n_frames=1, gripper_x_at={35})
        raw.gripper_status[35] = 'T'
        ep, report = self.converted(raw, StubIK(np.zeros((70, 6))))
        self.assertEqual(report.n_steps_out, 34)
        self.assertEqual(report.rejects['gripper_gap_missing'], 1)
        np.testing.assert_allclose(np.diff(ep.state_timestamp), 1 / RATE)

    def test_multichar_status_must_not_be_truncated_to_D(self):
        raw = make_raw(40, n_frames=1)
        raw.gripper_status = np.full(40, 'DEAD', dtype='<U4')
        self.assertTrue(validate_raw(raw), 'DEAD was silently interpreted as D')

    def test_two_radian_branch_jump_not_full_acceptance(self):
        raw = make_raw(70, n_frames=1)
        q = np.zeros((70, 6))
        q[:35, 0], q[35:, 0] = -1, 1
        try:
            _, report = self.converted(raw, StubIK(q))
        except ConversionError:
            return
        self.assertLess(report.n_steps_out, 70,
                        '2 rad in one 30 Hz tick (60 rad/s) accepted as full trajectory')

    def test_nan_solver_position_error_rejected(self):
        class BadIK(StubIK):
            def solve(self, *args, **kwargs):
                sol = super().solve(*args, **kwargs)
                return IKSolution(q_rad=sol.q_rad, pos_error_m=float('nan'),
                    axis_error_deg=.1, roll_residual_deg=.1,
                    within_limits=True, converged=True)
        with self.assertRaises(ConversionError):
            self.converted(make_raw(40, n_frames=1), BadIK(np.zeros((40, 6))))

    def test_unknown_axis_residual_not_full_acceptance(self):
        class BadIK(StubIK):
            def solve(self, *args, **kwargs):
                sol = super().solve(*args, **kwargs)
                return IKSolution(q_rad=sol.q_rad, pos_error_m=.0001,
                    axis_error_deg=float('nan'), roll_residual_deg=.1,
                    within_limits=True, converged=True)
        with self.assertRaises(ConversionError):
            self.converted(make_raw(40, n_frames=1), BadIK(np.zeros((40, 6))))

    def messages(self):
        return (NS(header=NS(stamp=NS(sec=123, nanosec=0)), points=[]),
                NS(positions=[0]*5, velocities=[9]*5, accelerations=[8]*5,
                   effort=[7]*5, time_from_start=NS(sec=0, nanosec=0)), NS(data=[.04]))

    def test_reused_message_clears_derivatives(self):
        trajectory, point, gripper = self.messages()
        fill_ros_command_messages(ArmCommand(np.zeros(5), .04),
            trajectory, point, gripper, duration_s=.03)
        self.assertEqual(point.velocities, [])
        self.assertEqual(point.accelerations, [])
        self.assertEqual(point.effort, [])

    def test_bad_gap_does_not_partially_mutate_arm(self):
        trajectory, point, gripper = self.messages()
        with self.assertRaises(ValueError):
            fill_ros_command_messages(ArmCommand(np.ones(5), .1),
                trajectory, point, gripper, duration_s=.03)
        self.assertEqual(point.positions, [0]*5)
        self.assertEqual(trajectory.points, [])
        self.assertEqual(gripper.data, [.04])

    def test_positive_duration_cannot_round_to_zero(self):
        with self.assertRaises(ValueError):
            arm_trajectory_fields(ArmCommand(np.zeros(5), .04), duration_s=1e-12)


if __name__ == '__main__':
    if is_shared_gpu_server():
        raise SystemExit('Local-only audit: shared GPU server is training-only.')
    unittest.main(verbosity=2)
