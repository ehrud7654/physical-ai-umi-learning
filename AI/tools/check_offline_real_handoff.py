"""ROS-free tests of the timed JointState -> relative EEF -> joint bridge."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.arm_preflight import ArmSolution
from umi.chunk_runtime import Snapshot
from umi.offline_real_handoff import (
    build_v10_real_observation, prepare_offline_real_handoff,
)
from umi.real_fk import ARM_JOINTS, SO101UrdfFK
from umi.ros_adapter import joint_state_to_runtime_snapshot


AI_ROOT = Path(__file__).resolve().parents[1]


class FixedPoseSolver:
    """Fixture only: accept the one exact FK pose, never stand in for real IK."""

    def __init__(self, pose: np.ndarray, arm: np.ndarray | None = None) -> None:
        self.pose = pose.copy()
        self.arm = np.zeros(5) if arm is None else np.asarray(arm, dtype=float)

    def solve(self, target_pose, *, seed_rad, gripper_width_m):
        del seed_rad, gripper_width_m
        return ArmSolution(
            positions_rad=self.arm.copy(), position_error_m=0.0,
            axis_error_deg=0.0, roll_error_deg=0.0,
            converged=bool(np.allclose(target_pose, self.pose, rtol=0, atol=1e-9)),
        )


class OfflineRealHandoffTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((AI_ROOT / "configs/real/so101_ver1.json")
                                 .read_text(encoding="utf-8"))
        self.fk = SO101UrdfFK(AI_ROOT / "configs/real/so101_ver1.urdf")
        message = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=0)),
            name=list(ARM_JOINTS) + ["gripper"],
            position=[0.0] * 5 + [0.045],
        )
        self.snapshot = joint_state_to_runtime_snapshot(
            message, self.fk, received_ros_time_s=100.005,
            received_monotonic_s=10.005, max_transport_age_s=0.02,
        )
        self.action = np.tile(
            np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.045], dtype=float),
            (8, 1),
        )
        self.arguments = dict(
            camera_timestamp_s=[9.9, 10.0],
            prediction_observation_timestamp_s=10.0,
            state_snapshot=self.snapshot, now_monotonic_s=10.005,
            fk=self.fk, solver=FixedPoseSolver(self.fk.pose(np.zeros(5))),
            real_config=self.config, execute_steps=4,
            max_age_s=0.5, max_sync_error_s=0.01,
            max_history_period_error_s=0.01,
            max_time_scale=2.0, max_position_error_m=0.005,
            max_axis_error_deg=5.0, max_roll_error_deg=5.0,
        )

    def test_exact_first_four_future_targets_and_no_ros_output(self):
        result = prepare_offline_real_handoff(self.action, **self.arguments)
        self.assertEqual(result.status, "OFFLINE_DIAGNOSTIC_ONLY")
        self.assertEqual(result.chunk.arm_positions_rad.shape, (8, 5))
        self.assertEqual(result.chunk.gripper_width_m.shape, (8,))
        self.assertEqual(result.execution_prefix.future_target_indices,
                         (0, 1, 2, 3))
        np.testing.assert_allclose(result.execution_prefix.time_from_start_s,
                                   [0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(result.camera_state_skew_s, 0.0, places=9)
        self.assertFalse(hasattr(result, "ros_goal"))

    def test_two_frame_rgb_and_proprio_feed_the_offline_boundary(self):
        previous = Snapshot(9.9, copy.deepcopy(self.snapshot.payload))
        images = np.zeros((2, 3, 224, 224), dtype=np.uint8)
        observation = build_v10_real_observation(
            images, camera_timestamp_s=[9.9, 10.0],
            state_history=(previous, self.snapshot),
            now_monotonic_s=10.005, fk=self.fk,
            calibration_id="offline-test-only", max_age_s=0.5,
            max_sync_error_s=0.01, max_history_period_error_s=0.01,
        )
        self.assertEqual(observation.image.shape, (2, 3, 224, 224))
        self.assertEqual(observation.proprio.shape, (2, 10))
        np.testing.assert_allclose(observation.proprio[:, 9], [0.045, 0.045])
        np.testing.assert_allclose(observation.proprio[-1, :9],
                                   [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
        self.assertFalse(observation.image.flags.writeable)
        self.assertFalse(observation.current_state_snapshot.payload.arm_rad.flags.writeable)
        self.assertFalse(observation.current_state_snapshot.payload.t_base_tcp.flags.writeable)
        self.snapshot.payload.arm_rad[0] = 0.1
        np.testing.assert_allclose(
            observation.current_state_snapshot.payload.arm_rad, np.zeros(5))
        self.snapshot.payload.arm_rad[0] = 0.0
        result = prepare_offline_real_handoff(
            self.action,
            **{**self.arguments,
               "camera_timestamp_s": observation.camera_timestamp_s,
               "state_snapshot": observation.current_state_snapshot},
        )
        self.assertEqual(result.status, "OFFLINE_DIAGNOSTIC_ONLY")
        with self.assertRaisesRegex(ValueError, "RGB and joint-state"):
            build_v10_real_observation(
                images, camera_timestamp_s=[9.88, 10.0],
                state_history=(previous, self.snapshot),
                now_monotonic_s=10.005, fk=self.fk,
                calibration_id="offline-test-only", max_age_s=0.5,
                max_sync_error_s=0.01, max_history_period_error_s=0.03,
            )

    def test_stale_skewed_or_mismatched_prediction_rejected(self):
        for change in (
            {"now_monotonic_s": 10.6},
            {"camera_timestamp_s": [9.9, 10.02]},
            {"camera_timestamp_s": [10.0, 9.9]},
            {"camera_timestamp_s": [9.98, 10.0]},
            {"prediction_observation_timestamp_s": 9.9},
        ):
            with self.subTest(change=change), self.assertRaisesRegex(
                ValueError, "observation timestamps"
            ):
                prepare_offline_real_handoff(
                    self.action, **{**self.arguments, **change})

    def test_fk_mismatch_and_installed_wrist_cap_rejected(self):
        wrong = copy.deepcopy(self.snapshot)
        wrong.payload.t_base_tcp[0, 3] += 0.01
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_offline_real_handoff(
                self.action, **{**self.arguments, "state_snapshot": wrong})
        solver = FixedPoseSolver(
            self.fk.pose(np.zeros(5)),
            arm=[0, 0, 0, 0, np.deg2rad(61)],
        )
        with self.assertRaisesRegex(ValueError, "arm range"):
            prepare_offline_real_handoff(
                self.action, **{**self.arguments, "solver": solver})

    def test_invalid_prediction_or_excessive_time_scale_rejected(self):
        with self.assertRaisesRegex(ValueError, "eight future"):
            prepare_offline_real_handoff(self.action[:7], **self.arguments)
        invalid = self.action.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            prepare_offline_real_handoff(invalid, **self.arguments)
        too_fast = {**self.arguments, "max_time_scale": 0.5}
        with self.assertRaisesRegex(ValueError, "invalid current state"):
            prepare_offline_real_handoff(self.action, **too_fast)

    def test_solver_cannot_claim_zero_error_for_wrong_urdf_fk(self):
        liar = FixedPoseSolver(
            self.fk.pose(np.zeros(5)), arm=[0.2, 0, 0, 0, 0],
        )
        with self.assertRaisesRegex(ValueError, "URDF FK contradicts"):
            prepare_offline_real_handoff(
                self.action, **{**self.arguments, "solver": liar})

    def test_out_of_contract_observation_gap_rejected(self):
        invalid = copy.deepcopy(self.snapshot)
        object.__setattr__(invalid.payload, 'gripper_width_m', -0.01)
        images = np.zeros((2, 3, 224, 224), dtype=np.uint8)
        previous = Snapshot(9.9, copy.deepcopy(self.snapshot.payload))
        with self.assertRaisesRegex(ValueError, "invalid joint/TCP"):
            build_v10_real_observation(
                images, camera_timestamp_s=[9.9, 10.0],
                state_history=(previous, invalid),
                now_monotonic_s=10.005, fk=self.fk,
                calibration_id="offline-test-only", max_age_s=0.5,
                max_sync_error_s=0.01, max_history_period_error_s=0.01,
            )


if __name__ == "__main__":
    unittest.main()
