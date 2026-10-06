#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.arm_preflight import ArmCommand
from umi.real_fk import ARM_JOINTS, SO101UrdfFK
from umi.ros_adapter import (ARM_COMMAND_TOPIC, GRIPPER_COMMAND_TOPIC,
    JOINT_STATE_TOPIC, arm_trajectory_fields, fill_ros_command_messages,
    gripper_command_data, joint_state_to_runtime_snapshot)


class RosAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fk = SO101UrdfFK(ROOT / "configs/real/so101_ver1.urdf")

    def message(self, *, sec=10, nanosec=0, gap=0.045):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
            name=("gripper", *reversed(ARM_JOINTS)),
            position=(gap, *([0.0] * 5)))

    def test_joint_state_clock_translation_and_fk(self):
        snap = joint_state_to_runtime_snapshot(
            self.message(sec=10), self.fk, received_ros_time_s=10.04,
            received_monotonic_s=50.0, max_transport_age_s=0.1)
        self.assertAlmostEqual(snap.timestamp, 49.96)
        np.testing.assert_array_equal(snap.payload.arm_rad, np.zeros(5))
        self.assertEqual(snap.payload.gripper_width_m, 0.045)
        self.assertEqual(snap.payload.t_base_tcp.shape, (4, 4))

    def test_future_stale_and_bad_gap_are_rejected(self):
        common = dict(fk=self.fk, received_monotonic_s=50.0, max_transport_age_s=0.1)
        with self.assertRaises(ValueError):
            joint_state_to_runtime_snapshot(self.message(sec=11),
                received_ros_time_s=10.0, **common)
        with self.assertRaises(ValueError):
            joint_state_to_runtime_snapshot(self.message(sec=9),
                received_ros_time_s=10.0, **common)
        with self.assertRaises(ValueError):
            joint_state_to_runtime_snapshot(self.message(gap=0.091),
                received_ros_time_s=10.0, **common)

    def test_command_fields_use_five_radians_and_full_gap(self):
        command = ArmCommand(np.array([0.1, 0.2, -0.3, 0.4, 0.5]), 0.09)
        fields = arm_trajectory_fields(command, duration_s=1 / 30)
        self.assertEqual(fields.joint_names, ARM_JOINTS)
        self.assertEqual(fields.positions, (0.1, 0.2, -0.3, 0.4, 0.5))
        self.assertEqual((fields.time_from_start_sec, fields.time_from_start_nanosec),
                         (0, 33_333_333))
        self.assertEqual(gripper_command_data(command), (0.09,))

    def test_command_rejects_config_gap_contract_mismatch(self):
        command = ArmCommand(np.zeros(5), 0.08)
        with self.assertRaisesRegex(ValueError, "does not match the episode contract"):
            gripper_command_data(command, max_gap_m=0.08)

    def test_real_message_instances_are_populated(self):
        command = ArmCommand(np.zeros(5), 0.045)
        trajectory = SimpleNamespace(joint_names=None, points=None)
        point = SimpleNamespace(positions=None,
            time_from_start=SimpleNamespace(sec=None, nanosec=None))
        gripper = SimpleNamespace(data=None)
        fill_ros_command_messages(command, trajectory, point, gripper, duration_s=0.05)
        self.assertEqual(trajectory.joint_names, list(ARM_JOINTS))
        self.assertEqual(len(trajectory.points), 1)
        self.assertEqual(point.time_from_start.nanosec, 50_000_000)
        self.assertEqual(gripper.data, [0.045])

    def test_topics_match_hardware_contract(self):
        config = json.loads((ROOT / "configs/real/so101_ver1.json").read_text(encoding="utf-8"))
        interfaces = config["ros_interfaces"]
        self.assertEqual(JOINT_STATE_TOPIC, "/joint_states")
        self.assertEqual(ARM_COMMAND_TOPIC, "/arm_controller/joint_trajectory")
        self.assertEqual(GRIPPER_COMMAND_TOPIC, "/gripper_controller/commands")
        self.assertEqual(interfaces["joint_state_topic"], JOINT_STATE_TOPIC)
        self.assertEqual(interfaces["arm_stream_topic"], ARM_COMMAND_TOPIC)
        self.assertEqual(interfaces["gripper_topic"], GRIPPER_COMMAND_TOPIC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
