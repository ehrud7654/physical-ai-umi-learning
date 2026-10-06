"""Regression checks for the confirmed SO-101 URDF and JointState contract."""
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from umi.real_fk import (ARM_JOINTS, SO101UrdfFK, canonical_file_sha256,
                         snapshot_from_joint_state)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real" / "so101_ver1.json"


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.urdf = ROOT / cls.config["urdf"]
        cls.fk = SO101UrdfFK(cls.urdf, cls.config["base_frame"], cls.config["tcp_frame"])

    def test_release_hash_and_limits_match_config(self):
        digest = canonical_file_sha256(self.urdf)
        self.assertEqual(digest, self.config["urdf_sha256"])
        expected = np.array([self.config["limits_rad"][name] for name in ARM_JOINTS])
        np.testing.assert_allclose(self.fk.limits_rad, expected, atol=1e-11)
        self.assertEqual(tuple(self.config["joint_order"]), ARM_JOINTS)

    def test_tcp_gripper_and_motor_calibration_contract(self):
        root = ET.parse(self.urdf).getroot()
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        tcp = joints["gripper_tcp_fixed"]
        expected_tcp = self.config["tcp_fixed_from_custom_hand"]
        np.testing.assert_allclose(np.fromstring(tcp.find("origin").attrib["xyz"], sep=" "),
                                   expected_tcp["xyz_m"], atol=1e-12)
        np.testing.assert_allclose(np.fromstring(tcp.find("origin").attrib["rpy"], sep=" "),
                                   expected_tcp["rpy_rad"], atol=1e-12)
        gripper = joints["gripper"]
        self.assertEqual(gripper.find("parent").attrib["link"], "custom_hand_link")
        self.assertEqual(float(gripper.find("limit").attrib["lower"]), 0.0)
        self.assertEqual(float(gripper.find("limit").attrib["upper"]), 0.09)
        for name in ("gripper_right", "gripper_mirror"):
            mimic = joints[name].find("mimic")
            self.assertEqual(mimic.attrib["joint"], "gripper")
            self.assertEqual(float(mimic.attrib["multiplier"]), 0.5)

        control = root.find("ros2_control")
        hardware = {joint.attrib["name"]: joint for joint in control.findall("joint")}
        for name in ARM_JOINTS:
            params = {param.attrib["name"]: param.text for param in hardware[name].findall("param")}
            self.assertEqual(int(params["direction"]), self.config["direction"][name])
            self.assertEqual(int(params["zero_tick"]), self.config["zero_tick"][name])

    def test_zero_pose_fixture(self):
        pose = self.fk.pose(np.zeros(5))
        self.assertTrue(np.isfinite(pose).all())
        np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-10)

    def test_joint_state_uses_full_gap_and_ros_time(self):
        names = ["unused", *ARM_JOINTS, "gripper"]
        positions = [12.0, 0, 0, 0, 0, 0, 0.09]
        snap = snapshot_from_joint_state(names, positions, 10, 250_000_000, self.fk)
        self.assertEqual(snap.arm_rad.shape, (5,))
        self.assertEqual(snap.gripper_width_m, 0.09)
        self.assertEqual(snap.ros_time_s, 10.25)
        np.testing.assert_allclose(snap.t_base_tcp, self.fk.pose(np.zeros(5)))

    def test_joint_state_rejects_config_gap_contract_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match the episode contract"):
            snapshot_from_joint_state(
                (*ARM_JOINTS, "gripper"), [0] * 5 + [0.08], 0, 0,
                self.fk, max_gap_m=0.08)

    def test_invalid_or_incomplete_state_rejected(self):
        with self.assertRaises(ValueError):
            snapshot_from_joint_state(ARM_JOINTS, np.zeros(5), 0, 0, self.fk)
        with self.assertRaises(ValueError):
            snapshot_from_joint_state((*ARM_JOINTS, "gripper"), [0]*5+[0.091], 0, 0, self.fk)
        q = np.zeros(5)
        q[2] = -1.58
        with self.assertRaises(ValueError):
            self.fk.pose(q)


if __name__ == "__main__":
    unittest.main()
