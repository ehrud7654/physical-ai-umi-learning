"""Checks that ARCore/OpenGL and OpenCV chains describe the same pinch pose."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from umi.camera_frames import (
    GRAVITY_PINCH_TO_SIDE_GRASP_ROTATION,
    SUPPLIED_HAND_TO_CANONICAL_PINCH_ROTATION,
    camera_gravity_pinch_to_side_grasp,
    camera_supplied_hand_to_canonical_pinch,
    camera_opencv_to_arcore_child,
    gripper_camera_from_pinch,
    load_arcore_pinch_calibration,
    s22_arcore_to_pinch,
    s22_opencv_to_pinch,
    world_arcore_to_world_opencv,
)
from sim.mujoco.build_scene import load_config


class Tests(unittest.TestCase):
    def test_method_a_and_b_are_identical(self):
        angle = np.deg2rad(31.0)
        world_arcore = np.eye(4)
        world_arcore[:3, :3] = [
            [np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
        world_arcore[:3, 3] = [0.2, 0.4, -0.1]
        method_a = world_arcore_to_world_opencv(world_arcore) @ s22_opencv_to_pinch()
        method_b = world_arcore @ s22_arcore_to_pinch()
        np.testing.assert_allclose(method_a, method_b, atol=1e-12)

    def test_exact_rotation_is_rigid_and_expected_translation_is_flipped(self):
        transform = s22_arcore_to_pinch()
        np.testing.assert_allclose(transform[:3, 3], [0, -0.0246, -0.1763])
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3),
                                   atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0)

    def test_checked_config_matches_implementation(self):
        config = json.loads((Path(__file__).resolve().parents[1]
                             / "configs/real/umi_s22_pinch_provisional.json")
                            .read_text(encoding="utf-8"))
        np.testing.assert_allclose(config["t_arcore_camera_to_umi_pinch"],
                                   s22_arcore_to_pinch(), atol=1e-12)
        self.assertEqual(config["status"], "provisional")

    def test_video_marker_midpoint_config_is_gl_and_matches_cv_source(self):
        config = json.loads((Path(__file__).resolve().parents[1]
                             / "configs/real/umi_s22_marker_midpoint_provisional.json")
                            .read_text(encoding="utf-8"))
        transform_gl = np.asarray(config["t_cam_to_pinch"], dtype=float)
        source = config["source_opencv"]
        angle = np.deg2rad(-105.0)
        transform_cv = np.eye(4)
        transform_cv[:3, :3] = [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
        transform_cv[:3, 3] = source["translation_m"]
        np.testing.assert_allclose(transform_gl,
                                   camera_opencv_to_arcore_child(transform_cv),
                                   atol=1e-12)
        self.assertEqual(config["t_cam_to_pinch_axes"],
                         "+X right, +Y up, -Z forward (ARCore/OpenGL)")
        self.assertEqual(config["status"], "provisional_marker_midpoint")
        loaded, checked = load_arcore_pinch_calibration(
            Path(__file__).resolve().parents[1]
            / "configs/real/umi_s22_marker_midpoint_provisional.json")
        self.assertEqual(loaded["calibration_id"], config["calibration_id"])
        np.testing.assert_allclose(checked, transform_gl, atol=1e-12)

    def test_loader_rejects_implicit_or_opencv_axes(self):
        import tempfile
        for axes in (None, "+X right, +Y down, +Z forward (OpenCV)"):
            with self.subTest(axes=axes), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "calibration.json"
                path.write_text(json.dumps({"t_cam_to_pinch_axes": axes,
                                            "t_cam_to_pinch": np.eye(4).tolist()}),
                                encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_arcore_pinch_calibration(path)

    def test_mujoco_wrist_camera_includes_gripper_to_pinch_axes(self):
        _, camera_pinch = load_arcore_pinch_calibration(
            Path(__file__).resolve().parents[1]
            / "configs/real/umi_s22_canonical_pinch_side_grasp_provisional.json")
        expected = gripper_camera_from_pinch(camera_pinch)
        camera = load_config()["cameras"]["cam_wrist"]
        np.testing.assert_allclose(camera["pos"], expected[:3, 3], atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(camera["xyaxes"]).reshape(2, 3).T,
            expected[:3, :2], atol=1e-8)

    def test_gravity_checked_axes_are_mapped_to_side_grasp_axes(self):
        root = Path(__file__).resolve().parents[1]
        _, gravity_pinch = load_arcore_pinch_calibration(
            root
            / "configs/real/umi_s22_canonical_pinch_gravity_checked_provisional.json")
        config, side_pinch = load_arcore_pinch_calibration(
            root
            / "configs/real/umi_s22_canonical_pinch_side_grasp_provisional.json")
        expected = np.eye(4)
        expected[:3, :3] = GRAVITY_PINCH_TO_SIDE_GRASP_ROTATION
        np.testing.assert_allclose(side_pinch, gravity_pinch @ expected, atol=1e-12)
        np.testing.assert_allclose(
            side_pinch,
            camera_gravity_pinch_to_side_grasp(gravity_pinch),
            atol=1e-12,
        )
        self.assertEqual(
            config["child_frame"], "canonical_pinch_marker_midpoint_side_grasp")

    def test_supplied_axes_are_explicitly_mapped_to_canonical_pinch(self):
        root = Path(__file__).resolve().parents[1]
        _, camera_hand = load_arcore_pinch_calibration(
            root / "configs/real/umi_s22_marker_midpoint_provisional.json")
        config, camera_pinch = load_arcore_pinch_calibration(
            root / "configs/real/umi_s22_canonical_pinch_gravity_checked_provisional.json")
        expected = np.eye(4)
        expected[:3, :3] = SUPPLIED_HAND_TO_CANONICAL_PINCH_ROTATION
        np.testing.assert_allclose(camera_pinch, camera_hand @ expected, atol=1e-12)
        np.testing.assert_allclose(
            camera_pinch,
            camera_supplied_hand_to_canonical_pinch(camera_hand),
            atol=1e-12,
        )
        # Canonical +Z points toward the fingertips.  A negative camera Z is
        # therefore physically behind/above a top-down pinch, not below table.
        self.assertLess(float(np.linalg.inv(camera_pinch)[2, 3]), 0.0)
        self.assertEqual(config["child_frame"], "canonical_pinch_marker_midpoint")

    def test_bad_transform_is_rejected(self):
        bad = np.eye(4); bad[0, 0] = 2
        with self.assertRaises(ValueError):
            camera_opencv_to_arcore_child(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
