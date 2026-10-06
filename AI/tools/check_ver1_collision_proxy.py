"""Checks for the provisional SO-101 ver1 parallel-jaw collision proxy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from sim.mujoco.build_scene import (
    active_gripper_pad_ids,
    build_model,
    load_config,
    sync_gripper_collision_proxy,
)
from sim.mujoco.env import MujocoPickEnv
from tools.umi_mujoco import apply_kinematic_grasp
from umi.convert import invert_gap_curve


AI_ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = copy.deepcopy(load_config(AI_ROOT / "configs" / "so101.yaml"))
        payload = json.loads((
            AI_ROOT / "configs" / "real"
            / "so101_ver1_phone_holder_kinematic.json"
        ).read_text(encoding="utf-8"))
        apply_kinematic_grasp(self.cfg, payload)
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)

    def test_proxy_replaces_legacy_jaw_collision(self) -> None:
        names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, value)
            for value in active_gripper_pad_ids(self.model, self.cfg)
        }
        self.assertEqual(names, {"ver1_pad_negative", "ver1_pad_positive"})
        disabled = set(self.cfg["gripper_pads"]["disable_mesh_collision"])
        active_disabled_meshes = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(self.model.ngeom)
            if self.model.geom_contype[index] != 0
            and self.model.geom_dataid[index] >= 0
            and mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_MESH,
                int(self.model.geom_dataid[index])) in disabled
        }
        self.assertEqual(active_disabled_meshes, set())

    def test_inner_surface_distance_tracks_metric_gap(self) -> None:
        block = self.cfg["gripper_pads"]
        half = np.asarray(block["finger_half_size_m"], dtype=float)
        centres = []
        for gap_m in (0.0053, 0.039, 0.0794):
            self.data.qpos[5] = invert_gap_curve(
                gap_m, self.cfg["grasp"]["gap_curve"])
            self.assertTrue(sync_gripper_collision_proxy(
                self.model, self.data, self.cfg))
            mujoco.mj_forward(self.model, self.data)
            hand_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                block["visual_model"]["custom_hand_body"])
            jaw_world = self.data.xmat[hand_id].reshape(3, 3)[:, 0]
            negative, positive = [
                self.data.geom_xpos[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, name)].copy()
                for name in block["pad_names"]
            ]
            measured_gap = float(
                np.dot(positive - negative, jaw_world) - 2.0 * half[0])
            self.assertAlmostEqual(measured_gap, gap_m, places=7)
            centres.append((negative + positive) / 2.0)
        for centre in centres[1:]:
            np.testing.assert_allclose(centre, centres[0], atol=1e-9)

    def test_visible_fingers_and_collision_pads_share_moving_bodies(self) -> None:
        block = self.cfg["gripper_pads"]
        visual = block["visual_model"]
        expected = {
            float(row["sign"]): str(row["name"])
            for row in visual["finger_bodies"]
        }
        for pad_name, sign in zip(block["pad_names"], (-1.0, 1.0)):
            pad_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, pad_name)
            body_id = int(self.model.geom_bodyid[pad_id])
            self.assertEqual(
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                expected[sign],
            )
            visible_meshes = [
                geom_id for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
                and int(self.model.geom_type[geom_id])
                == int(mujoco.mjtGeom.mjGEOM_MESH)
                and float(self.model.geom_rgba[geom_id, 3]) > 0.0
            ]
            self.assertGreaterEqual(len(visible_meshes), 2)

    def test_pad_contact_planes_match_ver1_finger_meshes(self) -> None:
        block = self.cfg["gripper_pads"]
        visual = block["visual_model"]
        asset_dir = AI_ROOT / visual["asset_dir"]
        half = np.asarray(block["finger_half_size_m"], dtype=float)
        rows = {float(row["sign"]): row for row in visual["finger_bodies"]}
        for pad_name, sign in zip(block["pad_names"], (-1.0, 1.0)):
            row = rows[sign]
            finger_file = next(
                filename for filename in row["meshes"]
                if Path(filename).stem.startswith("finger_"))
            vertices = np.asarray([
                [float(value) for value in line.split()[1:4]]
                for line in (asset_dir / finger_file).read_text(
                    encoding="utf-8", errors="ignore").splitlines()
                if line.startswith("v ")
            ])
            self.assertGreater(len(vertices), 0)
            pad_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, pad_name)
            centre = self.model.geom_pos[pad_id]
            # At width zero the supplied finger contact surfaces meet at hand
            # X=0.  The diagnostic boxes must expose the same inner plane.
            inner_x = float(centre[0] - sign * half[0])
            mesh_inner_x = (
                float(vertices[:, 0].max()) if sign < 0.0
                else float(vertices[:, 0].min()))
            self.assertAlmostEqual(inner_x, 0.0, places=9)
            self.assertAlmostEqual(mesh_inner_x, 0.0, places=6)
            # The reference package fits a short contact patch at the finger
            # tip, not a collision box spanning the whole finger.  Its tip-side
            # edge must coincide with the visible mesh and the patch must stay
            # within that mesh's longitudinal extent.
            self.assertAlmostEqual(
                float(vertices[:, 1].min()), centre[1] - half[2], places=5)
            self.assertLess(float(centre[1] + half[2]), float(vertices[:, 1].max()))
            # The package rounds the fitted patch half-height to 7.7mm while
            # the CAD strip is about 7.740mm; keep only that sub-0.1mm fit
            # tolerance rather than silently enlarging the collision box.
            self.assertGreaterEqual(
                float(vertices[:, 2].min()), centre[2] - half[1] - 1e-4)
            self.assertLessEqual(
                float(vertices[:, 2].max()), centre[2] + half[1] + 1e-4)

    def test_proxy_matches_hardware_handoff_fitted_patch(self) -> None:
        block = self.cfg["gripper_pads"]
        reference = block["reference_fit"]
        self.assertEqual(
            reference["source_archive_sha256"],
            "fd7e49fe1680bef55e21ab97be71293e22f19a34abef731df0529508971c2ac2",
        )
        self.assertFalse(block["measured_collision_geometry"])
        np.testing.assert_allclose(
            block["finger_half_size_m"], [0.003, 0.0077, 0.013], atol=0.0)
        np.testing.assert_allclose(
            block["friction"], [1.2, 0.01, 0.001], atol=0.0)
        expected_centres = {
            -1.0: np.asarray([-0.003, -0.065, 0.027]),
            1.0: np.asarray([0.003, -0.065, 0.027]),
        }
        for pad_name, sign in zip(block["pad_names"], (-1.0, 1.0)):
            pad_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, pad_name)
            np.testing.assert_allclose(
                self.model.geom_size[pad_id], block["finger_half_size_m"],
                atol=1e-12,
            )
            # The source rounds Y to -0.065m; the local builder anchors the
            # same patch exactly to the configured TCP tip at -0.078018753m.
            np.testing.assert_allclose(
                self.model.geom_pos[pad_id], expected_centres[sign], atol=2e-5)

    def test_legacy_rotary_jaw_is_hidden(self) -> None:
        hidden = set(
            self.cfg["gripper_pads"]["visual_model"]["hide_legacy_meshes"])
        found = 0
        for geom_id in range(self.model.ngeom):
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name in hidden:
                found += 1
                self.assertEqual(float(self.model.geom_rgba[geom_id, 3]), 0.0)
        self.assertEqual(found, 4)

    def test_proxy_remains_explicitly_provisional(self) -> None:
        block = self.cfg["gripper_pads"]
        self.assertTrue(block["diagnostic_dynamic_ready"])
        self.assertFalse(block["measured_collision_geometry"])
        self.assertFalse(
            self.cfg["grasp"]["kinematic_overlay"]["dynamic_mujoco_ready"])

    def test_contact_report_is_split_by_pad(self) -> None:
        with MujocoPickEnv(self.cfg, render=False, object_jitter_m=0.0) as env:
            env.reset(seed=0)
            self.assertEqual(
                set(env.jaw_contact_counts()),
                {"ver1_pad_negative", "ver1_pad_positive"},
            )
            self.assertEqual(
                env.jaw_contacts(), sum(env.jaw_contact_counts().values()))
            self.assertEqual(
                env.has_bilateral_jaw_contact(),
                all(value > 0 for value in env.jaw_contact_counts().values()),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
