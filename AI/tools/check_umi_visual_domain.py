"""Contract checks for the provisional S22-to-MuJoCo visual domain."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import build_model, load_config
from umi.visual_domain import apply_visual_domain, load_visual_domain


AI_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = AI_ROOT / "configs" / "real" / "umi_s22_mujoco_visual_alignment_provisional_20260917.json"


class VisualDomainTest(unittest.TestCase):
    def test_camera_and_appearance_reach_compiled_model(self):
        payload = load_visual_domain(DOMAIN)
        cfg = apply_visual_domain(load_config(DEFAULT_CONFIG), payload)
        self.assertEqual(cfg["cameras"]["cam_wrist"]["pos"], payload["camera"]["pos"])
        model = build_model(cfg)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
        self.assertAlmostEqual(float(model.cam_fovy[camera_id]), payload["camera"]["fovy"])
        for key, name in (("object_rgba", "target_object_geom"),
                          ("table_rgba", "table"), ("floor_rgba", "floor")):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            np.testing.assert_allclose(
                model.geom_rgba[geom_id], payload["appearance"][key], atol=1e-7)
        for row in payload["appearance"]["visual_geometry"]:
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, row["name"])
            self.assertGreaterEqual(geom_id, 0)
            self.assertEqual(int(model.geom_contype[geom_id]), 0)
            self.assertEqual(int(model.geom_conaffinity[geom_id]), 0)
            np.testing.assert_allclose(model.geom_rgba[geom_id], row["rgba"], atol=1e-7)

    def test_rejects_config_without_provisional_status(self):
        payload = load_visual_domain(DOMAIN)
        payload["status"] = "MEASURED_HARDWARE_CALIBRATION"
        with self.assertRaisesRegex(ValueError, "provisional"):
            apply_visual_domain(load_config(DEFAULT_CONFIG), payload)

    def test_rejects_collapsed_visual_geometry(self):
        payload = load_visual_domain(DOMAIN)
        payload["appearance"]["visual_geometry"][0]["size_m"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "geometry"):
            apply_visual_domain(load_config(DEFAULT_CONFIG), payload)

    def test_visual_domain_does_not_change_task_physics(self):
        base_cfg = load_config(DEFAULT_CONFIG)
        base = build_model(base_cfg)
        visual = build_model(apply_visual_domain(
            base_cfg, load_visual_domain(DOMAIN)))
        np.testing.assert_allclose(base.jnt_range, visual.jnt_range, atol=0.0)
        np.testing.assert_allclose(base.actuator_ctrlrange, visual.actuator_ctrlrange, atol=0.0)
        for body_name in ("target_object", "gripper"):
            base_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, body_name)
            visual_id = mujoco.mj_name2id(visual, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.assertAlmostEqual(
                float(base.body_mass[base_id]), float(visual.body_mass[visual_id]))
        for geom_name in ("target_object_geom", "ver1_pad_negative", "ver1_pad_positive"):
            base_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            visual_id = mujoco.mj_name2id(visual, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            np.testing.assert_allclose(
                base.geom_friction[base_id], visual.geom_friction[visual_id], atol=0.0)


if __name__ == "__main__":
    unittest.main()
