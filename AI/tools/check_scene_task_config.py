"""Regression: task object/table configuration must reach the compiled model."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from sim.mujoco.build_scene import build_model, load_config


def main() -> int:
    cfg = copy.deepcopy(load_config())
    cfg["task"]["object"]["half_size_m"] = [0.0195, 0.021, 0.05]
    cfg["task"]["object"]["init_pos"] = [0.18, 0.31, 0.051]
    cfg["task"]["object"]["mass_kg"] = 0.031
    cfg["task"]["table"]["half_size_m"] = [0.4, 0.41, 0.02]
    model = build_model(cfg)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    assert np.allclose(model.geom_size[gid], cfg["task"]["object"]["half_size_m"])
    assert np.allclose(model.body_pos[bid], cfg["task"]["object"]["init_pos"])
    assert np.isclose(model.body_mass[bid], cfg["task"]["object"]["mass_kg"])
    assert np.allclose(model.geom_size[tid], cfg["task"]["table"]["half_size_m"])
    assert np.allclose(model.geom_pos[tid], cfg["task"]["table"]["pos"])
    print("PASS: task object/table config reaches compiled MuJoCo model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
