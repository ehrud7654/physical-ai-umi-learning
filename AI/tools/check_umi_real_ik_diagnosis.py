"""Synthetic reachability check for the position-only diagnostic solver."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np

from paths import DEFAULT_CONFIG, DEFAULT_SCENE
from sim.mujoco.build_scene import build_model, load_config
from sim.mujoco.kinematics import grasp_point
from tools.diagnose_real_umi_ik import solve_position_only, vector_angle_deg


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    model = build_model(cfg, DEFAULT_SCENE)
    offset = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    target_q = np.array([0.18, -0.24, 0.31, -0.16, 0.2, 0.6])
    data = mujoco.MjData(model)
    data.qpos[:6] = target_q
    mujoco.mj_forward(model, data)
    target = grasp_point(model, data, offset).copy()
    solved, error_m, _ = solve_position_only(model, target, offset, np.zeros(6))
    assert solved[2] >= -1.57079632679
    assert error_m <= 5e-3, error_m
    assert abs(vector_angle_deg([1, 0, 0], [0, 1, 0]) - 90.0) < 1e-12
    print(f"PASS: synthetic position-only IK error={error_m * 1000:.4f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
