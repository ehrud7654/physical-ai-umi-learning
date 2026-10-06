"""Separate position reachability from approach-axis compatibility on real UMI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from paths import DEFAULT_CONFIG, DEFAULT_SCENE
from sim.mujoco.build_scene import build_model, load_config
from sim.mujoco.kinematics import approach_axis, grasp_point
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_mujoco import MujocoIK
from tools.umi_real_placement_study import (REAL_J3_LOWER_RAD, load_relative_run,
                                            register_relative)
from umi.camera_frames import rigid
from umi.ik import matrix_to_quat


def solve_position_only(model, target_xyz, offset_local, q_init, *, max_iters=200,
                        pos_tol_m=1e-5):
    """Damped least-squares pinch-position IK; wrist roll is held fixed."""
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(q_init, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    lo, hi = model.jnt_range[:6, 0].copy(), model.jnt_range[:6, 1].copy()
    lo[2] = max(lo[2], REAL_J3_LOWER_RAD)
    free = [0, 1, 2, 3]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    hit_limit = False
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        point = grasp_point(model, data, offset_local)
        error = target - point
        if np.linalg.norm(error) < pos_tol_m:
            break
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
        jac = jacp[:, free]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 5e-3 * np.eye(3), error)
        q = data.qpos[:6].copy()
        raw = q[free] + np.clip(dq, -0.1, 0.1)
        hit_limit |= bool(np.any(raw < lo[free] - 1e-9) or np.any(raw > hi[free] + 1e-9))
        q[free] = np.clip(raw, lo[free], hi[free])
        data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    position_error_m = float(np.linalg.norm(target - grasp_point(model, data, offset_local)))
    return data.qpos[:6].copy(), position_error_m, hit_limit


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def achieved_axis(model, q_rad):
    data = mujoco.MjData(model)
    data.qpos[:6] = np.asarray(q_rad, dtype=float)
    mujoco.mj_forward(model, data)
    return approach_axis(model, data).copy()


def vector_angle_deg(a, b):
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    aa, bb = aa / np.linalg.norm(aa), bb / np.linalg.norm(bb)
    return float(np.degrees(np.arccos(np.clip(np.dot(aa, bb), -1.0, 1.0))))


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit("Shared GPU server is training-only; run this validation locally")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    if args.stride < 1 or args.out.exists():
        raise SystemExit("stride must be >=1 and output must not already exist")

    extrinsic = json.loads(args.extrinsic.read_text(encoding="utf-8"))
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    t_camera_pinch = rigid(extrinsic["t_arcore_camera_to_umi_pinch"])
    t_start = rigid(placement["selected"]["start_pose"])
    start_arm = np.asarray(placement["selected"]["start_arm_rad"], dtype=float)
    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg)

    totals = {key: 0 for key in ("evaluated", "position_only_under_5mm",
                                  "full_position_under_5mm", "full_axis_under_5deg",
                                  "full_limit_clean", "full_accepted")}
    pos_only_errors, full_pos_errors, full_axis_errors, feasible_axis_losses = [], [], [], []
    episodes = []
    for bundle in sorted(args.bundles.glob("rec_*.zip")):
        quality = json.loads(bundle.with_suffix(".quality.json").read_text(encoding="utf-8"))
        relative, gaps, _ = load_relative_run(bundle, quality, t_camera_pinch)
        targets = register_relative(relative, t_start)
        indices = np.arange(0, len(targets), args.stride, dtype=int)
        if indices[-1] != len(targets) - 1:
            indices = np.r_[indices, len(targets) - 1]
        q_pos = np.r_[start_arm, 0.60]
        q_full = q_pos.copy()
        row = {key: 0 for key in totals}
        for index in indices:
            target = targets[index]
            q_pos, pos_error, _ = solve_position_only(
                model, target[:3, 3], ik.pinch, q_pos)
            feasible_axis_losses.append(
                vector_angle_deg(achieved_axis(model, q_pos), target[:3, 2]))
            full = ik.solve(target[:3, 3], matrix_to_quat(target[:3, :3]),
                            float(gaps[index]), q_init=q_full)
            q_full = full.q_rad
            position_ok = pos_error <= 5e-3
            full_position_ok = full.pos_error_m <= 5e-3
            axis_ok = full.axis_error_deg <= 5.0
            limit_ok = full.within_limits and full.q_rad[2] >= REAL_J3_LOWER_RAD
            accepted = full_position_ok and axis_ok and limit_ok
            row["evaluated"] += 1
            row["position_only_under_5mm"] += int(position_ok)
            row["full_position_under_5mm"] += int(full_position_ok)
            row["full_axis_under_5deg"] += int(axis_ok)
            row["full_limit_clean"] += int(limit_ok)
            row["full_accepted"] += int(accepted)
            pos_only_errors.append(pos_error * 1000)
            full_pos_errors.append(full.pos_error_m * 1000)
            full_axis_errors.append(full.axis_error_deg)
        for key in totals:
            totals[key] += row[key]
        row["episode_id"] = bundle.stem
        row["position_only_rate"] = row["position_only_under_5mm"] / row["evaluated"]
        row["full_acceptance"] = row["full_accepted"] / row["evaluated"]
        episodes.append(row)

    n = totals["evaluated"]
    rates = {key.replace("_under_5mm", "_rate").replace("_under_5deg", "_rate")
             .replace("_clean", "_rate").replace("full_accepted", "full_acceptance"):
             value / n for key, value in totals.items() if key != "evaluated"}
    result = {
        "purpose": "separate_position_reachability_from_approach_axis_constraint",
        "stride": args.stride, "totals": totals, "rates": rates,
        "errors": {
            "position_only_mm": {"p50": percentile(pos_only_errors, 50),
                                  "p95": percentile(pos_only_errors, 95)},
            "full_position_mm": {"p50": percentile(full_pos_errors, 50),
                                  "p95": percentile(full_pos_errors, 95)},
            "full_axis_deg": {"p50": percentile(full_axis_errors, 50),
                               "p95": percentile(full_axis_errors, 95)},
            "position_projected_axis_loss_deg": {
                "p50": percentile(feasible_axis_losses, 50),
                "p95": percentile(feasible_axis_losses, 95),
                "max": percentile(feasible_axis_losses, 100)},
        },
        "interpretation": {
            "position_only_high_full_low": "approach-axis/5-DoF compatibility is the main loss",
            "position_only_low": "workspace placement or translation/extrinsic is the main loss",
        },
        "per_episode": episodes,
        "limitations": placement["limitations"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated": n, "rates": rates, "errors": result["errors"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
