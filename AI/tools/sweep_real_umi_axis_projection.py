"""Measure the reachability/fidelity trade-off of UMI approach-axis projection."""
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
from sim.mujoco.kinematics import solve_pose_ik
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_mujoco import MujocoIK
from tools.umi_real_placement_study import (REAL_J3_LOWER_RAD, load_relative_run,
                                            register_relative)
from umi.camera_frames import load_arcore_pinch_calibration, rigid


def unit(vector):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def project_axis(original, anchor, retain: float):
    """Spherical interpolation from anchor (0) to demonstrated axis (1)."""
    if not 0.0 <= retain <= 1.0:
        raise ValueError("retain must be in [0,1]")
    start, end = unit(anchor), unit(original)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle < 1e-9:
        return start
    if np.pi - angle < 1e-7:
        # The shortest path is ambiguous at 180 degrees. Choose a deterministic
        # perpendicular so the result remains reproducible.
        basis = np.array([1.0, 0.0, 0.0]) if abs(start[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perpendicular = unit(basis - np.dot(basis, start) * start)
        return unit(np.cos(np.pi * retain) * start + np.sin(np.pi * retain) * perpendicular)
    return unit(np.sin((1.0 - retain) * angle) / np.sin(angle) * start
                + np.sin(retain * angle) / np.sin(angle) * end)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


def indices_for(length: int, stride: int):
    result = np.arange(0, length, stride, dtype=int)
    return result if result[-1] == length - 1 else np.r_[result, length - 1]


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit("Shared GPU server is training-only; run this validation locally")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--retains", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    retains = [float(value) for value in args.retains.split(",")]
    if args.stride < 1 or not retains or any(not 0 <= value <= 1 for value in retains):
        raise SystemExit("invalid stride or retain values")
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")

    extrinsic, t_camera_pinch = load_arcore_pinch_calibration(args.extrinsic)
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    t_start = rigid(placement["selected"]["start_pose"])
    start_arm = np.asarray(placement["selected"]["start_arm_rad"], dtype=float)
    anchor_axis = unit(t_start[:3, 2])
    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    pinch = MujocoIK(model, cfg).pinch

    runs = []
    for bundle in sorted(args.bundles.glob("rec_*.zip")):
        quality = json.loads(bundle.with_suffix(".quality.json").read_text(encoding="utf-8"))
        relative, _, _ = load_relative_run(bundle, quality, t_camera_pinch)
        targets = register_relative(relative, t_start)
        runs.append((bundle.stem, targets, indices_for(len(targets), args.stride)))

    results = []
    for retain in retains:
        position_errors, projected_axis_errors, original_axis_losses = [], [], []
        accepted = limit_clean = 0
        per_episode = []
        for episode_id, targets, indices in runs:
            q_prev = np.r_[start_arm, 0.60]
            episode_accepted = 0
            for index in indices:
                target = targets[index]
                original_axis = unit(target[:3, 2])
                desired_axis = project_axis(original_axis, anchor_axis, retain)
                solution = solve_pose_ik(
                    model, target[:3, 3], pinch, desired_axis, q_init=q_prev,
                    wrist_roll=float(q_prev[4]), max_iters=200, axis_weight=0.15,
                    pos_tol=1e-5, axis_tol_deg=0.05)
                q_prev = np.asarray(solution.qpos, dtype=float)
                real_limits = solution.within_limits and q_prev[2] >= REAL_J3_LOWER_RAD
                ok = solution.pos_error_m <= 5e-3 and solution.axis_error_deg <= 5.0 and real_limits
                accepted += int(ok); episode_accepted += int(ok); limit_clean += int(real_limits)
                position_errors.append(solution.pos_error_m * 1000.0)
                projected_axis_errors.append(solution.axis_error_deg)
                original_axis_losses.append(angle_deg(desired_axis, original_axis))
            per_episode.append({"episode_id": episode_id, "evaluated": len(indices),
                                "accepted": episode_accepted,
                                "acceptance": episode_accepted / len(indices)})
        n = len(position_errors)
        results.append({
            "retain": retain, "evaluated": n, "accepted": accepted,
            "acceptance": accepted / n, "limit_clean_rate": limit_clean / n,
            "position_error_mm": {"p50": float(np.percentile(position_errors, 50)),
                                  "p95": float(np.percentile(position_errors, 95))},
            "projected_axis_error_deg": {
                "p50": float(np.percentile(projected_axis_errors, 50)),
                "p95": float(np.percentile(projected_axis_errors, 95))},
            "orientation_loss_from_original_deg": {
                "p50": float(np.percentile(original_axis_losses, 50)),
                "p95": float(np.percentile(original_axis_losses, 95))},
            "per_episode": per_episode,
        })
    result = {
        "purpose": "approach_axis_reachability_vs_fidelity_sweep",
        "axis_anchor": "selected SO101 start-pose approach axis",
        "retain_definition": "0=anchor axis only, 1=original UMI approach axis",
        "stride": args.stride, "results": results,
        "selection_policy": "No retain is auto-selected; task semantics must set the orientation-loss budget.",
        "limitations": placement["limitations"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{key: row[key] for key in
                       ("retain", "evaluated", "acceptance", "limit_clean_rate",
                        "position_error_mm", "orientation_loss_from_original_deg")}
                      for row in results], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
