"""Search collision-free object registration candidates for one real UMI clip."""
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
from tools.replay_real_umi_trajectory import (build_commands, collision_counts,
                                               register_close_to_object)
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_real_placement_study import load_relative_run
from umi.camera_frames import rigid


def values(spec: str):
    return [float(value) for value in spec.split(",")]


def direct_collision_summary(model, commands):
    data = mujoco.MjData(model)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    table_frames = self_frames = 0
    for command in commands:
        data.qpos[:6] = command
        mujoco.mj_forward(model, data)
        table, self_count = collision_counts(model, data, object_geom, table_geom)
        table_frames += int(table > 0)
        self_frames += int(self_count > 0)
    return table_frames, self_frames


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit("Shared GPU server is training-only; run this search locally")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--x", default="0.16,0.19,0.22,0.25")
    parser.add_argument("--y", default="-0.10,-0.05,0,0.05,0.10")
    parser.add_argument("--yaw", default="-90,-45,0,45,90")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    if args.out.exists() or args.stride < 1:
        raise SystemExit("output must not exist and stride must be >=1")
    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    extrinsic = json.loads(args.extrinsic.read_text(encoding="utf-8"))
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    relative, gaps, _ = load_relative_run(
        args.bundle, quality, rigid(extrinsic["t_arcore_camera_to_umi_pinch"]))
    sample = np.arange(0, len(relative), args.stride, dtype=int)
    if sample[-1] != len(relative) - 1:
        sample = np.r_[sample, len(relative) - 1]
    object_z = float(cfg["task"]["object"]["init_pos"][2])
    rows = []
    for x in values(args.x):
        for y in values(args.y):
            for yaw in values(args.yaw):
                targets, close_index = register_close_to_object(
                    relative[sample], gaps[sample], [x, y, object_z],
                    cfg["grasp"]["grasp_z_offset_m"], yaw)
                commands, modes, losses, _ = build_commands(
                    model, cfg, targets, gaps[sample], placement["selected"]["start_arm_rad"])
                table_frames, self_frames = direct_collision_summary(model, commands)
                step = np.max(np.abs(np.diff(commands, axis=0)), axis=0)
                rows.append({"x": x, "y": y, "yaw_deg": yaw,
                             "sample_frames": len(sample), "sample_close_index": close_index,
                             "full_pose_rate": modes.count("full_pose") / len(modes),
                             "fallback_axis_loss_p95_deg": float(np.percentile(losses, 95)),
                             "table_collision_frames": table_frames,
                             "self_collision_frames": self_frames,
                             "max_arm_step_rad": float(np.max(step[:5]))})
    # Collision freedom is lexicographically mandatory, not traded for IK rate.
    rows.sort(key=lambda row: (row["table_collision_frames"], row["self_collision_frames"],
                               -row["full_pose_rate"], row["fallback_axis_loss_p95_deg"],
                               row["max_arm_step_rad"]))
    result = {"purpose": "collision_free_grasp_registration_search",
              "episode_id": args.bundle.stem, "stride": args.stride,
              "evaluated_candidates": len(rows), "selected": rows[0],
              "candidates": rows,
              "limitations": ["Kinematic collision screen; selected candidate needs dynamic replay.",
                              "Object xyz is inferred from the configured scene, not measured in S22 video.",
                              "Object geometry must match the object visible in the selected recording."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated_candidates": len(rows), "selected": rows[0]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
