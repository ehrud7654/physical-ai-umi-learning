"""Policy-free one-chunk MuJoCo contact probe for shifted-target candidates.

This probes only the first 0.8 s recorded chunk, retimed to robot limits. A
missing lift in this short prefix is not a full-episode failure. Passing this
probe does not approve these counterfactual labels for training or deployment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import load_config, normalize
from sim.mujoco.env import MujocoPickEnv
from tools.render_relative_chunk_rollout import (
    _episode_approach_start,
    _interpolated_joint_target,
    _registration,
)
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp, gap_from_angle
from umi.arm_preflight import prepare_arm_commands
from umi.convert import invert_gap_curve
from umi.relative_robot_preflight import (
    FixedActionPolicy,
    MujocoArmAdapter,
    arm_ranges,
    matrix_from_fk,
    real_limits,
)
from umi.timing import uniform_time_schedule
from umi.visual_domain import apply_visual_domain, load_visual_domain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--visual-domain", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--real-config", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "configs/real/so101_ver1.json")
    parser.add_argument("--preload-m", type=float, default=0.002)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("output already exists")
    if not np.isfinite(args.preload_m) or not 0 <= args.preload_m <= 0.005:
        raise SystemExit("simulation-only preload must be between 0 and 5 mm")
    index = json.loads((args.data / "dataset.json").read_text(encoding="utf-8"))
    if index.get("status") != "CANDIDATE_NOT_TRAINING_READY":
        raise SystemExit("only unapproved counterfactual candidates are accepted")

    registration = _registration(args.registration)
    cfg = copy.deepcopy(load_config(args.config))
    cfg = apply_visual_domain(cfg, load_visual_domain(args.visual_domain))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    cfg["task"]["object"]["half_size_m"] = (
        registration["object_size_m"] / 2).tolist()
    cfg["task"]["object"]["init_pos"] = registration["object_xyz_m"].tolist()
    cfg["task"]["table"]["half_size_m"][:2] = (
        registration["table_half_size_xy_m"].tolist())
    curve = cfg["grasp"]["gap_curve"]
    real = real_limits(args.real_config)
    ranges = arm_ranges(real)
    results = []
    with MujocoPickEnv(cfg, render=False, object_jitter_m=0.0,
                       max_ticks=1000) as env:
        ik = MujocoIK(env.model, cfg)
        solver = MujocoArmAdapter(ik, curve)
        for sample_id in index["episodes"]:
            meta = json.loads((args.data / f"{sample_id}.json").read_text(encoding="utf-8"))
            with np.load(args.data / f"{sample_id}.npz", allow_pickle=False) as stored:
                action = stored["action"][0].copy()
            episode = str(meta["source_episode"])
            row, start_arm, start_gap = _episode_approach_start(registration, episode)
            if row != int(meta["source_row"]):
                raise ValueError(f"{sample_id}: source row differs from fixed start")
            start_q = np.r_[start_arm, invert_gap_curve(start_gap, curve)]
            env.reset(seed=0, object_xy=tuple(meta["object_shifted_xyz_m"][:2]),
                      initial_q_rad=start_q)
            initial_object = env.object_position()
            initial_up = env.object_rotation()[:, 2]
            current_q = env.joint_positions()
            current_pose = matrix_from_fk(env.model, ik, current_q)
            current_gap = gap_from_angle(float(current_q[5]), curve)
            # Runtime preload changes contact commands, never stored labels.
            active = action[:, 9] <= 0.055
            action[active, 9] = np.maximum(0, action[active, 9] - args.preload_m)
            result = {
                "sample_id": sample_id,
                "object_offset_world_xy_m": meta["object_offset_world_xy_m"],
                "source_episode": episode,
                "start_row": row,
                "ik_pass": False,
            }
            try:
                commands = prepare_arm_commands(
                    FixedActionPolicy(action), {},
                    t_current=current_pose,
                    arm_current_rad=current_q[:5],
                    gripper_current_m=current_gap,
                    solver=solver,
                    ranges_rad=ranges,
                    gap_range_m=[0.0, float(real["max_gap_m"])],
                    max_step_rad=ranges[:, 1] - ranges[:, 0],
                    max_gap_step_m=float(real["max_gap_m"]),
                    max_position_error_m=0.005,
                    max_axis_error_deg=5.0,
                    max_roll_error_deg=5.0,
                )
            except ValueError as exc:
                result["ik_error"] = str(exc).splitlines()[0]
                results.append(result)
                continue
            result["ik_pass"] = True
            arm_path = np.vstack([current_q[:5], *[
                command.arm_positions_rad for command in commands]])
            gap_path = np.asarray([current_gap, *[
                command.gripper_width_m for command in commands]])
            schedule = uniform_time_schedule(
                arm_path, gap_path, source_period_s=0.1,
                max_arm_speed=float(real["max_speed_rad_s"]),
                max_arm_accel=float(real["max_accel_rad_s2"]),
                max_gap_speed=float(real["max_gap_speed_m_s"]),
                max_gap_accel=float(real["max_gap_accel_m_s2"]),
            )
            command_period = schedule.waypoint_time_s[1]
            bilateral_ticks = 0
            max_lift = 0.0
            max_pre_lift_xy_m = 0.0
            max_tilt_deg = 0.0
            max_tracking_rad = 0.0
            for command in commands:
                ticks = max(1, int(np.ceil(command_period * env.control_rate_hz)))
                segment_start_q = env.joint_positions()
                segment_start_gap = gap_from_angle(float(segment_start_q[5]), curve)
                for tick in range(ticks):
                    target = _interpolated_joint_target(
                        segment_start_q=segment_start_q,
                        target_arm_rad=command.arm_positions_rad,
                        segment_start_gap_m=segment_start_gap,
                        target_gap_m=command.gripper_width_m,
                        alpha=(tick + 1) / ticks, curve=curve,
                    )
                    env.step(normalize(target, cfg, clip=True))
                    max_tracking_rad = max(max_tracking_rad, float(np.max(np.abs(
                        env.joint_positions()[:5] - target[:5]))))
                    lift = env.lift_height()
                    max_lift = max(max_lift, lift)
                    xy = float(np.linalg.norm(
                        env.object_position()[:2] - initial_object[:2]))
                    if lift <= 0.005:
                        max_pre_lift_xy_m = max(max_pre_lift_xy_m, xy)
                    tilt = float(np.rad2deg(np.arccos(np.clip(
                        np.dot(initial_up, env.object_rotation()[:, 2]), -1.0, 1.0))))
                    max_tilt_deg = max(max_tilt_deg, tilt)
                    bilateral_ticks += env.has_bilateral_jaw_contact()
            result.update({
                "time_scale": schedule.time_scale,
                "simulated_ticks": env._ticks,
                "bilateral_contact_ticks": bilateral_ticks,
                "bilateral_contact_final": env.has_bilateral_jaw_contact(),
                "max_lift_m": max_lift,
                "final_lift_m": env.lift_height(),
                "max_pre_lift_xy_displacement_m": max_pre_lift_xy_m,
                "max_tilt_deg": max_tilt_deg,
                "max_arm_tracking_error_rad": max_tracking_rad,
                "one_chunk_stable_lift": bool(
                    env.is_success() and max_pre_lift_xy_m < 0.01
                    and max_tilt_deg < 30.0),
            })
            results.append(result)
    report = {
        "status": "POLICY_FREE_ONE_CHUNK_CONTACT_DIAGNOSTIC_NOT_TRAINING_GATE",
        "samples": len(results),
        "ik_pass": sum(bool(value["ik_pass"]) for value in results),
        "bilateral_contact_seen": sum(value.get("bilateral_contact_ticks", 0) > 0
                                      for value in results),
        "one_chunk_stable_lift": sum(bool(value.get("one_chunk_stable_lift"))
                                     for value in results),
        "training_ready": False,
        "preload_m": args.preload_m,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
