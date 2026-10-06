"""Policy-free duration sweep for a smooth moving-H=2 command profile.

Durations are ranked only by IK, contact, speed and acceleration.  No learned
checkpoint is loaded and no image-response result participates in selection.
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
from sim.mujoco.build_scene import load_config
from sim.mujoco.env import MujocoPickEnv
from tools.probe_fixed_pose_object_only import _source_path, _suite_conditions
from tools.probe_moving_history_object_only import (
    _history_target, _render_history, _simulate_history,
)
from tools.render_relative_chunk_rollout import (
    _diagnostic_object_xyz, _episode_approach_start, _registration,
)
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp, gap_from_angle
from umi.convert import invert_gap_curve
from umi.relative_robot_preflight import matrix_from_fk, real_limits
from umi.visual_domain import apply_visual_domain, load_visual_domain


def audit(suite_path: Path, durations_s: tuple[float, ...], *,
          backoff_m: float, history_target_step_m: float,
          open_gap_m: float, snapshot_duration_s: float | None = None,
          snapshot_dir: Path | None = None) -> dict:
    if (not durations_s or any(not np.isfinite(value) or value < 0.1
                               for value in durations_s)
            or tuple(sorted(set(durations_s))) != durations_s):
        raise ValueError("durations must be unique, sorted and >=0.1s")
    if (snapshot_duration_s is None) != (snapshot_dir is None):
        raise ValueError("snapshot duration and directory must be provided together")
    if snapshot_duration_s is not None and snapshot_duration_s not in durations_s:
        raise ValueError("snapshot duration must be in the predeclared sweep")
    conditions, episodes, _ = _suite_conditions(suite_path)
    registration_path = _source_path(conditions["registration"])
    visual_domain_path = _source_path(conditions["visual_domain"])
    real_config_path = (Path(__file__).resolve().parents[1]
                        / "configs/real/so101_ver1.json")
    registration = _registration(registration_path)
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
    cfg = apply_visual_domain(
        cfg, load_visual_domain(visual_domain_path))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    cfg["task"]["object"]["half_size_m"] = (
        registration["object_size_m"] / 2).tolist()
    cfg["task"]["object"]["init_pos"] = registration["object_xyz_m"].tolist()
    cfg["task"]["table"]["half_size_m"][:2] = (
        registration["table_half_size_xy_m"].tolist())
    curve = cfg["grasp"]["gap_curve"]
    approach = np.asarray(
        registration["desired_world_approach_axis"], dtype=float)
    approach /= np.linalg.norm(approach)
    real = real_limits(real_config_path)
    speed_limit = float(real["max_speed_rad_s"])
    accel_limit = float(real["max_accel_rad_s2"])

    rows = []
    pending_snapshots: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    with MujocoPickEnv(
            cfg, render=snapshot_dir is not None, object_jitter_m=0.0) as env:
        ik = MujocoIK(env.model, cfg)
        for episode in episodes:
            try:
                source_row, arm, gap = _episode_approach_start(
                    registration, episode)
                _, object_xyz = _diagnostic_object_xyz(
                    registration, episode, np.zeros(2))
                anchor_q = np.r_[arm, invert_gap_curve(gap, curve)]
                previous_q, current_q, ik_details = _history_target(
                    ik, env.model, anchor_q, approach,
                    backoff_m=backoff_m,
                    history_target_step_m=history_target_step_m,
                    open_gap_m=open_gap_m)
            except (ValueError, OSError, KeyError) as exc:
                rows.append({"episode": episode, "duration_s": None,
                             "endpoint_ik_valid": False,
                             "reason": str(exc).splitlines()[0]})
                continue
            for duration in durations_s:
                try:
                    snapshots, motion = _simulate_history(
                        env, previous_q=previous_q,
                        current_target_q=current_q,
                        object_xy=object_xyz[:2],
                        speed_limit_rad_s=speed_limit,
                        command_profile="quintic",
                        motion_duration_s=duration)
                    previous_pose = matrix_from_fk(
                        env.model, ik, snapshots[0]["qpos"][:6])
                    current_pose = matrix_from_fk(
                        env.model, ik, snapshots[1]["qpos"][:6])
                    motion["tcp_translation_m"] = float(np.linalg.norm(
                        current_pose[:3, 3] - previous_pose[:3, 3]))
                    motion["configured_accel_limit_rad_s2"] = accel_limit
                    row = {
                        "episode": episode, "duration_s": duration,
                        "source_row": source_row,
                        "endpoint_ik_valid": True, "ik": ik_details,
                        "motion": motion,
                        "acceleration_valid": bool(
                            motion["finite_difference_peak_accel_rad_s2"]
                            <= accel_limit + 1e-9),
                        "current_arm_rad": snapshots[1]["qpos"][:5].tolist(),
                        "current_gap_m": gap_from_angle(
                            float(snapshots[1]["qpos"][5]), curve),
                    }
                    rows.append(row)
                    if (snapshot_duration_s is not None
                            and duration == snapshot_duration_s
                            and row["acceleration_valid"]):
                        baseline = _render_history(
                            env, snapshots, [(0.0, 0.0)])[0]
                        pending_snapshots[(episode, duration)] = {
                            "image": baseline["images"].copy(),
                            "qpos": np.stack([
                                snapshots[0]["qpos"], snapshots[1]["qpos"]]),
                            "qvel": np.stack([
                                snapshots[0]["qvel"], snapshots[1]["qvel"]]),
                            "timestamp": np.asarray([
                                snapshots[0]["time"], snapshots[1]["time"]],
                                dtype=float),
                        }
                except (ValueError, OSError, KeyError) as exc:
                    rows.append({"episode": episode, "duration_s": duration,
                                 "endpoint_ik_valid": True,
                                 "reason": str(exc).splitlines()[0],
                                 "acceleration_valid": False})
    eligible = sorted({row["episode"] for row in rows
                       if row["duration_s"] is not None})
    summary = []
    for duration in durations_s:
        selected = [row for row in rows if row["duration_s"] == duration]
        valid = [row for row in selected if row.get("acceleration_valid")]
        summary.append({
            "duration_s": duration,
            "eligible_episodes": len(eligible),
            "acceleration_valid_episodes": len(valid),
            "all_eligible_acceleration_valid": bool(eligible)
            and len(selected) == len(eligible) and len(valid) == len(eligible),
            "peak_accel_rad_s2": (max(row["motion"][
                "finite_difference_peak_accel_rad_s2"] for row in selected
                if "motion" in row) if any("motion" in row for row in selected)
                else None),
            "median_tcp_translation_m": (float(np.median([
                row["motion"]["tcp_translation_m"] for row in valid]))
                if valid else None),
            "minimum_command_fraction_at_current": (min(
                row["motion"]["command_fraction_at_current"]
                for row in valid) if valid else None),
        })
    passing = [row["duration_s"] for row in summary
               if row["all_eligible_acceleration_valid"]]
    shortest_passing = passing[0] if passing else None
    if snapshot_duration_s is not None:
        if snapshot_duration_s != shortest_passing:
            raise ValueError(
                "snapshot duration must equal the shortest all-episode "
                "acceleration-valid duration")
        expected = {(episode, snapshot_duration_s) for episode in eligible}
        if set(pending_snapshots) != expected:
            raise ValueError("selected snapshot duration is incomplete")
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        for row in rows:
            key = (row.get("episode"), row.get("duration_s"))
            if key not in pending_snapshots:
                continue
            snapshot_path = snapshot_dir / f"{key[0]}.npz"
            np.savez_compressed(snapshot_path, **pending_snapshots[key])
            row["snapshot_npz"] = str(snapshot_path)
    return {
        "status": "POLICY_FREE_MOVING_H2_SMOOTH_PROFILE_AUDIT",
        "suite": str(suite_path),
        "registration": str(registration_path),
        "visual_domain": str(visual_domain_path),
        "real_config": str(real_config_path),
        "command_profile": "quintic",
        "history_interval_s": 0.1,
        "endpoint_translation_span_m": history_target_step_m,
        "backoff_m": backoff_m,
        "open_gap_m": open_gap_m,
        "configured_speed_limit_rad_s": speed_limit,
        "configured_accel_limit_rad_s2": accel_limit,
        "eligible_episodes": eligible,
        "endpoint_ik_rejected_episodes": sorted(set(episodes) - set(eligible)),
        "predeclared_duration_sweep": list(durations_s),
        "summary": summary,
        "shortest_all_episode_acceleration_valid_duration_s": (
            shortest_passing),
        "snapshot_duration_s": snapshot_duration_s,
        "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
        "rows": rows,
        "training_ready": False,
        "limitations": [
            "Policy-free simulation audit only; no checkpoint or visual metric is used.",
            "The 3mm value is the full command span; the first 100ms observation pair traverses only the reported fraction.",
            "The 3rad/s^2 limit is provisional and not a hardware approval.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--durations-s", type=float, nargs="+",
                        default=(0.2, 0.3, 0.4, 0.5, 0.6))
    parser.add_argument("--backoff-m", type=float, default=0.01)
    parser.add_argument("--history-target-step-m", type=float, default=0.003)
    parser.add_argument("--open-gap-m", type=float, default=0.045)
    parser.add_argument("--snapshot-duration-s", type=float)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = audit(
        args.suite, tuple(args.durations_s), backoff_m=args.backoff_m,
        history_target_step_m=args.history_target_step_m,
        open_gap_m=args.open_gap_m,
        snapshot_duration_s=args.snapshot_duration_s,
        snapshot_dir=args.snapshot_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"summary": result["summary"],
                      "shortest_all_episode_acceleration_valid_duration_s":
                      result["shortest_all_episode_acceleration_valid_duration_s"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
