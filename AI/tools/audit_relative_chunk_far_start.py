"""Policy-free fixed-start reachability and timing audit for the UMI side grasp.

The same provisional home joint pose is used for every episode.  The TCP path
is a straight position interpolation with quaternion-slerped orientation to
each predeclared near-contact approach pose.  This is a diagnostic candidate,
not a robot-base registration, collision certificate, or motor command.
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
    _detected_bbox_norm,
    _episode_approach_start,
    _episode_object_xyz,
    _interpolated_joint_target,
    _registration,
)
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp
from umi.convert import invert_gap_curve
from umi.ik import matrix_to_quat
from umi.relative_robot_preflight import arm_ranges, matrix_from_fk, real_limits
from umi.timing import uniform_time_schedule
from umi.visual_domain import apply_visual_domain, load_visual_domain


AI_ROOT = Path(__file__).resolve().parents[1]


def _slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc interpolation of two wxyz unit quaternions."""
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("slerp expects two quaternions")
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + fraction * (second - first)
    else:
        angle = float(np.arccos(dot))
        result = (np.sin((1.0 - fraction) * angle) * first
                  + np.sin(fraction * angle) * second) / np.sin(angle)
    return result / np.linalg.norm(result)


def _valid_ik(solution: object, ranges: np.ndarray) -> bool:
    return bool(
        solution.converged and solution.within_limits
        and solution.pos_error_m <= 0.005
        and solution.axis_error_deg <= 5.0
        and solution.roll_residual_deg <= 5.0
        and np.all(solution.q_rad[:5] >= ranges[:, 0])
        and np.all(solution.q_rad[:5] <= ranges[:, 1])
    )


def _profile_fraction(value: float, profile: str) -> float:
    if profile == "linear":
        return value
    if profile == "quintic":
        return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))
    raise ValueError(f"unsupported approach profile {profile!r}")


def audit(*, registration_path: Path, visual_domain_path: Path,
          real_config_path: Path, episodes: list[str] | None,
          waypoints: int, source_period_s: float, profile: str,
          execution_margin: float, diagnostic_time_scale_cap: float,
          start_anchor_episode: str | None, start_fraction: float,
          start_gap_m: float | None,
          handoff_waypoint: int | None,
          snapshot_dir: Path | None) -> dict:
    if (waypoints < 2 or source_period_s <= 0 or execution_margin < 1
            or diagnostic_time_scale_cap < 1
            or not 0 <= start_fraction <= 1):
        raise ValueError("invalid waypoint, source period, or diagnostic cap")
    handoff = waypoints if handoff_waypoint is None else handoff_waypoint
    if not 1 <= handoff <= waypoints:
        raise ValueError("handoff waypoint must lie within planned path")
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
    registration = _registration(registration_path)
    eligible = [str(row["episode"]) for row in
                registration.get("selected", {}).get("per_episode", [])
                if row.get("scene_constraints_ok") is True]
    chosen = eligible if episodes is None else episodes
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(eligible):
        raise ValueError("episodes must be unique and scene-valid in registration")

    real = real_limits(real_config_path)
    ranges = arm_ranges(real)
    start_arm = np.asarray([real["home_rad"][name] for name in real["joint_order"]],
                           dtype=float)
    if start_fraction:
        if start_anchor_episode is None or start_anchor_episode not in eligible:
            raise ValueError("nonzero start fraction requires a scene-valid anchor episode")
        _, anchor_arm, _ = _episode_approach_start(registration,
                                                   start_anchor_episode)
        start_arm = start_arm + start_fraction * (anchor_arm - start_arm)
    start_gap = float(real["home_gap_m"] if start_gap_m is None else start_gap_m)
    if np.any(start_arm < ranges[:, 0]) or np.any(start_arm > ranges[:, 1]):
        raise ValueError("configured home is outside installed diagnostic arm ranges")
    if not 0 <= start_gap <= real["max_gap_m"]:
        raise ValueError("configured home gap is outside the gap contract")

    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
    cfg = apply_visual_domain(cfg, load_visual_domain(visual_domain_path))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    cfg["task"]["object"]["half_size_m"] = (
        registration["object_size_m"] / 2).tolist()
    cfg["task"]["object"]["init_pos"] = registration["object_xyz_m"].tolist()
    cfg["task"]["table"]["half_size_m"][:2] = (
        registration["table_half_size_xy_m"].tolist())
    curve = cfg["grasp"]["gap_curve"]
    start_q = np.r_[start_arm, invert_gap_curve(start_gap, curve)]
    rows = []
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.0) as env:
        ik = MujocoIK(env.model, cfg)
        start_pose = matrix_from_fk(env.model, ik, start_q)
        start_quat = matrix_to_quat(start_pose[:3, :3])
        for episode in chosen:
            object_xyz = _episode_object_xyz(registration, episode)
            _, endpoint_arm, endpoint_gap = _episode_approach_start(
                registration, episode)
            endpoint_q = np.r_[endpoint_arm,
                               invert_gap_curve(start_gap, curve)]
            endpoint_pose = matrix_from_fk(env.model, ik, endpoint_q)
            end_quat = matrix_to_quat(endpoint_pose[:3, :3])
            observation = env.reset(seed=0, object_xy=tuple(object_xyz[:2]),
                                    initial_q_rad=start_q)
            start_time = float(env.data.time)
            initial_object = env.object_position()
            start_distance = float(np.linalg.norm(
                start_pose[:3, 3] - initial_object))
            start_xy_distance = float(np.linalg.norm(
                start_pose[:2, 3] - initial_object[:2]))
            row = {
                "episode": episode,
                "fixed_start_arm_rad": start_arm.tolist(),
                "start_tcp_to_object_m": start_distance,
                "start_tcp_to_object_xy_m": start_xy_distance,
                "start_object_bbox_norm": _detected_bbox_norm(
                    observation.images["cam_wrist"]),
                "initial_jaw_object_contacts": env.jaw_contacts(),
                "endpoint_arm_rad": endpoint_arm.tolist(),
                "recorded_near_start_gap_m": endpoint_gap,
                "planned_open_approach_gap_m": start_gap,
                "endpoint_tcp_distance_m": float(np.linalg.norm(
                    endpoint_pose[:3, 3] - start_pose[:3, 3])),
                "ik_pass": False,
                "diagnostic_cap_pass": False,
            }
            arm_path = [start_arm.copy()]
            gap_path = [start_gap]
            seed = start_q.copy()
            peak_position_error = peak_axis_error = peak_roll_error = 0.0
            for index in range(1, waypoints + 1):
                fraction = _profile_fraction(index / waypoints, profile)
                target_pos = ((1 - fraction) * start_pose[:3, 3]
                              + fraction * endpoint_pose[:3, 3])
                target_quat = _slerp(start_quat, end_quat, fraction)
                # The policy-free warmup is an open-jaw arm approach. Closing
                # toward a recorded near-start gap can move the object before
                # the learned policy ever receives its first observation.
                target_gap = start_gap
                solution = ik.solve(target_pos, target_quat, target_gap,
                                    q_init=seed)
                peak_position_error = max(peak_position_error,
                                          float(solution.pos_error_m))
                peak_axis_error = max(peak_axis_error,
                                      float(solution.axis_error_deg))
                peak_roll_error = max(peak_roll_error,
                                      float(solution.roll_residual_deg))
                if not _valid_ik(solution, ranges):
                    row.update(first_failed_waypoint=index,
                               failure_fraction=fraction,
                               failure_reason="IK residual, convergence, or installed range")
                    break
                seed = solution.q_rad.copy()
                arm_path.append(seed[:5].copy())
                gap_path.append(target_gap)
            row.update(max_ik_position_error_m=peak_position_error,
                       max_ik_axis_error_deg=peak_axis_error,
                       max_ik_roll_error_deg=peak_roll_error)
            if len(arm_path) != waypoints + 1:
                rows.append(row)
                continue
            row["ik_pass"] = True
            schedule = uniform_time_schedule(
                np.stack(arm_path), np.asarray(gap_path),
                source_period_s=source_period_s,
                max_arm_speed=float(real["max_speed_rad_s"]),
                max_arm_accel=float(real["max_accel_rad_s2"]),
                max_gap_speed=float(real["max_gap_speed_m_s"]),
                max_gap_accel=float(real["max_gap_accel_m_s2"]),
            )
            row.update(
                source_arm_speed_rad_s=schedule.source.arm_speed_rad_s,
                source_arm_accel_rad_s2=schedule.source.arm_accel_rad_s2,
                required_time_scale=schedule.time_scale,
                execution_time_scale=schedule.time_scale * execution_margin,
                planned_full_duration_s=(schedule.waypoint_time_s[-1]
                                         * execution_margin),
                handoff_scheduled_duration_s=(schedule.waypoint_time_s[handoff]
                                              * execution_margin),
                scheduled_arm_speed_rad_s=(
                    schedule.scheduled.arm_speed_rad_s / execution_margin),
                scheduled_arm_accel_rad_s2=(
                    schedule.scheduled.arm_accel_rad_s2 / execution_margin**2),
                diagnostic_cap_pass=(
                    schedule.time_scale * execution_margin
                    <= diagnostic_time_scale_cap + 1e-9),
            )
            # Measure the achieved trajectory separately from the retimed
            # waypoint commands. Position-actuator transients are not hidden by
            # a command-space acceleration calculation.
            prior_q = env.joint_positions()
            prior_velocity = np.asarray(env.data.qvel[:5], dtype=float).copy()
            max_actual_speed = max_actual_accel = max_tracking = 0.0
            max_object_xy_drift = max_object_tilt = 0.0
            contact_ticks = 0
            initial_up = env.object_rotation()[:, 2]
            final_snapshots: list[dict[str, np.ndarray | float]] = []
            for index in range(1, handoff + 1):
                duration = ((schedule.waypoint_time_s[index]
                             - schedule.waypoint_time_s[index - 1])
                            * execution_margin)
                ticks = max(1, int(np.ceil(duration * env.control_rate_hz)))
                segment_start = env.joint_positions()
                segment_gap = gap_path[index - 1]
                for tick in range(ticks):
                    target = _interpolated_joint_target(
                        segment_start_q=segment_start,
                        target_arm_rad=arm_path[index],
                        segment_start_gap_m=segment_gap,
                        target_gap_m=gap_path[index],
                        alpha=(tick + 1) / ticks,
                        curve=curve,
                    )
                    before = float(env.data.time)
                    observed = env.step(normalize(target, cfg, clip=True))
                    actual_q = env.joint_positions()
                    actual_dt = float(env.data.time) - before
                    actual_velocity = (actual_q[:5] - prior_q[:5]) / actual_dt
                    max_actual_speed = max(max_actual_speed,
                                           float(np.max(np.abs(actual_velocity))))
                    max_actual_accel = max(max_actual_accel, float(np.max(
                        np.abs(actual_velocity - prior_velocity))) / actual_dt)
                    max_tracking = max(max_tracking, float(np.max(
                        np.abs(actual_q[:5] - target[:5]))))
                    prior_q = actual_q
                    prior_velocity = actual_velocity
                    contact_ticks += int(env.jaw_contacts() > 0)
                    max_object_xy_drift = max(max_object_xy_drift, float(
                        np.linalg.norm(env.object_position()[:2]
                                       - initial_object[:2])))
                    tilt = float(np.rad2deg(np.arccos(np.clip(
                        np.dot(initial_up, env.object_rotation()[:, 2]),
                        -1.0, 1.0))))
                    max_object_tilt = max(max_object_tilt, tilt)
                    if index == handoff:
                        final_snapshots.append({
                            "timestamp": float(observed.timestamp),
                            "image": observed.images["cam_wrist"].copy(),
                            "qpos": env.data.qpos.copy(),
                            "qvel": env.data.qvel.copy(),
                        })
            current = final_snapshots[-1]
            history_steps = int(round(0.1 * env.control_rate_hz))
            previous = final_snapshots[-1 - history_steps]
            history_interval = (float(current["timestamp"])
                                - float(previous["timestamp"]))
            history_joint_change = float(np.max(np.abs(
                current["qpos"][:5] - previous["qpos"][:5])))
            history_image_change = float(np.mean(np.abs(
                current["image"].astype(float)
                - previous["image"].astype(float))))
            row.update(
                executed_duration_s=float(env.data.time - start_time),
                actual_arm_speed_peak_rad_s=max_actual_speed,
                actual_arm_accel_peak_rad_s2=max_actual_accel,
                max_joint_tracking_error_rad=max_tracking,
                jaw_object_contact_ticks_before_policy=contact_ticks,
                max_object_xy_drift_m=max_object_xy_drift,
                max_object_tilt_deg=max_object_tilt,
                final_object_bbox_norm=_detected_bbox_norm(current["image"]),
                final_history_interval_s=history_interval,
                final_history_joint_change_rad=history_joint_change,
                final_history_image_mae_uint8=history_image_change,
                handoff_tcp_distance_to_planned_endpoint_m=float(
                    np.linalg.norm(matrix_from_fk(env.model, ik,
                        env.joint_positions())[:3, 3] - endpoint_pose[:3, 3])),
                diagnostic_motion_pass=bool(
                    max_actual_speed <= float(real["max_speed_rad_s"]) + 1e-9
                    and max_actual_accel <= float(real["max_accel_rad_s2"]) + 1e-9
                    and max_tracking <= float(real["tracking_rad"])
                    and contact_ticks == 0
                    and max_object_xy_drift <= 0.001
                    and max_object_tilt <= 5.0
                    and abs(history_interval - 0.1) <= 0.01
                    and history_joint_change >= 1e-4
                    and _detected_bbox_norm(current["image"]) is not None
                ),
            )
            if row["diagnostic_motion_pass"] and snapshot_dir is not None:
                snapshot_path = snapshot_dir / f"{episode}.npz"
                if snapshot_path.exists():
                    raise ValueError(f"snapshot output already exists: {snapshot_path}")
                np.savez_compressed(
                    snapshot_path,
                    image=np.stack([previous["image"], current["image"]]),
                    qpos=np.stack([previous["qpos"], current["qpos"]]),
                    qvel=np.stack([previous["qvel"], current["qvel"]]),
                    timestamp=np.asarray([previous["timestamp"],
                                          current["timestamp"]], dtype=float),
                )
                row["snapshot_npz"] = str(snapshot_path)
            rows.append(row)

    return {
        "status": "POLICY_FREE_FIXED_START_TO_APPROACH_IK_TIMING_AUDIT",
        "registration": str(registration_path),
        "visual_domain": str(visual_domain_path),
        "real_config": str(real_config_path),
        "episodes": chosen,
        "fixed_start_arm_rad": start_arm.tolist(),
        "fixed_start_gap_m": start_gap,
        "fixed_start_selection": {
            "kind": "home_to_anchor_joint_interpolation",
            "anchor_episode": start_anchor_episode,
            "fraction": start_fraction,
            "candidate_search_not_performance_test": True,
        },
        "waypoints": waypoints,
        "handoff_waypoint": handoff,
        "handoff_path_fraction": _profile_fraction(handoff / waypoints,
                                                    profile),
        "approach_profile": profile,
        "source_period_s": source_period_s,
        "execution_margin": execution_margin,
        "diagnostic_time_scale_cap": diagnostic_time_scale_cap,
        "diagnostic_cap_is_team_approved": False,
        "ik_pass": sum(row["ik_pass"] for row in rows),
        "diagnostic_cap_pass": sum(row["diagnostic_cap_pass"] for row in rows),
        "first_view_object_visible": sum(row["start_object_bbox_norm"] is not None
                                         for row in rows),
        "diagnostic_motion_pass": sum(row.get("diagnostic_motion_pass") is True
                                      for row in rows),
        "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
        "rows": rows,
        "limitations": [
            "Policy-free straight TCP interpolation from one fixed candidate start; not a learned rollout.",
            "Checks waypoint IK, command-space timing and simulator-achieved motion; not certified swept installation clearance.",
            "Configured home, 5-axis IK, collision proxy, visual alignment, and dynamic limits are provisional.",
            "The caller's time-scale cap is diagnostic and is not a team-approved robot execution limit.",
            "Failure blocks subsequent learned-policy far-start claims; success alone would not authorise motors.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--visual-domain", type=Path, required=True)
    parser.add_argument("--real-config", type=Path, default=AI_ROOT / "configs/real/so101_ver1.json")
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument("--waypoints", type=int, default=16)
    parser.add_argument("--handoff-waypoint", type=int)
    parser.add_argument("--profile", choices=("linear", "quintic"),
                        default="quintic")
    parser.add_argument("--source-period-s", type=float, default=0.1)
    parser.add_argument("--execution-margin", type=float, default=1.0)
    parser.add_argument("--start-anchor-episode")
    parser.add_argument("--start-fraction", type=float, default=0.0)
    parser.add_argument("--start-gap-m", type=float)
    parser.add_argument("--snapshot-dir", type=Path,
                        help="new directory for two achieved-state H=2 snapshots")
    parser.add_argument("--diagnostic-time-scale-cap", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite an existing audit")
    result = audit(
        registration_path=args.registration,
        visual_domain_path=args.visual_domain,
        real_config_path=args.real_config,
        episodes=args.episodes,
        waypoints=args.waypoints,
        source_period_s=args.source_period_s,
        profile=args.profile,
        execution_margin=args.execution_margin,
        diagnostic_time_scale_cap=args.diagnostic_time_scale_cap,
        start_anchor_episode=args.start_anchor_episode,
        start_fraction=args.start_fraction,
        start_gap_m=args.start_gap_m,
        handoff_waypoint=args.handoff_waypoint,
        snapshot_dir=args.snapshot_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"IK {result['ik_pass']}/{len(result['episodes'])}; "
          f"diagnostic time cap {result['diagnostic_cap_pass']}/"
          f"{len(result['episodes'])}; first-view object "
          f"{result['first_view_object_visible']}/{len(result['episodes'])}; "
          f"achieved motion {result['diagnostic_motion_pass']}/"
          f"{len(result['episodes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
