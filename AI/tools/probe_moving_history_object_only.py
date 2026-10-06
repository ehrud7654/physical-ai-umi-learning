"""Probe held-out v10 episodes with a moving H=2 and object-only XY shifts.

One policy-free nominal-100ms approach supplies two robot/camera states per episode.
Both states are reused exactly for every object placement. This is an offline
first-chunk diagnostic, not a closed-loop grasp or physical calibration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from paths import DEFAULT_CONFIG
from policy.relative_chunk_bc import RelativeChunkBCPolicy
from sim.mujoco.build_scene import load_config, normalize, sync_gripper_collision_proxy
from sim.mujoco.env import MujocoPickEnv
from tools.build_sim_relative_localization_pilot import _bbox_xywh
from tools.probe_fixed_pose_object_only import (
    _mean_fill,
    _response,
    _source_path,
    _suite_conditions,
)
from tools.render_relative_chunk_rollout import (
    _diagnostic_object_xyz,
    _episode_approach_start,
    _registration,
)
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp, gap_from_angle
from umi.convert import invert_gap_curve
from umi.ik import matrix_to_quat
from umi.relative_dataset import relative_vector
from umi.relative_robot_preflight import matrix_from_fk, real_limits
from umi.visual_domain import apply_visual_domain, load_visual_domain


BACKOFF_M = 0.03
HISTORY_TARGET_STEP_M = 0.005
OPEN_GAP_M = 0.065
HISTORY_SECONDS = 0.1


def _command_fraction(profile: str, elapsed_s: float,
                      motion_duration_s: float) -> float:
    """Return a causal command fraction without looking at policy outputs."""
    if profile == "step":
        return 1.0
    if profile != "quintic":
        raise ValueError(f"unknown history command profile: {profile}")
    if (not np.isfinite(elapsed_s) or not np.isfinite(motion_duration_s)
            or elapsed_s < 0 or motion_duration_s < HISTORY_SECONDS):
        raise ValueError("invalid quintic motion duration")
    u = float(np.clip(elapsed_s / motion_duration_s, 0.0, 1.0))
    return 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5


def _valid_ik(solution: object) -> bool:
    return bool(solution.converged and solution.within_limits
                and solution.pos_error_m <= 0.005
                and solution.axis_error_deg <= 5.0
                and solution.roll_residual_deg <= 5.0)


def _history_target(ik: MujocoIK, model: mujoco.MjModel,
                    anchor_q: np.ndarray, approach: np.ndarray, *,
                    backoff_m: float = BACKOFF_M,
                    history_target_step_m: float = HISTORY_TARGET_STEP_M,
                    open_gap_m: float = OPEN_GAP_M,
                    require_valid: bool = True,
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve two predeclared backoff poses; do not search after seeing results."""
    anchor = matrix_from_fk(model, ik, anchor_q)
    rotation = anchor[:3, :3]
    quat = matrix_to_quat(rotation)
    previous_xyz = anchor[:3, 3] - approach * (backoff_m + history_target_step_m)
    current_xyz = anchor[:3, 3] - approach * backoff_m
    current = ik.solve(current_xyz, quat, open_gap_m, q_init=anchor_q)
    previous = ik.solve(previous_xyz, quat, open_gap_m, q_init=current.q_rad)
    details = {
        "backoff_m": backoff_m,
        "history_target_step_m": history_target_step_m,
        "open_gap_m": open_gap_m,
        "previous_ik_error_m": float(previous.pos_error_m),
        "current_ik_error_m": float(current.pos_error_m),
        "previous_axis_error_deg": float(previous.axis_error_deg),
        "current_axis_error_deg": float(current.axis_error_deg),
        "previous_roll_error_deg": float(previous.roll_residual_deg),
        "current_roll_error_deg": float(current.roll_residual_deg),
        "previous_within_limits": bool(previous.within_limits),
        "current_within_limits": bool(current.within_limits),
        "previous_ik_valid": _valid_ik(previous),
        "current_ik_valid": _valid_ik(current),
    }
    if require_valid and not (_valid_ik(previous) and _valid_ik(current)):
        raise ValueError(f"predeclared backoff IK failed: {details}")
    return previous.q_rad.copy(), current.q_rad.copy(), details


def _simulate_history(env: MujocoPickEnv, *, previous_q: np.ndarray,
                      current_target_q: np.ndarray, object_xy: np.ndarray,
                      speed_limit_rad_s: float,
                      command_profile: str = "step",
                      motion_duration_s: float = HISTORY_SECONDS,
                      ) -> tuple[list[dict], dict]:
    """Run one policy-free approach, then preserve its two exact snapshots."""
    env.reset(seed=0, object_xy=tuple(object_xy), initial_q_rad=previous_q)
    previous = {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "time": float(env.data.time),
        "object_xyz": env.object_position(),
    }
    pad_contact_ticks = int(env._n_jaw_contacts() > 0)
    max_arm_speed_rad_s = 0.0
    max_arm_accel_rad_s2 = 0.0
    last_velocity = np.asarray(env.data.qvel[:5], dtype=float).copy()
    control_ticks = int(round(HISTORY_SECONDS * env.control_rate_hz))
    if control_ticks < 1 or not np.isclose(
            control_ticks / env.control_rate_hz, HISTORY_SECONDS, atol=1e-9):
        raise ValueError("control rate cannot represent the 100ms history exactly")
    command_fractions = []
    for tick in range(control_ticks):
        fraction = _command_fraction(
            command_profile, (tick + 1) / env.control_rate_hz,
            motion_duration_s)
        command_fractions.append(fraction)
        target_q = previous_q + fraction * (current_target_q - previous_q)
        target = normalize(target_q, env.cfg, clip=True)
        last_q = env.joint_positions()
        last_time = float(env.data.time)
        env.step(target)
        elapsed = float(env.data.time) - last_time
        if elapsed <= 0:
            raise ValueError("MuJoCo control tick did not advance time")
        velocity = (env.joint_positions()[:5] - last_q[:5]) / elapsed
        max_arm_speed_rad_s = max(max_arm_speed_rad_s, float(np.max(np.abs(
            velocity))))
        max_arm_accel_rad_s2 = max(max_arm_accel_rad_s2, float(np.max(np.abs(
            velocity - last_velocity)) / elapsed))
        last_velocity = velocity
        pad_contact_ticks += int(env._n_jaw_contacts() > 0)
    current = {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "time": float(env.data.time),
        "object_xyz": env.object_position(),
    }
    object_drift = float(np.linalg.norm(
        current["object_xyz"][:2] - previous["object_xyz"][:2]))
    if object_drift > 0.001:
        raise ValueError(f"object moved {object_drift:.4f}m during approach")
    robot_motion = float(np.max(np.abs(
        current["qpos"][:6] - previous["qpos"][:6])))
    if robot_motion < 1e-4:
        raise ValueError("100ms approach did not move the robot measurably")
    actual_interval = current["time"] - previous["time"]
    if abs(actual_interval - HISTORY_SECONDS) > 0.01:
        raise ValueError(f"history interval {actual_interval:.4f}s exceeds 10ms skew")
    if pad_contact_ticks:
        raise ValueError(f"moving history touched object on {pad_contact_ticks} ticks")
    if max_arm_speed_rad_s > speed_limit_rad_s + 1e-9:
        raise ValueError(
            f"moving history peak speed {max_arm_speed_rad_s:.3f}rad/s exceeds "
            f"configured {speed_limit_rad_s:.3f}rad/s")
    return [previous, current], {
        "history_interval_s": actual_interval,
        "control_ticks": control_ticks,
        "max_joint_motion_rad": robot_motion,
        "max_arm_speed_rad_s": max_arm_speed_rad_s,
        "finite_difference_peak_accel_rad_s2": max_arm_accel_rad_s2,
        "command_profile": command_profile,
        "command_motion_duration_s": motion_duration_s,
        "command_fraction_at_current": command_fractions[-1],
        "pad_contact_ticks": pad_contact_ticks,
        "object_xy_drift_m": object_drift,
    }


def _render_history(env: MujocoPickEnv, snapshots: list[dict],
                    offsets: list[tuple[float, float]]) -> list[dict]:
    camera_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    if camera_id < 0:
        raise ValueError("cam_wrist is absent")
    views = []
    baseline_cameras = []
    baseline_objects = []
    for offset in offsets:
        images = []
        bboxes = []
        camera_error = 0.0
        robot_error = 0.0
        object_error = 0.0
        for step, snapshot in enumerate(snapshots):
            env.data.qpos[:] = snapshot["qpos"]
            env.data.qpos[env._obj_qadr:env._obj_qadr + 2] += offset
            env.data.qvel[:] = snapshot["qvel"]
            env.data.ctrl[:6] = snapshot["qpos"][:6]
            sync_gripper_collision_proxy(env.model, env.data, env.cfg)
            mujoco.mj_forward(env.model, env.data)
            image = env._observe().images["cam_wrist"].copy()
            images.append(image)
            bboxes.append(_bbox_xywh(image))
            camera = np.r_[env.data.cam_xpos[camera_id],
                           env.data.cam_xmat[camera_id].ravel()].copy()
            if len(baseline_cameras) <= step:
                baseline_cameras.append(camera)
                baseline_objects.append(env.object_position())
            camera_error = max(camera_error, float(np.max(np.abs(
                camera - baseline_cameras[step]))))
            robot_error = max(robot_error, float(np.max(np.abs(
                env.data.qpos[:6] - snapshot["qpos"][:6]))))
            object_error = max(object_error, float(np.linalg.norm(
                env.object_position() - baseline_objects[step] - np.r_[offset, 0.0])))
        views.append({
            "offset": offset,
            "images": np.stack(images),
            "bboxes": bboxes,
            "max_camera_pose_difference": camera_error,
            "max_robot_qpos_difference_rad": robot_error,
            "max_object_shift_error_m": object_error,
        })
    return views


def _predict(policy: RelativeChunkBCPolicy, images: np.ndarray,
             proprio: np.ndarray) -> np.ndarray:
    prediction = policy.predict_action({"image": images,
                                        "proprio": proprio})["action_pred"]
    if prediction.shape != (8, 10):
        raise ValueError(f"unexpected action shape {prediction.shape}")
    return prediction


def _montage(rows: list[dict], offsets: list[tuple[float, float]]) -> Image.Image:
    width, height = 224, 244
    sheet = Image.new("RGB", (width * len(offsets), height * 2 * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        for step in range(2):
            top = (row_index * 2 + step) * height
            for column, view in enumerate(row["views"]):
                image = Image.fromarray(np.transpose(
                    view["images"][step], (1, 2, 0)))
                sheet.paste(image, (column * width, top + 20))
                x, y = view["offset"]
                draw.text((column * width + 3, top + 3),
                          f"{row['episode'][-6:]} t{step} x={x * 1000:+.0f} y={y * 1000:+.0f}",
                          fill="black")
    return sheet


def evaluate(suite_path: Path, checkpoint: Path,
             max_episodes: int | None = None, *,
             backoff_m: float = BACKOFF_M,
             history_target_step_m: float = HISTORY_TARGET_STEP_M,
             open_gap_m: float = OPEN_GAP_M,
             command_profile: str = "step",
             motion_duration_s: float = HISTORY_SECONDS,
             ) -> tuple[dict, Image.Image]:
    if is_shared_gpu_server():
        raise RuntimeError("learned-policy inference is forbidden on the shared GPU server")
    conditions, episodes, offsets = _suite_conditions(suite_path)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError("suite has no complete source episode")
    registration = _registration(_source_path(conditions["registration"]))
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
    cfg = apply_visual_domain(
        cfg, load_visual_domain(_source_path(conditions["visual_domain"])))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    cfg["task"]["object"]["half_size_m"] = (
        registration["object_size_m"] / 2).tolist()
    cfg["task"]["object"]["init_pos"] = registration["object_xyz_m"].tolist()
    cfg["task"]["table"]["half_size_m"][:2] = (
        registration["table_half_size_xy_m"].tolist())
    curve = cfg["grasp"]["gap_curve"]
    real = real_limits(Path(__file__).resolve().parents[1]
                       / "configs/real/so101_ver1.json")
    speed_limit = float(real["max_speed_rad_s"])
    accel_limit = float(real["max_accel_rad_s2"])
    approach = np.asarray(registration["desired_world_approach_axis"], dtype=float)
    approach /= np.linalg.norm(approach)
    policy = RelativeChunkBCPolicy(checkpoint, device="cpu")
    held_out = set(policy.meta["val_episodes"])
    if any(episode not in held_out for episode in episodes):
        raise ValueError("a selected source episode was used in checkpoint training")

    rows = []
    excluded = []
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.0) as env:
        ik = MujocoIK(env.model, cfg)
        for episode in episodes:
            try:
                source_row, anchor_arm, anchor_gap = _episode_approach_start(
                    registration, episode)
                _, object_xyz = _diagnostic_object_xyz(
                    registration, episode, np.zeros(2))
                anchor_q = np.r_[anchor_arm, invert_gap_curve(anchor_gap, curve)]
                previous_q, current_target_q, ik_details = _history_target(
                    ik, env.model, anchor_q, approach,
                    backoff_m=backoff_m,
                    history_target_step_m=history_target_step_m,
                    open_gap_m=open_gap_m)
                snapshots, motion = _simulate_history(
                    env, previous_q=previous_q,
                    current_target_q=current_target_q,
                    object_xy=object_xyz[:2], speed_limit_rad_s=speed_limit,
                    command_profile=command_profile,
                    motion_duration_s=motion_duration_s)
                views = _render_history(env, snapshots, offsets)
                if any(None in view["bboxes"] for view in views):
                    raise ValueError("object not visible in both frames at every offset")
                if any(view["max_camera_pose_difference"] > 1e-12
                       or view["max_robot_qpos_difference_rad"] > 1e-12
                       or view["max_object_shift_error_m"] > 1e-9
                       for view in views):
                    raise ValueError("object-only paired-frame invariant failed")

                previous_pose = matrix_from_fk(
                    env.model, ik, snapshots[0]["qpos"][:6])
                current_pose = matrix_from_fk(
                    env.model, ik, snapshots[1]["qpos"][:6])
                previous_gap = gap_from_angle(float(snapshots[0]["qpos"][5]), curve)
                current_gap = gap_from_angle(float(snapshots[1]["qpos"][5]), curve)
                proprio = np.stack([
                    relative_vector(current_pose, previous_pose, previous_gap),
                    relative_vector(current_pose, current_pose, current_gap),
                ]).astype(np.float32)
                baseline_images = views[0]["images"]
                base_prediction = _predict(policy, baseline_images, proprio)
                base_mean = _predict(policy, _mean_fill(baseline_images), proprio)
                comparisons = []
                for view in views[1:]:
                    offset_world = np.r_[view["offset"], 0.0]
                    offset_local = current_pose[:3, :3].T @ offset_world
                    moved_prediction = _predict(policy, view["images"], proprio)
                    moved_mean = _predict(policy, _mean_fill(view["images"]), proprio)
                    comparisons.append({
                        "offset_world_xy_m": list(view["offset"]),
                        "expected_local_xyz_m": offset_local.tolist(),
                        "normal": _response(moved_prediction - base_prediction,
                                            offset_local),
                        "mean_fill_per_view": _response(moved_mean - base_mean,
                                                        offset_local),
                    })
                motion["tcp_translation_m"] = float(np.linalg.norm(
                    current_pose[:3, 3] - previous_pose[:3, 3]))
                motion["configured_accel_limit_rad_s2"] = accel_limit
                motion["acceleration_limit_exceeded"] = bool(
                    motion["finite_difference_peak_accel_rad_s2"] > accel_limit)
                rows.append({"episode": episode, "source_row": source_row,
                             "ik": ik_details, "motion": motion,
                             "views": views, "comparisons": comparisons})
            except (ValueError, OSError, KeyError) as exc:
                excluded.append({"episode": episode, "reason": str(exc).splitlines()[0]})
    if not rows:
        raise ValueError(f"no feasible moving-history episode; excluded={excluded}")

    def episode_median(condition: str, metric: str) -> float:
        return float(np.median([
            np.median([pair[condition][metric] for pair in row["comparisons"]])
            for row in rows
        ]))

    report = {
        "status": "LOCAL_HELDOUT_MOVING_H2_OBJECT_ONLY_DIAGNOSTIC",
        "suite": str(suite_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "inference_device": "cpu",
        "all_included_episodes_held_out_from_training": True,
        "backoff_m": backoff_m,
        "history_target_step_m": history_target_step_m,
        "history_interval_s": HISTORY_SECONDS,
        "open_gap_m": open_gap_m,
        "history_command_profile": command_profile,
        "history_motion_duration_s": motion_duration_s,
        "configured_speed_limit_rad_s": speed_limit,
        "configured_accel_limit_rad_s2": accel_limit,
        "eligible_source_episodes": len(episodes),
        "evaluated_source_episodes": len(rows),
        "excluded": excluded,
        "paired_offsets": 4 * len(rows),
        "episodes_exceeding_configured_accel_limit": sum(
            row["motion"]["acceleration_limit_exceeded"] for row in rows),
        "execution_feasible_under_config": all(
            not row["motion"]["acceleration_limit_exceeded"] for row in rows),
        "normal_positive_terminal_direction": sum(
            pair["normal"]["terminal_direction_positive"]
            for row in rows for pair in row["comparisons"]),
        "mean_fill_positive_terminal_direction": sum(
            pair["mean_fill_per_view"]["terminal_direction_positive"]
            for row in rows for pair in row["comparisons"]),
        "episodes_with_all_four_normal_directions_positive": sum(
            all(pair["normal"]["terminal_direction_positive"]
                for pair in row["comparisons"]) for row in rows),
        "episode_median_of_offset_medians": {
            condition: {metric: episode_median(condition, metric)
                        for metric in ("terminal_cosine", "terminal_gain",
                                       "chunk_translation_change_l2_mean_m")}
            for condition in ("normal", "mean_fill_per_view")
        },
        "per_episode": [{
            "episode": row["episode"], "source_row": row["source_row"],
            "ik": row["ik"], "motion": row["motion"],
            "views": [{key: value for key, value in view.items()
                       if key != "images"} for view in row["views"]],
            "comparisons": row["comparisons"],
        } for row in rows],
        "training_ready": False,
        "limitations": [
            "Backoff and gap values are explicit diagnostic inputs, not a recorded human start.",
            "Only the baseline placement undergoes nominal-100ms policy-free motion; actual MuJoCo interval is recorded and exact robot snapshots are reused for shifted-object renders without further physics.",
            "The geometric object-shift direction is not a measured human command or grasp-success label.",
            "One unchanged v10 checkpoint and its validation episodes are tested; no closed-loop success is scored.",
            "Finite-difference acceleration is reported against the provisional config; exceeding it makes the history non-executable as-is.",
            "Camera, contact, 2mm-preload registration and robot-base alignment remain provisional.",
        ],
    }
    return report, _montage(rows, offsets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--backoff-m", type=float, default=BACKOFF_M)
    parser.add_argument("--history-target-step-m", type=float,
                        default=HISTORY_TARGET_STEP_M)
    parser.add_argument("--open-gap-m", type=float, default=OPEN_GAP_M)
    parser.add_argument("--history-command-profile", choices=("step", "quintic"),
                        default="step")
    parser.add_argument("--motion-duration-s", type=float, default=HISTORY_SECONDS)
    args = parser.parse_args()
    if args.out.exists() or args.montage.exists():
        raise SystemExit("refusing to overwrite an existing report or montage")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise SystemExit("--max-episodes must be positive")
    if (not 0 <= args.backoff_m <= 0.05
            or not 0 < args.history_target_step_m <= 0.01
            or not 0.03 <= args.open_gap_m <= 0.09
            or not np.isfinite(args.motion_duration_s)
            or args.motion_duration_s < HISTORY_SECONDS):
        raise SystemExit("backoff/gap inputs are outside diagnostic bounds")
    report, montage = evaluate(
        args.suite, args.policy_ckpt, max_episodes=args.max_episodes,
        backoff_m=args.backoff_m,
        history_target_step_m=args.history_target_step_m,
        open_gap_m=args.open_gap_m,
        command_profile=args.history_command_profile,
        motion_duration_s=args.motion_duration_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.montage.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.montage)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "eligible_source_episodes", "evaluated_source_episodes",
        "excluded", "paired_offsets", "normal_positive_terminal_direction",
        "episodes_exceeding_configured_accel_limit",
        "mean_fill_positive_terminal_direction",
        "episodes_with_all_four_normal_directions_positive",
        "episode_median_of_offset_medians")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
