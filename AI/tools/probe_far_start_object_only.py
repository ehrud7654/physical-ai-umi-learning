"""Paired object-only image probe at a policy-free far-start H=2 handoff.

The robot's achieved qpos/qvel, camera, proprioception, and checkpoint stay
fixed. Only the object's XY position changes before each of the two images is
rendered. This measures first-four-target visual localisation, not grasp
success or hardware readiness. Learned inference is local CPU only.
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

from paths import DEFAULT_CONFIG
from policy.relative_chunk_bc import RelativeChunkBCPolicy
from sim.mujoco.build_scene import load_config, sync_gripper_collision_proxy
from sim.mujoco.env import MujocoPickEnv
from tools.build_sim_relative_localization_pilot import _bbox_xywh
from tools.probe_fixed_pose_object_only import _mean_fill
from tools.render_relative_chunk_rollout import _audit_file_path, _registration
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp, gap_from_angle
from umi.relative_dataset import relative_vector
from umi.relative_robot_preflight import matrix_from_fk
from umi.visual_domain import apply_visual_domain, load_visual_domain


OFFSETS = ((0.0, 0.0), (0.01, 0.0), (-0.01, 0.0),
           (0.0, 0.01), (0.0, -0.01))
AI_ROOT = Path(__file__).resolve().parents[1]


def _first_four_response(change: np.ndarray,
                         expected_local: np.ndarray) -> dict:
    """Check whether the executed prefix moves with the shifted object."""
    if change.shape != (8, 10) or expected_local.shape != (3,):
        raise ValueError("expected (8,10) prediction change and (3,) object shift")
    shifts = change[:4, :3]
    end = shifts[-1]
    expected_sq = float(np.dot(expected_local, expected_local))
    if expected_sq <= 0:
        raise ValueError("object shift must be nonzero")
    return {
        "first_translation_m": shifts[0].tolist(),
        "executed_prefix_last_translation_m": end.tolist(),
        "executed_prefix_last_gain": float(np.dot(end, expected_local) / expected_sq),
        "executed_prefix_last_direction_positive": bool(
            np.dot(end, expected_local) > 0),
        "executed_prefix_mean_change_l2_m": float(
            np.linalg.norm(shifts, axis=1).mean()),
    }


def _render_views(env: MujocoPickEnv, snapshot: dict,
                  offsets: tuple[tuple[float, float], ...]) -> list[dict]:
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                  "cam_wrist")
    if camera_id < 0:
        raise ValueError("cam_wrist absent")
    original_qpos = snapshot["qpos"]
    views = []
    baseline_camera = baseline_object = None
    for dx, dy in offsets:
        images, bboxes, contacts = [], [], []
        max_camera_error = max_robot_error = max_object_error = 0.0
        for frame in range(2):
            env.data.qpos[:] = original_qpos[frame]
            env.data.qpos[env._obj_qadr:env._obj_qadr + 2] += (dx, dy)
            env.data.qvel[:] = snapshot["qvel"][frame]
            env.data.time = float(snapshot["timestamp"][frame])
            env.data.ctrl[:6] = original_qpos[frame, :6]
            sync_gripper_collision_proxy(env.model, env.data, env.cfg)
            mujoco.mj_forward(env.model, env.data)
            image = env._observe().images["cam_wrist"].copy()
            images.append(image)
            bboxes.append(_bbox_xywh(image))
            contacts.append(int(env._n_jaw_contacts()))
            camera = np.r_[env.data.cam_xpos[camera_id],
                           env.data.cam_xmat[camera_id].ravel()]
            object_xyz = env.object_position()
            if baseline_camera is None:
                baseline_camera = [None, None]
                baseline_object = [None, None]
            if baseline_camera[frame] is None:
                baseline_camera[frame] = camera.copy()
                baseline_object[frame] = object_xyz.copy()
            max_camera_error = max(max_camera_error, float(np.max(np.abs(
                camera - baseline_camera[frame]))))
            max_robot_error = max(max_robot_error, float(np.max(np.abs(
                env.data.qpos[:6] - original_qpos[frame, :6]))))
            max_object_error = max(max_object_error, float(np.linalg.norm(
                object_xyz - baseline_object[frame] - np.array([dx, dy, 0.0]))))
        views.append({
            "offset_xy_m": [dx, dy],
            "images": np.stack(images),
            "object_bbox_xywh": bboxes,
            "jaw_object_contacts": contacts,
            "max_camera_pose_difference": max_camera_error,
            "max_robot_qpos_difference_rad": max_robot_error,
            "max_object_shift_error_m": max_object_error,
        })
    return views


def _predict(policy: RelativeChunkBCPolicy, images: np.ndarray,
             proprio: np.ndarray) -> np.ndarray:
    action = np.asarray(policy.predict_action({
        "image": images, "proprio": proprio})["action_pred"], dtype=float)
    if action.shape != (8, 10) or not np.isfinite(action).all():
        raise ValueError(f"invalid relative policy output {action.shape}")
    return action


def run(audit_path: Path, checkpoint_path: Path, *,
        max_episodes: int | None = None, noise_gray: float = 96.0,
        seed: int = 0) -> dict:
    if is_shared_gpu_server():
        raise RuntimeError("learned-policy inference is forbidden on the shared GPU server")
    if (not np.isfinite(noise_gray) or noise_gray < 0
            or max_episodes is not None and max_episodes < 1):
        raise ValueError("invalid noise or episode count")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "POLICY_FREE_FIXED_START_TO_APPROACH_IK_TIMING_AUDIT":
        raise ValueError("expected policy-free fixed-start audit")
    if audit.get("diagnostic_motion_pass") != len(audit.get("rows", [])):
        raise ValueError("all selected audit rows must pass achieved motion")
    rows = audit["rows"][:max_episodes]
    registration = _registration(_audit_file_path(audit["registration"]))
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
    cfg = apply_visual_domain(cfg, load_visual_domain(
        _audit_file_path(audit["visual_domain"])))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    cfg["task"]["object"]["half_size_m"] = (
        registration["object_size_m"] / 2).tolist()
    cfg["task"]["object"]["init_pos"] = registration["object_xyz_m"].tolist()
    cfg["task"]["table"]["half_size_m"][:2] = (
        registration["table_half_size_xy_m"].tolist())
    curve = cfg["grasp"]["gap_curve"]
    policy = RelativeChunkBCPolicy(checkpoint_path, device="cpu")
    held_out = set(policy.meta["val_episodes"])
    if any(row["episode"] not in held_out for row in rows):
        raise ValueError("selected source episode was used in checkpoint training")
    results = []
    excluded = []
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.0) as env:
        ik = MujocoIK(env.model, cfg)
        for episode_index, row in enumerate(rows):
            snapshot_path = _audit_file_path(row["snapshot_npz"])
            with np.load(snapshot_path, allow_pickle=False) as data:
                snapshot = {key: data[key].copy() for key in
                            ("image", "qpos", "qvel", "timestamp")}
            if snapshot["image"].shape != (2, 3, 224, 224):
                raise ValueError(f"invalid H=2 snapshot: {snapshot_path}")
            qpos = snapshot["qpos"]
            env.reset(seed=0, object_xy=tuple(qpos[0, env._obj_qadr:
                                                 env._obj_qadr + 2]),
                      initial_q_rad=qpos[0, :6])
            views = _render_views(env, snapshot, OFFSETS)
            for view in views:
                view["image_mae_from_nominal_uint8"] = float(np.mean(np.abs(
                    view["images"].astype(float)
                    - views[0]["images"].astype(float))))
            if any(bbox is None for view in views
                   for bbox in view["object_bbox_xywh"]):
                excluded.append({"episode": row["episode"],
                                 "reason": "object not visible at every offset/frame"})
                continue
            if any(any(view["jaw_object_contacts"]) for view in views):
                excluded.append({
                    "episode": row["episode"],
                    "reason": "object shift creates pre-policy jaw contact",
                    "contacts_by_offset": [{
                        "offset_xy_m": view["offset_xy_m"],
                        "contacts": view["jaw_object_contacts"],
                    } for view in views],
                })
                continue
            if any(view["max_camera_pose_difference"] > 1e-12
                   or view["max_robot_qpos_difference_rad"] > 1e-12
                   or view["max_object_shift_error_m"] > 1e-9
                   for view in views):
                raise ValueError(f"object-only invariant failed: {row['episode']}")
            baseline_mae = float(np.mean(np.abs(
                views[0]["images"].astype(float) - snapshot["image"].astype(float))))
            if baseline_mae > 0:
                raise ValueError(f"nominal snapshot re-render mismatch: {row['episode']} {baseline_mae}")
            previous_pose = matrix_from_fk(env.model, ik, qpos[0, :6])
            current_pose = matrix_from_fk(env.model, ik, qpos[1, :6])
            gaps = [gap_from_angle(float(qpos[frame, 5]), curve)
                    for frame in range(2)]
            proprio = np.stack([
                relative_vector(current_pose, previous_pose, gaps[0]),
                relative_vector(current_pose, current_pose, gaps[1]),
            ]).astype(np.float32)
            rng = np.random.default_rng(seed + episode_index)
            shared_noise = rng.normal(0.0, noise_gray,
                                      size=views[0]["images"].shape)
            predictions = []
            for view in views:
                image = view["images"]
                predictions.append({
                    "normal": _predict(policy, image, proprio),
                    "mean_fill": _predict(policy, _mean_fill(image), proprio),
                    "additive_noise": _predict(policy, np.clip(
                        image.astype(float) + shared_noise, 0, 255
                    ).astype(np.uint8), proprio),
                })
            comparisons = []
            for view, prediction in zip(views[1:], predictions[1:]):
                offset_world = np.r_[view["offset_xy_m"], 0.0]
                offset_local = current_pose[:3, :3].T @ offset_world
                comparisons.append({
                    "offset_world_xy_m": view["offset_xy_m"],
                    "expected_local_xyz_m": offset_local.tolist(),
                    "conditions": {
                        condition: _first_four_response(
                            prediction[condition] - predictions[0][condition],
                            offset_local)
                        for condition in predictions[0]
                    },
                })
            results.append({
                "episode": row["episode"],
                "snapshot_npz": str(snapshot_path),
                "snapshot_rerender_mae_uint8": baseline_mae,
                "history_interval_s": float(np.diff(snapshot["timestamp"])[0]),
                "views": [{key: value for key, value in view.items()
                           if key != "images"} for view in views],
                "comparisons": comparisons,
            })
    if not results:
        raise ValueError(f"no eligible object-only paired episodes: {excluded}")
    return {
        "status": "LOCAL_FAR_START_H2_OBJECT_ONLY_FIRST_FOUR_DIAGNOSTIC",
        "audit": str(audit_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "inference_device": "cpu",
        "all_episodes_held_out_from_training": True,
        "robot_and_camera_fixed_across_object_shifts": True,
        "object_offsets_world_xy_m": [list(offset) for offset in OFFSETS],
        "noise_condition": f"inference-time additive Gaussian sigma={noise_gray} on real render, shared draw across offsets",
        "evaluated_episodes": len(results),
        "eligible_audit_episodes": len(rows),
        "excluded": excluded,
        "positive_prefix_last_direction_by_condition": {
            condition: sum(pair["conditions"][condition][
                "executed_prefix_last_direction_positive"]
                for item in results for pair in item["comparisons"])
            for condition in ("normal", "mean_fill", "additive_noise")
        },
        "median_prefix_last_gain_by_condition": {
            condition: float(np.median([
                pair["conditions"][condition]["executed_prefix_last_gain"]
                for item in results for pair in item["comparisons"]]))
            for condition in ("normal", "mean_fill", "additive_noise")
        },
        "median_image_mae_from_nominal_uint8": float(np.median([
            view["image_mae_from_nominal_uint8"]
            for item in results for view in item["views"][1:]])),
        "episodes_all_four_directions_positive_by_condition": {
            condition: sum(all(pair["conditions"][condition][
                "executed_prefix_last_direction_positive"]
                for pair in item["comparisons"]) for item in results)
            for condition in ("normal", "mean_fill", "additive_noise")
        },
        "rows": results,
        "limitations": [
            "A paired first-four-action diagnostic, not a closed-loop rollout or physical grasp.",
            "A fixed policy-free scripted warmup produced each H=2 robot snapshot; the policy did not navigate the far start.",
            "Object offsets are a geometric test signal, not recorded human commands.",
            "Mean fill is recalculated for each shifted image, so global colour/area changes remain and may drive a non-spatial response.",
            "Additive noise preserves some spatial image content and is not an information-free image.",
            "Visual alignment, 2mm simulation-only preload, contact proxy, and base registration are provisional.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--noise-gray", type=float, default=96.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = run(args.audit, args.policy_ckpt,
                 max_episodes=args.max_episodes,
                 noise_gray=args.noise_gray, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "eligible_audit_episodes", "evaluated_episodes",
        "positive_prefix_last_direction_by_condition",
        "episodes_all_four_directions_positive_by_condition",
        "median_prefix_last_gain_by_condition",
        "median_image_mae_from_nominal_uint8")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
