"""Test v10 visual localisation with a fixed robot/camera and shifted object.

This is a local, inference-time diagnostic. It does not generate training data,
score a grasp, or establish physical camera/robot calibration.
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

from paths import AI_ROOT, DEFAULT_CONFIG
from policy.relative_chunk_bc import RelativeChunkBCPolicy
from sim.mujoco.build_scene import load_config, sync_gripper_collision_proxy
from sim.mujoco.env import MujocoPickEnv
from tools.build_sim_relative_localization_pilot import _bbox_xywh
from tools.render_relative_chunk_rollout import (
    _diagnostic_object_xyz,
    _episode_approach_start,
    _registration,
)
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp, gap_from_angle
from umi.convert import invert_gap_curve
from umi.relative_dataset import relative_vector
from umi.relative_robot_preflight import matrix_from_fk
from umi.visual_domain import apply_visual_domain, load_visual_domain


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else AI_ROOT.parent / path


def _suite_conditions(path: Path) -> tuple[dict, list[str], list[tuple[float, float]]]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if (suite.get("status") != "COMPLETE_WITH_REJECTIONS_POLICY_FREE_10HZ_DIAGNOSTIC"
            or suite.get("training_ready") is not False):
        raise ValueError("expected the completed, non-training-ready 10Hz suite")
    conditions = suite["conditions"]
    offsets = [tuple(float(item) for item in row)
               for row in conditions["offsets_xy_m"]]
    if (len(offsets) != 5 or len(set(offsets)) != 5
            or offsets[0] != (0.0, 0.0)
            or not all(np.isfinite(row).all() and max(map(abs, row)) <= 0.01
                       for row in offsets)):
        raise ValueError("expected baseline plus four unique offsets within 10mm")
    by_episode: dict[str, dict[tuple[float, float], str]] = {}
    for case in suite["cases"]:
        episode = str(case["episode"])
        offset = tuple(float(item) for item in case["offset_xy_m"])
        if offset in by_episode.setdefault(episode, {}):
            raise ValueError(f"duplicate suite condition: {episode} {offset}")
        by_episode[episode][offset] = str(case["status"])
    episodes = []
    for episode in conditions["episodes"]:
        if set(by_episode.get(episode, {})) != set(offsets):
            raise ValueError(f"missing suite placement for {episode}")
        if all(by_episode[episode][offset] == "CANDIDATE_CHECKED"
               for offset in offsets):
            episodes.append(episode)
    return conditions, episodes, offsets


def _mean_fill(image: np.ndarray) -> np.ndarray:
    """Remove spatial information but retain each frame's channel means."""
    source = np.asarray(image, dtype=np.uint8)
    if source.shape != (2, 3, 224, 224):
        raise ValueError("expected two 224x224 RGB observations")
    values = np.rint(source.mean(axis=(-2, -1), keepdims=True)).astype(np.uint8)
    return np.broadcast_to(values, source.shape).copy()


def _response(predicted: np.ndarray, expected_xyz: np.ndarray) -> dict[str, object]:
    """Compare the change in an 8-step target with a local object translation."""
    prediction = np.asarray(predicted, dtype=float)
    expected = np.asarray(expected_xyz, dtype=float)
    if (prediction.shape != (8, 10) or expected.shape != (3,)
            or not np.isfinite(prediction).all() or not np.isfinite(expected).all()
            or np.linalg.norm(expected) < 1e-9):
        raise ValueError("expected finite (8,10) action change and nonzero XYZ shift")
    translated = prediction[:, :3]
    expected_norm_sq = float(np.dot(expected, expected))
    terminal = translated[-1]
    terminal_norm = float(np.linalg.norm(terminal))
    return {
        "first_translation_m": translated[0].tolist(),
        "terminal_translation_m": terminal.tolist(),
        "first_gain": float(np.dot(translated[0], expected) / expected_norm_sq),
        "terminal_gain": float(np.dot(terminal, expected) / expected_norm_sq),
        "terminal_cosine": (float(np.dot(terminal, expected)
                                  / (terminal_norm * np.sqrt(expected_norm_sq)))
                            if terminal_norm > 1e-9 else 0.0),
        "terminal_direction_positive": bool(np.dot(terminal, expected) > 0),
        "chunk_translation_change_l2_mean_m": float(
            np.linalg.norm(translated, axis=1).mean()),
        "chunk_expected_error_l2_mean_m": float(
            np.linalg.norm(translated - expected, axis=1).mean()),
    }


def _predict(policy: RelativeChunkBCPolicy, image: np.ndarray,
             proprio: np.ndarray) -> np.ndarray:
    prediction = policy.predict_action({"image": np.stack([image, image]),
                                        "proprio": proprio})["action_pred"]
    if prediction.shape != (8, 10):
        raise ValueError(f"unexpected policy action shape {prediction.shape}")
    return prediction


def _render_fixed_views(env: MujocoPickEnv, *, base_qpos: np.ndarray,
                        offsets: list[tuple[float, float]]) -> list[dict]:
    """Use one settled snapshot; change only the object's free-joint XY."""
    camera_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    if camera_id < 0:
        raise ValueError("cam_wrist is absent")
    image_rows = []
    base_camera = None
    for x, y in offsets:
        env.data.qpos[:] = base_qpos
        env.data.qpos[env._obj_qadr:env._obj_qadr + 2] += (x, y)
        env.data.qvel[:] = 0.0
        env.data.ctrl[:6] = base_qpos[:6]
        sync_gripper_collision_proxy(env.model, env.data, env.cfg)
        mujoco.mj_forward(env.model, env.data)
        observed = env._observe()
        camera = np.r_[env.data.cam_xpos[camera_id],
                       env.data.cam_xmat[camera_id].ravel()].copy()
        if base_camera is None:
            base_camera = camera
        image = observed.images["cam_wrist"].copy()
        image_rows.append({
            "offset": (x, y),
            "image": image,
            "bbox": _bbox_xywh(image),
            "object_xyz_m": env.object_position().tolist(),
            "max_robot_qpos_change_rad": float(np.max(np.abs(
                env.data.qpos[:6] - base_qpos[:6]))),
            "max_camera_pose_change": float(np.max(np.abs(camera - base_camera))),
        })
    reference = np.asarray(image_rows[0]["object_xyz_m"])
    for row in image_rows:
        target = reference + np.r_[row["offset"], 0.0]
        row["object_shift_error_m"] = float(np.linalg.norm(
            np.asarray(row["object_xyz_m"]) - target))
    return image_rows


def _montage(rows: list[dict], offsets: list[tuple[float, float]]) -> Image.Image:
    width, height = 224, 244
    sheet = Image.new("RGB", (width * len(offsets), height * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        for column, view in enumerate(row["views"]):
            image = Image.fromarray(np.transpose(view["image"], (1, 2, 0)))
            sheet.paste(image, (column * width, row_index * height + 20))
            x, y = view["offset"]
            draw.text((column * width + 3, row_index * height + 3),
                      f"{row['episode'][-6:]}  x={x * 1000:+.0f} y={y * 1000:+.0f}mm",
                      fill="black")
    return sheet


def evaluate(suite_path: Path, checkpoint: Path, *, max_episodes: int | None = None
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
    policy = RelativeChunkBCPolicy(checkpoint, device="cpu")
    rows = []
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.0) as env:
        ik = MujocoIK(env.model, cfg)
        for episode in episodes:
            source_row, arm, gap = _episode_approach_start(registration, episode)
            _, object_xyz = _diagnostic_object_xyz(
                registration, episode, np.zeros(2))
            start_q = np.r_[arm, invert_gap_curve(gap, curve)]
            env.reset(seed=0, object_xy=tuple(object_xyz[:2]),
                      initial_q_rad=start_q)
            base_qpos = env.data.qpos.copy()
            start_pose = matrix_from_fk(env.model, ik, base_qpos[:6])
            actual_gap = gap_from_angle(float(base_qpos[5]), curve)
            static_state = relative_vector(start_pose, start_pose, actual_gap)
            proprio = np.stack([static_state, static_state]).astype(np.float32)
            views = _render_fixed_views(env, base_qpos=base_qpos, offsets=offsets)
            if any(view["bbox"] is None for view in views):
                raise ValueError(f"{episode}: object not visible in every fixed-pose view")
            if any(view["max_robot_qpos_change_rad"] > 1e-12
                   or view["max_camera_pose_change"] > 1e-12
                   or view["object_shift_error_m"] > 1e-9 for view in views):
                raise ValueError(f"{episode}: fixed-pose/object-only invariant failed")
            baseline_image = views[0]["image"]
            baseline_prediction = _predict(policy, baseline_image, proprio)
            baseline_mean_prediction = _predict(
                policy, _mean_fill(np.stack([baseline_image, baseline_image]))[0],
                proprio)
            comparisons = []
            for view in views[1:]:
                image = view["image"]
                displacement_world = np.r_[view["offset"], 0.0]
                displacement_local = start_pose[:3, :3].T @ displacement_world
                normal = _predict(policy, image, proprio)
                mean_fill = _predict(
                    policy, _mean_fill(np.stack([image, image]))[0], proprio)
                comparisons.append({
                    "offset_world_xy_m": list(view["offset"]),
                    "expected_local_xyz_m": displacement_local.tolist(),
                    "rgb_difference_mean_255": float(np.abs(
                        image.astype(np.int16)
                        - baseline_image.astype(np.int16)).mean()),
                    "normal": _response(normal - baseline_prediction,
                                        displacement_local),
                    "mean_fill_per_view": _response(
                        mean_fill - baseline_mean_prediction,
                        displacement_local),
                })
            rows.append({"episode": episode, "source_row": source_row,
                         "views": views, "comparisons": comparisons})

    report_rows = [{
        "episode": row["episode"],
        "source_row": row["source_row"],
        "views": [{key: value for key, value in view.items() if key != "image"}
                  for view in row["views"]],
        "comparisons": row["comparisons"],
    } for row in rows]
    def episode_median(condition: str, metric: str) -> float:
        return float(np.median([
            np.median([pair[condition][metric] for pair in row["comparisons"]])
            for row in rows
        ]))

    report = {
        "status": "LOCAL_FIXED_ROBOT_POSE_OBJECT_ONLY_IMAGE_DIAGNOSTIC",
        "suite": str(suite_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "inference_device": "cpu",
        "source_episodes": len(rows),
        "paired_offsets": sum(len(row["comparisons"]) for row in rows),
        "robot_and_camera_pose_identical": True,
        "training_ready": False,
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
            condition: {
                metric: episode_median(condition, metric)
                for metric in ("terminal_cosine", "terminal_gain",
                               "chunk_translation_change_l2_mean_m",
                               "chunk_expected_error_l2_mean_m")
            }
            for condition in ("normal", "mean_fill_per_view")
        },
        "per_episode": report_rows,
        "limitations": [
            "Static duplicate H=2 observations are outside moving demonstration histories.",
            "Expected direction is a geometric object shift, not a measured human command or achieved rollout future.",
            "Only first 8-step policy output is tested; no contact, lift, or closed-loop success is scored.",
            "The K=4 near-contact registration, camera proxy and collision proxy are provisional.",
            "The five placements of one source episode are paired, not independent demonstrations.",
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
    args = parser.parse_args()
    if args.out.exists() or args.montage.exists():
        raise SystemExit("refusing to overwrite an existing report or montage")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise SystemExit("--max-episodes must be positive")
    report, montage = evaluate(
        args.suite, args.policy_ckpt, max_episodes=args.max_episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.montage.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.montage)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "source_episodes", "paired_offsets",
        "normal_positive_terminal_direction",
        "mean_fill_positive_terminal_direction",
        "episodes_with_all_four_normal_directions_positive",
        "episode_median_of_offset_medians")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
