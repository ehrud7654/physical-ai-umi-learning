"""Locally refine the provisional wrist camera against actual MuJoCo RGB.

The geometric projection fit is a fast initializer, but perspective faces,
occlusion and the colour-component detector make its box differ from the box
measured on a rendered image.  This policy-free tool keeps the visual/physical
scene fixed and searches a small camera neighbourhood using the rendered box
itself as the objective.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image

from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import build_model, load_config, sync_gripper_collision_proxy
from tools.estimate_umi_object_geometry import largest_object_bbox
from tools.fit_s22_mujoco_visual_alignment import (
    _bbox_norm,
    _camera_rotation,
    _loss,
    _rotation_xyz,
    _summary,
)
from tools.umi_mujoco import apply_kinematic_grasp
from umi.convert import invert_gap_curve
from umi.visual_domain import apply_visual_domain, load_visual_domain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--visual-domain", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--candidates", type=int, default=0,
                    help="optional random candidates after coordinate refinement")
    ap.add_argument("--coordinate-rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--render-size", type=int, default=112)
    ap.add_argument("--max-position-delta-m", type=float, default=0.025)
    ap.add_argument("--max-rotation-delta-deg", type=float, default=5.0)
    ap.add_argument("--max-fovy-delta-deg", type=float, default=6.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("output already exists")

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    domain = load_visual_domain(args.visual_domain)
    cfg = copy.deepcopy(load_config(args.config))
    kinematic = registration.get("kinematic_grasp")
    if isinstance(kinematic, dict):
        apply_kinematic_grasp(cfg, kinematic)
    object_size = np.asarray(registration["object_size_m"], dtype=float)
    cfg["task"]["object"]["half_size_m"] = (object_size / 2.0).tolist()
    cfg["task"]["object"]["init_pos"] = list(registration["object_xyz_m"])
    cfg["task"]["table"]["half_size_m"][:2] = list(
        registration["table_half_size_xy_m"])
    cfg = apply_visual_domain(cfg, domain)

    model = build_model(cfg)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    object_qadr = int(model.jnt_qposadr[object_joint])
    samples: list[dict[str, object]] = []
    for row in registration.get("selected", {}).get("per_episode", []):
        if row.get("scene_constraints_ok") is not True:
            continue
        episode = str(row["episode"])
        path = args.data / f"{episode}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as stored:
            image_chw = stored["image"][int(row["approach_start_row"]), -1]
        image = Image.fromarray(np.transpose(image_chw, (1, 2, 0)))
        component = largest_object_bbox(image)
        if component is None:
            continue
        target = _bbox_norm(component[3], image.width, image.height)
        gap = float(row["approach_start_executed_gap_m"])
        q = np.r_[np.asarray(row["approach_start_arm_rad"], dtype=float),
                  invert_gap_curve(gap, cfg["grasp"]["gap_curve"], clamp=True)]
        state = mujoco.MjData(model)
        state.qpos[:6] = q
        state.qpos[object_qadr:object_qadr + 3] = row["object_xyz_m"]
        state.qpos[object_qadr + 3:object_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        sync_gripper_collision_proxy(model, state, cfg)
        mujoco.mj_forward(model, state)
        samples.append({"episode": episode, "target": target, "state": state})
    if len(samples) < 3:
        raise SystemExit("need at least three registered colour detections")

    targets = np.stack([np.asarray(row["target"], dtype=float) for row in samples])
    holdout = np.arange(3, len(samples), 4, dtype=int)
    if not len(holdout):
        holdout = np.array([len(samples) - 1], dtype=int)
    train = np.setdiff1d(np.arange(len(samples)), holdout)
    base_position = np.asarray(domain["camera"]["pos"], dtype=float)
    base_rotation = _camera_rotation(domain["camera"]["xyaxes"])
    base_fovy = float(domain["camera"]["fovy"])

    renderer = mujoco.Renderer(
        model, height=args.render_size, width=args.render_size)

    def evaluate(position: np.ndarray, rotation: np.ndarray, fovy: float):
        model.cam_pos[camera_id] = position
        mujoco.mju_mat2Quat(model.cam_quat[camera_id], rotation.reshape(-1))
        model.cam_fovy[camera_id] = fovy
        predictions = []
        for sample in samples:
            state = sample["state"]
            sync_gripper_collision_proxy(model, state, cfg)
            mujoco.mj_forward(model, state)
            renderer.update_scene(state, camera="cam_wrist")
            rgb = renderer.render().copy()
            component = largest_object_bbox(Image.fromarray(rgb))
            if component is None:
                return float("inf"), None
            predictions.append(_bbox_norm(
                component[3], args.render_size, args.render_size))
        values = np.stack(predictions)
        return _loss(targets[train], values[train]), values

    base_loss, base_predictions = evaluate(base_position, base_rotation, base_fovy)
    if base_predictions is None:
        raise SystemExit("base camera does not produce a colour-component bbox")
    best = (base_loss, base_position.copy(), base_rotation.copy(), base_fovy,
            base_predictions)
    position_step = args.max_position_delta_m / 2.0
    rotation_step = np.deg2rad(args.max_rotation_delta_deg / 2.0)
    fovy_step = args.max_fovy_delta_deg / 2.0
    for round_index in range(args.coordinate_rounds):
        round_best = best
        _, centre_position, centre_rotation, centre_fovy, _ = best
        candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                position = centre_position.copy()
                position[axis] += sign * position_step
                candidates.append((position, centre_rotation, centre_fovy))
                angles = np.zeros(3)
                angles[axis] = sign * rotation_step
                candidates.append((
                    centre_position, centre_rotation @ _rotation_xyz(angles), centre_fovy))
        candidates.extend([
            (centre_position, centre_rotation, centre_fovy - fovy_step),
            (centre_position, centre_rotation, centre_fovy + fovy_step),
        ])
        for position, rotation, fovy in candidates:
            score, predictions = evaluate(position, rotation, fovy)
            if predictions is not None and score < round_best[0]:
                round_best = (
                    score, position.copy(), rotation.copy(), float(fovy),
                    predictions.copy())
        best = round_best
        print(
            f"coordinate round {round_index + 1}/{args.coordinate_rounds}: "
            f"best={best[0]:.6f}", flush=True)
        position_step *= 0.6
        rotation_step *= 0.6
        fovy_step *= 0.6

    rng = np.random.default_rng(args.seed)
    max_angle = np.deg2rad(args.max_rotation_delta_deg)
    for index in range(args.candidates):
        delta_position = rng.uniform(
            -args.max_position_delta_m, args.max_position_delta_m, 3)
        delta_rotation = rng.uniform(-max_angle, max_angle, 3)
        fovy = float(base_fovy + rng.uniform(
            -args.max_fovy_delta_deg, args.max_fovy_delta_deg))
        position = base_position + delta_position
        rotation = base_rotation @ _rotation_xyz(delta_rotation)
        score, predictions = evaluate(position, rotation, fovy)
        if predictions is None:
            continue
        score += 0.005 * float(np.sum(
            (delta_position / args.max_position_delta_m) ** 2))
        score += 0.003 * float(np.sum((delta_rotation / max_angle) ** 2))
        if score < best[0]:
            best = (score, position.copy(), rotation.copy(), fovy, predictions.copy())
        if (index + 1) % 250 == 0:
            print(f"render candidates {index + 1}/{args.candidates}: best={best[0]:.6f}",
                  flush=True)
    renderer.close()

    _, position, rotation, fovy, predictions = best
    result = {
        "schema": "s22_mujoco_render_alignment/0.1.0-provisional",
        "status": "RENDER_OBJECTIVE_VISUAL_PROXY_NOT_HARDWARE_CALIBRATION",
        "source_dataset": str(args.data),
        "source_registration": str(args.registration),
        "source_visual_domain": str(args.visual_domain),
        "samples": len(samples),
        "train_episodes": [samples[i]["episode"] for i in train],
        "holdout_episodes": [samples[i]["episode"] for i in holdout],
        "render_size": args.render_size,
        "base": {
            "camera": domain["camera"],
            "loss": float(_loss(targets, base_predictions)),
            "holdout_loss": float(_loss(targets[holdout], base_predictions[holdout])),
            "summary": _summary(base_predictions),
        },
        "fitted": {
            "pos": position.tolist(),
            "xyaxes": rotation[:, :2].T.reshape(-1).tolist(),
            "fovy": float(fovy),
            "loss": float(_loss(targets, predictions)),
            "train_loss": float(_loss(targets[train], predictions[train])),
            "holdout_loss": float(_loss(targets[holdout], predictions[holdout])),
            "summary": _summary(predictions),
        },
        "real_summary": _summary(targets),
        "per_episode": [
            {
                "episode": sample["episode"],
                "real_bbox_xywh_norm": target.tolist(),
                "rendered_bbox_xywh_norm": prediction.tolist(),
            }
            for sample, target, prediction in zip(samples, targets, predictions)
        ],
        "limitations": [
            "Colour-component boxes are not manual object masks.",
            "This is a local render fit on eight starts, not hardware calibration.",
            "Policy rollout, contact physics and sim-to-real transfer are not evaluated.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "base": result["base"], "fitted": result["fitted"], "out": str(args.out)
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
