"""Compare real S22 frames with the provisional MuJoCo visual domain.

This is an appearance diagnostic, not a rollout score.  It renders the same
registered approach starts used by the camera fit, reports RGB/contrast/edge
statistics on the fixed 6/2 episode split, and proves that render-only clutter
does not hide the target.  No learned checkpoint is loaded.
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
from PIL import Image, ImageDraw

from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import (
    build_model,
    load_config,
    sync_gripper_collision_proxy,
)
from tools.umi_mujoco import apply_kinematic_grasp
from umi.convert import invert_gap_curve
from umi.visual_domain import apply_visual_domain, load_visual_domain


def image_stats(image: np.ndarray) -> dict[str, object]:
    rgb = np.asarray(image, dtype=np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edges = (float(np.mean(dx > 20.0)) + float(np.mean(dy > 20.0))) / 2.0
    return {
        "rgb_mean": np.mean(rgb, axis=(0, 1)).tolist(),
        "rgb_std": np.std(rgb, axis=(0, 1)).tolist(),
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "edge_fraction_abs_gradient_gt_20": edges,
    }


def aggregate(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--visual-domain", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("output already exists")

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(load_config(args.config))
    kinematic = registration.get("kinematic_grasp")
    if isinstance(kinematic, dict):
        apply_kinematic_grasp(cfg, kinematic)
    object_size = np.asarray(registration["object_size_m"], dtype=float)
    cfg["task"]["object"]["half_size_m"] = (object_size / 2.0).tolist()
    cfg["task"]["object"]["init_pos"] = list(registration["object_xyz_m"])
    cfg["task"]["table"]["half_size_m"][:2] = list(
        registration["table_half_size_xy_m"])
    cfg = apply_visual_domain(cfg, load_visual_domain(args.visual_domain))

    model = build_model(cfg)
    data = mujoco.MjData(model)
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    object_qadr = int(model.jnt_qposadr[object_joint])
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    samples: list[dict[str, object]] = []
    tiles: list[tuple[str, np.ndarray, np.ndarray]] = []

    with mujoco.Renderer(model, height=args.size, width=args.size) as renderer:
        for row in registration.get("selected", {}).get("per_episode", []):
            if row.get("scene_constraints_ok") is not True:
                continue
            episode = str(row["episode"])
            path = args.data / f"{episode}.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as stored:
                real_chw = stored["image"][int(row["approach_start_row"]), -1]
            real = np.asarray(Image.fromarray(np.transpose(real_chw, (1, 2, 0))).resize(
                (args.size, args.size), Image.Resampling.BILINEAR))
            gap = float(row["approach_start_executed_gap_m"])
            q = np.r_[np.asarray(row["approach_start_arm_rad"], dtype=float),
                      invert_gap_curve(gap, cfg["grasp"]["gap_curve"], clamp=True)]
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = q
            data.qpos[object_qadr:object_qadr + 3] = row["object_xyz_m"]
            data.qpos[object_qadr + 3:object_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            sync_gripper_collision_proxy(model, data, cfg)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="cam_wrist")
            sim = renderer.render().copy()

            camera_xyz = data.cam_xpos[camera_id].copy()
            centre = data.xpos[object_body].copy()
            visible = 0
            hit_counts: dict[str, int] = {}
            offsets = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.25 * object_size[1], 0.0],
                [0.0, -0.25 * object_size[1], 0.0],
                [0.0, 0.0, 0.25 * object_size[2]],
                [0.0, 0.0, -0.25 * object_size[2]],
            ])
            for point in centre + offsets:
                direction = point - camera_xyz
                direction /= np.linalg.norm(direction)
                hit = np.zeros(1, dtype=np.int32)
                mujoco.mj_ray(model, data, camera_xyz, direction, None, True, -1, hit)
                hit_body = -1 if hit[0] < 0 else int(model.geom_bodyid[int(hit[0])])
                visible += int(hit_body == object_body)
                name = ("MISS" if hit[0] < 0 else str(mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0]))))
                hit_counts[name] = hit_counts.get(name, 0) + 1

            samples.append({
                "episode": episode,
                "real": image_stats(real),
                "sim": image_stats(sim),
                "target_visible_rays": f"{visible}/5",
                "first_hit_geom_counts": hit_counts,
            })
            tiles.append((episode, real, sim))

    if len(samples) < 3:
        raise SystemExit(f"only {len(samples)} registered episodes were rendered")
    holdout = list(range(3, len(samples), 4)) or [len(samples) - 1]
    train = [index for index in range(len(samples)) if index not in holdout]

    def split_summary(indices: list[int]) -> dict[str, object]:
        real = [samples[index]["real"] for index in indices]
        sim = [samples[index]["sim"] for index in indices]
        keys = ("rgb_mean", "rgb_std", "gray_mean", "gray_std",
                "edge_fraction_abs_gradient_gt_20")
        return {
            "episodes": [samples[index]["episode"] for index in indices],
            "real": {key: aggregate(real, key) for key in keys},
            "sim": {key: aggregate(sim, key) for key in keys},
        }

    result = {
        "schema": "s22_mujoco_visual_domain_measurement/0.1.0",
        "status": "APPEARANCE_DIAGNOSTIC_NOT_ROLLOUT_SCORE",
        "policy_loaded": False,
        "source_dataset": str(args.data),
        "source_registration": str(args.registration),
        "visual_domain": str(args.visual_domain),
        "samples": len(samples),
        "train": split_summary(train),
        "holdout": split_summary(holdout),
        "target_visibility": {
            "visible_rays": int(sum(int(str(row["target_visible_rays"]).split("/")[0])
                                    for row in samples)),
            "total_rays": 5 * len(samples),
        },
        "per_episode": samples,
        "interpretation": (
            "RGB moments and edge density diagnose appearance coverage only; "
            "they are not success gates and do not establish sim-to-real transfer."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    count = min(6, len(tiles))
    canvas = Image.new("RGB", (args.size * count, args.size * 2 + 44), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (episode, real, sim) in enumerate(tiles[:count]):
        canvas.paste(Image.fromarray(real), (index * args.size, 22))
        canvas.paste(Image.fromarray(sim), (index * args.size, args.size + 44))
        draw.text((index * args.size + 2, 3), episode.replace("rec_20260911_", ""), fill="black")
    draw.text((2, args.size + 25), "MuJoCo visual-domain proxy", fill="black")
    montage = args.out.with_suffix(".png")
    canvas.save(montage)
    print(json.dumps({
        "report": str(args.out),
        "montage": str(montage),
        "samples": len(samples),
        "target_visibility": result["target_visibility"],
        "holdout": result["holdout"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
