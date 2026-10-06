"""Fit a *visual-domain proxy* wrist camera to real S22 object framing.

This tool compares the upright snack-case bounding box in canonical v10
training images with the projection of the registered MuJoCo box at the same
fixed approach starts.  It fits camera translation/orientation and vertical
FOV only.  The result is for simulation-domain matching; it is not a measured
IMX708 optical extrinsic and must never replace hardware calibration.
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
from tools.estimate_umi_object_geometry import largest_object_bbox
from tools.umi_mujoco import apply_kinematic_grasp
from umi.convert import invert_gap_curve


def _rotation_xyz(angles_rad: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(angles_rad, dtype=float)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _camera_rotation(xyaxes: list[float]) -> np.ndarray:
    values = np.asarray(xyaxes, dtype=float).reshape(2, 3)
    x = values[0] / np.linalg.norm(values[0])
    y = values[1] - x * float(x @ values[1])
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def _bbox_norm(bbox: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = np.asarray(bbox, dtype=float)
    return np.array([
        (x0 + x1 + 1.0) / (2.0 * width),
        (y0 + y1 + 1.0) / (2.0 * height),
        (x1 - x0 + 1.0) / width,
        (y1 - y0 + 1.0) / height,
    ])


def _project_bbox(
    corners_world: np.ndarray,
    body_position: np.ndarray,
    body_rotation: np.ndarray,
    camera_position_body: np.ndarray,
    camera_rotation_body: np.ndarray,
    fovy_deg: float,
) -> np.ndarray | None:
    camera_position_world = body_position + body_rotation @ camera_position_body
    camera_rotation_world = body_rotation @ camera_rotation_body
    local = (corners_world - camera_position_world) @ camera_rotation_world
    depth = -local[:, 2]
    if np.any(depth <= 1e-5):
        return None
    half = np.tan(np.deg2rad(float(fovy_deg)) / 2.0)
    u = 0.5 + 0.5 * local[:, 0] / (depth * half)
    v = 0.5 - 0.5 * local[:, 1] / (depth * half)
    return np.array([
        (float(np.min(u)) + float(np.max(u))) / 2.0,
        (float(np.min(v)) + float(np.max(v))) / 2.0,
        float(np.max(u) - np.min(u)),
        float(np.max(v) - np.min(v)),
    ])


def _loss(targets: np.ndarray, predictions: np.ndarray) -> float:
    centre = np.mean(np.square(predictions[:, :2] - targets[:, :2]))
    size = np.mean(np.square(np.log(np.maximum(predictions[:, 2:], 1e-5)
                                    / np.maximum(targets[:, 2:], 1e-5))))
    outside = np.mean(np.square(np.maximum(np.abs(predictions[:, :2] - 0.5) - 0.5, 0.0)))
    return float(4.0 * centre + size + 10.0 * outside)


def _summary(values: np.ndarray) -> dict[str, list[float] | float]:
    return {
        "centre_xy_median": np.median(values[:, :2], axis=0).tolist(),
        "size_wh_median": np.median(values[:, 2:], axis=0).tolist(),
        "bbox_area_fraction_median": float(np.median(values[:, 2] * values[:, 3])),
    }


def _object_corners(object_xyz: np.ndarray, object_size: np.ndarray) -> np.ndarray:
    signs = np.array([
        [x, y, z] for x in (-1.0, 1.0)
        for y in (-1.0, 1.0) for z in (-1.0, 1.0)
    ])
    return object_xyz + signs * object_size / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidates", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--montage-episodes", type=int, default=6)
    ap.add_argument("--max-position-delta-m", type=float, default=0.06)
    ap.add_argument("--max-rotation-delta-deg", type=float, default=95.0)
    ap.add_argument("--fovy-min-deg", type=float, default=30.0)
    ap.add_argument("--fovy-max-deg", type=float, default=110.0)
    ap.add_argument("--refine-from", type=Path, default=None,
                    help="previous alignment report whose fitted camera seeds a local search")
    ap.add_argument("--min-visible-ray-fraction", type=float, default=0.90)
    ap.add_argument("--shortlist-size", type=int, default=2048,
                    help="number of best projection candidates checked with expensive ray gates")
    ap.add_argument(
        "--visual-object-size-m", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help=("optional render-proxy full size; its bottom remains on the same "
              "table plane while physical registration dimensions stay unchanged"),
    )
    args = ap.parse_args()

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(load_config(args.config))
    kinematic = registration.get("kinematic_grasp")
    if isinstance(kinematic, dict):
        apply_kinematic_grasp(cfg, kinematic)
    physical_object_size = np.asarray(registration["object_size_m"], dtype=float)
    object_size = (
        np.asarray(args.visual_object_size_m, dtype=float)
        if args.visual_object_size_m is not None else physical_object_size.copy()
    )
    if object_size.shape != (3,) or not np.isfinite(object_size).all() or np.any(object_size <= 0):
        raise SystemExit("--visual-object-size-m must contain three positive finite values")
    visual_centre_shift = np.array(
        [0.0, 0.0, 0.5 * (object_size[2] - physical_object_size[2])],
        dtype=float,
    )
    cfg["task"]["object"]["half_size_m"] = (object_size / 2.0).tolist()
    cfg["task"]["object"]["init_pos"] = (
        np.asarray(registration["object_xyz_m"], dtype=float) + visual_centre_shift
    ).tolist()
    cfg["task"]["table"]["half_size_m"][:2] = list(
        registration["table_half_size_xy_m"])
    model = build_model(cfg)
    data = mujoco.MjData(model)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    samples: list[dict] = []
    for row in registration.get("selected", {}).get("per_episode", []):
        if row.get("scene_constraints_ok") is not True:
            continue
        episode_id = str(row["episode"])
        path = args.data / f"{episode_id}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as stored:
            image_chw = stored["image"][int(row["approach_start_row"]), -1]
        image = Image.fromarray(np.transpose(image_chw, (1, 2, 0)))
        component = largest_object_bbox(image)
        if component is None:
            continue
        real_bbox = _bbox_norm(component[3], image.width, image.height)
        gap = float(row["approach_start_executed_gap_m"])
        q = np.r_[np.asarray(row["approach_start_arm_rad"], dtype=float),
                  invert_gap_curve(gap, cfg["grasp"]["gap_curve"], clamp=True)]
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        physical_object_xyz = np.asarray(row["object_xyz_m"], dtype=float)
        object_xyz = physical_object_xyz + visual_centre_shift
        samples.append({
            "episode": episode_id,
            "image": np.asarray(image),
            "real_bbox_px": list(component[3]),
            "target": real_bbox,
            "body_position": data.xpos[gripper_id].copy(),
            "body_rotation": data.xmat[gripper_id].reshape(3, 3).copy(),
            "corners": _object_corners(object_xyz, object_size),
            "q": q,
            "object_xyz": object_xyz,
        })
    if len(samples) < 3:
        raise SystemExit(f"only {len(samples)} usable real object detections; need at least 3")

    targets = np.stack([sample["target"] for sample in samples])
    # Deterministic episode-level holdout.  Camera fitting on all episodes would
    # make the reported agreement a training score rather than validation.
    holdout_indices = np.arange(3, len(samples), 4, dtype=int)
    if holdout_indices.size == 0:
        holdout_indices = np.array([len(samples) - 1], dtype=int)
    train_indices = np.setdiff1d(np.arange(len(samples)), holdout_indices)
    base_position = np.asarray(cfg["cameras"]["cam_wrist"]["pos"], dtype=float)
    base_rotation = _camera_rotation(cfg["cameras"]["cam_wrist"]["xyaxes"])
    base_fovy = float(cfg["cameras"]["cam_wrist"]["fovy"])
    search_position = base_position.copy()
    search_rotation = base_rotation.copy()
    search_fovy = base_fovy
    if args.refine_from is not None:
        previous = json.loads(args.refine_from.read_text(encoding="utf-8"))["fitted"]
        search_position = np.asarray(previous["pos"], dtype=float)
        search_rotation = _camera_rotation(previous["xyaxes"])
        search_fovy = float(previous["fovy"])
    object_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    object_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    object_qadr = int(model.jnt_qposadr[object_joint_id])
    ray_states = []
    for sample in samples:
        state = mujoco.MjData(model)
        state.qpos[:6] = sample["q"]
        state.qpos[object_qadr:object_qadr + 3] = sample["object_xyz"]
        state.qpos[object_qadr + 3:object_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        sync_gripper_collision_proxy(model, state, cfg)
        mujoco.mj_forward(model, state)
        ray_states.append(state)

    def visibility_fraction(position: np.ndarray) -> float:
        visible = 0
        total = 0
        for sample, state in zip(samples, ray_states):
            sync_gripper_collision_proxy(model, state, cfg)
            mujoco.mj_forward(model, state)
            camera_world = sample["body_position"] + sample["body_rotation"] @ position
            centre = sample["object_xyz"]
            offsets = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.25 * object_size[1], 0.0],
                [0.0, -0.25 * object_size[1], 0.0],
                [0.0, 0.0, 0.25 * object_size[2]],
                [0.0, 0.0, -0.25 * object_size[2]],
            ])
            for point in centre + offsets:
                direction = point - camera_world
                direction /= np.linalg.norm(direction)
                hit = np.zeros(1, dtype=np.int32)
                mujoco.mj_ray(model, state, camera_world, direction, None, True,
                              -1, hit)
                visible += int(int(hit[0]) == object_geom_id)
                total += 1
        return float(visible / total)

    def visibility_hit_counts(position: np.ndarray) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample, state in zip(samples, ray_states):
            sync_gripper_collision_proxy(model, state, cfg)
            mujoco.mj_forward(model, state)
            camera_world = sample["body_position"] + sample["body_rotation"] @ position
            centre = sample["object_xyz"]
            offsets = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.25 * object_size[1], 0.0],
                [0.0, -0.25 * object_size[1], 0.0],
                [0.0, 0.0, 0.25 * object_size[2]],
                [0.0, 0.0, -0.25 * object_size[2]],
            ])
            for point in centre + offsets:
                direction = point - camera_world
                direction /= np.linalg.norm(direction)
                hit = np.zeros(1, dtype=np.int32)
                mujoco.mj_ray(model, state, camera_world, direction, None, True,
                              -1, hit)
                name = ("MISS" if int(hit[0]) < 0 else
                        str(mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0]))))
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def screen_up_errors_deg(rotation: np.ndarray) -> np.ndarray:
        errors = []
        for sample in samples:
            camera_world = sample["body_rotation"] @ rotation
            up_camera = camera_world.T @ np.array([0.0, 0.0, 1.0])
            screen = up_camera[:2]
            norm = float(np.linalg.norm(screen))
            if norm < 1e-8:
                errors.append(180.0)
                continue
            screen /= norm
            errors.append(float(np.rad2deg(np.arccos(np.clip(screen[1], -1.0, 1.0)))))
        return np.asarray(errors)

    def evaluate(position: np.ndarray, rotation: np.ndarray, fovy: float):
        predictions = []
        for sample in samples:
            projected = _project_bbox(
                sample["corners"], sample["body_position"], sample["body_rotation"],
                position, rotation, fovy)
            if projected is None or not np.isfinite(projected).all():
                return float("inf"), None
            predictions.append(projected)
        values = np.stack(predictions)
        return _loss(targets[train_indices], values[train_indices]), values

    base_loss, base_predictions = evaluate(base_position, base_rotation, base_fovy)
    assert base_predictions is not None
    base_candidate = (base_loss, base_position.copy(), base_rotation.copy(), base_fovy,
                      np.zeros(3), np.zeros(3), base_predictions)
    shortlist = [base_candidate]
    if args.refine_from is not None:
        seed_loss, seed_predictions = evaluate(search_position, search_rotation, search_fovy)
        assert seed_predictions is not None
        shortlist.append((seed_loss, search_position.copy(), search_rotation.copy(),
                          search_fovy, search_position - base_position,
                          np.zeros(3), seed_predictions))
    rng = np.random.default_rng(args.seed)
    # Broad but bounded visual-proxy search around the existing provisional camera.
    # The regularizer prevents replacing a missing hardware calibration with an
    # arbitrary virtual camera that merely overfits a handful of images.
    for _ in range(args.candidates):
        max_position = float(args.max_position_delta_m)
        max_rotation = float(args.max_rotation_delta_deg)
        delta_position = rng.uniform(-max_position, max_position, size=3)
        delta_angle = np.deg2rad(rng.uniform(-max_rotation, max_rotation, size=3))
        position = search_position + delta_position
        rotation = search_rotation @ _rotation_xyz(delta_angle)
        if args.refine_from is None:
            fovy = float(rng.uniform(args.fovy_min_deg, args.fovy_max_deg))
        else:
            fovy = float(np.clip(search_fovy + rng.uniform(-15.0, 15.0),
                                 args.fovy_min_deg, args.fovy_max_deg))
        score, predictions = evaluate(position, rotation, fovy)
        score += 0.04 * float(np.sum((delta_position / max_position) ** 2))
        score += 0.01 * float(np.sum((delta_angle / np.deg2rad(max_rotation)) ** 2))
        upright = screen_up_errors_deg(rotation)
        score += 0.10 * float(np.mean((upright / 15.0) ** 2))
        if predictions is not None:
            shortlist.append((score, position, rotation, fovy, delta_position,
                              delta_angle, predictions))
            if len(shortlist) > max(8192, 4 * args.shortlist_size):
                shortlist = sorted(shortlist, key=lambda item: item[0])[:args.shortlist_size]

    shortlist = sorted(shortlist, key=lambda item: item[0])[:args.shortlist_size]
    diagnostics = [
        (visibility_fraction(item[1]), screen_up_errors_deg(item[2]), item)
        for item in shortlist
    ]
    valid_candidates = [item for visibility, up_errors, item in diagnostics
                        if visibility >= args.min_visible_ray_fraction
                        and np.percentile(up_errors, 95) <= 20.0]
    if valid_candidates:
        best = min(valid_candidates, key=lambda item: item[0])
        visibility_gate = "PASS"
    else:
        # Keep the failure inspectable rather than silently returning no file.
        best = min(
            diagnostics,
            key=lambda row: (row[2][0] + 4.0 * (1.0 - row[0])
                             + 0.10 * float(np.mean((row[1] / 15.0) ** 2))),
        )[2]
        visibility_gate = "FAIL_NO_CANDIDATE_AT_VISIBILITY_AND_UPRIGHT_GATE"
    best_visibility = visibility_fraction(best[1])
    best_up_errors = screen_up_errors_deg(best[2])

    _, fit_position, fit_rotation, fit_fovy, delta_position, delta_angle, fitted = best
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample, target, before, after in zip(samples, targets, base_predictions, fitted):
        rows.append({
            "episode": sample["episode"],
            "real_bbox_xywh_norm": target.tolist(),
            "base_sim_bbox_xywh_norm": before.tolist(),
            "fitted_sim_bbox_xywh_norm": after.tolist(),
        })
    result = {
        "schema": "s22_mujoco_visual_alignment/0.1.0-provisional",
        "status": "VISUAL_DOMAIN_PROXY_NOT_HARDWARE_CALIBRATION",
        "source_dataset": str(args.data),
        "source_registration": str(args.registration),
        "physical_object_size_m": physical_object_size.tolist(),
        "visual_object_size_m": object_size.tolist(),
        "visual_object_centre_shift_m": visual_centre_shift.tolist(),
        "samples": len(samples),
        "train_episodes": [samples[index]["episode"] for index in train_indices],
        "holdout_episodes": [samples[index]["episode"] for index in holdout_indices],
        "object_detection": "largest yellow/orange HSV component in canonical v10 frame",
        "objective": "normalised object bbox centre and log-size agreement",
        "visibility_gate": {
            "status": visibility_gate,
            "minimum_ray_fraction": args.min_visible_ray_fraction,
            "method": "five object-interior rays per episode; first visible geom must be target_object_geom",
            "base_visible_ray_fraction": visibility_fraction(base_position),
            "fitted_visible_ray_fraction": best_visibility,
            "fitted_first_hit_geom_counts": visibility_hit_counts(best[1]),
            "shortlisted_candidates": len(shortlist),
        },
        "camera_direction_gate": {
            "status": ("PASS" if np.percentile(best_up_errors, 95) <= 20.0 else "FAIL"),
            "definition": "world +Z projected toward image up; 95th percentile error <= 20 deg",
            "base_error_deg_median_p95": [
                float(np.median(screen_up_errors_deg(base_rotation))),
                float(np.percentile(screen_up_errors_deg(base_rotation), 95)),
            ],
            "fitted_error_deg_median_p95": [
                float(np.median(best_up_errors)),
                float(np.percentile(best_up_errors, 95)),
            ],
        },
        "base": {
            "pos": base_position.tolist(),
            "xyaxes": base_rotation[:, :2].T.reshape(-1).tolist(),
            "fovy": base_fovy,
            "loss": base_loss,
            "holdout_loss": _loss(targets[holdout_indices],
                                    base_predictions[holdout_indices]),
            "projection_summary": _summary(base_predictions),
        },
        "fitted": {
            "pos": fit_position.tolist(),
            "xyaxes": fit_rotation[:, :2].T.reshape(-1).tolist(),
            "fovy": fit_fovy,
            "regularised_search_score": float(best[0]),
            "unregularised_loss": _loss(targets, fitted),
            "train_loss": _loss(targets[train_indices], fitted[train_indices]),
            "holdout_loss": _loss(targets[holdout_indices], fitted[holdout_indices]),
            "delta_position_m": (fit_position - base_position).tolist(),
            "last_search_delta_rotation_xyz_deg": np.rad2deg(delta_angle).tolist(),
            "projection_summary": _summary(fitted),
        },
        "real_summary": _summary(targets),
        "per_episode": rows,
        "limitations": [
            "The S22 package mask is colour-based and is not a manual annotation.",
            "The physical MuJoCo object remains the registration box with user-reported 40 mm grasp width.",
            "A differing visual size is an image-derived render proxy and is not a physical measurement.",
            "The fitted camera is a visual-domain proxy, not an IMX708 optical extrinsic.",
            "Only an episode-level holdout score, not rollout success, is reported here.",
        ],
        "refine_from": str(args.refine_from) if args.refine_from is not None else None,
    }
    report_path = args.out / "s22_mujoco_visual_alignment.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Render the fitted virtual camera at the exact same registered starts.
    fit_cfg = copy.deepcopy(cfg)
    fit_cfg["cameras"]["cam_wrist"]["pos"] = fit_position.tolist()
    fit_cfg["cameras"]["cam_wrist"]["xyaxes"] = (
        fit_rotation[:, :2].T.reshape(-1).tolist())
    fit_cfg["cameras"]["cam_wrist"]["fovy"] = float(fit_fovy)
    fit_model = build_model(fit_cfg)
    fit_data = mujoco.MjData(fit_model)
    object_joint = mujoco.mj_name2id(
        fit_model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    object_qadr = int(fit_model.jnt_qposadr[object_joint])

    # Visualise real frames and fitted MuJoCo frames.  Green is the colour mask
    # measurement, magenta is the geometric box projection.
    count = min(args.montage_episodes, len(samples))
    tile = 224
    canvas = Image.new("RGB", (tile * count, tile * 2 + 52), "white")
    draw = ImageDraw.Draw(canvas)
    with mujoco.Renderer(fit_model, height=tile, width=tile) as renderer:
        for index, (sample, after) in enumerate(zip(samples[:count], fitted[:count])):
            real_image = Image.fromarray(sample["image"]).copy()
            ImageDraw.Draw(real_image).rectangle(
                sample["real_bbox_px"], outline=(0, 255, 0), width=3)
            mujoco.mj_resetData(fit_model, fit_data)
            fit_data.qpos[:6] = sample["q"]
            fit_data.qpos[object_qadr:object_qadr + 3] = sample["object_xyz"]
            fit_data.qpos[object_qadr + 3:object_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            sync_gripper_collision_proxy(fit_model, fit_data, fit_cfg)
            mujoco.mj_forward(fit_model, fit_data)
            renderer.update_scene(fit_data, camera="cam_wrist")
            sim_image = Image.fromarray(renderer.render().copy())
            cx, cy, bw, bh = after
            fitted_px = [(cx - bw / 2) * tile, (cy - bh / 2) * tile,
                         (cx + bw / 2) * tile, (cy + bh / 2) * tile]
            ImageDraw.Draw(sim_image).rectangle(
                fitted_px, outline=(255, 0, 255), width=2)
            canvas.paste(real_image, (index * tile, 28))
            canvas.paste(sim_image, (index * tile, tile + 52))
            draw.text((index * tile + 3, 5),
                      sample["episode"].replace("rec_20260911_", ""), fill="black")
    draw.text((3, tile + 31), "fitted MuJoCo camera", fill="black")
    montage_path = args.out / "s22_real_bbox_vs_fitted_projection.png"
    canvas.save(montage_path)
    print(json.dumps({
        "samples": len(samples),
        "base_loss": base_loss,
        "fitted_loss": result["fitted"]["unregularised_loss"],
        "base_holdout_loss": result["base"]["holdout_loss"],
        "fitted_holdout_loss": result["fitted"]["holdout_loss"],
        "real_summary": result["real_summary"],
        "base_summary": result["base"]["projection_summary"],
        "fitted_summary": result["fitted"]["projection_summary"],
        "report": str(report_path),
        "montage": str(montage_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
