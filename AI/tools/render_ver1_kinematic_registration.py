"""Render a selected ver1 registration without running a learned policy.

This is deliberately a kinematic inspection movie.  The handoff only defines
the ver1 TCP/jaw frame; it does not contain camera optical extrinsics, collision
geometry, inertias, or actuator validation.  Consequently this tool places a
clearly coloured parallel-jaw *visual proxy* at the achieved EEF pose and never
steps object-contact dynamics or reports task success.
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

from paths import DEFAULT_CONFIG, DEFAULT_SCENE
from sim.mujoco.build_scene import build_model, load_config
from tools.search_relative_chunk_registration import NumpyRelativeDataset
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp
from umi.relative_robot_preflight import arm_ranges, episode_anchors, real_limits


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH = (
    AI_ROOT / "out" /
    "relative_chunk_v9_ver1_kinematic_registration_search32_refined_val14_20260915.json"
)
DEFAULT_KINEMATIC_GRASP = (
    AI_ROOT / "configs" / "real" / "so101_ver1_phone_holder_kinematic.json"
)
DEFAULT_REAL_CONFIG = AI_ROOT / "configs" / "real" / "so101_ver1.json"


def _selected_episode(report: dict, episode_id: str | None) -> tuple[dict, dict]:
    selected = report["selected"]
    wanted = episode_id or selected["representative_episode"]
    matches = [row for row in selected["per_episode"] if row["episode"] == wanted]
    if len(matches) != 1:
        available = [row["episode"] for row in selected["per_episode"]]
        raise ValueError(f"episode {wanted!r} is not in selected candidate; available={available}")
    return selected, matches[0]


def _hide_legacy_gripper(model: mujoco.MjModel) -> None:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(model.body_parentid):
            if int(parent_id) in descendants and body_id not in descendants:
                descendants.add(body_id)
                changed = True
    for geom_id, body_id in enumerate(model.geom_bodyid):
        if int(body_id) in descendants:
            model.geom_rgba[geom_id, 3] = 0.0


def _set_object(model: mujoco.MjModel, data: mujoco.MjData,
                xyz: np.ndarray, size: np.ndarray) -> None:
    geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    model.geom_size[geom] = size / 2.0
    qadr = int(model.jnt_qposadr[joint])
    data.qpos[qadr:qadr + 3] = xyz
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])


def _add_box(scene: mujoco.MjvScene, *, size: np.ndarray, pos: np.ndarray,
             rotation: np.ndarray, rgba: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_ver1_proxy(scene: mujoco.MjvScene, pose: np.ndarray, gap_m: float) -> None:
    """Add an intentionally simple, non-colliding parallel-jaw visual proxy."""
    rotation = pose[:3, :3]
    jaw, approach = rotation[:, 0], rotation[:, 2]
    pinch = pose[:3, 3]
    finger_half = np.array([0.005, 0.012, 0.040])
    for sign in (-1.0, 1.0):
        centre = (
            pinch - finger_half[2] * approach
            + sign * (float(gap_m) / 2.0 + finger_half[0]) * jaw
        )
        _add_box(
            scene, size=finger_half, pos=centre, rotation=rotation,
            rgba=(0.08, 0.85, 0.95, 0.92),
        )
    palm_half = np.array([max(0.025, float(gap_m) / 2.0 + 0.012), 0.018, 0.010])
    _add_box(
        scene,
        size=palm_half,
        pos=pinch - 0.090 * approach,
        rotation=rotation,
        rgba=(0.12, 0.35, 0.95, 0.88),
    )


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_height = 15
    width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 16
    height = line_height * len(lines) + 12
    draw.rectangle((6, 6, 6 + width, 6 + height), fill=(0, 0, 0, 190))
    for index, line in enumerate(lines):
        draw.text((14, 11 + index * line_height), line, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=max(1, round(1000 / fps)), loop=0, optimize=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--search", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--kinematic-grasp", type=Path, default=DEFAULT_KINEMATIC_GRASP)
    parser.add_argument("--real-config", type=Path, default=DEFAULT_REAL_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--episode", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")

    report = json.loads(args.search.read_text(encoding="utf-8"))
    selected, episode_report = _selected_episode(report, args.episode)
    grasp_payload = json.loads(args.kinematic_grasp.read_text(encoding="utf-8"))
    if grasp_payload.get("dynamic_mujoco_ready") is not False:
        raise SystemExit("this renderer requires a kinematic-only handoff overlay")

    dataset = NumpyRelativeDataset(args.data)
    episode = next(
        (row for row in dataset.episodes if row["id"] == episode_report["episode"]),
        None,
    )
    if episode is None:
        raise SystemExit(f"episode not found in dataset: {episode_report['episode']}")

    cfg = apply_kinematic_grasp(copy.deepcopy(load_config(args.config)), grasp_payload)
    model = build_model(cfg, args.scene)
    _hide_legacy_gripper(model)
    data = mujoco.MjData(model)
    object_xyz = np.asarray(episode_report["object_xyz_m"], dtype=float)
    object_size = np.asarray(report["object_size_m"], dtype=float)
    _set_object(model, data, object_xyz, object_size)

    ik = MujocoIK(model, cfg)
    ranges = arm_ranges(real_limits(args.real_config))
    poses, arms = episode_anchors(
        episode, model=model, ik=ik, curve=cfg["grasp"]["gap_curve"],
        arm_start=np.asarray(selected["start_arm_rad"], dtype=float), ranges=ranges,
    )
    valid_indices = [index for index, arm in enumerate(arms) if arm is not None]
    if len(valid_indices) < 2:
        raise SystemExit("fewer than two IK-valid trajectory rows")

    gaps = np.asarray(episode["proprio"][:, -1, 9], dtype=float)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    object_qadr = int(model.jnt_qposadr[object_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.33, 0.0, 0.12])
    camera.distance = 0.78
    camera.azimuth = 128.0
    camera.elevation = -18.0

    frames: list[np.ndarray] = []
    snapshots: dict[str, np.ndarray] = {}
    contact_row = int(episode_report["contact_row"])
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for order, index in enumerate(valid_indices):
            arm = np.asarray(arms[index], dtype=float)
            data.qpos[:5] = arm
            data.qpos[5] = 0.0
            data.qpos[object_qadr:object_qadr + 3] = object_xyz
            data.qpos[object_qadr + 3:object_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            _add_ver1_proxy(renderer.scene, poses[index], float(gaps[index]))
            label = "CONTACT ROW" if index == contact_row else "KINEMATIC TRAJECTORY"
            frame = _annotate(renderer.render().copy(), [
                "VER1 KINEMATIC INSPECTION - NO POLICY",
                f"episode {episode['id']}  row {index + 1}/{len(arms)}  {label}",
                f"gap {gaps[index] * 1000.0:.1f} mm",
                "cyan: non-colliding visual jaw proxy",
                "camera extrinsic + dynamics unavailable; NOT a success rollout",
            ])
            if order == 0:
                snapshots["start"] = frame
            if index == contact_row:
                snapshots["contact"] = frame
            if order == len(valid_indices) - 1:
                snapshots["end"] = frame
            repeats = args.hold_frames if order in (0, len(valid_indices) - 1) else 1
            frames.extend([frame] * repeats)

    _save_gif(frames, args.out, args.fps)
    from PIL import Image
    snapshot_paths = {}
    for name, frame in snapshots.items():
        path = args.out.with_name(f"{args.out.stem}_{name}.png")
        Image.fromarray(frame).save(path)
        snapshot_paths[name] = str(path)
    summary = {
        "status": "KINEMATIC_INSPECTION_ONLY",
        "policy_inference": False,
        "episode": episode["id"],
        "candidate_id": int(selected["candidate_id"]),
        "source_rows": len(arms),
        "rendered_valid_rows": len(valid_indices),
        "skipped_ik_invalid_rows": len(arms) - len(valid_indices),
        "contact_row": contact_row,
        "object_xyz_m": object_xyz.tolist(),
        "object_size_m": object_size.tolist(),
        "media": str(args.out),
        "snapshots": snapshot_paths,
        "limitations": [
            "No learned-policy inference was run.",
            "The cyan gripper is a non-colliding visual proxy, not a dynamic model.",
            "The handoff has no camera optical extrinsic, validated collision geometry, or inertia.",
            "This video cannot establish grasp or rollout success.",
        ],
    }
    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
