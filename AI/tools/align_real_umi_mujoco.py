"""Align a MuJoCo pilot scene to one real-UMI episode and render its first view.

The object XY is inferred from the FK pinch point at the first frame where the
observed gap reaches the supplied grasp width.  It is evidence for a comparison
render, not a measured world-to-base calibration.
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

from contract.episode import read_episode
from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import denormalize, load_config
from sim.mujoco.env import MujocoPickEnv
from sim.mujoco.kinematics import grasp_point
from umi.camera_frames import gripper_camera_from_pinch, load_arcore_pinch_calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grasp-width-m", type=float, default=0.039)
    parser.add_argument("--object-height-m", type=float, default=0.10)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registration", type=Path, default=None,
                        help="optional optimize_real_umi_registration.py result")
    parser.add_argument("--rotate-real-deg", type=int, default=0,
                        help="display-only rotation for comparing a legacy episode to canonical images")
    args = parser.parse_args()

    ep = read_episode(args.episode)
    cfg = copy.deepcopy(load_config(args.config))
    # Only the measured/inferred grasp-axis width is evidence-backed.  The other
    # horizontal extent and height remain an explicit snack-case proxy.
    cfg["task"]["object"]["half_size_m"] = [
        args.grasp_width_m / 2.0, args.grasp_width_m / 2.0,
        args.object_height_m / 2.0,
    ]
    cfg["task"]["object"]["init_pos"][2] = args.object_height_m / 2.0 + 0.001
    # The pilot registration can put the inferred contact outside the nominal
    # 0.6m-wide evaluation table.  Keep the inferred object on support so its
    # absence from the image diagnoses camera geometry, not a falling object.
    cfg["task"]["table"]["half_size_m"][:2] = [0.40, 0.40]

    initial_q = denormalize(ep.state[0], cfg)
    gaps = (np.asarray(ep.state[:, 5], dtype=float) + 1.0) * 0.045
    candidates = np.flatnonzero(gaps <= args.grasp_width_m + 0.002)
    contact_index = int(candidates[0]) if candidates.size else int(np.argmin(gaps))
    contact_q = denormalize(ep.state[contact_index], cfg)
    optimized = None
    if args.registration is not None:
        registration = json.loads(args.registration.read_text(encoding="utf-8"))
        optimized = registration["selected"]
        _, t_camera_pinch = load_arcore_pinch_calibration(
            Path("configs/real/umi_s22_marker_midpoint_provisional.json"))
        if registration.get("camera_frame_interpretation") == "hardware_body":
            t_body_camera = np.eye(4)
            t_body_camera[:3, 3] = [0.0, 0.0, -0.080]
            t_body_camera = t_body_camera @ np.linalg.inv(t_camera_pinch)
        else:
            t_body_camera = gripper_camera_from_pinch(t_camera_pinch)
        cfg["cameras"]["cam_wrist"]["pos"] = t_body_camera[:3, 3].tolist()
        cfg["cameras"]["cam_wrist"]["xyaxes"] = t_body_camera[:3, :2].T.reshape(-1).tolist()
        initial_q[:5] = np.asarray(optimized["start_arm_rad"], dtype=float)
        contact_index = int(optimized["contact_index"])
        contact_q = np.asarray(optimized["contact_q_rad"], dtype=float)

    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.0) as env:
        env.data.qpos[:6] = contact_q
        mujoco.mj_forward(env.model, env.data)
        pinch = grasp_point(
            env.model, env.data, np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float))
        object_xy = ((float(optimized["object_xyz_m"][0]),
                      float(optimized["object_xyz_m"][1]))
                     if optimized is not None else (float(pinch[0]), float(pinch[1])))
        obs = env.reset(seed=0, object_xy=object_xy, initial_q_rad=initial_q)
        sim = np.transpose(obs.images["cam_wrist"], (1, 2, 0))
        env.data.qpos[:6] = contact_q
        mujoco.mj_forward(env.model, env.data)
        debug_camera = mujoco.MjvCamera()
        debug_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        debug_camera.lookat[:] = [object_xy[0], object_xy[1], args.object_height_m / 2]
        debug_camera.distance = 0.65
        debug_camera.azimuth = 135
        debug_camera.elevation = -25
        with mujoco.Renderer(env.model, height=480, width=640) as debug_renderer:
            debug_renderer.update_scene(env.data, camera=debug_camera)
            contact_external = debug_renderer.render().copy()

    real = np.transpose(ep.images["cam_wrist"][0], (1, 2, 0))
    left = Image.fromarray(real).rotate(args.rotate_real_deg, expand=False).resize(
        (448, 448), Image.Resampling.NEAREST)
    right = Image.fromarray(sim).resize((448, 448), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (896, 488), "white")
    canvas.paste(left, (0, 40))
    canvas.paste(right, (448, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "REAL UMI: first training observation", fill="black")
    draw.text((460, 12), "MUJOCO: aligned initial observation", fill="black")

    args.out.mkdir(parents=True, exist_ok=True)
    comparison = args.out / f"{args.episode.stem}_first_view_comparison.png"
    canvas.save(comparison)
    external_path = args.out / f"{args.episode.stem}_contact_external.png"
    Image.fromarray(contact_external).save(external_path)
    report = {
        "status": "PROVISIONAL_ALIGNMENT_FOR_VISUAL_COMPARISON_ONLY",
        "episode": args.episode.name,
        "initial_q_rad": initial_q.tolist(),
        "contact_index": contact_index,
        "contact_gap_m": (float(optimized["contact_gap_m"])
                          if optimized is not None else float(gaps[contact_index])),
        "object_xy_inferred_from_contact_pinch_m": list(object_xy),
        "registration_source": (str(args.registration) if args.registration is not None
                                else "episode_contact_fk"),
        "object_size_m": [args.grasp_width_m, args.grasp_width_m, args.object_height_m],
        "object_size_evidence": {
            "grasp_width": "inferred_from_stable_contact_gap",
            "other_horizontal_extent": "provisional_equal_to_grasp_width",
            "height": "caller-supplied provisional estimate; see registration evidence",
        },
        "table_half_size_xy_m_for_alignment": [0.40, 0.40],
        "comparison_image": comparison.name,
        "contact_external_image": external_path.name,
    }
    report_path = args.out / f"{args.episode.stem}_alignment.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
