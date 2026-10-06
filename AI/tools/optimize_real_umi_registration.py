"""Search a provisional SO-101 registration for the upright snack-case pilot.

Objectives are explicit and inspectable: reachable relative motion, horizontal
jaw and side-approach axes at contact, object supported by the table, and the
inferred object centre inside the first wrist-camera view.  This does not turn
monocular pilot data into a measured robot/world calibration.
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

from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import build_model, load_config
from sim.mujoco.kinematics import grasp_point
from tools.diagnose_real_umi_ik import solve_position_only
from tools.replay_real_umi_trajectory import collision_counts
from tools.umi_mujoco import MujocoIK, eef_pose_from_joints
from tools.umi_real_placement_study import load_relative_run, register_relative
from umi.camera_frames import gripper_camera_from_pinch, load_arcore_pinch_calibration
from umi.ik import matrix_to_quat, quat_to_matrix


def camera_metrics(model, data, object_xyz):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    camera_xyz = data.cam_xpos[cid]
    camera_rot = data.cam_xmat[cid].reshape(3, 3)
    local = camera_rot.T @ (object_xyz - camera_xyz)
    depth = -float(local[2])
    if depth <= 1e-6:
        return False, 1e3, local.tolist(), camera_xyz.tolist()
    # Vertical fovy and square training images make both normalized coordinates
    # use the same tangent bound.
    half = np.tan(np.deg2rad(float(model.cam_fovy[cid])) / 2.0)
    uv = np.asarray([local[0], local[1]]) / (depth * half)
    return (bool(np.max(np.abs(uv)) < 0.85), float(np.linalg.norm(uv)),
            local.tolist(), camera_xyz.tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--extrinsic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidates", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--grasp-width-m", type=float, default=0.039)
    ap.add_argument("--object-height-m", type=float, default=0.10)
    ap.add_argument("--grasp-height-m", type=float, default=0.08,
                    help="provisional pinch height above table for the upright case")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--camera-frame-interpretation",
                    choices=("contract_pinch", "hardware_body"),
                    default="contract_pinch")
    ap.add_argument("--refine-from", type=Path, default=None,
                    help="search locally around a previous selected candidate")
    args = ap.parse_args()
    quality = json.loads(args.bundle.with_suffix(".quality.json").read_text(encoding="utf-8"))
    _, tcp = load_arcore_pinch_calibration(args.extrinsic)
    relative, gaps, _ = load_relative_run(args.bundle, quality, tcp)
    contact_candidates = np.flatnonzero(gaps <= args.grasp_width_m + 0.002)
    contact_index = int(contact_candidates[0]) if contact_candidates.size else int(np.argmin(gaps))

    cfg = copy.deepcopy(load_config(args.config))
    if args.camera_frame_interpretation == "hardware_body":
        t_body_camera = np.eye(4)
        t_body_camera[:3, 3] = [0.0, 0.0, -0.080]
        t_body_camera = t_body_camera @ np.linalg.inv(tcp)
    else:
        t_body_camera = gripper_camera_from_pinch(tcp)
    cfg["cameras"]["cam_wrist"]["pos"] = t_body_camera[:3, 3].tolist()
    cfg["cameras"]["cam_wrist"]["xyaxes"] = t_body_camera[:3, :2].T.reshape(-1).tolist()
    cfg["task"]["object"]["half_size_m"] = [args.grasp_width_m / 2] * 2 + [args.object_height_m / 2]
    cfg["task"]["object"]["init_pos"][2] = args.object_height_m / 2 + 0.001
    cfg["task"]["table"]["half_size_m"][:2] = [0.40, 0.40]
    model = build_model(cfg)
    ik = MujocoIK(model, cfg)
    data = mujoco.MjData(model)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    lo, hi = ik.ranges[:5, 0].copy(), ik.ranges[:5, 1].copy()
    lo[2] = max(lo[2], -1.57079632679)
    margin = 0.08 * (hi - lo)
    rng = np.random.default_rng(args.seed)
    if args.refine_from is None:
        candidates = [np.zeros(5)] + [
            rng.uniform(lo + margin, hi - margin) for _ in range(args.candidates - 1)]
    else:
        centre = np.asarray(json.loads(args.refine_from.read_text(encoding="utf-8"))
                            ["selected"]["start_arm_rad"], dtype=float)
        sigma = 0.12 * (hi - lo)
        candidates = [centre] + [
            np.clip(centre + rng.normal(0.0, sigma), lo + margin, hi - margin)
            for _ in range(args.candidates - 1)]
    sampled = np.unique(np.r_[np.linspace(0, contact_index, 12, dtype=int), contact_index])
    rows = []
    for candidate_id, arm in enumerate(candidates):
        q_start = np.r_[arm, 0.60]
        start_pos, start_quat = eef_pose_from_joints(model, data, q_start, ik.pinch)
        start_pose = np.eye(4); start_pose[:3, :3] = quat_to_matrix(start_quat); start_pose[:3, 3] = start_pos
        targets = register_relative(relative, start_pose)
        q = q_start.copy(); errors = []; contact_q = None
        for index in sampled:
            q, error, hit = solve_position_only(model, targets[index, :3, 3], ik.pinch, q)
            errors.append(error + (0.05 if hit else 0.0))
            if index == contact_index:
                contact_q = q.copy()
        assert contact_q is not None
        data.qpos[:6] = contact_q; mujoco.mj_forward(model, data)
        contact_table_contacts, contact_self_contacts = collision_counts(
            model, data, object_geom, table_geom)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        rot = data.xmat[bid].reshape(3, 3)
        jaw_vertical = abs(float(rot[2, 0]))
        approach_vertical = abs(float((-rot[:, 2])[2]))
        pinch = grasp_point(model, data, ik.pinch)
        object_xyz = np.asarray([pinch[0], pinch[1], args.object_height_m / 2])
        grasp_height_error_m = abs(float(pinch[2]) - args.grasp_height_m)
        on_table = bool(abs(object_xyz[0] - cfg["task"]["table"]["pos"][0]) < 0.37
                        and abs(object_xyz[1]) < 0.37)
        data.qpos[:6] = q_start; mujoco.mj_forward(model, data)
        initial_table_contacts, initial_self_contacts = collision_counts(
            model, data, object_geom, table_geom)
        visible, centre_error, object_camera, camera_world = camera_metrics(
            model, data, object_xyz)
        camera_depth_m = -float(object_camera[2])
        depth_error_m = abs(camera_depth_m - 0.20)
        unobstructed_height = float(camera_world[2]) >= args.object_height_m + 0.03
        reach_p95_mm = float(np.percentile(errors, 95) * 1000)
        feasible = (reach_p95_mm <= 5.0 and jaw_vertical <= 0.25
                    and approach_vertical <= 0.35 and on_table and visible
                    and unobstructed_height and centre_error <= 0.60
                    and 0.12 <= camera_depth_m <= 0.30
                    and grasp_height_error_m <= 0.02
                    and initial_table_contacts == 0 and initial_self_contacts == 0
                    and contact_table_contacts == 0 and contact_self_contacts == 0)
        score = ((0 if feasible else 1000) + reach_p95_mm + 80 * jaw_vertical
                 + 60 * approach_vertical + 50 * centre_error + 250 * depth_error_m)
        score += 500 * grasp_height_error_m
        score += 100 * (initial_table_contacts + initial_self_contacts
                        + contact_table_contacts + contact_self_contacts)
        rows.append({
            "candidate_id": candidate_id, "score": score, "feasible": feasible,
            "start_arm_rad": arm.tolist(), "start_pose": start_pose.tolist(),
            "contact_index": contact_index, "contact_gap_m": float(gaps[contact_index]),
            "contact_q_rad": contact_q.tolist(),
            "contact_pinch_xyz_m": pinch.tolist(), "object_xyz_m": object_xyz.tolist(),
            "grasp_height_target_m": args.grasp_height_m,
            "grasp_height_error_m": grasp_height_error_m,
            "reach_p95_mm": reach_p95_mm, "jaw_vertical_abs": jaw_vertical,
            "approach_vertical_abs": approach_vertical, "object_on_table": on_table,
            "first_view_visible": visible, "first_view_centre_error_norm": centre_error,
            "first_view_depth_m": camera_depth_m,
            "first_view_depth_target_m": 0.20,
            "object_in_initial_camera_xyz_m": object_camera,
            "initial_camera_world_xyz_m": camera_world,
            "camera_above_object_clearance": unobstructed_height,
            "initial_table_contacts": initial_table_contacts,
            "initial_self_contacts": initial_self_contacts,
            "contact_table_contacts": contact_table_contacts,
            "contact_self_contacts": contact_self_contacts,
        })
    rows.sort(key=lambda row: row["score"])
    result = {
        "status": "PROVISIONAL_REGISTRATION_SEARCH_NOT_PHYSICAL_CALIBRATION",
        "camera_frame_interpretation": args.camera_frame_interpretation,
        "constraints": {"reach_p95_mm_max": 5.0, "jaw_vertical_abs_max": 0.25,
                        "approach_vertical_abs_max": 0.35, "first_view_margin": 0.85,
                        "first_view_centre_error_norm_max": 0.60,
                        "first_view_depth_m": [0.12, 0.30],
                        "first_view_depth_target_m": 0.20,
                        "camera_world_z_min_m": args.object_height_m + 0.03,
                        "grasp_height_target_m": args.grasp_height_m,
                        "grasp_height_error_m_max": 0.02,
                        "initial_and_contact_table_self_contacts": 0},
        "evaluated_candidates": len(rows),
        "feasible_candidates": sum(r["feasible"] for r in rows),
        "selected": rows[0], "top10": rows[:10],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated": len(rows), "feasible": result["feasible_candidates"],
                      "selected": rows[0]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
