"""Render one real-UMI trajectory through SO-101 control without policy training.

This is a diagnostic replay. Original pose IK is preferred; position-only IK is
an explicit fallback so coordinate/control failures can be watched before any
BC model is trained. The fallback trajectory is never marked training-ready.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from paths import DEFAULT_CONFIG, DEFAULT_SCENE
from sim.mujoco.build_scene import build_model, load_config
from sim.mujoco.kinematics import approach_axis, grasp_point
from sim.mujoco.grasp import contact_on_object, gripper_geom_ids
from tools.diagnose_real_umi_ik import (solve_position_only, vector_angle_deg)
from tools.umi_mujoco import MujocoIK
from tools.umi_real_placement_study import (REAL_J3_LOWER_RAD, load_relative_run,
                                            register_relative)
from umi.camera_frames import load_arcore_pinch_calibration, rigid
from umi.convert import invert_gap_curve
from umi.episode_physics import load_episode_physics
from umi.ik import matrix_to_quat


def save_frames(frames, out_stem: Path, fps: int) -> Path:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        for suffix, kwargs in ((".mp4", {"fps": fps}),
                               (".gif", {"fps": fps, "loop": 0})):
            path = out_stem.with_suffix(suffix)
            try:
                iio.imwrite(path, np.stack(frames), **kwargs)
                return path
            except Exception:
                continue
    except ImportError:
        pass
    try:
        from PIL import Image
        path = out_stem.with_suffix(".gif")
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=round(1000 / fps), loop=0)
        return path
    except (ImportError, OSError):
        pass
    path = out_stem.with_suffix(".frames.npz")
    np.savez_compressed(path, frames=np.stack(frames))
    return path


def build_commands(model, cfg, targets, gaps, start_arm):
    ik = MujocoIK(model, cfg)
    q_prev = np.r_[np.asarray(start_arm, dtype=float), 0.60]
    commands, modes, orientation_loss = [], [], []
    gap_clamped = 0
    curve = cfg["grasp"]["gap_curve"]
    min_gap, max_gap = curve[0][1] / 100.0, curve[-1][1] / 100.0
    for target, gap in zip(targets, gaps):
        full = ik.solve(target[:3, 3], matrix_to_quat(target[:3, :3]),
                        float(gap), q_init=q_prev)
        real_limits = full.within_limits and full.q_rad[2] >= REAL_J3_LOWER_RAD
        if full.pos_error_m <= 5e-3 and full.axis_error_deg <= 5.0 and real_limits:
            q_arm = full.q_rad[:5].copy()
            mode = "full_pose"
            loss = full.axis_error_deg
        else:
            solved, _, _ = solve_position_only(model, target[:3, 3], ik.pinch, q_prev)
            q_arm = solved[:5]
            data = mujoco.MjData(model)
            data.qpos[:6] = solved
            mujoco.mj_forward(model, data)
            loss = vector_angle_deg(approach_axis(model, data), target[:3, 2])
            mode = "position_only_fallback"
        bounded_gap = float(np.clip(gap, min_gap, max_gap))
        gap_clamped += int(bounded_gap != float(gap))
        gripper = invert_gap_curve(bounded_gap, curve)
        q_prev = np.r_[q_arm, gripper]
        commands.append(q_prev.copy())
        modes.append(mode)
        orientation_loss.append(loss)
    return np.stack(commands), modes, orientation_loss, gap_clamped


def rate_limit_commands(commands, max_step):
    """Causal per-joint command limiter used only for diagnostic simulation."""
    source = np.asarray(commands, dtype=float)
    limits = np.asarray(max_step, dtype=float)
    if source.ndim != 2 or source.shape[1] != 6 or limits.shape != (6,):
        raise ValueError("commands must be (T,6) and max_step must be (6,)")
    if not np.isfinite(source).all() or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("commands and limits must be finite; limits must be positive")
    result = source.copy()
    for index in range(1, len(result)):
        result[index] = result[index - 1] + np.clip(
            source[index] - result[index - 1], -limits, limits)
    return result


def time_scale_commands(commands, max_step):
    """Insert linear joint waypoints so every step respects the supplied bound."""
    source = np.asarray(commands, dtype=float)
    limits = np.asarray(max_step, dtype=float)
    if source.ndim != 2 or source.shape[1] != 6 or limits.shape != (6,):
        raise ValueError("commands must be (T,6) and max_step must be (6,)")
    if not np.isfinite(source).all() or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("commands and limits must be finite; limits must be positive")
    if len(source) == 0:
        return source.copy(), np.empty(0, dtype=int)
    result = [source[0].copy()]
    source_waypoint = [0]
    for index in range(1, len(source)):
        delta = source[index] - source[index - 1]
        subdivisions = max(1, int(np.ceil(np.max(np.abs(delta) / limits))))
        for step in range(1, subdivisions + 1):
            result.append(source[index - 1] + delta * (step / subdivisions))
            source_waypoint.append(index)
    return np.stack(result), np.asarray(source_waypoint, dtype=int)


def command_pose_errors(model, commands, targets, pinch):
    data = mujoco.MjData(model)
    position_mm, axis_deg = [], []
    for command, target in zip(commands, targets):
        data.qpos[:6] = command
        mujoco.mj_forward(model, data)
        position_mm.append(float(np.linalg.norm(
            grasp_point(model, data, pinch) - target[:3, 3])) * 1000.0)
        axis_deg.append(vector_angle_deg(approach_axis(model, data), target[:3, 2]))
    return position_mm, axis_deg


def completed_close_index(gaps, completion_fraction=0.1):
    """Return the first post-transition frame near the completed closing plateau."""
    gaps = np.asarray(gaps, dtype=float)
    if len(gaps) < 2 or not 0.0 <= completion_fraction <= 1.0:
        raise ValueError("gaps need length >= 2 and completion_fraction must be in [0,1]")
    transition = int(np.argmin(np.diff(gaps)) + 1)
    post = gaps[transition:]
    closed = float(np.min(post))
    opened = float(np.max(gaps[:transition]))
    threshold = closed + completion_fraction * max(0.0, opened - closed)
    reached = np.flatnonzero(post <= threshold)
    return transition + (int(reached[0]) if len(reached) else 0)


def register_close_to_object(relative, gaps, object_xyz, grasp_z_offset_m, yaw_deg=0.0,
                             object_center_from_pinch_m=None, grasp_rotation_base=None):
    """Register the completed closing plateau to a vertical simulated grasp."""
    if len(relative) != len(gaps) or len(gaps) < 2:
        raise ValueError("relative poses and gaps must have the same length >= 2")
    close_index = completed_close_index(gaps)
    desired = np.eye(4)
    if grasp_rotation_base is None:
        # Legacy top-grasp default. New real episodes provide an explicit frame.
        yaw = np.deg2rad(float(yaw_deg))
        jaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        approach = np.array([0.0, 0.0, -1.0])
        middle = np.cross(approach, jaw)
        desired[:3, :3] = np.column_stack([jaw, middle, approach])
    else:
        rotation = np.asarray(grasp_rotation_base, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
            raise ValueError("grasp_rotation_base must be a proper 3x3 rotation")
        desired[:3, :3] = rotation
    object_xyz = np.asarray(object_xyz, dtype=float)
    offset = (np.zeros(3) if object_center_from_pinch_m is None
              else np.asarray(object_center_from_pinch_m, dtype=float))
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("object_center_from_pinch_m must be a finite 3-vector")
    desired[:3, 3] = object_xyz
    if object_center_from_pinch_m is None:
        desired[2, 3] += float(grasp_z_offset_m)
    else:
        # observed object centre = pinch position + R_pinch @ local offset
        desired[:3, 3] -= desired[:3, :3] @ offset
    anchor = desired @ np.linalg.inv(relative[close_index])
    return np.einsum("ij,tjk->tik", anchor, relative), close_index


def collision_counts(model, data, object_geom, table_geom):
    table_robot = self_collision = 0
    for index in range(data.ncon):
        geom1, geom2 = int(data.contact[index].geom1), int(data.contact[index].geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        if table_geom in (geom1, geom2):
            other = geom2 if geom1 == table_geom else geom1
            if other != object_geom and model.geom_bodyid[other] != 0:
                table_robot += 1
        if (body1 != 0 and body2 != 0 and body1 != body2
                and object_geom not in (geom1, geom2)):
            self_collision += 1
    return table_robot, self_collision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="output stem; writes video/frames and .json")
    parser.add_argument("--camera", default="cam_wrist")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-arm-step-rad", type=float, default=None,
                        help="diagnostic causal rate limit; not a verified hardware limit")
    parser.add_argument("--max-gripper-step-rad", type=float, default=None)
    parser.add_argument("--time-scale", action="store_true",
                        help="insert waypoints instead of causally clipping commands")
    parser.add_argument("--register-close-to-object", action="store_true",
                        help="anchor the strongest gap-closing frame to the nominal object grasp")
    parser.add_argument("--object-x", type=float, default=None)
    parser.add_argument("--object-y", type=float, default=None)
    parser.add_argument("--object-half-size-m", type=float, nargs=3, default=None,
                        metavar=("HX", "HY", "HZ"),
                        help="provisional MuJoCo box half-size override")
    parser.add_argument("--grasp-height-m", type=float, default=None,
                        help="provisional pinch height above the table (z=0)")
    parser.add_argument("--grasp-yaw-deg", type=float, default=0.0)
    parser.add_argument("--episode-physics", type=Path, default=None,
                        help="per-episode object type, full size and grasp-height metadata")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    report_path = args.out.with_suffix(".json")
    if report_path.exists():
        raise SystemExit(f"output already exists: {report_path}")

    extrinsic, t_camera_pinch = load_arcore_pinch_calibration(args.extrinsic)
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    physics_meta = (load_episode_physics(args.episode_physics, args.bundle.stem)
                    if args.episode_physics is not None else None)
    if physics_meta is not None and (args.object_half_size_m is not None
                                     or args.grasp_height_m is not None):
        raise SystemExit("episode physics metadata cannot be mixed with object size/height overrides")
    relative, gaps, source_rows = load_relative_run(
        args.bundle, quality, t_camera_pinch)
    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    if physics_meta is not None:
        half_size = physics_meta.half_size_m
        model.geom_size[object_geom] = half_size
    elif args.object_half_size_m is not None:
        half_size = np.asarray(args.object_half_size_m, dtype=float)
        if np.any(half_size <= 0):
            raise SystemExit("object half sizes must be positive")
        model.geom_size[object_geom] = half_size
    else:
        half_size = model.geom_size[object_geom].copy()
    close_index = None
    if args.register_close_to_object:
        object_xyz = np.asarray(cfg["task"]["object"]["init_pos"], dtype=float).copy()
        if args.object_x is not None:
            object_xyz[0] = args.object_x
        if args.object_y is not None:
            object_xyz[1] = args.object_y
        # A free object's qpos stores its centre. Keep its bottom on z=0 when its
        # provisional geometry is overridden.
        object_xyz[2] = float(half_size[2])
        grasp_height_m = (physics_meta.grasp_height_m if physics_meta is not None
                          else args.grasp_height_m)
        grasp_offset = (float(grasp_height_m) - object_xyz[2]
                        if grasp_height_m is not None
                        else float(cfg["grasp"]["grasp_z_offset_m"]))
        targets, close_index = register_close_to_object(
            relative, gaps, object_xyz, grasp_offset,
            args.grasp_yaw_deg,
            (physics_meta.object_center_from_pinch_m if physics_meta is not None else None),
            (physics_meta.grasp_rotation_base if physics_meta is not None else None))
        registration = "completed closing plateau -> provisional object grasp pose"
    else:
        targets = register_relative(relative, rigid(placement["selected"]["start_pose"]))
        registration = "first valid frame -> selected SO101 start pose"
    raw_commands, modes, losses, gap_clamped = build_commands(
        model, cfg, targets, gaps, placement["selected"]["start_arm_rad"])
    commands = raw_commands
    source_waypoint = np.arange(len(raw_commands), dtype=int)
    limiter = None
    if args.max_arm_step_rad is not None or args.max_gripper_step_rad is not None:
        if args.max_arm_step_rad is None or args.max_gripper_step_rad is None:
            raise SystemExit("both max step arguments are required together")
        max_step = np.array([args.max_arm_step_rad] * 5 + [args.max_gripper_step_rad])
        if args.time_scale:
            commands, source_waypoint = time_scale_commands(raw_commands, max_step)
            method = "linear_time_scaling"
        else:
            commands = rate_limit_commands(raw_commands, max_step)
            method = "causal_rate_limit"
        limiter = {"status": "simulation_provisional", "method": method,
                   "max_step": max_step.tolist()}
    pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    # Retargeting fidelity is judged at original waypoints. Inserted time-scaling
    # points intentionally have no new camera/EEF observation attached to them.
    pose_position_mm, pose_axis_deg = command_pose_errors(model, raw_commands, targets, pinch)

    camera_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
                    for index in range(model.ncam)]
    if args.camera not in camera_names:
        raise SystemExit(f"camera {args.camera!r} not found; available={camera_names}")
    data = mujoco.MjData(model)
    data.qpos[:6] = commands[0]
    data.ctrl[:6] = commands[0]
    if args.register_close_to_object:
        object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
        object_qpos = model.jnt_qposadr[object_joint]
        data.qpos[object_qpos:object_qpos + 3] = object_xyz
    mujoco.mj_forward(model, data)
    for _ in range(max(1, int(0.4 / model.opt.timestep))):
        mujoco.mj_step(model, data)
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    jaws = gripper_geom_ids(model)
    object_z0 = float(data.xpos[object_body, 2])
    substeps = max(1, int((1.0 / args.fps) / model.opt.timestep))
    frames, tracking_errors = [], []
    object_lifts, jaw_contacts, jaw_forces = [], [], []
    table_robot_contacts = self_contacts = 0
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        for command in commands:
            data.ctrl[:6] = command
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            tracking_errors.append(np.abs(data.qpos[:6] - command))
            object_lifts.append(float(data.xpos[object_body, 2]) - object_z0)
            force, contacts = contact_on_object(model, data, object_geom, jaws)
            jaw_forces.append(force); jaw_contacts.append(contacts)
            table_count, self_count = collision_counts(model, data, object_geom, table_geom)
            table_robot_contacts += table_count
            self_contacts += self_count
            renderer.update_scene(data, camera=args.camera)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    media = save_frames(frames, args.out, args.fps)
    tracking = np.stack(tracking_errors)
    raw_step = np.abs(np.diff(raw_commands, axis=0)) if len(commands) > 1 else np.zeros((0, 6))
    step = np.abs(np.diff(commands, axis=0)) if len(commands) > 1 else np.zeros((0, 6))
    result = {
        "purpose": "diagnostic_trajectory_replay_without_policy_training",
        "training_ready": False,
        "episode_id": args.bundle.stem,
        "registration": registration,
        "object_model": {
            "object_type": (physics_meta.object_type if physics_meta is not None
                            else "unspecified_proxy"),
            "shape": "box",
            "size_m": (half_size * 2.0).tolist(),
            "half_size_m": half_size.tolist(),
            "grasp_height_above_table_m": (physics_meta.grasp_height_m
                                             if physics_meta is not None else
                                             args.grasp_height_m if args.grasp_height_m is not None
                                             else (float(half_size[2]) +
                                                   float(cfg["grasp"]["grasp_z_offset_m"]))),
            "evidence": (physics_meta.evidence if physics_meta is not None else "provisional"),
            "source": (physics_meta.source if physics_meta is not None
                       else "command-line override or default MuJoCo scene"),
            "object_center_from_pinch_m": (list(physics_meta.object_center_from_pinch_m)
                                            if physics_meta is not None else [0.0, 0.0, 0.0]),
            "registration_evidence": (physics_meta.registration_evidence
                                       if physics_meta is not None else "provisional"),
            "grasp_rotation_base": (physics_meta.grasp_rotation_base
                                     if physics_meta is not None else None),
        },
        "closing_source_index": close_index,
        "frames": len(commands), "source_frames": len(raw_commands),
        "duration_s": len(commands) / args.fps,
        "source_duration_s": len(raw_commands) / args.fps,
        "source_rows": source_rows,
        "full_pose_frames": modes.count("full_pose"),
        "position_only_fallback_frames": modes.count("position_only_fallback"),
        "full_pose_rate": modes.count("full_pose") / len(modes),
        "fallback_orientation_loss_deg": {
            "p50": float(np.percentile([loss for loss, mode in zip(losses, modes)
                                         if mode != "full_pose"], 50)),
            "p95": float(np.percentile([loss for loss, mode in zip(losses, modes)
                                         if mode != "full_pose"], 95)),
        },
        "gap_clamped_to_mujoco_curve_frames": gap_clamped,
        "rate_limiter": limiter,
        "source_waypoint_index": source_waypoint.tolist(),
        "raw_max_command_step": np.max(raw_step, axis=0).tolist() if len(raw_step) else [0.0] * 6,
        "max_command_step": np.max(step, axis=0).tolist() if len(step) else [0.0] * 6,
        "command_pose_error": {
            "position_mm_p50": float(np.percentile(pose_position_mm, 50)),
            "position_mm_p95": float(np.percentile(pose_position_mm, 95)),
            "axis_deg_p50": float(np.percentile(pose_axis_deg, 50)),
            "axis_deg_p95": float(np.percentile(pose_axis_deg, 95)),
        },
        "tracking_error_rad_or_gripper_joint": {
            "p95_per_joint": np.percentile(tracking, 95, axis=0).tolist(),
            "max_per_joint": np.max(tracking, axis=0).tolist(),
        },
        "physics": {
            "max_object_lift_m": max(object_lifts),
            "end_object_lift_m": object_lifts[-1],
            "ticks_with_jaw_object_contact": int(sum(value > 0 for value in jaw_contacts)),
            "max_jaw_object_contact_count": max(jaw_contacts),
            "max_jaw_object_normal_force_n": max(jaw_forces),
            "table_robot_contact_count_sum": table_robot_contacts,
            "self_collision_count_sum": self_contacts,
            "pick_success": bool(object_lifts[-1] >= cfg["grasp"]["success_lift_m"]
                                 and jaw_contacts[-1] > 0),
        },
        "media": str(media),
        "warnings": [
            "Position-only fallback changes grasp orientation and is not a training label.",
            ("The grasp frame was inferred from the completed closing plateau; "
             "the object itself has no measured 3-D label in the S22 recording."
             if args.register_close_to_object else
             "Object placement is the nominal MuJoCo scene, not registered from the S22 video."),
            "This replay checks plumbing and motion shape, not task success or policy performance.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
