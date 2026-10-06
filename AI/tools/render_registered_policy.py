"""Render a learned/replay policy at an episode's inferred registered condition."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from contract.episode import read_episode
from policy.baselines import ReplayPolicy
from sim.mujoco.build_scene import denormalize, load_config
from sim.mujoco.env import MujocoPickEnv
from sim.mujoco.kinematics import grasp_point
from tools.replay_real_umi_trajectory import save_frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy-ckpt", type=Path)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--ticks", type=int, default=250)
    ap.add_argument("--object-height-m", type=float, default=0.0767)
    ap.add_argument("--grasp-width-m", type=float, default=0.039)
    args = ap.parse_args()
    if args.replay == (args.policy_ckpt is not None):
        raise SystemExit("choose exactly one of --replay or --policy-ckpt")

    ep = read_episode(args.episode)
    cfg = copy.deepcopy(load_config())
    cfg["task"]["object"]["half_size_m"] = [args.grasp_width_m / 2] * 2 + [
        args.object_height_m / 2]
    cfg["task"]["object"]["init_pos"][2] = args.object_height_m / 2 + 0.001
    cfg["task"]["table"]["half_size_m"][:2] = [0.40, 0.40]

    initial_q = denormalize(ep.state[0], cfg)
    gaps = (np.asarray(ep.state[:, 5], dtype=float) + 1.0) * 0.045
    contact = np.flatnonzero(gaps <= args.grasp_width_m + 0.002)
    contact_index = int(contact[0]) if contact.size else int(np.argmin(gaps))

    with MujocoPickEnv(cfg, render=True, object_jitter_m=0, max_ticks=args.ticks) as env:
        env.data.qpos[:6] = denormalize(ep.state[contact_index], cfg)
        mujoco.mj_forward(env.model, env.data)
        object_xy = grasp_point(env.model, env.data,
                                np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float))[:2]
        if args.replay:
            policy = ReplayPolicy(ep.action, source=args.episode.name)
            label = "v5 trajectory replay"
        else:
            from policy.bc import BCPolicy
            policy = BCPolicy(args.policy_ckpt, device="cpu")
            label = args.policy_ckpt.stem
        obs = env.reset(seed=0, object_xy=tuple(object_xy), initial_q_rad=initial_q)
        policy.reset(seed=0)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [object_xy[0], object_xy[1], args.object_height_m / 2]
        camera.distance = 0.65
        camera.azimuth = 135
        camera.elevation = -25
        frames = []
        object_geom = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
        table_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        object_start = env.object_position().copy()
        first_object_contact = None
        max_arm_tracking_error = 0.0
        closest = (float("inf"), -1)
        with mujoco.Renderer(env.model, height=448, width=448) as renderer:
            for tick in range(args.ticks):
                wrist = np.transpose(obs.images["cam_wrist"], (1, 2, 0))
                wrist = np.asarray(Image.fromarray(wrist).resize((448, 448)))
                renderer.update_scene(env.data, camera=camera)
                external = renderer.render().copy()
                canvas = Image.new("RGB", (896, 484), "white")
                canvas.paste(Image.fromarray(wrist), (0, 36))
                canvas.paste(Image.fromarray(external), (448, 36))
                xy, d3 = env.pinch_to_object_m()
                ImageDraw.Draw(canvas).text(
                    (8, 8), f"{label}  tick={tick:03d}  xy={xy*1000:.1f}mm  "
                    f"3d={d3*1000:.1f}mm  contacts={env.jaw_contacts()}", fill="black")
                frames.append(np.asarray(canvas))
                action = policy.act(obs)
                commanded = denormalize(action, cfg)
                obs = env.step(action)
                max_arm_tracking_error = max(
                    max_arm_tracking_error,
                    float(np.max(np.abs(env.joint_positions()[:5] - commanded[:5]))))
                xy_after, _ = env.pinch_to_object_m()
                if xy_after < closest[0]:
                    closest = (xy_after, tick + 1)
                if first_object_contact is None:
                    for ci in range(env.data.ncon):
                        con = env.data.contact[ci]
                        if object_geom not in (int(con.geom1), int(con.geom2)):
                            continue
                        other = int(con.geom2 if int(con.geom1) == object_geom else con.geom1)
                        if other == table_geom:
                            continue
                        first_object_contact = {
                            "tick": tick + 1,
                            "other_geom": mujoco.mj_id2name(
                                env.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                            "gripper_command_normalized": float(action[5]),
                            "pinch_xy_mm": float(xy_after * 1000),
                        }
                        bid = mujoco.mj_name2id(
                            env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
                        body_rot = env.data.xmat[bid].reshape(3, 3)
                        pinch_world = grasp_point(
                            env.model, env.data,
                            np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float))
                        object_world = env.object_position()
                        first_object_contact["object_minus_pinch_world_m"] = (
                            object_world - pinch_world).tolist()
                        first_object_contact["object_minus_pinch_gripper_local_m"] = (
                            body_rot.T @ (object_world - pinch_world)).tolist()
                        break
        output = save_frames(frames, args.out, int(cfg["control"]["rate_hz"]))
        print(f"video={output}")
        print(f"object_xy={object_xy.tolist()} contact_index={contact_index} "
              f"lift_m={env.lift_height():.5f} success={env.is_success()}")
        print(f"first_object_contact={first_object_contact}")
        print(f"closest_xy_mm={closest[0]*1000:.2f} at_tick={closest[1]} "
              f"max_arm_tracking_error_rad={max_arm_tracking_error:.5f} "
              f"object_displacement_m={(env.object_position()-object_start).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
