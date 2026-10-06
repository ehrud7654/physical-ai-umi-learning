"""Render every camera at rest and at a pre-grasp pose to verify the scene.
휴식자세와 파지직전 자세에서 모든 카메라를 렌더해 씬이 쓸 만한지 확인한다.

This is a measurement tool, not a data collector. It answers two questions:
does each camera actually see the workspace, and is the pre-grasp pose
reachable within joint limits?
이건 계측 도구지 수집기가 아니다. 답하는 질문은 둘이다 —
각 카메라가 작업공간을 실제로 보는가, 파지직전 자세가 관절한계 안에서 도달 가능한가?

Headless rendering needs an EGL context: MUJOCO_GL=egl python render_check.py
헤드리스 렌더링에는 EGL이 필요하다: MUJOCO_GL=egl python render_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from sim.mujoco.build_scene import build_model, dls_ik, load_config, normalize, verify_against_config


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    """Step the sim so the free-floating object rests on the table.
    자유물체가 테이블에 안착하도록 시뮬을 진행한다."""
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def shoot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    out_dir: Path,
    tag: str,
) -> None:
    """Render one frame per camera and write it to disk.
    카메라마다 한 프레임씩 렌더해 파일로 쓴다."""
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        renderer.update_scene(data, camera=name)
        frame = renderer.render()
        path = out_dir / f"{tag}_{name}.png"
        iio.imwrite(path, frame)
        print(f"  {tag}/{name}: {frame.shape} {frame.dtype} -> {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument("--approach-height", type=float, default=0.06,
                        help="pre-grasp height above the object top / 물체 윗면 위 접근 높이 (m)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    width, height = cfg["cameras"]["resolution"]
    model = build_model(cfg)
    data = mujoco.MjData(model)

    problems = verify_against_config(model, cfg)
    print("config vs mjcf mismatches:", problems or "none")

    mujoco.mj_forward(model, data)
    settle(model, data, args.settle_sec)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_xyz = data.xpos[obj_id].copy()
    half_z = float(cfg["task"]["object"]["half_size_m"][2])
    pre_grasp = obj_xyz + np.array([0.0, 0.0, half_z + args.approach_height])

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        print("[rest pose / 휴식자세]")
        shoot(model, data, renderer, args.out, "rest")
        q_rest = data.qpos[:6].copy()

        q_pre, err, within = dls_ik(model, data, pre_grasp)
        print(f"[pre-grasp / 파지직전] target={np.array2string(pre_grasp, precision=4)}")
        print(f"  ik position error: {err * 1000:.2f} mm   within joint limits: {within}")
        shoot(model, data, renderer, args.out, "pregrasp")

    print("rest    qpos[:6] rad:", np.array2string(q_rest, precision=4))
    print("rest    normalized  :", np.array2string(normalize(q_rest, cfg), precision=4))
    print("pregrasp qpos[:6] rad:", np.array2string(q_pre, precision=4))
    print("pregrasp normalized  :", np.array2string(normalize(q_pre, cfg), precision=4))
    print("object xpos:", np.array2string(obj_xyz, precision=4))


if __name__ == "__main__":
    main()
