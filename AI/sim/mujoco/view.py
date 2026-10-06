"""Open the MuJoCo viewer and watch the scene, or watch a policy run in real time.
MuJoCo 뷰어를 열어 씬을 보거나, 정책이 실시간으로 도는 것을 본다.

This is for looking, not for measuring. Numbers come from `tools/grasp_check.py`
and `tools/eval_rollout.py` — a viewer runs at wall-clock speed and whatever you
see here is one episode, not a success rate.
이건 **보는** 도구이고 재는 도구가 아니다. 수치는 `tools/grasp_check.py` 와
`tools/eval_rollout.py` 에서 나온다. 뷰어는 실시간으로 돌고, 여기서 보이는 것은
에피소드 하나이지 성공률이 아니다.

Requires a display. It will not work over EGL headless — that is the opposite
setting from the measurement tools.
디스플레이가 필요하다. EGL 헤드리스에서는 안 된다 — 계측 도구와 반대 설정이다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import mujoco
import numpy as np

from pathlib import Path

from paths import DEFAULT_CONFIG, DEFAULT_SCENE
from policy.base import Policy
from policy.baselines import HoldPolicy, ReplayPolicy, ScriptedPickPolicy, ZeroPolicy
from sim.mujoco.build_scene import build_model, load_config
from sim.mujoco.env import MujocoPickEnv

POLICY_NAMES = ("none", "scripted", "hold", "zero", "replay", "bc")


def _check_display() -> None:
    """Fail early and clearly when there is no display to draw into.
    그릴 화면이 없을 때 일찍, 분명하게 실패한다.

    `MUJOCO_GL=egl` is what the measurement tools need and exactly what the
    viewer cannot use. Getting this wrong produces an opaque GL error, so it is
    worth naming.
    `MUJOCO_GL=egl` 은 계측 도구에 필요한 값이고 뷰어가 쓸 수 없는 값이다.
    잘못 두면 알아보기 힘든 GL 에러가 나므로 이름을 붙여 알려준다.
    """
    gl = os.environ.get("MUJOCO_GL", "").lower()
    if gl in ("egl", "osmesa"):
        print(f"⚠️ MUJOCO_GL={gl} 로 설정돼 있다. 이 값은 헤드리스 렌더링용이라 뷰어 창이 안 열린다.")
        print("   창을 띄우려면 이 변수를 지운다:")
        print("     Windows  PowerShell :  Remove-Item Env:MUJOCO_GL")
        print("     Windows  cmd        :  set MUJOCO_GL=")
        print("     Linux / macOS       :  unset MUJOCO_GL")
        sys.exit(2)
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("⚠️ DISPLAY 가 없다. 원격 셸이나 컨테이너에서는 뷰어 창을 띄울 수 없다.")
        print("   계측만 필요하면 MUJOCO_GL=egl 로 tools/grasp_check.py 를 쓴다.")
        sys.exit(2)


def build_policy(
    name: str,
    env: MujocoPickEnv,
    cfg: dict[str, Any],
    ckpt: "Path | None" = None,
) -> Policy | None:
    """Return the requested policy, or None for free inspection.
    요청한 정책을 반환한다. 자유 관찰이면 None."""
    if name == "none":
        return None
    if name == "bc":
        from policy.bc import BCPolicy

        if ckpt is None:
            print("⚠️ --policy bc 는 --policy-ckpt 가 필요하다.")
            sys.exit(2)
        pol = BCPolicy(ckpt)
        print(f"학습 정책 로드: {pol.describe()}")
        if pol.meta.get("trained_on") == "random_tensors":
            print("⚠️ 이 체크포인트는 랜덤 텐서로 학습된 것이다. 보이는 동작에 의미가 없다.")
        return pol
    if name == "scripted":
        return ScriptedPickPolicy(env)
    if name == "hold":
        return HoldPolicy()
    if name == "zero":
        return ZeroPolicy()
    if name == "replay":
        from paths import AI_ROOT

        episodes = sorted((AI_ROOT / "datasets" / "sim_pick_v0").glob("*.npz"))
        if not episodes:
            print("⚠️ datasets/sim_pick_v0 에 에피소드가 없다. 먼저 tools/collect_sim.py 를 돌린다.")
            sys.exit(2)
        return ReplayPolicy.from_episode(episodes[0])
    raise ValueError(f"모르는 정책: {name}")


def free_look(cfg: dict[str, Any]) -> None:
    """Just open the scene and let the user move the camera around.
    씬만 열고 카메라를 자유롭게 움직이게 둔다."""
    import mujoco.viewer

    model = build_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = data.qpos[:6]

    cams = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)
    ]
    print(f"씬: {DEFAULT_SCENE}")
    print(f"설정: {DEFAULT_CONFIG}")
    print(f"nq={model.nq} nu={model.nu} 카메라 {model.ncam}대: {', '.join(cams)}")
    print()
    print("조작:")
    print("  마우스 좌드래그   회전        마우스 우드래그  이동")
    print("  스크롤            줌          더블클릭         그 물체를 추적")
    print("  Ctrl+좌드래그     물체 끌기   백스페이스       초기화")
    print("  Tab               카메라 전환 (cam_front / cam_wrist 도 여기서 본다)")
    print("  스페이스          일시정지/재생")
    print()
    print("⚠️ 물체를 끌어다 놓아도 성공률이 바뀌지 않는다. 여기는 보는 곳이다.")
    mujoco.viewer.launch(model, data)


def watch(
    policy_name: str,
    episodes: int,
    seed_base: int,
    jitter_m: float,
    speed: float,
    cfg: dict[str, Any],
    ckpt: "Path | None" = None,
) -> int:
    """Run rollouts at wall-clock speed inside a passive viewer.
    수동 뷰어 안에서 롤아웃을 실시간 속도로 돌린다."""
    import mujoco.viewer

    # A vision policy must see real pixels. With render=False the observation
    # carries zero-filled images of the correct shape, so the policy runs but on
    # a blank world -- what you would watch is not the behaviour the rollout
    # harness measures.
    # 시각 정책은 실제 픽셀을 봐야 한다. render=False 면 관측에 올바른 shape 의
    # 0 배열이 들어가서, 정책은 돌지만 빈 세계 위에서 돈다. 그러면 화면에 보이는
    # 것은 롤아웃 harness 가 재는 그 동작이 아니다.
    needs_pixels = policy_name == "bc"

    n_ok = 0
    with MujocoPickEnv(cfg, render=needs_pixels, object_jitter_m=jitter_m) as env:
        policy = build_policy(policy_name, env, cfg, ckpt)
        assert policy is not None
        dt = 1.0 / env.control_rate_hz / max(speed, 1e-6)
        print(f"정책 {policy.name} · {episodes} 에피소드 · 물체 xy ±{jitter_m * 1000:.0f}mm "
              f"· {env.control_rate_hz:.0f}Hz × {speed:g}배속")
        if policy.uses_privileged_state:
            print("⚠️ 이 정책은 물체의 정답 위치를 시뮬에서 직접 읽는다. 실물에서는 돌지 않는다.")
        if needs_pixels:
            print("카메라 2대를 매 틱 렌더한다 — 실시간보다 느릴 수 있다. --speed 로 맞춘다.")
        print("스페이스 일시정지 · Tab 카메라 전환 · 창을 닫으면 종료\n")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for i in range(episodes):
                if not viewer.is_running():
                    break
                obs = env.reset(seed=seed_base + i)
                policy.reset(seed=seed_base + i)
                viewer.sync()
                success = False
                for _ in range(env.max_ticks):
                    if not viewer.is_running():
                        break
                    t0 = time.perf_counter()
                    obs = env.step(policy.act(obs))
                    viewer.sync()
                    if env.is_success():
                        success = True
                        break
                    slack = dt - (time.perf_counter() - t0)
                    if slack > 0:
                        time.sleep(slack)
                n_ok += int(success)
                obj = env.object_position()
                print(f"  [{i + 1}/{episodes}] {'성공' if success else '실패'}  "
                      f"물체 ({obj[0]:+.3f}, {obj[1]:+.3f})  상승 {env.lift_height() * 100:5.2f}cm")
                if success and viewer.is_running():
                    time.sleep(0.8)

    print(f"\n{n_ok}/{episodes} 성공")
    print("⚠️ 이건 실시간으로 본 결과다. 보고할 성공률은 tools/eval_rollout.py 로 낸다 —")
    print("   고정 시드, 모든 정책 동일 조건, 게이트 판정까지 함께 나온다.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MuJoCo 뷰어로 씬을 보거나 정책이 도는 것을 본다"
    )
    parser.add_argument("--policy", choices=POLICY_NAMES, default="none",
                        help="none 이면 씬만 열어 자유 관찰 / 그 외는 실시간 실행")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--jitter", type=float, default=0.03,
                        help="물체 xy 무작위 범위 (m)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="1.0 이 실시간. 2 면 2배속, 0.5 면 절반 속도")
    parser.add_argument("--policy-ckpt", type=Path, default=None,
                        help="--policy bc 일 때 볼 체크포인트")
    args = parser.parse_args()

    _check_display()
    cfg = load_config()

    if args.policy == "none":
        free_look(cfg)
        return 0
    return watch(
        args.policy,
        args.episodes,
        args.seed_base,
        args.jitter,
        args.speed,
        cfg,
        args.policy_ckpt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
