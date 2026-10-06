"""One rollout budget, many verdicts: can our policy commands reach the servos?
한 번의 롤아웃 예산으로 여러 판정을 낸다 — 우리 정책 명령이 서보에 도달하는가.

    python tools/probe_exec_envelope.py --help

## 왜 한 스크립트인가

롤아웃은 비싸고(EGL 렌더) **결정적이다** 🟢 (2026-09-10, eval_rollout 2회 완전 동일).
그래서 명령열을 한 번 모아두면 그 뒤 판정은 **재롤아웃 없이 후처리**로 계산된다.
시간 스케일링·데시메이션 스윕이 공짜가 되는 이유다.

## 무엇을 한 번에 판정하나

| # | 질문 | 비용 |
|---|---|---|
| 1 | 시뮬 롤아웃 성공률이 기존 수치를 재현하는가 (계측기 검증) | 패스 A |
| 2 | ROS2 백엔드가 명령열을 거부하는가 · 어느 제약이 지배하는가 | 후처리 |
| 3 | `elbow_flex` 하한 위반 비율과 **초과량 분포** | 후처리 |
| 4 | direct-motor 전사 검사 통과율 | 후처리 |
| 5 | 시간 스케일링 k 를 얼마나 키워야 direct-motor 를 통과하나 | 후처리 |
| 6 | **URDF 한계로 클램프해도 시뮬 성공률이 유지되나** ← 처방 검증 | 패스 B |
| 7 | 폐쇄 시각의 상한 (에피소드별 최근접 틱에 닫기) | 패스 C |

## 왜 6번이 핵심인가

**ROS2 검사는 위치와 tick 만 본다.** 속도를 늦추든 waypoint 를 솎든 같은 위치를
지나므로 통과율이 바뀌지 않는다. 즉 ROS2 가 실패하면 **시간 스케일링은 처방이 아니다.**
남는 처방은 궤적이 지나는 위치를 바꾸는 것이고, 그 최소 개입이 클램프다.
클램프하고도 시뮬 성공률이 유지되면 config 정합만으로 끝난다.

## 한계

- 명령점 검사다. `JointTrajectoryController` 의 30Hz 보간점은 재현하지 않는다 →
  **ROS2 통과율은 하한이다.** 통과해도 보간점이 걸릴 수 있다
- 실물 추종 오차·충돌·설치 안전을 판정하지 않는다
- `real_calibration_verified`·`installation_verified`·`collision_reviewed` 가 false 인 동안
  어떤 결과도 **실물 안전 승인이 아니다**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402

runtime_limits.claim("probe_exec_envelope")
runtime_limits.torch_threads()

import numpy as np  # noqa: E402

from eval.exec_limits import (  # noqa: E402
    check_direct,
    check_ros2,
    denorm,
    dominant_constraints,
    tightest_margins,
    gripper_rad_to_gap_m,
    load_real_config,
    load_ros2_limits,
    merge_reports,
    norm,
)
from policy.bc import BCPolicy  # noqa: E402
from policy.baselines import ScriptedPickPolicy  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL = AI_ROOT / "configs" / "real" / "so101_ver1.json"
DEFAULT_URDF = AI_ROOT / "configs" / "real" / "so101_ver1.urdf"
CODE_SHA = code_digest()

TIME_SCALES = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0)


class ClampWrapper:
    """Clamp a policy's arm commands to the ROS2 backend position limits.
    정책의 팔 명령을 ROS2 백엔드 위치 한계로 자른다.

    개입은 이것뿐이다 — 팔이 가는 자리만 바꾸고 그리퍼와 정책 가중치는 건드리지 않는다.
    한 번에 하나만 바꿔야 성공률 차이를 귀속시킬 수 있다.
    """

    def __init__(self, base: Any, lo_norm: np.ndarray, hi_norm: np.ndarray) -> None:
        self.base = base
        self._lo = lo_norm
        self._hi = hi_norm
        self.clamped_frames = 0
        self.total_frames = 0

    def reset(self, seed: int) -> None:
        self.base.reset(seed=seed)

    def act(self, obs: Any) -> np.ndarray:
        a = np.asarray(self.base.act(obs), dtype=np.float64).copy()
        out = np.clip(a, self._lo, self._hi)
        self.total_frames += 1
        self.clamped_frames += int(bool(np.any(out != a)))
        return out


class CloseAtTick:
    """Force the gripper shut at one chosen tick. Upper bound on close timing.
    지정한 틱에 그리퍼를 강제로 닫는다. 폐쇄 시각의 상한 계측용.

    인과적으로 불가능하다(미래를 봐야 한다). 배포 처방이 아니라 상한 계측기다.
    """

    def __init__(self, base: Any, close_tick: int, open_norm: float, close_norm: float,
                 gripper_index: int) -> None:
        self.base = base
        self.close_tick = int(close_tick)
        self._open = float(open_norm)
        self._close = float(close_norm)
        self._g = int(gripper_index)
        self._t = 0

    def reset(self, seed: int) -> None:
        self.base.reset(seed=seed)
        self._t = 0

    def act(self, obs: Any) -> np.ndarray:
        a = np.asarray(self.base.act(obs), dtype=np.float64).copy()
        a[self._g] = self._close if self._t >= self.close_tick else self._open
        self._t += 1
        return a


def rollout_once(
    env: MujocoPickEnv,
    policy: Any,
    seed: int,
    joint_ranges: list[tuple[float, float]],
    gap_curve: list[list[float]],
    gripper_index: int,
    stop_on_success: bool,
) -> dict[str, Any]:
    """Run one episode and keep the commanded stream plus closing diagnostics.
    에피소드 하나를 돌리고 명령열과 폐쇄 진단을 남긴다."""
    obs = env.reset(seed=seed)
    policy.reset(seed)

    # reset 직후 실제 상태를 첫 명령 앞에 붙인다 — 시작 자세에서 첫 명령까지의
    # 이동도 검사 대상이다. 이게 없으면 첫 구간이 통째로 빠진다.
    q0 = np.asarray(env.joint_positions(), dtype=np.float64)
    arm_idx = [i for i in range(len(joint_ranges)) if i != gripper_index]
    arm_rows: list[np.ndarray] = [q0[arm_idx]]
    gap_rows: list[float] = [float(gripper_rad_to_gap_m(q0[gripper_index : gripper_index + 1], gap_curve)[0])]

    xy_series: list[float] = []
    success = False
    ticks = 0
    for _ in range(env.max_ticks):
        action = np.asarray(policy.act(obs), dtype=np.float64).reshape(len(joint_ranges))
        rad = np.array(
            [denorm(action[i], *joint_ranges[i]) for i in range(len(joint_ranges))],
            dtype=np.float64,
        )
        arm_rows.append(rad[arm_idx])
        gap_rows.append(float(gripper_rad_to_gap_m(rad[gripper_index : gripper_index + 1], gap_curve)[0]))

        obs = env.step(action)
        ticks += 1
        xy, _ = env.pinch_to_object_m()
        xy_series.append(float(xy))
        if env.is_success():
            success = True
            if stop_on_success:
                break

    xy_arr = np.asarray(xy_series, dtype=np.float64)
    nearest_tick = int(np.argmin(xy_arr)) if xy_arr.size else -1
    return {
        "seed": seed,
        "success": success,
        "ticks": ticks,
        "arm_rad": np.asarray(arm_rows, dtype=np.float64),
        "gap_m": np.asarray(gap_rows, dtype=np.float64),
        "nearest_tick": nearest_tick,
        "nearest_xy_mm": float(xy_arr.min() * 1000.0) if xy_arr.size else float("nan"),
        "lift_cm": float(env.lift_height() * 100.0),
    }


def decide(ros2_rate: float, direct_rate: float) -> str:
    """Pre-registered verdict from the two pass rates.
    두 통과율에서 나오는 사전등록 판정."""
    if ros2_rate >= 0.80:
        return "ros2_ok__pick_execution_path"
    if ros2_rate >= 0.30:
        return "ros2_partial__align_config_or_change_trajectory"
    return "ros2_blocked__config_or_ros_hw_agreement_first"


def fmt_table(title: str, rows: list[tuple[str, Any, Any, Any, Any]]) -> None:
    print(f"\n{title}")
    print(f"{'constraint':<36}{'viol':>9}{'total':>9}{'rate':>10}{'max x':>10}")
    for name, viol, total, rate, mx in rows:
        rate_s = "-" if rate is None else f"{100.0 * rate:.2f}%"
        mx_s = "-" if mx is None else f"{mx:.2f}x"
        print(f"{name:<36}{viol:>9}{total:>9}{rate_s:>10}{mx_s:>10}")


def summarise_rows(summary: dict[str, Any], only_violations: bool) -> list[tuple[str, Any, Any, Any, Any]]:
    rows = []
    for key, r in summary["metrics"].items():
        if only_violations and not r["violations"]:
            continue
        rows.append((key, r["violations"], r["total"], r["violation_rate"], r["max_ratio"]))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-ckpt", type=Path, action="append", default=[])
    p.add_argument("--expected", type=int, action="append", default=[],
                   help="--policy-ckpt 순서와 같은 기존 성공 수. 계측기 재현 검사")
    p.add_argument("--include-scripted", action="store_true")
    p.add_argument("--expected-scripted", type=int, default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=3000)
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--policy-device", type=str, default="cpu")
    p.add_argument("--real-config", type=Path, default=DEFAULT_REAL)
    p.add_argument("--ros2-urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--clamp-pass", action="store_true",
                   help="패스 B — URDF 한계로 클램프한 정책의 시뮬 성공률")
    p.add_argument("--best-tick-pass", action="store_true",
                   help="패스 C — 에피소드별 최근접 틱에 닫는 폐쇄 시각 상한")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    if not args.policy_ckpt and not args.include_scripted:
        p.error("--policy-ckpt 또는 --include-scripted 가 필요하다")
    if args.expected and len(args.expected) != len(args.policy_ckpt):
        p.error("--expected 개수가 --policy-ckpt 개수와 다르다")
    if args.out.exists():
        raise FileExistsError(args.out)
    for path in (args.real_config, args.ros2_urdf):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = load_config()
    real = load_real_config(args.real_config)
    ros2 = load_ros2_limits(args.ros2_urdf)

    joints = sorted(cfg["joints"], key=lambda r: int(r["index"]))
    joint_ranges = [(float(j["range_rad"][0]), float(j["range_rad"][1])) for j in joints]
    gap_curve = cfg["grasp"]["gap_curve"]
    gripper_index = next(int(j["index"]) for j in joints if j["name"] == "gripper")
    arm_names = ros2.arm_names
    seeds = [args.seed_base + i for i in range(args.episodes)]

    # 클램프 경계를 계약 정규화 공간으로 옮긴다. 값은 URDF 에서 온다 — 하드코딩 없음.
    lo_norm = np.full(len(joint_ranges), -1.0)
    hi_norm = np.full(len(joint_ranges), 1.0)
    for j in ros2.arm:
        idx = next(int(row["index"]) for row in joints if row["name"] == j.name)
        lo, hi = joint_ranges[idx]
        lo_norm[idx] = float(norm(np.array([j.min_rad]), lo, hi)[0])
        hi_norm[idx] = float(norm(np.array([j.max_rad]), lo, hi)[0])

    env_rate = float(cfg["control"]["rate_hz"])

    print("실행 포락선 계측 — 명령열이 서보에 도달하는가")
    print(f"조건: {args.episodes}편 · seeds {seeds[0]}~{seeds[-1]} · "
          f"{env_rate:.0f}Hz · render · "
          f"policy-device {args.policy_device} · jitter ±{args.jitter * 1000:.0f}mm")
    print(f"ROS2 URDF  {args.ros2_urdf.name} sha {file_digest(args.ros2_urdf)}")
    print(f"real cfg   {args.real_config.name} sha {file_digest(args.real_config)}")
    print(f"sim cfg    {Path(DEFAULT_CONFIG).name} sha {file_digest(DEFAULT_CONFIG)}")
    print(f"실물 플래그 calibration={real.get('real_calibration_verified')} "
          f"installation={real.get('installation_verified')} "
          f"collision={real.get('collision_reviewed')}")
    print("⚠️ 명령점 검사다. controller 보간·실물 추종·충돌을 판정하지 않는다. "
          "어떤 결과도 실물 안전 승인이 아니다.\n")
    print("클램프 경계 (URDF → 계약 정규화):")
    for j in ros2.arm:
        idx = next(int(row["index"]) for row in joints if row["name"] == j.name)
        print(f"  {j.name:<15} URDF [{j.min_rad:+.5f}, {j.max_rad:+.5f}] rad  "
              f"→ norm [{lo_norm[idx]:+.4f}, {hi_norm[idx]:+.4f}]"
              + ("   ← 우리 config 보다 좁다" if lo_norm[idx] > -0.999 or hi_norm[idx] < 0.999 else ""))

    dt = 1.0 / env_rate
    results: dict[str, Any] = {}

    with MujocoPickEnv(cfg=cfg, render=True, object_jitter_m=args.jitter, max_ticks=200) as env:
        targets: list[tuple[str, Any, int | None]] = []
        if args.include_scripted:
            targets.append(("scripted", ScriptedPickPolicy(env), args.expected_scripted))
        for i, ckpt in enumerate(args.policy_ckpt):
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            targets.append((ckpt.stem, BCPolicy(ckpt, device=args.policy_device),
                            args.expected[i] if args.expected else None))

        for label, policy, expected in targets:
            print(f"\n===== {label} =====")
            # ---- 패스 A: 그대로 --------------------------------------------
            episodes = []
            for k, seed in enumerate(seeds):
                episodes.append(rollout_once(env, policy, seed, joint_ranges,
                                             gap_curve, gripper_index, True))
                if (k + 1) % 20 == 0:
                    ok = sum(e["success"] for e in episodes)
                    print(f"  A {k + 1}/{len(seeds)} success={ok}", flush=True)
            success_a = sum(e["success"] for e in episodes)
            if expected is not None and success_a != expected:
                raise RuntimeError(
                    f"instrument invalid: {label} {success_a}/{args.episodes}, "
                    f"expected {expected}/{args.episodes} — "
                    "git_rev·config sha·체크포인트 경로를 먼저 대조하라"
                )
            print(f"  A 재현 fixture: {success_a}/{args.episodes}"
                  + (" PASS" if expected is not None else " (fixture 미지정)"))

            # ---- 후처리: 두 경로 판정 + 시간 스케일링 -----------------------
            ros2_reports = [check_ros2(e["arm_rad"], e["gap_m"], ros2) for e in episodes]
            ros2_sum = merge_reports(ros2_reports)

            scale_curve = []
            for k in TIME_SCALES:
                reps = [
                    check_direct(
                        e["arm_rad"], e["gap_m"], dt * k, arm_names, real["limits_rad"],
                        float(real["max_speed_rad_s"]), float(real["max_accel_rad_s2"]),
                        float(real["max_gap_m"]), float(real["max_gap_speed_m_s"]),
                        float(real["max_gap_accel_m_s2"]),
                    )
                    for e in episodes
                ]
                s = merge_reports(reps)
                scale_curve.append({"scale": k, "dt_s": dt * k,
                                    "pass_rate": s["episode_pass_rate"]})
                if k == 1.0:
                    direct_sum = s

            # ---- 패스 B: 클램프 --------------------------------------------
            clamp: dict[str, Any] | None = None
            if args.clamp_pass:
                wrapper = ClampWrapper(policy, lo_norm, hi_norm)
                clamp_eps = [
                    rollout_once(env, wrapper, seed, joint_ranges, gap_curve,
                                 gripper_index, True)
                    for seed in seeds
                ]
                success_b = sum(e["success"] for e in clamp_eps)
                clamp_ros2 = merge_reports(
                    [check_ros2(e["arm_rad"], e["gap_m"], ros2) for e in clamp_eps]
                )
                clamp = {
                    "rollout_successes": success_b,
                    "rollout_success_rate": success_b / args.episodes,
                    "delta_vs_pass_a": (success_b - success_a) / args.episodes,
                    "clamped_frame_rate": (
                        wrapper.clamped_frames / max(1, wrapper.total_frames)
                    ),
                    "ros2": clamp_ros2,
                }
                print(f"  B 클램프: {success_b}/{args.episodes} "
                      f"(A 대비 {100.0 * clamp['delta_vs_pass_a']:+.1f}%p) · "
                      f"클램프된 프레임 {100.0 * clamp['clamped_frame_rate']:.1f}% · "
                      f"ROS2 통과 {100.0 * clamp_ros2['episode_pass_rate']:.1f}%")

            # ---- 패스 C: 폐쇄 시각 상한 ------------------------------------
            best_tick: dict[str, Any] | None = None
            if args.best_tick_pass:
                from policy.bc import gripper_command_norms  # noqa: PLC0415

                open_norm_v, close_norm_v = gripper_command_norms()
                hits = 0
                for e in episodes:
                    if e["nearest_tick"] < 0:
                        continue
                    forced = CloseAtTick(policy, e["nearest_tick"], open_norm_v,
                                         close_norm_v, gripper_index)
                    r = rollout_once(env, forced, e["seed"], joint_ranges, gap_curve,
                                     gripper_index, True)
                    hits += int(r["success"])
                best_tick = {
                    "rollout_successes": hits,
                    "rollout_success_rate": hits / args.episodes,
                    "delta_vs_pass_a": (hits - success_a) / args.episodes,
                    "note": "causally impossible; upper bound on close timing only",
                }
                print(f"  C 폐쇄시각 상한: {hits}/{args.episodes} "
                      f"(A 대비 {100.0 * best_tick['delta_vs_pass_a']:+.1f}%p)")

            verdict = decide(ros2_sum["episode_pass_rate"], direct_sum["episode_pass_rate"])
            results[label] = {
                "rollout_successes": success_a,
                "rollout_success_rate": success_a / args.episodes,
                "nearest_xy_mm_median": float(
                    np.median([e["nearest_xy_mm"] for e in episodes])
                ),
                "ros2": ros2_sum,
                "direct_motor_transcribed": direct_sum,
                "time_scale_curve": scale_curve,
                "clamp_pass": clamp,
                "best_tick_pass": best_tick,
                "verdict": verdict,
                "ros2_tightest_margins": tightest_margins(ros2_sum),
                "direct_tightest_margins": tightest_margins(direct_sum),
                "episode_detail": [
                    {"seed": e["seed"], "success": bool(e["success"]),
                     "ticks": int(e["ticks"]), "nearest_tick": int(e["nearest_tick"]),
                     "nearest_xy_mm": e["nearest_xy_mm"],
                     "ros2_passes": bool(rep["passes"])}
                    for e, rep in zip(episodes, ros2_reports)
                ],
            }

            print(f"  ROS2 통과 {100.0 * ros2_sum['episode_pass_rate']:.1f}% · "
                  f"direct(전사) {100.0 * direct_sum['episode_pass_rate']:.1f}% · "
                  f"판정 {verdict}")
            ros2_rows = summarise_rows(ros2_sum, only_violations=True)
            if ros2_rows:
                fmt_table(f"  {label} — ROS2 위반 제약 (원문 🟢)", ros2_rows)
                print("  ROS2 지배 제약: " + " · ".join(
                    f"{k} {100.0 * v:.1f}%" for k, v in dominant_constraints(ros2_sum)))
            else:
                # 위반이 0 이면 남는 질문은 "얼마나 여유 있게 통과했나" 다.
                # 0.98x 통과와 0.05x 통과는 전혀 다른 상태다.
                fmt_table(
                    f"  {label} — ROS2 위반 0건. 한계 대비 최대 배수 (여유)",
                    [(k, ros2_sum["metrics"][k]["violations"],
                      ros2_sum["metrics"][k]["total"],
                      ros2_sum["metrics"][k]["violation_rate"], v)
                     for k, v in tightest_margins(ros2_sum)],
                )
                head = tightest_margins(ros2_sum, top=1)
                if head:
                    print(f"  ROS2 최소 여유: {head[0][0]} 가 한계의 "
                          f"{head[0][1]:.3f}배까지 갔다 "
                          f"(1.0 을 넘으면 거부)")

            direct_rows = summarise_rows(direct_sum, only_violations=True)
            fmt_table(f"  {label} — direct-motor 위반 제약 (전사 🔵, dt={dt:.4f}s)",
                      direct_rows or [(k, 0, direct_sum["metrics"][k]["total"], 0.0, v)
                                      for k, v in tightest_margins(direct_sum)])
            print("  시간 스케일링 → direct 통과율: " +
                  " · ".join(f"x{c['scale']:g} {100.0 * c['pass_rate']:.0f}%"
                             for c in scale_curve))
            print("  ⚠️ ROS2 는 위치·tick 만 본다 — 시간 스케일링으로 통과율이 바뀌지 않는다")

    payload = {
        "experiment": "exec_envelope",
        "conditions": {
            "episodes": args.episodes, "seed_base": args.seed_base,
            "seed_end": seeds[-1], "jitter_m": args.jitter, "render": True,
            "policy_device": args.policy_device, "control_rate_hz": env_rate,
            "max_ticks": 200, "stop_on_first_success": True,
            "initial_state_prepended": True,
            "clamp_pass": args.clamp_pass, "best_tick_pass": args.best_tick_pass,
            "policy_checkpoints": [str(c) for c in args.policy_ckpt],
            "include_scripted": args.include_scripted,
            "real_config": str(args.real_config),
            "real_config_sha": file_digest(args.real_config),
            "ros2_urdf": str(args.ros2_urdf),
            "ros2_urdf_sha": file_digest(args.ros2_urdf),
            "sim_config_sha": file_digest(DEFAULT_CONFIG),
            "code_sha_at_launch": CODE_SHA,
            "time_scales": list(TIME_SCALES),
        },
        "results": results,
        "interpretation_limit": (
            "Sampled-command check only. Controller interpolation, tracking error, "
            "collision and installation safety are out of scope. Real-robot "
            "verification flags are false; this is not a safety approval."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장 {args.out}")

    if args.log:
        rec = log_run(experiment="exec_envelope", author=args.author,
                      issue="project-task", conditions=payload["conditions"],
                      result={"results": results,
                              "interpretation_limit": payload["interpretation_limit"]})
        print(f"EXP_LOG 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    print("판정은 사람이 한다. docs/PREREG_exec_envelope_0911.md 를 열고 대조하라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
