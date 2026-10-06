"""Do our recorded action streams fit inside the real robot's limits?
우리가 기록한 행동열이 실물 로봇의 한계 안에 들어가는가?

Until 2026-09-08 this could not be asked. `configs/so101.yaml` carries no joint
velocity or acceleration limit, so issue 113 reported `dynamic: NOT_EVALUABLE` and
every executability claim stopped at geometry. The direct-motor pipeline handover
(`so101_direct_motor_pipeline`) supplies those numbers, and the question becomes
answerable without touching the robot.
2026-09-08 까지 이 질문은 던질 수 없었다. `configs/so101.yaml` 에 관절 속도·가속도
한계가 없어서 이슈 113 은 `dynamic: NOT_EVALUABLE` 을 냈고, 실행가능성 주장은 전부
기하에서 멈췄다. 직접 모터 제어 파이프라인 인계본이 그 값을 주면서, 로봇을 만지지
않고도 답할 수 있는 질문이 됐다.

What it decides: whether the 98 scripted episodes are executable at all. A
trajectory that exceeds `max_speed_rad_s` or `max_accel_rad_s2` is rejected by
`core.models.validate_trajectory` before a single tick reaches the servos -- so if
our demonstrations violate the limits, no amount of policy work matters and the
scripted expert has to be re-tuned first.
무엇을 판정하는가: 98편 scripted 에피소드가 애초에 실행 가능한지. `max_speed_rad_s`
또는 `max_accel_rad_s2` 를 넘는 궤적은 서보에 tick 하나 도달하기 전에
`core.models.validate_trajectory` 가 거부한다. 즉 우리 시연이 한계를 위반하면
정책 작업이 아무 의미가 없고 scripted 전문가부터 다시 맞춰야 한다.

⚠️ The limits are read from the pipeline's own config, never hardcoded here. The
   pipeline is the source of truth; `AI/configs/real/so101_ver1.json` is a tracked
   copy so a change on the hardware side shows up as a diff instead of silently
   moving our verdicts.
⚠️ 한계값은 파이프라인 자신의 설정에서 읽고 여기에 하드코딩하지 않는다. 정본은
   파이프라인이고 `AI/configs/real/so101_ver1.json` 은 추적되는 사본이다. 그래야
   하드웨어 쪽 변경이 우리 판정을 조용히 옮기는 대신 diff 로 드러난다.

⚠️ This is a check on the *commanded* stream. It does not say the real arm follows
   it -- that needs the robot, and `real_calibration_verified` is still false.
⚠️ 이것은 **명령** 열에 대한 검사다. 실물 팔이 그것을 따라간다는 말이 아니다 —
   그건 로봇이 필요하고 `real_calibration_verified` 는 아직 false 다.

    # [서버]
    python tools/check_real_limits.py datasets/sim_pick_v5 --log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("check_real_limits")

from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA_AT_LAUNCH = code_digest()
AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CONFIG = AI_ROOT / "configs" / "real" / "so101_ver1.json"

# The pipeline rejects any waypoint shorter than this
# (`core.models.validate_trajectory`). Our contract is 30Hz = 0.0333s.
# 파이프라인은 이보다 짧은 waypoint 를 거부한다
# (`core.models.validate_trajectory`). 우리 계약은 30Hz = 0.0333초다.
PIPELINE_MIN_WAYPOINT_S = 0.05


def denorm(x_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Contract [-1,1] back to radians, per `configs/so101.yaml` normalization.
    계약 [-1,1] 을 라디안으로. `configs/so101.yaml` 의 정규화 규칙을 따른다."""
    return (x_norm + 1.0) / 2.0 * (hi - lo) + lo


def gripper_rad_to_gap_m(rad: np.ndarray, curve: list[list[float]]) -> np.ndarray:
    """Stock revolute gripper angle to pad gap, via the measured curve.
    공식 회전 그리퍼 각도를 패드 간격으로. 실측 곡선을 쓴다.

    The curve is in cm and the pipeline's API is in metres, so the conversion is
    also the bridge between two different mechanisms: our sim gripper is a hinge
    and the real Ver1 gripper is a slide. Only the gap is common to both.
    곡선은 cm 이고 파이프라인 API 는 m 다. 그래서 이 변환은 서로 다른 두 기구를
    잇는 다리이기도 하다 — 우리 시뮬 그리퍼는 hinge 이고 실물 Ver1 은 slide 다.
    **둘에 공통인 것은 간격뿐이다.**
    """
    xs = np.array([p[0] for p in curve], dtype=np.float64)
    ys = np.array([p[1] for p in curve], dtype=np.float64) / 100.0
    order = np.argsort(xs)
    return np.interp(rad, xs[order], ys[order])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--real-config", type=Path, default=DEFAULT_REAL_CONFIG,
                   help="파이프라인 configs/so101_ver1.json 의 사본")
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    if not args.real_config.exists():
        print(f"실물 설정이 없다: {args.real_config}")
        print("so101_direct_motor_pipeline/configs/so101_ver1.json 을 복사해라")
        return 1

    cfg: dict[str, Any] = load_config()
    real = json.loads(args.real_config.read_text(encoding="utf-8"))
    rate = float(cfg["control"]["rate_hz"])
    dt = 1.0 / rate
    joints = sorted(cfg["joints"], key=lambda j: j["index"])
    names = [j["name"] for j in joints]
    ranges = [(float(j["range_rad"][0]), float(j["range_rad"][1])) for j in joints]
    g_idx = next(j["index"] for j in joints if j["name"] == "gripper")
    curve = cfg["grasp"]["gap_curve"]

    v_max = float(real["max_speed_rad_s"])
    a_max = float(real["max_accel_rad_s2"])
    gv_max = float(real["max_gap_speed_m_s"])
    ga_max = float(real["max_gap_accel_m_s2"])
    gap_max = float(real["max_gap_m"])

    eps = sorted(args.data.glob("ep_*.npz"))
    if not eps:
        print(f"에피소드가 없다: {args.data}")
        return 1

    print(f"{args.data.name}: {len(eps)}편 · {rate:.0f}Hz (dt {dt * 1000:.1f}ms)")
    print(f"실물 한계 ({args.real_config.name}, {file_digest(args.real_config)}): "
          f"속도 {v_max} rad/s · 가속도 {a_max} rad/s² · "
          f"gap 속도 {gv_max} m/s · gap 가속도 {ga_max} m/s²")
    print(f"검증 플래그: real_calibration_verified="
          f"{real.get('real_calibration_verified')} · "
          f"installation_verified={real.get('installation_verified')} · "
          f"collision_reviewed={real.get('collision_reviewed')}")
    print()

    arm = [i for i in range(6) if i != g_idx]
    vmax_seen = np.zeros(6)
    amax_seen = np.zeros(6)
    v_all: list[np.ndarray] = []
    a_all: list[np.ndarray] = []
    gap_lo, gap_hi = np.inf, -np.inf
    gv_seen = ga_seen = 0.0
    bad_v = {n: 0 for n in names}
    bad_a = {n: 0 for n in names}
    gap_out = 0
    n_steps_total = 0

    for npz in eps:
        with np.load(npz) as z:
            act = np.asarray(z["action"], dtype=np.float64)
        rad = np.empty_like(act)
        for j in range(6):
            rad[:, j] = denorm(act[:, j], *ranges[j])
        gap = gripper_rad_to_gap_m(rad[:, g_idx], curve)
        gap_lo, gap_hi = min(gap_lo, gap.min()), max(gap_hi, gap.max())
        gap_out += int(((gap < 0.0) | (gap > gap_max)).sum())

        vel = np.diff(rad, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        gvel = np.diff(gap) / dt
        gacc = np.diff(gvel) / dt
        n_steps_total += rad.shape[0]

        vmax_seen = np.maximum(vmax_seen, np.abs(vel).max(axis=0))
        amax_seen = np.maximum(amax_seen, np.abs(acc).max(axis=0))
        v_all.append(np.abs(vel))
        a_all.append(np.abs(acc))
        gv_seen = max(gv_seen, float(np.abs(gvel).max()))
        ga_seen = max(ga_seen, float(np.abs(gacc).max()))
        for j in arm:
            bad_v[names[j]] += int((np.abs(vel[:, j]) > v_max).sum())
            bad_a[names[j]] += int((np.abs(acc[:, j]) > a_max).sum())
        bad_v[names[g_idx]] += int((np.abs(gvel) > gv_max).sum())
        bad_a[names[g_idx]] += int((np.abs(gacc) > ga_max).sum())

    V = np.concatenate(v_all)
    A = np.concatenate(a_all)

    print(f"{'관절':<15}{'최대 속도':>12}{'p99':>10}{'한계':>8}{'위반':>8}"
          f"{'최대 가속':>12}{'p99':>10}{'한계':>8}{'위반':>8}")
    for j in arm:
        print(f"{names[j]:<15}{vmax_seen[j]:>12.4f}{np.percentile(V[:, j], 99):>10.4f}"
              f"{v_max:>8.2f}{bad_v[names[j]]:>8}"
              f"{amax_seen[j]:>12.4f}{np.percentile(A[:, j], 99):>10.4f}"
              f"{a_max:>8.2f}{bad_a[names[j]]:>8}")
    print(f"{'gripper(gap)':<15}{gv_seen:>12.4f}{'':>10}{gv_max:>8.2f}"
          f"{bad_v[names[g_idx]]:>8}{ga_seen:>12.4f}{'':>10}{ga_max:>8.2f}"
          f"{bad_a[names[g_idx]]:>8}")
    print("\n팔 단위 rad/s · rad/s² · 그리퍼 단위 m/s · m/s²")

    print(f"\n그리퍼 간격 범위 {gap_lo * 1000:.1f} ~ {gap_hi * 1000:.1f} mm "
          f"(실물 허용 0 ~ {gap_max * 1000:.0f} mm) · 범위 밖 {gap_out} 스텝")

    v_bad = sum(bad_v.values())
    a_bad = sum(bad_a.values())

    print(f"\n총 {n_steps_total} 스텝 · 속도 위반 {v_bad} · 가속도 위반 {a_bad}")

    # ---- 판정 (결과 보기 전 확정) -------------------------------------------
    print("\n판정 (결과 보기 전 확정):")
    print("  위반 0 이면 명령 열 자체는 실물 안전검사를 통과할 수 있다")
    print("  위반이 있으면 그 에피소드는 실물에서 거부된다 → scripted 전문가의 "
          "timing 을 늘려 재수집해야 하고, 정책 작업보다 그것이 먼저다")
    ok = (v_bad == 0 and a_bad == 0 and gap_out == 0)
    print(f"  실측 → {'**통과.** 명령 열은 한계 안에 있다' if ok else '**위반 있음.** 위 표의 위반 열을 보라'}")

    # ---- 구조적 불일치 — 에피소드 수치와 무관하게 항상 성립한다 --------------
    print("\n계약과 파이프라인의 구조적 불일치 (이 실행 결과와 무관)")
    print(f"  1. 제어 주기: 계약 {rate:.0f}Hz = {dt * 1000:.1f}ms 이고 "
          f"파이프라인 waypoint 최소값은 {PIPELINE_MIN_WAYPOINT_S * 1000:.0f}ms "
          f"({1 / PIPELINE_MIN_WAYPOINT_S:.0f}Hz).")
    print(f"     → {rate:.0f}Hz 행동열을 waypoint 마다 하나씩 넣을 수 없다. "
          "거부된다. 재표집 또는 묶음 전송이 필요하다")
    print("  2. 그리퍼 기구: 계약은 hinge 관절각, 실물 Ver1 은 slide (gap_m).")
    print("     → 공통인 것은 간격뿐이다. gap_curve 가 다리 역할을 하지만, "
          "패드 형상이 달라 같은 간격에서도 접촉이 다르다")
    print("  3. 이 스크립트는 **명령** 열만 검사한다. 실물이 따라가는지는 "
          "로봇이 필요하고 real_calibration_verified 는 아직 false 다")

    if args.log:
        rec = log_run(
            experiment="check_real_limits",
            author=args.author,
            issue="project-task",
            conditions={
                "dataset": str(args.data), "n_episodes": len(eps),
                "rate_hz": rate,
                "real_config": str(args.real_config),
                "real_config_sha": file_digest(args.real_config),
                "max_speed_rad_s": v_max, "max_accel_rad_s2": a_max,
                "max_gap_speed_m_s": gv_max, "max_gap_accel_m_s2": ga_max,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "code_sha_at_launch": CODE_SHA_AT_LAUNCH,
            },
            result={
                "n_steps": n_steps_total,
                "vel_max_rad_s": {names[j]: round(float(vmax_seen[j]), 5) for j in arm},
                "acc_max_rad_s2": {names[j]: round(float(amax_seen[j]), 5) for j in arm},
                "vel_p99_rad_s": {names[j]: round(float(np.percentile(V[:, j], 99)), 5)
                                  for j in arm},
                "gap_vel_max_m_s": round(gv_seen, 6),
                "gap_acc_max_m_s2": round(ga_seen, 6),
                "gap_range_mm": [round(gap_lo * 1000, 2), round(gap_hi * 1000, 2)],
                "violations_speed": bad_v,
                "violations_accel": bad_a,
                "gap_out_of_range_steps": gap_out,
                "passes": ok,
                "contract_rate_hz": rate,
                "pipeline_min_waypoint_s": PIPELINE_MIN_WAYPOINT_S,
                "rate_mismatch": dt < PIPELINE_MIN_WAYPOINT_S,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
