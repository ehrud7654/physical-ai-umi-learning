"""G0 gate for DAgger: can the feedback expert recover from off-path states?
DAgger 의 G0 게이트 — 피드백 전문가가 경로 밖 상태에서 복구하는가?

    python tools/probe_recovery.py --episodes 100 --log

Runs the open-loop `scripted` and the state-feedback `scripted_fb` over the SAME
seeds, undisturbed and then disturbed at several ticks, and prints the two gate
verdicts. No render -- both policies are privileged and never look at an image.
개루프 `scripted` 와 상태 피드백 `scripted_fb` 를 **같은 시드**로, 교란 없이 그리고
여러 틱에서 교란해서 돌리고 게이트 두 개를 판정한다. 렌더는 안 한다 — 둘 다
특권 정보를 쓰고 이미지를 보지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("probe_recovery")

from eval.recovery import (  # noqa: E402
    G0A_MIN,
    G0B_MIN,
    GATES,
    PERTURB_HOLD_TICKS,
    PerturbSpec,
    as_records,
    overlaps,
    probe,
    rows,
    summarise,
)
from policy.baselines import ScriptedFeedbackPolicy, ScriptedPickPolicy  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import file_digest, log_run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=3000)
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--perturb-ticks", type=str, default="-1,20,40,60",
                   help="교란을 주입할 틱, 콤마 구분. -1 = 초기 관절 배치 (G0-b 판정 조건)")
    p.add_argument("--perturb-rad", type=str, default="0.05,0.15,0.30",
                   help="관절당 오프셋 크기 (rad), 콤마 구분. 부호는 시드로 정한다. "
                        "여러 개를 주면 복구 곡선이 나온다 — 크기를 하나 고르지 않아도 된다")
    p.add_argument("--hold-ticks", type=int, default=PERTURB_HOLD_TICKS)
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    cfg: dict[str, Any] = load_config()
    seeds = [args.seed_base + i for i in range(args.episodes)]
    rads = [float(x) for x in args.perturb_rad.split(",") if x.strip()]
    specs = [PerturbSpec(int(t), r, args.hold_ticks)
             for t in args.perturb_ticks.split(",") if t.strip()
             for r in rads]

    print("게이트 기준 (결과 확인 전 확정):")
    for k, text in GATES.items():
        print(f"  [{k}] {text}")
    print(f"\n에피소드 {args.episodes} · 시드 {seeds[0]}~{seeds[-1]} · 물체 xy ±{args.jitter * 1000:.0f}mm")
    print(f"교란 크기 {rads} rad · 주입 지점 "
          f"{sorted({('초기배치' if s.is_start else f'틱{s.tick}') for s in specs})} "
          f"(중간 주입은 {args.hold_ticks}틱 유지) · 부호는 시드별 고정\n")

    out: dict[str, dict[str, Any]] = {}
    detail: dict[str, list[dict[str, Any]]] = {}
    print(f"{'정책':<14} {'조건':<14} {'성공':<8} {'성공률':<7}  95% 구간")
    print("-" * 62)
    with MujocoPickEnv(cfg, render=False, object_jitter_m=args.jitter) as env:
        builders = (("scripted", ScriptedPickPolicy), ("scripted_fb", ScriptedFeedbackPolicy))
        for name, ctor in builders:
            for spec in [None, *specs]:
                pol = ctor(env)
                res = [probe(env, pol, s, cfg, spec) for s in seeds]
                cond = spec.label if spec is not None else "none"
                key = f"{name}|{cond}"
                out[key] = summarise(res)
                detail[key] = as_records(res)
                print(rows(name, cond, out[key]))
            print("-" * 62)

    print()
    a_open, a_fb = out["scripted|none"], out["scripted_fb|none"]
    g0a_min_ok = a_fb["rate"] >= G0A_MIN
    g0a_overlap = overlaps(a_fb["ci95"], a_open["ci95"])
    g0a = g0a_min_ok and g0a_overlap
    print("[G0-a] 교란 없는 성공률")
    print(f"  scripted_fb {a_fb['rate'] * 100:.1f}% >= {G0A_MIN:.0%} → "
          f"{'통과' if g0a_min_ok else '실패'}")
    print(f"  개루프 scripted {a_open['rate'] * 100:.1f}% 와 구간 겹침 → "
          f"{'통과' if g0a_overlap else '실패 — 다른 일을 하고 있다'}")
    print(f"  → {'통과' if g0a else '실패'}")

    # 판정은 초기 배치 교란으로 한다. 중간 명령 교란은 위치 제어에서 저절로
    # 지워져서(2026-09-07 🟢 개루프도 무영향) 분포 밖 상태를 만들지 못한다.
    starts = [s for s in specs if s.is_start]
    judged = starts or specs
    fb_perturb = [out[f"scripted_fb|{s.label}"] for s in judged]
    worst = min(fb_perturb, key=lambda s: s["rate"]) if fb_perturb else None
    g0b = bool(worst and worst["rate"] >= G0B_MIN)
    if not starts:
        print("\n⚠️ 초기 배치 교란(-1) 조건이 없다. 중간 명령 교란만으로는 "
              "위치 제어에서 분포 밖 상태가 안 만들어지므로 G0-b 판정 근거가 약하다")
    print("\n[G0-b] 교란 후 복구 (가장 나쁜 조건으로 판정)")
    if worst:
        print(f"  최저 {worst['rate'] * 100:.1f}% (CI {worst['ci95'][0] * 100:.1f}~"
              f"{worst['ci95'][1] * 100:.1f}%) >= {G0B_MIN:.0%} → {'통과' if g0b else '실패'}")
        base = [out[f"scripted|{s.label}"] for s in judged]
        wb = min(base, key=lambda s: s["rate"])
        # 2026-09-07 🟢 정정: 개루프가 복구를 **한다.** ±0.3rad 초기 변위에서도
        # 87.0% 그대로였다. 계획이 절대 관절 목표이고 제어가 위치 제어라, 변위는
        # 다음 명령에서 흡수된다. "개루프는 복구 못 한다"는 내 전제가 틀렸다.
        # 개루프가 못 하는 것은 복구가 아니라 **질문에 답하는 것**이다 — obs 를
        # 읽지 않으므로 임의 상태에 라벨을 붙일 수 없다. 그것이 DAgger 가
        # 피드백 전문가를 요구하는 이유이고, 복구 능력은 그 이유가 아니었다.
        print(f"  개루프 scripted 최저 {wb['rate'] * 100:.1f}%")
        if wb["rate"] >= worst["rate"]:
            print("  ⚠️ 개루프가 같거나 더 낫다. 절대 관절 목표 + 위치 제어는 그 자체로 "
                  "변위를 흡수한다 — 피드백 전문가의 존재 이유는 복구가 아니라 "
                  "**임의 상태에 라벨을 붙일 수 있다는 것**이다")
    print(f"\n→ DAgger {'착수 가능' if (g0a and g0b) else '착수 불가'}")
    if not (g0a and g0b):
        print("  라벨을 모으지 않는다. 게이트를 옮기지 말고 전문가를 고친다.")

    print("\n실패 원인 분해 (교란 없는 조건)")
    for name in ("scripted", "scripted_fb"):
        s = out[f"{name}|none"]
        print(f"  {name}: 실패 {s['n'] - s['successes']}건 · "
              f"닫는순간거리 중앙 {s['fail_median_xy_at_close_mm']}mm · "
              f"물체밀림(닫기완료) 실패 {s['fail_median_obj_push_at_grasp_mm']}mm / "
              f"성공 {s['ok_median_obj_push_at_grasp_mm']}mm · "
              f"턱접촉 실패 {s['fail_jaws_at_grasp']} / 성공 {s['ok_jaws_at_grasp']} · "
              f"닫지않음 {s['never_closed']} · IK실패 {s['ik_failures_total']} "
              f"{s['ik_fail_by_phase']}")
        if s["phase_at_end"]:
            print(f"    종료 phase: {s['phase_at_end']}")
    print("\n⚠️ 물체밀림 성공/실패 중앙값이 비슷하면 밀림은 실패 원인이 아니다. "
          "턱접촉수 1 이 많으면 한쪽 턱만 닿아 집히지 않은 것이다.")

    if args.log:
        rec = log_run(
            experiment="recovery_probe",
            author=args.author,
            issue="project-task",
            conditions={
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "jitter_m": args.jitter,
                "perturb_rads": rads,
                "perturb_ticks": sorted({s.tick for s in specs}),
                "hold_ticks": args.hold_ticks,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "gates": GATES,
                "g0a_min": G0A_MIN,
                "g0b_min": G0B_MIN,
            },
            result={
                "summary": out,
                "g0a_passed": g0a,
                "g0b_passed": g0b,
                "dagger_go": bool(g0a and g0b),
                "detail": detail,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0 if (g0a and g0b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
