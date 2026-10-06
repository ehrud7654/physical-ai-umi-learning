"""kick test: is the frozen policy at an observation fixpoint?
kick 검사 — 얼어붙은 정책이 관측 고정점에 갇힌 것인가?

    python tools/probe_freeze.py --policy-ckpt checkpoints/bc/sim_pick_v5_seed0.pt --episodes 100 --log
    python tools/probe_freeze.py --policy scripted_fb --episodes 20      # 연기시험 (torch 불필요)

No training. Runs the same seeds twice -- once untouched, once with a single
5-tick scripted lift handed over the moment the arm stops moving -- and reports
both rates with their intervals plus the re-freeze fraction. Gates are in
`eval/freeze.py`, fixed before the first number.
학습 없음. 같은 시드를 두 번 돌린다 — 그대로 한 번, 팔이 멈추는 순간 scripted 들기를
5틱 한 번 넘겨주고 한 번 — 그리고 두 성공률과 구간, 재동결 비율을 보고한다.
게이트는 `eval/freeze.py` 에 있고 첫 수치 전에 확정했다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("probe_freeze")

from eval.freeze import (  # noqa: E402
    FREEZE_EPS_RAD,
    paired_on_frozen,
    FREEZE_TICKS,
    GATES,
    KICK_MIN_TICK,
    KICK_TICKS,
    as_records,
    probe_freeze,
    summarise,
)
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA_AT_LAUNCH = code_digest()


def build(env: MujocoPickEnv, name: str, ckpt: Path | None, device: str = "cuda") -> Any:
    """The policy under test. A baseline by name, or a checkpoint.
    검사 대상 정책. 이름으로 baseline, 또는 체크포인트."""
    if ckpt is not None:
        from policy.bc import BCPolicy

        bc = BCPolicy(ckpt, device=device)
        print(f"학습 정책 로드: {bc.describe()}")
        return bc
    from policy.baselines import ScriptedFeedbackPolicy, ScriptedPickPolicy

    if name == "scripted":
        return ScriptedPickPolicy(env)
    if name == "scripted_fb":
        return ScriptedFeedbackPolicy(env)
    raise SystemExit(f"--policy 는 scripted 또는 scripted_fb 여야 한다: {name!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-ckpt", type=Path, default=None)
    p.add_argument("--policy", type=str, default="scripted_fb",
                   help="체크포인트가 없을 때 쓸 baseline. 연기시험용")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=3000)
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda",
                   help="학습 정책을 올릴 장치. BCPolicy 기본값은 cpu 라 200편 × 200틱 "
                        "forward 가 CPU 2스레드로 돌아간다 — 공유 머신에서 그게 더 나쁘다")
    p.add_argument("--render", action="store_true",
                   help="시각 정책에는 필수다. baseline 연기시험에는 불필요")
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    if args.policy_ckpt is not None and not args.render:
        print("⚠️ 학습 정책인데 --render 가 없다. 이미지가 전부 0 이 되어 "
              "이 결과는 무의미하다. --render 를 붙여라")
        return 1

    cfg: dict[str, Any] = load_config()
    seeds = [args.seed_base + i for i in range(args.episodes)]

    print("사전등록 (결과 확인 전 확정 · eval/freeze.py):")
    print(f"  동결 판정: 팔 관절 이동 < {FREEZE_EPS_RAD} rad/틱 이 {FREEZE_TICKS}틱 연속 "
          f"(틱 {KICK_MIN_TICK} 이후만)")
    print(f"  kick: scripted 들기 {KICK_TICKS}틱, **1회만**. 재동결에는 다시 kick 하지 않는다")
    for k, text in GATES.items():
        print(f"  [{k}] {text}")
    print(f"\n에피소드 {args.episodes} · 시드 {seeds[0]}~{seeds[-1]} · 물체 xy ±{args.jitter * 1000:.0f}mm\n")

    out: dict[str, dict[str, Any]] = {}
    detail: dict[str, list[dict[str, Any]]] = {}
    raw: dict[str, list[Any]] = {}
    with MujocoPickEnv(cfg, render=args.render, object_jitter_m=args.jitter) as env:
        for kick in (False, True):
            pol = build(env, args.policy, args.policy_ckpt, args.device)
            res = [probe_freeze(env, pol, s, cfg, kick) for s in seeds]
            key = "kick" if kick else "none"
            out[key] = summarise(res)
            detail[key] = as_records(res)
            raw[key] = res
            s = out[key]
            print(f"{key:<6} {s['successes']:>3}/{s['n']:<4} {s['rate'] * 100:>5.1f}%   "
                  f"CI {s['ci95'][0] * 100:>4.1f}~{s['ci95'][1] * 100:<5.1f}%   "
                  f"동결 {s['froze']}편({s['froze_frac'] * 100:.0f}%)"
                  + (f" 중앙 틱 {s['freeze_tick_median']:.0f}" if s['freeze_tick_median'] else "")
                  + f" · 상승 중앙 {s['lift_mm_median']:.1f}mm")

    a, b = out["none"], out["kick"]
    print(f"\n동결 시 턱접촉 분포: {a['jaws_at_freeze']}")
    print(f"kick 한 편수 {b['kicked']} · 그중 성공 "
          f"{(b['success_given_kicked'] or 0) * 100:.1f}% · 재동결 {b['refroze']}편 "
          f"({(b['refroze_frac'] or 0) * 100:.0f}%)")

    print(f"kick 이 팔을 움직인 양 중앙 {(b['kick_moved_rad_median'] or 0):.4f} rad · "
          f"무동작 {(b['kick_noop_frac'] or 0) * 100:.0f}% · "
          f"kick 중 IK 실패 {(b['kick_ik_fail_frac'] or 0) * 100:.0f}%")
    if (b["kick_noop_frac"] or 0) > 0.30:
        print("⚠️ **kick 이 대부분 무동작이다.** 들어올림 목표에 IK 해가 없으면 kick 은 "
              "아무것도 하지 않는다. 이 상태의 '밀어줘도 안 된다'는 고정점 가설의 "
              "반증이 아니라 **계측 실패**다. 판정하지 마라")

    pair = paired_on_frozen(raw["none"], raw["kick"])
    one_push = (b["refroze_frac"] is not None) and b["refroze_frac"] < 0.30
    print(f"\n얼어붙은 편만 (n={pair['n']}) — **이것이 판정 통계다.** "
          "전체 성공률은 동결 편이 소수면 완벽한 처방도 못 움직인다")
    if pair["n"]:
        print(f"  무개입 {pair['none_successes']}/{pair['n']} = {pair['none_rate'] * 100:.1f}%  "
              f"CI {pair['none_ci95'][0] * 100:.1f}~{pair['none_ci95'][1] * 100:.1f}%")
        print(f"  kick   {pair['kick_successes']}/{pair['n']} = {pair['kick_rate'] * 100:.1f}%  "
              f"CI {pair['kick_ci95'][0] * 100:.1f}~{pair['kick_ci95'][1] * 100:.1f}%")
        print(f"  살린 시드 {len(pair['rescued'])}건 · 망친 시드 {len(pair['broken'])}건")
        for key, label in (("grasped", "파지 후 동결 (턱접촉 있음)"),
                           ("airborne", "공중에서 닫고 멈춤 (턱접촉 0)")):
            g = pair.get(key) or {}
            if not g.get("n"):
                continue
            print(f"  {label}  n={g['n']} · 무개입 {g['none']}/{g['n']} "
                  f"CI {g['none_ci95'][0] * 100:.1f}~{g['none_ci95'][1] * 100:.1f}% · "
                  f"kick {g['kick']}/{g['n']} "
                  f"CI {g['kick_ci95'][0] * 100:.1f}~{g['kick_ci95'][1] * 100:.1f}%")
        print("  ⚠️ 턱접촉 0 인 동결은 파지 후 동결이 아니다. 들기 밀기가 도울 수 없다 — "
              "합산하면 kick 효과가 희석된다")
    fixpoint = bool(pair["n"] and pair["kick_ci95"][0] > pair["none_ci95"][1])
    print("\n[fixpoint] " + (
        f"kick 하한 {pair['kick_ci95'][0] * 100:.1f}% > 무개입 상한 "
        f"{pair['none_ci95'][1] * 100:.1f}% → "
        f"{'통과 — 고정점 확정' if fixpoint else '실패 — 구간이 겹친다'}"
        if pair["n"] else "동결 편이 없다. 판정 불가"))
    print(f"[one_push_enough] 재동결 {(b['refroze_frac'] or 0) * 100:.0f}% < 30% → "
          f"{'통과' if one_push else '실패'}")
    if fixpoint and one_push:
        print("\n→ **고정점이 병목이다.** 처방은 정지 창 재표집·청킹·위상정보. "
              "DAgger 는 같은 상태에 라벨을 주므로 유효하지만 우회로다")
    elif fixpoint and not one_push:
        print("\n→ kick 이 성공률을 올렸지만 계속 다시 얼어붙는다. 고정점 하나가 병목이 "
              "아니라 **들기 구간 전체를 못 배웠다.** 처방은 재표집이 아니라 행동 표현·청킹")
    elif a["froze_frac"] < 0.30:
        print("\n→ 동결이 애초에 드물다. 이 계열을 내리고 접근 발산 계열(A′·T4)을 우선한다")
    else:
        print("\n→ 얼기는 하는데 밀어줘도 안 된다. 고정점 가설 기각. "
              "파지 후 정책이 더 깊게 망가진 것이고 이벤트 핸드오프로 국소화한다")

    if args.log:
        rec = log_run(
            experiment="freeze_kick",
            author=args.author,
            issue="project-task",
            conditions={
                "policy": str(args.policy_ckpt) if args.policy_ckpt else args.policy,
                "episodes": args.episodes, "seed_base": args.seed_base,
                "jitter_m": args.jitter, "render": args.render,
                "device": args.device,
                "freeze_eps_rad": FREEZE_EPS_RAD, "freeze_ticks": FREEZE_TICKS,
                "kick_ticks": KICK_TICKS, "kick_min_tick": KICK_MIN_TICK,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "code_sha_at_launch": CODE_SHA_AT_LAUNCH,
                "gates": GATES,
            },
            result={"summary": out, "paired_on_frozen": pair, "fixpoint_passed": fixpoint,
                    "one_push_passed": one_push, "detail": detail},
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
