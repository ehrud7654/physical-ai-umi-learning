"""Does commanding a taller lift than success requires cost us the ceiling?
성공에 필요한 것보다 높게 들라고 명령하는 것이 천장을 깎고 있나?

`success_lift_m` is 0.05 but `lift_height_m` commands 0.08. The extra 3 cm buys
nothing the score counts, and it costs reachability: the lift target sits higher,
so more object placements have no IK solution there. Measured 2026-09-07 🟢 --
`scripted_fb` failed 39/300 and 38 of those were lift IK failures, and a separate
measurement found the lift IK rejecting 25 of 100 placements (project-task).
`success_lift_m` 은 0.05 인데 `lift_height_m` 은 0.08 을 명령한다. 남는 3cm 는
채점에 아무것도 보태지 않고, 도달성을 깎는다 — 목표가 더 높으니 IK 해가 없는
물체 배치가 늘어난다. 2026-09-07 실측 🟢: `scripted_fb` 실패 39/300 중 38건이
lift IK 실패였고, 별도 계측에서 들어올림 IK 가 100개 중 25개를 탈락시켰다(이슈 64).

This sweeps the commanded height and reports the ceiling at each. It changes the
demonstrator, so a height that wins here means **re-collecting data**, not just
swapping a number -- that cost is stated with the result rather than discovered
after.
명령 높이를 훑어 각 높이의 천장을 보고한다. 시연자를 바꾸는 값이므로, 여기서
이기는 높이는 숫자 교체가 아니라 **재수집**을 뜻한다. 그 비용은 결과를 보고
발견하는 것이 아니라 결과와 함께 적는다.

    python tools/lift_height_sweep.py --episodes 300 --heights 0.055,0.06,0.07,0.08 --log
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("lift_height_sweep")

import numpy as np  # noqa: E402

from eval.recovery import as_records, probe, summarise  # noqa: E402
from policy.baselines import ScriptedFeedbackPolicy, ScriptedPickPolicy  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import file_digest, log_run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed-base", type=int, default=3000)
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--heights", type=str, default="0.055,0.06,0.07,0.08",
                   help="명령할 들어올림 높이 (m), 콤마 구분")
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    base: dict[str, Any] = load_config()
    need = float(base["grasp"]["success_lift_m"])
    seeds = [args.seed_base + i for i in range(args.episodes)]
    heights = [float(h) for h in args.heights.split(",") if h.strip()]

    print(f"성공 판정 상승량 {need * 100:.1f}cm · 현재 명령 높이 "
          f"{float(base['grasp']['lift_height_m']) * 100:.1f}cm")
    print(f"에피소드 {args.episodes} · 시드 {seeds[0]}~{seeds[-1]} · 물체 xy ±{args.jitter * 1000:.0f}mm\n")
    too_low = [h for h in heights if h <= need]
    if too_low:
        print(f"⚠️ {too_low} 는 성공 판정 상승량 {need}m 이하다. 명령 높이가 판정선보다 "
              "낮으면 성공할 수 없다 — 비교 대상이 아니라 하한 확인용이다\n")

    out: dict[str, dict[str, Any]] = {}
    detail: dict[str, list[dict[str, Any]]] = {}
    print(f"{'명령':<7} {'정책':<13} {'성공':<9} {'성공률':<7}  95% 구간        "
          f"{'달성 상승량 (mm)':<22} IK실패")
    print("-" * 96)
    for h in heights:
        cfg = copy.deepcopy(base)
        cfg["grasp"]["lift_height_m"] = h
        with MujocoPickEnv(cfg, render=False, object_jitter_m=args.jitter) as env:
            for name, ctor in (("scripted", ScriptedPickPolicy),
                               ("scripted_fb", ScriptedFeedbackPolicy)):
                pol = ctor(env)
                # 성공에서 끊지 않는다 — 여기서 묻는 것은 합격 여부가 아니라
                # 팔이 얼마나 올릴 수 있는가다. 기본값으로는 성공한 편의 상승량이
                # 판정선을 넘는 순간의 값이 되어 물리 한계처럼 읽힌다 🟢 2026-09-07.
                res = [probe(env, pol, s, cfg, None, stop_on_success=False) for s in seeds]
                key = f"{h}|{name}"
                out[key] = summarise(res)
                detail[key] = as_records(res)
                s = out[key]
                lifts = [r.max_lift_mm for r in res]
                s["lift_mm_median"] = float(np.median(lifts))
                s["lift_mm_q1"] = float(np.percentile(lifts, 25))
                s["lift_mm_q3"] = float(np.percentile(lifts, 75))
                s["commanded_mm"] = h * 1000
                s["achieved_over_commanded"] = s["lift_mm_median"] / (h * 1000)
                print(f"{h * 100:>4.1f}cm {name:<13} {s['successes']:>3}/{s['n']:<4} "
                      f"{s['rate'] * 100:>5.1f}%   "
                      f"CI {s['ci95'][0] * 100:>4.1f}~{s['ci95'][1] * 100:<5.1f}%  "
                      f"최대 중앙 {s['lift_mm_median']:>5.1f} "
                      f"(Q1 {s['lift_mm_q1']:>5.1f} / Q3 {s['lift_mm_q3']:>5.1f}) "
                      f"명령대비 {s['achieved_over_commanded'] * 100:>3.0f}%  "
                      f"{s['ik_fail_by_phase'].get('lift', 0)}")
        print("-" * 76)

    print(f"\n판정 기준선 {need * 1000:.0f}mm. 상승량은 **에피소드 전체의 최댓값**이다 "
          "— 성공에서 끊지 않고 틱 제한까지 돌린다. 그렇게 하지 않으면 성공한 편의 "
          "값이 판정선을 넘은 순간의 높이가 되어 물리 한계처럼 읽힌다")

    valid = [h for h in heights if h > need]
    if valid:
        best = max(valid, key=lambda h: out[f"{h}|scripted_fb"]["rate"])
        cur = float(base["grasp"]["lift_height_m"])
        b, c = out[f"{best}|scripted_fb"], out.get(f"{cur}|scripted_fb")
        print(f"\nscripted_fb 최고: {best * 100:.1f}cm {b['rate'] * 100:.1f}% "
              f"(CI {b['ci95'][0] * 100:.1f}~{b['ci95'][1] * 100:.1f}%)")
        if c is not None:
            sep = not (b["ci95"][0] <= c["ci95"][1] and c["ci95"][0] <= b["ci95"][1])
            print(f"현재 {cur * 100:.1f}cm {c['rate'] * 100:.1f}% "
                  f"(CI {c['ci95'][0] * 100:.1f}~{c['ci95'][1] * 100:.1f}%) 와 "
                  f"{'구간이 갈라진다 — 천장이 실제로 오른다' if sep else '구간이 겹친다 — 이 n 으로 구분되지 않는다'}")
            if sep:
                print("⚠️ 이 값을 채택하면 시연자가 바뀐다. **데이터 재수집이 따라온다.** "
                      "숫자 교체로 끝나지 않는다")

    if args.log:
        rec = log_run(
            experiment="lift_height_sweep",
            author=args.author,
            issue="project-task",
            conditions={
                "episodes": args.episodes, "seed_base": args.seed_base,
                "jitter_m": args.jitter, "heights_m": heights,
                "success_lift_m": need,
                "current_lift_height_m": float(base["grasp"]["lift_height_m"]),
                "config_sha": file_digest(DEFAULT_CONFIG),
            },
            result={"summary": out, "detail": detail},
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
