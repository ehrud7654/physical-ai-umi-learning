"""Do the demonstrations actually lift the cube, or only grasp it?
시연이 실제로 큐브를 들어올리는가, 잡기만 하는가?

The collector's log says "파지 98/98" -- **grasp**, not lift. And the expert fails
its lift on 12~16% of placements: measured 2026-09-07 🟢, `scripted` freezes at
tick 97 (the first lift tick) with 8 jaw contacts on 5/40 episodes, and a
joint-space nudge rescues 5/5, so those are "grasped, lift target unreachable".
수집 로그는 "파지 98/98" 이라고만 말한다 — **파지**고 들기가 아니다. 그리고 전문가는
배치의 12~16% 에서 들기에 실패한다. 2026-09-07 실측 🟢: `scripted` 가 40편 중 5편에서
턱접촉 8개로 틱 97(들기 첫 틱)에 정지하고, 관절공간 밀기가 5/5 를 살린다. 즉
"잡았는데 들기 목표가 도달 불가" 다.

If that failure mode is present in the 98 training episodes, then some fraction of
the demonstrations **is** a grasp followed by holding still -- which is precisely
the behaviour the policy exhibits and precisely what it would be right to learn.
That would make the freeze taught by the data, not an artefact of covariate shift,
and it changes what to do next. So this is measured before anything else runs.
그 실패 모드가 98편 학습 에피소드에 섞여 있으면, 시연의 일부가 **잡고 나서 가만히
있는 것**이다 — 정책이 실제로 보이는 행동이고, 배우는 것이 옳은 행동이다. 그러면
동결은 공변량 이동의 산물이 아니라 **데이터가 가르친 것**이고 다음 할 일이 바뀐다.
그래서 다른 무엇보다 먼저 잰다.

Method: replay each episode's recorded actions with the object pinned to the
position that episode recorded, and read the lift off the simulator. Deterministic,
no training, no GPU. Reuses `ReplayPolicy` and `rollout` unchanged -- the same code
that scores every baseline.
방법: 각 에피소드의 기록 행동을 그 에피소드가 기록한 물체 위치에 고정해 재생하고
시뮬에서 상승량을 읽는다. 결정론적이고 학습도 GPU 도 필요 없다. `ReplayPolicy` 와
`rollout` 을 그대로 쓴다 — 모든 baseline 을 채점하는 바로 그 코드다.

    python tools/audit_demo_lift.py datasets/sim_pick_v5 --log
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

runtime_limits.claim("audit_demo_lift")

from eval.rollout import rollout  # noqa: E402
from policy.baselines import ReplayPolicy  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA_AT_LAUNCH = code_digest()


def object_xy(npz: Path) -> tuple[float, float] | None:
    """Where this episode recorded the object. None if the metadata lacks it.
    이 에피소드가 기록한 물체 위치. 메타에 없으면 None."""
    meta_path = npz.with_suffix(".json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = (meta.get("notes") or {}).get("object_init_xy")
    if xy is None:
        return None
    return (float(xy[0]), float(xy[1]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--jitter", type=float, default=0.05,
                   help="물체 위치가 메타에 없는 에피소드에만 쓰인다")
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    cfg: dict[str, Any] = load_config()
    need_mm = float(cfg["grasp"]["success_lift_m"]) * 1000
    eps = sorted(args.data.glob("ep_*.npz"))
    if not eps:
        print(f"에피소드가 없다: {args.data}")
        return 1

    print(f"{args.data.name}: {len(eps)}편 · 성공 판정 상승량 {need_mm:.0f}mm")
    print("기록 행동을 그 에피소드의 물체 위치에 고정해 재생한다 (결정론적)\n")

    rows: list[dict[str, Any]] = []
    missing_xy = 0
    with MujocoPickEnv(cfg, render=False, object_jitter_m=args.jitter) as env:
        for i, npz in enumerate(eps):
            xy = object_xy(npz)
            if xy is None:
                missing_xy += 1
            pol = ReplayPolicy.from_episode(npz)
            r = rollout(env, pol, seed=1_000_000 + i, object_xy=xy)
            rows.append({
                "episode": npz.stem,
                "object_xy_from_meta": xy is not None,
                "success": bool(r.success),
                "lift_mm": r.lift_height_m * 1000,
                "contact_tick": r.first_contact_tick,
                "min_pinch_xy_mm": r.min_pinch_xy_mm,
                "close_tick": r.close_tick,
            })

    if missing_xy:
        print(f"⚠️ {missing_xy}편은 메타에 `notes.object_init_xy` 가 없다. "
              "그 편들은 물체가 config 기본값에 놓이므로 **재생이 무효**다. "
              "아래 수치에서 분리해 읽어라\n")

    valid = [r for r in rows if r["object_xy_from_meta"]]
    base = valid if valid else rows
    lifted = [r for r in base if r["success"]]
    grasped_not_lifted = [r for r in base if not r["success"] and r["contact_tick"] > 0]
    never_touched = [r for r in base if not r["success"] and r["contact_tick"] < 0]

    print(f"{'들기 성공':<16} {len(lifted):>3}/{len(base)} = {len(lifted) / len(base) * 100:5.1f}%")
    print(f"{'잡았으나 못 들음':<16} {len(grasped_not_lifted):>3}/{len(base)} = "
          f"{len(grasped_not_lifted) / len(base) * 100:5.1f}%   ← 이것이 정책에게 "
          "'파지 후 정지' 를 가르친다")
    print(f"{'접촉조차 없음':<16} {len(never_touched):>3}/{len(base)}")
    print(f"\n상승량 중앙 {np.median([r['lift_mm'] for r in base]):.1f}mm "
          f"(Q1 {np.percentile([r['lift_mm'] for r in base], 25):.1f} / "
          f"Q3 {np.percentile([r['lift_mm'] for r in base], 75):.1f})")
    if grasped_not_lifted:
        g = [r["lift_mm"] for r in grasped_not_lifted]
        print(f"못 든 편의 상승량 중앙 {np.median(g):.1f}mm · 최대 {max(g):.1f}mm")
        print("해당 편: " + ", ".join(r["episode"] for r in grasped_not_lifted[:12])
              + (" ..." if len(grasped_not_lifted) > 12 else ""))

    frac = len(grasped_not_lifted) / len(base)
    print("\n판정 (결과 보기 전 확정):")
    print("  '잡았으나 못 들음' 이 0 이면 데이터는 깨끗하다 → 동결은 데이터가 "
          "가르친 것이 아니다. 정지 창·공변량 이동 계열로 진행")
    print("  5% 이상이면 **데이터가 파지 후 정지를 직접 가르친다** → 그 편들을 "
          "제외한 데이터로 재학습하는 것이 Z1 보다 먼저다")
    print(f"  실측 {frac * 100:.1f}% → "
          + ("**데이터 오염 확인. 재학습이 최우선**" if frac >= 0.05
             else "데이터는 깨끗하다. 다음 계열로 진행"))
    print("\n⚠️ 재생은 시연을 만든 전문가가 아니라 **기록된 행동**을 되돌린다. "
          "재생이 실패하는 편은 수집 당시에도 실패했거나, 수집 이후 씬·config 가 "
          "바뀐 것이다. 후자면 config_sha 를 대조해야 한다")

    if args.log:
        rec = log_run(
            experiment="audit_demo_lift",
            author=args.author,
            issue="project-task",
            conditions={
                "dataset": str(args.data), "n_episodes": len(eps),
                "success_lift_m": float(cfg["grasp"]["success_lift_m"]),
                "config_sha": file_digest(DEFAULT_CONFIG),
                "code_sha_at_launch": CODE_SHA_AT_LAUNCH,
                "episodes_with_meta_xy": len(valid),
            },
            result={
                "lifted": len(lifted),
                "grasped_not_lifted": len(grasped_not_lifted),
                "never_touched": len(never_touched),
                "grasped_not_lifted_frac": frac,
                "lift_mm_median": float(np.median([r["lift_mm"] for r in base])),
                "detail": rows,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
