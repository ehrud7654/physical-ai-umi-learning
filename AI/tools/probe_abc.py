"""A / B' / C — does the policy's own on-demo error break the recorded trajectory?
A / B' / C — 정책 자신의 시연상 오차가 기록된 궤적을 깨뜨리는가?

    A       a_t = a_t^demo            the recording. Known 98/98 with the object pinned
    B'_all  a_t = pi(o_t^demo)        the policy's own sequence, replayed open-loop
    B'_arm  arm from pi, gripper from the recording
    B'_grip gripper from pi, arm from the recording
    C       a_t = pi(o_t^actual)      the real closed loop

B' preserves the signed bias, the joint-to-joint covariance, the phase dependence
and the temporal correlation of the policy's residual exactly, because it *is* the
policy's residual rather than a model of it. Nothing has to be chosen: no block
length, no injection space, no scale factor.
B' 는 정책 잔차의 부호·관절 간 공분산·위상 의존성·시간 상관을 정확히 보존한다.
잔차의 모형이 아니라 잔차 **그 자체**이기 때문이다. 고를 것이 없다 — 블록 길이도,
주입 공간도, 배율도.

The comparison only means something if the three run under identical conditions.
The numbers already on record do not: replay 98/98 pinned the object, the BC rollout
9.3% jittered it by +-50mm. So C is measured again here, pinned.
세 조건이 동일 설정에서 돌아야만 비교가 성립한다. 기록에 남은 수치는 그렇지 않다 —
재생 98/98 은 물체를 고정했고 BC 롤아웃 9.3% 는 +-50mm 지터를 줬다. 그래서 C 를
여기서 고정 조건으로 다시 잰다.

Pre-registered in `docs/PREREG_ABC_0908.md` before any of this ran. The judgment
rules and the "round 2 runs regardless of round 1" commitment are there, not here.
`docs/PREREG_ABC_0908.md` 에 실행 전 사전등록했다. 판정 규칙과 "2차는 1차 결과와
무관하게 돈다" 는 약속은 여기가 아니라 거기 있다.

    # [서버]
    python tools/probe_abc.py datasets/sim_pick_v5 \
        out/residual_dump/sim_pick_v5_seed0__sim_pick_v5 \
        checkpoints/bc/sim_pick_v5_seed0.pt --device cpu --log
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

runtime_limits.claim("probe_abc")

from eval.residual_dump import (  # noqa: E402
    CONDITIONS, assemble, heldout_episode_ids, read_dump, seed_from_ckpt_name,
)
from eval.rollout import rollout  # noqa: E402
from eval.stats import wilson_ci  # noqa: E402
from policy.baselines import ReplayPolicy  # noqa: E402
from policy.bc import BCPolicy, gripper_index  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA_AT_LAUNCH = code_digest()

# The rollout gate this project judges policies against. Not moved for this run.
# 이 프로젝트가 정책을 판정하는 롤아웃 게이트. 이번 실행에서 옮기지 않는다.
ROLLOUT_GATE = 0.20

# Seed base shared with `tools/audit_demo_lift.py`, so condition A re-runs the exact
# rollouts that produced 98/98 and any difference is an instrument failure, not a
# different experiment.
# `tools/audit_demo_lift.py` 와 같은 시드 기준. 그래서 조건 A 는 98/98 을 낸 바로 그
# 롤아웃을 다시 돌리고, 차이가 나면 그건 다른 실험이 아니라 계측기 고장이다.
SEED_BASE = 1_000_000


def _fmt(successes: int, n: int) -> str:
    if n == 0:
        return "   -/0"
    lo, hi = wilson_ci(successes, n)
    return f"{successes:>3}/{n} = {successes / n * 100:5.1f}%   [{lo * 100:5.1f}, {hi * 100:5.1f}]"


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def object_xy(npz: Path) -> tuple[float, float] | None:
    """Where this episode recorded the object, read from its own metadata.
    이 에피소드가 기록한 물체 위치. 자기 메타데이터에서 읽는다.

    Read here rather than from the dump so that condition A does not depend on a
    dump existing. A is the recording replayed against itself: it has nothing to
    do with any policy, and needing a policy's dump to run it would be an
    accidental coupling that hides when the baseline itself has drifted.
    덤프가 아니라 여기서 읽는다. 그래야 조건 A 가 덤프의 존재에 의존하지 않는다.
    A 는 기록을 자기 자신에 대해 재생하는 것이고 어떤 정책과도 무관하다. 그것을
    돌리려고 정책의 덤프가 필요하다면, 기준선 자체가 밀렸을 때 그것을 가리는
    우연한 결합이 된다.
    """
    meta_path = npz.with_suffix(".json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    xy = (meta.get("notes") or {}).get("object_init_xy")
    return (float(xy[0]), float(xy[1])) if xy is not None else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("ckpt", type=Path)
    p.add_argument("--dump", type=Path, default=None,
                   help="tools/dump_residuals.py 의 출력 디렉터리. "
                        "B_* 조건에만 필요하다 — A 와 C 는 없어도 돈다")
    p.add_argument("--device", type=str, default="cpu",
                   help="덤프를 만든 장치와 같아야 한다 (L58)")
    p.add_argument("--only", type=str, default=None,
                   help="조건 하나만 (A|B_all|B_arm|B_grip|C). 계측기 검증용")
    p.add_argument("--train-seed", type=int, default=None,
                   help="이 ckpt 를 학습한 시드. 생략하면 파일명 _seed{N} 에서 "
                        "추론한다. 홀드아웃 분리 보고에만 쓴다")
    p.add_argument("--jitter-c", action="store_true",
                   help="C 를 물체 고정이 아니라 지터로 돌린다. V0-5 검증 전용 — "
                        "이 결과는 A·B' 와 비교 불가다")
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    runtime_limits.torch_threads()

    cfg: dict[str, Any] = load_config()
    g = gripper_index()
    eps = sorted(args.data.glob("ep_*.npz"))
    if not eps:
        print(f"에피소드가 없다: {args.data}")
        return 1

    todo = [args.only] if args.only else [*CONDITIONS, "C"]
    for c in todo:
        if c not in (*CONDITIONS, "C"):
            print(f"모르는 조건: {c} (가능한 값 {[*CONDITIONS, 'C']})")
            return 1
    needs_dump = [c for c in todo if c.startswith("B_")]

    dumps: dict[str, Any] = {}
    if needs_dump:
        if args.dump is None:
            print(f"조건 {needs_dump} 은 덤프가 필요하다. --dump 를 줘라")
            print("  python tools/dump_residuals.py <데이터셋> <ckpt> --device cpu --log")
            return 1
        dumps = {d.stem: read_dump(d) for d in sorted(args.dump.glob("ep_*.npz"))}
        missing = [e.stem for e in eps if e.stem not in dumps]
        if missing:
            print(f"덤프에 없는 에피소드 {len(missing)}편: {missing[:5]}")
            print("덤프와 데이터셋이 어긋났다. 같은 데이터셋으로 다시 덤프해라")
            return 1

    xy_of = {e.stem: object_xy(e) for e in eps}
    no_xy = [s for s, v in xy_of.items() if v is None]
    if no_xy:
        print(f"⚠️ {len(no_xy)}편은 `notes.object_init_xy` 가 없어 물체를 고정할 수 "
              "없다. **그 편의 재생은 무효다.** 전량 제외하고 돈다\n")
    usable = [e for e in eps if xy_of[e.stem] is not None]
    n = len(usable)
    if n == 0:
        print("물체 위치를 가진 에피소드가 없다. 비교가 성립하지 않는다")
        return 1

    # ---- 홀드아웃 분리 ------------------------------------------------------
    # A policy scored only on the episodes it trained on reports memorisation
    # mixed with generalisation, and the two cannot be separated afterwards.
    # 학습에 쓴 편에서만 채점한 정책 수치는 암기와 일반화가 섞인 값이고,
    # 사후에는 둘을 분리할 수 없다.
    train_seed = (args.train_seed if args.train_seed is not None
                  else seed_from_ckpt_name(args.ckpt.name))
    val_fraction = float(BCPolicy(args.ckpt, device="cpu").meta["train_config"]
                         ["train"]["val_fraction"])
    ckpt_n_eps = int(BCPolicy(args.ckpt, device="cpu").meta.get("n_episodes", -1))
    heldout: set[str] = set()
    split_ok = False
    if train_seed is None:
        print("⚠️ ckpt 파일명에서 학습 시드를 못 읽었다. 홀드아웃 분리를 생략한다 "
              "— 아래 수치는 학습편과 홀드아웃편이 섞인 값이다")
    elif ckpt_n_eps != len(eps):
        print(f"⚠️ ckpt 는 {ckpt_n_eps}편으로 학습됐다고 기록하는데 데이터셋에는 "
              f"{len(eps)}편이 있다. 계약 위반으로 탈락한 편이 있으면 에피소드 "
              "인덱스가 밀린다. **홀드아웃 분리를 생략한다**")
    else:
        ids = set(heldout_episode_ids(len(eps), val_fraction, train_seed))
        heldout = {e.stem for i, e in enumerate(eps) if i in ids}
        split_ok = True
        print(f"학습 시드 {train_seed} (ckpt 파일명에서 추론 — 관례이지 기록이 "
              f"아니다) · val_fraction {val_fraction} → 홀드아웃 {len(heldout)}편")

    ckpt_sha = file_digest(args.ckpt)
    # Only condition C looks at the observation, and `BCPolicy.act` raises if the
    # cameras are missing. A and B' ignore it, so rendering is pure overhead for
    # them -- but every condition in one run must see the same environment, so the
    # switch is per-run and not per-condition.
    # 조건 C 만 관측을 본다. 카메라가 없으면 `BCPolicy.act` 가 예외를 낸다.
    # A·B' 는 관측을 무시하니 렌더는 순수 낭비지만, 한 실행 안의 모든 조건은 같은
    # 환경을 봐야 하므로 스위치는 조건별이 아니라 실행별이다.
    render = "C" in todo
    print(f"{args.data.name}: {n}편 · ckpt {args.ckpt.name} ({ckpt_sha}) · "
          f"장치 {args.device}")
    print(f"그리퍼 인덱스 {g} (configs/so101.yaml 에서 읽음) · "
          f"lift_height_m {cfg['grasp']['lift_height_m']}")
    print(f"조건 {todo} · render {render}")
    if render and "A" in todo:
        print("⚠️ A 를 render=True 로 돈다. `audit_demo_lift` 의 98/98 은 "
              "render=False 였다. 물리가 렌더에 무관하다는 것은 **미검증 가정**이고, "
              "아래 V0-1 이 그것을 검사한다")
    print()

    bc = BCPolicy(args.ckpt, device=args.device) if "C" in todo else None
    results: dict[str, list[bool]] = {}
    detail: dict[str, list[dict[str, Any]]] = {}

    with MujocoPickEnv(cfg, render=render, object_jitter_m=args.jitter) as env:
        for cond in todo:
            hits: list[bool] = []
            rows: list[dict[str, Any]] = []
            for i, npz in enumerate(usable):
                xy = None if (cond == "C" and args.jitter_c) else xy_of[npz.stem]
                if cond == "C":
                    pol = bc
                elif cond == "A":
                    pol = ReplayPolicy.from_episode(npz)
                else:
                    pol = ReplayPolicy(assemble(cond, dumps[npz.stem], g),
                                       source=f"{cond}:{npz.stem}")
                r = rollout(env, pol, seed=SEED_BASE + i, object_xy=xy)
                hits.append(bool(r.success))
                rows.append({
                    "episode": npz.stem,
                    "heldout": npz.stem in heldout,
                    "success": bool(r.success),
                    "lift_mm": round(r.lift_height_m * 1000, 2),
                    "contact_tick": r.first_contact_tick,
                    "close_tick": r.close_tick,
                    "min_pinch_xy_mm": r.min_pinch_xy_mm,
                    "ticks": r.ticks,
                })
            results[cond] = hits
            detail[cond] = rows
            line = f"{cond:<8} {_fmt(sum(hits), n)}"
            if split_ok:
                tr = [r["success"] for r in rows if not r["heldout"]]
                ho = [r["success"] for r in rows if r["heldout"]]
                line += (f"   학습 {_fmt(sum(tr), len(tr))}"
                         f"   홀드아웃 {_fmt(sum(ho), len(ho))}")
            print(line)

    if split_ok:
        print("\n⚠️ 홀드아웃 n 이 작아 구간이 넓다. 학습·홀드아웃 차이는 이 n 으로 "
              "구분되지 않을 수 있다 — 겹치면 '차이 없음' 이 아니라 '구분 불가' 다")
        print("⚠️ 홀드아웃 집합은 학습 시드가 정한다. 즉 seed0·1·2 는 서로 다른 "
              "편을 홀드아웃한다. 시드 간 비교에는 초기화 분산과 분할 분산이 "
              "섞여 있고 분리 불가다 (LIMITS 등재 대상)")
    print()

    # ---- 계측기 검증 --------------------------------------------------------
    # A failed check invalidates the comparison, so it is reported before any
    # reading of the numbers.
    # 검증 실패는 비교 자체를 무효로 만든다. 그래서 숫자를 읽기 전에 먼저 낸다.
    checks: dict[str, Any] = {}
    if "A" in results:
        a_hits = sum(results["A"])
        checks["V0-1_A_all_success"] = (a_hits == n)
        if a_hits != n:
            print(f"❌ V0-1 실패 — A(기록 재생)가 {a_hits}/{n} 이다. "
                  "`audit_demo_lift` 는 98/98 이었다.")
            print("   후보: (1) 렌더가 물리를 바꾼다 — `--only A` 로 render=False "
                  "재실행해 갈라라. (2) 씬·config·코드가 그 사이에 바뀌었다 — "
                  "config_sha·code_sha 대조.")
            print("   어느 쪽인지 가리기 전에는 아래 수치를 읽지 마라\n")
        else:
            print(f"✅ V0-1 — A 가 {n}/{n}. 기준선 재현됨")
    if "B_all" in results and "C" in results and not args.jitter_c:
        # B'_all and C differ only in which observation the policy saw. If they are
        # bit-identical the closed loop never left the demonstrated states, which
        # would mean this experiment cannot separate anything.
        # B'_all 과 C 는 정책이 어느 관측을 봤는지만 다르다. 둘이 비트 동일하면
        # 폐루프가 시연 상태를 한 번도 벗어나지 않은 것이고, 그러면 이 실험은
        # 아무것도 분리하지 못한다.
        same = results["B_all"] == results["C"]
        checks["B_all_equals_C"] = same
        if same:
            print("⚠️ B'_all 과 C 의 성공/실패가 편별로 완전히 같다. "
                  "폐루프가 시연 상태를 벗어나지 않았거나 두 경로가 같은 코드를 "
                  "타고 있다. 분리 실패 — 원인을 찾기 전에는 결론을 내지 마라")

    # ---- 판정 (사전등록 docs/PREREG_ABC_0908.md) ----------------------------
    verdict = "미판정"
    if "A" in results and "B_all" in results:
        a_ci = wilson_ci(sum(results["A"]), n)
        b_ci = wilson_ci(sum(results["B_all"]), n)
        print(f"\n판정 (사전등록 docs/PREREG_ABC_0908.md §4, 게이트 "
              f"{ROLLOUT_GATE * 100:.0f}%)")
        if b_ci[1] < ROLLOUT_GATE:
            verdict = "J1"
            print("  **J1** — B' 의 구간 상한이 게이트 아래다. 되먹임 없이도 "
                  "시연 관측에서의 예측 오차만으로 배포 기준 미만이다.")
            print("  다음: one-step 정확도 · 행동 표현 · 시퀀스 모델. "
                  "DAgger 착수하지 않는다")
        elif _overlaps(a_ci, b_ci):
            verdict = "J2"
            print("  **J2** — B' 구간이 A 와 겹친다. 시연 관측에서의 예측 오차는 "
                  "개루프에서 검출 가능한 피해를 주지 않는다.")
            print("  다음: O(alpha) vs C(alpha), 그리고 exact-state phase start")
        else:
            verdict = "J3"
            print("  **J3** — B' 가 A 보다 유의하게 낮지만 상한이 게이트 이상이다. "
                  "예측 오차가 일부 기여하나 단독 설명은 아니다.")
            print("  다음: 사전등록 단계 2 전체")
        print("\n  ⚠️ 이 판정은 체크포인트 하나의 것이다. 같은 데이터·설정에서 "
              "학습 실행 간 성공률이 25%p 갈린다 🟢 — 2차(seed1·seed2)는 이 결과와 "
              "무관하게 돈다. 그때까지 처방을 확정하지 않는다")

    if "B_arm" in results and "B_grip" in results:
        arm_ci = wilson_ci(sum(results["B_arm"]), n)
        grip_ci = wilson_ci(sum(results["B_grip"]), n)
        a_ci = wilson_ci(sum(results["A"]), n) if "A" in results else None
        print("\n귀속 (그리퍼 모델링 변경 대비)")
        arm_hurt = a_ci is not None and not _overlaps(a_ci, arm_ci)
        grip_hurt = a_ci is not None and not _overlaps(a_ci, grip_ci)
        if arm_hurt and not grip_hurt:
            print("  팔 잔차가 주원인 → **그리퍼 기하가 바뀌어도 결론이 살아남는다**")
        elif grip_hurt and not arm_hurt:
            print("  그리퍼 잔차가 주원인 → **그리퍼 모델링이 바뀌면 전량 재측정**이다")
        elif arm_hurt and grip_hurt:
            print("  양쪽 모두 기여 → 그리퍼 변경 시 최소한 그리퍼 조건은 재측정")
        else:
            print("  둘 다 A 와 구간이 겹친다 → 이 n 으로는 귀속 불가")
        print("  ⚠️ B'_arm 의 그리퍼는 기록 명령이다. 전체 BC 엔드포인트가 아니다")

    if args.log:
        rec = log_run(
            experiment="probe_abc",
            author=args.author,
            issue="project-task",
            conditions={
                "dataset": str(args.data), "dump": str(args.dump),
                "n_episodes": n, "excluded_no_object_xy": len(no_xy),
                "ckpt": args.ckpt.name, "ckpt_sha": ckpt_sha,
                "device": args.device, "conditions": todo,
                "gripper_index": g, "render": render,
                "train_seed_inferred": train_seed, "val_fraction": val_fraction,
                "heldout_split_reported": split_ok,
                "heldout_episodes": sorted(heldout),
                "lift_height_m": float(cfg["grasp"]["lift_height_m"]),
                "jitter_c": bool(args.jitter_c),
                "seed_base": SEED_BASE, "gate": ROLLOUT_GATE,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "code_sha_at_launch": CODE_SHA_AT_LAUNCH,
                "prereg": "docs/PREREG_ABC_0908.md",
            },
            result={
                # Per-checkpoint only. Rollouts that share a training run are not
                # independent Bernoulli draws, so nothing here may be pooled
                # across checkpoints into one interval.
                # 체크포인트 단위로만 낸다. 같은 학습 실행을 공유하는 롤아웃은
                # 독립 베르누이 시행이 아니므로, 여기 값을 체크포인트 간에 합쳐
                # 하나의 구간으로 만들면 안 된다.
                "success": {c: sum(v) for c, v in results.items()},
                "wilson": {c: [round(x, 4) for x in wilson_ci(sum(v), n)]
                           for c, v in results.items()},
                "success_heldout": {
                    c: sum(r["success"] for r in detail[c] if r["heldout"])
                    for c in results
                } if split_ok else None,
                "n_heldout": len(heldout) if split_ok else None,
                "verdict": verdict,
                "checks": checks,
                "detail": detail,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
