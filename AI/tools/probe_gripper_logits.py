"""Is the gripper probability pinned low, or does it oscillate?
그리퍼 확률이 눌려 있는가, 요동치는가.

    python tools/probe_gripper_logits.py --help

## 왜

`joint_delta_gripper_binary` 에서 그리퍼 채널은 **로짓**이고 `to_action` 이 부호만
남긴다. 크기는 아무도 안 본다. 그래서 미폐쇄 67/100(seed0) 이 두 가지 중 무엇인지
구분되지 않는다 — **처방이 서로 다르다.**

- **눌림** p 가 내내 0.5 아래 → 다수 클래스 붕괴가 BCE 에서도 재발. 임계값·pos_weight
- **요동** p 가 넘나들되 유지 안 됨 → 관측이 순간을 분해 못 함. 시간 문맥·카메라

판정 기준은 `docs/PREREG_gripper_logits_0911.md` 에 결과 보기 전에 박았다.

## 한계

- 시뮬이다. 실물 전이를 보장하지 않는다
- 기본 패스는 **조기종료를 끈다** (성공 후 p 궤적도 봐야 한다) → 그 성공률은
  기존 6/69/25 와 직접 비교하지 않는다. 비교는 임계값 0.5 스윕이 한다
- 임계값 하향은 처방 **후보**다. 채택하려면 3회 반복 + 사전등록이 필요하다
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402

runtime_limits.claim("probe_gripper_logits")
runtime_limits.torch_threads()

import numpy as np  # noqa: E402

from policy.bc import BCPolicy, gripper_command_norms  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA = code_digest()
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic.
    수치적으로 안정한 로지스틱."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class ThresholdPolicy:
    """Close the gripper when p(close) crosses a chosen threshold.
    p(close) 가 정한 임계값을 넘으면 닫는다.

    0.5 는 현행 `to_action`(로짓 부호)과 같아야 한다 — **그 자체가 계측기 검증이다.**
    """

    def __init__(self, base: BCPolicy, threshold: float, gripper_index: int,
                 open_norm: float, close_norm: float) -> None:
        self.base = base
        self.threshold = float(threshold)
        self._g = int(gripper_index)
        self._open = float(open_norm)
        self._close = float(close_norm)

    def reset(self, seed: int) -> None:
        self.base.reset(seed=seed)

    def act(self, obs: Any) -> np.ndarray:
        a = np.asarray(self.base.act(obs), dtype=np.float64).copy()
        p = float(sigmoid(np.asarray([self.base.last_raw[self._g]]))[0])
        a[self._g] = self._close if p >= self.threshold else self._open
        return a


def run_episode(
    env: MujocoPickEnv, policy: Any, base: BCPolicy, seed: int,
    gripper_index: int, stop_on_success: bool,
) -> dict[str, Any]:
    """One episode, keeping the per-frame close probability.
    에피소드 하나. 프레임별 폐쇄 확률을 남긴다."""
    obs = env.reset(seed=seed)
    policy.reset(seed)
    ps: list[float] = []
    xys: list[float] = []
    success = False
    for _ in range(env.max_ticks):
        obs = env.step(policy.act(obs))
        ps.append(float(sigmoid(np.asarray([base.last_raw[gripper_index]]))[0]))
        xy, _ = env.pinch_to_object_m()
        xys.append(float(xy))
        if env.is_success():
            success = True
            if stop_on_success:
                break

    p = np.asarray(ps, dtype=np.float64)
    xy_arr = np.asarray(xys, dtype=np.float64)
    over = p >= 0.5
    # 첫 **유지** 통과 — 스치고 되돌아오는 것은 닫은 것이 아니다.
    # `closing_moment` 과 같은 전제를 쓴다.
    first_hold = -1
    for i in range(p.size):
        if over[i] and bool(over[i:].all()):
            first_hold = i
            break
    return {
        "seed": seed,
        "success": success,
        "ticks": int(p.size),
        "p_max": float(p.max()) if p.size else float("nan"),
        "p_median": float(np.median(p)) if p.size else float("nan"),
        "frames_over_half": int(over.sum()),
        "ever_over_half": bool(over.any()),
        "first_hold_tick": int(first_hold),
        "nearest_tick": int(np.argmin(xy_arr)) if xy_arr.size else -1,
        "nearest_xy_mm": float(xy_arr.min() * 1000.0) if xy_arr.size else float("nan"),
    }


def dist(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "max": None, "n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max()), "n": int(a.size)}


def shape_verdict(p_max_median: float | None) -> str:
    """Pre-registered read of the un-closed p_max median.
    미폐쇄 p_max 중앙값에 대한 사전등록 판정."""
    if p_max_median is None:
        return "no_unclosed_episodes"
    if p_max_median < 0.35:
        return "pinned__probability_never_rises"
    if p_max_median <= 0.5:
        return "edge_pinned__threshold_may_be_cheap"
    return "oscillating__observation_cannot_resolve_moment"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-ckpt", type=Path, action="append", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--policy-device", type=str, default="cpu")
    ap.add_argument("--threshold-sweep", action="store_true",
                    help=f"임계값 {THRESHOLDS} 로 재롤아웃 (조기종료 켬)")
    ap.add_argument("--expected-at-half", type=int, action="append", default=[],
                    help="임계값 0.5 스윕의 기존 성공 수. 계측기 검증")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        raise FileExistsError(args.out)
    if args.expected_at_half and len(args.expected_at_half) != len(args.policy_ckpt):
        ap.error("--expected-at-half 개수가 --policy-ckpt 개수와 다르다")

    cfg = load_config()
    joints = sorted(cfg["joints"], key=lambda r: int(r["index"]))
    gripper_index = next(int(j["index"]) for j in joints if j["name"] == "gripper")
    open_norm, close_norm = gripper_command_norms()
    seeds = [args.seed_base + i for i in range(args.episodes)]

    print("그리퍼 로짓 분포 계측 — 눌림인가 요동인가")
    print(f"조건: {args.episodes}편 · seeds {seeds[0]}~{seeds[-1]} · render · "
          f"policy-device {args.policy_device} · jitter ±{args.jitter * 1000:.0f}mm")
    print(f"sim cfg sha {file_digest(DEFAULT_CONFIG)} · "
          f"개폐 명령 norm open {open_norm:+.4f} / close {close_norm:+.4f}")
    print("⚠️ 기본 패스는 조기종료를 끈다 — 그 성공률은 기존 6/69/25 와 직접 비교하지 않는다\n")

    results: dict[str, Any] = {}
    with MujocoPickEnv(cfg=cfg, render=True, object_jitter_m=args.jitter,
                       max_ticks=200) as env:
        for idx, ckpt in enumerate(args.policy_ckpt):
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            base = BCPolicy(ckpt, device=args.policy_device)
            label = ckpt.stem
            print(f"===== {label} =====")

            eps = []
            for k, seed in enumerate(seeds):
                eps.append(run_episode(env, base, base, seed, gripper_index, False))
                if (k + 1) % 25 == 0:
                    print(f"  기본 {k + 1}/{len(seeds)}", flush=True)

            closed = [e for e in eps if e["first_hold_tick"] >= 0]
            unclosed = [e for e in eps if e["first_hold_tick"] < 0]
            p_unclosed = dist([e["p_max"] for e in unclosed])
            p_closed = dist([e["p_max"] for e in closed])
            lag = [e["first_hold_tick"] - e["nearest_tick"] for e in closed
                   if e["nearest_tick"] >= 0]
            verdict = shape_verdict(p_unclosed["p50"])

            print(f"  미폐쇄 {len(unclosed)}/{args.episodes} · "
                  f"폐쇄 {len(closed)}/{args.episodes}")
            print(f"  미폐쇄 p_max  p50 {p_unclosed['p50']} · p90 {p_unclosed['p90']} · "
                  f"max {p_unclosed['max']}")
            print(f"  폐쇄   p_max  p50 {p_closed['p50']}")
            if lag:
                print(f"  폐쇄 시점 − 최근접 틱  중앙 {float(np.median(lag)):+.1f}틱 "
                      f"(n={len(lag)})")
            print(f"  판정 {verdict}")

            sweep = []
            if args.threshold_sweep:
                for thr in THRESHOLDS:
                    wrapped = ThresholdPolicy(base, thr, gripper_index,
                                              open_norm, close_norm)
                    hits = sum(
                        run_episode(env, wrapped, base, s, gripper_index, True)["success"]
                        for s in seeds
                    )
                    sweep.append({"threshold": thr, "successes": int(hits),
                                  "success_rate": hits / args.episodes})
                    print(f"  임계값 {thr:.1f} → {hits}/{args.episodes}", flush=True)
                at_half = next(s["successes"] for s in sweep if s["threshold"] == 0.5)
                if args.expected_at_half:
                    want = args.expected_at_half[idx]
                    if at_half != want:
                        raise RuntimeError(
                            f"instrument invalid: {label} 임계값 0.5 에서 "
                            f"{at_half}/{args.episodes}, 기대 {want}. "
                            "git_rev·config sha·체크포인트 경로를 먼저 대조하라"
                        )
                    print(f"  임계값 0.5 재현 fixture PASS ({at_half})")

            results[label] = {
                "episodes": args.episodes,
                "unclosed": len(unclosed),
                "closed": len(closed),
                "p_max_unclosed": p_unclosed,
                "p_max_closed": p_closed,
                "hold_lag_vs_nearest_median": (
                    float(np.median(lag)) if lag else None
                ),
                "shape_verdict": verdict,
                "threshold_sweep": sweep,
                "episode_detail": eps,
            }

    payload = {
        "experiment": "gripper_logits",
        "conditions": {
            "episodes": args.episodes, "seed_base": args.seed_base,
            "seed_end": seeds[-1], "jitter_m": args.jitter, "render": True,
            "policy_device": args.policy_device, "max_ticks": 200,
            "base_pass_stop_on_success": False,
            "sweep_stop_on_success": True,
            "thresholds": list(THRESHOLDS) if args.threshold_sweep else [],
            "policy_checkpoints": [str(c) for c in args.policy_ckpt],
            "sim_config_sha": file_digest(DEFAULT_CONFIG),
            "code_sha_at_launch": CODE_SHA,
        },
        "results": results,
        "interpretation_limit": (
            "Simulation only. Base-pass success rates are not comparable to the "
            "recorded 6/69/25 because early stopping is disabled. Threshold "
            "lowering is a candidate, not a decision."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장 {args.out}")
    if args.log:
        rec = log_run(experiment="gripper_logits", author=args.author,
                      issue="project-task", conditions=payload["conditions"],
                      result={"results": results,
                              "interpretation_limit": payload["interpretation_limit"]})
        print(f"EXP_LOG 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    print("판정은 사람이 한다. docs/PREREG_gripper_logits_0911.md 를 열고 대조하라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
