"""Is the mapping the policy is asked to learn well-posed at the grasp boundary?
정책에게 학습시키는 사상이 파지 경계에서 애초에 잘 정의돼 있는가?

No training, no rollouts, no GPU. Reads the demonstrations and asks three
questions that a loss value cannot answer.
학습도 롤아웃도 GPU 도 필요 없다. 시연을 읽고, 손실값으로는 답할 수 없는 세 가지를 묻는다.

1. **Zero-delta census.** With `joint_delta_gripper_abs` the arm target is
   `action - state`. While the gripper closes, the arm holds -- so the arm target
   is ~0 for that whole stretch. Under L1 the optimal prediction on a set of
   targets is their **median**, so a run of ~0 targets pulls the prediction to 0
   for every observation that looks like those frames. This is the exact mechanism
   `dwell_s: 0.5` demonstrated: adding more ~0 frames took the policy from 9.3%
   to 0/300 🟢. dwell was removed, but the close phase still produces them.
   `joint_delta_gripper_abs` 에서 팔 목표는 `action - state` 다. 그리퍼가 닫히는
   동안 팔은 정지하므로 그 구간 내내 팔 목표가 ~0 이다. L1 에서 목표 집합의 최적
   예측은 **중앙값**이라, ~0 목표가 연속되면 그 프레임처럼 보이는 모든 관측에서
   예측이 0 으로 끌린다. `dwell_s: 0.5` 가 정확히 이 메커니즘을 실증했다 —
   ~0 프레임을 더 넣자 9.3% → 0/300 🟢. dwell 은 뺐지만 닫기 구간은 여전히 만든다.

2. **Boundary jump.** At close-end -> lift-start the label jumps from ~0 to a
   large delta while the observation barely changes. If the observation does not
   change and the label does, no single-frame model can learn the transition --
   that is not a capacity problem, it is an ill-posed mapping.
   닫기 종료 → 상승 시작에서 라벨은 ~0 에서 큰 델타로 뛰는데 관측은 거의 안 변한다.
   관측이 안 변하고 라벨만 변하면 단일 프레임 모델은 그 전환을 배울 수 없다.
   용량 문제가 아니라 **사상이 잘 정의되지 않은** 것이다.

3. **Aliasing index.** For every sample, the nearest other sample in state space,
   and how far apart their labels are. Same-episode neighbours are the strong
   evidence: same object position means near-identical images too, so a small
   state distance with a large label distance is genuine aliasing and not
   something the images could resolve.
   각 샘플의 state 공간 최근접 이웃과 그 라벨 거리. **같은 에피소드** 이웃이 강한
   증거다 — 물체 위치가 같으니 이미지도 거의 같고, state 거리가 작은데 라벨 거리가
   크면 이미지로 해소될 수 없는 진짜 앨리어싱이다.

    python tools/audit_demos.py datasets/sim_pick_v5 --log

This is a measurement, not a gate. It does not pass or fail anything -- it says
which of several explanations the data is consistent with, before a GPU is used.
게이트가 아니라 계측이다. 무엇도 통과/실패시키지 않는다. GPU 를 쓰기 전에,
데이터가 어느 설명과 부합하는지를 말한다.
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

runtime_limits.claim("audit_demos")

from tracking.exp_log import log_run  # noqa: E402

ARM = slice(0, 5)
GRIP = 5


def load(root: Path) -> list[dict[str, Any]]:
    """Every episode's state/action tracks plus its metadata.
    에피소드별 state/action 트랙과 메타."""
    out = []
    for npz in sorted(root.glob("ep_*.npz")):
        with np.load(npz) as z:
            state = np.asarray(z["state"], dtype=np.float64)
            action = np.asarray(z["action"], dtype=np.float64)
        meta_path = npz.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        out.append({"name": npz.stem, "state": state, "action": action, "meta": meta})
    return out


def target(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """`joint_delta_gripper_abs`, the tensor the loss is computed against.
    손실이 계산되는 대상 텐서."""
    t = action - state
    t[..., GRIP] = action[..., GRIP]
    return t


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Length and start index of the longest True run.
    가장 긴 True 구간의 길이와 시작 인덱스."""
    best_len = best_start = cur_len = 0
    cur_start = 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_len, best_start


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--zero-frac", type=float, default=0.10,
                   help="팔 델타 크기가 그 에피소드 최대치의 이 비율 미만이면 '~0' 으로 본다")
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    eps = load(args.data)
    if not eps:
        print(f"에피소드가 없다: {args.data}")
        return 1
    print(f"{args.data.name}: 에피소드 {len(eps)}개 · "
          f"샘플 {sum(len(e['state']) for e in eps)}개\n")

    # --- 1. zero-delta census -------------------------------------------------
    zero_fracs, run_lens, run_starts, ep_lens = [], [], [], []
    close_zero_fracs = []
    for e in eps:
        tg = target(e["state"], e["action"])
        mag = np.linalg.norm(tg[:, ARM], axis=1)
        thr = args.zero_frac * mag.max() if mag.max() > 0 else 0.0
        near0 = mag <= thr
        zero_fracs.append(float(near0.mean()))
        L, s = longest_run(near0)
        run_lens.append(L); run_starts.append(s); ep_lens.append(len(mag))
        # 그리퍼가 닫히는 쪽으로 움직이는 구간
        g = e["action"][:, GRIP]
        closing = np.zeros(len(g), dtype=bool)
        if g.max() - g.min() > 1e-6:
            mid = (g.max() + g.min()) / 2.0
            closing = g <= mid
        close_zero_fracs.append(float(near0[closing].mean()) if closing.any() else float("nan"))

    print("1. 팔 델타 ~0 프레임 인구조사  (~0 기준: 에피소드 최대 크기의 "
          f"{args.zero_frac:.0%} 미만)")
    print(f"   전체 프레임 중 ~0 비율      중앙값 {np.median(zero_fracs) * 100:5.1f}%  "
          f"(최소 {min(zero_fracs) * 100:.1f} / 최대 {max(zero_fracs) * 100:.1f})")
    czf = [v for v in close_zero_fracs if v == v]
    if czf:
        print(f"   그리퍼 닫힘 구간 안 ~0 비율 중앙값 {np.median(czf) * 100:5.1f}%")
    print(f"   가장 긴 연속 ~0 구간        중앙값 {np.median(run_lens):5.1f}틱 "
          f"= {np.median(run_lens) / 30:.2f}초  (에피소드 길이 중앙 {np.median(ep_lens):.0f}틱)")
    print(f"   그 구간 시작 위치           중앙값 {np.median(run_starts):5.1f}틱 "
          f"(에피소드의 {np.median(np.array(run_starts) / np.array(ep_lens)) * 100:.0f}% 지점)")
    print("   ⚠️ v4(dwell 0.5s) 가 0/300 을 낸 메커니즘이 이것이다. dwell 을 뺀 v5 에도 "
          "이만큼 남아 있다면, 정책이 그 구간에서 멈추도록 배우는 것은 버그가 아니라 "
          "L1 하에서 **정답**이다\n")

    # --- 2. boundary jump -----------------------------------------------------
    jumps = []
    for e in eps:
        tg = target(e["state"], e["action"])
        mag = np.linalg.norm(tg[:, ARM], axis=1)
        thr = args.zero_frac * mag.max() if mag.max() > 0 else 0.0
        near0 = mag <= thr
        L, s = longest_run(near0)
        end = s + L
        if L == 0 or end >= len(mag) - 1:
            continue
        # ~0 구간 마지막 프레임 -> 그 다음 프레임
        d_obs = float(np.linalg.norm(e["state"][end] - e["state"][end - 1]))
        d_lab = float(np.linalg.norm(tg[end, ARM] - tg[end - 1, ARM]))
        jumps.append({"obs": d_obs, "label": d_lab,
                      "ratio": d_lab / d_obs if d_obs > 1e-12 else float("inf")})
    if len(jumps) < max(5, len(eps) // 10):
        print(f"2. ~0 구간 종료 경계 — **해석 불가.** ~0 구간이 있는 에피소드가 "
              f"{len(jumps)}/{len(eps)} 뿐이다. 아래 수치를 읽지 마라\n")
    if jumps:
        print("2. ~0 구간 종료 경계 — 관측은 얼마나 변하고 라벨은 얼마나 뛰는가")
        print(f"   관측 변화 |Δstate|   중앙값 {np.median([j['obs'] for j in jumps]):.6f}")
        print(f"   라벨 변화 |Δtarget|  중앙값 {np.median([j['label'] for j in jumps]):.6f}")
        r = [j["ratio"] for j in jumps if j["ratio"] != float("inf")]
        if r:
            print(f"   비율 라벨/관측       중앙값 {np.median(r):8.1f}")
        print("   ⚠️ 비율이 크면 관측이 거의 그대로인데 라벨만 뛴다는 뜻이다. "
              "단일 프레임 모델의 용량 문제가 아니라 사상이 잘 정의되지 않은 것이고, "
              "처방은 데이터량이나 DAgger 가 아니라 **관측에 위상/이력을 넣거나 청킹**이다\n")

    # --- 3. aliasing index ----------------------------------------------------
    S = np.concatenate([e["state"] for e in eps])
    T = np.concatenate([target(e["state"], e["action"]) for e in eps])
    ep_id = np.concatenate([np.full(len(e["state"]), i) for i, e in enumerate(eps)])
    t_idx = np.concatenate([np.arange(len(e["state"])) for e in eps])
    scale = float(np.linalg.norm(T[:, ARM], axis=1).max())

    rng = np.random.default_rng(0)
    sample = rng.choice(len(S), size=min(3000, len(S)), replace=False)
    same_ep, cross_ep = [], []
    for i in sample:
        d = np.linalg.norm(S - S[i], axis=1)
        # 자기 자신과 시간상 바로 옆(±5틱)은 이웃으로 세지 않는다 — 연속 프레임은
        # 당연히 가깝고, 그것을 앨리어싱이라 부르면 아무 의미가 없다.
        near_time = (ep_id == ep_id[i]) & (np.abs(t_idx - t_idx[i]) <= 5)
        d[near_time] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            continue
        lab = float(np.linalg.norm(T[j, ARM] - T[i, ARM])) / scale
        rec = {"d_state": float(d[j]), "d_label_rel": lab}
        (same_ep if ep_id[j] == ep_id[i] else cross_ep).append(rec)

    print("3. 앨리어싱 지표 — state 최근접 이웃의 라벨 거리 (라벨은 최대 크기로 정규화)")
    for label, rows in (("같은 에피소드 (이미지도 거의 같다)", same_ep),
                        ("다른 에피소드 (이미지는 다를 수 있다)", cross_ep)):
        if not rows:
            print(f"   {label}: 없음")
            continue
        ds = np.array([r["d_state"] for r in rows])
        dl = np.array([r["d_label_rel"] for r in rows])
        print(f"   {label}  n={len(rows)}")
        print(f"     state 거리 중앙 {np.median(ds):.5f} · 라벨 거리 중앙 "
              f"{np.median(dl) * 100:5.1f}% · 라벨거리>30% 인 비율 {np.mean(dl > 0.30) * 100:5.1f}%")
    if same_ep:
        n_bad = sum(1 for r in same_ep if r["d_label_rel"] > 0.30)
        print(f"   → 전체 표본 대비: {n_bad}/{len(sample)} = {n_bad / len(sample) * 100:.1f}%")
        print("     (같은에피소드 쌍 안의 비율만 보면 과장된다. 두 분모를 함께 읽는다)")
    print("   ⚠️ 같은 에피소드 쪽에서 state 거리가 작은데 라벨 거리가 크면, "
          "이미지로도 해소되지 않는 앨리어싱이다\n")

    # --- 4. 관측 정지 창 -------------------------------------------------------
    # 앨리어싱의 실체는 "팔 델타 0" 이 아니다. state 6차원에 그리퍼 위치가 들어 있어서
    # 닫는 동안 그리퍼 state 가 단조 변하는 구간은 관측상 구분된다. 진짜 모호한 곳은
    # **그리퍼가 물체에 물려 포화되고 팔도 멈춘** 창이다. 거기서는 관측이 정지하고
    # 라벨만 바뀌므로 단일 프레임 모델이 "언제 들어올릴지"를 알 수 없다.
    # dwell(v4) 이 주입한 15틱이 정확히 이 창의 프레임이었다 — v4 가 0/300 을 낸 것이
    # 단순 라벨 불균형이 아니라 이 창의 확대였을 가능성이 여기서 갈린다.
    print("4. 관측 정지 창 — 그리퍼 state 포화 AND 팔 정지 (단일 프레임으로 구분 불가한 구간)")
    win_lens, win_starts, win_label_mags, tail_after = [], [], [], []
    for e in eps:
        st = e["state"]
        tg = target(st, e["action"])
        d_arm = np.linalg.norm(np.diff(st[:, ARM], axis=0), axis=1)
        d_grip = np.abs(np.diff(st[:, GRIP]))
        ta = 0.05 * d_arm.max() if d_arm.max() > 0 else 0.0
        tgp = 0.05 * d_grip.max() if d_grip.max() > 0 else 0.0
        static = (d_arm <= ta) & (d_grip <= tgp)
        L, s0 = longest_run(static)
        win_lens.append(L); win_starts.append(s0)
        if L > 0:
            win_label_mags.append(float(np.median(np.linalg.norm(tg[s0:s0 + L, ARM], axis=1))))
            tail_after.append(len(st) - (s0 + L))
    print(f"   가장 긴 정지 창   중앙값 {np.median(win_lens):5.1f}틱 = {np.median(win_lens) / 30:.2f}초 "
          f"(Q1 {np.percentile(win_lens, 25):.0f} / Q3 {np.percentile(win_lens, 75):.0f})")
    print(f"   시작 위치         중앙값 {np.median(win_starts):5.1f}틱 "
          f"(에피소드 길이 중앙 {np.median(ep_lens):.0f}틱)")
    if win_label_mags:
        print(f"   창 안 라벨 크기   중앙값 {np.median(win_label_mags):.6f} "
              f"(에피소드 최대 대비 {np.median(win_label_mags) / scale * 100:.1f}%)")
        print(f"   창 이후 남은 틱   중앙값 {np.median(tail_after):5.1f}틱")
    print("   ⚠️ 판정 규칙 (결과 보기 전 확정): 창 중앙값 >= 10틱 이면 zero-delta/앨리어싱 "
          "계열이 살아 있다 → 재표집 ablation 착수. < 3틱 이면 이 계열을 내리고 "
          "접근 발산 계열을 우선한다\n")

    if args.log:
        rec = log_run(
            experiment="audit_demos",
            author=args.author,
            issue="project-task",
            conditions={"dataset": str(args.data), "n_episodes": len(eps),
                        "n_samples": int(len(S)), "zero_frac_threshold": args.zero_frac,
                        "action_space": "joint_delta_gripper_abs"},
            result={
                "zero_frac_median": float(np.median(zero_fracs)),
                "close_phase_zero_frac_median": float(np.median(czf)) if czf else None,
                "longest_zero_run_ticks_median": float(np.median(run_lens)),
                "longest_zero_run_start_median": float(np.median(run_starts)),
                "episode_len_median": float(np.median(ep_lens)),
                "boundary_obs_delta_median": float(np.median([j["obs"] for j in jumps])) if jumps else None,
                "boundary_label_delta_median": float(np.median([j["label"] for j in jumps])) if jumps else None,
                "alias_same_ep_label_dist_median": float(np.median([r["d_label_rel"] for r in same_ep])) if same_ep else None,
                "alias_same_ep_frac_over_30pct": float(np.mean([r["d_label_rel"] > 0.30 for r in same_ep])) if same_ep else None,
                "static_window_ticks_median": float(np.median(win_lens)),
                "static_window_ticks_q1": float(np.percentile(win_lens, 25)),
                "static_window_ticks_q3": float(np.percentile(win_lens, 75)),
                "static_window_start_median": float(np.median(win_starts)),
                "static_window_label_mag_median": float(np.median(win_label_mags)) if win_label_mags else None,
                "alias_same_ep_frac_of_all_samples": (
                    float(sum(1 for r in same_ep if r["d_label_rel"] > 0.30) / len(sample))
                    if same_ep else None),
                "alias_cross_ep_label_dist_median": float(np.median([r["d_label_rel"] for r in cross_ep])) if cross_ep else None,
            },
        )
        print(f"EXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
