"""두 계약 데이터셋을 배열 수준에서 대조한다. 학습 없이 변환 정확성을 판정한다.
Compare two contract datasets array-by-array — conversion accuracy without training.

왜 롤아웃으로 판정하지 않는가: 현행 baseline 은 v5 9.3%(CI 6.5~13.2%)이고 학습 실행 간
변동이 25%p 다 🟢. 0% 는 "변환이 깨졌다"와 "정책이 원래 나쁘다" 둘 다에서 나온다.
**배열이 같으면 학습 결과가 같을 수밖에 없다** — 그게 더 강하고 더 싸다.

    # [로컬]
    cd AI && python tools/diff_datasets.py datasets/umi_src_sim_v1 datasets/umi_pick_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from contract.episode import read_episode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--state-tol", type=float, default=1e-3,
                    help="정규화 단위. IK+정규화 왕복 오차의 허용치")
    args = ap.parse_args()

    lmap = {p.stem: p for p in sorted(args.left.glob("*.npz"))}
    rmap = {p.stem: p for p in sorted(args.right.glob("*.npz"))}
    common = sorted(set(lmap) & set(rmap))
    if not common:
        print(f"공통 에피소드가 없다. left {len(lmap)}편 · right {len(rmap)}편")
        return 1
    only_l, only_r = sorted(set(lmap) - set(rmap)), sorted(set(rmap) - set(lmap))

    print(f"{args.left}  vs  {args.right}")
    print(f"공통 {len(common)}편" + (f" · 왼쪽만 {len(only_l)}" if only_l else "")
          + (f" · 오른쪽만 {len(only_r)}" if only_r else "") + "\n")

    acc = {k: [] for k in ("state", "action", "state_ts", "action_ts")}
    per_joint = {"state": [], "action": []}   # 관절별 최대차
    motion = []                               # 왼쪽의 관절별 mean|action-state|
    signal = []                               # 왼쪽의 RMS(action-state)
    signal_r = []                             # 오른쪽의 RMS. 판정 대상은 이쪽이다
    img_same = img_total = 0
    step_mismatch = []

    for name in common:
        a, b = read_episode(lmap[name]), read_episode(rmap[name])
        n = min(a.meta.n_steps, b.meta.n_steps)
        if a.meta.n_steps != b.meta.n_steps:
            step_mismatch.append((name, a.meta.n_steps, b.meta.n_steps))
        acc["state"].append(float(np.abs(a.state[:n] - b.state[:n]).max()))
        acc["action"].append(float(np.abs(a.action[:n] - b.action[:n]).max()))
        acc["state_ts"].append(float(np.abs(a.state_timestamp[:n] - b.state_timestamp[:n]).max()))
        acc["action_ts"].append(float(np.abs(a.action_timestamp[:n] - b.action_timestamp[:n]).max()))
        per_joint["state"].append(np.abs(a.state[:n] - b.state[:n]).max(axis=0))
        per_joint["action"].append(np.abs(a.action[:n] - b.action[:n]).max(axis=0))
        motion.append(np.abs(a.action[:n] - a.state[:n]).mean(axis=0))
        signal.append(float(np.sqrt(((a.action[:n] - a.state[:n]) ** 2).mean())))
        signal_r.append(float(np.sqrt(((b.action[:n] - b.state[:n]) ** 2).mean())))
        for cam in sorted(set(a.images) & set(b.images)):
            img_total += 1
            if np.array_equal(a.images[cam][:n], b.images[cam][:n]):
                img_same += 1

    print(f"{'필드':<12}{'최대차 중앙':>14}{'최대차 최대':>14}")
    print("-" * 40)
    for k in ("state", "action"):
        v = np.array(acc[k])
        print(f"{k:<12}{np.median(v):14.3e}{v.max():14.3e}")
    for k in ("state_ts", "action_ts"):
        v = np.array(acc[k])
        print(f"{k:<12}{np.median(v):14.3e}{v.max():14.3e}  [초]")

    print(f"\n이미지 바이트 완전 일치 {img_same}/{img_total}")
    if step_mismatch:
        print(f"스텝 수 불일치 {len(step_mismatch)}편: {step_mismatch[:5]}")

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    js = np.stack(per_joint["state"]).max(axis=0)
    ja = np.stack(per_joint["action"]).max(axis=0)
    mo = np.stack(motion).mean(axis=0)
    print(f"\n{'관절':<15}{'state 최대차':>14}{'action 최대차':>15}{'움직임 평균':>14}{'오차/움직임':>13}")
    print("-" * 74)
    for i, nm in enumerate(names):
        ratio = js[i] / mo[i] if mo[i] > 0 else float("nan")
        # 분모가 왕복 오차와 같은 자릿수면 비율은 아무것도 판정하지 못한다.
        floor_hit = mo[i] < 10.0 * js[i]
        mark = "  ← 분모 과소, 비교 불가" if floor_hit else ""
        print(f"{nm:<15}{js[i]:14.3e}{ja[i]:15.3e}{mo[i]:14.3e}{ratio * 100:12.2f}%{mark}")

    st = np.array(acc["state"]).max()
    at = np.array(acc["action"]).max()
    print("\n=== 판정 ===")
    print(f"state 왕복 절대 최대차 {st:.3e} (정규화 단위)")
    print(f"action 왕복 절대 최대차 {at:.3e}")
    rms_l, rms_r = float(np.mean(signal)), float(np.mean(signal_r))
    rms = min(rms_l, rms_r)
    print(f"\n=== 주 판정: 학습 신호 RMS 대비 ===")
    print(f"RMS(action-state)  왼쪽 {rms_l:.4e} · 오른쪽 {rms_r:.4e}")
    # ⚠️ 2026-09-08 정정. 왼쪽 기준으로만 나누면 낙관적이 된다 — 왼쪽이 시뮬 수집이면
    #    action=ctrl(추종오차 포함)이라 신호가 크게 잡힌다. 실측: 왼쪽 4.6987e-02 vs
    #    오른쪽(q[t+1] 기준) 1.602e-02 로 3배 차이. **작은 쪽으로 나눈다.**
    if rms_l / max(rms_r, 1e-12) > 1.5 or rms_r / max(rms_l, 1e-12) > 1.5:
        print("  ⚠️ 양쪽 신호 크기가 1.5배 이상 다르다 — action 의미가 다른 두 출처다")
        print("     (시뮬 수집 action=ctrl 추종오차 포함 / UMI 경로 action=q[t+1])")
    print(f"state 왕복 최대차 / RMS(작은 쪽 {rms:.4e}) = {st / rms * 100:.2f}%")
    print("\n관절별 '오차/움직임' 은 보조 지표다. 관절 평균 움직임이 왕복 오차와")
    print("같은 자릿수면(위 표의 '분모 과소') 비율이 무의미해진다 — 실측 🟢 2026-09-08:")
    print("wrist_roll 이 57.72% 로 나왔는데 절대 오차는 0.0035도였다. 아무것도 아닌 것의 58%다.")
    print("절대 허용치도 비율도 단독으로는 판정하지 못한다. RMS 대비를 주 지표로 쓴다.")
    ok = [i for i in range(6) if mo[i] >= 10.0 * js[i]]
    if ok:
        worst = max(js[i] / mo[i] for i in ok)
        print(f"분모가 충분한 관절 중 최악 비율 {worst * 100:.2f}%")
    print("  ⚠️ action 은 0 이 아닌 것이 정상일 수 있다 — 시뮬 수집의 action 은 ctrl(명령)이고")
    print("     UMI 경로는 q[t+1] 이다. 두 출처의 구조적 차이다 (umi_dump_from_dataset 참조).")
    print("     원본이 UMI 경로로 만들어진 데이터셋이면 0 에 가까워야 한다.")
    return 0 if st <= args.state_tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
