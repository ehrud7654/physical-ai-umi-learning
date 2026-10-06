"""다양체 밖으로 밀었을 때 무엇이 먼저 무너지는가 — 수용률의 실체.
What gives way first when the target leaves the reachable manifold.

project-task 후속 · project-task(UMI 실행가능성 검사기 + 수용률) 방향.

지금까지의 관통 계측은 입력이 **전부 도달 가능**했다 (유효한 관절 배치의 FK).
사람 손 pose 는 그렇지 않다. 실측 🟢 2026-09-07 로 알게 된 자유도 배분:

    pan lift elbow_flex wrist_flex (4개)  →  위치 3 + 접근축 2 = 5구속  (과결정)
    wrist_roll (1개)                      →  roll 전담, 완전 독립

roll 은 늘 맞춰진다. **무너지는 것은 (위치, 접근축) 조합**이고, 4관절로 5구속을
만족시킬 수 없을 때 감쇠 최소자승이 무엇을 내주는지가 수용률을 정한다.

`solve_pose_ik` 가 최소화하는 것은 `[e_pos ; axis_weight * e_axis]` 이고
`axis_weight=0.15` 다. 위치 가중이 축의 6.7배다.

    # [로컬]
    cd AI && python tools/umi_reach_study.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from paths import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from sim.mujoco.build_scene import build_model, load_config  # noqa: E402
from sim.mujoco.kinematics import IKResult  # noqa: E402
from tools.make_umi_synth import gripper_profile  # noqa: E402
from tools.umi_mujoco import (  # noqa: E402
    MujocoIK,
    eef_pose_from_joints,
    gap_from_angle,
    joint_ranges,
    smooth_joint_trajectory,
)
from umi.ik import matrix_to_quat, quat_to_matrix  # noqa: E402

POS_BUDGET_MM = 5.0
AXIS_BUDGET_DEG = 5.0


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def tilt_frame(quat_wxyz: np.ndarray, deg: float, rng: np.random.Generator) -> np.ndarray:
    """Tilt the whole EEF frame by `deg` about a random axis perpendicular to the approach.
    EEF 프레임 전체를 접근축에 수직인 무작위 축 둘레로 `deg` 만큼 기울인다.

    Tilting the frame rather than the approach vector alone keeps the frame
    orthonormal, which is what a wrist actually does.
    접근 벡터만이 아니라 프레임을 기울이면 정규직교가 유지된다. 손목이 실제로
    하는 일이 그것이다.
    """
    if deg == 0.0:
        return np.asarray(quat_wxyz, dtype=float)
    rot = quat_to_matrix(quat_wxyz)
    z = rot[:, 2]
    v = rng.normal(size=3)
    perp = v - np.dot(v, z) * z
    perp /= np.linalg.norm(perp)
    return matrix_to_quat(_rodrigues(perp, np.radians(deg)) @ rot)


def jitter_pos(pos: np.ndarray, mm: float, rng: np.random.Generator) -> np.ndarray:
    if mm == 0.0:
        return np.asarray(pos, dtype=float)
    d = rng.normal(size=3)
    return np.asarray(pos, dtype=float) + (d / np.linalg.norm(d)) * (mm / 1000.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-mm", type=float, nargs="*", default=[0, 2, 5, 10, 20, 50])
    ap.add_argument("--tilt-deg", type=float, nargs="*", default=[0, 2, 5, 10, 20])
    ap.add_argument("--multistart", action="store_true",
                    help="시드 없는 스텝에 다중시작. 다양체 밖 비용 배수를 재려고 노출한다")
    ap.add_argument("--advance-seed-always", action="store_true", default=True)
    ap.add_argument("--advance-seed-on-success-only", dest="advance_seed_always",
                    action="store_false",
                    help="옛 규칙. 다양체 밖에서는 모든 스텝이 시드 없는 스텝이 된다")
    ap.add_argument("--max-iters", type=int, default=120,
                    help="다양체 밖 목표는 수렴하지 않으므로 기본 500 을 전부 돈다")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    # 다중시작을 끈다. 이 실험은 **기하**를 재는 것이고, 다양체 밖에서는 수렴이
    # 없어서 다중시작이 스텝당 8배 비용만 낸다 (실측: 170초 타임아웃).
    ik = MujocoIK(model, cfg, multistart=args.multistart)
    ik.max_iters = args.max_iters
    data = mujoco.MjData(model)
    ranges = joint_ranges(cfg)
    pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    curve = cfg["grasp"]["gap_curve"]

    # 궤적은 조건마다 동일하게 재생성한다. 섭동만 바뀌어야 한다.
    def trajectories():
        rng = np.random.default_rng(args.seed)
        for _ in range(args.episodes):
            q = smooth_joint_trajectory(ranges, args.steps, rng)
            q[:, 5] = gripper_profile(args.steps, ranges[5, 0], ranges[5, 1])
            yield q

    n = args.episodes * args.steps
    print(f"n={n} per 조건 ({args.episodes}편 x {args.steps}스텝) · "
          f"예산 위치 {POS_BUDGET_MM}mm · 축 {AXIS_BUDGET_DEG}도 (IKResult 상수)")
    print(f"다중시작 {args.multistart} · 시드 무조건 이어감 {args.advance_seed_always} · "
          f"max_iters {args.max_iters}")
    print(f"solve_pose_ik axis_weight=0.15 — 위치 가중이 축의 6.7배\n")

    head = (f"{'섭동':<14}{'위치중앙':>10}{'위치p95':>10}{'축중앙':>9}{'축p95':>9}"
            f"{'위치>5mm':>10}{'축>5도':>9}{'한계':>6}{'수용률':>9}")
    print(head)
    print("-" * len(head))

    results = []
    for kind, values in (("pos", args.pos_mm), ("tilt", args.tilt_deg)):
        for v in values:
            rng = np.random.default_rng(1234)  # 섭동 방향은 조건 간 동일
            pos_e, axis_e, lim_bad, accepted = [], [], 0, 0
            for q_true in trajectories():
                poses = [eef_pose_from_joints(model, data, row, pinch) for row in q_true]
                prev = None
                for (pos, quat), qref in zip(poses, q_true):
                    if kind == "pos":
                        tgt_pos, tgt_quat = jitter_pos(pos, v, rng), quat
                    else:
                        tgt_pos, tgt_quat = pos, tilt_frame(quat, v, rng)
                    sol = ik.solve(tgt_pos, tgt_quat, gap_from_angle(qref[5], curve), q_init=prev)
                    pos_e.append(sol.pos_error_m * 1000.0)
                    axis_e.append(sol.axis_error_deg)
                    if not sol.within_limits:
                        lim_bad += 1
                    ok = (
                        sol.within_limits
                        and sol.pos_error_m * 1000.0 <= POS_BUDGET_MM
                        and sol.axis_error_deg <= AXIS_BUDGET_DEG
                    )
                    accepted += int(ok)
                    # 시드를 **무조건** 이어간다. 다양체 밖에서는 최소자승 해가
                    # "가장 가까운 도달 가능 자세"이므로 다음 스텝의 좋은 시드다.
                    # (파이프라인의 "실패 시 갱신 안 함" 규칙과 다르다 — 그 규칙은
                    #  시드 없는 풀이가 나쁜 분지에 빠지는 경우를 막으려는 것이고,
                    #  여기서는 시드가 항상 있다.)
                    if args.advance_seed_always or (sol.converged and sol.within_limits):
                        prev = sol.q_rad
            pe, ae = np.array(pos_e), np.array(axis_e)
            over_pos = int((pe > POS_BUDGET_MM).sum())
            over_axis = int((ae > AXIS_BUDGET_DEG).sum())
            label = f"{kind} {v:g}{'mm' if kind == 'pos' else '도'}"
            results.append((kind, v, np.median(pe), np.median(ae), over_pos, over_axis, accepted / n))
            print(f"{label:<14}{np.median(pe):10.4f}{np.percentile(pe,95):10.4f}"
                  f"{np.median(ae):9.3f}{np.percentile(ae,95):9.3f}"
                  f"{over_pos:10d}{over_axis:9d}{lim_bad:6d}{accepted / n * 100:8.1f}%")
        print()

    print("=== 예측 대조 ===")
    def verdict(ok): return "맞음" if ok else "틀림"
    pos_rows = [r for r in results if r[0] == "pos" and r[1] > 0]
    if pos_rows:
        worst = max(pos_rows, key=lambda r: r[1])
        print(f"1. 잔여 위치 오차가 지터보다 작다   지터 {worst[1]:g}mm → 위치오차 중앙 "
              f"{worst[2]:.3f}mm  {verdict(worst[2] < worst[1])}")
    first_axis = next((r for r in results if r[0] == "pos" and r[5] > 0), None)
    first_pos = next((r for r in results if r[0] == "pos" and r[4] > 0), None)
    if first_axis and first_pos:
        print(f"2. 축이 위치보다 먼저 무너진다     축>5도 처음 = pos {first_axis[1]:g}mm · "
              f"위치>5mm 처음 = pos {first_pos[1]:g}mm  {verdict(first_axis[1] < first_pos[1])}")
    elif first_axis and not first_pos:
        print(f"2. 축이 위치보다 먼저 무너진다     축>5도 처음 = pos {first_axis[1]:g}mm · "
              f"위치>5mm 는 끝까지 안 나옴  맞음")
    print("\n⚠️ roll 은 이 실험의 대상이 아니다. wrist_roll 이 결합 0 이라 늘 맞춰진다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
