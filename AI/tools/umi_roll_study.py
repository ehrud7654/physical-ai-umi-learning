"""Can the discarded roll be recovered? Three strategies, measured against each other.
버린 roll 을 되찾을 수 있는가. 세 전략을 서로 대조해 잰다.

project-task · MEASURE_umi_roundtrip_0907 M4 의 후속.

M4 🟢: FK→IK 왕복에서 roll 잔차 중앙값이 47도였다. 그 입력의 6자유도 pose 는
유효한 관절 배치에서 나왔으므로 **전부 달성 가능**했다. 즉 47도는 5자유도의
원리적 한계가 아니라 되돌릴 수 있었던 정보를 버린 것이다.

## 왜 `wrist_roll` 고정이 자명한 답이 아닌가

팔 관절 5개, 구속 5개(위치 3 + 접근축 2)면 해는 **이산집합**이다. roll 은
"자유"가 아니라 **분기가 정해지면 함께 정해진다.** 그래서 레버는 두 개다.

  B) `wrist_roll` 을 목표값으로 고정 → 자유관절 4개에 구속 5개 = **과결정.**
     위치 오차가 밀린다. 위치에는 실측 예산(5mm)이 걸려 있다
  C) `q_init` 의 `wrist_roll` 을 스윕해 **분기를 고르고** 자유 풀이 →
     여전히 정확결정이므로 위치 오차를 내주지 않는다

    # [로컬]
    cd AI && python tools/umi_roll_study.py --episodes 2 --steps 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from paths import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from sim.mujoco.build_scene import build_model, load_config  # noqa: E402
from sim.mujoco.kinematics import IKResult, solve_pose_ik  # noqa: E402
from umi.ik import (  # noqa: E402
    IKSolution,
    approach_axis_from_quat,
    jaw_axis_from_quat,
    roll_residual_deg,
    summarize,
)
from tools.umi_mujoco import (  # noqa: E402
    GRIPPER_BODY,
    MujocoIK,
    N_JOINTS,
    eef_pose_from_joints,
    joint_ranges,
    smooth_joint_trajectory,
)

WRIST_ROLL_IDX = 4


def signed_roll_error_rad(
    quat_desired: np.ndarray, achieved_approach: np.ndarray, achieved_jaw: np.ndarray
) -> float:
    """Signed rotation about the achieved approach axis, folded to (-90, 90] degrees.
    달성된 접근축 둘레의 부호 있는 회전. (-90, 90] 도로 접는다.

    `umi.ik.roll_residual_deg` is unsigned because reporting wants a magnitude.
    Correcting needs a direction, and the parallel jaw's 180-degree symmetry means
    a +170 degree error is really -10 degrees -- correcting toward +170 would walk
    away from the answer.
    `umi.ik.roll_residual_deg` 는 크기만 보고하면 되므로 부호가 없다. 보정에는
    방향이 필요하고, 평행 턱의 180도 대칭 때문에 +170도 오차는 실제로 -10도다.
    +170 쪽으로 보정하면 답에서 멀어진다.
    """
    axis = np.asarray(achieved_approach, dtype=float)
    axis /= np.linalg.norm(axis)

    def proj(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        p = v - np.dot(v, axis) * axis
        n = float(np.linalg.norm(p))
        return p / n if n > 1e-9 else np.zeros(3)

    want = proj(jaw_axis_from_quat(quat_desired))
    got = proj(achieved_jaw)
    if not np.any(want) or not np.any(got):
        return 0.0
    ang = float(np.arctan2(float(np.dot(np.cross(got, want), axis)), float(np.dot(got, want))))
    # 180도 대칭 접기: (-pi/2, pi/2] 로
    while ang > np.pi / 2:
        ang -= np.pi
    while ang <= -np.pi / 2:
        ang += np.pi
    return ang


class RollStudy:
    def __init__(self, model, cfg, *, pos_tol_m: float = 1e-5, axis_tol_deg: float = 0.05):
        self.model = model
        self.cfg = cfg
        self.data = mujoco.MjData(model)
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
        self.pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
        self.ranges = joint_ranges(cfg)
        self.pos_tol_m = pos_tol_m
        self.axis_tol_deg = axis_tol_deg

    def _achieved(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:N_JOINTS] = q
        mujoco.mj_forward(self.model, self.data)
        rot = self.data.xmat[self.bid].reshape(3, 3)
        return -rot[:, 2], rot[:, 0]

    def _wrap(self, res: IKResult, quat: np.ndarray) -> IKSolution:
        q = np.asarray(res.qpos, dtype=float).copy()
        appr, jaw = self._achieved(q)
        return IKSolution(
            q_rad=q,
            pos_error_m=float(res.pos_error_m),
            axis_error_deg=float(res.axis_error_deg),
            roll_residual_deg=float(roll_residual_deg(quat, appr, jaw)),
            within_limits=bool(res.within_limits),
            converged=bool(
                res.pos_error_m < IKResult.POS_TOL_M
                and res.axis_error_deg < IKResult.AXIS_TOL_DEG
            ),
        )

    def _solve(self, pos, quat, q_init, wrist_roll=None) -> IKSolution:
        res = solve_pose_ik(
            self.model,
            target_xyz=np.asarray(pos, dtype=float),
            offset_local=self.pinch,
            desired_axis=approach_axis_from_quat(quat),
            q_init=None if q_init is None else np.asarray(q_init, dtype=float),
            wrist_roll=wrist_roll,
            pos_tol=self.pos_tol_m,
            axis_tol_deg=self.axis_tol_deg,
        )
        return self._wrap(res, quat)

    # A — 현행: 자유 풀이, 직전 해에서 이어간다
    def solve_free(self, pos, quat, q_init) -> IKSolution:
        return self._solve(pos, quat, q_init)

    # B — wrist_roll 고정. 과결정
    def solve_fixed_roll(self, pos, quat, q_init, *, iters: int = 2) -> IKSolution:
        sol = self._solve(pos, quat, q_init)
        best = sol
        wr = float(sol.q_rad[WRIST_ROLL_IDX])
        lo, hi = self.ranges[WRIST_ROLL_IDX]
        for _ in range(iters):
            appr, jaw = self._achieved(best.q_rad)
            err = signed_roll_error_rad(quat, appr, jaw)
            if abs(np.degrees(err)) < 0.5:
                break
            for sign in (+1.0, -1.0):
                cand_wr = float(np.clip(wr + sign * err, lo, hi))
                cand = self._solve(pos, quat, best.q_rad, wrist_roll=cand_wr)
                if cand.roll_residual_deg < best.roll_residual_deg:
                    best, wr = cand, cand_wr
        return best

    # C — q_init 의 wrist_roll 을 스윕해 분기를 고른다. 정확결정 유지
    def solve_branch_seeded(self, pos, quat, q_init, *, n_seeds: int = 9) -> IKSolution:
        lo, hi = self.ranges[WRIST_ROLL_IDX]
        seeds = np.linspace(lo, hi, n_seeds)
        base = np.asarray(q_init, dtype=float).copy() if q_init is not None else np.zeros(N_JOINTS)
        best: IKSolution | None = None
        for wr in seeds:
            seed = base.copy()
            seed[WRIST_ROLL_IDX] = wr
            cand = self._solve(pos, quat, seed)
            if cand.pos_error_m > IKResult.POS_TOL_M or not cand.within_limits:
                continue
            if best is None or cand.roll_residual_deg < best.roll_residual_deg:
                best = cand
        return best if best is not None else self._solve(pos, quat, q_init)


def rotate_about_approach(quat_wxyz: np.ndarray, delta_deg: float) -> np.ndarray:
    """Rotate a pose about its own approach axis, leaving position and axis untouched.
    pose 를 자기 접근축 둘레로만 회전시킨다. 위치와 접근축은 그대로다.

    This is the perturbation that separates the strategies. Position and approach
    direction stay exactly achievable; only the roll becomes something the 5-DOF
    arm may not be able to produce at that position and axis. A human wrist rolls
    freely, so this is the shape of the real mismatch -- not a random pose.
    전략을 구분하는 섭동이 이것이다. 위치와 접근방향은 정확히 달성 가능한 채로
    남고, roll 만 그 위치·축에서 5자유도 팔이 못 만들 수도 있는 값이 된다.
    사람 손목은 자유롭게 돌아가므로 실제 불일치의 형태가 이것이다 — 무작위 pose 가 아니다.
    """
    from umi.ik import matrix_to_quat, quat_to_matrix

    rot = quat_to_matrix(quat_wxyz)
    axis = rot[:, 2]
    th = np.radians(float(delta_deg))
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rodrigues = np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)
    return matrix_to_quat(rodrigues @ rot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-jitter-deg", type=float, nargs="*", default=[0.0, 15.0, 30.0, 60.0],
                    help="목표 pose 를 접근축 둘레로 이만큼 돌린다. 0 은 달성 가능한 입력")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=9, help="C안 wrist_roll 시드 개수")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    MujocoIK(model, cfg)  # 관절 한계·정규화 대조를 여기서도 통과시킨다
    st = RollStudy(model, cfg)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)

    strategies = ("A_free", "B_fixed_roll", "C_branch_seed")
    n = args.episodes * args.steps
    print(f"n={n} per 조건 ({args.episodes}편 x {args.steps}스텝) · C 시드 {args.seeds}개 · "
          f"솔버 정지 pos {st.pos_tol_m:.0e}m axis {st.axis_tol_deg}도")
    print("δ = 목표 pose 를 접근축 둘레로 돌린 각도. δ=0 은 달성 가능한 입력.")
    print("q복원 = 팔 5축이 참 관절각과 얼마나 다른가 (달성 가능 입력에서만 의미)\n")

    head = (f"{'δ':>5} {'안':<15}{'위치중앙':>11}{'위치p95':>10}{'위치최대':>10}"
            f"{'roll중앙':>10}{'roll p95':>10}{'5mm초과':>9}{'q복원최대':>11}{'초/스텝':>9}")
    print(head)
    print("-" * len(head))

    table: dict[float, dict[str, tuple]] = {}
    for delta in args.roll_jitter_deg:
        out = {k: [] for k in strategies}
        qerr = {k: 0.0 for k in strategies}
        timing = {k: 0.0 for k in strategies}
        rng_d = np.random.default_rng(args.seed)
        for _ in range(args.episodes):
            q_true = smooth_joint_trajectory(st.ranges, args.steps, rng_d)
            poses = [eef_pose_from_joints(model, data, q, st.pinch) for q in q_true]
            prev = {k: None for k in strategies}
            for (pos, quat0), q_ref in zip(poses, q_true):
                quat = rotate_about_approach(quat0, delta) if delta else quat0
                for name, fn in (
                    ("A_free", st.solve_free),
                    ("B_fixed_roll", st.solve_fixed_roll),
                    ("C_branch_seed",
                     lambda p, q, qi: st.solve_branch_seeded(p, q, qi, n_seeds=args.seeds)),
                ):
                    t0 = time.perf_counter()
                    sol = fn(pos, quat, prev[name])
                    timing[name] += time.perf_counter() - t0
                    out[name].append(sol)
                    prev[name] = sol.q_rad
                    qerr[name] = max(qerr[name],
                                     float(np.abs(sol.q_rad[:5] - q_ref[:5]).max()))
        table[delta] = {}
        for name in strategies:
            s = summarize(out[name])
            table[delta][name] = (s, qerr[name], timing[name] / n)
            print(f"{delta:5.0f} {name:<15}{s.pos_error_median_mm:10.4f}mm"
                  f"{s.pos_error_p95_mm:10.4f}{s.pos_error_max_mm:10.4f}"
                  f"{s.roll_residual_median_deg:9.2f}도{s.roll_residual_p95_deg:10.2f}"
                  f"{s.n_over_5mm:8d}{np.degrees(qerr[name]):10.1f}도{timing[name] / n:8.4f}s")
        print()

    def verdict(ok: bool) -> str:
        return "맞음" if ok else "틀림"

    print("=== 예측 대조 ===")
    d0 = table.get(0.0)
    if d0:
        a, b, c = (d0[k][0] for k in strategies)
        print(f"δ=0  A roll 중앙 47도 근처       실측 {a.roll_residual_median_deg:.2f}도  "
              f"{verdict(abs(a.roll_residual_median_deg - 47.0) < 10.0)}")
        print(f"δ=0  B roll <5도                실측 {b.roll_residual_median_deg:.2f}도  "
              f"{verdict(b.roll_residual_median_deg < 5.0)}")
        print(f"δ=0  B 위치 악화 (>=0.05mm)     실측 {b.pos_error_median_mm:.4f}mm  "
              f"{verdict(b.pos_error_median_mm >= 0.05)}  ← 달성 가능 입력에선 무모순이라 안 밀린다")
        print(f"δ=0  C 위치 0.01mm대 유지       실측 {c.pos_error_median_mm:.4f}mm  "
              f"{verdict(c.pos_error_median_mm < 0.05)}")
    big = max((d for d in table if d > 0), default=None)
    if big is not None:
        b_big, c_big = table[big]["B_fixed_roll"][0], table[big]["C_branch_seed"][0]
        print(f"δ={big:.0f} B 위치가 무너진다        실측 {b_big.pos_error_median_mm:.4f}mm  "
              f"{verdict(b_big.pos_error_median_mm > 0.05)}")
        print(f"δ={big:.0f} C 위치 0.0088mm 유지     실측 {c_big.pos_error_median_mm:.4f}mm  "
              f"{verdict(c_big.pos_error_median_mm < 0.05)}")

    print("\n⚠️ 입력은 유효한 관절 배치에서 FK 로 만든 것이고, δ 로 roll 만 달성 불가로 밀었다.")
    print("   실제 사람 손 궤적은 위치·접근축까지 다양체 밖일 수 있다. 그건 project-task 이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
