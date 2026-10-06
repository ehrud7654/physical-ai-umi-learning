"""Inverse kinematics for the SO-101, position and approach-axis constrained.
SO-101용 역기구학. 위치와 접근축을 함께 구속한다.

Why this exists: position-only IK is not enough to grasp anything. It leaves
the finger axis wherever the solver happens to land, so the jaws arrive at the
object edge-on and push it instead of straddling it — measured, not assumed.
왜 필요한가: 위치만 맞추는 IK로는 아무것도 못 집는다. 손가락 축이 아무렇게나
정해져서 턱이 물체를 감싸는 대신 옆에서 밀어버린다 — 추정이 아니라 실측 결과다.

DOF budget: the SO-101 has 5 arm joints. Position (3) + approach direction (2)
= 5 constraints, exactly determined. Full 6-DOF pose control is NOT available —
the rotation about the approach axis is whatever wrist_roll is commanded to be,
and cannot be chosen independently of the other five.
자유도 계산: SO-101 팔 관절은 5개다. 위치 3 + 접근방향 2 = 5로 정확히 결정된다.
6자유도 완전 자세 제어는 **불가능**하다. 접근축 둘레 회전은 wrist_roll 명령
그대로일 뿐, 나머지 다섯과 독립적으로 고를 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

N_ARM = 6  # 5 joints + gripper; the gripper is never used for reaching / 그리퍼는 도달에 안 씀
N_REACH = 5  # joints the IK is allowed to move / IK가 움직여도 되는 관절


@dataclass(frozen=True)
class IKResult:
    """Outcome of one IK solve.
    IK 풀이 한 번의 결과."""

    qpos: np.ndarray
    pos_error_m: float
    axis_error_deg: float
    within_limits: bool
    """False when the solver was clamped by a joint limit at any iteration.
    풀이가 어느 반복에서든 관절 한계에 걸렸으면 False."""

    POS_TOL_M = 5e-3
    AXIS_TOL_DEG = 5.0

    @property
    def ok(self) -> bool:
        """Converged, and never pushed against a joint limit to get there.
        수렴했고, 거기 도달하려고 관절 한계에 밀어붙이지 않았는가."""
        return (
            self.pos_error_m < self.POS_TOL_M
            and self.axis_error_deg < self.AXIS_TOL_DEG
            and self.within_limits
        )


@dataclass(frozen=True)
class PickSegment:
    """One leg of the scripted pick: where to go, how to hold the jaws, how long.
    스크립트 파지의 한 구간. 어디로 갈지, 턱을 어떻게 둘지, 얼마나 걸릴지.

    `target is None` means hold the arm's current commanded pose -- used for the
    dwell and for closing in place.
    `target is None` 이면 팔의 현재 명령 자세를 유지한다. dwell 과 제자리 닫기에 쓴다.
    """

    target: np.ndarray | None
    grip: float
    seconds: float
    # Whether unreachability disqualifies the episode. Approach and descend do:
    # if the arm cannot get to the object, there is nothing to demonstrate. The
    # lift does not, and this is not a style choice -- isolated 🟢 2026-09-03,
    # over the same 100 placements:
    #
    #     lift not gated, seed chained      2 rejected
    #     lift not gated, seed not chained  2      (chaining is irrelevant here)
    #     lift GATED,     seed chained     24      <- the cause
    #     lift gated,     seed not chained 25
    #
    # `lift_height_m` is 0.08 m and configs/so101.yaml records the measurement
    # that IK converges only up to 8 cm above the grasp point with the fingers
    # held vertical. The lift target sits exactly on that boundary, so gating on
    # it discards a quarter of otherwise fine placements.
    # 도달 불가가 에피소드를 탈락시키는가. 접근·하강은 그렇다 — 팔이 물체에 못 가면
    # 시연할 것이 없다. 들어올리기는 아니고, 이건 취향이 아니라 원인 분리 결과다.
    # `lift_height_m` 이 0.08m 이고, so101.yaml 에 "손가락을 수직으로 유지한 채로는
    # 파지점 위 8cm 까지만 IK 가 수렴한다"는 실측이 적혀 있다. 들어올리기 목표가
    # 정확히 그 경계에 있으므로, 여기에 게이트를 걸면 멀쩡한 배치의 1/4 이 버려진다.
    required: bool = True


def pick_waypoints(cfg: dict[str, Any], obj_xyz: np.ndarray) -> list[PickSegment]:
    """The one definition of the scripted demonstration.
    스크립트 시연의 **유일한** 정의.

    It lived in two places -- `data/collect.py` (collection) and
    `policy/baselines.py` (evaluation) -- and they had already drifted: collection
    produced 141 ticks, the evaluation baseline 165. That means the `scripted`
    ceiling was not the demonstrator that made the data, and neither number alone
    would have shown it. 🟢 2026-09-03
    이 계획이 두 곳에 있었다 — `data/collect.py`(수집)와 `policy/baselines.py`(평가) —
    그리고 **이미 갈라져 있었다**: 수집은 141틱, 평가 baseline 은 165틱을 냈다.
    즉 `scripted` 상한선이 데이터를 만든 시연자와 같은 것이 아니었고, 어느 한쪽
    수치만 봐서는 아무도 몰랐다.

    Timing and the dwell come from `configs/so101.yaml` so the demonstrator can be
    changed without touching code, and so a change shows up in `config_sha`.
    타이밍과 dwell 은 `configs/so101.yaml` 에서 온다. 코드를 건드리지 않고 시연자를
    바꿀 수 있고, 변경이 `config_sha` 에 남는다.
    """
    g = cfg["grasp"]
    t = g.get("timing", {})
    grasp_pt = np.asarray(obj_xyz, dtype=float) + np.array(
        [0.0, 0.0, float(g["grasp_z_offset_m"])]
    )
    approach = grasp_pt + np.array([0.0, 0.0, float(g["approach_height_m"])])
    lift = grasp_pt + np.array([0.0, 0.0, float(g["lift_height_m"])])
    open_cmd, close_cmd = float(g["open_cmd"]), float(g["close_cmd"])

    segs = [
        PickSegment(approach, open_cmd, float(t.get("approach_s", 1.2))),
        PickSegment(grasp_pt, open_cmd, float(t.get("descend_s", 1.0))),
    ]
    dwell_s = float(g.get("dwell_s", 0.0))
    if dwell_s > 0.0:
        segs.append(PickSegment(None, open_cmd, dwell_s))
    segs.append(PickSegment(None, close_cmd, float(t.get("close_s", 1.0))))
    segs.append(PickSegment(lift, close_cmd, float(t.get("lift_s", 1.5)), required=False))
    return segs


def grasp_point(model: mujoco.MjModel, data: mujoco.MjData, offset_local: np.ndarray) -> np.ndarray:
    """World position of a point fixed in the gripper body frame.
    그리퍼 body 좌표계에 고정된 점의 월드 위치.

    The TCP site sits at the very fingertip, which is not where an object ends
    up when the jaws close. The usable pinch pocket is an offset from it.
    TCP site는 손가락 끝에 있는데, 턱이 닫힐 때 물체가 실제로 놓이는 자리는
    거기가 아니다. 쓸 수 있는 파지 포켓은 거기서 떨어진 지점이다.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    rot = data.xmat[bid].reshape(3, 3)
    return data.xpos[bid] + rot @ np.asarray(offset_local, dtype=float)


def pinch_offset_for_gap(grasp_cfg: dict[str, Any], gap_m: float) -> np.ndarray:
    """Return the contact-surface midpoint in the gripper body frame.

    ``pinch_offset_local`` remains the legacy fixed point.  An asymmetric
    parallel jaw has one fixed and one moving contact surface, so its physical
    midpoint moves by half the requested pad gap.  Keeping that dependence in
    one function prevents FK, IK and rollout scoring from silently aiming at
    different points.
    """
    legacy = np.asarray(grasp_cfg["pinch_offset_local"], dtype=float)
    block = grasp_cfg.get("gap_dependent_pinch_midpoint")
    if block is None:
        return legacy.copy()
    kind = block.get("kind")
    if kind == "symmetric_parallel_jaw":
        # Both contact surfaces travel by equal and opposite half-gaps, so
        # their midpoint is fixed in the wrist-roll body frame.
        return legacy.copy()
    if kind != "asymmetric_parallel_jaw":
        raise ValueError("unknown gap_dependent_pinch_midpoint kind")
    fixed = np.asarray(block["fixed_inner_surface_local_m"], dtype=float)
    axis = np.asarray(block["gap_axis_local"], dtype=float)
    if fixed.shape != (3,) or axis.shape != (3,) or not np.isfinite(fixed).all():
        raise ValueError("gap-dependent pinch geometry must contain finite 3-vectors")
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        raise ValueError("gap_axis_local must be a non-zero finite vector")
    gap = float(gap_m)
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("gap_m must be finite and non-negative")
    return fixed + 0.5 * gap * axis / axis_norm


def _unit_local_axis(value: np.ndarray, *, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return axis / norm


def approach_axis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    local_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Unit vector the fingers point along, in world coordinates.
    손가락이 향하는 방향의 월드 단위벡터.

    The legacy model defaults to body-local -Z.  A reviewed kinematic overlay
    may supply another local axis; the caller must not infer one from visuals.
    레거시 모델은 body 로컬 -Z가 기본이다. 검토된 기구학 overlay가 다른 로컬
    축을 줄 수 있으며, 호출자가 visual mesh를 보고 축을 추정하면 안 된다.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    axis = _unit_local_axis(
        np.array([0.0, 0.0, -1.0]) if local_axis is None else local_axis,
        name="approach_axis_local",
    )
    return data.xmat[bid].reshape(3, 3) @ axis


def solve_pose_ik(
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    offset_local: np.ndarray,
    desired_axis: np.ndarray,
    approach_axis_local: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    wrist_roll: float | None = None,
    max_iters: int = 500,
    damping: float = 5e-3,
    axis_weight: float = 0.15,
    pos_tol: float = 1e-3,
    axis_tol_deg: float = 1.0,
) -> IKResult:
    """Solve for a joint configuration that puts the pinch point on target, fingers along axis.
    파지점을 목표에 두고 손가락을 지정 방향으로 향하게 하는 관절각을 푼다.

    `wrist_roll`, when given, is held fixed and excluded from the solve — use it
    to choose which way the jaws open, since that is not otherwise controllable.
    wrist_roll을 주면 고정하고 풀이에서 제외한다. 턱이 열리는 방향을 고르는
    유일한 수단이다 — 다른 방법으로는 제어되지 않는다.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    lo = model.jnt_range[:N_ARM, 0].copy()
    hi = model.jnt_range[:N_ARM, 1].copy()
    target = np.asarray(target_xyz, dtype=float)
    axis_goal = np.asarray(desired_axis, dtype=float)
    axis_goal = axis_goal / np.linalg.norm(axis_goal)

    free = [i for i in range(N_REACH) if not (wrist_roll is not None and i == 4)]

    data = mujoco.MjData(model)
    if q_init is not None:
        data.qpos[:N_ARM] = np.asarray(q_init, dtype=float)
    if wrist_roll is not None:
        data.qpos[4] = wrist_roll

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pos_err = np.inf
    axis_err_rad = np.inf
    hit_limit = False

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        point = grasp_point(model, data, offset_local)
        e_pos = target - point
        axis_now = approach_axis(model, data, approach_axis_local)
        e_axis = np.cross(axis_now, axis_goal)
        pos_err = float(np.linalg.norm(e_pos))
        axis_err_rad = float(np.arcsin(np.clip(np.linalg.norm(e_axis), 0.0, 1.0)))
        if np.dot(axis_now, axis_goal) < 0:
            axis_err_rad = np.pi - axis_err_rad
        if pos_err < pos_tol and np.degrees(axis_err_rad) < axis_tol_deg:
            break

        mujoco.mj_jac(model, data, jacp, jacr, point, bid)
        jac = np.vstack([jacp[:, free], axis_weight * jacr[:, free]])
        err = np.concatenate([e_pos, axis_weight * e_axis])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
        step = np.clip(dq, -0.1, 0.1)
        q = data.qpos[:N_ARM].copy()
        raw = q[free] + step
        # Record that the solver wanted to leave the joint range BEFORE clipping.
        # Judging limits after the clip would make within_limits trivially true —
        # every solution is inside the range because we put it there.
        # 클립하기 **전에** 풀이가 관절 범위를 벗어나려 했는지 기록한다.
        # 클립 후에 판정하면 within_limits 는 항상 True 가 된다 — 우리가 넣어놨으니까.
        if np.any(raw < lo[free] - 1e-9) or np.any(raw > hi[free] + 1e-9):
            hit_limit = True
        q[free] = np.clip(raw, lo[free], hi[free])
        data.qpos[:N_ARM] = q

    mujoco.mj_forward(model, data)
    q_final = data.qpos[:N_ARM].copy()
    return IKResult(
        qpos=q_final,
        pos_error_m=pos_err,
        axis_error_deg=float(np.degrees(axis_err_rad)),
        within_limits=not hit_limit,
    )
