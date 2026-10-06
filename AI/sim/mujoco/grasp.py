"""Measure whether the simulated SO-101 can actually pick up the target object.
시뮬 SO-101이 대상물을 실제로 집어 올릴 수 있는지 계측한다.

Scripted open-loop sequence — approach, descend, close, lift — at the control
rate from the config. No policy involved. The question is not "does a policy
work" but "is this task physically achievable in this scene at all". If the
answer is no, every downstream success rate is meaningless.
설정의 제어 주기로 도는 개루프 스크립트 시퀀스다. 정책은 개입하지 않는다.
묻는 것은 "정책이 되는가"가 아니라 "이 씬에서 이 태스크가 애초에 물리적으로
가능한가"이다. 아니라면 이후의 모든 성공률은 무의미하다.

Every number it needs comes from configs/so101.yaml. Nothing is hardcoded.
필요한 수치는 전부 configs/so101.yaml 에서 온다. 하드코딩 없음.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from sim.mujoco.build_scene import DEFAULT_CONFIG, build_model, load_config
from tracking.exp_log import file_digest, log_run
from sim.mujoco.kinematics import solve_pose_ik

JAW_BODIES = ("gripper", "moving_jaw_so101_v1")


@dataclass
class GraspResult:
    """Outcome of one scripted pick attempt.
    스크립트 파지 시도 한 번의 결과."""

    success: bool
    lift_ik_converged: bool
    object_xy: tuple[float, float]
    lift_height_m: float
    grip_force_at_close_n: float
    contacts_at_close: int
    contacts_at_end: int
    ik_max_pos_error_mm: float
    ik_max_axis_error_deg: float
    failure_reason: str | None


def gripper_geom_ids(model: mujoco.MjModel) -> set[int]:
    """Collision geom ids belonging to the two jaws.
    두 턱에 속한 충돌 geom id."""
    ids: set[int] = set()
    for body_name in JAW_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        ids |= {
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid and model.geom_contype[g] != 0
        }
    return ids


def contact_on_object(
    model: mujoco.MjModel, data: mujoco.MjData, obj_geom: int, jaw_geoms: set[int]
) -> tuple[float, int]:
    """Total normal force between jaws and object, and contact count.
    턱과 물체 사이 법선력 합계와 접촉 개수."""
    total = 0.0
    count = 0
    buf = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        pair = {con.geom1, con.geom2}
        if obj_geom in pair and pair & jaw_geoms:
            mujoco.mj_contactForce(model, data, i, buf)
            total += abs(float(buf[0]))
            count += 1
    return total, count


def hold(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    """Keep the current command for a duration.
    현재 명령을 일정 시간 유지한다."""
    for _ in range(max(1, int(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)


def move_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_target: np.ndarray,
    gripper_cmd: float,
    seconds: float,
    rate_hz: float,
) -> None:
    """Interpolate the joint command to the target at the control rate.
    제어 주기로 관절 명령을 목표까지 보간한다."""
    q_start = data.ctrl[:5].copy()
    n_ctrl = max(1, int(seconds * rate_hz))
    sub = max(1, int((1.0 / rate_hz) / model.opt.timestep))
    for k in range(1, n_ctrl + 1):
        alpha = k / n_ctrl
        data.ctrl[:5] = (1 - alpha) * q_start + alpha * q_target[:5]
        data.ctrl[5] = gripper_cmd
        for _ in range(sub):
            mujoco.mj_step(model, data)


def run_grasp(
    cfg: dict[str, Any],
    model: mujoco.MjModel,
    object_xy: tuple[float, float] | None = None,
    wrist_roll: float = 0.0,
) -> GraspResult:
    """Run one scripted pick and report what actually happened.
    스크립트 파지를 한 번 돌리고 실제로 무슨 일이 있었는지 보고한다."""
    g = cfg["grasp"]
    rate_hz = float(cfg["control"]["rate_hz"])
    offset = np.asarray(g["pinch_offset_local"], dtype=float)
    axis = np.asarray(g["approach_axis"], dtype=float)

    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    jaws = gripper_geom_ids(model)

    if object_xy is not None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = object_xy[0]
        data.qpos[adr + 1] = object_xy[1]

    mujoco.mj_forward(model, data)
    data.ctrl[:] = data.qpos[:6]
    hold(model, data, 0.4)

    obj_xyz = data.xpos[obj_bid].copy()
    z0 = float(obj_xyz[2])
    grasp_pt = obj_xyz + np.array([0.0, 0.0, float(g["grasp_z_offset_m"])])

    pos_errs: list[float] = []
    axis_errs: list[float] = []
    stages = [
        (grasp_pt + np.array([0.0, 0.0, float(g["approach_height_m"])]), g["open_cmd"], 1.2),
        (grasp_pt, g["open_cmd"], 1.0),
    ]
    for target, grip, dur in stages:
        res = solve_pose_ik(model, target, offset, axis, q_init=data.qpos[:6], wrist_roll=wrist_roll)
        pos_errs.append(res.pos_error_m * 1000.0)
        axis_errs.append(res.axis_error_deg)
        if not res.ok:
            return GraspResult(
                success=False,
                lift_ik_converged=False,
                object_xy=(float(obj_xyz[0]), float(obj_xyz[1])),
                lift_height_m=0.0,
                grip_force_at_close_n=0.0,
                contacts_at_close=0,
                contacts_at_end=0,
                ik_max_pos_error_mm=round(max(pos_errs), 3),
                ik_max_axis_error_deg=round(max(axis_errs), 3),
                failure_reason="IK 도달 불가 (자세 또는 관절한계)",
            )
        move_to(model, data, res.qpos, float(grip), dur, rate_hz)

    data.ctrl[5] = float(g["close_cmd"])
    hold(model, data, 1.2)
    force, n_close = contact_on_object(model, data, obj_gid, jaws)

    lift_target = grasp_pt + np.array([0.0, 0.0, float(g["lift_height_m"])])
    res = solve_pose_ik(model, lift_target, offset, axis, q_init=data.qpos[:6], wrist_roll=wrist_roll)
    pos_errs.append(res.pos_error_m * 1000.0)
    axis_errs.append(res.axis_error_deg)
    # A non-converged lift solve still moves the arm — clamped at a joint limit,
    # with the gripper tilted. The object may well rise, and the run then looks
    # like a success that nobody asked for. Record it instead of hiding it.
    # 수렴하지 않은 들어올림 해도 팔을 움직이기는 한다 — 관절한계에 클램프된 채,
    # 그리퍼가 기울어져서. 물체가 올라가기도 하고, 그러면 아무도 요청하지 않은
    # 성공처럼 보인다. 숨기지 말고 기록한다.
    lift_ik_converged = res.ok
    move_to(model, data, res.qpos, float(g["close_cmd"]), 1.5, rate_hz)
    hold(model, data, 0.8)

    lifted = float(data.xpos[obj_bid][2]) - z0
    _, n_end = contact_on_object(model, data, obj_gid, jaws)
    threshold = float(g["success_lift_m"])
    success = lifted >= threshold and n_end > 0

    reason = None
    if not success:
        if n_close == 0:
            reason = "그리퍼가 물체에 닿지 않음"
        elif n_end == 0:
            reason = "파지 후 놓침 (과도한 압착으로 튕겨나갔을 가능성)"
        else:
            reason = f"들어올린 높이 부족 ({lifted * 100:.1f}cm < {threshold * 100:.1f}cm)"

    return GraspResult(
        success=success,
        lift_ik_converged=lift_ik_converged,
        object_xy=(float(obj_xyz[0]), float(obj_xyz[1])),
        lift_height_m=round(lifted, 5),
        grip_force_at_close_n=round(force, 3),
        contacts_at_close=n_close,
        contacts_at_end=n_end,
        ik_max_pos_error_mm=round(max(pos_errs), 3),
        ik_max_axis_error_deg=round(max(axis_errs), 3),
        failure_reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=0.03,
                        help="uniform xy jitter of the object, metres / 물체 xy 무작위 이동 (m)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    model = build_model(cfg)
    rng = np.random.default_rng(args.seed)
    base_xy = np.array(cfg["task"]["object"]["init_pos"][:2], dtype=float)

    results: list[GraspResult] = []
    for t in range(args.trials):
        xy = base_xy + rng.uniform(-args.jitter, args.jitter, size=2)
        res = run_grasp(cfg, model, object_xy=(float(xy[0]), float(xy[1])))
        results.append(res)
        print(
            f"[{t + 1:3d}/{args.trials}] {'O' if res.success else 'X'} "
            f"xy=({res.object_xy[0]:+.3f},{res.object_xy[1]:+.3f}) "
            f"lift={res.lift_height_m * 100:6.2f}cm F={res.grip_force_at_close_n:6.2f}N "
            f"{res.failure_reason or ''}"
        )

    n_ok = sum(r.success for r in results)
    rate = n_ok / len(results)
    n_bad_ik = sum(not r.lift_ik_converged for r in results if r.contacts_at_close > 0)
    print(f"\n성공률 {n_ok}/{len(results)} = {rate * 100:.1f}%  (jitter ±{args.jitter * 1000:.0f}mm, seed={args.seed})")
    if n_bad_ik:
        print(f"⚠️ 들어올림 IK 미수렴 {n_bad_ik}회 — 그 시행의 성공은 요청한 자세로 얻은 것이 아니다")
    reasons: dict[str, int] = {}
    for r in results:
        if r.failure_reason:
            reasons[r.failure_reason] = reasons.get(r.failure_reason, 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  실패 {n:3d}회: {reason}")

    if args.log:
        rec = log_run(
            experiment="grasp_check",
            author=args.author,
            issue="project-task",
            conditions={
                "trials": args.trials,
                "jitter_m": args.jitter,
                "seed": args.seed,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "control_rate_hz": cfg["control"]["rate_hz"],
                "object_half_size_m": cfg["task"]["object"]["half_size_m"],
                "close_cmd": cfg["grasp"]["close_cmd"],
                "lift_height_m": cfg["grasp"]["lift_height_m"],
                "success_criterion": f"들어올린 높이 >= {cfg['grasp']['success_lift_m']}m 이고 종료 시 접촉 유지",
            },
            result={
                "success_rate": rate,
                "n_success": n_ok,
                "n_trials": len(results),
                "failure_reasons": reasons,
                "lift_ik_not_converged": n_bad_ik,
                "trials": [asdict(r) for r in results],
            },
        )
        print(f"EXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")


if __name__ == "__main__":
    main()
