"""Measure the two gripper facts the config depends on, reproducibly.
설정 파일이 의존하는 그리퍼 사실 두 가지를 재현 가능하게 계측한다.

`configs/so101.yaml` hardcodes a gap curve and a pad geometry. Numbers in a
config that no committed script can reproduce are assertions, not measurements —
this script exists so both can be re-derived after any model or config change.
`configs/so101.yaml` 에는 간격 곡선과 패드 형상이 박혀 있다. 커밋된 스크립트로
재현할 수 없는 설정값은 계측이 아니라 주장이다. 모델이나 설정이 바뀐 뒤에도
두 값을 다시 유도할 수 있도록 이 스크립트를 둔다.

Two probes:
  gap    — pad-to-pad opening as a function of the gripper joint angle
  hull   — whether the target object fits between the jaws at all, with the
           finger pads applied and with them removed
계측 두 가지:
  gap    — 그리퍼 관절각에 따른 패드 간 간격
  hull   — 손가락 패드를 적용했을 때와 제거했을 때, 대상물이 턱 사이에 들어가는지
"""

from __future__ import annotations

import argparse
import copy

import mujoco
import numpy as np

from sim.mujoco.build_scene import DEFAULT_CONFIG, build_model, load_config
from tracking.exp_log import file_digest, log_run


def gap_curve(model: mujoco.MjModel, angles: np.ndarray) -> list[tuple[float, float]]:
    """Pad-to-pad opening along the jaw axis, per gripper angle.
    그리퍼 각도별로 턱 축 방향 패드 간 간격을 잰다."""
    data = mujoco.MjData(model)
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    pf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad_fixed")
    pm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad_moving")
    if pf < 0 or pm < 0:
        raise RuntimeError("pad geoms not found — gripper_pads 설정이 적용되지 않았다")

    out: list[tuple[float, float]] = []
    for q in angles:
        data.qpos[:6] = 0.0
        data.qpos[5] = q
        mujoco.mj_forward(model, data)
        rot = data.xmat[gb].reshape(3, 3)
        origin = data.xpos[gb]
        cf = rot.T @ (data.geom_xpos[pf] - origin)
        cm = rot.T @ (data.geom_xpos[pm] - origin)
        # Jaws separate along gripper-local x (measured, see MEASURE_grasp_0827).
        # 턱은 gripper 로컬 x 축으로 벌어진다 (실측).
        gap = (cm[0] - model.geom_size[pm][0]) - (cf[0] + model.geom_size[pf][0])
        out.append((float(q), float(gap)))
    return out


def point_is_free(model: mujoco.MjModel, offset_local: np.ndarray, open_cmd: float) -> dict:
    """Whether the object fits at one specific point in the gripper frame.
    그리퍼 좌표계의 특정 한 점에 물체가 들어가는지.

    This is the number that matters: the configured pinch point must be free.
    The grid below is a map, and it deliberately includes points inside the pads
    themselves — penetration there is expected, not a defect.
    실제로 중요한 건 이 수치다. 설정된 파지점이 비어 있어야 한다.
    아래 격자는 지도일 뿐이고 패드 내부 점도 일부러 포함한다 — 거기서의 관통은
    결함이 아니라 당연한 결과다.
    """
    data = mujoco.MjData(model)
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    og = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    adr = model.jnt_qposadr[jid]
    jaw_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("gripper", "moving_jaw_so101_v1")
    }
    data.qpos[:6] = 0.0
    data.qpos[5] = open_cmd
    mujoco.mj_forward(model, data)
    rot = data.xmat[gb].reshape(3, 3)
    data.qpos[adr : adr + 3] = data.xpos[gb] + rot @ np.asarray(offset_local, dtype=float)
    data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    depth = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        if og not in (con.geom1, con.geom2):
            continue
        other = con.geom2 if con.geom1 == og else con.geom1
        if model.geom_bodyid[other] in jaw_bodies:
            depth = min(depth, float(con.dist))
    return {
        "offset_local": [round(float(v), 5) for v in offset_local],
        "penetration_m": round(abs(depth), 5),
        "free": depth >= 0.0,
    }


def hull_penetration(model: mujoco.MjModel, cfg: dict, open_cmd: float) -> dict:
    """How deep the object sinks into the jaw collision shapes, fully open.
    그리퍼를 완전히 벌린 상태에서 물체가 턱 충돌 형상에 얼마나 파고드는가.

    Zero penetration everywhere means there is a real pocket to grasp into.
    Non-zero means the collision shape has filled the opening and no grasp is
    geometrically possible, whatever the controller does.
    어디서도 관통이 없다면 실제로 물체가 들어갈 포켓이 있다는 뜻이다.
    관통이 있으면 충돌 형상이 입구를 메운 것이고, 제어를 어떻게 하든
    기하학적으로 파지가 불가능하다.
    """
    data = mujoco.MjData(model)
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    og = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    adr = model.jnt_qposadr[jid]
    jaw_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("gripper", "moving_jaw_so101_v1")
    }

    worst = 0.0
    n_blocked = 0
    probes = 0
    for dz in (-0.045, -0.055, -0.065, -0.075, -0.085, -0.095):
        for dx in (-0.010, 0.0, 0.010, 0.020):
            data.qpos[:6] = 0.0
            data.qpos[5] = open_cmd
            mujoco.mj_forward(model, data)
            rot = data.xmat[gb].reshape(3, 3)
            world = data.xpos[gb] + rot @ np.array([dx, 0.0, dz])
            data.qpos[adr : adr + 3] = world
            data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            probes += 1
            depth = 0.0
            for i in range(data.ncon):
                con = data.contact[i]
                if og not in (con.geom1, con.geom2):
                    continue
                other = con.geom2 if con.geom1 == og else con.geom1
                if model.geom_bodyid[other] in jaw_bodies:
                    depth = min(depth, float(con.dist))
            if depth < 0:
                n_blocked += 1
            worst = min(worst, depth)
    return {
        "probe_points": probes,
        "blocked_points": n_blocked,
        "max_penetration_m": round(abs(worst), 5),
        "graspable": n_blocked == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    lo, hi = cfg["joints"][5]["range_rad"]
    angles = np.linspace(lo, hi, 13)

    model = build_model(cfg)
    curve = gap_curve(model, angles)
    with_pads = hull_penetration(model, cfg, float(cfg["grasp"]["open_cmd"]))

    # Same scene with the pad workaround removed: what the official MJCF gives us.
    # 패드 우회를 제거한 동일 씬 — 공식 MJCF 가 주는 그대로의 상태.
    cfg_raw = copy.deepcopy(cfg)
    cfg_raw.pop("gripper_pads", None)
    model_raw = build_model(cfg_raw)
    without_pads = hull_penetration(model_raw, cfg_raw, float(cfg["grasp"]["open_cmd"]))

    print("패드 간 간격 곡선 (그리퍼 각도 → 간격):")
    for q, gap in curve:
        print(f"  {q:+.4f} rad  →  {gap * 100:6.2f} cm")

    offset = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    pt_with = point_is_free(model, offset, float(cfg["grasp"]["open_cmd"]))
    pt_without = point_is_free(model_raw, offset, float(cfg["grasp"]["open_cmd"]))

    print("\n설정된 파지점에 2cm 물체가 들어가는가 (그리퍼 벌린 상태):")
    for label, res in (("패드 적용 (현재)", pt_with), ("패드 없음 (공식 MJCF 그대로)", pt_without)):
        verdict = "들어감 — 파지 가능" if res["free"] else f"관통 {res['penetration_m'] * 100:.2f}cm — 파지 불가"
        print(f"  {label:28s} {verdict}")

    print("\n주변 격자 24점 (패드 내부 점도 포함 — 지도 용도):")
    for label, res in (("패드 적용 (현재)", with_pads), ("패드 없음 (공식 MJCF 그대로)", without_pads)):
        print(
            f"  {label:28s} 관통 {res['blocked_points']}/{res['probe_points']}점, "
            f"최대 {res['max_penetration_m'] * 100:.2f}cm"
        )

    if args.log:
        rec = log_run(
            experiment="gripper_probe",
            author=args.author,
            issue="project-task",
            conditions={
                "config_sha": file_digest(DEFAULT_CONFIG),
                "open_cmd": cfg["grasp"]["open_cmd"],
                "object_half_size_m": cfg["task"]["object"]["half_size_m"],
                "probe_grid": "dz -0.045..-0.095 x dx -0.010..0.020, gripper body local",
            },
            result={
                "gap_curve": [[round(q, 5), round(gp, 5)] for q, gp in curve],
                "pinch_point_with_pads": pt_with,
                "pinch_point_without_pads": pt_without,
                "grid_with_pads": with_pads,
                "grid_without_pads": without_pads,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']})")


if __name__ == "__main__":
    main()
