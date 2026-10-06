"""The wiring layer: MuJoCo kinematics behind the `umi.ik.IKSolver` interface.
배선 층. MuJoCo 기구학을 `umi.ik.IKSolver` 인터페이스 뒤에 둔다.

**This is the only file allowed to import both `umi/` and `sim/`.** `umi/` stays
neutral so the real-data path (project-task) can inject a MoveIt/URDF solver
instead; the simulator dependency lives here, at the call site, not in the seam.
**`umi/` 와 `sim/` 을 함께 임포트할 수 있는 유일한 파일이다.** `umi/` 를 중립으로
두어야 실데이터 경로(이슈 31)가 MoveIt/URDF 솔버를 대신 주입할 수 있다. 시뮬
의존성은 이음매가 아니라 호출 지점인 여기 있다.

⚠️ `sim/mujoco/kinematics.py` 와 `sim/mujoco/build_scene.py` 는 시뮬·정책 대화
   단독 소유다. **임포트만 하고 고치지 않는다.** 그쪽이 `solve_pose_ik` 의
   인자·반환을 바꾸면 이 파일이 깨진다 — 소유권 문서 "변경 예고" 에 올려뒀다.

    # [로컬]
    cd AI && python tools/umi_mujoco.py --episodes 5 --steps 90

`main()` 은 관통 검증의 **계측기 검증**이다. 유효한 관절 배치에서 FK 로 pose 를
만들고 IK 로 되돌린다. 그 pose 는 5자유도 다양체 위에 정확히 놓여 있으므로
**오차는 솔버 수렴 한계뿐이어야 한다.** 크게 나오면 좌표계·오프셋·부호 버그다.
이 수치를 UMI 실행가능성으로 보고하지 않는다 — 도달 가능성이 구조적으로 보장된
입력이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from paths import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from sim.mujoco.build_scene import (  # noqa: E402
    build_model,
    load_config,
    normalize as sim_normalize,
    verify_against_config,
)
from sim.mujoco.kinematics import (  # noqa: E402
    IKResult,
    approach_axis,
    grasp_point,
    pinch_offset_for_gap,
    solve_pose_ik,
)
from umi.convert import (  # noqa: E402
    ConversionError,
    invert_gap_curve,
    normalize_gap,
    normalize_joints,
)
from umi.ik import (  # noqa: E402
    IKSolution,
    approach_axis_from_quat,
    jaw_axis_from_quat,
    matrix_to_quat,
    roll_residual_deg,
)
from umi.ik import summarize  # noqa: E402

N_JOINTS = 6
GRIPPER_BODY = "gripper"
WRIST_ROLL_IDX = 4
ROLL_CORRECTION_SIGN = -1.0
"""Sign of the wrist_roll correction, MEASURED 🟢 2026-09-07 not derived.

`wrist_roll` rotates about the gripper body's local +z while the approach axis is
its local -z, so the sign flips somewhere -- and which way depends on the MJCF's
axis convention, not on anything I can read off a docstring. Both signs were run:
-1 gave roll p95 0.0000 deg, +1 gave 88.0767 deg.
`wrist_roll` 은 그리퍼 body 로컬 +z 둘레로 돌고 접근축은 로컬 -z 라서 어딘가에서
부호가 뒤집힌다. 어느 쪽인지는 MJCF 의 축 규약에 달렸고 docstring 으로 알 수 없다.
두 부호를 다 돌려봤다 — -1 이 roll p95 0.0000도, +1 이 88.0767도.

⚠️ 그리퍼 형상이 바뀌면(로봇팔 변형 예정) 재측정해야 한다."""

NORM_CROSSCHECK_ATOL = 1e-6
"""`sim.build_scene.normalize` computes in float32, `umi` in float64. Anything
tighter than this compares the two dtypes, not the two formulas.
`sim.build_scene.normalize` 는 float32, `umi` 는 float64 로 계산한다. 이보다
빡빡하게 잡으면 공식이 아니라 dtype 을 비교하는 것이 된다."""


def joint_ranges(cfg: dict[str, Any]) -> np.ndarray:
    """(6, 2) of [lo, hi] in radians, from the config, in joint-index order.
    설정에서 읽은 (6,2) [lo, hi] 라디안. 관절 인덱스 순서.

    Read straight from the YAML rather than through `joint_specs`, which casts to
    float32 -- the contract's normalisation is defined on these exact numbers.
    `joint_specs` 를 거치지 않고 YAML 에서 바로 읽는다. 그쪽은 float32 로
    캐스팅하는데, 계약의 정규화는 이 정확한 수치 위에 정의돼 있다.
    """
    rows = sorted(cfg["joints"], key=lambda j: int(j["index"]))
    if len(rows) != N_JOINTS:
        raise ConversionError(f"관절이 {len(rows)}개다. {N_JOINTS}개여야 한다")
    return np.array([[float(j["range_rad"][0]), float(j["range_rad"][1])] for j in rows])


def gap_from_angle(angle_rad: float, curve: Sequence[Sequence[float]]) -> float:
    """Gripper joint angle to pad-to-pad gap in metres. Forward direction of the
    measured curve; `umi.convert.invert_gap_curve` is the inverse.
    그리퍼 관절각을 패드 간 간격[m]으로. 실측 곡선의 정방향."""
    table = np.asarray(curve, dtype=float)
    return float(np.interp(float(angle_rad), table[:, 0], table[:, 1] / 100.0))


def apply_kinematic_grasp(
    cfg: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Apply a reviewed kinematic-only gripper frame to one runtime config.

    This changes FK/IK geometry only.  It does not replace the MuJoCo visual,
    collision, inertia or actuator model, and callers must preserve that
    limitation in every result produced with the overlay.
    """
    if payload.get("schema") != "so101_kinematic_grasp/0.1.0":
        raise ValueError("unsupported kinematic grasp schema")
    if payload.get("dynamic_mujoco_ready") is not False:
        raise ValueError(
            "kinematic grasp payload must explicitly declare dynamic_mujoco_ready=false")
    offset = np.asarray(payload.get("pinch_offset_local_m"), dtype=float)
    axes = payload.get("eef_frame_local")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("pinch_offset_local_m must be a finite 3-vector")
    if not isinstance(axes, dict):
        raise ValueError("eef_frame_local is required")
    frame = np.column_stack([
        np.asarray(axes.get("jaw_axis"), dtype=float),
        np.asarray(axes.get("up_axis"), dtype=float),
        np.asarray(axes.get("approach_axis"), dtype=float),
    ])
    if frame.shape != (3, 3) or not np.isfinite(frame).all():
        raise ValueError("eef_frame_local axes must be finite 3-vectors")
    gram = frame.T @ frame
    if not np.allclose(gram, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("eef_frame_local axes must be orthonormal")
    if not np.isclose(np.linalg.det(frame), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("eef_frame_local must be right-handed")
    if payload.get("gap_midpoint_kind") != "symmetric_parallel_jaw":
        raise ValueError("only a symmetric parallel-jaw kinematic overlay is supported")

    grasp = cfg["grasp"]
    grasp["pinch_offset_local"] = offset.tolist()
    grasp["eef_frame_local"] = {
        "jaw_axis": frame[:, 0].tolist(),
        "up_axis": frame[:, 1].tolist(),
        "approach_axis": frame[:, 2].tolist(),
    }
    grasp["gap_dependent_pinch_midpoint"] = {
        "kind": "symmetric_parallel_jaw",
    }
    grasp["kinematic_overlay"] = {
        "schema": payload["schema"],
        "source_archive_sha256": payload.get("source_archive_sha256"),
        "camera_optical_extrinsic_present": payload.get(
            "camera_optical_extrinsic_present") is True,
        "dynamic_mujoco_ready": False,
    }
    collision = payload.get("collision_proxy")
    if collision is not None:
        if (not isinstance(collision, dict)
                or collision.get("schema")
                != "so101_parallel_jaw_collision_proxy/0.1.0-provisional"
                or collision.get("diagnostic_dynamic_ready") is not True
                or collision.get("measured_collision_geometry") is not False):
            raise ValueError("invalid provisional ver1 collision proxy declaration")
        cfg["gripper_pads"] = {
            **collision,
            "kind": "symmetric_parallel_jaw_runtime_proxy",
        }
        visual_model = payload.get("visual_model")
        if (not isinstance(visual_model, dict)
                or visual_model.get("schema")
                != "so101_parallel_jaw_visual/0.1.0-provisional"
                or visual_model.get("visual_collision_alignment_required") is not True):
            raise ValueError(
                "provisional ver1 collision proxy requires its matching visible model"
            )
        cfg["gripper_pads"]["visual_model"] = visual_model
        grasp["kinematic_overlay"]["diagnostic_collision_proxy_ready"] = True
    return cfg


def _eef_frame_local(grasp_cfg: dict[str, Any] | None) -> np.ndarray:
    if grasp_cfg is None or "eef_frame_local" not in grasp_cfg:
        return np.diag([1.0, -1.0, -1.0])
    axes = grasp_cfg["eef_frame_local"]
    return np.column_stack([
        np.asarray(axes["jaw_axis"], dtype=float),
        np.asarray(axes["up_axis"], dtype=float),
        np.asarray(axes["approach_axis"], dtype=float),
    ])


def eef_pose_from_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_rad: np.ndarray,
    pinch_offset_local: np.ndarray,
    grasp_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics to the raw schema's EEF pose: pinch position and orientation.
    raw 스키마의 EEF pose 로 가는 순기구학. 파지점 위치와 자세.

    The orientation follows `umi/raw.py`'s convention.  The legacy model uses:

        R[:, 2] = approach axis = **minus** the gripper body's local z 🟢
        R[:, 0] = jaw axis      = the gripper body's local x
        R[:, 1] = z x x         = minus the body's local y (keeps it right-handed)

    A reviewed `eef_frame_local` overlay replaces all three local axes together;
    mixing one overlay axis with the legacy signs is rejected upstream.
    자세는 `umi/raw.py` 규약을 따른다. 검토된 `eef_frame_local` overlay가 있으면
    로컬 세 축을 한꺼번에 교체한다. overlay 한 축과 레거시 부호를 섞지 않는다.

    Position is the pinch point, not the TCP site -- the same point
    `solve_pose_ik` aims at, so the round trip compares like with like.
    위치는 TCP site 가 아니라 파지점이다. `solve_pose_ik` 가 조준하는 바로 그
    점이라서 왕복이 같은 것끼리 비교된다.
    """
    data.qpos[:N_JOINTS] = np.asarray(q_rad, dtype=float)
    mujoco.mj_forward(model, data)
    pos = grasp_point(model, data, np.asarray(pinch_offset_local, dtype=float))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    rot = data.xmat[bid].reshape(3, 3)
    local_frame = _eef_frame_local(grasp_cfg)
    r_eef = rot @ local_frame
    # 규약 자기검사: 이 행렬의 z열은 sim 의 approach_axis 와 같아야 한다.
    if not np.allclose(
            r_eef[:, 2], approach_axis(model, data, local_frame[:, 2]),
            atol=1e-12):
        raise ConversionError("EEF 규약과 sim.approach_axis 가 불일치 — 부호 규약을 다시 봐라")
    return np.asarray(pos, dtype=float).copy(), matrix_to_quat(r_eef)


def signed_roll_error_rad(
    quat_desired: np.ndarray, achieved_approach: np.ndarray, achieved_jaw: np.ndarray
) -> float:
    """Signed rotation about the achieved approach axis, folded to (-90, 90] degrees.
    달성된 접근축 둘레의 부호 있는 회전. (-90, 90] 도로 접는다.

    `umi.ik.roll_residual_deg` is unsigned because reporting wants a magnitude.
    Correcting needs a direction, and the parallel jaw's 180-degree symmetry means a
    +170 degree error is really -10 degrees -- correcting toward +170 walks away.
    보고에는 크기만 필요해서 `roll_residual_deg` 는 부호가 없다. 보정에는 방향이
    필요하고, 평행 턱의 180도 대칭 때문에 +170도 오차는 실제로 -10도다.
    """
    axis = np.asarray(achieved_approach, dtype=float)
    axis = axis / np.linalg.norm(axis)

    def proj(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        pp = v - np.dot(v, axis) * axis
        n = float(np.linalg.norm(pp))
        return pp / n if n > 1e-9 else np.zeros(3)

    want = proj(jaw_axis_from_quat(quat_desired))
    got = proj(achieved_jaw)
    if not np.any(want) or not np.any(got):
        return 0.0
    ang = float(np.arctan2(float(np.dot(np.cross(got, want), axis)), float(np.dot(got, want))))
    while ang > np.pi / 2:
        ang -= np.pi
    while ang <= -np.pi / 2:
        ang += np.pi
    return ang


class MujocoArmAdapter:
    """Five-joint API over the simulation-only six-joint solver."""

    def __init__(self, solver, gap_curve):
        self.solver, self.gap_curve = solver, gap_curve

    def solve(self, target_pose, *, seed_rad, gripper_width_m):
        from umi.arm_preflight import ArmSolution
        from umi.convert import invert_gap_curve
        seed = np.r_[seed_rad, invert_gap_curve(gripper_width_m, self.gap_curve)]
        sol = self.solver.solve(target_pose[:3, 3],
                                matrix_to_quat(target_pose[:3, :3]),
                                gripper_width_m, q_init=seed)
        return ArmSolution(sol.q_rad[:5].copy(), sol.pos_error_m,
                           sol.axis_error_deg, sol.roll_residual_deg,
                           sol.converged, sol.within_limits)


class MujocoIK:
    """`umi.ik.IKSolver` over `sim.mujoco.kinematics.solve_pose_ik`.
    `solve_pose_ik` 를 감싼 `umi.ik.IKSolver` 구현.

    ## 자유도 배분 — 실측 🟢 2026-09-07, docstring 을 믿지 말고 재라

    레거시 고정 파지점에서 `wrist_roll` 만 흔들었을 때 (무작위 40자세 x 15각도):

        파지점 이동   0.0000 mm      접근축 변화   0.000 도      턱축 변화 88.57 도
        (wrist_flex: 282mm / 176도,  elbow_flex: 552mm / 180도)

    이 결과는 `grasp.pinch_offset_local = [0, 0, -0.08]` 이 roll 축 위에 있을 때만
    성립한다. SO-101의 실제 접촉면 중점은 고정 턱과 가동 턱 사이에 있어 gap/2만큼
    로컬 x로 이동한다. 따라서 현재 경로는 gap 종속 중점을 쓰며, roll 보정 뒤
    위치·접근축을 다시 풀어 결합 오차를 제거한다.

        pan · lift · elbow_flex · wrist_flex (4개)  →  위치 3 + 접근축 2  (과결정)
        wrist_roll (1개)                            →  roll 전담, 완전 독립

    따라서 **roll 은 버릴 필요가 없지만**, gap 종속 중점에서는 한 번의 보정이
    더는 충분하지 않다. roll과 4관절 위치·접근축 풀이를 번갈아 수행한다.

    ⚠️ `wrist_roll` 을 풀이에 열어두면(이전 구현) 야코비안에 영향 0 인 열이 생겨
       감쇠 최소자승이 그 널 방향으로 표류한다. 그게 roll 잔차 47도의 원인이었다.
       그래서 여기서는 `wrist_roll` 을 고정해 **4관절 문제로 풀고** 나중에 정한다.

    ⚠️ 이 분해는 파지점이 roll 축 위에 있다는 데 전적으로 의존한다. 그리퍼 형상이
       바뀌면(로봇팔 변형 예정) 무너진다. 재측정 대상이다.

    ⚠️ `sim/mujoco/kinematics.py` docstring 은 "위치 3 + 접근방향 2 = 5로 정확히
       결정된다 / 접근축 둘레 회전은 나머지 다섯과 독립적으로 고를 수 없다"고 쓰고
       있다. 실측과 다르다 — 그 파일은 시뮬·정책 대화 소유이므로 고치지 않고
       소유권 문서 "변경 예고" 로 전달한다.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: dict[str, Any],
        *,
        crosscheck: bool = True,
        pos_tol_m: float = 1e-7,
        axis_tol_deg: float = 0.05,
        match_roll: bool = True,
        multistart: bool = True,
    ) -> None:
        """`pos_tol_m` and `axis_tol_deg` are tightened from `solve_pose_ik`'s
        defaults (1e-3 m, 1.0 deg) on purpose, and recorded as conditions.

        MEASURED 🟢 2026-09-07: with the defaults, the round-trip position error
        piled up against 1.0mm — median 0.830, p95 0.983, max 0.9989, with 0/450
        non-converged. That distribution is the stopping tolerance, not the
        geometry. 1mm is 20% of the conversion's 5mm budget spent on an early
        exit, and there were iterations left over.
        기본값(1e-3 m, 1.0도)보다 조인 값이고, 실행 조건으로 기록한다.

        실측 🟢 2026-09-07: 기본값으로는 왕복 위치 오차가 1.0mm 에 쌓였다 —
        중앙 0.830, p95 0.983, 최대 0.9989, 미수렴 0/450. 그 분포는 기하가 아니라
        정지 허용오차다. 변환 예산 5mm 의 20% 를 조기종료로 쓰는 것이고, 반복
        여유는 남아 있었다.

        ⚠️ `solve_pose_ik` 의 기본값을 바꾸지 않는다 — 그 함수는 시뮬·정책 대화
           소유이고 스크립트 수집이 그 기본값으로 검증돼 있다. 여기서 인자로만
           덮는다.

        2026-09-08: 기본값을 1e-5 → **1e-7** 로 더 조였다. 관절 복원 정확도가
        걸려 있다 — 계약 왕복에서 관절별 최대차가 움직임 대비 8.91% → 0.96% 로
        떨어진다 (`MEASURE_umi_roundtrip_0907.md` M10). 비용은 1.23배(1.93s→2.37s),
        미수렴 0/270 불변.

        `axis_tol_deg` 는 0.05 를 유지한다. 100배 조여봤지만 수치가 **바이트 단위로
        동일**했다 — `pos_tol=1e-7` 이면 축 오차가 이미 훨씬 아래라 그 조건이
        구속하지 않는다. 아무것도 사지 못하는 제약은 걸지 않는다.
        """
        self.model = model
        self.cfg = cfg
        self.pos_tol_m = float(pos_tol_m)
        self.axis_tol_deg = float(axis_tol_deg)
        self.match_roll = bool(match_roll)
        self.multistart = bool(multistart)
        self.max_iters = 500
        """`solve_pose_ik` 의 반복 한도. 도달 가능한 목표는 조기 종료하므로 기본값이
        비용이 아니지만, **다양체 밖 목표는 한도를 전부 돈다** — 실험에서는 낮춘다.
        실측: 다양체 밖 + 다중시작이 스텝당 8배 x 500반복으로 170초 타임아웃을 냈다."""
        self.data = mujoco.MjData(model)
        self.pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
        self.eef_frame_local = _eef_frame_local(cfg["grasp"])
        self.ranges = joint_ranges(cfg)
        if crosscheck:
            self._crosscheck()

    def pinch_for_gap(self, gap_m: float) -> np.ndarray:
        """Contact-surface midpoint for one commanded pad gap."""
        return pinch_offset_for_gap(self.cfg["grasp"], gap_m)

    def forward_pose(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FK using the same gap-dependent pinch point used by IK."""
        q = np.asarray(q_rad, dtype=float)
        gap = gap_from_angle(float(q[5]), self.cfg["grasp"]["gap_curve"])
        return eef_pose_from_joints(
            self.model, self.data, q, self.pinch_for_gap(gap), self.cfg["grasp"])

    def _crosscheck(self) -> None:
        """Fail now if the config, the compiled model and the two normalisations disagree.
        설정·컴파일된 모델·두 정규화 구현이 어긋나면 지금 실패한다.

        Duplication across the track boundary cannot be removed, so make divergence
        detectable. A converter that normalises differently from the simulator
        produces a dataset whose numbers mean something else, and nothing downstream
        can tell.
        트랙 경계 때문에 중복을 없앨 수 없으니 갈라지는 것을 탐지 가능하게 만든다.
        시뮬과 다르게 정규화하는 변환기는 수치의 의미가 다른 데이터셋을 만들고,
        하류에서는 아무도 그걸 알 수 없다.
        """
        problems = verify_against_config(self.model, self.cfg)
        if problems:
            raise ConversionError(
                "컴파일된 모델과 설정의 관절 한계가 다르다:\n  " + "\n  ".join(problems)
            )
        rng = np.random.default_rng(20260907)
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        q = lo + rng.random((64, N_JOINTS)) * (hi - lo)
        mine = np.stack([normalize_joints(row, self.ranges) for row in q])
        gaps = np.array([gap_from_angle(row[5], self.cfg["grasp"]["gap_curve"])
                         for row in q])
        mine[:, 5] = normalize_gap(gaps)
        theirs = np.stack([sim_normalize(row, self.cfg) for row in q])
        worst = float(np.abs(mine - theirs).max())
        if worst > NORM_CROSSCHECK_ATOL:
            raise ConversionError(
                f"umi.normalize_joints 와 sim.build_scene.normalize 가 최대 {worst:.3e} "
                f"다르다 (허용 {NORM_CROSSCHECK_ATOL:.0e}). 두 구현이 갈라졌다"
            )
        self.norm_crosscheck_max = worst

    def _seed_grid(self) -> list[np.ndarray]:
        """Deterministic seeds for a solve that has no previous solution to start from.
        직전 해가 없는 풀이에 쓸 결정적 시드 목록.

        MEASURED 🟢 2026-09-07: every IK failure in the 20-episode pass-through was
        an **unseeded** solve. `convert()` does not advance `q_prev` on failure, so a
        failed first step leaves the next steps unseeded too -- ep7 steps 0,1,2 all
        solved from home and all failed, recovering at step 3 when the target drifted
        into home's basin. Not a singularity: the failures' worst constraint-Jacobian
        sigma_min (0.02762) is larger than that of 13.5% of the successful steps, and
        seeding with the true configuration converged 3/3 to 0.0000mm.
        실측 🟢: 20편 관통에서 IK 실패는 전부 **시드 없는** 풀이였다. `convert()` 는
        실패 시 `q_prev` 를 갱신하지 않으므로 첫 스텝이 실패하면 다음 스텝들도 시드가
        없다 — ep7 의 0·1·2 가 모두 원점에서 풀려 모두 실패하고, 목표가 원점의 수렴
        분지로 들어온 step 3 에서 회복했다. 특이자세가 아니다: 실패의 최악
        sigma_min(0.02762)보다 작은 성공 스텝이 13.5% 있고, 참값 시딩은 3/3 이
        0.0000mm 로 수렴했다.

        Only unseeded steps pay for this -- normally one per episode.
        비용은 시드 없는 스텝만 낸다. 보통 에피소드당 한 번이다.
        """
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        seeds = [np.zeros(N_JOINTS), mid.copy()]
        for frac in (0.35, -0.35, 0.6, -0.6, 0.85, -0.85):
            seeds.append(mid + frac * half)
        return seeds

    def _best_unseeded(
        self, pos_m: np.ndarray, desired_axis: np.ndarray, held_roll: float,
        pinch_offset: np.ndarray, gripper_angle: float,
    ) -> IKResult:
        """Multi-start over `_seed_grid`, keeping the usable solve with least error.
        `_seed_grid` 다중시작. 쓸 수 있는 해 중 오차가 가장 작은 것을 남긴다."""
        best: IKResult | None = None

        def worse(cand: IKResult, ref: IKResult) -> bool:
            key_c = (not cand.within_limits, cand.pos_error_m, cand.axis_error_deg)
            key_r = (not ref.within_limits, ref.pos_error_m, ref.axis_error_deg)
            return key_c >= key_r

        for seed in self._seed_grid():
            seed = seed.copy()
            seed[5] = gripper_angle
            cand = solve_pose_ik(
                self.model,
                target_xyz=np.asarray(pos_m, dtype=float),
                offset_local=pinch_offset,
                desired_axis=desired_axis,
                approach_axis_local=self.eef_frame_local[:, 2],
                q_init=seed,
                wrist_roll=held_roll if self.match_roll else None,
                pos_tol=self.pos_tol_m,
                axis_tol_deg=self.axis_tol_deg,
                max_iters=self.max_iters,
            )
            if best is None or not worse(cand, best):
                best = cand
            if (
                best.within_limits
                and best.pos_error_m < self.pos_tol_m
                and best.axis_error_deg < self.axis_tol_deg
            ):
                break  # 더 볼 필요 없다
        assert best is not None
        return best

    def _achieved(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Approach axis and jaw axis at a joint configuration.
        주어진 관절 배치에서의 접근축과 턱 축."""
        self.data.qpos[:N_JOINTS] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self.model, self.data)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
        rot = self.data.xmat[bid].reshape(3, 3)
        frame = rot @ self.eef_frame_local
        return frame[:, 2], frame[:, 0]

    def _roll_for(self, q: np.ndarray, quat_wxyz: np.ndarray) -> float:
        """The `wrist_roll` value that puts the jaws where the demonstration had them.
        시연의 턱 방향을 재현하는 `wrist_roll` 값.

        The angular correction itself is linear.  A gap-dependent pinch midpoint
        is off the roll axis, so the caller must re-solve position afterwards.

        If the corrected value leaves the joint range, the two equivalents at plus
        and minus 180 degrees are tried first -- the parallel jaw is the same grasp
        with the jaws swapped. Only if none fits is it clipped, and then the
        shortfall shows up in `roll_residual_deg` rather than being hidden.
        보정값이 관절 범위를 벗어나면 ±180도 등가값을 먼저 시도한다. 평행 턱은
        180도 돌려도 두 턱이 자리를 바꿀 뿐 같은 파지다. 전부 안 맞으면 클립하고,
        그 부족분은 숨기지 않고 `roll_residual_deg` 로 드러난다.
        """
        appr, jaw = self._achieved(q)
        err = signed_roll_error_rad(quat_wxyz, appr, jaw)
        lo, hi = self.ranges[WRIST_ROLL_IDX]
        target = float(q[WRIST_ROLL_IDX]) + ROLL_CORRECTION_SIGN * err
        valid = [cand for cand in (target, target + np.pi, target - np.pi)
                 if lo <= cand <= hi]
        if valid:
            # Parallel jaws make +/-180 degrees physically equivalent.  Choose
            # the equivalent nearest the seeded/previous wrist value so a
            # continuous demonstration does not acquire artificial pi jumps.
            return float(min(valid, key=lambda cand: abs(cand - q[WRIST_ROLL_IDX])))
        return float(np.clip(target, lo, hi))

    def match_wrist_roll(self, q_rad: np.ndarray, quat_wxyz: np.ndarray) -> float:
        """Return the continuous wrist-roll label matching the demonstrated jaws."""
        return self._roll_for(np.asarray(q_rad, dtype=float), quat_wxyz)

    def solve(
        self,
        pos_m: np.ndarray,
        quat_wxyz: np.ndarray,
        gripper_gap_m: float,
        q_init: np.ndarray | None = None,
    ) -> IKSolution:
        desired_axis = approach_axis_from_quat(quat_wxyz)
        seed = None if q_init is None else np.asarray(q_init, dtype=float)
        gripper_angle = invert_gap_curve(
            float(gripper_gap_m), self.cfg["grasp"]["gap_curve"])
        pinch_offset = self.pinch_for_gap(float(gripper_gap_m))
        if seed is not None:
            seed = seed.copy()
            seed[5] = gripper_angle
        held_roll = float(seed[WRIST_ROLL_IDX]) if seed is not None else 0.0
        if seed is None and self.multistart:
            res: IKResult = self._best_unseeded(
                pos_m, desired_axis, held_roll, pinch_offset, gripper_angle)
        else:
            res = solve_pose_ik(
                self.model,
                target_xyz=np.asarray(pos_m, dtype=float),
                offset_local=pinch_offset,
                desired_axis=desired_axis,
                approach_axis_local=self.eef_frame_local[:, 2],
                q_init=seed,
                wrist_roll=held_roll if self.match_roll else None,
                pos_tol=self.pos_tol_m,
                axis_tol_deg=self.axis_tol_deg,
                max_iters=self.max_iters,
            )
        q = np.asarray(res.qpos, dtype=float).copy()
        q[5] = gripper_angle

        if self.match_roll:
            # The gap-dependent midpoint is off the wrist-roll axis.  Roll
            # correction therefore moves the point, unlike the legacy fixed
            # offset.  Alternate roll matching and the four-joint position /
            # approach solve so the final residual is measured at one pose.
            within_limits = bool(res.within_limits)
            for _ in range(3):
                q[WRIST_ROLL_IDX] = self._roll_for(q, quat_wxyz)
                refined = solve_pose_ik(
                    self.model,
                    target_xyz=np.asarray(pos_m, dtype=float),
                    offset_local=pinch_offset,
                    desired_axis=desired_axis,
                    approach_axis_local=self.eef_frame_local[:, 2],
                    q_init=q,
                    wrist_roll=float(q[WRIST_ROLL_IDX]),
                    pos_tol=self.pos_tol_m,
                    axis_tol_deg=self.axis_tol_deg,
                    max_iters=self.max_iters,
                )
                within_limits = within_limits and bool(refined.within_limits)
                q = np.asarray(refined.qpos, dtype=float).copy()
                q[5] = gripper_angle
                res = refined
        else:
            within_limits = bool(res.within_limits)

        appr, jaw = self._achieved(q)
        roll = roll_residual_deg(quat_wxyz, appr, jaw)
        actual_pos = grasp_point(self.model, self.data, pinch_offset)
        pos_error = float(np.linalg.norm(np.asarray(pos_m, dtype=float) - actual_pos))
        axis_cos = float(np.clip(np.dot(appr, desired_axis), -1.0, 1.0))
        axis_error = float(np.degrees(np.arccos(axis_cos)))

        # 수렴 판정은 sim 쪽 상수(5mm, 5도)를 그대로 쓴다. 임계값을 새로 지어내지
        # 않는다. 솔버에 준 정지 허용오차(pos_tol_m)와는 다른 것이다 — 이쪽은
        # "쓸 수 있는 해인가", 저쪽은 "언제 멈추는가" 다.
        converged = (
            pos_error < IKResult.POS_TOL_M
            and axis_error < IKResult.AXIS_TOL_DEG
        )
        return IKSolution(
            q_rad=q,
            pos_error_m=pos_error,
            axis_error_deg=axis_error,
            roll_residual_deg=float(roll),
            within_limits=within_limits,
            converged=bool(converged),
        )


def smooth_joint_trajectory(
    ranges: np.ndarray, n: int, rng: np.random.Generator, *, span_frac: float = 0.45
) -> np.ndarray:
    """A slow trajectory strictly inside the joint limits, one sinusoid per axis.
    관절 한계 안에서만 움직이는 느린 궤적. 축마다 사인 하나.

    Inside the limits by construction, so a limit rejection during the round trip
    means the IK wandered, not that the input was impossible.
    구조적으로 한계 안이므로, 왕복 중 한계 폐기가 나오면 입력이 불가능했던 것이
    아니라 IK 가 헤맨 것이다.
    """
    lo, hi = ranges[:, 0], ranges[:, 1]
    mid = 0.5 * (lo + hi)
    span = span_frac * 0.5 * (hi - lo)
    phase = rng.random(N_JOINTS) * 2.0 * np.pi
    freq = 0.5 + rng.random(N_JOINTS)
    s = np.linspace(0.0, 1.0, n)[:, None]
    return mid + span * np.sin(2.0 * np.pi * freq * s + phase)


def main() -> int:
    ap = argparse.ArgumentParser(description="FK→IK 왕복으로 변환 계측기를 검증한다")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--kinematic-grasp", type=Path,
        help="optional reviewed kinematic-only TCP/jaw-frame overlay",
    )
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--pos-tol-m", type=float, default=1e-5,
                    help="솔버 정지 허용오차. sim 기본값은 1e-3 이고 그게 왕복 오차의 천장이었다")
    ap.add_argument("--axis-tol-deg", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.kinematic_grasp is not None:
        apply_kinematic_grasp(
            cfg,
            json.loads(args.kinematic_grasp.read_text(encoding="utf-8")),
        )
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg, pos_tol_m=args.pos_tol_m, axis_tol_deg=args.axis_tol_deg)
    data = mujoco.MjData(model)
    ranges = ik.ranges
    curve = cfg["grasp"]["gap_curve"]

    print(f"씬 {args.scene.name} · 설정 {args.config.name}")
    print(f"관절 한계 대조 OK · 정규화 두 구현 최대차 {ik.norm_crosscheck_max:.3e}")
    print(f"솔버 정지 허용오차 pos {ik.pos_tol_m:.1e}m · axis {ik.axis_tol_deg}도 "
          f"(sim 기본값 1e-3m · 1.0도)\n")

    rng = np.random.default_rng(args.seed)
    all_sols = []
    q_err_max = 0.0
    gap_err_max = 0.0

    for e in range(args.episodes):
        q_true = smooth_joint_trajectory(ranges, args.steps, rng)
        poses = [ik.forward_pose(q) for q in q_true]
        gaps = [gap_from_angle(q[5], curve) for q in q_true]

        sols = []
        q_prev = None
        for i, ((pos, quat), gap) in enumerate(zip(poses, gaps)):
            sol = ik.solve(pos, quat, gap, q_init=q_prev)
            sols.append(sol)
            q_prev = sol.q_rad
        summary = summarize(sols)
        all_sols.extend(sols)

        q_rec = np.stack([s.q_rad for s in sols])
        q_err = float(np.abs(q_rec[:, :5] - q_true[:, :5]).max())
        q_err_max = max(q_err_max, q_err)
        gap_back = [
            gap_from_angle(
                float(np.interp(g, np.asarray(curve)[:, 1] / 100.0, np.asarray(curve)[:, 0])),
                curve,
            )
            for g in gaps
        ]
        gap_err_max = max(gap_err_max, float(np.abs(np.array(gap_back) - np.array(gaps)).max()))

        print(
            f"ep{e}  위치 중앙 {summary.pos_error_median_mm:6.3f}mm  "
            f"p95 {summary.pos_error_p95_mm:6.3f}  최대 {summary.pos_error_max_mm:6.3f}  "
            f"축 {summary.axis_error_median_deg:5.2f}도  "
            f"roll잔차 중앙 {summary.roll_residual_median_deg:6.2f}도  "
            f"미수렴 {summary.n_not_converged:3d}  한계 {summary.n_limit_clamped:3d}  "
            f"5mm초과 {summary.n_over_5mm:3d}"
        )

    total = summarize(all_sols)
    print(f"\n=== 합계 n={total.n} ({args.episodes}편 x {args.steps}스텝) ===")
    print(f"위치 오차   중앙 {total.pos_error_median_mm:.4f}mm · p95 {total.pos_error_p95_mm:.4f} · 최대 {total.pos_error_max_mm:.4f}")
    print(f"접근축 오차 중앙 {total.axis_error_median_deg:.4f}도")
    print(f"roll 잔차   중앙 {total.roll_residual_median_deg:.3f}도 · p95 {total.roll_residual_p95_deg:.3f}")
    print(f"미수렴 {total.n_not_converged}/{total.n} · 한계클램프 {total.n_limit_clamped}/{total.n} · 5mm초과 {total.n_over_5mm}/{total.n}")
    print(f"관절각 복원 최대오차 (팔 5축) {q_err_max:.4f} rad = {np.degrees(q_err_max):.2f}도")
    # 이 검사는 표의 자기일관성만 본다. 실물 리그의 간격과는 무관하다.
    print(f"gap 곡선 자기일관성 최대오차 {gap_err_max * 1000:.4f} mm (실물 리그와 무관)")
    print(f"\n127 게이트 (위치 중앙값 5mm 이내): {'통과' if total.passes_127_gate else '미통과'}")
    print(
        "⚠️ 이 입력은 유효한 관절 배치에서 FK 로 만든 것이라 5자유도 다양체 위에 정확히\n"
        "   놓여 있다. 도달 가능성이 구조적으로 보장된다. 이 수치를 UMI 실행가능성으로\n"
        "   보고하지 않는다 — 재는 것은 좌표계·오프셋·부호가 맞는가다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
