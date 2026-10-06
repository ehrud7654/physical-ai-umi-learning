"""What the converter needs from an IK solver, and how to measure what it lost.
변환기가 IK 솔버에게 요구하는 것, 그리고 잃어버린 것을 재는 방법.

No IK is implemented here. The solver is injected, because the two callers need
different ones and neither may drag the other's dependencies along:

  - project-task (관통 검증) injects a MuJoCo adapter over
    `sim/mujoco/kinematics.py:solve_pose_ik`, which is already validated 🟢
  - project-task (실데이터) will inject a MoveIt/URDF solver on the robot side

여기에 IK 를 구현하지 않는다. 솔버는 **주입받는다.** 두 호출자가 서로 다른 솔버를
필요로 하고, 어느 쪽도 상대의 의존성을 끌고 들어와선 안 되기 때문이다.

## 6자유도 pose 를 5개 구속으로 투영한다 — 이게 계약 경계다

SO-101 팔 관절은 5개다. 위치 3 + 접근방향 2 = 5 로 정확히 결정되고,
**접근축 둘레 회전은 독립적으로 고를 수 없다** (🟢 `sim/mujoco/kinematics.py`
docstring, 그리고 `solve_pose_ik` 는 애초에 6D pose 를 받는 인자가 없다).

UMI raw 의 6자유도 pose 중 실제로 쓰이는 것:

    eef_pos          → 위치 3개 구속        (전부 쓴다)
    quat 의 R[:, 2]  → 접근축 2개 구속      (전부 쓴다)
    quat 의 R[:, 0]  → 턱 방향 = 접근축 둘레 회전 1개   ← **버린다**

버린 1개는 `roll_residual_deg` 로 계측해서 보고한다. 이슈 127 이 요구하는
"자세 오차" 가 이것이다. 사라진 것을 0 으로 보고하지 않는다.

⚠️ 이 투영 규칙은 계약 경계 결정이다. 작성자 AI 학습 담당자(트랙B, 트랙A 대행),
   D-AI 기록 필요. **되돌릴 조건**: pick 이외 스킬(`align_fixture`,
   `present_inspect`)에서 접근축 둘레 회전이 태스크 성공에 유의하게 기여한다는
   계측이 나오면 뒤집는다 — 그때는 6자유도 팔이 필요하다는 뜻이고 HW 결정이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix for a unit quaternion in (w, x, y, z) order.
    (w, x, y, z) 순서 단위 사원수의 회전 행렬.

    Order matters and is not conventional across libraries -- MuJoCo uses wxyz,
    scipy uses xyzw. Getting it backwards is a silent 180-degree-class error, so
    the order lives in the function name's docstring and nowhere else.
    순서는 중요하고 라이브러리마다 다르다 — MuJoCo 는 wxyz, scipy 는 xyzw 다.
    거꾸로 넣으면 조용한 180도급 오류가 되므로 순서는 이 docstring 에만 둔다.
    """
    q = np.asarray(q_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must be (4,) wxyz, got {q.shape}")
    n = float(np.linalg.norm(q))
    if n == 0.0:
        raise ValueError("zero quaternion is not a rotation")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a rotation matrix. Sign is not canonical.
    회전 행렬의 단위 사원수 (w, x, y, z). 부호는 정규형이 아니다.

    q and -q are the same rotation. Never compare quaternions elementwise --
    compare the rotations they describe, or the axes you actually care about.
    q 와 -q 는 같은 회전이다. 사원수를 원소별로 비교하지 마라. 기술하는 회전을,
    또는 실제로 관심 있는 축을 비교하라.
    """
    r = np.asarray(rot, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {r.shape}")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def approach_axis_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """The direction the fingers point, per the raw schema's EEF convention.
    raw 스키마의 EEF 규약에 따른 손가락 방향.

    `umi/raw.py` fixes this as R[:, 2]. Do not read it off anywhere else.
    `umi/raw.py` 가 R[:, 2] 로 고정한다. 다른 데서 임의로 읽지 않는다.
    """
    return quat_to_matrix(q_wxyz)[:, 2]


def jaw_axis_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """The direction the jaws open along, per the raw schema's EEF convention.
    raw 스키마의 EEF 규약에 따른 턱이 열리는 방향. R[:, 0]."""
    return quat_to_matrix(q_wxyz)[:, 0]


def roll_residual_deg(
    q_desired_wxyz: np.ndarray,
    achieved_approach: np.ndarray,
    achieved_jaw: np.ndarray,
) -> float:
    """Rotation about the approach axis that the 5-DOF arm could not deliver, in degrees.
    5자유도 팔이 낼 수 없었던 접근축 둘레 회전 [도].

    This is the discarded degree of freedom, measured rather than assumed to be
    zero. Both jaw axes are projected onto the plane normal to the *achieved*
    approach axis -- using the desired axis instead would fold position error
    into an orientation number.
    버린 자유도를 0 으로 가정하지 않고 실제로 잰 값이다. 두 턱 축을 **달성된**
    접근축에 수직인 평면으로 투영한다. 목표 축을 쓰면 위치 오차가 자세 수치에
    섞여 들어간다.

    The jaw axis is **undirected**: a parallel gripper rotated 180 degrees about
    the approach axis is the same physical grasp with the two jaws swapped. So the
    result is folded into [0, 90] degrees. Measuring on [0, 180] reports a
    perfectly good grasp as a 170-degree error — measured 🟢 2026-09-07, the FK→IK
    round trip produced a p95 of 106.6 degrees that way, which was the metric
    lying, not the solver failing.
    턱 축은 **방향이 없다.** 평행 그리퍼를 접근축 둘레로 180도 돌리면 두 턱이
    자리를 바꿀 뿐 같은 파지다. 그래서 결과를 [0, 90] 도로 접는다. [0, 180] 으로
    재면 멀쩡한 파지를 170도 오차로 보고한다 — 실측 🟢 2026-09-07, FK→IK 왕복에서
    p95 106.6도가 나왔고 그건 솔버가 실패한 게 아니라 계측기가 거짓말한 것이었다.

    Returns 90.0 when the projection is degenerate (the jaw axis is parallel to
    the approach axis), which cannot happen for a valid right-handed frame and
    therefore means the input frame is malformed.
    투영이 퇴화하면(턱 축이 접근축과 평행) 90.0 을 반환한다. 유효한 오른손
    좌표계에서는 일어날 수 없으므로, 그 값은 입력 프레임이 잘못됐다는 뜻이다.
    """
    axis = np.asarray(achieved_approach, dtype=float)
    axis = axis / np.linalg.norm(axis)

    def _project(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        p = v - np.dot(v, axis) * axis
        n = float(np.linalg.norm(p))
        return p / n if n > 1e-9 else np.zeros(3)

    want = _project(jaw_axis_from_quat(q_desired_wxyz))
    got = _project(achieved_jaw)
    if not np.any(want) or not np.any(got):
        return 90.0
    # abs() 가 180도 대칭을 접는다. 결과는 [0, 90].
    cos = float(np.clip(abs(np.dot(want, got)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


@dataclass(frozen=True)
class IKSolution:
    """One pose solved into joint angles, with what the solve cost.
    pose 하나를 관절각으로 푼 결과와 그 대가."""

    q_rad: np.ndarray
    """(6,) joint angles in radians, arm 5 + gripper. 관절각 [rad], 팔 5 + 그리퍼."""

    pos_error_m: float
    """Distance from the commanded position to where the pinch point landed.
    명령 위치와 파지점이 실제로 도달한 지점 사이 거리 [m]."""

    axis_error_deg: float
    """Angle between the commanded approach axis and the achieved one.
    명령 접근축과 달성된 접근축 사이 각도 [도]."""

    roll_residual_deg: float
    """The discarded degree of freedom, measured. See `roll_residual_deg`.
    버린 자유도의 실측값. 위 `roll_residual_deg` 참조."""

    within_limits: bool
    """False when the solver was clamped by a joint limit. 관절 한계에 걸렸으면 False."""

    converged: bool
    """False when the solver ran out of iterations. 반복 한도에서 끝났으면 False."""

    @property
    def ok(self) -> bool:
        """Converged, inside limits, and position is usable.
        수렴했고, 한계 안이고, 위치가 쓸 만하다.

        The 5mm threshold is not arbitrary: replay position tolerance measures
        4/4 at plus/minus 5mm and 3/4 at plus/minus 10mm 🟢, so conversion alone
        must not eat half the budget.
        5mm 기준은 임의값이 아니다. 재생 위치 허용오차가 ±5mm 4/4, ±10mm 3/4 🟢
        이므로 변환만으로 예산의 절반을 써선 안 된다.
        """
        return self.converged and self.within_limits and self.pos_error_m <= 5e-3


class IKSolver(Protocol):
    """What the converter requires. Implementations live outside `umi/`.
    변환기가 요구하는 것. 구현은 `umi/` 밖에 둔다.

    `q_init` is the previous timestep's solution. Passing it matters: IK is
    multi-solution and seeding from the last pose is what keeps a trajectory
    continuous instead of flipping the elbow mid-motion.
    `q_init` 은 직전 스텝의 해다. 넘기는 게 중요하다 — IK 는 다해이고, 직전
    pose 에서 출발해야 궤적이 연속으로 유지된다. 안 그러면 동작 중간에 팔꿈치가
    뒤집힌다.
    """

    def solve(
        self,
        pos_m: np.ndarray,
        quat_wxyz: np.ndarray,
        gripper_gap_m: float,
        q_init: np.ndarray | None = None,
    ) -> IKSolution:
        """Solve one pose. 하나의 pose 를 푼다."""
        ...


@dataclass(frozen=True)
class ResidualSummary:
    """What a whole trajectory's worth of IK cost, as the numbers we report.
    궤적 전체의 IK 대가를 보고용 수치로 정리한 것.

    Median rather than mean, and p95 rather than max: a single unreachable frame
    at the start of a demonstration would drag a mean anywhere, and the gate in
    project-task is written on the median for that reason.
    평균이 아니라 중앙값, 최대가 아니라 p95 다. 시연 앞부분의 도달 불가 프레임
    하나가 평균을 아무 데로나 끌고 가고, 그래서 127 의 게이트도 중앙값에 걸려 있다.
    """

    n: int
    pos_error_median_mm: float
    pos_error_p95_mm: float
    pos_error_max_mm: float
    axis_error_median_deg: float
    roll_residual_median_deg: float
    roll_residual_p95_deg: float
    n_not_converged: int
    n_limit_clamped: int
    n_over_5mm: int

    @property
    def passes_127_gate(self) -> bool:
        """The gate fixed before results were seen: median position error within 5mm.
        결과를 보기 전에 확정된 게이트 — 위치 오차 중앙값 5mm 이내.

        ⚠️ Passing this proves the conversion plumbing, NOT that human UMI
        trajectories are reproducible. A round-trip through poses that came from
        valid joint configurations lies exactly on the 5-DOF manifold and is
        reachable by construction.
        ⚠️ 통과는 **변환 배관**을 증명하고, 사람 UMI 궤적의 재현 가능성은
        증명하지 않는다. 유효한 관절 배치에서 나온 pose 를 왕복시키면 5자유도
        다양체 위에 정확히 놓여서 구조적으로 도달 가능하다.
        """
        return self.pos_error_median_mm <= 5.0


def summarize(solutions: Sequence[IKSolution]) -> ResidualSummary:
    """Reduce per-step IK results to the numbers that get reported.
    스텝별 IK 결과를 보고할 수치로 줄인다."""
    if not solutions:
        raise ValueError("빈 해 목록은 요약할 수 없다")
    pos_mm = np.array([s.pos_error_m for s in solutions], dtype=float) * 1000.0
    axis_deg = np.array([s.axis_error_deg for s in solutions], dtype=float)
    roll_deg = np.array([s.roll_residual_deg for s in solutions], dtype=float)
    return ResidualSummary(
        n=len(solutions),
        pos_error_median_mm=float(np.median(pos_mm)),
        pos_error_p95_mm=float(np.percentile(pos_mm, 95)),
        pos_error_max_mm=float(pos_mm.max()),
        axis_error_median_deg=float(np.median(axis_deg)),
        roll_residual_median_deg=float(np.median(roll_deg)),
        roll_residual_p95_deg=float(np.percentile(roll_deg, 95)),
        n_not_converged=int(sum(not s.converged for s in solutions)),
        n_limit_clamped=int(sum(not s.within_limits for s in solutions)),
        n_over_5mm=int((pos_mm > 5.0).sum()),
    )
