"""Raw recordings into contract episodes. The one converter project-task and -127 share.
raw 기록을 계약 에피소드로. 이슈 31 과 127 이 공유하는 유일한 변환기.

Imports `contract/` and numpy only. IK, image resizing and the joint/gap tables all
arrive as arguments -- so the same code converts a simulator round-trip and a real
ARCore recording without either dragging the other's dependencies in.
`contract/` 와 numpy 만 임포트한다. IK·이미지 리사이즈·관절/간격 표는 전부 인자로
받는다. 그래서 같은 코드가 시뮬 왕복과 실제 ARCore 기록을 둘 다 변환하고, 어느
쪽도 상대의 의존성을 끌고 들어오지 않는다.

## 세 가지 판단이 여기 들어 있다

**1. 스텝을 중간에서 버리지 않는다. 가장 긴 연속 구간만 쓴다.**
계약 검증기는 제어 주기 표류를 20% 이내로 검사한다. 중간 스텝 하나를 버리면 그
자리의 dt 가 두 배가 되어 **계약 위반이 된다.** 구멍을 메우면 없던 데이터를
만드는 것이다. 그래서 IK 가 실패한 스텝이 있으면 성공 스텝의 최장 연속 구간을
잘라 쓰고, 얼마나 버렸는지 보고한다.

**2. `action[t] = q[t+1]`, 타임스탬프는 `state` 와 같다.**
계약 0.3의 `action[t]`은 출처와 무관하게 다음 스텝의 관절 배치다. 타임스탬프를
t+1로 두면 33ms 차이가 나서 계약의 10ms 게이트를 위반한다.
명령을 낸 시각은 t 다. 마지막 raw 상태에는 다음 목표가 없으므로 학습 행에서 삭제한다.

`action[t] = q[t]` 로 두면 `action - state` 가 전부 0 이 되어 정책이 "가만히 있기"를
배운다. 손실은 잘 떨어지고 로봇은 안 움직인다.

**3. 이미지 리사이즈를 여기서 하지 않는다.**
`policy.resize` 가 없으면 224x224 가 아닌 이미지를 **거부**한다. 폰 프레임을
numpy 최근접으로 줄이면 아무도 고르지 않은 에일리어싱이 계약 안으로 들어온다.
리샘플러를 고르는 것은 이 이음매의 일이 아니다.

## 계약에 자리가 없는 계측값은 `notes` 에 넣는다

pose 스트림과 RGB 스트림 사이 오프셋은 계약에 필드가 없다. `state`/`action` 은 둘 다
pose 스트림에서 나오므로 계약의 10ms 게이트는 이 오차를 보지 못한다. 그래서
`EpisodeMeta.notes["image_sync_offset_ms"]` 에 실측값을 넣는다 — 계약을 바꾸지 않고
project-task 의 계측이 데이터와 함께 이동한다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from contract.episode import (
    ACTION_DIM,
    CONTRACT_VERSION,
    GRIPPER_GAP_RANGE_M,
    GRIPPER_MAX_GAP_M,
    IMAGE_SHAPE,
    RANGE_TOLERANCE,
    STATE_DIM,
    Episode,
    EpisodeMeta,
)

from umi.ik import IKSolution, IKSolver, ResidualSummary, summarize
from umi.raw import GRIPPER_STATUS_VALID, RawEpisode, require_convertible

# Rejection reasons. Strings rather than an enum because they end up in a report
# a human reads and in EXP_LOG as JSON.
# 폐기 이유. 사람이 읽는 리포트와 EXP_LOG JSON 으로 나가므로 enum 대신 문자열이다.
REJECT_NOT_CONVERGED = "ik_not_converged"
REJECT_LIMIT = "ik_joint_limit"
REJECT_POS_ERROR = "ik_pos_error_over_budget"
REJECT_GAP_OUT_OF_RANGE = "gripper_gap_outside_curve"
REJECT_GRIPPER_MISSING = "gripper_gap_missing"
"""SPEC 의 `status` 가 T(템플릿 보강) 또는 X(실패)인 프레임. `gap_m` 이 결측이다.
채워 넣지 않고 폐기한다 — 0 으로 채우면 "완전히 닫혔다"가 되어 학습이 그것을 배운다."""


class ConversionError(ValueError):
    """The recording cannot be converted, and pretending otherwise would produce
    an episode that lies.
    이 기록은 변환할 수 없다. 억지로 하면 거짓말하는 에피소드가 나온다."""


@dataclass
class ConversionPolicy:
    """Every threshold the conversion applies, fixed before results are seen.
    변환이 적용하는 모든 기준값. 결과를 보기 전에 확정한다."""

    max_pos_error_m: float = 5e-3
    """Position budget for the conversion alone. Replay tolerance measures 4/4 at
    plus/minus 5mm and 3/4 at plus/minus 10mm 🟢 -- conversion may not eat half of it.
    변환만의 위치 예산. 재생 허용오차가 ±5mm 4/4, ±10mm 3/4 🟢 이므로 변환이
    절반을 써선 안 된다."""

    trajectory_limits: object | None = None
    min_steps: int = 30
    """Shorter than a second at 30Hz is not a demonstration. 30Hz 에서 1초 미만은 시연이 아니다."""

    resize: Callable[[np.ndarray, int, int], np.ndarray] | None = None
    """(frame_hwc, height, width) -> resized frame_hwc. None means refuse non-224 input.
    None 이면 224 가 아닌 입력을 거부한다."""

    clamp_gap: bool = False
    """When True, a gap outside the measured curve is clamped to its ends instead of
    rejecting the step. Off by default: the UMI rig's jaws are not the SO-101's, and
    silently clamping hides how often the human asked for something the robot cannot do.
    True 면 실측 곡선 밖의 간격을 양 끝으로 물린다. 기본은 끈다 — UMI 리그의 턱은
    SO-101 의 턱이 아니고, 조용히 물리면 사람이 로봇이 못 하는 것을 얼마나 자주
    요구했는지가 가려진다."""


@dataclass
class ConversionReport:
    """What the conversion accepted, what it threw away, and what it cost.
    변환이 무엇을 받고 무엇을 버렸으며 대가가 얼마였는가.

    This is the instrument project-task (실행가능성 검사기 + 수용률) needs. It is a
    by-product of converting, not a separate tool -- a separate tool would measure a
    different pipeline than the one that produced the data.
    이슈 113 이 필요한 계측기가 이것이다. 별도 도구가 아니라 변환의 부산물이다.
    별도 도구는 데이터를 만든 파이프라인과 다른 것을 재게 된다."""

    recording_id: str
    n_steps_in: int
    n_steps_out: int
    rejects: dict[str, int]
    kept_span: tuple[int, int]
    residual: ResidualSummary
    image_sync_offset_ms: dict[str, dict[str, float]]
    identity_residual: dict[str, float]
    """`mean|action - state|` per joint, in normalised units, plus the MSE an
    identity predictor would score. A dataset property -- no model needed.
    관절별 `mean|action - state|` (정규화 단위)와 identity 예측기의 MSE.
    모델 없이 나오는 데이터셋 속성이다.

    ⚠️ 학습 손실은 이 `identity_mse` 보다 **뚜렷하게** 낮아야 의미가 있다.
       절대 관절각 표현은 손실 대부분을 "팔이 지금 어디 있나"로 채운다."""

    @property
    def acceptance_rate(self) -> float:
        """Fraction of possible next-tick transitions that survived."""
        possible = max(0, self.n_steps_in - 1)
        return self.n_steps_out / possible if possible else 0.0


def normalize_joints(
    q_rad: np.ndarray, ranges_rad: np.ndarray, *, clip: bool = False
) -> np.ndarray:
    """Joint angles to the contract's [-1, 1], per configs/so101.yaml normalization.
    관절각을 계약의 [-1,1] 로. configs/so101.yaml normalization 규칙 그대로.

        x_norm = 2 * (x_rad - lo) / (hi - lo) - 1

    `clip` defaults to False, matching `sim/mujoco/build_scene.normalize`, and for
    the same reason it gives there: MuJoCo joint limits are soft, contact forces
    push qpos past the range, and clipping inside the mapping hides that. Observed
    max +1.0062 during scripted collection — one episode in 17 failed contract
    validation because of it 🟢. Whether to clip, widen the range or reject the
    episode is the caller's decision, and hiding it here removes the choice.
    `clip` 기본값 False 는 `sim/mujoco/build_scene.normalize` 와 같고 이유도 같다.
    MuJoCo 관절 한계는 soft 라 접촉력이 qpos 를 범위 밖으로 밀어내는데, 매핑
    안에서 클립하면 그 사실이 가려진다. 스크립트 수집 중 최대 +1.0062 관측,
    그 때문에 17편 중 1편이 계약 검증에 실패했다 🟢. 클립할지·범위를 넓힐지·
    에피소드를 버릴지는 호출자의 결정이고, 여기서 숨기면 결정 자체가 사라진다.

    ⚠️ 같은 공식이 `sim/mujoco/build_scene.normalize` 에도 있다. 경계 규칙상
       `umi/` 는 `sim/` 을 임포트할 수 없어 중복이 불가피하다. 중복을 없앨 수
       없으면 **갈라지는 것을 탐지 가능하게** 만든다 — 어댑터 층
       (`tools/umi_mujoco.py`)이 생성 시 두 구현을 대조하고 다르면 실패한다.
       그쪽은 float32 로 계산하므로 대조 허용오차는 1e-6 수준이다.
    """
    q = np.asarray(q_rad, dtype=float)
    lo = ranges_rad[:, 0]
    hi = ranges_rad[:, 1]
    if np.any(hi <= lo):
        raise ConversionError(f"관절 범위가 뒤집혔다: lo={lo} hi={hi}")
    x = 2.0 * (q - lo) / (hi - lo) - 1.0
    if clip:
        x = np.clip(x, -1.0, 1.0)
    return x.astype(np.float32)


def normalize_gap(gap_m: np.ndarray | float) -> np.ndarray:
    """Full contact-surface gap 0..0.09 m to the contract's [-1, 1]."""
    gap = np.asarray(gap_m, dtype=float)
    return 2.0 * gap / GRIPPER_MAX_GAP_M - 1.0


def denormalize_joints(x_norm: np.ndarray, ranges_rad: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize_joints`. configs/so101.yaml `normalization.inverse`.
    normalize_joints 의 역변환."""
    x = np.asarray(x_norm, dtype=float)
    lo = ranges_rad[:, 0]
    hi = ranges_rad[:, 1]
    return (x + 1.0) / 2.0 * (hi - lo) + lo


def invert_gap_curve(
    gap_m: float,
    curve: Sequence[Sequence[float]],
    *,
    clamp: bool = False,
) -> float:
    """Finger gap in metres to the gripper joint angle in radians.
    손가락 간격[m]을 그리퍼 관절각[rad]으로.

    `curve` is `configs/so101.yaml` `grasp.gap_curve`: rows of [angle_rad, gap_cm].
    **Note the unit change** -- the config is in centimetres, the raw schema in metres.
    `curve` 는 `grasp.gap_curve` 다. 행이 [각도 rad, 간격 cm] 이다.
    **단위가 다르다** — config 는 cm, raw 스키마는 m 다.

    Measured range 🟢: -0.1745 rad -> 0.53 cm (완전 닫힘), 1.7453 rad -> 7.94 cm.
    The jaws never touch: fully closed still leaves 5.3mm of pad-to-pad gap. A UMI
    recording that asks for less is asking for something the robot cannot do, which
    is a finding, not a rounding error.
    턱은 닿지 않는다. 완전히 닫아도 패드 간 5.3mm 가 남는다. 그보다 좁은 간격을
    요구하는 UMI 기록은 로봇이 못 하는 것을 요구하는 것이고, 그건 반올림 오차가
    아니라 발견이다.

    Raises ConversionError outside the measured range unless `clamp` is set --
    extrapolating a measured curve invents numbers.
    실측 범위 밖에서는 `clamp` 가 아니면 예외를 던진다. 실측 곡선의 외삽은
    수치를 지어내는 것이다.
    """
    table = np.asarray(curve, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 or len(table) < 2:
        raise ConversionError(f"gap_curve 형식이 [angle_rad, gap_cm] 행들이 아니다: {table.shape}")

    angles = table[:, 0]
    gaps_m = table[:, 1] / 100.0  # cm -> m
    if not np.all(np.diff(gaps_m) > 0):
        raise ConversionError("gap_curve 가 단조증가가 아니다 — 역변환이 유일하지 않다")

    g = float(gap_m)
    if g < gaps_m[0] or g > gaps_m[-1]:
        if not clamp:
            raise ConversionError(
                f"간격 {g * 100:.2f}cm 가 실측 곡선 밖이다 "
                f"[{gaps_m[0] * 100:.2f}, {gaps_m[-1] * 100:.2f}]cm"
            )
        g = float(np.clip(g, gaps_m[0], gaps_m[-1]))
    return float(np.interp(g, gaps_m, angles))


def _nearest_frames(
    pose_ts: np.ndarray, image_ts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each pose timestamp, the nearest image frame index and the offset in ms.
    pose 타임스탬프마다 가장 가까운 이미지 프레임 인덱스와 오차[ms].

    Nearest, never interpolated. Blending two frames produces an image that was
    never seen, and a policy trained on invented pixels fails in a way that looks
    like a model problem.
    최근접만 쓰고 보간하지 않는다. 두 프레임을 섞으면 실제로 본 적 없는 이미지가
    되고, 지어낸 픽셀로 학습한 정책은 모델 문제처럼 보이는 방식으로 실패한다.
    """
    idx = np.searchsorted(image_ts, pose_ts)
    idx = np.clip(idx, 1, len(image_ts) - 1) if len(image_ts) > 1 else np.zeros_like(idx)
    if len(image_ts) > 1:
        left = image_ts[idx - 1]
        right = image_ts[idx]
        take_left = (pose_ts - left) <= (right - pose_ts)
        idx = np.where(take_left, idx - 1, idx)
    offset_ms = np.abs(image_ts[idx] - pose_ts) * 1000.0
    return idx.astype(int), offset_ms


def _longest_ok_run(ok: np.ndarray) -> tuple[int, int]:
    """Half-open [start, end) of the longest contiguous True run. Empty gives (0, 0).
    가장 긴 연속 True 구간의 반열린 구간 [start, end). 없으면 (0, 0)."""
    best = (0, 0)
    start = None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(ok) - start > best[1] - best[0]:
        best = (start, len(ok))
    return best


def convert(
    raw: RawEpisode,
    ik: IKSolver,
    ranges_rad: np.ndarray,
    gap_curve: Sequence[Sequence[float]],
    *,
    policy: ConversionPolicy | None = None,
    control_rate_hz: float | None = None,
    collected_by: str = "",
    config_sha: str = "",
    git_rev: str = "",
) -> tuple[Episode, ConversionReport]:
    """Convert one raw recording into one contract episode, and report what it cost.
    raw 기록 하나를 계약 에피소드 하나로 변환하고, 대가를 보고한다.

    `ranges_rad` is (6, 2) of [lo, hi] from `configs/so101.yaml` `joints[].range_rad`.
    `gap_curve` remains an API-compatibility argument for older callers; contract
    0.3 never maps physical gap through a hinge representation.
    `ranges_rad` 는 (6,2) [lo,hi]다. `gap_curve` 인자는 이전 호출자 호환용으로
    유지하지만 계약 0.3에서는 물리 gap을 hinge 표현으로 변환하지 않는다.
    """
    pol = policy or ConversionPolicy()
    require_convertible(raw)
    if not np.isfinite(pol.max_pos_error_m) or pol.max_pos_error_m <= 0:
        raise ConversionError('invalid position error budget')
    if type(pol.min_steps) is not int or pol.min_steps < 2:
        raise ConversionError('min_steps must be an integer >= 2')
    if raw.meta.source in ('arcore', 'arcore_imu') and pol.trajectory_limits is None:
        raise ConversionError('real recordings require explicit trajectory limits')
    if pol.trajectory_limits is not None:
        pol.trajectory_limits.arrays()

    ranges = np.asarray(ranges_rad, dtype=float)
    if (ranges.shape != (STATE_DIM, 2) or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] >= ranges[:, 1])):
        raise ConversionError(f"ranges_rad 는 {(STATE_DIM, 2)} 여야 한다, got {ranges.shape}")

    t_in = raw.meta.n_steps
    rate = control_rate_hz if control_rate_hz is not None else raw.meta.pose_rate_hz
    if not np.isfinite(rate) or rate <= 0:
        raise ConversionError('invalid control rate')

    # --- IK, 스텝별 ---------------------------------------------------------
    solutions: list[IKSolution | None] = []
    rejects: Counter[str] = Counter()
    q_prev: np.ndarray | None = None

    status = np.asarray(raw.gripper_status).astype("<U1")

    for i in range(t_in):
        if status[i] not in GRIPPER_STATUS_VALID:
            rejects[REJECT_GRIPPER_MISSING] += 1
            solutions.append(None)
            continue
        gap_m = float(raw.gripper_gap_m[i])
        if (not np.isfinite(gap_m)
                or not GRIPPER_GAP_RANGE_M[0] <= gap_m <= GRIPPER_GAP_RANGE_M[1]):
            rejects[REJECT_GAP_OUT_OF_RANGE] += 1
            solutions.append(None)
            continue

        sol = ik.solve(
            raw.eef_pos[i], raw.eef_quat[i], float(raw.gripper_gap_m[i]), q_init=q_prev
        )
        q = np.asarray(sol.q_rad, dtype=float).copy()
        errors = np.array([sol.pos_error_m, sol.axis_error_deg, sol.roll_residual_deg])
        if (q.shape != (STATE_DIM,) or not np.isfinite(q).all()
                or not np.isfinite(errors).all() or np.any(errors < 0)):
            rejects['invalid_ik_solution'] += 1
            solutions.append(None)
            continue
        # SO-101 pose IK is J1~J5. Channel 6 remains physical gap and must not
        # pass through the legacy MuJoCo hinge representation.
        q[STATE_DIM - 1] = 0.0
        if np.any(q[:5] < ranges[:5, 0]) or np.any(q[:5] > ranges[:5, 1]):
            rejects[REJECT_LIMIT] += 1
            solutions.append(None)
            continue
        sol = IKSolution(
            q_rad=q,
            pos_error_m=sol.pos_error_m,
            axis_error_deg=sol.axis_error_deg,
            roll_residual_deg=sol.roll_residual_deg,
            within_limits=sol.within_limits,
            converged=sol.converged,
        )

        # 시드는 **채택된 스텝만** 넘긴다. 실패한 해를 넘기면 나쁜 분지가 연쇄한다
        # (TS_umi_ik_unseeded_0907: 진단기가 이 규칙을 어겨 실패 3건이 11건이 됐다).
        #
        # 2026-09-07 에 이 규칙을 "무조건 이어감"으로 바꿨다가 **되돌렸다.**
        # 근거로 든 "다양체 밖에서 다중시작이 스텝당 8배" 는 추론이었고 실측은
        # 1.33배였다 — 타임아웃의 실제 원인은 다양체 밖 목표에 조기 종료가 없어
        # `max_iters` 를 전부 도는 것이었다.
        #
        # 실측 🟢 (tilt 20도, n=300): 무조건 이어가면 관절 한계 클램프가
        # 44/300 → 114/300 (6.7 SE, 유의) 로 늘고 위치 오차 p95 가 28.4 → 85.3mm
        # 로 3배 나빠진다. 실패한 자세를 시드로 쓰면 궤적이 한계 쪽으로 끌려간다.
        # 비용은 1.12배뿐이었다. 정확도를 사는 게 맞다.
        #   (수용률 11.7% vs 7.7% 은 차이 1.66 SE 로 유의하지 않다. 인용 금지)

        if not sol.converged:
            rejects[REJECT_NOT_CONVERGED] += 1
            solutions.append(None)
            continue
        if not sol.within_limits:
            rejects[REJECT_LIMIT] += 1
            solutions.append(None)
            continue
        if sol.pos_error_m > pol.max_pos_error_m:
            rejects[REJECT_POS_ERROR] += 1
            solutions.append(None)
            continue

        solutions.append(sol)
        q_prev = q

    ok = np.array([s is not None for s in solutions], dtype=bool)
    if pol.trajectory_limits is not None:
        from umi.trajectory import continuous_mask
        positions = np.array([s.q_rad[:5] if s is not None else np.full(5, np.nan)
                              for s in solutions])
        ok, dynamics_rejects = continuous_mask(positions, raw.gripper_gap_m,
            raw.pose_timestamp, ok, pol.trajectory_limits)
        rejects.update(dynamics_rejects)
    start, end = _longest_ok_run(ok)
    n_raw_kept = end - start
    n_out = n_raw_kept - 1
    if n_out < pol.min_steps:
        raise ConversionError(
            f"{raw.meta.recording_id}: 다음-tick 목표가 있는 학습 행이 {n_out} 로 "
            f"최소 {pol.min_steps} 미만이다. 폐기 내역 {dict(rejects)}\n"
            "  중간 스텝을 버리고 이어붙이지 않는다 — 제어 주기 표류로 계약 위반이 된다"
        )

    kept = [s for s in solutions[start:end] if s is not None]
    q_seq = np.stack([np.asarray(s.q_rad, dtype=float) for s in kept])

    # --- 계약 배열 -----------------------------------------------------------
    arm = np.stack([normalize_joints(q, ranges)[:5] for q in q_seq])
    gap = normalize_gap(np.asarray(raw.gripper_gap_m[start:end]))[:, None]
    normalized = np.concatenate([arm, gap], axis=1).astype(np.float32)
    # Each row is a transition. The final raw state has no q[t+1], so it is not
    # repeated or masked into the training contract.
    state = normalized[:-1]
    action = normalized[1:]

    raw_ts = np.asarray(raw.pose_timestamp[start:end], dtype=np.float64)
    ts = raw_ts[:-1]
    state_ts = ts.copy()
    action_ts = ts.copy()

    if state.shape != (n_out, STATE_DIM) or action.shape != (n_out, ACTION_DIM):
        raise ConversionError(f"배열 형태가 계약과 다르다: {state.shape} {action.shape}")
    # 이 검사는 살아 있어야 한다. normalize_joints 가 클립하지 않는 이유가 이것이다.
    # IK 는 해를 model.jnt_range 안으로 클립하므로, 여기서 범위를 벗어난다는 것은
    # 넘겨받은 ranges_rad(=configs/so101.yaml joints[].range_rad)가 컴파일된
    # 모델의 jnt_range 와 다르다는 뜻이다. 스텝 폐기로 넘기지 않고 크게 실패한다 —
    # 설정과 모델이 어긋난 채로 만든 데이터셋은 전량 무효다.
    worst = float(max(np.abs(state).max(), np.abs(action).max()))
    if worst > 1.0 + RANGE_TOLERANCE:
        raise ConversionError(
            f"정규화 결과가 [-1,1] 을 {worst - 1.0:.3e} 벗어났다 (허용 {RANGE_TOLERANCE:.0e}).\n"
            "  IK 해는 model.jnt_range 안으로 클립되므로, 이는 넘겨받은 ranges_rad 가\n"
            "  컴파일된 모델의 jnt_range 와 다르다는 뜻이다. "
            "build_scene.verify_against_config 로 대조해라"
        )

    # --- 이미지: 최근접 정렬, 리사이즈는 주입받은 것만 -------------------------
    images: dict[str, np.ndarray] = {}
    sync: dict[str, dict[str, float]] = {}
    _, height, width = IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]

    for cam, frames in raw.images.items():
        idx, offset_ms = _nearest_frames(ts, np.asarray(raw.image_timestamp[cam]))
        picked = frames[idx]  # (n_out, H, W, 3)
        if picked.shape[1:3] != (height, width):
            if pol.resize is None:
                raise ConversionError(
                    f"images[{cam}] 가 {picked.shape[1:3]} 인데 계약은 {(height, width)} 다. "
                    "리사이즈 함수를 policy.resize 로 주입해라 — 여기서 최근접으로 줄이면 "
                    "아무도 고르지 않은 에일리어싱이 계약 안으로 들어온다"
                )
            picked = np.stack([pol.resize(f, height, width) for f in picked])
        chw = np.ascontiguousarray(picked.transpose(0, 3, 1, 2)).astype(np.uint8)
        images[cam] = chw
        sync[cam] = {
            "median_ms": float(np.median(offset_ms)),
            "p95_ms": float(np.percentile(offset_ms, 95)),
            "max_ms": float(offset_ms.max()),
        }

    # --- identity 기준선: 모델 없이 나오는 데이터셋 속성 -----------------------
    diff = np.abs(action - state)
    identity = {f"joint{j}_mean_abs": float(diff[:, j].mean()) for j in range(ACTION_DIM)}
    identity["mean_abs"] = float(diff.mean())
    identity["identity_mse"] = float((diff ** 2).mean())

    residual = summarize(kept)

    meta = EpisodeMeta(
        episode_id=raw.meta.recording_id,
        skill_id=raw.meta.skill_id,
        task=raw.meta.notes.get("task", raw.meta.skill_id),
        source=raw.meta.source,
        success=bool(raw.meta.notes.get("success", False)),
        n_steps=n_out,
        control_rate_hz=float(rate),
        cameras=sorted(images),
        contract_version=CONTRACT_VERSION,
        collected_by=collected_by,
        config_sha=config_sha,
        git_rev=git_rev,
        notes={
            **raw.meta.notes,
            "converted_from": "umi_raw",
            "trajectory_checked": pol.trajectory_limits is not None,
            "trajectory_limits": ({name: list(getattr(pol.trajectory_limits, name))
                for name in ('max_step', 'max_velocity', 'max_acceleration')}
                if pol.trajectory_limits is not None else None),
            "raw_version": raw.meta.raw_version,
            "raw_frame": raw.meta.frame,
            "calibration_id": raw.meta.calibration_id,
            "kept_span": [start, end],
            "final_raw_tick_rule": "drop_no_next_target",
            "n_steps_in": t_in,
            "rejects": dict(rejects),
            # 계약에 필드가 없는 계측값. project-task 이 이 값을 본다.
            "image_sync_offset_ms": sync,
            "ik_pos_error_median_mm": residual.pos_error_median_mm,
            "ik_roll_residual_median_deg": residual.roll_residual_median_deg,
            "identity_residual": identity,
        },
    )

    episode = Episode(
        meta=meta,
        images=images,
        state=state,
        state_timestamp=state_ts,
        action=action,
        action_timestamp=action_ts,
    )
    report = ConversionReport(
        recording_id=raw.meta.recording_id,
        n_steps_in=t_in,
        n_steps_out=n_out,
        rejects=dict(rejects),
        kept_span=(start, end),
        residual=residual,
        image_sync_offset_ms=sync,
        identity_residual=identity,
    )
    return episode, report
