"""Verify the UMI converter's invariants with a stub IK solver. No MuJoCo needed.
스텁 IK 솔버로 UMI 변환기의 불변식을 검증한다. MuJoCo 불필요.

The instrument gets checked before the instrument is used. A converter that
silently produces contract violations, or that quietly reindexes actions by one
step, does not announce itself -- the training loss looks fine and only the robot
fails.
계측기를 쓰기 전에 계측기를 검증한다. 계약 위반을 조용히 만들거나 액션 인덱스를
한 칸 밀어버리는 변환기는 스스로 알리지 않는다. 학습 손실은 멀쩡하고 로봇만 실패한다.

    # [로컬]
    cd AI && python tools/check_umi_convert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from contract.episode import CONTRACT_VERSION, IMAGE_SHAPE, validate  # noqa: E402
from umi.convert import (  # noqa: E402
    ConversionError,
    ConversionPolicy,
    convert,
    denormalize_joints,
    invert_gap_curve,
    normalize_joints,
    normalize_gap,
)
from umi.ik import (  # noqa: E402
    IKSolution,
    approach_axis_from_quat,
    matrix_to_quat,
    quat_to_matrix,
    roll_residual_deg,
)
from umi.raw import FRAME_ROBOT_BASE, RawEpisode, RawMeta, validate_raw  # noqa: E402

# configs/so101.yaml 실측값을 복사한 것이 아니라, 이 검사가 쓰는 고정 표다.
# 실제 변환은 config 에서 읽은 값을 인자로 받는다.
RANGES = np.array(
    [
        [-1.9198621771937616, 1.9198621771937634],
        [-1.7453292519943224, 1.7453292519943366],
        [-1.69, 1.69],
        [-1.6580628494556928, 1.6580627293335335],
        [-2.7438472969992493, 2.841206309382605],
        [-0.17453297762778586, 1.7453291995659765],
    ]
)
GAP_CURVE = [
    [-0.1745, 0.53], [-0.0145, 1.39], [0.1454, 2.28], [0.3054, 3.17],
    [0.4654, 4.03], [0.6254, 4.86], [0.9454, 6.30], [1.2654, 7.34],
    [1.7453, 7.94],
]

RATE = 30.0
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


class StubIK:
    """A solver that returns a prescribed joint trajectory, optionally failing at
    chosen steps. Lets the converter be tested without any kinematics at all.
    지정된 관절 궤적을 돌려주고, 고른 스텝에서 실패하는 솔버. 기구학 없이
    변환기만 시험할 수 있다."""

    def __init__(self, q_seq: np.ndarray, fail_at: set[int] | None = None) -> None:
        self.q_seq = q_seq
        self.fail_at = fail_at or set()
        self.i = 0
        self.q_init_seen: list[np.ndarray | None] = []

    def solve(self, pos_m, quat_wxyz, gripper_gap_m, q_init=None):  # noqa: ANN001
        i = self.i
        self.i += 1
        self.q_init_seen.append(None if q_init is None else np.asarray(q_init).copy())
        failed = i in self.fail_at
        return IKSolution(
            q_rad=self.q_seq[i],
            pos_error_m=0.020 if failed else 0.0004,
            axis_error_deg=0.3,
            roll_residual_deg=12.0,
            within_limits=True,
            converged=True,
        )


def make_raw(
    n: int = 90,
    *,
    frame: str = FRAME_ROBOT_BASE,
    image_hw: tuple[int, int] = (IMAGE_SHAPE[1], IMAGE_SHAPE[2]),
    n_frames: int | None = None,
    image_lag_s: float = 0.004,
    gripper_x_at: set[int] | None = None,
) -> RawEpisode:
    """A synthetic recording. Images deliberately have their own count and their own
    timestamps, offset from the pose stream — that is the case the schema exists for.
    합성 기록. 이미지는 의도적으로 개수와 타임스탬프가 pose 와 다르다.
    스키마가 그 경우를 위해 존재한다."""
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64) / RATE + 1000.0

    # 매끄러운 EEF 궤적. 기구학적으로 유효할 필요는 없다 — 스텁 IK 가 받는다.
    s = np.linspace(0.0, 1.0, n)
    pos = np.stack([0.18 + 0.06 * s, -0.05 + 0.10 * s, 0.12 + 0.03 * np.sin(np.pi * s)], axis=1)
    quats = np.empty((n, 4), dtype=np.float64)
    for i, ang in enumerate(np.linspace(0.0, 0.6, n)):
        c, sn = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        quats[i] = matrix_to_quat(rot)
    gap = np.linspace(0.045, 0.019, n)  # 4.5cm -> 1.9cm, 실측 곡선 안
    status = np.full(n, "D", dtype="<U1")
    if gripper_x_at:
        for i in gripper_x_at:          # SPEC 의 X(검출 실패) — gap 은 결측이어야 한다
            status[i] = "X"
            gap[i] = np.nan

    nf = n_frames if n_frames is not None else n - 7
    ft = np.linspace(t[0] + image_lag_s, t[-1] + image_lag_s, nf).astype(np.float64)
    h, w = image_hw
    imgs = {
        cam: rng.integers(0, 256, size=(nf, h, w, 3), dtype=np.uint8)
        for cam in ("cam_wrist",)
    }
    return RawEpisode(
        meta=RawMeta(
            recording_id="synth_0000",
            skill_id="pick_place",
            source="sim_synth",
            frame=frame,
            n_steps=n,
            pose_rate_hz=RATE,
            cameras=["cam_wrist"],
            notes={"task": "합성 관통 검증", "success": True},
        ),
        eef_pos=pos,
        eef_quat=quats,
        gripper_gap_m=gap,
        gripper_status=status,
        pose_timestamp=t,
        images=imgs,
        image_timestamp={cam: ft.copy() for cam in imgs},
    )


def q_trajectory(n: int) -> np.ndarray:
    """A joint trajectory strictly inside the limits, moving on every axis.
    관절 한계 안에서 모든 축이 움직이는 궤적."""
    s = np.linspace(0.0, 1.0, n)
    q = np.empty((n, 6))
    for j in range(6):
        lo, hi = RANGES[j]
        mid = 0.5 * (lo + hi)
        span = 0.25 * (hi - lo)
        q[:, j] = mid + span * np.sin(2.0 * np.pi * (s + 0.1 * j))
    return q


def main() -> int:
    print("== 1. 사원수 왕복 ==")
    rot = quat_to_matrix(np.array([0.5, 0.5, 0.5, 0.5]))
    back = quat_to_matrix(matrix_to_quat(rot))
    check("quat -> R -> quat -> R 동일", np.allclose(rot, back, atol=1e-12),
          f"최대차 {np.abs(rot - back).max():.2e}")
    check("R 이 직교", np.allclose(rot @ rot.T, np.eye(3), atol=1e-12))
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    check("항등 사원수의 접근축 = +z", np.allclose(approach_axis_from_quat(ident), [0, 0, 1]))

    print("== 2. roll 잔차 ==")
    c, s90 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    rot30 = np.array([[c, -s90, 0.0], [s90, c, 0.0], [0.0, 0.0, 1.0]])
    r = roll_residual_deg(matrix_to_quat(rot30), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    check("z축 30도 회전 -> 잔차 30도", abs(r - 30.0) < 1e-6, f"{r:.6f}도")
    r0 = roll_residual_deg(ident, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    check("일치하면 잔차 0", abs(r0) < 1e-9, f"{r0:.2e}도")

    print("== 3. 정규화 ==")
    q = q_trajectory(5)
    check("정규화 왕복", np.allclose(denormalize_joints(normalize_joints(q[0], RANGES), RANGES), q[0], atol=1e-6))
    lo_norm = normalize_joints(RANGES[:, 0], RANGES)
    hi_norm = normalize_joints(RANGES[:, 1], RANGES)
    check("하한 -> -1", np.allclose(lo_norm, -1.0), f"{lo_norm}")
    check("상한 -> +1", np.allclose(hi_norm, 1.0), f"{hi_norm}")
    check("그리퍼 닫힘(하한) -> -1  (config gripper_closed_norm)", abs(lo_norm[5] + 1.0) < 1e-6)

    print("== 4. gap 곡선 역변환 ==")
    check("4.86cm -> 0.6254 rad (실측점)", abs(invert_gap_curve(0.0486, GAP_CURVE) - 0.6254) < 1e-9,
          f"{invert_gap_curve(0.0486, GAP_CURVE):.6f}")
    mid = invert_gap_curve(0.0184, GAP_CURVE)
    check("1.84cm -> close_cmd 0.06 근처", abs(mid - 0.06) < 0.02, f"{mid:.4f} rad")
    try:
        invert_gap_curve(0.002, GAP_CURVE)
        check("0.2cm 는 거부", False, "예외가 안 났다")
    except ConversionError:
        check("0.2cm 는 거부 (완전 닫힘 0.53cm 미만)", True)
    check("clamp=True 면 하한으로", abs(invert_gap_curve(0.002, GAP_CURVE, clamp=True) - (-0.1745)) < 1e-9)

    print("== 5. 정상 변환 -> 계약 위반 0 ==")
    n = 90
    raw = make_raw(n)
    check("raw 스키마 위반 0", validate_raw(raw) == [], str(validate_raw(raw)))
    ik = StubIK(q_trajectory(n))
    ep, rep = convert(raw, ik, RANGES, GAP_CURVE, control_rate_hz=RATE, collected_by="check")
    problems = validate(ep)
    check("계약 위반 0", problems == [], str(problems))
    check("가능한 전 transition 채택", rep.n_steps_out == n - 1, f"{rep.n_steps_out}/{n - 1}")
    check("수용률 1.0", abs(rep.acceptance_rate - 1.0) < 1e-12)

    print("== 6. action = q[t+1], 타임스탬프 동일 ==")
    check("action[:-1] == state[1:]", np.array_equal(ep.action[:-1], ep.state[1:]))
    check("마지막 action은 마지막 raw 상태 목표", not np.array_equal(ep.action[-1], ep.state[-1]))
    check("마지막 raw tick 학습 행 삭제", ep.meta.n_steps == n - 1)
    check("state_ts == action_ts", np.array_equal(ep.state_timestamp, ep.action_timestamp))
    resid = float(np.abs(ep.action - ep.state).mean())
    check("action-state 잔차가 0 이 아니다", resid > 1e-4, f"mean|Δ| = {resid:.5f}")
    check("identity_mse 리포트됨", rep.identity_residual["identity_mse"] > 0,
          f"{rep.identity_residual['identity_mse']:.3e}")
    check("gap 0/45/90mm 정규화", np.array_equal(
        normalize_gap(np.array([0.0, 0.045, 0.09])), np.array([-1.0, 0.0, 1.0])))
    current = ep.meta.contract_version
    ep.meta.contract_version = "0.2.0-provisional"
    check("구 계약은 신규 학습 검증에서 거부", any("contract_version" in p for p in validate(ep)))
    ep.meta.contract_version = current
    check("현행 계약 버전 상승", CONTRACT_VERSION == "0.3.0-provisional", CONTRACT_VERSION)
    wrist = ep.images["cam_wrist"]
    ep.images["cam_front"] = wrist
    ep.meta.cameras = ["cam_front", "cam_wrist"]
    check("구 2카메라 계약 거부", any("cameras must" in p for p in validate(ep)))
    del ep.images["cam_front"]
    ep.meta.cameras = ["cam_wrist"]

    print("== 7. 이미지 최근접 정렬 ==")
    check("이미지 CHW uint8", all(v.shape == (n - 1, *IMAGE_SHAPE) and v.dtype == np.uint8 for v in ep.images.values()))
    off = rep.image_sync_offset_ms["cam_wrist"]
    check("동기 오차가 notes 에 기록됨", "image_sync_offset_ms" in ep.meta.notes)
    # 상한은 pose 주기의 반이 아니라 **이미지 주기의 반**이다. 이미지 스트림이
    # pose 보다 느리면 최근접만으로는 그만큼의 오차가 남는다 — 그게 L29 의 실체다.
    ft = raw.image_timestamp["cam_wrist"]
    img_period_ms = float(np.diff(ft).mean()) * 1000.0
    check("최근접 오차가 이미지 반주기 이내", off["max_ms"] <= img_period_ms / 2 + 1e-6,
          f"median {off['median_ms']:.2f} / max {off['max_ms']:.2f} ms "
          f"(이미지 주기 {img_period_ms:.2f}ms, 반주기 {img_period_ms / 2:.2f}ms)")
    check("느린 이미지 스트림은 계약 10ms 게이트를 넘는다 — 최근접만으로 부족",
          off["max_ms"] > 10.0,
          f"max {off['max_ms']:.2f}ms > 10ms. 손 0.3m/s 가정 시 {off['max_ms'] * 0.3:.1f}mm 🟡")

    print("== 7b. 이미지 1프레임 (병리적 입력) ==")
    ep1f, rep1f = convert(make_raw(n, n_frames=1), StubIK(q_trajectory(n)), RANGES,
                          GAP_CURVE, control_rate_hz=RATE)
    check("1프레임이어도 계약 위반 0", validate(ep1f) == [], str(validate(ep1f)))
    check("1프레임은 전 스텝이 같은 이미지",
          all(np.array_equal(v[0], v[-1]) for v in ep1f.images.values()))

    print("== 8. IK 실패는 구간을 자른다. 중간을 이어붙이지 않는다 ==")
    raw2 = make_raw(n)
    ik2 = StubIK(q_trajectory(n), fail_at={40})
    ep2, rep2 = convert(raw2, ik2, RANGES, GAP_CURVE, control_rate_hz=RATE)
    check("계약 위반 0 (자른 뒤에도)", validate(ep2) == [], str(validate(ep2)))
    check("최장 구간을 골랐다", rep2.n_steps_out == n - 42, f"{rep2.n_steps_out} (span {rep2.kept_span})")
    check("폐기 이유가 기록됨", rep2.rejects.get("ik_pos_error_over_budget") == 1, str(rep2.rejects))
    dt = np.diff(ep2.state_timestamp)
    check("제어 주기에 구멍 없음", float(np.abs(dt - 1.0 / RATE).max()) < 1e-9, f"최대편차 {np.abs(dt - 1/RATE).max():.2e}s")

    print("== 8b. 그리퍼 결측 프레임은 폐기한다 (SPEC D/M/T/X) ==")
    rawx = make_raw(n, gripper_x_at={5, 6})
    check("결측이 있어도 raw 스키마 위반 0", validate_raw(rawx) == [], str(validate_raw(rawx)))
    epx, repx = convert(rawx, StubIK(q_trajectory(n)), RANGES, GAP_CURVE, control_rate_hz=RATE)
    check("계약 위반 0", validate(epx) == [], str(validate(epx)))
    check("X 프레임이 폐기됨", repx.rejects.get("gripper_gap_missing") == 2, str(repx.rejects))
    check("최장 구간을 골랐다", repx.n_steps_out == n - 8, f"{repx.n_steps_out} (span {repx.kept_span})")
    bad_status = make_raw(n)
    bad_status.gripper_status[3] = "Z"
    check("알 수 없는 상태값은 거부", any("D/M/T/X" in m for m in validate_raw(bad_status)),
          str(validate_raw(bad_status))[:80])
    filled = make_raw(n, gripper_x_at={5})
    filled.gripper_gap_m[5] = 0.0          # X 인데 0 으로 채운 경우
    check("X 프레임을 0 으로 채우면 거부",
          any("채우지 않고 NaN" in m for m in validate_raw(filled)),
          str(validate_raw(filled))[:80])

    print("== 9. 거부해야 하는 것 ==")
    try:
        convert(make_raw(n, frame="arcore_world"), StubIK(q_trajectory(n)), RANGES, GAP_CURVE)
        check("frame != robot_base 거부", False, "예외가 안 났다")
    except Exception as exc:
        check("frame != robot_base 거부", "robot_base" in str(exc), type(exc).__name__)
    try:
        convert(make_raw(n, image_hw=(480, 640)), StubIK(q_trajectory(n)), RANGES, GAP_CURVE)
        check("224 아닌 이미지 + resize 없음 -> 거부", False, "예외가 안 났다")
    except ConversionError as exc:
        check("224 아닌 이미지 + resize 없음 -> 거부", "policy.resize" in str(exc))
    resized = ConversionPolicy(resize=lambda f, h, w: f[:h, :w])
    ep3, _ = convert(make_raw(n, image_hw=(480, 640)), StubIK(q_trajectory(n)), RANGES,
                     GAP_CURVE, policy=resized, control_rate_hz=RATE)
    check("resize 주입하면 통과", validate(ep3) == [], str(validate(ep3)))
    try:
        convert(make_raw(n), StubIK(q_trajectory(n), fail_at=set(range(10, 80))), RANGES, GAP_CURVE)
        check("최소 길이 미달 거부", False, "예외가 안 났다")
    except ConversionError as exc:
        check("최소 길이 미달 거부", "최소" in str(exc))

    print("== 10. q_init 연속성 ==")
    check("2번째 이후 q_init 이 전달됨", ik.q_init_seen[0] is None and ik.q_init_seen[1] is not None)
    # 팔 5축만 비교한다. 그리퍼는 IK 대상이 아니며 계약에서 물리 gap으로 분리된다.
    check("q_init 팔 5축이 직전 해와 같다",
          np.allclose(ik.q_init_seen[5][:5], ik.q_seq[4][:5], atol=1e-12))
    check("q_init 그리퍼 hinge는 사용하지 않음", ik.q_init_seen[5][5] == 0.0)

    print()
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
