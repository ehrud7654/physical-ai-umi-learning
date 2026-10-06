"""Check the five arm-angle and physical gripper-gap contract boundaries.
팔 5축 각도와 물리 gripper gap 계약 경계를 각각 검사한다.

Why this check exists.
왜 이 검사가 있는가.

The HW block-command runtime (ROBOT_COMMAND_API v1.0) reads joint limits from the
MJCF ``actuator ctrlrange``, converts them to degrees and rounds to three decimals
before handing them to the frontend. Our contract's source of truth is the
``joint range`` in ``configs/so101.yaml``, and MJCF rounds ``ctrlrange`` to five
digits — so the two differ slightly. If the frontend commands the maximum it was
given, the value can normalise to slightly more than 1.0, and
``contract.episode.validate`` rejects the whole episode for it.
HW 블록 명령 런타임(ROBOT_COMMAND_API v1.0)은 관절 한계를 MJCF ``actuator
ctrlrange`` 에서 읽어 degree 로 바꾸고 소수 3자리로 반올림해 프런트에 내려준다.
우리 계약의 정본은 ``configs/so101.yaml`` 의 ``joint range`` 이고, MJCF 의
``ctrlrange`` 는 5자리로 반올림돼 있어 둘이 미세하게 다르다. 프런트가 내려받은
최대값을 그대로 명령하면 정규화 결과가 1.0 을 아주 조금 넘을 수 있고, 그러면
``contract.episode.validate`` 가 그 에피소드를 통째로 거부한다.

This is not a hypothetical: one collected episode has already been rejected for
``state max 1.0062``. That one came from a different cause, but the failure mode
is the same and it is silent until an entire collection run is thrown away.
가정이 아니다. ``state max 1.0062`` 로 이미 한 건이 거부된 적이 있다. 그 건은
원인이 달랐지만 실패 양상은 같고, 수집분을 통째로 버리기 전까지 드러나지 않는다.

⚠️ 이 검사는 통과/실패만 말한다. 통과가 "규격이 일치한다"는 뜻은 아니다 —
   허용오차가 막아준 것일 수 있고, 로봇팔이 변형되면 다시 재야 한다.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contract.episode import GRIPPER_GAP_RANGE_M, RANGE_TOLERANCE, STATE_RANGE  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, build_model, load_config  # noqa: E402
from tracking.exp_log import file_digest, log_run  # noqa: E402

LIMIT_SOURCES = ("ctrlrange", "jntrange")


@dataclass
class AxisReport:
    """One joint's round trip: our range → their degrees → back to our units.
    관절 하나의 왕복: 우리 범위 → 그쪽 degree → 다시 우리 단위."""

    joint: str
    cfg_lo: float
    cfg_hi: float
    src_lo: float
    src_hi: float
    min_deg: float
    max_deg: float
    norm_min: float
    norm_max: float
    overshoot: float

    def verdict(self) -> str:
        """Human-readable pass/fail for this axis.
        이 축의 통과 여부를 사람이 읽을 수 있게."""
        return "통과" if self.overshoot < RANGE_TOLERANCE else "위반"


def _normalize(x_rad: float, lo: float, hi: float) -> float:
    """Arm contract normalisation. Must match `normalization.arm_formula`."""
    return 2.0 * (x_rad - lo) / (hi - lo) - 1.0


def limits_from_model(
    model: mujoco.MjModel, source: str = "ctrlrange"
) -> dict[str, tuple[float, float]]:
    """Joint limits as the runtime would read them out of the model.
    런타임이 모델에서 읽어갈 관절 한계.

    ``ctrlrange`` is what the HW runtime uses today; ``jntrange`` is what we would
    prefer, because it is the same number our config is generated from.
    ``ctrlrange`` 는 현재 HW 런타임이 쓰는 것이고, ``jntrange`` 는 우리 config 가
    생성된 값과 같아서 우리가 선호하는 것이다.
    """
    if source not in LIMIT_SOURCES:
        raise ValueError(f"source 는 {LIMIT_SOURCES} 중 하나여야 한다: {source!r}")

    out: dict[str, tuple[float, float]] = {}
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not name:
            raise RuntimeError(f"actuator {actuator_id} 에 이름이 없다")
        if source == "ctrlrange":
            lo, hi = model.actuator_ctrlrange[actuator_id]
        else:
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            lo, hi = model.jnt_range[joint_id]
        out[name] = (float(lo), float(hi))
    return out


def check(
    model: mujoco.MjModel,
    cfg: dict[str, Any],
    *,
    decimals: int = 3,
    source: str = "ctrlrange",
) -> list[AxisReport]:
    """Round every joint limit the way the API does, then normalise it back.
    API 가 하는 방식대로 관절 한계를 반올림한 뒤 다시 정규화한다."""
    limits = limits_from_model(model, source)
    reports: list[AxisReport] = []

    # Channel 6 is not the MuJoCo hinge in contract 0.3. Only J1~J5 cross the
    # degree-based arm API; physical gap is checked independently below.
    for spec in cfg["joints"][:5]:
        name = str(spec["name"])
        cfg_lo, cfg_hi = (float(v) for v in spec["range_rad"])
        if name not in limits:
            raise KeyError(f"모델에 액추에이터 {name!r} 이 없다. config 와 MJCF 가 어긋났다")
        src_lo, src_hi = limits[name]

        min_deg = round(math.degrees(src_lo), decimals)
        max_deg = round(math.degrees(src_hi), decimals)
        norm_min = _normalize(math.radians(min_deg), cfg_lo, cfg_hi)
        norm_max = _normalize(math.radians(max_deg), cfg_lo, cfg_hi)
        overshoot = max(
            abs(norm_min) - abs(STATE_RANGE[1]),
            abs(norm_max) - abs(STATE_RANGE[1]),
            0.0,
        )

        reports.append(
            AxisReport(
                joint=name,
                cfg_lo=cfg_lo,
                cfg_hi=cfg_hi,
                src_lo=src_lo,
                src_hi=src_hi,
                min_deg=min_deg,
                max_deg=max_deg,
                norm_min=norm_min,
                norm_max=norm_max,
                overshoot=overshoot,
            )
        )
    return reports


def worst_overshoot(reports: list[AxisReport]) -> float:
    """Largest excursion outside the contract range across all axes.
    전 축에서 계약 범위를 벗어난 최대치."""
    return max((r.overshoot for r in reports), default=0.0)


def gap_contract_samples() -> list[tuple[float, float]]:
    """Physical full gap metres and their contract-normalised values."""
    lo, hi = GRIPPER_GAP_RANGE_M
    return [(gap, 2.0 * (gap - lo) / (hi - lo) - 1.0)
            for gap in (lo, (lo + hi) / 2.0, hi)]


def format_table(reports: list[AxisReport]) -> str:
    """The table that goes into the MEASURE document verbatim.
    MEASURE 문서에 그대로 들어갈 표."""
    head = (
        f"{'joint':15s} {'config lo/hi (rad, 정본)':>30s} "
        f"{'API min/max_deg':>22s}  {'정규화 결과':>24s}  판정"
    )
    lines = [head, "-" * len(head)]
    for r in reports:
        lines.append(
            f"{r.joint:15s} [{r.cfg_lo:+.9f},{r.cfg_hi:+.9f}] "
            f"{r.min_deg:+9.3f} /{r.max_deg:+9.3f}   "
            f"{r.norm_min:+.7f} /{r.norm_max:+.7f}  {r.verdict()}"
        )
    return "\n".join(lines)


def main() -> int:
    """CLI. Exit code 1 means a degree command can invalidate an episode.
    CLI. 종료코드 1 은 degree 명령이 에피소드를 무효화할 수 있다는 뜻이다."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="API 가 degree 를 반올림하는 자릿수 (HW 런타임 기본값 3)",
    )
    parser.add_argument("--source", choices=LIMIT_SOURCES, default="ctrlrange")
    parser.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    model = build_model(cfg)
    reports = check(model, cfg, decimals=args.decimals, source=args.source)
    worst = worst_overshoot(reports)
    gap_samples = gap_contract_samples()
    expected_gap_norm = (-1.0, 0.0, 1.0)
    gap_passed = all(abs(actual[1] - expected) < RANGE_TOLERANCE
                     for actual, expected in zip(gap_samples, expected_gap_norm))
    passed = worst < RANGE_TOLERANCE and gap_passed

    print(format_table(reports))
    print()
    print("gripper_gap (full contact-surface width; MuJoCo hinge와 분리)")
    for gap_m, norm in gap_samples:
        print(f"  {gap_m:.3f} m -> {norm:+.1f}")
    print(f"  판정: {'통과' if gap_passed else '위반'}")
    print()
    print(
        f"[-1,1] 초과 최대치 {worst:.3e}  ·  계약 허용오차 {RANGE_TOLERANCE:.0e}  →  "
        f"{'통과' if passed else '위반 — 경계값 에피소드가 거부된다'}"
    )
    if passed:
        print(
            "⚠️ 통과는 '규격이 일치한다'가 아니라 '허용오차가 막아준다'는 뜻이다. "
            "허용오차를 조이거나 관절 범위가 바뀌면 다시 재야 한다."
        )

    if args.log:
        rec = log_run(
            experiment="angle_contract",
            author=args.author,
            issue="project-task",
            conditions={
                "config_sha": file_digest(DEFAULT_CONFIG),
                "limit_source": args.source,
                "degree_decimals": args.decimals,
                "range_tolerance": RANGE_TOLERANCE,
                "counterpart": "HW ROBOT_COMMAND_API v1.0",
            },
            result={
                "worst_overshoot": worst,
                "gap_samples": gap_samples,
                "gap_passed": gap_passed,
                "passed": passed,
                "axes": [asdict(r) for r in reports],
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']})")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
