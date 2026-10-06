"""Shared real-execution command limit checks.
실물 실행 계층 명령 한계 검사의 공용 구현.

**이 파일이 판정 로직의 유일한 구현이다.** 두 벌이 되면 어느 쪽으로 잰 수치인지
사후에 가릴 수 없다 — 2026-09-10 에 같은 계열 사고를 겪었다. 새 도구는 여기서
import 하고 복제하지 않는다.

두 실행 경로는 **검사식이 다르다.** 하나의 "실물 한계" 수치로 합치지 않는다.

| 경로 | 검사 | 확신도 |
|---|---|---|
| ROS2 `SO101System` | 비유한값 · URDF 위치 한계 · encoder tick 범위 · gripper width | 🟢 원문 |
| direct-motor waypoint | `0.05<=dt<=60` · quintic 피크 가속 | 🔵 전사 |

ROS2 판정의 범위 한계: `JointTrajectoryController` 는 30Hz 로 **보간점**을 내고
하드웨어 인터페이스는 그 보간점을 검사한다. 여기서는 정책이 만든 **샘플 명령점**만
검사하므로 보간 overshoot 는 재현하지 않는다. 따라서 이 결과는 **하한**이다 —
통과해도 보간점이 걸릴 수 있다.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# --- direct-motor 전사 상수 -------------------------------------------------
# 출처: `so101_direct_motor_pipeline` 의 `core/models.py` 🔵 (원문 미보유).
# `tools/umi_rate_budget.py` 헤더가 같은 출처를 기록한다.
DIRECT_DT_MIN_S = 0.05
DIRECT_DT_MAX_S = 60.0
DIRECT_ACCEL_FACTOR = 10.0 / math.sqrt(3.0)   # 5.7735 — 정지→정지 quintic 피크 가속
# ⚠️ 아래는 전사본에 **없는** 값이다. quintic 피크 속도 계수 15/8 로 수학적으로는
#    맞지만 `core/models.py` 에 그 검사가 실제로 있는지는 모른다 → 🟡.
#    결과에서 별도 키로 분리하고 가속 검사와 같은 확신도로 읽지 않는다.
DIRECT_SPEED_FACTOR_UNVERIFIED = 15.0 / 8.0

TICKS_PER_TURN = 4096.0


@dataclass(frozen=True)
class ArmBackendLimit:
    """One ROS2 arm joint calibration record.
    ROS2 팔 관절 하나의 캘리브레이션 레코드."""

    name: str
    zero_tick: int
    direction: int
    min_tick: int
    max_tick: int
    min_rad: float
    max_rad: float


@dataclass(frozen=True)
class GripperBackendLimit:
    """ROS2 gripper calibration record.
    ROS2 그리퍼 캘리브레이션 레코드."""

    closed_tick: int
    open_tick: int
    min_tick: int
    max_tick: int
    max_width_m: float


@dataclass(frozen=True)
class Ros2Limits:
    """Limits parsed from the deployed ros2_control description.
    배포된 ros2_control 기술에서 읽은 한계."""

    arm: tuple[ArmBackendLimit, ...]
    gripper: GripperBackendLimit

    @property
    def arm_names(self) -> list[str]:
        return [joint.name for joint in self.arm]


ARM_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def load_ros2_limits(path: Path) -> Ros2Limits:
    """Parse the embedded ros2_control block from a URDF or xacro.
    URDF 또는 xacro 의 ros2_control 블록을 파싱한다.

    값을 여기에 적지 않는다. 하드웨어 정본이 바뀌면 사본 교체만으로 반영되어야 한다.
    """
    control = ET.parse(path).getroot().find(".//ros2_control")
    if control is None:
        raise ValueError(f"{path}: ros2_control 블록이 없다")

    parsed: dict[str, dict[str, str]] = {}
    for joint in control.findall("joint"):
        name = joint.attrib.get("name")
        if name:
            parsed[name] = {
                p.attrib["name"]: (p.text or "").strip()
                for p in joint.findall("param")
            }

    arm: list[ArmBackendLimit] = []
    for name in ARM_ORDER:
        if name not in parsed:
            raise ValueError(f"{path}: ros2_control 에 {name} 이 없다")
        p = parsed[name]
        arm.append(
            ArmBackendLimit(
                name=name,
                zero_tick=int(p["zero_tick"]),
                direction=int(p["direction"]),
                min_tick=int(p["min_tick"]),
                max_tick=int(p["max_tick"]),
                min_rad=float(p["min_rad"]),
                max_rad=float(p["max_rad"]),
            )
        )

    if "gripper" not in parsed:
        raise ValueError(f"{path}: ros2_control 에 gripper 가 없다")
    g = parsed["gripper"]
    gripper = GripperBackendLimit(
        closed_tick=int(g["closed_tick"]),
        open_tick=int(g["open_tick"]),
        min_tick=int(g["min_tick"]),
        max_tick=int(g["max_tick"]),
        max_width_m=float(g["max_width_m"]),
    )
    return Ros2Limits(tuple(arm), gripper)


def load_real_config(path: Path) -> dict[str, Any]:
    """Load the tracked direct-motor real configuration.
    추적 중인 direct-motor 실물 설정을 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))


def denorm(x_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Contract [-1,1] to physical units. Inverse of the contract formula.
    계약 [-1,1] 을 물리 단위로. 계약 공식의 역변환."""
    return (np.asarray(x_norm, dtype=np.float64) + 1.0) / 2.0 * (hi - lo) + lo


def norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Physical units to contract [-1,1].
    물리 단위를 계약 [-1,1] 로."""
    return 2.0 * (np.asarray(x, dtype=np.float64) - lo) / (hi - lo) - 1.0


def gripper_rad_to_gap_m(rad: np.ndarray, curve: list[list[float]]) -> np.ndarray:
    """Simulated hinge angle to pad gap in metres, via the measured curve.
    시뮬 hinge 각도를 실측 곡선으로 패드 간격(m)으로 바꾼다.

    `configs/so101.yaml` 의 `grasp.gap_curve` 는 `[rad, cm]` 이다 — 여기서 m 로 바꾼다.
    실물 슬라이드 변환이 아니다. 실물은 백엔드가 미터를 직접 받는다.
    """
    xs = np.asarray([p[0] for p in curve], dtype=np.float64)
    ys = np.asarray([p[1] for p in curve], dtype=np.float64) / 100.0
    order = np.argsort(xs)
    return np.interp(np.asarray(rad, dtype=np.float64), xs[order], ys[order])


class Tally:
    """Violation counts and limit-ratio distributions per constraint.
    제약별 위반 수와 한계 대비 배수 분포."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def add(self, key: str, ratio: np.ndarray, bad: np.ndarray) -> None:
        r = np.asarray(ratio, dtype=np.float64).reshape(-1)
        b = np.asarray(bad, dtype=bool).reshape(-1)
        if r.shape != b.shape:
            raise ValueError(f"{key}: ratio {r.shape} != bad {b.shape}")
        row = self._rows.setdefault(
            key, {"total": 0, "bad": 0, "ratios": [], "bad_ratios": []}
        )
        finite = np.isfinite(r)
        row["total"] += int(r.size)
        row["bad"] += int(b.sum())
        row["ratios"].extend(r[finite].tolist())
        row["bad_ratios"].extend(r[finite & b].tolist())

    @staticmethod
    def _dist(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p50": None, "p95": None, "p99": None, "max": None}
        a = np.asarray(values, dtype=np.float64)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    def result(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, row in sorted(self._rows.items()):
            total = int(row["total"])
            out[key] = {
                "violations": int(row["bad"]),
                "total": total,
                "violation_rate": (row["bad"] / total) if total else None,
                "ratio_all": self._dist(row["ratios"]),
                "ratio_violations": self._dist(row["bad_ratios"]),
            }
        return out


def _span_ratio(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Position as a multiple of the limit on its own side.
    위치를 같은 부호 쪽 한계 대비 배수로."""
    a = np.asarray(values, dtype=np.float64)
    pos = abs(upper) if upper != 0.0 else np.nan
    neg = abs(lower) if lower != 0.0 else np.nan
    return np.where(a >= 0.0, np.abs(a) / pos, np.abs(a) / neg)


def check_ros2(
    arm_rad: np.ndarray, gap_m: np.ndarray, limits: Ros2Limits
) -> dict[str, Any]:
    """Reproduce `SO101System` rejection on a sampled command stream.
    샘플 명령열에 `SO101System` 거부 조건을 재현한다.

    원문 `src/so101_system.cpp` 🟢:
      joint_to_tick   : finite -> min_rad<=r<=max_rad -> min_tick<=raw<=max_tick
      gripper_to_tick : finite -> 0<=w<=max_width_m   -> min_tick<=raw<=max_tick
      raw_arm     = zero_tick + direction * rad * TICKS_PER_TURN/(2*pi)
      raw_gripper = closed_tick + (open_tick-closed_tick) * w / max_width_m
    한 프레임이라도 걸리면 그 쓰기 주기 전체가 거부된다 (wrap·clamp 없음).
    """
    arm = np.asarray(arm_rad, dtype=np.float64)
    gap = np.asarray(gap_m, dtype=np.float64).reshape(-1)
    if arm.ndim != 2 or arm.shape[1] != len(limits.arm):
        raise ValueError(f"arm_rad shape {arm.shape}")
    if arm.shape[0] != gap.shape[0]:
        raise ValueError("팔과 그리퍼 명령 길이가 다르다")

    t = Tally()
    for i, j in enumerate(limits.arm):
        v = arm[:, i]
        finite = np.isfinite(v)
        t.add(f"{j.name}.finite", np.where(finite, 0.0, np.inf), ~finite)

        t.add(
            f"{j.name}.position",
            _span_ratio(v, j.min_rad, j.max_rad),
            finite & ((v < j.min_rad) | (v > j.max_rad)),
        )
        # 하한·상한을 나눠 센다. 어느 쪽이 지배하는지가 처방을 정한다.
        t.add(
            f"{j.name}.position_lower",
            np.where(v < 0.0, np.abs(v) / abs(j.min_rad), 0.0),
            finite & (v < j.min_rad),
        )
        t.add(
            f"{j.name}.position_upper",
            np.where(v > 0.0, np.abs(v) / abs(j.max_rad), 0.0),
            finite & (v > j.max_rad),
        )

        raw = j.zero_tick + j.direction * v * TICKS_PER_TURN / (2.0 * math.pi)
        up = max(1.0, float(j.max_tick - j.zero_tick))
        lo = max(1.0, float(j.zero_tick - j.min_tick))
        t.add(
            f"{j.name}.encoder_tick",
            np.where(
                raw >= j.zero_tick,
                np.abs(raw - j.zero_tick) / up,
                np.abs(raw - j.zero_tick) / lo,
            ),
            finite & ((raw < j.min_tick) | (raw > j.max_tick)),
        )

    g = limits.gripper
    finite_g = np.isfinite(gap)
    t.add("gripper.finite", np.where(finite_g, 0.0, np.inf), ~finite_g)
    t.add(
        "gripper.width",
        np.abs(gap) / g.max_width_m,
        finite_g & ((gap < 0.0) | (gap > g.max_width_m)),
    )
    raw_g = g.closed_tick + (g.open_tick - g.closed_tick) * gap / g.max_width_m
    span = max(1.0, float(abs(g.open_tick - g.closed_tick)))
    t.add(
        "gripper.encoder_tick",
        np.abs(raw_g - g.closed_tick) / span,
        finite_g & ((raw_g < g.min_tick) | (raw_g > g.max_tick)),
    )

    metrics = t.result()
    bad = sum(r["violations"] for r in metrics.values())
    return {
        "path": "ros2",
        "confidence": "green_source",
        "passes": bad == 0,
        "n_points": int(arm.shape[0]),
        "metrics": metrics,
        "scope_note": (
            "sampled policy commands only; JointTrajectoryController "
            "interpolation not emulated -> this is a lower bound"
        ),
    }


def check_direct(
    arm_rad: np.ndarray,
    gap_m: np.ndarray,
    dt: float,
    joint_names: list[str],
    limits_rad: dict[str, list[float]],
    max_speed_rad_s: float,
    max_accel_rad_s2: float,
    max_gap_m: float,
    max_gap_speed_m_s: float,
    max_gap_accel_m_s2: float,
) -> dict[str, Any]:
    """Transcribed direct-motor waypoint check.
    전사된 direct-motor waypoint 검사.

    가속 계수와 dt 범위는 🔵 전사, 속도 계수는 🟡 미검증이므로 키를 분리한다.
    """
    arm = np.asarray(arm_rad, dtype=np.float64)
    gap = np.asarray(gap_m, dtype=np.float64).reshape(-1)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    n = max(0, arm.shape[0] - 1)

    t = Tally()
    t.add(
        "waypoint.duration",
        np.full(n, DIRECT_DT_MIN_S / dt, dtype=np.float64),
        np.full(n, not (DIRECT_DT_MIN_S <= dt <= DIRECT_DT_MAX_S), dtype=bool),
    )

    d = np.diff(arm, axis=0)
    for i, name in enumerate(joint_names):
        v = arm[:, i]
        lo, hi = (float(x) for x in limits_rad[name])
        finite = np.isfinite(v)
        t.add(
            f"{name}.position",
            _span_ratio(v, lo, hi),
            finite & ((v < lo) | (v > hi)),
        )
        accel = DIRECT_ACCEL_FACTOR * np.abs(d[:, i]) / (dt * dt) / max_accel_rad_s2
        t.add(f"{name}.peak_accel", accel, accel > 1.0)
        speed = (
            DIRECT_SPEED_FACTOR_UNVERIFIED
            * np.abs(d[:, i])
            / dt
            / max_speed_rad_s
        )
        t.add(f"{name}.peak_speed_UNVERIFIED", speed, speed > 1.0)

    t.add(
        "gripper.width",
        np.abs(gap) / max_gap_m,
        np.isfinite(gap) & ((gap < 0.0) | (gap > max_gap_m)),
    )
    gv = np.diff(gap) / dt
    t.add("gripper.speed", np.abs(gv) / max_gap_speed_m_s, np.abs(gv) > max_gap_speed_m_s)
    ga = np.diff(gv) / dt
    t.add("gripper.accel", np.abs(ga) / max_gap_accel_m_s2, np.abs(ga) > max_gap_accel_m_s2)

    metrics = t.result()
    # 미검증 속도 검사는 통과 판정에서 뺀다. 없는 검사로 불통과를 부풀리지 않는다.
    bad = sum(
        r["violations"] for k, r in metrics.items() if not k.endswith("_UNVERIFIED")
    )
    return {
        "path": "direct_motor_transcribed",
        "confidence": "blue_transcribed",
        "passes": bad == 0,
        "dt_s": dt,
        "n_points": int(arm.shape[0]),
        "metrics": metrics,
        "scope_note": (
            "constants transcribed from core/models.py; source unavailable. "
            "*_UNVERIFIED keys are excluded from the pass verdict"
        ),
    }


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode reports into one summary.
    에피소드별 결과를 하나로 합산한다."""
    if not reports:
        raise ValueError("빈 결과")
    merged: dict[str, dict[str, Any]] = {}
    for rep in reports:
        for key, row in rep["metrics"].items():
            tgt = merged.setdefault(
                key, {"violations": 0, "total": 0, "max_ratio": None, "p99": []}
            )
            tgt["violations"] += int(row["violations"])
            tgt["total"] += int(row["total"])
            m = row["ratio_all"]["max"]
            if m is not None:
                tgt["max_ratio"] = m if tgt["max_ratio"] is None else max(tgt["max_ratio"], m)
            q = row["ratio_all"]["p99"]
            if q is not None:
                tgt["p99"].append(float(q))

    metrics = {
        key: {
            "violations": row["violations"],
            "total": row["total"],
            "violation_rate": (row["violations"] / row["total"]) if row["total"] else None,
            "max_ratio": row["max_ratio"],
            "p99_of_episode_p99": (
                float(np.percentile(row["p99"], 99)) if row["p99"] else None
            ),
        }
        for key, row in sorted(merged.items())
    }
    passed = sum(bool(r["passes"]) for r in reports)
    return {
        "episodes": len(reports),
        "episodes_passed": passed,
        "episode_pass_rate": passed / len(reports),
        "metrics": metrics,
    }


def tightest_margins(
    summary: dict[str, Any], top: int = 6
) -> list[tuple[str, float]]:
    """Constraints ranked by how close they came to the limit.
    한계에 얼마나 가까웠는지 순위.

    위반이 0 이어도 이 값은 남는다. **통과했을 때 알고 싶은 숫자가 이것이다** —
    0.98x 로 통과한 것과 0.05x 로 통과한 것은 전혀 다른 상태이고, 전자는 다음
    체크포인트에서 깨진다.
    """
    rows = [
        (k, r["max_ratio"])
        for k, r in summary["metrics"].items()
        if r.get("max_ratio") is not None
    ]
    return sorted(rows, key=lambda kv: kv[1], reverse=True)[:top]


def dominant_constraints(summary: dict[str, Any], top: int = 5) -> list[tuple[str, float]]:
    """Constraints ranked by violation rate. What to fix first.
    위반율 순 제약 순위. 무엇부터 고칠지."""
    rows = [
        (k, r["violation_rate"])
        for k, r in summary["metrics"].items()
        if r["violation_rate"]
    ]
    return sorted(rows, key=lambda kv: kv[1], reverse=True)[:top]
