"""Fail-closed gripper phase guard for real UMI policy execution.

The policy predicts an absolute contact-surface gap.  A physical robot may
start wider than the demonstrations, so ``observed < predicted`` or
``predicted closes relative to observed`` is *not* a grasp-phase signal.

This guard keeps the dataset-specific open reference until a close intent has
been observed for a configured number of consecutive policy cycles.  All
dataset/task-dependent numbers come from a sidecar config; no grasp width is
hard-coded in the implementation.

This module only transforms gap commands.  It is not a substitute for IK,
collision, joint-limit, speed, acceleration, or command-publication gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


SCHEMA = "umi_gripper_phase_guard/0.1.0"
TASK_MODES = {"pick_hold", "policy_reopen"}


class GripperGuardConfigError(ValueError):
    """Raised when a guard contract is missing or unsafe."""


@dataclass(frozen=True)
class GripperGuardConfig:
    command_gap_min_m: float
    command_gap_max_m: float
    observation_gap_min_m: float
    observation_gap_max_m: float
    pregrasp_open_m: float
    close_threshold_m: float
    close_confirmation_cycles: int
    close_offset_m: float
    reopen_threshold_m: float
    task_mode: str

    @classmethod
    def from_mapping(cls, raw: dict) -> "GripperGuardConfig":
        if raw.get("schema") != SCHEMA:
            raise GripperGuardConfigError(
                f"gripper guard schema must be {SCHEMA!r}, got {raw.get('schema')!r}"
            )
        required = {
            "command_gap_range_m",
            "observation_gap_range_m",
            "pregrasp_open_m",
            "close_threshold_m",
            "close_confirmation_cycles",
            "close_offset_m",
            "reopen_threshold_m",
            "task_mode",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise GripperGuardConfigError(f"missing gripper guard keys: {missing}")

        def pair(name: str) -> tuple[float, float]:
            value = raw[name]
            if not isinstance(value, list) or len(value) != 2:
                raise GripperGuardConfigError(f"{name} must be [min, max]")
            lo, hi = map(float, value)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise GripperGuardConfigError(f"invalid {name}: {value!r}")
            return lo, hi

        command_lo, command_hi = pair("command_gap_range_m")
        observation_lo, observation_hi = pair("observation_gap_range_m")
        numeric = {
            key: float(raw[key])
            for key in (
                "pregrasp_open_m",
                "close_threshold_m",
                "close_offset_m",
                "reopen_threshold_m",
            )
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise GripperGuardConfigError("gripper guard values must be finite")
        confirmations = raw["close_confirmation_cycles"]
        if isinstance(confirmations, bool) or int(confirmations) != confirmations:
            raise GripperGuardConfigError("close_confirmation_cycles must be an integer")
        confirmations = int(confirmations)
        mode = str(raw["task_mode"])
        if mode not in TASK_MODES:
            raise GripperGuardConfigError(
                f"task_mode must be one of {sorted(TASK_MODES)}, got {mode!r}"
            )
        if confirmations < 1:
            raise GripperGuardConfigError("close_confirmation_cycles must be >= 1")
        if not command_lo <= numeric["close_threshold_m"] < numeric["pregrasp_open_m"] <= command_hi:
            raise GripperGuardConfigError(
                "require command_min <= close_threshold < pregrasp_open <= command_max"
            )
        if not numeric["close_threshold_m"] < numeric["reopen_threshold_m"] <= command_hi:
            raise GripperGuardConfigError(
                "require close_threshold < reopen_threshold <= command_max"
            )
        if not 0 <= numeric["close_offset_m"] < numeric["close_threshold_m"] - command_lo:
            raise GripperGuardConfigError("close_offset_m is outside the safe command range")
        if observation_lo > command_lo or observation_hi < command_hi:
            raise GripperGuardConfigError(
                "observation range must contain the command range"
            )
        return cls(
            command_gap_min_m=command_lo,
            command_gap_max_m=command_hi,
            observation_gap_min_m=observation_lo,
            observation_gap_max_m=observation_hi,
            close_confirmation_cycles=confirmations,
            task_mode=mode,
            **numeric,
        )


@dataclass(frozen=True)
class GripperGuardState:
    latched: bool = False
    close_votes: int = 0
    held_gap_m: float | None = None


@dataclass(frozen=True)
class GripperGuardResult:
    commanded_gap_m: tuple[float, ...]
    state: GripperGuardState
    report: dict


def load_gripper_guard_config(path: Path) -> GripperGuardConfig:
    path = Path(path)
    if not path.is_file():
        raise GripperGuardConfigError(f"gripper guard config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise GripperGuardConfigError("gripper guard config root must be an object")
    return GripperGuardConfig.from_mapping(raw)


def _finite_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} is empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def apply_gripper_phase_guard(
    predicted_gap_m: Iterable[float],
    observed_gap_m: float,
    state: GripperGuardState,
    config: GripperGuardConfig,
) -> GripperGuardResult:
    """Return an all-or-nothing guarded gap chunk and the next phase state.

    Before close confirmation, commands are held at ``pregrasp_open_m`` even
    if the policy starts closing.  A close vote requires *every executable
    point* in the chunk to be at or below ``close_threshold_m``.  This avoids
    using a far-horizon final point as proof that the gripper has arrived.

    ``pick_hold`` is monotonic after latching and is appropriate only for a
    pick/lift task.  ``policy_reopen`` permits an explicit reopen once every
    executable point is at or above ``reopen_threshold_m``.
    """
    predicted = _finite_tuple(predicted_gap_m, "predicted_gap_m")
    observed = float(observed_gap_m)
    if not math.isfinite(observed):
        raise ValueError("observed_gap_m is NaN/Inf")
    if not config.observation_gap_min_m <= observed <= config.observation_gap_max_m:
        raise ValueError(
            f"observed gap {observed:.6f} outside configured observation range "
            f"[{config.observation_gap_min_m:.6f}, {config.observation_gap_max_m:.6f}]"
        )
    for index, gap in enumerate(predicted):
        if not config.command_gap_min_m <= gap <= config.command_gap_max_m:
            raise ValueError(
                f"predicted gap[{index}]={gap:.6f} outside command range "
                f"[{config.command_gap_min_m:.6f}, {config.command_gap_max_m:.6f}]"
            )

    close_candidate = all(gap <= config.close_threshold_m for gap in predicted)
    reopen_candidate = all(gap >= config.reopen_threshold_m for gap in predicted)
    latched = state.latched
    close_votes = state.close_votes
    held = state.held_gap_m
    event = None

    if latched and config.task_mode == "policy_reopen" and reopen_candidate:
        latched, close_votes, held = False, 0, None
        event = "reopened"

    if not latched:
        close_votes = close_votes + 1 if close_candidate else 0
        if close_votes >= config.close_confirmation_cycles:
            latched = True
            held = config.pregrasp_open_m
            event = "close_confirmed"

    if not latched:
        commanded = tuple(max(gap, config.pregrasp_open_m) for gap in predicted)
        phase = "pregrasp_hold"
    elif config.task_mode == "pick_hold":
        values = []
        held = config.pregrasp_open_m if held is None else held
        for gap in predicted:
            requested = max(config.command_gap_min_m, gap - config.close_offset_m)
            held = min(held, requested)
            values.append(held)
        commanded = tuple(values)
        phase = "grasp_hold"
    else:
        commanded = tuple(
            max(config.command_gap_min_m, gap - config.close_offset_m)
            for gap in predicted
        )
        held = commanded[-1]
        phase = "grasp_policy"

    next_state = GripperGuardState(
        latched=latched,
        close_votes=close_votes,
        held_gap_m=held,
    )
    report = {
        "schema": SCHEMA,
        "phase": phase,
        "event": event,
        "task_mode": config.task_mode,
        "observed_gap_m": observed,
        "raw_predicted_gap_m": list(predicted),
        "commanded_gap_m": list(commanded),
        "close_candidate": close_candidate,
        "reopen_candidate": reopen_candidate,
        "close_votes": close_votes,
        "close_confirmation_cycles": config.close_confirmation_cycles,
        "latched": latched,
    }
    return GripperGuardResult(commanded, next_state, report)
