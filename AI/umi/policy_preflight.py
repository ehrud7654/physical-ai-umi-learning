"""Hardware-free UMI policy preflight.

This boundary intentionally stops at joint targets.  It never imports a motor
driver and cannot write to the robot.  ``predict_action`` output is already in
physical units; normalization statistics are therefore neither accepted nor
applied here.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from umi.action_ik import actions_to_joints


@dataclass(frozen=True)
class PreflightResult:
    action_pred: np.ndarray
    joint_targets_rad: np.ndarray
    predict_ms: float
    convert_ms: float


def _to_numpy(value: Any) -> np.ndarray:
    """Convert numpy or a torch-like CPU/GPU tensor without importing torch."""
    for method in ("detach", "cpu"):
        fn = getattr(value, method, None)
        if callable(fn):
            value = fn()
    fn = getattr(value, "numpy", None)
    if callable(fn):
        value = fn()
    return np.asarray(value, dtype=np.float64)


def extract_action_pred(output: Any) -> np.ndarray:
    """Extract one-arm unnormalized ``(T, 10)`` output.

    A mapping must use the explicit ``action_pred`` key so an unrelated tensor
    cannot silently become a robot action.  A leading batch dimension of one is
    accepted because many policy APIs return ``(1, T, 10)``.
    """
    if isinstance(output, Mapping):
        if "action_pred" not in output:
            raise ValueError("policy mapping output has no 'action_pred' key")
        output = output["action_pred"]
    action = _to_numpy(output)
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] != 10:
        raise ValueError("policy action_pred must have shape (T, 10) or (1, T, 10)")
    if not np.isfinite(action).all():
        raise ValueError("policy action_pred contains NaN or infinity")
    return np.array(action, copy=True)


def run_policy_preflight(policy: Any, observation: Any, *, t_current: np.ndarray,
                         q_current: np.ndarray, solver: Any, ranges_rad: Any,
                         gap_curve: Any, max_step_rad: Any,
                         max_axis_deg: float = 5.0, max_roll_deg: float = 5.0,
                         clock: Callable[[], float] = perf_counter) -> PreflightResult:
    """Run exactly one inference and convert it to checked joint targets.

    The caller remains responsible for constructing the checkpoint-specific
    observation and proving its camera/frame/timestamp contract.  This function
    deliberately exposes no servo callback.
    """
    predict = getattr(policy, "predict_action", None)
    if not callable(predict):
        raise TypeError("policy must provide callable predict_action(observation)")
    t_current = np.array(t_current, dtype=float, copy=True)
    q_current = np.array(q_current, dtype=float, copy=True)
    start = clock()
    action = extract_action_pred(predict(observation))
    predicted = clock()
    joints = actions_to_joints(
        action, t_current, q_current, solver, ranges_rad, gap_curve,
        max_step_rad=max_step_rad, max_axis_deg=max_axis_deg,
        max_roll_deg=max_roll_deg,
    )
    converted = clock()
    return PreflightResult(
        action_pred=action,
        joint_targets_rad=joints,
        predict_ms=(predicted - start) * 1000.0,
        convert_ms=(converted - predicted) * 1000.0,
    )
