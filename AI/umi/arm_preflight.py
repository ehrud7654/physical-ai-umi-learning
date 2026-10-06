"""Offline real-robot command preparation: five radians plus gap metres.

The injected arm solver must use the same zero, signs, frame and TCP as the
state snapshot. It returns five joints; existing six-axis simulator solvers
require an explicit adapter. No motor driver or encoder mapping lives here.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from umi.action_decode import decode_relative_actions
from umi.policy_preflight import extract_action_pred


@dataclass(frozen=True)
class ArmSolution:
    positions_rad: np.ndarray
    position_error_m: float
    axis_error_deg: float
    roll_error_deg: float
    converged: bool
    within_limits: bool = True


class ArmSolver(Protocol):
    def solve(self, target_pose: np.ndarray, *, seed_rad: np.ndarray,
              gripper_width_m: float) -> ArmSolution: ...


@dataclass(frozen=True)
class ArmCommand:
    arm_positions_rad: np.ndarray
    gripper_width_m: float


def prepare_arm_commands(policy, observation, *, t_current, arm_current_rad,
                         gripper_current_m, solver: ArmSolver, ranges_rad,
                         gap_range_m, max_step_rad, max_gap_step_m,
                         max_position_error_m, max_axis_error_deg,
                         max_roll_error_deg):
    """Snapshot, infer once and validate the entire chunk before returning it.

    Limits are explicitly supplied, never assumed from the proposed dataset
    range. This checks jumps, not velocity or collision. Caller must acquire a
    synchronized observation/state snapshot and handle freshness and scheduling.
    """
    anchor = np.array(t_current, dtype=float, copy=True)
    previous = np.array(arm_current_rad, dtype=float, copy=True)
    limits = np.array(ranges_rad, dtype=float, copy=True)
    gap_limits = np.array(gap_range_m, dtype=float, copy=True)
    step = np.broadcast_to(np.asarray(max_step_rad, dtype=float), (5,)).copy()
    budgets = np.array([max_position_error_m, max_axis_error_deg,
                        max_roll_error_deg], dtype=float)
    gap = float(gripper_current_m)
    if (previous.shape != (5,) or limits.shape != (5, 2)
            or gap_limits.shape != (2,) or not np.isfinite(previous).all()
            or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1])
            or not np.isfinite(gap_limits).all() or gap_limits[0] < 0
            or gap_limits[0] >= gap_limits[1]
            or not np.isfinite(step).all() or np.any(step <= 0)
            or not np.isfinite(budgets).all() or np.any(budgets <= 0)
            or not np.isfinite(max_gap_step_m) or max_gap_step_m <= 0
            or not np.isfinite(gap) or not gap_limits[0] <= gap <= gap_limits[1]
            or np.any(previous < limits[:, 0]) or np.any(previous > limits[:, 1])):
        raise ValueError('invalid state or command limits')
    # Validate anchor before invoking an expensive policy.
    identity = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, gap]])
    decode_relative_actions(identity, anchor)
    output = policy.predict_action(deepcopy(observation))
    targets = decode_relative_actions(extract_action_pred(output), anchor)
    commands = []
    for i, (pose, width) in enumerate(zip(targets.poses, targets.gripper_width_m)):
        if not gap_limits[0] <= width <= gap_limits[1] or abs(width - gap) > max_gap_step_m:
            raise ValueError(f'step {i}: gripper range or jump exceeded')
        solution = solver.solve(pose.copy(), seed_rad=previous.copy(),
                                gripper_width_m=float(width))
        joints = np.array(solution.positions_rad, dtype=float, copy=True)
        errors = np.array([solution.position_error_m, solution.axis_error_deg,
                           solution.roll_error_deg], dtype=float)
        if (not solution.converged or not solution.within_limits
                or joints.shape != (5,)
                or not np.isfinite(joints).all() or not np.isfinite(errors).all()
                or np.any(errors < 0) or np.any(errors > budgets)):
            raise ValueError(
                f'step {i}: invalid arm IK solution '
                f'(converged={solution.converged}, '
                f'within_limits={solution.within_limits}, '
                f'position_error_m={solution.position_error_m:.6g}/'
                f'{max_position_error_m:.6g}, '
                f'axis_error_deg={solution.axis_error_deg:.6g}/'
                f'{max_axis_error_deg:.6g}, '
                f'roll_error_deg={solution.roll_error_deg:.6g}/'
                f'{max_roll_error_deg:.6g})'
            )
        if (np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1])
                or np.any(abs(joints - previous) > step)):
            raise ValueError(f'step {i}: arm range or jump exceeded')
        commands.append(ArmCommand(joints, float(width)))
        previous, gap = joints.copy(), float(width)
    return tuple(commands)
