"""Offline decoded-target IK adapter. Does not authorize motor execution."""
import numpy as np
from umi.action_decode import decode_relative_actions
from umi.convert import invert_gap_curve
from umi.ik import IKSolver, matrix_to_quat


def actions_to_joints(action_pred, t_current, q_current, solver: IKSolver,
                      ranges_rad, gap_curve, *, max_step_rad, max_axis_deg=5.0,
                      max_roll_deg=5.0):
    """Return (T,6) radians or fail the whole chunk at its first invalid step.

    gap_curve is the calibrated [angle_rad, gap_cm] table. Previous accepted
    joint solutions seed IK, but all Cartesian targets share t_current.
    max_step_rad is a caller-specified six-channel jump bound, not a velocity
    or collision check. Timing/dynamic validation remains the caller's job.
    """
    targets = decode_relative_actions(action_pred, t_current)
    previous = np.array(q_current, dtype=float, copy=True)
    ranges = np.asarray(ranges_rad, dtype=float)
    bound = np.broadcast_to(np.asarray(max_step_rad, dtype=float), (6,))
    if (previous.shape != (6,) or ranges.shape != (6, 2)
            or not np.isfinite(previous).all() or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] >= ranges[:, 1])
            or not np.isfinite(bound).all() or np.any(bound <= 0)
            or not np.isfinite([max_axis_deg, max_roll_deg]).all()
            or min(max_axis_deg, max_roll_deg) <= 0):
        raise ValueError('invalid initial joints, ranges or error/jump budgets')
    if np.any(previous < ranges[:, 0]) or np.any(previous > ranges[:, 1]):
        raise ValueError('initial joints outside limits')
    result = []
    for i, (pose, gap) in enumerate(zip(targets.poses, targets.gripper_width_m)):
        grip = invert_gap_curve(float(gap), gap_curve)
        seed = previous.copy()
        seed[5] = grip
        sol = solver.solve(pose[:3, 3].copy(), matrix_to_quat(pose[:3, :3]),
                           float(gap), q_init=seed)
        errors = [sol.pos_error_m, sol.axis_error_deg, sol.roll_residual_deg]
        if (not sol.ok or not np.isfinite(errors).all() or min(errors) < 0
                or sol.axis_error_deg > max_axis_deg or sol.roll_residual_deg > max_roll_deg):
            raise ValueError(f'step {i}: IK rejected, residuals={errors}')
        q = np.array(sol.q_rad, dtype=float, copy=True)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError(f'step {i}: invalid IK joint output')
        q[5] = grip  # IK adapter does not implement gripper width conversion
        if np.any(q < ranges[:, 0]) or np.any(q > ranges[:, 1]):
            raise ValueError(f'step {i}: joint limit exceeded')
        if np.any(np.abs(q - previous) > bound):
            raise ValueError(f'step {i}: joint jump exceeds supplied budget')
        result.append(q)
        previous = q
    return np.stack(result)
