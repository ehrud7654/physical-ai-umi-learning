"""Decode unnormalized predict_action output using the HW reply of 2026-09-09.

Columns are position in metres, first two rotation rows, gripper width in metres.
Every target is relative to the SAME observation-time EEF pose. This module
performs no normalization, IK, motor writes or action scheduling.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecodedTargets:
    """Targets in the reference frame of t_current (robot base if supplied there)."""

    poses: np.ndarray  # (T, 4, 4), same TCP definition as t_current
    gripper_width_m: np.ndarray  # (T,), absolute width, not a relative delta


def rotation_6d_rows_to_matrix(values: np.ndarray) -> np.ndarray:
    """Gram-Schmidt decode (..., 6), stacking orthonormal vectors as ROWS.

    Degenerate/nonfinite predictions are rejected rather than replaced with
    an arbitrary orientation. The threshold applies to each input vector and
    to their relative angular separation.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != 6 or not np.isfinite(a).all():
        raise ValueError("rotation_6d must be finite with shape (..., 6)")
    first, second = a[..., :3], a[..., 3:]
    n1 = np.linalg.norm(first, axis=-1, keepdims=True)
    n2 = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(n1 < 1e-8) or np.any(n2 < 1e-8):
        raise ValueError("rotation_6d contains a zero or near-zero row")
    b1 = first / n1
    unit_second = second / n2
    orthogonal = unit_second - np.sum(b1 * unit_second, axis=-1, keepdims=True) * b1
    norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("rotation_6d rows are parallel or nearly parallel")
    b2 = orthogonal / norm
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def decode_relative_actions(action_pred: np.ndarray, t_current: np.ndarray) -> DecodedTargets:
    """Restore (T,10) physical-unit outputs with current @ relative[i].

    t_current must be the EEF pose used as the model's relative observation
    reference, not a newly sampled pose after inference. Caller owns frame,
    TCP and timestamp consistency; a numeric matrix cannot establish these.
    Negative widths are invalid. Hardware-specific maximum width, reachability
    and dynamic limits must be checked downstream before execution.
    """
    actions = np.asarray(action_pred, dtype=np.float64)
    current = np.array(t_current, dtype=np.float64, copy=True)
    if actions.ndim != 2 or actions.shape[1] != 10 or actions.shape[0] == 0:
        raise ValueError("action_pred must be a nonempty (T, 10) array")
    if not np.isfinite(actions).all():
        raise ValueError("action_pred contains NaN or infinity")
    if np.any(actions[:, 9] < 0):
        raise ValueError("gripper_width_m must be nonnegative")
    if current.shape != (4, 4) or not np.isfinite(current).all():
        raise ValueError("t_current must be a finite (4, 4) transform")
    if not np.allclose(current[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("t_current has an invalid homogeneous bottom row")
    rotation = current[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("t_current rotation must be orthonormal with determinant +1")
    relative = np.broadcast_to(np.eye(4), (len(actions), 4, 4)).copy()
    relative[:, :3, :3] = rotation_6d_rows_to_matrix(actions[:, 3:9])
    relative[:, :3, 3] = actions[:, :3]
    return DecodedTargets(current @ relative, actions[:, 9].copy())
