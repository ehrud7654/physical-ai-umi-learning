"""Offline finite-difference gates; budgets are caller supplied, not HW claims."""
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class TrajectoryLimits:
    # Five arm channels in radians followed by full gripper width in metres.
    max_step: tuple
    max_velocity: tuple
    max_acceleration: tuple

    def arrays(self):
        arrays = [np.asarray(x, dtype=float) for x in
                  (self.max_step, self.max_velocity, self.max_acceleration)]
        if any(x.shape != (6,) or not np.isfinite(x).all() or np.any(x <= 0)
               for x in arrays):
            raise ValueError('each trajectory budget must contain six positive finite values')
        return arrays


def continuous_mask(arm, gap, timestamps, valid, limits):
    step, speed, accel = limits.arrays()
    q = np.column_stack((arm, gap))
    t = np.asarray(timestamps, dtype=float)
    keep = np.array(valid, dtype=bool, copy=True)
    if (q.shape != (len(t), 6) or keep.shape != t.shape
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('invalid trajectory shape or timestamps')
    counts = Counter()
    previous_velocity = None
    previous_dt = None
    for i in range(len(t)):
        if not keep[i]:
            previous_velocity = None
            continue
        reason = None
        if not np.isfinite(q[i]).all():
            reason = 'trajectory_nonfinite'
        elif i and keep[i-1]:
            dt = t[i] - t[i-1]
            delta = q[i] - q[i-1]
            velocity = delta / dt
            if np.any(abs(delta) > step):
                reason = 'trajectory_step'
            elif np.any(abs(velocity) > speed):
                reason = 'trajectory_velocity'
            elif previous_velocity is not None and np.any(
                    abs(velocity - previous_velocity) / ((dt + previous_dt) / 2) > accel):
                reason = 'trajectory_acceleration'
            previous_velocity, previous_dt = velocity, dt
        else:
            previous_velocity = None
        if reason:
            keep[i] = False
            counts[reason] += 1
            previous_velocity = None
    return keep, counts
