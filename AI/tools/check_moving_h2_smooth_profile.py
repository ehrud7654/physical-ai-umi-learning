"""Focused, MuJoCo-free checks for moving-H=2 quintic command fractions."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.probe_moving_history_object_only import _command_fraction
from tools.render_relative_chunk_rollout import (
    _PostGraspGapRetentionPolicy,
    _smooth_history_row,
)


class _FixturePolicy:
    def predict_action(self, observation):
        del observation
        action = np.zeros((3, 10), dtype=np.float32)
        action[:, -1] = [0.050, 0.036, 0.048]
        return {"action_pred": action}


def main() -> int:
    assert _command_fraction("step", 0.01, 0.1) == 1.0
    values = [_command_fraction("quintic", value, 0.4)
              for value in (0.0, 0.1, 0.2, 0.4, 0.5)]
    assert np.allclose(values, [0.0, 0.103515625, 0.5, 1.0, 1.0])
    assert values == sorted(values)
    for profile, elapsed, duration in (
            ("bad", 0.1, 0.4), ("quintic", -0.1, 0.4),
            ("quintic", 0.1, 0.09)):
        try:
            _command_fraction(profile, elapsed, duration)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid smooth-profile input was accepted")
    valid_row = {
        "episode": "fixture", "duration_s": 0.3,
        "source_row": 4, "current_arm_rad": [0.0] * 5,
        "current_gap_m": 0.045, "acceleration_valid": True,
        "snapshot_npz": "fixture.npz",
    }
    audit = {
        "status": "POLICY_FREE_MOVING_H2_SMOOTH_PROFILE_AUDIT",
        "snapshot_duration_s": 0.3,
        "shortest_all_episode_acceleration_valid_duration_s": 0.3,
        "rows": [valid_row],
    }
    assert _smooth_history_row(audit, "fixture") is valid_row
    invalid = dict(audit, snapshot_duration_s=0.4)
    try:
        _smooth_history_row(invalid, "fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("non-shortest smooth snapshot was accepted")
    retention = _PostGraspGapRetentionPolicy(
        _FixturePolicy(), maximum_gap_m=0.037)
    retained = retention.predict_action({})["action_pred"]
    assert np.allclose(retained[:, -1], [0.037, 0.036, 0.037])
    assert np.allclose(retained[:, :-1], 0.0)
    assert np.allclose(retention.requested_gap_m, [0.050, 0.036, 0.048])
    print("moving H2 smooth profile checks: 9/9 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
