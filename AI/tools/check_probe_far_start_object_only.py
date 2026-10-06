"""Fast checks for the executed-prefix object-direction diagnostic."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.probe_far_start_object_only import _first_four_response


def main() -> int:
    expected = np.array([0.01, 0.0, 0.0])
    change = np.zeros((8, 10))
    change[:4, :3] = expected
    change[4:, :3] = -expected
    result = _first_four_response(change, expected)
    assert result["executed_prefix_last_direction_positive"] is True
    assert np.isclose(result["executed_prefix_last_gain"], 1.0)
    assert np.isclose(result["executed_prefix_mean_change_l2_m"], 0.01)
    opposite = _first_four_response(-change, expected)
    assert opposite["executed_prefix_last_direction_positive"] is False
    assert np.isclose(opposite["executed_prefix_last_gain"], -1.0)
    try:
        _first_four_response(change[:4], expected)
    except ValueError:
        pass
    else:
        raise AssertionError("short chunk must be rejected")
    print("far-start object-only prefix checks: 3/3 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
