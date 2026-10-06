"""Fast mathematical and input-control checks for fixed-pose image probing."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.probe_fixed_pose_object_only import _mean_fill, _response, _suite_conditions


def main() -> int:
    expected = np.array([0.01, 0.0, 0.0])
    correct = np.zeros((8, 10), dtype=float)
    correct[:, :3] = expected
    result = _response(correct, expected)
    assert result["terminal_direction_positive"] is True
    assert np.isclose(result["terminal_gain"], 1.0)
    assert np.isclose(result["terminal_cosine"], 1.0)
    opposite = _response(-correct, expected)
    assert opposite["terminal_direction_positive"] is False
    assert np.isclose(opposite["terminal_cosine"], -1.0)

    image = np.arange(2 * 3 * 224 * 224, dtype=np.uint32).reshape(2, 3, 224, 224)
    filled = _mean_fill((image % 256).astype(np.uint8))
    assert filled.shape == (2, 3, 224, 224)
    assert filled.dtype == np.uint8
    assert (filled == filled[:, :, :1, :1]).all()

    suite = (Path(__file__).resolve().parents[1] / "out"
             / "sim_relative_10hz_sweep_20260917.json")
    if suite.is_file():
        _, episodes, offsets = _suite_conditions(suite)
        assert len(episodes) == 7
        assert len(offsets) == 5
    print("fixed-pose object-only checks: 3/3 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
