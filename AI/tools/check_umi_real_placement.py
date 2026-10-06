"""Dependency-light checks for real UMI trajectory registration helpers."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from tools.umi_real_placement_study import (REAL_J3_LOWER_RAD, candidate_starts,
                                            register_relative, sample_indices)


def main() -> int:
    relative = np.repeat(np.eye(4)[None], 3, axis=0)
    relative[1, 0, 3] = 0.01
    relative[2, 1, 3] = -0.02
    start = np.eye(4)
    start[:3, 3] = [0.2, 0.1, 0.3]
    placed = register_relative(relative, start)
    np.testing.assert_allclose(placed[0], start)
    np.testing.assert_allclose(placed[1, :3, 3], [0.21, 0.1, 0.3])
    np.testing.assert_array_equal(sample_indices(10, 4), [0, 3, 6, 9])
    ranges = np.array([[-2, 2], [-2, 2], [-1.69, 1.69], [-2, 2], [-3, 3], [-1, 1.]])
    candidates = candidate_starts(ranges, 20, 7)
    assert len(candidates) == 20
    assert all(q.shape == (5,) and q[2] >= REAL_J3_LOWER_RAD for q in candidates)
    print("PASS: placement composition, sampling, and real-J3 candidate bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
