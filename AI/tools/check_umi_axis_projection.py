"""Checks for deterministic spherical approach-axis projection."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from tools.sweep_real_umi_axis_projection import angle_deg, project_axis


def main() -> int:
    anchor = np.array([0.0, 0.0, 1.0])
    original = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(project_axis(original, anchor, 0), anchor, atol=1e-12)
    np.testing.assert_allclose(project_axis(original, anchor, 1), original, atol=1e-12)
    halfway = project_axis(original, anchor, 0.5)
    assert abs(angle_deg(anchor, halfway) - 45.0) < 1e-9
    assert abs(angle_deg(original, halfway) - 45.0) < 1e-9
    opposite = project_axis(-anchor, anchor, 0.5)
    assert np.isfinite(opposite).all() and abs(np.linalg.norm(opposite) - 1) < 1e-12
    print("PASS: axis projection endpoints, midpoint, and antipodal case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
