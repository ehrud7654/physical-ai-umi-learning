"""MuJoCo and real UMI must give channel 6 the same gap meaning."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sim.mujoco.build_scene import denormalize, load_config, normalize


def main() -> None:
    cfg = load_config()
    curve = np.asarray(cfg["grasp"]["gap_curve"], dtype=float)
    q = np.zeros((len(curve), 6), dtype=float)
    q[:, 5] = curve[:, 0]
    normalized = normalize(q, cfg)
    expected = 2.0 * (curve[:, 1] / 100.0) / 0.09 - 1.0
    np.testing.assert_allclose(normalized[:, 5], expected, atol=2e-6)

    restored = denormalize(normalized, cfg)
    np.testing.assert_allclose(restored[:, 5], curve[:, 0], atol=2e-6)

    closed = denormalize(np.array([0, 0, 0, 0, 0, -1], dtype=float), cfg)
    opened = denormalize(np.array([0, 0, 0, 0, 0, 1], dtype=float), cfg)
    assert np.isclose(closed[5], curve[0, 0])
    assert np.isclose(opened[5], curve[-1, 0])
    print("PASS: sim channel 6 uses full gap metres; endpoint saturation explicit")


if __name__ == "__main__":
    main()
