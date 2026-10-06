"""Checks for the diagnostic trajectory rate limiter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from tools.replay_real_umi_trajectory import (completed_close_index,
                                              rate_limit_commands,
                                              register_close_to_object,
                                              time_scale_commands)


def main() -> int:
    commands = np.array([[0, 0, 0, 0, 0, 0],
                         [1, -1, .2, -.2, .01, .8],
                         [.5, -.5, 0, 0, 0, 0]], dtype=float)
    limits = np.array([.1, .1, .05, .05, .02, .2])
    limited = rate_limit_commands(commands, limits)
    assert np.all(np.abs(np.diff(limited, axis=0)) <= limits + 1e-12)
    np.testing.assert_allclose(limited[1], [.1, -.1, .05, -.05, .01, .2])
    scaled, source_index = time_scale_commands(commands, limits)
    assert len(scaled) > len(commands)
    assert np.all(np.abs(np.diff(scaled, axis=0)) <= limits + 1e-12)
    np.testing.assert_allclose(scaled[0], commands[0])
    np.testing.assert_allclose(scaled[-1], commands[-1])
    assert source_index[0] == 0 and source_index[-1] == len(commands) - 1
    relative = np.repeat(np.eye(4)[None], 3, axis=0)
    gaps = np.array([.07, .04, .039])
    registered, close_index = register_close_to_object(relative, gaps, [.22, 0, .011], .008)
    assert close_index == 1
    np.testing.assert_allclose(registered[close_index, :3, 3], [.22, 0, .019])
    np.testing.assert_allclose(registered[close_index, :3, 2], [0, 0, -1])
    taller, taller_close = register_close_to_object(relative, gaps, [.22, 0, .05], .04)
    assert taller_close == close_index
    np.testing.assert_allclose(taller[taller_close, :3, 3], [.22, 0, .09])
    assert completed_close_index([.067, .0577, .0453, .0411, .0406, .0403]) == 3
    try:
        rate_limit_commands(commands, np.zeros(6))
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive limits must be rejected")
    print("PASS: causal limiting, path-preserving time scaling, and invalid rejection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
