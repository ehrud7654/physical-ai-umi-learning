"""Focused checks for the optional v10 lateral-correction training sampler."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from tools.train_relative_chunk_bc import lateral_sampling_weights


def main() -> int:
    actions = np.zeros((5, 8, 10), dtype=np.float32)
    actions[:, 3, 0] = [0.02, -0.03, 0.02, 0.01, 0.0]
    gaps = np.array([0.065, 0.060, 0.045, 0.070, 0.065])
    weights, count = lateral_sampling_weights(
        actions, gaps, threshold_m=0.01, factor=4.0)
    assert count == 2  # current open gap and >10mm, either correction sign
    assert np.array_equal(weights, [4.0, 4.0, 1.0, 1.0, 1.0])
    baseline, count_baseline = lateral_sampling_weights(
        actions, gaps, threshold_m=0.01, factor=1.0)
    assert count_baseline == count and np.array_equal(baseline, np.ones(5))
    first = list(WeightedRandomSampler(
        torch.from_numpy(weights), 20, replacement=True,
        generator=torch.Generator().manual_seed(0)))
    second = list(WeightedRandomSampler(
        torch.from_numpy(weights), 20, replacement=True,
        generator=torch.Generator().manual_seed(0)))
    assert first == second
    for invalid in (0.0, 0.009, float("nan"), float("inf")):
        try:
            lateral_sampling_weights(actions, gaps, threshold_m=0.01,
                                     factor=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sampling factor was accepted")
    print("v10 lateral sampler checks: 4/4 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
