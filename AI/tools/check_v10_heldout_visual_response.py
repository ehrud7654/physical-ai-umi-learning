"""Focused, inference-free checks for held-out visual response accounting."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.analyze_v10_heldout_episode_effect import _paired_effect
from tools.eval_v10_heldout_visual_response import (
    _bin_name, _mean_fill, _noise, _summarise, _validate_contract,
)


def main() -> int:
    assert _bin_name(0.44, 0.011) == ("left", "large_gt_10mm")
    assert _bin_name(0.55, -0.01) == ("centre", "small_le_10mm")
    assert _bin_name(0.56, -0.011) == ("right", "large_gt_10mm")

    rows = [
        {"episode": "a", "target_local_x_m": 0.02,
         "predicted_local_x_m": {"baseline": {"normal": 0.01}}},
        {"episode": "a", "target_local_x_m": 0.02,
         "predicted_local_x_m": {"baseline": {"normal": -0.01}}},
        {"episode": "b", "target_local_x_m": 0.0,
         "predicted_local_x_m": {"baseline": {"normal": 0.003}}},
    ]
    summary = _summarise(rows, "baseline", "normal")
    assert summary["rows"] == 3 and summary["episodes"] == 2
    assert np.isclose(summary["row_mae_m"], (0.01 + 0.03 + 0.003) / 3)
    assert np.isclose(summary["episode_equal_mae_m"], (0.02 + 0.003) / 2)
    assert summary["large_target_direction"] == {
        "rows": 2, "episodes": 1, "correct_rows": 1,
        "median_signed_gain": 0.0,
    }
    assert _summarise([], "baseline", "normal")["row_mae_m"] is None

    image = np.full((2, 3, 224, 224), 128, dtype=np.uint8)
    image[1] = 64
    filled = _mean_fill(image)
    assert np.all(filled[0] == 128) and np.all(filled[1] == 64)
    assert np.array_equal(_noise(image, seed=3, sigma=96),
                          _noise(image, seed=3, sigma=96))
    assert not np.array_equal(_noise(image, seed=3, sigma=96),
                              _noise(image, seed=4, sigma=96))

    heldout = [f"h{i}" for i in range(14)]
    training = [f"t{i}" for i in range(56)]
    index = {"schema": "umi_relative_chunk/0.2.0-provisional",
             "episodes": heldout + training}
    match = {"status": "REAL_V10_OPEN_GAP_SIM_SCALE_IMAGE_SUPPORT_DIAGNOSTIC",
             "rows": ([{"episode": name, "split": "heldout"} for name in heldout]
                      + [{"episode": name, "split": "training"} for name in training])}
    meta = {"val_episodes": heldout, "observation_horizon": 2,
            "action_horizon": 8, "action_dim": 10, "rate_hz": 10.0,
            "seed": 0, "trained_on": "datasets/umi_real_relative_20260911_v10"}
    assert _validate_contract(index, match,
                              {"baseline": SimpleNamespace(meta=meta)}) == sorted(heldout)
    bad = {**meta, "val_episodes": heldout[:-1]}
    try:
        _validate_contract(index, match,
                           {"bad": SimpleNamespace(meta=bad)})
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched holdout split must fail")

    paired = _paired_effect([
        {"difference": 0.01}, {"difference": -0.01},
    ], "difference", seed=0, draws=1000)
    assert paired["episodes"] == 2
    assert paired["positive_episodes"] == paired["negative_episodes"] == 1
    assert np.isclose(paired["episode_equal_mean_m"], 0)
    assert paired["episode_cluster_bootstrap_95pct_m"] == [-0.01, 0.01]
    print("held-out visual response checks: 6/6 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
