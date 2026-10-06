"""Focused compatibility checks for the optional spatial-grid image encoder."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from policy.relative_chunk_bc import RelativeChunkBCNet, RelativeImageEncoder


def config(grid: int | None = None) -> dict:
    model = {
        "encoder_channels": [8, 16],
        "feature_dim": 12,
        "hidden_dims": [24],
        "dropout": 0.0,
    }
    if grid is not None:
        model["spatial_grid_size"] = grid
    return {"model": model}


def main() -> int:
    image = torch.zeros((2, 2, 3, 224, 224), dtype=torch.float32)
    proprio = torch.zeros((2, 2, 10), dtype=torch.float32)

    # A missing setting is the exact legacy global-average architecture.
    legacy = RelativeChunkBCNet(
        obs_horizon=2, action_horizon=8, action_dim=10, config=config())
    assert legacy.encoder.spatial_grid_size == 1
    assert legacy.encoder.projection.in_features == 16
    assert legacy(image, proprio).shape == (2, 8, 10)

    spatial = RelativeChunkBCNet(
        obs_horizon=2, action_horizon=8, action_dim=10, config=config(4))
    assert spatial.encoder.spatial_grid_size == 4
    assert spatial.encoder.projection.in_features == 16 * 4 * 4
    assert spatial(image, proprio).shape == (2, 8, 10)
    assert spatial.n_params() > legacy.n_params()

    # Checkpoint reconstruction uses only the saved config and state dict.
    restored = RelativeChunkBCNet(
        obs_horizon=2, action_horizon=8, action_dim=10, config=config(4))
    restored.load_state_dict(spatial.state_dict())
    assert torch.equal(spatial(image, proprio), restored(image, proprio))

    for invalid in (0, 9):
        try:
            RelativeImageEncoder([8], 8, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid spatial grid size was accepted")

    print("relative chunk spatial encoder checks: 5/5 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
