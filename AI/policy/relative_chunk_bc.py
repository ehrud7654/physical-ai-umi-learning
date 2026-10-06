"""Small action-chunk BC baseline for relative UMI trajectories.

This is deliberately a diagnostic baseline, not an ACT or Diffusion Policy
claim.  It predicts physical-unit relative SE(3)+gap chunks compatible with
``umi.policy_preflight`` after inverse target standardisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class RelativeImageEncoder(nn.Module):
    """Small image encoder kept independent from SO-101 joint-policy code."""

    def __init__(self, channels: list[int], feature_dim: int,
                 spatial_grid_size: int = 1) -> None:
        super().__init__()
        self.spatial_grid_size = int(spatial_grid_size)
        if not 1 <= self.spatial_grid_size <= 8:
            raise ValueError("spatial_grid_size must be in [1, 8]")
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            layers.extend([
                nn.Conv2d(input_channels, output_channels, 3, stride=2,
                          padding=1, bias=False),
                nn.GroupNorm(min(8, output_channels), output_channels),
                nn.ReLU(inplace=True),
            ])
            input_channels = output_channels
        self.network = nn.Sequential(*layers)
        projection_input = input_channels * self.spatial_grid_size ** 2
        self.projection = nn.Linear(projection_input, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.network(image)
        encoded = nn.functional.adaptive_avg_pool2d(
            encoded, self.spatial_grid_size).flatten(1)
        return self.projection(encoded)


class RelativeChunkBCNet(nn.Module):
    def __init__(self, *, obs_horizon: int, action_horizon: int,
                 action_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        model = config["model"]
        feature_dim = int(model["feature_dim"])
        self.encoder = RelativeImageEncoder(
            [int(value) for value in model["encoder_channels"]], feature_dim,
            spatial_grid_size=int(model.get("spatial_grid_size", 1)))
        input_dim = self.obs_horizon * (feature_dim + self.action_dim)
        dims = [input_dim, *[int(value) for value in model["hidden_dims"]]]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(left, right), nn.ReLU(inplace=True),
                nn.Dropout(float(model["dropout"])),
            ])
        layers.append(nn.Linear(dims[-1], self.action_horizon * self.action_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != self.obs_horizon:
            raise ValueError("image must have shape (B, observation_horizon, 3, H, W)")
        if proprio.shape[1:] != (self.obs_horizon, self.action_dim):
            raise ValueError("proprio shape does not match the checkpoint contract")
        batch = image.shape[0]
        encoded = self.encoder(image.reshape(-1, *image.shape[2:]))
        encoded = encoded.reshape(batch, self.obs_horizon, -1)
        fused = torch.cat((encoded, proprio), dim=-1).reshape(batch, -1)
        return self.head(fused).reshape(batch, self.action_horizon, self.action_dim)

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


class RelativeChunkBCPolicy:
    """Inference wrapper exposing the existing predict_action chunk boundary."""

    def __init__(self, checkpoint: Path | str, device: str = "cpu") -> None:
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        self.meta = blob["meta"]
        self.device = torch.device(device)
        self.model = RelativeChunkBCNet(
            obs_horizon=int(self.meta["observation_horizon"]),
            action_horizon=int(self.meta["action_horizon"]),
            action_dim=int(self.meta["action_dim"]), config=self.meta["config"])
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device).eval()
        self.mean = torch.tensor(
            self.meta["target_mean"], dtype=torch.float32, device=self.device)
        self.std = torch.tensor(
            self.meta["target_std"], dtype=torch.float32, device=self.device)
        self.image_mean = float(self.meta["config"]["data"]["image_mean"])
        self.image_std = float(self.meta["config"]["data"]["image_std"])

    @torch.inference_mode()
    def predict_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(observation, dict) or not {"image", "proprio"} <= observation.keys():
            raise ValueError("observation must contain image and proprio histories")
        image = np.asarray(observation["image"])
        proprio = np.asarray(observation["proprio"], dtype=np.float32)
        expected_image = (int(self.meta["observation_horizon"]), 3, 224, 224)
        expected_proprio = (int(self.meta["observation_horizon"]),
                            int(self.meta["action_dim"]))
        if image.shape != expected_image or proprio.shape != expected_proprio:
            raise ValueError(
                f"observation shapes must be {expected_image} and {expected_proprio}, "
                f"got {image.shape} and {proprio.shape}")
        if image.dtype == np.uint8:
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            image_tensor = (image_tensor - self.image_mean) / self.image_std
        else:
            image_tensor = torch.from_numpy(image.astype(np.float32))
        prediction = self.model(
            image_tensor[None].to(self.device),
            torch.from_numpy(proprio)[None].to(self.device),
        )[0]
        physical = prediction * self.std + self.mean
        return {"action_pred": physical.detach().cpu().numpy()}
