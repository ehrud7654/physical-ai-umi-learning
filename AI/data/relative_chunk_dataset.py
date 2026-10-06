"""Torch dataset for provisional camera-centric UMI relative action chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from umi.relative_dataset import SCHEMA, validate_arrays


class RelativeChunkDataset(Dataset):
    """Load validated episodes and expose image history, proprio and a chunk."""

    def __init__(self, root: Path, *, image_mean: float = 0.5,
                 image_std: float = 0.5) -> None:
        self.root = Path(root)
        self.image_mean = float(image_mean)
        self.image_std = float(image_std)
        if self.image_std <= 0:
            raise ValueError("image_std must be positive")
        index = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        if index.get("schema") != SCHEMA:
            raise ValueError(f"unsupported relative dataset schema: {index.get('schema')}")
        if index.get("status") == "CANDIDATE_NOT_TRAINING_READY":
            raise ValueError(
                "counterfactual localisation candidate is not training-ready; "
                "shifted-target IK and contact dynamics must be validated first"
            )
        self.obs_horizon = int(index["observation_horizon"])
        self.action_horizon = int(index["action_horizon"])
        self.rate_hz = float(index["rate_hz"])
        self.episodes: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        self.episode_sample_indices: list[list[int]] = []
        for episode_id in index["episodes"]:
            path = self.root / f"{episode_id}.npz"
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(path, allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            problems = validate_arrays(
                arrays, obs_horizon=self.obs_horizon,
                action_horizon=self.action_horizon)
            if problems:
                raise ValueError(f"{episode_id}: " + "; ".join(problems))
            episode_index = len(self.episodes)
            start = len(self.index)
            self.episodes.append({"id": episode_id, "meta": meta, **arrays})
            self.index.extend((episode_index, row) for row in range(len(arrays["action"])))
            self.episode_sample_indices.append(list(range(start, len(self.index))))
        if not self.index:
            raise ValueError(f"relative dataset has no samples: {self.root}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        episode_index, row = self.index[item]
        episode = self.episodes[episode_index]
        image = torch.from_numpy(
            episode["image"][row].astype(np.float32) / 255.0)
        image = (image - self.image_mean) / self.image_std
        proprio = torch.from_numpy(episode["proprio"][row].copy())
        action = torch.from_numpy(episode["action"][row].copy())
        return image, proprio, action

    @property
    def episode_ids(self) -> list[str]:
        return [str(episode["id"]) for episode in self.episodes]

    def action_rows(self, sample_indices: list[int]) -> np.ndarray:
        return np.stack([
            self.episodes[episode_index]["action"][row]
            for episode_index, row in (self.index[index] for index in sample_indices)
        ]).astype(np.float32)

    def summary(self) -> str:
        return (
            f"{self.root.name}: episodes={len(self.episodes)}, samples={len(self)}, "
            f"obs={self.obs_horizon}, action={self.action_horizon}, rate={self.rate_hz:g}Hz"
        )
