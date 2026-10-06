"""Compose the official UMI Hydra config with this package's profile overrides."""
from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_NAME = "train_diffusion_unet_timm_umi_workspace"


def load_profile(path: Path) -> dict:
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile.get("hydra_overrides"), list) or profile.get("policy_frequency_hz", 0) <= 0:
        raise ValueError(f"{path}: expected policy_frequency_hz > 0 and a hydra_overrides list")
    return profile


def down_sample_steps(native_hz: float, policy_hz: float) -> int:
    steps = max(1, round(native_hz / policy_hz))
    effective = native_hz / steps
    if abs(effective - policy_hz) / policy_hz > 0.1:
        raise ValueError(f"{native_hz} Hz cannot be down-sampled to {policy_hz} Hz within 10% "
                         f"(closest is {effective:.2f} Hz)")
    return steps


def compose_config(umi_root: Path, profile: dict, dataset: Path, native_hz: float,
                   epochs: int | None = None, batch: int | None = None, seed: int | None = None,
                   max_train_steps: int | None = None) -> DictConfig:
    overrides = list(profile["hydra_overrides"]) + [
        f"task.dataset_path='{Path(dataset).resolve().as_posix()}'",
        f"task.obs_down_sample_steps={down_sample_steps(native_hz, profile['policy_frequency_hz'])}",
    ]
    if epochs is not None:
        overrides.append(f"training.num_epochs={epochs}")
    if batch is not None:
        overrides += [f"dataloader.batch_size={batch}", f"val_dataloader.batch_size={batch}"]
    if seed is not None:
        overrides += [f"training.seed={seed}", f"task.dataset.seed={seed}"]
    if max_train_steps is not None:
        overrides.append(f"training.max_train_steps={max_train_steps}")
    config_dir = Path(umi_root) / "diffusion_policy/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=overrides)
