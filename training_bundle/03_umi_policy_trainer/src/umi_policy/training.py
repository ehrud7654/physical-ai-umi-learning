"""Run the official TrainDiffusionUnetImageWorkspace inside a new run directory."""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def train(cfg: DictConfig, run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an existing run
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("WANDB_DIR", str(run_dir))
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    import torch
    torch.set_num_threads(4)
    from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
        TrainDiffusionUnetImageWorkspace)
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(run_dir))
    workspace.run()
    if workspace._saving_thread is not None:
        workspace._saving_thread.join()
    return Path(workspace.save_checkpoint(use_thread=False))
