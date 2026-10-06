"""Train a diagnostic BC baseline on camera-centric relative action chunks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import yaml

from data.relative_chunk_dataset import RelativeChunkDataset
from policy.relative_chunk_bc import RelativeChunkBCNet
from umi.relative_dataset import ACTION_DIM, SCHEMA


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/train/relative_chunk_bc.yaml"


def split_by_episode(dataset: RelativeChunkDataset, fraction: float,
                     seed: int) -> tuple[list[int], list[int], list[str]]:
    if not 0 < fraction < 1 or len(dataset.episodes) < 2:
        raise ValueError("episode validation split requires at least two episodes")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.episodes))
    n_val = max(1, int(round(len(order) * fraction)))
    val_episodes = set(int(value) for value in order[:n_val])
    train, val = [], []
    for episode_index, indices in enumerate(dataset.episode_sample_indices):
        (val if episode_index in val_episodes else train).extend(indices)
    names = [dataset.episode_ids[index] for index in sorted(val_episodes)]
    if not train or not val:
        raise ValueError("empty train or validation split")
    return train, val, names


def lateral_sampling_weights(actions: np.ndarray, gaps_m: np.ndarray, *,
                             threshold_m: float, factor: float,
                             gap_min_m: float = 0.06,
                             gap_max_m: float = 0.07) -> tuple[np.ndarray, int]:
    """Weight open-jaw rows with a large fourth future local-x correction.

    The rule uses training rows only. It is a diagnostic sampling intervention,
    not a claim that these rows provide causal object-position supervision.
    """
    if actions.ndim != 3 or actions.shape[1:] != (8, ACTION_DIM):
        raise ValueError("expected (N, 8, 10) relative actions")
    if gaps_m.shape != (len(actions),):
        raise ValueError("one current metric gap is required per action")
    if not (np.isfinite(threshold_m) and np.isfinite(factor) and
            0 < threshold_m and factor >= 1 and
            0 <= gap_min_m < gap_max_m <= 0.09):
        raise ValueError("invalid lateral sampling thresholds")
    if not np.isfinite(actions).all() or not np.isfinite(gaps_m).all():
        raise ValueError("non-finite sampling inputs")
    selected = ((gaps_m >= gap_min_m) & (gaps_m <= gap_max_m) &
                (np.abs(actions[:, 3, 0]) > threshold_m))
    weights = np.ones(len(actions), dtype=np.float64)
    weights[selected] = factor
    return weights, int(selected.sum())


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, mean: torch.Tensor, std: torch.Tensor,
              optimizer: torch.optim.Optimizer | None, grad_clip: float,
              cancel_event=None) -> float:
    model.train(optimizer is not None)
    total, count = 0.0, 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for image, proprio, action in loader:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("RelativeChunkBC training cancelled")
            image = image.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            target = (action.to(device, non_blocking=True) - mean) / std
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(image, proprio)
            loss = criterion(prediction, target)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch = image.shape[0]
            total += float(loss.detach()) * batch
            count += batch
    return total / count


def save(path: Path, model: RelativeChunkBCNet, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lateral-sampling-factor", type=float, default=1.0,
                        help="Diagnostic oversampling of open-gap rows with "
                             ">10mm fourth-target local-x correction; 1 keeps baseline")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_config = config["train"]
    epochs = int(args.epochs or train_config["epochs"])
    batch_size = int(args.batch_size or train_config["batch_size"])
    seed = int(train_config["seed"] if args.seed is None else args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    dataset = RelativeChunkDataset(
        args.data, image_mean=float(config["data"]["image_mean"]),
        image_std=float(config["data"]["image_std"]))
    train_indices, val_indices, val_episodes = split_by_episode(
        dataset, float(train_config["val_fraction"]), seed)
    train_actions = torch.from_numpy(dataset.action_rows(train_indices))
    train_gaps = np.asarray([
        dataset.episodes[episode_index]["proprio"][row, -1, -1]
        for episode_index, row in (dataset.index[index] for index in train_indices)
    ], dtype=np.float64)
    sampling_weights, lateral_count = lateral_sampling_weights(
        train_actions.numpy(), train_gaps, threshold_m=0.01,
        factor=args.lateral_sampling_factor)
    mean = train_actions.mean(dim=0)
    std = train_actions.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    trivial = float((((train_actions - mean) / std).abs()).mean())
    print(dataset.summary())
    print(f"split: train={len(train_indices)}, val={len(val_indices)}, "
          f"val episodes={len(val_episodes)}")
    print(f"target: ({dataset.action_horizon}, {ACTION_DIM}) relative SE(3)+gap; "
          "each future step uses the same current-pose anchor")
    print("source time is preserved; robot base alignment and IK are not training labels")
    print(f"mean-predictor standardised L1 baseline: {trivial:.5f}")
    print(f"open-gap large lateral rows in train: {lateral_count}/{len(train_indices)} "
          f"(sampling factor {args.lateral_sampling_factor:g})")
    selected_draw_probability = float(
        lateral_count * args.lateral_sampling_factor / sampling_weights.sum())

    generator = torch.Generator().manual_seed(seed)
    sampler = (WeightedRandomSampler(
        torch.from_numpy(sampling_weights), num_samples=len(train_indices),
        replacement=True, generator=torch.Generator().manual_seed(seed))
        if args.lateral_sampling_factor != 1.0 else None)
    loaders = {
        "train": DataLoader(
            Subset(dataset, train_indices), batch_size=batch_size,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=int(train_config["num_workers"]), generator=generator),
        "val": DataLoader(
            Subset(dataset, val_indices), batch_size=batch_size, shuffle=False,
            num_workers=int(train_config["num_workers"])),
    }
    model = RelativeChunkBCNet(
        obs_horizon=dataset.obs_horizon, action_horizon=dataset.action_horizon,
        action_dim=ACTION_DIM, config=config).to(device)
    spatial_grid_size = int(config["model"].get("spatial_grid_size", 1))
    print(f"image encoder spatial grid: {spatial_grid_size}x{spatial_grid_size} "
          f"(trainable parameters {model.n_params():,})")
    mean, std = mean.to(device), std.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]))
    criterion = nn.L1Loss()
    best, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, loaders["train"], criterion, device, mean, std, optimizer,
            float(train_config["grad_clip"]))
        val_loss = run_epoch(
            model, loaders["val"], criterion, device, mean, std, None,
            float(train_config["grad_clip"]))
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        marker = ""
        if val_loss < best:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            marker = " <- best val"
        print(f"epoch {epoch:3d}/{epochs} train {train_loss:.5f} "
              f"val {val_loss:.5f}{marker}")

    meta = {
        "policy": "relative_chunk_bc_diagnostic",
        "dataset_schema": SCHEMA,
        "trained_on": str(args.data),
        "observation_horizon": dataset.obs_horizon,
        "action_horizon": dataset.action_horizon,
        "action_dim": ACTION_DIM,
        "rate_hz": dataset.rate_hz,
        "target_mean": mean.detach().cpu().tolist(),
        "target_std": std.detach().cpu().tolist(),
        "config": config,
        "seed": seed,
        "epochs": epochs,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "val_episodes": val_episodes,
        "mean_predictor_standardised_l1": trivial,
        "lateral_sampling": {
            "factor": args.lateral_sampling_factor,
            "threshold_m": 0.01,
            "gap_band_m": [0.06, 0.07],
            "train_rows_selected": lateral_count,
            "selected_draw_probability": selected_draw_probability,
            "replacement": sampler is not None,
            "validation_sampler": "all_heldout_rows_once",
        },
        "best_val_loss": best,
        "history": history,
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            "Diagnostic BC baseline; not ACT or Diffusion Policy.",
            ("Spatial-grid encoder ablation; improvement must be established "
             "on episode-held-out object-placement tests."
             if spatial_grid_size > 1 else
             "Global-average image encoder may attenuate object position."),
            "Camera-to-marker-midpoint extrinsic remains provisional.",
            "Held-out episodes are not a certified held-out object-position split.",
        ],
    }
    save(args.out, model, meta)
    best_path = args.out.with_name(args.out.stem + "_bestval" + args.out.suffix)
    if best_state is not None:
        model.load_state_dict(best_state)
        save(best_path, model, {**meta, "checkpoint_selection": "best_validation"})
    print(f"last checkpoint: {args.out}")
    print(f"best checkpoint: {best_path}")
    print("Loss is diagnostic only. Next: same-checkpoint image ablation, then "
          "robot-time IK and camera-matched rollout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
