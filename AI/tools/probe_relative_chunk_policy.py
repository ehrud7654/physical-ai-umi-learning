"""Diagnose image dependence and constant-velocity shortcuts in relative BC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.relative_chunk_dataset import RelativeChunkDataset
from policy.relative_chunk_bc import RelativeChunkBCPolicy
from tools.run_umi_regression import is_shared_gpu_server
from umi.relative_dataset import relative_vector, vector_to_transform


def balanced_indices(dataset: RelativeChunkDataset, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    each = max(1, int(np.ceil(count / len(dataset.episode_sample_indices))))
    selected: list[int] = []
    for indices in dataset.episode_sample_indices:
        take = min(each, len(indices))
        selected.extend(int(value) for value in rng.choice(indices, take, replace=False))
    rng.shuffle(selected)
    return selected[:count]


def shortcut_metrics(dataset: RelativeChunkDataset) -> dict[str, float]:
    truth, predicted = [], []
    for episode in dataset.episodes:
        proprio, action = episode["proprio"], episode["action"]
        if proprio.shape[1] < 2:
            continue
        for history, target in zip(proprio, action):
            current_to_previous = vector_to_transform(history[-2])
            previous_to_current = np.linalg.inv(current_to_previous)
            transform = np.eye(4)
            chunk = []
            for step in range(target.shape[0]):
                transform = transform @ previous_to_current
                chunk.append(relative_vector(
                    np.eye(4), transform, float(history[-1, 9])))
            truth.append(target)
            predicted.append(np.stack(chunk))
    expected, baseline = np.stack(truth), np.stack(predicted)
    return {
        "samples": int(len(expected)),
        "translation_l2_mae_m": float(np.linalg.norm(
            expected[..., :3] - baseline[..., :3], axis=-1).mean()),
        "rotation6d_mae": float(np.abs(
            expected[..., 3:9] - baseline[..., 3:9]).mean()),
        "gap_mae_m": float(np.abs(expected[..., 9] - baseline[..., 9]).mean()),
    }


def differences(reference: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    delta = changed - reference
    return {
        "translation_l2_mean_m": float(np.linalg.norm(delta[..., :3], axis=-1).mean()),
        "rotation6d_mae": float(np.abs(delta[..., 3:9]).mean()),
        "gap_mae_m": float(np.abs(delta[..., 9]).mean()),
        "all_components_mae": float(np.abs(delta).mean()),
    }


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit(
            "Shared GPU server is training-only; run policy probes locally")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-gray", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    policy = RelativeChunkBCPolicy(args.policy_ckpt, args.device)
    config = policy.meta["config"]["data"]
    dataset = RelativeChunkDataset(
        args.data, image_mean=float(config["image_mean"]),
        image_std=float(config["image_std"]))
    indices = balanced_indices(dataset, min(args.samples, len(dataset)), args.seed)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    original, targets = [], []
    changed = {name: [] for name in ("black", "shuffled", "noise")}
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    mean, std = policy.mean, policy.std
    policy.model.eval()
    with torch.inference_mode():
        for image, proprio, action in loader:
            image, proprio = image.to(policy.device), proprio.to(policy.device)
            prediction = policy.model(image, proprio) * std + mean
            original.append(prediction.cpu().numpy())
            targets.append(action.numpy())
            variants = {
                "black": torch.full_like(
                    image, (0.0 - policy.image_mean) / policy.image_std),
                "shuffled": torch.roll(image, shifts=1, dims=0),
                "noise": image + torch.randn(
                    image.shape, generator=generator, device=policy.device,
                    dtype=image.dtype) * (args.noise_gray / 255.0 / policy.image_std),
            }
            for name, variant in variants.items():
                output = policy.model(variant, proprio) * std + mean
                changed[name].append(output.cpu().numpy())
    base = np.concatenate(original)
    target = np.concatenate(targets)
    result = {
        "dataset": str(args.data),
        "checkpoint": str(args.policy_ckpt),
        "samples": len(base),
        "constant_velocity_shortcut": shortcut_metrics(dataset),
        "original_prediction_error": differences(target, base),
        "same_checkpoint_image_replacement": {
            name: differences(base, np.concatenate(values))
            for name, values in changed.items()
        },
        "interpretation": (
            "Image replacement must materially change translation/orientation, not only gap. "
            "This remains diagnostic; rollout is the performance test."
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
