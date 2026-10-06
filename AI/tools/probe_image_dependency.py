"""Measure trajectory predictability and image dependence of a BC checkpoint.

This is a diagnostic, not a rollout score.  It answers two narrower questions:

1. Can constant-velocity extrapolation predict the next arm state without images?
2. With state held fixed, does one checkpoint change its action when only the
   image is replaced by black, shuffled, or noisy pixels?

Example:
    python tools/probe_image_dependency.py \
      --data datasets/umi_real_20260911_v4 \
      --policy-ckpt checkpoints/bc/umi_real_20260911_v4_seed0.pt \
      --samples 2000 --device cuda --out out/image_dependency_v4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.dataset import EpisodeDataset
from policy.bc import BCNet, to_action
from tools.run_umi_regression import is_shared_gpu_server


def _mae(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=0)


def trajectory_probe(dataset: EpisodeDataset) -> dict[str, Any]:
    """Compare hold and constant-velocity next-state predictors at 30/15/10 Hz."""
    result: dict[str, Any] = {}
    for hz, stride in ((30, 1), (15, 2), (10, 3)):
        truth: list[np.ndarray] = []
        hold: list[np.ndarray] = []
        extrap: list[np.ndarray] = []
        for ep in dataset.episodes:
            # Arm only. Gripper is an absolute set-point in this BC action space.
            state = np.asarray(ep.state[:, :5], dtype=np.float64)
            if len(state) <= 2 * stride:
                continue
            truth.append(state[2 * stride :: stride])
            hold.append(state[stride:-stride:stride])
            extrap.append(2.0 * state[stride:-stride:stride] - state[:-2 * stride:stride])
        y = np.concatenate(truth)
        p_hold = np.concatenate(hold)
        p_extrap = np.concatenate(extrap)
        hold_joint = _mae(p_hold, y)
        extrap_joint = _mae(p_extrap, y)
        hold_mae = float(np.mean(hold_joint))
        extrap_mae = float(np.mean(extrap_joint))
        explained = 100.0 * (1.0 - extrap_mae / hold_mae) if hold_mae else float("nan")
        result[str(hz)] = {
            "samples": int(len(y)),
            "hold_mae": hold_mae,
            "constant_velocity_mae": extrap_mae,
            "improvement_over_hold_percent": explained,
            "hold_mae_by_arm_joint": hold_joint.tolist(),
            "constant_velocity_mae_by_arm_joint": extrap_joint.tolist(),
        }
    return result


def _indices_balanced(dataset: EpisodeDataset, count: int, seed: int) -> np.ndarray:
    """Draw roughly equal counts per episode so long, slowed episodes do not dominate."""
    rng = np.random.default_rng(seed)
    by_episode: list[list[int]] = [[] for _ in dataset.episodes]
    for flat, (episode, _) in enumerate(dataset.index):
        by_episode[episode].append(flat)
    per_episode = max(1, int(np.ceil(count / len(by_episode))))
    chosen: list[int] = []
    for candidates in by_episode:
        take = min(per_episode, len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen[:count], dtype=np.int64)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.reshape(-1), b.reshape(-1)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def image_probe(
    dataset: EpisodeDataset,
    checkpoint: Path,
    *,
    samples: int,
    batch_size: int,
    noise_gray: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    model = BCNet(meta["camera_names"], meta["train_config"])
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    mean = float(meta["train_config"]["data"]["image_mean"])
    std = float(meta["train_config"]["data"]["image_std"])
    target_mean = meta.get("target_mean")
    target_std = meta.get("target_std")
    tm = torch.tensor(target_mean, device=device) if target_mean is not None else None
    ts = torch.tensor(target_std, device=device) if target_std is not None else None
    action_space = str(meta.get("action_space", "joint_absolute"))
    indices = _indices_balanced(dataset, min(samples, len(dataset)), seed)
    rng = np.random.default_rng(seed + 17)

    all_original: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    changed: dict[str, list[np.ndarray]] = {"black": [], "shuffled": [], "noise": []}

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        states_np = np.stack([
            dataset.episodes[dataset.index[int(i)][0]].state[dataset.index[int(i)][1]]
            for i in batch_idx
        ]).astype(np.float32)
        truth_np = np.stack([
            dataset.episodes[dataset.index[int(i)][0]].action[dataset.index[int(i)][1]]
            for i in batch_idx
        ]).astype(np.float32)
        raw_by_cam: dict[str, np.ndarray] = {}
        for cam in model.camera_names:
            raw_by_cam[cam] = np.stack([
                dataset.episodes[dataset.index[int(i)][0]].images[cam][dataset.index[int(i)][1]]
                for i in batch_idx
            ]).astype(np.float32)
        state = torch.from_numpy(states_np).to(device)

        def predict(raw_images: dict[str, np.ndarray]) -> np.ndarray:
            images = {
                cam: ((torch.from_numpy(arr).to(device) / 255.0) - mean) / std
                for cam, arr in raw_images.items()
            }
            raw = model(images, state)
            return to_action(raw, state, action_space, tm, ts).cpu().numpy()

        original = predict(raw_by_cam)
        black = predict({cam: np.zeros_like(arr) for cam, arr in raw_by_cam.items()})
        perm = rng.permutation(len(batch_idx))
        shuffled = predict({cam: arr[perm] for cam, arr in raw_by_cam.items()})
        noisy_raw = {
            cam: np.clip(arr + rng.normal(0.0, noise_gray, arr.shape), 0, 255).astype(np.float32)
            for cam, arr in raw_by_cam.items()
        }
        noisy = predict(noisy_raw)
        all_original.append(original)
        all_truth.append(truth_np)
        changed["black"].append(black)
        changed["shuffled"].append(shuffled)
        changed["noise"].append(noisy)

    original = np.concatenate(all_original)
    truth = np.concatenate(all_truth)
    original_mae = float(np.mean(np.abs(original - truth)))
    report: dict[str, Any] = {
        "samples": int(len(original)),
        "sampling": "episode_balanced",
        "checkpoint": str(checkpoint),
        "original_prediction_mae": original_mae,
        "conditions": {},
    }
    for name, chunks in changed.items():
        pred = np.concatenate(chunks)
        delta = np.abs(pred - original)
        mae = float(np.mean(np.abs(pred - truth)))
        report["conditions"][name] = {
            "action_delta_mae": float(np.mean(delta)),
            "action_delta_p95": float(np.percentile(delta, 95)),
            "action_delta_mae_by_joint": np.mean(delta, axis=0).tolist(),
            "prediction_mae": mae,
            "prediction_mae_ratio_to_original": mae / original_mae if original_mae else float("nan"),
            "prediction_correlation_with_original": _corr(pred, original),
        }
    return report


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit(
            "Shared GPU server is training-only; run policy probes locally")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-gray", type=float, default=96.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.policy_ckpt:
        blob = torch.load(args.policy_ckpt, map_location="cpu", weights_only=False)
        cfg = blob["meta"]["train_config"]
    else:
        from policy.bc import load_train_config
        cfg = load_train_config()
    dataset = EpisodeDataset(args.data, cfg)
    report: dict[str, Any] = {
        "data": str(args.data),
        "dataset_summary": dataset.summary(),
        "trajectory_predictability": trajectory_probe(dataset),
    }
    if args.policy_ckpt:
        report["image_dependency"] = image_probe(
            dataset, args.policy_ckpt, samples=args.samples, batch_size=args.batch_size,
            noise_gray=args.noise_gray, seed=args.seed, device=args.device)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
