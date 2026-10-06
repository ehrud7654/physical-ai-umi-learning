"""Full held-out evaluation of one checkpoint against a hold-current-pose baseline."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    from scipy.spatial.transform import Rotation
    from umi.common.pose_util import rot6d_to_mat
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 10:
        raise ValueError("expected matching [samples, horizon, 10] arrays")
    if not prediction.size or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("empty or non-finite actions")
    p, t = prediction.astype(np.float64), target.astype(np.float64)
    for x in (p, t):
        if np.any(np.linalg.norm(x[..., 3:6], axis=-1) < 1e-8) or np.any(
                np.linalg.norm(np.cross(x[..., 3:6], x[..., 6:9]), axis=-1) < 1e-8):
            raise ValueError("degenerate rotation representation")
    distance = np.linalg.norm(p[..., :3] - t[..., :3], axis=-1) * 1000
    relative = np.swapaxes(rot6d_to_mat(t[..., 3:9].reshape(-1, 6)), -1, -2) @ rot6d_to_mat(
        p[..., 3:9].reshape(-1, 6))
    angles = np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude())
    width = (p[..., 9] - t[..., 9]) * 1000
    return {
        "position_component_rmse_mm": float(np.sqrt(np.mean((p[..., :3] - t[..., :3]) ** 2)) * 1000),
        "position_distance_rmse_mm": float(np.sqrt(np.mean(distance ** 2))),
        "position_distance_p95_mm": float(np.percentile(distance, 95)),
        "rotation_mean_deg": float(np.mean(angles)),
        "rotation_p95_deg": float(np.percentile(angles, 95)),
        "width_rmse_mm": float(np.sqrt(np.mean(width ** 2))),
        "predicted_width_min_max_m": [float(p[..., 9].min()), float(p[..., 9].max())],
    }


def self_check() -> None:
    a = np.zeros((2, 3, 10)); a[..., 3] = 1; a[..., 7] = 1
    b = a.copy(); b[..., 0] = .003; b[..., 9] = .004
    result = metrics(a, b)
    assert abs(result["position_distance_rmse_mm"] - 3) < 1e-8
    assert abs(result["width_rmse_mm"] - 4) < 1e-8
    assert result["rotation_mean_deg"] == 0
    assert metrics(a, a)["position_component_rmse_mm"] == 0
    try:
        metrics(np.zeros_like(a), b)
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate rotations accepted")


def load_policy(checkpoint: Path, device: str):
    import dill
    import hydra
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model" if cfg.training.use_ema else "model"])
    policy.to(device).eval()
    return payload, cfg, policy, int(dill.loads(payload["pickles"]["epoch"]))


def deterministic_image_transform(policy, cfg) -> dict:
    """Swap random training augmentation for its deterministic centre-crop equivalent.

    Jetson inference must apply the same transform so evaluation matches deployment."""
    import torch.nn as nn
    import torchvision
    transforms = cfg.policy.obs_encoder.transforms
    ratio = float(transforms[0].ratio) if transforms and transforms[0].get("type") == "RandomCrop" else 1.0
    encoder = policy.obs_encoder
    for key in encoder.rgb_keys:
        size = int(encoder.key_shape_map[key][-1])
        crop = int(size * ratio)
        encoder.key_transform_map[key] = nn.Identity() if crop == size else nn.Sequential(
            torchvision.transforms.CenterCrop(crop),
            torchvision.transforms.Resize(size, antialias=True))
    return {"type": "identity" if ratio == 1.0 else "center_crop_then_resize", "crop_ratio": ratio}


def build_datasets(cfg, dataset: Path):
    import hydra
    dataset = str(Path(dataset).resolve())
    cfg.task.dataset_path = dataset
    cfg.task.dataset.dataset_path = dataset
    cfg.task.dataset.cache_dir = None
    training = hydra.utils.instantiate(cfg.task.dataset)
    return training, training.get_validation_dataset()


def assert_deterministic(validation) -> None:
    first, second = validation[0], validation[0]
    for key in first["obs"]:
        if not torch.equal(first["obs"][key], second["obs"][key]):
            raise RuntimeError(f"validation sample is not deterministic ({key}); "
                               "is the start-pose noise patch active with val_start_pose_noise_scale=0?")


def evaluate(checkpoint: Path, dataset: Path, output_dir: Path, batch: int = 4, seed: int = 42,
             device: str | None = None) -> dict:
    checkpoint = Path(checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    _, cfg, policy, epoch = load_policy(checkpoint, device)
    if cfg.task.pose_repr.obs_pose_repr != "relative" or cfg.task.pose_repr.action_pose_repr != "relative":
        raise ValueError("hold baseline assumes relative observation and action poses")
    transform = deterministic_image_transform(policy, cfg)
    training, validation = build_datasets(cfg, dataset)
    if not len(validation):
        raise ValueError("validation split is empty")
    assert_deterministic(validation)
    ends = np.asarray(training.replay_buffer.episode_ends)
    predictions, targets, baselines, episode_ids, latency = [], [], [], [], []
    with torch.inference_mode():
        for begin in range(0, len(validation), batch):
            indices = range(begin, min(begin + batch, len(validation)))
            samples = [validation[i] for i in indices]
            obs = {k: torch.stack([s["obs"][k] for s in samples]).to(device) for k in samples[0]["obs"]}
            target = np.stack([s["action"].numpy() for s in samples])
            started = time.perf_counter()
            prediction = policy.predict_action(obs)["action"].cpu().numpy()
            latency.append((time.perf_counter() - started) * 1000)
            metrics(prediction, target)  # fail immediately on invalid model output
            baseline = np.zeros_like(target); baseline[..., 3] = 1; baseline[..., 7] = 1
            baseline[..., 9] = np.array([s["obs"]["robot0_gripper_width"][-1, 0].item() for s in samples])[:, None]
            predictions.append(prediction); targets.append(target); baselines.append(baseline)
            episode_ids.extend(int(np.searchsorted(ends, validation.sampler.indices[i][0], side="right"))
                               for i in indices)
    prediction, target, baseline = map(np.concatenate, (predictions, targets, baselines))
    episode_ids = np.asarray(episode_ids)
    if set(episode_ids.tolist()) != set(np.flatnonzero(training.val_mask).tolist()):
        raise RuntimeError("not every held-out episode produced evaluation sequences")
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": epoch,
        "dataset": str(Path(dataset).resolve()),
        "seed": seed, "batch_size": batch, "device": device,
        "inference_image_transform": transform,
        "evaluated_sequences": int(len(target)),
        "held_out_episode_ids": sorted(set(episode_ids.tolist())),
        "action_shape": list(target.shape),
        "policy": metrics(prediction, target),
        "hold_current_pose_and_width": metrics(baseline, target),
        "per_episode": [{
            "episode_index": int(e), "sequences": int(np.sum(episode_ids == e)),
            "policy": metrics(prediction[episode_ids == e], target[episode_ids == e]),
            "hold": metrics(baseline[episode_ids == e], target[episode_ids == e]),
        } for e in sorted(set(episode_ids.tolist()))],
        "median_batch_inference_ms": float(np.median(latency)),
        "limitations": [
            "Labels come from provisional camera->TCP and ArUco width, not calibrated robot truth.",
            "Sampled sequences overlap; pooled errors are not independent trial success rates.",
            "Offline metric only: no motor commands, closed-loop control, or physical success.",
        ],
    }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem
    np.savez_compressed(output_dir / f"{stem}.predictions.npz", prediction=prediction, target=target,
                        baseline=baseline, episode_ids=episode_ids)
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
