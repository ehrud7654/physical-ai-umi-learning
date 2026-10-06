"""Compare v10 checkpoints on episode-held-out, scene-scale-matched real frames.

The colour bbox is an image-position proxy, not a measured object placement.
Frames from one demonstration are correlated; episode-level means are reported
separately from row-level means. No robot commands or server inference occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from policy.relative_chunk_bc import RelativeChunkBCPolicy
from tools.audit_v10_real_visual_coverage import _bbox
from tools.run_umi_regression import is_shared_gpu_server


def _bin_name(x: float, target_x: float) -> tuple[str, str]:
    position = ("left" if x < 0.45 else "right" if x > 0.55 else "centre")
    displacement = "large_gt_10mm" if abs(target_x) > 0.01 else "small_le_10mm"
    return position, displacement


def _summarise(rows: list[dict], model: str, condition: str) -> dict:
    if not rows:
        return {"rows": 0, "episodes": 0, "row_mae_m": None,
                "episode_equal_mae_m": None, "large_target_direction": None}
    by_episode: dict[str, list[float]] = {}
    errors = []
    large_directions = []
    large_gains = []
    for row in rows:
        target = row["target_local_x_m"]
        predicted = row["predicted_local_x_m"][model][condition]
        error = abs(predicted - target)
        errors.append(error)
        by_episode.setdefault(row["episode"], []).append(error)
        if abs(target) > 0.01:
            large_directions.append(bool(predicted * target > 0))
            large_gains.append(float(predicted / target))
    return {
        "rows": len(rows),
        "episodes": len(by_episode),
        "row_mae_m": float(np.mean(errors)),
        "episode_equal_mae_m": float(np.mean([
            np.mean(values) for values in by_episode.values()])),
        "large_target_direction": {
            "rows": len(large_directions),
            "episodes": len({row["episode"] for row in rows
                             if abs(row["target_local_x_m"]) > 0.01}),
            "correct_rows": int(sum(large_directions)),
            "median_signed_gain": (float(np.median(large_gains))
                                   if large_gains else None),
        },
    }


def _validate_contract(index: dict, match: dict, policies: dict) -> list[str]:
    if (index.get("schema") != "umi_relative_chunk/0.2.0-provisional"
            or match.get("status")
            != "REAL_V10_OPEN_GAP_SIM_SCALE_IMAGE_SUPPORT_DIAGNOSTIC"):
        raise ValueError("expected real v10 dataset and matching visual audit")
    heldout = sorted(row["episode"] for row in match["rows"]
                     if row["split"] == "heldout")
    training = {row["episode"] for row in match["rows"]
                if row["split"] == "training"}
    if (len(heldout) != 14 or len(set(heldout)) != 14
            or set(heldout) & training
            or set(heldout) | training != set(index["episodes"])):
        raise ValueError("visual audit is not the expected 56/14 episode split")
    for name, policy in policies.items():
        meta = policy.meta
        if (sorted(meta["val_episodes"]) != heldout
                or int(meta["observation_horizon"]) != 2
                or int(meta["action_horizon"]) != 8
                or int(meta["action_dim"]) != 10
                or float(meta["rate_hz"]) != 10.0
                or int(meta["seed"]) != 0
                or Path(meta["trained_on"]).name
                != "umi_real_relative_20260911_v10"):
            raise ValueError(f"{name}: checkpoint does not match v10 heldout contract")
    return heldout


def _noise(image: np.ndarray, *, seed: int, sigma: float) -> np.ndarray:
    draw = np.random.default_rng(seed).normal(0.0, sigma, size=image.shape)
    return np.clip(image.astype(np.float32) + draw, 0, 255).astype(np.uint8)


def _mean_fill(image: np.ndarray) -> np.ndarray:
    values = np.rint(image.mean(axis=(-2, -1), keepdims=True)).astype(np.uint8)
    return np.broadcast_to(values, image.shape).copy()


def evaluate(data: Path, match_path: Path, checkpoints: dict[str, Path],
             *, noise_gray: float = 96.0) -> dict:
    if is_shared_gpu_server():
        raise RuntimeError("learned-policy inference is forbidden on shared GPU server")
    if not np.isfinite(noise_gray) or noise_gray < 0:
        raise ValueError("noise_gray must be finite and nonnegative")
    torch.set_num_threads(2)
    index = json.loads((data / "dataset.json").read_text(encoding="utf-8"))
    match = json.loads(match_path.read_text(encoding="utf-8"))
    policies = {name: RelativeChunkBCPolicy(path, device="cpu")
                for name, path in checkpoints.items()}
    heldout = _validate_contract(index, match, policies)
    target_size = np.asarray(
        match["sim_nominal_bbox_width_height_norm_median"], dtype=float)
    gap_min, gap_max = (float(value) for value in match["gap_band_m"])
    tolerance = float(match["relative_size_tolerance"])
    if (target_size.shape != (2,) or not np.isfinite(target_size).all()
            or np.any(target_size <= 0)
            or not 0 <= gap_min < gap_max <= 0.09
            or not 0 < tolerance <= 1):
        raise ValueError("invalid visual-match selection thresholds")

    rows = []
    coverage = []
    for episode in heldout:
        count = 0
        with np.load(data / f"{episode}.npz", allow_pickle=False) as stored:
            images = stored["image"]
            proprio = stored["proprio"]
            actions = stored["action"]
            if (images.shape[1:] != (2, 3, 224, 224)
                    or proprio.shape != (len(images), 2, 10)
                    or actions.shape != (len(images), 8, 10)):
                raise ValueError(f"{episode}: invalid v10 array shapes")
            for anchor in np.flatnonzero(
                    (proprio[:, -1, -1] >= gap_min)
                    & (proprio[:, -1, -1] <= gap_max)):
                image = images[anchor]
                box = _bbox(image[-1])
                if box is None or np.any(
                        np.abs(np.asarray(box[2:4]) / target_size - 1) > tolerance):
                    continue
                count += 1
                variants = {
                    "normal": image,
                    "mean_fill": _mean_fill(image),
                    "additive_noise": _noise(
                        image, seed=int(hashlib.sha256(
                            f"{episode}:{anchor}".encode()).hexdigest()[:8], 16),
                        sigma=noise_gray),
                }
                predictions = {}
                for name, policy in policies.items():
                    predictions[name] = {}
                    for condition, variant in variants.items():
                        action = policy.predict_action({
                            "image": variant, "proprio": proprio[anchor]
                        })["action_pred"]
                        if action.shape != (8, 10) or not np.isfinite(action).all():
                            raise ValueError(f"{name}: invalid action prediction")
                        predictions[name][condition] = float(action[3, 0])
                target_x = float(actions[anchor, 3, 0])
                position_bin, target_bin = _bin_name(box[0], target_x)
                rows.append({
                    "episode": episode, "anchor_index": int(anchor),
                    "object_bbox_xywh_norm": box,
                    "gap_m": float(proprio[anchor, -1, -1]),
                    "target_local_x_m": target_x,
                    "position_bin": position_bin, "target_bin": target_bin,
                    "predicted_local_x_m": predictions,
                })
        coverage.append({"episode": episode, "matched_rows": count})

    def summaries(subset: list[dict]) -> dict:
        return {name: {condition: _summarise(subset, name, condition)
                       for condition in ("normal", "mean_fill", "additive_noise")}
                for name in checkpoints}

    return {
        "status": "LOCAL_V10_HELDOUT_ALL_MATCHED_VISUAL_RESPONSE_DIAGNOSTIC",
        "dataset": str(data), "match_report": str(match_path),
        "checkpoints": {name: {"path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in checkpoints.items()},
        "device": "local_cpu", "split": "14 episode-heldout, 56 training",
        "selection": {"gap_band_m": [gap_min, gap_max],
                      "relative_object_bbox_size_tolerance": tolerance,
                      "target_size_wh_norm": target_size.tolist(),
                      "all_matching_rows_per_episode": True},
        "perturbation": {"phase": "inference_only_same_checkpoints",
                         "noise": f"additive Gaussian sigma={noise_gray} grayscale values; fixed draw per row across checkpoints",
                         "mean_fill": "per-history-frame/channel mean; removes spatial information"},
        "coverage": coverage, "overall": summaries(rows),
        "by_image_position": {
            band: summaries([row for row in rows if row["position_bin"] == band])
            for band in ("left", "centre", "right")},
        "by_recorded_lateral_target": {
            band: summaries([row for row in rows if row["target_bin"] == band])
            for band in ("small_le_10mm", "large_gt_10mm")},
        "rows": rows,
        "limitations": [
            "Object image position is not measured world placement; camera pose, phase and background vary.",
            "Multiple rows per episode are correlated. Row counts are not independent demonstrations.",
            "This checks recorded fourth-target local-x error, not robot execution or contact success.",
            "Gaussian noise retains some image information; mean fill removes spatial information.",
            "No hardware readiness or real object-position generalisation claim follows.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lateral4", type=Path,
                        help="Optional historical factor-4 sampler checkpoint")
    parser.add_argument("--candidate", type=Path,
                        help="Optional new checkpoint under the generic candidate label")
    parser.add_argument("--noise-gray", type=float, default=96.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    checkpoints = {"baseline": args.baseline}
    if args.lateral4 is not None:
        checkpoints["lateral4"] = args.lateral4
    if args.candidate is not None:
        checkpoints["candidate"] = args.candidate
    if len(checkpoints) == 1:
        raise SystemExit("provide at least one of --lateral4 or --candidate")
    result = evaluate(args.data, args.match, checkpoints,
                      noise_gray=args.noise_gray)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("status", "coverage", "overall", "by_image_position",
                       "by_recorded_lateral_target")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
