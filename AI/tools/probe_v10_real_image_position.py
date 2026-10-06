"""Test v10 checkpoint response to real-image object position locally.

One selected frame per episode has comparable apparent object size/open gap.
Holding proprio fixed isolates the image channel across real episodes, but
background, hand pixels, and camera pose also differ: this is not an
object-only causal image intervention or a rollout success test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from policy.relative_chunk_bc import RelativeChunkBCPolicy
from tools.probe_fixed_pose_object_only import _mean_fill
from tools.run_umi_regression import is_shared_gpu_server


def _association(rows: list[dict], x_key: str, y_key: str) -> dict:
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y = np.asarray([row[y_key] for row in rows], dtype=float)
    if len(x) < 3 or np.std(x) <= 0:
        raise ValueError("too few samples or constant object position")
    if np.std(y) <= 1e-12:
        return {"samples": len(x), "slope_m_per_normalized_image_x": 0.0,
                "pearson_r": 0.0, "constant_prediction": True}
    slope = float(np.polyfit(x, y, 1)[0])
    return {"samples": len(x),
            "slope_m_per_normalized_image_x": slope,
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "constant_prediction": False}


def _predict_x(policy: RelativeChunkBCPolicy, image: np.ndarray,
               proprio: np.ndarray) -> float:
    predicted = np.asarray(policy.predict_action({
        "image": image, "proprio": proprio})["action_pred"], dtype=float)
    if predicted.shape != (8, 10) or not np.isfinite(predicted).all():
        raise ValueError(f"invalid action prediction {predicted.shape}")
    return float(predicted[3, 0])


def _position_bins(rows: list[dict]) -> dict:
    """Expose off-centre errors instead of hiding them in a global MAE."""
    bands = {
        "left_x_lt_0.45": lambda x: x < 0.45,
        "centre_x_0.45_to_0.55": lambda x: 0.45 <= x <= 0.55,
        "right_x_gt_0.55": lambda x: x > 0.55,
    }
    result = {}
    for name, within in bands.items():
        subset = [row for row in rows if within(row["bbox_center_x"])]
        result[name] = {
            "episodes": len(subset),
            "recorded_large_x_gt_10mm": sum(
                abs(row["recorded_target_x_m"]) > 0.01 for row in subset),
            "original_proprio_x_mae_m": (
                float(np.mean([abs(row["pred_original_proprio_x_m"]
                                   - row["recorded_target_x_m"])
                               for row in subset])) if subset else None),
        }
    return result


def run(data_root: Path, match_path: Path, checkpoint: Path) -> dict:
    if is_shared_gpu_server():
        raise RuntimeError("learned-checkpoint inference is forbidden on shared GPU server")
    match = json.loads(match_path.read_text(encoding="utf-8"))
    if match.get("status") != "REAL_V10_OPEN_GAP_SIM_SCALE_IMAGE_SUPPORT_DIAGNOSTIC":
        raise ValueError("expected real-v10 open-gap visual-match report")
    selected = [row for row in match["rows"] if row["selected"] is not None]
    training = [row for row in selected if row["split"] == "training"]
    heldout = [row for row in selected if row["split"] == "heldout"]
    if not training or not heldout:
        raise ValueError("need distinct training and held-out image groups")
    policy = RelativeChunkBCPolicy(checkpoint, device="cpu")
    checkpoint_val = set(policy.meta["val_episodes"])
    report_val = {row["episode"] for row in match["rows"]
                  if row["split"] == "heldout"}
    if checkpoint_val != report_val:
        raise ValueError("match split does not equal checkpoint validation split")

    centre = float(np.median([row["selected"]["object_bbox_xywh_norm"][0]
                              for row in training]))
    reference = min(training, key=lambda row: (
        abs(row["selected"]["object_bbox_xywh_norm"][0] - centre),
        row["episode"]))
    reference_path = data_root / f"{reference['episode']}.npz"
    with np.load(reference_path, allow_pickle=False) as stored:
        fixed_proprio = stored["proprio"][
            reference["selected"]["anchor_index"]].copy()

    rows = []
    for item in selected:
        episode = item["episode"]
        choice = item["selected"]
        with np.load(data_root / f"{episode}.npz", allow_pickle=False) as stored:
            image = stored["image"][choice["anchor_index"]].copy()
            proprio = stored["proprio"][choice["anchor_index"]].copy()
            target_x = float(stored["action"][choice["anchor_index"], 3, 0])
        if image.shape != (2, 3, 224, 224) or proprio.shape != (2, 10):
            raise ValueError(f"bad v10 input: {episode}")
        rows.append({
            "episode": episode,
            "split": item["split"],
            "anchor_index": choice["anchor_index"],
            "bbox_center_x": choice["object_bbox_xywh_norm"][0],
            "recorded_target_x_m": target_x,
            "pred_original_proprio_x_m": _predict_x(policy, image, proprio),
            "pred_fixed_proprio_x_m": _predict_x(policy, image, fixed_proprio),
            "pred_fixed_proprio_mean_fill_x_m": _predict_x(
                policy, _mean_fill(image), fixed_proprio),
        })

    def summary(split: str) -> dict:
        group = [row for row in rows if row["split"] == split]
        return {
            "episodes": len(group),
            "recorded_label": _association(group, "bbox_center_x",
                                           "recorded_target_x_m"),
            "prediction_original_proprio": _association(
                group, "bbox_center_x", "pred_original_proprio_x_m"),
            "prediction_fixed_proprio": _association(
                group, "bbox_center_x", "pred_fixed_proprio_x_m"),
            "prediction_fixed_proprio_mean_fill": _association(
                group, "bbox_center_x", "pred_fixed_proprio_mean_fill_x_m"),
            "original_proprio_mae_m": float(np.mean([
                abs(row["pred_original_proprio_x_m"]
                    - row["recorded_target_x_m"]) for row in group])),
            "object_centre_x_bins": _position_bins(group),
        }

    return {
        "status": "LOCAL_V10_REAL_IMAGE_POSITION_CHECKPOINT_RESPONSE_DIAGNOSTIC",
        "data": str(data_root),
        "image_match_report": str(match_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "inference_device": "cpu",
        "fixed_proprio_reference": {
            "episode": reference["episode"],
            "anchor_index": reference["selected"]["anchor_index"],
            "selection_rule": "training object bbox centre nearest training median",
        },
        "training": summary("training"),
        "heldout": summary("heldout"),
        "rows": rows,
        "limitations": [
            "This is a cross-episode comparison, not moving one object within the same real image.",
            "Camera pose, human/hand pixels, lighting and background vary with bbox centre.",
            "Fixed proprio may be inconsistent with some real images and is only an input-channel diagnostic.",
            "No robot command or closed-loop success is evaluated.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = run(args.data, args.match, args.policy_ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("fixed_proprio_reference", "training", "heldout")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
