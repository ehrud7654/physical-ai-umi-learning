"""Pair object offsets while separating inference RGB from proprio changes.

This is an offline diagnostic on policy-free simulator candidates. It does not
approve those candidates as training data or a learned policy for real motors.
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
from tools.run_umi_regression import is_shared_gpu_server


def common_source_row(case_paths: list[Path]) -> tuple[int, list[int]]:
    """Choose the earliest source row shared by every object placement."""
    source_rows = []
    for path in case_paths:
        with np.load(path, allow_pickle=False) as stored:
            source_rows.append(stored["source_row"].astype(int).tolist())
    shared = set(source_rows[0]).intersection(*(set(rows) for rows in source_rows[1:]))
    if not shared:
        raise ValueError("object placements have no common source row")
    row = min(shared)
    return row, [rows.index(row) for rows in source_rows]


def translation_response(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compare the *change* in a predicted chunk with an oracle change."""
    predicted = np.asarray(prediction, dtype=np.float64)[..., :3]
    expected = np.asarray(target, dtype=np.float64)[..., :3]
    if predicted.shape != expected.shape or predicted.shape != (8, 3):
        raise ValueError("expected paired 8-step translation differences")
    pred_norm = float(np.linalg.norm(predicted, axis=-1).mean())
    target_norm = float(np.linalg.norm(expected, axis=-1).mean())
    target_energy = float(np.sum(expected * expected))
    pred_energy = float(np.sum(predicted * predicted))
    dot = float(np.sum(predicted * expected))
    return {
        "prediction_change_l2_mean_m": pred_norm,
        "oracle_change_l2_mean_m": target_norm,
        "alignment_gain": dot / target_energy if target_energy > 1e-12 else 0.0,
        "cosine": (dot / np.sqrt(pred_energy * target_energy)
                   if pred_energy > 1e-12 and target_energy > 1e-12 else 0.0),
        "delta_error_l2_mean_m": float(np.linalg.norm(predicted - expected, axis=-1).mean()),
    }


def _read_row(path: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name][index].copy()
                for name in ("image", "proprio", "action")}


def evaluate(suite_path: Path, checkpoint: Path) -> dict:
    if is_shared_gpu_server():
        raise RuntimeError("learned-checkpoint inference is forbidden on the shared GPU server")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if (suite.get("status") != "COMPLETE_WITH_REJECTIONS_POLICY_FREE_10HZ_DIAGNOSTIC"
            or suite.get("training_ready") is not False):
        raise ValueError("expected complete, non-training-ready policy-free suite")
    offsets = [tuple(value) for value in suite["conditions"]["offsets_xy_m"]]
    if len(offsets) != 5 or len(set(offsets)) != 5 or (0.0, 0.0) not in offsets:
        raise ValueError("expected unique baseline and four object offsets")
    by_episode: dict[str, dict[tuple[float, float], dict]] = {}
    for case in suite["cases"]:
        episode = str(case["episode"])
        offset = tuple(case["offset_xy_m"])
        if offset in by_episode.setdefault(episode, {}):
            raise ValueError(f"duplicate placement: {episode} {offset}")
        by_episode[episode][offset] = case
    if set(by_episode) != set(suite["conditions"]["episodes"]):
        raise ValueError("missing declared source episode")

    policy = RelativeChunkBCPolicy(checkpoint, device="cpu")
    per_episode = []
    excluded = []
    for episode in suite["conditions"]["episodes"]:
        group = by_episode[episode]
        if set(group) != set(offsets):
            raise ValueError(f"missing placement in {episode}")
        if any(group[offset]["status"] != "CANDIDATE_CHECKED" for offset in offsets):
            excluded.append({"episode": episode, "reason": "incomplete candidate set"})
            continue
        paths = [Path(group[offset]["candidate"]) for offset in offsets]
        source_row, indices = common_source_row(paths)
        observations = {offset: _read_row(path, index)
                        for offset, path, index in zip(offsets, paths, indices)}
        baseline = observations[(0.0, 0.0)]

        def predict(image: np.ndarray, proprio: np.ndarray) -> np.ndarray:
            return policy.predict_action({"image": image, "proprio": proprio})[
                "action_pred"]

        reference = predict(baseline["image"], baseline["proprio"])
        comparisons = []
        for offset in offsets:
            if offset == (0.0, 0.0):
                continue
            moved = observations[offset]
            target_change = moved["action"] - baseline["action"]
            rgb_only = predict(moved["image"], baseline["proprio"]) - reference
            proprio_only = predict(baseline["image"], moved["proprio"]) - reference
            combined = predict(moved["image"], moved["proprio"]) - reference
            comparisons.append({
                "offset_world_xy_m": list(offset),
                "rgb_abs_difference_mean_255": float(np.abs(
                    moved["image"].astype(np.int16)
                    - baseline["image"].astype(np.int16)).mean()),
                "proprio_abs_difference_max": float(np.max(np.abs(
                    moved["proprio"] - baseline["proprio"]))),
                "rgb_only": translation_response(rgb_only, target_change),
                "proprio_only": translation_response(proprio_only, target_change),
                "rgb_and_proprio": translation_response(combined, target_change),
            })
        per_episode.append({"episode": episode, "source_row": source_row,
                            "offset_comparisons": comparisons})

    def episode_median(input_name: str, field: str) -> float:
        return float(np.median([
            np.median([row[input_name][field]
                       for row in episode["offset_comparisons"]])
            for episode in per_episode
        ]))

    fields = ("prediction_change_l2_mean_m", "oracle_change_l2_mean_m",
              "alignment_gain", "cosine", "delta_error_l2_mean_m")

    return {
        "status": "LOCAL_OFFLINE_IMAGE_VS_PROPRIO_POSITION_DIAGNOSTIC",
        "suite": str(suite_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "inference_device": "cpu",
        "same_checkpoint": True,
        "source_episodes": len(per_episode),
        "paired_offsets": sum(len(item["offset_comparisons"]) for item in per_episode),
        "excluded_episodes": excluded,
        "episode_median_of_offset_medians": {
            input_name: {name: episode_median(input_name, name) for name in fields}
            for input_name in ("rgb_only", "proprio_only", "rgb_and_proprio")
        } if per_episode else {},
        "per_episode": per_episode,
        "limitations": [
            "Only the earliest common source_row per eligible episode is used; offsets from one episode are paired, not independent trials.",
            "Policy-free oracle target differences are simulator-achieved futures, not human commands or training labels.",
            "RGB frames may also differ in robot pixels because the first recorded observations are after motion; holding proprio fixed isolates the model input channel but not object pixels alone.",
            "K=4 near-contact, simulation-only 2mm preload, provisional contact/camera models and one seed do not validate real-robot visual localisation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = evaluate(args.suite, args.policy_ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "source_episodes", "paired_offsets", "excluded_episodes",
        "episode_median_of_offset_medians")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
