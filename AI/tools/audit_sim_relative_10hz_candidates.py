"""Summarize policy-free 10Hz candidates; never promote them to training data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from tools.check_sim_relative_10hz_candidate import check
from tools.estimate_umi_object_geometry import largest_object_bbox
from umi.relative_dataset import relative_vector, vector_to_transform


def shortcut_metrics(proprio: np.ndarray, action: np.ndarray) -> dict[str, float]:
    """Compare future translation with hold and two-frame SE(3) extrapolation."""
    predicted = []
    for history, target in zip(proprio, action):
        previous_to_current = np.linalg.inv(vector_to_transform(history[0]))
        future = np.eye(4)
        chunk = []
        for _ in target:
            future = future @ previous_to_current
            chunk.append(relative_vector(
                np.eye(4), future, float(history[1, 9])))
        predicted.append(np.stack(chunk))
    baseline = np.stack(predicted)
    return {
        "hold_translation_l2_mean_m": float(np.linalg.norm(
            action[..., :3], axis=-1).mean()),
        "constant_velocity_translation_l2_mean_m": float(np.linalg.norm(
            action[..., :3] - baseline[..., :3], axis=-1).mean()),
        "constant_velocity_rotation6d_mae": float(np.abs(
            action[..., 3:9] - baseline[..., 3:9]).mean()),
        "constant_velocity_gap_mae_m": float(np.abs(
            action[..., 9] - baseline[..., 9]).mean()),
    }


def _initial_object_center(image_chw: np.ndarray) -> list[float]:
    image = Image.fromarray(np.transpose(image_chw, (1, 2, 0)))
    component = largest_object_bbox(image)
    if component is None:
        raise ValueError("initial image has no visible object colour component")
    x0, y0, x1, y1 = component[3]
    return [(x0 + x1) / 2, (y0 + y1) / 2]


def pair_metrics(left: dict, right: dict) -> dict:
    """Compare a same-episode, fixed-start object-only placement pair."""
    a, b = left["arrays"], right["arrays"]
    ma, mb = left["meta"], right["meta"]
    if (ma["source_episode"] != mb["source_episode"]
            or ma.get("robot_start_is_fixed") is not True
            or mb.get("robot_start_is_fixed") is not True
            or int(a["source_row"][0]) != int(b["source_row"][0])
            or not np.allclose(a["observation_timestamp"][0, 0],
                               b["observation_timestamp"][0, 0], atol=1e-8)):
        raise ValueError("paired candidates must share episode, start and clock")
    image_a, image_b = a["image"][0, 0], b["image"][0, 0]
    centre_a = _initial_object_center(image_a)
    centre_b = _initial_object_center(image_b)
    return {
        "source_episode": ma["source_episode"],
        "candidate_a": str(left["path"]),
        "candidate_b": str(right["path"]),
        "object_offset_difference_world_xy_m": (
            np.asarray(mb["object_offset_world_xy_m"])
            - np.asarray(ma["object_offset_world_xy_m"])).tolist(),
        "initial_rgb_abs_difference_mean": float(np.abs(
            image_a.astype(np.int16) - image_b.astype(np.int16)).mean()),
        "object_center_a_px": centre_a,
        "object_center_b_px": centre_b,
        "object_center_shift_px": (np.asarray(centre_b)
                                   - np.asarray(centre_a)).tolist(),
        "first_future_tcp_translation_difference_m": (
            b["action"][0, 0, :3] - a["action"][0, 0, :3]).tolist(),
        "interpretation": (
            "RGB and achieved target vary at a fixed start; this does not "
            "prove a learned policy uses RGB"),
    }


def audit(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        validation = check(path)
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        metrics = shortcut_metrics(arrays["proprio"], arrays["action"])
        rows.append({"path": path, "meta": meta, "arrays": arrays,
                     "validation": validation, "shortcut": metrics})
    pairs = []
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            if (first["meta"]["source_episode"] == second["meta"]["source_episode"]
                    and first["meta"]["object_offset_world_xy_m"]
                    != second["meta"]["object_offset_world_xy_m"]):
                pairs.append(pair_metrics(first, second))
    weights = np.asarray([len(row["arrays"]["action"]) for row in rows], dtype=float)
    names = tuple(rows[0]["shortcut"]) if rows else ()
    return {
        "status": ("DIAGNOSTIC_CANDIDATES_CHECKED_NOT_TRAINING_READY"
                   if rows and all(not row["validation"]["problems"] for row in rows)
                   else "CANDIDATE_SUITE_REJECTED"),
        "candidates": len(rows),
        "source_episodes": len({row["meta"]["source_episode"] for row in rows}),
        "rows": int(weights.sum()),
        "checks": [row["validation"] for row in rows],
        "shortcut_by_candidate": [
            {"candidate": str(row["path"]), "rows": len(row["arrays"]["action"]),
             **row["shortcut"]} for row in rows],
        "shortcut_weighted_mean": {
            name: float(np.average([row["shortcut"][name] for row in rows],
                                   weights=weights)) for name in names},
        "same_episode_object_offset_pairs": pairs,
        "training_ready": False,
        "interpretation": (
            "Executed future simulator states are not controller commands. "
            "Neither visible RGB differences nor shortcut errors establish "
            "policy image dependence, real-world calibration or deployment readiness."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = audit(args.candidates)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"].startswith("DIAGNOSTIC_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
