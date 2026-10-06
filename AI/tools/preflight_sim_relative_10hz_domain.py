"""Leakage-safe source split and real-vs-sim RGB diagnostics for 10Hz candidates.

The matching source_row is only a trajectory-phase proxy: robot-time was
stretched in MuJoCo, so these are not synchronized physical camera frames.
Nothing here changes labels or approves synthetic candidates for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from tools.estimate_umi_object_geometry import largest_object_bbox


def source_episode_split(episode_ids: list[str], *, seed: int = 0,
                         holdout_fraction: float = 0.25) -> dict[str, list[str]]:
    unique = sorted(set(episode_ids))
    if len(unique) != len(episode_ids) or len(unique) < 3:
        raise ValueError("need at least three distinct source episodes")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    shuffled = np.random.default_rng(seed).permutation(unique).tolist()
    n_holdout = max(1, int(round(len(unique) * holdout_fraction)))
    return {
        "train_source_episodes": sorted(shuffled[n_holdout:]),
        "holdout_source_episodes": sorted(shuffled[:n_holdout]),
    }


def first_index_per_source_row(source_rows: np.ndarray) -> list[int]:
    """Prevent robot-time repetition from masquerading as independent frames."""
    first: dict[int, int] = {}
    for index, row in enumerate(source_rows):
        first.setdefault(int(row), index)
    return [first[row] for row in sorted(first)]


def _bbox_xywh(image: np.ndarray) -> np.ndarray | None:
    component = largest_object_bbox(Image.fromarray(image))
    if component is None:
        return None
    x0, y0, x1, y1 = component[3]
    height, width = image.shape[:2]
    return np.asarray([
        (x0 + x1 + 1) / (2 * width),
        (y0 + y1 + 1) / (2 * height),
        (x1 - x0 + 1) / width,
        (y1 - y0 + 1) / height,
    ], dtype=float)


def _object_texture(image: np.ndarray, bbox: np.ndarray) -> dict[str, float]:
    height, width = image.shape[:2]
    cx, cy, bw, bh = bbox
    x0 = max(0, int(round((cx - bw / 2) * width)))
    x1 = min(width, int(round((cx + bw / 2) * width)))
    y0 = max(0, int(round((cy - bh / 2) * height)))
    y1 = min(height, int(round((cy + bh / 2) * height)))
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        raise ValueError("object crop is too small")
    gray = crop @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    edges = 0.5 * (np.mean(np.abs(np.diff(gray, axis=0)) > 20)
                   + np.mean(np.abs(np.diff(gray, axis=1)) > 20))
    return {
        "gray_std": float(gray.std()),
        "edge_fraction_abs_gradient_gt_20": float(edges),
    }


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"paired_source_rows": 0}
    metrics = ("bbox_center_l2_norm", "bbox_width_abs_norm",
               "bbox_height_abs_norm", "real_object_gray_std",
               "sim_object_gray_std", "real_object_edge_fraction",
               "sim_object_edge_fraction")
    return {
        "paired_source_rows": len(rows),
        **{name + "_median": float(np.median([row[name] for row in rows]))
           for name in metrics},
    }


def _compare_episode(real_path: Path, candidate_path: Path) -> dict:
    with np.load(real_path, allow_pickle=False) as stored:
        real_images = stored["image"][:, -1]
    with np.load(candidate_path, allow_pickle=False) as stored:
        sim_images = stored["image"][:, -1]
        source_rows = stored["source_row"]
    rows = []
    problems = []
    for index in first_index_per_source_row(source_rows):
        source_row = int(source_rows[index])
        if source_row < 0 or source_row >= len(real_images):
            problems.append(f"source_row {source_row} is outside real episode")
            continue
        real = np.transpose(real_images[source_row], (1, 2, 0))
        sim = np.transpose(sim_images[index], (1, 2, 0))
        real_bbox, sim_bbox = _bbox_xywh(real), _bbox_xywh(sim)
        if real_bbox is None or sim_bbox is None:
            problems.append(f"object bbox absent at source_row {source_row}")
            continue
        real_texture = _object_texture(real, real_bbox)
        sim_texture = _object_texture(sim, sim_bbox)
        rows.append({
            "source_row": source_row,
            "sim_candidate_row": index,
            "bbox_center_l2_norm": float(np.linalg.norm(real_bbox[:2] - sim_bbox[:2])),
            "bbox_width_abs_norm": float(abs(real_bbox[2] - sim_bbox[2])),
            "bbox_height_abs_norm": float(abs(real_bbox[3] - sim_bbox[3])),
            "real_object_gray_std": real_texture["gray_std"],
            "sim_object_gray_std": sim_texture["gray_std"],
            "real_object_edge_fraction": real_texture["edge_fraction_abs_gradient_gt_20"],
            "sim_object_edge_fraction": sim_texture["edge_fraction_abs_gradient_gt_20"],
        })
    return {
        "real_episode": str(real_path),
        "candidate": str(candidate_path),
        "candidate_rows": len(source_rows),
        "distinct_source_rows": len(set(int(value) for value in source_rows)),
        "paired": _summary(rows),
        "problems": problems,
    }


def preflight(suite_path: Path, real_root: Path, *, seed: int = 0) -> dict:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if (suite.get("status") not in (
            "COMPLETE_POLICY_FREE_10HZ_DIAGNOSTIC",
            "COMPLETE_WITH_REJECTIONS_POLICY_FREE_10HZ_DIAGNOSTIC")
            or suite.get("training_ready") is not False):
        raise ValueError("expected the complete, non-training-ready 10Hz diagnostic suite")
    declared = suite["conditions"]["episodes"]
    cases = suite["cases"]
    accepted = [item for item in cases if item["status"] == "CANDIDATE_CHECKED"]
    eligible = sorted({item["episode"] for item in accepted})
    split = source_episode_split(eligible, seed=seed)
    if set(split["train_source_episodes"]) & set(split["holdout_source_episodes"]):
        raise AssertionError("source episode leaked across folds")
    by_episode: dict[str, list[dict]] = {}
    for case in cases:
        by_episode.setdefault(case["episode"], []).append(case)
    if (set(by_episode) != set(declared)
            or any(len(items) != len(suite["conditions"]["offsets_xy_m"])
                   for items in by_episode.values())):
        raise ValueError("suite cases do not cover every declared episode and offset")
    baseline = []
    for episode in eligible:
        matches = [item for item in by_episode[episode]
                   if item["offset_xy_m"] == [0.0, 0.0]
                   and item["status"] == "CANDIDATE_CHECKED"]
        if len(matches) != 1:
            raise ValueError(f"missing unique baseline candidate: {episode}")
        path = real_root / f"{episode}.npz"
        if not path.is_file():
            raise ValueError(f"missing real episode: {path}")
        record = _compare_episode(path, Path(matches[0]["candidate"]))
        record["episode"] = episode
        record["fold"] = ("holdout" if episode in split["holdout_source_episodes"]
                          else "train")
        baseline.append(record)
    paired_rows = [item["paired"] for item in baseline
                   if item["paired"]["paired_source_rows"] > 0]
    if len(paired_rows) != len(baseline):
        raise ValueError("at least one baseline episode has no comparable object frames")
    keys = ("bbox_center_l2_norm_median", "bbox_width_abs_norm_median",
            "bbox_height_abs_norm_median", "real_object_gray_std_median",
            "sim_object_gray_std_median", "real_object_edge_fraction_median",
            "sim_object_edge_fraction_median")
    return {
        "status": "SOURCE_SPLIT_AND_VISUAL_DOMAIN_DIAGNOSTIC_NOT_TRAINING_READY",
        "suite": str(suite_path),
        "real_dataset": str(real_root),
        "seed": seed,
        "declared_source_episodes": declared,
        "eligible_source_episodes": eligible,
        "ineligible_source_episodes": sorted(set(declared) - set(eligible)),
        "source_episode_split": split,
        "case_fold_counts": {
            fold: sum(item["episode"] in set(split[fold + "_source_episodes"])
                      for item in accepted)
            for fold in ("train", "holdout")
        },
        "visual_comparison": {
            "baseline_offset_only": True,
            "paired_source_rows": sum(row["paired"]["paired_source_rows"]
                                      for row in baseline),
            "distinct_episodes": len(baseline),
            "episode_median_of_medians": {
                key: float(np.median([row[key] for row in paired_rows]))
                for key in keys
            } if paired_rows else {},
            "per_episode": baseline,
        },
        "training_ready": False,
        "interpretation": (
            "Five offsets of one source episode must stay in one fold. "
            "Real and simulator images are matched only by source_row; "
            "retimed MuJoCo is not time-synchronized to the human capture. "
            "Object HSV bbox and texture statistics are diagnostic proxies, "
            "not proof of sim-to-real transfer or learned visual dependence."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = preflight(args.suite, args.real_data, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "source_episode_split", "ineligible_source_episodes",
        "case_fold_counts", "visual_comparison", "training_ready")
        if key != "visual_comparison"} | {
            "visual_comparison_summary": {key: value for key, value in
                                          result["visual_comparison"].items()
                                          if key != "per_episode"}},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
