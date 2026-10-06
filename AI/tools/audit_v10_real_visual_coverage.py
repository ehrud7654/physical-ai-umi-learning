"""Measure visible snack-case coverage in the real v10 training images.

This is a read-only colour-mask audit, not physical object localisation.
Between-episode image variation mixes object placement with handheld camera
pose, crop, and trajectory-phase variation.
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


def _bbox(image_chw: np.ndarray) -> list[float] | None:
    if image_chw.shape != (3, 224, 224):
        raise ValueError(f"expected canonical RGB (3,224,224), got {image_chw.shape}")
    image = Image.fromarray(np.transpose(image_chw, (1, 2, 0)))
    component = largest_object_bbox(image)
    if component is None:
        return None
    x0, y0, x1, y1 = component[3]
    return [float((x0 + x1 + 1) / 448),
            float((y0 + y1 + 1) / 448),
            float((x1 - x0 + 1) / 224),
            float((y1 - y0 + 1) / 224)]


def _stats(values: list[list[float]]) -> dict:
    if not values:
        return {"samples": 0}
    array = np.asarray(values, dtype=float)
    return {
        "samples": len(values),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "p05": np.quantile(array, 0.05, axis=0).tolist(),
        "p50": np.quantile(array, 0.50, axis=0).tolist(),
        "p95": np.quantile(array, 0.95, axis=0).tolist(),
    }


def audit(root: Path, *, close_gap_m: float = 0.045,
          heldout_audit: Path | None = None) -> dict:
    index = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if index.get("schema") != "umi_relative_chunk/0.2.0-provisional":
        raise ValueError("expected v10 relative-chunk dataset schema")
    if not 0 < close_gap_m < 0.09:
        raise ValueError("close gap threshold must be within (0, 0.09)m")
    rows = []
    for episode in index["episodes"]:
        with np.load(root / f"{episode}.npz", allow_pickle=False) as stored:
            images = stored["image"]
            proprio = stored["proprio"]
            action = stored["action"]
            if (images.ndim != 5 or images.shape[1:] != (2, 3, 224, 224)
                    or proprio.shape != (len(images), 2, 10)
                    or action.shape != (len(images), 8, 10)):
                raise ValueError(f"invalid v10 arrays: {episode}")
            gap = proprio[:, -1, -1]
            close_indices = np.flatnonzero(gap <= close_gap_m)
            if not len(close_indices):
                raise ValueError(f"no close-gap anchor: {episode}")
            close_index = int(close_indices[0])
            samples = {}
            for name, position in (("first", 0), ("first_gap_le_45mm", close_index)):
                samples[name] = {
                    "anchor_index": position,
                    "gap_m": float(gap[position]),
                    "object_bbox_xywh_norm": _bbox(images[position, -1]),
                    "first_four_target_last_translation_local_m":
                        action[position, 3, :3].astype(float).tolist(),
                    "history_translation_local_m":
                        proprio[position, 0, :3].astype(float).tolist(),
                }
            rows.append({"episode": episode, "n_anchors": len(images),
                         "samples": samples})

    heldout = set()
    if heldout_audit is not None:
        source = json.loads(heldout_audit.read_text(encoding="utf-8"))
        if source.get("status") != "POLICY_FREE_FIXED_START_TO_APPROACH_IK_TIMING_AUDIT":
            raise ValueError("expected fixed-start audit for held-out episode IDs")
        heldout = set(source["episodes"])
        if len(heldout) != len(source["episodes"]) or heldout - set(index["episodes"]):
            raise ValueError("invalid held-out episode IDs")

    def summarize(selected: list[dict]) -> dict:
        groups = {}
        for name in ("first", "first_gap_le_45mm"):
            entries = [row["samples"][name] for row in selected]
            boxes = [entry["object_bbox_xywh_norm"] for entry in entries
                     if entry["object_bbox_xywh_norm"] is not None]
            targets = [entry["first_four_target_last_translation_local_m"]
                       for entry in entries]
            groups[name] = {
                "detections": len(boxes),
                "episodes": len(entries),
                "bbox_columns": ["center_x", "center_y", "width", "height"],
                "bbox_xywh_norm": _stats(boxes),
                "first_four_target_last_translation_local_m": _stats(targets),
                "gap_m": _stats([[entry["gap_m"]] for entry in entries]),
            }
        return groups

    return {
        "status": "REAL_V10_IMAGE_COVERAGE_NOT_OBJECT_WORLD_PLACEMENT",
        "dataset": str(root),
        "episodes": len(rows),
        "close_gap_threshold_m": close_gap_m,
        "groups": summarize(rows),
        "heldout_episode_source": str(heldout_audit) if heldout_audit else None,
        "training_episodes_excluding_heldout": (
            summarize([row for row in rows if row["episode"] not in heldout])
            if heldout else None),
        "heldout_episodes": (
            summarize([row for row in rows if row["episode"] in heldout])
            if heldout else None),
        "rows": rows,
        "limitations": [
            "The yellow/orange connected-component bbox is an automatic proxy, not manual ground truth.",
            "Between-episode bbox differences conflate object placement, camera pose, image crop, and phase.",
            "The first <=45mm-gap anchor is a phase proxy; it does not synchronize physical contact.",
            "This audit does not infer object XY in a robot base frame or prove label-response causality.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--heldout-audit", type=Path,
                        help="optional fixed-start report listing held-out episodes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = audit(args.data, heldout_audit=args.heldout_audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"episodes": result["episodes"],
                      "groups": result["groups"],
                      "training_episodes_excluding_heldout":
                          result["training_episodes_excluding_heldout"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
