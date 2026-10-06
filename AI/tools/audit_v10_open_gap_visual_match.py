"""Find real-v10 open-gap frames with object scale like a MuJoCo handoff.

One matching row is selected per episode. This checks image support without
pretending that apparent object scale identifies a physical task phase.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.audit_v10_real_visual_coverage import _bbox, _stats


def audit(data_root: Path, object_probe: Path, *,
          gap_min_m: float = 0.06, gap_max_m: float = 0.07,
          relative_size_tolerance: float = 0.25) -> dict:
    if (not 0 <= gap_min_m < gap_max_m <= 0.09
            or not 0 < relative_size_tolerance <= 1):
        raise ValueError("invalid gap band or size tolerance")
    index = json.loads((data_root / "dataset.json").read_text(encoding="utf-8"))
    probe = json.loads(object_probe.read_text(encoding="utf-8"))
    if (index.get("schema") != "umi_relative_chunk/0.2.0-provisional"
            or probe.get("status") != "LOCAL_FAR_START_H2_OBJECT_ONLY_FIRST_FOUR_DIAGNOSTIC"):
        raise ValueError("expected v10 dataset and object-only handoff probe")
    sim_boxes = np.asarray([row["views"][0]["object_bbox_xywh"][1]
                            for row in probe["rows"]], dtype=float)
    target_size = np.median(sim_boxes[:, 2:4], axis=0)
    heldout = {str(row["episode"]) for row in probe["rows"]}
    heldout.update(str(row["episode"]) for row in probe["excluded"])
    rows = []
    for episode in index["episodes"]:
        candidate = None
        visible_open_rows = 0
        size_matched_rows = 0
        size_matched_large_x_rows = 0
        with np.load(data_root / f"{episode}.npz", allow_pickle=False) as stored:
            gap = stored["proprio"][:, -1, -1]
            for position in np.flatnonzero((gap >= gap_min_m) & (gap <= gap_max_m)):
                box = _bbox(stored["image"][position, -1])
                if box is None:
                    continue
                visible_open_rows += 1
                size = np.asarray(box[2:4])
                relative_error = np.abs(size / target_size - 1)
                if np.any(relative_error > relative_size_tolerance):
                    continue
                size_matched_rows += 1
                size_matched_large_x_rows += int(
                    abs(float(stored["action"][position, 3, 0])) > 0.01)
                score = float(np.sum(relative_error**2))
                if candidate is None or score < candidate["size_error_score"]:
                    candidate = {
                        "anchor_index": int(position),
                        "gap_m": float(gap[position]),
                        "object_bbox_xywh_norm": box,
                        "size_error_score": score,
                        "first_four_target_last_translation_local_m":
                            stored["action"][position, 3, :3].astype(float).tolist(),
                    }
        rows.append({"episode": episode,
                     "split": "heldout" if episode in heldout else "training",
                     "visible_open_rows": visible_open_rows,
                     "size_matched_rows": size_matched_rows,
                     "size_matched_large_x_rows_gt_10mm":
                         size_matched_large_x_rows,
                     "selected": candidate})

    def summary(split: str | None) -> dict:
        subset = [row for row in rows if split is None or row["split"] == split]
        selected = [row["selected"] for row in subset if row["selected"] is not None]
        return {
            "episodes": len(subset),
            "episodes_with_matching_frame": len(selected),
            "visible_open_rows": sum(row["visible_open_rows"] for row in subset),
            "size_matched_rows": sum(row["size_matched_rows"] for row in subset),
            "size_matched_large_x_rows_gt_10mm": sum(
                row["size_matched_large_x_rows_gt_10mm"] for row in subset),
            "selected_episodes_large_x_gt_10mm": sum(
                abs(row["selected"]["first_four_target_last_translation_local_m"][0])
                > 0.01 for row in subset if row["selected"] is not None),
            "bbox_xywh_norm": _stats([
                item["object_bbox_xywh_norm"] for item in selected]),
            "first_four_target_last_translation_local_m": _stats([
                item["first_four_target_last_translation_local_m"]
                for item in selected]),
        }

    return {
        "status": "REAL_V10_OPEN_GAP_SIM_SCALE_IMAGE_SUPPORT_DIAGNOSTIC",
        "data": str(data_root),
        "object_probe": str(object_probe),
        "sim_nominal_bbox_width_height_norm_median": target_size.tolist(),
        "gap_band_m": [gap_min_m, gap_max_m],
        "relative_size_tolerance": relative_size_tolerance,
        "all": summary(None),
        "training": summary("training"),
        "heldout": summary("heldout"),
        "rows": rows,
        "limitations": [
            "Colour-mask size matching is visual, not a physical pose or contact-phase match.",
            "Only one nearest-scale row per episode enters centre statistics; selection is not a success score.",
            "The probe reports 13 evaluable episodes plus one pre-policy-contact exclusion; all 14 are treated as held out.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--object-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = audit(args.data, args.object_probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("sim_nominal_bbox_width_height_norm_median",
                       "training", "heldout")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
