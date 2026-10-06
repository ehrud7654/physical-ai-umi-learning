"""Estimate provisional object/grasp dimensions from contact-plane image scale.

The 15 mm gripper markers provide a local pixel-to-mm scale.  This is only a
monocular, contact-plane estimate: it must not be labelled as a physical
measurement, but it is useful for replacing an unsupported replay constant.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi.gripper_gap import MARKER_DIAMETER_MM, _components, detect_markers


def yellow_mask(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    # Pillow hue: yellow/orange packaging occupies roughly 18..55 / 255.
    return ((hsv[..., 0] >= 18) & (hsv[..., 0] <= 55)
            & (hsv[..., 1] >= 105) & (hsv[..., 2] >= 90))


def largest_object_bbox(image: Image.Image):
    minimum = image.width * image.height * 0.015
    candidates = []
    for area, x, y, bbox in _components(yellow_mask(image)):
        x0, y0, x1, y1 = bbox
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if area >= minimum and height >= image.height * 0.18:
            candidates.append((area, x, y, bbox))
    return max(candidates, default=None, key=lambda item: item[0])


def estimate_frame(raw: bytes, frame_index: int, gap_m: float) -> dict | None:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    markers = detect_markers(image)
    obj = largest_object_bbox(image)
    if len(markers) != 2 or obj is None:
        return None
    mean_diameter = np.mean([marker.diameter_px for marker in markers])
    scale = MARKER_DIAMETER_MM / mean_diameter
    marker_x = np.mean([marker.x_px for marker in markers])
    marker_y = np.mean([marker.y_px for marker in markers])
    _, _, _, (x0, y0, x1, y1) = obj
    # Delivered JPEGs are upside-down. In raw image coordinates the physical
    # table-side edge is y0, so grasp height is marker_y - y0.
    if not (x0 - 0.2 * (x1 - x0) <= marker_x <= x1 + 0.2 * (x1 - x0)):
        return None
    height_mm = (y1 - y0 + 1) * scale
    marker_height_mm = (marker_y - y0) * scale
    if not (50 <= height_mm <= 300 and -10 <= marker_height_mm <= height_mm + 10):
        return None
    return {"frame_index": frame_index, "gap_m": gap_m,
            "marker_diameter_px": float(mean_diameter),
            "scale_mm_per_px": float(scale), "object_bbox_xyxy_raw": [x0, y0, x1, y1],
            "object_height_mm": float(height_mm),
            "marker_midpoint_height_above_object_bottom_mm": float(marker_height_mm)}


def summary(values: list[float]) -> dict:
    return {"count": len(values), "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)), "p75": float(np.percentile(values, 75))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    args = parser.parse_args()
    rows, per_episode = [], []
    for bundle in sorted(args.bundles.glob("rec_*.zip")):
        with zipfile.ZipFile(bundle) as archive:
            gaps = list(csv.DictReader(io.StringIO(
                archive.read("gripper.csv").decode("utf-8-sig"))))
            valid = [(i, float(row["gap_m"])) for i, row in enumerate(gaps)
                     if row["status"] in ("D", "M") and row["gap_m"].strip()]
            if not valid:
                continue
            # Smallest-gap frames best approximate contact and equal depth.
            chosen = sorted(valid, key=lambda item: item[1])[:args.frames_per_episode]
            episode_rows = []
            for index, gap in chosen:
                name = f"frames/{index:06d}.jpg"
                if name not in archive.namelist():
                    continue
                estimate = estimate_frame(archive.read(name), index, gap)
                if estimate:
                    episode_rows.append(estimate); rows.append(estimate)
            if episode_rows:
                per_episode.append({"episode_id": bundle.stem, "frames": episode_rows})
    if not rows:
        raise SystemExit("no usable marker/object co-detections")
    result = {
        "schema": "umi_object_geometry_estimate/0.1.0",
        "status": "provisional_monocular_contact_plane_estimate",
        "method": "15mm marker local scale; largest yellow/orange component; raw JPEG upside-down",
        "limitations": [
            "Marker and object are assumed to be at approximately equal depth near contact.",
            "HSV component bounds may omit non-yellow packaging edges.",
            "Replace with ruler/caliper measurements before deployment acceptance."],
        "episodes_total": len(list(args.bundles.glob("rec_*.zip"))),
        "episodes_usable": len(per_episode), "frames_usable": len(rows),
        "object_height_mm": summary([row["object_height_mm"] for row in rows]),
        "marker_midpoint_height_above_object_bottom_mm": summary([
            row["marker_midpoint_height_above_object_bottom_mm"] for row in rows]),
        "per_episode": per_episode,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("episodes_total", "episodes_usable", "frames_usable",
                       "object_height_mm",
                       "marker_midpoint_height_above_object_bottom_mm")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
