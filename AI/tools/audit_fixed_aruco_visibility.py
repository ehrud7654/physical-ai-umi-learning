#!/usr/bin/env python3
"""Audit visibility of one fixed ArUco marker across intake episodes."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _maximum_missing_run(detected: list[bool]) -> int:
    maximum = current = 0
    for value in detected:
        if value:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _scan(task: tuple[str, str, str | None, int]) -> dict:
    episode, video_text, trajectory_text, marker_id = task
    video = Path(video_text)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"episode": episode, "status": "ERROR", "error": "cannot open video"}
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
    detected: list[bool] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        _, ids, _ = detector.detectMarkers(frame)
        detected.append(bool(ids is not None and np.any(ids.reshape(-1) == marker_id)))
    capture.release()
    indices = [index for index, value in enumerate(detected) if value]
    result = {
        "episode": episode,
        "status": "OK",
        "frames": len(detected),
        "detected_frames": len(indices),
        "detection_rate": len(indices) / len(detected) if detected else 0.0,
        "first_detected_frame": indices[0] if indices else None,
        "last_detected_frame": indices[-1] if indices else None,
        "maximum_missing_run_frames": _maximum_missing_run(detected),
    }
    if trajectory_text is not None:
        trajectory = Path(trajectory_text)
        if not trajectory.is_file():
            result["trajectory_status"] = "MISSING"
        else:
            with trajectory.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != len(detected):
                result["trajectory_status"] = "FRAME_COUNT_MISMATCH"
            else:
                pose_valid = [
                    row.get("state") == "2"
                    and row.get("is_lost", "true").lower() == "false"
                    for row in rows
                ]
                overlap = [a and b for a, b in zip(detected, pose_valid)]
                union = [a or b for a, b in zip(detected, pose_valid)]
                overlap_indices = [i for i, value in enumerate(overlap) if value]
                result.update({
                    "trajectory_status": "OK",
                    "orb_valid_frames": sum(pose_valid),
                    "marker_orb_overlap_frames": sum(overlap),
                    "marker_orb_overlap_first_last_frame": (
                        [overlap_indices[0], overlap_indices[-1]]
                        if overlap_indices else None
                    ),
                    "marker_orb_union_coverage": sum(union) / len(union) if union else 0.0,
                    "marker_orb_union_maximum_missing_run_frames": _maximum_missing_run(union),
                })
    return result


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path)
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    batch = json.loads(args.batch_summary.read_text(encoding="utf-8"))
    episode_names = sorted(
        entry["episode"]
        for key in ("quality_pass", "quality_reject")
        for entry in batch.get(key, [])
    )
    tasks = [
        (
            episode,
            str(args.raw_root / episode / "video.mp4"),
            (str(args.trajectory_root / episode / "camera_trajectory.csv")
             if args.trajectory_root else None),
            args.marker_id,
        )
        for episode in episode_names
    ]
    if args.workers <= 1:
        episodes = [_scan(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            episodes = list(executor.map(_scan, tasks))

    ok = [entry for entry in episodes if entry["status"] == "OK"]
    errors = [entry for entry in episodes if entry["status"] != "OK"]
    report = {
        "status": "PASS" if not errors else "INCOMPLETE",
        "marker_id": args.marker_id,
        "episodes_requested": len(tasks),
        "episodes_scanned": len(ok),
        "episodes_with_detection": sum(entry["detected_frames"] > 0 for entry in ok),
        "detection_rate": _summary([entry["detection_rate"] for entry in ok]),
        "first_detected_frame": _summary([
            entry["first_detected_frame"] for entry in ok
            if entry["first_detected_frame"] is not None
        ]),
        "maximum_missing_run_frames": _summary([
            entry["maximum_missing_run_frames"] for entry in ok
        ]),
        "marker_orb_overlap_frames": _summary([
            entry["marker_orb_overlap_frames"] for entry in ok
            if entry.get("trajectory_status") == "OK"
        ]),
        "marker_orb_union_coverage": _summary([
            entry["marker_orb_union_coverage"] for entry in ok
            if entry.get("trajectory_status") == "OK"
        ]),
        "marker_orb_union_maximum_missing_run_frames": _summary([
            entry["marker_orb_union_maximum_missing_run_frames"] for entry in ok
            if entry.get("trajectory_status") == "OK"
        ]),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "marker_id", "episodes_requested", "episodes_scanned",
        "episodes_with_detection", "detection_rate", "first_detected_frame",
        "maximum_missing_run_frames", "marker_orb_overlap_frames",
        "marker_orb_union_coverage", "marker_orb_union_maximum_missing_run_frames",
    )}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
