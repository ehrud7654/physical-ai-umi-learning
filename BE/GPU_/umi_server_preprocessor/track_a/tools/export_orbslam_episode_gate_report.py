#!/usr/bin/env python3
"""Export per-episode ORB/gap quality and complete gate rejection provenance.

The report is intentionally independent of RGB access.  It lets Track A audit
why each input episode was accepted or rejected after a private runner executes
the raw pipeline.  Rejection reason counts are episode counts; one episode may
contribute to more than one category, so category counts need not sum to N.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_row(row: dict[str, str]) -> bool:
    return row.get("state") == "2" and row.get("is_lost", "true").lower() == "false"


def false_intervals(values: list[bool]) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate([*values, True]):
        if not value and start is None:
            start = index
        elif value and start is not None:
            intervals.append({
                "start_frame": start,
                "end_frame": index - 1,
                "length_frames": index - start,
            })
            start = None
    return intervals


def fixed_marker_summary(video: Path, marker_id: int = 13) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--scan-fixed-marker requires OpenCV with the aruco module and NumPy"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("--scan-fixed-marker requires an OpenCV build with aruco")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"status": "error", "error": "cannot open video"}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = (
        cv2.aruco.ArucoDetector(dictionary)
        if hasattr(cv2.aruco, "ArucoDetector") else None
    )
    detected: list[bool] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if detector is not None:
            _, ids, _ = detector.detectMarkers(frame)
        else:
            _, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
        detected.append(bool(ids is not None and np.any(ids.reshape(-1) == marker_id)))
    capture.release()
    intervals = false_intervals(detected)
    count = sum(detected)
    return {
        "status": "ok",
        "marker_id": marker_id,
        "frames": len(detected),
        "detected_frames": count,
        "detection_rate": count / len(detected) if detected else 0.0,
        "maximum_missing_run_frames": max(
            (item["length_frames"] for item in intervals), default=0),
    }


def trajectory_summary(path: Path) -> dict[str, Any]:
    pose_rows = rows(path)
    tracked = [tracked_row(row) for row in pose_rows]
    indices = [index for index, value in enumerate(tracked) if value]
    timestamps = []
    for index in indices:
        try:
            timestamps.append(float(pose_rows[index]["timestamp"]))
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "frames": len(pose_rows),
        "tracked_frames": len(indices),
        "coverage": len(indices) / len(pose_rows) if pose_rows else 0.0,
        "first_tracked_frame": indices[0] if indices else None,
        "first_tracked_time_s": timestamps[0] if timestamps else None,
        "final_frame_tracked": bool(tracked and tracked[-1]),
        "tracking_failure_intervals": false_intervals(tracked),
    }


def reason_categories(reason: str) -> set[str]:
    text = reason.lower()
    categories: set[str] = set()
    if text.startswith("outcome="):
        categories.add("outcome_not_success")
    if "missing gripper_report" in text:
        categories.add("missing_gripper_report")
    if "gripper status=" in text or "missing run=" in text:
        categories.add("gripper_quality")
    if "no trajectory attempts" in text:
        categories.add("no_trajectory_attempt")
    if "coverage" in text:
        categories.add("slam_coverage")
    if "first pose time" in text:
        categories.add("slam_initialization_time")
    if "tracking loss" in text:
        categories.add("slam_tracking_loss")
    if "final frame" in text:
        categories.add("slam_final_frame_untracked")
    if "timestamp error" in text:
        categories.add("timestamp_alignment")
    if "v10 build" in text:
        categories.add("v10_build")
    return categories or {"other"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--gripper-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path)
    parser.add_argument("--scan-fixed-marker", action="store_true")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.out_json.exists() or args.out_csv.exists():
        raise SystemExit("refusing to overwrite an existing gate report")

    selection = load_json(args.selection)
    selected = {item["episode_id"]: item for item in selection.get("episodes", [])}
    excluded = selection.get("excluded", {})
    dataset = load_json(args.dataset_index) if args.dataset_index else None
    dataset_accepted = set(dataset.get("episodes", [])) if dataset else set()
    dataset_rejected = dataset.get("rejected", {}) if dataset else {}
    dataset_root = args.dataset_index.parent if args.dataset_index else None
    candidates = selection.get("candidates", [])

    episodes: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    exact_reason_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()

    for raw in sorted(path for path in args.raw_root.glob("rec_*") if path.is_dir()):
        episode_id = raw.name
        manifest = load_json(raw / "manifest.json")
        outcome = str(manifest.get("outcome", "missing"))
        outcome_counts[outcome] += 1
        frame_count = len(rows(raw / "frames.csv"))

        gripper_path = args.gripper_root / episode_id / "gripper_report.json"
        gripper = load_json(gripper_path) if gripper_path.is_file() else {}
        attempts: list[dict[str, Any]] = []
        for candidate in candidates:
            batch = Path(candidate["batch"])
            for attempt_dir in sorted((batch / episode_id).glob("attempt*")):
                trajectory = attempt_dir / "camera_trajectory.csv"
                if not trajectory.is_file():
                    continue
                item = {
                    "candidate": candidate.get("name"),
                    "attempt": attempt_dir.name,
                    **trajectory_summary(trajectory),
                }
                validation = attempt_dir / "trajectory_validation_v10.json"
                if validation.is_file():
                    checked = load_json(validation)
                    item["gate_status"] = checked.get("status")
                    item["gate_errors"] = checked.get("errors", [])
                attempts.append(item)

        reasons = list(excluded.get(episode_id, []))
        if episode_id in selected:
            gate_status = "selected"
            if dataset is not None and episode_id not in dataset_accepted:
                gate_status = "rejected_v10_build"
                detail = dataset_rejected.get(episode_id, "not present in final dataset")
                reasons = [f"v10 build: {detail}"]
        else:
            gate_status = "rejected_selection"
        categories = sorted({
            category
            for reason in reasons
            for category in reason_categories(reason)
        })
        for category in categories:
            category_counts[category] += 1
        for reason in set(reasons):
            exact_reason_counts[reason] += 1

        chosen = selected.get(episode_id, {})
        v10_metadata: dict[str, Any] = {}
        if dataset_root is not None and episode_id in dataset_accepted:
            metadata_path = dataset_root / f"{episode_id}.json"
            if metadata_path.is_file():
                metadata = load_json(metadata_path)
                v10_metadata = {
                    "start_tcp_pose_6d": metadata.get("start_tcp_pose_6d"),
                    "first_training_anchor_tcp_pose_6d": metadata.get(
                        "first_training_anchor_tcp_pose_6d"),
                }
        episode = {
            "episode_id": episode_id,
            "outcome": outcome,
            "gate_status": gate_status,
            "rejection_categories": categories,
            "rejection_reasons": reasons,
            "frame_count": frame_count,
            "selected_candidate": chosen.get("atlas_candidate"),
            "selected_attempt": chosen.get("attempt"),
            "slam_attempts": attempts,
            "fixed_table_marker": (
                fixed_marker_summary(raw / "video.mp4")
                if args.scan_fixed_marker else {"status": "not_scanned"}
            ),
            "gripper": {
                "status": gripper.get("status"),
                "jaw_marker_pair_detection_rate": gripper.get("detection_rate"),
                "maximum_missing_run_frames": gripper.get("maximum_missing_run_frames"),
                "gap_min_mm": gripper.get("width_min_mm"),
                "gap_median_mm": gripper.get("width_median_mm"),
                "gap_max_mm": gripper.get("width_max_mm"),
            },
            "v10_metadata": v10_metadata,
        }
        episodes.append(episode)

    selected_n = sum(item["gate_status"] == "selected" for item in episodes)
    report = {
        "schema": "orbslam_episode_gate_report/0.1",
        "scope": "per-episode quality and rejection provenance; no RGB included",
        "inputs": {
            "selection_sha256": sha256_file(args.selection),
            "dataset_index_sha256": (
                sha256_file(args.dataset_index) if args.dataset_index else None),
        },
        "count_semantics": (
            "reason counts are episode counts; one episode may appear in multiple categories"
        ),
        "summary": {
            "input_episodes": len(episodes),
            "passed_episodes": selected_n,
            "rejected_episodes": len(episodes) - selected_n,
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "rejection_category_episode_counts": dict(sorted(category_counts.items())),
            "exact_rejection_reason_episode_counts": dict(sorted(exact_reason_counts.items())),
            "selection_thresholds": selection.get("thresholds", {}),
            "dataset_index_included": dataset is not None,
        },
        "episodes": episodes,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    columns = [
        "episode_id", "outcome", "gate_status", "rejection_categories",
        "rejection_reasons", "frame_count", "selected_candidate", "selected_attempt",
        "slam_attempt_count", "slam_best_coverage", "slam_best_tracked_frames",
        "slam_best_first_tracked_time_s", "slam_tracking_failure_intervals",
        "jaw_marker_pair_detection_rate", "maximum_gap_missing_run_frames",
        "gap_min_mm", "gap_median_mm", "gap_max_mm",
        "fixed_table_marker_detection_rate", "fixed_table_marker_maximum_missing_run_frames",
        "start_tcp_xyz_m", "start_tcp_rotation_vector_rad",
        "first_training_anchor_tcp_xyz_m", "first_training_anchor_tcp_rotation_vector_rad",
    ]
    with args.out_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in episodes:
            attempts = item["slam_attempts"]
            best = max(attempts, key=lambda row: row["coverage"], default={})
            grip = item["gripper"]
            fixed = item["fixed_table_marker"]
            v10 = item["v10_metadata"]
            start = v10.get("start_tcp_pose_6d") or {}
            first_anchor = v10.get("first_training_anchor_tcp_pose_6d") or {}
            writer.writerow({
                "episode_id": item["episode_id"],
                "outcome": item["outcome"],
                "gate_status": item["gate_status"],
                "rejection_categories": ";".join(item["rejection_categories"]),
                "rejection_reasons": " | ".join(item["rejection_reasons"]),
                "frame_count": item["frame_count"],
                "selected_candidate": item["selected_candidate"],
                "selected_attempt": item["selected_attempt"],
                "slam_attempt_count": len(attempts),
                "slam_best_coverage": best.get("coverage"),
                "slam_best_tracked_frames": best.get("tracked_frames"),
                "slam_best_first_tracked_time_s": best.get("first_tracked_time_s"),
                "slam_tracking_failure_intervals": json.dumps(
                    best.get("tracking_failure_intervals", []), separators=(",", ":")),
                "jaw_marker_pair_detection_rate": grip["jaw_marker_pair_detection_rate"],
                "maximum_gap_missing_run_frames": grip["maximum_missing_run_frames"],
                "gap_min_mm": grip["gap_min_mm"],
                "gap_median_mm": grip["gap_median_mm"],
                "gap_max_mm": grip["gap_max_mm"],
                "fixed_table_marker_detection_rate": fixed.get("detection_rate"),
                "fixed_table_marker_maximum_missing_run_frames": fixed.get(
                    "maximum_missing_run_frames"),
                "start_tcp_xyz_m": json.dumps(
                    start.get("xyz_m"), separators=(",", ":")),
                "start_tcp_rotation_vector_rad": json.dumps(
                    start.get("rotation_vector_rad"), separators=(",", ":")),
                "first_training_anchor_tcp_xyz_m": json.dumps(
                    first_anchor.get("xyz_m"), separators=(",", ":")),
                "first_training_anchor_tcp_rotation_vector_rad": json.dumps(
                    first_anchor.get("rotation_vector_rad"), separators=(",", ":")),
            })
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
