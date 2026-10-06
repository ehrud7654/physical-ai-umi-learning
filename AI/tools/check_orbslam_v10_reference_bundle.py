"""Verify a Track-A ORB-SLAM/v10 reference bundle without rerunning SLAM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    root = args.bundle
    manifest_path = root / "manifest.json"
    verification_path = root / "verification_reference.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    if manifest.get("schema") not in {
            "track_a_orbslam_v10_reference/0.1",
            "track_a_orbslam_v10_reference/0.2",
            "track_a_orbslam_v10_reference/0.3"}:
        problems.append("unexpected manifest schema")
    gate_report = manifest.get("episode_gate_report", {})
    if manifest.get("schema") == "track_a_orbslam_v10_reference/0.3":
        if not gate_report.get("included"):
            problems.append("0.3 bundle is missing episode gate report")
        for label in ("json", "csv"):
            info = gate_report.get("files", {}).get(label)
            if not info:
                problems.append(f"missing episode gate report {label} metadata")
                continue
            path = root / info["path"]
            if not path.is_file():
                problems.append(f"missing episode gate report {label} file")
            elif sha256_file(path) != info["sha256"]:
                problems.append(f"episode gate report {label} hash mismatch")
    episodes = manifest.get("episodes", [])
    order = manifest.get("selection", {}).get("episode_order", [])
    if [item.get("episode_id") for item in episodes] != order:
        problems.append("episode order differs from manifest selection order")

    total_pose = total_gripper = total_x = 0
    for episode in episodes:
        episode_id = episode.get("episode_id", "unknown")
        pose_info = episode["pose_csv"]
        grip_info = episode["gripper_csv"]
        if manifest.get("schema") == "track_a_orbslam_v10_reference/0.3":
            if not pose_info.get("start_tcp_pose_6d"):
                problems.append(f"{episode_id}: missing full-episode start TCP pose")
            v10_info = episode.get("v10", {})
            if not v10_info.get("start_tcp_pose_6d"):
                problems.append(f"{episode_id}: missing selected-segment start TCP pose")
            if not v10_info.get("first_training_anchor_tcp_pose_6d"):
                problems.append(f"{episode_id}: missing first-anchor TCP pose")
        pose_path = root / pose_info["path"]
        grip_path = root / grip_info["path"]
        for label, path, info in (
            ("pose", pose_path, pose_info), ("gripper", grip_path, grip_info)
        ):
            if not path.is_file():
                problems.append(f"{episode_id}: missing {label}.csv")
                continue
            if sha256_file(path) != info["sha256"]:
                problems.append(f"{episode_id}: {label}.csv hash mismatch")
        if not pose_path.is_file() or not grip_path.is_file():
            continue
        pose_rows = csv_rows(pose_path)
        grip_rows = csv_rows(grip_path)
        total_pose += len(pose_rows)
        total_gripper += len(grip_rows)
        if len(pose_rows) != int(pose_info["rows"]):
            problems.append(f"{episode_id}: pose row count mismatch")
        if len(grip_rows) != int(grip_info["rows"]):
            problems.append(f"{episode_id}: gripper row count mismatch")
        if len(pose_rows) != len(grip_rows):
            problems.append(f"{episode_id}: pose/gripper row count differs")
            continue
        last_timestamp = None
        counts = {"D": 0, "M": 0, "X": 0}
        for expected, (pose, grip) in enumerate(zip(pose_rows, grip_rows)):
            if int(pose["frame_index"]) != expected or int(grip["frame_index"]) != expected:
                problems.append(f"{episode_id}: non-contiguous index at {expected}")
                break
            if pose["sensor_timestamp_ns"] != grip["sensor_timestamp_ns"]:
                problems.append(f"{episode_id}: pose/gripper timestamp mismatch at {expected}")
                break
            timestamp = int(pose["sensor_timestamp_ns"])
            if last_timestamp is not None and timestamp <= last_timestamp:
                problems.append(f"{episode_id}: non-monotonic timestamp at {expected}")
                break
            last_timestamp = timestamp
            status = grip["status"]
            if status not in counts:
                problems.append(f"{episode_id}: invalid gripper status {status!r}")
                break
            counts[status] += 1
            if status != "X":
                gap = float(grip["gap_m"])
                if not 0 <= gap <= 0.09:
                    problems.append(f"{episode_id}: gap outside [0,0.09] at {expected}")
                    break
        if counts != grip_info["status_counts"]:
            problems.append(f"{episode_id}: gripper status counts mismatch")
        total_x += counts["X"]

    expected = verification.get("checks", {})
    if total_pose != expected.get("pose_rows"):
        problems.append("total pose rows differ from verification reference")
    if total_gripper != expected.get("gripper_rows"):
        problems.append("total gripper rows differ from verification reference")
    if total_x != expected.get("invalid_gap_rows"):
        problems.append("total invalid gap rows differ from verification reference")
    if sha256_file(manifest_path) != verification.get("manifest_sha256"):
        problems.append("manifest hash differs from verification reference")

    report = {
        "status": "PASS_TRACK_A_REFERENCE_BUNDLE" if not problems else "FAIL_TRACK_A_REFERENCE_BUNDLE",
        "episodes": len(episodes),
        "pose_rows": total_pose,
        "gripper_rows": total_gripper,
        "invalid_gap_rows": total_x,
        "problems": problems,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
