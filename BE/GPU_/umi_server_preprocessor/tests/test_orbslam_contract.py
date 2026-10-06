#!/usr/bin/env python3
"""Dependency-light checks for the v2 ORB-SLAM backend contract."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_builder/slam"))

import validate_orbslam_trajectory  # noqa: E402


FIELDS = (
    "frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
    "x", "y", "z", "q_x", "q_y", "q_z", "q_w",
)


def write_trajectory(path: Path, positions: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, x in enumerate(positions):
            writer.writerow({
                "frame_idx": index,
                "timestamp": index / 30,
                "state": 2,
                "is_lost": "false",
                "is_keyframe": str(index == 0).lower(),
                "x": x, "y": 0, "z": 0,
                "q_x": 0, "q_y": 0, "q_z": 0, "q_w": 1,
            })


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        normal = Path(directory) / "normal.csv"
        write_trajectory(normal, [0.0, 0.01, 0.02])
        report = validate_orbslam_trajectory.validate(normal)
        assert report["status"] == "pass"
        assert report["pose_continuity"]["status"] == "pass"
        assert report["pose_coverage"] == 1.0
        assert report["final_frame_tracked"] is True
        assert report["first_tracked_pose"]["position_m"] == [0.0, 0.0, 0.0]

        jumped = Path(directory) / "jumped.csv"
        write_trajectory(jumped, [0.0, 1.0, 1.01])
        report = validate_orbslam_trajectory.validate(jumped)
        assert report["status"] == "fail"
        assert report["pose_continuity"]["status"] == "fail"
        assert any("pose translation jump" in error for error in report["errors"])

    linux_runner = (ROOT / "dataset_builder/slam/run_orbslam_linux.sh").read_text(encoding="utf-8")
    assert '--load_map "$ORB_SLAM_LOAD_MAP"' in linux_runner
    assert '--load_map "$ORB_SLAM_LOAD_MAP" --localization_only' not in linux_runner
    assert "--use_sensor_timestamps --save_map --load_map" in linux_runner
    print("ORB_SLAM_CONTRACT_SELF_TEST_OK")


if __name__ == "__main__":
    main()
