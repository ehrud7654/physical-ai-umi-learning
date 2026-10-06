#!/usr/bin/env python3
"""Validate one ORB-SLAM3 + gripper UMI intake episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.orbslam_intake import validate_orbslam_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-episode", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--tracking", type=Path)
    parser.add_argument("--gripper-report", type=Path, required=True)
    parser.add_argument("--fixed-marker-id", type=int, required=True)
    parser.add_argument("--fixed-marker-size-m", type=float, required=True)
    parser.add_argument("--fixed-marker-size-verified", action="store_true")
    parser.add_argument("--jaw-marker-size-verified", action="store_true")
    parser.add_argument("--max-timestamp-error-ms", type=float, default=2.0)
    parser.add_argument("--max-first-pose-time-s", type=float, default=3.0)
    parser.add_argument("--min-pose-coverage", type=float, default=0.70)
    parser.add_argument("--min-pose-frames", type=int, default=24)
    parser.add_argument("--max-gripper-missing-run", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tracking = args.tracking or Path(str(args.trajectory) + ".tracking.csv")
    report = validate_orbslam_episode(
        args.raw_episode,
        args.trajectory,
        tracking,
        args.gripper_report,
        fixed_marker_id=args.fixed_marker_id,
        fixed_marker_size_m=args.fixed_marker_size_m,
        fixed_marker_size_verified=args.fixed_marker_size_verified,
        jaw_marker_size_verified=args.jaw_marker_size_verified,
        maximum_timestamp_error_ms=args.max_timestamp_error_ms,
        maximum_first_pose_time_s=args.max_first_pose_time_s,
        minimum_pose_coverage=args.min_pose_coverage,
        minimum_pose_frames=args.min_pose_frames,
        maximum_gripper_missing_run_frames=args.max_gripper_missing_run,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "PASS_ORB_INTAKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
