#!/usr/bin/env python3
"""Run the strict intake validator over a prepared ORB-SLAM3 batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.orbslam_intake import validate_orbslam_episode


SCALE_ERRORS = {
    "jaw marker metric scale is not verified",
    "fixed ArUco marker metric size is provisional, not verified",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--orb-root", type=Path, required=True)
    parser.add_argument("--fixed-marker-id", type=int, default=13)
    parser.add_argument("--fixed-marker-size-m", type=float, default=0.16)
    parser.add_argument("--fixed-marker-size-verified", action="store_true")
    parser.add_argument("--jaw-marker-size-verified", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results = []
    missing_outputs = []
    for prepared in sorted(args.processed_root.iterdir()):
        if not prepared.is_dir():
            continue
        gripper_path = prepared / "gripper_report.json"
        if not gripper_path.exists():
            continue
        gripper = json.loads(gripper_path.read_text(encoding="utf-8"))
        if int(gripper["maximum_missing_run_frames"]) > 7:
            continue
        name = prepared.name
        trajectory = args.orb_root / name / "camera_trajectory.csv"
        tracking = Path(str(trajectory) + ".tracking.csv")
        if not trajectory.exists() or not tracking.exists():
            missing_outputs.append(name)
            continue
        report = validate_orbslam_episode(
            args.raw_root / name,
            trajectory,
            tracking,
            gripper_path,
            fixed_marker_id=args.fixed_marker_id,
            fixed_marker_size_m=args.fixed_marker_size_m,
            fixed_marker_size_verified=args.fixed_marker_size_verified,
            jaw_marker_size_verified=args.jaw_marker_size_verified,
        )
        quality_errors = [error for error in report["errors"] if error not in SCALE_ERRORS]
        scale_errors = [error for error in report["errors"] if error in SCALE_ERRORS]
        results.append({
            "episode": name,
            "quality_status": "PASS_QUALITY" if not quality_errors else "REJECT_QUALITY",
            "quality_errors": quality_errors,
            "scale_blockers": scale_errors,
            "pose": report["pose"],
            "camera": report["camera"],
            "gripper": report["gripper"],
        })

    quality_pass = [item for item in results if item["quality_status"] == "PASS_QUALITY"]
    quality_reject = [item for item in results if item["quality_status"] == "REJECT_QUALITY"]
    report = {
        "status": (
            "BATCH_INCOMPLETE" if missing_outputs else
            "QUALITY_COMPLETE_SCALE_UNVERIFIED"
            if not (args.fixed_marker_size_verified and args.jaw_marker_size_verified) else
            "QUALITY_COMPLETE"
        ),
        "candidate_episodes": len(results) + len(missing_outputs),
        "processed_orb_episodes": len(results),
        "missing_orb_outputs": missing_outputs,
        "quality_pass_episodes": len(quality_pass),
        "quality_reject_episodes": len(quality_reject),
        "fixed_marker": {
            "id": args.fixed_marker_id,
            "black_square_size_m": args.fixed_marker_size_m,
            "physical_size_verified": args.fixed_marker_size_verified,
        },
        "jaw_marker": {
            "physical_scale_verified": args.jaw_marker_size_verified,
        },
        "quality_pass": quality_pass,
        "quality_reject": quality_reject,
        "interpretation": (
            "PASS_QUALITY excludes metric-scale blockers only; no episode is final until "
            "the fixed-board and jaw-marker physical scales are verified by explicit "
            "calibration evidence; raw manifests are never rewritten"
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "status", "candidate_episodes", "processed_orb_episodes",
            "quality_pass_episodes", "quality_reject_episodes",
        )
    }, ensure_ascii=False, indent=2))
    return 0 if not missing_outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
