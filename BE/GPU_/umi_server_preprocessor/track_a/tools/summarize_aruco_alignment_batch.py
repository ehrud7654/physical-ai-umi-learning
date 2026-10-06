#!/usr/bin/env python3
"""Summarize fixed-marker alignment reports for ORB quality-pass episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orb-batch-summary", type=Path, required=True)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    batch = json.loads(args.orb_batch_summary.read_text(encoding="utf-8"))
    episodes = []
    missing = []
    for item in batch.get("quality_pass", []):
        episode = item["episode"]
        path = args.alignment_root / episode / "aruco_alignment.json"
        if not path.is_file():
            missing.append(episode)
            continue
        alignment = json.loads(path.read_text(encoding="utf-8"))
        episodes.append({
            "episode": episode,
            "status": alignment["status"],
            "errors": alignment["errors"],
            "joint_pose_marker": alignment["observations"]["joint_pose_marker"],
            "robust_inliers": alignment["observations"]["robust_inliers"],
            "translation_residual_m": alignment["translation_residual_m"],
            "rotation_residual_deg": alignment["rotation_residual_deg"],
            "reprojection_error_px": alignment["reprojection_error_px"],
        })

    accepted = [
        item for item in episodes
        if item["status"] in {"PROVISIONAL_ARUCO_ALIGNMENT", "PASS_ARUCO_ALIGNMENT"}
    ]
    report = {
        "status": "INCOMPLETE" if missing else "COMPLETE",
        "input_quality_pass_episodes": len(batch.get("quality_pass", [])),
        "alignment_reports": len(episodes),
        "alignment_pass_episodes": len(accepted),
        "alignment_reject_episodes": len(episodes) - len(accepted),
        "missing_reports": missing,
        "accepted_episode_ids": [item["episode"] for item in accepted],
        "episodes": episodes,
        "interpretation": (
            "PROVISIONAL_ARUCO_ALIGNMENT is not final metric evidence until the "
            "physical fixed-marker and jaw-marker scales are verified"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "input_quality_pass_episodes", "alignment_reports",
        "alignment_pass_episodes", "alignment_reject_episodes", "missing_reports",
    )}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
