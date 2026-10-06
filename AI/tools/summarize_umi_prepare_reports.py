#!/usr/bin/env python3
"""Summarize telemetry/gripper preparation reports before ORB batch work."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.floor(fraction * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--maximum-missing-run", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    reports = []
    for path in sorted(args.processed_root.glob("*/prepare_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_path"] = str(path)
        reports.append(report)
    if not reports:
        raise SystemExit(f"no prepare_report.json under {args.processed_root}")

    accepted = []
    rejected = []
    missing_runs = []
    detection_rates = []
    scale_verified = 0
    for report in reports:
        episode = report["session"]
        gripper = report["gripper"]
        run = int(gripper["maximum_missing_run_frames"])
        rate = float(gripper["detection_rate"])
        missing_runs.append(run)
        detection_rates.append(rate)
        reasons = []
        if report.get("outcome") != "success":
            reasons.append(f"outcome={report.get('outcome')!r}")
        if gripper.get("status") != "pass":
            reasons.append(f"gripper_status={gripper.get('status')!r}")
        if run > args.maximum_missing_run:
            reasons.append(
                f"maximum_missing_run_frames={run}>{args.maximum_missing_run}"
            )
        if args.raw_root:
            manifest_path = args.raw_root / episode / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verified = bool(manifest.get("marker_black_square_size_verified", False))
            scale_verified += int(verified)
            if not verified:
                reasons.append("jaw_marker_metric_scale_unverified")
        item = {
            "episode": episode,
            "frames": int(report["frame_count"]),
            "detection_rate": rate,
            "maximum_missing_run_frames": run,
            "reasons": reasons,
        }
        if reasons:
            rejected.append(item)
        else:
            accepted.append(item)

    gap_candidates = [item for item in accepted + rejected
                      if not any(reason.startswith("maximum_missing_run_frames=")
                                 for reason in item["reasons"])]
    result = {
        "status": (
            "PREPROCESS_COMPLETE_SCALE_UNVERIFIED"
            if args.raw_root and scale_verified < len(reports)
            else "PREPROCESS_COMPLETE"
        ),
        "processed_root": str(args.processed_root),
        "reports": len(reports),
        "frames_total": sum(int(report["frame_count"]) for report in reports),
        "gap_gate": {
            "maximum_missing_run_frames": args.maximum_missing_run,
            "candidate_episodes": len(gap_candidates),
            "rejected_episodes": len(reports) - len(gap_candidates),
        },
        "jaw_marker_scale": {
            "verified_episodes": scale_verified if args.raw_root else None,
            "unverified_episodes": (len(reports) - scale_verified) if args.raw_root else None,
        },
        "maximum_missing_run_frames": {
            "min": min(missing_runs),
            "p50": percentile(missing_runs, 0.50),
            "p95": percentile(missing_runs, 0.95),
            "max": max(missing_runs),
        },
        "detection_rate": {
            "min": min(detection_rates),
            "p05": percentile(detection_rates, 0.05),
            "p50": percentile(detection_rates, 0.50),
            "max": max(detection_rates),
        },
        "accepted_after_all_requested_gates": accepted,
        "rejected_or_blocked": rejected,
        "interpretation": (
            "gap-gate candidates are not final dataset episodes until metric scale, "
            "ORB pose, ArUco alignment and timestamp gates also pass"
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "reports": result["reports"],
        "frames_total": result["frames_total"],
        "gap_gate": result["gap_gate"],
        "jaw_marker_scale": result["jaw_marker_scale"],
        "missing_run_distribution": result["maximum_missing_run_frames"],
        "detection_rate_distribution": result["detection_rate"],
        "out": str(args.out) if args.out else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
