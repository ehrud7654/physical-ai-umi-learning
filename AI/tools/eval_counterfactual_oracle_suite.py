"""Resumable local, policy-free displaced-object recorded-oracle suite.

This evaluates existing v10 recorded targets, not a learned checkpoint.
The same fixed robot start, provisional contact proxy, and simulation-only
preload are used across object-only offsets. Reports never approve training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def parse_offset(value: str) -> tuple[float, float]:
    try:
        x, y = (float(item) for item in value.split(",", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("offset must be X,Y metres") from exc
    if not math.isfinite(x) or not math.isfinite(y) or abs(x) > 0.03 or abs(y) > 0.03:
        raise argparse.ArgumentTypeError("offsets must stay within +/-30 mm")
    return x, y


def _slug(value: float) -> str:
    return f"{value:+.3f}".replace("-", "minus").replace("+", "plus").replace(".", "p")


def _record(report: dict) -> dict:
    return {
        "episode": report["representative_episode"],
        "offset_xy_m": report["diagnostic_object_offset_xy_m"],
        "report": report.get("report_path"),
        "executed_commands": report["executed_commands"],
        "geometry_error": report["geometry_error"],
        "full_trajectory_geometry_valid": report["geometry_error"] is None,
        "bilateral_contact_ticks": report["bilateral_contact_ticks"],
        "final_bilateral_contact": report["bilateral_jaw_contact_final"],
        "final_lift_m": report["lift_height_m"],
        "max_pre_lift_xy_displacement_m": report[
            "max_pre_lift_object_xy_displacement_m"],
        "max_tilt_deg": report["max_object_tilt_deg"],
        "stable_side_grasp_success": report["stable_side_grasp_success"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--visual-domain", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--offset-m", action="append", type=parse_offset,
                        required=True)
    parser.add_argument("--grip-preload-m", type=float, default=0.002)
    parser.add_argument("--max-new-cases", type=int,
                        help="stop after this many newly run cases; resume later")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.episodes) != len(set(args.episodes)):
        raise SystemExit("episode ids must be unique")
    if len(args.offset_m) != len(set(args.offset_m)):
        raise SystemExit("offsets must be unique")
    if args.max_new_cases is not None and args.max_new_cases <= 0:
        raise SystemExit("max-new-cases must be positive")
    if not 0 <= args.grip_preload_m <= 0.005:
        raise SystemExit("simulation-only preload must be between 0 and 5 mm")
    if args.out.suffix.lower() != ".json":
        raise SystemExit("--out must end in .json")

    config = {
        "data": str(args.data),
        "registration": str(args.registration),
        "visual_domain": str(args.visual_domain),
        "episodes": args.episodes,
        "offsets_xy_m": [list(value) for value in args.offset_m],
        "grip_preload_m": args.grip_preload_m,
        "execute_steps": 1,
        "validation_horizon_steps": 8,
        "execute_final_oracle_tail": True,
        "object_only_shift": True,
        "target_shift_once_in_world": True,
    }
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("conditions") != config:
            raise SystemExit("existing suite has different conditions")
    case_dir = args.out.with_name(args.out.stem + "_episodes")
    case_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    new_cases = 0
    total = len(args.episodes) * len(args.offset_m)
    for episode in args.episodes:
        for offset_x, offset_y in args.offset_m:
            base = case_dir / (
                f"{episode}_x{_slug(offset_x)}_y{_slug(offset_y)}")
            report_path = base.with_suffix(".json")
            if not report_path.exists():
                if args.max_new_cases is not None and new_cases >= args.max_new_cases:
                    continue
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("render_relative_chunk_rollout.py")),
                    "--data", str(args.data),
                    "--registration", str(args.registration),
                    "--visual-domain", str(args.visual_domain),
                    "--oracle", "--oracle-shift-target-with-object",
                    "--episode", episode,
                    "--use-registration-approach-start",
                    "--cycles", "30", "--execute-steps", "1",
                    "--validation-horizon-steps", "8",
                    "--execute-final-oracle-tail",
                    "--grip-preload-m", str(args.grip_preload_m),
                    "--no-video",
                    "--object-offset-x-m", str(offset_x),
                    "--object-offset-y-m", str(offset_y),
                    "--out", str(base.with_suffix(".gif")),
                ]
                process = subprocess.run(command, capture_output=True, text=True)
                if process.returncode:
                    raise RuntimeError(
                        f"{episode} {offset_x},{offset_y}: "
                        f"{process.stderr[-1200:]} {process.stdout[-1200:]}")
                new_cases += 1
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("representative_episode") != episode
                    or report.get("diagnostic_object_offset_xy_m")
                    != [offset_x, offset_y]
                    or report.get("counterfactual_oracle_target_shift_once") is not True
                    or report.get("full_chunk_validation") is not True
                    or report.get("final_oracle_tail_executed") is not True):
                raise ValueError(f"case report has mismatched conditions: {report_path}")
            item = _record(report)
            item["report"] = str(report_path)
            results.append(item)
            print(f"{len(results)}/{total} {episode} ({offset_x:+.3f},{offset_y:+.3f}) "
                  f"stable={item['stable_side_grasp_success']} "
                  f"lift={item['final_lift_m']:.3f}m", flush=True)
            aggregate = {
                "status": "COMPLETE_POLICY_FREE_COUNTERFACTUAL_DIAGNOSTIC"
                          if len(results) == total else "PARTIAL_RESUMABLE",
                "conditions": config,
                "completed_cases": len(results),
                "total_cases": total,
                "stable_lifts": sum(bool(value["stable_side_grasp_success"])
                                    for value in results),
                "stable_lifts_with_full_geometry": sum(
                    bool(value["stable_side_grasp_success"])
                    and value["full_trajectory_geometry_valid"]
                    for value in results),
                "geometry_errors": sum(value["geometry_error"] is not None
                                       for value in results),
                "training_ready": False,
                "cases": results,
            }
            args.out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2)
                                + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
