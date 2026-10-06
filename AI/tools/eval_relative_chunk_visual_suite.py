"""Run the five-condition local visual ablation for a relative-chunk policy.

The checkpoint, simulator state, registration and scoring remain identical.
Only inference RGB changes.  This wrapper is deliberately local-only and
delegates each episode to render_relative_chunk_rollout.py, whose shared-server
guard also rejects learned-checkpoint inference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


AI_ROOT = Path(__file__).resolve().parents[1]
RENDER_TOOL = AI_ROOT / "tools" / "render_relative_chunk_rollout.py"
PERTURB = {
    "normal": "none",
    "noise": "noise",
    "blackout": "blackout",
    "freeze": "freeze",
    "mean_fill": "mean_fill",
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [centre - radius, centre + radius]


def scene_valid(registration: dict[str, Any]) -> list[str]:
    rows = registration.get("selected", {}).get("per_episode", [])
    return [str(row["episode"]) for row in rows
            if isinstance(row, dict) and row.get("scene_constraints_ok") is True]


def classify(report: dict[str, Any]) -> str:
    if report.get("timing_error"):
        return "timing_reject"
    if report.get("object_tipped") is True:
        return "object_tipped"
    if report.get("object_displaced") is True:
        return "object_displaced"
    if report.get("stable_side_grasp_success") is True:
        return "success"
    if report.get("geometry_error"):
        return "geometry_error"
    if report.get("first_bilateral_contact_cycle") is None:
        return "no_bilateral_contact"
    if float(report.get("max_lift_height_m", 0.0)) > 0.005:
        return "partial_lift"
    return "contact_without_lift"


def parse_offset(value: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "object offsets must be X,Y pairs in metres, for example 0.01,-0.01"
        ) from exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--policy-ckpt", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--visual-domain", type=Path, required=True)
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--allow-rejected-diagnostic-subset", action="store_true")
    ap.add_argument("--conditions", nargs="+", choices=tuple(PERTURB),
                    default=list(PERTURB))
    ap.add_argument(
        "--object-offsets-m", action="append", type=parse_offset,
        default=None,
        help=("diagnostic object-only XY offsets as X,Y metre pairs; the robot "
              "start is deliberately not shifted; repeat this option for each pair"),
    )
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--far-start-audit", type=Path)
    ap.add_argument("--min-policy-cycles", type=int, default=1)
    ap.add_argument("--max-policy-time-scale", type=float)
    ap.add_argument("--execute-steps", type=int, default=1)
    ap.add_argument("--complete-execute-prefix", action="store_true")
    ap.add_argument("--validation-horizon-steps", type=int, default=1)
    ap.add_argument("--tracking-time-margin", type=float, default=1.0)
    ap.add_argument("--grip-preload-m", type=float, default=0.002)
    ap.add_argument("--grip-preload-activation-gap-m", type=float, default=0.055)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    object_offsets = args.object_offsets_m or [(0.0, 0.0)]
    if args.far_start_audit is not None and (
            object_offsets != [(0.0, 0.0)]
            or args.max_policy_time_scale is None):
        raise SystemExit("far-start suite requires nominal object and explicit time cap")
    if args.out.suffix.lower() != ".json" or args.out.exists():
        raise SystemExit("--out must be a new .json path")
    if not args.policy_ckpt.is_file():
        raise SystemExit(f"checkpoint is missing: {args.policy_ckpt}")

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    eligible = scene_valid(registration)
    if not eligible:
        raise SystemExit("registration has no scene-valid episodes")
    if registration.get("status") == "REJECTED_DIAGNOSTIC_CANONICAL_REPLAY":
        if not args.allow_rejected_diagnostic_subset or not args.episodes:
            raise SystemExit(
                "rejected diagnostic registration requires explicit --episodes "
                "and --allow-rejected-diagnostic-subset")
    episodes = list(args.episodes) if args.episodes else eligible
    unknown = sorted(set(episodes) - set(eligible))
    if unknown or len(episodes) != len(set(episodes)):
        raise SystemExit(f"invalid or duplicate episode ids: {unknown}")

    report_dir = args.out.parent / f"{args.out.stem}_episodes"
    if report_dir.exists():
        raise SystemExit(f"episode report directory exists: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=False)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in args.conditions:
        rows: list[dict[str, Any]] = []
        trials = [
            (episode, offset_x, offset_y)
            for episode in episodes
            for offset_x, offset_y in object_offsets
        ]
        for index, (episode, offset_x, offset_y) in enumerate(trials, start=1):
            offset_slug = f"x{offset_x:+.3f}_y{offset_y:+.3f}".replace(".", "p")
            pseudo = (
                report_dir / f"{condition}_{episode}_{offset_slug}.gif"
            ).resolve()
            command = [
                sys.executable, str(RENDER_TOOL),
                "--data", str(args.data.resolve()),
                "--policy-ckpt", str(args.policy_ckpt.resolve()),
                "--registration", str(args.registration.resolve()),
                "--device", "cpu",
                "--observation-source", "simulation",
                "--image-perturb", PERTURB[condition],
                "--episode", episode,
                "--cycles", str(args.cycles),
                "--execute-steps", str(args.execute_steps),
                "--validation-horizon-steps", str(args.validation_horizon_steps),
                "--no-video", "--capture-every", "1000000",
                "--tracking-time-margin", str(args.tracking_time_margin),
                "--grip-preload-m", str(args.grip_preload_m),
                "--grip-preload-activation-gap-m",
                str(args.grip_preload_activation_gap_m),
                "--object-offset-x-m", str(offset_x),
                "--object-offset-y-m", str(offset_y),
                "--visual-domain", str(args.visual_domain.resolve()),
                "--out", str(pseudo),
            ]
            if args.complete_execute_prefix:
                command.append("--complete-execute-prefix")
            if args.far_start_audit is None:
                command.append("--use-registration-approach-start")
            else:
                command.extend([
                    "--far-start-audit", str(args.far_start_audit.resolve()),
                    "--min-policy-cycles", str(args.min_policy_cycles),
                    "--max-policy-time-scale", str(args.max_policy_time_scale),
                ])
            completed = subprocess.run(
                command, cwd=AI_ROOT, capture_output=True, text=True, check=False)
            report_path = pseudo.with_suffix(".json")
            if completed.returncode != 0 or not report_path.is_file():
                row = {
                    "episode": episode,
                    "object_offset_xy_m": [offset_x, offset_y],
                    "classification": "runner_error",
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                }
            else:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                row = {
                    "episode": episode,
                    "object_offset_xy_m": [offset_x, offset_y],
                    "classification": classify(report),
                    "success": bool(report.get("stable_side_grasp_success")),
                    "geometry_error": report.get("geometry_error"),
                    "timing_error": report.get("timing_error"),
                    "executed_commands": report.get("executed_commands"),
                    "completed_cycles": len(report.get("cycle_trace", [])),
                    "bilateral_contact": report.get("first_bilateral_contact_cycle") is not None,
                    "max_lift_height_m": report.get("max_lift_height_m"),
                    "closest_pinch_3d_m": report.get("closest_pinch_3d_m"),
                    "episode_report": str(report_path),
                }
            rows.append(row)
            print(f"{condition} {index}/{len(trials)} {episode} "
                  f"offset=({offset_x:+.3f},{offset_y:+.3f}): "
                  f"{row['classification']}", flush=True)
        all_rows[condition] = rows

    summary: dict[str, Any] = {}
    for condition, rows in all_rows.items():
        successes = sum(row.get("success") is True for row in rows)
        total = len(rows)
        summary[condition] = {
            "successes": successes,
            "episodes": total,
            "success_fraction": successes / total,
            "wilson95": wilson(successes, total),
            "bilateral_contacts": sum(row.get("bilateral_contact") is True for row in rows),
            "geometry_errors": sum(bool(row.get("geometry_error")) for row in rows),
            "timing_rejections": sum(bool(row.get("timing_error")) for row in rows),
            "runner_errors": sum(row["classification"] == "runner_error" for row in rows),
            "classification_counts": {
                label: sum(row["classification"] == label for row in rows)
                for label in sorted({row["classification"] for row in rows})
            },
        }
    normal = summary.get("normal")
    if normal:
        for condition, row in summary.items():
            row["drop_from_normal_percentage_points"] = 100.0 * (
                normal["success_fraction"] - row["success_fraction"])

    result = {
        "status": "LOCAL_LEARNED_POLICY_VISUAL_ABLATION_DIAGNOSTIC",
        "policy_ckpt": str(args.policy_ckpt),
        "data": str(args.data),
        "registration": str(args.registration),
        "registration_status": registration.get("status"),
        "visual_domain": str(args.visual_domain),
        "episodes": episodes,
        "object_offsets_xy_m": [list(offset) for offset in object_offsets],
        "trials_per_condition": len(episodes) * len(object_offsets),
        "inference_device": "cpu",
        "same_checkpoint_all_conditions": True,
        "conditions_change_inference_rgb_only": True,
        "complete_execute_prefix_after_success": bool(
            args.complete_execute_prefix),
        "far_start_audit": (str(args.far_start_audit)
                            if args.far_start_audit is not None else None),
        "min_policy_cycles": args.min_policy_cycles,
        "max_policy_time_scale": args.max_policy_time_scale,
        "policy_rgb_rendered_with_no_video": True,
        "summary": summary,
        "per_condition": all_rows,
        "decision_rule": (
            "Normal must exceed spatial-information-free controls by >=15pp "
            "with non-overlapping Wilson 95% intervals; n=8 is diagnostic and "
            "normally underpowered for that claim."
        ),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
