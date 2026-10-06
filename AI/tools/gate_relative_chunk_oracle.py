"""Run a local multi-episode dynamic gate for recorded relative UMI targets.

This tool never loads a learned policy.  It replays recorded target chunks in
MuJoCo without rendering, then aggregates contact, lift, IK and displacement
outcomes for the registration's scene-valid episodes.  Passing this diagnostic
is necessary before learned-policy rollout, but is not hardware validation.
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
DEFAULT_CONFIG = AI_ROOT / "configs" / "so101.yaml"
DEFAULT_REAL_CONFIG = AI_ROOT / "configs" / "real" / "so101_ver1.json"
RENDER_TOOL = AI_ROOT / "tools" / "render_relative_chunk_rollout.py"
ACCEPTED_PROVISIONAL_REGISTRATION_STATUSES = {
    "PROVISIONAL_NOT_PHYSICAL_CALIBRATION",
    "DIAGNOSTIC_COLLISION_PROXY_CANDIDATE_NOT_PHYSICAL_VALIDATION",
    "DIAGNOSTIC_CANONICAL_REPLAY_NOT_REGISTRATION",
}


def _scene_valid_episodes(registration: dict[str, Any]) -> list[str]:
    selected = registration.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("registration has no selected candidate report")
    rows = selected.get("per_episode")
    if not isinstance(rows, list):
        raise ValueError("registration has no per-episode scene report")
    episodes = [
        str(row["episode"])
        for row in rows
        if isinstance(row, dict) and row.get("scene_constraints_ok") is True
    ]
    if not episodes:
        raise ValueError("registration has no scene-valid episodes to gate")
    return episodes


def _require_provisional_registration(
        registration: dict[str, Any], *,
        allow_rejected_diagnostic_subset: bool = False) -> None:
    status = registration.get("status")
    if (allow_rejected_diagnostic_subset
            and status == "REJECTED_DIAGNOSTIC_CANONICAL_REPLAY"):
        return
    if status not in ACCEPTED_PROVISIONAL_REGISTRATION_STATUSES:
        raise ValueError(
            "registration must be an explicitly accepted provisional "
            f"diagnostic, got {status!r}"
        )


def _classification(report: dict[str, Any]) -> str:
    # A transient lift must not hide a tip or large horizontal shove.  The
    # dynamic gate is specifically the stable side-grasp gate, while the raw
    # MuJoCo success bit remains available in each episode report.
    if report.get("object_tipped") is True:
        return "object_tipped"
    if report.get("object_displaced") is True:
        return "object_displaced"
    if report.get("stable_side_grasp_success", report.get("success")) is True:
        return "success"
    contacted = report.get("first_contact_cycle") is not None
    geometry_error = report.get("geometry_error")
    if geometry_error:
        return "ik_reject_after_contact" if contacted else "ik_reject_before_contact"
    if not contacted:
        return "no_contact"
    final_lift = float(report.get("lift_height_m", 0.0))
    min_lift = float(report.get("min_lift_height_m", min(0.0, final_lift)))
    max_lift = float(report.get("max_lift_height_m", max(0.0, final_lift)))
    if min_lift <= -0.01:
        return "object_displaced_or_tipped"
    if max_lift > 0.005:
        return "partial_lift"
    return "contact_without_lift"


def _finite_nonnegative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _kinematic_grasp_args(path: Path | None) -> list[str]:
    """Build the explicit renderer override without changing old registrations."""
    if path is None:
        return []
    return ["--kinematic-grasp", str(path.resolve())]


def _aggregate(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    """Aggregate dynamic outcomes without hiding simultaneous failures."""
    if not rows:
        raise ValueError("dynamic oracle gate needs at least one episode")
    successes = sum(row.get("success") is True for row in rows)
    contacts = sum(row.get("first_contact_cycle") is not None for row in rows)
    runner_errors = sum(row["classification"] == "runner_error" for row in rows)
    success_fraction = successes / len(rows)
    counts = {
        label: sum(row["classification"] == label for row in rows)
        for label in sorted({row["classification"] for row in rows})
    }
    failure_counts = {
        "runner_error": sum(
            row["classification"] == "runner_error" for row in rows),
        "ik_reject_after_contact": sum(
            row["classification"] == "ik_reject_after_contact" for row in rows),
        "object_tipped": sum(
            row.get("object_tipped") is True
            or row["classification"] == "object_tipped" for row in rows),
        "object_displaced": sum(
            row.get("object_displaced") is True
            or row["classification"] == "object_displaced" for row in rows),
        "static_scene_reject": sum(
            row["classification"] == "static_scene_reject" for row in rows),
    }
    return {
        "episodes": len(rows),
        "contacts": contacts,
        "successes": successes,
        "success_fraction": success_fraction,
        "required_success_fraction": float(threshold),
        "runner_errors": runner_errors,
        "classification_counts": counts,
        "failure_counts": failure_counts,
        "passed": runner_errors == 0 and success_fraction >= threshold,
    }


def _run_recorded_oracle_episode(
        *, data: Path, registration_path: Path, episode: str,
        report_dir: Path, cycles: int, tracking_time_margin: float,
        grip_preload_m: float, grip_preload_activation_gap_m: float,
        config: Path, real_config: Path,
        kinematic_grasp: Path | None = None,
        use_registration_approach_start: bool = False) -> dict[str, Any]:
    """Run one local recorded-oracle episode and return its classified report."""
    # The child runs with AI_ROOT as cwd.  An output rooted at a relative
    # `AI/out/...` directory would otherwise become `AI/AI/out/...` on Windows,
    # while the parent checks the original path and reports a false runner
    # error even though the child completed successfully.
    pseudo_video = (report_dir / f"{episode}.gif").resolve()
    command = [
        sys.executable,
        str(RENDER_TOOL),
        "--data", str(data.resolve()),
        "--registration", str(registration_path.resolve()),
        "--device", "cpu",
        "--observation-source", "dataset",
        "--oracle",
        "--episode", episode,
        "--cycles", str(cycles),
        "--execute-steps", "1",
        "--validation-horizon-steps", "1",
        "--execute-final-oracle-tail",
        "--no-video",
        "--capture-every", "1000000",
        "--tracking-time-margin", str(tracking_time_margin),
        "--grip-preload-m", str(grip_preload_m),
        "--grip-preload-activation-gap-m", str(
            grip_preload_activation_gap_m),
        "--config", str(config.resolve()),
        "--real-config", str(real_config.resolve()),
        "--out", str(pseudo_video),
    ]
    command.extend(_kinematic_grasp_args(kinematic_grasp))
    if use_registration_approach_start:
        command.append("--use-registration-approach-start")
    completed = subprocess.run(
        command, cwd=AI_ROOT, capture_output=True, text=True, check=False)
    report_path = pseudo_video.with_suffix(".json")
    if completed.returncode != 0 or not report_path.is_file():
        return {
            "episode": episode,
            "classification": "runner_error",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
            "stdout_tail": completed.stdout[-2000:],
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "episode": episode,
        "classification": _classification(report),
        "success": bool(report.get(
            "stable_side_grasp_success", report["success"])),
        "raw_lift_contact_success": bool(report["success"]),
        "first_contact_cycle": report["first_contact_cycle"],
        "first_bilateral_contact_cycle": report.get(
            "first_bilateral_contact_cycle"),
        "bilateral_contact_ticks": report.get("bilateral_contact_ticks"),
        "closest_pinch_xy_m": report["closest_pinch_xy_m"],
        "final_lift_height_m": report["lift_height_m"],
        "max_lift_height_m": report["max_lift_height_m"],
        "min_lift_height_m": report["min_lift_height_m"],
        "max_object_xy_displacement_m": report.get(
            "max_object_xy_displacement_m"),
        "max_pre_lift_object_xy_displacement_m": report.get(
            "max_pre_lift_object_xy_displacement_m"),
        "tabletop_motion_lift_cutoff_m": report.get(
            "tabletop_motion_lift_cutoff_m"),
        "max_object_tilt_deg": report.get("max_object_tilt_deg"),
        "object_xy_motion_exceeded_threshold": report.get(
            "object_xy_motion_exceeded_threshold"),
        "total_object_xy_motion_exceeded_threshold": report.get(
            "total_object_xy_motion_exceeded_threshold"),
        "object_displaced": report.get("object_displaced"),
        "object_tipped": report.get("object_tipped"),
        "jaw_contacts_final": report["jaw_contacts_final"],
        "jaw_contact_counts_final": report.get("jaw_contact_counts_final"),
        "bilateral_jaw_contact_final": report.get(
            "bilateral_jaw_contact_final"),
        "geometry_error": report["geometry_error"],
        "episode_report": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--episodes", nargs="*",
                        help="explicit episode ids; default is every scene-valid episode")
    parser.add_argument(
        "--allow-rejected-diagnostic-subset", action="store_true",
        help=(
            "allow an explicitly listed scene-valid subset from a rejected "
            "diagnostic canonical replay; never promotes the source report"
        ),
    )
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--tracking-time-margin", type=float, default=1.0)
    parser.add_argument("--grip-preload-m", type=float, default=0.001)
    parser.add_argument("--grip-preload-activation-gap-m", type=float, default=0.055)
    parser.add_argument(
        "--use-registration-approach-start", action="store_true",
        help="start each oracle replay from its fixed static-gate approach row",
    )
    parser.add_argument("--min-success-fraction", type=float, default=0.8)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--real-config", type=Path, default=DEFAULT_REAL_CONFIG)
    parser.add_argument(
        "--kinematic-grasp", type=Path,
        help=(
            "override a stale registration's kinematic block with the current "
            "visible-hand/collision payload"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.suffix.lower() != ".json":
        raise SystemExit("--out must end in .json")
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    if args.cycles <= 0:
        raise SystemExit("--cycles must be positive")
    preload = _finite_nonnegative("grip-preload-m", args.grip_preload_m)
    activation = _finite_nonnegative(
        "grip-preload-activation-gap-m", args.grip_preload_activation_gap_m)
    margin = float(args.tracking_time_margin)
    threshold = float(args.min_success_fraction)
    if not math.isfinite(margin) or margin < 1.0:
        raise SystemExit("--tracking-time-margin must be finite and >= 1")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit("--min-success-fraction must be within [0, 1]")

    registration_path = args.registration.resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    _require_provisional_registration(
        registration,
        allow_rejected_diagnostic_subset=args.allow_rejected_diagnostic_subset,
    )
    rejected_subset = (
        registration.get("status") == "REJECTED_DIAGNOSTIC_CANONICAL_REPLAY"
    )
    if rejected_subset and not args.episodes:
        raise ValueError(
            "a rejected diagnostic registration requires explicit --episodes"
        )
    eligible = _scene_valid_episodes(registration)
    episodes = list(args.episodes) if args.episodes else eligible
    unknown = sorted(set(episodes) - set(eligible))
    if unknown:
        raise ValueError(
            "explicit episodes are not scene-valid in this registration: "
            + ", ".join(unknown))
    if len(set(episodes)) != len(episodes):
        raise ValueError("episode ids must be unique")

    report_dir = (
        args.out.parent / f"{args.out.stem}_episodes"
    ).resolve()
    if report_dir.exists():
        raise SystemExit(f"episode report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        row = _run_recorded_oracle_episode(
            data=args.data, registration_path=registration_path,
            episode=episode, report_dir=report_dir, cycles=args.cycles,
            tracking_time_margin=margin, grip_preload_m=preload,
            grip_preload_activation_gap_m=activation,
            config=args.config, real_config=args.real_config,
            kinematic_grasp=args.kinematic_grasp,
            use_registration_approach_start=(
                args.use_registration_approach_start))
        rows.append(row)
        print(
            f"{index}/{len(episodes)} {episode}: {row['classification']}",
            flush=True,
        )

    aggregate = _aggregate(rows, threshold=threshold)
    passed = bool(aggregate["passed"])
    result = {
        "status": (
            "PASS_LOCAL_DYNAMIC_ORACLE_GATE"
            if passed else "FAIL_LOCAL_DYNAMIC_ORACLE_GATE"
        ),
        "purpose": "recorded-oracle dynamic registration gate; not policy or hardware validation",
        "dataset": str(args.data),
        "registration": str(args.registration),
        "source_registration_status": registration.get("status"),
        "evaluated_static_subset_of_rejected_registration": rejected_subset,
        "kinematic_grasp_override": (
            str(args.kinematic_grasp) if args.kinematic_grasp else None),
        "eligible_scene_valid_episodes": eligible,
        "evaluated_episodes": episodes,
        "episodes": aggregate["episodes"],
        "contacts": aggregate["contacts"],
        "successes": aggregate["successes"],
        "success_fraction": aggregate["success_fraction"],
        "required_success_fraction": aggregate["required_success_fraction"],
        "runner_errors": aggregate["runner_errors"],
        "simulation_grip_preload_m": preload,
        "simulation_grip_preload_activation_gap_m": activation,
        "registration_approach_start_used": bool(
            args.use_registration_approach_start),
        "simulation_tracking_time_margin": margin,
        "simulation_waypoint_execution": (
            "linear arm-and-gap target interpolation across scheduled ticks"
        ),
        "classification_counts": aggregate["classification_counts"],
        "failure_counts": aggregate["failure_counts"],
        "per_episode": rows,
        "limitations": [
            "Recorded target chunks are replayed; no learned policy is loaded.",
            "The configured preload is a MuJoCo-only contact diagnostic, not hardware calibration.",
            "Passing does not approve real motor commands or establish physical workspace calibration.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
