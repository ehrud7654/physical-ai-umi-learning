"""Resumable, policy-free exact-10Hz MuJoCo candidate sweep.

Every declared condition stays in the manifest, including failed or interrupted
conditions. Outputs are diagnostics, never human demonstrations or training data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_sim_relative_10hz_candidate import check


def parse_offset(value: str) -> tuple[float, float]:
    try:
        x, y = (float(item) for item in value.split(",", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("offset must be X,Y metres") from exc
    if not math.isfinite(x) or not math.isfinite(y) or abs(x) > 0.03 or abs(y) > 0.03:
        raise argparse.ArgumentTypeError("offsets must stay within +/-30mm")
    return x, y


def _slug(value: float) -> str:
    return f"{value:+.3f}".replace("-", "minus").replace("+", "plus").replace(".", "p")


def _case_paths(case_dir: Path, episode: str, offset: tuple[float, float],
                attempt: int = 0) -> tuple[Path, Path]:
    base = case_dir / f"{episode}_x{_slug(offset[0])}_y{_slug(offset[1])}"
    if attempt:
        base = base.with_name(base.name + f"_retry{attempt}")
    return base.with_suffix(".json"), base.with_name(base.name + "_candidate.npz")


def _resume_paths(case_dir: Path, episode: str,
                  offset: tuple[float, float]) -> tuple[Path, Path, int]:
    """Reuse a complete attempt; skip partial artifacts without deleting them."""
    for attempt in range(100):
        report, candidate = _case_paths(case_dir, episode, offset, attempt)
        if (report.exists() and candidate.exists()
                and candidate.with_suffix(".json").exists()):
            return report, candidate, attempt
        if not (report.exists() or candidate.exists()
                or candidate.with_suffix(".json").exists()):
            return report, candidate, attempt
    raise RuntimeError("too many preserved partial case attempts")


def _inspect_case(report_path: Path, candidate_path: Path, *, episode: str,
                  offset: tuple[float, float], config: dict) -> dict:
    case = {
        "episode": episode,
        "offset_xy_m": list(offset),
        "report": str(report_path),
        "candidate": str(candidate_path),
        "training_ready": False,
    }
    if not report_path.exists() or not candidate_path.exists() or not candidate_path.with_suffix(".json").exists():
        case["status"] = "INCOMPLETE_OUTPUT"
        case["problems"] = ["report, candidate NPZ, or candidate sidecar is missing"]
        return case
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        meta = json.loads(candidate_path.with_suffix(".json").read_text(encoding="utf-8"))
        if (report.get("representative_episode") != episode
                or report.get("diagnostic_object_offset_xy_m") != list(offset)
                or report.get("dataset") != config["data"]
                or report.get("registration") != config["registration"]
                or report.get("mode") != "recorded_oracle"
                or report.get("observation_source") != "simulation"
                or report.get("counterfactual_oracle_target_shift_once") is not True
                or report.get("full_chunk_validation") is not True
                or report.get("final_oracle_tail_executed") is not True
                or not math.isclose(report.get("simulation_grip_preload_m", float("nan")),
                                    config["grip_preload_m"], abs_tol=1e-9)
                or meta.get("source_oracle_report") != str(report_path)
                or meta.get("source_visual_domain") != config["visual_domain"]
                or meta.get("source_episode") != episode
                or meta.get("object_offset_world_xy_m") != list(offset)
                or meta.get("control_rate_hz") != 50.0):
            case["status"] = "CONDITION_MISMATCH"
            case["problems"] = ["stored report/candidate does not match declared suite"]
            return case
        validation = check(candidate_path)
        case.update({
            "status": ("CANDIDATE_CHECKED" if not validation["problems"]
                       else "CANDIDATE_REJECTED"),
            "problems": validation["problems"],
            "rows": validation["rows"],
            "stable_side_grasp_success": report.get("stable_side_grasp_success"),
            "geometry_error": report.get("geometry_error"),
            "lift_height_m": report.get("lift_height_m"),
            "object_visible_rows": validation["object_visible_rows"],
        })
    except (ValueError, KeyError, OSError) as exc:
        case["status"] = "INSPECTION_ERROR"
        case["problems"] = [f"{type(exc).__name__}: {exc}"]
    return case


def _summary(config: dict, cases: list[dict]) -> dict:
    complete = all(item["status"] != "PENDING" for item in cases)
    valid = sum(item["status"] == "CANDIDATE_CHECKED" for item in cases)
    rejected = sum(item["status"] not in ("PENDING", "CANDIDATE_CHECKED")
                   for item in cases)
    return {
        "status": (("COMPLETE_WITH_REJECTIONS_POLICY_FREE_10HZ_DIAGNOSTIC"
                    if rejected else "COMPLETE_POLICY_FREE_10HZ_DIAGNOSTIC")
                   if complete else "PARTIAL_RESUMABLE_POLICY_FREE_10HZ_DIAGNOSTIC"),
        "conditions": config,
        "completed_cases": sum(item["status"] != "PENDING" for item in cases),
        "total_cases": len(cases),
        "checked_candidates": valid,
        "rejected_or_error_cases": rejected,
        "candidate_rows": sum(item.get("rows", 0) for item in cases
                              if item["status"] == "CANDIDATE_CHECKED"),
        "training_ready": False,
        "interpretation": ("Policy-free simulator achieved-future candidates only; "
                           "not controller commands, human data, policy success, "
                           "hardware validation or deployment approval."),
        "cases": cases,
    }


def _save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--visual-domain", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--offset-m", action="append", type=parse_offset, required=True)
    parser.add_argument("--grip-preload-m", type=float, default=0.002)
    parser.add_argument("--max-new-cases", type=int)
    parser.add_argument("--retry-errors", action="store_true",
                        help="rerun recorded runner errors using a new attempt path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (len(set(args.episodes)) != len(args.episodes)
            or len(set(args.offset_m)) != len(args.offset_m)):
        raise SystemExit("duplicate episode or offset")
    if args.max_new_cases is not None and args.max_new_cases <= 0:
        raise SystemExit("max-new-cases must be positive")
    if not 0 <= args.grip_preload_m <= 0.005:
        raise SystemExit("simulation preload must be 0..5mm")
    if args.out.suffix.lower() != ".json":
        raise SystemExit("--out must end with .json")
    index = json.loads((args.data / "dataset.json").read_text(encoding="utf-8"))
    if any(episode not in index["episodes"] for episode in args.episodes):
        raise SystemExit("episode not in dataset index")
    config = {
        "data": str(args.data),
        "registration": str(args.registration),
        "visual_domain": str(args.visual_domain),
        "episodes": args.episodes,
        "offsets_xy_m": [list(value) for value in args.offset_m],
        "grip_preload_m": args.grip_preload_m,
        "policy": "none; recorded-oracle shifted with object once",
        "observation_source": "simulation",
        "control_rate_hz": 50,
        "sample_rate_hz": 10,
        "observation_horizon": 2,
        "action_horizon": 8,
        "execute_steps": 1,
        "full_chunk_validation": True,
        "final_oracle_tail": True,
    }
    previous_cases = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("conditions") != config:
            raise SystemExit("existing suite has different conditions")
        previous_cases = {
            (item["episode"], tuple(item["offset_xy_m"])): item
            for item in existing.get("cases", [])
        }
    case_dir = args.out.with_name(args.out.stem + "_cases")
    cases = [previous_cases.get(
                 (episode, offset),
                 dict(episode=episode, offset_xy_m=list(offset), status="PENDING",
                      training_ready=False))
             for episode in args.episodes for offset in args.offset_m]
    _save(args.out, _summary(config, cases))
    new_cases = 0
    for position, case in enumerate(cases):
        episode = case["episode"]
        offset = tuple(case["offset_xy_m"])
        if case["status"] == "RUNNER_ERROR" and not args.retry_errors:
            continue
        report_path, candidate_path, attempt = _resume_paths(case_dir, episode, offset)
        if not report_path.exists():
            if args.max_new_cases is not None and new_cases >= args.max_new_cases:
                continue
            command = [
                sys.executable,
                str(Path(__file__).with_name("render_relative_chunk_rollout.py")),
                "--data", str(args.data), "--registration", str(args.registration),
                "--visual-domain", str(args.visual_domain),
                "--oracle", "--oracle-shift-target-with-object",
                "--episode", episode, "--use-registration-approach-start",
                "--cycles", "30", "--execute-steps", "1",
                "--validation-horizon-steps", "8", "--execute-final-oracle-tail",
                "--grip-preload-m", str(args.grip_preload_m),
                "--observation-source", "simulation", "--no-video",
                "--object-offset-x-m", str(offset[0]),
                "--object-offset-y-m", str(offset[1]),
                "--dense-candidate-stream-out", str(candidate_path),
                "--out", str(report_path.with_suffix(".gif")),
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            new_cases += 1
            if process.returncode:
                case.update(status="RUNNER_ERROR", attempt=attempt,
                            report=str(report_path),
                            candidate=str(candidate_path),
                            problems=[(process.stderr or process.stdout)[-1500:]],
                            training_ready=False)
                _save(args.out, _summary(config, cases))
                print(f"{position+1}/{len(cases)} {episode} {offset}: RUNNER_ERROR",
                      flush=True)
                continue
        cases[position] = _inspect_case(
            report_path, candidate_path, episode=episode, offset=offset, config=config)
        cases[position]["attempt"] = attempt
        _save(args.out, _summary(config, cases))
        print(f"{position+1}/{len(cases)} {episode} {offset}: "
              f"{cases[position]['status']}", flush=True)
    result = _summary(config, cases)
    print(f"{result['status']}: {result['checked_candidates']}/{result['total_cases']} "
          f"checked, {result['rejected_or_error_cases']} rejected/error; "
          f"{result['candidate_rows']} rows; training_ready=false", flush=True)
    return 0 if result["rejected_or_error_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
