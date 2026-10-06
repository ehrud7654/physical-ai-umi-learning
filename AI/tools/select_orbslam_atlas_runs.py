"""Select reproducible ORB-SLAM runs across fixed-marker-aligned Atlas candidates.

An Atlas candidate is supplied as four values: name, Atlas file, ArUco
alignment report, and batch output root.  Each batch root contains
``<episode>/attempt*/camera_trajectory.csv``.  The selector never merges pose
rows from different runs; it chooses one complete run per episode and records
the Atlas/alignment hashes needed to reproduce the common table frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _load_reviewed_gap_exceptions(path: Path | None) -> tuple[dict[str, dict], dict | None]:
    if path is None:
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "reviewed_gap_exceptions/0.1":
        raise SystemExit(f"unsupported reviewed gap exception schema: {payload.get('schema')!r}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, dict):
        raise SystemExit("reviewed gap exceptions must contain an episodes object")
    for episode_id, exception in episodes.items():
        if not isinstance(exception, dict):
            raise SystemExit(f"{episode_id}: reviewed gap exception must be an object")
        if exception.get("reviewed") is not True:
            raise SystemExit(f"{episode_id}: reviewed must be true")
        if not isinstance(exception.get("maximum_missing_run_frames"), int):
            raise SystemExit(f"{episode_id}: maximum_missing_run_frames must be an integer")
        if not str(exception.get("reason", "")).strip():
            raise SystemExit(f"{episode_id}: reason is required")
    return episodes, {
        "path": str(path),
        "sha256": _sha256(path),
        "schema": payload["schema"],
    }


def _loss_episodes(values: list[bool], start: int) -> int:
    count = 0
    in_loss = False
    for value in values[start:]:
        if not value and not in_loss:
            count += 1
            in_loss = True
        elif value:
            in_loss = False
    return count


def validate_run(raw_episode: Path, trajectory: Path, *,
                 minimum_coverage: float, maximum_first_time_s: float,
                 maximum_timestamp_error_ms: float) -> dict:
    frames = _rows(raw_episode / "frames.csv")
    poses = _rows(trajectory)
    errors: list[str] = []
    if len(frames) != len(poses):
        errors.append(f"frame count mismatch {len(poses)} != {len(frames)}")
    count = min(len(frames), len(poses))
    if [int(row.get("frame_idx", -1)) for row in poses] != list(range(len(poses))):
        errors.append("frame_idx is not contiguous from zero")
    sensor_ns = np.asarray(
        [int(row["sensor_timestamp_ns"]) for row in frames], dtype=np.int64)
    if len(sensor_ns) and np.any(np.diff(sensor_ns) <= 0):
        errors.append("camera timestamps are not strictly increasing")
    expected = ((sensor_ns - sensor_ns[0]) / 1e9)[:count] if len(sensor_ns) else np.empty(0)
    tracked = [
        row.get("state") == "2" and row.get("is_lost", "true").lower() == "false"
        for row in poses[:count]
    ]
    indices = [index for index, value in enumerate(tracked) if value]
    first = indices[0] if indices else None
    first_time = float(expected[first]) if first is not None else None
    coverage = len(indices) / len(frames) if frames else 0.0
    losses = _loss_episodes(tracked, first) if first is not None else 0
    timestamp_errors = [
        abs(float(poses[index]["timestamp"]) - expected[index]) * 1000
        for index in indices
    ]
    maximum_error = max(timestamp_errors, default=float("inf"))
    if coverage < minimum_coverage:
        errors.append(f"coverage {coverage:.3f} below {minimum_coverage:.3f}")
    if first_time is None or first_time > maximum_first_time_s:
        errors.append(f"first pose time {first_time!r}s exceeds {maximum_first_time_s:.3f}s")
    if losses:
        errors.append(f"{losses} tracking loss episode(s) after initialization")
    if tracked and not tracked[-1]:
        errors.append("final frame is not tracked")
    if maximum_error > maximum_timestamp_error_ms:
        errors.append(
            f"timestamp error {maximum_error:.6f}ms exceeds {maximum_timestamp_error_ms:.3f}ms")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "frames": len(frames),
        "tracked_frames": len(indices),
        "coverage": coverage,
        "first_tracked_frame": first,
        "first_tracked_time_s": first_time,
        "loss_episodes_after_initialization": losses,
        "maximum_timestamp_error_ms": maximum_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--gripper-root", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", nargs=4,
        metavar=("NAME", "ATLAS", "ALIGNMENT", "BATCH_ROOT"), required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--maximum-first-time-s", type=float, default=0.5)
    parser.add_argument("--maximum-timestamp-error-ms", type=float, default=2.0)
    parser.add_argument("--maximum-gripper-missing-run", type=int, default=15)
    parser.add_argument(
        "--reviewed-gap-exceptions", type=Path,
        help=("JSON containing narrowly reviewed per-episode exceptions. The global "
              "missing-run threshold remains unchanged."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gap_exceptions, gap_exception_source = _load_reviewed_gap_exceptions(
        args.reviewed_gap_exceptions)

    candidates = []
    for name, atlas_text, alignment_text, batch_text in args.candidate:
        atlas, alignment, batch = map(Path, (atlas_text, alignment_text, batch_text))
        alignment_report = json.loads(alignment.read_text(encoding="utf-8"))
        if alignment_report.get("status") != "PASS_ARUCO_ALIGNMENT":
            raise SystemExit(f"{name}: alignment is not PASS_ARUCO_ALIGNMENT")
        candidates.append({
            "name": name,
            "atlas": atlas,
            "atlas_sha256": _sha256(atlas),
            "alignment": alignment,
            "alignment_sha256": _sha256(alignment),
            "batch": batch,
        })

    selected, excluded = [], {}
    raw_episodes = sorted(path for path in args.raw_root.glob("rec_*") if path.is_dir())
    for raw in raw_episodes:
        manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("outcome") != "success":
            excluded[raw.name] = [f"outcome={manifest.get('outcome')!r}"]
            continue
        gripper_report_path = args.gripper_root / raw.name / "gripper_report.json"
        if not gripper_report_path.is_file():
            excluded[raw.name] = ["missing gripper_report.json"]
            continue
        gripper_report = json.loads(gripper_report_path.read_text(encoding="utf-8"))
        missing_run = int(gripper_report.get("maximum_missing_run_frames", 10**9))
        gap_exception = gap_exceptions.get(raw.name)
        exception_applies = False
        if gap_exception is not None:
            declared_run = gap_exception["maximum_missing_run_frames"]
            if missing_run != declared_run:
                raise SystemExit(
                    f"{raw.name}: reviewed exception says {declared_run} missing frames, "
                    f"but current gripper report says {missing_run}")
            exception_applies = missing_run > args.maximum_gripper_missing_run
        if (gripper_report.get("status") != "pass" or
                (missing_run > args.maximum_gripper_missing_run and not exception_applies)):
            excluded[raw.name] = [
                f"gripper status={gripper_report.get('status')!r}, "
                f"missing run={missing_run} > {args.maximum_gripper_missing_run}"
            ]
            continue
        choices, failures = [], []
        for candidate_index, candidate in enumerate(candidates):
            episode_root = candidate["batch"] / raw.name
            for attempt_dir in sorted(episode_root.glob("attempt*")):
                trajectory = attempt_dir / "camera_trajectory.csv"
                if not trajectory.is_file():
                    continue
                report = validate_run(
                    raw, trajectory,
                    minimum_coverage=args.minimum_coverage,
                    maximum_first_time_s=args.maximum_first_time_s,
                    maximum_timestamp_error_ms=args.maximum_timestamp_error_ms,
                )
                report_path = attempt_dir / "trajectory_validation_v10.json"
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                attempt = int(attempt_dir.name.removeprefix("attempt"))
                row = {
                    "candidate": candidate["name"],
                    "candidate_index": candidate_index,
                    "attempt": attempt,
                    "trajectory": str(trajectory),
                    "trajectory_validation": str(report_path),
                    "report": report,
                    "alignment": str(candidate["alignment"]),
                    "atlas_sha256": candidate["atlas_sha256"],
                }
                if report["status"] == "pass":
                    choices.append(row)
                else:
                    failures.append(row)
        if not choices:
            excluded[raw.name] = [
                f"{row['candidate']}/attempt{row['attempt']}: "
                + "; ".join(row["report"]["errors"])
                for row in failures
            ] or ["no trajectory attempts"]
            continue
        choices.sort(key=lambda row: (
            -row["report"]["coverage"],
            row["report"]["first_tracked_time_s"],
            row["candidate_index"], row["attempt"],
        ))
        chosen = choices[0]
        selected.append({
            "episode_id": raw.name,
            "raw_episode": str(raw),
            "gripper_csv": str(args.gripper_root / raw.name / "gripper_width.csv"),
            "trajectory": chosen["trajectory"],
            "trajectory_validation": chosen["trajectory_validation"],
            "alignment": chosen["alignment"],
            "atlas_candidate": chosen["candidate"],
            "atlas_sha256": chosen["atlas_sha256"],
            "attempt": chosen["attempt"],
            "trajectory_summary": chosen["report"],
            "gripper_quality": {
                "status": gripper_report.get("status"),
                "maximum_missing_run_frames": missing_run,
                "global_threshold_frames": args.maximum_gripper_missing_run,
                "reviewed_exception": gap_exception if exception_applies else None,
            },
        })
    result = {
        "schema": "orbslam_atlas_selection/0.1",
        "policy": "one complete run per episode; no cross-run pose-row splice",
        "common_frame": "DICT_4X4_50 ID 13, black-square side 0.16 m",
        "thresholds": {
            "minimum_coverage": args.minimum_coverage,
            "maximum_first_time_s": args.maximum_first_time_s,
            "maximum_timestamp_error_ms": args.maximum_timestamp_error_ms,
            "maximum_gripper_missing_run_frames": args.maximum_gripper_missing_run,
            "maximum_loss_episodes_after_initialization": 0,
            "require_final_frame_tracked": True,
        },
        "candidates": [{key: str(value) if isinstance(value, Path) else value
                        for key, value in candidate.items() if key != "batch"}
                       | {"batch": str(candidate["batch"])}
                       for candidate in candidates],
        "reviewed_gap_exception_source": gap_exception_source,
        "reviewed_gap_exceptions_applied": [
            episode["episode_id"] for episode in selected
            if episode["gripper_quality"]["reviewed_exception"] is not None
        ],
        "episodes": selected,
        "n_selected": len(selected),
        "n_excluded": len(excluded),
        "excluded": excluded,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"selected": len(selected), "excluded": len(excluded)},
                     ensure_ascii=False))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
