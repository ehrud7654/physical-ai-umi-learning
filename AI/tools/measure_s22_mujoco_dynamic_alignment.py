"""Measure real/S22 versus MuJoCo object framing along recorded-oracle replays.

This is a policy-free diagnostic.  Each selected episode is replayed with the
recorded relative target chunk while the real dataset frame and simulated wrist
frame are compared at the same dataset row.  The detector is deliberately the
same simple yellow/orange largest-component proxy used by the camera-fit tools;
the result therefore diagnoses framing drift, not semantic segmentation quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


AI_ROOT = Path(__file__).resolve().parents[1]
RENDER_TOOL = AI_ROOT / "tools" / "render_relative_chunk_rollout.py"


def _scene_valid(registration: dict[str, Any]) -> list[str]:
    rows = registration.get("selected", {}).get("per_episode", [])
    return [
        str(row["episode"])
        for row in rows
        if isinstance(row, dict) and row.get("scene_constraints_ok") is True
    ]


def _percentiles(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None, None, None]
    return [float(value) for value in np.percentile(values, [0, 50, 95, 100])]


def _pair_metrics(real: list[float], sim: list[float]) -> dict[str, float]:
    rcx, rcy, rw, rh = (float(value) for value in real)
    scx, scy, sw, sh = (float(value) for value in sim)
    rx0, ry0, rx1, ry1 = rcx - rw / 2, rcy - rh / 2, rcx + rw / 2, rcy + rh / 2
    sx0, sy0, sx1, sy1 = scx - sw / 2, scy - sh / 2, scx + sw / 2, scy + sh / 2
    iw = max(0.0, min(rx1, sx1) - max(rx0, sx0))
    ih = max(0.0, min(ry1, sy1) - max(ry0, sy0))
    intersection = iw * ih
    union = rw * rh + sw * sh - intersection
    return {
        "center_l2_norm": math.hypot(scx - rcx, scy - rcy),
        "width_abs_norm": abs(sw - rw),
        "height_abs_norm": abs(sh - rh),
        "width_log_ratio_abs": abs(math.log(max(sw, 1e-9) / max(rw, 1e-9))),
        "height_log_ratio_abs": abs(math.log(max(sh, 1e-9) / max(rh, 1e-9))),
        "bbox_iou": intersection / union if union > 0 else 0.0,
    }


def _summarise_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in pairs]
    keys = (
        "center_l2_norm",
        "width_abs_norm",
        "height_abs_norm",
        "width_log_ratio_abs",
        "height_log_ratio_abs",
        "bbox_iou",
    )
    return {
        "paired_frames": len(pairs),
        **{f"{key}_p0_p50_p95_max": _percentiles([row[key] for row in metrics])
           for key in keys},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--visual-domain", type=Path, required=True)
    ap.add_argument("--episodes", nargs="*", required=True)
    ap.add_argument("--allow-rejected-diagnostic-subset", action="store_true")
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--execute-steps", type=int, default=1)
    ap.add_argument("--validation-horizon-steps", type=int, default=1)
    ap.add_argument("--tracking-time-margin", type=float, default=1.0)
    ap.add_argument("--grip-preload-m", type=float, default=0.002)
    ap.add_argument("--grip-preload-activation-gap-m", type=float, default=0.055)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.suffix.lower() != ".json" or args.out.exists():
        raise SystemExit("--out must be a new .json path")

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    eligible = _scene_valid(registration)
    unknown = sorted(set(args.episodes) - set(eligible))
    if unknown or len(args.episodes) != len(set(args.episodes)):
        raise SystemExit(f"invalid or duplicate episode ids: {unknown}")
    if registration.get("status") == "REJECTED_DIAGNOSTIC_CANONICAL_REPLAY":
        if not args.allow_rejected_diagnostic_subset:
            raise SystemExit(
                "rejected diagnostic registration requires "
                "--allow-rejected-diagnostic-subset")

    report_dir = args.out.parent / f"{args.out.stem}_episodes"
    if report_dir.exists():
        raise SystemExit(f"episode report directory exists: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=False)

    episode_rows: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for index, episode in enumerate(args.episodes, start=1):
        pseudo = (report_dir / f"{episode}.gif").resolve()
        command = [
            sys.executable, str(RENDER_TOOL),
            "--data", str(args.data.resolve()),
            "--registration", str(args.registration.resolve()),
            "--oracle", "--episode", episode,
            "--use-registration-approach-start",
            "--cycles", str(args.cycles),
            "--execute-steps", str(args.execute_steps),
            "--validation-horizon-steps", str(args.validation_horizon_steps),
            "--execute-final-oracle-tail",
            "--no-video", "--measure-visual-alignment",
            "--tracking-time-margin", str(args.tracking_time_margin),
            "--grip-preload-m", str(args.grip_preload_m),
            "--grip-preload-activation-gap-m",
            str(args.grip_preload_activation_gap_m),
            "--visual-domain", str(args.visual_domain.resolve()),
            "--out", str(pseudo),
        ]
        completed = subprocess.run(
            command, cwd=AI_ROOT, capture_output=True, text=True, check=False)
        report_path = pseudo.with_suffix(".json")
        if completed.returncode != 0 or not report_path.is_file():
            episode_rows.append({
                "episode": episode,
                "runner_error": True,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            })
            print(f"{index}/{len(args.episodes)} {episode}: runner_error", flush=True)
            continue

        source = json.loads(report_path.read_text(encoding="utf-8"))
        trace = source.get("visual_alignment_trace", [])
        pairs: list[dict[str, Any]] = []
        real_detections = 0
        sim_detections = 0
        for trace_row in trace:
            real = trace_row.get("real_bbox_xywh_norm")
            sim = trace_row.get("simulation_bbox_xywh_norm")
            real_detections += real is not None
            sim_detections += sim is not None
            if real is None or sim is None:
                continue
            pair = {
                "episode": episode,
                "cycle": trace_row.get("cycle"),
                "dataset_row": trace_row.get("dataset_row"),
                "metrics": _pair_metrics(real, sim),
            }
            pairs.append(pair)
            all_pairs.append(pair)
        episode_rows.append({
            "episode": episode,
            "runner_error": False,
            "trace_frames": len(trace),
            "real_detections": real_detections,
            "simulation_detections": sim_detections,
            "recorded_oracle_success": bool(source.get("stable_side_grasp_success")),
            "summary": _summarise_pairs(pairs),
            "episode_report": str(report_path),
        })
        print(
            f"{index}/{len(args.episodes)} {episode}: "
            f"paired={len(pairs)}/{len(trace)}",
            flush=True,
        )

    third_pairs: dict[str, list[dict[str, Any]]] = {
        "early": [], "middle": [], "late": []}
    by_episode = {episode: [] for episode in args.episodes}
    for pair in all_pairs:
        by_episode[pair["episode"]].append(pair)
    for pairs in by_episode.values():
        pairs.sort(key=lambda row: int(row["cycle"]))
        for position, pair in enumerate(pairs):
            fraction = position / max(1, len(pairs) - 1)
            label = "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"
            third_pairs[label].append(pair)

    result = {
        "status": "POLICY_FREE_RECORDED_ORACLE_DYNAMIC_VISUAL_ALIGNMENT_DIAGNOSTIC",
        "data": str(args.data),
        "registration": str(args.registration),
        "registration_status": registration.get("status"),
        "visual_domain": str(args.visual_domain),
        "episodes": args.episodes,
        "learned_policy_used": False,
        "detector": "largest yellow/orange connected component; framing proxy only",
        "summary": _summarise_pairs(all_pairs),
        "phase_summary": {
            label: _summarise_pairs(pairs) for label, pairs in third_pairs.items()
        },
        "recorded_oracle_successes": sum(
            row.get("recorded_oracle_success") is True for row in episode_rows),
        "runner_errors": sum(row.get("runner_error") is True for row in episode_rows),
        "per_episode": episode_rows,
        "interpretation": (
            "Low centre/scale error and high bbox IoU must hold throughout the replay. "
            "This proxy does not validate texture, lighting, contact physics or policy success."
        ),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    return 0 if result["runner_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
