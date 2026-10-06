"""Audit early gripper closure from real-time policy-runner JSONL logs.

This tool is deliberately read-only.  It never imports the robot driver and
never emits motor commands.  It separates three signals which are otherwise
easy to conflate:

* raw policy gap prediction;
* command after the runtime latch/close offset;
* observed gripper position inferred from the same logged gap-to-tick map.

Optionally, an official UMI Zarr dataset and a deploy checkpoint can be
inspected to find the training gap range and normalizer buffers.  Checkpoint
inspection only lists tensors; it does not instantiate or run the policy.

Examples
--------
python AI/tools/analyze_realtime_gripper_logs.py --selftest
python AI/tools/analyze_realtime_gripper_logs.py \
  --logs runs_0922_logs.tgz --out-dir AI/out/gripper_0922
python AI/tools/analyze_realtime_gripper_logs.py \
  --logs runs_0922_logs.tgz --dataset ds.zarr.zip \
  --checkpoint so101_pick_v1.ckpt --out-dir out/gripper_audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import tarfile
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable


OPEN_LATCH_WARNING_MM = 60.0
GRASP_WIDTH_WARNING_MM = 45.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def load_runs(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read cycles.jsonl members without extracting the archive."""
    runs: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(path, "r:*") as archive:
        members = [member for member in archive.getmembers()
                   if member.isfile() and member.name.endswith("cycles.jsonl")]
        if not members:
            raise ValueError(f"cycles.jsonl not found in {path}")
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read {member.name}")
            rows = []
            for number, raw in enumerate(io.TextIOWrapper(stream, encoding="utf-8"), 1):
                if not raw.strip():
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{member.name}:{number}: {exc}") from exc
            run = Path(member.name).parent.name
            if run in runs:
                raise ValueError(f"duplicate run name: {run}")
            runs[run] = rows
    return runs


def fit_gap_tick(runs: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Fit logged predicted gap -> predicted tick and report fit residual."""
    pairs = []
    for cycles in runs.values():
        for cycle in cycles:
            for candidate in cycle.get("candidates", []):
                gap = candidate.get("predicted_gripper_mm")
                tick = candidate.get("predicted_gripper_tick")
                if gap is not None and tick is not None:
                    pairs.append((float(gap), float(tick)))
    if len(pairs) < 2:
        raise ValueError(f"only {len(pairs)} gap/tick pairs; need at least two")
    mean_x = statistics.fmean(x for x, _ in pairs)
    mean_y = statistics.fmean(y for _, y in pairs)
    denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
    if denominator <= 1e-12:
        raise ValueError("logged gaps have no variation; cannot infer tick map")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denominator
    intercept = mean_y - slope * mean_x
    residual = max(abs((intercept + slope * x) - y) for x, y in pairs)
    if slope >= 0:
        raise ValueError(f"unexpected gap/tick slope {slope}; closing should increase ticks")
    return {
        "pairs": len(pairs),
        "tick_intercept": intercept,
        "tick_per_mm": slope,
        "max_abs_tick_residual": residual,
    }


def tick_to_gap_mm(tick: float | int | None, calibration: dict[str, float]) -> float | None:
    if tick is None:
        return None
    return ((float(tick) - calibration["tick_intercept"])
            / calibration["tick_per_mm"])


def _last_commanded_tick(cycle: dict[str, Any]) -> float | None:
    trace = cycle.get("execution_trace") or []
    if not trace:
        return None
    return (trace[-1].get("targets") or {}).get("6")


def analyze_runs(runs: dict[str, list[dict[str, Any]]],
                 calibration: dict[str, float]) -> tuple[list[dict[str, Any]],
                                                         list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run, cycles in sorted(runs.items()):
        raw_gaps: list[float] = []
        applied_gaps: list[float] = []
        target_errors: list[float] = []
        first_latch: dict[str, Any] | None = None
        clamped = 0
        motor_commands = 0
        for cycle_index, cycle in enumerate(cycles):
            observed_tick = (cycle.get("observation_ticks") or {}).get("6")
            final_tick = (cycle.get("final_ticks") or {}).get("6")
            observed_gap = tick_to_gap_mm(observed_tick, calibration)
            final_gap = tick_to_gap_mm(final_tick, calibration)
            motor_commands += int(cycle.get("motor_commands_sent") or 0)
            last_command = _last_commanded_tick(cycle)
            if last_command is not None and final_tick is not None:
                target_errors.append(abs(float(last_command) - float(final_tick)))
            for candidate_index, candidate in enumerate(cycle.get("candidates") or []):
                raw_gap = float(candidate["predicted_gripper_mm"])
                applied_tick = candidate.get("gripper_tick")
                applied_gap = tick_to_gap_mm(applied_tick, calibration)
                raw_gaps.append(raw_gap)
                if applied_gap is not None:
                    applied_gaps.append(applied_gap)
                if candidate.get("gripper_latched") and first_latch is None:
                    first_latch = {
                        "cycle": cycle_index,
                        "action_index": candidate.get("action_index", candidate_index),
                        "raw_policy_gap_mm": raw_gap,
                        "policy_tick": candidate.get("predicted_gripper_tick"),
                        "applied_tick": applied_tick,
                        "offset_ticks": candidate.get("gripper_close_offset_ticks", 0),
                    }
                curves.append({
                    "run": run,
                    "cycle": cycle_index,
                    "action_index": candidate.get("action_index", candidate_index),
                    "motor_commands_sent": int(cycle.get("motor_commands_sent") or 0),
                    "raw_policy_gap_mm": raw_gap,
                    "latch_applied": bool(candidate.get("gripper_latched")),
                    "close_offset_ticks": candidate.get("gripper_close_offset_ticks", 0),
                    "command_gap_mm": applied_gap,
                    "observed_cycle_start_gap_mm": observed_gap,
                    "observed_cycle_end_gap_mm": final_gap,
                    "target_x_mm": (candidate.get("target_table_mm") or [None] * 3)[0],
                    "target_y_mm": (candidate.get("target_table_mm") or [None] * 3)[1],
                    "target_z_mm": (candidate.get("target_table_mm") or [None] * 3)[2],
                })
            for trace in cycle.get("execution_trace") or []:
                clamped += len(trace.get("clamped_targets") or {})
        is_live = motor_commands > 0
        summaries.append({
            "run": run,
            "cycles": len(cycles),
            "live": is_live,
            "motor_commands_sent": motor_commands,
            "predictions": len(raw_gaps),
            "raw_gap_min_mm": min(raw_gaps) if raw_gaps else None,
            "raw_gap_max_mm": max(raw_gaps) if raw_gaps else None,
            "raw_gap_median_mm": statistics.median(raw_gaps) if raw_gaps else None,
            "first_latch": first_latch,
            "early_latch_above_mm": bool(
                first_latch and first_latch["raw_policy_gap_mm"] > OPEN_LATCH_WARNING_MM),
            "policy_reached_grasp_width": bool(
                raw_gaps and min(raw_gaps) <= GRASP_WIDTH_WARNING_MM),
            "applied_gap_min_mm": min(applied_gaps) if applied_gaps else None,
            "clamped_target_entries": clamped,
            "target_follow_abs_error_tick_median": (
                statistics.median(target_errors) if target_errors else None),
            "target_follow_abs_error_tick_p95": percentile(target_errors, 0.95),
            "target_follow_abs_error_tick_last": target_errors[-1] if target_errors else None,
        })
    return curves, summaries


def numeric_summary(values: Any) -> dict[str, Any]:
    import numpy as np
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": int(len(array)), "finite": 0}
    return {
        "count": int(len(array)), "finite": int(len(finite)),
        "min": float(finite.min()), "p01": float(np.quantile(finite, 0.01)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)), "max": float(finite.max()),
    }


def inspect_zarr_dataset(path: Path, close_threshold_m: float) -> dict[str, Any]:
    """Inspect gap observation/action support in an official UMI Zarr."""
    try:
        import numpy as np
        import zarr
    except ImportError as exc:
        raise RuntimeError("dataset audit needs numpy and zarr") from exc
    if str(path).endswith(".zip"):
        # Official UMI commonly ships a .zarr.zip.  Opening the filename
        # directly is version-dependent, whereas ZipStore states the intent.
        store_type = getattr(zarr, "ZipStore", None)
        if store_type is None:
            from zarr.storage import ZipStore as store_type
        root = zarr.open(store_type(str(path), mode="r"), mode="r")
    else:
        root = zarr.open(str(path), mode="r")
    data = root["data"]
    gap = np.asarray(data["robot0_gripper_width"][:], dtype=np.float64).reshape(-1)
    action = np.asarray(data["action"][:], dtype=np.float64)
    action_gap = action[:, -1]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    initial = gap[starts]
    preclose = []
    close_rows = []
    for lo, hi in zip(starts, ends):
        local = gap[lo:hi]
        idx = np.flatnonzero(local <= close_threshold_m)
        cut = int(idx[0]) if len(idx) else len(local)
        preclose.extend(local[:cut].tolist())
        close_rows.append(None if not len(idx) else int(lo + idx[0]))
    return {
        "format": "official_umi_zarr", "path": str(path),
        "rows": int(len(gap)), "episodes": int(len(ends)),
        "gap_observation_m": numeric_summary(gap),
        "gap_action_m": numeric_summary(action_gap),
        "episode_initial_gap_m": numeric_summary(initial),
        "preclose_gap_m": numeric_summary(preclose),
        "close_threshold_m": close_threshold_m,
        "episodes_with_close": sum(row is not None for row in close_rows),
        "episodes_without_close": sum(row is None for row in close_rows),
    }


def inspect_v10_dataset(path: Path, close_threshold_m: float) -> dict[str, Any]:
    """Inspect Track-A v10 NPZ gap support before official-Zarr conversion."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("dataset audit needs numpy") from exc
    index = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
    obs, actions, initial = [], [], []
    for episode in index.get("episodes", []):
        npz = path / f"{episode}.npz"
        if not npz.is_file():
            raise FileNotFoundError(npz)
        with np.load(npz, allow_pickle=False) as archive:
            proprio = np.asarray(archive["proprio"], dtype=np.float64)
            action = np.asarray(archive["action"], dtype=np.float64)
        if not len(proprio):
            continue
        obs.append(proprio[..., -1].reshape(-1))
        actions.append(action[..., -1].reshape(-1))
        initial.append(float(proprio[0, -1, -1]))
    if not obs:
        raise ValueError(f"no v10 episode rows in {path}")
    gap = np.concatenate(obs)
    action_gap = np.concatenate(actions)
    preclose = gap[gap > close_threshold_m]
    return {
        "format": "track_a_v10_npz", "path": str(path),
        "schema": index.get("schema"), "rows": int(index.get("n_rows", 0)),
        "episodes": len(obs),
        "gap_observation_m": numeric_summary(gap),
        "gap_action_m": numeric_summary(action_gap),
        "episode_initial_gap_m": numeric_summary(initial),
        "preclose_gap_m": numeric_summary(preclose),
        "close_threshold_m": close_threshold_m,
        "note": ("v10 stores overlapping H=2/K=8 anchors; counts are tensor values, not "
                 "unique source frames. Official-Zarr parity must still be checked."),
    }


def inspect_training_dataset(path: Path, close_threshold_m: float) -> dict[str, Any]:
    if path.is_dir() and (path / "dataset.json").is_file():
        return inspect_v10_dataset(path, close_threshold_m)
    return inspect_zarr_dataset(path, close_threshold_m)


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """List normalizer-like checkpoint tensors without running the model."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("checkpoint audit needs torch") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload.get("state_dicts", {}) if isinstance(payload, dict) else {}
    model = states.get("ema_model", states.get("model", {}))
    found = {}
    for key, value in model.items():
        low = key.lower()
        if not any(token in low for token in ("normalizer", "scale", "offset", "stat")):
            continue
        if not hasattr(value, "detach"):
            continue
        array = value.detach().cpu().numpy()
        item = {"shape": list(array.shape), "summary": numeric_summary(array)}
        if array.size <= 32:
            item["values"] = array.reshape(-1).tolist()
        found[key] = item
    return {
        "path": str(path), "sha256": sha256(path),
        "state_dict": "ema_model" if "ema_model" in states else "model",
        "normalizer_like_tensors": found,
        "found": len(found),
        "note": ("Tensor names are reported, not guessed. Identify the action tensor and its "
                 "gap channel from the checkpoint's own policy implementation."),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_curves_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a dependency-free diagnostic plot, one panel per live run."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["motor_commands_sent"] > 0:
            grouped.setdefault(str(row["run"]), []).append(row)
    if not grouped:
        return
    width, panel_h, left, right = 1100, 170, 70, 20
    top, bottom = 34, 28
    height = top + panel_h * len(grouped) + bottom
    gap_lo, gap_hi = 30.0, 95.0
    colors = {"raw_policy_gap_mm": "#2563eb", "command_gap_mm": "#ea580c",
              "observed_cycle_end_gap_mm": "#16a34a"}

    def polyline(items: list[dict[str, Any]], key: str, y0: float) -> str:
        points = []
        n = max(1, len(items) - 1)
        for index, item in enumerate(items):
            value = item.get(key)
            if value is None:
                continue
            x = left + (width - left - right) * index / n
            clipped = min(gap_hi, max(gap_lo, float(value)))
            y = y0 + 18 + (panel_h - 38) * (gap_hi - clipped) / (gap_hi - gap_lo)
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Segoe UI,Arial,sans-serif;font-size:12px}'
           '.title{font-size:15px;font-weight:600}.axis{stroke:#cbd5e1;stroke-width:1}'
           '.line{fill:none;stroke-width:2}</style>',
           '<text x="70" y="20" class="title">0922 gripper: raw policy / latch command / observed</text>']
    legend_x = 610
    for index, (key, label) in enumerate((
        ("raw_policy_gap_mm", "raw policy"),
        ("command_gap_mm", "after latch/offset"),
        ("observed_cycle_end_gap_mm", "observed end"),
    )):
        x = legend_x + index * 155
        svg += [f'<line x1="{x}" y1="17" x2="{x + 24}" y2="17" '
                f'stroke="{colors[key]}" stroke-width="3"/>',
                f'<text x="{x + 29}" y="21">{label}</text>']
    for panel, (run, items) in enumerate(grouped.items()):
        y0 = top + panel * panel_h
        svg += [f'<text x="8" y="{y0 + 16}" class="title">{escape(run)}</text>']
        for gap in (40, 50, 60, 79, 90):
            y = y0 + 18 + (panel_h - 38) * (gap_hi - gap) / (gap_hi - gap_lo)
            svg += [f'<line class="axis" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>',
                    f'<text x="38" y="{y + 4:.1f}">{gap}</text>']
        for key, color in colors.items():
            points = polyline(items, key, y0)
            if points:
                svg.append(f'<polyline class="line" stroke="{color}" points="{points}"/>')
        svg.append(f'<text x="{width - 145}" y="{y0 + panel_h - 5}">candidate order →</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def build_report(log_path: Path, runs: dict[str, list[dict[str, Any]]],
                 calibration: dict[str, float], summaries: list[dict[str, Any]],
                 dataset: dict[str, Any] | None,
                 checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    live = [row for row in summaries if row["live"]]
    early = [row for row in live if row["early_latch_above_mm"]]
    closed = [row for row in live if row["policy_reached_grasp_width"]]
    offsets = Counter()
    for row in live:
        latch = row["first_latch"]
        if latch:
            offsets[str(latch["offset_ticks"])] += 1
    verdict = {
        "hardware_is_primary_cause": False,
        "policy_output_contributes": bool(closed),
        "runtime_latch_contributes": bool(early),
        "exact_latch_off_closed_loop_replay_possible_from_logs": False,
        "reason": (
            "The log stores actions and ticks but not the image/history tensors needed to rerun "
            "the checkpoint. Raw-policy versus latch-applied commands are still reconstructed."),
    }
    return {
        "schema": "realtime_gripper_log_audit/0.1",
        "status": "DIAGNOSTIC_NOT_HARDWARE_APPROVAL",
        "source": {"path": str(log_path), "sha256": sha256(log_path)},
        "counts": {"runs": len(runs), "live_runs": len(live),
                   "live_early_latch_runs": len(early),
                   "live_policy_at_or_below_45mm_runs": len(closed)},
        "gap_tick_calibration_from_logs": calibration,
        "offset_ticks_first_latch_counts": dict(offsets),
        "verdict": verdict,
        "runs": summaries,
        "dataset": dataset,
        "checkpoint": checkpoint,
    }


def selftest() -> int:
    def candidate(gap: float, tick: int, applied: int, latch: bool) -> dict[str, Any]:
        return {"action_index": 4, "predicted_gripper_mm": gap,
                "predicted_gripper_tick": tick, "gripper_tick": applied,
                "gripper_latched": latch,
                "gripper_close_offset_ticks": applied - tick,
                "target_table_mm": [1, 2, 3]}
    runs = {"live": [
        {"motor_commands_sent": 1, "observation_ticks": {"6": 130},
         "final_ticks": {"6": 355},
         "candidates": [candidate(79.0, 330, 355, True)],
         "execution_trace": [{"targets": {"6": 355}, "clamped_targets": {}}]},
        {"motor_commands_sent": 1, "observation_ticks": {"6": 355},
         "final_ticks": {"6": 1100},
         "candidates": [candidate(38.0, 1073, 1098, True)],
         "execution_trace": [{"targets": {"6": 1098}, "clamped_targets": {}}]},
    ], "predict_only": [
        {"motor_commands_sent": 0, "candidates": [candidate(78.0, 348, 348, False)]},
    ]}
    calibration = fit_gap_tick(runs)
    curves, summaries = analyze_runs(runs, calibration)
    live = next(row for row in summaries if row["run"] == "live")
    checks = [
        ("negative tick/mm", calibration["tick_per_mm"] < 0),
        ("curve denominator", len(curves) == 3),
        ("early latch detected", live["early_latch_above_mm"]),
        ("policy reaches grasp width", live["policy_reached_grasp_width"]),
        ("predict-only is not live", not next(row for row in summaries
                                                if row["run"] == "predict_only")["live"]),
        ("no clamp", live["clamped_target_entries"] == 0),
    ]
    for index, (name, ok) in enumerate(checks, 1):
        print(f"[{index}] {name:<36} {'OK' if ok else 'FAIL'}")
    print(f"selftest {sum(ok for _, ok in checks)} / {len(checks)}")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--close-threshold-m", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=Path("gripper_log_audit"))
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.logs:
        parser.error("--logs is required (or use --selftest)")
    runs = load_runs(args.logs)
    calibration = fit_gap_tick(runs)
    curves, summaries = analyze_runs(runs, calibration)
    dataset = inspect_training_dataset(args.dataset, args.close_threshold_m) if args.dataset else None
    checkpoint = inspect_checkpoint(args.checkpoint) if args.checkpoint else None
    report = build_report(args.logs, runs, calibration, summaries, dataset, checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "gripper_curves.csv", curves)
    write_curves_svg(args.out_dir / "gripper_curves.svg", curves)
    flat = []
    for row in summaries:
        item = dict(row)
        item["first_latch"] = json.dumps(item["first_latch"], ensure_ascii=False)
        flat.append(item)
    write_csv(args.out_dir / "run_summary.csv", flat)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"tick/mm {calibration['tick_per_mm']:.6f} · "
          f"max fit residual {calibration['max_abs_tick_residual']:.3f} tick")
    print(f"-> {args.out_dir / 'report.json'}")
    print(f"-> {args.out_dir / 'gripper_curves.csv'}")
    print(f"-> {args.out_dir / 'gripper_curves.svg'}")
    print(f"-> {args.out_dir / 'run_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
