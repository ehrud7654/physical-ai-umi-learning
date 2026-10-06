"""Validate and summarize a provisional UMI relative action-chunk dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from umi.relative_dataset import (
    SCHEMA,
    relative_vector,
    validate_arrays,
    vector_to_transform,
)


def percentile(values: list[np.ndarray], points=(0, 50, 95, 100)) -> list[float]:
    merged = np.concatenate(values) if values else np.asarray([], dtype=float)
    return [float(value) for value in np.percentile(merged, points)] if len(merged) else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    index_path = args.dataset / "dataset.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if index.get("schema") != SCHEMA:
        problems.append(f"dataset schema={index.get('schema')!r}, expected {SCHEMA!r}")
    expected = list(index.get("episodes", []))
    actual = sorted(path.stem for path in args.dataset.glob("*.npz"))
    if actual != sorted(expected):
        problems.append("dataset.json episode list does not match NPZ files")

    rows: list[int] = []
    durations: list[float] = []
    translation: list[np.ndarray] = []
    rotation_deg: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    horizons: list[np.ndarray] = []
    source_steps: list[np.ndarray] = []
    velocity_translation_error: list[np.ndarray] = []
    velocity_rotation6d_error: list[np.ndarray] = []
    velocity_gap_error: list[np.ndarray] = []
    for episode_id in actual:
        npz_path = args.dataset / f"{episode_id}.npz"
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.is_file():
            problems.append(f"{episode_id}: missing JSON metadata")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema") != SCHEMA:
            problems.append(f"{episode_id}: wrong schema")
        if meta.get("time_scale") != 1.0 or meta.get("source_time_preserved") is not True:
            problems.append(f"{episode_id}: source time is not preserved")
        if meta.get("robot_base_alignment_applied") is not False:
            problems.append(f"{episode_id}: robot base alignment must not be baked in")
        if meta.get("robot_ik_applied") is not False:
            problems.append(f"{episode_id}: robot IK must not be baked in")
        with np.load(npz_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        episode_problems = validate_arrays(
            arrays,
            obs_horizon=int(meta["observation_horizon"]),
            action_horizon=int(meta["action_horizon"]),
        )
        problems.extend(f"{episode_id}: {value}" for value in episode_problems)
        if episode_problems:
            continue
        action = arrays["action"]
        rows.append(len(action))
        durations.append(float(meta["source_duration_s"]))
        translation.append(np.linalg.norm(action[..., :3], axis=-1).reshape(-1))
        gaps.append(action[..., 9].reshape(-1))
        horizons.append(
            arrays["action_timestamp"][:, -1] - arrays["observation_timestamp"][:, -1])
        source_steps.append(np.diff(arrays["source_row"]).astype(float))
        rotations = np.stack([
            vector_to_transform(value)[:3, :3]
            for value in action.reshape(-1, action.shape[-1])
        ])
        traces = np.trace(rotations, axis1=1, axis2=2)
        rotation_deg.append(np.rad2deg(np.arccos(np.clip((traces - 1) / 2, -1, 1))))
        if arrays["proprio"].shape[1] >= 2:
            predicted_chunks = []
            for history, expected_chunk in zip(arrays["proprio"], action):
                previous_to_current = np.linalg.inv(vector_to_transform(history[-2]))
                future = np.eye(4)
                predicted = []
                for _ in range(expected_chunk.shape[0]):
                    future = future @ previous_to_current
                    predicted.append(relative_vector(
                        np.eye(4), future, float(history[-1, 9])))
                predicted_chunks.append(np.stack(predicted))
            predicted = np.stack(predicted_chunks)
            velocity_translation_error.append(np.linalg.norm(
                action[..., :3] - predicted[..., :3], axis=-1).reshape(-1))
            velocity_rotation6d_error.append(np.abs(
                action[..., 3:9] - predicted[..., 3:9]).reshape(-1))
            velocity_gap_error.append(np.abs(
                action[..., 9] - predicted[..., 9]).reshape(-1))

    if sum(rows) != int(index.get("n_rows", -1)):
        problems.append(f"row total={sum(rows)} but dataset.json says {index.get('n_rows')}")
    if len(rows) != int(index.get("n_episodes", -1)):
        problems.append(
            f"valid episode total={len(rows)} but dataset.json says {index.get('n_episodes')}")
    result = {
        "dataset": str(args.dataset),
        "episodes": len(rows),
        "rows": sum(rows),
        "rows_per_episode_min_median_max": (
            [int(min(rows)), float(np.median(rows)), int(max(rows))] if rows else []),
        "source_duration_s_min_median_max": (
            [float(min(durations)), float(np.median(durations)), float(max(durations))]
            if durations else []),
        "future_translation_m_p0_p50_p95_max": percentile(translation),
        "future_rotation_deg_p0_p50_p95_max": percentile(rotation_deg),
        "gap_m_p0_p50_p95_max": percentile(gaps),
        "last_target_horizon_s_p0_p50_p95_max": percentile(horizons),
        "source_row_step_p0_p50_p95_max": percentile(source_steps),
        "constant_velocity_shortcut": {
            "translation_l2_error_m_p0_p50_p95_max": percentile(
                velocity_translation_error),
            "rotation6d_abs_error_p0_p50_p95_max": percentile(
                velocity_rotation6d_error),
            "gap_abs_error_m_p0_p50_p95_max": percentile(velocity_gap_error),
        },
        "problems": problems,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved: {args.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
