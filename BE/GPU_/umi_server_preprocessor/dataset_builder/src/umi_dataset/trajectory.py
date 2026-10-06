from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _longest_contiguous(indices: list[int]) -> list[int]:
    runs = []
    current = []
    for index in indices:
        if current and index != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(index)
    if current:
        runs.append(current)
    return max(runs, key=len, default=[])


def load_tracked_trajectory(path: Path, frame_times_s: np.ndarray,
                            minimum_frames: int, maximum_timestamp_error_ms: float,
                            maximum_lost_after_initialization: int | None = None):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(frame_times_s):
        raise ValueError(f"trajectory/frame count mismatch {len(rows)} != {len(frame_times_s)}")
    frame_indices = [int(row["frame_idx"]) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError("trajectory frame indices are not contiguous")
    tracked = [index for index, row in enumerate(rows)
               if row["state"] == "2" and row["is_lost"].lower() == "false"]
    selected = _longest_contiguous(tracked)
    if len(selected) < minimum_frames:
        raise ValueError(f"longest tracked segment has {len(selected)} frames, need {minimum_frames}")
    # Frames before the first tracked one are initialisation lag; losses after it mean unreliable tracking.
    lost_after_initialization = len(rows) - tracked[0] - len(tracked)
    if maximum_lost_after_initialization is not None and lost_after_initialization > maximum_lost_after_initialization:
        raise ValueError(f"{lost_after_initialization} frames lost after initialization, "
                         f"limit {maximum_lost_after_initialization}")
    timestamp_error_ms = np.abs(
        np.asarray([float(rows[index]["timestamp"]) for index in selected]) - frame_times_s[selected]
    ) * 1000
    if timestamp_error_ms.max() > maximum_timestamp_error_ms:
        raise ValueError(f"trajectory timestamps differ by {timestamp_error_ms.max():.3f} ms")
    position = np.asarray([[float(rows[index][key]) for key in ("x", "y", "z")]
                           for index in selected], dtype=np.float64)
    quaternion = np.asarray([[float(rows[index][key]) for key in ("q_x", "q_y", "q_z", "q_w")]
                             for index in selected], dtype=np.float64)
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("trajectory contains non-finite values")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.max(np.abs(norms - 1)) > 1e-3:
        raise ValueError("trajectory quaternion is not normalized")
    return np.asarray(selected, dtype=np.int64), position, quaternion, {
        "source_frames": len(rows),
        "tracked_frames": len(tracked),
        "first_tracked_frame": tracked[0],
        "lost_frames_after_initialization": lost_after_initialization,
        "selected_start_frame": selected[0],
        "selected_end_frame": selected[-1],
        "selected_frames": len(selected),
        "maximum_timestamp_error_ms": float(timestamp_error_ms.max()),
    }
