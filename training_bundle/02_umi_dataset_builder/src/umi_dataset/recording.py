from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "video.mp4",
    "frames.csv",
    "encoded.csv",
    "accelerometer.csv",
    "gyroscope.csv",
    "manifest.json",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_recording(session: Path, require_success: bool = True) -> dict:
    session = session.resolve()
    missing = [name for name in REQUIRED_FILES if not (session / name).is_file()]
    if missing:
        raise ValueError(f"{session.name}: missing {', '.join(missing)}")
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "status": "recorded",
        "schema_version": 2,
        "app_version": "0.3",
        "image_orientation_contract": "upright_v1",
        "rotation_baked_into_pixels": True,
        "decoder_rotation_degrees": 0,
    }
    wrong = {key: (manifest.get(key), value) for key, value in expected.items()
             if manifest.get(key) != value}
    if wrong:
        raise ValueError(f"{session.name}: incompatible manifest fields {wrong}")
    if require_success and manifest.get("outcome") != "success":
        raise ValueError(f"{session.name}: outcome must be success, got {manifest.get('outcome')!r}")

    frames = read_csv(session / "frames.csv")
    encoded = read_csv(session / "encoded.csv")
    if not frames or len(frames) != len(encoded):
        raise ValueError(f"{session.name}: frame/encoder row mismatch {len(frames)} != {len(encoded)}")
    frame_ns = np.asarray([int(row["sensor_timestamp_ns"]) for row in frames], dtype=np.int64)
    pts_us = np.asarray([int(row["pts_us"]) for row in encoded], dtype=np.int64)
    if np.any(np.diff(frame_ns) <= 0) or np.any(np.diff(pts_us) <= 0):
        raise ValueError(f"{session.name}: non-increasing camera timestamps")
    if [int(row["sample_index"]) for row in encoded] != list(range(len(encoded))):
        raise ValueError(f"{session.name}: encoded sample indices are not contiguous")
    alignment_us = np.abs(frame_ns // 1000 - pts_us)
    if alignment_us.max() > 2:
        raise ValueError(f"{session.name}: camera/encoder timestamps differ by {alignment_us.max()} us")
    return {
        "session": session,
        "manifest": manifest,
        "frame_count": len(frames),
        "times_s": (pts_us - pts_us[0]).astype(np.float64) / 1e6,
        "duration_s": float((pts_us[-1] - pts_us[0]) / 1e6),
        "camera_encoder_max_error_us": int(alignment_us.max()),
    }


def longest_false_run(values) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest
