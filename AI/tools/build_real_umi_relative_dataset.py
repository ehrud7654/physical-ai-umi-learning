"""Build a no-time-dilation, camera-centric action-chunk pilot from 0911 UMI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from umi.camera_frames import load_arcore_pinch_calibration
from umi.relative_dataset import (
    SCHEMA,
    load_longest_relative_run,
    relative_vector,
    write_episode,
)


def resize_rgb(data: bytes, rotation_deg: int) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        if rotation_deg not in (0, 180):
            raise ValueError("camera.rotate_to_canonical_deg must be 0 or 180")
        if rotation_deg == 180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def nearest_grid_indices(timestamps: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Select real frames nearest a fixed-rate grid; never stretch the source time."""
    period = 1.0 / rate_hz
    grid = timestamps[0] + np.arange(
        int(np.floor((timestamps[-1] - timestamps[0]) / period)) + 1) * period
    indices = np.asarray([int(np.argmin(np.abs(timestamps - value))) for value in grid])
    keep = np.r_[True, np.diff(indices) > 0]
    indices, grid = indices[keep], grid[keep]
    if len(indices) < 2 or np.max(np.abs(timestamps[indices] - grid)) > 0.55 * period:
        raise ValueError("source frames do not cover the requested real-time grid")
    return indices, grid


def digest(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in paths:
        value.update(path.name.encode())
        value.update(hashlib.sha256(path.read_bytes()).digest())
    return value.hexdigest()


def build_one(bundle: Path, quality_path: Path, t_camera_pinch: np.ndarray, *,
              rate_hz: float, obs_horizon: int, action_horizon: int,
              output_dir: Path, calibration: dict,
              image_rotation_override: int | None) -> dict[str, object]:
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    relative, gaps, source_rows = load_longest_relative_run(
        bundle, quality, t_camera_pinch)
    with zipfile.ZipFile(bundle) as archive:
        poses = list(csv.DictReader(io.StringIO(
            archive.read("poses.csv").decode("utf-8-sig"))))
        frames = list(csv.DictReader(io.StringIO(
            archive.read("frames.csv").decode("utf-8-sig"))))
        frame_by_index = {int(row["frame_index"]): row for row in frames}
        episode_json = json.loads(archive.read("episode.json"))
        declared_rotation = episode_json["camera"].get("rotate_to_canonical_deg")
        if declared_rotation is None and image_rotation_override is None:
            raise ValueError(
                "camera.rotate_to_canonical_deg is absent; pass "
                "--image-rotation-deg only with reviewed mounting evidence")
        rotation = int(declared_rotation if declared_rotation is not None
                       else image_rotation_override)
        if (declared_rotation is not None and image_rotation_override is not None
                and int(declared_rotation) != image_rotation_override):
            raise ValueError("image rotation override conflicts with episode metadata")
        timestamps = np.asarray(
            [int(poses[int(row)]["timestamp_ns"]) * 1e-9 for row in source_rows],
            dtype=np.float64)
        sampled, _ = nearest_grid_indices(timestamps, rate_hz)
        poses_sampled = relative[sampled]
        gaps_sampled = gaps[sampled]
        source_sampled = np.asarray(source_rows, dtype=np.int64)[sampled]
        first = obs_horizon - 1
        last = len(sampled) - action_horizon
        if last <= first:
            raise ValueError(
                f"real-time samples={len(sampled)} cannot fit obs={obs_horizon}, "
                f"action={action_horizon}")

        rows = np.arange(first, last, dtype=int)
        images: list[np.ndarray] = []
        proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        observation_times: list[np.ndarray] = []
        action_times: list[np.ndarray] = []
        decoded: dict[int, np.ndarray] = {}
        for current in rows:
            hist_indices = range(current - obs_horizon + 1, current + 1)
            future_indices = range(current + 1, current + action_horizon + 1)
            image_history = []
            proprio_history = []
            for index in hist_indices:
                source_row = int(source_sampled[index])
                frame_index = int(poses[source_row]["index"])
                if frame_index not in frame_by_index:
                    raise ValueError(f"missing RGB for frame index {frame_index}")
                if frame_index not in decoded:
                    decoded[frame_index] = resize_rgb(
                        archive.read(frame_by_index[frame_index]["image"]), rotation)
                image_history.append(decoded[frame_index].transpose(2, 0, 1))
                proprio_history.append(relative_vector(
                    poses_sampled[current], poses_sampled[index], gaps_sampled[index]))
            action_history = [relative_vector(
                poses_sampled[current], poses_sampled[index], gaps_sampled[index])
                for index in future_indices]
            images.append(np.stack(image_history))
            proprio.append(np.stack(proprio_history))
            actions.append(np.stack(action_history))
            observation_times.append(timestamps[sampled[list(hist_indices)]])
            action_times.append(timestamps[sampled[list(future_indices)]])

    arrays = {
        "image": np.stack(images).astype(np.uint8),
        "proprio": np.stack(proprio).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "observation_timestamp": np.stack(observation_times).astype(np.float64),
        "action_timestamp": np.stack(action_times).astype(np.float64),
        "source_row": source_sampled[rows].astype(np.int64),
    }
    source_duration = float(timestamps[sampled[-1]] - timestamps[sampled[0]])
    meta = {
        "schema": SCHEMA,
        "episode_id": bundle.stem,
        "source_bundle": bundle.name,
        "source_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "calibration_id": calibration["calibration_id"],
        "calibration_status": calibration["status"],
        "camera_rotation_to_canonical_deg": rotation,
        "camera_rotation_source": (
            "episode_metadata" if declared_rotation is not None else "cli_reviewed_override"),
        "rate_hz": rate_hz,
        "observation_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "action_semantics": "all future poses relative to the same current pinch pose; gap absolute metres",
        "action_columns": ["x_m", "y_m", "z_m", "r0x", "r0y", "r0z",
                           "r1x", "r1y", "r1z", "gap_m"],
        "time_scale": 1.0,
        "source_time_preserved": True,
        "source_duration_s": source_duration,
        "training_rows": int(len(rows)),
        "success_label": "unverified",
        "robot_base_alignment_applied": False,
        "robot_ik_applied": False,
    }
    write_episode(output_dir / f"{bundle.stem}.npz", arrays, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--image-rotation-deg", type=int, choices=(0, 180),
                        help="reviewed fallback only when old canonical bundles lack metadata")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    if args.rate_hz <= 0 or args.obs_horizon < 1 or args.action_horizon < 1:
        raise SystemExit("invalid rate or horizons")
    calibration, transform = load_arcore_pinch_calibration(args.extrinsic)
    bundles = sorted(args.bundles.glob("rec_*.zip"))
    if args.limit is not None:
        bundles = bundles[:args.limit]
    if not bundles:
        raise SystemExit(f"no bundles: {args.bundles}")
    args.out.mkdir(parents=True)
    accepted, rejected = [], {}
    for bundle in bundles:
        quality = bundle.with_suffix(".quality.json")
        try:
            meta = build_one(
                bundle, quality, transform, rate_hz=args.rate_hz,
                obs_horizon=args.obs_horizon, action_horizon=args.action_horizon,
                output_dir=args.out, calibration=calibration,
                image_rotation_override=args.image_rotation_deg)
            accepted.append(meta)
            print(f"{bundle.stem}: {meta['training_rows']} rows")
        except Exception as exc:
            rejected[bundle.stem] = str(exc).splitlines()[0]
            print(f"{bundle.stem}: rejected - {rejected[bundle.stem]}")
    index = {
        "schema": SCHEMA,
        "status": "PROVISIONAL — relative action contract pending Track A/B agreement",
        "source": str(args.bundles),
        "source_digest": digest(bundles),
        "episodes": [item["episode_id"] for item in accepted],
        "n_episodes": len(accepted),
        "n_rows": sum(int(item["training_rows"]) for item in accepted),
        "rejected": rejected,
        "rate_hz": args.rate_hz,
        "observation_horizon": args.obs_horizon,
        "action_horizon": args.action_horizon,
        "time_scale": 1.0,
        "source_time_preserved": True,
        "robot_base_alignment_applied": False,
        "robot_ik_applied": False,
        "calibration_id": calibration["calibration_id"],
    }
    (args.out / "dataset.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: index[key] for key in
                      ("n_episodes", "n_rows", "rejected")}, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
