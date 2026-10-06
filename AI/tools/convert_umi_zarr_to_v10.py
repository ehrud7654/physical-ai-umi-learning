"""Convert an official-UMI observation Zarr into RelativeChunkBC v10.

The conversion uses only RGB, absolute EEF pose and metric gripper width.
Global/base alignment cancels because every proprio/action target is expressed
relative to the current EEF pose.  When the Zarr has no timestamps, a rate must
be supplied explicitly or recovered from the converter provenance sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.relative_dataset import (  # noqa: E402
    ACTION_DIM,
    SCHEMA,
    validate_arrays,
    write_episode,
)


REQUIRED_KEYS = (
    "camera0_rgb",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
)
ACTION_COLUMNS = [
    "x_m", "y_m", "z_m", "r0x", "r0y", "r0z",
    "r1x", "r1y", "r1z", "gap_m",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Vectorised Rodrigues conversion without a SciPy dependency."""
    values = np.asarray(rotvec, dtype=np.float64)
    if values.shape[-1:] != (3,) or not np.isfinite(values).all():
        raise ValueError("axis-angle must be a finite (..., 3) array")
    flat = values.reshape(-1, 3)
    result = np.empty((len(flat), 3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for index, vector in enumerate(flat):
        angle = float(np.linalg.norm(vector))
        if angle < 1e-12:
            # First-order form is more stable and preserves tiny rotations.
            x, y, z = vector
            skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            result[index] = eye + skew
            continue
        axis = vector / angle
        x, y, z = axis
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        result[index] = eye + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return result.reshape(*values.shape[:-1], 3, 3)


def poses_from_arrays(position: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or rotvec.shape != position.shape:
        raise ValueError("EEF position and axis-angle must both have shape (N, 3)")
    if not np.isfinite(position).all():
        raise ValueError("EEF position contains NaN/Inf")
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(position), axis=0)
    transforms[:, :3, :3] = rotvec_to_matrix(rotvec)
    transforms[:, :3, 3] = position
    return transforms


def relative_vector(reference: np.ndarray, target: np.ndarray, gap_m: float) -> np.ndarray:
    relative = np.linalg.inv(reference) @ target
    gap = float(gap_m)
    if not np.isfinite(gap) or not 0.0 <= gap <= 0.09:
        raise ValueError(f"gripper width {gap!r} is outside [0, 0.09] m")
    return np.r_[relative[:3, 3], relative[:2, :3].reshape(-1), gap]


def episode_to_v10(
    rgb: np.ndarray,
    position: np.ndarray,
    rotvec: np.ndarray,
    gripper_width: np.ndarray,
    *,
    rate_hz: float,
    observation_horizon: int = 2,
    action_horizon: int = 8,
) -> dict[str, np.ndarray]:
    """Build one v10 episode from one contiguous official-UMI episode."""
    rgb = np.asarray(rgb)
    position = np.asarray(position)
    rotvec = np.asarray(rotvec)
    gripper = np.asarray(gripper_width, dtype=np.float64).reshape(-1)
    count = len(rgb)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("camera0_rgb must be uint8 (N, H, W, 3)")
    if rgb.shape[1:3] != (224, 224):
        raise ValueError(f"camera0_rgb must already be 224x224, got {rgb.shape[1:3]}")
    if len(position) != count or len(rotvec) != count or len(gripper) != count:
        raise ValueError("official UMI arrays have inconsistent frame counts")
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be a finite positive value")
    if observation_horizon < 1 or action_horizon < 1:
        raise ValueError("observation/action horizons must be positive")
    transforms = poses_from_arrays(position, rotvec)
    anchors = range(observation_horizon - 1, count - action_horizon)
    n_rows = max(0, count - action_horizon - observation_horizon + 1)
    if n_rows <= 0:
        raise ValueError(
            f"episode has {count} frames; need at least "
            f"{observation_horizon + action_horizon}"
        )

    images = np.empty((n_rows, observation_horizon, 3, 224, 224), dtype=np.uint8)
    proprio = np.empty((n_rows, observation_horizon, ACTION_DIM), dtype=np.float32)
    action = np.empty((n_rows, action_horizon, ACTION_DIM), dtype=np.float32)
    observation_timestamp = np.empty((n_rows, observation_horizon), dtype=np.float64)
    action_timestamp = np.empty((n_rows, action_horizon), dtype=np.float64)
    source_row = np.empty((n_rows,), dtype=np.int64)
    time_s = np.arange(count, dtype=np.float64) / float(rate_hz)

    for output_row, current in enumerate(anchors):
        obs_indices = list(range(current - observation_horizon + 1, current + 1))
        future_indices = list(range(current + 1, current + action_horizon + 1))
        images[output_row] = np.transpose(rgb[obs_indices], (0, 3, 1, 2))
        for history_row, source in enumerate(obs_indices):
            proprio[output_row, history_row] = relative_vector(
                transforms[current], transforms[source], gripper[source]
            ).astype(np.float32)
        for future_row, source in enumerate(future_indices):
            action[output_row, future_row] = relative_vector(
                transforms[current], transforms[source], gripper[source]
            ).astype(np.float32)
        observation_timestamp[output_row] = time_s[obs_indices]
        action_timestamp[output_row] = time_s[future_indices]
        source_row[output_row] = current

    arrays = {
        "image": images,
        "proprio": proprio,
        "action": action,
        "observation_timestamp": observation_timestamp,
        "action_timestamp": action_timestamp,
        "source_row": source_row,
    }
    problems = validate_arrays(
        arrays,
        obs_horizon=observation_horizon,
        action_horizon=action_horizon,
    )
    if problems:
        raise ValueError("converted episode violates v10: " + "; ".join(problems))
    return arrays


def _sidecar_path(path: Path) -> Path:
    text = str(path)
    for suffix in (".zarr.zip", ".zarr"):
        if text.endswith(suffix):
            return Path(text[: -len(suffix)] + ".provenance.json")
    return Path(text + ".provenance.json")


def _episode_names(source: Path, count: int, sidecar: Mapping[str, Any]) -> list[str]:
    names = sidecar.get("episodes_kept_names")
    if isinstance(names, list) and len(names) == count and len(set(map(str, names))) == count:
        result = [str(value) for value in names]
        if all(re.fullmatch(r"[A-Za-z0-9_.-]+", name)
               and name not in {".", ".."} for name in result):
            return result
        raise ValueError("provenance episode names contain unsafe path characters")
    stem = source.name.replace(".zarr.zip", "").replace(".zarr", "")
    return [f"{stem}_episode_{index:05d}" for index in range(count)]


def convert_zarr(
    source: Path,
    output: Path,
    *,
    rate_hz: float | None,
    observation_horizon: int = 2,
    action_horizon: int = 8,
) -> dict[str, Any]:
    if observation_horizon != 2 or action_horizon != 8:
        raise ValueError(
            "umi_relative_chunk/0.2.0-provisional requires "
            "observation_horizon=2 and action_horizon=8"
        )
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required for UMI Zarr conversion") from exc

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    sidecar_path = _sidecar_path(source)
    sidecar = (json.loads(sidecar_path.read_text(encoding="utf-8"))
               if sidecar_path.is_file() else {})
    if rate_hz is None:
        candidate = sidecar.get("rate_hz")
        if not isinstance(candidate, (int, float)):
            raise ValueError("rate_hz is absent; pass --rate-hz or provide provenance rate_hz")
        rate_hz = float(candidate)

    store = None
    try:
        if source.is_file():
            zip_store = getattr(zarr, "ZipStore", None)
            if zip_store is None:
                zip_store = zarr.storage.ZipStore
            store = zip_store(str(source), mode="r")
            root = zarr.open_group(store=store, mode="r")
        else:
            root = zarr.open_group(str(source), mode="r")
        data = root["data"]
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"official UMI Zarr is missing arrays: {missing}")
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(ends) <= 0):
            raise ValueError("meta/episode_ends must be a non-empty increasing vector")
        total = int(ends[-1])
        arrays = {key: np.asarray(data[key][:]) for key in REQUIRED_KEYS}
    finally:
        if store is not None:
            store.close()
    if any(len(value) != total for value in arrays.values()):
        raise ValueError("Zarr array length does not match final episode_end")

    names = _episode_names(source, len(ends), sidecar)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    start = 0
    for episode_index, (name, end_value) in enumerate(zip(names, ends)):
        end = int(end_value)
        try:
            converted = episode_to_v10(
                arrays["camera0_rgb"][start:end],
                arrays["robot0_eef_pos"][start:end],
                arrays["robot0_eef_rot_axis_angle"][start:end],
                arrays["robot0_gripper_width"][start:end],
                rate_hz=float(rate_hz),
                observation_horizon=observation_horizon,
                action_horizon=action_horizon,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            dropped.append({"episode": name, "index": episode_index, "reason": str(exc)})
            start = end
            continue
        metadata = {
            "schema": SCHEMA,
            "episode_id": name,
            "source": "official UMI Zarr",
            "source_zarr": str(source),
            "source_episode_index": episode_index,
            "source_frame_range": [start, end],
            "rate_hz": float(rate_hz),
            "observation_horizon": observation_horizon,
            "action_horizon": action_horizon,
            "action_semantics": (
                "all future TCP poses relative to the same current TCP pose; "
                "gap absolute metres"
            ),
            "action_columns": ACTION_COLUMNS,
            "rotation_6d": "first two rows; row-major; Gram-Schmidt decode",
            "source_time_preserved": False,
            "timestamp_provenance": "synthesised from rate_hz; source Zarr has no timestamps",
            "training_rows": int(len(converted["action"])),
            "robot_base_alignment_applied": False,
            "robot_ik_applied": False,
            "physical_deployment_ready": False,
        }
        write_episode(output / f"{name}.npz", converted, metadata)
        kept.append(name)
        episode_rows.append({"episode": name, "source_frames": end - start,
                             "training_rows": int(len(converted["action"]))})
        start = end

    if not kept:
        raise RuntimeError("conversion kept zero episodes; refusing an empty v10 dataset")

    index = {
        "schema": SCHEMA,
        "status": "DERIVED_FROM_OFFICIAL_UMI_ZARR_NOT_FOR_PHYSICAL_DEPLOYMENT",
        "episodes": kept,
        "n_episodes": len(kept),
        "n_rows": sum(row["training_rows"] for row in episode_rows),
        "rejected": {row["episode"]: row["reason"] for row in dropped},
        "rate_hz": float(rate_hz),
        "observation_horizon": observation_horizon,
        "action_horizon": action_horizon,
        "robot_base_alignment_applied": False,
        "robot_ik_applied": False,
        "physical_deployment_ready": False,
    }
    (output / "dataset.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "umi_zarr_to_relative_chunk_v10/0.1.0",
        "source": str(source),
        "source_sha256": sha256(source) if source.is_file() else None,
        "source_provenance": str(sidecar_path) if sidecar_path.is_file() else None,
        "output": str(output),
        "rate_hz": float(rate_hz),
        "observation_horizon": observation_horizon,
        "action_horizon": action_horizon,
        "episodes_requested": len(ends),
        "episodes_kept": len(kept),
        "episodes_dropped": len(dropped),
        "rows": index["n_rows"],
        "episode_rows": episode_rows,
        "dropped": dropped,
        "limitations": [
            "Timestamps are synthesised from rate_hz because official UMI Zarr omits source timestamps.",
            "Original v10 source_row and H=2 source-frame identities cannot be recovered bit-for-bit.",
            "Global pose alignment cancels in relative labels; this does not create physical base calibration.",
        ],
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def selftest() -> int:
    length = 14
    rgb = np.zeros((length, 224, 224, 3), dtype=np.uint8)
    rgb[:, 0, 0, 0] = np.arange(length, dtype=np.uint8)
    position = np.zeros((length, 3), dtype=np.float64)
    position[:, 0] = np.arange(length) * 0.01
    rotvec = np.zeros((length, 3), dtype=np.float64)
    rotvec[:, 2] = np.arange(length) * math.radians(2.0)
    gap = np.linspace(0.079, 0.04, length, dtype=np.float64)[:, None]
    arrays = episode_to_v10(rgb, position, rotvec, gap, rate_hz=10.0)
    checks = [
        ("row count", arrays["action"].shape == (5, 8, 10)),
        ("H=2 image order", arrays["image"][0, :, 0, 0, 0].tolist() == [0, 1]),
        ("current proprio identity", np.allclose(arrays["proprio"][:, -1, :3], 0.0)),
        ("first future translation norm", np.allclose(
            np.linalg.norm(arrays["action"][:, 0, :3], axis=1), 0.01, atol=1e-8)),
        ("future timestamps", np.all(arrays["action_timestamp"] > arrays["observation_timestamp"][:, -1, None])),
        ("v10 validator", not validate_arrays(arrays, obs_horizon=2, action_horizon=8)),
    ]
    for name, passed in checks:
        print(("PASS" if passed else "FAIL") + f"  {name}")
    print(f"{sum(bool(value) for _, value in checks)}/{len(checks)} checks passed")
    return 0 if all(value for _, value in checks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate-hz", type=float)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.input is None or args.output is None:
        parser.error("--input and --output are required")
    report = convert_zarr(
        args.input,
        args.output,
        rate_hz=args.rate_hz,
        observation_horizon=args.observation_horizon,
        action_horizon=args.action_horizon,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
