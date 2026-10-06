"""Validate a 02_umi_dataset_builder ReplayBuffer and its report before training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

EXPECTED_SHAPES = {
    "camera0_rgb": (224, 224, 3),
    "robot0_eef_pos": (3,),
    "robot0_eef_rot_axis_angle": (3,),
    "robot0_gripper_width": (1,),
    "robot0_demo_start_pose": (6,),
    "robot0_demo_end_pose": (6,),
}
TRAINABLE_STATUSES = {"ready", "provisional_camera_tcp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_path(dataset: Path) -> Path:
    # 02 writes dataset.zarr.zip -> dataset.zarr.report.json
    return dataset.with_suffix(".report.json")


def check_dataset(dataset: Path, allow_development: bool = False,
                  native_hz: float | None = None) -> dict:
    dataset = Path(dataset).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    with zarr.ZipStore(str(dataset), mode="r") as store:
        root = zarr.group(store=store)
        ends = np.asarray(root["meta/episode_ends"][:])
        if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(np.r_[0, ends]) <= 0):
            raise ValueError("meta/episode_ends must be non-empty and strictly increasing")
        total = int(ends[-1])
        arrays = {}
        for key, shape in EXPECTED_SHAPES.items():
            if key not in root["data"]:
                raise ValueError(f"missing data/{key}")
            array = root["data"][key]
            if tuple(array.shape) != (total, *shape):
                raise ValueError(f"data/{key} shape {array.shape} != {(total, *shape)}")
            arrays[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            if key != "camera0_rgb" and not np.isfinite(array[:]).all():
                raise ValueError(f"data/{key} contains non-finite values")
        unexpected = sorted(set(root["data"].keys()) - set(EXPECTED_SHAPES))
    digest = sha256(dataset)
    report_file = report_path(dataset)
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else {}
    if report and report.get("dataset_sha256") != digest:
        raise ValueError(f"{dataset.name} does not match dataset_sha256 in {report_file.name}")
    status = report.get("training_input_status", "no_builder_report")
    if status not in TRAINABLE_STATUSES and not allow_development:
        raise ValueError(f"training_input_status={status}; pass --allow-development to train anyway")
    rate = native_hz or report.get("native_sample_rate_hz")
    if not rate:
        raise ValueError("native sample rate unknown: builder report missing, pass --native-hz")
    return {
        "dataset": str(dataset),
        "dataset_sha256": digest,
        "frames": total,
        "episodes": int(len(ends)),
        "episode_lengths": np.diff(np.r_[0, ends]).tolist(),
        "arrays": arrays,
        "unexpected_arrays": unexpected,
        "native_sample_rate_hz": float(rate),
        "training_input_status": status,
        "camera_tcp_status": report.get("camera_tcp_status"),
        "physical_deployment_ready": bool(report.get("physical_deployment_ready", False)),
        "builder_report": str(report_file) if report else None,
    }
