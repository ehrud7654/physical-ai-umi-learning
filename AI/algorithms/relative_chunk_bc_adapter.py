"""Platform entrypoint for the diagnostic RelativeChunkBC algorithm.

Public contract:

    train(dataset, profile, run_dir, reporter, cancel_event) -> manifest

``dataset`` must be a v10 ``umi_relative_chunk/0.2.0-provisional`` directory.
Official UMI Zarr input is converted by ``tools/convert_umi_zarr_to_v10.py``
before this entrypoint is called.  The function never commands a robot and
does not perform policy inference outside validation loss computation.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import yaml

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from data.relative_chunk_dataset import RelativeChunkDataset
from policy.relative_chunk_bc import RelativeChunkBCNet
from tools.train_relative_chunk_bc import (
    lateral_sampling_weights,
    run_epoch,
    save,
    split_by_episode,
)
from umi.relative_dataset import ACTION_DIM, SCHEMA


ADAPTER_SCHEMA = "relative_chunk_bc_training_manifest/0.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _emit(reporter: Any, status: str, stage: str, percent: int, **extra: Any) -> None:
    """Support the current ProgressReporter and a one-payload callable."""
    if reporter is None:
        return
    target = getattr(reporter, "report", reporter)
    if not callable(target):
        raise TypeError("reporter must be callable or expose report()")
    signature = inspect.signature(target)
    positional = [
        parameter for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY,
                              parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_keyword = any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    payload = {"status": status, "stage": stage, "percent": int(percent), **extra}
    if len(positional) <= 1 and not has_var_keyword:
        target(payload)
    else:
        target(status, stage, int(percent), **extra)


def _cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    method = getattr(cancel_event, "is_set", None)
    if not callable(method):
        raise TypeError("cancel_event must expose is_set()")
    return bool(method())


def _resolve_config(profile: Mapping[str, Any]) -> Path:
    raw = profile.get("config")
    if not raw:
        raise ValueError("RelativeChunkBC profile requires an explicit config path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = AI_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RelativeChunkBC config not found: {path}")
    return path


def _base_manifest(dataset: Path, run_dir: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": ADAPTER_SCHEMA,
        "algorithm": "RelativeChunkBC",
        "status": "CREATED",
        "dataset": str(dataset),
        "run_dir": str(run_dir),
        "profile": json.loads(json.dumps(dict(profile), default=str)),
        "started_at_unix_s": time.time(),
        "artifacts": {},
        "metrics": {},
        "limitations": [
            "Diagnostic BC baseline; not the validated product diffusion policy.",
            "Validation loss alone is not robot success or physical deployment evidence.",
            "The v10 source may contain provisional calibration provenance.",
        ],
    }


def train(dataset, profile, run_dir, reporter, cancel_event) -> dict[str, Any]:
    """Train RelativeChunkBC and return the persisted run manifest.

    ``profile`` is a mapping. Required: ``config``. Optional overrides:
    ``device``, ``epochs``, ``batch_size``, ``seed``,
    ``lateral_sampling_factor`` and ``num_workers``.
    """
    dataset_path = Path(dataset).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    if not isinstance(profile, Mapping):
        raise TypeError("profile must be a mapping")
    profile = dict(profile)
    manifest_path = run_path / "manifest.json"
    manifest = _base_manifest(dataset_path, run_path, profile)
    run_path.mkdir(parents=True, exist_ok=True)
    if (run_path / "checkpoints" / "last.pt").exists() or (run_path / "checkpoints" / "best.pt").exists():
        raise FileExistsError(f"run_dir already contains RelativeChunkBC checkpoints: {run_path}")

    try:
        if _cancelled(cancel_event):
            manifest.update(status="CANCELLED", finished_at_unix_s=time.time())
            _write_manifest(manifest_path, manifest)
            _emit(reporter, "CANCELLED", "PREPROCESSING", 0)
            return manifest

        config_path = _resolve_config(profile)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        train_config = config["train"]
        epochs = int(profile.get("epochs", train_config["epochs"]))
        batch_size = int(profile.get("batch_size", train_config["batch_size"]))
        seed = int(profile.get("seed", train_config["seed"]))
        factor = float(profile.get("lateral_sampling_factor", 1.0))
        workers = int(profile.get("num_workers", train_config["num_workers"]))
        device_name = str(profile.get("device", "cuda"))
        if epochs < 1 or batch_size < 1 or workers < 0:
            raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        _emit(reporter, "RUNNING", "PREPROCESSING", 2,
              epoch=0, totalEpochs=epochs)
        dataset_index = dataset_path / "dataset.json"
        if not dataset_index.is_file():
            raise FileNotFoundError(f"v10 dataset.json not found: {dataset_index}")
        dataset_meta = json.loads(dataset_index.read_text(encoding="utf-8"))
        if dataset_meta.get("schema") != SCHEMA:
            raise ValueError(
                f"RelativeChunkBC requires {SCHEMA}, got {dataset_meta.get('schema')!r}"
            )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        training_data = RelativeChunkDataset(
            dataset_path,
            image_mean=float(config["data"]["image_mean"]),
            image_std=float(config["data"]["image_std"]),
        )
        train_indices, val_indices, val_episodes = split_by_episode(
            training_data, float(train_config["val_fraction"]), seed
        )
        train_actions = torch.from_numpy(training_data.action_rows(train_indices))
        train_gaps = np.asarray([
            training_data.episodes[episode_index]["proprio"][row, -1, -1]
            for episode_index, row in
            (training_data.index[index] for index in train_indices)
        ], dtype=np.float64)
        sampling_weights, lateral_count = lateral_sampling_weights(
            train_actions.numpy(), train_gaps, threshold_m=0.01, factor=factor
        )
        mean = train_actions.mean(dim=0)
        std = train_actions.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        trivial = float((((train_actions - mean) / std).abs()).mean())
        sampler = (WeightedRandomSampler(
            torch.from_numpy(sampling_weights), num_samples=len(train_indices),
            replacement=True, generator=torch.Generator().manual_seed(seed))
            if factor != 1.0 else None)
        loaders = {
            "train": DataLoader(
                Subset(training_data, train_indices), batch_size=batch_size,
                shuffle=(sampler is None), sampler=sampler, num_workers=workers,
                generator=torch.Generator().manual_seed(seed)),
            "val": DataLoader(
                Subset(training_data, val_indices), batch_size=batch_size,
                shuffle=False, num_workers=workers),
        }
        model = RelativeChunkBCNet(
            obs_horizon=training_data.obs_horizon,
            action_horizon=training_data.action_horizon,
            action_dim=ACTION_DIM,
            config=config,
        ).to(device)
        mean_device, std_device = mean.to(device), std.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(train_config["lr"]),
            weight_decay=float(train_config["weight_decay"]),
        )
        criterion = nn.L1Loss()
        best = float("inf")
        best_state = None
        history: list[dict[str, float | int]] = []
        started = time.perf_counter()

        for epoch in range(1, epochs + 1):
            if _cancelled(cancel_event):
                raise InterruptedError("RelativeChunkBC training cancelled")
            train_loss = run_epoch(
                model, loaders["train"], criterion, device, mean_device,
                std_device, optimizer, float(train_config["grad_clip"]), cancel_event,
            )
            val_loss = run_epoch(
                model, loaders["val"], criterion, device, mean_device,
                std_device, None, float(train_config["grad_clip"]), cancel_event,
            )
            history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
            if val_loss < best:
                best = val_loss
                best_state = copy.deepcopy(model.state_dict())
            _emit(
                reporter, "RUNNING", "TRAINING",
                min(94, 5 + int(89 * epoch / epochs)),
                epoch=epoch, totalEpochs=epochs,
                metrics={"trainLoss": train_loss, "valLoss": val_loss,
                         "bestValLoss": best},
            )

        _emit(reporter, "RUNNING", "CHECKPOINT_CREATING", 95,
              epoch=epochs, totalEpochs=epochs)
        checkpoint_meta = {
            "policy": "relative_chunk_bc_diagnostic",
            "dataset_schema": SCHEMA,
            "trained_on": str(dataset_path),
            "observation_horizon": training_data.obs_horizon,
            "action_horizon": training_data.action_horizon,
            "action_dim": ACTION_DIM,
            "rate_hz": training_data.rate_hz,
            "target_mean": mean.tolist(),
            "target_std": std.tolist(),
            "config": config,
            "seed": seed,
            "epochs": epochs,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "val_episodes": val_episodes,
            "mean_predictor_standardised_l1": trivial,
            "lateral_sampling": {
                "factor": factor,
                "threshold_m": 0.01,
                "gap_band_m": [0.06, 0.07],
                "train_rows_selected": lateral_count,
            },
            "best_val_loss": best,
            "history": history,
            "elapsed_s": time.perf_counter() - started,
        }
        checkpoint_dir = run_path / "checkpoints"
        last_path = checkpoint_dir / "last.pt"
        best_path = checkpoint_dir / "best.pt"
        save(last_path, model, {**checkpoint_meta, "checkpoint_selection": "last_epoch"})
        if best_state is None:
            raise RuntimeError("training completed without a best checkpoint")
        model.load_state_dict(best_state)
        save(best_path, model, {**checkpoint_meta, "checkpoint_selection": "best_validation"})
        manifest.update({
            "status": "COMPLETED",
            "finished_at_unix_s": time.time(),
            "dataset_schema": SCHEMA,
            "dataset_index_sha256": _sha256(dataset_index),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "split": {
                "seed": seed,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "val_episodes": val_episodes,
            },
            "metrics": {
                "best_val_loss": best,
                "mean_predictor_standardised_l1": trivial,
                "history": history,
            },
            "artifacts": {
                "last": {"path": str(last_path), "size_bytes": last_path.stat().st_size,
                         "sha256": _sha256(last_path)},
                "best": {"path": str(best_path), "size_bytes": best_path.stat().st_size,
                         "sha256": _sha256(best_path)},
            },
        })
        manifest.update({
            "checkpointStorageKey": str(best_path).replace("\\", "/"),
            "sizeBytes": best_path.stat().st_size,
            "sha256": manifest["artifacts"]["best"]["sha256"],
            "metadata": {
                "algorithm": "RelativeChunkBC",
                "datasetSchema": SCHEMA,
                "checkpointSelection": "best_validation",
                "bestValLoss": best,
            },
        })
        _write_manifest(manifest_path, manifest)
        _emit(reporter, "COMPLETED", "COMPLETED", 100,
              epoch=epochs, totalEpochs=epochs,
              manifest=str(manifest_path), artifacts=manifest["artifacts"])
        return manifest
    except InterruptedError as exc:
        manifest.update(status="CANCELLED", finished_at_unix_s=time.time(),
                        cancellation_reason=str(exc))
        _write_manifest(manifest_path, manifest)
        _emit(reporter, "CANCELLED", "TRAINING", 0, error={"message": str(exc)})
        return manifest
    except Exception as exc:
        manifest.update(status="FAILED", finished_at_unix_s=time.time(),
                        error={"type": type(exc).__name__, "message": str(exc)})
        _write_manifest(manifest_path, manifest)
        _emit(reporter, "FAILED", "FAILED", 0,
              error={"code": "RELATIVE_CHUNK_BC_FAILED", "message": str(exc)})
        raise
