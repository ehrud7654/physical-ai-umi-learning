"""Describe the local Zarr input consumed directly by the GPU training server."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_training_input(dataset: Path) -> dict:
    dataset = Path(dataset).resolve()
    report = dataset.with_suffix(".report.json")
    if not dataset.is_file() or not report.is_file():
        raise FileNotFoundError("Zarr archive or its report is missing")
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "report": str(report),
        "report_sha256": sha256(report),
    }
    manifest_path = dataset.parent / f"{dataset.name.removesuffix('.zarr.zip')}_gpu_input.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = dataset.parent / f"{dataset.name.removesuffix('.zarr.zip')}_gpu_input.tgz"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        bundle.add(dataset, arcname=dataset.name)
        bundle.add(report, arcname=report.name)
        bundle.add(manifest_path, arcname=manifest_path.name)
    temporary.replace(archive)
    return {
        **manifest,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
