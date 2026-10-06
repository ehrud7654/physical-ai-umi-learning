#!/usr/bin/env python3
"""Run RAW -> Atlas -> UMI Zarr and package the GPU training input."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from umi_preprocessor.contracts import (  # noqa: E402
    extract_archive,
    validate_config_dir,
    validate_name,
    validate_raw_layout,
)
from umi_preprocessor.packaging import package_training_input  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def job_local_file(root: Path, value: Path | None, field: str) -> Path | None:
    if value is None:
        return None
    target = value.resolve() if value.is_absolute() else (root / value).resolve()
    if root not in target.parents or not target.is_file():
        raise ValueError(f"{field} is missing or outside the job root: {value}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="job directory containing raw/ and raw_mapping/")
    parser.add_argument("--archive", type=Path, help="optional .tgz/.tar.gz/.zip to extract into an empty root")
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--require-mapping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mapping-session-id")
    parser.add_argument("--demonstration-session-id", action="append", default=[])
    parser.add_argument("--external-atlas", type=Path)
    parser.add_argument("--external-alignment", type=Path)
    parser.add_argument("--external-atlas-sha256")
    parser.add_argument("--external-alignment-sha256")
    parser.add_argument("--allow-unrated", action="store_true")
    parser.add_argument("--allow-tracking-loss", action="store_true")
    parser.add_argument("--allow-missing-tag", action="store_true")
    args = parser.parse_args()

    validate_name(args.dataset_name, "datasetName")
    root = args.root.resolve()
    status_path = root / "job_status.json"
    status = {"status": "running", "startedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        if args.archive:
            extract_archive(args.archive, root)
        else:
            root.mkdir(parents=True, exist_ok=True)
        write_json(status_path, status)
        external_atlas = job_local_file(root, args.external_atlas, "externalAtlas")
        external_alignment = job_local_file(root, args.external_alignment, "externalAlignment")
        if (external_atlas is None) != (external_alignment is None):
            raise ValueError("external Atlas and alignment must be provided together")
        if external_atlas is not None:
            if not args.external_atlas_sha256 or not args.external_alignment_sha256:
                raise ValueError("external Atlas use requires both expected SHA-256 values")
            actual_atlas = sha256(external_atlas)
            actual_alignment = sha256(external_alignment)
            if actual_atlas.lower() != args.external_atlas_sha256.lower():
                raise ValueError("external Atlas SHA-256 mismatch")
            if actual_alignment.lower() != args.external_alignment_sha256.lower():
                raise ValueError("external alignment SHA-256 mismatch")
        status["input"] = validate_raw_layout(
            root,
            args.require_mapping,
            args.mapping_session_id,
            args.demonstration_session_id or None,
            external_atlas,
            external_alignment,
        )
        config_dir = args.config_dir or HERE.parent / "dataset_builder/configs"
        os.environ["UMI_CONFIG_DIR"] = str(validate_config_dir(config_dir))
        from umi_preprocessor.pipeline import run

        report = run(
            root,
            dataset_name=args.dataset_name,
            allow_unrated=args.allow_unrated,
            allow_tracking_loss=args.allow_tracking_loss,
            require_mapping=args.require_mapping,
            allow_missing_tag=args.allow_missing_tag,
            mapping_session_id=args.mapping_session_id,
            demonstration_session_ids=args.demonstration_session_id or None,
            external_atlas=external_atlas,
            external_alignment=external_alignment,
        )
        if report.get("status") != "ready":
            raise RuntimeError(report.get("stopped", "pipeline did not produce a ready dataset"))
        dataset_report = report.get("dataset") or {}
        if dataset_report.get("training_input_status") != "ready":
            raise RuntimeError(
                "dataset training_input_status is not ready: "
                f"{dataset_report.get('training_input_status', 'missing')}"
            )
        artifact = package_training_input(Path(report["dataset"]["dataset"]))
        status.update(status="ready", completedUtc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      report=str(root / "pipeline_report.json"), artifact=artifact)
        write_json(status_path, status)
        print(json.dumps(status, indent=2))
    except Exception as error:
        root.mkdir(parents=True, exist_ok=True)
        status.update(status="failed", completedUtc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      error={"type": type(error).__name__, "message": str(error)})
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
