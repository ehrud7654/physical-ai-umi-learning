#!/usr/bin/env python3
"""Fail-closed validation for the backend ORB-SLAM role contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root: Path, session_id: str) -> dict:
    path = root / session_id / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{session_id}: missing manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def require_session_files(root: Path, session_id: str) -> None:
    required = (
        "manifest.json", "video.mp4", "frames.csv", "encoded.csv",
        "accelerometer.csv", "gyroscope.csv",
    )
    missing = [name for name in required if not (root / session_id / name).is_file()]
    if missing:
        raise ValueError(f"{session_id}: missing {', '.join(missing)}")


def job_local_file(job_root: Path, relative: str, field: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"{field} must be relative to the job root")
    target = (job_root / value).resolve()
    if job_root.resolve() not in target.parents or not target.is_file():
        raise ValueError(f"{field} is missing or escapes the job root")
    return target


def validate(payload: dict, job_root: Path) -> dict:
    mapping_id = payload.get("mapping_session_id")
    external_path = payload.get("external_atlas_path")
    if bool(mapping_id) == bool(external_path):
        raise ValueError("provide exactly one of mapping_session_id or external Atlas")
    demonstrations = payload.get("demonstration_session_ids")
    if not isinstance(demonstrations, list) or not demonstrations:
        raise ValueError("demonstration_session_ids must be a non-empty list")
    if len(demonstrations) != len(set(demonstrations)):
        raise ValueError("duplicate demonstration session ID")
    if mapping_id in demonstrations:
        raise ValueError("a session cannot be both Mapping and Demonstration")

    mapping_summary = None
    if mapping_id:
        mapping_root = job_root / "raw_mapping"
        require_session_files(mapping_root, mapping_id)
        mapping_manifest = manifest(mapping_root, mapping_id)
        mapping_summary = {
            "session_id": mapping_id,
            "outcome": mapping_manifest.get("outcome"),
            "outcome_used_as_training_label": False,
        }
    else:
        required = (
            "external_atlas_path", "external_atlas_sha256",
            "external_alignment_path", "external_alignment_sha256",
        )
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise ValueError(f"external Atlas request missing {', '.join(missing)}")
        atlas_path = job_local_file(job_root, payload["external_atlas_path"], "external_atlas_path")
        alignment_path = job_local_file(job_root, payload["external_alignment_path"], "external_alignment_path")
        if sha256(atlas_path).lower() != payload["external_atlas_sha256"].lower():
            raise ValueError("external Atlas SHA-256 mismatch")
        if sha256(alignment_path).lower() != payload["external_alignment_sha256"].lower():
            raise ValueError("external alignment SHA-256 mismatch")
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        if alignment.get("status") not in (None, "pass", "PASS", "PASS_ARUCO_ALIGNMENT"):
            raise ValueError("external alignment status is not pass")
        mapping_summary = {
            "external_atlas": str(atlas_path),
            "external_atlas_sha256": sha256(atlas_path),
            "external_alignment": str(alignment_path),
            "external_alignment_sha256": sha256(alignment_path),
            "outcome_used_as_training_label": False,
        }

    demo_summaries = []
    demo_root = job_root / "raw"
    for session_id in demonstrations:
        require_session_files(demo_root, session_id)
        data = manifest(demo_root, session_id)
        outcome = data.get("outcome")
        demo_summaries.append({
            "session_id": session_id,
            "outcome": outcome,
            "training_candidate": outcome == "success",
            "reason": None if outcome == "success" else f"outcome={outcome!r}",
        })
    return {
        "status": "PASS_ROLE_PREFLIGHT",
        "mapping": mapping_summary,
        "demonstrations": demo_summaries,
        "notes": [
            "Mapping role was explicit; no outcome/duration inference was used",
            "Demonstration ID 13 visibility is not evaluated at role preflight",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("job_root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(json.loads(args.job.read_text(encoding="utf-8")), args.job_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "REJECT_ROLE_PREFLIGHT", "error": str(exc)}
        code = 2
    else:
        code = 0
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
