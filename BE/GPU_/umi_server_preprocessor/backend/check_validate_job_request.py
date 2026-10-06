#!/usr/bin/env python3
"""Focused self-test for validate_job_request.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from validate_job_request import validate


REQUIRED = (
    "video.mp4", "frames.csv", "encoded.csv", "accelerometer.csv", "gyroscope.csv",
)


def make_session(root: Path, name: str, outcome: str) -> None:
    session = root / name
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps({"outcome": outcome}), encoding="utf-8")
    for filename in REQUIRED:
        (session / filename).write_bytes(b"test")


def reject(payload: dict, root: Path, message: str) -> None:
    try:
        validate(payload, root)
    except ValueError:
        return
    raise AssertionError(message)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mapping_root = root / "raw_mapping"
        demo_root = root / "raw"
        mapping_root.mkdir()
        demo_root.mkdir()
        make_session(mapping_root, "mapping", "unrated")
        make_session(demo_root, "demo_ok", "success")
        make_session(demo_root, "demo_unrated", "unrated")

        payload = {
            "job_id": "test-job",
            "mapping_session_id": "mapping",
            "demonstration_session_ids": ["demo_ok", "demo_unrated"],
        }
        result = validate(payload, root)
        assert result["status"] == "PASS_ROLE_PREFLIGHT"
        assert result["mapping"]["outcome"] == "unrated"
        assert result["mapping"]["outcome_used_as_training_label"] is False
        assert result["demonstrations"][0]["training_candidate"] is True
        assert result["demonstrations"][1]["training_candidate"] is False

        reject(
            {"job_id": "test-job", "demonstration_session_ids": ["demo_ok"]},
            root, "missing Mapping/Atlas must reject")
        reject(
            {
                "job_id": "test-job",
                "mapping_session_id": "mapping",
                "demonstration_session_ids": ["mapping"],
            },
            root, "role overlap must reject")
    print("VALIDATE_JOB_REQUEST_SELF_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
