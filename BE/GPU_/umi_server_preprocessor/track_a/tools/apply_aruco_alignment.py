#!/usr/bin/env python3
"""Apply a fixed-marker alignment to an ORB-SLAM3 camera trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_alignment(trajectory: Path, alignment_path: Path, output: Path,
                    *, allow_provisional: bool = False) -> dict:
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    status = alignment.get("status")
    if status != "PASS_ARUCO_ALIGNMENT":
        if not (allow_provisional and status == "PROVISIONAL_ARUCO_ALIGNMENT"):
            raise ValueError(
                f"alignment status {status!r} is not publishable; "
                "use --allow-provisional for diagnostic output only"
            )
    transform_marker_world = np.asarray(alignment["T_marker_world"], dtype=np.float64)
    if transform_marker_world.shape != (4, 4) or not np.isfinite(transform_marker_world).all():
        raise ValueError("T_marker_world must be a finite 4x4 matrix")
    if not np.allclose(transform_marker_world[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("T_marker_world has an invalid homogeneous last row")

    with trajectory.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError("trajectory has no CSV header")

    transformed = 0
    for row in rows:
        valid = row.get("state") == "2" and row.get("is_lost", "true").lower() == "false"
        if not valid:
            continue
        transform_world_camera = np.eye(4)
        transform_world_camera[:3, :3] = Rotation.from_quat([
            float(row["q_x"]), float(row["q_y"]),
            float(row["q_z"]), float(row["q_w"]),
        ]).as_matrix()
        transform_world_camera[:3, 3] = [
            float(row["x"]), float(row["y"]), float(row["z"]),
        ]
        transform_marker_camera = transform_marker_world @ transform_world_camera
        quaternion = Rotation.from_matrix(transform_marker_camera[:3, :3]).as_quat()
        for key, value in zip(("x", "y", "z"), transform_marker_camera[:3, 3]):
            row[key] = f"{value:.12g}"
        for key, value in zip(("q_x", "q_y", "q_z", "q_w"), quaternion):
            row[key] = f"{value:.12g}"
        transformed += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "status": "PROVISIONAL_OUTPUT" if status == "PROVISIONAL_ARUCO_ALIGNMENT" else "PASS",
        "source_trajectory": str(trajectory),
        "source_trajectory_sha256": _sha256(trajectory),
        "alignment": str(alignment_path),
        "alignment_status": status,
        "output_trajectory": str(output),
        "output_trajectory_sha256": _sha256(output),
        "pose_semantics": "T_marker_camera",
        "rows": len(rows),
        "transformed_valid_rows": transformed,
        "training_input_ready": status == "PASS_ARUCO_ALIGNMENT",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()
    report = apply_alignment(
        args.trajectory, args.alignment, args.out,
        allow_provisional=args.allow_provisional,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
