"""Audit relative UMI pinch motion without inventing a robot-base alignment."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from umi.camera_frames import rigid
from umi.ik import quat_to_matrix
from tools.run_umi_regression import is_shared_gpu_server


def percentile(values, q):
    return None if not values else float(np.percentile(values, q))


def audit_episode(bundle: Path, quality: dict, t_camera_pinch: np.ndarray) -> dict:
    with zipfile.ZipFile(bundle) as archive:
        poses = list(csv.DictReader(io.StringIO(archive.read("poses.csv").decode("utf-8-sig"))))
        gaps = list(csv.DictReader(io.StringIO(archive.read("gripper.csv").decode("utf-8-sig"))))
    if len(poses) != len(gaps):
        raise ValueError(f"pose/gap row mismatch: {bundle.name}")
    allowed = np.zeros(len(poses), dtype=bool)
    segment_id = np.full(len(poses), -1, dtype=int)
    for index, (start, end) in enumerate(quality["usable_segments"]):
        allowed[start:end] = True
        segment_id[start:end] = index
    transforms, stamps, valid = [], [], []
    for index, (pose, gap) in enumerate(zip(poses, gaps)):
        q_wxyz = np.array([float(pose[k]) for k in ("qw", "qx", "qy", "qz")])
        world_camera = np.eye(4)
        world_camera[:3, :3] = quat_to_matrix(q_wxyz)
        world_camera[:3, 3] = [float(pose[k]) for k in ("x", "y", "z")]
        transforms.append(world_camera @ t_camera_pinch)
        stamps.append(int(pose["timestamp_ns"]) / 1e9)
        valid.append(bool(allowed[index] and pose["tracking"] == "TRACKING"
                          and gap["status"] in ("D", "M") and gap["gap_m"].strip()))
    step_m, speed_m_s, angle_deg, angle_deg_s = [], [], [], []
    for index in range(1, len(poses)):
        if (not valid[index - 1] or not valid[index]
                or segment_id[index - 1] != segment_id[index]):
            continue
        dt = stamps[index] - stamps[index - 1]
        if dt <= 0:
            raise ValueError(f"non-increasing timestamp: {bundle.name}")
        relative = np.linalg.inv(transforms[index - 1]) @ transforms[index]
        distance = float(np.linalg.norm(relative[:3, 3]))
        cosine = np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)
        angle = float(np.rad2deg(np.arccos(cosine)))
        step_m.append(distance); speed_m_s.append(distance / dt)
        angle_deg.append(angle); angle_deg_s.append(angle / dt)
    return {
        "episode_id": bundle.stem, "frames": len(poses),
        "valid_gap_frames": int(sum(valid)), "relative_transitions": len(step_m),
        "translation_step_m": {"p50": percentile(step_m, 50),
                               "p95": percentile(step_m, 95), "max": percentile(step_m, 100)},
        "translation_speed_m_s": {"p50": percentile(speed_m_s, 50),
                                  "p95": percentile(speed_m_s, 95), "max": percentile(speed_m_s, 100)},
        "rotation_step_deg": {"p50": percentile(angle_deg, 50),
                              "p95": percentile(angle_deg, 95), "max": percentile(angle_deg, 100)},
        "rotation_speed_deg_s": {"p50": percentile(angle_deg_s, 50),
                                 "p95": percentile(angle_deg_s, 95), "max": percentile(angle_deg_s, 100)},
        "_values": {"step_m": step_m, "speed_m_s": speed_m_s,
                    "angle_deg": angle_deg, "angle_deg_s": angle_deg_s},
    }


def main() -> None:
    if is_shared_gpu_server():
        raise SystemExit("Shared GPU server is training-only")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.extrinsic.read_text(encoding="utf-8"))
    transform = rigid(config["t_arcore_camera_to_umi_pinch"], name="t_camera_pinch")
    episodes, aggregate = [], {key: [] for key in
        ("step_m", "speed_m_s", "angle_deg", "angle_deg_s")}
    for bundle in sorted(args.bundles.glob("rec_*.zip")):
        quality = json.loads(bundle.with_suffix(".quality.json").read_text(encoding="utf-8"))
        report = audit_episode(bundle, quality, transform)
        for key, values in report.pop("_values").items():
            aggregate[key].extend(values)
        episodes.append(report)
    result = {
        "extrinsic_calibration_id": config["calibration_id"],
        "extrinsic_status": config["status"],
        "robot_base_alignment_applied": False,
        "episodes": len(episodes), "relative_transitions": len(aggregate["step_m"]),
        "aggregate": {
            "translation_step_m": {"p50": percentile(aggregate["step_m"], 50),
                                   "p95": percentile(aggregate["step_m"], 95),
                                   "max": percentile(aggregate["step_m"], 100)},
            "translation_speed_m_s": {"p50": percentile(aggregate["speed_m_s"], 50),
                                      "p95": percentile(aggregate["speed_m_s"], 95),
                                      "max": percentile(aggregate["speed_m_s"], 100)},
            "rotation_step_deg": {"p50": percentile(aggregate["angle_deg"], 50),
                                  "p95": percentile(aggregate["angle_deg"], 95),
                                  "max": percentile(aggregate["angle_deg"], 100)},
            "rotation_speed_deg_s": {"p50": percentile(aggregate["angle_deg_s"], 50),
                                     "p95": percentile(aggregate["angle_deg_s"], 95),
                                     "max": percentile(aggregate["angle_deg_s"], 100)},
        },
        "per_episode": episodes,
    }
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("episodes", "relative_transitions", "aggregate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
