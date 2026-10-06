"""Search a common SO-101 start pose for relative real-UMI demonstrations.

This is a geometry/label-generation study, not a physical-safety approval. Each
recording's first valid pinch pose is registered to the same robot start pose;
the demonstrated relative motion is then solved with the existing MuJoCo IK.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from paths import DEFAULT_CONFIG, DEFAULT_SCENE
from sim.mujoco.build_scene import build_model, load_config
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_mujoco import MujocoIK, eef_pose_from_joints
from umi.camera_frames import rigid
from umi.ik import matrix_to_quat, quat_to_matrix


REAL_J3_LOWER_RAD = -1.57079632679


def load_relative_run(bundle: Path, quality: dict, t_camera_pinch: np.ndarray):
    """Return the longest contiguous tracking+gap-valid pinch trajectory."""
    with zipfile.ZipFile(bundle) as archive:
        poses = list(csv.DictReader(io.StringIO(archive.read("poses.csv").decode("utf-8-sig"))))
        grips = list(csv.DictReader(io.StringIO(archive.read("gripper.csv").decode("utf-8-sig"))))
    if len(poses) != len(grips):
        raise ValueError(f"pose/gap row mismatch: {bundle.name}")
    allowed = np.zeros(len(poses), dtype=bool)
    segment_ids = np.full(len(poses), -1, dtype=int)
    for segment_id, (start, end) in enumerate(quality["usable_segments"]):
        allowed[start:end] = True
        segment_ids[start:end] = segment_id
    transforms, gaps, valid = [], [], []
    for index, (pose, grip) in enumerate(zip(poses, grips)):
        t_world_camera = np.eye(4)
        t_world_camera[:3, :3] = quat_to_matrix(
            np.array([float(pose[key]) for key in ("qw", "qx", "qy", "qz")]))
        t_world_camera[:3, 3] = [float(pose[key]) for key in ("x", "y", "z")]
        transforms.append(t_world_camera @ t_camera_pinch)
        gap = float(grip["gap_m"]) if grip["gap_m"].strip() else np.nan
        gaps.append(gap)
        valid.append(bool(allowed[index] and pose["tracking"] == "TRACKING"
                          and grip["status"] in ("D", "M") and np.isfinite(gap)))
    runs: list[list[int]] = []
    for index, ok in enumerate(valid):
        if not ok:
            continue
        if (not runs or index != runs[-1][-1] + 1
                or segment_ids[index] != segment_ids[runs[-1][-1]]):
            runs.append([])
        runs[-1].append(index)
    chosen = max(runs, key=len, default=[])
    if len(chosen) < 2:
        raise ValueError(f"no contiguous measured run: {bundle.name}")
    origin_inv = np.linalg.inv(transforms[chosen[0]])
    relative = np.stack([origin_inv @ transforms[index] for index in chosen])
    return relative, np.asarray([gaps[index] for index in chosen]), chosen


def register_relative(relative: np.ndarray, t_base_start: np.ndarray) -> np.ndarray:
    """Left-compose a relative pinch trajectory onto one robot start pose."""
    start = rigid(t_base_start, name="t_base_start")
    rel = np.asarray(relative, dtype=float)
    if rel.ndim != 3 or rel.shape[1:] != (4, 4):
        raise ValueError("relative trajectory must have shape (T,4,4)")
    for value in rel:
        rigid(value, name="relative pose")
    return np.einsum("ij,tjk->tik", start, rel)


def candidate_starts(ranges: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    """Deterministic interior joint candidates; zero is always evaluated first."""
    if count < 1:
        raise ValueError("candidate count must be positive")
    lo, hi = np.asarray(ranges)[:5, 0], np.asarray(ranges)[:5, 1]
    lo = lo.copy()
    lo[2] = max(lo[2], REAL_J3_LOWER_RAD)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    margin = 0.1 * (hi - lo)
    result = [np.clip(np.zeros(5), lo + margin, hi - margin)]
    rng = np.random.default_rng(seed)
    while len(result) < count:
        result.append(mid + rng.uniform(-0.65, 0.65, size=5) * half)
    return result


def sample_indices(length: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=int))


def solve_run(ik: MujocoIK, targets: np.ndarray, gaps: np.ndarray,
              indices: np.ndarray | None = None, q_start: np.ndarray | None = None) -> dict:
    selected = np.arange(len(targets)) if indices is None else np.asarray(indices, dtype=int)
    q_prev = None if q_start is None else np.asarray(q_start, dtype=float).copy()
    ok = 0
    position_mm, axis_deg = [], []
    real_limit_rejects = 0
    for index in selected:
        target = targets[index]
        solution = ik.solve(target[:3, 3], matrix_to_quat(target[:3, :3]),
                            float(gaps[index]), q_init=q_prev)
        real_limits = bool(solution.q_rad[2] >= REAL_J3_LOWER_RAD)
        accepted = solution.ok and real_limits
        real_limit_rejects += int(not real_limits)
        # Keep continuity even when a frame misses the acceptance gate. Resetting
        # to an unseeded multi-start after every miss is both slower and less
        # representative of an online trajectory solver.
        q_prev = solution.q_rad
        ok += int(accepted)
        position_mm.append(solution.pos_error_m * 1000.0)
        axis_deg.append(solution.axis_error_deg)
    return {
        "frames": int(len(selected)), "accepted": ok,
        "acceptance": ok / len(selected), "real_j3_limit_rejects": real_limit_rejects,
        "position_error_p95_mm": float(np.percentile(position_mm, 95)),
        "axis_error_p95_deg": float(np.percentile(axis_deg, 95)),
    }


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit("Shared GPU server is training-only; run this validation locally")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--full-stride", type=int, default=1,
                        help="evaluate every Nth valid frame after candidate selection")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    if args.full_stride < 1:
        raise SystemExit("--full-stride must be >= 1")
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")

    from umi.camera_frames import load_arcore_pinch_calibration
    camera_config, t_camera_pinch = load_arcore_pinch_calibration(args.extrinsic)
    runs = []
    for bundle in sorted(args.bundles.glob("rec_*.zip")):
        quality = json.loads(bundle.with_suffix(".quality.json").read_text(encoding="utf-8"))
        relative, gaps, source_rows = load_relative_run(bundle, quality, t_camera_pinch)
        runs.append((bundle.stem, relative, gaps, source_rows))

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg)
    fk_data = mujoco.MjData(model)
    candidates = candidate_starts(ik.ranges, args.candidates, args.seed)
    screened = []
    for candidate_id, arm in enumerate(candidates):
        q = np.r_[arm, 0.60]
        pos, quat = eef_pose_from_joints(model, fk_data, q, ik.pinch)
        t_start = np.eye(4)
        t_start[:3, :3], t_start[:3, 3] = quat_to_matrix(quat), pos
        episode_scores = []
        for _, relative, gaps, _ in runs:
            targets = register_relative(relative, t_start)
            episode_scores.append(solve_run(
                ik, targets, gaps, sample_indices(len(targets), args.sample_frames), q_start=q))
        frames = sum(row["frames"] for row in episode_scores)
        accepted = sum(row["accepted"] for row in episode_scores)
        screened.append({"candidate_id": candidate_id, "start_arm_rad": arm.tolist(),
                         "start_pose": t_start.tolist(), "sample_frames": frames,
                         "sample_accepted": accepted,
                         "sample_acceptance": accepted / frames})
    best = max(screened, key=lambda row: (row["sample_acceptance"], -row["candidate_id"]))
    t_best = np.asarray(best["start_pose"])
    episode_reports = []
    for episode_id, relative, gaps, source_rows in runs:
        q_start = np.r_[np.asarray(best["start_arm_rad"]), 0.60]
        full_indices = np.arange(0, len(relative), args.full_stride, dtype=int)
        if full_indices[-1] != len(relative) - 1:
            full_indices = np.r_[full_indices, len(relative) - 1]
        report = solve_run(ik, register_relative(relative, t_best), gaps,
                           indices=full_indices, q_start=q_start)
        report["source_valid_frames"] = len(relative)
        report.update({"episode_id": episode_id, "source_rows": source_rows})
        episode_reports.append(report)
    frames = sum(row["frames"] for row in episode_reports)
    accepted = sum(row["accepted"] for row in episode_reports)
    result = {
        "purpose": "pilot_geometry_and_label_generation_only",
        "physical_safety_verified": False,
        "camera_extrinsic_status": camera_config.get("status"),
        "registration": "each longest valid run first pinch -> common SO101 start pose",
        "selection": {"seed": args.seed, "candidates": args.candidates,
                      "sample_frames_per_episode": args.sample_frames},
        "screened_candidates": screened, "selected": best,
        "full_result": {"episodes": len(episode_reports), "evaluated_frames": frames,
                        "source_valid_frames": sum(len(row[1]) for row in runs),
                        "stride": args.full_stride,
                        "accepted": accepted, "acceptance": accepted / frames},
        "per_episode": episode_reports,
        "limitations": [
            "No robot/table collision or dynamic-limit validation.",
            "Camera-to-pinch extrinsic is provisional, not hand-eye measured.",
            "Candidate selection uses this pilot set and is not held-out performance.",
            "S22 demonstration and IMX708 deployment image domains are not matched.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_candidate": best["candidate_id"],
                      **result["full_result"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
