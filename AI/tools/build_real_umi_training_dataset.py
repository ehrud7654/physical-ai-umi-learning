"""Build a provisional contract dataset from the 2026-09-11 UMI delivery.

The provisional hand-eye transform is used to retain the demonstrated pinch
position and parallel-jaw direction.  SO-101 has only five arm DoF: a continuous
position IK supplies J1..J4 and wrist roll is matched to the demonstrated jaws.
The unpreserved approach-axis residual is recorded explicitly.  This remains a
pilot dataset, not a deployment calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from contract.episode import (CAMERA_NAMES, CONTRACT_VERSION, Episode, EpisodeMeta,
                              validate, write_dataset_index, write_episode)
from paths import DEFAULT_CONFIG, DEFAULT_SCENE
from sim.mujoco.build_scene import build_model, load_config
from tools.diagnose_real_umi_ik import solve_position_only
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_mujoco import MujocoIK
from tools.umi_real_placement_study import (REAL_J3_LOWER_RAD, load_relative_run,
                                            register_relative)
from umi.camera_frames import load_arcore_pinch_calibration, rigid
from umi.convert import normalize_gap, normalize_joints
from umi.ik import (approach_axis_from_quat, matrix_to_quat,
                    roll_residual_deg)


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = None
    for i, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def resize_rgb(data: bytes, *, rotate_to_canonical_deg: int) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        if rotate_to_canonical_deg not in (0, 180):
            raise ValueError("rotate_to_canonical_deg must be explicitly 0 or 180")
        if rotate_to_canonical_deg == 180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        image = image.resize((224, 224), resample=Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def digest_inputs(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit("Build/IK is local-only; upload only the finished dataset")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--extrinsic", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-steps", type=int, default=30)
    parser.add_argument("--author", default="AI UMI")
    parser.add_argument("--real-config", type=Path,
                        default=Path("configs/real/so101_ver1.json"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    if args.min_steps < 2:
        raise SystemExit("--min-steps must be >= 2")

    bundles = sorted(args.bundles.glob("rec_*.zip"))
    if not bundles:
        raise SystemExit(f"no normalized bundles: {args.bundles}")
    extrinsic, t_camera_pinch = load_arcore_pinch_calibration(args.extrinsic)
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    t_start = rigid(placement["selected"]["start_pose"])
    start_arm = np.asarray(placement["selected"]["start_arm_rad"], dtype=float)

    cfg = load_config(args.config)
    real_cfg = json.loads(args.real_config.read_text(encoding="utf-8"))
    arm_speed_limit = float(real_cfg["max_speed_rad_s"])
    arm_accel_limit = float(real_cfg["max_accel_rad_s2"])
    gap_speed_limit = float(real_cfg["max_gap_speed_m_s"])
    gap_accel_limit = float(real_cfg["max_gap_accel_m_s2"])
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg)
    ranges = np.asarray(ik.ranges, dtype=float)
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    args.out.mkdir(parents=True)
    reports = []
    failures = {}
    for bundle in bundles:
        quality_path = bundle.with_suffix(".quality.json")
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        try:
            relative, gaps, source_rows = load_relative_run(bundle, quality, t_camera_pinch)
            targets = register_relative(relative, t_start)
            q_prev = np.r_[start_arm, 0.60]
            joints, errors, axis_errors, roll_errors, solution_ok = [], [], [], [], []
            for target, gap_m in zip(targets, gaps):
                quat = matrix_to_quat(target[:3, :3])
                q_prev, error, hit_limit = solve_position_only(
                    model, target[:3, 3], ik.pinch, q_prev)
                q_prev[4] = ik.match_wrist_roll(q_prev, quat)
                achieved_axis, achieved_jaw = ik._achieved(q_prev)
                desired_axis = approach_axis_from_quat(quat)
                axis_error = float(np.degrees(np.arccos(np.clip(
                    np.dot(desired_axis, achieved_axis), -1.0, 1.0))))
                joints.append(q_prev.copy())
                errors.append(error)
                axis_errors.append(axis_error)
                roll_errors.append(roll_residual_deg(quat, achieved_axis, achieved_jaw))
                solution_ok.append(
                    not hit_limit and error <= 5e-3 and q_prev[2] >= REAL_J3_LOWER_RAD)
            joints = np.asarray(joints)
            errors = np.asarray(errors)
            axis_errors = np.asarray(axis_errors)
            roll_errors = np.asarray(roll_errors)
            ok = (np.asarray(solution_ok) & np.isfinite(gaps)
                  & (gaps >= 0.0) & (gaps <= 0.09))
            start, end = longest_true_run(ok)
            if end - start - 1 < args.min_steps:
                raise ValueError(f"longest feasible training run={end-start-1}")

            with zipfile.ZipFile(bundle) as archive:
                poses = list(csv.DictReader(io.StringIO(
                    archive.read("poses.csv").decode("utf-8-sig"))))
                episode_json = json.loads(archive.read("episode.json"))
                try:
                    rotate_to_canonical_deg = int(
                        episode_json["camera"]["rotate_to_canonical_deg"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "camera.rotate_to_canonical_deg must be explicit before training"
                    ) from exc
                frame_rows = list(csv.DictReader(io.StringIO(
                    archive.read("frames.csv").decode("utf-8-sig"))))
                frame_by_index = {int(row["frame_index"]): row for row in frame_rows}
                available_frame_indices = np.asarray(sorted(frame_by_index), dtype=int)
                chosen_rows = source_rows[start:end]
                source_timestamps = np.asarray(
                    [float(poses[int(row)]["timestamp_ns"]) * 1e-9 for row in chosen_rows],
                    dtype=np.float64)
                # The recorder can omit an RGB frame while ARCore keeps advancing.
                # Contract episodes are a fixed 30 Hz control stream, so interpolate
                # joint/gap labels on a uniform grid and select the nearest real RGB.
                period = 1.0 / 30.0
                probe_timestamps = source_timestamps[0] + np.arange(
                    int(np.floor((source_timestamps[-1] - source_timestamps[0]) / period)) + 1
                ) * period
                q_source = joints[start:end]
                gap_source = gaps[start:end]
                q_probe = np.column_stack([
                    np.interp(probe_timestamps, source_timestamps, q_source[:, joint])
                    for joint in range(q_source.shape[1])])
                gap_probe = np.interp(probe_timestamps, source_timestamps, gap_source)
                dq = np.diff(q_probe[:, :5], axis=0) / period
                dg = np.diff(gap_probe) / period
                velocity_ratio = max(
                    float(np.max(np.abs(dq))) / arm_speed_limit,
                    float(np.max(np.abs(dg))) / gap_speed_limit)
                joint_velocity_peak_rad_s = np.max(np.abs(dq), axis=0)
                arm_accel = np.diff(dq, axis=0) / period
                gap_accel = np.diff(dg) / period
                accel_ratio = max(
                    float(np.max(np.abs(arm_accel))) / arm_accel_limit
                    if arm_accel.size else 0.0,
                    float(np.max(np.abs(gap_accel))) / gap_accel_limit
                    if gap_accel.size else 0.0)
                # Uniform time dilation reduces velocity by s and acceleration by s^2.
                time_scale = max(1.0, velocity_ratio, np.sqrt(accel_ratio)) * 1.02
                output_duration = (source_timestamps[-1] - source_timestamps[0]) * time_scale
                timestamps = source_timestamps[0] + np.arange(
                    int(np.floor(output_duration / period)) + 1) * period
                label_timestamps = source_timestamps[0] + (
                    timestamps - source_timestamps[0]) / time_scale
                if len(timestamps) - 1 < args.min_steps:
                    raise ValueError(f"uniform training run={len(timestamps)-1}")
                q = np.column_stack([
                    np.interp(label_timestamps, source_timestamps, q_source[:, joint])
                    for joint in range(q_source.shape[1])])
                selected_gaps = np.interp(label_timestamps, source_timestamps, gap_source)
                arm = np.stack([normalize_joints(row, ranges)[:5] for row in q])
                values = np.column_stack((arm, normalize_gap(selected_gaps))).astype(np.float32)
                images = []
                decoded_images: dict[int, np.ndarray] = {}
                for label_timestamp in label_timestamps[:-1]:
                    nearest = int(np.argmin(np.abs(source_timestamps - label_timestamp)))
                    row_index = int(chosen_rows[nearest])
                    if row_index not in frame_by_index:
                        row_index = int(available_frame_indices[
                            np.argmin(np.abs(available_frame_indices - row_index))])
                    frame = frame_by_index[row_index]
                    if row_index not in decoded_images:
                        decoded_images[row_index] = resize_rgb(
                            archive.read(frame["image"]),
                            rotate_to_canonical_deg=rotate_to_canonical_deg)
                    images.append(decoded_images[row_index])
            images_chw = np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2))
            state_timestamps = np.asarray(timestamps[:-1], dtype=np.float64)
            rate = 30.0
            n = len(state_timestamps)
            ep = Episode(
                meta=EpisodeMeta(
                    episode_id=bundle.stem, skill_id=episode_json.get("skill_id", "pick_place"),
                    task="real UMI pick/place demonstration", source="real", success=False,
                    n_steps=n, control_rate_hz=rate, cameras=list(CAMERA_NAMES),
                    contract_version=CONTRACT_VERSION, collected_by=args.author,
                    git_rev=git_rev,
                    notes={
                        "success_label": "unverified",
                        "label_projection": "pinch_position_continuous; wrist_roll_matched",
                        "orientation_fidelity": "jaw_preserved; approach_axis_not_constrained_and_residual_recorded",
                        "time_scale": float(time_scale),
                        "source_velocity_ratio": float(velocity_ratio),
                        "joint_velocity_peak_rad_s": joint_velocity_peak_rad_s.tolist(),
                        "source_acceleration_ratio": float(accel_ratio),
                        "dynamic_limits_status": "provisional",
                        "camera_extrinsic_status": extrinsic.get("status", "unknown"),
                        "image_rotation_to_canonical_deg": rotate_to_canonical_deg,
                        "source_bundle": bundle.name,
                        "source_rows_half_open": [int(chosen_rows[0]), int(chosen_rows[-1]) + 1],
                        "ik_position_error_p95_mm": float(np.percentile(errors[start:end], 95) * 1000),
                        "ik_axis_error_p95_deg": float(np.percentile(axis_errors[start:end], 95)),
                        "ik_roll_residual_p95_deg": float(np.percentile(roll_errors[start:end], 95)),
                    }),
                images={"cam_wrist": images_chw}, state=values[:-1],
                state_timestamp=state_timestamps, action=values[1:],
                action_timestamp=state_timestamps.copy())
            problems = validate(ep)
            if problems:
                raise ValueError("; ".join(problems))
            write_episode(ep, args.out)
            reports.append({"episode_id": bundle.stem, "source_frames": len(targets),
                            "training_steps": n, "kept_span": [start, end],
                            "time_scale": float(time_scale),
                            "source_velocity_ratio": float(velocity_ratio),
                            "joint_velocity_peak_rad_s": joint_velocity_peak_rad_s.tolist(),
                            "source_acceleration_ratio": float(accel_ratio),
                            "position_error_p95_mm": ep.meta.notes["ik_position_error_p95_mm"],
                            "axis_error_p95_deg": ep.meta.notes["ik_axis_error_p95_deg"],
                            "roll_residual_p95_deg": ep.meta.notes["ik_roll_residual_p95_deg"]})
            print(f"{bundle.stem}: {n} steps")
        except Exception as exc:
            failures[bundle.stem] = str(exc).splitlines()[0]
            print(f"{bundle.stem}: rejected - {failures[bundle.stem]}")

    if not reports:
        raise SystemExit("no contract episodes were produced")
    index = write_dataset_index(args.out, extra={
        "status": "PROVISIONAL — position+jaw labels; not deployment calibrated",
        "source": "real UMI arpose delivery normalized to umi_raw/0.1.0",
        "label_projection": "pinch_position_continuous; wrist_roll_matched",
        "orientation_fidelity": "jaw_preserved; approach_axis_not_constrained_and_residual_recorded",
        "camera_extrinsic_status": extrinsic.get("status", "unknown"),
        "success_labels": "unverified",
        "dynamic_limits_status": "provisional",
        "dynamic_limits": {
            "arm_velocity_rad_s": arm_speed_limit,
            "arm_acceleration_rad_s2": arm_accel_limit,
            "gap_velocity_m_s": gap_speed_limit,
            "gap_acceleration_m_s2": gap_accel_limit,
        },
        "source_sha256": digest_inputs(bundles + [p.with_suffix(".quality.json") for p in bundles]),
        "episodes_rejected": failures,
        "build_report": reports,
    })
    print(json.dumps({"dataset": str(index), "episodes": len(reports),
                      "rejected": len(failures),
                      "steps": sum(row["training_steps"] for row in reports)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
