"""Export Track-A pose/gap reference CSVs and a reproducibility manifest.

The reviewed Atlas selection is the ordering and provenance boundary.  This
tool does not rerun ORB-SLAM3 and does not reinterpret rejected episodes.  It
exports, for every selected episode:

* ``pose.csv``: ArUco-marker-frame camera and TCP poses for every source frame;
* ``gripper.csv``: canonical gap in metres, preserving direct/interpolated state;
* ``manifest.json``: inputs, ordering, hashes, calibration and environment;
* ``verification_reference.json``: precomputed structural/numeric checks.

The bundle is a Track-A reference artifact, not an official UMI Zarr and not
physical-deployment approval.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from umi.ik import matrix_to_quat, quat_to_matrix

try:
    import scipy
    from scipy.spatial.transform import Rotation
except ModuleNotFoundError:  # NumPy fallback keeps reference export portable.
    scipy = None
    Rotation = None

try:
    import cv2
except ModuleNotFoundError:  # OpenCV is not required to export stored rows.
    cv2 = None


BUNDLE_SCHEMA = "track_a_orbslam_v10_reference/0.3"


def rotation_from_quat_xyzw(value: list[float]) -> np.ndarray:
    if Rotation is not None:
        return Rotation.from_quat(value).as_matrix()
    x, y, z, w = value
    return quat_to_matrix(np.asarray([w, x, y, z], dtype=np.float64))


def proper_rotation(value: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        return Rotation.from_matrix(value).as_matrix()
    u, _, vh = np.linalg.svd(np.asarray(value, dtype=np.float64))
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def quat_xyzw_from_matrix(value: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        return Rotation.from_matrix(value).as_quat()
    w, x, y, z = matrix_to_quat(value)
    return np.asarray([x, y, z, w], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def rigid(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999:
        raise ValueError(f"{name} rotation is reflected")
    return matrix


def canonical_array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    dtype = value.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
        value = value.astype(dtype.newbyteorder("<"), copy=False)
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def array_reference(array: np.ndarray, *, rgb: bool = False) -> dict[str, Any]:
    value = np.asarray(array)
    item: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "elements": int(value.size),
        "canonical_c_little_endian_sha256": canonical_array_sha256(value),
        "finite": bool(np.isfinite(value).all()) if value.dtype.kind in "fc" else True,
    }
    if rgb:
        item["comparison_policy"] = (
            "diagnostic_only; shape/dtype/elements required, content hash not a parity gate"
        )
    else:
        item["comparison_policy"] = "bit_identical_after_c_contiguous_little_endian_normalisation"
    return item


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, encoding="utf-8",
        errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout.strip()


def environment_reference() -> dict[str, Any]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__ if scipy is not None else "not-installed",
        "opencv": cv2.__version__ if cv2 is not None else "not-installed",
        "os": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "endianness": sys.byteorder,
        "numpy_blas_config": output.getvalue(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
    }


def pose_components(transform: np.ndarray) -> list[float]:
    quaternion = quat_xyzw_from_matrix(transform[:3, :3])
    return [
        float(transform[0, 3]), float(transform[1, 3]), float(transform[2, 3]),
        float(quaternion[0]), float(quaternion[1]),
        float(quaternion[2]), float(quaternion[3]),
    ]


def pose_6d(transform: np.ndarray, *, frame_index: int,
            timestamp_s: float, definition: str) -> dict[str, Any]:
    rotation = proper_rotation(transform[:3, :3])
    quaternion = quat_xyzw_from_matrix(rotation)
    if Rotation is not None:
        rotvec = Rotation.from_matrix(rotation).as_rotvec()
    else:
        vector = quaternion[:3]
        length = float(np.linalg.norm(vector))
        if length < 1e-12:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            # q and -q are equivalent.  Keep the principal rotation angle.
            q = quaternion if quaternion[3] >= 0 else -quaternion
            vector, length = q[:3], float(np.linalg.norm(q[:3]))
            angle = 2.0 * np.arctan2(length, float(q[3]))
            rotvec = vector / length * angle
    return {
        "frame": "aruco_id13_table",
        "definition": definition,
        "source_frame_index": int(frame_index),
        "timestamp_s": float(timestamp_s),
        "xyz_m": [float(value) for value in transform[:3, 3]],
        "rotation_vector_rad": [float(value) for value in rotvec],
        "quaternion_xyzw": [float(value) for value in quaternion],
    }


def _private_source_path(path: Path, *, fallback: str,
                         redact_source_paths: bool) -> str:
    """Keep hashes reproducible without leaking a user's absolute home path."""
    return fallback if redact_source_paths else str(path)


def write_episode_reference(
    selection: dict[str, Any], output: Path, camera_tcp: np.ndarray,
    *, redact_source_paths: bool = False,
) -> dict[str, Any]:
    episode_id = str(selection["episode_id"])
    raw = Path(selection["raw_episode"])
    trajectory_path = Path(selection["trajectory"])
    gripper_path = Path(selection["gripper_csv"])
    alignment_path = Path(selection["alignment"])
    validation_path = Path(selection["trajectory_validation"])

    frame_rows = rows(raw / "frames.csv")
    pose_rows = rows(trajectory_path)
    gripper_rows = rows(gripper_path)
    if not frame_rows or not (len(frame_rows) == len(pose_rows) == len(gripper_rows)):
        raise ValueError(f"{episode_id}: frame/pose/gripper row count mismatch")

    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment.get("status") != "PASS_ARUCO_ALIGNMENT":
        raise ValueError(f"{episode_id}: alignment did not pass")
    marker_world = rigid(alignment["T_marker_world"], "T_marker_world")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "pass":
        raise ValueError(f"{episode_id}: trajectory validation did not pass")

    episode_dir = output / "episodes" / episode_id
    episode_dir.mkdir(parents=True)
    pose_path = episode_dir / "pose.csv"
    gripper_out = episode_dir / "gripper.csv"

    pose_header = [
        "frame_index", "sensor_timestamp_ns", "timestamp_s", "tracking_state",
        "is_lost", "is_keyframe",
        "camera_x_m", "camera_y_m", "camera_z_m",
        "camera_qx", "camera_qy", "camera_qz", "camera_qw",
        "tcp_x_m", "tcp_y_m", "tcp_z_m",
        "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
    ]
    gripper_header = [
        "frame_index", "sensor_timestamp_ns", "timestamp_s", "status",
        "marker_detected", "gap_m", "marker_center_distance_m",
        "marker_center_interpolated_m",
    ]

    first_ns = int(frame_rows[0]["sensor_timestamp_ns"])
    pose_output: list[list[Any]] = []
    gripper_output: list[list[Any]] = []
    direct = interpolated = invalid = tracked = 0
    start_tcp_pose: dict[str, Any] | None = None
    timestamps: list[int] = []
    for expected, (frame, pose, grip) in enumerate(
        zip(frame_rows, pose_rows, gripper_rows)
    ):
        frame_index = int(frame["frame_number"])
        if frame_index != expected or int(pose["frame_idx"]) != expected:
            raise ValueError(f"{episode_id}: non-contiguous frame/pose index at {expected}")
        if int(grip["frame_index"]) != expected:
            raise ValueError(f"{episode_id}: non-contiguous gripper index at {expected}")
        sensor_ns = int(frame["sensor_timestamp_ns"])
        timestamps.append(sensor_ns)
        timestamp_s = (sensor_ns - first_ns) / 1e9

        is_lost = pose.get("is_lost", "true").lower() == "true"
        tracking_state = int(pose.get("state", "0"))
        pose_valid = tracking_state == 2 and not is_lost
        if pose_valid:
            world_camera = np.eye(4)
            world_camera[:3, :3] = rotation_from_quat_xyzw([
                float(pose[key]) for key in ("q_x", "q_y", "q_z", "q_w")
            ])
            world_camera[:3, 3] = [float(pose[key]) for key in ("x", "y", "z")]
            marker_camera = marker_world @ world_camera
            marker_camera[:3, :3] = proper_rotation(marker_camera[:3, :3])
            marker_tcp = marker_camera @ camera_tcp
            marker_tcp[:3, :3] = proper_rotation(marker_tcp[:3, :3])
            camera_values: list[Any] = pose_components(marker_camera)
            tcp_values: list[Any] = pose_components(marker_tcp)
            tracked += 1
            if start_tcp_pose is None:
                start_tcp_pose = pose_6d(
                    marker_tcp, frame_index=frame_index, timestamp_s=timestamp_s,
                    definition="first valid tracked TCP row in the full source episode",
                )
        else:
            camera_values = [""] * 7
            tcp_values = [""] * 7
        pose_output.append([
            frame_index, sensor_ns, f"{timestamp_s:.9f}", tracking_state,
            str(is_lost).lower(), pose.get("is_keyframe", "false").lower(),
            *camera_values, *tcp_values,
        ])

        detected = grip.get("marker_detected", "false").lower() == "true"
        width_key = "gripper_width_mm" if "gripper_width_mm" in grip else "width_mm"
        try:
            gap_m = float(grip[width_key]) / 1000.0
        except (KeyError, TypeError, ValueError):
            gap_m = float("nan")
        valid_gap = bool(np.isfinite(gap_m) and 0 <= gap_m <= 0.09)
        status = "D" if detected and valid_gap else "M" if valid_gap else "X"
        direct += int(status == "D")
        interpolated += int(status == "M")
        invalid += int(status == "X")

        def optional_mm(name: str) -> str:
            try:
                value = float(grip[name]) / 1000.0
            except (KeyError, TypeError, ValueError):
                return ""
            return f"{value:.12g}" if np.isfinite(value) else ""

        gripper_output.append([
            frame_index, sensor_ns, f"{timestamp_s:.9f}", status,
            str(detected).lower(), f"{gap_m:.12g}" if valid_gap else "",
            optional_mm("marker_center_distance_mm"),
            optional_mm("marker_center_interpolated_mm"),
        ])

    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{episode_id}: non-monotonic sensor timestamps")
    if start_tcp_pose is None:
        raise ValueError(f"{episode_id}: no valid tracked TCP row")

    with pose_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(pose_header)
        writer.writerows(pose_output)
    with gripper_out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(gripper_header)
        writer.writerows(gripper_output)

    return {
        "episode_id": episode_id,
        "source": {
            "raw_episode": _private_source_path(
                raw, fallback=episode_id, redact_source_paths=redact_source_paths),
            "video": _private_source_path(
                raw / "video.mp4", fallback=f"{episode_id}/video.mp4",
                redact_source_paths=redact_source_paths),
            "video_sha256": sha256_file(raw / "video.mp4"),
            "frames_csv_sha256": sha256_file(raw / "frames.csv"),
            "source_trajectory": _private_source_path(
                trajectory_path,
                fallback=(f"{selection.get('atlas_candidate', 'atlas')}/{episode_id}/"
                          f"attempt{selection.get('attempt', 'unknown')}/camera_trajectory.csv"),
                redact_source_paths=redact_source_paths),
            "source_trajectory_sha256": sha256_file(trajectory_path),
            "source_gripper": _private_source_path(
                gripper_path, fallback=f"{episode_id}/gripper_width.csv",
                redact_source_paths=redact_source_paths),
            "source_gripper_sha256": sha256_file(gripper_path),
            "trajectory_validation": _private_source_path(
                validation_path, fallback=f"{episode_id}/trajectory_validation_v10.json",
                redact_source_paths=redact_source_paths),
            "trajectory_validation_sha256": sha256_file(validation_path),
            "alignment": _private_source_path(
                alignment_path, fallback=alignment_path.name,
                redact_source_paths=redact_source_paths),
            "alignment_sha256": sha256_file(alignment_path),
            "atlas_candidate": selection.get("atlas_candidate"),
            "atlas_sha256": selection.get("atlas_sha256"),
            "attempt": selection.get("attempt"),
        },
        "pose_csv": {
            "path": pose_path.relative_to(output).as_posix(),
            "sha256": sha256_file(pose_path),
            "rows": len(pose_output),
            "tracked_rows": tracked,
            "semantics": {
                "camera": "T_marker_camera; ArUco ID 13 table frame",
                "tcp": "T_marker_tcp = T_marker_camera @ T_camera_tcp",
                "quaternion": "qx,qy,qz,qw",
                "translation_unit": "m",
            },
            "start_tcp_pose_6d": start_tcp_pose,
        },
        "gripper_csv": {
            "path": gripper_out.relative_to(output).as_posix(),
            "sha256": sha256_file(gripper_out),
            "rows": len(gripper_output),
            "status_counts": {"D": direct, "M": interpolated, "X": invalid},
            "gap_unit": "m",
            "gap_definition": "absolute inner contact-pad spacing",
        },
    }


def v10_episode_reference(root: Path, episode_id: str,
                          *, redact_source_paths: bool = False) -> dict[str, Any]:
    npz = root / f"{episode_id}.npz"
    metadata = root / f"{episode_id}.json"
    if not npz.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"missing v10 episode artifact for {episode_id}")
    metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
    for key in ("start_tcp_pose_6d", "first_training_anchor_tcp_pose_6d"):
        if not isinstance(metadata_payload.get(key), dict):
            raise ValueError(f"{episode_id}: v10 metadata is missing {key}")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {
            key: array_reference(archive[key], rgb=(key == "image"))
            for key in sorted(archive.files)
        }
    return {
        "npz_path": npz.name if redact_source_paths else str(npz),
        "npz_sha256": sha256_file(npz),
        "metadata_path": metadata.name if redact_source_paths else str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "start_tcp_pose_6d": metadata_payload.get("start_tcp_pose_6d"),
        "first_training_anchor_tcp_pose_6d": metadata_payload.get(
            "first_training_anchor_tcp_pose_6d"),
        "arrays": arrays,
    }


def add_file_hashes(root: Path, manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "bundle_checksums.sha256":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "bundle_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    manifest["bundle_checksums"] = "bundle_checksums.sha256"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--camera-tcp", type=Path, required=True)
    parser.add_argument("--v10", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--episode-gate-report-json", type=Path, required=True)
    parser.add_argument("--episode-gate-report-csv", type=Path, required=True)
    parser.add_argument(
        "--redact-source-paths", action="store_true",
        help="store logical source names instead of absolute local paths in manifest")
    args = parser.parse_args()

    args.selection = args.selection.resolve()
    args.camera_tcp = args.camera_tcp.resolve()
    args.v10 = args.v10.resolve()
    args.out = args.out.resolve()
    args.tar = args.tar.resolve() if args.tar else None
    args.episode_gate_report_json = (
        args.episode_gate_report_json.resolve() if args.episode_gate_report_json else None)
    args.episode_gate_report_csv = (
        args.episode_gate_report_csv.resolve() if args.episode_gate_report_csv else None)
    for report in (args.episode_gate_report_json, args.episode_gate_report_csv):
        if not report.is_file():
            raise FileNotFoundError(report)

    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    gate_report_files: dict[str, dict[str, str]] = {}
    quality_root = args.out / "quality"
    quality_root.mkdir()
    for label, source in (
        ("json", args.episode_gate_report_json),
        ("csv", args.episode_gate_report_csv),
    ):
        target = quality_root / f"episode_gate_report.{label}"
        shutil.copyfile(source, target)
        gate_report_files[label] = {
            "path": target.relative_to(args.out).as_posix(),
            "sha256": sha256_file(target),
        }

    repo = Path(__file__).resolve().parents[2]
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    camera_config = json.loads(args.camera_tcp.read_text(encoding="utf-8"))
    camera_tcp = rigid(camera_config["T_camera_tcp"], "T_camera_tcp")
    v10_index_path = args.v10 / "dataset.json"
    v10_index = json.loads(v10_index_path.read_text(encoding="utf-8"))

    selected = list(selection.get("episodes", []))
    order = [str(item["episode_id"]) for item in selected]
    if order != list(v10_index.get("episodes", [])):
        raise ValueError("selection order and v10 dataset episode order differ")

    reference_episodes = []
    total_pose = total_gripper = total_d = total_m = total_x = 0
    for index, item in enumerate(selected):
        episode = write_episode_reference(
            item, args.out, camera_tcp,
            redact_source_paths=args.redact_source_paths)
        episode["order"] = index
        episode["v10"] = v10_episode_reference(
            args.v10, episode["episode_id"],
            redact_source_paths=args.redact_source_paths)
        reference_episodes.append(episode)
        total_pose += int(episode["pose_csv"]["rows"])
        total_gripper += int(episode["gripper_csv"]["rows"])
        counts = episode["gripper_csv"]["status_counts"]
        total_d += int(counts["D"])
        total_m += int(counts["M"])
        total_x += int(counts["X"])
        print(f"{index + 1:02d}/{len(selected)} {episode['episode_id']}", flush=True)

    exporter_path = Path(__file__).resolve()
    code_paths = [
        exporter_path,
        repo / "AI/tools/export_orbslam_episode_gate_report.py",
        repo / "AI/tools/build_orbslam_umi_relative_dataset.py",
        repo / "AI/umi/relative_dataset.py",
    ]
    calibration_paths = [
        args.camera_tcp,
        repo / "AI/configs/real/umi_orbslam_intake_0918.json",
        repo / "AI/configs/real/s22_orbslam3_mono_inertial_0918.yaml",
        repo / "AI/configs/real/s22_slam_mask_0918.json",
    ]
    reviewed_exception_source = selection.get("reviewed_gap_exception_source")
    if isinstance(reviewed_exception_source, dict) and reviewed_exception_source.get("path"):
        calibration_paths.append(Path(reviewed_exception_source["path"]).resolve())

    def manifest_path(path: Path) -> str:
        if args.redact_source_paths:
            return path.name
        try:
            return str(path.relative_to(repo)).replace("\\", "/")
        except ValueError:
            return str(path)

    git_tracked_paths: list[str] = []
    for path in code_paths + calibration_paths:
        try:
            git_tracked_paths.append(str(path.relative_to(repo)))
        except ValueError:
            # An external calibration file is hashed into the manifest but is
            # not part of this Git worktree, so Git cannot report its dirtiness.
            pass

    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "status": "PROVISIONAL_CALIBRATION_NOT_FOR_PHYSICAL_DEPLOYMENT",
        "artifact": "Track A reference; not official UMI Zarr",
        "reference_scope": (
            f"Track A reviewed {len(selected)}-episode accepted selection; ordering is fixed "
            "by the selection file SHA-256"
        ),
        "selection": {
            "path": manifest_path(args.selection),
            "sha256": sha256_file(args.selection),
            "episode_order": order,
            "selected": len(selected),
            "excluded": len(selection.get("excluded", {})),
            "total_inputs": len(selected) + len(selection.get("excluded", {})),
            "excluded_episodes": selection.get("excluded", {}),
        },
        "episode_gate_report": {
            "included": bool(gate_report_files),
            "files": gate_report_files,
            "count_semantics": (
                "rejection counts are episode counts; one episode may have multiple categories"
            ),
        },
        "v10": {
            "artifact": "v10",
            "not_official_umi_zarr": True,
            "root": manifest_path(args.v10),
            "dataset_json_sha256": sha256_file(v10_index_path),
            "schema": v10_index.get("schema"),
            "episodes": v10_index.get("n_episodes"),
            "rows": v10_index.get("n_rows"),
            "rate_hz": v10_index.get("rate_hz"),
            "observation_horizon": v10_index.get("observation_horizon"),
            "action_horizon": v10_index.get("action_horizon"),
        },
        "code": {
            "git_commit": git_output(repo, "rev-parse", "HEAD"),
            "git_branch": git_output(repo, "branch", "--show-current"),
            "git_dirty": bool(git_output(
                repo, "status", "--short", "--",
                *git_tracked_paths,
            )),
            "files": {
                manifest_path(path): sha256_file(path)
                for path in code_paths
            },
        },
        "calibration": {
            "calibration_id": camera_config.get(
                "calibration_id",
                "s22_camera_tcp_cad_video_aligned_provisional_0918+aruco_id13_0p16m",
            ),
            "camera_tcp_status": camera_config.get("status"),
            "physical_deployment_allowed": bool(
                camera_config.get("physical_deployment_allowed", False)),
            "fixed_marker": {
                "dictionary": "DICT_4X4_50", "id": 13,
                "black_square_size_m": 0.16,
            },
            "files": {
                manifest_path(path): sha256_file(path)
                for path in calibration_paths if path.is_file()
            },
            "alignment_files": sorted({
                episode["source"]["alignment_sha256"] for episode in reference_episodes
            }),
            "gripper_metric_scale_verified": bool(
                v10_index.get("gripper_scale_verified", False)),
        },
        "pose_csv_contract": {
            "frame": "ArUco DICT_4X4_50 ID 13 marker/table frame",
            "camera_semantics": "T_marker_camera",
            "tcp_semantics": "T_marker_tcp = T_marker_camera @ T_camera_tcp",
            "translation_unit": "m",
            "quaternion_order": "qx,qy,qz,qw",
        },
        "gripper_csv_contract": {
            "gap_unit": "m",
            "gap_definition": "absolute inner contact-pad spacing",
            "status": {"D": "direct marker detection", "M": "interpolated", "X": "invalid"},
        },
        "source_image_identity": {
            "policy": (
                "video.mp4 SHA-256 plus frames.csv SHA-256 binds the ordered original input; "
                "decoder-dependent RGB hashes are diagnostics in v10 array references"
            ),
            "episode_video_hashes_are_in_episode_entries": True,
        },
        "environment": environment_reference(),
        "episodes": reference_episodes,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checks = {
        "status": "PASS_REFERENCE_BUNDLE_PROVISIONAL_CALIBRATION",
        "physical_deployment_ready": False,
        "reference_scope": manifest["reference_scope"],
        "checks": {
            "selection_v10_order_identical": True,
            "pose_csv_count": len(reference_episodes),
            "gripper_csv_count": len(reference_episodes),
            "pose_gripper_rows_equal": total_pose == total_gripper,
            "pose_rows": total_pose,
            "gripper_rows": total_gripper,
            "gripper_status_counts": {"D": total_d, "M": total_m, "X": total_x},
            "invalid_gap_rows": total_x,
            "v10_contract_problems": [],
            "track_a_selection_is_canonical": True,
        },
        "manifest_sha256": sha256_file(manifest_path),
        "limitations": [
            "camera-to-TCP origin remains provisional",
            "gripper metric scale lacks the empty-gripper full-stroke verification",
            "this is not official UMI Zarr and not physical-deployment approval",
        ],
    }
    checks["limitations"] = [item for item in checks["limitations"] if item]
    verification_path = args.out / "verification_reference.json"
    verification_path.write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    add_file_hashes(args.out, manifest)

    if args.tar:
        if args.tar.exists():
            raise SystemExit(f"tar already exists: {args.tar}")
        with tarfile.open(args.tar, "w") as archive:
            archive.add(args.out, arcname=args.out.name)

    print(json.dumps({
        "status": checks["status"],
        "episodes": len(reference_episodes),
        "pose_rows": total_pose,
        "gripper_rows": total_gripper,
        "gripper_status_counts": {"D": total_d, "M": total_m, "X": total_x},
        "output": str(args.out),
        "tar": str(args.tar) if args.tar else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
