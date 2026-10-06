"""Compatibility adapter for ``arpose.episode/1`` delivery archives.

The recorder delivered one outer ZIP containing many short recording folders.
This adapter creates one lossless ``umi_raw/0.1.0`` ZIP per recording.  JPEGs
are copied byte-for-byte; no image decode, resize, rotation, or recompression is
performed here.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


SOURCE_SCHEMA = "arpose.episode/1"
TARGET_SCHEMA = "umi_raw/0.1.0"
SKILL_ALIASES = {"pick_place_cup": "pick_place"}


def seconds_segments_to_rows(timestamps_ns: list[int], segments) -> list[list[int]]:
    """Convert inclusive time endpoints from the recorder into row half-ranges."""
    if len(timestamps_ns) < 2 or any(b <= a for a, b in zip(timestamps_ns, timestamps_ns[1:])):
        raise ValueError("pose timestamps must be strictly increasing")
    origin = timestamps_ns[0]
    offsets = [(stamp - origin) / 1e9 for stamp in timestamps_ns]
    result = []
    for segment in segments:
        if (not isinstance(segment, list) or len(segment) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in segment)):
            raise ValueError("source usable_segments must contain time pairs")
        start_s, end_s = map(float, segment)
        if not 0 <= start_s <= end_s:
            raise ValueError("invalid source usable segment")
        start = bisect_left(offsets, start_s)
        end = bisect_right(offsets, end_s)
        if start < end:
            if result and start < result[-1][1]:
                raise ValueError("overlapping source usable segments")
            result.append([start, end])
    if not result:
        raise ValueError("no usable pose rows")
    return result


def reviewed_segments_to_rows(timestamps_ns: list[int], segments) -> list[list[int]] | None:
    """Return None for an episode explicitly quarantined by recorder tracking QC."""
    if not segments:
        return None
    return seconds_segments_to_rows(timestamps_ns, segments)


def _producer_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (Path("umi/gripper_gap.py"), Path("tools/gripper_gap.py")):
        digest.update(relative.as_posix().encode())
        digest.update((root / relative).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_meta(source: dict, producer_sha: str, delivery_sha: str,
                    source_episode_sha: str) -> dict:
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unsupported source schema")
    contract, camera = source["contract"], source["camera"]
    modes = camera.get("modes", "").lower()
    if not all(value in modes for value in ("ois=0", "eis=0", "ae_lock=true", "awb_lock=true")):
        raise ValueError("camera stabilization/exposure contract mismatch")
    width, height = contract["image_size"]
    intrinsics = camera["intrinsics"]
    skill = source.get("session", {}).get("skill_id", "")
    skill = SKILL_ALIASES.get(skill, skill)
    return {
        "schema": TARGET_SCHEMA,
        "episode_id": source["episode"],
        "skill_id": skill,
        "collected_by": source.get("session", {}).get("user_id", ""),
        "recorded_at_ms": source.get("recorded_at_ms"),
        "device": {"model": source.get("device", ""), "os": source.get("android", ""),
                   "app_version": "arpose.episode/1"},
        "clock": {"source": "elapsedRealtimeNanos", "all_streams_same_clock": True},
        "camera": {
            "which": "main", "width": width, "height": height, "nominal_hz": 30,
            "focus_mode": "fixed", "ae": "locked", "awb": "locked",
            "ois": "off", "vdis": "off",
            "intrinsics": {key: intrinsics[key] for key in ("fx", "fy", "cx", "cy")},
            "distortion": None,
            "stored_image_orientation": "180deg from canonical upright",
            "rotate_to_canonical_deg": 180,
        },
        "pose": {
            "provider": "ARCore", "pose_is": contract["pose_frame"],
            "frame": "arcore_world", "handedness": "right",
            "units": contract["translation_units"],
            "quaternion_order": contract["quaternion_order"],
            "camera_local_axes": "+X right, +Y up, -Z forward (ARCore/OpenGL)",
            "display_orientation_applied": False,
        },
        "gripper": {
            "method": "hsv_magenta_marker_scale", "marker_mm": 15,
            "calib_closed_mm": 0, "calib_open_mm": 70,
            "marker_distance_closed_mm": 37.6,
            "marker_distance_open_mm": 134.0,
            "midpoint_check": {"contact_gap_mm": 30, "estimated_gap_mm": 31.0568491,
                               "status": "provisional_single_image"},
            "producer": {"name": "AI/tools/gripper_gap.py",
                         "version": "1.0.0-provisional", "sha256": producer_sha},
        },
        "source": {"schema": SOURCE_SCHEMA, "episode": source["episode"],
                   "delivery_sha256": delivery_sha,
                   "episode_json_sha256": source_episode_sha},
    }


def normalize_delivery(*, delivery_zip: Path, gripper_root: Path,
                       output_dir: Path, pre_stabilized: bool) -> dict:
    """Create canonical per-episode ZIPs and quality JSON sidecars."""
    if pre_stabilized is not True:
        raise ValueError("this delivery requires explicit pre_stabilized=True evidence")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    producer_sha = _producer_sha()
    delivery_sha = _file_sha256(delivery_zip)
    manifest = []
    rejected = []
    with zipfile.ZipFile(delivery_zip) as source_zip:
        names = source_zip.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate delivery ZIP entries")
        name_set = set(names)
        episode_files = sorted(name for name in names if name.endswith("/episode.json"))
        for episode_name in episode_files:
            prefix = PurePosixPath(episode_name).parent
            source_episode_bytes = source_zip.read(episode_name)
            source_meta = json.loads(source_episode_bytes)
            episode_id = source_meta["episode"]
            if prefix.name != episode_id:
                raise ValueError("episode folder/id mismatch")
            pose_name = str(prefix / "poses.csv")
            pose_rows = list(csv.DictReader(io.StringIO(
                source_zip.read(pose_name).decode("utf-8-sig"))))
            required = {"index", "timestamp_ns", "tracking", "x", "y", "z",
                        "qx", "qy", "qz", "qw", "image"}
            if not pose_rows or not required.issubset(pose_rows[0]):
                raise ValueError(f"invalid poses.csv: {episode_id}")
            timestamps = [int(row["timestamp_ns"]) for row in pose_rows]
            source_segments = source_meta.get("summary", {}).get("usable_segments", [])
            segments = reviewed_segments_to_rows(timestamps, source_segments)
            if segments is None:
                rejected.append({"episode_id": episode_id,
                                 "reason": "no_usable_segments_after_tracking_qc"})
                continue

            grip_path = gripper_root / episode_id / "gripper.csv"
            if not grip_path.is_file():
                raise ValueError(f"missing generated gripper.csv: {episode_id}")
            grip_bytes = grip_path.read_bytes()
            grip_rows = list(csv.DictReader(io.StringIO(grip_bytes.decode("utf-8-sig"))))
            if len(grip_rows) != len(pose_rows):
                raise ValueError(f"gripper/pose row mismatch: {episode_id}")

            pose_stream, frame_stream = io.StringIO(), io.StringIO()
            pose_writer = csv.writer(pose_stream, lineterminator="\n")
            frame_writer = csv.writer(frame_stream, lineterminator="\n")
            pose_columns = ["index", "timestamp_ns", "tracking", "x", "y", "z",
                            "qx", "qy", "qz", "qw"]
            pose_writer.writerow(pose_columns)
            frame_writer.writerow(["frame_index", "timestamp_ns", "image"])
            image_members = []
            for row in pose_rows:
                pose_writer.writerow([row[key] for key in pose_columns])
                image = PurePosixPath(row["image"])
                if image.parent == PurePosixPath('.'):
                    image = PurePosixPath("frames") / image
                member = str(prefix / image)
                if member not in name_set:
                    raise ValueError(f"missing image: {member}")
                frame_writer.writerow([row["index"], row["timestamp_ns"], str(image)])
                image_members.append((str(image), member))

            quality = {
                "tracking_valid_values": ["TRACKING"], "warmup_s": 0.0,
                "pre_stabilized": True, "usable_segments": segments,
                "frames_dropped": int(source_meta.get("summary", {}).get("frames_dropped", 0)),
                "source_usable_segments_seconds": source_meta["summary"]["usable_segments"],
            }
            canonical = _canonical_meta(source_meta, producer_sha, delivery_sha,
                hashlib.sha256(source_episode_bytes).hexdigest())
            target = output_dir / f"{episode_id}.zip"
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as out:
                out.writestr("episode.json", json.dumps(canonical, ensure_ascii=False, indent=2))
                out.writestr("poses.csv", pose_stream.getvalue())
                out.writestr("frames.csv", frame_stream.getvalue())
                out.writestr("gripper.csv", grip_bytes)
                for plain in ("imu.csv", "summary.txt", "meta.txt", "marker_qc.txt"):
                    source_name = str(prefix / plain)
                    if source_name in name_set:
                        out.writestr(plain, source_zip.read(source_name))
                out.writestr("source_episode.json", source_episode_bytes)
                for target_name, source_name in image_members:
                    out.writestr(target_name, source_zip.read(source_name))
            quality_path = output_dir / f"{episode_id}.quality.json"
            quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({"episode_id": episode_id, "bundle": target.name,
                             "quality": quality_path.name, "frames": len(pose_rows),
                             "usable_segments": segments,
                             "bundle_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    result = {"source_schema": SOURCE_SCHEMA, "target_schema": TARGET_SCHEMA,
              "source_delivery_sha256": delivery_sha,
              "pre_stabilized": True, "warmup_s": 0.0,
              "producer_sha256": producer_sha, "episodes": manifest,
              "rejected": rejected}
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
