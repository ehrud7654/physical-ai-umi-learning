from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import zarr
from numcodecs import Blosc

from .camera_tcp import load_camera_tcp
from .episode import build_episode
from .recording import load_recording
from .slam_tag import load_slam_tag


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _inpaint_markers(frame, detector):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for marker in corners:
        cv2.fillConvexPoly(mask, np.rint(marker[0]).astype(np.int32), 255)
    mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8))
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def _write_images(array, offset: int, session: Path, selected: np.ndarray,
                  output_resolution: list[int], inpaint: bool):
    wanted = {int(frame_index): offset + index for index, frame_index in enumerate(selected)}
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ) if inpaint else None
    capture = cv2.VideoCapture(str(session / "video.mp4"))
    written = 0
    frame_index = 0
    out_width, out_height = output_resolution
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        destination = wanted.get(frame_index)
        if destination is not None:
            if detector is not None:
                frame = _inpaint_markers(frame, detector)
            height, width = frame.shape[:2]
            crop_width = round(height * out_width / out_height)
            if crop_width > width:
                raise ValueError("output aspect ratio exceeds source image width")
            left = (width - crop_width) // 2
            frame = frame[:, left:left + crop_width]
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            array[destination] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            written += 1
        frame_index += 1
    capture.release()
    if written != len(selected):
        raise ValueError(f"decoded {written}/{len(selected)} selected images from {session.name}")


def build_replay_buffer(plan_path: Path, output: Path, dataset_config_path: Path,
                        camera_tcp_path: Path, allow_unrated: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or not plan.get("episodes"):
        raise ValueError("episode plan must contain at least one schema_version 1 episode")
    dataset_config = json.loads(dataset_config_path.read_text(encoding="utf-8"))
    camera_tcp, camera_tcp_config = load_camera_tcp(camera_tcp_path)
    base = plan_path.resolve().parent
    # Table-marker world frame: one shared tag transform (mapping session) or one per episode
    # (marker seen during each recording's warm-up). Mixing tagged and untagged episodes would put
    # them in different coordinate systems, so that is refused.
    shared_tag = _resolve(base, plan["slam_tag"]) if plan.get("slam_tag") else None
    episode_tags = [_resolve(base, item["slam_tag"]) if item.get("slam_tag") else shared_tag
                    for item in plan["episodes"]]
    if any(tag is None for tag in episode_tags) and any(tag is not None for tag in episode_tags):
        raise ValueError("some episodes have a slam_tag and others do not; world frames would be inconsistent")
    world_frame = ("slam_origin_of_each_recording" if episode_tags[0] is None
                   else "mapping_table_tag" if shared_tag and all(t == shared_tag for t in episode_tags)
                   else "per_recording_table_tag")
    episodes = []
    reports = []
    for item, tag_path in zip(plan["episodes"], episode_tags):
        session = _resolve(base, item["session"])
        trajectory = _resolve(base, item["trajectory"])
        gripper = _resolve(base, item["gripper"])
        world_slam = np.linalg.inv(load_slam_tag(tag_path)) if tag_path else None
        recording = load_recording(session, require_success=not allow_unrated)
        data, selected, report = build_episode(
            recording, trajectory, gripper, camera_tcp, dataset_config, world_slam
        )
        episodes.append((recording, selected, data))
        reports.append({
            "session": session.name,
            "outcome": recording["manifest"].get("outcome"),
            "duration_s": recording["duration_s"],
            "camera_encoder_max_error_us": recording["camera_encoder_max_error_us"],
            "source_sha256": {
                "manifest.json": _sha256(session / "manifest.json"),
                "video.mp4": _sha256(session / "video.mp4"),
                "camera_trajectory.csv": _sha256(trajectory),
                "gripper_width.csv": _sha256(gripper),
                **({"tx_slam_tag.json": _sha256(tag_path)} if tag_path else {}),
            },
            **report,
        })

    episode_lengths = [len(selected) for _, selected, _ in episodes]
    episode_ends = np.cumsum(episode_lengths, dtype=np.int64)
    total = int(episode_ends[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    try:
        with zarr.ZipStore(str(temporary), mode="w") as store:
            root = zarr.group(store=store)
            data_group = root.create_group("data")
            meta_group = root.create_group("meta")
            meta_group.array("episode_ends", episode_ends, compressor=None)
            for key in episodes[0][2]:
                values = np.concatenate([episode[2][key] for episode in episodes], axis=0)
                data_group.array(key, values, chunks=(min(1024, total), values.shape[1]),
                                 compressor=compressor)
            out_width, out_height = dataset_config["output_resolution"]
            images = data_group.create_dataset(
                "camera0_rgb", shape=(total, out_height, out_width, 3),
                chunks=(1, out_height, out_width, 3), dtype=np.uint8, compressor=compressor,
            )
            offset = 0
            for recording, selected, _ in episodes:
                _write_images(images, offset, recording["session"], selected,
                              dataset_config["output_resolution"],
                              dataset_config["inpaint_aruco_markers"])
                offset += len(selected)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    with zarr.ZipStore(str(output), mode="r") as store:
        root = zarr.group(store=store)
        if root["meta/episode_ends"][-1] != total or root["data/camera0_rgb"].shape[0] != total:
            raise RuntimeError("saved replay buffer failed reload validation")
    statuses = {report["outcome"] for report in reports}
    report = {
        "status": "pass",
        "dataset": str(output.resolve()),
        "dataset_sha256": _sha256(output),
        "episodes": len(episodes),
        "frames": total,
        "episode_ends": episode_ends.tolist(),
        "native_sample_rate_hz": float(np.median([item["native_sample_rate_hz"] for item in reports])),
        "schema": "Stanford UMI ReplayBuffer data/meta layout",
        "image_contract": "upright_v1 pixels; center crop; RGB; ArUco inpainted",
        "pose_contract": "T_world_tcp = T_world_slam @ T_slam_camera @ T_camera_tcp; metres; axis-angle radians",
        "world_frame": world_frame,
        "slam_tag": str(shared_tag) if shared_tag else None,
        "episode_start_trim_s": dataset_config.get("episode_start_trim_s", 0.0),
        "camera_tcp_status": camera_tcp_config["status"],
        "training_input_status": (
            "ready" if statuses == {"success"} and camera_tcp_config["status"] == "physically_validated"
            else "provisional_camera_tcp" if statuses == {"success"}
            else "development_only_unrated_input"
        ),
        "physical_deployment_ready": bool(camera_tcp_config.get("physical_deployment_allowed", False)),
        "config_sha256": {
            "dataset": _sha256(dataset_config_path),
            "camera_tcp": _sha256(camera_tcp_path),
            "episode_plan": _sha256(plan_path),
            **({"slam_tag": _sha256(shared_tag)} if shared_tag else {}),
        },
        "episode_reports": reports,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def inspect_replay_buffer(path: Path) -> dict:
    with zarr.ZipStore(str(path), mode="r") as store:
        root = zarr.group(store=store)
        return {
            "episode_ends": root["meta/episode_ends"][:].tolist(),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for name, value in root["data"].items()},
        }
