"""Overlay observed jaw-marker and candidate camera-to-TCP axes on 0918 RGB.

This is a read-only calibration diagnostic.  It never rotates or rewrites the
source video.  The jaw opening axis is the undirected line between ArUco IDs
0 and 1.  Candidate TCP +X/+Z axes are projected using the S22 intrinsics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np


def load_transform(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = np.asarray(data["T_camera_tcp"], dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"invalid T_camera_tcp: {path}")
    return value


def yaml_number(text: str, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {key}")
    return float(match.group(1))


def load_intrinsics(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    calibration_width = yaml_number(text, "Camera.width")
    calibration_height = yaml_number(text, "Camera.height")
    scale_x = width / calibration_width
    scale_y = height / calibration_height
    camera = np.asarray([
        [yaml_number(text, "Camera1.fx") * scale_x, 0,
         yaml_number(text, "Camera1.cx") * scale_x],
        [0, yaml_number(text, "Camera1.fy") * scale_y,
         yaml_number(text, "Camera1.cy") * scale_y],
        [0, 0, 1],
    ], dtype=np.float64)
    distortion = np.asarray([
        yaml_number(text, "Camera1.k1"), yaml_number(text, "Camera1.k2"),
        yaml_number(text, "Camera1.p1"), yaml_number(text, "Camera1.p2"), 0,
    ], dtype=np.float64)
    return camera, distortion


def project_axes(
    transform: np.ndarray, camera: np.ndarray, distortion: np.ndarray,
    axis_length_m: float,
) -> dict[str, np.ndarray]:
    points_tcp = np.asarray([
        [0, 0, 0], [axis_length_m, 0, 0], [0, 0, axis_length_m],
    ], dtype=np.float64)
    points_camera = (
        transform[:3, :3] @ points_tcp.T + transform[:3, 3, None]
    ).T
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("projected TCP axes are behind the camera")
    pixels, _ = cv2.projectPoints(
        points_camera, np.zeros(3), np.zeros(3), camera, distortion)
    pixels = pixels.reshape(-1, 2)
    return {"origin": pixels[0], "x": pixels[1], "z": pixels[2]}


def angle(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def undirected_error(left_deg: float, right_deg: float) -> float:
    difference = abs((left_deg - right_deg + 180.0) % 360.0 - 180.0)
    return min(difference, 180.0 - difference)


def point(value: np.ndarray) -> tuple[int, int]:
    return int(round(float(value[0]))), int(round(float(value[1])))


def draw_arrow(
    image: np.ndarray, start: np.ndarray, end: np.ndarray,
    color: tuple[int, int, int], label: str,
) -> None:
    cv2.arrowedLine(image, point(start), point(end), color, 7, cv2.LINE_AA, tipLength=0.18)
    target = point(end)
    cv2.putText(image, label, (target[0] + 8, target[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)


def detected_frames(video: Path, detector: cv2.aruco.ArucoDetector) -> list[dict]:
    capture = cv2.VideoCapture(str(video))
    found: list[dict] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                by_id = {
                    int(marker_id): np.asarray(marker_corners).reshape(-1, 2).mean(axis=0)
                    for marker_corners, marker_id in zip(corners, ids.reshape(-1))
                }
                if 0 in by_id and 1 in by_id:
                    found.append({"frame_index": index, "centres": by_id})
            index += 1
    finally:
        capture.release()
    return found


def read_frame(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot read {video} frame {index}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--old-camera-tcp", type=Path, required=True)
    parser.add_argument("--candidate-camera-tcp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--axis-length-m", type=float, default=0.06)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    all_episodes = list(selection["episodes"])
    indices = np.linspace(0, len(all_episodes) - 1, min(args.episodes, len(all_episodes)))
    episodes = [all_episodes[int(round(value))] for value in indices]
    old = load_transform(args.old_camera_tcp)
    candidate = load_transform(args.candidate_camera_tcp)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    results: list[dict] = []
    thumbnails: list[np.ndarray] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        video = Path(episode["raw_episode"]) / "video.mp4"
        found = detected_frames(video, detector)
        if not found:
            results.append({"episode_id": episode_id, "status": "NO_ID0_ID1_PAIR"})
            continue
        chosen_indices = np.linspace(
            0, len(found) - 1, min(args.frames_per_episode, len(found)))
        chosen = [found[int(round(value))] for value in chosen_indices]
        for item in chosen:
            frame = read_frame(video, int(item["frame_index"]))
            height, width = frame.shape[:2]
            camera, distortion = load_intrinsics(args.intrinsics, width, height)
            old_axes = project_axes(old, camera, distortion, args.axis_length_m)
            candidate_axes = project_axes(candidate, camera, distortion, args.axis_length_m)
            marker0 = item["centres"][0]
            marker1 = item["centres"][1]
            marker_angle = angle(marker1 - marker0)
            old_angle = angle(old_axes["x"] - old_axes["origin"])
            candidate_angle = angle(candidate_axes["x"] - candidate_axes["origin"])
            result = {
                "episode_id": episode_id,
                "frame_index": int(item["frame_index"]),
                "marker_axis_angle_deg": marker_angle,
                "old_tcp_x_angle_deg": old_angle,
                "candidate_tcp_x_angle_deg": candidate_angle,
                "old_undirected_error_deg": undirected_error(old_angle, marker_angle),
                "candidate_undirected_error_deg": undirected_error(candidate_angle, marker_angle),
                "candidate_tcp_z_angle_deg": angle(
                    candidate_axes["z"] - candidate_axes["origin"]),
            }
            results.append(result)

            overlay = frame.copy()
            cv2.line(overlay, point(marker0), point(marker1), (0, 255, 255), 8, cv2.LINE_AA)
            cv2.putText(overlay, "observed jaw axis ID0-ID1", point((marker0 + marker1) / 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3, cv2.LINE_AA)
            draw_arrow(overlay, old_axes["origin"], old_axes["x"], (0, 0, 255), "old +X")
            draw_arrow(overlay, candidate_axes["origin"], candidate_axes["x"],
                       (0, 255, 0), "candidate +X")
            draw_arrow(overlay, candidate_axes["origin"], candidate_axes["z"],
                       (255, 0, 0), "candidate +Z")
            title = (
                f"{episode_id} f={item['frame_index']}  "
                f"jaw err old={result['old_undirected_error_deg']:.1f}deg "
                f"candidate={result['candidate_undirected_error_deg']:.1f}deg"
            )
            cv2.rectangle(overlay, (0, 0), (width, 58), (0, 0, 0), -1)
            cv2.putText(overlay, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
            output_path = args.out / f"{episode_id}_f{item['frame_index']:04d}.jpg"
            cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
            thumbnails.append(cv2.resize(overlay, (640, 360), interpolation=cv2.INTER_AREA))

    measured = [item for item in results if "old_undirected_error_deg" in item]
    old_errors = [item["old_undirected_error_deg"] for item in measured]
    candidate_errors = [item["candidate_undirected_error_deg"] for item in measured]
    summary = {
        "status": "PASS_VISUAL_AXIS_DIAGNOSTIC" if measured else "NO_COMPARABLE_FRAMES",
        "source_pixels_modified": False,
        "episodes_requested": len(episodes),
        "frames_compared": len(measured),
        "observed_axis": "undirected line between ArUco IDs 0 and 1",
        "axis_length_m": args.axis_length_m,
        "old_error_deg_p50_max": [
            float(np.median(old_errors)), float(np.max(old_errors))
        ] if old_errors else None,
        "candidate_error_deg_p50_max": [
            float(np.median(candidate_errors)), float(np.max(candidate_errors))
        ] if candidate_errors else None,
        "preferred_by_jaw_axis": (
            "candidate" if candidate_errors and np.median(candidate_errors) < np.median(old_errors)
            else "old" if old_errors else None
        ),
        "results": results,
        "interpretation": (
            "Jaw-axis agreement validates projected TCP +X only. Inspect candidate +Z overlays "
            "against the physical forward/approach direction; this does not validate TCP origin."
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if thumbnails:
        columns = 3
        blank = np.zeros_like(thumbnails[0])
        while len(thumbnails) % columns:
            thumbnails.append(blank.copy())
        rows = [cv2.hconcat(thumbnails[index:index + columns])
                for index in range(0, len(thumbnails), columns)]
        cv2.imwrite(str(args.out / "contact_sheet.jpg"), cv2.vconcat(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if measured else 1


if __name__ == "__main__":
    raise SystemExit(main())
