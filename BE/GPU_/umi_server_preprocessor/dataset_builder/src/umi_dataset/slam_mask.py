"""SLAM mask for the gripper, Stanford UMI style (`draw_predefined_mask`).

Features on the handle and fingers move with the camera, so they have zero parallax and
corrupt ORB-SLAM3 two-view initialisation and tracking. The fingers slide sideways as the
gripper opens, so the mask covers the whole region they can occupy plus the handle body.
Regions are image fractions in configs/s22_slam_mask.json; the PNG is white where masked."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def build_mask(config_path: Path, output_png: Path, width: int = 1920, height: int = 1080) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in config["regions"]:
        points = np.rint(np.asarray(region["polygon_xy_fraction"], dtype=np.float64) * [width, height]).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_png), mask):
        raise OSError(f"cannot write {output_png}")
    return {"config": str(config_path), "regions": [r["name"] for r in config["regions"]],
            "masked_fraction": float((mask > 0).mean()), "mask_size": [width, height], "output": str(output_png)}


def overlay(video: Path, mask_png: Path, output_jpg: Path, frame_index: int = 0) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("cannot read frame")
    mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    frame[mask > 0] = (0.45 * frame[mask > 0] + np.array([0, 0, 140])).astype(np.uint8)
    cv2.imwrite(str(output_jpg), cv2.resize(frame, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
