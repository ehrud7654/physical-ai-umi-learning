from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_camera_tcp(path: Path) -> tuple[np.ndarray, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    camera_lens = np.asarray(config["T_camera_lens"], dtype=np.float64)
    lens_tcp = np.asarray(config["T_lens_tcp"], dtype=np.float64)
    declared = np.asarray(config["T_camera_tcp"], dtype=np.float64)
    for name, value in (("T_camera_lens", camera_lens), ("T_lens_tcp", lens_tcp),
                        ("T_camera_tcp", declared)):
        if value.shape != (4, 4) or not np.allclose(value[3], [0, 0, 0, 1]):
            raise ValueError(f"invalid {name}")
        rotation = value[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
            raise ValueError(f"non-rigid {name}")
    composed = camera_lens @ lens_tcp
    if not np.allclose(composed, declared, atol=1e-9):
        raise ValueError("T_camera_tcp does not equal T_camera_lens @ T_lens_tcp")
    return declared, config


def camera_pose_to_tcp(position: np.ndarray, quaternion_xyzw: np.ndarray,
                       camera_tcp: np.ndarray, world_slam: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """T_world_tcp = T_world_slam @ T_slam_camera @ T_camera_tcp.

    world_slam is inverse(T_slam_tag) from the mapping session's table marker; without it the
    SLAM origin of this recording is the world frame."""
    count = len(position)
    slam_camera = np.zeros((count, 4, 4), dtype=np.float64)
    slam_camera[:, 3, 3] = 1
    slam_camera[:, :3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    slam_camera[:, :3, 3] = position
    world_camera = slam_camera if world_slam is None else world_slam @ slam_camera
    world_tcp = world_camera @ camera_tcp
    return world_tcp[:, :3, 3].astype(np.float32), Rotation.from_matrix(
        world_tcp[:, :3, :3]).as_rotvec().astype(np.float32)
