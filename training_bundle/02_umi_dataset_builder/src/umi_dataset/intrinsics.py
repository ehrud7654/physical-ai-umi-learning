from __future__ import annotations

import numpy as np


def scaled_camera_matrix(camera: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole matrix from the calibrated 960x540 intrinsics, scaled to the decoded video size."""
    source_width, source_height = camera["resolution"]
    fx, fy, cx, cy = camera["intrinsics_960x540"]
    matrix = np.asarray([
        [fx * width / source_width, 0, cx * width / source_width],
        [0, fy * height / source_height, cy * height / source_height],
        [0, 0, 1],
    ], dtype=np.float64)
    return matrix, np.asarray(camera["distortion_coeffs"], dtype=np.float64)
