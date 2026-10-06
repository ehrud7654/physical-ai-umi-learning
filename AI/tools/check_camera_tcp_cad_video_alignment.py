"""Cross-check the HW CAD TCP against the historical video hand-eye result.

The 2026-09-11 images used by the historical calibration were stored in the
physical upside-down orientation.  The 2026-09-18 collector bakes a 180-degree
pixel rotation into ``upright_v1``.  This check makes that frame change
explicit before comparing the two transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAD = ROOT / "configs/real/s22_camera_tcp_cad_provisional_0918.json"
DEFAULT_ALIGNED = ROOT / "configs/real/s22_camera_tcp_cad_video_aligned_provisional_0918.json"
DEFAULT_VIDEO = ROOT / "configs/real/umi_s22_canonical_pinch_side_grasp_provisional.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rigid(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"invalid {name}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"non-orthonormal {name}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"improper {name}")
    return matrix


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--aligned", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cad = _load(args.cad)
    aligned = _load(args.aligned)
    video = _load(args.video)

    old_camera_tcp = _rigid(cad["T_camera_tcp"], "old T_camera_tcp")
    lens_tcp = _rigid(cad["T_lens_tcp"], "T_lens_tcp")
    camera_lens = _rigid(aligned["T_camera_lens"], "aligned T_camera_lens")
    proposed = _rigid(aligned["T_camera_tcp"], "aligned T_camera_tcp")
    canonical = _rigid(
        aligned["T_cad_tcp_v10_canonical"], "T_cad_tcp_v10_canonical"
    )

    if not np.allclose(camera_lens @ lens_tcp @ canonical, proposed, atol=1e-9):
        raise ValueError("aligned transform does not compose from CAD inputs")

    gl_from_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    upright_from_old_jpeg = np.diag([-1.0, -1.0, 1.0, 1.0])
    video_old_gl = _rigid(video["t_cam_to_pinch"], "video t_cam_to_pinch")
    video_upright = upright_from_old_jpeg @ gl_from_cv @ video_old_gl

    marker_to_cad = np.linalg.inv(video_upright) @ proposed
    old_rotation_error = _rotation_error_deg(video_upright, old_camera_tcp)
    proposed_rotation_error = _rotation_error_deg(video_upright, proposed)
    camera_delta_mm = (proposed[:3, 3] - video_upright[:3, 3]) * 1000.0
    marker_delta_mm = marker_to_cad[:3, 3] * 1000.0

    result = {
        "status": "PASS_CAD_TCP_VIDEO_AXIS_ALIGNMENT_ORIGIN_PROVISIONAL",
        "cad_tcp_to_v10_canonical_is_identity": bool(
            np.allclose(canonical, np.eye(4), atol=1e-12)
        ),
        "old_axis_assumption_rotation_error_deg": old_rotation_error,
        "proposed_axis_rotation_error_deg": proposed_rotation_error,
        "proposed_minus_video_marker_in_upright_camera_mm": camera_delta_mm.tolist(),
        "video_marker_to_cad_tcp_translation_in_canonical_mm": marker_delta_mm.tolist(),
        "origin_delta_norm_mm": float(np.linalg.norm(camera_delta_mm)),
        "physical_deployment_allowed": False,
        "remaining_blockers": [
            "CAD design TCP to physical contact-pad centre is not measured",
            "CAD lens origin to optical entrance pupil is not measured",
        ],
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")

    if not result["cad_tcp_to_v10_canonical_is_identity"]:
        return 1
    if proposed_rotation_error > 1e-3:
        return 1
    if old_rotation_error < 80.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
