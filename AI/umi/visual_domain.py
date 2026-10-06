"""Validated application of a simulation-only visual-domain proxy."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "s22_mujoco_visual_domain/0.1.0-provisional"
STATUS = "VISUAL_DOMAIN_PROXY_NOT_HARDWARE_CALIBRATION"


def load_visual_domain(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != STATUS:
        raise ValueError("unsupported or non-provisional visual-domain config")
    return payload


def apply_visual_domain(cfg: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copied runtime config with camera and appearance overrides.

    The explicit status check prevents this fitted visual proxy from being
    mistaken for the robot's measured optical extrinsic.
    """
    if payload.get("schema") != SCHEMA or payload.get("status") != STATUS:
        raise ValueError("visual-domain payload lacks the required provisional contract")
    out = copy.deepcopy(cfg)
    camera = payload.get("camera", {})
    name = str(camera.get("name", ""))
    if name not in out.get("cameras", {}) or not isinstance(out["cameras"][name], dict):
        raise ValueError(f"unknown visual-domain camera {name!r}")
    position = np.asarray(camera.get("pos"), dtype=float)
    xyaxes = np.asarray(camera.get("xyaxes"), dtype=float)
    fovy = float(camera.get("fovy", float("nan")))
    if (position.shape != (3,) or xyaxes.shape != (6,)
            or not np.isfinite(position).all() or not np.isfinite(xyaxes).all()
            or not np.isfinite(fovy) or not 1.0 < fovy < 179.0):
        raise ValueError("invalid visual-domain camera geometry")
    out["cameras"][name]["pos"] = position.tolist()
    out["cameras"][name]["xyaxes"] = xyaxes.tolist()
    out["cameras"][name]["fovy"] = fovy

    appearance = payload.get("appearance", {})
    required = ("object_rgba", "table_rgba", "floor_rgba", "light_diffuse_rgb")
    if not isinstance(appearance, dict) or any(key not in appearance for key in required):
        raise ValueError("visual-domain appearance block is incomplete")
    for key in required:
        values = np.asarray(appearance[key], dtype=float)
        expected = (3,) if key == "light_diffuse_rgb" else (4,)
        if values.shape != expected or not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"invalid visual-domain appearance {key}")
    geometry = appearance.get("visual_geometry", [])
    if not isinstance(geometry, list):
        raise ValueError("visual-domain visual_geometry must be a list")
    seen: set[str] = set()
    for row in geometry:
        if not isinstance(row, dict):
            raise ValueError("visual-domain geometry entry must be an object")
        name = str(row.get("name", ""))
        parent = str(row.get("parent", ""))
        kind = str(row.get("type", "box"))
        pos = np.asarray(row.get("pos_m"), dtype=float)
        size = np.asarray(row.get("size_m"), dtype=float)
        rgba = np.asarray(row.get("rgba"), dtype=float)
        expected_size = (3,) if kind == "box" else (2,) if kind == "cylinder" else ()
        if (not name or name in seen or parent not in {"world", "target_object"}
                or expected_size == () or pos.shape != (3,) or size.shape != expected_size
                or rgba.shape != (4,) or not np.isfinite(pos).all()
                or not np.isfinite(size).all() or np.any(size <= 0.0)
                or not np.isfinite(rgba).all() or np.any(rgba < 0.0)
                or np.any(rgba > 1.0)):
            raise ValueError(f"invalid visual-domain geometry entry {name!r}")
        seen.add(name)
    out["visual_domain"] = copy.deepcopy(appearance)
    return out
