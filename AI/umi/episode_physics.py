"""Per-demonstration object metadata used by diagnostic physics replay."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "umi_episode_physics/0.1.0"
EVIDENCE_LEVELS = ("verified", "provisional")
SHAPES = ("box",)


@dataclass(frozen=True)
class EpisodePhysics:
    episode_id: str
    object_type: str
    shape: str
    size_m: tuple[float, float, float]
    grasp_height_m: float
    evidence: str
    source: str
    object_center_from_pinch_m: tuple[float, float, float]
    registration_evidence: str
    grasp_rotation_base: tuple[tuple[float, float, float], ...]

    @property
    def half_size_m(self) -> np.ndarray:
        return np.asarray(self.size_m, dtype=float) / 2.0


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def validate_document(document: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if document.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    episodes = document.get("episodes")
    if not isinstance(episodes, dict) or not episodes:
        problems.append("episodes must be a non-empty object keyed by episode_id")
        return problems
    for episode_id, raw in episodes.items():
        prefix = f"episodes[{episode_id!r}]"
        if not isinstance(episode_id, str) or not episode_id:
            problems.append("episode_id keys must be non-empty strings")
        if not isinstance(raw, dict):
            problems.append(f"{prefix} must be an object")
            continue
        if not isinstance(raw.get("object_type"), str) or not raw.get("object_type", "").strip():
            problems.append(f"{prefix}.object_type must be a non-empty string")
        if raw.get("shape") not in SHAPES:
            problems.append(f"{prefix}.shape must be one of {SHAPES}")
        size = raw.get("size_m")
        if not isinstance(size, list) or len(size) != 3:
            problems.append(f"{prefix}.size_m must be [x, y, z] full dimensions")
        else:
            for index, value in enumerate(size):
                try:
                    _positive_number(value, f"{prefix}.size_m[{index}]")
                except ValueError as exc:
                    problems.append(str(exc))
        try:
            height = _positive_number(raw.get("grasp_height_m"), f"{prefix}.grasp_height_m")
            if isinstance(size, list) and len(size) == 3:
                object_height = _positive_number(size[2], f"{prefix}.size_m[2]")
                if height > object_height:
                    problems.append(f"{prefix}.grasp_height_m cannot exceed object height")
        except ValueError as exc:
            problems.append(str(exc))
        if raw.get("evidence") not in EVIDENCE_LEVELS:
            problems.append(f"{prefix}.evidence must be one of {EVIDENCE_LEVELS}")
        if not isinstance(raw.get("source"), str) or not raw.get("source", "").strip():
            problems.append(f"{prefix}.source must explain where the values came from")
        offset = raw.get("object_center_from_pinch_m")
        if not isinstance(offset, list) or len(offset) != 3:
            problems.append(f"{prefix}.object_center_from_pinch_m must be [x, y, z] in pinch axes")
        elif any(isinstance(value, bool) or not isinstance(value, (int, float))
                 or not np.isfinite(float(value)) for value in offset):
            problems.append(f"{prefix}.object_center_from_pinch_m must contain finite numbers")
        if raw.get("registration_evidence") not in EVIDENCE_LEVELS:
            problems.append(f"{prefix}.registration_evidence must be one of {EVIDENCE_LEVELS}")
        rotation = np.asarray(raw.get("grasp_rotation_base"), dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
            problems.append(
                f"{prefix}.grasp_rotation_base must be a proper 3x3 rotation; "
                "columns are jaw, remaining axis, approach"
            )
    return problems


def load_episode_physics(path: Path, episode_id: str) -> EpisodePhysics:
    document = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_document(document)
    if problems:
        raise ValueError("invalid episode physics metadata:\n  " + "\n  ".join(problems))
    try:
        raw = document["episodes"][episode_id]
    except KeyError as exc:
        raise ValueError(f"no physics metadata for episode {episode_id!r}") from exc
    return EpisodePhysics(
        episode_id=episode_id,
        object_type=raw["object_type"].strip(),
        shape=raw["shape"],
        size_m=tuple(float(value) for value in raw["size_m"]),
        grasp_height_m=float(raw["grasp_height_m"]),
        evidence=raw["evidence"],
        source=raw["source"].strip(),
        object_center_from_pinch_m=tuple(float(value) for value in raw["object_center_from_pinch_m"]),
        registration_evidence=raw["registration_evidence"],
        grasp_rotation_base=tuple(tuple(float(value) for value in row)
                                  for row in raw["grasp_rotation_base"]),
    )
