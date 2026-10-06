"""Compare real v10 H=2 observation motion with a scripted MuJoCo handoff.

This read-only audit separates a possible proprioception/tempo shift from the
image appearance shift. It does not load or evaluate a learned checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import build_model, load_config
from tools.render_relative_chunk_rollout import _audit_file_path, _registration
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp, gap_from_angle
from umi.action_decode import rotation_6d_rows_to_matrix
from umi.relative_dataset import relative_vector
from umi.relative_robot_preflight import matrix_from_fk


def _rotation_deg(rows6d: np.ndarray) -> float:
    rotation = rotation_6d_rows_to_matrix(rows6d)
    return float(np.rad2deg(np.arccos(np.clip(
        (np.trace(rotation) - 1) / 2, -1.0, 1.0))))


def _features(image: np.ndarray, proprio: np.ndarray, *,
              gap_min_m: float, gap_max_m: float) -> dict | None:
    gap = float(proprio[-1, -1])
    if not gap_min_m <= gap <= gap_max_m:
        return None
    return {
        "gap_m": gap,
        "h2_translation_m": float(np.linalg.norm(proprio[0, :3])),
        "h2_rotation_deg": _rotation_deg(proprio[0, 3:9]),
        "h2_rgb_mae_uint8": float(np.mean(np.abs(
            image[0].astype(float) - image[1].astype(float)))),
    }


def _distribution(values: list[float]) -> dict | None:
    array = np.asarray(values, dtype=float)
    return (None if not len(array) else {
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "mean": float(array.mean()),
    })


def _summarize(rows: list[dict]) -> dict:
    result = {"samples": len(rows)}
    for key in ("gap_m", "h2_translation_m", "h2_rotation_deg",
                "h2_rgb_mae_uint8"):
        result[key] = _distribution([row[key] for row in rows])
    return result


def audit(data_root: Path, audit_path: Path, *,
          gap_min_m: float = 0.06, gap_max_m: float = 0.07) -> dict:
    if not 0 <= gap_min_m < gap_max_m <= 0.09:
        raise ValueError("invalid comparison gap band")
    index = json.loads((data_root / "dataset.json").read_text(encoding="utf-8"))
    far = json.loads(audit_path.read_text(encoding="utf-8"))
    if (index.get("schema") != "umi_relative_chunk/0.2.0-provisional"
            or far.get("status") != "POLICY_FREE_FIXED_START_TO_APPROACH_IK_TIMING_AUDIT"):
        raise ValueError("expected real v10 dataset and policy-free far-start audit")
    heldout = set(far["episodes"])
    if heldout - set(index["episodes"]):
        raise ValueError("audit episodes are absent from dataset")
    real_train: list[dict] = []
    real_heldout: list[dict] = []
    for episode in index["episodes"]:
        with np.load(data_root / f"{episode}.npz", allow_pickle=False) as stored:
            image = stored["image"]
            proprio = stored["proprio"]
            if image.shape[0] != proprio.shape[0]:
                raise ValueError(f"image/proprio row mismatch: {episode}")
            destination = real_heldout if episode in heldout else real_train
            for index_in_episode in range(len(image)):
                feature = _features(image[index_in_episode],
                                    proprio[index_in_episode],
                                    gap_min_m=gap_min_m,
                                    gap_max_m=gap_max_m)
                if feature is not None:
                    destination.append({"episode": episode, **feature})

    registration = _registration(_audit_file_path(far["registration"]))
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    model = build_model(cfg)
    ik = MujocoIK(model, cfg)
    curve = cfg["grasp"]["gap_curve"]
    sim: list[dict] = []
    for row in far["rows"]:
        if row.get("diagnostic_motion_pass") is not True:
            raise ValueError(f"not a passing H=2 snapshot: {row['episode']}")
        with np.load(_audit_file_path(row["snapshot_npz"]),
                     allow_pickle=False) as stored:
            image = stored["image"]
            qpos = stored["qpos"]
            timestamps = stored["timestamp"]
        if image.shape != (2, 3, 224, 224) or qpos.shape[0] != 2:
            raise ValueError(f"bad snapshot arrays: {row['episode']}")
        poses = [matrix_from_fk(model, ik, qpos[i, :6]) for i in range(2)]
        gaps = [gap_from_angle(float(qpos[i, 5]), curve) for i in range(2)]
        proprio = np.stack([
            relative_vector(poses[1], poses[0], gaps[0]),
            relative_vector(poses[1], poses[1], gaps[1]),
        ])
        feature = _features(image, proprio, gap_min_m=gap_min_m,
                            gap_max_m=gap_max_m)
        if feature is None:
            raise ValueError(f"sim handoff outside comparison gap band: {row['episode']}")
        sim.append({"episode": row["episode"],
                    "h2_interval_s": float(timestamps[1] - timestamps[0]),
                    **feature})
    if not real_train or not sim:
        raise ValueError("comparison has no real training rows or sim snapshots")
    return {
        "status": "REAL_V10_VS_SCRIPTED_HANDOFF_H2_INPUT_SUPPORT_DIAGNOSTIC",
        "data": str(data_root),
        "far_start_audit": str(audit_path),
        "gap_band_m": [gap_min_m, gap_max_m],
        "real_training": _summarize(real_train),
        "real_heldout": _summarize(real_heldout),
        "sim_handoff": _summarize(sim),
        "real_training_episode_count": len({row["episode"] for row in real_train}),
        "sim_h2_interval_s": _distribution([
            row["h2_interval_s"] for row in sim]),
        "sim_rows": sim,
        "limitations": [
            "Real rows are correlated within episodes; row counts are not independent trials.",
            "The scripted sim handoff is not phase-matched to each real human demonstration.",
            "An input-range mismatch is diagnostic and does not establish causal failure origin.",
            "No learned checkpoint inference or robot actuation occurs.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = audit(args.data, args.audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("gap_band_m", "real_training", "real_heldout",
                       "sim_handoff")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
