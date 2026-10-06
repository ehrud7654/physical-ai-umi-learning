"""Offline robot-time preflight for a trained relative UMI action-chunk policy.

This tool never writes to a motor.  It runs real validation observations through
the checkpoint, anchors every predicted chunk at one explicitly registered
SO-101 pose, then applies the production decoding boundary and MuJoCo IK.  The
result separates geometric rejection from commanded velocity/acceleration
rejection before any camera-matched rollout is attempted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data.relative_chunk_dataset import RelativeChunkDataset
from policy.relative_chunk_bc import RelativeChunkBCPolicy
from tools.umi_mujoco import (
    DEFAULT_CONFIG,
    DEFAULT_SCENE,
    MujocoIK,
    build_model,
    load_config,
)
from tools.run_umi_regression import is_shared_gpu_server
from umi.relative_robot_preflight import (
    FixedActionPolicy,
    arm_ranges,
    episode_anchors,
    evaluate_chunk,
    real_limits,
)


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CONFIG = AI_ROOT / "configs" / "real" / "so101_ver1.json"
DEFAULT_REGISTRATION = (
    AI_ROOT / "configs" / "real" / "umi_so101_registration_provisional.json"
)


def _registration(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PROVISIONAL_NOT_PHYSICAL_CALIBRATION":
        raise ValueError(
            "registration must explicitly declare "
            "PROVISIONAL_NOT_PHYSICAL_CALIBRATION")
    arm = np.asarray(payload.get("start_arm_rad"), dtype=float)
    object_xyz = np.asarray(payload.get("object_xyz_m"), dtype=float)
    if (arm.shape != (5,) or object_xyz.shape != (3,)
            or not np.isfinite(arm).all() or not np.isfinite(object_xyz).all()):
        raise ValueError("registration must contain finite start_arm_rad[5] and object_xyz_m[3]")
    return arm, object_xyz, payload


def _balanced_indices(dataset: RelativeChunkDataset, episode_ids: list[str],
                      count: int, seed: int) -> list[int]:
    wanted = set(episode_ids)
    groups = [
        indices for episode, indices in zip(
            dataset.episode_ids, dataset.episode_sample_indices)
        if episode in wanted
    ]
    if not groups:
        raise ValueError("none of the requested episodes exists in the dataset")
    rng = np.random.default_rng(seed)
    each = max(1, int(np.ceil(count / len(groups))))
    selected: list[int] = []
    for indices in groups:
        take = min(each, len(indices))
        selected.extend(int(value) for value in rng.choice(indices, take, replace=False))
    rng.shuffle(selected)
    return selected[:count]


def _percentiles(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None, None, None]
    return [float(value) for value in np.percentile(values, [0, 50, 95, 100])]


def _empty_stats() -> dict[str, Any]:
    return {
        "geometry_accepted": 0,
        "dynamic_accepted": 0,
        "rejects": Counter(),
        "position_errors": [],
        "axis_errors": [],
        "roll_errors": [],
        "arm_speed_peaks": [],
        "arm_accel_peaks": [],
        "gap_speed_peaks": [],
        "gap_accel_peaks": [],
        "required_time_scales": [],
        "scheduled_dynamic_accepted": 0,
        "scheduled_arm_speed_peaks": [],
        "scheduled_arm_accel_peaks": [],
        "scheduled_gap_speed_peaks": [],
        "scheduled_gap_accel_peaks": [],
        "examples": [],
    }


def _accumulate(stats: dict[str, Any], outcome: dict[str, Any],
                identity: dict[str, Any]) -> None:
    if outcome["result"] == "geometry_reject":
        stats["rejects"][outcome["reason"]] += 1
    else:
        stats["geometry_accepted"] += 1
        stats["position_errors"].extend(outcome["position_errors"])
        stats["axis_errors"].extend(outcome["axis_errors"])
        stats["roll_errors"].extend(outcome["roll_errors"])
        peak_keys = {
            "arm_speed_peak_rad_s": "arm_speed_peaks",
            "arm_accel_peak_rad_s2": "arm_accel_peaks",
            "gap_speed_peak_m_s": "gap_speed_peaks",
            "gap_accel_peak_m_s2": "gap_accel_peaks",
            "required_time_scale": "required_time_scales",
        }
        for source, destination in peak_keys.items():
            stats[destination].append(outcome[source])
        scheduled_peak_keys = {
            "scheduled_arm_speed_peak_rad_s": "scheduled_arm_speed_peaks",
            "scheduled_arm_accel_peak_rad_s2": "scheduled_arm_accel_peaks",
            "scheduled_gap_speed_peak_m_s": "scheduled_gap_speed_peaks",
            "scheduled_gap_accel_peak_m_s2": "scheduled_gap_accel_peaks",
        }
        for source, destination in scheduled_peak_keys.items():
            stats[destination].append(outcome[source])
        if not outcome["scheduled_violations"]:
            stats["scheduled_dynamic_accepted"] += 1
        if outcome["violations"]:
            stats["rejects"].update(outcome["violations"])
        else:
            stats["dynamic_accepted"] += 1
    if len(stats["examples"]) < 8:
        stats["examples"].append({**identity, **{
            key: value for key, value in outcome.items()
            if key not in ("position_errors", "axis_errors", "roll_errors")
        }})


def _summary(stats: dict[str, Any], total: int) -> dict[str, Any]:
    return {
        "geometry": {
            "accepted_chunks": stats["geometry_accepted"],
            "accepted_fraction": stats["geometry_accepted"] / total if total else 0.0,
            "position_error_m_p0_p50_p95_max": _percentiles(stats["position_errors"]),
            "axis_error_deg_p0_p50_p95_max": _percentiles(stats["axis_errors"]),
            "roll_error_deg_p0_p50_p95_max": _percentiles(stats["roll_errors"]),
        },
        "dynamics": {
            "accepted_chunks": stats["dynamic_accepted"],
            "accepted_fraction": stats["dynamic_accepted"] / total if total else 0.0,
            "arm_speed_peak_rad_s_p0_p50_p95_max": _percentiles(
                stats["arm_speed_peaks"]),
            "arm_accel_peak_rad_s2_p0_p50_p95_max": _percentiles(
                stats["arm_accel_peaks"]),
            "gap_speed_peak_m_s_p0_p50_p95_max": _percentiles(
                stats["gap_speed_peaks"]),
            "gap_accel_peak_m_s2_p0_p50_p95_max": _percentiles(
                stats["gap_accel_peaks"]),
            "required_uniform_time_scale_p0_p50_p95_max": _percentiles(
                stats["required_time_scales"]),
            "time_scale_semantics": (
                "robot-time duration multiplier; source timestamps and training "
                "labels are not modified"
            ),
            "after_uniform_retime": {
                "accepted_chunks": stats["scheduled_dynamic_accepted"],
                "accepted_fraction": (
                    stats["scheduled_dynamic_accepted"] / total if total else 0.0),
                "arm_speed_peak_rad_s_p0_p50_p95_max": _percentiles(
                    stats["scheduled_arm_speed_peaks"]),
                "arm_accel_peak_rad_s2_p0_p50_p95_max": _percentiles(
                    stats["scheduled_arm_accel_peaks"]),
                "gap_speed_peak_m_s_p0_p50_p95_max": _percentiles(
                    stats["scheduled_gap_speed_peaks"]),
                "gap_accel_peak_m_s2_p0_p50_p95_max": _percentiles(
                    stats["scheduled_gap_accel_peaks"]),
                "semantics": (
                    "same waypoints; only cumulative robot execution timestamps "
                    "are stretched uniformly"
                ),
            },
        },
        "rejections": dict(sorted(stats["rejects"].items())),
        "examples": stats["examples"],
    }


def main() -> int:
    if is_shared_gpu_server():
        raise SystemExit(
            "Shared GPU server is training-only; run policy preflight locally")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--real-config", type=Path, default=DEFAULT_REAL_CONFIG)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")

    policy = RelativeChunkBCPolicy(args.policy_ckpt, args.device)
    data_config = policy.meta["config"]["data"]
    dataset = RelativeChunkDataset(
        args.data, image_mean=float(data_config["image_mean"]),
        image_std=float(data_config["image_std"]))
    if float(policy.meta["rate_hz"]) != float(dataset.rate_hz):
        raise ValueError("checkpoint and dataset rates differ")
    val_episodes = [str(value) for value in policy.meta.get("val_episodes", [])]
    if not val_episodes:
        raise ValueError("checkpoint has no recorded validation episode split")
    indices = _balanced_indices(
        dataset, val_episodes, min(args.samples, len(dataset)), args.seed)

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg)
    arm_start, object_xyz, registration = _registration(args.registration)
    real = real_limits(args.real_config)
    ranges = arm_ranges(real)
    if np.any(arm_start < ranges[:, 0]) or np.any(arm_start > ranges[:, 1]):
        raise ValueError("registered start pose exceeds real arm limits")

    rate_hz = float(dataset.rate_hz)
    dt = 1.0 / rate_hz
    max_arm_speed = float(real["max_speed_rad_s"])
    max_arm_accel = float(real["max_accel_rad_s2"])
    max_gap_speed = float(real["max_gap_speed_m_s"])
    max_gap_accel = float(real["max_gap_accel_m_s2"])
    max_gap = float(real["max_gap_m"])
    curve = cfg["grasp"]["gap_curve"]

    stats = {"policy_prediction": _empty_stats(), "recorded_oracle": _empty_stats()}
    anchor_cache: dict[int, tuple[list[np.ndarray], list[np.ndarray | None]]] = {}
    anchor_unavailable = 0

    for sample_index in indices:
        episode_index, row = dataset.index[sample_index]
        episode = dataset.episodes[episode_index]
        image = episode["image"][row]
        proprio = episode["proprio"][row]
        current_gap = float(proprio[-1, 9])
        observation = {"image": image.copy(), "proprio": proprio.copy()}
        if episode_index not in anchor_cache:
            anchor_cache[episode_index] = episode_anchors(
                episode, model=model, ik=ik, curve=curve,
                arm_start=arm_start, ranges=ranges)
        current_poses, current_arms = anchor_cache[episode_index]
        arm_current = current_arms[row]
        identity = {
            "episode": str(episode["id"]), "row": int(row),
            "source_row": int(episode["source_row"][row]),
            "anchor": "episode_oracle_cumulative_pose",
        }
        if arm_current is None:
            anchor_unavailable += 1
            outcome = {
                "result": "geometry_reject",
                "reason": "current_anchor_ik_unavailable",
                "detail": "recorded cumulative current pose has no valid continuous IK",
            }
            for name in stats:
                _accumulate(stats[name], outcome, identity)
            continue
        candidates = {
            "policy_prediction": policy,
            "recorded_oracle": FixedActionPolicy(episode["action"][row]),
        }
        for name, candidate in candidates.items():
            outcome = evaluate_chunk(
                candidate, observation,
                ik=ik, curve=curve,
                current_pose=current_poses[row], arm_current=arm_current,
                current_gap=current_gap,
                ranges=ranges, gap_max=max_gap, dt=dt,
                max_arm_speed=max_arm_speed, max_arm_accel=max_arm_accel,
                max_gap_speed=max_gap_speed, max_gap_accel=max_gap_accel,
            )
            _accumulate(stats[name], outcome, identity)

    total = len(indices)
    result = {
        "status": "OFFLINE_PREFLIGHT_NOT_ROLLOUT_NOT_HARDWARE_VALIDATION",
        "dataset": str(args.data),
        "checkpoint": str(args.policy_ckpt),
        "samples": total,
        "sampling": "checkpoint_validation_episodes_balanced",
        "validation_episodes": val_episodes,
        "anchor": {
            "registration": str(args.registration),
            "registration_status": registration["status"],
            "start_arm_rad": arm_start.tolist(),
            "object_xyz_m_for_next_render_stage": object_xyz.tolist(),
            "note": (
                "The first sample of each episode uses the registered SO-101 pose. "
                "Later current poses are composed from recorded action[i,0] and "
                "seeded continuously through oracle IK. Object contact and "
                "closed-loop visual feedback are not evaluated here."
            ),
            "unavailable_sample_anchors": anchor_unavailable,
        },
        "limits": {
            "rate_hz": rate_hz,
            "max_arm_speed_rad_s": max_arm_speed,
            "max_arm_accel_rad_s2": max_arm_accel,
            "max_gap_speed_m_s": max_gap_speed,
            "max_gap_accel_m_s2": max_gap_accel,
            "max_position_error_m": 0.005,
            "max_axis_error_deg": 5.0,
            "max_roll_error_deg": 5.0,
            "geometry_jump_budget": (
                "full in-range span; velocity and acceleration are evaluated "
                "separately from source timestamps"
            ),
        },
        "comparison": {
            "policy_prediction": _summary(stats["policy_prediction"], total),
            "recorded_oracle": _summary(stats["recorded_oracle"], total),
            "interpretation": (
                "If the recorded oracle also fails, fix registration, IK or timing "
                "before blaming the learned policy. If the oracle passes but the "
                "policy fails, the learned output is the bottleneck."
            ),
        },
        "next_gate": (
            "Only after geometry and commanded dynamics are measured should a "
            "camera-matched MuJoCo diagnostic video be interpreted."
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
