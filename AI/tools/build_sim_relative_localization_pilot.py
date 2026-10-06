"""Build a policy-free *candidate* visual-localisation set for v10 chunks.

Each sample has one fixed robot start and an independently displaced object.
The same world displacement is applied to all eight recorded future TCP
positions before re-expressing them in the unchanged start-TCP frame.  Thus
the image and label both vary with object position while proprio cannot reveal
the displacement.  This is a counterfactual label diagnostic, NOT a measured
demonstration or an approved simulation pretraining dataset.  Kinematic and
dynamic grasp feasibility must be gated separately before training use.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from paths import DEFAULT_CONFIG
from sim.mujoco.build_scene import load_config
from sim.mujoco.env import MujocoPickEnv
from tools.estimate_umi_object_geometry import largest_object_bbox
from tools.render_relative_chunk_rollout import (
    _diagnostic_object_xyz,
    _episode_approach_start,
    _registration,
)
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp, gap_from_angle
from umi.arm_preflight import prepare_arm_commands
from umi.convert import invert_gap_curve
from umi.counterfactual import shifted_chunk_from_reference
from umi.relative_dataset import (
    SCHEMA,
    relative_vector,
    write_episode,
)
from umi.relative_robot_preflight import (
    FixedActionPolicy,
    RecordingArmAdapter,
    arm_ranges,
    matrix_from_fk,
    real_limits,
)
from umi.visual_domain import apply_visual_domain, load_visual_domain


def shifted_chunk(action: np.ndarray, start_pose: np.ndarray,
                  delta_world_xyz: np.ndarray) -> np.ndarray:
    """Move target positions once in world space, retaining rotation and gap."""
    return shifted_chunk_from_reference(
        action, reference_pose=start_pose, actual_pose=start_pose,
        delta_world_xyz=delta_world_xyz)


def parse_offset(value: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "offset must be X,Y metres, e.g. --offset-m=-0.01,0"
        ) from exc


def _bbox_xywh(image_chw: np.ndarray) -> list[float] | None:
    image = Image.fromarray(np.transpose(image_chw, (1, 2, 0)))
    component = largest_object_bbox(image)
    if component is None:
        return None
    x0, y0, x1, y1 = component[3]
    return [
        (x0 + x1 + 1) / (2 * image.width),
        (y0 + y1 + 1) / (2 * image.height),
        (x1 - x0 + 1) / image.width,
        (y1 - y0 + 1) / image.height,
    ]


def _input_digest(*, data: Path, registration: Path, visual_domain: Path,
                  config: Path, real_config: Path, episode_ids: list[str],
                  offsets: list[tuple[float, float]]) -> str:
    """Bind a candidate to source bytes and the exact counterfactual offsets."""
    sha = hashlib.sha256()
    sha.update(json.dumps({"episodes": episode_ids, "offsets": offsets},
                          sort_keys=True).encode("utf-8"))
    for path in (registration, visual_domain, config, real_config,
                 *(data / f"{episode_id}.npz" for episode_id in episode_ids)):
        sha.update(path.name.encode("utf-8"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                sha.update(block)
    return sha.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--visual-domain", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--real-config", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "configs/real/so101_ver1.json")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--offset-m", action="append", type=parse_offset, required=True,
                    help="repeat for each object-only XY displacement in metres")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("output already exists")
    offsets = args.offset_m
    if (len(offsets) != len(set(offsets)) or
            any(not np.isfinite(x) or not np.isfinite(y) or
                abs(x) > 0.03 or abs(y) > 0.03 for x, y in offsets)):
        raise SystemExit("offsets must be unique, finite and within +/-30mm per axis")
    if len(args.episodes) != len(set(args.episodes)):
        raise SystemExit("duplicate episode ids")

    registration = _registration(args.registration)
    cfg = copy.deepcopy(load_config(args.config))
    cfg = apply_visual_domain(cfg, load_visual_domain(args.visual_domain))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    cfg["task"]["object"]["half_size_m"] = (
        registration["object_size_m"] / 2).tolist()
    cfg["task"]["object"]["init_pos"] = registration["object_xyz_m"].tolist()
    cfg["task"]["table"]["half_size_m"][:2] = (
        registration["table_half_size_xy_m"].tolist())
    curve = cfg["grasp"]["gap_curve"]
    real = real_limits(args.real_config)
    ranges = arm_ranges(real)

    args.out.mkdir(parents=True, exist_ok=False)
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    samples: list[dict[str, object]] = []
    full_chunk_ik_pass = 0
    with MujocoPickEnv(cfg, render=True, object_jitter_m=0.0) as env:
        ik = MujocoIK(env.model, cfg)
        for episode_id in args.episodes:
            try:
                row, arm, gap = _episode_approach_start(registration, episode_id)
                with np.load(args.data / f"{episode_id}.npz", allow_pickle=False) as stored:
                    source_action = np.asarray(stored["action"][row], dtype=np.float32)
                start_q = np.r_[arm, invert_gap_curve(gap, curve)]
                start_pose = matrix_from_fk(env.model, ik, start_q)
                base_xyz, _ = _diagnostic_object_xyz(
                    registration, episode_id, np.zeros(2))
                for offset_x, offset_y in offsets:
                    delta = np.array([offset_x, offset_y, 0.0], dtype=float)
                    _, object_xyz = _diagnostic_object_xyz(
                        registration, episode_id, delta[:2])
                    offset_slug = (
                        f"x{offset_x:+.3f}_y{offset_y:+.3f}".replace(".", "p")
                    )
                    sample_id = f"{episode_id}_{offset_slug}"
                    obs = env.reset(
                        seed=0, object_xy=tuple(object_xyz[:2]),
                        initial_q_rad=start_q)
                    image = obs.images["cam_wrist"].copy()
                    bbox = _bbox_xywh(image)
                    if bbox is None:
                        rejected[sample_id] = "object_colour_component_not_visible"
                        continue
                    # Both observations are the same static start by design;
                    # only the two timestamps differ. No fictitious motion is
                    # inserted into the proprio history.
                    current = relative_vector(start_pose, start_pose, gap)
                    action = shifted_chunk(source_action, start_pose, delta)
                    solver = RecordingArmAdapter(ik, curve)
                    ik_error = None
                    try:
                        prepare_arm_commands(
                            FixedActionPolicy(action), {},
                            t_current=start_pose,
                            arm_current_rad=arm,
                            gripper_current_m=gap_from_angle(float(start_q[5]), curve),
                            solver=solver,
                            ranges_rad=ranges,
                            gap_range_m=[0.0, float(real["max_gap_m"])],
                            max_step_rad=ranges[:, 1] - ranges[:, 0],
                            max_gap_step_m=float(real["max_gap_m"]),
                            max_position_error_m=0.005,
                            max_axis_error_deg=5.0,
                            max_roll_error_deg=5.0,
                        )
                        full_chunk_ik_pass += 1
                    except ValueError as exc:
                        ik_error = str(exc).splitlines()[0]
                    arrays = {
                        "image": np.stack([image, image])[None].astype(np.uint8),
                        "proprio": np.stack([current, current])[None].astype(np.float32),
                        "action": action[None],
                        "observation_timestamp": np.array([[-0.1, 0.0]], dtype=np.float64),
                        "action_timestamp": np.arange(1, 9, dtype=np.float64)[None] / 10.0,
                        "source_row": np.array([row], dtype=np.int64),
                    }
                    meta = {
                        "schema": SCHEMA,
                        "episode_id": sample_id,
                        "source_episode": episode_id,
                        "source_row": row,
                        "source": "policy_free_mujoco_counterfactual_localisation",
                        "status": "CANDIDATE_NOT_TRAINING_READY",
                        "object_offset_world_xy_m": [offset_x, offset_y],
                        "object_base_xyz_m": base_xyz.tolist(),
                        "object_shifted_xyz_m": object_xyz.tolist(),
                        "object_bbox_xywh_norm": bbox,
                        "robot_start_is_fixed": True,
                        "two_observations_are_static_duplicates": True,
                        "label_translation_shift_is_world_object_offset": True,
                        "full_chunk_ik_validated": ik_error is None,
                        "full_chunk_ik_failure": ik_error,
                        "contact_dynamics_validated": False,
                        "rate_hz": 10.0,
                        "observation_horizon": 2,
                        "action_horizon": 8,
                        "action_semantics": (
                            "same-start relative SE3 target chunk, recorded target "
                            "translation shifted once in world by object offset; "
                            "absolute gap unchanged"
                        ),
                        "success_label": "unverified",
                    }
                    write_episode(args.out / f"{sample_id}.npz", arrays, meta)
                    accepted.append(sample_id)
                    samples.append({
                        "episode": episode_id,
                        "sample_id": sample_id,
                        "offset_xy_m": [offset_x, offset_y],
                        "bbox_xywh_norm": bbox,
                        "first_target_xyz_m": arrays["action"][0, 0, :3].tolist(),
                        "full_chunk_ik_pass": ik_error is None,
                        "full_chunk_ik_failure": ik_error,
                    })
            except (OSError, ValueError, KeyError) as exc:
                rejected[episode_id] = str(exc).splitlines()[0]

    index = {
        "schema": SCHEMA,
        "status": "CANDIDATE_NOT_TRAINING_READY",
        "source": "policy_free_mujoco_counterfactual_localisation",
        "source_real_dataset": str(args.data),
        "source_registration": str(args.registration),
        "source_visual_domain": str(args.visual_domain),
        "source_digest": _input_digest(
            data=args.data, registration=args.registration,
            visual_domain=args.visual_domain, config=args.config,
            real_config=args.real_config, episode_ids=args.episodes,
            offsets=offsets),
        "episodes": accepted,
        "n_episodes": len(accepted),
        "n_rows": len(accepted),
        "rejected": rejected,
        "rate_hz": 10.0,
        "observation_horizon": 2,
        "action_horizon": 8,
        "time_scale": 1.0,
        "robot_base_alignment_applied": True,
        "robot_ik_applied": False,
        "full_chunk_ik_validated": bool(accepted) and full_chunk_ik_pass == len(accepted),
        "full_chunk_ik_pass_count": full_chunk_ik_pass,
        "contact_dynamics_validated": False,
        "notes": (
            "Not a human demonstration or a deployable policy training set. "
            "Full-chunk IK alone does not check dynamics or contact. Do not "
            "combine with real v10 or train until trajectory feasibility and "
            "contact are measured; split by source_episode to avoid leakage."
        ),
    }
    (args.out / "dataset.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "measurement.json").write_text(
        json.dumps({"samples": samples}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": index["status"],
        "episodes": len(accepted),
        "rejected": rejected,
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
