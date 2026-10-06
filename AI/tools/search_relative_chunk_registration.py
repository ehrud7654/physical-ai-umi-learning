"""Search a provisional SO-101 start registration across real UMI episodes.

The recorded oracle is evaluated before the learned policy.  A candidate is
useful only when its sampled oracle chunks are IK-reachable while the inferred
upright snack case is on the table, visible in the initial wrist-camera view,
and grasped around the middle of its body from the side.  Both the horizontal
approach direction and the orthogonal horizontal jaw-closing direction are
explicit task-frame constraints.  This is an offline MuJoCo search, never a
physical robot calibration or motor-safety approval.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from sim.mujoco.build_scene import (
    active_gripper_pad_ids,
    sync_gripper_collision_proxy,
)
from umi.relative_robot_preflight import (
    FixedActionPolicy,
    arm_ranges,
    episode_anchors,
    evaluate_chunk,
    real_limits,
)
from tools.umi_mujoco import (
    DEFAULT_CONFIG,
    DEFAULT_SCENE,
    MujocoArmAdapter,
    MujocoIK,
    apply_kinematic_grasp,
    build_model,
    load_config,
)
from tools.gate_relative_chunk_oracle import (
    _aggregate as aggregate_dynamic_oracle,
    _run_recorded_oracle_episode,
)
from umi.convert import invert_gap_curve
from umi.ik import matrix_to_quat, quat_to_matrix
from umi.relative_dataset import vector_to_transform


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CONFIG = AI_ROOT / "configs" / "real" / "so101_ver1.json"
DEFAULT_REGISTRATION = (
    AI_ROOT / "configs" / "real" / "umi_so101_registration_provisional.json"
)
PROGRESS_SCHEMA = "umi_registration_search_progress/0.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress_signature(*, args: argparse.Namespace,
                        candidates: list[np.ndarray],
                        validation_ids: list[str],
                        screen_ids: list[str]) -> tuple[str, dict[str, Any]]:
    ignored = {"out", "progress", "resume"}
    arguments = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if key not in ignored
    }
    input_paths = {
        "dataset_index": args.data / "dataset.json",
        "registration": args.registration,
        "real_config": args.real_config,
        "sim_config": args.config,
        "scene": args.scene,
        "search_code": Path(__file__),
        "oracle_gate_code": AI_ROOT / "tools" / "gate_relative_chunk_oracle.py",
        "rollout_code": AI_ROOT / "tools" / "render_relative_chunk_rollout.py",
        "ik_code": AI_ROOT / "tools" / "umi_mujoco.py",
    }
    if args.preflight is not None:
        input_paths["preflight"] = args.preflight
    if args.kinematic_grasp is not None:
        input_paths["kinematic_grasp"] = args.kinematic_grasp
    fingerprints = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in input_paths.items()
    }
    payload = {
        "arguments": arguments,
        "inputs": fingerprints,
        "validation_episodes": validation_ids,
        "screen_episodes": screen_ids,
        "candidate_start_arm_rad": [
            np.asarray(value, dtype=float).tolist() for value in candidates
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _load_progress(*, path: Path, resume: bool, signature: str,
                   signature_payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise ValueError(
                f"progress already exists: {path}; pass --resume or choose another path")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema") != PROGRESS_SCHEMA:
            raise ValueError(f"unsupported progress schema: {state.get('schema')!r}")
        if state.get("signature") != signature:
            raise ValueError(
                "progress signature differs from the current data, code, model or arguments")
        return state
    if resume:
        raise ValueError(f"cannot resume because progress is absent: {path}")
    state = {
        "schema": PROGRESS_SCHEMA,
        "signature": signature,
        "signature_payload": signature_payload,
        "stages": {"screen": {}, "refine": {}, "final": {}, "dynamic": {}},
        "complete": False,
    }
    _atomic_json(path, state)
    return state


def _cached_candidate(*, state: dict[str, Any], stage: str,
                      candidate_id: int, arm: np.ndarray) -> dict[str, Any] | None:
    row = state["stages"][stage].get(str(candidate_id))
    if row is None:
        return None
    stored = np.asarray(row.get("start_arm_rad"), dtype=float)
    if stored.shape != (5,) or not np.allclose(
            stored, np.asarray(arm, dtype=float), atol=1e-12, rtol=0):
        raise ValueError(
            f"progress {stage} candidate {candidate_id} has a different arm seed")
    return row


def _save_candidate(*, state: dict[str, Any], progress_path: Path,
                    stage: str, report: dict[str, Any]) -> None:
    candidate_id = str(int(report["candidate_id"]))
    state["stages"][stage][candidate_id] = report
    _atomic_json(progress_path, state)


class NumpyRelativeDataset:
    """Load only geometry arrays; registration search does not need images."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        index = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        self.rate_hz = float(index["rate_hz"])
        self.episodes = []
        for episode_id in index["episodes"]:
            path = self.root / f"{episode_id}.npz"
            with np.load(path, allow_pickle=False) as stored:
                action = stored["action"].copy()
                proprio = stored["proprio"].copy()
            self.episodes.append({
                "id": str(episode_id), "action": action, "proprio": proprio,
            })

    @property
    def episode_ids(self) -> list[str]:
        return [str(episode["id"]) for episode in self.episodes]


def _sample_rows(length: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=int))


def _contact_row(gaps: np.ndarray) -> int:
    """Choose the first near-closed row, not an arbitrary late plateau row."""
    values = np.asarray(gaps, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("episode gaps must be a finite non-empty vector")
    opened = float(np.percentile(values, 90))
    closed = float(np.min(values))
    threshold = closed + 0.10 * max(0.0, opened - closed)
    reached = np.flatnonzero(values <= threshold)
    return int(reached[0]) if len(reached) else int(np.argmin(values))


def _mean_contact_rotation(dataset: NumpyRelativeDataset,
                           episode_indices: list[int]) -> np.ndarray:
    """Average start-to-contact rotations on SO(3), independent of base registration."""
    rotations = []
    for episode_index in episode_indices:
        episode = dataset.episodes[episode_index]
        gaps = np.asarray(episode["proprio"][:, -1, 9], dtype=float)
        contact = _contact_row(gaps)
        pose = np.eye(4)
        for row in range(contact):
            pose = pose @ vector_to_transform(episode["action"][row, 0])
        rotations.append(pose[:3, :3])
    summed = np.sum(rotations, axis=0)
    u, _, vt = np.linalg.svd(summed)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _candidate_starts(current: np.ndarray, ranges: np.ndarray, *, count: int,
                      seed: int, ik: MujocoIK,
                      initial_gap_m: float,
                      targeted_start_rotation: np.ndarray) -> tuple[list[np.ndarray], int]:
    if count <= 0:
        raise ValueError("candidate count must be positive")
    lo, hi = ranges[:, 0], ranges[:, 1]
    margin = 0.08 * (hi - lo)
    lower, upper = lo + margin, hi - margin
    source = np.asarray(current, dtype=float)
    if source.shape != (5,) or not np.isfinite(source).all():
        raise ValueError("source registration arm must be finite shape (5,)")
    if np.any(source < lo) or np.any(source > hi):
        raise ValueError("source registration arm exceeds real joint limits")
    # Preserve the source registration exactly.  It can legitimately sit inside
    # the real limit but outside the random-search safety margin.  Clipping it
    # silently changed the known 150943-successful seed and made the log line
    # 'source registration 1' false.
    centre = np.clip(source, lower, upper)
    rng = np.random.default_rng(seed)
    candidates = [source.copy()]
    if count == 1:
        return candidates, 0

    # Random joint configurations almost never land in the task's required side
    # approach orientation. Build candidates around the declared task-frame
    # approach and jaw axes. Jaw sign is irrelevant for parallel jaws and is
    # handled by the wrist-roll symmetry; approach sign is not interchangeable.
    rotation = np.asarray(targeted_start_rotation, dtype=float)
    target_count = max(0, int(round((count - 1) * 0.65)))
    target_quat = matrix_to_quat(rotation)
    source_q = np.r_[source, invert_gap_curve(
        initial_gap_m, ik.cfg["grasp"]["gap_curve"])]
    source_pos, source_quat = ik.forward_pose(source_q)

    # Registration translation is a transform of the complete recorded
    # trajectory, not an object-only correction.  Preserve the already valid
    # side-grasp rotation and solve Cartesian translations of the start TCP
    # before spending candidates on unconstrained joint-space samples.  The
    # old generator produced only two directed IK seeds out of 32 and filled
    # the remainder with joint-space samples that almost always lost the
    # horizontal side-grasp frame.
    local_translations = [
        np.asarray([dx, dy, dz], dtype=float)
        for dx in (-0.04, -0.08, 0.04)
        for dz in (0.0, -0.015, 0.015)
        for dy in (0.0, -0.04, 0.04)
    ]
    for translation in local_translations:
        target_xyz = source_pos + translation
        solution = ik.solve(
            target_xyz, source_quat, initial_gap_m, q_init=source_q)
        directed_solutions = [solution]
        for delta in (np.pi, -np.pi):
            seed_q = np.asarray(solution.q_rad, dtype=float).copy()
            seed_q[4] += delta
            if ranges[4, 0] <= seed_q[4] <= ranges[4, 1]:
                directed_solutions.append(ik.solve(
                    target_xyz, source_quat, initial_gap_m, q_init=seed_q))

        source_rotation = quat_to_matrix(source_quat)

        def source_frame_score(value: Any) -> tuple[float, float, float]:
            _, achieved_quat = ik.forward_pose(value.q_rad)
            achieved_rotation = quat_to_matrix(achieved_quat)
            return (
                float(np.dot(
                    achieved_rotation[:, 1], source_rotation[:, 1])),
                -float(value.pos_error_m),
                -float(value.axis_error_deg),
            )

        solution = max(directed_solutions, key=source_frame_score)
        _, achieved_quat = ik.forward_pose(solution.q_rad)
        achieved_rotation = quat_to_matrix(achieved_quat)
        directed_up_error = float(np.degrees(np.arccos(np.clip(
            np.dot(achieved_rotation[:, 1], source_rotation[:, 1]),
            -1.0, 1.0))))
        arm = np.asarray(solution.q_rad[:5], dtype=float)
        valid = bool(
            solution.converged
            and solution.pos_error_m <= 0.005
            and solution.axis_error_deg <= 5.0
            and directed_up_error <= 5.0
            and np.all(arm >= ranges[:, 0])
            and np.all(arm <= ranges[:, 1])
        )
        if valid and all(np.linalg.norm(arm - value) > 1e-3
                         for value in candidates):
            candidates.append(arm)
        if len(candidates) >= 1 + target_count:
            break

    if len(candidates) < 1 + target_count:
        directed_source = ik.solve(
            source_pos, target_quat, initial_gap_m, q_init=source_q)
        directed_arm = np.asarray(directed_source.q_rad[:5], dtype=float)
        if (
            directed_source.converged
            and directed_source.pos_error_m <= 0.005
            and directed_source.axis_error_deg <= 5.0
            and np.all(directed_arm >= ranges[:, 0])
            and np.all(directed_arm <= ranges[:, 1])
            and np.linalg.norm(directed_arm - source) > 1e-3
        ):
            candidates.append(directed_arm)
    xs = np.linspace(0.10, 0.32, 4)
    ys = np.linspace(-0.20, 0.20, 5)
    # The standing case centre is 38.35 mm above the table and the configured
    # grasp offset is only a few millimetres.  The old 0.08--0.14 m range could
    # select a pose whose pad merely clipped the case's top edge.  Include the
    # case-centre height directly; collision and IK checks reject unsafe seeds.
    zs = np.linspace(0.035, 0.10, 6)
    targets = [(float(x), float(y), float(z), rotation)
               for z in zs for y in ys for x in xs]
    rng.shuffle(targets)
    for x, y, z, rotation in targets:
        if len(candidates) >= 1 + target_count:
            break
        target_xyz = np.asarray([x, y, z])
        seed_q = np.r_[centre, invert_gap_curve(
            initial_gap_m, ik.cfg["grasp"]["gap_curve"])]
        solution = ik.solve(
            target_xyz, target_quat, initial_gap_m, q_init=seed_q)
        # A parallel jaw is unchanged by 180 degrees about the approach axis,
        # but the attached wrist camera is not.  The generic IK intentionally
        # folds that symmetry.  For registration candidate generation, test
        # both wrist-roll equivalents and retain the one whose canonical +Y
        # points in the requested (directed) up direction.
        directed_solutions = [solution]
        for delta in (np.pi, -np.pi):
            seed_q = np.asarray(solution.q_rad, dtype=float).copy()
            seed_q[4] += delta
            if ranges[4, 0] <= seed_q[4] <= ranges[4, 1]:
                directed_solutions.append(ik.solve(
                    target_xyz, target_quat, initial_gap_m, q_init=seed_q))

        def directed_score(value: Any) -> tuple[float, float, float]:
            _, achieved_quat = ik.forward_pose(value.q_rad)
            achieved_rotation = quat_to_matrix(achieved_quat)
            return (
                float(np.dot(achieved_rotation[:, 1], rotation[:, 1])),
                -float(value.pos_error_m),
                -float(value.axis_error_deg),
            )

        solution = max(directed_solutions, key=directed_score)
        _, achieved_quat = ik.forward_pose(solution.q_rad)
        achieved_rotation = quat_to_matrix(achieved_quat)
        directed_up_error = float(np.degrees(np.arccos(np.clip(
            np.dot(achieved_rotation[:, 1], rotation[:, 1]), -1.0, 1.0))))
        arm = np.asarray(solution.q_rad[:5], dtype=float)
        valid = bool(
            solution.converged
            and solution.pos_error_m <= 0.005
            and solution.axis_error_deg <= 5.0
            and directed_up_error <= 5.0
            and np.all(arm >= ranges[:, 0])
            and np.all(arm <= ranges[:, 1])
        )
        if valid and all(np.linalg.norm(arm - value) > 1e-3
                         for value in candidates):
            candidates.append(arm)
    targeted_count = len(candidates) - 1
    local_count = max(0, int(round((count - len(candidates)) * 0.65)))
    sigma = 0.10 * (hi - lo)
    for _ in range(local_count):
        candidates.append(np.clip(centre + rng.normal(0.0, sigma), lower, upper))
    while len(candidates) < count:
        candidates.append(rng.uniform(lower, upper))
    return candidates, targeted_count


def _require_source_registration(registration: dict[str, Any]) -> None:
    """Accept only source files that explicitly deny physical validation."""
    accepted = {
        "PROVISIONAL_NOT_PHYSICAL_CALIBRATION",
        "DIAGNOSTIC_COLLISION_PROXY_CANDIDATE_NOT_PHYSICAL_VALIDATION",
    }
    status = registration.get("status")
    if status not in accepted:
        raise ValueError(
            "source registration must be an explicitly nonphysical "
            f"diagnostic, got {status!r}"
        )


def _refined_starts(*, seeds: list[np.ndarray], ranges: np.ndarray,
                    existing: list[np.ndarray], count: int,
                    sigma_fraction: float, seed: int) -> list[np.ndarray]:
    """Sample small joint-space neighbourhoods around promising static seeds."""
    if count <= 0:
        return []
    if not seeds:
        raise ValueError("refinement requires at least one seed")
    lo, hi = ranges[:, 0], ranges[:, 1]
    sigma = float(sigma_fraction) * (hi - lo)
    rng = np.random.default_rng(seed)
    refined: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(100, count * 100)
    while len(refined) < count and attempts < max_attempts:
        base = np.asarray(seeds[len(refined) % len(seeds)], dtype=float)
        # Refinement is deliberately local around a verified real-limit seed.
        # Applying the broad-search 8% margin here would move a valid seed by a
        # large amount before the small perturbation is even evaluated.
        arm = np.clip(base + rng.normal(0.0, sigma), lo, hi)
        attempts += 1
        if all(np.linalg.norm(arm - value) > 1e-4
               for value in [*existing, *refined]):
            refined.append(arm)
    if len(refined) != count:
        raise RuntimeError(
            f"could generate only {len(refined)}/{count} unique refinements")
    return refined


def _collision_counts(model: Any, data: Any) -> tuple[int, int]:
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    table_robot = self_collision = 0
    for index in range(data.ncon):
        geom1 = int(data.contact[index].geom1)
        geom2 = int(data.contact[index].geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if table_geom in (geom1, geom2):
            other = geom2 if geom1 == table_geom else geom1
            if other != object_geom and int(model.geom_bodyid[other]) != 0:
                table_robot += 1
        if (body1 != 0 and body2 != 0 and body1 != body2
                and object_geom not in (geom1, geom2)):
            self_collision += 1
    return table_robot, self_collision


def _jaw_object_contact_details(model: Any, data: Any, cfg: dict[str, Any],
                                object_xyz: np.ndarray) -> list[dict[str, Any]]:
    """Describe static contacts between the two active jaws and the object."""
    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    jaw_geoms = active_gripper_pad_ids(model, cfg)
    details = []
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if object_geom not in pair or not pair & jaw_geoms:
            continue
        jaw_geom = next(iter(pair & jaw_geoms))
        position = np.asarray(contact.pos, dtype=float).copy()
        details.append({
            "jaw_geom": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, jaw_geom),
            "position_world_m": position.tolist(),
            "position_minus_object_m": (
                position - np.asarray(object_xyz, dtype=float)).tolist(),
            "distance_m": float(contact.dist),
        })
    return details


def _jaw_object_contacts(model: Any, data: Any, cfg: dict[str, Any]) -> int:
    """Count static contacts between the two active jaw bodies and the object."""
    object_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    return len(_jaw_object_contact_details(
        model, data, cfg, np.asarray(data.xpos[object_body], dtype=float)))


def _set_static_object_pose(model: Any, data: Any,
                            object_xyz: np.ndarray) -> None:
    """Place the free object at one inferred episode position without stepping."""
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free")
    if joint_id < 0:
        raise ValueError("required target_object_free joint is missing")
    qpos_address = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_address:qpos_address + 3] = np.asarray(
        object_xyz, dtype=float)
    data.qpos[qpos_address + 3:qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]


def _planar_contact_offsets(offset: np.ndarray,
                            approach_axis: np.ndarray) -> tuple[float, float]:
    """Signed approach depth and absolute lateral error in the table plane."""
    offset = np.asarray(offset, dtype=float)
    approach_xy = np.asarray(approach_axis, dtype=float)[:2]
    if (offset.shape != (3,) or not np.isfinite(offset).all()
            or approach_xy.shape != (2,) or not np.isfinite(approach_xy).all()):
        raise ValueError("contact offset and approach axis must be finite")
    norm = float(np.linalg.norm(approach_xy))
    if norm <= 1e-9:
        raise ValueError("side-grasp approach axis must have a planar component")
    approach_xy /= norm
    lateral_xy = np.asarray([-approach_xy[1], approach_xy[0]])
    return (
        float(np.dot(offset[:2], approach_xy)),
        abs(float(np.dot(offset[:2], lateral_xy))),
    )


def _canonical_replay_translation(
        *, contact_position_m: np.ndarray,
        desired_world_approach_axis: np.ndarray,
        object_approach_offset_m: float,
        canonical_object_xyz_m: np.ndarray,
        align_contact_to_body_center_3d: bool,
) -> np.ndarray:
    """Return an explicit diagnostic trajectory/object registration shift.

    The default preserves the source vertical trajectory and changes XY only.
    The optional 3-D mode additionally treats the recorded near-closed pinch
    as the source object's body centre (modulo the configured approach offset)
    and aligns it to the canonical on-table object.  That mode isolates missing
    vertical registration; it is not measured calibration.
    """
    contact = np.asarray(contact_position_m, dtype=float)
    approach = np.asarray(desired_world_approach_axis, dtype=float)
    canonical = np.asarray(canonical_object_xyz_m, dtype=float)
    if (contact.shape != (3,) or approach.shape != (3,)
            or canonical.shape != (3,)
            or not np.isfinite(contact).all()
            or not np.isfinite(approach).all()
            or not np.isfinite(canonical).all()
            or not np.isfinite(object_approach_offset_m)):
        raise ValueError("invalid canonical replay alignment inputs")
    inferred_object = (
        contact + approach * float(object_approach_offset_m))
    if not align_contact_to_body_center_3d:
        inferred_object[2] = canonical[2]
    return canonical - inferred_object


def _translated_episode_anchors(
        *, poses: list[np.ndarray], gaps: np.ndarray,
        translation_m: np.ndarray, ik: MujocoIK, curve: Any,
        ranges: np.ndarray, seed_arm: np.ndarray,
        rotation_world_eef: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """Translate the complete trajectory, optionally project rotation, and solve.

    This is a simulation-only workspace diagnostic. It deliberately translates
    every recorded pose by the same vector as the inferred object; translating
    the object alone would manufacture contact and is forbidden.  The optional
    constant rotation is an explicit 5-DoF side-frame retargeting diagnostic;
    it must never silently replace the recorded orientation in training data.
    """
    translation = np.asarray(translation_m, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    seed = np.asarray(seed_arm, dtype=float).copy()
    projected_rotation = None
    if rotation_world_eef is not None:
        projected_rotation = np.asarray(rotation_world_eef, dtype=float)
        if (projected_rotation.shape != (3, 3)
                or not np.isfinite(projected_rotation).all()
                or not np.allclose(
                    projected_rotation.T @ projected_rotation,
                    np.eye(3), rtol=0.0, atol=1e-6)
                or not np.isclose(
                    np.linalg.det(projected_rotation), 1.0,
                    rtol=0.0, atol=1e-6)):
            raise ValueError(
                "projected side-frame rotation must be a proper rotation")
    if (translation.shape != (3,) or not np.isfinite(translation).all()
            or len(poses) != len(gaps) or seed.shape != (5,)):
        raise ValueError("invalid canonical replay translation inputs")
    adapter = MujocoArmAdapter(ik, curve)
    translated: list[np.ndarray] = []
    arms: list[np.ndarray | None] = []
    for pose, gap in zip(poses, gaps, strict=True):
        target = np.asarray(pose, dtype=float).copy()
        target[:3, 3] += translation
        if projected_rotation is not None:
            target[:3, :3] = projected_rotation
        translated.append(target)
        solution = adapter.solve(
            target, seed_rad=seed, gripper_width_m=float(gap))
        valid = bool(
            solution.converged
            and solution.position_error_m <= 0.005
            and solution.axis_error_deg <= 5.0
            and solution.roll_error_deg <= 5.0
            and np.all(solution.positions_rad >= ranges[:, 0])
            and np.all(solution.positions_rad <= ranges[:, 1])
        )
        if valid:
            seed = solution.positions_rad.copy()
            arms.append(seed.copy())
        else:
            arms.append(None)
    return translated, arms


def _apply_grip_preload(gaps: np.ndarray, *, preload_m: float,
                        activation_gap_m: float) -> np.ndarray:
    """Return simulation execution gaps without mutating recorded labels."""
    source = np.asarray(gaps, dtype=float)
    if (source.ndim != 1 or not np.isfinite(source).all()
            or not np.isfinite(preload_m) or preload_m < 0.0
            or not np.isfinite(activation_gap_m)
            or activation_gap_m <= 0.0):
        raise ValueError("invalid diagnostic grip preload inputs")
    executed = source.copy()
    active = source <= float(activation_gap_m)
    executed[active] = np.maximum(0.0, source[active] - float(preload_m))
    return executed


def _approach_start_row(*, contact_row: int, episode_length: int,
                        lookback_rows: int | None) -> int:
    """Choose a fixed pre-contact segment start without success-based search."""
    if (episode_length <= 0 or contact_row < 0
            or contact_row >= episode_length
            or (lookback_rows is not None and lookback_rows < 0)):
        raise ValueError("invalid diagnostic approach segment inputs")
    if lookback_rows is None:
        return 0
    return max(0, contact_row - int(lookback_rows))


def _first_approach_contact(*, model: Any, data: Any, ik: MujocoIK,
                            cfg: dict[str, Any],
                            arms: list[np.ndarray | None], gaps: np.ndarray,
                            start_row: int, contact_row: int,
                            object_xyz: np.ndarray,
                            curve: Any, interpolation_substeps: int,
                            max_pinch_xy_m: float,
                            desired_world_approach_axis: np.ndarray,
                            max_body_depth_offset_m: float) -> dict[str, Any]:
    """Measure the first swept jaw contact before the near-closed pose.

    Dataset rows are only 10 Hz, while MuJoCo executes continuously between
    commands. Interpolating joint targets catches an off-centre collision that
    occurs between recorded rows without a dynamic rollout for every candidate.
    """
    if interpolation_substeps <= 0:
        raise ValueError("interpolation_substeps must be positive")
    if start_row < 0 or start_row > contact_row or contact_row >= len(gaps):
        raise ValueError("invalid first-contact approach segment")
    _set_static_object_pose(model, data, object_xyz)
    previous_q: np.ndarray | None = None
    previous_gap: float | None = None
    checked = 0
    for row in range(start_row, min(contact_row, len(gaps) - 1) + 1):
        arm = arms[row]
        if arm is None:
            previous_q = None
            previous_gap = None
            continue
        gap = float(gaps[row])
        q = np.r_[arm, invert_gap_curve(gap, curve)]
        if previous_q is None:
            samples = [(0, q, gap)]
        else:
            samples = [
                (
                    substep,
                    previous_q + (q - previous_q) *
                    (substep / interpolation_substeps),
                    float(previous_gap + (gap - previous_gap) *
                          (substep / interpolation_substeps)),
                )
                for substep in range(1, interpolation_substeps + 1)
            ]
        for substep, sample_q, sample_gap in samples:
            data.qpos[:6] = sample_q
            sync_gripper_collision_proxy(model, data, cfg)
            mujoco.mj_forward(model, data)
            checked += 1
            contact_details = _jaw_object_contact_details(
                model, data, cfg, object_xyz)
            if not contact_details:
                continue
            pinch_xyz, _ = ik.forward_pose(sample_q)
            offset = np.asarray(pinch_xyz, dtype=float) - object_xyz
            xy = float(np.linalg.norm(offset[:2]))
            approach_offset, lateral_offset = _planar_contact_offsets(
                offset, desired_world_approach_axis)
            contact_centre_offset = np.mean([
                np.asarray(value["position_minus_object_m"], dtype=float)
                for value in contact_details
            ], axis=0)
            body_depth_offset = float(np.dot(
                contact_centre_offset, desired_world_approach_axis))
            return {
                "detected": True,
                "row": row,
                "start_row": start_row,
                "substep": substep,
                "interpolation_substeps": interpolation_substeps,
                "samples_checked": checked,
                "gap_m": sample_gap,
                "contacts": len(contact_details),
                "contact_details": contact_details,
                "pinch_xyz_m": np.asarray(pinch_xyz, dtype=float).tolist(),
                "pinch_minus_object_m": offset.tolist(),
                "pinch_xy_offset_m": xy,
                "pinch_approach_offset_m": approach_offset,
                "pinch_lateral_offset_m": lateral_offset,
                "pinch_3d_offset_m": float(np.linalg.norm(offset)),
                "max_pinch_xy_m": max_pinch_xy_m,
                "max_pinch_lateral_m": max_pinch_xy_m,
                "centering_metric": "planar_perpendicular_to_approach_axis",
                "centered": lateral_offset <= max_pinch_xy_m,
                "legacy_full_xy_centered": xy <= max_pinch_xy_m,
                "contact_centre_minus_object_m": (
                    contact_centre_offset.tolist()),
                "body_depth_offset_m": body_depth_offset,
                "max_body_depth_offset_m": max_body_depth_offset_m,
                "body_depth_centered": (
                    abs(body_depth_offset) <= max_body_depth_offset_m),
            }
        previous_q = q
        previous_gap = gap
    return {
        "detected": False,
        "row": None,
        "start_row": start_row,
        "substep": None,
        "interpolation_substeps": interpolation_substeps,
        "samples_checked": checked,
        "gap_m": None,
        "contacts": 0,
        "contact_details": [],
        "pinch_xyz_m": None,
        "pinch_minus_object_m": None,
        "pinch_xy_offset_m": None,
        "pinch_approach_offset_m": None,
        "pinch_lateral_offset_m": None,
        "pinch_3d_offset_m": None,
        "max_pinch_xy_m": max_pinch_xy_m,
        "max_pinch_lateral_m": max_pinch_xy_m,
        "centering_metric": "planar_perpendicular_to_approach_axis",
        "centered": False,
        "legacy_full_xy_centered": False,
        "contact_centre_minus_object_m": None,
        "body_depth_offset_m": None,
        "max_body_depth_offset_m": max_body_depth_offset_m,
        "body_depth_centered": False,
    }


def _camera_metrics(model: Any, data: Any,
                    object_xyz: np.ndarray) -> dict[str, Any]:
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
    camera_xyz = data.cam_xpos[camera_id].copy()
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    local = camera_rotation.T @ (object_xyz - camera_xyz)
    depth = -float(local[2])
    if depth <= 1e-9:
        return {"visible": False, "centre_error_norm": None,
                "depth_m": depth, "object_camera_xyz_m": local.tolist(),
                "camera_world_xyz_m": camera_xyz.tolist()}
    half_fov = np.tan(np.deg2rad(float(model.cam_fovy[camera_id])) / 2.0)
    uv = np.asarray([local[0], local[1]]) / (depth * half_fov)
    centre_error = float(np.linalg.norm(uv))
    # The requirement here is only that the object appears in the initial view.
    # A 0.30 m far-depth cutoff incorrectly rejected centred objects that were
    # still inside the camera frustum.  Keep a near clearance, but let the
    # frustum test decide the far limit.
    visible = bool(np.max(np.abs(uv)) < 0.85 and centre_error <= 0.60
                   and depth >= 0.12)
    return {"visible": visible, "centre_error_norm": centre_error,
            "depth_m": depth, "object_camera_xyz_m": local.tolist(),
            "camera_world_xyz_m": camera_xyz.tolist()}


def _geom_vertical_interval(model: Any, data: Any,
                            geom_name: str) -> tuple[float, float, float]:
    """World-Z interval and centre of an oriented box collision pad."""
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"required gripper pad is missing: {geom_name}")
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError(f"gripper pad must be a box: {geom_name}")
    centre_z = float(data.geom_xpos[geom_id, 2])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    half_extent_z = float(
        np.dot(np.abs(rotation[2]), model.geom_size[geom_id, :3]))
    return centre_z - half_extent_z, centre_z + half_extent_z, centre_z


def _mean(rows: list[bool]) -> float:
    return float(np.mean(rows)) if rows else 0.0


def _scene_gate(checks: dict[str, bool]) -> tuple[bool, float, list[str]]:
    """Return the hard scene decision and every failed named constraint."""
    if not checks:
        raise ValueError("scene gate requires at least one named constraint")
    normalised = {str(name): bool(value) for name, value in checks.items()}
    failed = [name for name, value in normalised.items() if not value]
    return not failed, float(np.mean(list(normalised.values()))), failed


def _workspace_audit(*, episode_reports: list[dict[str, Any]],
                     table_xy: np.ndarray, table_half_xy: np.ndarray,
                     object_size_m: np.ndarray,
                     object_size_evidence: dict[str, Any] | None = None,
                     edge_margin_m: float) -> dict[str, Any]:
    """Summarise what one common registration can and cannot translate away."""
    if not episode_reports:
        raise ValueError("workspace audit requires per-episode reports")
    object_xyz = np.asarray([
        report["object_xyz_m"] for report in episode_reports
    ], dtype=float)
    displacements = np.asarray([
        report["contact_displacement_from_common_start_m"]
        for report in episode_reports
    ], dtype=float)
    grasp_height = np.asarray([
        report["grasp_height_error_m"] for report in episode_reports
    ], dtype=float)
    contact_target_gaps = np.asarray([
        report["contact_target_gap_m"] for report in episode_reports
    ], dtype=float)
    executed_contact_target_gaps = np.asarray([
        report.get("executed_contact_target_gap_m",
                   report["contact_target_gap_m"])
        for report in episode_reports
    ], dtype=float)
    minimum_recorded_gaps = np.asarray([
        report["minimum_recorded_gap_m"] for report in episode_reports
    ], dtype=float)
    minimum_executed_gaps = np.asarray([
        report.get("minimum_executed_gap_m",
                   report["minimum_recorded_gap_m"])
        for report in episode_reports
    ], dtype=float)
    contact_detected = np.asarray([
        bool(report["first_approach_contact"]["detected"])
        for report in episode_reports
    ], dtype=bool)
    if (object_xyz.shape != (len(episode_reports), 3)
            or displacements.shape != (len(episode_reports), 3)
            or not np.isfinite(object_xyz).all()
            or not np.isfinite(displacements).all()
            or not np.isfinite(grasp_height).all()
            or not np.isfinite(contact_target_gaps).all()
            or not np.isfinite(executed_contact_target_gaps).all()
            or not np.isfinite(minimum_recorded_gaps).all()
            or not np.isfinite(minimum_executed_gaps).all()):
        raise ValueError("workspace audit received invalid episode geometry")

    table_xy = np.asarray(table_xy, dtype=float)
    table_half_xy = np.asarray(table_half_xy, dtype=float)
    object_size_m = np.asarray(object_size_m, dtype=float)
    usable_half = (
        table_half_xy - object_size_m[:2] / 2.0 - float(edge_margin_m))
    lower_per_episode = table_xy - usable_half - object_xyz[:, :2]
    upper_per_episode = table_xy + usable_half - object_xyz[:, :2]
    common_lower = np.max(lower_per_episode, axis=0)
    common_upper = np.min(upper_per_episode, axis=0)
    common_translation_exists = bool(np.all(common_lower <= common_upper))

    failures: Counter[str] = Counter()
    for report in episode_reports:
        failures.update(report.get("scene_failure_reasons", []))

    def distribution(values: np.ndarray) -> dict[str, Any]:
        return {
            "min": np.min(values, axis=0).tolist(),
            "median": np.median(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "span": np.ptp(values, axis=0).tolist(),
        }

    def scalar_distribution(values: np.ndarray) -> dict[str, float] | None:
        if not len(values):
            return None
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }

    return {
        "anchor_rule": (
            "fixed contact-relative lookback per episode; no reachability- or "
            "success-based row search"
            if any(report.get("diagnostic_approach_lookback_rows") is not None
                   for report in episode_reports)
            else "full recorded episode start"
        ),
        "episode_count": len(episode_reports),
        "contact_displacement_from_common_start_m": distribution(displacements),
        "inferred_object_xyz_m": distribution(object_xyz),
        "grasp_height_abs_error_m": {
            "min": float(np.min(grasp_height)),
            "median": float(np.median(grasp_height)),
            "p95": float(np.percentile(grasp_height, 95)),
            "max": float(np.max(grasp_height)),
        },
        "recorded_gap_contact_gate": {
            "object_width_m": float(object_size_m[1]),
            "object_width_status": (
                str(object_size_evidence.get(
                    "grasp_axis_width_source", "configured_without_source"))
                if object_size_evidence else "configured_without_source"
            ),
            "object_width_evidence": object_size_evidence,
            "contact_target_gap_m": scalar_distribution(
                contact_target_gaps),
            "executed_contact_target_gap_m": scalar_distribution(
                executed_contact_target_gaps),
            "minimum_recorded_gap_m": scalar_distribution(
                minimum_recorded_gaps),
            "minimum_executed_gap_m": scalar_distribution(
                minimum_executed_gaps),
            "contact_detected_episodes": int(np.sum(contact_detected)),
            "contact_not_detected_episodes": int(np.sum(~contact_detected)),
            "contact_target_gap_m_when_detected": scalar_distribution(
                contact_target_gaps[contact_detected]),
            "contact_target_gap_m_when_not_detected": scalar_distribution(
                contact_target_gaps[~contact_detected]),
            "executed_contact_target_gap_m_when_detected": (
                scalar_distribution(executed_contact_target_gaps[contact_detected])),
            "executed_contact_target_gap_m_when_not_detected": (
                scalar_distribution(executed_contact_target_gaps[~contact_detected])),
            "interpretation": (
                "A gap/contact split diagnoses consistency between the "
                "recorded gripper gap, configured object width and fitted "
                "pad geometry. Object-width provenance is reported "
                "separately and is not inferred from this diagnostic."
            ),
        },
        "scene_failure_counts": dict(sorted(failures.items())),
        "common_xy_translation_for_table_margin_m": {
            "exists": common_translation_exists,
            "lower": common_lower.tolist(),
            "upper": common_upper.tolist(),
        },
        "interpretation": (
            "A common XY interval only proves table-footprint feasibility; "
            "it does not prove 5-DoF IK reachability or justify an "
            "episode-specific object offset."
        ),
    }


def _table_placement_metrics(*, object_xyz: np.ndarray,
                             object_size_m: np.ndarray,
                             table_xy: np.ndarray,
                             table_half_xy: np.ndarray,
                             edge_margin_m: float) -> dict[str, Any]:
    """Measure full object-footprint clearance from every table edge."""
    object_xyz = np.asarray(object_xyz, dtype=float)
    object_size_m = np.asarray(object_size_m, dtype=float)
    table_xy = np.asarray(table_xy, dtype=float)
    table_half_xy = np.asarray(table_half_xy, dtype=float)
    if (object_xyz.shape != (3,) or object_size_m.shape != (3,)
            or table_xy.shape != (2,) or table_half_xy.shape != (2,)
            or not np.isfinite(object_xyz).all()
            or not np.isfinite(object_size_m).all()
            or not np.isfinite(table_xy).all()
            or not np.isfinite(table_half_xy).all()
            or np.any(object_size_m <= 0.0)
            or np.any(table_half_xy <= 0.0)
            or not np.isfinite(edge_margin_m) or edge_margin_m < 0.0):
        raise ValueError("invalid object/table placement geometry")
    object_half_xy = object_size_m[:2] / 2.0
    centre_offset = np.abs(object_xyz[:2] - table_xy)
    clearance = table_half_xy - centre_offset - object_half_xy
    usable_half = table_half_xy - object_half_xy - float(edge_margin_m)
    if np.any(usable_half <= 0.0):
        centre_distance = float("inf")
    else:
        centre_distance = float(np.linalg.norm(centre_offset / usable_half))
    return {
        "object_on_table_with_margin": bool(
            np.all(clearance >= float(edge_margin_m))),
        "table_edge_clearance_xy_m": clearance.tolist(),
        "minimum_table_edge_clearance_m": float(np.min(clearance)),
        "table_centre_distance_norm": centre_distance,
    }


def _evaluate_candidate(*, candidate_id: int, arm_start: np.ndarray,
                        dataset: NumpyRelativeDataset,
                        episode_indices: list[int], samples_per_episode: int,
                        model: Any, ik: MujocoIK, cfg: dict[str, Any],
                        ranges: np.ndarray, real: dict[str, Any],
                        object_size_m: np.ndarray,
                        object_size_evidence: dict[str, Any] | None,
                        object_edge_margin_m: float,
                        pad_center_margin_m: float,
                        desired_world_jaw_axis: np.ndarray,
                        jaw_yaw_tolerance_deg: float,
                        desired_world_approach_axis: np.ndarray,
                        approach_axis_tolerance_deg: float,
                        desired_world_up_axis: np.ndarray,
                        up_axis_tolerance_deg: float,
                        body_midheight_tolerance_m: float,
                        first_contact_max_pinch_xy_m: float,
                        first_contact_max_body_depth_offset_m: float,
                        approach_contact_substeps: int,
                        object_approach_offset_m: float,
                        diagnostic_grip_preload_m: float = 0.0,
                        diagnostic_grip_preload_activation_gap_m: float = 0.055,
                        diagnostic_approach_lookback_rows: int | None = None,
                        diagnostic_canonical_object_xyz: np.ndarray | None = None,
                        diagnostic_project_side_frame: bool = False,
                        diagnostic_align_contact_3d: bool = False,
                        kinematic_only: bool = False,
                        camera_screen_available: bool = True) -> dict[str, Any]:
    curve = cfg["grasp"]["gap_curve"]
    data = mujoco.MjData(model)
    first_episode = dataset.episodes[episode_indices[0]]
    first_gap = float(first_episode["proprio"][0, -1, 9])
    q_start = np.r_[arm_start, invert_gap_curve(first_gap, curve)]
    data.qpos[:6] = q_start
    sync_gripper_collision_proxy(model, data, cfg)
    mujoco.mj_forward(model, data)
    start_table_contacts, start_self_contacts = _collision_counts(model, data)
    start_pos, start_quat = ik.forward_pose(q_start)
    start_pose = np.eye(4)
    start_pose[:3, :3] = quat_to_matrix(start_quat)
    start_pose[:3, 3] = start_pos

    table = cfg["task"]["table"]
    table_top = float(table["pos"][2]) + float(table["half_size_m"][2])
    object_size_m = np.asarray(object_size_m, dtype=float)
    object_z = table_top + float(object_size_m[2]) / 2.0
    object_bottom = table_top
    object_top = table_top + float(object_size_m[2])
    episode_reports: list[dict[str, Any]] = []
    total_samples = anchor_accepted = geometry_accepted = combined_accepted = 0
    geometry_rejections: Counter[str] = Counter()

    for episode_index in episode_indices:
        episode = dataset.episodes[episode_index]
        source_gaps = np.asarray(
            episode["proprio"][:, -1, 9], dtype=float)
        execution_gaps = _apply_grip_preload(
            source_gaps, preload_m=diagnostic_grip_preload_m,
            activation_gap_m=diagnostic_grip_preload_activation_gap_m)
        poses, arms = episode_anchors(
            episode, model=model, ik=ik, curve=curve,
            arm_start=arm_start, ranges=ranges)
        contact = _contact_row(source_gaps)
        approach_start = _approach_start_row(
            contact_row=contact, episode_length=len(source_gaps),
            lookback_rows=diagnostic_approach_lookback_rows)
        rows = approach_start + _sample_rows(
            len(source_gaps) - approach_start, samples_per_episode)
        contact_pose = poses[contact]
        contact_position = np.asarray(contact_pose[:3, 3], dtype=float).copy()
        object_xyz = contact_position.copy()
        object_xyz += desired_world_approach_axis * object_approach_offset_m
        object_xyz[2] = object_z
        canonical_translation = np.zeros(3, dtype=float)
        if diagnostic_canonical_object_xyz is not None:
            canonical_object = np.asarray(
                diagnostic_canonical_object_xyz, dtype=float)
            if (canonical_object.shape != (3,)
                    or not np.isfinite(canonical_object).all()
                    or not np.isclose(canonical_object[2], object_z,
                                      rtol=0.0, atol=1e-9)):
                raise ValueError(
                    "diagnostic canonical object xyz must be finite and lie "
                    "on the configured table")
            canonical_translation = _canonical_replay_translation(
                contact_position_m=contact_position,
                desired_world_approach_axis=desired_world_approach_axis,
                object_approach_offset_m=object_approach_offset_m,
                canonical_object_xyz_m=canonical_object,
                align_contact_to_body_center_3d=(
                    diagnostic_align_contact_3d),
            )
            projected_rotation = None
            if diagnostic_project_side_frame:
                projected_rotation = np.column_stack([
                    desired_world_jaw_axis,
                    desired_world_up_axis,
                    desired_world_approach_axis,
                ])
            poses, arms = _translated_episode_anchors(
                poses=poses, gaps=execution_gaps,
                translation_m=canonical_translation,
                ik=ik, curve=curve, ranges=ranges,
                seed_arm=arm_start,
                rotation_world_eef=projected_rotation,
            )
            contact_pose = poses[contact]
            object_xyz = canonical_object.copy()
        if kinematic_only:
            first_approach_contact = {
                "available": False,
                "reason": "ver1 collision geometry is absent",
                "detected": None,
                "centered": None,
                "body_depth_centered": None,
                "pinch_xy_offset_m": None,
            }
        else:
            first_approach_contact = _first_approach_contact(
                model=model, data=data, ik=ik, cfg=cfg,
                arms=arms, gaps=execution_gaps,
                start_row=approach_start, contact_row=contact,
                object_xyz=object_xyz, curve=curve,
                interpolation_substeps=approach_contact_substeps,
                max_pinch_xy_m=first_contact_max_pinch_xy_m,
                desired_world_approach_axis=desired_world_approach_axis,
                max_body_depth_offset_m=(
                    first_contact_max_body_depth_offset_m),
            )

        episode_start_ik_available = arms[approach_start] is not None
        episode_q_start = (
            np.r_[arms[approach_start], invert_gap_curve(
                float(execution_gaps[approach_start]), curve)]
            if episode_start_ik_available else q_start)
        data.qpos[:6] = episode_q_start
        sync_gripper_collision_proxy(model, data, cfg)
        mujoco.mj_forward(model, data)
        episode_start_table_contacts, episode_start_self_contacts = (
            _collision_counts(model, data))
        camera = _camera_metrics(model, data, object_xyz)
        camera_world_z = float(data.cam_xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist"), 2]
        )
        camera_above_table = camera_world_z >= table_top
        jaw_vertical = abs(float(contact_pose[2, 0]))
        jaw_axis = np.asarray(contact_pose[:3, 0], dtype=float)
        approach_axis = np.asarray(contact_pose[:3, 2], dtype=float)
        jaw_alignment = abs(float(np.dot(
            jaw_axis, desired_world_jaw_axis)))
        jaw_yaw_error = float(np.degrees(np.arccos(np.clip(
            jaw_alignment, -1.0, 1.0))))
        jaw_yaw_aligned = jaw_yaw_error <= jaw_yaw_tolerance_deg
        approach_alignment = float(np.dot(
            approach_axis, desired_world_approach_axis))
        approach_axis_error = float(np.degrees(np.arccos(np.clip(
            approach_alignment, -1.0, 1.0))))
        approach_axis_aligned = (
            approach_axis_error <= approach_axis_tolerance_deg)
        up_axis = np.asarray(contact_pose[:3, 1], dtype=float)
        up_axis_error = float(np.degrees(np.arccos(np.clip(
            np.dot(up_axis, desired_world_up_axis), -1.0, 1.0))))
        up_axis_aligned = up_axis_error <= up_axis_tolerance_deg
        side_body_grasp = bool(
            jaw_vertical <= 0.25
            and abs(float(approach_axis[2])) <= 0.25
            and approach_axis_aligned
            and up_axis_aligned
        )
        placement = _table_placement_metrics(
            object_xyz=object_xyz,
            object_size_m=object_size_m,
            table_xy=np.asarray(table["pos"][:2], dtype=float),
            table_half_xy=np.asarray(table["half_size_m"][:2], dtype=float),
            edge_margin_m=object_edge_margin_m,
        )
        object_on_table = bool(placement["object_on_table_with_margin"])
        grasp_height_error = abs(float(contact_pose[2, 3]) - object_z)
        pinch_z = float(contact_pose[2, 3])
        body_midheight_aligned = (
            grasp_height_error <= body_midheight_tolerance_m)

        contact_ik_available = arms[contact] is not None
        contact_collision_free = False
        pad_block = cfg.get("gripper_pads") or {}
        pad_names = (
            list(pad_block.get("pad_names", []))
            if pad_block.get("kind") == "symmetric_parallel_jaw_runtime_proxy"
            else [str(value["name"]) for value in pad_block.get("pads", [])]
        )
        pad_vertical_overlaps = {name: 0.0 for name in pad_names}
        pad_center_offsets = {name: None for name in pad_names}
        object_center_inside_both_pads = False
        if contact_ik_available and not kinematic_only:
            data.qpos[:6] = np.r_[
                arms[contact], invert_gap_curve(
                    float(execution_gaps[contact]), curve)]
            sync_gripper_collision_proxy(model, data, cfg)
            mujoco.mj_forward(model, data)
            table_contacts, self_contacts = _collision_counts(model, data)
            contact_collision_free = table_contacts == 0 and self_contacts == 0
            pad_centered = []
            for pad_name in pad_vertical_overlaps:
                low, high, centre = _geom_vertical_interval(
                    model, data, pad_name)
                pad_vertical_overlaps[pad_name] = max(
                    0.0, min(high, object_top) - max(low, object_bottom))
                pad_center_offsets[pad_name] = centre - object_z
                pad_centered.append(
                    low + pad_center_margin_m <= object_z
                    <= high - pad_center_margin_m)
            object_center_inside_both_pads = all(pad_centered)
        else:
            table_contacts = self_contacts = None

        contact_collision_free = bool(contact_collision_free)
        start_collision_free = bool(
            episode_start_ik_available
            and episode_start_table_contacts == 0
            and episode_start_self_contacts == 0)
        # Use this episode's actual (possibly diagnostic-translated) first
        # pose.  The common candidate start pose is a registration seed, not
        # the beginning of the recorded motion.
        contact_displacement = np.asarray(
            contact_pose[:3, 3] - poses[approach_start][:3, 3], dtype=float)
        if kinematic_only:
            scene_checks = {
                "start_ik_available": episode_start_ik_available,
                "upright_body_side_grasp": side_body_grasp,
                "jaw_yaw_aligned": jaw_yaw_aligned,
                "up_axis_aligned": up_axis_aligned,
                "object_on_table_with_margin": object_on_table,
                "body_midheight_aligned": body_midheight_aligned,
                "contact_ik_available": contact_ik_available,
            }
            if camera_screen_available:
                scene_checks = {
                    "initial_object_visible": bool(camera["visible"]),
                    "camera_above_table": camera_above_table,
                    **scene_checks,
                }
        else:
            scene_checks = {
                "start_collision_free": start_collision_free,
                "start_ik_available": episode_start_ik_available,
                "initial_object_visible": bool(camera["visible"]),
                "camera_above_table": camera_above_table,
                "upright_body_side_grasp": side_body_grasp,
                "jaw_yaw_aligned": jaw_yaw_aligned,
                "up_axis_aligned": up_axis_aligned,
                "object_on_table_with_margin": object_on_table,
                "body_midheight_aligned": body_midheight_aligned,
                "contact_ik_available": contact_ik_available,
                "object_center_inside_both_pads_with_margin": (
                    object_center_inside_both_pads),
                "contact_collision_free": contact_collision_free,
                # These swept-contact checks were previously reported and
                # ranked but were not part of scene_constraints_ok.  That let
                # every 0911 candidate miss the declared 8 mm centre gate and
                # still appear static-valid.  Keep the threshold hard.
                "first_approach_contact_detected": bool(
                    first_approach_contact["detected"]),
                "first_approach_contact_centered": bool(
                    first_approach_contact["centered"]),
                "first_approach_contact_body_depth_centered": bool(
                    first_approach_contact["body_depth_centered"]),
            }
        scene_ok, scene_component_fraction, scene_failure_reasons = (
            _scene_gate(scene_checks))
        episode_anchor = episode_geometry = episode_combined = 0
        for row in rows:
            total_samples += 1
            arm_current = arms[int(row)]
            if arm_current is None:
                continue
            anchor_accepted += 1
            episode_anchor += 1
            outcome = evaluate_chunk(
                FixedActionPolicy(episode["action"][int(row)]),
                {},
                ik=ik, curve=curve, current_pose=poses[int(row)],
                arm_current=arm_current,
                current_gap=float(source_gaps[int(row)]), ranges=ranges,
                gap_max=float(real["max_gap_m"]), dt=1.0 / dataset.rate_hz,
                max_arm_speed=float(real["max_speed_rad_s"]),
                max_arm_accel=float(real["max_accel_rad_s2"]),
                max_gap_speed=float(real["max_gap_speed_m_s"]),
                max_gap_accel=float(real["max_gap_accel_m_s2"]),
            )
            if outcome["result"] == "geometry_reject":
                geometry_rejections[outcome["reason"]] += 1
                continue
            geometry_accepted += 1
            episode_geometry += 1
            if scene_ok:
                combined_accepted += 1
                episode_combined += 1

        episode_reports.append({
            "episode": str(episode["id"]),
            "sampled_rows": int(len(rows)),
            "anchor_accepted": episode_anchor,
            "oracle_geometry_accepted": episode_geometry,
            "combined_accepted": episode_combined,
            "contact_row": contact,
            "approach_start_row": approach_start,
            "approach_start_arm_rad": (
                None if arms[approach_start] is None
                else np.asarray(arms[approach_start], dtype=float).tolist()),
            "approach_start_source_gap_m": float(
                source_gaps[approach_start]),
            "approach_start_executed_gap_m": float(
                execution_gaps[approach_start]),
            "diagnostic_approach_lookback_rows": (
                diagnostic_approach_lookback_rows),
            "contact_target_gap_m": float(source_gaps[contact]),
            "executed_contact_target_gap_m": float(execution_gaps[contact]),
            "minimum_recorded_gap_m": float(np.min(source_gaps)),
            "minimum_executed_gap_m": float(np.min(execution_gaps)),
            "diagnostic_grip_preload_m": diagnostic_grip_preload_m,
            "diagnostic_grip_preload_activation_gap_m": (
                diagnostic_grip_preload_activation_gap_m),
            "object_xyz_m": object_xyz.tolist(),
            "diagnostic_canonical_translation_m": (
                canonical_translation.tolist()),
            "diagnostic_canonical_replay": bool(
                diagnostic_canonical_object_xyz is not None),
            "diagnostic_side_frame_projection": bool(
                diagnostic_project_side_frame),
            "diagnostic_contact_to_body_center_3d_alignment": bool(
                diagnostic_align_contact_3d),
            "contact_displacement_from_common_start_m": (
                contact_displacement.tolist()),
            "contact_displacement_norm_m": float(np.linalg.norm(
                contact_displacement)),
            "object_approach_offset_m": object_approach_offset_m,
            "first_view": camera,
            "camera_screen_available": camera_screen_available,
            "camera_above_table": camera_above_table,
            "jaw_vertical_abs": jaw_vertical,
            "approach_vertical_abs": abs(float(approach_axis[2])),
            "approach_axis_world": approach_axis.tolist(),
            "desired_world_approach_axis": desired_world_approach_axis.tolist(),
            "approach_axis_error_deg": approach_axis_error,
            "approach_axis_tolerance_deg": approach_axis_tolerance_deg,
            "approach_axis_aligned": approach_axis_aligned,
            "up_axis_world": up_axis.tolist(),
            "desired_world_up_axis": desired_world_up_axis.tolist(),
            "up_axis_error_deg": up_axis_error,
            "up_axis_tolerance_deg": up_axis_tolerance_deg,
            "up_axis_aligned": up_axis_aligned,
            "upright_body_side_grasp": side_body_grasp,
            "jaw_axis_world": jaw_axis.tolist(),
            "desired_world_jaw_axis": desired_world_jaw_axis.tolist(),
            "jaw_yaw_error_deg": jaw_yaw_error,
            "jaw_yaw_tolerance_deg": jaw_yaw_tolerance_deg,
            "jaw_yaw_aligned": jaw_yaw_aligned,
            "object_on_table": object_on_table,
            "object_edge_margin_required_m": object_edge_margin_m,
            "table_edge_clearance_xy_m": placement[
                "table_edge_clearance_xy_m"],
            "minimum_table_edge_clearance_m": placement[
                "minimum_table_edge_clearance_m"],
            "table_centre_distance_norm": placement[
                "table_centre_distance_norm"],
            "contact_pinch_z_m": pinch_z,
            "grasp_height_error_m": grasp_height_error,
            "body_midheight_tolerance_m": body_midheight_tolerance_m,
            "body_midheight_aligned": body_midheight_aligned,
            "contact_ik_available": contact_ik_available,
            "pad_vertical_overlap_m": pad_vertical_overlaps,
            "pad_center_z_offset_m": pad_center_offsets,
            "pad_center_margin_m": pad_center_margin_m,
            "object_center_inside_both_pads_with_margin": (
                object_center_inside_both_pads),
            "contact_table_contacts": table_contacts,
            "contact_self_contacts": self_contacts,
            "start_ik_available": episode_start_ik_available,
            "start_table_contacts": episode_start_table_contacts,
            "start_self_contacts": episode_start_self_contacts,
            "first_approach_contact": first_approach_contact,
            "scene_checks": scene_checks,
            "scene_failure_reasons": scene_failure_reasons,
            "scene_component_fraction": scene_component_fraction,
            "scene_constraints_ok": scene_ok,
            "screen_mode": (
                "KINEMATIC_ONLY_NO_VER1_COLLISION_OR_OPTICAL_EXTRINSIC"
                if kinematic_only else "LEGACY_DYNAMIC_MODEL"
            ),
        })

    scene_fraction = _mean([
        bool(report["scene_constraints_ok"]) for report in episode_reports])
    scene_component_fraction = float(np.mean([
        report["scene_component_fraction"] for report in episode_reports]))
    side_grasp_fraction = _mean([
        bool(report["upright_body_side_grasp"])
        for report in episode_reports])
    approach_aligned_fraction = _mean([
        bool(report["approach_axis_aligned"])
        for report in episode_reports])
    yaw_aligned_fraction = _mean([
        bool(report["jaw_yaw_aligned"]) for report in episode_reports])
    up_aligned_fraction = _mean([
        bool(report["up_axis_aligned"]) for report in episode_reports])
    visible_fraction = _mean([
        bool(report["first_view"]["visible"]) for report in episode_reports])
    centered_fraction = _mean([
        bool(report["object_center_inside_both_pads_with_margin"])
        for report in episode_reports])
    contact_ik_reports = [
        report for report in episode_reports if report["contact_ik_available"]]
    contact_ik_fraction = len(contact_ik_reports) / len(episode_reports)
    table_margin_fraction = _mean([
        bool(report["object_on_table"]) for report in episode_reports])
    minimum_table_clearance = min(
        float(report["minimum_table_edge_clearance_m"])
        for report in episode_reports)
    mean_table_centre_distance = float(np.mean([
        report["table_centre_distance_norm"] for report in episode_reports]))
    centered_given_contact_ik = _mean([
        bool(report["object_center_inside_both_pads_with_margin"])
        for report in contact_ik_reports])
    collision_free_given_contact_ik = _mean([
        report["contact_table_contacts"] == 0
        and report["contact_self_contacts"] == 0
        for report in contact_ik_reports])
    first_contact_detected_fraction = _mean([
        bool(report["first_approach_contact"]["detected"])
        for report in episode_reports])
    first_contact_centered_fraction = _mean([
        bool(report["first_approach_contact"]["centered"])
        for report in episode_reports])
    first_contact_body_depth_centered_fraction = _mean([
        bool(report["first_approach_contact"]["body_depth_centered"])
        for report in episode_reports])
    first_contact_offsets = [
        float(report["first_approach_contact"]["pinch_xy_offset_m"])
        for report in episode_reports
        if report["first_approach_contact"]["pinch_xy_offset_m"] is not None
    ]
    first_contact_lateral_offsets = [
        float(report["first_approach_contact"]["pinch_lateral_offset_m"])
        for report in episode_reports
        if report["first_approach_contact"]["pinch_lateral_offset_m"] is not None
    ]
    # Prefer an episode that passes both the scene contract and at least one
    # sampled full-chunk geometry check. Choosing only by camera centring can
    # produce a neat view whose recorded tail is not executable.
    representative_pool = [
        report for report in episode_reports
        if report["scene_constraints_ok"] and report["combined_accepted"] > 0]
    if not representative_pool:
        representative_pool = [
            report for report in episode_reports if report["scene_constraints_ok"]]
    if not representative_pool:
        representative_pool = episode_reports
    representative = min(
        representative_pool,
        key=lambda report: (
            -report["combined_accepted"],
            -report["oracle_geometry_accepted"],
            report["table_centre_distance_norm"],
            (report["first_view"]["centre_error_norm"]
             if camera_screen_available
             and report["first_view"]["centre_error_norm"] is not None
             else 0.0),
            report["episode"],
        ),
    )
    workspace_audit = _workspace_audit(
        episode_reports=episode_reports,
        table_xy=np.asarray(table["pos"][:2], dtype=float),
        table_half_xy=np.asarray(table["half_size_m"][:2], dtype=float),
        object_size_m=object_size_m,
        object_size_evidence=object_size_evidence,
        edge_margin_m=object_edge_margin_m,
    )
    return {
        "candidate_id": candidate_id,
        "screen_mode": (
            "KINEMATIC_ONLY_NO_VER1_COLLISION_OR_OPTICAL_EXTRINSIC"
            if kinematic_only else "LEGACY_DYNAMIC_MODEL"
        ),
        "start_arm_rad": arm_start.tolist(),
        "start_pose": start_pose.tolist(),
        "start_table_contacts": start_table_contacts,
        "start_self_contacts": start_self_contacts,
        "sampled_chunks": total_samples,
        "anchor_accepted": anchor_accepted,
        "anchor_accepted_fraction": anchor_accepted / total_samples,
        "oracle_geometry_accepted": geometry_accepted,
        "oracle_geometry_accepted_fraction": geometry_accepted / total_samples,
        "oracle_geometry_rejections": dict(sorted(geometry_rejections.items())),
        "combined_accepted": combined_accepted,
        "combined_accepted_fraction": combined_accepted / total_samples,
        "scene_constraint_episode_fraction": scene_fraction,
        "scene_component_fraction": scene_component_fraction,
        "side_grasp_episode_fraction": side_grasp_fraction,
        "approach_axis_aligned_episode_fraction": approach_aligned_fraction,
        "jaw_yaw_aligned_episode_fraction": yaw_aligned_fraction,
        "up_axis_aligned_episode_fraction": up_aligned_fraction,
        "initial_object_visible_episode_fraction": visible_fraction,
        "pad_centered_episode_fraction": centered_fraction,
        "contact_ik_available_episode_fraction": contact_ik_fraction,
        "object_table_margin_episode_fraction": table_margin_fraction,
        "minimum_table_edge_clearance_m": minimum_table_clearance,
        "mean_table_centre_distance_norm": mean_table_centre_distance,
        "pad_centered_given_contact_ik_fraction": centered_given_contact_ik,
        "collision_free_given_contact_ik_fraction": (
            collision_free_given_contact_ik),
        "first_approach_contact_detected_episode_fraction": (
            first_contact_detected_fraction),
        "first_approach_contact_centered_episode_fraction": (
            first_contact_centered_fraction),
        "first_approach_contact_body_depth_centered_episode_fraction": (
            first_contact_body_depth_centered_fraction),
        "first_approach_contact_pinch_xy_m_mean": (
            float(np.mean(first_contact_offsets)) if first_contact_offsets else None),
        "first_approach_contact_pinch_xy_m_max": (
            float(np.max(first_contact_offsets)) if first_contact_offsets else None),
        "first_approach_contact_lateral_m_mean": (
            float(np.mean(first_contact_lateral_offsets))
            if first_contact_lateral_offsets else None),
        "first_approach_contact_lateral_m_max": (
            float(np.max(first_contact_lateral_offsets))
            if first_contact_lateral_offsets else None),
        "object_approach_offset_m": object_approach_offset_m,
        "representative_episode": representative["episode"],
        "representative_object_xyz_m": representative["object_xyz_m"],
        "workspace_audit": workspace_audit,
        "per_episode": episode_reports,
    }


def _evaluate_dynamic_finalist(
        *, report: dict[str, Any], episode_ids: list[str], data_root: Path,
        output_root: Path, object_size: np.ndarray, table_half_xy: list[float],
        cycles: int, tracking_time_margin: float, grip_preload_m: float,
        grip_preload_activation_gap_m: float, threshold: float,
        config: Path, real_config: Path,
        kinematic_grasp: dict[str, Any] | None) -> dict[str, Any]:
    """Replay fixed challenge episodes so IK rejects and tipping affect rank."""
    candidate_id = int(report["candidate_id"])
    candidate_root = output_root / f"candidate_{candidate_id:03d}"
    retry = 0
    while candidate_root.exists():
        retry += 1
        candidate_root = output_root / (
            f"candidate_{candidate_id:03d}_retry_{retry:03d}")
    candidate_root.mkdir(parents=True, exist_ok=False)
    registration_path = candidate_root / "registration.json"
    registration = {
        "status": "PROVISIONAL_NOT_PHYSICAL_CALIBRATION",
        "source": "registration-search dynamic finalist",
        "purpose": "local recorded-oracle candidate comparison only",
        "start_arm_rad": report["start_arm_rad"],
        "object_xyz_m": report["representative_object_xyz_m"],
        "object_size_m": object_size.tolist(),
        "table_half_size_xy_m": table_half_xy,
        "representative_episode": report["representative_episode"],
        "selected": report,
    }
    if kinematic_grasp is not None:
        registration["kinematic_grasp"] = kinematic_grasp
    registration_path.write_text(
        json.dumps(registration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    static_by_episode = {
        str(row["episode"]): row for row in report["per_episode"]
    }
    rows: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        static = static_by_episode[episode_id]
        if static["scene_constraints_ok"] is not True:
            row = {
                "episode": episode_id,
                "classification": "static_scene_reject",
                "success": False,
                "first_contact_cycle": None,
                "static_scene_component_fraction": static[
                    "scene_component_fraction"],
            }
        else:
            row = _run_recorded_oracle_episode(
                data=data_root, registration_path=registration_path,
                episode=episode_id, report_dir=candidate_root,
                cycles=cycles, tracking_time_margin=tracking_time_margin,
                grip_preload_m=grip_preload_m,
                grip_preload_activation_gap_m=(
                    grip_preload_activation_gap_m),
                config=config, real_config=real_config)
        rows.append(row)
    aggregate = aggregate_dynamic_oracle(rows, threshold=threshold)
    lifts = [
        float(row["max_lift_height_m"])
        for row in rows if row.get("max_lift_height_m") is not None
    ]
    return {
        **{key: value for key, value in aggregate.items() if key != "passed"},
        "passed": bool(aggregate["passed"]),
        "mean_max_lift_height_m": float(np.mean(lifts)) if lifts else 0.0,
        "candidate_registration": str(registration_path),
        "per_episode": rows,
    }


def _dynamic_rank(report: dict[str, Any],
                  static_rank: tuple[Any, ...]) -> tuple[Any, ...]:
    """Rank with non-exclusive dynamic failure counts before static quality."""
    dynamic = report["dynamic_oracle"]
    counts = dynamic["failure_counts"]
    return (
        dynamic["passed"],
        dynamic["success_fraction"],
        -dynamic["runner_errors"],
        -counts.get("ik_reject_after_contact", 0),
        -counts.get("object_tipped", 0),
        -counts.get("object_displaced", 0),
        -counts.get("static_scene_reject", 0),
        dynamic["contacts"],
        dynamic["mean_max_lift_height_m"],
        *static_rank,
    )


def _static_rank(report: dict[str, Any]) -> tuple[Any, ...]:
    """Rank registration candidates without sacrificing task semantics.

    When every cheap-screen candidate has zero fully accepted scenes, table
    centrality used to outrank the required side-grasp axes.  That selected a
    centred but top-down gripper for expensive finalist evaluation.  Keep the
    full acceptance metrics first, then require the four task-frame checks to
    break a zero/zero tie before convenience metrics such as table clearance.
    """
    task_frame_fraction = min(
        report["jaw_yaw_aligned_episode_fraction"],
        report["approach_axis_aligned_episode_fraction"],
        report["up_axis_aligned_episode_fraction"],
        report["side_grasp_episode_fraction"],
    )
    return (
        report["combined_accepted_fraction"],
        report["scene_constraint_episode_fraction"],
        task_frame_fraction,
        report["side_grasp_episode_fraction"],
        report["jaw_yaw_aligned_episode_fraction"],
        report["approach_axis_aligned_episode_fraction"],
        report["up_axis_aligned_episode_fraction"],
        report["first_approach_contact_body_depth_centered_episode_fraction"],
        report["object_table_margin_episode_fraction"],
        -report["mean_table_centre_distance_norm"],
        report["minimum_table_edge_clearance_m"],
        report["pad_centered_episode_fraction"],
        report["contact_ik_available_episode_fraction"],
        report["pad_centered_given_contact_ik_fraction"],
        report["scene_component_fraction"],
        report["oracle_geometry_accepted_fraction"],
        report["anchor_accepted_fraction"],
        -report["start_table_contacts"],
        -report["start_self_contacts"],
        -report["candidate_id"],
    )


def _static_gate_passed(report: dict[str, Any], threshold: float) -> bool:
    """Require the configured episode fraction, not merely one lucky episode."""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("static scene threshold must be within [0, 1]")
    return bool(
        report["combined_accepted_fraction"] > 0.0
        and report["scene_constraint_episode_fraction"] >= threshold
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--preflight", type=Path,
                        help="phase-aligned preflight JSON containing validation episodes")
    parser.add_argument("--episodes", nargs="*",
                        help="explicit episode ids; otherwise use --preflight or all episodes")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--real-config", type=Path, default=DEFAULT_REAL_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--kinematic-grasp", type=Path,
        help=(
            "reviewed kinematic-only TCP/jaw-frame overlay; this does not "
            "replace MuJoCo collision or inertia"
        ),
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument(
        "--candidate-ik-max-iters", type=int, default=80,
        help="maximum IK iterations during start-candidate generation",
    )
    parser.add_argument("--samples-per-episode", type=int, default=2)
    parser.add_argument("--screen-episodes", type=int, default=3,
                        help="episode count for the cheap first-stage screen")
    parser.add_argument(
        "--refine-candidates", type=int, default=0,
        help="additional local candidates sampled around top first-stage seeds",
    )
    parser.add_argument("--refine-seeds", type=int, default=2)
    parser.add_argument("--refine-sigma-fraction", type=float, default=0.01)
    parser.add_argument("--finalists", type=int, default=4,
                        help="top candidates evaluated on every selected episode")
    parser.add_argument(
        "--dynamic-finalists", type=int, default=0,
        help=(
            "top full-screen finalists replayed dynamically before selection; "
            "zero keeps the legacy static-only search"
        ),
    )
    parser.add_argument(
        "--dynamic-episodes", nargs="*",
        help=(
            "fixed challenge episodes for every dynamic finalist; required when "
            "--dynamic-finalists is positive"
        ),
    )
    parser.add_argument("--dynamic-cycles", type=int, default=30)
    parser.add_argument("--dynamic-tracking-time-margin", type=float, default=1.0)
    parser.add_argument("--dynamic-grip-preload-m", type=float, default=0.001)
    parser.add_argument(
        "--dynamic-grip-preload-activation-gap-m", type=float, default=0.055)
    parser.add_argument("--min-dynamic-success-fraction", type=float, default=0.8)
    parser.add_argument(
        "--min-static-scene-fraction", type=float, default=0.8,
        help="minimum validation-episode fraction passing every static scene gate",
    )
    parser.add_argument(
        "--pad-center-margin-m", type=float, default=0.005,
        help=(
            "required clearance between the inferred object centre and either "
            "end of the fixed pad (default: 5 mm)"
        ),
    )
    parser.add_argument(
        "--object-edge-margin-m", type=float, default=0.03,
        help=(
            "minimum clearance from the full object footprint to every table "
            "edge (default: 30 mm)"
        ),
    )
    parser.add_argument(
        "--first-contact-max-pinch-xy-m", type=float, default=0.008,
        help=(
            "provisional maximum planar pinch-centre error perpendicular to "
            "the approach axis at first swept contact (default: 8 mm; option "
            "name retained for CLI compatibility)"
        ),
    )
    parser.add_argument(
        "--first-contact-max-body-depth-offset-m", type=float, default=0.010,
        help=(
            "maximum approach-axis offset of the first pad/object contact "
            "from the upright object's side-body centre (default: 10 mm)"
        ),
    )
    parser.add_argument(
        "--object-approach-offset-m", type=float,
        help=(
            "inferred object-centre offset from the near-closed pinch along "
            "desired_world_approach_axis; defaults to the registration value "
            "or zero"
        ),
    )
    parser.add_argument(
        "--diagnostic-canonical-object-xyz", type=float, nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "simulation-only workspace diagnostic: translate each episode's "
            "complete recorded TCP trajectory and inferred object by the same "
            "vector so the object centre is X Y Z; never a physical "
            "registration or deployment calibration"
        ),
    )
    parser.add_argument(
        "--diagnostic-project-side-frame", action="store_true",
        help=(
            "with --diagnostic-canonical-object-xyz only, preserve recorded "
            "positions/gap/time but replace every TCP rotation with the fixed "
            "desired jaw/up/approach side frame before re-solving IK; this is "
            "a simulation-only retargeting diagnostic, never a source-data "
            "rewrite or physical calibration"
        ),
    )
    parser.add_argument(
        "--diagnostic-align-contact-3d", action="store_true",
        help=(
            "with --diagnostic-canonical-object-xyz only, translate the whole "
            "trajectory in XYZ so its recorded near-closed pinch aligns with "
            "the canonical object's body centre (plus the configured approach "
            "offset); isolates missing vertical registration and is never a "
            "measured physical calibration"
        ),
    )
    parser.add_argument(
        "--diagnostic-grip-preload-m", type=float, default=0.0,
        help=(
            "simulation-only under-closure applied to static canonical "
            "replay; source gap labels remain unchanged"
        ),
    )
    parser.add_argument(
        "--diagnostic-grip-preload-activation-gap-m",
        type=float, default=0.055,
        help="apply diagnostic preload only at or below this source gap",
    )
    parser.add_argument(
        "--diagnostic-approach-lookback-rows", type=int,
        help=(
            "simulation-only fixed segment start: begin this many 10Hz rows "
            "before the recorded near-close row; no reachability-based search"
        ),
    )
    parser.add_argument(
        "--approach-contact-substeps", type=int, default=5,
        help="joint-space interpolation samples per 10 Hz approach interval",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--progress", type=Path,
        help="candidate checkpoint JSON; defaults beside --out",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse candidate stages only when the saved run signature matches",
    )
    args = parser.parse_args()
    if (args.candidates <= 0 or args.candidate_ik_max_iters <= 0
            or args.samples_per_episode <= 0
            or args.screen_episodes <= 0 or args.finalists <= 0
            or args.finalists > args.candidates + args.refine_candidates
            or args.refine_candidates < 0 or args.refine_seeds <= 0
            or args.refine_sigma_fraction <= 0.0
            or args.dynamic_finalists < 0
            or args.dynamic_finalists > args.finalists
            or args.dynamic_cycles <= 0
            or args.pad_center_margin_m < 0.0
            or args.object_edge_margin_m < 0.0
            or args.first_contact_max_pinch_xy_m <= 0.0
            or args.first_contact_max_body_depth_offset_m <= 0.0
            or args.approach_contact_substeps <= 0
            or args.dynamic_tracking_time_margin < 1.0
            or args.dynamic_grip_preload_m < 0.0
            or args.dynamic_grip_preload_activation_gap_m <= 0.0
            or args.diagnostic_grip_preload_m < 0.0
            or args.diagnostic_grip_preload_activation_gap_m <= 0.0
            or (args.diagnostic_approach_lookback_rows is not None
                and args.diagnostic_approach_lookback_rows < 0)
            or not 0.0 <= args.min_dynamic_success_fraction <= 1.0
            or not 0.0 <= args.min_static_scene_fraction <= 1.0):
        raise SystemExit("candidate and sample counts must be positive")
    if args.dynamic_finalists > 0 and not args.dynamic_episodes:
        raise SystemExit(
            "--dynamic-episodes is required when --dynamic-finalists is positive")
    if args.dynamic_finalists == 0 and args.dynamic_episodes:
        raise SystemExit(
            "--dynamic-episodes requires a positive --dynamic-finalists")
    if (args.diagnostic_canonical_object_xyz is not None
            and (args.candidates != 1 or args.refine_candidates != 0
                 or args.finalists != 1 or args.dynamic_finalists != 0)):
        raise SystemExit(
            "--diagnostic-canonical-object-xyz requires --candidates 1, "
            "--refine-candidates 0, --finalists 1 and "
            "--dynamic-finalists 0; it diagnoses one fixed source seed and "
            "must not be used as registration search")
    if (args.diagnostic_project_side_frame
            and args.diagnostic_canonical_object_xyz is None):
        raise SystemExit(
            "--diagnostic-project-side-frame requires "
            "--diagnostic-canonical-object-xyz")
    if (args.diagnostic_align_contact_3d
            and args.diagnostic_canonical_object_xyz is None):
        raise SystemExit(
            "--diagnostic-align-contact-3d requires "
            "--diagnostic-canonical-object-xyz")
    if (args.diagnostic_grip_preload_m > 0.0
            and args.diagnostic_canonical_object_xyz is None):
        raise SystemExit(
            "--diagnostic-grip-preload-m requires "
            "--diagnostic-canonical-object-xyz")
    if (args.diagnostic_approach_lookback_rows is not None
            and args.diagnostic_canonical_object_xyz is None):
        raise SystemExit(
            "--diagnostic-approach-lookback-rows requires "
            "--diagnostic-canonical-object-xyz")
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    progress_path = (
        args.progress if args.progress is not None
        else args.out.with_name(args.out.stem + ".progress.json")
    )
    if progress_path.resolve() == args.out.resolve():
        raise SystemExit("--progress and --out must be different paths")
    dynamic_output_root = (
        args.out.parent / f"{args.out.stem}_dynamic_finalists"
    )
    if (args.dynamic_finalists > 0 and dynamic_output_root.exists()
            and not args.resume):
        raise SystemExit(
            f"dynamic finalist output already exists: {dynamic_output_root}")

    source_registration = json.loads(
        args.registration.read_text(encoding="utf-8"))
    _require_source_registration(source_registration)
    current = np.asarray(source_registration["start_arm_rad"], dtype=float)
    object_size = np.asarray(source_registration["object_size_m"], dtype=float)
    object_size_evidence = source_registration.get("object_size_evidence")
    if (object_size_evidence is not None
            and not isinstance(object_size_evidence, dict)):
        raise ValueError("registration object_size_evidence must be an object")
    if current.shape != (5,) or object_size.shape != (3,):
        raise ValueError("registration start_arm_rad/object_size_m shape mismatch")
    desired_jaw = np.asarray(
        source_registration.get("desired_world_jaw_axis"), dtype=float)
    if (desired_jaw.shape != (3,) or not np.isfinite(desired_jaw).all()
            or np.linalg.norm(desired_jaw[:2]) <= 1e-9
            or abs(float(desired_jaw[2])) > 1e-6):
        raise ValueError(
            "registration needs a finite horizontal desired_world_jaw_axis")
    desired_jaw = desired_jaw / np.linalg.norm(desired_jaw)
    jaw_yaw_tolerance = float(
        source_registration.get("jaw_yaw_tolerance_deg", 15.0))
    if not 0.0 <= jaw_yaw_tolerance < 90.0:
        raise ValueError("jaw_yaw_tolerance_deg must be within [0, 90)")
    desired_approach = np.asarray(
        source_registration.get("desired_world_approach_axis"), dtype=float)
    if (desired_approach.shape != (3,)
            or not np.isfinite(desired_approach).all()
            or np.linalg.norm(desired_approach[:2]) <= 1e-9
            or abs(float(desired_approach[2])) > 1e-6):
        raise ValueError(
            "registration needs a finite horizontal "
            "desired_world_approach_axis")
    desired_approach = desired_approach / np.linalg.norm(desired_approach)
    if abs(float(np.dot(desired_jaw, desired_approach))) > 1e-6:
        raise ValueError("desired jaw and approach axes must be orthogonal")
    approach_axis_tolerance = float(
        source_registration.get("approach_axis_tolerance_deg", 15.0))
    if not 0.0 <= approach_axis_tolerance < 90.0:
        raise ValueError("approach_axis_tolerance_deg must be within [0, 90)")
    desired_up = np.asarray(
        source_registration.get("desired_world_up_axis"), dtype=float)
    if (desired_up.shape != (3,) or not np.isfinite(desired_up).all()
            or np.linalg.norm(desired_up) <= 1e-9):
        raise ValueError("registration needs a finite desired_world_up_axis")
    desired_up = desired_up / np.linalg.norm(desired_up)
    if (abs(float(np.dot(desired_up, desired_jaw))) > 1e-6
            or abs(float(np.dot(desired_up, desired_approach))) > 1e-6):
        raise ValueError("desired up axis must be orthogonal to jaw and approach")
    up_axis_tolerance = float(
        source_registration.get("up_axis_tolerance_deg", 15.0))
    if not 0.0 <= up_axis_tolerance < 90.0:
        raise ValueError("up_axis_tolerance_deg must be within [0, 90)")
    body_midheight_tolerance = float(
        source_registration.get("body_midheight_tolerance_m", 0.012))
    if body_midheight_tolerance <= 0.0:
        raise ValueError("body_midheight_tolerance_m must be positive")
    object_approach_offset = float(
        source_registration.get("object_approach_offset_m", 0.0)
        if args.object_approach_offset_m is None
        else args.object_approach_offset_m
    )
    if not np.isfinite(object_approach_offset):
        raise ValueError("object_approach_offset_m must be finite")

    dataset = NumpyRelativeDataset(args.data)
    if args.episodes:
        validation_ids = [str(value) for value in args.episodes]
        source = "explicit --episodes"
    elif args.preflight is not None:
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        validation_ids = [str(value) for value in preflight["validation_episodes"]]
        source = str(args.preflight)
    else:
        validation_ids = dataset.episode_ids
        source = "all dataset episodes"
    episode_by_id = {episode_id: index for index, episode_id
                     in enumerate(dataset.episode_ids)}
    missing = sorted(set(validation_ids) - set(episode_by_id))
    if missing:
        raise ValueError(f"validation episodes missing from dataset: {missing}")
    episode_indices = [episode_by_id[value] for value in validation_ids]
    dynamic_episode_ids = [str(value) for value in (args.dynamic_episodes or [])]
    if len(set(dynamic_episode_ids)) != len(dynamic_episode_ids):
        raise ValueError("dynamic episode ids must be unique")
    dynamic_missing = sorted(set(dynamic_episode_ids) - set(validation_ids))
    if dynamic_missing:
        raise ValueError(
            "dynamic episodes must be included in the selected validation set: "
            + ", ".join(dynamic_missing))

    mean_contact_rotation = _mean_contact_rotation(dataset, episode_indices)
    lateral = np.cross(desired_approach, desired_jaw)
    desired_contact_rotation = np.column_stack([
        desired_jaw, lateral, desired_approach])
    targeted_start_rotation = (
        desired_contact_rotation @ mean_contact_rotation.T)

    cfg = load_config(args.config)
    kinematic_grasp = None
    if args.kinematic_grasp is not None:
        kinematic_grasp = json.loads(
            args.kinematic_grasp.read_text(encoding="utf-8"))
        apply_kinematic_grasp(cfg, kinematic_grasp)
        diagnostic_proxy_ready = bool(
            kinematic_grasp.get("collision_proxy", {}).get(
                "diagnostic_dynamic_ready") is True
        )
        if args.dynamic_finalists > 0 and not diagnostic_proxy_ready:
            raise ValueError(
                "dynamic finalists are forbidden with a kinematic-only ver1 "
                "overlay because its collision/inertia model is absent")
    else:
        diagnostic_proxy_ready = False
    kinematic_only = bool(kinematic_grasp is not None and not diagnostic_proxy_ready)
    camera_screen_available = bool(
        kinematic_grasp is None
        or kinematic_grasp.get("camera_optical_extrinsic_present") is True
    )
    cfg["task"]["object"]["half_size_m"] = (object_size / 2.0).tolist()
    if "table_half_size_xy_m" in source_registration:
        cfg["task"]["table"]["half_size_m"][:2] = [
            float(value) for value in source_registration["table_half_size_xy_m"]]
    table_top = (float(cfg["task"]["table"]["pos"][2])
                 + float(cfg["task"]["table"]["half_size_m"][2]))
    object_z = table_top + float(object_size[2]) / 2.0
    cfg["task"]["object"]["init_pos"][2] = object_z
    diagnostic_canonical_object_xyz = None
    if args.diagnostic_canonical_object_xyz is not None:
        diagnostic_canonical_object_xyz = np.asarray(
            args.diagnostic_canonical_object_xyz, dtype=float)
        if (not np.isfinite(diagnostic_canonical_object_xyz).all()
                or not np.isclose(
                    diagnostic_canonical_object_xyz[2], object_z,
                    rtol=0.0, atol=1e-9)):
            raise ValueError(
                "diagnostic canonical Z must equal the configured object "
                f"centre height {object_z:.9f} m")
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg)
    full_ik_max_iters = ik.max_iters
    ik.max_iters = args.candidate_ik_max_iters
    real = real_limits(args.real_config)
    ranges = arm_ranges(real)
    first_selected_episode = dataset.episodes[episode_indices[0]]
    initial_gap = float(first_selected_episode["proprio"][0, -1, 9])
    candidates, targeted_candidate_count = _candidate_starts(
        current, ranges, count=args.candidates, seed=args.seed, ik=ik,
        initial_gap_m=initial_gap,
        targeted_start_rotation=targeted_start_rotation)
    ik.max_iters = full_ik_max_iters
    print(
        f"candidate generation: side-grasp IK {targeted_candidate_count}, "
        f"random/local {len(candidates) - targeted_candidate_count - 1}, "
        "source registration 1",
        flush=True,
    )
    screen_count = min(args.screen_episodes, len(episode_indices))
    screen_positions = np.unique(np.linspace(
        0, len(episode_indices) - 1, screen_count, dtype=int))
    screen_indices = [episode_indices[int(value)] for value in screen_positions]
    # Fixed dynamic challenge episodes must also participate in the cheap
    # screen. Otherwise a candidate can be discarded on unrelated evenly
    # spaced episodes before the failures we are trying to improve are seen.
    for episode_id in dynamic_episode_ids:
        episode_index = episode_by_id[episode_id]
        if episode_index not in screen_indices:
            screen_indices.append(episode_index)
    screen_ids = [dataset.episode_ids[index] for index in screen_indices]
    signature, signature_payload = _progress_signature(
        args=args, candidates=candidates, validation_ids=validation_ids,
        screen_ids=screen_ids,
    )
    progress = _load_progress(
        path=progress_path, resume=args.resume, signature=signature,
        signature_payload=signature_payload,
    )

    rank = _static_rank

    screen_reports = []
    for candidate_id, arm in enumerate(candidates):
        report = _cached_candidate(
            state=progress, stage="screen", candidate_id=candidate_id, arm=arm)
        resumed = report is not None
        if report is None:
            report = _evaluate_candidate(
                candidate_id=candidate_id, arm_start=arm,
                dataset=dataset, episode_indices=screen_indices,
                samples_per_episode=1,
                model=model, ik=ik, cfg=cfg, ranges=ranges, real=real,
                object_size_m=object_size,
                object_size_evidence=object_size_evidence,
                object_edge_margin_m=args.object_edge_margin_m,
                pad_center_margin_m=args.pad_center_margin_m,
                desired_world_jaw_axis=desired_jaw,
                jaw_yaw_tolerance_deg=jaw_yaw_tolerance,
                desired_world_approach_axis=desired_approach,
                approach_axis_tolerance_deg=approach_axis_tolerance,
                desired_world_up_axis=desired_up,
                up_axis_tolerance_deg=up_axis_tolerance,
                body_midheight_tolerance_m=body_midheight_tolerance,
                first_contact_max_pinch_xy_m=(
                    args.first_contact_max_pinch_xy_m),
                first_contact_max_body_depth_offset_m=(
                    args.first_contact_max_body_depth_offset_m),
                approach_contact_substeps=args.approach_contact_substeps,
                object_approach_offset_m=object_approach_offset,
                diagnostic_grip_preload_m=args.diagnostic_grip_preload_m,
                diagnostic_grip_preload_activation_gap_m=(
                    args.diagnostic_grip_preload_activation_gap_m),
                diagnostic_approach_lookback_rows=(
                    args.diagnostic_approach_lookback_rows),
                diagnostic_canonical_object_xyz=(
                    diagnostic_canonical_object_xyz),
                diagnostic_project_side_frame=(
                    args.diagnostic_project_side_frame),
                diagnostic_align_contact_3d=(
                    args.diagnostic_align_contact_3d),
                kinematic_only=kinematic_only,
                camera_screen_available=camera_screen_available,
            )
            _save_candidate(
                state=progress, progress_path=progress_path,
                stage="screen", report=report)
        screen_reports.append(report)
        print(
            f"screen {candidate_id + 1:3d}/{len(candidates)} "
            f"{'[resume] ' if resumed else ''}"
            f"combined={report['combined_accepted_fraction']:.1%} "
            f"geometry={report['oracle_geometry_accepted_fraction']:.1%} "
            f"scene={report['scene_constraint_episode_fraction']:.1%} "
            f"side-contact-centred="
            f"{report['first_approach_contact_body_depth_centered_episode_fraction']:.1%} "
            f"side={report['side_grasp_episode_fraction']:.1%} "
            f"partial={report['scene_component_fraction']:.1%}",
            flush=True,
        )

    if args.refine_candidates > 0:
        seed_reports = sorted(screen_reports, key=rank, reverse=True)[
            :min(args.refine_seeds, len(screen_reports))]
        refinements = _refined_starts(
            seeds=[np.asarray(report["start_arm_rad"], dtype=float)
                   for report in seed_reports],
            ranges=ranges, existing=candidates,
            count=args.refine_candidates,
            sigma_fraction=args.refine_sigma_fraction,
            seed=args.seed + 1,
        )
        for refinement_index, arm in enumerate(refinements, start=1):
            candidate_id = len(candidates)
            candidates.append(arm)
            report = _cached_candidate(
                state=progress, stage="refine", candidate_id=candidate_id,
                arm=arm)
            resumed = report is not None
            if report is None:
                report = _evaluate_candidate(
                    candidate_id=candidate_id, arm_start=arm,
                    dataset=dataset, episode_indices=screen_indices,
                    samples_per_episode=1,
                    model=model, ik=ik, cfg=cfg, ranges=ranges, real=real,
                    object_size_m=object_size,
                    object_size_evidence=object_size_evidence,
                    object_edge_margin_m=args.object_edge_margin_m,
                    pad_center_margin_m=args.pad_center_margin_m,
                    desired_world_jaw_axis=desired_jaw,
                    jaw_yaw_tolerance_deg=jaw_yaw_tolerance,
                    desired_world_approach_axis=desired_approach,
                    approach_axis_tolerance_deg=approach_axis_tolerance,
                    desired_world_up_axis=desired_up,
                    up_axis_tolerance_deg=up_axis_tolerance,
                    body_midheight_tolerance_m=body_midheight_tolerance,
                    first_contact_max_pinch_xy_m=(
                        args.first_contact_max_pinch_xy_m),
                    first_contact_max_body_depth_offset_m=(
                        args.first_contact_max_body_depth_offset_m),
                    approach_contact_substeps=args.approach_contact_substeps,
                    object_approach_offset_m=object_approach_offset,
                    diagnostic_grip_preload_m=args.diagnostic_grip_preload_m,
                    diagnostic_grip_preload_activation_gap_m=(
                        args.diagnostic_grip_preload_activation_gap_m),
                    diagnostic_approach_lookback_rows=(
                        args.diagnostic_approach_lookback_rows),
                    diagnostic_canonical_object_xyz=(
                        diagnostic_canonical_object_xyz),
                    diagnostic_project_side_frame=(
                        args.diagnostic_project_side_frame),
                    diagnostic_align_contact_3d=(
                        args.diagnostic_align_contact_3d),
                    kinematic_only=kinematic_only,
                    camera_screen_available=camera_screen_available,
                )
                _save_candidate(
                    state=progress, progress_path=progress_path,
                    stage="refine", report=report)
            screen_reports.append(report)
            print(
                f"refine {refinement_index:3d}/{len(refinements)} "
                f"candidate={candidate_id} "
                f"{'[resume] ' if resumed else ''}"
                f"combined={report['combined_accepted_fraction']:.1%} "
                f"geometry={report['oracle_geometry_accepted_fraction']:.1%} "
                f"scene={report['scene_constraint_episode_fraction']:.1%} "
                f"side-contact-centred="
                f"{report['first_approach_contact_body_depth_centered_episode_fraction']:.1%} "
                f"side={report['side_grasp_episode_fraction']:.1%} "
                f"partial={report['scene_component_fraction']:.1%}",
                flush=True,
            )

    screen_reports.sort(key=rank, reverse=True)
    finalist_ids = [int(report["candidate_id"])
                    for report in screen_reports[:args.finalists]]
    # Always retain the exact source registration as a measured baseline.  It
    # may fail the cheap screen while remaining the only candidate with a known
    # dynamic success on a challenge episode.
    if args.finalists > 1 and 0 not in finalist_ids:
        finalist_ids[-1] = 0
    reports = []
    for finalist_index, candidate_id in enumerate(finalist_ids):
        arm = candidates[candidate_id]
        report = _cached_candidate(
            state=progress, stage="final", candidate_id=candidate_id, arm=arm)
        resumed = report is not None
        if report is None:
            report = _evaluate_candidate(
                candidate_id=candidate_id, arm_start=arm,
                dataset=dataset, episode_indices=episode_indices,
                samples_per_episode=args.samples_per_episode,
                model=model, ik=ik, cfg=cfg, ranges=ranges, real=real,
                object_size_m=object_size,
                object_size_evidence=object_size_evidence,
                object_edge_margin_m=args.object_edge_margin_m,
                pad_center_margin_m=args.pad_center_margin_m,
                desired_world_jaw_axis=desired_jaw,
                jaw_yaw_tolerance_deg=jaw_yaw_tolerance,
                desired_world_approach_axis=desired_approach,
                approach_axis_tolerance_deg=approach_axis_tolerance,
                desired_world_up_axis=desired_up,
                up_axis_tolerance_deg=up_axis_tolerance,
                body_midheight_tolerance_m=body_midheight_tolerance,
                first_contact_max_pinch_xy_m=(
                    args.first_contact_max_pinch_xy_m),
                first_contact_max_body_depth_offset_m=(
                    args.first_contact_max_body_depth_offset_m),
                approach_contact_substeps=args.approach_contact_substeps,
                object_approach_offset_m=object_approach_offset,
                diagnostic_grip_preload_m=args.diagnostic_grip_preload_m,
                diagnostic_grip_preload_activation_gap_m=(
                    args.diagnostic_grip_preload_activation_gap_m),
                diagnostic_approach_lookback_rows=(
                    args.diagnostic_approach_lookback_rows),
                diagnostic_canonical_object_xyz=(
                    diagnostic_canonical_object_xyz),
                diagnostic_project_side_frame=(
                    args.diagnostic_project_side_frame),
                diagnostic_align_contact_3d=(
                    args.diagnostic_align_contact_3d),
                kinematic_only=kinematic_only,
                camera_screen_available=camera_screen_available,
            )
            _save_candidate(
                state=progress, progress_path=progress_path,
                stage="final", report=report)
        reports.append(report)
        print(
            f"final {finalist_index + 1:2d}/{len(finalist_ids)} "
            f"candidate={candidate_id} "
            f"{'[resume] ' if resumed else ''}"
            f"combined={report['combined_accepted_fraction']:.1%} "
            f"geometry={report['oracle_geometry_accepted_fraction']:.1%} "
            f"scene={report['scene_constraint_episode_fraction']:.1%} "
            f"side-contact-centred="
            f"{report['first_approach_contact_body_depth_centered_episode_fraction']:.1%} "
            f"side={report['side_grasp_episode_fraction']:.1%} "
            f"partial={report['scene_component_fraction']:.1%}",
            flush=True,
        )
    reports.sort(key=rank, reverse=True)
    static_finalists = reports.copy()
    if args.dynamic_finalists > 0:
        dynamic_output_root.mkdir(parents=True, exist_ok=args.resume)
        dynamic_reports = []
        for dynamic_index, report in enumerate(
                reports[:args.dynamic_finalists], start=1):
            candidate_id = int(report["candidate_id"])
            cached = _cached_candidate(
                state=progress, stage="dynamic", candidate_id=candidate_id,
                arm=np.asarray(report["start_arm_rad"], dtype=float))
            resumed = cached is not None
            if cached is None:
                dynamic = _evaluate_dynamic_finalist(
                    report=report, episode_ids=dynamic_episode_ids,
                    data_root=args.data, output_root=dynamic_output_root,
                    object_size=object_size,
                    table_half_xy=[
                        float(value)
                        for value in cfg["task"]["table"]["half_size_m"][:2]
                    ],
                    cycles=args.dynamic_cycles,
                    tracking_time_margin=args.dynamic_tracking_time_margin,
                    grip_preload_m=args.dynamic_grip_preload_m,
                    grip_preload_activation_gap_m=(
                        args.dynamic_grip_preload_activation_gap_m),
                    threshold=args.min_dynamic_success_fraction,
                    config=args.config, real_config=args.real_config,
                    kinematic_grasp=kinematic_grasp,
                )
                completed_report = {**report, "dynamic_oracle": dynamic}
                _save_candidate(
                    state=progress, progress_path=progress_path,
                    stage="dynamic", report=completed_report)
            else:
                completed_report = cached
                dynamic = completed_report["dynamic_oracle"]
            dynamic_reports.append(completed_report)
            counts = dynamic["failure_counts"]
            print(
                f"dynamic {dynamic_index:2d}/{args.dynamic_finalists} "
                f"candidate={candidate_id} "
                f"{'[resume] ' if resumed else ''}"
                f"success={dynamic['successes']}/{dynamic['episodes']} "
                f"ik_after={counts.get('ik_reject_after_contact', 0)} "
                f"tipped={counts.get('object_tipped', 0)} "
                f"displaced={counts.get('object_displaced', 0)} "
                f"static_reject={counts.get('static_scene_reject', 0)}",
                flush=True,
            )

        reports = sorted(
            dynamic_reports,
            key=lambda report: _dynamic_rank(report, rank(report)),
            reverse=True,
        )

    selected = reports[0]
    static_accepted = _static_gate_passed(
        selected, args.min_static_scene_fraction)
    dynamic_accepted = bool(
        args.dynamic_finalists == 0
        or selected["dynamic_oracle"]["passed"])
    accepted = static_accepted and dynamic_accepted
    if diagnostic_canonical_object_xyz is not None:
        status = (
            "DIAGNOSTIC_CANONICAL_REPLAY_NOT_REGISTRATION"
            if accepted else "REJECTED_DIAGNOSTIC_CANONICAL_REPLAY"
        )
    elif accepted and diagnostic_proxy_ready:
        status = "DIAGNOSTIC_COLLISION_PROXY_CANDIDATE_NOT_PHYSICAL_VALIDATION"
    elif accepted and kinematic_grasp is not None:
        status = (
            "KINEMATIC_REGISTRATION_CANDIDATE_CAMERA_AND_DYNAMICS_UNVERIFIED"
            if not camera_screen_available
            else "KINEMATIC_REGISTRATION_CANDIDATE_NOT_DYNAMIC_READY"
        )
    elif accepted:
        status = "PROVISIONAL_NOT_PHYSICAL_CALIBRATION"
    elif static_accepted and args.dynamic_finalists > 0:
        status = "REJECTED_DYNAMIC_ORACLE_GATE"
    else:
        status = "REJECTED_NO_VALID_REGISTRATION"
    result = {
        "status": status,
        "source": source,
        "purpose": (
            "Multi-episode recorded-oracle registration search for offline IK "
            "preflight and camera-matched MuJoCo diagnostics only"
        ),
        "start_arm_rad": selected["start_arm_rad"],
        "object_xyz_m": selected["representative_object_xyz_m"],
        "object_size_m": object_size.tolist(),
        "table_half_size_xy_m": [
            float(value) for value in cfg["task"]["table"]["half_size_m"][:2]
        ],
        "object_edge_margin_m": args.object_edge_margin_m,
        "desired_world_jaw_axis": desired_jaw.tolist(),
        "jaw_yaw_tolerance_deg": jaw_yaw_tolerance,
        "desired_world_approach_axis": desired_approach.tolist(),
        "approach_axis_tolerance_deg": approach_axis_tolerance,
        "desired_world_up_axis": desired_up.tolist(),
        "up_axis_tolerance_deg": up_axis_tolerance,
        "body_midheight_tolerance_m": body_midheight_tolerance,
        "first_contact_max_pinch_xy_m": args.first_contact_max_pinch_xy_m,
        "first_contact_centring_metric": (
            "planar_perpendicular_to_approach_axis"),
        "first_contact_max_body_depth_offset_m": (
            args.first_contact_max_body_depth_offset_m),
        "approach_contact_substeps": args.approach_contact_substeps,
        "object_approach_offset_m": object_approach_offset,
        "diagnostic_canonical_object_xyz_m": (
            None if diagnostic_canonical_object_xyz is None
            else diagnostic_canonical_object_xyz.tolist()),
        "diagnostic_canonical_replay": bool(
            diagnostic_canonical_object_xyz is not None),
        "diagnostic_side_frame_projection": bool(
            args.diagnostic_project_side_frame),
        "diagnostic_contact_to_body_center_3d_alignment": bool(
            args.diagnostic_align_contact_3d),
        "diagnostic_grip_preload_m": args.diagnostic_grip_preload_m,
        "diagnostic_grip_preload_activation_gap_m": (
            args.diagnostic_grip_preload_activation_gap_m),
        "diagnostic_approach_lookback_rows": (
            args.diagnostic_approach_lookback_rows),
        "diagnostic_side_frame_rotation_world_eef": (
            desired_contact_rotation.tolist()
            if args.diagnostic_project_side_frame else None),
        "kinematic_grasp": kinematic_grasp,
        "camera_screen_available": camera_screen_available,
        "grasp_mode": source_registration.get("grasp_mode"),
        "axis_source": source_registration.get("axis_source"),
        "mean_start_to_contact_rotation": mean_contact_rotation.tolist(),
        "targeted_start_rotation": targeted_start_rotation.tolist(),
        "representative_episode": selected["representative_episode"],
        "selection": {
            "validation_episodes": validation_ids,
            "candidates": len(candidates),
            "base_candidates": args.candidates,
            "refine_candidates": args.refine_candidates,
            "refine_seeds": args.refine_seeds,
            "refine_sigma_fraction": args.refine_sigma_fraction,
            "side_grasp_ik_candidates": targeted_candidate_count,
            "screen_episodes": [dataset.episode_ids[index]
                                for index in screen_indices],
            "finalists": args.finalists,
            "dynamic_finalists": args.dynamic_finalists,
            "dynamic_episodes": dynamic_episode_ids,
            "dynamic_cycles": args.dynamic_cycles,
            "dynamic_tracking_time_margin": args.dynamic_tracking_time_margin,
            "dynamic_grip_preload_m": args.dynamic_grip_preload_m,
            "dynamic_grip_preload_activation_gap_m": (
                args.dynamic_grip_preload_activation_gap_m),
            "min_dynamic_success_fraction": (
                args.min_dynamic_success_fraction),
            "min_static_scene_fraction": args.min_static_scene_fraction,
            "samples_per_episode": args.samples_per_episode,
            "pad_center_margin_m": args.pad_center_margin_m,
            "object_edge_margin_m": args.object_edge_margin_m,
            "first_contact_max_pinch_xy_m": (
                args.first_contact_max_pinch_xy_m),
            "first_contact_centring_metric": (
                "planar_perpendicular_to_approach_axis"),
            "first_contact_max_body_depth_offset_m": (
                args.first_contact_max_body_depth_offset_m),
            "approach_contact_substeps": args.approach_contact_substeps,
            "object_approach_offset_m": object_approach_offset,
            "diagnostic_canonical_object_xyz_m": (
                None if diagnostic_canonical_object_xyz is None
                else diagnostic_canonical_object_xyz.tolist()),
            "diagnostic_side_frame_projection": bool(
                args.diagnostic_project_side_frame),
            "diagnostic_contact_to_body_center_3d_alignment": bool(
                args.diagnostic_align_contact_3d),
            "diagnostic_grip_preload_m": args.diagnostic_grip_preload_m,
            "diagnostic_grip_preload_activation_gap_m": (
                args.diagnostic_grip_preload_activation_gap_m),
            "diagnostic_approach_lookback_rows": (
                args.diagnostic_approach_lookback_rows),
            "seed": args.seed,
            "ranking": (
                "dynamic gate pass/success first, then separate post-contact IK, "
                "object-tip and static-scene penalties; static geometry+scene "
                "ranking breaks ties"
                if args.dynamic_finalists > 0 else
                "combined oracle geometry+scene fraction, exact scene fraction, "
                "full-footprint table margin, table centrality, object-centred "
                "pad grasp, side-approach task fidelity, partial "
                "scene constraints, oracle geometry and anchor reachability"
            ),
        },
        "screened_candidates": [{
            key: report[key] for key in (
                "candidate_id", "start_arm_rad", "anchor_accepted_fraction",
                "oracle_geometry_accepted_fraction",
                "combined_accepted_fraction",
                "scene_constraint_episode_fraction",
                "scene_component_fraction",
                "jaw_yaw_aligned_episode_fraction",
                "approach_axis_aligned_episode_fraction",
                "up_axis_aligned_episode_fraction",
                "side_grasp_episode_fraction",
                "initial_object_visible_episode_fraction",
                "pad_centered_episode_fraction",
                "contact_ik_available_episode_fraction",
                "object_table_margin_episode_fraction",
                "minimum_table_edge_clearance_m",
                "mean_table_centre_distance_norm",
                "pad_centered_given_contact_ik_fraction",
                "collision_free_given_contact_ik_fraction",
                "first_approach_contact_detected_episode_fraction",
                "first_approach_contact_centered_episode_fraction",
                "first_approach_contact_body_depth_centered_episode_fraction",
                "first_approach_contact_pinch_xy_m_mean",
                "first_approach_contact_pinch_xy_m_max",
                "first_approach_contact_lateral_m_mean",
                "first_approach_contact_lateral_m_max",
                "object_approach_offset_m",
            )
        } for report in screen_reports],
        "selected": selected,
        "static_finalists": static_finalists,
        "top10": reports[:10],
        "limitations": [
            "Search registration, not measured robot-base/workspace calibration.",
            "A diagnostic canonical replay translates the complete recorded "
            "TCP trajectory and inferred object together. It is only a "
            "simulation workspace/IK test and can never establish physical "
            "T_base_world, object pose, deployment calibration or policy success.",
            "Object locations are inferred from each episode's near-closed pinch "
            "pose plus the explicitly reported provisional approach-axis offset.",
            "Table placement requires the full object footprint plus the configured "
            "edge margin; centrality is a tie-breaker, not a measured object label.",
            "Swept first-contact metrics are diagnostic only and do not by "
            "themselves establish a measured marker-to-pad offset.",
            "Marker-midpoint hand-eye and object geometry remain provisional.",
            "Collision checks are kinematic MuJoCo screens, not hardware approval.",
            "A kinematic gripper overlay does not replace the legacy MuJoCo "
            "collision, inertia or actuator model.",
            "Kinematic-only candidate statuses must not be passed to rollout "
            "or a dynamic oracle gate.",
            "Must not be used for real motor commands.",
            "A rejected status must not be passed to policy preflight.",
            "Dynamic finalist replay uses fixed challenge episodes so candidates "
            "cannot win by making difficult episodes fail the static scene screen.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    progress["complete"] = True
    progress["result"] = {
        "path": str(args.out.resolve()),
        "sha256": _sha256(args.out),
        "status": status,
        "selected_candidate_id": int(selected["candidate_id"]),
    }
    _atomic_json(progress_path, progress)
    print(json.dumps({
        "status": result["status"],
        "selected_candidate": selected["candidate_id"],
        "start_arm_rad": selected["start_arm_rad"],
        "representative_episode": selected["representative_episode"],
        "object_xyz_m": selected["representative_object_xyz_m"],
        "combined_accepted_fraction": selected["combined_accepted_fraction"],
        "oracle_geometry_accepted_fraction": selected[
            "oracle_geometry_accepted_fraction"],
        "scene_constraint_episode_fraction": selected[
            "scene_constraint_episode_fraction"],
        "scene_component_fraction": selected["scene_component_fraction"],
        "jaw_yaw_aligned_episode_fraction": selected[
            "jaw_yaw_aligned_episode_fraction"],
        "approach_axis_aligned_episode_fraction": selected[
            "approach_axis_aligned_episode_fraction"],
        "up_axis_aligned_episode_fraction": selected[
            "up_axis_aligned_episode_fraction"],
        "side_grasp_episode_fraction": selected[
            "side_grasp_episode_fraction"],
        "dynamic_oracle": selected.get("dynamic_oracle"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
