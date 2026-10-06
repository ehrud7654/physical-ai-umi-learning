"""Policy-free bounded IK audit before a moving-H2 visual probe.

No learned checkpoint is loaded, no collision tolerance is relaxed, and this
report does not select or approve a physical robot registration.
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
from sim.mujoco.build_scene import load_config
from sim.mujoco.env import MujocoPickEnv
from tools.probe_fixed_pose_object_only import _source_path, _suite_conditions
from tools.probe_moving_history_object_only import _history_target
from tools.render_relative_chunk_rollout import _episode_approach_start, _registration
from tools.umi_mujoco import MujocoIK, apply_kinematic_grasp
from umi.convert import invert_gap_curve
from umi.visual_domain import apply_visual_domain, load_visual_domain


BACKOFFS_M = (0.01, 0.02, 0.03)
OPEN_GAPS_M = (0.045, 0.065)
HISTORY_TARGET_STEP_M = 0.005


def audit(suite_path: Path) -> dict:
    conditions, episodes, _ = _suite_conditions(suite_path)
    registration = _registration(_source_path(conditions["registration"]))
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG))
    cfg = apply_visual_domain(
        cfg, load_visual_domain(_source_path(conditions["visual_domain"])))
    if isinstance(registration.get("kinematic_grasp"), dict):
        apply_kinematic_grasp(cfg, registration["kinematic_grasp"])
    cfg["task"]["object"]["half_size_m"] = (
        registration["object_size_m"] / 2).tolist()
    cfg["task"]["object"]["init_pos"] = registration["object_xyz_m"].tolist()
    cfg["task"]["table"]["half_size_m"][:2] = (
        registration["table_half_size_xy_m"].tolist())
    curve = cfg["grasp"]["gap_curve"]
    approach = np.asarray(registration["desired_world_approach_axis"], dtype=float)
    approach /= np.linalg.norm(approach)

    rows = []
    with MujocoPickEnv(cfg, render=False, object_jitter_m=0.0) as env:
        ik = MujocoIK(env.model, cfg)
        for episode in episodes:
            _, arm, gap = _episode_approach_start(registration, episode)
            anchor_q = np.r_[arm, invert_gap_curve(gap, curve)]
            for backoff in BACKOFFS_M:
                for open_gap in OPEN_GAPS_M:
                    _, _, details = _history_target(
                        ik, env.model, anchor_q, approach,
                        backoff_m=backoff,
                        history_target_step_m=HISTORY_TARGET_STEP_M,
                        open_gap_m=open_gap, require_valid=False)
                    rows.append({"episode": episode, **details,
                                 "both_valid": bool(details["previous_ik_valid"]
                                                    and details["current_ik_valid"])})
    summary = [{
        "backoff_m": backoff,
        "open_gap_m": open_gap,
        "valid_episodes": sum(row["both_valid"] for row in rows
                              if row["backoff_m"] == backoff
                              and row["open_gap_m"] == open_gap),
        "eligible_episodes": len(episodes),
    } for backoff in BACKOFFS_M for open_gap in OPEN_GAPS_M]
    return {
        "status": "POLICY_FREE_MOVING_H2_BACKOFF_IK_AUDIT_ONLY",
        "suite": str(suite_path),
        "episodes": episodes,
        "predeclared_backoffs_m": BACKOFFS_M,
        "predeclared_open_gaps_m": OPEN_GAPS_M,
        "history_target_step_m": HISTORY_TARGET_STEP_M,
        "summary": summary,
        "rows": rows,
        "training_ready": False,
        "limitations": [
            "Checks two endpoint IK solves only; no motion, collision, image, or contact gate.",
            "A feasible pair is not a valid common robot registration or a human demonstration.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite an existing report")
    report = audit(args.suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
