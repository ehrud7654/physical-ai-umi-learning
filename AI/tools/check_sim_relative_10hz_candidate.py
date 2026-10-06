"""Audit exact-10Hz achieved-future MuJoCo rows without approving training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from tools.estimate_umi_object_geometry import largest_object_bbox
from umi.relative_dataset import SCHEMA, validate_arrays, vector_to_transform


def check(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    report = json.loads(Path(meta["source_oracle_report"]).read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    problems = validate_arrays(arrays, obs_horizon=2, action_horizon=8)
    if (meta.get("schema") != SCHEMA
            or meta.get("status") != "CANDIDATE_NOT_TRAINING_READY"
            or meta.get("training_ready") is not False
            or meta.get("human_demonstration") is not False
            or meta.get("source")
            != "policy_free_mujoco_achieved_future_tcp_10hz_candidate"):
        problems.append("candidate provenance or training block is missing")
    if (report.get("mode") != "recorded_oracle"
            or report.get("counterfactual_oracle_target_shift_once") is not True
            or report.get("full_chunk_validation") is not True
            or report.get("final_oracle_tail_executed") is not True
            or report.get("geometry_error") is not None
            or report.get("stable_side_grasp_success") is not True
            or report.get("dense_candidate_stream_rows") != len(arrays["action"])):
        problems.append("source policy-free physical oracle did not pass")
    obs_time = arrays["observation_timestamp"]
    action_time = arrays["action_timestamp"]
    if len(obs_time) < 2:
        problems.append("need at least two dense rows")
    if not np.allclose(np.diff(obs_time, axis=1), 0.1, atol=1e-6, rtol=0):
        problems.append("observation history is not exactly 10Hz")
    if not np.allclose(np.diff(obs_time[:, -1]), 0.1, atol=1e-6, rtol=0):
        problems.append("adjacent current observations are not exactly 10Hz")
    expected_future = obs_time[:, -1, None] + 0.1 * np.arange(1, 9)
    if not np.allclose(action_time, expected_future, atol=1e-6, rtol=0):
        problems.append("future targets are not on the measured 10Hz clock")
    if not np.allclose(obs_time[1:, 0], obs_time[:-1, 1], atol=1e-9, rtol=0):
        problems.append("adjacent observation histories are not continuous")
    if not np.array_equal(arrays["image"][1:, 0], arrays["image"][:-1, 1]):
        problems.append("adjacent RGB histories are not continuous")
    if not np.allclose(
            arrays["proprio"][1:, 0, 9], arrays["proprio"][:-1, 1, 9],
            atol=1e-7, rtol=0):
        problems.append("adjacent measured gap histories are not continuous")
    if np.any(np.diff(arrays["source_row"]) < 0):
        problems.append("source rows moved backwards")
    # The next achieved pose is expressed from both the current and the next
    # row anchor. Their relative transforms must compose exactly.
    for index in range(len(arrays["action"]) - 1):
        future = vector_to_transform(arrays["action"][index, 0])
        previous = vector_to_transform(arrays["proprio"][index + 1, 0])
        if not np.allclose(future @ previous, np.eye(4), atol=2e-5, rtol=0):
            problems.append(f"future/current achieved TCP mismatch at row {index}")
            break
        if not np.isclose(arrays["action"][index, 0, 9],
                          arrays["proprio"][index + 1, 1, 9], atol=1e-7):
            problems.append(f"future/current achieved gap mismatch at row {index}")
            break
    image_difference = float(np.abs(
        arrays["image"][:, 0].astype(np.int16)
        - arrays["image"][:, 1].astype(np.int16)).mean())
    if image_difference == 0:
        problems.append("RGB history is static")
    object_boxes = []
    for chw in arrays["image"][:, -1]:
        image = Image.fromarray(np.transpose(chw, (1, 2, 0)))
        component = largest_object_bbox(image)
        object_boxes.append(component[3] if component is not None else None)
    if any(box is None for box in object_boxes):
        problems.append("object colour component is absent from a wrist frame")
    step = np.diff(obs_time[:, -1])
    return {
        "status": ("DIAGNOSTIC_CANDIDATE_CHECKED_NOT_TRAINING_READY"
                   if not problems else "CANDIDATE_REJECTED"),
        "candidate": str(path),
        "rows": len(arrays["action"]),
        "observation_step_s_min_median_max": (
            [float(np.min(step)), float(np.median(step)), float(np.max(step))]
            if len(step) else None),
        "image_history_abs_difference_mean": image_difference,
        "object_visible_rows": sum(box is not None for box in object_boxes),
        "physical_oracle_stable": report.get("stable_side_grasp_success"),
        "geometry_error": report.get("geometry_error"),
        "training_ready": False,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    result = check(args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
