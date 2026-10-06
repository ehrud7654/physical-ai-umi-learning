"""Audit one policy-free MuJoCo counterfactual stream; never approve training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from tools.estimate_umi_object_geometry import largest_object_bbox
from umi.relative_dataset import SCHEMA, validate_arrays


def check(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    report = json.loads(Path(meta["source_oracle_report"]).read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    problems = validate_arrays(arrays, obs_horizon=2, action_horizon=8)
    if (meta.get("schema") != SCHEMA
            or meta.get("status") != "CANDIDATE_NOT_TRAINING_READY"
            or meta.get("training_ready") is not False
            or meta.get("human_demonstration") is not False):
        problems.append("candidate provenance/training block is missing")
    if (report.get("mode") != "recorded_oracle"
            or report.get("counterfactual_oracle_target_shift_once") is not True
            or report.get("full_chunk_validation") is not True
            or report.get("geometry_error") is not None
            or report.get("stable_side_grasp_success") is not True
            or report.get("candidate_stream_rows") != len(arrays["action"])):
        problems.append("source policy-free oracle report does not support this stream")
    if len(arrays["action"]) < 2:
        problems.append("need at least two distinct dynamic training-row candidates")
    if arrays["source_row"][0] <= report["episode_start_row"]:
        problems.append("duplicated first observation was not excluded")
    if not np.all(np.diff(arrays["source_row"]) > 0):
        problems.append("source rows do not increase")
    obs_time = arrays["observation_timestamp"]
    if not np.array_equal(obs_time[1:, 0], obs_time[:-1, 1]):
        problems.append("adjacent observation histories are not continuous")
    if not np.array_equal(arrays["image"][1:, 0], arrays["image"][:-1, 1]):
        problems.append("adjacent RGB histories are not continuous")
    if not np.allclose(arrays["proprio"][1:, 0, 9],
                       arrays["proprio"][:-1, 1, 9], atol=1e-6, rtol=0):
        problems.append("adjacent measured gripper gaps disagree")
    image_history_difference = float(np.abs(
        arrays["image"][:, 0].astype(np.int16)
        - arrays["image"][:, 1].astype(np.int16)).mean())
    if image_history_difference == 0:
        problems.append("RGB history is static or rendering was disabled")
    object_boxes = []
    for image_chw in arrays["image"][:, -1]:
        image = Image.fromarray(np.transpose(image_chw, (1, 2, 0)))
        component = largest_object_bbox(image)
        object_boxes.append(component[3] if component is not None else None)
    if any(box is None for box in object_boxes):
        problems.append("object colour component is absent from a wrist frame")
    alignment_s = (obs_time[1:, 1] - arrays["action_timestamp"][:-1, 0])
    max_alignment_s = float(np.max(np.abs(alignment_s))) if len(alignment_s) else None
    if max_alignment_s is not None and max_alignment_s > 0.010000001:
        problems.append("planned first target vs next observed frame exceeds 10ms")
    observation_step_s = np.diff(obs_time[:, 1])
    return {
        "status": "DIAGNOSTIC_CANDIDATE_CHECKED_NOT_TRAINING_READY"
                  if not problems else "CANDIDATE_REJECTED",
        "candidate": str(path),
        "rows": len(arrays["action"]),
        "physical_oracle_stable": report["stable_side_grasp_success"],
        "geometry_error": report["geometry_error"],
        "image_history_abs_difference_mean": image_history_difference,
        "object_visible_rows": sum(box is not None for box in object_boxes),
        "observation_step_s_min_median_max": (
            [float(np.min(observation_step_s)),
             float(np.median(observation_step_s)),
             float(np.max(observation_step_s))]
            if len(observation_step_s) else None),
        "max_planned_target_to_next_observation_offset_ms": (
            1000 * max_alignment_s if max_alignment_s is not None else None),
        "source_nominal_period_s": 1 / float(meta["source_nominal_rate_hz"]),
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
