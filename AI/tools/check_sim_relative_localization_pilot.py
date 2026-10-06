"""Check counterfactual image/target coupling without approving training use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from umi.relative_dataset import SCHEMA, validate_arrays


def check(root: Path) -> dict:
    index = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    if index.get("schema") != SCHEMA:
        problems.append("wrong schema")
    if index.get("status") != "CANDIDATE_NOT_TRAINING_READY":
        problems.append("candidate status missing")
    if index.get("full_chunk_ik_pass_count", -1) > len(index.get("episodes", [])):
        problems.append("IK pass count exceeds sample count")
    if index.get("contact_dynamics_validated") is not False:
        problems.append("contact status must be unvalidated")
    ids = list(index.get("episodes", []))
    ik_pass_count = 0
    if sorted(ids) != sorted(path.stem for path in root.glob("*.npz")):
        problems.append("index/NPZ mismatch")
    groups: dict[str, list[dict]] = {}
    for sample_id in ids:
        path = root / f"{sample_id}.npz"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        problems.extend(
            f"{sample_id}: {message}" for message in validate_arrays(
                arrays, obs_horizon=2, action_horizon=8)
        )
        if meta.get("status") != "CANDIDATE_NOT_TRAINING_READY":
            problems.append(f"{sample_id}: training-readiness status missing")
        ik_pass_count += meta.get("full_chunk_ik_validated") is True
        if not np.array_equal(arrays["image"][0, 0], arrays["image"][0, 1]):
            problems.append(f"{sample_id}: two static images differ")
        if not np.array_equal(arrays["proprio"][0, 0], arrays["proprio"][0, 1]):
            problems.append(f"{sample_id}: two static proprio values differ")
        groups.setdefault(str(meta["source_episode"]), []).append({
            "id": sample_id,
            "offset": np.asarray(meta["object_offset_world_xy_m"], dtype=float),
            "bbox": np.asarray(meta["object_bbox_xywh_norm"], dtype=float),
            "image": arrays["image"],
            "proprio": arrays["proprio"],
            "action": arrays["action"],
            "row": arrays["source_row"],
        })

    pair_results = []
    for source, samples in groups.items():
        for first_index, first in enumerate(samples):
            for second in samples[first_index + 1:]:
                offset_distance = float(np.linalg.norm(first["offset"] - second["offset"]))
                if offset_distance < 1e-9:
                    problems.append(f"{source}: duplicate object offset")
                    continue
                if not np.array_equal(first["proprio"], second["proprio"]):
                    problems.append(f"{source}: proprio reveals the object offset")
                if not np.array_equal(first["row"], second["row"]):
                    problems.append(f"{source}: source row differs across offsets")
                a, b = first["action"][0], second["action"][0]
                if not np.allclose(a[:, 3:], b[:, 3:], atol=1e-5, rtol=0):
                    problems.append(f"{source}: rotation or gap changed")
                translation_shift = b[:, :3] - a[:, :3]
                if not np.allclose(translation_shift, translation_shift[0], atol=1e-5, rtol=0):
                    problems.append(f"{source}: target shift is not constant across chunk")
                shift_distance = float(np.linalg.norm(translation_shift[0]))
                if not np.isclose(shift_distance, offset_distance, atol=1e-5, rtol=0):
                    problems.append(f"{source}: target shift magnitude differs from object shift")
                image_mae = float(np.mean(np.abs(
                    first["image"].astype(np.float32) - second["image"].astype(np.float32))))
                bbox_shift = float(np.linalg.norm(first["bbox"][:2] - second["bbox"][:2]))
                if image_mae <= 0 or bbox_shift <= 0:
                    problems.append(f"{source}: displaced object did not change the visible image")
                pair_results.append({
                    "source": source,
                    "offset_distance_m": offset_distance,
                    "target_shift_distance_m": shift_distance,
                    "image_mae_0_255": image_mae,
                    "bbox_center_shift_norm": bbox_shift,
                })
    if not pair_results:
        problems.append("need at least two offsets of one source episode")
    if index.get("full_chunk_ik_pass_count") != ik_pass_count:
        problems.append("IK pass count differs from per-sample results")
    if index.get("full_chunk_ik_validated") is not (bool(ids) and ik_pass_count == len(ids)):
        problems.append("IK validation flag differs from per-sample results")
    return {
        "status": "PASS_COUNTERFACTUAL_COUPLING_ONLY" if not problems else "FAIL",
        "samples": len(ids),
        "paired_comparisons": len(pair_results),
        "pairs": pair_results,
        "problems": problems,
        "training_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    report = check(args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["problems"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
