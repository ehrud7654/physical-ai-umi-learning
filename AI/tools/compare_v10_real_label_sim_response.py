"""Compare real-image label slope with paired simulator policy response.

The real slope is an observational association, not a causal object-motion
effect. The simulator slope is a causal image intervention with robot/proprio
fixed. Comparing them diagnoses transfer, not policy success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _real_pair(rows: list[dict], split: str) -> np.ndarray:
    values = [[row["selected"]["object_bbox_xywh_norm"][0],
               row["selected"]["first_four_target_last_translation_local_m"][0]]
              for row in rows if row["split"] == split
              and row["selected"] is not None]
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 3:
        raise ValueError(f"too few matched real {split} episodes")
    return result


def _association(pairs: np.ndarray) -> dict:
    x, y = pairs.T
    if np.std(x) <= 0 or np.std(y) <= 0:
        raise ValueError("object centres or action targets are constant")
    slope, intercept = np.polyfit(x, y, deg=1)
    return {"episodes": len(pairs),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "slope_m_per_normalized_image_x": float(slope),
            "intercept_m": float(intercept)}


def _quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"p05": float(np.quantile(array, 0.05)),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95))}


def compare(real_path: Path, probe_path: Path) -> dict:
    real = json.loads(real_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if (real.get("status") != "REAL_V10_OPEN_GAP_SIM_SCALE_IMAGE_SUPPORT_DIAGNOSTIC"
            or probe.get("status") != "LOCAL_FAR_START_H2_OBJECT_ONLY_FIRST_FOUR_DIAGNOSTIC"):
        raise ValueError("expected matched real-image and object-only probe reports")
    training = _real_pair(real["rows"], "training")
    heldout = _real_pair(real["rows"], "heldout")
    train_fit = _association(training)
    heldout_fit = _association(heldout)
    fitted_heldout = np.polyval([
        train_fit["slope_m_per_normalized_image_x"],
        train_fit["intercept_m"]], heldout[:, 0])
    constant = float(np.median(training[:, 1]))
    sim_rows = []
    for row in probe["rows"]:
        by_offset = {tuple(pair["offset_world_xy_m"]): (index, pair)
                     for index, pair in enumerate(row["comparisons"], start=1)}
        plus_index, plus = by_offset[(0.0, 0.01)]
        minus_index, minus = by_offset[(0.0, -0.01)]
        plus_bbox = row["views"][plus_index]["object_bbox_xywh"][1]
        minus_bbox = row["views"][minus_index]["object_bbox_xywh"][1]
        screen_difference = float(plus_bbox[0] - minus_bbox[0])
        if abs(screen_difference) < 0.01:
            raise ValueError(f"object screen shift too small: {row['episode']}")
        policy_difference = float(
            plus["conditions"]["normal"]["executed_prefix_last_translation_m"][0]
            - minus["conditions"]["normal"]["executed_prefix_last_translation_m"][0])
        ideal_difference = float(
            plus["expected_local_xyz_m"][0] - minus["expected_local_xyz_m"][0])
        sim_rows.append({
            "episode": row["episode"],
            "screen_center_x_plus_minus_difference_norm": screen_difference,
            "policy_local_x_plus_minus_difference_m": policy_difference,
            "geometric_local_x_plus_minus_difference_m": ideal_difference,
            "policy_slope_m_per_normalized_image_x":
                policy_difference / screen_difference,
            "geometric_slope_m_per_normalized_image_x":
                ideal_difference / screen_difference,
        })
    if not sim_rows:
        raise ValueError("no object-only simulator episodes")
    return {
        "status": "REAL_LABEL_ASSOCIATION_VS_SIM_CAUSAL_IMAGE_RESPONSE_DIAGNOSTIC",
        "real_report": str(real_path),
        "sim_report": str(probe_path),
        "train_association": train_fit,
        "heldout_association": heldout_fit,
        "heldout_action_x_mae_m_using_train_line": float(
            np.mean(np.abs(fitted_heldout - heldout[:, 1]))),
        "heldout_action_x_mae_m_using_train_median": float(
            np.mean(np.abs(constant - heldout[:, 1]))),
        "sim_episode_count": len(sim_rows),
        "sim_policy_slope_distribution": _quantiles([
            row["policy_slope_m_per_normalized_image_x"] for row in sim_rows]),
        "sim_geometric_slope_distribution": _quantiles([
            row["geometric_slope_m_per_normalized_image_x"] for row in sim_rows]),
        "sim_policy_local_x_difference_abs_m": _quantiles([
            abs(row["policy_local_x_plus_minus_difference_m"])
            for row in sim_rows]),
        "sim_geometric_local_x_difference_abs_m": _quantiles([
            abs(row["geometric_local_x_plus_minus_difference_m"])
            for row in sim_rows]),
        "sim_rows": sim_rows,
        "limitations": [
            "Real x association is exploratory: selected frame, camera pose and phase may confound it.",
            "Real and simulated RGB content, H=2 motion, and camera calibration remain different.",
            "The simulator intervention shifts only the object; slopes are not grasp-success rates.",
            "These data cannot attribute the transfer failure to one factor without a controlled retraining ablation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = compare(args.real, args.sim_probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("train_association", "heldout_association",
                       "sim_episode_count", "sim_policy_slope_distribution",
                       "sim_geometric_slope_distribution")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
