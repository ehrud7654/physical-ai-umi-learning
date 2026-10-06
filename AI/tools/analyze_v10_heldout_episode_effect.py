"""Paired episode-level uncertainty for the v10 held-out image diagnostic.

This reuses saved local-CPU predictions. It does not run checkpoint inference.
The bootstrap unit is a demonstration episode, never an individual frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _episode_errors(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["episode"], []).append(row)
    result = []
    for episode, group in sorted(grouped.items()):
        values = {}
        for model in ("baseline", "lateral4"):
            values[model] = {}
            for condition in ("normal", "mean_fill", "additive_noise"):
                values[model][condition] = float(np.mean([
                    abs(row["predicted_local_x_m"][model][condition]
                        - row["target_local_x_m"]) for row in group]))
        result.append({
            "episode": episode, "matched_rows": len(group),
            "large_target_rows": sum(abs(row["target_local_x_m"]) > 0.01
                                     for row in group),
            "mae_m": values,
            "baseline_minus_lateral4_normal_mae_m": (
                values["baseline"]["normal"] - values["lateral4"]["normal"]),
            "lateral4_mean_fill_minus_normal_mae_m": (
                values["lateral4"]["mean_fill"]
                - values["lateral4"]["normal"]),
            "lateral4_noise_minus_normal_mae_m": (
                values["lateral4"]["additive_noise"]
                - values["lateral4"]["normal"]),
        })
    return result


def _paired_effect(episodes: list[dict], key: str, *,
                   seed: int, draws: int) -> dict:
    values = np.asarray([row[key] for row in episodes], dtype=float)
    if not len(values) or not np.isfinite(values).all() or draws < 100:
        raise ValueError("need finite episode effects and >=100 bootstrap draws")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "episodes": len(values), "positive_episodes": int(np.sum(values > 0)),
        "negative_episodes": int(np.sum(values < 0)),
        "zero_episodes": int(np.sum(values == 0)),
        "episode_equal_mean_m": float(np.mean(values)),
        "episode_equal_median_m": float(np.median(values)),
        "episode_cluster_bootstrap_95pct_m": np.quantile(
            boot, [0.025, 0.975]).astype(float).tolist(),
    }


def analyze(source: dict, *, seed: int = 0, draws: int = 20000) -> dict:
    if source.get("status") != "LOCAL_V10_HELDOUT_ALL_MATCHED_VISUAL_RESPONSE_DIAGNOSTIC":
        raise ValueError("expected held-out visual response report")
    rows = source["rows"]
    episodes = _episode_errors(rows)
    if len(episodes) != 10 or len(rows) != 28:
        raise ValueError("the compared 2026-09-18 v10 cohort has changed")
    keys = (
        "baseline_minus_lateral4_normal_mae_m",
        "lateral4_mean_fill_minus_normal_mae_m",
        "lateral4_noise_minus_normal_mae_m",
    )
    return {
        "status": "LOCAL_V10_EPISODE_PAIRED_UNCERTAINTY_EXPLORATORY",
        "source_checkpoint_sha256": source["checkpoints"],
        "bootstrap": {"resampling_unit": "episode", "draws": draws, "seed": seed,
                      "interval": "percentile_95pct_exploratory_not_preregistered"},
        "effects": {key: _paired_effect(episodes, key, seed=seed + index,
                                        draws=draws)
                    for index, key in enumerate(keys)},
        "large_target_coverage": {
            "rows": sum(row["large_target_rows"] for row in episodes),
            "episodes": sum(row["large_target_rows"] > 0 for row in episodes),
            "note": "Too few independent episodes for a reliable large-target interval."},
        "per_episode": episodes,
        "limitations": [
            "This is post-hoc uncertainty on an image-scale-selected cohort, not a preregistered performance gate.",
            "The held-out split does not verify that physical object placements are new.",
            "Bootstrap intervals summarize ten correlated demonstrations, not robot success.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    result = analyze(json.loads(args.source.read_text(encoding="utf-8")),
                     seed=args.seed, draws=args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"effects": result["effects"],
                      "large_target_coverage": result["large_target_coverage"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
