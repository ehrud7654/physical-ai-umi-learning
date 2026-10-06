"""Measure how predictable episode trajectories are without looking at images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mae(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=0)


def probe(root: Path) -> dict[str, object]:
    episodes: list[np.ndarray] = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            episodes.append(np.asarray(data["state"][:, :5], dtype=np.float64))
    if not episodes:
        raise FileNotFoundError(f"no episode npz files: {root}")

    rates: dict[str, object] = {}
    for hz, stride in ((30, 1), (15, 2), (10, 3)):
        truth: list[np.ndarray] = []
        hold: list[np.ndarray] = []
        extrap: list[np.ndarray] = []
        for state in episodes:
            if len(state) <= 2 * stride:
                continue
            truth.append(state[2 * stride :: stride])
            hold.append(state[stride:-stride:stride])
            extrap.append(2.0 * state[stride:-stride:stride] - state[:-2 * stride:stride])
        y = np.concatenate(truth)
        p_hold = np.concatenate(hold)
        p_extrap = np.concatenate(extrap)
        hold_joint = mae(p_hold, y)
        extrap_joint = mae(p_extrap, y)
        hold_all = float(np.mean(hold_joint))
        extrap_all = float(np.mean(extrap_joint))
        rates[str(hz)] = {
            "samples": int(len(y)),
            "hold_mae": hold_all,
            "constant_velocity_mae": extrap_all,
            "improvement_over_hold_percent": 100.0 * (1.0 - extrap_all / hold_all),
            "hold_mae_by_arm_joint": hold_joint.tolist(),
            "constant_velocity_mae_by_arm_joint": extrap_joint.tolist(),
        }
    return {"data": str(root), "episodes": len(episodes), "rates_hz": rates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = probe(args.data)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
