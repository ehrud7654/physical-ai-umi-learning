"""Hardware-free regression checks for the real-policy gripper phase guard."""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.gripper_runtime_guard import (  # noqa: E402
    GripperGuardConfig,
    GripperGuardConfigError,
    GripperGuardState,
    apply_gripper_phase_guard,
    load_gripper_guard_config,
)


CONFIG = ROOT / "configs" / "real" / "gripper_guard_onsite260921_pick.json"


def main() -> int:
    passed = total = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, total
        total += 1
        passed += bool(condition)
        print(("PASS" if condition else "FAIL") + f"  {name}")

    cfg = load_gripper_guard_config(CONFIG)
    state = GripperGuardState()

    # A 91.5 mm physical observation may align to the demonstrated 79 mm open
    # reference, but that alignment is not a close-phase latch.
    first = apply_gripper_phase_guard([0.07897] * 4, 0.09154, state, cfg)
    check("90->79 mm initial alignment does not latch", not first.state.latched)
    check("initial alignment commands the configured open reference",
          min(first.commanded_gap_m) >= cfg.pregrasp_open_m - 1e-12)

    # The old runner progressively closed here.  One low-gap cycle is not enough.
    vote1 = apply_gripper_phase_guard([0.0434, 0.0410, 0.0389, 0.0368],
                                      0.04734, first.state, cfg)
    check("first close-looking cycle remains open", not vote1.state.latched)
    check("first close-looking cycle is held open",
          min(vote1.commanded_gap_m) >= cfg.pregrasp_open_m - 1e-12)

    vote2 = apply_gripper_phase_guard([0.0399, 0.0388, 0.0376, 0.0372],
                                      0.03930, vote1.state, cfg)
    check("second consecutive close cycle latches", vote2.state.latched)
    check("close offset applies only after confirmation",
          vote2.commanded_gap_m[0] < vote2.report["raw_predicted_gap_m"][0])
    check("pick-hold commands are monotonic within the chunk",
          all(a >= b for a, b in zip(vote2.commanded_gap_m, vote2.commanded_gap_m[1:])))

    reopened = apply_gripper_phase_guard([0.079] * 4, 0.037, vote2.state, cfg)
    check("pick_hold does not reopen during lift", reopened.state.latched)

    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["task_mode"] = "policy_reopen"
    place_cfg = GripperGuardConfig.from_mapping(raw)
    place_reopen = apply_gripper_phase_guard([0.079] * 4, 0.037, vote2.state, place_cfg)
    check("policy_reopen task accepts an explicit reopen", not place_reopen.state.latched)

    try:
        apply_gripper_phase_guard([float("nan")] * 4, 0.05, state, cfg)
    except ValueError:
        check("NaN policy gap is rejected", True)
    else:
        check("NaN policy gap is rejected", False)

    try:
        bad = dict(raw)
        bad.pop("close_threshold_m")
        GripperGuardConfig.from_mapping(bad)
    except GripperGuardConfigError:
        check("missing dataset-specific threshold fails closed", True)
    else:
        check("missing dataset-specific threshold fails closed", False)

    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
