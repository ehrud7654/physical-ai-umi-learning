"""Check invariants and aggregate counts in a moving-H2 object-only report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(report: dict) -> None:
    rows = report["per_episode"]
    assert report["inference_device"] == "cpu"
    assert report["all_included_episodes_held_out_from_training"] is True
    assert report["training_ready"] is False
    assert len(rows) == report["evaluated_source_episodes"]
    assert len(rows) + len(report["excluded"]) == report["eligible_source_episodes"]
    assert len({row["episode"] for row in rows}) == len(rows)
    assert len({row["episode"] for row in report["excluded"]}) == len(report["excluded"])
    assert not ({row["episode"] for row in rows}
                & {row["episode"] for row in report["excluded"]})
    assert report["paired_offsets"] == 4 * len(rows)

    normal_positive = 0
    mean_positive = 0
    four_positive = 0
    accel_exceeded = 0
    for row in rows:
        motion = row["motion"]
        assert row["ik"]["previous_ik_valid"] and row["ik"]["current_ik_valid"]
        assert abs(motion["history_interval_s"] - report["history_interval_s"]) <= 0.01
        assert 0 < motion["max_arm_speed_rad_s"] <= report["configured_speed_limit_rad_s"] + 1e-9
        assert motion["pad_contact_ticks"] == 0
        assert motion["object_xy_drift_m"] <= 0.001
        assert motion["acceleration_limit_exceeded"] == (
            motion["finite_difference_peak_accel_rad_s2"]
            > report["configured_accel_limit_rad_s2"])
        accel_exceeded += motion["acceleration_limit_exceeded"]
        assert len(row["views"]) == 5
        assert len(row["comparisons"]) == 4
        assert row["views"][0]["offset"] == [0.0, 0.0]
        for view in row["views"]:
            assert len(view["bboxes"]) == 2 and all(view["bboxes"])
            assert view["max_camera_pose_difference"] <= 1e-12
            assert view["max_robot_qpos_difference_rad"] <= 1e-12
            assert view["max_object_shift_error_m"] <= 1e-9
        n = [pair["normal"]["terminal_direction_positive"]
             for pair in row["comparisons"]]
        m = [pair["mean_fill_per_view"]["terminal_direction_positive"]
             for pair in row["comparisons"]]
        normal_positive += sum(n)
        mean_positive += sum(m)
        four_positive += all(n)

    assert normal_positive == report["normal_positive_terminal_direction"]
    assert mean_positive == report["mean_fill_positive_terminal_direction"]
    assert four_positive == report["episodes_with_all_four_normal_directions_positive"]
    assert accel_exceeded == report["episodes_exceeding_configured_accel_limit"]
    assert report["execution_feasible_under_config"] == (accel_exceeded == 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.report.read_text(encoding="utf-8")))
    print("moving-H2 object-only report: invariants and counts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
