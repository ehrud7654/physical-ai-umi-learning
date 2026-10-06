"""Hardware-free regression checks for the ORB-SLAM3 intake gate."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.orbslam_intake import validate_orbslam_episode


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class OrbslamIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "rec_fixture"
        self.raw.mkdir()
        (self.raw / "manifest.json").write_text(json.dumps({
            "outcome": "success",
            "marker_black_square_size_verified": True,
        }), encoding="utf-8")
        count = 40
        _write_csv(
            self.raw / "frames.csv",
            ["frame_number", "sensor_timestamp_ns"],
            [{"frame_number": index, "sensor_timestamp_ns": 1_000_000_000 + index * 100_000_000}
             for index in range(count)],
        )
        self.trajectory = self.root / "camera_trajectory.csv"
        self.tracking = self.root / "camera_trajectory.csv.tracking.csv"
        self.gripper = self.root / "gripper_report.json"
        self.gripper.write_text(json.dumps({
            "status": "pass",
            "frames_with_both_markers": count,
            "detection_rate": 1.0,
            "maximum_missing_run_frames": 0,
        }), encoding="utf-8")
        self.write_outputs(first_valid=5)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_outputs(self, *, first_valid: int, timestamp_offset_s: float = 0.0,
                      loss_frame: int | None = None) -> None:
        trajectory_rows = []
        tracking_rows = []
        for index in range(40):
            valid = index >= first_valid and index != loss_frame
            timestamp = index * 0.1 + timestamp_offset_s
            trajectory_rows.append({
                "frame_idx": index, "timestamp": timestamp, "state": 2 if valid else 1,
                "is_lost": str(not valid).lower(), "is_keyframe": "false",
                "x": 0.001 * index, "y": 0, "z": 0,
                "q_x": 0, "q_y": 0, "q_z": 0, "q_w": 1,
            })
            tracking_rows.append({
                "timestamp": timestamp,
                "tracking_ok": int(valid),
                "imu_initialized": int(index >= max(first_valid, 10)),
            })
        _write_csv(self.trajectory, list(trajectory_rows[0]), trajectory_rows)
        _write_csv(self.tracking, list(tracking_rows[0]), tracking_rows)

    def validate(self):
        return validate_orbslam_episode(
            self.raw, self.trajectory, self.tracking, self.gripper,
            fixed_marker_id=13, fixed_marker_size_m=0.16,
            fixed_marker_size_verified=True,
        )

    def test_accepts_complete_fixture(self):
        self.assertEqual(self.validate()["status"], "PASS_ORB_INTAKE")

    def test_rejects_late_initialization_and_low_coverage(self):
        self.write_outputs(first_valid=35)
        report = self.validate()
        self.assertEqual(report["status"], "REJECT_ORB_INTAKE")
        self.assertTrue(any("coverage" in error for error in report["errors"]))
        self.assertTrue(any("first valid pose" in error for error in report["errors"]))

    def test_rejects_timestamp_offset(self):
        self.write_outputs(first_valid=5, timestamp_offset_s=0.011)
        self.assertTrue(any("timestamp error" in error for error in self.validate()["errors"]))

    def test_rejects_tracking_loss_after_initialization(self):
        self.write_outputs(first_valid=5, loss_frame=20)
        self.assertTrue(any("loss episode" in error for error in self.validate()["errors"]))

    def test_rejects_gripper_gap_and_unverified_scales(self):
        self.gripper.write_text(json.dumps({
            "status": "pass", "maximum_missing_run_frames": 8,
        }), encoding="utf-8")
        manifest = json.loads((self.raw / "manifest.json").read_text(encoding="utf-8"))
        manifest["marker_black_square_size_verified"] = False
        (self.raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = validate_orbslam_episode(
            self.raw, self.trajectory, self.tracking, self.gripper,
            fixed_marker_id=13, fixed_marker_size_m=0.16,
            fixed_marker_size_verified=False,
        )
        self.assertTrue(any("missing run" in error for error in report["errors"]))
        self.assertTrue(any("jaw marker metric scale" in error for error in report["errors"]))
        self.assertTrue(any("fixed ArUco marker metric size" in error for error in report["errors"]))

    def test_explicit_jaw_calibration_does_not_require_manifest_rewrite(self):
        manifest = json.loads((self.raw / "manifest.json").read_text(encoding="utf-8"))
        manifest["marker_black_square_size_verified"] = False
        (self.raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = validate_orbslam_episode(
            self.raw, self.trajectory, self.tracking, self.gripper,
            fixed_marker_id=13, fixed_marker_size_m=0.16,
            fixed_marker_size_verified=True,
            jaw_marker_size_verified=True,
        )
        self.assertEqual(report["status"], "PASS_ORB_INTAKE")
        self.assertTrue(report["gripper"]["jaw_marker_size_verified"])
        self.assertFalse(report["gripper"]["jaw_marker_size_verified_in_manifest"])


if __name__ == "__main__":
    unittest.main()
