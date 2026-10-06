"""Hardware-free checks for applying fixed-marker trajectory alignment."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("apply_aruco_alignment.py")
SPEC = importlib.util.spec_from_file_location("apply_aruco_alignment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ApplyAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trajectory = self.root / "trajectory.csv"
        self.alignment = self.root / "alignment.json"
        self.output = self.root / "aligned.csv"
        fields = [
            "frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
            "x", "y", "z", "q_x", "q_y", "q_z", "q_w",
        ]
        with self.trajectory.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(zip(fields, [0, 0, 2, "false", "false", 1, 2, 3, 0, 0, 0, 1])))
            writer.writerow(dict(zip(fields, [1, 0.1, 1, "true", "false", 0, 0, 0, 0, 0, 0, 0])))
        self.alignment.write_text(json.dumps({
            "status": "PASS_ARUCO_ALIGNMENT",
            "T_marker_world": [
                [1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1],
            ],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_transforms_only_valid_rows(self) -> None:
        report = MODULE.apply_alignment(self.trajectory, self.alignment, self.output)
        with self.output.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(rows[0][key]) for key in ("x", "y", "z")], [11, 22, 33])
        self.assertEqual([float(rows[1][key]) for key in ("x", "y", "z")], [0, 0, 0])
        self.assertEqual(report["pose_semantics"], "T_marker_camera")
        self.assertTrue(report["training_input_ready"])

    def test_rejects_provisional_without_explicit_override(self) -> None:
        payload = json.loads(self.alignment.read_text(encoding="utf-8"))
        payload["status"] = "PROVISIONAL_ARUCO_ALIGNMENT"
        self.alignment.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not publishable"):
            MODULE.apply_alignment(self.trajectory, self.alignment, self.output)
        report = MODULE.apply_alignment(
            self.trajectory, self.alignment, self.output, allow_provisional=True
        )
        self.assertFalse(report["training_input_ready"])
        self.assertEqual(report["status"], "PROVISIONAL_OUTPUT")


if __name__ == "__main__":
    unittest.main()
