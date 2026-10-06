"""Unit checks for the current-ver1 side-workspace diagnostic."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np

from probe_ver1_side_workspace import largest_true_rectangle, load_task_envelope


class Tests(unittest.TestCase):
    def test_largest_rectangle_uses_only_true_cells(self) -> None:
        grid = np.asarray([
            [False, True, True, False],
            [True, True, True, False],
            [True, True, False, False],
        ])
        area, bounds = largest_true_rectangle(grid)
        self.assertEqual(area, 4)
        self.assertIn(bounds, {(1, 2, 0, 1), (0, 1, 1, 2)})

    def test_rejects_non_grid_input(self) -> None:
        with self.assertRaises(ValueError):
            largest_true_rectangle(np.zeros(3, dtype=bool))

    def test_loads_only_explicit_simulation_envelope(self) -> None:
        payload = {
            "schema": "simulation_task_envelope/0.1.0",
            "status": "SIMULATION_ONLY_NOT_PHYSICAL_CALIBRATION",
            "object_xy_bounds_m": {"x": [0.385, 0.415], "y": [-0.035, 0.035]},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "envelope.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_task_envelope(path)
            self.assertEqual(loaded["object_xy_bounds_m"]["x"], [0.385, 0.415])
            payload["status"] = "PHYSICAL_CALIBRATION"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_task_envelope(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
