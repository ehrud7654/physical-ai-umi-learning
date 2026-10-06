"""Fast unit checks for 10Hz policy-free suite bookkeeping and resume safety."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_sim_relative_10hz_suite import (
    _case_paths, _resume_paths, _summary, parse_offset,
)


class DenseSuiteTests(unittest.TestCase):
    def test_partial_attempt_is_preserved_and_new_attempt_used(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            offset = (-0.01, 0.0)
            report0, candidate0 = _case_paths(directory, "episode", offset)
            candidate0.touch()
            report1, candidate1, attempt = _resume_paths(directory, "episode", offset)
            self.assertEqual(attempt, 1)
            self.assertNotEqual(candidate0, candidate1)
            self.assertFalse(report1.exists())
            self.assertTrue(candidate0.exists())

            report1.touch()
            candidate1.touch()
            candidate1.with_suffix(".json").touch()
            resumed_report, resumed_candidate, attempt = _resume_paths(
                directory, "episode", offset)
            self.assertEqual((resumed_report, resumed_candidate, attempt),
                             (report1, candidate1, 1))

    def test_failures_stay_in_denominator(self) -> None:
        cases = [
            {"status": "CANDIDATE_CHECKED", "rows": 14},
            {"status": "CANDIDATE_REJECTED", "rows": 12},
            {"status": "PENDING"},
        ]
        result = _summary({"episodes": ["one"]}, cases)
        self.assertEqual(result["total_cases"], 3)
        self.assertEqual(result["checked_candidates"], 1)
        self.assertEqual(result["rejected_or_error_cases"], 1)
        self.assertEqual(result["candidate_rows"], 14)
        self.assertFalse(result["training_ready"])
        cases[2]["status"] = "RUNNER_ERROR"
        self.assertEqual(
            _summary({"episodes": ["one"]}, cases)["status"],
            "COMPLETE_WITH_REJECTIONS_POLICY_FREE_10HZ_DIAGNOSTIC")

    def test_offset_is_bounded(self) -> None:
        self.assertEqual(parse_offset("-0.01,0"), (-0.01, 0.0))
        with self.assertRaises(Exception):
            parse_offset("nan,0")
        with self.assertRaises(Exception):
            parse_offset("0.04,0")


if __name__ == "__main__":
    unittest.main()
