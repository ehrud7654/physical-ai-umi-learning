"""Unit checks for the five-condition relative-policy visual suite."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_relative_chunk_visual_suite import classify, scene_valid, wilson


class VisualSuiteTest(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = wilson(0, 8)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.32440756488388023)
        lo, hi = wilson(4, 8)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_classification_priorities(self):
        self.assertEqual(classify({"object_tipped": True}), "object_tipped")
        self.assertEqual(classify({"object_displaced": True}), "object_displaced")
        self.assertEqual(classify({"stable_side_grasp_success": True}), "success")
        self.assertEqual(classify({"geometry_error": "bad IK"}), "geometry_error")
        self.assertEqual(classify({"first_bilateral_contact_cycle": None}),
                         "no_bilateral_contact")
        self.assertEqual(classify({"first_bilateral_contact_cycle": 0,
                                   "max_lift_height_m": 0.01}), "partial_lift")

    def test_scene_valid_only(self):
        payload = {"selected": {"per_episode": [
            {"episode": "a", "scene_constraints_ok": True},
            {"episode": "b", "scene_constraints_ok": False},
        ]}}
        self.assertEqual(scene_valid(payload), ["a"])


if __name__ == "__main__":
    unittest.main()
