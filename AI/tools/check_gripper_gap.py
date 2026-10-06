"""Synthetic checks for the marker-to-gap calibration and conservative statuses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from PIL import Image, ImageDraw

from umi.gripper_gap import (
    expected_marker_distance_mm,
    marker_distance_to_gap_m,
    measure_gap,
)


class Tests(unittest.TestCase):
    def test_affine_anchors_and_midpoint(self):
        self.assertAlmostEqual(marker_distance_to_gap_m(37.6), 0.0)
        self.assertAlmostEqual(marker_distance_to_gap_m(134.0), 0.070)
        expected = expected_marker_distance_mm(30.0)
        self.assertAlmostEqual(expected, 78.9142857143)
        self.assertAlmostEqual(marker_distance_to_gap_m(expected), 0.030)

    def test_two_direct_markers(self):
        image = Image.new("RGB", (800, 300), "white")
        draw = ImageDraw.Draw(image)
        # 60 px diameter represents 15 mm.  Centres 315.657 px apart represent
        # the measured 30 mm contact-gap calibration point.
        centres = (200, 515.657142857)
        for x in centres:
            draw.ellipse((x - 30, 120 - 30, x + 30, 120 + 30), fill=(235, 35, 125))
        result = measure_gap(image, min_area_px=100)
        self.assertEqual(result.status, "D")
        self.assertAlmostEqual(result.gap_m, 0.030, delta=0.001)

    def test_missing_and_out_of_range_are_x_not_clamped(self):
        blank = Image.new("RGB", (300, 200), "white")
        self.assertEqual(measure_gap(blank, min_area_px=50).status, "X")

        image = Image.new("RGB", (1000, 300), "white")
        draw = ImageDraw.Draw(image)
        for x in (100, 900):
            draw.ellipse((x - 20, 100 - 20, x + 20, 100 + 20), fill=(235, 35, 125))
        result = measure_gap(image, min_area_px=50)
        self.assertEqual(result.status, "X")
        self.assertIsNone(result.json_dict()["gap_m"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
