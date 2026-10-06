"""Extract UMI gripper contact gap from the two fluorescent magenta markers.

Calibration supplied with the 2026-09-11 real capture:

* marker: fluorescent magenta circle, 15 mm diameter
* marker-centre distance 37.6 mm -> contact gap 0 mm
* marker-centre distance 134.0 mm -> contact gap 70 mm

Only direct two-marker detections are emitted as ``D``.  Missing or invalid
detections are ``X`` with a NaN gap; this module deliberately does not invent
``M``/``T`` estimates or silently clamp out-of-range measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot, pi, sqrt
from typing import Iterable

import numpy as np
from PIL import Image


MARKER_DIAMETER_MM = 15.0
CLOSED_MARKER_DISTANCE_MM = 37.6
OPEN_MARKER_DISTANCE_MM = 134.0
OPEN_CONTACT_GAP_MM = 70.0


@dataclass(frozen=True)
class Marker:
    x_px: float
    y_px: float
    diameter_px: float
    area_px: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class GapMeasurement:
    gap_m: float
    status: str
    reason: str
    marker_distance_px: float | None = None
    marker_distance_mm: float | None = None
    markers: tuple[Marker, ...] = ()

    def json_dict(self) -> dict:
        value = asdict(self)
        value["gap_m"] = None if not np.isfinite(self.gap_m) else self.gap_m
        return value


def marker_distance_to_gap_m(marker_distance_mm: float) -> float:
    """Apply the measured two-point affine calibration without clamping."""
    return ((float(marker_distance_mm) - CLOSED_MARKER_DISTANCE_MM)
            * OPEN_CONTACT_GAP_MM
            / (OPEN_MARKER_DISTANCE_MM - CLOSED_MARKER_DISTANCE_MM)
            / 1000.0)


def expected_marker_distance_mm(contact_gap_mm: float) -> float:
    """Inverse calibration, useful for measured calibration-point checks."""
    return (CLOSED_MARKER_DISTANCE_MM + float(contact_gap_mm)
            * (OPEN_MARKER_DISTANCE_MM - CLOSED_MARKER_DISTANCE_MM)
            / OPEN_CONTACT_GAP_MM)


def _hue_distance(hue: np.ndarray, centre: int) -> np.ndarray:
    raw = np.abs(hue.astype(np.int16) - int(centre))
    return np.minimum(raw, 256 - raw)


def magenta_mask(image: Image.Image, *, hue=242, hue_tolerance=7,
                  min_saturation=90, min_value=120) -> np.ndarray:
    """Return a conservative HSV mask using Pillow's 0..255 HSV convention."""
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    return ((_hue_distance(hsv[..., 0], hue) <= hue_tolerance)
            & (hsv[..., 1] >= min_saturation)
            & (hsv[..., 2] >= min_value))


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(row.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _components(mask: np.ndarray) -> Iterable[tuple[int, float, float, tuple[int, int, int, int]]]:
    """Yield 8-connected run-length components without an OpenCV dependency."""
    parent: list[int] = []
    stats: list[list[float]] = []  # area, sum_x, sum_y, x0, y0, x1, y1

    def root(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def merge(a: int, b: int) -> int:
        a, b = root(a), root(b)
        if a == b:
            return a
        if stats[a][0] < stats[b][0]:
            a, b = b, a
        parent[b] = a
        for i in range(3):
            stats[a][i] += stats[b][i]
        stats[a][3] = min(stats[a][3], stats[b][3])
        stats[a][4] = min(stats[a][4], stats[b][4])
        stats[a][5] = max(stats[a][5], stats[b][5])
        stats[a][6] = max(stats[a][6], stats[b][6])
        return a

    previous: list[tuple[int, int, int]] = []
    for y, row in enumerate(mask):
        current: list[tuple[int, int, int]] = []
        for x0, x1 in _runs(row):
            overlaps = [label for p0, p1, label in previous
                        if p1 >= x0 - 1 and p0 <= x1]
            width = x1 - x0
            sum_x = (x0 + x1 - 1) * width / 2
            label = len(parent)
            parent.append(label)
            stats.append([width, sum_x, y * width, x0, y, x1 - 1, y])
            for other in overlaps:
                label = merge(label, other)
            current.append((x0, x1, label))
        previous = current

    seen: set[int] = set()
    for label in range(len(parent)):
        label = root(label)
        if label in seen:
            continue
        seen.add(label)
        area, sx, sy, x0, y0, x1, y1 = stats[label]
        yield int(area), sx / area, sy / area, (int(x0), int(y0), int(x1), int(y1))


def detect_markers(image: Image.Image, *, min_area_px=500,
                   max_area_fraction=0.08, **mask_options) -> tuple[Marker, ...]:
    mask = magenta_mask(image, **mask_options)
    max_area = image.width * image.height * max_area_fraction
    markers = []
    for area, x, y, bbox in _components(mask):
        if not min_area_px <= area <= max_area:
            continue
        width, height = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
        aspect = width / height
        if not 0.25 <= aspect <= 4.0:
            continue
        markers.append(Marker(x, y, 2 * sqrt(area / pi), area, bbox))
    return tuple(sorted(markers, key=lambda marker: marker.area_px, reverse=True)[:2])


def measure_gap(image: Image.Image, **detection_options) -> GapMeasurement:
    markers = detect_markers(image, **detection_options)
    if len(markers) != 2:
        return GapMeasurement(float("nan"), "X", "two_markers_not_detected", markers=markers)
    first, second = markers
    mean_diameter_px = (first.diameter_px + second.diameter_px) / 2
    if mean_diameter_px <= 0:
        return GapMeasurement(float("nan"), "X", "invalid_marker_diameter", markers=markers)
    distance_px = hypot(second.x_px - first.x_px, second.y_px - first.y_px)
    distance_mm = distance_px * MARKER_DIAMETER_MM / mean_diameter_px
    gap_m = marker_distance_to_gap_m(distance_mm)
    if not 0.0 <= gap_m <= OPEN_CONTACT_GAP_MM / 1000:
        return GapMeasurement(float("nan"), "X", "gap_out_of_calibrated_range",
                              distance_px, distance_mm, markers)
    return GapMeasurement(gap_m, "D", "direct_two_marker_detection",
                          distance_px, distance_mm, markers)
