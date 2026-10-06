"""Synthetic checks for pixel/table-plane object registration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from umi.object_registration import (intersect_ray_plane, object_center_from_bottom_pixel,
                                      object_center_from_upright_bbox_opencv,
                                      offset_in_pinch, opencv_point_to_arcore_camera,
                                      pixel_ray_opencv)


def main() -> int:
    intrinsics = {"fx": 100, "fy": 100, "cx": 50, "cy": 40}
    np.testing.assert_allclose(pixel_ray_opencv([50, 40], intrinsics), [0, 0, 1])
    np.testing.assert_allclose(intersect_ray_plane([0, 0, 1], [0, 0, -1], [0, 0, 1], 0),
                               [0, 0, 0])
    camera = np.eye(4); camera[:3, 3] = [0, 0, 1]
    camera[:3, :3] = np.diag([1, -1, -1])
    centre = object_center_from_bottom_pixel([50, 40], intrinsics, camera, [0, 0, 1], 0, .10)
    np.testing.assert_allclose(centre, [0, 0, .05])
    pinch = np.eye(4); pinch[:3, 3] = [.01, 0, .08]
    np.testing.assert_allclose(offset_in_pinch(centre, pinch), [-.01, 0, -.03])
    box = object_center_from_upright_bbox_opencv(
        [40, 30.5, 60, 49.5], .10, {"fx": 100, "fy": 100, "cx": 50, "cy": 40})
    np.testing.assert_allclose(box, [0, 0, .5])
    np.testing.assert_allclose(opencv_point_to_arcore_camera([1, 2, 3]), [1, -2, -3])
    for call in (lambda: intersect_ray_plane([0, 0, 1], [1, 0, 0], [0, 0, 1], 0),
                 lambda: object_center_from_bottom_pixel([50, 40], intrinsics, camera,
                                                         [0, 0, 1], 0, 0),
                 lambda: object_center_from_upright_bbox_opencv([0, 2, 1, 1], .1, intrinsics)):
        try: call()
        except ValueError: pass
        else: raise AssertionError("invalid registration geometry must be rejected")
    print("PASS: pixel ray, table-plane intersection and pinch-relative object offset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
