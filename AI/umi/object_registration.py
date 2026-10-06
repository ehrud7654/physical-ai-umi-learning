"""Geometry for registering a demonstrated object to a camera/EEF trajectory."""
from __future__ import annotations

import numpy as np


def pixel_ray_opencv(pixel_uv, intrinsics) -> np.ndarray:
    """Return a unit OpenCV-camera ray (+x right, +y down, +z forward)."""
    u, v = np.asarray(pixel_uv, dtype=float)
    fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
    if not np.isfinite([u, v, fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("pixel and camera intrinsics must be finite; focal lengths must be positive")
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return ray / np.linalg.norm(ray)


def intersect_ray_plane(origin, direction, plane_normal, plane_offset) -> np.ndarray:
    """Intersect origin+s*direction with n·x+offset=0, refusing behind-camera hits."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    values = np.r_[origin, direction, normal, float(plane_offset)]
    if origin.shape != (3,) or direction.shape != (3,) or normal.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("ray and plane values must be finite 3-vectors")
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-9:
        raise ValueError("camera ray is parallel to the table plane")
    distance = -(float(normal @ origin) + float(plane_offset)) / denominator
    if distance <= 0:
        raise ValueError("table-plane intersection is behind the camera")
    return origin + distance * direction


def object_center_from_bottom_pixel(pixel_uv, intrinsics, t_world_camera_opencv,
                                    plane_normal_world, plane_offset_world,
                                    object_height_m) -> np.ndarray:
    """Back-project an object's table-contact pixel and raise by half its height."""
    transform = np.asarray(t_world_camera_opencv, dtype=float)
    normal = np.asarray(plane_normal_world, dtype=float)
    if transform.shape != (4, 4) or object_height_m <= 0:
        raise ValueError("camera transform must be 4x4 and object height must be positive")
    ray_world = transform[:3, :3] @ pixel_ray_opencv(pixel_uv, intrinsics)
    bottom = intersect_ray_plane(transform[:3, 3], ray_world, normal, plane_offset_world)
    unit_normal = normal / np.linalg.norm(normal)
    return bottom + unit_normal * (float(object_height_m) / 2.0)


def offset_in_pinch(object_center_world, t_world_pinch) -> np.ndarray:
    transform = np.asarray(t_world_pinch, dtype=float)
    point = np.asarray(object_center_world, dtype=float)
    if transform.shape != (4, 4) or point.shape != (3,):
        raise ValueError("pinch transform must be 4x4 and object centre a 3-vector")
    return transform[:3, :3].T @ (point - transform[:3, 3])


def object_center_from_upright_bbox_opencv(bbox_xyxy, object_height_m, intrinsics) -> np.ndarray:
    """Estimate an upright object's camera-frame centre from its full-height bbox.

    This is a single-view pinhole estimate and must remain provisional unless the
    object height and segmentation bounds were independently measured.
    """
    x0, y0, x1, y1 = np.asarray(bbox_xyxy, dtype=float)
    height_px = y1 - y0 + 1.0
    height_m = float(object_height_m)
    fy = float(intrinsics["fy"])
    if (not np.isfinite([x0, y0, x1, y1, height_m, fy]).all()
            or height_px <= 0 or height_m <= 0 or fy <= 0):
        raise ValueError("bbox, object height and fy must be finite and positive")
    depth = fy * height_m / height_px
    centre = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
    return pixel_ray_opencv(centre, intrinsics) * (
        depth / pixel_ray_opencv(centre, intrinsics)[2])


def opencv_point_to_arcore_camera(point_opencv) -> np.ndarray:
    """OpenCV (+y down,+z forward) to ARCore/OpenGL (+y up,-z forward)."""
    point = np.asarray(point_opencv, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("camera point must be a finite 3-vector")
    return np.diag([1.0, -1.0, -1.0]) @ point
