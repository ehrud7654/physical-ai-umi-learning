"""Map the current ver1 horizontal side-grasp workspace.

This is a policy-free MuJoCo diagnostic.  It uses the reviewed ver1 TCP/frame,
visible fingers and fitted (still provisional) collision patches.  It does not
estimate ``T_base_world`` and must not be used as physical calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sim.mujoco.build_scene import (
    build_model,
    load_config,
    sync_gripper_collision_proxy,
)
from tools.search_relative_chunk_registration import (
    _camera_metrics,
    _collision_counts,
    _geom_vertical_interval,
    _require_source_registration,
    _set_static_object_pose,
    _table_placement_metrics,
)
from tools.umi_mujoco import (
    MujocoIK,
    apply_kinematic_grasp,
)
from umi.ik import matrix_to_quat


AI_ROOT = Path(__file__).resolve().parents[1]


def load_task_envelope(path: Path) -> dict[str, Any]:
    """Load a predeclared simulation envelope without promoting it to calibration."""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("schema") != "simulation_task_envelope/0.1.0":
        raise ValueError("unsupported task-envelope schema")
    if envelope.get("status") != "SIMULATION_ONLY_NOT_PHYSICAL_CALIBRATION":
        raise ValueError("task envelope must be explicitly simulation-only")
    bounds = envelope.get("object_xy_bounds_m") or {}
    for axis in ("x", "y"):
        values = np.asarray(bounds.get(axis), dtype=float)
        if (values.shape != (2,) or not np.isfinite(values).all()
                or values[0] >= values[1]):
            raise ValueError(f"invalid {axis} bounds in task envelope")
    return envelope


def largest_true_rectangle(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None]:
    """Return cell count and inclusive ``(x0, x1, y0, y1)`` indices."""
    values = np.asarray(grid, dtype=bool)
    if values.ndim != 2:
        raise ValueError("workspace grid must be two-dimensional")
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for y0 in range(values.shape[0]):
        live = np.ones(values.shape[1], dtype=bool)
        for y1 in range(y0, values.shape[0]):
            live &= values[y1]
            x0 = 0
            while x0 < len(live):
                if not live[x0]:
                    x0 += 1
                    continue
                x1 = x0
                while x1 + 1 < len(live) and live[x1 + 1]:
                    x1 += 1
                area = (y1 - y0 + 1) * (x1 - x0 + 1)
                if area > best_area:
                    best_area, best = area, (x0, x1, y0, y1)
                x0 = x1 + 1
    return best_area, best


def _pad_names(cfg: dict[str, Any]) -> list[str]:
    block = cfg.get("gripper_pads") or {}
    if block.get("kind") != "symmetric_parallel_jaw_runtime_proxy":
        raise ValueError("ver1 runtime parallel-jaw collision proxy is required")
    names = [str(value) for value in block.get("pad_names", [])]
    if len(names) != 2:
        raise ValueError("exactly two ver1 collision pads are required")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--kinematic-grasp", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=AI_ROOT / "configs" / "so101.yaml")
    parser.add_argument("--scene", type=Path,
                        default=AI_ROOT / "sim" / "mujoco" / "scenes" / "pick_place.xml")
    parser.add_argument(
        "--task-envelope", type=Path,
        help=("predeclared simulation-only XY envelope; when provided it "
              "overrides --x-range and --y-range"))
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.25, 0.60))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.20, 0.20))
    parser.add_argument("--step-m", type=float, default=0.025)
    parser.add_argument("--gap-m", type=float, default=0.040)
    parser.add_argument("--object-approach-offset-m", type=float, default=-0.020)
    parser.add_argument("--object-edge-margin-m", type=float, default=0.030)
    parser.add_argument("--pad-center-margin-m", type=float, default=0.005)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    task_envelope = None
    if args.task_envelope is not None:
        task_envelope = load_task_envelope(args.task_envelope)
        bounds = task_envelope["object_xy_bounds_m"]
        args.x_range = tuple(float(value) for value in bounds["x"])
        args.y_range = tuple(float(value) for value in bounds["y"])
    if (args.step_m <= 0.0 or args.gap_m <= 0.0
            or args.object_edge_margin_m < 0.0
            or args.pad_center_margin_m < 0.0):
        raise SystemExit("step, gap and margins must be valid positive values")
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    _require_source_registration(registration)
    overlay = json.loads(args.kinematic_grasp.read_text(encoding="utf-8"))
    if overlay.get("collision_proxy", {}).get("diagnostic_dynamic_ready") is not True:
        raise ValueError("fitted ver1 collision proxy must be explicitly enabled")

    cfg = load_config(args.config)
    apply_kinematic_grasp(cfg, overlay)
    object_size = np.asarray(registration["object_size_m"], dtype=float)
    cfg["task"]["object"]["half_size_m"] = (object_size / 2.0).tolist()
    if "table_half_size_xy_m" in registration:
        cfg["task"]["table"]["half_size_m"][:2] = [
            float(value) for value in registration["table_half_size_xy_m"]]
    table = cfg["task"]["table"]
    table_top = float(table["pos"][2]) + float(table["half_size_m"][2])
    object_z = table_top + float(object_size[2]) / 2.0
    cfg["task"]["object"]["init_pos"][2] = object_z

    jaw = np.asarray(registration["desired_world_jaw_axis"], dtype=float)
    approach = np.asarray(
        registration["desired_world_approach_axis"], dtype=float)
    up = np.asarray(registration["desired_world_up_axis"], dtype=float)
    jaw /= np.linalg.norm(jaw)
    approach /= np.linalg.norm(approach)
    up /= np.linalg.norm(up)
    rotation = np.column_stack([jaw, up, approach])
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or np.linalg.det(rotation) < 0.999999):
        raise ValueError("registration side-grasp axes must form a right-handed frame")
    quaternion = matrix_to_quat(rotation)

    model = build_model(cfg, args.scene)
    data = mujoco.MjData(model)
    ik = MujocoIK(model, cfg)
    pad_names = _pad_names(cfg)
    xs = np.arange(args.x_range[0], args.x_range[1] + 1e-12, args.step_m)
    ys = np.arange(args.y_range[0], args.y_range[1] + 1e-12, args.step_m)
    shape = (len(ys), len(xs))
    ik_grid = np.zeros(shape, dtype=bool)
    collision_grid = np.zeros(shape, dtype=bool)
    pad_grid = np.zeros(shape, dtype=bool)
    camera_grid = np.zeros(shape, dtype=bool)
    table_grid = np.zeros(shape, dtype=bool)
    accepted_grid = np.zeros(shape, dtype=bool)
    position_error_mm = np.full(shape, np.nan)
    axis_error_deg = np.full(shape, np.nan)
    roll_error_deg = np.full(shape, np.nan)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            object_xyz = np.asarray([x, y, object_z], dtype=float)
            placement = _table_placement_metrics(
                object_xyz=object_xyz, object_size_m=object_size,
                table_xy=np.asarray(table["pos"][:2], dtype=float),
                table_half_xy=np.asarray(table["half_size_m"][:2], dtype=float),
                edge_margin_m=args.object_edge_margin_m)
            table_grid[iy, ix] = placement["object_on_table_with_margin"]
            target_xyz = (
                object_xyz - approach * args.object_approach_offset_m)
            solution = ik.solve(
                target_xyz, quaternion, args.gap_m, q_init=None)
            position_error_mm[iy, ix] = solution.pos_error_m * 1000.0
            axis_error_deg[iy, ix] = solution.axis_error_deg
            roll_error_deg[iy, ix] = solution.roll_residual_deg
            ik_ok = bool(
                solution.converged and solution.within_limits
                and solution.pos_error_m <= 0.005
                and solution.axis_error_deg <= 5.0
                and solution.roll_residual_deg <= 5.0)
            ik_grid[iy, ix] = ik_ok
            if not ik_ok:
                continue
            _set_static_object_pose(model, data, object_xyz)
            data.qpos[:6] = solution.q_rad
            sync_gripper_collision_proxy(model, data, cfg)
            mujoco.mj_forward(model, data)
            table_contacts, self_contacts = _collision_counts(model, data)
            collision_grid[iy, ix] = table_contacts == 0 and self_contacts == 0
            centred = []
            for name in pad_names:
                low, high, _ = _geom_vertical_interval(model, data, name)
                centred.append(
                    low + args.pad_center_margin_m <= object_z
                    <= high - args.pad_center_margin_m)
            pad_grid[iy, ix] = all(centred)
            camera_grid[iy, ix] = bool(
                _camera_metrics(model, data, object_xyz)["visible"])
            accepted_grid[iy, ix] = bool(
                table_grid[iy, ix] and collision_grid[iy, ix]
                and pad_grid[iy, ix] and camera_grid[iy, ix])

    area, rectangle = largest_true_rectangle(accepted_grid)
    rectangle_m = None
    if rectangle is not None:
        x0, x1, y0, y1 = rectangle
        rectangle_m = [float(xs[x0]), float(xs[x1]),
                       float(ys[y0]), float(ys[y1])]
    result = {
        "status": "DIAGNOSTIC_VER1_SIDE_WORKSPACE_NOT_PHYSICAL_CALIBRATION",
        "purpose": "policy-free current-ver1 side-grasp workspace support",
        "grid": {
            "x_m": xs.tolist(), "y_m": ys.tolist(),
            "step_m": args.step_m,
            "ik": ik_grid.tolist(),
            "collision_free": collision_grid.tolist(),
            "pad_centered": pad_grid.tolist(),
            "camera_visible": camera_grid.tolist(),
            "table_margin": table_grid.tolist(),
            "accepted": accepted_grid.tolist(),
            "position_error_mm": position_error_mm.tolist(),
            "axis_error_deg": axis_error_deg.tolist(),
            "roll_error_deg": roll_error_deg.tolist(),
        },
        "counts": {
            "total": int(accepted_grid.size),
            "ik": int(ik_grid.sum()),
            "accepted": int(accepted_grid.sum()),
        },
        "largest_axis_aligned_rectangle": {
            "cells": int(area), "xy_bounds_m": rectangle_m,
        },
        "conditions": {
            "acceptance": (
                "IK + table/self collision-free + pad vertical centring + "
                "initial camera visibility + table footprint margin; static "
                "bilateral contact and lift are not claimed"),
            "object_size_m": object_size.tolist(),
            "object_z_m": object_z,
            "gap_m": args.gap_m,
            "object_approach_offset_m": args.object_approach_offset_m,
            "object_edge_margin_m": args.object_edge_margin_m,
            "pad_center_margin_m": args.pad_center_margin_m,
            "desired_world_jaw_axis": jaw.tolist(),
            "desired_world_approach_axis": approach.tolist(),
            "desired_world_up_axis": up.tolist(),
            "registration": str(args.registration.resolve()),
            "kinematic_grasp": str(args.kinematic_grasp.resolve()),
            "camera_optical_extrinsic_present": bool(
                overlay.get("camera_optical_extrinsic_present") is True),
            "task_envelope": (
                None if args.task_envelope is None
                else str(args.task_envelope.resolve())),
            "task_envelope_status": (
                None if task_envelope is None
                else task_envelope["status"]),
            "task_envelope_source": (
                None if task_envelope is None
                else {
                    "source_archive": task_envelope["source_archive"],
                    "source_archive_sha256": task_envelope[
                        "source_archive_sha256"],
                    "source_paths": task_envelope["source_paths"],
                    "source_task": task_envelope["source_task"],
                }),
        },
        "limitations": [
            "This grid is simulation-only and is not measured T_base_world.",
            "The fitted collision patches and dynamics remain provisional.",
            "A static accepted cell does not prove a full trajectory or lift.",
            "No learned checkpoint or policy inference was used.",
            *(
                [] if overlay.get("camera_optical_extrinsic_present") is True
                else [
                    "The wrist-camera visibility check uses the provisional "
                    "MuJoCo camera because the ver1 optical extrinsic is absent."
                ]
            ),
            *(task_envelope.get("limitations", [])
              if task_envelope is not None else []),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "total": result["counts"]["total"],
        "ik": result["counts"]["ik"],
        "accepted": result["counts"]["accepted"],
        "largest_axis_aligned_rectangle": result[
            "largest_axis_aligned_rectangle"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
