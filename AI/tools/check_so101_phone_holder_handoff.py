"""Validate the SO-101 ver1 phone-holder handoff without trusting its prose.

The archive is a mechanical-model input, not an instruction source.  This
check verifies every packaged SHA-256, parses the URDF, and derives the
gripper TCP frame expressed in the wrist-roll link.  It deliberately does not
claim that the package is a complete MuJoCo dynamics model: optical camera
extrinsics, custom-part inertias and collision-safe joint limits are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


SCHEMA = "so101_phone_holder_handoff_audit/0.1.0"
PREFIX = "handoff/"
URDF_PATH = PREFIX + "model/urdf/so101_phone_holder.urdf"
SUMS_PATH = PREFIX + "SHA256SUMS.txt"
LIMITATIONS_PATH = PREFIX + "validation/KNOWN_LIMITATIONS.md"


def _numbers(value: str, *, count: int) -> list[float]:
    result = [float(item) for item in value.split()]
    if len(result) != count or not all(math.isfinite(item) for item in result):
        raise ValueError(f"expected {count} finite values, got {value!r}")
    return result


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4))
             for j in range(4)] for i in range(4)]


def _transform(xyz: list[float], rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return [
        [*rotation[0], xyz[0]],
        [*rotation[1], xyz[1]],
        [*rotation[2], xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _joint_transform(root: ET.Element, name: str) -> list[list[float]]:
    joint = root.find(f"./joint[@name='{name}']")
    if joint is None:
        raise ValueError(f"required joint is absent: {name}")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"joint has no origin: {name}")
    return _transform(
        _numbers(origin.attrib.get("xyz", "0 0 0"), count=3),
        _numbers(origin.attrib.get("rpy", "0 0 0"), count=3),
    )


def _column(matrix: list[list[float]], index: int) -> list[float]:
    return [matrix[row][index] for row in range(3)]


def _joint_summary(root: ET.Element, name: str) -> dict[str, object]:
    joint = root.find(f"./joint[@name='{name}']")
    if joint is None:
        raise ValueError(f"required joint is absent: {name}")
    parent = joint.find("parent")
    child = joint.find("child")
    axis = joint.find("axis")
    limit = joint.find("limit")
    result: dict[str, object] = {
        "type": joint.attrib.get("type"),
        "parent": None if parent is None else parent.attrib.get("link"),
        "child": None if child is None else child.attrib.get("link"),
    }
    if axis is not None:
        result["axis"] = _numbers(axis.attrib["xyz"], count=3)
    if limit is not None:
        result["limit"] = {
            key: float(limit.attrib[key])
            for key in ("lower", "upper", "velocity", "effort")
            if key in limit.attrib
        }
    if "mimic" in [child.tag for child in joint]:
        mimic = joint.find("mimic")
        assert mimic is not None
        result["mimic"] = {
            "joint": mimic.attrib.get("joint"),
            "multiplier": float(mimic.attrib.get("multiplier", "1")),
            "offset": float(mimic.attrib.get("offset", "0")),
        }
    return result


def audit(path: Path) -> dict[str, object]:
    archive_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing_required = sorted(
            {URDF_PATH, SUMS_PATH, LIMITATIONS_PATH} - names)
        if missing_required:
            raise ValueError(f"handoff is missing required files: {missing_required}")

        problems: list[str] = []
        checked = 0
        sums = archive.read(SUMS_PATH).decode("utf-8-sig")
        for line in sums.splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            member = PREFIX + relative.strip()
            if member not in names:
                problems.append(f"missing checksum member: {member}")
                continue
            actual = hashlib.sha256(archive.read(member)).hexdigest()
            checked += 1
            if actual != expected:
                problems.append(
                    f"sha256 mismatch: {relative.strip()} expected={expected} actual={actual}")

        root = ET.fromstring(archive.read(URDF_PATH))
        custom_hand = _joint_transform(root, "custom_hand_fixed")
        hand_tcp = _joint_transform(root, "gripper_tcp_fixed")
        wrist_tcp = _matmul(custom_hand, hand_tcp)
        limitations = archive.read(LIMITATIONS_PATH).decode("utf-8-sig")

    links = root.findall("./link")
    joints = root.findall("./joint")
    if root.attrib.get("name") != "so101_ver1_phone_holder_v010":
        problems.append(f"unexpected robot name: {root.attrib.get('name')!r}")
    if len(links) != 15 or len(joints) != 14:
        problems.append(f"unexpected URDF size: links={len(links)} joints={len(joints)}")

    gripper = _joint_summary(root, "gripper")
    right = _joint_summary(root, "gripper_right")
    left = _joint_summary(root, "gripper_mirror")
    if gripper.get("type") != "prismatic":
        problems.append("gripper source joint is not prismatic")
    for name, row in (("gripper_right", right), ("gripper_mirror", left)):
        mimic = row.get("mimic")
        if not isinstance(mimic, dict) or mimic.get("joint") != "gripper":
            problems.append(f"{name} does not mimic the full-width gripper joint")

    return {
        "schema": SCHEMA,
        "archive": str(path.resolve()),
        "archive_sha256": archive_sha,
        "internal_sha256_checked": checked,
        "internal_sha256_problems": problems,
        "passed": not problems,
        "urdf": {
            "name": root.attrib.get("name"),
            "links": len(links),
            "joints": len(joints),
            "gripper": gripper,
            "gripper_right": right,
            "gripper_mirror": left,
        },
        "t_wrist_roll_to_gripper_tcp": wrist_tcp,
        "tcp_origin_in_wrist_roll_m": _column(wrist_tcp, 3),
        "tcp_axes_in_wrist_roll": {
            "jaw_positive_x": _column(wrist_tcp, 0),
            "up_positive_y": _column(wrist_tcp, 1),
            "approach_positive_z": _column(wrist_tcp, 2),
        },
        "simulation_boundaries": {
            "camera_optical_extrinsic_present": False,
            "custom_part_inertias_validated": False,
            "collision_safe_joint_limits_present": False,
            "known_bracket_interference_documented": (
                "브래킷" in limitations and "교차" in limitations),
            "dynamic_mujoco_ready": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.archive)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"output already exists: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
