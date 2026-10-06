"""Hardware-contract FK and ROS JointState value conversion, without ROS imports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np

from contract.episode import GRIPPER_MAX_GAP_M


ARM_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
)


def canonical_file_sha256(path: Path) -> str:
    """Hash text content with LF newlines, independent of Git checkout policy."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class JointStateSnapshot:
    ros_time_s: float
    arm_rad: np.ndarray
    gripper_width_m: float
    t_base_tcp: np.ndarray


def _numbers(text: str | None, count: int) -> np.ndarray:
    values = np.fromstring(text or "", sep=" ", dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"expected {count} finite values, got {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _transform(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, v = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [x*x*v+c, x*y*v-z*s, x*z*v+y*s],
        [y*x*v+z*s, y*y*v+c, y*z*v-x*s],
        [z*x*v-y*s, z*y*v+x*s, z*z*v+c],
    ])


class SO101UrdfFK:
    """FK from the released base_link to fixed gripper_tcp chain."""

    def __init__(self, urdf_path: Path, base="base_link", tcp="gripper_tcp"):
        self.path = Path(urdf_path)
        root = ET.parse(self.path).getroot()
        by_child = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            parent = joint.find("parent")
            if child is None or parent is None:
                continue
            by_child[child.attrib["link"]] = joint
        chain = []
        link = tcp
        while link != base:
            if link not in by_child:
                raise ValueError(f"no URDF chain from {base} to {tcp}: stopped at {link}")
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        self.chain = tuple(reversed(chain))
        movable = tuple(j.attrib["name"] for j in self.chain if j.attrib["type"] == "revolute")
        if movable != ARM_JOINTS:
            raise ValueError(f"unexpected base-to-TCP joints: {movable}")
        self.limits_rad = np.array([
            [float(j.find("limit").attrib["lower"]), float(j.find("limit").attrib["upper"])]
            for j in self.chain if j.attrib["type"] == "revolute"
        ])

    def pose(self, arm_rad) -> np.ndarray:
        q = np.asarray(arm_rad, dtype=float)
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("arm_rad must contain five finite radians")
        if np.any(q < self.limits_rad[:, 0]) or np.any(q > self.limits_rad[:, 1]):
            raise ValueError("arm_rad outside released URDF limits")
        values = dict(zip(ARM_JOINTS, q))
        result = np.eye(4)
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = _numbers(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", 3)
            rpy = _numbers(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", 3)
            result = result @ _transform(xyz, _rpy_matrix(rpy))
            if joint.attrib["type"] == "revolute":
                axis = _numbers(joint.find("axis").attrib["xyz"], 3)
                result = result @ _transform(np.zeros(3), _axis_rotation(axis, values[joint.attrib["name"]]))
        return result


def snapshot_from_joint_state(names, positions, stamp_sec, stamp_nanosec,
                              fk: SO101UrdfFK,
                              *, max_gap_m=GRIPPER_MAX_GAP_M) -> JointStateSnapshot:
    """Convert one published JointState sample; timestamp remains ROS Time."""
    names = tuple(names)
    values = np.asarray(positions, dtype=float)
    if len(names) != len(set(names)) or values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("invalid JointState names or positions")
    state = dict(zip(names, values))
    required = (*ARM_JOINTS, "gripper")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"JointState missing {missing}")
    arm = np.array([state[name] for name in ARM_JOINTS])
    gap = float(state["gripper"])
    if not math.isclose(float(max_gap_m), GRIPPER_MAX_GAP_M, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("robot max_gap_m does not match the episode contract")
    if not 0.0 <= gap <= max_gap_m:
        raise ValueError("gripper must be full contact-surface width in 0..max_gap_m")
    if type(stamp_sec) is not int or type(stamp_nanosec) is not int or stamp_sec < 0 \
            or not 0 <= stamp_nanosec < 1_000_000_000:
        raise ValueError("invalid ROS Time stamp")
    ros_time_s = stamp_sec + stamp_nanosec * 1e-9
    return JointStateSnapshot(ros_time_s, arm.copy(), gap, fk.pose(arm))
