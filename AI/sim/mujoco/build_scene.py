"""Build the SO-101 pick-place MuJoCo model without touching the official MJCF.
공식 MJCF를 수정하지 않고 SO-101 픽앤플레이스 모델을 조립한다.

The official SO-101 description ships with zero cameras, and MJCF `include`
cannot reopen a body that lives in an included file — so a wrist camera cannot
be declared in the scene XML at all. Both cameras are therefore injected at
load time through `mujoco.MjSpec`, driven entirely by `configs/so101.yaml`.
공식 SO-101 기술서에는 카메라가 하나도 없고, include된 파일 안의 body는 다시
열 수 없다. 그래서 손목 카메라는 씬 XML에 선언 자체가 불가능하다.
두 카메라 모두 configs/so101.yaml 값으로 로드 시점에 주입한다.

Hardware-dependent numbers live only in the config. Nothing here is hardcoded.
하드웨어 의존 값은 설정 파일에만 있다. 여기에 하드코딩된 값은 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from paths import AI_ROOT, DEFAULT_CONFIG, DEFAULT_SCENE, resolve_for_mujoco  # noqa: F401

REPO_AI_ROOT = AI_ROOT


@dataclass(frozen=True)
class JointSpec:
    """One actuated joint and its hardware limit in radians.
    관절 하나와 그 하드웨어 한계(rad)."""

    name: str
    index: int
    lo: float
    hi: float


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the single-source-of-truth hardware config.
    하드웨어 값 단일 출처 설정을 읽는다."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def joint_specs(cfg: dict[str, Any]) -> list[JointSpec]:
    """Return joints ordered by their state/action vector index.
    state/action 벡터 인덱스 순서대로 관절을 반환한다."""
    specs = [
        JointSpec(j["name"], j["index"], float(j["range_rad"][0]), float(j["range_rad"][1]))
        for j in cfg["joints"]
    ]
    specs.sort(key=lambda s: s.index)
    if [s.index for s in specs] != list(range(len(specs))):
        raise ValueError(f"joint index must be contiguous from 0: {[s.index for s in specs]}")
    return specs


def _set_camera(cam: Any, spec: dict[str, Any], resolution: list[int]) -> None:
    """Apply one camera block from the config onto an MjSpec camera.
    설정의 카메라 블록 하나를 MjSpec 카메라에 적용한다."""
    xyaxes = np.asarray(spec["xyaxes"], dtype=float)
    if xyaxes.shape != (6,):
        raise ValueError(f"xyaxes must have 6 numbers, got {xyaxes.shape}")
    cam.pos = np.asarray(spec["pos"], dtype=float)
    cam.fovy = float(spec["fovy"])
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    cam.alt.xyaxes = xyaxes
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.resolution = np.asarray(resolution, dtype=float)


def build_model(
    cfg: dict[str, Any] | None = None,
    scene_path: Path = DEFAULT_SCENE,
) -> mujoco.MjModel:
    """Compile the scene with every camera injected from the config.
    설정에서 읽은 카메라를 전부 주입해 씬을 컴파일한다."""
    cfg = cfg if cfg is not None else load_config()
    cams = cfg["cameras"]
    resolution = cams["resolution"]
    # MuJoCo opens XML from C++ and cannot handle non-ASCII paths on Windows.
    # MuJoCo 는 C++ 에서 XML 을 열고 Windows 에서 비ASCII 경로를 처리하지 못한다.
    spec = mujoco.MjSpec.from_file(str(resolve_for_mujoco(scene_path)))

    # The XML supplies defaults, while the reviewed runtime contract lives in
    # so101.yaml.  Apply the task block before compilation so configuration
    # overrides are real model changes rather than misleading report metadata.
    task = cfg["task"]
    object_body = spec.body("target_object")
    if object_body is None:
        raise ValueError("scene has no target_object body")
    object_body.pos = np.asarray(task["object"]["init_pos"], dtype=float)
    object_geoms = [g for g in object_body.geoms if g.name == "target_object_geom"]
    if len(object_geoms) != 1:
        raise ValueError("scene must have exactly one target_object_geom")
    object_geoms[0].size = np.asarray(task["object"]["half_size_m"], dtype=float)
    object_geoms[0].mass = float(task["object"]["mass_kg"])
    table_geoms = [g for g in spec.worldbody.geoms if g.name == "table"]
    if len(table_geoms) != 1:
        raise ValueError("scene must have exactly one table geom")
    table_geoms[0].size = np.asarray(task["table"]["half_size_m"], dtype=float)
    table_geoms[0].pos = np.asarray(task["table"]["pos"], dtype=float)

    names = [k for k in cams if isinstance(cams[k], dict)]
    if len(names) != int(cams["count"]):
        raise ValueError(f"cameras.count={cams['count']} but {len(names)} camera blocks defined")

    _add_visual_domain_geometry(spec, cfg)

    for name in names:
        block = cams[name]
        kind = block["kind"]
        if kind == "fixed":
            parent = spec.worldbody
        elif kind == "attached":
            parent = spec.body(block["parent_body"])
            if parent is None:
                raise ValueError(
                    f"body {block['parent_body']!r} not found in {scene_path.name}; "
                    "공식 MJCF의 body 이름이 바뀌었는지 확인할 것"
                )
        else:
            raise ValueError(f"unknown camera kind {kind!r} for {name!r}")
        cam = parent.add_camera()
        cam.name = name
        _set_camera(cam, block, resolution)

    _apply_gripper_pads(spec, cfg)
    # Render determinism knob, from config -- see cameras.offsamples in so101.yaml.
    # 렌더 결정성 노브. 설정에서 읽는다 — so101.yaml 의 cameras.offsamples.
    if "offsamples" in cams:
        spec.visual.quality.offsamples = int(cams["offsamples"])
    model = spec.compile()

    _apply_visual_domain(model, cfg)

    found = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    if found != set(names):
        raise RuntimeError(f"camera injection failed: expected {set(names)}, got {found}")
    return model


def _apply_visual_domain(model: mujoco.MjModel, cfg: dict[str, Any]) -> None:
    """Apply optional simulation appearance after material compilation."""
    appearance = cfg.get("visual_domain")
    if not appearance:
        return
    names = {
        "object_rgba": "target_object_geom",
        "table_rgba": "table",
        "floor_rgba": "floor",
    }
    for key, geom_name in names.items():
        rgba = np.asarray(appearance[key], dtype=float)
        if rgba.shape != (4,) or not np.isfinite(rgba).all():
            raise ValueError(f"visual_domain.{key} must be finite RGBA")
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"visual-domain geom {geom_name!r} is missing")
        model.geom_rgba[geom_id] = rgba
    diffuse = np.asarray(appearance["light_diffuse_rgb"], dtype=float)
    if diffuse.shape != (3,) or not np.isfinite(diffuse).all():
        raise ValueError("visual_domain.light_diffuse_rgb must be finite RGB")
    if model.nlight:
        model.light_diffuse[:] = diffuse


def _add_visual_domain_geometry(spec: Any, cfg: dict[str, Any]) -> None:
    """Add render-only package details and background clutter before compile.

    Every added geom has both collision masks cleared and zero mass.  This lets
    the visual domain resemble the recorded S22 workspace without silently
    changing contact, inertia, IK, or recorded-oracle physics.
    """
    appearance = cfg.get("visual_domain")
    if not appearance:
        return
    for row in appearance.get("visual_geometry", []):
        parent_name = str(row["parent"])
        parent = spec.worldbody if parent_name == "world" else spec.body(parent_name)
        if parent is None:
            raise ValueError(f"visual-domain parent {parent_name!r} is missing")
        geom = parent.add_geom()
        geom.name = str(row["name"])
        kind = str(row.get("type", "box"))
        if kind == "box":
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
        elif kind == "cylinder":
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        else:
            raise ValueError(f"unsupported visual-domain geom type {kind!r}")
        geom.pos = np.asarray(row["pos_m"], dtype=float)
        size = np.asarray(row["size_m"], dtype=float)
        if kind == "cylinder":
            size = np.r_[size, 0.0]
        geom.size = size
        geom.rgba = np.asarray(row["rgba"], dtype=float)
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geom.mass = 0.0


def _apply_gripper_pads(spec: Any, cfg: dict[str, Any]) -> None:
    """Replace the jaw hull collision with explicit finger pads.
    턱의 볼록껍질 충돌을 명시적 손가락 패드로 교체한다.

    MuJoCo collides meshes as convex hulls, and the hull of a two-pronged jaw is
    a solid wedge with no opening — measured, the 2 cm cube penetrates it by up
    to 3.1 cm at every candidate grasp point with the gripper fully open. Without
    this the arm can only ever push the object.
    MuJoCo 는 메시를 볼록껍질로 충돌시키는데, 두 갈래 턱의 껍질은 틈이 없는
    덩어리다 — 실측으로 그리퍼를 완전히 벌려도 2cm 정육면체가 모든 후보 파지점에서
    최대 3.1cm 관통한다. 이 처리가 없으면 팔은 물체를 밀 수만 있다.

    The official MJCF is not edited; contype is cleared at load time instead.
    공식 MJCF 는 수정하지 않는다. 로드 시점에 contype 을 끌 뿐이다.
    """
    block = cfg.get("gripper_pads")
    if not block:
        return

    disabled = set(block.get("disable_mesh_collision", []))
    n_disabled = 0
    configured_bodies = (
        {str(block["parent_body"])}
        if block.get("kind") == "symmetric_parallel_jaw_runtime_proxy"
        else {str(value["body"]) for value in block["pads"]}
    )
    for body_name in configured_bodies | {"gripper", "moving_jaw_so101_v1"}:
        body = spec.body(body_name)
        if body is None:
            continue
        for geom in body.geoms:
            if geom.meshname in disabled and geom.contype != 0:
                geom.contype = 0
                geom.conaffinity = 0
                n_disabled += 1
    if n_disabled != len(disabled):
        raise RuntimeError(
            f"expected to disable {len(disabled)} collision meshes, disabled {n_disabled}; "
            "공식 MJCF 의 메시 이름이 바뀌었는지 확인할 것"
        )

    friction = np.asarray(block["friction"], dtype=float)
    rgba = np.asarray(block["rgba"], dtype=float)
    if block.get("kind") == "symmetric_parallel_jaw_runtime_proxy":
        _add_symmetric_parallel_jaw_proxy(spec, cfg, block, friction, rgba)
        return
    for pad in block["pads"]:
        body = spec.body(pad["body"])
        if body is None:
            raise ValueError(f"body {pad['body']!r} not found for pad {pad['name']!r}")
        geom = body.add_geom()
        geom.name = pad["name"]
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.pos = np.asarray(pad["pos"], dtype=float)
        geom.size = np.asarray(pad["half_size"], dtype=float)
        geom.contype = 1
        geom.conaffinity = 1
        geom.condim = int(block["condim"])
        geom.friction = friction
        geom.rgba = rgba
        geom.group = 3
        geom.mass = 0.0


def _add_symmetric_parallel_jaw_proxy(
    spec: Any,
    cfg: dict[str, Any],
    block: dict[str, Any],
    friction: np.ndarray,
    rgba: np.ndarray,
) -> None:
    """Add the reviewed ver1 visual hand and aligned provisional pad boxes.

    The legacy rotary-jaw meshes are hidden.  The handoff's parallel-jaw meshes
    and the diagnostic collision pads share the same two moving bodies, so the
    rendered fingers cannot disagree with the contact geometry.  Their local
    positions are synchronised from the measured gripper gap before every
    MuJoCo step.  This remains a contact diagnostic, not validated dynamics.
    """
    if block.get("schema") != "so101_parallel_jaw_collision_proxy/0.1.0-provisional":
        raise ValueError("unsupported symmetric gripper collision proxy schema")
    parent = spec.body(str(block["parent_body"]))
    if parent is None:
        raise ValueError("collision proxy parent body is missing")
    names = list(block.get("pad_names", []))
    half = np.asarray(block.get("finger_half_size_m"), dtype=float)
    gap_range = np.asarray(block.get("gap_range_m"), dtype=float)
    frame_cfg = cfg["grasp"].get("eef_frame_local", {})
    frame = np.column_stack([
        np.asarray(frame_cfg.get("jaw_axis"), dtype=float),
        np.asarray(frame_cfg.get("up_axis"), dtype=float),
        np.asarray(frame_cfg.get("approach_axis"), dtype=float),
    ])
    if (names != ["ver1_pad_negative", "ver1_pad_positive"]
            or half.shape != (3,) or np.any(~np.isfinite(half))
            or np.any(half <= 0.0)
            or gap_range.shape != (2,) or np.any(~np.isfinite(gap_range))
            or gap_range[0] < 0.0 or gap_range[0] >= gap_range[1]
            or frame.shape != (3, 3)
            or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-6, rtol=0.0)
            or not np.isclose(np.linalg.det(frame), 1.0, atol=1e-6, rtol=0.0)):
        raise ValueError("invalid symmetric gripper collision proxy geometry")
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, frame.reshape(-1))
    pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    initial_gap = float(gap_range[1])
    visual = block.get("visual_model")
    if (not isinstance(visual, dict)
            or visual.get("schema")
            != "so101_parallel_jaw_visual/0.1.0-provisional"
            or visual.get("visual_collision_alignment_required") is not True):
        raise ValueError(
            "symmetric collision proxy requires the matching visible ver1 hand"
        )

    transform = visual.get("custom_hand_fixed_from_legacy_gripper", {})
    xyz = np.asarray(transform.get("xyz_m"), dtype=float)
    rpy = np.asarray(transform.get("rpy_rad"), dtype=float)
    fingers = visual.get("finger_bodies")
    if (xyz.shape != (3,) or rpy.shape != (3,)
            or not np.isfinite(xyz).all() or not np.isfinite(rpy).all()
            or not isinstance(fingers, list) or len(fingers) != 2
            or sorted(float(row.get("sign", 0.0)) for row in fingers) != [-1.0, 1.0]):
        raise ValueError("invalid ver1 visible-hand transform or finger declaration")

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    hand_quat = np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)

    asset_dir = (REPO_AI_ROOT / str(visual.get("asset_dir", ""))).resolve()
    mesh_files = list(visual.get("static_meshes", []))
    for row in fingers:
        mesh_files.extend(row.get("meshes", []))
    if len(mesh_files) != len(set(mesh_files)):
        raise ValueError("ver1 visible-hand mesh declarations must be unique")
    mesh_names: dict[str, str] = {}
    for filename in mesh_files:
        path = asset_dir / str(filename)
        if path.suffix.lower() != ".obj" or not path.is_file():
            raise ValueError(f"missing ver1 visible-hand mesh: {path}")
        mesh_name = f"ver1_visual_{path.stem}"
        spec.add_mesh(name=mesh_name, file=str(resolve_for_mujoco(path)))
        mesh_names[str(filename)] = mesh_name

    hidden = set(visual.get("hide_legacy_meshes", []))
    if hidden != set(block.get("disable_mesh_collision", [])):
        raise ValueError("legacy visual and collision replacement lists must match")
    hidden_count = 0
    for body_name in {str(block["parent_body"]), "moving_jaw_so101_v1"}:
        body = spec.body(body_name)
        if body is None:
            continue
        for geom in body.geoms:
            if geom.meshname in hidden:
                geom.rgba = np.asarray([1.0, 1.0, 1.0, 0.0])
                hidden_count += 1
    if hidden_count != 4:
        raise RuntimeError(
            f"expected four legacy jaw visual/collision geoms, hid {hidden_count}"
        )

    hand = parent.add_body(
        name=str(visual["custom_hand_body"]), pos=xyz, quat=hand_quat)
    for filename in visual.get("static_meshes", []):
        hand.add_geom(
            name=f"ver1_static_{Path(filename).stem}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_names[str(filename)],
            contype=0,
            conaffinity=0,
            group=2,
            rgba=np.asarray([0.65, 0.69, 0.73, 1.0]),
            mass=0.0,
        )

    # In custom_hand coordinates the TCP is [0, -0.078018753, 0.027000012].
    # The pad's long axis points from palm to fingertip (-Y).  Attach both the
    # visual meshes and the box to the same translated body so a contact can
    # never occur at an invisible, differently moving location.
    hand_rotation = frame @ np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]).T
    hand_origin = pinch - hand_rotation @ np.asarray(
        [0.0, -0.0780187530518, 0.0270000119209])
    if not np.allclose(hand_origin, xyz, atol=2e-6, rtol=0.0):
        raise ValueError("visible hand transform disagrees with the reviewed pinch frame")
    local_pad_frame = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    local_pad_quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(local_pad_quat, local_pad_frame.reshape(-1))
    finger_by_sign = {float(row["sign"]): row for row in fingers}
    for name, sign in zip(names, (-1.0, 1.0)):
        row = finger_by_sign[sign]
        finger = hand.add_body(
            name=str(row["name"]), pos=np.asarray([sign * initial_gap / 2.0, 0.0, 0.0]))
        for filename in row["meshes"]:
            color = [0.12, 0.35, 0.8, 1.0] if "holder" in filename else [0.75, 0.78, 0.82, 1.0]
            finger.add_geom(
                name=f"{row['name']}_{Path(filename).stem}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_names[str(filename)],
                contype=0,
                conaffinity=0,
                group=2,
                rgba=np.asarray(color),
                mass=0.0,
            )
        pad = finger.add_geom()
        pad.name = name
        pad.type = mujoco.mjtGeom.mjGEOM_BOX
        pad.pos = np.asarray([
            sign * half[0],
            -0.0780187530518 + half[2],
            0.0270000119209,
        ])
        pad.quat = local_pad_quat
        pad.size = half
        pad.contype = 1
        pad.conaffinity = 1
        pad.condim = int(block["condim"])
        pad.friction = friction
        pad.rgba = rgba
        pad.group = 3
        pad.mass = 0.0


def active_gripper_pad_ids(model: mujoco.MjModel, cfg: dict[str, Any]) -> set[int]:
    """Return only configured pad geoms, excluding motor and palm collisions."""
    block = cfg.get("gripper_pads") or {}
    names = (
        list(block.get("pad_names", []))
        if block.get("kind") == "symmetric_parallel_jaw_runtime_proxy"
        else [str(value["name"]) for value in block.get("pads", [])]
    )
    ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in names
    }
    if -1 in ids or len(ids) != len(names):
        raise ValueError("configured gripper pad geoms are missing or duplicated")
    return ids


def sync_gripper_collision_proxy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: dict[str, Any],
) -> bool:
    """Synchronise provisional symmetric pads to the measured current gap.

    Returns True when the proxy exists. Callers must invoke this before contact
    generation whenever qpos changes outside normal MuJoCo integration.
    """
    block = cfg.get("gripper_pads") or {}
    if block.get("kind") != "symmetric_parallel_jaw_runtime_proxy":
        return False
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    if joint_id < 0:
        raise ValueError("gripper joint is missing for collision proxy")
    qpos_address = int(model.jnt_qposadr[joint_id])
    curve = np.asarray(cfg["grasp"]["gap_curve"], dtype=float)
    gap_m = float(np.interp(float(data.qpos[qpos_address]), curve[:, 0], curve[:, 1])) / 100.0
    gap_range = np.asarray(block["gap_range_m"], dtype=float)
    if not gap_range[0] - 1e-9 <= gap_m <= gap_range[1] + 1e-9:
        raise ValueError("measured gripper gap is outside collision proxy range")
    visual = block.get("visual_model", {})
    fingers = visual.get("finger_bodies", [])
    if len(fingers) != 2:
        raise ValueError("visible finger bodies are missing for collision proxy")
    for row in fingers:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, str(row["name"]))
        if body_id < 0:
            raise ValueError(f"visible finger body is missing: {row['name']}")
        model.body_pos[body_id] = np.asarray([
            float(row["sign"]) * gap_m / 2.0, 0.0, 0.0])
    return True


def verify_against_config(model: mujoco.MjModel, cfg: dict[str, Any]) -> list[str]:
    """Compare compiled joint limits with the config; return mismatch messages.
    컴파일된 관절 한계를 설정과 대조하고 불일치 목록을 반환한다."""
    problems: list[str] = []
    for spec in joint_specs(cfg):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.name)
        if jid < 0:
            problems.append(f"{spec.name}: joint not found in compiled model")
            continue
        lo, hi = model.jnt_range[jid]
        # rtol=0 on purpose: the default rtol=1e-5 would wave through a ~2.8e-5 rad
        # difference on wrist_roll — looser than the ~3e-6 rad ctrlrange rounding
        # this check exists to catch.
        # rtol=0 은 의도적이다. 기본 rtol=1e-5 는 wrist_roll 기준 약 2.8e-5 rad 차이를
        # 통과시켜, 이 검사가 잡으려는 ctrlrange 반올림 차이(~3e-6 rad)보다 느슨해진다.
        if not (
            np.isclose(lo, spec.lo, rtol=0.0, atol=1e-9)
            and np.isclose(hi, spec.hi, rtol=0.0, atol=1e-9)
        ):
            problems.append(
                f"{spec.name}: mjcf=({lo!r}, {hi!r}) != config=({spec.lo!r}, {spec.hi!r})"
            )
    return problems


def normalize(
    q_rad: np.ndarray, cfg: dict[str, Any], clip: bool = False
) -> np.ndarray:
    """Map physical simulator joints to the shared data-contract range.

    J1..J5 use their radian limits. Channel 6 is *not* a hinge angle in the
    contract: the measured MuJoCo gap curve converts it to full jaw gap metres
    before the common 0..0.09m -> [-1,1] mapping.
    J1..J5 는 rad 범위를 쓰고, 6번은 hinge 각도가 아니라 전체 gap(m) 계약이다.

    MEASURED: joint limits in MuJoCo are soft, so contact forces push qpos a
    little past the range and this mapping then emits values outside [-1, 1].
    Observed max +1.0062 during scripted collection — one episode in 17 failed
    contract validation because of it. Real encoders past a calibrated limit will
    do the same thing, so this is not a simulator artefact to paper over.
    실측: MuJoCo 의 관절 한계는 soft 라 접촉력이 qpos 를 범위 밖으로 밀어내고,
    그러면 이 매핑이 [-1,1] 을 벗어난 값을 낸다. 스크립트 수집 중 최대 +1.0062
    관측, 그 때문에 17개 중 1개가 계약 검증에 실패했다. 실물 엔코더도 캘리브된
    한계를 넘으면 같은 일이 생긴다. 시뮬 특유의 잡음이 아니다.

    `clip` is False by default so callers see the real value. Whoever writes a
    dataset must decide explicitly — clip, widen the range, or reject the
    episode — and that decision belongs to the data contract (project-task),
    not to this function.
    기본값 False 로 두어 호출자가 실제 값을 보게 한다. 데이터셋을 쓰는 쪽이
    명시적으로 정해야 한다 — 클립할지, 범위를 넓힐지, 에피소드를 버릴지.
    그 결정은 데이터 계약(project-task)의 몫이지 이 함수의 몫이 아니다.
    """
    specs = joint_specs(cfg)
    lo = np.array([s.lo for s in specs], dtype=np.float32)
    hi = np.array([s.hi for s in specs], dtype=np.float32)
    out = (2.0 * (np.asarray(q_rad, dtype=np.float32) - lo) / (hi - lo) - 1.0).astype(np.float32)
    curve = np.asarray(cfg["grasp"]["gap_curve"], dtype=np.float32)
    gap_m = np.interp(np.asarray(q_rad, dtype=np.float32)[..., 5],
                      curve[:, 0], curve[:, 1]) / 100.0
    out[..., 5] = 2.0 * gap_m / 0.09 - 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32) if clip else out


def denormalize(q_norm: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Convert contract actions to simulator joint targets.

    The gripper channel is first decoded as gap metres, then mapped through the
    inverse measured gap curve to the MuJoCo hinge target. Values outside the
    simulated physical gap endpoints saturate at the nearest measured endpoint.
    """
    specs = joint_specs(cfg)
    lo = np.array([s.lo for s in specs], dtype=np.float32)
    hi = np.array([s.hi for s in specs], dtype=np.float32)
    norm = np.asarray(q_norm, dtype=np.float32)
    out = ((norm + 1.0) / 2.0 * (hi - lo) + lo).astype(np.float32)
    curve = np.asarray(cfg["grasp"]["gap_curve"], dtype=np.float32)
    gap_cm = ((norm[..., 5] + 1.0) * 0.045) * 100.0
    out[..., 5] = np.interp(gap_cm, curve[:, 1], curve[:, 0])
    return out


def dls_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xyz: np.ndarray,
    site_name: str = "gripperframe",
    max_iters: int = 300,
    damping: float = 1e-2,
    tol: float = 1e-3,
) -> tuple[np.ndarray, float, bool]:
    """Damped least-squares IK for the gripper site, clamped to joint limits.
    그리퍼 site에 대한 감쇠최소자승 IK. 관절 한계로 클램프한다.

    Returns (qpos_arm, final_position_error_m, within_limits).
    반환: (팔 관절각, 최종 위치오차 m, 관절한계 내 여부).
    이건 '실행가능성 검사'의 씨앗이다 — 목표 자세가 애초에 도달 불가면 여기서 걸린다.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"site {site_name!r} not found")
    n_arm = 6
    lo = model.jnt_range[:n_arm, 0].copy()
    hi = model.jnt_range[:n_arm, 1].copy()
    jacp = np.zeros((3, model.nv))
    target = np.asarray(target_xyz, dtype=float)

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[sid]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        j = jacp[:, :n_arm]
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), err)
        data.qpos[:n_arm] = np.clip(data.qpos[:n_arm] + dq, lo, hi)

    mujoco.mj_forward(model, data)
    final_err = float(np.linalg.norm(target - data.site_xpos[sid]))
    q = data.qpos[:n_arm].copy()
    within = bool(np.all(q > lo + 1e-6) and np.all(q < hi - 1e-6))
    return q, final_err, within


def solve_ik(
    model: mujoco.MjModel,
    target_xyz: np.ndarray,
    q_init: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, float, bool]:
    """Run :func:`dls_ik` on a scratch state so the live sim is untouched.
    별도 상태에서 dls_ik를 돌린다. 진행 중인 시뮬을 건드리지 않는다."""
    scratch = mujoco.MjData(model)
    if q_init is not None:
        scratch.qpos[:6] = np.asarray(q_init, dtype=float)
    return dls_ik(model, scratch, target_xyz, **kwargs)


if __name__ == "__main__":
    config = load_config()
    mj_model = build_model(config)
    mismatches = verify_against_config(mj_model, config)
    print(f"nq={mj_model.nq} nu={mj_model.nu} ncam={mj_model.ncam}")
    print("cameras:", [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(mj_model.ncam)
    ])
    print("config mismatches:", mismatches or "none")
