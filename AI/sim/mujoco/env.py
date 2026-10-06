"""MuJoCo implementation of `sim/base.py`'s `RobotEnv`.
`sim/base.py` 의 `RobotEnv` 를 MuJoCo 로 구현한 것.

The protocol itself is NOT here — it is in `sim/base.py`, deliberately free of
any MuJoCo import, so a second backend (Isaac Sim) or the real arm satisfies the
same surface. This file is one backend among those.
프로토콜 자체는 여기 없다. `sim/base.py` 에 있고 MuJoCo 임포트가 의도적으로
없다. 두 번째 백엔드(Isaac Sim)든 실물 팔이든 같은 표면을 만족시키게 하려는
것이다. 이 파일은 그중 백엔드 하나다.

The real-robot implementation does not exist yet — project-task
([ROS] 정책 ckpt 로드→추론 노드). What the protocol guarantees is that it has
one shape to fill in.
실물 구현은 아직 존재하지 않는다 — project-task. 프로토콜이 보장하는 것은
채워 넣을 형태가 하나로 정해져 있다는 것뿐이다.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from sim.base import Observation, RobotEnv  # noqa: F401 — 재수출: 하위 호환용 아님, 타입 참조용
from sim.mujoco.build_scene import (
    active_gripper_pad_ids,
    build_model,
    denormalize,
    load_config,
    normalize,
    sync_gripper_collision_proxy,
)
from sim.mujoco.domain import DomainRandomizer


class MujocoPickEnv:
    """`RobotEnv` backed by the MuJoCo pick scene.
    MuJoCo 픽 씬으로 구현한 `RobotEnv`.

    Rendering is the expensive part — about 60 ms per tick for two 224x224
    cameras on CPU. `render=False` skips it for policies that ignore images,
    which is what makes a 200-rollout baseline sweep finish in minutes instead
    of hours. The observation then carries zero-filled image arrays of the
    correct shape, so a policy that *does* look at images cannot silently
    receive a differently-shaped input.
    렌더링이 비싸다 — CPU에서 224x224 카메라 2대 기준 틱당 약 60ms.
    이미지를 안 보는 정책에는 `render=False` 로 건너뛴다. 200회 롤아웃 스윕이
    시간 단위가 아니라 분 단위로 끝나는 이유다. 이때 관측에는 올바른 shape 의
    0으로 채운 배열이 들어가므로, 이미지를 **보는** 정책이 다른 shape 를 조용히
    받는 일은 생기지 않는다.
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        render: bool = True,
        object_jitter_m: float = 0.05,
        max_ticks: int = 200,
        domain: DomainRandomizer | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self._render = render
        self._jitter = float(object_jitter_m)
        self.max_ticks = int(max_ticks)
        # Bound once against the compiled model, re-applied every reset. None
        # means the nominal model, which is condition A of the transfer measure.
        # 컴파일된 모델에 한 번 결속하고 리셋마다 다시 적용한다. None 이면
        # 공칭 모델이고, 그것이 전이 계측의 조건 A 다.
        self._domain = domain
        if self._domain is not None:
            self._domain.bind(self.model)

        self._width, self._height = self.cfg["cameras"]["resolution"]
        self._rate = float(self.cfg["control"]["rate_hz"])
        self._substeps = max(1, int((1.0 / self._rate) / self.model.opt.timestep))
        self._cams = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        ]
        self._renderer = (
            mujoco.Renderer(self.model, height=self._height, width=self._width)
            if render
            else None
        )

        self._obj_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self._obj_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom"
        )
        self._obj_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free"
        )
        self._obj_qadr = self.model.jnt_qposadr[self._obj_joint]
        self._jaw_geoms = self._collect_jaw_geoms()
        self._base_xy = np.asarray(self.cfg["task"]["object"]["init_pos"][:2], dtype=float)
        self._success_lift = float(self.cfg["grasp"]["success_lift_m"])

        self._t0_z = 0.0
        self._ticks = 0

    # ---- RobotEnv -------------------------------------------------------

    @property
    def control_rate_hz(self) -> float:
        """Ticks per second the policy is called at.
        정책이 호출되는 주기."""
        return self._rate

    @property
    def camera_names(self) -> list[str]:
        """Cameras present in the observation.
        관측에 들어오는 카메라."""
        return list(self._cams)

    def reset(
        self, seed: int | None = None, object_xy: tuple[float, float] | None = None,
        initial_q_rad: np.ndarray | None = None,
    ) -> Observation:
        """Randomise the object position and settle the scene.
        물체 위치를 무작위로 놓고 씬을 안정시킨다.

        `object_xy` overrides the randomisation. It exists so a recorded episode
        can be replayed against the exact condition it was recorded under —
        without that, a replay baseline scoring 0% tells you nothing about the
        task, only that the object moved.
        `object_xy` 를 주면 무작위화를 덮어쓴다. 기록된 에피소드를 그것이 기록된
        **바로 그 조건**에서 재생하기 위한 것이다. 이게 없으면 replay baseline 의
        0% 는 태스크에 대해 아무것도 말해주지 않는다. 물체가 움직였다는 것만 말해준다.
        """
        mujoco.mj_resetData(self.model, self.data)
        if initial_q_rad is not None:
            initial_q_rad = np.asarray(initial_q_rad, dtype=float).reshape(6)
            if not np.isfinite(initial_q_rad).all():
                raise ValueError("initial_q_rad must contain six finite values")
            self.data.qpos[:6] = initial_q_rad
        if self._domain is not None:
            self._domain.apply(self.model, self.data, seed=seed)
        rng = np.random.default_rng(seed)
        xy = (
            np.asarray(object_xy, dtype=float)
            if object_xy is not None
            else self._base_xy + rng.uniform(-self._jitter, self._jitter, size=2)
        )
        self.data.qpos[self._obj_qadr] = xy[0]
        self.data.qpos[self._obj_qadr + 1] = xy[1]
        sync_gripper_collision_proxy(self.model, self.data, self.cfg)
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self.data.qpos[:6]
        for _ in range(int(0.4 / self.model.opt.timestep)):
            sync_gripper_collision_proxy(self.model, self.data, self.cfg)
            mujoco.mj_step(self.model, self.data)
        sync_gripper_collision_proxy(self.model, self.data, self.cfg)
        mujoco.mj_forward(self.model, self.data)
        self._t0_z = float(self.data.xpos[self._obj_body][2])
        self._ticks = 0
        return self._observe()

    def step(self, action: np.ndarray) -> Observation:
        """Apply one normalized action and advance one control tick.
        정규화된 행동 하나를 적용하고 제어 틱 하나만큼 진행한다."""
        action = np.asarray(action, dtype=np.float32).reshape(6)
        if not np.isfinite(action).all():
            raise ValueError(f"action must be finite, got {action}")
        # Clipping here is a safety limit, not a data-contract decision. The
        # contract question (what to do with out-of-range values in recorded
        # data) is open — see project-task.
        # 여기서의 클립은 안전 제한이지 데이터 계약 결정이 아니다. 기록 데이터의
        # 범위 초과 처리 문제는 미결이다 — project-task 참조.
        self.data.ctrl[:6] = denormalize(np.clip(action, -1.0, 1.0), self.cfg)
        for _ in range(self._substeps):
            sync_gripper_collision_proxy(self.model, self.data, self.cfg)
            mujoco.mj_step(self.model, self.data)
        sync_gripper_collision_proxy(self.model, self.data, self.cfg)
        mujoco.mj_forward(self.model, self.data)
        self._ticks += 1
        return self._observe()

    def is_success(self) -> bool:
        """Object lifted past the threshold and still held.
        물체가 기준 높이 이상 올라갔고 아직 잡혀 있는가."""
        lifted = float(self.data.xpos[self._obj_body][2]) - self._t0_z
        contact_ok = (
            self.has_bilateral_jaw_contact()
            if (self.cfg.get("gripper_pads") or {}).get("kind")
            == "symmetric_parallel_jaw_runtime_proxy"
            else self._n_jaw_contacts() > 0
        )
        return lifted >= self._success_lift and contact_ok

    # ---- simulator-only accessors --------------------------------------
    # A policy must not call these. They exist for scripted baselines and for
    # scoring, where reading privileged state is the point.
    # 정책이 호출하면 안 된다. 스크립트 baseline 과 채점용이다.
    # 특권 정보를 읽는 것 자체가 목적인 자리다.

    def object_position(self) -> np.ndarray:
        """Ground-truth object position — privileged, simulator only.
        물체의 정답 위치. 특권 정보이며 시뮬에만 존재한다."""
        return self.data.xpos[self._obj_body].copy()

    def object_rotation(self) -> np.ndarray:
        """Ground-truth object rotation matrix — privileged, simulator only.
        물체의 정답 회전행렬. 특권 정보이며 시뮬에만 존재한다."""
        return self.data.xmat[self._obj_body].reshape(3, 3).copy()

    @property
    def gripper_index(self) -> int:
        """Index of the gripper in the action vector, from the hardware config.
        행동 벡터에서 그리퍼의 위치. 하드웨어 설정에서 읽는다."""
        for j in self.cfg["joints"]:
            if j["name"] == "gripper":
                return int(j["index"])
        raise KeyError("configs 의 joints 에 'gripper' 가 없다")

    def pinch_to_object_m(self) -> tuple[float, float]:
        """Distance from the gripper's pinch pocket to the object centre — privileged.
        그리퍼 파지 포켓에서 물체 중심까지의 거리. 특권 정보.

        Returns (horizontal xy distance, full 3-D distance) in metres. The pinch
        pocket is `grasp.pinch_offset_local` from configs/so101.yaml -- the same
        point the scripted policy aims at -- so this number says how close a
        learned policy came to the place a grasp actually happens, not to the
        fingertip. Measured for the failure-shape question: a 0% policy that
        gets within 15 mm and one that never gets within 50 mm need different fixes.
        (수평 xy 거리, 3차원 거리) 를 m 로 돌려준다. 파지 포켓은 so101.yaml 의
        `grasp.pinch_offset_local` — 스크립트 정책이 조준하는 바로 그 점 — 이라서,
        이 수치는 학습 정책이 손끝이 아니라 **실제로 잡히는 자리**에 얼마나 가까이
        왔는지를 말한다. 실패의 형태를 묻기 위한 값이다. 15mm 안까지 오는 0% 와
        50mm 안에도 못 오는 0% 는 처방이 다르다.
        """
        from sim.mujoco.kinematics import grasp_point, pinch_offset_for_gap

        curve = np.asarray(self.cfg["grasp"]["gap_curve"], dtype=float)
        gap_m = float(np.interp(
            float(self.data.qpos[self.gripper_index]),
            curve[:, 0], curve[:, 1],
        )) / 100.0
        pinch = grasp_point(
            self.model, self.data,
            pinch_offset_for_gap(self.cfg["grasp"], gap_m),
        )
        obj = self.data.xpos[self._obj_body]
        d = pinch - obj
        return float(np.hypot(d[0], d[1])), float(np.linalg.norm(d))

    def jaw_contacts(self) -> int:
        """Number of jaw-object contacts right now — privileged, for scoring.
        현재 턱-물체 접촉 수. 특권 정보, 채점용."""
        return self._n_jaw_contacts()

    def jaw_contact_counts(self) -> dict[str, int]:
        """Per-pad object contact counts — privileged diagnostic state.

        A total contact count cannot distinguish a body pinch from one finger
        merely pushing the object.  Keep the pad names in the result so ver1
        recorded-oracle reports can prove that both sides participated.
        """
        counts = {
            str(mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)): 0
            for geom_id in self._jaw_geoms
        }
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == self._obj_geom and con.geom2 in self._jaw_geoms:
                geom_id = int(con.geom2)
            elif con.geom2 == self._obj_geom and con.geom1 in self._jaw_geoms:
                geom_id = int(con.geom1)
            else:
                continue
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            counts[str(name)] += 1
        return counts

    def has_bilateral_jaw_contact(self) -> bool:
        """Whether every configured jaw pad currently contacts the object."""
        counts = self.jaw_contact_counts()
        return len(counts) >= 2 and all(value > 0 for value in counts.values())

    def lift_height(self) -> float:
        """How far the object has risen from its settled height.
        물체가 안착 높이에서 얼마나 올라갔는가."""
        return float(self.data.xpos[self._obj_body][2]) - self._t0_z

    def joint_positions(self) -> np.ndarray:
        """Raw joint angles in radians.
        관절각 원값 (rad)."""
        return self.data.qpos[:6].copy()

    def close(self) -> None:
        """Release the renderer.
        렌더러를 해제한다."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "MujocoPickEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- internals ------------------------------------------------------

    def _collect_jaw_geoms(self) -> set[int]:
        return active_gripper_pad_ids(self.model, self.cfg)

    def _n_jaw_contacts(self) -> int:
        n = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if self._obj_geom in pair and pair & self._jaw_geoms:
                n += 1
        return n

    def displace_joints(self, delta_rad: np.ndarray) -> Observation:
        """Move the arm to an off-nominal configuration and re-observe.
        팔을 명목 배치 밖으로 옮기고 다시 관측한다.

        Not a disturbance model -- it chooses **where an episode starts**. DAgger
        asks the expert "what would you do here", and "here" means joint
        configurations the demonstrations never contain. Injecting a few wrong
        commands does not produce one: under position control the next command
        pulls the arm back, so both an open-loop plan and a feedback controller
        recover trivially. Measured 2026-09-07 🟢 -- open-loop `scripted` scored
        the same perturbed as undisturbed. Setting `qpos` is the honest way to
        put the robot somewhere it has never been.
        교란 모델이 아니다. **에피소드가 어디서 시작하는지**를 고르는 것이다.
        DAgger 는 전문가에게 "여기서 뭘 하겠나"를 묻고, 그 "여기"는 시연에 없는
        관절 배치를 뜻한다. 잘못된 명령 몇 개를 주입해서는 그런 상태가 안 만들어진다 —
        위치 제어에서는 다음 명령이 팔을 되돌리므로 개루프 계획도 피드백 제어기도
        똑같이 복구한다. 2026-09-07 실측 🟢: 개루프 `scripted` 가 교란 유무에
        같은 점수를 냈다. `qpos` 를 직접 놓는 것이 로봇을 가본 적 없는 자리에
        두는 정직한 방법이다.
        """
        delta = np.asarray(delta_rad, dtype=float)
        lo = self.model.jnt_range[:5, 0]
        hi = self.model.jnt_range[:5, 1]
        self.data.qpos[:5] = np.clip(self.data.qpos[:5] + delta[:5], lo, hi)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._observe()

    def _observe(self) -> Observation:
        images: dict[str, np.ndarray] = {}
        for cam in self._cams:
            if self._renderer is None:
                images[cam] = np.zeros((3, self._height, self._width), dtype=np.uint8)
                continue
            self._renderer.update_scene(self.data, camera=cam)
            images[cam] = np.transpose(self._renderer.render(), (2, 0, 1)).copy()
        return Observation(
            images=images,
            state=normalize(self.data.qpos[:6].copy(), self.cfg),
            timestamp=float(self.data.time),
        )
