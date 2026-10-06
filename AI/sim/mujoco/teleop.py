"""Drive the arm yourself in the viewer, and record what you did as episodes.
뷰어에서 팔을 직접 조작하고, 한 일을 에피소드로 기록한다.

Why this exists: the collected dataset is unusable for training because the
trajectories came from an open-loop script — one path, no diversity, so a model
memorises it instead of learning the task. Human-driven demonstrations fix that,
and they do not need the physical robot.
이게 있는 이유: 수집된 데이터셋이 학습에 못 쓰이는 까닭은 궤적이 개루프 스크립트에서
나왔기 때문이다. 경로가 하나이고 다양성이 없어서, 모델이 태스크를 배우는 대신
그 궤적을 외운다. 사람이 조작한 시연이 그 문제를 풀고, 실물 로봇이 필요 없다.

This is the sim counterpart of the real leader-follower rig: on hardware you move
the leader arm by hand and that IS the demonstration. Here you move it by key.
실물 리더-팔로워 구조의 시뮬 대응물이다. 하드웨어에서는 리더 암을 손으로 움직이면
그것이 곧 시연이고, 여기서는 키로 움직인다.

⚠️ 수집한 궤적의 품질은 조작자에게 달려 있다. 지침의 원칙이 그대로 적용된다:
   **환경은 다양하게, 시연 방식은 동일하게.** 물체 위치는 매번 바꾸되 접근 궤적·
   속도·그립 위치는 일관되게 유지한다. 매번 다르면 다양성이 아니라 노이즈다.

⚠️ 물체는 도달 가능 영역 안에만 둔다. 그 밖에서 찍은 에피소드는 로봇이 재현할 수
   없어 폐기 대상이다. 현재 실측 영역은 README §8 참조.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from contract.episode import CONTRACT_VERSION, Episode, EpisodeMeta, validate, write_episode
from contract.episode import write_dataset_index
from data.collect import EpisodeRecorder
from paths import AI_ROOT, DEFAULT_CONFIG
from sim.mujoco.build_scene import build_model, denormalize, load_config, normalize
from sim.mujoco.kinematics import solve_pose_ik
from tracking.exp_log import _git_rev, file_digest

# GLFW key codes the viewer hands to key_callback.
# 뷰어가 key_callback 으로 넘겨주는 GLFW 키 코드.
KEY = {
    "W": 87, "S": 83, "A": 65, "D": 68, "Q": 81, "E": 69,
    "R": 82, "F": 70, "G": 71, "T": 84, "N": 78, "M": 77,
    "O": 79, "C": 67, "X": 88, "H": 72,
    "1": 49, "2": 50, "3": 51, "4": 52, "5": 53, "6": 54,
    "LEFT": 263, "RIGHT": 262, "UP": 265, "DOWN": 264,
}

HELP = """
조작 — 데카르트 모드 (기본)
  W / S        파지점 +x / -x  (앞 / 뒤)
  A / D        파지점 +y / -y  (좌 / 우)
  Q / E        파지점 +z / -z  (위 / 아래)
  O / C        그리퍼 열기 / 닫기
  ← / →        wrist_roll 회전

조작 — 관절 모드
  1~6          제어할 관절 선택
  ↑ / ↓        선택한 관절 +/-
  M            데카르트 ↔ 관절 모드 전환

기록
  R            녹화 시작/정지 (정지 시 계약 검증 후 저장)
  X            녹화 취소 (버린다)
  N            새 에피소드 — 물체를 무작위 위치로, 팔은 초기 자세
  T            이동 스텝 크기 순환 (1 / 5 / 10 mm)
  G            도달 가능 영역 안에 있는지 표시
  H            이 도움말 다시 출력

참고: 키를 눌러도 팔이 즉시 가지 않는다. 관절 속도가 제한돼 있어서 목표를 향해
      천천히 움직인다 (기본 1.5 rad/s, --max-joint-speed 로 변경). 제한이 없으면
      팔이 자기 목표를 추종하지 못하고 물체를 쓸어버린다.

⚠️ 녹화 중에는 물체를 마우스로 끌지 마라. 손으로 옮긴 것은 조작이 아니라
   외력이고, 정책이 재현할 수 없는 궤적이 된다.
"""


@dataclass
class TeleopState:
    """Everything the key handler mutates. Kept separate so it can be driven
    programmatically in a test, without a window.
    키 핸들러가 바꾸는 전부. 창 없이 테스트에서 프로그램으로 구동할 수 있도록
    분리해 둔다."""

    target: np.ndarray                  # 파지점 목표 (world, m)
    q_goal: np.ndarray | None = None    # 목표 관절각 5개. tick() 이 여기로 천천히 간다
    wrist_roll: float = 0.0
    gripper_cmd: float = 0.60
    step_m: float = 0.005
    joint_mode: bool = False
    joint_idx: int = 0
    recording: bool = False
    quit_episode: bool = False
    discard: bool = False
    messages: list[str] = field(default_factory=list)
    ik_failures: int = 0

    def say(self, text: str) -> None:
        self.messages.append(text)
        print(f"  {text}")


STEP_CYCLE = (0.001, 0.005, 0.010)


class Teleop:
    """Keyboard control of the arm, with contract-conforming recording.
    키보드로 팔을 조작하고 계약을 만족하는 형태로 기록한다."""

    def __init__(
        self,
        cfg: dict[str, Any],
        out_dir: Path,
        author: str,
        render: bool = True,
        max_joint_speed: float = 1.5,
    ) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.author = author
        self.model = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.rate_hz = float(cfg["control"]["rate_hz"])
        self.substeps = max(1, int((1.0 / self.rate_hz) / self.model.opt.timestep))
        # A keypress must not teleport the command. Without this the arm cannot
        # track its own target and sweeps the object away — the headless test
        # caught exactly that: the cube was pushed 7 cm.
        # 키 입력이 명령을 순간이동시키면 안 된다. 이게 없으면 팔이 자기 목표를
        # 추종하지 못하고 물체를 쓸어버린다. 헤드리스 테스트가 그것을 잡았다 —
        # 큐브가 7cm 밀려났다.
        # ⚠️ 이 값은 도구 수준의 제한이고 실물 서보 사양이 아니다. 실물 속도는 미계측.
        self.max_joint_step = float(max_joint_speed) / self.rate_hz
        g = cfg["grasp"]
        self.offset = np.asarray(g["pinch_offset_local"], dtype=float)
        self.axis = np.asarray(g["approach_axis"], dtype=float)
        self.open_cmd = float(g["open_cmd"])
        self.close_cmd = float(g["close_cmd"])
        self.success_lift = float(g["success_lift_m"])
        self.base_xy = np.asarray(cfg["task"]["object"]["init_pos"][:2], dtype=float)

        w, h = cfg["cameras"]["resolution"]
        self.renderer = mujoco.Renderer(self.model, height=h, width=w) if render else None

        self.obj_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.obj_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
        self.obj_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free"
        )
        self.obj_qadr = self.model.jnt_qposadr[self.obj_joint]
        self.jaw_geoms = self._jaw_geoms()

        self.recorder: EpisodeRecorder | None = None
        self.state: TeleopState | None = None
        self.n_saved = 0
        self.n_rejected = 0
        self.object_xy = self.base_xy.copy()
        self.t0_z = 0.0

    # ---- scene ----------------------------------------------------------

    def reset(self, rng: np.random.Generator, jitter_m: float) -> None:
        """Put the object somewhere new and the arm back to its start pose.
        물체를 새 위치에 놓고 팔을 시작 자세로 되돌린다."""
        mujoco.mj_resetData(self.model, self.data)
        self.object_xy = self.base_xy + rng.uniform(-jitter_m, jitter_m, size=2)
        self.data.qpos[self.obj_qadr] = self.object_xy[0]
        self.data.qpos[self.obj_qadr + 1] = self.object_xy[1]
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self.data.qpos[:6]
        for _ in range(int(0.4 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.t0_z = float(self.data.xpos[self.obj_body][2])

        obj = self.data.xpos[self.obj_body].copy()
        start = obj + np.array([0.0, 0.0, float(self.cfg["grasp"]["approach_height_m"])])
        self.state = TeleopState(
            target=start, q_goal=self.data.ctrl[:5].copy(), gripper_cmd=self.open_cmd
        )
        self.recorder = None

    def _jaw_geoms(self) -> set[int]:
        ids: set[int] = set()
        for name in ("gripper", "moving_jaw_so101_v1"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            ids |= {
                gi for gi in range(self.model.ngeom)
                if self.model.geom_bodyid[gi] == bid and self.model.geom_contype[gi] != 0
            }
        return ids

    def n_contacts(self) -> int:
        n = 0
        for i in range(self.data.ncon):
            pair = {self.data.contact[i].geom1, self.data.contact[i].geom2}
            if self.obj_geom in pair and pair & self.jaw_geoms:
                n += 1
        return n

    def lift_height(self) -> float:
        return float(self.data.xpos[self.obj_body][2]) - self.t0_z

    def is_success(self) -> bool:
        return self.lift_height() >= self.success_lift and self.n_contacts() > 0

    # ---- control --------------------------------------------------------

    def on_key(self, code: int) -> None:
        """Apply one keypress. Pure state mutation — no stepping, no drawing.
        키 입력 하나를 적용한다. 상태만 바꾼다. 스텝도 그리기도 하지 않는다."""
        s = self.state
        assert s is not None
        d = s.step_m

        if s.joint_mode:
            for i, k in enumerate("123456"):
                if code == KEY[k]:
                    s.joint_idx = i
                    s.say(f"관절 {i} 선택")
                    return
            if code in (KEY["UP"], KEY["DOWN"]):
                sign = 1.0 if code == KEY["UP"] else -1.0
                lo, hi = self.model.jnt_range[s.joint_idx]
                base = (s.q_goal[s.joint_idx] if (s.q_goal is not None and s.joint_idx < 5)
                        else self.data.ctrl[s.joint_idx])
                q = base + sign * 0.02
                if q < lo or q > hi:
                    s.say(f"관절 {s.joint_idx} 한계 도달 ({lo:.3f}, {hi:.3f})")
                if s.q_goal is None:
                    s.q_goal = self.data.ctrl[:5].copy()
                if s.joint_idx < 5:
                    s.q_goal[s.joint_idx] = float(np.clip(q, lo, hi))
                else:
                    s.gripper_cmd = float(np.clip(q, lo, hi))
                return
        else:
            deltas = {
                KEY["W"]: (0, +d), KEY["S"]: (0, -d),
                KEY["A"]: (1, +d), KEY["D"]: (1, -d),
                KEY["Q"]: (2, +d), KEY["E"]: (2, -d),
            }
            if code in deltas:
                axis, amount = deltas[code]
                probe = s.target.copy()
                probe[axis] += amount
                res = solve_pose_ik(
                    self.model, probe, self.offset, self.axis,
                    q_init=self.data.qpos[:6], wrist_roll=s.wrist_roll,
                )
                if res.ok:
                    s.target = probe
                    s.q_goal = res.qpos[:5].copy()
                else:
                    s.ik_failures += 1
                    s.say(
                        f"IK 도달 불가 — 위치오차 {res.pos_error_m * 1000:.1f}mm "
                        f"접근축오차 {res.axis_error_deg:.1f}° "
                        f"관절한계내={res.within_limits}. 목표를 옮기지 않았다"
                    )
                return
            if code in (KEY["LEFT"], KEY["RIGHT"]):
                s.wrist_roll += (0.1 if code == KEY["RIGHT"] else -0.1)
                lo, hi = self.model.jnt_range[4]
                s.wrist_roll = float(np.clip(s.wrist_roll, lo, hi))
                res = solve_pose_ik(
                    self.model, s.target, self.offset, self.axis,
                    q_init=self.data.qpos[:6], wrist_roll=s.wrist_roll,
                )
                if res.ok:
                    s.q_goal = res.qpos[:5].copy()
                else:
                    s.say(f"wrist_roll {s.wrist_roll:+.2f} 에서 IK 실패")
                return

        if code == KEY["O"]:
            s.gripper_cmd = self.open_cmd
            s.say(f"그리퍼 열기 ({self.open_cmd})")
        elif code == KEY["C"]:
            s.gripper_cmd = self.close_cmd
            s.say(f"그리퍼 닫기 ({self.close_cmd})")
        elif code == KEY["M"]:
            s.joint_mode = not s.joint_mode
            s.say(f"{'관절' if s.joint_mode else '데카르트'} 모드")
        elif code == KEY["T"]:
            i = (STEP_CYCLE.index(s.step_m) + 1) % len(STEP_CYCLE)
            s.step_m = STEP_CYCLE[i]
            s.say(f"스텝 {s.step_m * 1000:.0f}mm")
        elif code == KEY["G"]:
            self._report_reachable()
        elif code == KEY["H"]:
            print(HELP)
        elif code == KEY["R"]:
            self._toggle_record()
        elif code == KEY["X"]:
            if s.recording:
                s.recording = False
                s.discard = True
                self.recorder = None
                s.say("녹화 취소 — 버렸다")
        elif code == KEY["N"]:
            s.quit_episode = True

    def _report_reachable(self) -> None:
        s = self.state
        assert s is not None
        obj = self.data.xpos[self.obj_body]
        # 실측된 최대 연속 가능 영역 (README §8). 하드코딩이 아니라 계측 결과의 인용이다.
        x_ok = 0.10 <= obj[0] <= 0.25
        y_ok = -0.175 <= obj[1] <= 0.175
        verdict = "안" if (x_ok and y_ok) else "**밖 — 이 에피소드는 폐기 대상이다**"
        s.say(
            f"물체 ({obj[0]:+.3f}, {obj[1]:+.3f}) 는 실측 도달영역 "
            f"x[0.10,0.25] y[±0.175] {verdict}"
        )

    def _toggle_record(self) -> None:
        s = self.state
        assert s is not None
        if s.recording:
            s.recording = False
            s.say("녹화 정지")
            self._save()
        else:
            if self.renderer is None:
                s.say("렌더러가 없어 녹화할 수 없다 (--no-render 로 실행됨)")
                return
            self.recorder = EpisodeRecorder(self.model, self.cfg, self.renderer)
            s.recording = True
            s.say("녹화 시작 — 다시 R 을 누르면 저장한다")

    def tick(self) -> None:
        """Advance one control tick, capturing a frame if recording.
        제어 틱 하나 진행. 녹화 중이면 한 프레임 담는다."""
        s = self.state
        assert s is not None
        if s.q_goal is not None:
            cur = self.data.ctrl[:5]
            delta = np.clip(s.q_goal - cur, -self.max_joint_step, self.max_joint_step)
            self.data.ctrl[:5] = cur + delta
        self.data.ctrl[5] = s.gripper_cmd
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if s.recording and self.recorder is not None:
            self.recorder.capture(self.data)

    # ---- saving ---------------------------------------------------------

    def _save(self) -> None:
        s = self.state
        assert s is not None
        if self.recorder is None or not self.recorder.state:
            s.say("기록된 프레임이 없다. 저장하지 않는다")
            return
        obj = self.data.xpos[self.obj_body]
        # bool() 로 감싼다 — numpy 비교는 np.bool_ 을 내고 json 이 직렬화하지 못한다.
        # 검증 테스트가 이걸 잡았다.
        in_region = bool(0.10 <= float(obj[0]) <= 0.25 and -0.175 <= float(obj[1]) <= 0.175)
        meta = EpisodeMeta(
            episode_id=f"teleop_{self.n_saved:05d}",
            task="pick_cube_2cm",
            source="sim",
            success=bool(self.is_success()),
            n_steps=0,
            control_rate_hz=self.rate_hz,
            cameras=[],
            contract_version=CONTRACT_VERSION,
            collected_by=self.author,
            config_sha=file_digest(DEFAULT_CONFIG),
            git_rev=_git_rev(),
            notes={
                "object_init_xy": [round(float(self.object_xy[0]), 5),
                                   round(float(self.object_xy[1]), 5)],
                "lift_height_m": round(self.lift_height(), 5),
                "contacts_at_end": int(self.n_contacts()),
                "simulator": "mujoco",
                "scripted": False,
                "teleop": True,
                "input": "keyboard",
                "ik_failures_during_episode": int(s.ik_failures),
                "object_in_measured_reach_region": in_region,
                "caveat": (
                    "사람이 키보드로 조작한 시뮬 시연이다. 실물 리더 암 시연이 아니다. "
                    "궤적 다양성은 있으나 조작 방식이 실물과 다르다."
                ),
            },
        )
        ep = self.recorder.build(meta)
        problems = validate(ep)
        if problems:
            self.n_rejected += 1
            s.say(f"계약 위반 {len(problems)}건 — 저장하지 않는다:")
            for p in problems:
                print(f"      - {p}")
            self.recorder = None
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = write_episode(ep, self.out_dir)
        self.n_saved += 1
        s.say(
            f"저장 {path.name} — {meta.n_steps}스텝, "
            f"파지 {'성공' if meta.success else '실패'}, "
            f"상승 {self.lift_height() * 100:.2f}cm, "
            f"도달영역 {'안' if in_region else '밖(폐기 대상)'}"
        )
        self.recorder = None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MuJoCo 뷰어에서 팔을 직접 조작하고 시연을 기록한다"
    )
    parser.add_argument("--out", type=Path, default=AI_ROOT / "datasets" / "sim_teleop_v0")
    parser.add_argument("--jitter", type=float, default=0.05,
                        help="새 에피소드 시 물체 xy 무작위 범위 (m)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    parser.add_argument("--max-joint-speed", type=float, default=1.5,
                        help="관절 최대 속도 (rad/s). 도구 제한이며 실물 사양이 아니다")
    args = parser.parse_args()

    from sim.mujoco.view import _check_display

    _check_display()
    import mujoco.viewer

    cfg = load_config()
    tele = Teleop(cfg, args.out, args.author, render=True,
                  max_joint_speed=args.max_joint_speed)
    rng = np.random.default_rng(args.seed)
    tele.reset(rng, args.jitter)
    print(HELP)
    print(f"저장 위치: {args.out}")
    print("⚠️ 뷰어는 실시간이다. 여기서 본 성공 몇 번은 성공률이 아니다.\n")

    pending: list[int] = []

    def on_key(code: int) -> None:
        pending.append(code)

    dt = 1.0 / tele.rate_hz
    slow_warned = False
    n_slow = 0
    try:
        with mujoco.viewer.launch_passive(
            tele.model, tele.data, key_callback=on_key
        ) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                while pending:
                    tele.on_key(pending.pop(0))
                assert tele.state is not None
                if tele.state.quit_episode:
                    if tele.state.recording:
                        tele.state.recording = False
                        tele._save()
                    tele.reset(rng, args.jitter)
                    print("  새 에피소드")
                tele.tick()
                viewer.sync()
                spent = time.perf_counter() - t0
                slack = dt - spent
                if slack > 0:
                    time.sleep(slack)
                elif spent > 2 * dt:
                    # Sim time still advances by exactly 1/rate per tick, so the
                    # recorded episode stays contract-valid — only the interaction
                    # feels slow. Say so instead of letting the user wonder.
                    # 시뮬 시간은 틱당 정확히 1/rate 씩 가므로 기록된 에피소드는
                    # 계약을 계속 만족한다. 조작감만 느려진다. 헷갈리지 않게 알린다.
                    n_slow += 1
                    if not slow_warned and n_slow > 15:
                        slow_warned = True
                        print(
                            f"\n⚠️ 실시간을 못 따라간다 — 틱당 {spent * 1000:.0f}ms "
                            f"(예산 {dt * 1000:.0f}ms). 녹화 중 이미지 렌더가 원인이다.\n"
                            "   기록된 데이터는 영향 없다 — 타임스탬프는 시뮬 시간이라\n"
                            "   계약(30Hz, 주기 흔들림 20% 이내)을 그대로 만족한다.\n"
                            "   조작감만 느려진다. GPU 렌더가 되는 환경이면 사라진다.\n"
                        )
    finally:
        tele.close()

    if tele.n_saved:
        write_dataset_index(args.out, extra={"collection": "teleop_keyboard"})
    print(f"\n저장 {tele.n_saved}건 / 계약 위반으로 거부 {tele.n_rejected}건 → {args.out}")
    print("검증: python tools/verify_dataset.py " + str(args.out) + " --log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
