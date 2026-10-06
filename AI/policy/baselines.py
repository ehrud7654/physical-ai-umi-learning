"""Baselines a learned policy has to be measured against.
학습된 정책이 반드시 대조되어야 할 기준선들.

Project rule: no training code before baselines are measured. The reason is
diagnostic, not ceremonial. Without the numbers below, a BC success rate of 40%
is uninterpretable — it could be excellent or it could be worse than standing
still, and you cannot tell which.
프로젝트 규칙: baseline 측정 전에 학습 코드를 쓰지 않는다. 의례가 아니라 진단
때문이다. 아래 수치가 없으면 BC 성공률 40% 는 해석이 불가능하다. 훌륭한
결과일 수도 있고 가만히 서 있는 것보다 못한 것일 수도 있는데, 구분할 수가 없다.

Four baselines, each answering a different question:
  Hold      가만히 있으면 몇 %인가 — 하한선. 이보다 낮으면 정책이 해롭다.
  Zero      의미 없이 움직이면 몇 %인가 — 우연 성공률.
  Replay    고정 궤적을 재생하면 몇 %인가 — 이 태스크가 관측을 봐야 하는
            태스크인지 판정한다. 높게 나오면 시각 정책이 필요 없다는 뜻이고,
            그건 태스크 설계를 다시 해야 한다는 신호다.
  Scripted  인식이 완벽하면 몇 %인가 — 상한선. 학습 정책이 넘을 수 없다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.mujoco.build_scene import denormalize, normalize
from sim.base import Observation
from sim.mujoco.env import MujocoPickEnv
from sim.mujoco.kinematics import grasp_point, pick_waypoints, solve_pose_ik
from policy.base import check_action


class HoldPolicy:
    """Command the joint positions observed at reset, forever.
    리셋 시점에 관측한 관절 위치를 계속 명령한다.

    The true do-nothing floor. Any policy scoring below this is actively worse
    than not moving.
    진짜 하한선. 이보다 낮은 정책은 안 움직이느니만 못한 것이다.
    """

    name = "hold"
    uses_privileged_state = False

    def __init__(self) -> None:
        self._target: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        """Forget the held pose.
        유지 중인 자세를 잊는다."""
        self._target = None

    def act(self, obs: Observation) -> np.ndarray:
        """Return the first observed state, unchanged.
        처음 관측한 상태를 그대로 반환한다."""
        if self._target is None:
            self._target = obs.state.copy()
        return check_action(self._target, self.name)


class ZeroPolicy:
    """Command the centre of every joint range.
    모든 관절 범위의 중앙을 명령한다.

    Moves, but toward nothing in particular. Separates "the policy learned to
    move usefully" from "moving at all sometimes works".
    움직이기는 하지만 아무 데도 향하지 않는다. "유용하게 움직이는 법을 배웠다"와
    "움직이기만 해도 가끔 된다"를 분리한다.
    """

    name = "zero"
    uses_privileged_state = False

    def reset(self, seed: int | None = None) -> None:
        """Nothing to reset.
        되돌릴 상태 없음."""

    def act(self, obs: Observation) -> np.ndarray:
        """Return an all-zero action.
        전부 0인 행동을 반환한다."""
        return check_action(np.zeros(6, dtype=np.float32), self.name)


class ReplayPolicy:
    """Replay a recorded action sequence, ignoring the observation entirely.
    기록된 행동열을 그대로 재생한다. 관측은 전혀 보지 않는다.

    This is the most informative baseline in the set. If replaying one fixed
    trajectory succeeds even when the object moves, the task does not require
    perception and a vision policy trained on it will learn nothing useful —
    it will learn the trajectory. That is a task-design failure, and it is
    much cheaper to find here than after collecting a hundred demonstrations.
    이 묶음에서 가장 정보가 많은 baseline 이다. 물체가 움직여도 고정 궤적 하나를
    재생해서 성공한다면, 그 태스크는 인식을 필요로 하지 않는다. 그런 데이터로
    학습한 시각 정책은 쓸모 있는 것을 배우지 않는다 — 궤적을 외울 뿐이다.
    그건 태스크 설계의 실패이고, 시연 100개를 찍은 뒤보다 여기서 발견하는 편이
    훨씬 싸다.
    """

    name = "replay"
    uses_privileged_state = False

    def __init__(self, actions: np.ndarray, source: str = "") -> None:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"actions must be (T, 6), got {arr.shape}")
        self._actions = arr
        self.source = source
        self._i = 0

    @classmethod
    def from_episode(cls, npz_path: Path) -> "ReplayPolicy":
        """Load the action track of a recorded episode.
        기록된 에피소드의 행동 트랙을 읽는다."""
        with np.load(npz_path) as z:
            return cls(z["action"], source=npz_path.name)

    def reset(self, seed: int | None = None) -> None:
        """Rewind to the first recorded action.
        첫 기록 행동으로 되감는다."""
        self._i = 0

    def act(self, obs: Observation) -> np.ndarray:
        """Return the next recorded action, holding the last one at the end.
        다음 기록 행동을 반환한다. 끝에 도달하면 마지막 행동을 유지한다."""
        idx = min(self._i, len(self._actions) - 1)
        self._i += 1
        return check_action(self._actions[idx], self.name)


class ScriptedPickPolicy:
    """Plan the whole pick from the ground-truth object position, then execute.
    물체의 정답 위치로 파지 전체를 계획한 뒤 실행한다.

    Not deployable — it reads simulator state. Its job is to be the ceiling:
    what the task allows when perception is perfect and the only error left is
    control. A learned policy cannot exceed this, and how far below it lands is
    the honest measure of how much perception is costing.
    배포 불가 — 시뮬 상태를 읽는다. 역할은 천장이다. 인식이 완벽하고 남은 오차가
    제어뿐일 때 이 태스크가 허용하는 상한선. 학습 정책은 이걸 넘을 수 없고,
    여기서 얼마나 아래에 떨어지는지가 인식이 치르는 비용의 정직한 척도다.
    """

    name = "scripted"
    uses_privileged_state = True

    def __init__(self, env: MujocoPickEnv) -> None:
        self._env = env
        self._cfg = env.cfg
        self._plan: list[np.ndarray] = []
        self._i = 0
        # Counted because instruments compare this policy against
        # `ScriptedFeedbackPolicy`, and a missing attribute reads as **zero
        # failures** through `getattr(..., 0)` -- which is what happened on
        # 2026-09-07: `lift_height_sweep` printed "lift IK 실패 0" for this policy
        # at every height while the feedback expert showed 15, and the comparison
        # was meaningless rather than informative.
        # 계측기가 이 정책을 `ScriptedFeedbackPolicy` 와 비교하는데, 속성이 없으면
        # `getattr(..., 0)` 을 지나 **실패 0** 으로 읽힌다. 2026-09-07 에 그것이
        # 일어났다 — `lift_height_sweep` 이 모든 높이에서 이 정책의 "lift IK 실패"
        # 를 0 으로 찍었고 피드백 전문가는 15 를 찍었는데, 그 비교는 정보가 아니라
        # 무의미였다.
        self.ik_failures = 0
        self.ik_fail_by_phase: dict[str, int] = {}

    def reset(self, seed: int | None = None) -> None:
        """Re-plan against the object's current position.
        물체의 현재 위치로 계획을 다시 세운다."""
        self.ik_failures = 0
        self.ik_fail_by_phase = {}
        self._plan = self._build_plan()
        self._i = 0

    def act(self, obs: Observation) -> np.ndarray:
        """Return the next planned action, holding the last one at the end.
        계획된 다음 행동을 반환한다. 끝에 도달하면 마지막을 유지한다."""
        if not self._plan:
            self.reset()
        idx = min(self._i, len(self._plan) - 1)
        self._i += 1
        return check_action(self._plan[idx], self.name)

    def _build_plan(self) -> list[np.ndarray]:
        """Expand the shared pick waypoints into one action per control tick.
        공유 파지 웨이포인트를 제어 틱당 행동 하나로 펼친다.

        The waypoints come from `pick_waypoints`, the same function the collector
        uses. Before 2026-09-03 this method had its own copy and the two had
        drifted (141 vs 165 ticks) -- the ceiling was not the demonstrator.
        웨이포인트는 수집기와 **같은 함수** `pick_waypoints` 에서 온다. 2026-09-03
        이전에는 이 메서드가 자기 사본을 갖고 있었고 둘이 갈라져 있었다(141 대 165틱)
        — 상한선이 시연자와 다른 것이었다.
        """
        cfg = self._cfg
        g = cfg["grasp"]
        env = self._env
        model = env.model
        rate = env.control_rate_hz
        offset = np.asarray(g["pinch_offset_local"], dtype=float)
        axis = np.asarray(g["approach_axis"], dtype=float)

        q_now = env.joint_positions()
        segments: list[tuple[np.ndarray, float, float]] = []
        seed_q = q_now
        for seg in pick_waypoints(cfg, env.object_position()):
            if seg.target is None:
                segments.append((seed_q, seg.grip, seg.seconds))
                continue
            res = solve_pose_ik(model, seg.target, offset, axis, q_init=seed_q, wrist_roll=0.0)
            if not res.ok:
                # 구간 이름을 모르므로 필수/선택으로 구분해 센다. 선택 구간(들기)의
                # 실패는 그 구간을 건너뛰는 것이고, 그러면 팔이 파지 자세에서 멈춘다.
                phase = "required" if seg.required else "lift"
                self.ik_failures += 1
                self.ik_fail_by_phase[phase] = self.ik_fail_by_phase.get(phase, 0) + 1
                if not seg.required:
                    # A failed lift skips its segment, as before the refactor.
                    # 들어올리기 실패는 그 구간만 건너뛴다. 리팩터링 이전과 같다.
                    continue
                # Unreachable: hold still rather than flail. The rollout scores
                # this a failure, which is the correct outcome.
                # 도달 불가면 휘젓지 말고 정지한다. 롤아웃은 실패로 채점하고 그게 맞다.
                return [normalize(q_now, cfg)]
            segments.append((res.qpos, seg.grip, seg.seconds))
            seed_q = res.qpos

        plan: list[np.ndarray] = []
        q_prev = q_now.copy()
        for q_target, grip, seconds in segments:
            n = max(1, int(seconds * rate))
            for k in range(1, n + 1):
                alpha = k / n
                q = (1 - alpha) * q_prev[:5] + alpha * np.asarray(q_target[:5], dtype=float)
                plan.append(normalize(np.concatenate([q, [grip]]), cfg))
            q_prev = np.concatenate([np.asarray(q_target[:5], dtype=float), [grip]])
        return plan


class ScriptedFeedbackPolicy:
    """A state-feedback version of the scripted expert. DAgger requires one.
    스크립트 전문가의 상태 피드백 판. DAgger 가 요구하는 형태다.

    `ScriptedPickPolicy` plans once at reset and replays by tick index -- it never
    reads `obs` at all. DAgger asks "what would the expert do in **this** state",
    and an open-loop plan cannot answer that question: it does not know where it
    is. So the phase boundaries move off the clock and onto the geometry. Each
    phase's target is recomputed from the object's true position, and the command
    is a rate-limited step from the joint angles the robot is **actually** at.
    `ScriptedPickPolicy` 는 리셋에서 계획을 한 번 세우고 틱 번호로 재생한다 —
    `obs` 를 아예 읽지 않는다. DAgger 는 "**이** 상태에서 전문가라면 뭘 했겠나"를
    묻는데, 개루프 계획은 그 질문에 답할 수 없다. 자기가 어디 있는지 모르니까.
    그래서 phase 경계를 시계에서 기하로 옮긴다. phase 마다 목표를 물체 참값에서
    다시 계산하고, 명령은 로봇이 **실제로** 있는 관절각에서 출발한 속도제한 스텝이다.

    Two honesty notes, written here rather than hidden:
    숨기지 않고 여기 적는 두 가지:

    1. Still privileged. It reads the object's true position, exactly like
       `ScriptedPickPolicy`. It is an expert for labelling, never a deployable
       policy.
       여전히 특권 정보다. `ScriptedPickPolicy` 와 똑같이 물체 참값을 읽는다.
       라벨을 붙이는 전문가이고, 배포 가능한 정책이 아니다.
    2. Closing is still on the clock. There is no contact signal in the
       observation to close on, so the close phase runs for `timing.close_s`.
       That is the one part of the plan that stays open-loop.
       닫기는 여전히 시계로 한다. 관측에는 접촉 신호가 없어서 닫을 계기가 없다.
       닫기 phase 는 `timing.close_s` 만큼 돈다. 계획 중 개루프로 남는 부분이다.

    Whether this actually recovers from an off-distribution state is **not**
    assumed here -- `eval/recovery.py` measures it, and its G0-b gate stops
    DAgger before a single label is collected if it does not.
    이것이 실제로 분포 밖 상태에서 복구하는지는 여기서 가정하지 않는다 —
    `eval/recovery.py` 가 계측하고, 못 하면 G0-b 게이트가 라벨 한 장 모으기 전에
    DAgger 를 멈춘다.
    """

    name = "scripted_fb"
    uses_privileged_state = True

    APPROACH, DESCEND, CLOSE, LIFT, DONE = "approach", "descend", "close", "lift", "done"

    def __init__(self, env: MujocoPickEnv) -> None:
        self._env = env
        self._cfg = env.cfg
        g = env.cfg["grasp"]
        fb = g.get("feedback", {})
        self._offset = np.asarray(g["pinch_offset_local"], dtype=float)
        self._axis = np.asarray(g["approach_axis"], dtype=float)
        self._open, self._close = float(g["open_cmd"]), float(g["close_cmd"])
        self._max_step = float(fb.get("max_joint_step_rad", 0.05))
        self._approach_tol = float(fb.get("approach_tol_m", 0.010))
        self._grasp_tol = float(fb.get("grasp_tol_m", 0.005))
        self._resolve_move = float(fb.get("resolve_obj_move_m", 0.002))
        self._close_ticks = max(1, int(float(g["timing"]["close_s"]) * env.control_rate_hz))
        self.phase = self.APPROACH
        self.phase_entry: dict[str, int] = {}
        self._tick = 0
        self._close_left = self._close_ticks
        self._q_target: np.ndarray | None = None
        self._solved_at: np.ndarray | None = None
        self._solved_phase = ""
        self._obj_frozen: np.ndarray | None = None
        self.ik_solves = 0
        self.ik_failures = 0
        self.ik_fail_by_phase: dict[str, int] = {}

    def reset(self, seed: int | None = None) -> None:
        """Return to the approach phase and forget the cached IK solution.
        접근 phase 로 돌아가고 캐시된 IK 해를 잊는다."""
        self.phase = self.APPROACH
        self.phase_entry = {self.APPROACH: 0}
        self._tick = 0
        self._close_left = self._close_ticks
        self._q_target = None
        self._solved_at = None
        self._solved_phase = ""
        self._obj_frozen = None
        self.ik_solves = 0
        self.ik_failures = 0
        self.ik_fail_by_phase = {}

    def _targets(self, obj: np.ndarray) -> dict[str, np.ndarray]:
        """The three Cartesian waypoints, from the object's current position.
        물체의 현재 위치에서 계산한 세 데카르트 웨이포인트."""
        g = self._cfg["grasp"]
        grasp_pt = np.asarray(obj, dtype=float) + np.array(
            [0.0, 0.0, float(g["grasp_z_offset_m"])]
        )
        return {
            self.APPROACH: grasp_pt + np.array([0.0, 0.0, float(g["approach_height_m"])]),
            self.DESCEND: grasp_pt,
            self.LIFT: grasp_pt + np.array([0.0, 0.0, float(g["lift_height_m"])]),
        }

    def _advance(self, pinch: np.ndarray, targets: dict[str, np.ndarray]) -> None:
        """Move to the next phase when this phase's condition is met.
        이 phase 의 조건이 충족되면 다음 phase 로 넘어간다."""
        prev = self.phase
        if self.phase == self.APPROACH:
            if float(np.linalg.norm(pinch - targets[self.APPROACH])) < self._approach_tol:
                self.phase = self.DESCEND
        elif self.phase == self.DESCEND:
            if float(np.linalg.norm(pinch - targets[self.DESCEND])) < self._grasp_tol:
                self.phase = self.CLOSE
        elif self.phase == self.CLOSE:
            self._close_left -= 1
            if self._close_left <= 0:
                self.phase = self.LIFT
        elif self.phase == self.LIFT:
            if float(np.linalg.norm(pinch - targets[self.LIFT])) < self._approach_tol:
                self.phase = self.DONE
        if self.phase != prev:
            self.phase_entry.setdefault(self.phase, self._tick)

    def act(self, obs: Observation) -> np.ndarray:
        """One rate-limited step toward this phase's target.
        이 phase 의 목표로 향하는 속도제한 스텝 하나."""
        env = self._env
        self._tick += 1
        q_now = np.asarray(denormalize(obs.state, self._cfg), dtype=float)
        obj = env.object_position()
        pinch = grasp_point(env.model, env.data, self._offset)
        targets = self._targets(self._obj_frozen if self._obj_frozen is not None else obj)
        self._advance(pinch, targets)
        if self.phase in (self.CLOSE, self.LIFT, self.DONE) and self._obj_frozen is None:
            # Freeze the reference once the object is captured. Recomputing the lift
            # target from the object's *current* position makes the target rise with
            # the object the gripper is holding -- the arm then chases a point
            # 8.8 cm above whatever it lifts, forever, until the solve goes
            # infeasible. 2026-09-07 🟢: 22/22 lift IK failures, 0/4 success, every
            # episode ending in the lift phase.
            # 물체가 잡히면 기준을 고정한다. lift 목표를 물체의 **현재** 위치에서
            # 다시 계산하면, 그리퍼가 든 물체와 함께 목표도 올라가서 팔이 자기가
            # 들고 있는 것의 8.8cm 위를 영원히 쫓고 결국 해가 없어진다.
            # 2026-09-07 실측 🟢: lift IK 22/22 실패, 0/4 성공, 전부 lift phase 에서 종료.
            self._obj_frozen = np.asarray(obj, dtype=float).copy()
            targets = self._targets(self._obj_frozen)

        if self.phase in (self.CLOSE, self.DONE):
            # 닫는 중에는 팔을 세우지 않는다. 물체를 밀어내는 원인이 된다.
            q_arm = q_now[:5]
        else:
            target = targets[self.APPROACH if self.phase == self.APPROACH else self.phase]
            moved = (
                self._solved_at is None
                or self._solved_phase != self.phase
                or float(np.linalg.norm(np.asarray(obj) - self._solved_at)) > self._resolve_move
            )
            if moved:
                # Seed from the previous phase's solution, exactly as the open-loop
                # plan does (`seed_q = res.qpos`), and only then from the measured
                # joints. IK here is a local solver, so the seed picks the branch.
                # 2026-09-07 🟢: seeding from the measured joints instead failed the
                # lift solve 14/14 while the open-loop plan solved it -- the two
                # experts differed in IK seeding, not in feedback, which is exactly
                # what G0-a must not be measuring.
                # 개루프 계획과 **같은 방식으로** 직전 phase 의 해로 시드하고, 실패할
                # 때만 실측 관절각으로 재시도한다. 여기 IK 는 국소 해법이라 시드가
                # 분기를 고른다. 2026-09-07 🟢: 실측 관절각으로 시드했더니 lift 해가
                # 14/14 실패했는데 개루프는 같은 목표를 풀었다 — 두 전문가가
                # 피드백이 아니라 IK 시드에서 달랐고, 그건 G0-a 가 재야 할 것이 아니다.
                seeds = [q_now]
                if self._q_target is not None:
                    seeds.insert(0, np.concatenate([self._q_target, [q_now[5]]]))
                res = None
                for q_seed in seeds:
                    res = solve_pose_ik(
                        env.model, target, self._offset, self._axis,
                        q_init=q_seed, wrist_roll=0.0,
                    )
                    self.ik_solves += 1
                    if res.ok:
                        break
                assert res is not None
                if res.ok:
                    self._q_target = np.asarray(res.qpos[:5], dtype=float)
                else:
                    # Unreachable: hold. The rollout scores this a failure, correctly.
                    # 도달 불가면 정지한다. 롤아웃이 실패로 채점하고 그게 맞다.
                    self.ik_failures += 1
                    self.ik_fail_by_phase[self.phase] = (
                        self.ik_fail_by_phase.get(self.phase, 0) + 1
                    )
                    self._q_target = q_now[:5].copy()
                self._solved_at = np.asarray(obj, dtype=float).copy()
                self._solved_phase = self.phase
            assert self._q_target is not None
            delta = np.clip(self._q_target - q_now[:5], -self._max_step, self._max_step)
            q_arm = q_now[:5] + delta

        grip = self._close if self.phase in (self.CLOSE, self.LIFT, self.DONE) else self._open
        return check_action(
            normalize(np.concatenate([q_arm, [grip]]), self._cfg), self.name
        )
