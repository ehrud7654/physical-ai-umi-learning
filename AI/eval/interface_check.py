"""Prove that deployable policies do not depend on the simulator.
배포 가능한 정책이 시뮬레이터에 의존하지 않음을 증명한다.

project-task asks for the same policy code to run against sim and against the
real arm. That claim is easy to make and easy to be wrong about — a policy can
reach into the environment for something only MuJoCo has and nobody notices
until the ROS node is written and it does not work.
project-task 은 같은 정책 코드가 시뮬과 실물 양쪽에서 돌 것을 요구한다.
이 주장은 하기도 쉽고 틀리기도 쉽다. 정책이 MuJoCo 에만 있는 것을 슬쩍 꺼내 써도
ROS 노드를 짜서 안 돌아볼 때까지 아무도 모른다.

So this runs them against a stub environment that has no simulator in it at all.
If a policy touches anything outside the `RobotEnv` surface, it fails here —
now, on a laptop, instead of later on the robot.
그래서 시뮬레이터가 전혀 없는 더미 환경에 정책을 태워본다. `RobotEnv` 표면 밖의
무언가를 건드리는 정책은 여기서 실패한다. 나중에 로봇 앞에서가 아니라, 지금 노트북에서.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


from sim.base import Observation, RobotEnv
from policy.base import Policy
from policy.baselines import HoldPolicy, ReplayPolicy, ZeroPolicy

IMAGE_SHAPE = (3, 224, 224)


class StubEnv:
    """A `RobotEnv` with no physics — stands in for whatever is on the far side.
    물리가 없는 `RobotEnv`. 반대편에 무엇이 오든 그 자리를 대신한다.

    Deliberately dumb: it returns fixed-shape observations and records the
    actions it was given. Its only job is to have exactly the surface a real
    robot driver would have, and nothing more.
    일부러 멍청하게 만들었다. 고정된 shape 의 관측을 돌려주고 받은 행동을 기록할
    뿐이다. 실물 로봇 드라이버가 가질 표면을 정확히 갖고, 그 이상은 갖지 않는 것이
    유일한 임무다.
    """

    def __init__(self, ticks: int = 30, rate_hz: float = 30.0) -> None:
        self._ticks = ticks
        self._rate = rate_hz
        self._t = 0
        self.received: list[np.ndarray] = []

    @property
    def control_rate_hz(self) -> float:
        """Ticks per second the policy is called at.
        정책이 호출되는 주기."""
        return self._rate

    @property
    def camera_names(self) -> list[str]:
        """Cameras present in the observation.
        관측에 들어오는 카메라."""
        return ["cam_wrist"]

    def reset(self, seed: int | None = None) -> Observation:
        """Return the first observation of a fresh episode.
        새 에피소드의 첫 관측을 반환한다."""
        self._t = 0
        self.received.clear()
        return self._observe()

    def step(self, action: np.ndarray) -> Observation:
        """Record the action and advance one tick.
        행동을 기록하고 한 틱 진행한다."""
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != (6,):
            raise AssertionError(f"policy returned shape {arr.shape}, contract requires (6,)")
        if not np.isfinite(arr).all():
            raise AssertionError(f"policy returned non-finite action: {arr}")
        if arr.min() < -1.0 - 1e-4 or arr.max() > 1.0 + 1e-4:
            raise AssertionError(f"action outside [-1, 1]: min={arr.min()} max={arr.max()}")
        self.received.append(arr.copy())
        self._t += 1
        return self._observe()

    def is_success(self) -> bool:
        """A stub never succeeds; success is not what it measures.
        더미는 성공하지 않는다. 성공은 여기서 재는 것이 아니다."""
        return False

    def _observe(self) -> Observation:
        rng = np.random.default_rng(self._t)
        return Observation(
            images={c: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8) for c in self.camera_names},
            state=rng.uniform(-1.0, 1.0, 6).astype(np.float32),
            timestamp=self._t / self._rate,
        )


def check(policy: Policy, ticks: int = 30) -> tuple[bool, str]:
    """Run one policy against the stub and report whether it survived.
    정책 하나를 더미에 태우고 살아남았는지 보고한다."""
    env = StubEnv(ticks=ticks)
    try:
        if not isinstance(env, RobotEnv):
            return False, "StubEnv 가 RobotEnv 프로토콜을 만족하지 않는다"
        obs = env.reset(seed=0)
        policy.reset(seed=0)
        for _ in range(ticks):
            obs = env.step(policy.act(obs))
    except Exception as exc:  # noqa: BLE001 — reporting the failure is the point
        return False, f"{type(exc).__name__}: {exc}"
    if len(env.received) != ticks:
        return False, f"{len(env.received)} actions for {ticks} ticks"
    return True, f"{ticks} 틱 관통, 행동 {len(env.received)}개 전부 계약 준수"


def main() -> int:
    dataset = Path("datasets/sim_pick_v0")
    episodes = sorted(dataset.glob("*.npz"))

    policies: list[Policy] = [HoldPolicy(), ZeroPolicy()]
    if episodes:
        policies.append(ReplayPolicy.from_episode(episodes[0]))
    else:
        print(f"⚠️ {dataset} 에 에피소드가 없어 replay 는 건너뛴다")

    print("시뮬레이터 없는 더미 환경에 배포 가능 정책을 태운다:")
    failed = 0
    for policy in policies:
        ok, detail = check(policy)
        print(f"  {policy.name:10s} {'통과' if ok else '실패'}  {detail}")
        failed += int(not ok)

    print()
    print("⚠️ scripted 정책은 여기에 없다. 물체의 정답 위치를 읽으므로 실물에서 돌 수 없고,")
    print("   그 사실이 policy.uses_privileged_state 로 표시돼 있다.")
    print()
    if failed:
        print(f"{failed}개 정책이 시뮬레이터 밖에서 돌지 못한다 — project-task 요구 미충족")
    else:
        print("배포 가능 정책 전부가 RobotEnv 표면만으로 동작한다.")
        print("실물 구현(project-task)은 같은 표면을 채우면 정책 코드를 그대로 쓴다.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
