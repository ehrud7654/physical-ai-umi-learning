"""The one seam between a policy and whatever is on the other side of it.
정책과 그 반대편에 있는 것 사이의 유일한 이음매.

**Backend-agnostic on purpose.** Nothing here imports MuJoCo. A second simulator
(Isaac Sim) or the real arm satisfies exactly this and nothing else changes.
**의도적으로 백엔드에 무관하다.** 여기서 MuJoCo 를 임포트하지 않는다. 두 번째
시뮬레이터(Isaac Sim)든 실물 팔이든, 정확히 이것만 만족시키면 나머지는 그대로다.

project-task asks for the same policy code to run against simulation and
against the real arm. That is only true if the policy never learns which one it
is talking to — so everything simulator-specific lives behind `RobotEnv`, and a
policy only ever sees an `Observation` and returns an action in contract units.
project-task 의 요구는 같은 정책 코드가 시뮬과 실물 양쪽에서 도는 것이다.
정책이 자기가 어느 쪽과 얘기하는지 **모를 때만** 그게 성립한다.

Implementations:
구현체:
  sim/mujoco/env.py   MujocoPickEnv  — 있음
  sim/isaac/          — 없음. 검토 대상
  실물                — 없음. project-task ([ROS] 정책 ckpt 로드 -> 추론 노드)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Observation:
    """What a policy sees at one control tick — data-contract units.
    정책이 제어 틱 한 번에 보는 것. 데이터 계약 단위.

    Field shapes and dtypes match `contract/episode.py` exactly. If they drift
    apart, a policy trained on recorded data will silently receive something
    different at inference time.
    필드의 shape 과 dtype 은 `contract/episode.py` 와 정확히 일치한다. 둘이
    갈라지면, 기록 데이터로 학습한 정책이 추론 시점에 다른 것을 조용히 받게 된다.
    """

    images: dict[str, np.ndarray]  # camera name -> (3, 224, 224) uint8, CHW
    state: np.ndarray  # (6,) float32, normalized to [-1, 1]
    timestamp: float  # seconds since episode start


@runtime_checkable
class RobotEnv(Protocol):
    """What a policy is allowed to know about the world.
    정책이 세상에 대해 알아도 되는 전부.

    Anything beyond this surface is privileged information. A policy that reads
    it cannot be deployed, and `eval/interface_check.py` is what catches that
    now instead of the robot catching it later.
    이 표면 밖의 것은 특권 정보다. 그것을 읽는 정책은 배포할 수 없고,
    나중에 로봇 앞에서가 아니라 지금 `eval/interface_check.py` 가 잡는다.
    """

    @property
    def control_rate_hz(self) -> float:
        """Ticks per second the policy is called at.
        정책이 호출되는 주기."""
        ...

    @property
    def camera_names(self) -> list[str]:
        """Cameras present in the observation, in a stable order.
        관측에 들어오는 카메라. 순서는 고정."""
        ...

    def reset(self, seed: int | None = None) -> Observation:
        """Put the world back to a start state and return the first observation.
        세상을 시작 상태로 되돌리고 첫 관측을 반환한다."""
        ...

    def step(self, action: np.ndarray) -> Observation:
        """Apply one normalized action and advance one control tick.
        정규화된 행동 하나를 적용하고 제어 틱 하나만큼 진행한다."""
        ...

    def is_success(self) -> bool:
        """Whether the task is currently satisfied.
        지금 태스크가 달성된 상태인가."""
        ...
