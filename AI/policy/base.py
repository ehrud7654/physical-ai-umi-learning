"""What every policy in this project must look like, learned or not.
이 프로젝트의 모든 정책이 따라야 할 형태. 학습된 것이든 아니든.

One interface for scripted baselines and for BC/ACT/Diffusion alike. The point
is that the rollout harness cannot tell them apart, so a learned policy is
scored on exactly the same footing as the baseline it has to beat.
스크립트 baseline 과 BC/ACT/Diffusion 이 같은 인터페이스를 쓴다. 롤아웃
harness 가 둘을 구분하지 못하는 것이 핵심이다. 학습된 정책이 이겨야 할
baseline 과 정확히 같은 조건에서 채점된다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from sim.base import Observation


@runtime_checkable
class Policy(Protocol):
    """Maps an observation to a normalized joint-target action.
    관측을 정규화된 목표 관절각 행동으로 매핑한다."""

    @property
    def name(self) -> str:
        """Short identifier used in logs and result tables.
        로그와 결과표에 쓰는 짧은 식별자."""
        ...

    @property
    def uses_privileged_state(self) -> bool:
        """True when the policy reads simulator-only ground truth.
        시뮬에만 있는 정답 정보를 읽는 정책이면 True.

        A policy that reads this cannot run on the real robot and its success
        rate is an upper bound, never a performance claim. The harness records
        the flag so nobody has to remember which was which.
        이걸 읽는 정책은 실물에서 못 돈다. 그 성공률은 상한선이지 성능이 아니다.
        harness 가 이 플래그를 기록하므로 나중에 헷갈릴 일이 없다.
        """
        ...

    def reset(self, seed: int | None = None) -> None:
        """Clear any per-episode state before a new rollout.
        새 롤아웃 전에 에피소드별 상태를 비운다."""
        ...

    def act(self, obs: Observation) -> np.ndarray:
        """Return a (6,) float32 action in [-1, 1].
        (6,) float32, [-1,1] 범위의 행동을 반환한다."""
        ...


def check_action(action: np.ndarray, policy_name: str) -> np.ndarray:
    """Fail loudly on a malformed action instead of letting it reach the robot.
    형태가 틀린 행동이 로봇에 도달하기 전에 크게 실패시킨다.

    On hardware a NaN or a wrong-length action is a physical event, not a stack
    trace. Checking here costs nothing and the same check runs in simulation.
    실물에서 NaN 이나 길이가 틀린 행동은 스택 트레이스가 아니라 물리적 사고다.
    여기서 검사하는 비용은 없고, 시뮬에서도 같은 검사가 돈다.
    """
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape != (6,):
        raise ValueError(f"{policy_name}: action shape must be (6,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{policy_name}: action contains NaN or inf: {arr}")
    return arr
