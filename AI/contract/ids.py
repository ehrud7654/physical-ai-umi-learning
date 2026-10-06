"""The shared vocabulary: identifiers both the episode contract and the skill
registry have to agree on.
공유 어휘. 에피소드 계약과 스킬 레지스트리가 같이 합의해야 하는 식별자들.

This module exists to break a cycle, and the cycle is the point: the contract
needs to know which skill identifiers are legal, and the registry needs to know
which contract version it was trained under. Neither can own the other, so the
names they share live below both.
이 모듈은 순환 임포트를 끊으려고 있고, 그 순환 자체가 요점이다. 계약은 어떤 스킬
식별자가 유효한지 알아야 하고, 레지스트리는 자신이 어느 계약 버전으로 학습됐는지
알아야 한다. 어느 쪽도 다른 쪽을 소유할 수 없으므로 공유하는 이름은 둘 아래에 둔다.

⚠️ 이 파일의 값을 바꾸면 이미 수집한 에피소드가 무효가 될 수 있다.
   변경에는 양 트랙 합의와 D-AI 기록이 필요하다.
"""

from __future__ import annotations

SKILL_IDS: tuple[str, ...] = (
    "pick_place",
    "sort_two",
    "align_fixture",
    "present_inspect",
    "line_up",
)
"""The only place a skill identifier is defined. Everything derives from this.
스킬 식별자가 정의되는 유일한 자리. 나머지는 전부 여기서 파생된다."""

DESTINATIONS: tuple[str, ...] = (
    "left_tray",
    "right_tray",
    "fixture",
    "origin",
    "target_pose",
)

TIERS: tuple[str, ...] = ("tutorial", "demo")
"""Which layer a skill belongs to, and they are not the same product claim.
스킬이 어느 층에 속하는가. 둘은 같은 제품 주장이 아니다.

  tutorial — 플랫폼에 미리 학습시켜 넣어두는 기본 제공 범용 프리미티브
  demo     — 특정 페르소나를 겨냥해 **시연 자리에서 추가하는** 스킬

제품 주장은 "우리가 다섯 행동을 만들었다"가 아니라 "사용자가 시연으로 새 행동을
추가할 수 있다"이다. 튜토리얼 다섯이 되는 것보다 하나가 추가되어 동작하는 것이
그 주장의 더 강한 근거다."""

STATUSES: tuple[str, ...] = (
    "planned",      # 정의만 있다. 데이터 없음
    "collecting",   # 시연 수집 중
    "training",     # 학습 중
    "gate_failed",  # 학습했으나 게이트 미통과
    "deployed",     # 게이트 통과, 배포됨
)
"""Where a skill actually stands. Five planned skills are not five skills.
스킬이 실제로 어디까지 왔는가. 계획된 다섯은 다섯 개가 아니다."""
