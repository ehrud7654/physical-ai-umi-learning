"""Interval estimates shared by every tool that reports a success rate.
성공률을 보고하는 모든 도구가 공유하는 구간 추정.

Kept in one place because two implementations of the same interval eventually
disagree, and the disagreement shows up as two tools reporting different
confidence for the same measurement.
한 곳에 두는 이유는 같은 구간의 구현이 둘이면 언젠가 어긋나고, 그 어긋남이
같은 측정에 대해 두 도구가 다른 신뢰도를 보고하는 형태로 드러나기 때문이다.
"""

from __future__ import annotations


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a success rate.
    성공률의 95% Wilson 신뢰구간.

    A rate reported without an interval invites reading a 5-percentage-point
    difference as a finding. At n=20 the interval is roughly ±20 points, and
    this run already shows why: the same scripted policy scored 95% on one seed
    block and 70% on another. The interval is what stops that from becoming a
    conclusion.
    구간 없이 성공률만 보고하면 5%p 차이를 발견으로 읽게 된다. n=20 에서 구간은
    대략 ±20%p 이고, 이번 실행이 그 이유를 이미 보여줬다 — 같은 스크립트 정책이
    한 시드 블록에서 95%, 다른 블록에서 70% 였다. 구간은 그것이 결론이 되는 것을
    막는다.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))
