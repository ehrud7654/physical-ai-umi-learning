"""Policies — baselines now, learned policies (BC -> ACT -> Diffusion) later.
정책 — 지금은 baseline, 나중에 학습 정책(BC → ACT → Diffusion).

A new learned policy goes here as `bc.py` / `act.py` and must satisfy the
`Policy` protocol in `base.py`. Nothing else needs to change.
새 학습 정책은 `bc.py` / `act.py` 로 여기 들어오고, `base.py` 의 `Policy`
프로토콜만 만족하면 된다. 다른 것은 바꿀 필요가 없다.

⚠️ baseline 성능을 재기 전에 ACT/Diffusion 으로 넘어가지 않는다.
"""
