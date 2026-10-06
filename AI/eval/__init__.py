"""Evaluation — rollout success rate and the gates, NOT validation loss.
평가 — 롤아웃 성공률과 게이트. validation loss 가 아니다.

Gate thresholds live in `rollout.GATES` and are fixed before results are
looked at. Editing them after seeing a result is rationalisation.
게이트 기준은 `rollout.GATES` 에 있고 결과를 보기 전에 확정한다.
결과를 보고 고치면 사후 합리화다.
"""
