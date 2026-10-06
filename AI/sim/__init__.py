"""Simulation, keyed by backend.
시뮬레이션. 백엔드별로 나눈다.

    base.py     RobotEnv 프로토콜. 백엔드 무관. 정책이 보는 전부
    mujoco/     주력 백엔드 (헤드리스 렌더링이 어디서든 된다)
    isaac/      검토 대상. 구현 없음

A policy must never import from a backend directory.
정책은 백엔드 디렉터리를 임포트해서는 안 된다.
"""
