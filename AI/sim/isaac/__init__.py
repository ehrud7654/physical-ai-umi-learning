"""Isaac Sim backend — NOT IMPLEMENTED. Feasibility not yet measured.
Isaac Sim 백엔드 — **구현 없음. 실행가능성 미계측.**

To be a backend, something here must satisfy `sim/base.py`'s `RobotEnv` and
nothing else. Then policies, baselines, rollout and the gates run unchanged.
백엔드가 되려면 여기의 무언가가 `sim/base.py` 의 `RobotEnv` 만 만족시키면 된다.
그러면 정책·baseline·롤아웃·게이트가 그대로 돈다.

Known constraints — check these before spending time:
착수 전에 확인할 제약:

1. Isaac Sim needs RT cores, so it cannot render on the H200 server. It can
   only run on the local RTX 4070 / 5070 machines. MuJoCo stays the primary
   simulator because it renders headless anywhere, H200 included.
   Isaac Sim 은 RT 코어가 필요해서 H200 서버에서 렌더링할 수 없다. 로컬
   RTX 4070 / 5070 에서만 돈다. MuJoCo 가 주력인 이유는 H200 을 포함해
   어디서든 헤드리스 렌더링이 되기 때문이다.

2. Sim success rate is not the final metric either way. Friction, sensor noise,
   lighting and calibration error are what neither simulator reproduces —
   a prettier renderer does not close that gap.
   어느 쪽이든 시뮬 성공률은 최종 지표가 아니다. 마찰·센서 노이즈·조명·
   캘리브레이션 오차는 두 시뮬레이터 모두 재현하지 못한다. 렌더러가
   예뻐지는 것으로 그 간극이 닫히지 않는다.

3. The data contract must not change to accommodate a second simulator.
   If it has to, that is a both-tracks decision with a D-AI record.
   두 번째 시뮬레이터를 위해 데이터 계약을 바꾸지 않는다. 바꿔야 한다면
   양 트랙 합의와 D-AI 기록이 필요한 사안이다.

⚠️ 시뮬 환경 구축은 Jira 에서 [ROS] 로 분류돼 있다. 이슈를 새로 만들기 전에
   접두어를 확인하라.
"""
