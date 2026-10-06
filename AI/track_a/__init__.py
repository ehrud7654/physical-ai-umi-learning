"""Track A — 현실·기하. UMI pose estimation, calibration, Sim2Real, real data.
트랙 A — 현실·기하. UMI 포즈 추정, 캘리브레이션, Sim2Real, 실데이터.

Track A's question is "is this data real, and can the robot reproduce it".
Its output is a trustworthy dataset. Everything here turns raw recordings from
the real world into episodes that satisfy `contract/`.
트랙 A 의 질문은 "이 데이터가 진짜인가, 로봇이 재현할 수 있나" 이고,
산출물은 신뢰 가능한 데이터셋이다. 여기 있는 것은 현실에서 찍힌 raw 기록을
`contract/` 를 만족하는 에피소드로 바꾸는 일이다.

NOT IMPLEMENTED. This is a scaffold — the folder shape and the boundary, not
the code. Track A writes what goes inside.
**구현 없음.** 폴더 형태와 경계만 잡아둔 골격이다. 안에 들어갈 코드는 트랙 A 가 쓴다.

## 경계 — 접점은 `contract/` + `umi/` 둘이다 (D-AI-30, 2026-09-08)

    track_a/convert/  →  umi/  →  contract/  ←  policy/ eval/ sim/ data/ (트랙 B)

⚠️ **2026-09-08 변경.** 이전에는 접점이 `contract/` 하나였다. raw→계약 변환기를
   `AI/umi/` 로 두면서 둘이 됐다 — 이슈 31 과 127 이 같은 코드를 써야 하고, raw
   스키마는 양 트랙이 함께 읽어야 하는데 `track_a/` 안에 있으면 트랙 B 가 아래
   규칙을 깨야 하기 때문이다. 대행 결정이고 되돌릴 조건이 D-AI-30 에 있다.

- **트랙 B 는 `track_a/` 를 임포트하지 않는다.** 트랙 B 가 보는 것은 계약을 만족하는
  에피소드뿐이고, 그것이 어떻게 만들어졌는지 알 필요가 없다
- **트랙 A 는 `sim/` `policy/` `eval/` 을 임포트하지 않는다.** 필요한 것은 `contract/` 다
- **`track_a/` 는 `umi/` 와 `contract/` 만 임포트한다.** `sim/` `policy/` `eval/` 금지
- `umi/` 도 `sim/` 을 임포트하지 않는다. 둘을 함께 쓰는 곳은 `tools/umi_mujoco.py` 하나다
- 이 경계를 또 넘는 임포트가 생기면 접점이 셋이 된 것이다. D-AI 기록이 필요하다

⚠️ 현재 AI/ 의 나머지(`sim/` `policy/` `eval/` `data/` `vlm/` `tracking/`)는 전부
   트랙 B 작업이다. **비대칭이다** — 트랙 B 코드가 `track_b/` 로 묶여 있지 않다.
   대칭으로 만들 이유가 생기면 그때 옮긴다. 지금 옮기면 임포트 경로만 또 바뀐다.

## 하위 폴더와 이슈

| 폴더 | 이슈 | 내용 |
| --- | --- | --- |
| `calibration/` | project-task | 카메라 캘리브레이션 (내부·외부 파라미터) |
| `pose/`        | project-task | pose → 로봇 좌표 변환 |
| `sync/`        | project-task | pose ↔ 영상 시간동기화 |
| `convert/`     | project-task | raw → dataset 포맷 변환 |

## 착수 전에 알아둘 것 (트랙 B 가 실측한 것)

1. **물체는 도달 가능 영역 안에만 놓는다.** 그 밖에서 찍은 에피소드는 로봇이 재현할 수
   없어 통째로 폐기 대상이다. 현재 실측 영역과 근거는 `AI/README.md` 의 '측정 수치' 절 참조
2. **위치 허용오차는 약 ±10mm 다.** 캘리브레이션·좌표변환 오차가 이 수준을 넘으면
   그 데이터로 학습한 정책은 물체를 못 집는다. 이게 정밀도 요구의 근거다
2-b. **접근 방향 허용폭은 약 5도다** (실측 🟢 2026-09-08, `MEASURE_umi_reach_0907.md`).
   자세를 10도 기울이면 수용률이 20.5% 로 붕괴한다. 위치는 20mm 에서 92% 다.
   **각도가 위치보다 훨씬 비싸다** — 기전은 관절 한계다.
   그리고 **roll(접근축 둘레 회전)은 재현 가능하다** — `wrist_roll` 이 위치·접근축과
   결합이 0 이다 (D-AI-31). `pose/__init__.py` 의 이전 서술은 틀렸고 정정했다
3. **시간동기화는 수집 전에 계측한다.** 어긋나면 모델이 "과거 화면 보고 미래 행동"을
   학습한다. 학습 손실과 시뮬 성공률은 정상이고 실물에서만 실패해서, 이 프로젝트에서
   추적이 가장 어려운 실패 형태다. `contract/` 가 `state_timestamp` /
   `action_timestamp` 를 분리해 둔 이유가 이 계측을 가능하게 하려는 것이다
4. **시뮬에서는 이 오차가 구조적으로 0 이다.** 시뮬로 project-task 을 검증했다고
   볼 수 없다

## 인수인계 (2026-09-08)

트랙 A 코드의 기본 세팅을 AI 학습 담당자(트랙 B)가 대행했다. 무엇이 돌고 무엇이 비었고
무엇을 재야 하는지는 **`track_a/HANDOVER_track_a.md`** 가 정본이다. 그것부터 읽어라.
"""
