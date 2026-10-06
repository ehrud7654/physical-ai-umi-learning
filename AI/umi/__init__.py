"""UMI raw recordings and their conversion into contract episodes.
UMI raw 기록과 계약 에피소드로의 변환.

project-task (관통 검증) 과 project-task (raw→dataset 변환) 이 **같은 코드**를
쓴다. 두 벌 만들면 실물 데이터가 들어오는 날 어느 쪽이 맞는지 모른다.

## 왜 `track_a/convert/` 가 아니라 여기인가

`track_a/__init__.py` 가 "트랙 B 는 `track_a/` 를 임포트하지 않는다"를 명시한다.
그런데 raw 스키마는 양쪽이 읽어야 한다 — 트랙 A 는 실기록을 그 형식으로 쓰고,
트랙 B 는 그 형식을 읽어 학습 데이터로 만든다. 스키마가 `track_a/` 안에 있으면
트랙 B 가 규칙을 깨야 한다. 그래서 중립 위치에 둔다.

    track_a/convert/arcore.py  →  umi/  →  contract/  →  policy/ eval/
    (ARCore 로그 파싱)             (여기)     (계약)        (트랙 B)

인수인계 문장은 한 줄이다 — **`umi.raw.RawEpisode` 를 만들어 주면 아래는 다 돈다.**

## 임포트 규칙 — 이걸 깨면 실데이터 경로가 시뮬에 묶인다

- `umi/` 는 `contract/` 와 numpy 만 임포트한다
- **`sim/` `policy/` `eval/` `data/` `track_a/` 를 임포트하지 않는다**
- IK 는 구현하지 않고 **주입받는다** (`umi.ik.IKSolver`). 시뮬 어댑터는 `tools/` 에 둔다

⚠️ 접점이 `contract/` 하나에서 `contract/` + `umi/` 둘로 늘어났다.
   D-AI 기록 필요. 작성자 AI 학습 담당자(트랙B, 트랙A 대행), 되돌릴 조건은 HANDOVER 참조.

⚠️ 빚: 실물 경로는 `sim/` 에 의존해선 안 되는데 현재 유일한 IK 구현이
   `sim/mujoco/kinematics.py:solve_pose_ik` 다. 중립 위치로 옮기는 것은 양 트랙
   합의가 필요한 별건이다. 지금은 어댑터로 주입만 하고 빚으로 기록한다.
"""
