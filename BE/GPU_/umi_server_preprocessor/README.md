# UMI Server Preprocessor — ORB-SLAM Atlas v2

S22 수집 앱 RAW를 받아 Stanford UMI 학습 직전 입력까지 만드는 서버 인계 패키지다.

```text
RAW → IMU/그리퍼 전처리 → ORB-SLAM3 Atlas → 시연 재국지화
    → 품질 gate → UMI Zarr → GPU 입력 TGZ
```

학습, 체크포인트 export, Jetson 모터 제어는 포함하지 않는다. Spring은 업로드·권한·작업 DB를
담당하고 이 컨테이너는 영상/SLAM/Zarr 처리만 담당한다.

## 빠른 실행

Linux 서버에서 작업 데이터를 다음과 같이 둔다.

```text
jobs/demo-001/
├── raw_mapping/rec_<mapping-id>/...
└── raw/rec_<demo-id>/...
```

공개본에서는 원래 통합 Compose가 빠져 있으므로 이 폴더의 Dockerfile과 환경 설정을 자신의 Linux 환경에 맞춰 구성해야 합니다. 서버가 실행된 뒤에는 다음 API 형식으로 작업을 요청합니다.

```bash
curl -X POST http://127.0.0.1:18181/internal/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "job_id": "demo-001",
    "dataset_name": "s22_pick_v1",
    "calibration_profile": "s22_20260921",
    "require_mapping": true,
    "mapping_session_id": "rec_mapping_001",
    "demonstration_session_ids": ["rec_demo_001", "rec_demo_002"]
  }'

curl http://127.0.0.1:18181/internal/v1/jobs/demo-001
```

완료되면 다음 파일을 GPU 학습 서버로 전달한다.

```text
jobs/demo-001/dataset/s22_pick_v1_gpu_input.tgz
```

## 구성

```text
app/                    Python CLI와 내부 HTTP API
dataset_builder/        검증된 S22 → Stanford UMI Zarr 모듈
track_a/                Atlas 선택·ArUco 정렬·Episode gate 도구
docker/                 ORB-SLAM3 빌드 스크립트
Dockerfile              Ubuntu 24.04 + ORB-SLAM3 + Python
openapi.yaml             내부 API 계약
backend_contract.json   ORB-SLAM Atlas 백엔드 계약
backend/                역할 요청 스키마·검증기·게이트 정책
tests/test_package.py    빠른 자체검사
```

기본 보정 프로필은 `dataset_builder/configs`다. 다른 장비·카메라를 붙일 때는 이 값을 조용히
덮어쓰지 말고 `/profiles/<profile-id>`에 새 프로필을 추가한다.

## 범위와 판정

- `job_status.json.status == ready`
- `pipeline_report.json.dataset.training_input_status == ready`
- `<dataset>.zarr.report.json`의 해시와 calibration 상태 확인

위 조건을 통과해도 실제 로봇 파지 성공을 보장하지 않는다. 이는 GPU 학습 입력 승인까지만 뜻한다.

## 역할·Atlas 계약

- Mapping과 Demonstration 역할을 `outcome`, 길이, 파일명, ZIP/업로드 순서로 추정하지 않는다.
- Mapping은 `raw_mapping/`에 두고 가능하면 `mapping_session_id`로 정확한 세션을 지정한다.
  Mapping의 `outcome=unrated`는 허용된다. Mapping은 학습 label이 아니라 Atlas와 테이블 좌표 정렬용이다.
- Demonstration은 `raw/`에 두고 `demonstration_session_ids`로 처리 대상을 고정한다. 학습 후보는
  `outcome=success`만 통과한다.
- ArUco ID 13, 검은 정사각형 한 변 0.16 m는 **Mapping 정렬에 필수**다. 공유 Atlas로
  localization하는 각 Demonstration에서 ID 13이 계속 보이는 것은 필수가 아니며 진단값일 뿐이다.
- Demonstration은 Mapping Atlas를 `--load_map`으로만 재사용한다. 편 내부 local mapping은 허용하지만
  `--localization_only`와 `--save_map`은 사용하지 않고 Atlas 파일의 처리 전후 SHA-256을 검사한다. Mapping도 외부 Atlas도
  없는 상태에서 에피소드별 독립 map을 만드는 경로는 진단용일 뿐 학습 입력 정본이 아니다.
- 기존 Atlas를 외부 입력으로 쓸 때는 job 디렉터리 아래의 Atlas·`tx_slam_tag.json` 경로와 두 파일의
  SHA-256을 함께 요청에 보낸다. Mapping 세션과 외부 Atlas를 동시에 지정하면 거부한다.

외부 Atlas 요청 예시는 다음과 같다.

```json
{
  "job_id": "demo-002",
  "dataset_name": "s22_pick_v2",
  "demonstration_session_ids": ["rec_demo_101", "rec_demo_102"],
  "external_atlas_path": "atlas/map_atlas.osa",
  "external_alignment_path": "atlas/tx_slam_tag.json",
  "external_atlas_sha256": "<64 hex>",
  "external_alignment_sha256": "<64 hex>"
}
```

## 판정 원칙

`gopro_slam` 종료코드 0은 품질 통과가 아니다. `pipeline_report.json`에는 편마다 outcome, 입력 프레임 수,
gap 검출률·최대 미검출 구간, 첫 pose, tracking loss, 경로 길이, 연속 pose jump와 최종 포함/제외 사유가
남는다. 현재 pose 연속성 한계는 큰 재국지화 점프를 막는 잠정값(0.25 m/frame, 45 deg/frame)이며 각
trajectory report에 실제 최대값과 사용 임계값을 기록한다.

자세한 역할 계약은 `backend/ROLE_AND_GATE_POLICY.md`, 기계 검증용 요청 형식은
`backend/job_request.schema.json`을 본다.
