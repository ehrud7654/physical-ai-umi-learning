# 백엔드 적용 메모 — ORB-SLAM Atlas v2

이 ZIP은 전달받은 `umi_server_preprocessor/` 루트 구조를 그대로 유지한 전체 교체본이다.
기존 폴더에 일부 파일만 임의 병합하지 말고 별도 브랜치에서 전체 묶음으로 자체검사한 뒤 적용한다.

## 이번 2세션 실패에 대한 적용 방식

- `rec_001_...` 62초 세션이 Mapping이면 `raw_mapping/`에 두고 API의
  `mapping_session_id`로 정확히 지정한다. Mapping의 `outcome=unrated`는 탈락 사유가 아니다.
- `rec_000_...` 시연은 `raw/`에 두고 `demonstration_session_ids`에 넣는다.
  공유 Atlas를 불러온 Demonstration에서는 테이블 ArUco ID 13 직접 검출을 필수로 요구하지 않는다.
- 단, Mapping 자체에서는 ID 13(0.16 m)을 충분히 검출하여 `T_slam_tag`를 만들어야 한다.
- 두 세션을 모두 `raw/`에 넣고 길이/outcome/순서로 역할을 추정하는 방식은 금지한다.

예시:

```text
jobs/demo-001/
├── raw_mapping/rec_001_mapping/...
└── raw/rec_000_demo/...
```

```json
{
  "job_id": "demo-001",
  "dataset_name": "s22_pick_v1",
  "calibration_profile": "s22_20260921",
  "require_mapping": true,
  "mapping_session_id": "rec_001_mapping",
  "demonstration_session_ids": ["rec_000_demo"]
}
```

## 백엔드가 확인할 파일

- API: `openapi.yaml`, `app/server.py`
- 요청·경로 검증: `app/src/umi_preprocessor/contracts.py`, `backend/job_request.schema.json`
- 파이프라인: `app/src/umi_preprocessor/pipeline.py`, `app/run_preprocess.py`
- ORB 실행: `dataset_builder/slam/run_orbslam_linux.sh`
- trajectory 품질: `dataset_builder/slam/validate_orbslam_trajectory.py`
- 전체 계약: `backend_contract.json`, `backend/ROLE_AND_GATE_POLICY.md`

## 중요한 동작

1. Mapping: `--save_map`, ID 13/0.16 m로 Atlas→table 정렬.
2. Demonstration: 동일 Atlas를 `--load_map`으로만 사용하고 편 내부 local mapping은 허용.
   `--localization_only`와 `--save_map`은 사용하지 않으며 처리 전후 Atlas SHA-256이 같아야 함.
3. ORB 종료코드가 0이어도 trajectory 품질 gate를 별도로 적용.
4. 편별 `pipeline_report.json`에 outcome, 프레임 수, gap marker 검출률·최대 누락,
   시작 pose, tracking loss, pose jump, 포함/제외 사유를 기록.
5. 한 편도 통과하지 않으면 `status=stopped`이며 빈 학습 데이터셋을 만들지 않음.
6. 외부 Atlas를 쓰려면 job 내부 상대경로, 정렬 JSON, 두 SHA-256을 모두 제공.
7. 실제 로봇 배포 승인 기능은 포함하지 않음.

## 자체검사

컨테이너 의존성 설치 후 패키지 루트에서:

```bash
python -B backend/check_validate_job_request.py
python -B tests/test_package.py
python -B tests/test_orbslam_contract.py
python -B dataset_builder/tests/test_pipeline.py
python -B dataset_builder/tests/test_slam_tag.py
```

앞의 세 검사는 Python 표준 라이브러리만으로 실행된다. 마지막 두 검사는 `requirements.txt`의
OpenCV/NumPy/SciPy 설치 후 실행한다.
