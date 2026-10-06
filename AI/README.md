# AI 데이터·학습 코드

이 폴더는 UMI 시연을 로봇 학습 입력으로 바꾸고 상대 동작 정책을 실험·평가하는 코드입니다. 개인 작성 파일과 팀 구현이 함께 있으므로 개인 기여는 [루트 기여 기록](../CONTRIBUTIONS.md)에서 구분합니다.

## 단계별 코드

| 단계 | 주요 경로 | 입력과 산출물 |
|---|---|---|
| ORB-SLAM3·ArUco 정렬 | `umi/orbslam_intake.py`, `tools/align_orbslam_to_aruco.py`, `tools/select_orbslam_atlas_runs.py` | Mapping·시연 RAW → Atlas와 공통 좌표계의 카메라 궤적 |
| 상대 동작 데이터 | `tools/build_orbslam_umi_relative_dataset.py`, `umi/relative_dataset.py`, `tools/verify_umi_relative_dataset.py` | 궤적·그리퍼 간격 → v10 상대 EEF 동작 청크 |
| 공식 UMI 형식 변환 | `tools/convert_v10_to_umi.py` | v10 → Zarr 및 출처·검증 정보 |
| 개인 BC 기준선 | `tools/train_relative_chunk_bc.py`, `policy/relative_chunk_bc.py`, `configs/train/` | 상대 동작 데이터 → 진단용 정책 체크포인트 |
| 평가·시뮬레이션 | `eval/`, `sim/`, `tools/eval_v10_heldout_visual_response.py` | 정책 출력 → 홀드아웃·영상 반응·실행 전 검사 |

서비스형 RAW→Zarr 경로는 [`BE/GPU_/umi_server_preprocessor/`](../BE/GPU_/umi_server_preprocessor/)에, 팀의 별도 [UMI 데이터셋 빌더와 트레이너](../training_bundle/)는 `training_bundle/`에 있습니다. 학습 요청 API와 진행률·패키지 검사는 [`BE/GPU_/FastAPI_/`](../BE/GPU_/FastAPI_/)에 있습니다. 두 데이터셋 생성 경로는 서로 다른 인계 시점의 코드이므로 입력 계약을 확인해 사용해야 합니다.

## 외부 의존성과 한계

- ORB-SLAM3 실행 파일과 vocabulary는 포함하지 않았습니다. 저장소의 패치는 외부 ORB-SLAM3 소스에 적용하는 자료이며 원본 프로젝트의 출처·라이선스를 확인해야 합니다.
- 팀의 `03_umi_policy_trainer/train_policy.py`는 GitLab 추적 스냅샷에는 없었지만 로컬 인계본에서 확인해 포함했습니다. 이 코드는 별도로 받는 Stanford UMI 원본 워크스페이스의 학습 클래스를 호출합니다.
- RAW 영상, Atlas, 보정 원본, Zarr 데이터, 체크포인트는 포함하지 않았습니다. 로컬 장비·데이터·GPU가 없는 환경에서 전체 학습과 로봇 실행을 재현했다고 주장하지 않습니다.
- `configs/real/`에는 당시 리그의 잠정 보정 예시가 있습니다. 다른 장비에 그대로 적용하거나 물리 안전 검증으로 해석하면 안 됩니다.
