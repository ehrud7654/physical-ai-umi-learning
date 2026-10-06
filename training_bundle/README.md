# 팀 UMI 데이터셋·학습 인계 코드

이 폴더는 노트북의 프로젝트 로컬 인계본에서 확인한 팀 코드를 담습니다. 원본 GitLab `master`의 추적 파일에는 없던 자료이므로, 원본 커밋 이력을 이 폴더의 작성 근거로 삼지 않습니다. 황도경의 개인 AI 구현은 [기여 상세](../CONTRIBUTIONS.md)에서 따로 구분합니다.

| 경로 | 역할 |
|---|---|
| [`02_umi_dataset_builder/`](02_umi_dataset_builder/) | S22 수집 세션, ORB-SLAM3 궤적, ArUco·그리퍼 관측을 Stanford UMI 형식의 Zarr로 구성 |
| [`03_umi_policy_trainer/`](03_umi_policy_trainer/) | Zarr 계약 검사, Stanford UMI 학습 워크스페이스 호출, 체크포인트 평가·선택·기록 |

`AI/`의 Atlas→v10→Zarr 실험 경로, `BE/GPU_/umi_server_preprocessor/`의 서버 전처리 경로와 이 폴더의 `02_umi_dataset_builder/`는 서로 다른 개발 시점의 구현입니다. 하나의 실행 명령으로 자동 연결된다고 가정하지 말고 각 README의 입력 계약을 확인하세요. `BE/GPU_/FastAPI_/` 학습 어댑터는 `UMI_TRAINING_BUNDLE_ROOT` 아래 이 폴더의 `03_umi_policy_trainer/train_policy.py`를 호출하도록 설계됐습니다.

## 외부 준비물

- ORB-SLAM3 실행 파일, vocabulary, 장치에 맞는 보정과 실제 시연 데이터
- [Stanford UMI 원본 저장소](https://github.com/real-stanford/universal_manipulation_interface)의 고정 커밋 `d095ba9590df789df5189eea5ee7e431689038a6`와 `03_umi_policy_trainer/patches/`의 두 패치
- Python·PyTorch CUDA 환경과 두 패키지의 `requirements.txt`

Stanford UMI 원본은 여기에 복사하지 않았습니다. 트레이너는 기본적으로 `training_bundle/third_party/umi`를 찾으며 `UMI_ROOT` 또는 `--umi-root`로 위치를 바꿀 수 있습니다. 이 코드의 학습 실행 기록과 120 epoch 결과는 황도경의 개인 기여로 표시하지 않습니다.

실제 로봇 적용에는 카메라→TCP 보정과 안전 검증이 더 필요합니다. 포함된 기본 보정값과 오프라인 검증 지표는 실기 성공을 보증하지 않습니다.
