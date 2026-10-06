# 02 데이터셋 → Stanford UMI 정책 체크포인트

`02_umi_dataset_builder`가 만든 `dataset.zarr.zip`을 입력으로 공식 Stanford UMI Diffusion Policy를
학습하고, 전체 검증 지표로 고른 `best.ckpt`와 재현 정보를 `runs/<run_id>/`에 남긴다.
모터 제어와 Jetson export는 포함하지 않는다.

```text
dataset.zarr.zip (+ .report.json)
  ├─ check  → 키·shape·episode_ends·보고서 상태·샘플레이트 검사
  ├─ train  → 공식 TrainDiffusionUnetImageWorkspace, latest.ckpt + train_loss top-k
  ├─ select → 저장된 ckpt 전체를 검증 세트로 재추론, best.ckpt + manifest.json
  └─ inspect→ ckpt 재로딩, 입력·출력 계약 출력
```

## 1. Stanford UMI 소스

UMI 원본은 이 패키지에 복사하지 않고 경로로 참조한다. 기본 경로는
`training_bundle/third_party/umi`이며 `--umi-root` 또는 환경변수 `UMI_ROOT`로 바꿀 수 있다.
시작 시 커밋 `d095ba9590df789df5189eea5ee7e431689038a6`와 아래 패치 적용 여부를 검사한다.

| 패치 | 내용 |
|---|---|
| `patches/0001-umi-dataset-normalizer-workers.patch` | 정규화 통계 계산 DataLoader worker 32→0. Windows/8GB에서 프로세스 32개 복제 방지 |
| `patches/0002-umi-dataset-val-start-pose-noise.patch` | 시작 자세 잡음 크기를 `start_pose_noise_scale`, `val_start_pose_noise_scale` 인자로 노출. 기본값은 upstream과 동일(0.05) |

새 PC에서 처음 준비할 때:

```powershell
$umiRoot = Join-Path (Resolve-Path ..).Path 'third_party/umi'
$patchDir = (Resolve-Path .\patches).Path
git clone https://github.com/real-stanford/universal_manipulation_interface.git $umiRoot
git -C $umiRoot checkout d095ba9590df789df5189eea5ee7e431689038a6
git -C $umiRoot apply (Join-Path $patchDir '0001-umi-dataset-normalizer-workers.patch')
git -C $umiRoot apply (Join-Path $patchDir '0002-umi-dataset-val-start-pose-noise.patch')
```

## 2. Python 환경

PyTorch 2.6과 장비에 맞는 CUDA 빌드는 별도로 설치합니다. 아래 명령은 나머지 Python 의존성을 설치합니다.

```powershell
$py = "python"
& $py -m pip install -r requirements.txt
```

## 3. 실행

### 데이터셋 검사

```powershell
& $py train_policy.py check C:\data\derived\s22_v1\dataset.zarr.zip
```

02의 `dataset.zarr.report.json`을 함께 읽는다. `training_input_status`가 `ready` 또는
`provisional_camera_tcp`일 때만 학습한다. 보고서가 없거나 `development_only_unrated_input`이면
`--allow-development`를 명시해야 하며 manifest에 그 상태가 남는다. 보고서의
`native_sample_rate_hz`로 `obs_down_sample_steps`를 계산한다(30 Hz → 3, 10 Hz 정책).

### 학습 + 선택

```powershell
& $py train_policy.py train C:\data\derived\s22_v1\dataset.zarr.zip --run-id s22_v1_resnet18_e120
```

기본 프로파일은 `configs/policy_resnet18_8gb.yaml`이다(RTX 4070 Laptop 8GB 기준).
`--epochs`, `--batch`, `--seed`, `--profile`로 바꾼다. 같은 `run_id`가 있으면 덮어쓰지 않고 중단한다.
학습이 끝나면 저장된 ckpt 전체를 검증 세트로 재추론해 `best.ckpt`를 고른다. `--no-select`로 생략할 수 있다.

### 선택만 다시 실행 / 검사

```powershell
& $py train_policy.py select ..\..\runs\s22_v1_resnet18_e120
& $py train_policy.py inspect ..\..\runs\s22_v1_resnet18_e120\checkpoints\best.ckpt
```

## 4. 출력 구조

```text
runs/<run_id>/
  config.yaml            resolved Hydra 설정 (업스트림 + 프로파일 + 데이터 경로)
  dataset_check.json     check 결과
  normalizer.pkl, logs.json.txt   업스트림 워크스페이스가 남기는 정규화 통계·step 로그
  checkpoints/
    latest.ckpt          마지막 epoch, 재개용
    epoch=NNNN-train_loss=….ckpt   train_loss 기준 top-k 후보
    best.ckpt            전체 검증 위치 RMSE 최소 ckpt 복사본
  evaluation/
    <ckpt>.json, <ckpt>.predictions.npz   ckpt별 지표·예측·정답·유지 기준
    selection.json       후보별 지표와 선택 근거
  manifest.json          dataset/config sha256, UMI 커밋·패치, 환경, best 지표, 배포 가능 여부
```

## 5. 기본 학습 설정과 근거

| 항목 | 값 | 근거 |
|---|---|---|
| 인코더 | ResNet18 scratch, avg pooling | 사전학습 다운로드 불필요. 8GB에서 검증된 구성 |
| UNet | down_dims [128,256,512] | 동일 |
| batch | 4, gradient accumulation 4 | 유효 16 |
| epoch | 120 | `docs/CHECKPOINT_PLAN.md` 초안 |
| action_padding | true | 시연 끝 관측도 학습 샘플로 사용 |
| 분할 | episode 단위 15%, seed 42 | 고정 검증 세트 |
| 검증 잡음 | 0 | 패치 0002. 같은 index를 두 번 읽어 동일함을 평가 전에 검사 |
| 이미지 증강 | 학습: RandomCrop 0.95 + ColorJitter (upstream 기본) | 평가·배포: 같은 비율의 CenterCrop + Resize로 대체. manifest `inference_image_transform`에 기록 |
| best 선택 | 검증 위치 성분 RMSE(mm) 최소 | train_loss 기준 top-k는 후보 풀로만 사용 |

평가는 "현재 자세·폭 유지" 기준선과 나란히 보고한다. `beats_hold_baseline=false`이면
정책이 기준선보다 못하다는 뜻이며 실물 연결 근거가 없다.

## 6. 현재 제한

- camera→TCP 변환이 CAD 추정치인 동안 02 보고서는 `provisional_camera_tcp`이고, 이 패키지의
  manifest도 `physical_deployment_ready=false`다. 실측 후 02로 Zarr를 다시 만들고 재학습해야 한다.
- 검증 지표는 오프라인 궤적 오차다. 파지 성공률이나 실제 로봇 동작을 뜻하지 않는다.
- Jetson용 TorchScript export와 실기 추론은 별도 단계다.

## 7. 검사

```powershell
& $py -B tests\test_pipeline.py
```

합성 3 episode × 90 frame Zarr → 공식 워크스페이스 1 epoch(2 step) 학습 → ckpt 전체 재추론 →
`best.ckpt`·`manifest.json` 생성 → 새 로딩까지 한 번 통과한다. 2026-09-17 `TRAINER_SELF_TEST_OK` 확인.
