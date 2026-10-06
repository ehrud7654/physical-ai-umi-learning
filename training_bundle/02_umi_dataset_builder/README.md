# S22 앱 데이터 → Stanford UMI 학습 입력

이 패키지는 수집 앱 원본을 보존하면서 다음 파생물을 만든다.

```text
raw session
  ├─ prepare → telemetry.json, gripper_width.csv
  ├─ ORB-SLAM3 on Jetson → camera_trajectory.csv
  └─ build → dataset.zarr.zip, dataset.zarr.report.json
```

모터 제어 코드는 포함하지 않는다.

## 1. Python 환경

PC의 기존 환경을 그대로 쓸 수 있다.

```powershell
python -m pip install -r requirements.txt
```

## 2. 세션 전처리

```powershell
python build_dataset.py prepare `
  C:\data\s22_sessions\rec_123 `
  C:\data\processed\rec_123
```

생성 결과:

- `telemetry.json`: 실제 카메라 timestamp와 보정된 IMU timestamp
- `gripper_width.csv`: ArUco ID 0·1의 16 mm 마커로 계산한 개폐 폭(millimetre)
- `gripper_report.json`, `prepare_report.json`: 검출률·연속 누락·범위
- `s22_mono_inertial.yaml`: 검증된 S22 ORB-SLAM 설정

## 2b. 매핑 세션과 테이블 마커 (Stanford UMI 방식)

시연마다 SLAM을 처음부터 초기화하면 1~8초의 초기화 지연이 생기고 세션마다 원점이 달라진다.
UMI처럼 장소마다 **매핑 세션 1개**를 먼저 찍어 지도를 저장하고, 시연은 그 지도에 대해 재위치(relocalize)한다.

- 마커: [UMI table_marker.pdf](https://umi-gripper.github.io/share/table_marker.pdf)를 100% 배율로 인쇄한
  16 cm ArUco ID 13. 설정은 `configs/s22_slam_tag.json`. ChArUco 보정판(18 mm ID 13 포함)은 치운다.
- 매핑 녹화: 마커를 테이블 중앙에 두고 약 1분간 천천히 작업 공간을 훑은 뒤, 과제 동작을 흉내 내며 물체 주변을 여러 각도로 본다.
- SLAM 실행 스크립트는 `ORB_SLAM_SAVE_MAP`(매핑), `ORB_SLAM_LOAD_MAP`(시연) 환경변수로 `--save_map`/`--load_map`을 전달한다.
- 세계 좌표: 매핑 궤적과 마커 검출로 `T_slam_tag`를 구한다. 이후 `T_world_tcp = inv(T_slam_tag) @ T_slam_camera @ T_camera_tcp`.

```powershell
python build_dataset.py slam-tag C:\data\s22_sessions\rec_mapping C:\data\processed\rec_mapping
```

`tx_slam_tag.json`이 생기며, episode plan 최상위에 `"slam_tag": "<경로>"`를 적으면 `build`가 모든 에피소드를
마커 기준 좌표로 변환한다. 없으면 기존처럼 녹화별 SLAM 원점을 쓰고 보고서 `world_frame`에 표시된다.
`dataset.json`의 `maximum_lost_frames_after_initialization`(10)은 UMI의 "잃은 프레임 10개 초과 시 제외" 규칙이다.

SLAM 실행 설정(2026-09-17 확정): `slam/s22_mono_inertial.yaml`의 ORB 특징점 2000(1250에서 증가),
그리퍼 마스크 `configs/s22_slam_mask.json`(핸들 몸체 y≥0.76, 손가락 영역 x 0.20–0.80 · y 0.56–0.78, 화면의 35%;
`build_dataset.py slam-mask`로 PNG 생성, 실행 스크립트가 자동 적용, `ORB_SLAM_MASK=""`로 해제), 그리고
`slam/0002-relax-post-relocalization-inliers.patch`(재위치 직후 inlier 기준 30→15). 마스크를 43%로 크게 잡으면
근거리 특징점이 사라져 매핑 초기화가 56초까지 밀렸다. `dataset.json`의 `maximum_marker_missing_run_frames`는
15(0.5초)로, 그 안의 그리퍼 마커 누락은 선형 보간한다.

### 지도 없이 쓰는 방식: 세션별 초기화 + 준비 구간 잘라내기

단조로운 장면에서는 지도 재위치가 실패할 수 있다(2026-09-17 나무 벽·흰 테이블에서 3/3 실패). 이때는
시연마다 **처음 2~3초 동안 테이블 마커가 보이는 상태로 살짝 움직여** SLAM을 초기화한 뒤 과제를 수행한다.

- `dataset.json`의 `episode_start_trim_s`(3.0): 각 에피소드에서 녹화 시작 후 이 시간 이전 프레임을 버린다.
  준비 움직임이 학습에 섞이지 않는다. 남은 프레임이 `minimum_episode_frames` 미만이면 에피소드 제외.
- 같은 세션의 궤적과 영상으로 `slam-tag`를 돌리면 세션별 `tx_slam_tag.json`이 생긴다. plan의 각 episode에
  `"slam_tag"`를 적으면 세션마다 자기 마커 좌표로 변환되어 모든 에피소드가 같은 테이블 기준 좌표를 갖는다
  (`world_frame: per_recording_table_tag`). 태그가 있는 에피소드와 없는 에피소드를 섞으면 `build`가 거부한다.

## 3. Jetson에서 ORB-SLAM3

기존에 빌드한 `umi-orb-slam3:b741dca-s22` 이미지를 재사용한다.

새 Jetson에서 한 번만 빌드할 때는 Stanford UMI 포크를 commit
`b741dca39015330ef4bcc3a85f89493503ade04b`로 checkout하고 패치를 적용한다.

```sh
cd /path/to/umi-workspace/third_party/umi-orb-slam3
git apply /path/to/public-repo/training_bundle/02_umi_dataset_builder/slam/0001-use-recorded-sensor-timestamps.patch
sudo sh /path/to/public-repo/training_bundle/02_umi_dataset_builder/slam/build_orbslam_jetson.sh "$PWD"
```

```sh
sudo sh slam/run_orbslam_jetson.sh \
  /data/raw/rec_123 \
  /data/processed/rec_123 \
  /data/processed/rec_123/orbslam
```

`camera_trajectory.csv`를 PC의 해당 processed 폴더로 가져온다. 첫 3초 보드 구간에서 초기화되지 않거나 작업 중 추적이 끊긴 세션은 학습 입력에서 제외한다.

## 4. episode plan과 Zarr 생성

`configs/episodes.example.json`을 복사해 실제 경로를 적는다. 상대 경로는 plan JSON이 있는 폴더 기준이다.

```powershell
python build_dataset.py build configs/episodes.json data/derived/s22_v1/dataset.zarr.zip
python build_dataset.py inspect data/derived/s22_v1/dataset.zarr.zip
```

기본 빌드는 `manifest.json`의 `outcome=success`만 받는다. `--allow-unrated`는 코드 시험용이며 보고서가 `development_only_unrated_input`으로 남는다.

생성 키는 Stanford UMI의 ReplayBuffer 계약과 같다.

- `camera0_rgb`: 224×224 RGB, 중앙 crop, ArUco 마커 inpaint
- `robot0_eef_pos`: world 기준 TCP 위치(m)
- `robot0_eef_rot_axis_angle`: world 기준 TCP 회전축-각도(rad)
- `robot0_gripper_width`: 접촉면 전체 간격(m)
- `robot0_demo_start_pose`, `robot0_demo_end_pose`
- `meta/episode_ends`

앱 원본은 약 30 Hz다. 10 Hz 정책을 학습할 때 UMI shape/config의 observation/action
`down_sample_steps`를 `3`으로 둔다. 생성 보고서의 `native_sample_rate_hz`를 먼저 확인한다.

## 좌표와 현재 제한

`T_world_tcp = T_world_camera @ T_camera_tcp`를 사용한다. 현재 `configs/s22_camera_tcp.json`은 Blender의 S22 ultra-wide CAD 후보와 저장 영상의 180도 축 가정을 합성한 값이다. 렌즈 entrance pupil과 축 매핑을 실측하지 않았으므로 보고서의 `physical_deployment_ready`는 `false`다. 이 값을 물리 보정한 뒤 같은 파일의 상태와 행렬을 갱신해야 실제 로봇 배포 후보가 된다.

## 검사

```powershell
python -B tests/test_pipeline.py
python -B tests/test_slam_tag.py
```

첫 검사는 임시 영상 → TCP 라벨 → Zarr 저장 → 재로딩을 한 번 통과한다. 둘째는 알려진 자세로 렌더한
16 cm 마커 영상에서 `T_slam_tag`를 복원한다(위치 15 mm, 회전 1도 이내). 실제 파지 성공이나 ORB-SLAM 정확도 검사는 아니다.

2026-09-16 확인 결과 `PIPELINE_SELF_TEST_OK`를 통과했다. 실제 정지 세션
`rec_example`도 전처리했으며 81프레임, ArUco 동시 검출
80프레임, 최대 연속 누락 1프레임, 폭 중앙값 62.072 mm로 사용자 실측 62 mm를 재현했다.
