# configs/real/ 출처

이 디렉터리의 파일은 **다른 저장소에서 온 추적 사본**이다. 정본은 HW·ROS 파트이고,
여기 사본은 하드웨어 쪽 변경이 조용히 우리 판정을 옮기는 대신 **diff 로 드러나게** 하려고 둔다.
사본을 고치지 않는다. 정본이 바뀌면 교체하고 이 표를 갱신한다.

| 파일 | 출처 | sha256(앞 16) | 들여온 날 |
|---|---|---|---|
| `so101_ver1.json` | `so101_direct_motor_pipeline` 인계본 | (미기록 — 상류 커밋 해시 요청 중) | 2026-09-08 (커밋 cae686e) |
| `so101_ver1.urdf` | `so101_motor_control` ROS2 패키지 `urdf/so101_ver1.urdf` | `78e20d4e2a998bef` | 2026-09-11 |
| `ros2_controllers.yaml` | 같은 패키지 `config/controllers.yaml` | `fea2fa8eaf5e5e25` | 2026-09-11 |

## 왜 URDF 를 들여왔나

`AI/tools/check_policy_real_limits.py` 가 ROS2 백엔드의 거부 조건을 재현하려면
`ros2_control` 블록의 관절별 `zero_tick`·`direction`·`min_tick`/`max_tick`·
`min_rad`/`max_rad` 와 그리퍼 `closed_tick`/`open_tick`/`max_width_m` 이 필요하다.
그 값은 URDF 안에 embedded 돼 있고 AI 저장소에는 없었다.

## ⚠️ 알려진 불일치 — elbow_flex 하한

| 출처 | elbow_flex 하한 |
|---|---:|
| `so101_ver1.urdf` / xacro `min_rad` | **−1.57079632679** |
| `AI/configs/so101.yaml`, `so101_ver1.json` | **−1.69** |

0.119 rad = 6.8°. **우리가 더 넓다.** 시뮬 궤적이 −1.5708 아래로 내려가면 ROS2 가
쓰기 주기 전체를 거부한다. **어느 쪽이 정본인지 HW 확인 전까지 우리 config 를 고치지 않는다.**
근거: `AI/docs/MEASURE_ros2_exec_intake_0911.md`

## 상류 커밋 해시가 없다

세 파일 모두 상류 저장소의 커밋 해시를 기록하지 못했다. L70 과 같은 계열의 구멍이다 —
어느 버전으로 잰 수치인지 사후에 확인할 수 없다. **HW·ROS 파트에 해시 요청 중.**
받으면 위 표에 채운다.
