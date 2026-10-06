"""Waypoint-rate budget against the real execution API's trajectory validator.

실물 실행 API 의 궤적 검사식을 통과하는 waypoint 최소 간격을 계산한다.

왜 필요한가: 계약은 30Hz 인데 실물 API 는 waypoint 열을 받아 검사한다.
그 검사 중 하나가 정지→정지 quintic 모델의 피크 가속을 쓰기 때문에,
연속 궤적에서는 waypoint 를 촘촘하게 넣을수록 검사값이 커진다.
"정책 주기를 낮추면 된다"도 "시연을 느리게 하면 된다"도 답이 아니라는 것을
숫자로 보이는 것이 이 파일의 목적이다.

상수 출처 (`so101_direct_motor_pipeline` 원본 코드 🔵):
  core/models.py   dt 범위 0.05 <= dt <= 60          -> DT_MIN_S, DT_MAX_S
  core/models.py   가속 검사 5.774 * dq / dt**2      -> QUINTIC_ACC_FACTOR
  configs/real     max_accel 3.0 rad/s^2             -> ACC_LIMIT
  configs/real     max_speed 1.0 rad/s (wrist_flex)  -> VEL_LIMIT
  configs/real     그리퍼 개폐 0.08 m/s               -> GRIP_VEL_LIMIT

속도 통계 출처:
  MEAN_V   내 19편 UMI 변환 데이터 mean|action-state| 환산 (30Hz)  🟢
  P99_V    시뮬/정책 대화 check_real_limits, 98편 wrist_flex p99   🟢
  MAX_V    같은 계측의 최대                                        🟢

정정 이력 (2026-09-09): 최초 보고에서 "실물 속도 한계는 통과한다"고 썼는데
평균을 썼다. 한계 검사는 최대/p99 로 판정한다. 평균으로 판정하면 24.8% 의
위반 스텝이 보이지 않는다. 이 파일은 세 통계를 나란히 출력해서 같은 실수가
반복되지 않게 한다.
"""
from __future__ import annotations

DT_MIN_S = 0.05
DT_MAX_S = 60.0
QUINTIC_ACC_FACTOR = 5.774   # 10/sqrt(3), 정지->정지 quintic 피크 가속 계수
QUINTIC_VEL_FACTOR = 1.875   # 15/8,      정지->정지 quintic 피크 속도 계수
ACC_LIMIT = 3.0              # rad/s^2
VEL_LIMIT = 1.0              # rad/s
GRIP_VEL_LIMIT = 0.08        # m/s

MEAN_V = 0.4473              # rad/s, 내 19편 평균
P99_V = 1.3600               # rad/s, 98편 wrist_flex p99
MAX_V = 1.3630               # rad/s, 98편 wrist_flex 최대
QUINTIC_SAFE_V = VEL_LIMIT / QUINTIC_VEL_FACTOR   # 속도한계를 딱 지키는 평균속도 상한


def min_dt_for(v: float) -> tuple[float, str]:
    """Minimum waypoint spacing that passes both checks, and which one binds.

    두 검사를 동시에 통과하는 최소 waypoint 간격과 구속 조건을 돌려준다.
    연속 궤적에서 dq = v*dt 이므로 가속 검사값은 5.774*v/dt 이고,
    이것이 한계 이하가 되려면 dt >= 5.774*v/ACC_LIMIT 이다.
    """
    dt_acc = QUINTIC_ACC_FACTOR * v / ACC_LIMIT
    dt_vel = DT_MIN_S if v <= VEL_LIMIT else float("inf")
    dt = max(dt_acc, DT_MIN_S)
    if v > VEL_LIMIT:
        binding = "가속 (속도도 이미 위반)"
    elif dt_acc > DT_MIN_S:
        binding = "가속"
    else:
        binding = "dt 하한 0.05s"
    del dt_vel
    return dt, binding


def main() -> None:
    print("=== 필요 waypoint 간격 - 어느 속도 통계를 쓰느냐로 갈린다 ===")
    rows = [
        ("내 평균 (19편, mean|action-state|)", MEAN_V),
        ("시뮬/정책 p99 (98편 wrist_flex)", P99_V),
        ("시뮬/정책 최대 (98편 wrist_flex)", MAX_V),
        ("속도한계 준수시 quintic 상한 V/1.875", QUINTIC_SAFE_V),
    ]
    for label, v in rows:
        dt, binding = min_dt_for(v)
        print(f"  {label:38s} v={v:.4f} rad/s -> dt >= {dt:.3f}s = {1.0/dt:5.2f}Hz  (구속: {binding})")

    print()
    print("=== 왜 촘촘하게 넣어도 안 되는가 ===")
    print("연속 궤적에서 dq = v*dt 이므로 검사값 = 5.774*v*dt/dt^2 = 5.774*v/dt")
    print("-> dt 가 작아지면 검사값이 커진다. **촘촘할수록 더 나쁘게 판정된다.**")
    print("    (두 열을 같이 낸다 - 0909 에 한 열만 내고 라벨을 틀리게 붙인 적이 있다)")
    print(f"    {'':16s} {'v=%.4f (내 평균)' % MEAN_V:>26s} {'v=%.4f (속도한계 준수)' % QUINTIC_SAFE_V:>28s}")
    for hz in (30, 20, 10, 5, 2, 1):
        dt = 1.0 / hz
        a = QUINTIC_ACC_FACTOR * MEAN_V / dt
        b = QUINTIC_ACC_FACTOR * QUINTIC_SAFE_V / dt
        print(f"  {hz:3d}Hz dt={dt:.4f}s  "
              f"{a:8.2f} rad/s^2 = {a/ACC_LIMIT:5.1f}x    "
              f"{b:8.2f} rad/s^2 = {b/ACC_LIMIT:5.1f}x")

    print()
    print("=== 그리퍼 ===")
    print(f"  스텝 명령으로 gap 을 한 번에 닫으면 0.8771 m/s (계측 🟢) vs 한계 {GRIP_VEL_LIMIT} m/s")
    print("  램프로 펴면 통과한다 - close_s 1.0s = 30틱, 필요한 것은 19틱")

    print()
    print("=== 결론 ===")
    print(f"  dt 명시 하한 {DT_MIN_S}s (={1/DT_MIN_S:.0f}Hz) 은 계약 30Hz 와 이미 충돌한다.")
    print("  그러나 더 큰 문제는 암묵 가속 검사다 - 정책 주기를 낮춰도,")
    print("  시연을 속도한계까지 느리게 해도 dt >= "
          f"{min_dt_for(QUINTIC_SAFE_V)[0]:.3f}s 가 남는다.")
    print("  즉 리샘플러 파라미터로 해결되지 않는다. 실행 계층 인터페이스 문제다.")


if __name__ == "__main__":
    main()
