"""Fixture for the `joint_delta_gripper_binary` action space.
`joint_delta_gripper_binary` 행동공간의 검증 도구.

    python tools/check_gripper_binary.py

밤새 학습을 돌린 뒤에 라벨이 뒤집혀 있었다는 것을 알면 하룻밤이 날아간다.
이 공간은 세 곳(라벨 생성·표준화 예외·로짓 복호)이 서로 맞아야 동작하고,
어긋나도 예외가 나지 않는다 — 손실은 잘 떨어지고 롤아웃만 실패한다.
그래서 학습 전에 여기서 대조한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from policy.bc import (  # noqa: E402
    ACTION_SPACES,
    _GRIP_CLOSE_NORM,
    _GRIP_MID,
    _GRIP_OPEN_NORM,
    _GRIPPER,
    gripper_command_norms,
    target_scale,
    to_action,
    training_target,
)
from policy.train_bc import ArmL1GripperBCE, make_loss  # noqa: E402

SPACE = "joint_delta_gripper_binary"
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def main() -> int:
    torch.manual_seed(0)

    check("공간이 등록됐다", SPACE in ACTION_SPACES, str(ACTION_SPACES))

    open_norm, close_norm = gripper_command_norms()
    check(
        "개폐 명령이 계약 범위 안이다",
        all(-1.0 <= v <= 1.0 for v in (open_norm, close_norm)),
        f"open {open_norm:+.4f} · close {close_norm:+.4f}",
    )
    check(
        "닫힘이 열림보다 작다 (range_rad 하한 = 닫힘)",
        close_norm < open_norm,
        f"close {close_norm:+.4f} < open {open_norm:+.4f} · mid {_GRIP_MID:+.4f}",
    )

    # 라벨 생성 — 기록된 명령이 닫힘이면 1, 열림이면 0.
    n = 8
    state = torch.zeros(n, 6)
    action = torch.zeros(n, 6)
    action[:, :5] = torch.randn(n, 5) * 0.01
    action[: n // 2, _GRIPPER] = close_norm
    action[n // 2 :, _GRIPPER] = open_norm
    target = training_target(action, state, SPACE)
    labels = target[:, _GRIPPER]
    check(
        "라벨이 0/1 뿐이다",
        bool(((labels == 0.0) | (labels == 1.0)).all()),
        f"고유값 {sorted(set(labels.tolist()))}",
    )
    check(
        "닫힘 명령 -> 라벨 1",
        bool((labels[: n // 2] == 1.0).all()),
        f"{labels[: n // 2].tolist()}",
    )
    check(
        "열림 명령 -> 라벨 0",
        bool((labels[n // 2 :] == 0.0).all()),
        f"{labels[n // 2 :].tolist()}",
    )
    check(
        "팔 채널은 델타 그대로다",
        torch.allclose(target[:, :5], action[:, :5] - state[:, :5]),
    )

    # 표준화 예외 — 그리퍼 채널은 항등이어야 한다.
    mean, std = target_scale(target, SPACE)
    check(
        "그리퍼 채널 표준화가 항등이다",
        float(mean[_GRIPPER]) == 0.0 and float(std[_GRIPPER]) == 1.0,
        f"mean {float(mean[_GRIPPER])} · std {float(std[_GRIPPER])}",
    )
    plain_mean, _ = target_scale(target)
    check(
        "공간을 안 넘기면 예외가 적용되지 않는다 (인자 전달 확인)",
        float(plain_mean[_GRIPPER]) != 0.0
        or float(target[:, _GRIPPER].mean()) == 0.0,
        f"공간 없이 mean {float(plain_mean[_GRIPPER]):.4f}",
    )

    # 로짓 복호 — 중간값이 나올 수 없어야 한다.
    raw = torch.zeros(4, 6)
    raw[:, _GRIPPER] = torch.tensor([-5.0, -0.1, 0.1, 5.0])
    out = to_action(raw, torch.zeros(4, 6), SPACE, mean, std)
    grip = out[:, _GRIPPER]
    check(
        "출력이 개폐 두 값만 낸다 (중간값 없음)",
        bool(
            torch.isclose(
                grip, torch.tensor([open_norm, open_norm, close_norm, close_norm])
            ).all()
        ),
        f"{[round(v, 4) for v in grip.tolist()]}",
    )

    # 왕복 — 기록을 라벨로 만들고 다시 명령으로 복호하면 원래 명령이 나온다.
    logits = torch.zeros(n, 6)
    logits[:, _GRIPPER] = torch.where(
        labels > 0.5, torch.tensor(1.0), torch.tensor(-1.0)
    )
    back = to_action(logits, torch.zeros(n, 6), SPACE, mean, std)[:, _GRIPPER]
    check(
        "라벨 -> 명령 왕복이 원래 명령과 같다",
        bool(torch.isclose(back, action[:, _GRIPPER], atol=1e-6).all()),
    )

    # 손실 — 팔 항이 기존 L1 과 동일해야 한다 (한 번에 하나만 바꾼다).
    crit = make_loss("l1", SPACE, gripper_pos_weight=3.0)
    check("이진 공간이면 복합 손실이 나온다", isinstance(crit, ArmL1GripperBCE))
    pred = torch.randn(n, 6)
    arm_only = torch.nn.L1Loss()(pred[:, :5], target[:, :5])
    grip_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred[:, _GRIPPER], labels, pos_weight=torch.tensor(3.0)
    )
    check(
        "복합 손실 = L1(팔) + 1.0*BCE(그리퍼)",
        torch.isclose(crit(pred, target), arm_only + grip_bce, atol=1e-6),
        f"{float(crit(pred, target)):.6f} vs {float(arm_only + grip_bce):.6f}",
    )
    check(
        "pos_weight 없이 이진 공간을 요구하면 거부한다",
        _raises(lambda: make_loss("l1", SPACE)),
    )
    check(
        "다른 공간은 기존 손실 그대로다",
        isinstance(make_loss("l1", "joint_delta_gripper_abs"), torch.nn.L1Loss),
    )

    width = max(len(name) for name, _, _ in CHECKS)
    failed = 0
    for name, ok, detail in CHECKS:
        mark = "PASS" if ok else "FAIL"
        failed += int(not ok)
        print(f"[{mark}] {name.ljust(width)}  {detail}")
    print()
    if failed:
        print(f"✗ {failed}/{len(CHECKS)} 실패 — 학습을 돌리지 마라")
        return 1
    print(f"✓ {len(CHECKS)}/{len(CHECKS)} 통과")
    print(f"  open {open_norm:+.4f} · close {close_norm:+.4f} · mid {_GRIP_MID:+.4f}")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001 — 거부하는지만 본다
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
