"""Verify the domain randomiser before trusting any number it produces.
도메인 랜덤화기가 내놓을 수치를 믿기 전에 계측기 자체를 검증한다.

A broken instrument does not announce itself. If condition B silently perturbs
nothing, every policy scores the same under A and B, the collapse rate comes out
near zero, and the conclusion reads "simulation transfers well" — the most
expensive possible wrong answer. These checks exist so that reading cannot
happen by accident.
고장난 계측기는 스스로 알리지 않는다. 조건 B 가 조용히 아무것도 흔들지 않으면
모든 정책이 A 와 B 에서 같은 점수를 받고, 붕괴율은 0 근처로 나오고, 결론은
"시뮬 전이가 잘 된다"가 된다. 나올 수 있는 가장 비싼 오답이다. 이 검사들은
그 결론이 사고로 나오는 일을 막으려고 있다.

Run from AI/ before any transfer measurement. Exit code 1 means do not measure.
전이 계측 전에 AI/ 에서 실행한다. 종료코드 1 은 계측하지 말라는 뜻이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from sim.mujoco.build_scene import build_model, load_config  # noqa: E402
from sim.mujoco.domain import CONDITION_A, CONDITION_B, DomainRandomizer  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402

PIXEL_DIFF_FLOOR = 1.0  # 0~255 스케일. 이보다 작으면 외관 외란이 사실상 없는 것이다
PLACEMENT_TOL_M = 0.002  # A/B 물체 배치가 이보다 벌어지면 시드 스트림이 섞인 것이다


def _snapshot(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    return {
        "geom_friction": model.geom_friction.copy(),
        "geom_rgba": model.geom_rgba.copy(),
        "body_mass": model.body_mass.copy(),
        "cam_pos": model.cam_pos.copy(),
        "cam_quat": model.cam_quat.copy(),
    }


def _differs(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    return [k for k in a if not np.allclose(a[k], b[k])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="픽셀 차이 검사를 건너뛴다. 외관 외란 검증이 빠지므로 권장하지 않는다",
    )
    args = parser.parse_args()

    cfg = load_config()
    model = build_model(cfg)
    data = mujoco.MjData(model)
    nominal = _snapshot(model)
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  {'통과' if ok else '실패'}  {label}{('  — ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    print("도메인 랜덤화기 검증\n")

    # 1. 조건 A 는 공칭 모델을 바꾸지 않아야 한다.
    rand_a = DomainRandomizer(CONDITION_A)
    rand_a.bind(model)
    rand_a.apply(model, data, seed=args.seed)
    changed = _differs(nominal, _snapshot(model))
    check("조건 A 는 모델을 바꾸지 않는다", not changed, f"바뀐 항목 {changed}" if changed else "")

    # 2. 조건 B 는 실제로 무언가를 바꿔야 한다.
    rand_b = DomainRandomizer(CONDITION_B)
    rand_b.bind(model)
    rand_b.apply(model, data, seed=args.seed)
    after_b = _snapshot(model)
    changed_b = _differs(nominal, after_b)
    for key in ("geom_friction", "body_mass", "cam_pos", "cam_quat", "geom_rgba"):
        check(f"조건 B 가 {key} 를 흔든다", key in changed_b)

    # 3. 같은 시드는 같은 결과를 내야 한다 (계측 재현성).
    rand_b.apply(model, data, seed=args.seed)
    check("같은 시드 → 같은 외란", not _differs(after_b, _snapshot(model)))

    # 4. 반복 적용이 누적되면 안 된다 (매번 공칭에서 복원).
    rand_b.apply(model, data, seed=args.seed)
    rand_b.apply(model, data, seed=args.seed)
    check("반복 적용이 누적되지 않는다", not _differs(after_b, _snapshot(model)))

    # 5. 다른 시드는 다른 결과를 내야 한다.
    rand_b.apply(model, data, seed=args.seed + 1)
    check("다른 시드 → 다른 외란", bool(_differs(after_b, _snapshot(model))))

    # 6. 카메라 쿼터니언은 단위여야 한다. 아니면 렌더가 조용히 망가진다.
    norms = np.linalg.norm(model.cam_quat, axis=1)
    check("카메라 쿼터니언이 단위다", bool(np.allclose(norms, 1.0, atol=1e-9)),
          f"norm {np.round(norms, 9).tolist()}")

    # 7. 물체 배치는 A 와 B 에서 같아야 한다. 다르면 조건 비교가 아니라
    #    배치 비교가 되어 붕괴율 전체가 무의미해진다.
    with MujocoPickEnv(cfg, render=False, domain=DomainRandomizer(CONDITION_A)) as ea:
        ea.reset(seed=args.seed)
        xy_a = ea.object_position()[:2].copy()
    with MujocoPickEnv(cfg, render=False, domain=DomainRandomizer(CONDITION_B)) as eb:
        eb.reset(seed=args.seed)
        xy_b = eb.object_position()[:2].copy()
    gap = float(np.linalg.norm(xy_a - xy_b))
    check("A/B 물체 배치가 동일하다", gap < PLACEMENT_TOL_M,
          f"차이 {gap * 1000:.3f}mm (허용 {PLACEMENT_TOL_M * 1000:.0f}mm, 물리 외란에 의한 안착 차이)")

    # 8. 외관 외란이 실제로 픽셀을 바꾸는가. 이게 0 이면 시각 정책 입장에서
    #    조건 B 는 조건 A 와 완전히 같은 세계다.
    if args.skip_render:
        print("  건너뜀  픽셀 차이 검사 (--skip-render)")
    else:
        with MujocoPickEnv(cfg, render=True, domain=DomainRandomizer(CONDITION_A)) as ea:
            obs_a = ea.reset(seed=args.seed)
        with MujocoPickEnv(cfg, render=True, domain=DomainRandomizer(CONDITION_B)) as eb:
            obs_b = eb.reset(seed=args.seed)
        for cam in obs_a.images:
            d = float(
                np.mean(
                    np.abs(
                        obs_a.images[cam].astype(np.int16)
                        - obs_b.images[cam].astype(np.int16)
                    )
                )
            )
            check(f"조건 B 가 카메라 {cam} 픽셀을 바꾼다", d > PIXEL_DIFF_FLOOR,
                  f"평균 절대차 {d:.3f}/255 (하한 {PIXEL_DIFF_FLOOR})")

    info = DomainRandomizer(CONDITION_B)
    info.bind(model)
    desc = info.describe()
    print(f"\n광원 개수 {desc['n_lights']}")
    if "warning" in desc:
        print(f"⚠️ {desc['warning']}")

    print(f"\n{'전부 통과 — 전이 계측을 진행해도 된다' if not failures else '실패 항목: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
