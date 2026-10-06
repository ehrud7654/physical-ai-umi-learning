"""Does the feedback expert recover from states the demonstrations never contain?
피드백 전문가가 시연에 없는 상태에서 복구하는가?

DAgger's whole premise is that an expert can be asked "what would you do **here**",
including at states the expert's own trajectories never visit. `ScriptedPickPolicy`
cannot be asked -- it is open-loop. `ScriptedFeedbackPolicy` can be asked, but
being *askable* is not the same as being *able*: a feedback controller that only
tracks its nominal path will answer, and answer uselessly.
DAgger 의 전제는 전문가에게 "**여기서** 뭘 하겠나"를 물을 수 있다는 것이고, 그 "여기"는
전문가 자신의 궤적이 가본 적 없는 상태를 포함한다. `ScriptedPickPolicy` 는 개루프라서
물을 수 없다. `ScriptedFeedbackPolicy` 는 물을 수 있지만, **물을 수 있음**과
**답할 수 있음**은 다르다. 명목 경로만 따라가는 제어기는 답을 하고, 그 답이 쓸모없다.

So this is the instrument that validates the instrument, before any label is
collected. The gates below are fixed here, in code, before the first number.
그래서 이것은 **라벨 한 장 모으기 전에 계측기를 검증하는 계측기**다.
아래 게이트는 첫 수치가 나오기 전에 여기 코드에 확정한다.

It also answers a separate question that the 84% ceiling raised: scripted's 16
failures close at 4.7mm with 94% jaw contact -- inside `CLOSE_OK_MM` -- so they
are not approach-precision failures. Either the object is pushed away, or it is
touched by one jaw and never pinched. `obj_push_mm` and `jaw_contacts_at_close`
separate those two.
84% 천장이 남긴 다른 질문에도 답한다. scripted 실패 16건은 4.7mm·턱접촉 94% 에서
닫는다 — `CLOSE_OK_MM` 안이다. 즉 접근 정밀도 실패가 아니다. 물체가 밀려났거나,
한쪽 턱만 닿아서 집히지 않았거나 둘 중 하나다. `obj_push_mm` 과
`jaw_contacts_at_close` 가 그 둘을 가른다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from eval.stats import wilson_ci
from policy.base import Policy
from sim.mujoco.build_scene import denormalize, normalize
from sim.mujoco.env import MujocoPickEnv

# Fixed before any number was produced. G0-a is a construction check -- does the
# feedback expert do the same job as the open-loop one. G0-b is the decision:
# if it fails, DAgger stops here.
# 어떤 수치도 나오기 전에 확정했다. G0-a 는 제작 검사다 — 피드백 전문가가 개루프와
# 같은 일을 하는가. G0-b 가 결정이다. 실패하면 DAgger 는 여기서 멈춘다.
G0A_MIN = 0.80
G0B_MIN = 0.60
PERTURB_HOLD_TICKS = 5   # 30Hz 기준 0.17초. 정책이 몇 틱 헛나가는 것을 모사한다

GATES: dict[str, str] = {
    "G0-a": (
        f"교란 없는 성공률 >= {G0A_MIN:.0%}. 개루프 scripted 와 95% 구간이 겹쳐야 한다. "
        "안 겹치면 피드백 전문가가 원래 계획과 다른 일을 하고 있다는 뜻이고, "
        "그러면 이 전문가의 라벨은 시연 데이터와 다른 행동을 가르친다."
    ),
    "G0-b": (
        f"교란 후 복구 성공률 >= {G0B_MIN:.0%}. **못 넘으면 DAgger 를 착수하지 않는다.** "
        "전문가가 벗어난 상태에서 못 돌아오면 가르칠 것이 없고, "
        "라벨을 모아도 정책이 배우는 것은 '벗어난 채로 있기'다. "
        "개루프 대비는 참고값이다 — 2026-09-07 🟢 개루프도 ±0.3rad 에서 87.0% 를 "
        "유지했다. 절대 관절 목표 + 위치 제어가 변위를 흡수하기 때문이고, "
        "개루프가 못 하는 것은 복구가 아니라 임의 상태에 라벨을 붙이는 것이다."
    ),
}


@dataclass(frozen=True)
class PerturbSpec:
    """Where an episode starts, or when the arm gets knocked off its path.
    에피소드가 어디서 시작하는가, 또는 팔이 언제 경로에서 밀리는가.

    `tick < 0` means the displacement is applied to the initial joint
    configuration instead of mid-episode -- that is the condition G0-b judges on,
    because it is the only one that actually produces a state the demonstrations
    never contain. See `MujocoPickEnv.displace_joints`.
    `tick < 0` 이면 에피소드 중간이 아니라 **초기 관절 배치**에 적용한다.
    G0-b 는 이 조건으로 판정한다. 시연에 없는 상태를 실제로 만드는 유일한
    조건이기 때문이다. `MujocoPickEnv.displace_joints` 참조.
    """

    tick: int
    rad: float
    hold_ticks: int = PERTURB_HOLD_TICKS

    @property
    def is_start(self) -> bool:
        return self.tick < 0

    @property
    def label(self) -> str:
        return f"start±{self.rad}rad" if self.is_start else f"t{self.tick}±{self.rad}rad"


@dataclass
class ProbeResult:
    """One episode. Enough fields to say *why*, not only whether.
    에피소드 하나. 성공 여부뿐 아니라 **왜**를 말할 수 있는 필드까지."""

    seed: int
    condition: str
    success: bool
    ticks: int
    phase_at_end: str
    lift_height_mm: float
    min_pinch_xy_mm: float
    tick_at_min: int
    close_tick: int
    xy_at_close_mm: float
    jaw_contacts_at_grasp: int
    obj_push_at_grasp_mm: float
    obj_push_final_mm: float
    # `lift_height_mm` is read after the loop, and the loop **breaks on success**.
    # So for a successful episode it is the height at the tick the threshold was
    # crossed, not how high the arm can go. Measured 2026-09-07 🟢: commanded
    # 6.0 / 6.5 / 7.0 cm all reported ~50.1-50.6 mm with Q1 exactly 50.0 -- the
    # success criterion terminating the measurement, read as a physical ceiling.
    # `lift_height_mm` 은 루프 뒤에 읽고, 루프는 **성공에서 끊긴다.** 그래서 성공한
    # 에피소드에서는 판정선을 넘은 틱의 높이이고, 팔이 얼마나 올릴 수 있는지가
    # 아니다. 2026-09-07 실측 🟢: 명령 6.0/6.5/7.0cm 이 전부 ~50.1~50.6mm 로
    # Q1 이 정확히 50.0 이었다 — 판정 기준이 측정을 끝낸 것을 물리 한계로 읽었다.
    max_lift_mm: float
    ik_failures: int
    ik_fail_by_phase: dict[str, int]


def _perturbed(action: np.ndarray, signs: np.ndarray, rad: float, cfg: dict[str, Any]) -> np.ndarray:
    """Add a fixed joint offset to the arm part of a normalised action.
    정규화된 행동의 팔 부분에 고정 관절 오프셋을 더한다.

    Injected at the action, not at `qpos`. A policy's error arrives as a wrong
    command, so that is where the disturbance belongs; poking the simulator state
    would test a different thing (an external shove) and would not be reachable
    by the failure mode this is standing in for.
    `qpos` 가 아니라 **행동에** 주입한다. 정책의 오차는 잘못된 명령으로 도착하므로
    교란도 거기 있어야 한다. 시뮬 상태를 직접 건드리면 다른 것(외부 충격)을 재는
    것이고, 이것이 대리하려는 실패 모드로는 도달할 수 없는 상태가 된다.
    """
    raw = np.asarray(denormalize(action, cfg), dtype=float)
    raw[:5] = raw[:5] + signs * rad
    return normalize(raw, cfg, clip=True).astype(np.float32)


def probe(
    env: MujocoPickEnv,
    policy: Policy,
    seed: int,
    cfg: dict[str, Any],
    spec: PerturbSpec | None,
    stop_on_success: bool = True,
) -> ProbeResult:
    """Run one episode, optionally knocking the arm off its path partway through.
    에피소드 하나를 돌린다. 지정하면 중간에 팔을 경로에서 밀어낸다.

    `stop_on_success=False` runs to the tick limit so `max_lift_mm` measures how
    high the arm actually gets. Pass it whenever the question is capability rather
    than pass/fail -- with the default, a successful episode stops being measured
    the moment it crosses the threshold.
    `stop_on_success=False` 는 틱 제한까지 돌려서 `max_lift_mm` 이 팔이 실제로 어디까지
    올리는지를 재게 한다. 합격/불합격이 아니라 **능력**을 묻는 경우에 쓴다.
    기본값에서는 성공한 에피소드가 판정선을 넘는 순간 측정이 끝난다."""
    g_cfg = cfg["grasp"]
    close_mid = (float(g_cfg["open_cmd"]) + float(g_cfg["close_cmd"])) / 2.0
    obs = env.reset(seed=seed)
    policy.reset(seed=seed)
    obj0 = env.object_position().copy()
    gi = env.gripper_index
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=5)
    if spec is not None and spec.is_start:
        obs = env.displace_joints(signs * spec.rad)

    success = False
    ticks = 0
    max_lift = -float("inf")
    min_xy, tick_at_min = float("inf"), -1
    close_tick, xy_at_close = -1, float("nan")
    # 닫는 명령이 나간 **틱에** 재면 아직 안 닫혀 있다. 첫 실행에서 턱접촉이
    # 전부 0 으로 나온 것이 그 탓이다 🟢 2026-09-07. 닫기 구간이 끝난 뒤에 잰다.
    close_ticks = max(1, int(float(g_cfg["timing"]["close_s"]) * env.control_rate_hz))
    grasp_tick = -1
    jaws_at_grasp, push_at_grasp = -1, float("nan")

    for _ in range(env.max_ticks):
        action = np.asarray(policy.act(obs), dtype=np.float32)
        raw_grip = float(denormalize(action, cfg)[gi])
        if spec is not None and not spec.is_start and spec.tick <= ticks < spec.tick + spec.hold_ticks:
            action = _perturbed(action, signs, spec.rad, cfg)
        obs = env.step(action)
        ticks += 1
        xy, _d3 = env.pinch_to_object_m()
        if xy < min_xy:
            min_xy, tick_at_min = xy, ticks
        if close_tick < 0 and raw_grip <= close_mid:
            close_tick = ticks
            xy_at_close = xy
            grasp_tick = ticks + close_ticks
        if ticks == grasp_tick:
            obj_now = env.object_position()
            jaws_at_grasp = env.jaw_contacts()
            push_at_grasp = float(np.hypot(*(obj_now[:2] - obj0[:2])))
        max_lift = max(max_lift, env.lift_height())
        if env.is_success():
            success = True
            if stop_on_success:
                break

    obj_end = env.object_position()
    return ProbeResult(
        seed=seed,
        condition=spec.label if spec is not None else "none",
        success=success,
        ticks=ticks,
        phase_at_end=str(getattr(policy, "phase", "")),
        lift_height_mm=round(env.lift_height() * 1000, 2),
        min_pinch_xy_mm=round(min_xy * 1000, 2),
        tick_at_min=tick_at_min,
        close_tick=close_tick,
        xy_at_close_mm=round(xy_at_close * 1000, 2) if xy_at_close == xy_at_close else float("nan"),
        jaw_contacts_at_grasp=jaws_at_grasp,
        obj_push_at_grasp_mm=(
            round(push_at_grasp * 1000, 2) if push_at_grasp == push_at_grasp else float("nan")
        ),
        obj_push_final_mm=round(float(np.hypot(*(obj_end[:2] - obj0[:2]))) * 1000, 2),
        max_lift_mm=round(max_lift * 1000, 2),
        ik_failures=int(getattr(policy, "ik_failures", 0)),
        ik_fail_by_phase=dict(getattr(policy, "ik_fail_by_phase", {}) or {}),
    )


def summarise(results: list[ProbeResult]) -> dict[str, Any]:
    """Success rate with its interval, plus the fields that say why it failed.
    성공률과 구간, 그리고 왜 실패했는지 말하는 필드들."""
    n = len(results)
    ok = sum(1 for r in results if r.success)
    lo, hi = wilson_ci(ok, n) if n else (0.0, 1.0)
    fails = [r for r in results if not r.success]

    def med(vals: list[float]) -> float:
        clean = [v for v in vals if v == v]
        return round(float(np.median(clean)), 2) if clean else float("nan")

    return {
        "n": n,
        "successes": ok,
        "rate": ok / n if n else 0.0,
        "ci95": [lo, hi],
        "phase_at_end": {p: sum(1 for r in fails if r.phase_at_end == p)
                         for p in sorted({r.phase_at_end for r in fails})},
        "fail_median_xy_at_close_mm": med([r.xy_at_close_mm for r in fails]),
        "fail_median_obj_push_at_grasp_mm": med([r.obj_push_at_grasp_mm for r in fails]),
        "fail_median_obj_push_final_mm": med([r.obj_push_final_mm for r in fails]),
        "fail_jaws_at_grasp": {j: sum(1 for r in fails if r.jaw_contacts_at_grasp == j)
                               for j in sorted({r.jaw_contacts_at_grasp for r in fails})},
        "ok_jaws_at_grasp": {j: sum(1 for r in results if r.success and r.jaw_contacts_at_grasp == j)
                             for j in sorted({r.jaw_contacts_at_grasp for r in results if r.success})},
        "ok_median_obj_push_at_grasp_mm": med(
            [r.obj_push_at_grasp_mm for r in results if r.success]
        ),
        "never_closed": sum(1 for r in results if r.close_tick < 0),
        "ik_failures_total": sum(r.ik_failures for r in results),
        "ik_fail_by_phase": {
            k: sum(r.ik_fail_by_phase.get(k, 0) for r in results)
            for k in sorted({k for r in results for k in r.ik_fail_by_phase})
        },
    }


def overlaps(a: list[float], b: list[float]) -> bool:
    """Do two 95% intervals overlap?
    두 95% 구간이 겹치는가?"""
    return a[0] <= b[1] and b[0] <= a[1]


def rows(name: str, cond: str, s: dict[str, Any]) -> str:
    return (f"{name:<14} {cond:<14} {s['successes']:>3}/{s['n']:<4} "
            f"{s['rate'] * 100:>5.1f}%   CI {s['ci95'][0] * 100:>4.1f}~{s['ci95'][1] * 100:<5.1f}%")


def as_records(results: list[ProbeResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
