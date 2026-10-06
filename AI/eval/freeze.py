"""Is the frozen policy stuck at an observation fixpoint, or more deeply broken?
얼어붙은 정책이 관측 고정점에 갇힌 것인가, 더 깊게 망가진 것인가?

`audit_demos` measured a 22-tick window (Q1 = Q3 = 22, all 98 episodes) in which
the demonstration's observation is completely static -- gripper saturated on the
cube, arm stationary -- followed by a label that jumps to the lift delta. Under L1
the median over that neighbourhood is "hold", and holding keeps the observation
static, which keeps the prediction "hold". That is a self-sustaining fixpoint, and
it matches the observed behaviour exactly: bc grasps at tick 102 with xy 2.7 mm
and then does not move until the 200-tick limit 🟢.
`audit_demos` 가 정지 창 22틱(Q1=Q3=22, 98편 전부)을 실측했다. 그리퍼가 큐브에
물려 포화되고 팔도 멈춘 구간이고, 그 다음 프레임에서 라벨만 들기 델타로 뛴다.
L1 이면 그 이웃의 중앙값은 "유지"이고, 유지하면 관측이 계속 정지하므로 예측도
계속 "유지"다 — 자기유지 고정점. 관측된 행동과 정확히 맞는다: bc 는 틱 102 에
xy 2.7mm 로 잡고 틱 제한 200 까지 움직이지 않는다 🟢.

The test: detect the freeze, hand the arm one short scripted lift, give control
back. If the fixpoint is the whole bottleneck, one push is enough.
검사: 동결을 감지하고 scripted 들기를 짧게 한 번 넘겨준 뒤 제어권을 돌려준다.
고정점이 병목 전부라면 한 번으로 충분하다.

Two things are pre-registered here, in code, before the first number.
두 가지를 첫 수치가 나오기 전에 여기 코드에 확정한다.

1. **One kick only.** Repeated kicks would measure something else -- if the policy
   needs to be pushed through the whole lift, it did not learn that phase at all,
   and the prescription is different. So the re-freeze count is reported
   separately and a re-freeze is *not* kicked again.
   **kick 은 1회만.** 반복 kick 은 다른 것을 재게 된다 — 들기 구간 전체를 밀어줘야
   한다면 그 구간을 아예 못 배운 것이고 처방이 다르다. 그래서 재동결 횟수를 따로
   보고하고, 재동결에는 다시 kick 하지 않는다.
2. **The kick is privileged.** It uses the object's true position. This is a
   diagnostic, never a deployable policy.
   **kick 은 특권 정보다.** 물체 참값을 쓴다. 진단 전용이고 배포 가능한 정책이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import mujoco
import numpy as np

from eval.stats import wilson_ci
from policy.base import Policy
from sim.mujoco.build_scene import denormalize, normalize
from sim.mujoco.env import MujocoPickEnv

# Fixed before any number was produced -- but corrected once, by a smoke test,
# before any number was produced. Recorded rather than quietly changed.
# 어떤 수치도 나오기 전에 확정했다 — 단, 수치가 나오기 전에 연기시험으로 한 번
# 정정했다. 조용히 바꾸지 않고 기록한다.
#
# The first version used FREEZE_TICKS = 10 and fired on **100% of episodes at tick
# 64** while `scripted_fb` went on to succeed 70% of them 🟢 2026-09-07. Tick 64 is
# the end of the descend phase, where the arm legitimately slows to a stop before
# closing. A detector that calls normal behaviour a freeze cannot tell "the policy
# is stuck" from "the demonstration holds still here", and that is the whole
# distinction this instrument exists to make.
# 첫 판은 FREEZE_TICKS = 10 이었고 **100% 편에서 틱 64 에 발화**했는데 그 상태로
# `scripted_fb` 는 70% 성공했다 🟢 2026-09-07. 틱 64 는 하강 phase 의 끝이고 팔이
# 닫기 전에 정당하게 멈추는 지점이다. 정상 행동을 동결로 부르는 감지기는 "정책이
# 갇혔다"와 "시연이 여기서 멈춘다"를 구분할 수 없고, 그 구분이 이 계측기의 존재
# 이유 전부다.
#
# So the threshold now comes from the measurement instead of a guess: the
# demonstration's own static window is 22 ticks (Q1 = Q3 = 22, all 98 episodes,
# MEASURE_static_window_0907). Anything longer than that is not in the data.
# 그래서 문턱을 추측이 아니라 실측에서 가져온다. 시연 자신의 정지 창이 22틱이다
# (Q1=Q3=22, 98편 전부, MEASURE_static_window_0907). 그보다 길면 데이터에 없는 것이다.
FREEZE_EPS_RAD = 0.002    # 틱당 팔 관절 변화 (rad). 30Hz 에서 0.06 rad/s
FREEZE_TICKS = 30         # 실측 정지 창 22틱 + 여유 8. 1.0초
KICK_TICKS = 5            # 넘겨줄 틱 수. 0.17초
KICK_MIN_TICK = 20        # 이 틱 이전의 정지는 시작 정착으로 보고 kick 하지 않는다

GATES: dict[str, str] = {
    "fixpoint": (
        "**얼어붙은 편만 놓고** kick 성공률의 95% 구간 하한 > 무개입 구간 상한 "
        "→ **고정점 확정.** 처방은 정지 창 재표집·청킹·위상정보이고 DAgger 는 우회로다. "
        "전체 성공률로 재지 않는다 — 동결 편이 소수면 완벽한 처방도 총계를 못 움직인다. "
        "구간이 겹치면 고정점 가설을 내리고 접근 발산 계열을 우선한다."
    ),
    "one_push_enough": (
        "재동결 비율 < 30% 여야 '고정점 하나가 병목'이라고 말할 수 있다. "
        "그보다 높으면 정책이 들기 구간 전체를 못 배운 것이고, "
        "kick 이 성공률을 올려도 처방은 재표집이 아니라 행동 표현·청킹이다."
    ),
}


@dataclass
class FreezeResult:
    """One episode. Says whether it froze, when, and what the kick did.
    에피소드 하나. 얼었는지·언제·kick 이 무엇을 했는지."""

    seed: int
    condition: str
    success: bool
    ticks: int
    froze: bool
    freeze_tick: int
    kicked: bool
    refroze: bool
    refreeze_tick: int
    lift_height_mm: float
    min_pinch_xy_mm: float
    jaw_contacts_at_freeze: int
    # A kick that did nothing and a policy that ignored the kick look identical in
    # the success rate. These two say which happened. 🟢 2026-09-07: the expert's
    # own failures are the episodes whose lift target has no IK solution, so the
    # kick is a no-op there -- reading that as "no fixpoint" would be wrong.
    # 아무것도 안 한 kick 과 kick 을 무시한 정책은 성공률에서 똑같이 보인다.
    # 이 둘이 어느 쪽인지 말한다. 🟢 2026-09-07: 전문가 자신의 실패 편은 들어올림
    # 목표에 IK 해가 없는 편이라 거기서 kick 은 무동작이다 — 그걸 "고정점 아님"으로
    # 읽으면 틀린다.
    kick_moved_rad: float
    kick_ik_failures: int


def _arm_step(prev: np.ndarray, cur: np.ndarray, cfg: dict[str, Any]) -> float:
    """Arm joint movement between two observations, in radians.
    두 관측 사이 팔 관절 이동량 (rad)."""
    a = np.asarray(denormalize(prev, cfg), dtype=float)[:5]
    b = np.asarray(denormalize(cur, cfg), dtype=float)[:5]
    return float(np.linalg.norm(b - a))


def lift_kick_action(
    env: MujocoPickEnv, obs: Any, cfg: dict[str, Any], step_rad: float = 0.04
) -> np.ndarray:
    """A joint-space nudge that raises the grasp point. No IK.
    파지점을 올리는 관절공간 밀기. IK 를 쓰지 않는다.

    The first version of this kick asked `ScriptedFeedbackPolicy` for a LIFT
    action, which needs an IK solution at the lift pose. Measured 2026-09-07 🟢:
    on the expert's **own** failure episodes that solve fails 100% of the time --
    those are exactly the placements whose lift target is unreachable -- so the
    kick moved the arm 0.0022 rad and did nothing. "Pushing it did not help" was
    then a statement about the instrument, not about the policy.
    이 kick 의 첫 판은 `ScriptedFeedbackPolicy` 에 LIFT 행동을 물었는데, 그건 들기
    자세의 IK 해를 요구한다. 2026-09-07 실측 🟢: 전문가 **자신의** 실패 편에서 그
    해가 100% 실패한다 — 그 편들이 바로 들기 목표가 도달 불가인 배치다 — 그래서
    kick 이 팔을 0.0022 rad 움직이고 아무것도 하지 않았다. 그 상태의 "밀어줘도
    안 된다"는 정책에 대한 진술이 아니라 계측기에 대한 진술이었다.

    So the kick asks a smaller question with a guaranteed answer: which joint
    direction raises the pinch point? A five-point numerical gradient of the grasp
    point's z against the arm joints, stepped along. Always defined, never fails.
    그래서 kick 은 답이 보장된 더 작은 질문을 한다 — 어느 관절 방향이 파지점을
    올리는가. 팔 관절에 대한 파지점 z 의 5점 수치 기울기를 따라 한 걸음.
    항상 정의되고 실패하지 않는다.
    """
    from sim.mujoco.kinematics import grasp_point

    offset = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    model, data = env.model, env.data
    q0 = data.qpos.copy()
    v0 = data.qvel.copy()
    h = 1e-3
    grad = np.zeros(5)
    try:
        for j in range(5):
            data.qpos[:] = q0
            data.qpos[j] += h
            mujoco.mj_forward(model, data)
            zp = float(grasp_point(model, data, offset)[2])
            data.qpos[:] = q0
            data.qpos[j] -= h
            mujoco.mj_forward(model, data)
            zm = float(grasp_point(model, data, offset)[2])
            grad[j] = (zp - zm) / (2 * h)
    finally:
        data.qpos[:] = q0
        data.qvel[:] = v0
        mujoco.mj_forward(model, data)

    norm = float(np.linalg.norm(grad))
    q_now = np.asarray(denormalize(obs.state, cfg), dtype=float)
    if norm < 1e-9:
        return np.asarray(obs.state, dtype=np.float32)
    q_arm = q_now[:5] + (grad / norm) * step_rad
    grip = float(cfg["grasp"]["close_cmd"])
    return normalize(np.concatenate([q_arm, [grip]]), cfg, clip=True).astype(np.float32)


def probe_freeze(
    env: MujocoPickEnv,
    policy: Policy,
    seed: int,
    cfg: dict[str, Any],
    kick: bool,
) -> FreezeResult:
    """Run one episode, optionally handing the arm one short scripted lift.
    에피소드 하나를 돌린다. 지정하면 scripted 들기를 짧게 한 번 넘겨준다."""
    obs = env.reset(seed=seed)
    policy.reset(seed=seed)
    g = cfg["grasp"]
    close_mid = (float(g["open_cmd"]) + float(g["close_cmd"])) / 2.0
    gi = env.gripper_index

    success = False
    ticks = 0
    static_run = 0
    closed = False
    froze = kicked = refroze = False
    freeze_tick = refreeze_tick = -1
    jaws_at_freeze = -1
    kick_left = 0
    min_xy = float("inf")
    prev_state = obs.state.copy()
    kick_moved = 0.0

    for _ in range(env.max_ticks):
        if kick_left > 0:
            action = lift_kick_action(env, obs, cfg)
            kick_left -= 1
            in_kick = True
        else:
            action = np.asarray(policy.act(obs), dtype=np.float32)
            in_kick = False

        if float(denormalize(action, cfg)[gi]) <= close_mid:
            closed = True
        obs = env.step(action)
        ticks += 1
        xy, _d3 = env.pinch_to_object_m()
        min_xy = min(min_xy, xy)

        moved = _arm_step(prev_state, obs.state, cfg)
        prev_state = obs.state.copy()
        if in_kick:
            kick_moved += moved
        static_run = static_run + 1 if moved < FREEZE_EPS_RAD else 0

        # 파지를 시도한 뒤의 정체만 센다. 파지 전 정체는 접근 실패이고 다른 처방이다.
        if closed and static_run >= FREEZE_TICKS and ticks >= KICK_MIN_TICK:
            if not froze:
                froze = True
                freeze_tick = ticks
                jaws_at_freeze = env.jaw_contacts()
                if kick:
                    kicked = True
                    kick_left = KICK_TICKS
                    static_run = 0

            elif kicked and not refroze:
                # kick 이후 다시 얼었다. 다시 kick 하지 않는다 — 사전등록된 규칙이다.
                refroze = True
                refreeze_tick = ticks

        if env.is_success():
            success = True
            break

    return FreezeResult(
        seed=seed,
        condition="kick" if kick else "none",
        success=success,
        ticks=ticks,
        froze=froze,
        freeze_tick=freeze_tick,
        kicked=kicked,
        refroze=refroze,
        refreeze_tick=refreeze_tick,
        lift_height_mm=round(env.lift_height() * 1000, 2),
        min_pinch_xy_mm=round(min_xy * 1000, 2),
        jaw_contacts_at_freeze=jaws_at_freeze,
        kick_moved_rad=round(kick_moved, 5),
        kick_ik_failures=0,  # IK 를 쓰지 않는다. 남겨둔 것은 기록 호환용
    )


def summarise(results: list[FreezeResult]) -> dict[str, Any]:
    """Success rate with its interval, plus what the freeze looked like.
    성공률과 구간, 그리고 동결의 모양."""
    n = len(results)
    ok = sum(1 for r in results if r.success)
    lo, hi = wilson_ci(ok, n) if n else (0.0, 1.0)
    froze = [r for r in results if r.froze]
    kicked = [r for r in results if r.kicked]
    return {
        "n": n,
        "successes": ok,
        "rate": ok / n if n else 0.0,
        "ci95": [lo, hi],
        "froze": len(froze),
        "froze_frac": len(froze) / n if n else 0.0,
        "freeze_tick_median": float(np.median([r.freeze_tick for r in froze])) if froze else None,
        "jaws_at_freeze": {j: sum(1 for r in froze if r.jaw_contacts_at_freeze == j)
                           for j in sorted({r.jaw_contacts_at_freeze for r in froze})},
        "kicked": len(kicked),
        "refroze": sum(1 for r in kicked if r.refroze),
        "refroze_frac": (sum(1 for r in kicked if r.refroze) / len(kicked)) if kicked else None,
        "success_given_kicked": (
            sum(1 for r in kicked if r.success) / len(kicked)) if kicked else None,
        # kick 이 실제로 팔을 움직였나. 이 값이 0 에 가까우면 kick 은 무동작이고
        # "밀어줘도 안 된다"는 결론을 낼 수 없다.
        "kick_moved_rad_median": float(np.median([r.kick_moved_rad for r in kicked])) if kicked else None,
        "kick_noop_frac": (
            sum(1 for r in kicked if r.kick_moved_rad < 0.01) / len(kicked)) if kicked else None,
        "kick_ik_fail_frac": (
            sum(1 for r in kicked if r.kick_ik_failures > 0) / len(kicked)) if kicked else None,
        "lift_mm_median": float(np.median([r.lift_height_mm for r in results])),
        "frozen_seeds": sorted(r.seed for r in froze),
    }


def paired_on_frozen(none: list[FreezeResult], kick: list[FreezeResult]) -> dict[str, Any]:
    """Success among only the episodes that froze -- the sharp comparison.
    얼어붙은 편만 놓고 본 성공률. 이것이 예리한 비교다.

    The overall rate cannot separate: if only 15% of episodes freeze, even a
    perfect fix moves the total by 15 points, which is inside the interval at any
    n we can afford. The episodes that froze are the same seeds in both runs --
    the rollout is deterministic up to the kick -- so the two conditions are
    paired on exactly the population the kick can affect.
    전체 성공률로는 갈라지지 않는다. 동결 편이 15% 뿐이면 완벽한 처방도 총계를
    15%p 움직이고, 그건 우리가 감당할 수 있는 어떤 n 에서도 구간 안이다.
    얼어붙은 편은 두 실행에서 같은 시드다 — 롤아웃이 kick 지점까지 결정론적이므로 —
    그래서 kick 이 영향을 줄 수 있는 모집단에 정확히 짝지어 비교한다.
    """
    seeds = sorted({r.seed for r in none if r.froze} & {r.seed for r in kick if r.froze})
    if not seeds:
        return {"n": 0}
    a = {r.seed: r.success for r in none}
    b = {r.seed: r.success for r in kick}
    ok_a = sum(1 for x in seeds if a.get(x))
    ok_b = sum(1 for x in seeds if b.get(x))
    # Split by whether the jaws were actually on the object when it froze. The
    # first run pooled them and the pooled number said nothing: 11 frozen
    # episodes, but 7 of them had **zero** jaw contact 🟢 2026-09-07. Those did not
    # freeze after grasping -- they closed the gripper in mid-air and stopped, a
    # different failure with a different fix. A lift nudge cannot help them.
    # 얼었을 때 턱이 실제로 물체에 있었는지로 나눈다. 첫 실행은 둘을 합산했고 그
    # 합산값은 아무 말도 하지 않았다 — 동결 11편 중 **7편이 턱접촉 0** 이다 🟢
    # 2026-09-07. 그건 파지 후 동결이 아니라 공중에서 그리퍼를 닫고 멈춘 것이고,
    # 다른 실패이고 처방도 다르다. 들기 밀기가 도울 수 없는 편들이다.
    contact = {r.seed: r.jaw_contacts_at_freeze for r in none if r.froze}
    grasped = [x for x in seeds if contact.get(x, 0) > 0]
    airborne = [x for x in seeds if contact.get(x, 0) <= 0]

    def sub(group: list[int]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        ga = sum(1 for x in group if a.get(x))
        gb = sum(1 for x in group if b.get(x))
        return {"n": len(group), "none": ga, "kick": gb,
                "none_ci95": list(wilson_ci(ga, len(group))),
                "kick_ci95": list(wilson_ci(gb, len(group)))}

    return {
        "n": len(seeds),
        "none_successes": ok_a, "kick_successes": ok_b,
        "none_rate": ok_a / len(seeds), "kick_rate": ok_b / len(seeds),
        "none_ci95": list(wilson_ci(ok_a, len(seeds))),
        "kick_ci95": list(wilson_ci(ok_b, len(seeds))),
        "rescued": sorted(x for x in seeds if b.get(x) and not a.get(x)),
        "broken": sorted(x for x in seeds if a.get(x) and not b.get(x)),
        # 턱접촉 있음 = 실제로 파지 후 동결. 없음 = 공중에서 닫고 멈춤
        "grasped": sub(grasped),
        "airborne": sub(airborne),
    }


def as_records(results: list[FreezeResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
