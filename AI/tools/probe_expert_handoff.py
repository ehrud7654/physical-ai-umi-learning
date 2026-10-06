from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from policy.baselines import ScriptedFeedbackPolicy
from policy.bc import BCPolicy
from sim.mujoco.env import MujocoPickEnv
from tracking.exp_log import log_run


CONDITIONS = {
    "P": None,
    "H30": 30,
    "H60": 60,
    "H90": 90,
    "E": 0,
}


def select_expert(tick: int, handoff: int | None) -> bool:
    return handoff is not None and tick >= handoff


def fixture() -> None:
    assert not select_expert(0, None)
    assert not select_expert(29, 30)
    assert select_expert(30, 30)
    assert select_expert(0, 0)
    print("fixture boundary: PASS")


def run_condition(
    ckpt: Path,
    condition: str,
    handoff: int | None,
    episodes: int,
    seed_base: int,
) -> dict:
    success = 0
    lifts: list[float] = []
    ik_failure_episodes = 0
    ik_failures = 0
    phases: Counter[str] = Counter()

    with MujocoPickEnv(
        render=True,
        object_jitter_m=0.05,
        max_ticks=200,
    ) as env:
        bc = BCPolicy(ckpt, device="cpu")
        expert = ScriptedFeedbackPolicy(env)

        for i in range(episodes):
            seed = seed_base + i
            obs = env.reset(seed=seed)
            bc.reset(seed)
            expert.reset(seed)
            switched = handoff == 0

            episode_success = False
            for tick in range(env.max_ticks):
                use_expert = select_expert(tick, handoff)
                if use_expert and not switched:
                    expert.reset(seed)
                    switched = True
                action = expert.act(obs) if use_expert else bc.act(obs)
                obs = env.step(action)
                episode_success = episode_success or env.is_success()

            success += int(episode_success)
            lifts.append(env.lift_height())

            if handoff is not None:
                ik_failures += int(expert.ik_failures)
                ik_failure_episodes += int(expert.ik_failures > 0)
                phases[expert.phase] += 1

            if (i + 1) % 20 == 0:
                print(
                    f"{condition}: {i + 1}/{episodes} "
                    f"success={success}/{i + 1}",
                    flush=True,
                )

    rate = success / episodes
    return {
        "condition": condition,
        "handoff_tick": handoff,
        "success": success,
        "episodes": episodes,
        "success_rate": rate,
        "mean_lift_m": sum(lifts) / len(lifts),
        "ik_failures": ik_failures,
        "ik_failure_episodes": ik_failure_episodes,
        "ik_failure_episode_rate": ik_failure_episodes / episodes,
        "final_phases": dict(phases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ckpt", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=3000)
    parser.add_argument("--expected-policy-success", type=int, default=11)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    fixture()
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    # 계측기가 공식 rollout 결과를 재현하기 전에는 개입값을 읽지 않는다.
    results["P"] = run_condition(
        args.policy_ckpt, "P", None, args.episodes, args.seed_base
    )
    observed = results["P"]["success"]
    if observed != args.expected_policy_success:
        raise RuntimeError(
            f"instrument invalid: P={observed}/{args.episodes}, "
            f"expected={args.expected_policy_success}/{args.episodes}"
        )
    print("fixture official-policy reproduction: PASS")

    for name in ("H30", "H60", "H90", "E"):
        results[name] = run_condition(
            args.policy_ckpt,
            name,
            CONDITIONS[name],
            args.episodes,
            args.seed_base,
        )

    p_rate = results["P"]["success_rate"]
    recovery_candidates = []
    for name in ("H60", "H90"):
        r = results[name]
        recovery_candidates.append(
            r["success_rate"] - p_rate >= 0.20
            and r["success_rate"] >= 0.50
        )

    expert_gate = results["E"]["success_rate"] >= 0.80
    ik_gate = all(
        results[name]["ik_failure_episode_rate"] < 0.05
        for name in ("H30", "H60", "H90", "E")
    )
    dagger_gate = any(recovery_candidates) and expert_gate and ik_gate

    payload = {
        "checkpoint": str(args.policy_ckpt),
        "episodes_per_condition": args.episodes,
        "seed_base": args.seed_base,
        "render": True,
        "policy_device": "cpu",
        "jitter_m": 0.05,
        "max_ticks": 200,
        "results": results,
        "gates": {
            "policy_reproduction": observed == args.expected_policy_success,
            "expert_ceiling_ge_80pct": expert_gate,
            "ik_failure_episodes_lt_5pct": ik_gate,
            "handoff_gain_ge_20pp_and_absolute_ge_50pct": any(recovery_candidates),
            "dagger_go": dagger_gate,
        },
    }

    result_path = args.out / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\ncondition success rate gain_vs_P ik_failure_episodes")
    for name in ("P", "H30", "H60", "H90", "E"):
        r = results[name]
        gain = 100 * (r["success_rate"] - p_rate)
        print(
            f"{name:>4} {r['success']:3d}/{r['episodes']} "
            f"{100*r['success_rate']:5.1f}% "
            f"{gain:+6.1f}%p "
            f"{r['ik_failure_episodes']:3d}/{r['episodes']}"
        )

    print("\ngates:", json.dumps(payload["gates"], ensure_ascii=False))

    if args.log:
        log_run(
            experiment="expert_handoff",
            author="AI 학습 담당자(트랙B)",
            issue="project-task",
            conditions={
                "checkpoint": str(args.policy_ckpt),
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "render": True,
                "policy_device": "cpu",
                "jitter_m": 0.05,
                "max_ticks": 200,
                "handoff_ticks": [30, 60, 90],
            },
            result=payload,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
