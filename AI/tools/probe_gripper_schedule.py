from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 이 저장소의 tools/ 관례다 (repeat_runs·findings·check_determinism 전부 한다).
# 이 파일만 빠져 있어서 `PYTHONPATH` 없이 부르면 ModuleNotFoundError 가 났다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402

_claim_name = "probe_gripper_schedule"
if "--claim-name" in sys.argv:
    _claim_index = sys.argv.index("--claim-name")
    if _claim_index + 1 >= len(sys.argv):
        raise ValueError("--claim-name requires a value")
    _claim_name = sys.argv[_claim_index + 1]
runtime_limits.claim(_claim_name)
runtime_limits.torch_threads()

import numpy as np

from policy.bc import BCPolicy
from sim.mujoco.build_scene import normalize
from sim.mujoco.env import MujocoPickEnv
from sim.mujoco.kinematics import grasp_point
from tracking.exp_log import log_run


FIXED_TICKS = (60, 67, 75, 85)
CONDITIONS = (
    "learned",
    "fixed60",
    "fixed67",
    "fixed75",
    "fixed85",
    "oracle",
)


ORACLE_TOLERANCE_OVERRIDE: float | None = None

class GripperOverride:
    def __init__(
        self,
        base: BCPolicy,
        env: MujocoPickEnv,
        condition: str,
    ) -> None:
        self.base = base
        self.env = env
        self.condition = condition
        self.tick = 0
        self.closed = False
        self.first_close_tick: int | None = None

        cfg = env.cfg
        g = cfg["grasp"]
        self.offset = np.asarray(g["pinch_offset_local"], dtype=float)
        self.grasp_z = float(g["grasp_z_offset_m"])
        default_tolerance = float(
            g.get("feedback", {}).get("grasp_tol_m", 0.005)
        )
        self.tolerance = (
            float(ORACLE_TOLERANCE_OVERRIDE)
            if ORACLE_TOLERANCE_OVERRIDE is not None
            else default_tolerance
        )

        middle = np.asarray(
            [
                (float(j["range_rad"][0]) + float(j["range_rad"][1])) / 2
                for j in cfg["joints"]
            ],
            dtype=float,
        )
        opened = middle.copy()
        closed = middle.copy()
        opened[env.gripper_index] = float(g["open_cmd"])
        closed[env.gripper_index] = float(g["close_cmd"])
        self.open_norm = float(normalize(opened, cfg)[env.gripper_index])
        self.close_norm = float(normalize(closed, cfg)[env.gripper_index])

    def reset(self, seed: int) -> None:
        self.base.reset(seed)
        self.tick = 0
        self.closed = False
        self.first_close_tick = None

    def _oracle_ready(self) -> bool:
        env = self.env
        pinch = grasp_point(env.model, env.data, self.offset)
        object_position = env.object_position()
        target = object_position + np.asarray(
            [0.0, 0.0, self.grasp_z], dtype=float
        )
        return float(np.linalg.norm(pinch - target)) <= self.tolerance

    def act(self, obs) -> np.ndarray:
        action = self.base.act(obs).copy()

        if self.condition == "learned":
            self.tick += 1
            return action

        if self.condition.startswith("fixed"):
            close_tick = int(self.condition.removeprefix("fixed"))
            if self.tick >= close_tick:
                self.closed = True
        elif self.condition == "oracle":
            if self._oracle_ready():
                self.closed = True
        else:
            raise ValueError(self.condition)

        if self.closed and self.first_close_tick is None:
            self.first_close_tick = self.tick

        action[self.env.gripper_index] = (
            self.close_norm if self.closed else self.open_norm
        )
        self.tick += 1
        return action


def run_condition(
    checkpoint: Path,
    condition: str,
    episodes: int,
    seed_base: int,
) -> dict:
    successes = 0
    close_ticks: list[int] = []
    never_closed = 0
    max_lifts: list[float] = []

    with MujocoPickEnv(
        render=True,
        object_jitter_m=0.05,
        max_ticks=200,
    ) as env:
        base = BCPolicy(checkpoint, device="cpu")
        policy = GripperOverride(base, env, condition)

        for episode_index in range(episodes):
            seed = seed_base + episode_index
            obs = env.reset(seed=seed)
            policy.reset(seed)

            episode_success = False
            max_lift = 0.0

            for _ in range(env.max_ticks):
                obs = env.step(policy.act(obs))
                episode_success = episode_success or env.is_success()
                max_lift = max(max_lift, env.lift_height())

            successes += int(episode_success)
            max_lifts.append(max_lift)
            if condition != "learned":
                if policy.first_close_tick is None:
                    never_closed += 1
                else:
                    close_ticks.append(policy.first_close_tick)

            if (episode_index + 1) % 20 == 0:
                print(
                    f"{checkpoint.stem} {condition}: "
                    f"{episode_index + 1}/{episodes} "
                    f"success={successes}/{episode_index + 1}",
                    flush=True,
                )

    return {
        "success": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_max_lift_m": float(np.mean(max_lifts)),
        "median_close_tick": (
            float(np.median(close_ticks)) if close_ticks else None
        ),
        "never_closed": never_closed,
    }


def main() -> int:
    global ORACLE_TOLERANCE_OVERRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-ckpt", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--expected", type=int, action="append", default=[]
    )
    parser.add_argument("--no-expected", action="store_true")
    parser.add_argument(
        "--claim-name", type=str, default="probe_gripper_schedule"
    )
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--oracle-tolerance", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=3000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    ORACLE_TOLERANCE_OVERRIDE = args.oracle_tolerance

    if not args.no_expected and len(args.policy_ckpt) != len(args.expected):
        raise ValueError("checkpoint and expected counts differ")
    if args.no_expected:
        args.expected = [-1] * len(args.policy_ckpt)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    all_results: dict[str, dict[str, dict]] = {}

    # learned를 먼저 실행해 기존 harness 결과를 재현한다.
    for checkpoint, expected in zip(args.policy_ckpt, args.expected):
        key = checkpoint.stem
        all_results[key] = {}
        result = run_condition(
            checkpoint, "learned", args.episodes, args.seed_base
        )
        all_results[key]["learned"] = result
        if not args.no_expected and result["success"] != expected:
            raise RuntimeError(
                f"instrument invalid: {key} learned "
                f"{result['success']}/{args.episodes}, "
                f"expected {expected}/{args.episodes}"
            )
        print(f"{key} reproduction fixture: PASS")

    if args.oracle_only:
        for checkpoint in args.policy_ckpt:
            key = checkpoint.stem
            all_results[key]["oracle"] = run_condition(
                checkpoint,
                "oracle",
                args.episodes,
                args.seed_base,
            )

        payload = {
            "episodes_per_condition": args.episodes,
            "seed_base": args.seed_base,
            "render": True,
            "policy_device": "cpu",
            "jitter_m": 0.05,
            "oracle_tolerance_m": args.oracle_tolerance,
            "results": all_results,
        }
        result_path = args.out / "result.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\ncheckpoint learned oracle never_closed median_close_tick")
        for checkpoint in args.policy_ckpt:
            key = checkpoint.stem
            learned = all_results[key]["learned"]
            oracle = all_results[key]["oracle"]
            print(
                f"{key} "
                f"{learned['success']}/{learned['episodes']} "
                f"{oracle['success']}/{oracle['episodes']} "
                f"{oracle['never_closed']} "
                f"{oracle['median_close_tick']}"
            )

        if args.log:
            log_run(
                experiment="gripper_oracle_tolerance",
                author="AI 학습 담당자(트랙B)",
                issue="project-task",
                conditions={
                    "checkpoints": [str(p) for p in args.policy_ckpt],
                    "episodes": args.episodes,
                    "seed_base": args.seed_base,
                    "render": True,
                    "policy_device": "cpu",
                    "jitter_m": 0.05,
                    "oracle_tolerance_m": args.oracle_tolerance,
                },
                result=payload,
            )
        return 0

    for checkpoint in args.policy_ckpt:
        key = checkpoint.stem
        for condition in CONDITIONS[1:]:
            all_results[key][condition] = run_condition(
                checkpoint,
                condition,
                args.episodes,
                args.seed_base,
            )

    fixed_pass_counts: dict[str, int] = {}
    for condition in ("fixed60", "fixed67", "fixed75", "fixed85"):
        count = 0
        for checkpoint in args.policy_ckpt:
            key = checkpoint.stem
            base = all_results[key]["learned"]["success_rate"]
            candidate = all_results[key][condition]["success_rate"]
            count += int(candidate > 0.20 and candidate - base >= 0.20)
        fixed_pass_counts[condition] = count

    oracle_pass_count = 0
    for checkpoint in args.policy_ckpt:
        key = checkpoint.stem
        base = all_results[key]["learned"]["success_rate"]
        oracle = all_results[key]["oracle"]["success_rate"]
        oracle_pass_count += int(oracle > 0.20 and oracle - base >= 0.20)

    viable_fixed = [
        condition
        for condition, count in fixed_pass_counts.items()
        if count >= 2
    ]

    if viable_fixed:
        verdict = "fixed_schedule_candidate"
    elif oracle_pass_count >= 2:
        verdict = "visual_close_trigger_needed"
    else:
        verdict = "gripper_only_not_sufficient"

    payload = {
        "episodes_per_condition": args.episodes,
        "seed_base": args.seed_base,
        "render": True,
        "policy_device": "cpu",
        "jitter_m": 0.05,
        "results": all_results,
        "fixed_pass_counts": fixed_pass_counts,
        "oracle_pass_count": oracle_pass_count,
        "viable_fixed": viable_fixed,
        "verdict": verdict,
    }
    result_path = args.out / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\ncheckpoint condition success mean_max_lift_cm close_tick")
    for checkpoint in args.policy_ckpt:
        key = checkpoint.stem
        for condition in CONDITIONS:
            result = all_results[key][condition]
            print(
                f"{key} {condition:>8} "
                f"{result['success']:3d}/{result['episodes']} "
                f"{100*result['success_rate']:5.1f}% "
                f"{100*result['mean_max_lift_m']:6.2f} "
                f"{result['median_close_tick']}"
            )
    print(f"\nverdict={verdict}")

    if args.log:
        log_run(
            experiment="gripper_schedule_probe",
            author="AI 학습 담당자(트랙B)",
            issue="project-task",
            conditions={
                "checkpoints": [str(p) for p in args.policy_ckpt],
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "conditions": list(CONDITIONS),
                "render": True,
                "policy_device": "cpu",
                "jitter_m": 0.05,
            },
            result=payload,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
