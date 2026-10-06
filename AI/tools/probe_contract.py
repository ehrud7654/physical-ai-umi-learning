"""T0 -- does a known-correct action survive the policy's own deployment path?
T0 — 정답 행동이 정책 자신의 배포 경로를 통과하고도 그대로인가?

`rollout_baselines`' A section replays **recorded absolute actions** straight into
the controller. The learned policy's path is different: the network emits a scaled
residual, which is unscaled, added to the current state for the arm joints, kept
absolute for the gripper, range-checked, and only then applied. A consistent bug
anywhere in that chain -- a sign, a scale, a mixed gripper channel, an off-by-one
in the pairing -- produces exactly A pass / B pass / C fail, because training and
evaluation would be wrong the *same* way and only the closed loop would notice.
`rollout_baselines` 의 A 절은 **기록된 절대 행동**을 컨트롤러에 그대로 넣는다.
학습 정책의 경로는 다르다 — 신경망이 스케일된 잔차를 내고, 그것을 역스케일하고,
팔 관절은 현재 state 에 더하고, 그리퍼는 절대로 두고, 범위 검사를 거쳐 적용된다.
그 사슬 어디든 일관된 버그가 있으면(부호·스케일·혼합 그리퍼 채널·페어링 off-by-one)
정확히 A 통과·B 통과·C 실패가 나온다. 학습과 평가가 **같은 방식으로** 틀리므로
폐루프만 알아차리기 때문이다.

So this replaces the network with a policy that already knows the right answer and
sends it through the identical chain:
그래서 신경망을 **정답을 이미 아는 정책**으로 갈아끼우고 동일한 사슬로 보낸다.

    target = training_target(recorded_action_t, state_t, space)   # 학습이 쓰는 함수
    scaled = (target - target_mean) / target_std                  # 학습이 쓰는 스케일
    action = to_action(scaled, state_t, space, target_mean, target_std)  # 추론이 쓰는 함수

If those two functions are exact inverses, `action` equals `recorded_action_t` to
floating-point precision and the rollout must match direct replay **seed for seed**.
Any deviation is a defect in the pair, not in the policy.
두 함수가 정확한 역이면 `action` 은 부동소수점 정밀도로 `recorded_action_t` 와 같고,
롤아웃은 직접 재생과 **시드별로** 일치해야 한다. 어긋나면 정책이 아니라 그 쌍의 결함이다.

**This tool cannot test the pairing, and the first version claimed it could.**
2026-09-07 🟢: shift -1 and shift 0 returned *identical* deviations (1.86e-09 max,
5.82e-11 median) and identical 20/20 rollout vectors, and the verdict picked -1
only because ties break by list order. The reason is structural -- the deviation is
`|to_action(training_target(rec_j)) - rec_j|`, a round-trip error on whichever
action index was used, so it does not depend on the shift at all. And the rollout
cannot tell either: a one-tick lag in an absolute position command is absorbed by
the controller, which is the same thing measured when an initial displacement of
±0.3 rad left the open-loop expert at 87.0% unchanged.
**이 도구는 페어링을 검사할 수 없고, 첫 판은 할 수 있다고 주장했다.**
2026-09-07 🟢: shift -1 과 shift 0 이 **동일한** 편차(최대 1.86e-09, 중앙 5.82e-11)와
동일한 20/20 롤아웃 벡터를 냈고, 판정이 -1 을 고른 것은 동점이 목록 순서로 깨졌기
때문이다. 이유는 구조적이다 — 편차는 `|to_action(training_target(rec_j)) − rec_j|`,
즉 어느 인덱스를 썼든 그 행동의 왕복 오차이므로 shift 와 무관하다. 롤아웃도 못
가른다. 절대 위치 명령의 한 틱 지연은 제어기가 흡수하는데, 그건 초기 변위 ±0.3 rad
에서도 개루프 전문가가 87.0% 를 유지한 것과 같은 현상이다.

So the shift rows are printed as information, not as a gate. The pairing itself is
established by the collector's loop: `data.ctrl[...] = cmd` then `rec.capture(data)`
then `mj_step` -- `state_t` is the observation before `action_t` is applied, with no
index arithmetic anywhere. A sentinel-action test belongs to the real-robot and UMI
path (project-task), where separate streams make an off-by-one possible.
그래서 shift 행은 게이트가 아니라 정보로 찍는다. 페어링 자체는 수집 루프가
보장한다 — `data.ctrl[...] = cmd` → `rec.capture(data)` → `mj_step` 이므로
`state_t` 는 `action_t` 적용 **전**의 관측이고 인덱스 산술이 어디에도 없다.
sentinel action 검사는 스트림이 분리돼 off-by-one 이 가능한 **실물·UMI 경로**
(project-task)의 것이다.

    python tools/probe_contract.py datasets/sim_pick_v5 --ckpt checkpoints/bc/sim_pick_v5_seed0.pt --episodes 20 --log

⚠️ 이 도구는 torch 를 쓴다 (학습·추론과 **같은 함수**를 부르는 것이 목적이므로
수식을 다시 쓰지 않는다). 서버에서만 돈다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("probe_contract")

from eval.rollout import rollout  # noqa: E402
from policy.base import check_action  # noqa: E402
from policy.baselines import ReplayPolicy  # noqa: E402
from sim.base import Observation  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tools.audit_demo_lift import object_xy  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA_AT_LAUNCH = code_digest()


class OraclePolicy:
    """Recorded actions, sent through the learned policy's decode path.
    기록된 행동을 학습 정책의 디코드 경로로 보낸다."""

    name = "oracle_path"
    uses_privileged_state = False

    def __init__(self, actions: np.ndarray, space: str, mean: Any, std: Any, shift: int = 0) -> None:
        import torch

        from policy.bc import to_action, training_target

        self._torch = torch
        self._to_action = to_action
        self._training_target = training_target
        self._actions = np.asarray(actions, dtype=np.float32)
        self._space = space
        self._mean = mean
        self._std = std
        self._shift = shift
        self._i = 0
        self.max_dev = 0.0

    def reset(self, seed: int | None = None) -> None:
        self._i = 0
        self.max_dev = 0.0

    def act(self, obs: Observation) -> np.ndarray:
        torch = self._torch
        n = len(self._actions)
        j = min(max(self._i + self._shift, 0), n - 1)
        self._i += 1
        rec = torch.from_numpy(self._actions[j])
        st = torch.from_numpy(np.asarray(obs.state, dtype=np.float32))
        tgt = self._training_target(rec, st, self._space)
        if self._mean is not None and self._std is not None:
            scaled = (tgt - self._mean) / self._std
        else:
            scaled = tgt
        out = self._to_action(scaled, st, self._space, self._mean, self._std)
        arr = out.detach().cpu().numpy().astype(np.float32)
        self.max_dev = max(self.max_dev, float(np.max(np.abs(arr - self._actions[j]))))
        return check_action(arr, self.name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="target_mean/std 와 action_space 를 여기서 읽는다. "
                        "없으면 스케일 없이 표현만 검사한다")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--shifts", type=str, default="-1,0,1")
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    import torch

    cfg: dict[str, Any] = load_config()
    eps = sorted(args.data.glob("ep_*.npz"))[: args.episodes]
    if not eps:
        print(f"에피소드가 없다: {args.data}")
        return 1

    space, mean, std = "joint_delta_gripper_abs", None, None
    if args.ckpt is not None:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        meta = ck.get("meta", ck) if isinstance(ck, dict) else {}
        space = str(meta.get("action_space", space))
        tm, ts = meta.get("target_mean"), meta.get("target_std")
        if tm is not None and ts is not None:
            mean = torch.as_tensor(tm, dtype=torch.float32)
            std = torch.as_tensor(ts, dtype=torch.float32)
        print(f"체크포인트에서 읽음: action_space={space} · "
              f"target scale {'있음' if mean is not None else '없음'}")
    else:
        print(f"체크포인트 없음 — action_space={space}, 스케일 없이 표현만 검사한다")

    shifts = [int(s) for s in args.shifts.split(",") if s.strip()]
    print(f"\n에피소드 {len(eps)}편 · shift {shifts}")
    print("직접 재생 대비 시드별 성공/실패 벡터와 최대 편차를 본다\n")

    out: dict[str, Any] = {}
    with MujocoPickEnv(cfg, render=False, object_jitter_m=0.05) as env:
        direct: list[bool] = []
        for i, npz in enumerate(eps):
            r = rollout(env, ReplayPolicy.from_episode(npz), seed=2_000_000 + i,
                        object_xy=object_xy(npz))
            direct.append(bool(r.success))
        out["direct"] = {"successes": sum(direct), "vector": direct}
        print(f"{'직접 재생':<12} {sum(direct):>3}/{len(direct)}")

        for sh in shifts:
            got: list[bool] = []
            devs: list[float] = []
            for i, npz in enumerate(eps):
                with np.load(npz) as z:
                    acts = z["action"]
                pol = OraclePolicy(acts, space, mean, std, shift=sh)
                r = rollout(env, pol, seed=2_000_000 + i, object_xy=object_xy(npz))
                got.append(bool(r.success))
                devs.append(pol.max_dev)
            same = sum(1 for a, b in zip(direct, got) if a == b)
            out[f"shift{sh}"] = {
                "successes": sum(got), "vector": got,
                "agree_with_direct": same,
                "max_dev": float(max(devs)), "median_dev": float(np.median(devs)),
            }
            mark = "  ← 계약이 말하는 페어링" if sh == 0 else ""
            print(f"{'shift ' + f'{sh:+d}':<12} {sum(got):>3}/{len(got)}   "
                  f"직접재생과 일치 {same}/{len(got)}   "
                  f"최대 편차 {max(devs):.2e}   중앙 {np.median(devs):.2e}{mark}")

    z = out["shift0"]
    exact = z["max_dev"] < 1e-5
    agree = z["agree_with_direct"] == len(direct)
    devs = {s: out[f"shift{s}"]["max_dev"] for s in shifts}
    print("\n판정 (결과 보기 전 확정):")
    print(f"  [round_trip] shift 0 최대 편차 {z['max_dev']:.2e} < 1e-5 → "
          f"{'통과' if exact else '**실패 — training_target 과 to_action 이 역이 아니다**'}")
    print(f"  [rollout_match] 시드별 성공/실패 벡터 일치 {z['agree_with_direct']}/{len(direct)} → "
          f"{'통과' if agree else '**실패 — 같은 행동인데 결과가 다르다**'}")
    ok = exact and agree
    print(f"\n→ {'배포 경로 통과. 다음 실험으로 진행한다' if ok else '**전면 중지. 배포 경로부터 고친다**'}")
    if not ok:
        print("  이 상태의 롤아웃 수치는 정책에 대한 진술이 아니다.")
    print("\n[페어링] **이 도구로는 검사할 수 없다.** 편차 지표가 shift 와 무관하고"
          f"(측정 {devs}), 절대 위치 명령의 한 틱 지연은 제어기가 흡수한다.")
    print("  페어링은 수집 루프가 보장한다 — ctrl 설정 → capture → mj_step 이므로 "
          "state_t 는 action_t 적용 전의 관측이고 인덱스 산술이 없다 (코드 확인 🟡).")
    print("  sentinel action 검사는 스트림이 분리된 실물·UMI 경로(project-task)의 것이다.")

    if args.log:
        rec = log_run(
            experiment="contract_path", author=args.author, issue="project-task",
            conditions={"dataset": str(args.data), "n_episodes": len(eps),
                        "ckpt": str(args.ckpt) if args.ckpt else None,
                        "action_space": space, "shifts": shifts,
                        "target_scale": mean is not None,
                        "config_sha": file_digest(DEFAULT_CONFIG),
                        "code_sha_at_launch": CODE_SHA_AT_LAUNCH},
            result={"summary": out, "round_trip_passed": exact,
                    "rollout_match_passed": agree,
                    "pairing_testable": False,
                    "pairing_note": "편차 지표가 shift 와 무관하다. 이 도구로는 검사 불가",
                    "t0_passed": ok},
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
