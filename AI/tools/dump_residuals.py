"""Store what the policy outputs at every recorded observation of a dataset.
데이터셋의 기록된 모든 관측에서 정책이 출력하는 것을 저장한다.

No simulator, no rollout, no GPU -- one forward pass per recorded tick. The output
feeds `tools/probe_abc.py`, which replays the stored sequence open-loop (condition
B'), and it is also the residual pool any later bootstrap would need, so running
this once avoids re-collecting.
시뮬레이터도 롤아웃도 GPU 도 없다. 기록된 틱마다 전방추론 한 번이다. 출력은
`tools/probe_abc.py` 로 들어가 저장된 시퀀스를 개루프로 재생하고(조건 B'), 이후
어떤 부트스트랩이든 필요할 잔차 풀이기도 하다. 한 번 돌려두면 재수집이 없다.

⚠️ `--device` must match the device the rollouts will use. Device changes results:
   cpu 11/100 vs cuda 12/100 on the same checkpoint, measured 2026-09-07 🟢 (L58).
   Dumping on one device and rolling out on the other would compare two policies.
⚠️ `--device` 는 롤아웃이 쓸 장치와 같아야 한다. 장치가 결과를 바꾼다 —
   같은 체크포인트에서 cpu 11/100 vs cuda 12/100, 2026-09-07 실측 🟢 (L58).
   한 장치로 덤프하고 다른 장치로 롤아웃하면 서로 다른 정책 둘을 비교하게 된다.

    # [서버]
    python tools/dump_residuals.py datasets/sim_pick_v5 \
        checkpoints/bc/sim_pick_v5_seed0.pt --device cpu --log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("dump_residuals")

from eval.residual_dump import (  # noqa: E402
    DUMP_VERSION, dump_episode, phase_bounds, residual_summary, write_dump,
)
from policy.bc import BCPolicy  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA_AT_LAUNCH = code_digest()
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "out" / "residual_dump"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("ckpt", type=Path)
    p.add_argument("--device", type=str, default="cpu",
                   help="롤아웃이 쓸 장치와 같아야 한다 (L58)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    runtime_limits.torch_threads()

    cfg: dict[str, Any] = load_config()
    eps = sorted(args.data.glob("ep_*.npz"))
    if not eps:
        print(f"에피소드가 없다: {args.data}")
        return 1

    out_dir = args.out or (DEFAULT_OUT / f"{args.ckpt.stem}__{args.data.name}")
    policy = BCPolicy(args.ckpt, device=args.device)
    ckpt_sha = file_digest(args.ckpt)

    print(f"{args.data.name}: {len(eps)}편 · ckpt {args.ckpt.name} ({ckpt_sha})")
    print(f"장치 {args.device} · 행동공간 {policy.action_space}")
    print(f"저장 {out_dir}\n")

    dumps = []
    spans_by_episode: dict[str, list[tuple[str, int, int]]] = {}
    no_phase = 0
    for i, npz in enumerate(eps):
        d = dump_episode(policy, npz)
        spans = phase_bounds(cfg, d.n_steps)
        if not spans:
            no_phase += 1
        spans_by_episode[d.episode] = spans
        write_dump(d, out_dir)
        dumps.append(d)
        if (i + 1) % 20 == 0 or i + 1 == len(eps):
            print(f"  {i + 1}/{len(eps)}")

    if no_phase:
        print(f"\n⚠️ {no_phase}편은 config 타이밍의 합이 에피소드 길이와 맞지 않아 "
              "위상을 붙이지 못했다. 그 편의 위상은 'unknown' 이다 — "
              "틀린 라벨을 붙이는 것보다 낫다")

    summary = residual_summary(dumps, spans_by_episode)
    names = [j["name"] for j in sorted(cfg["joints"], key=lambda j: j["index"])]

    print(f"\n잔차 (계약 단위, n={summary['n_steps']} 틱)\n")
    print(f"{'관절':<16}{'signed':>10}{'|절대|':>10}{'편향분율':>10}"
          f"{'|추종오차|':>12}")
    for k, name in enumerate(names):
        print(f"{name:<16}{summary['residual_signed_mean'][k]:>10.5f}"
              f"{summary['residual_abs_mean'][k]:>10.5f}"
              f"{summary['bias_fraction'][k]:>10.3f}"
              f"{summary['tracking_delta_abs_mean'][k]:>12.5f}")
    print("\n편향분율 = |signed 평균| / |절대 평균|. "
          "1 에 가까우면 상수 편향, 0 에 가까우면 노이즈다.")
    print("⚠️ 추종오차(`action - state`) 는 비교용 규모일 뿐 잔차의 배율 기준이 "
          "아니다 — 그건 명령과 측정의 차이지 예측 오차가 아니다")
    print(f"\n±1 경계에 앉은 성분 {summary['at_bound_frac'] * 100:.3f}% "
          "(클립 횟수의 하한이다. 정확히 1.0 예측과 5.0 예측을 구분할 수 없다)")

    if summary["phase_n"]:
        print(f"\n위상별 |절대| 평균")
        print(f"{'위상':<12}{'n':>8}  " + "".join(f"{n[:9]:>10}" for n in names))
        for ph, vals in summary["phase_abs_mean"].items():
            print(f"{ph:<12}{summary['phase_n'][ph]:>8}  "
                  + "".join(f"{v:>10.5f}" for v in vals))

    print("\n다음: python tools/probe_abc.py "
          f"{args.data} {out_dir} {args.ckpt} --device {args.device}")
    print("⚠️ 이 덤프만으로는 아무 결론도 나지 않는다. 잔차 크기는 궤적이 그 크기에 "
          "견디는지 말해주지 않는다 — 그건 롤아웃이 답한다")

    if args.log:
        rec = log_run(
            experiment="residual_dump",
            author=args.author,
            issue="project-task",
            conditions={
                "dataset": str(args.data),
                "n_episodes": len(eps),
                "ckpt": args.ckpt.name,
                "ckpt_sha": ckpt_sha,
                "device": args.device,
                "action_space": policy.action_space,
                "dump_version": DUMP_VERSION,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "code_sha_at_launch": CODE_SHA_AT_LAUNCH,
                "episodes_without_phase": no_phase,
            },
            result=summary,
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
