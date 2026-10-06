"""Validate every episode in a dataset directory against the data contract.
데이터셋 디렉터리의 모든 에피소드를 데이터 계약에 대해 검증한다.

This is the data-quality gate. Run it on the whole dataset before anyone trains
on it — an episode that violates the contract in a way nobody noticed becomes a
policy that fails for reasons nobody can trace.
데이터 품질 게이트다. 누가 학습을 돌리기 전에 데이터셋 전체에 대해 실행한다.
아무도 못 본 계약 위반은, 아무도 원인을 못 찾는 정책 실패가 된다.

Exit code is non-zero when any episode fails, so it can gate a pipeline.
위반이 하나라도 있으면 종료 코드가 0이 아니다. 파이프라인 게이트로 쓸 수 있다.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np


from contract.episode import read_episode, validate, write_dataset_index

from tracking.exp_log import log_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    parser.add_argument("--write-index", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.dataset.glob("*.npz"))
    if not paths:
        print(f"에피소드가 없다: {args.dataset}")
        return 1

    failures: dict[str, list[str]] = {}
    kinds: Counter[str] = Counter()
    total_bytes = 0
    total_steps = 0
    n_success = 0
    state_min, state_max = np.inf, -np.inf

    for path in paths:
        ep = read_episode(path)
        problems = validate(ep)
        total_bytes += path.stat().st_size
        total_steps += ep.meta.n_steps
        n_success += int(ep.meta.success)
        state_min = min(state_min, float(ep.state.min()))
        state_max = max(state_max, float(ep.state.max()))
        if problems:
            failures[path.name] = problems
            for p in problems:
                kinds[p.split(":")[0]] += 1
        print(
            f"{path.name}  {ep.meta.n_steps:3d}스텝  {path.stat().st_size / 1e6:5.2f}MB  "
            f"파지={'성공' if ep.meta.success else '실패'}  "
            f"{'OK' if not problems else '위반 ' + str(len(problems))}"
        )

    n = len(paths)
    print(f"\n에피소드 {n}개, 계약 위반 {len(failures)}개")
    print(f"파지 성공 {n_success}/{n}")
    print(f"state 값 범위 실측 [{state_min:.4f}, {state_max:.4f}]  (계약 [-1, 1])")
    print(f"평균 {total_steps / n:.1f}스텝, 평균 {total_bytes / n / 1e6:.2f}MB, 합계 {total_bytes / 1e6:.1f}MB")
    for name, problems in failures.items():
        print(f"  {name}: {problems}")

    if args.write_index:
        index = write_dataset_index(
            args.dataset,
            extra={"verified_by": args.author, "contract_violations": len(failures)},
        )
        print(f"인덱스: {index}")

    if args.log:
        log_run(
            experiment="verify_dataset",
            author=args.author,
            issue="project-task",
            conditions={"dataset": str(args.dataset), "n_episodes": n},
            result={
                "violations": len(failures),
                "grasp_success": n_success,
                "state_min": state_min,
                "state_max": state_max,
                "mb_per_episode": round(total_bytes / n / 1e6, 3),
                "failure_kinds": dict(kinds),
            },
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
