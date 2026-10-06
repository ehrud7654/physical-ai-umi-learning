"""Where in the workspace does a policy actually succeed?
정책은 작업공간의 **어디에서** 실제로 성공하는가?

An aggregate success rate hides the shape of the competence. 25% can mean the
policy works one time in four everywhere, or that it works almost always in a
small patch and never outside it. Those two are different products: the first is
a policy that needs more training, the second is a policy that memorised a
region and needs different data. The rollout records already carry per-trial
object positions, so the distinction costs nothing to make — it only costs
something to ignore, because acting on the wrong one wastes the four weeks left.
집계 성공률은 능력의 **모양**을 감춘다. 25% 는 어디서나 네 번에 한 번 된다는
뜻일 수도, 좁은 구역에서는 거의 항상 되고 밖에서는 전혀 안 된다는 뜻일 수도
있다. 이 둘은 다른 제품이다. 앞은 더 학습하면 되는 정책이고, 뒤는 구역을 외운
정책이라 다른 데이터가 필요하다. 롤아웃 기록에 시행별 물체 위치가 이미 들어
있으므로 이 구분에는 비용이 들지 않는다. 무시하는 쪽에만 비용이 든다 — 잘못된
쪽에 대응하면 남은 4주를 버린다.

⚠️ 성공 건수가 적으면 지도는 지도가 아니라 점 몇 개다. 이 도구는 성공 건수를
   항상 함께 출력하고, 10건 미만이면 경고한다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from paths import AI_ROOT
from sim.mujoco.build_scene import load_config

MIN_SUCCESSES_FOR_SHAPE = 10


@dataclass
class Trial:
    """One rollout, reduced to what the map needs.
    롤아웃 하나를 지도에 필요한 것만 남긴 것."""

    x: float
    y: float
    success: bool


@dataclass
class Run:
    """One EXP_LOG record: trials that share code, conditions and timestamp.
    EXP_LOG 레코드 하나. 코드·조건·시각을 공유하는 시행 묶음."""

    idx: int
    experiment: str
    ts: str
    git_rev: str
    git_dirty: bool
    code_sha: str
    condition: str          # "" | "A" | "B"
    jitter_m: float | None
    seeds: list[int] | None
    trials: list[Trial]

    def label(self) -> str:
        """One line identifying this run, conditions included.
        이 실행을 식별하는 한 줄. 조건 포함."""
        cond = f":{self.condition}" if self.condition else ""
        dirty = "*" if self.git_dirty else " "
        jit = f"±{self.jitter_m * 1000:.0f}mm" if self.jitter_m is not None else "지터?"
        seeds = f"{self.seeds[0]}~{self.seeds[-1]}" if self.seeds else "시드?"
        return (
            f"[{self.idx:2d}] {self.ts[:16]} {self.git_rev[:7]}{dirty} "
            f"{self.experiment}{cond:3s} {jit:>8s} {seeds:>12s}"
        )

    def rate(self) -> float:
        return sum(t.success for t in self.trials) / len(self.trials)


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rows_to_trials(rows: list[dict[str, Any]]) -> list[Trial]:
    out: list[Trial] = []
    for r in rows:
        xy = r.get("object_xy")
        if xy:
            out.append(Trial(float(xy[0]), float(xy[1]), bool(r["success"])))
    return out


def collect_runs(exp_log: Path, policy: str) -> list[Run]:
    """Group per-trial rows by the record they came from.
    시행별 기록을 그것이 나온 레코드 단위로 묶는다.

    Pooling trials across records is the mistake this function exists to
    prevent. Records differ in code version, jitter and seed block, and the log
    still holds runs from a build where the scripted baseline scored 0%. Mixing
    those in produces an aggregate that describes no configuration that ever
    existed.
    레코드를 가로질러 시행을 합치는 것이 이 함수가 막으려는 실수다. 레코드마다
    코드 버전·지터·시드 블록이 다르고, 로그에는 scripted 기준선이 0% 나오던
    빌드의 실행분도 남아 있다. 그걸 섞으면 실제로 존재한 적 없는 설정을 기술하는
    집계값이 나온다.
    """
    runs: list[Run] = []
    for i, rec in enumerate(_iter_records(exp_log)):
        exp = rec.get("experiment")
        result = rec.get("result") or {}
        cond_block = rec.get("conditions") or {}
        common = dict(
            idx=i,
            experiment=str(exp),
            ts=str(rec.get("ts", "")),
            git_rev=str(rec.get("git_rev", "unknown")),
            git_dirty=bool(rec.get("git_dirty", False)),
            code_sha=str(rec.get("code_sha", "")),
            jitter_m=cond_block.get("jitter_m"),
            seeds=cond_block.get("seeds"),
        )
        if exp == "rollout_baselines":
            trials = _rows_to_trials((result.get("detail") or {}).get(policy) or [])
            if trials:
                runs.append(Run(condition="", trials=trials, **common))
        elif exp == "domain_transfer":
            for cond, rows in ((result.get("detail") or {}).get(policy) or {}).items():
                trials = _rows_to_trials(rows)
                if trials:
                    runs.append(Run(condition=str(cond), trials=trials, **common))
    return runs


def latest_per_key(runs: list[Run]) -> list[Run]:
    """Keep only the newest run for each (experiment, condition, seed block).
    (실험, 조건, 시드 블록)마다 가장 최근 실행만 남긴다."""
    keep: dict[tuple[str, str, str], Run] = {}
    for r in runs:
        seeds = f"{r.seeds}" if r.seeds else "?"
        keep[(r.experiment, r.condition, seeds)] = r
    return [keep[k] for k in sorted(keep, key=lambda k: keep[k].idx)]


def render_map(trials: list[Trial], base_xy: tuple[float, float], span_m: float,
               bins: int) -> str:
    """An ASCII grid of successes over failures, in millimetres from base.
    기준 위치로부터 mm 단위로, 성공/전체를 보여주는 ASCII 격자."""
    edges = np.linspace(-span_m, span_m, bins + 1)
    grid_s = np.zeros((bins, bins), dtype=int)
    grid_n = np.zeros((bins, bins), dtype=int)
    for t in trials:
        dx, dy = t.x - base_xy[0], t.y - base_xy[1]
        ix = int(np.clip(np.searchsorted(edges, dx) - 1, 0, bins - 1))
        iy = int(np.clip(np.searchsorted(edges, dy) - 1, 0, bins - 1))
        grid_n[iy, ix] += 1
        grid_s[iy, ix] += int(t.success)

    lines = [f"세로축 = y 편차(mm), 가로축 = x 편차(mm), 칸 = 성공/시행  ·  기준 {base_xy}"]
    header = "        " + "".join(
        f"{(edges[i] + edges[i + 1]) / 2 * 1000:>8.0f}" for i in range(bins)
    )
    lines.append(header)
    for iy in range(bins - 1, -1, -1):
        centre = (edges[iy] + edges[iy + 1]) / 2 * 1000
        cells = []
        for ix in range(bins):
            n = grid_n[iy, ix]
            cells.append("       ." if n == 0 else f"{grid_s[iy, ix]:>4d}/{n:<3d}")
        lines.append(f"{centre:>7.0f} " + "".join(cells))
    return "\n".join(lines)


def summarise(trials: list[Trial], base_xy: tuple[float, float]) -> str:
    """Distance-from-base statistics for successes versus failures.
    성공과 실패의 기준 위치로부터의 거리 통계."""
    def dist(t: Trial) -> float:
        return math.hypot(t.x - base_xy[0], t.y - base_xy[1]) * 1000.0

    ok = [dist(t) for t in trials if t.success]
    bad = [dist(t) for t in trials if not t.success]
    lines = [f"{'':10s} {'건수':>6s} {'거리 평균(mm)':>14s} {'최소':>8s} {'최대':>8s}"]
    for label, vals in (("성공", ok), ("실패", bad)):
        if vals:
            lines.append(
                f"{label:10s} {len(vals):6d} {float(np.mean(vals)):14.1f} "
                f"{min(vals):8.1f} {max(vals):8.1f}"
            )
        else:
            lines.append(f"{label:10s} {0:6d} {'—':>14s} {'—':>8s} {'—':>8s}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-log", type=Path, default=AI_ROOT / "EXP_LOG.jsonl")
    parser.add_argument("--policy", type=str, default="bc")
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--span", type=float, default=0.05, help="기준 위치 ± 범위 (m)")
    parser.add_argument("--list", action="store_true", help="실행 목록만 보고 끝낸다")
    parser.add_argument("--runs", type=int, nargs="*", default=None,
                        help="합칠 실행의 [n] 번호. 생략하면 조건별 최신 실행만 쓴다")
    args = parser.parse_args()

    cfg = load_config()
    base = cfg["task"]["object"]["init_pos"][:2]
    base_xy = (float(base[0]), float(base[1]))

    runs = collect_runs(args.exp_log, args.policy)
    if not runs:
        print(f"{args.policy!r} 의 시행 기록이 없다. 롤아웃을 --log 로 돌렸는지, "
              "정책 이름이 맞는지 확인하라.")
        return 1

    print(f"정책 {args.policy}  ·  EXP_LOG 에서 찾은 실행 {len(runs)}건 "
          "(* = git dirty)\n")
    for r in runs:
        print(f"  {r.label()}   {sum(t.success for t in r.trials)}/{len(r.trials)} "
              f"= {r.rate() * 100:5.1f}%")

    if args.list:
        return 0

    if args.runs:
        chosen = [r for r in runs if r.idx in set(args.runs)]
        missing = set(args.runs) - {r.idx for r in chosen}
        if missing:
            print(f"\n번호 {sorted(missing)} 에 해당하는 실행이 없다.")
            return 1
    else:
        chosen = latest_per_key(runs)

    revs = {r.git_rev for r in chosen}
    print(f"\n사용한 실행 {len(chosen)}건: {[r.idx for r in chosen]}")
    if len(revs) > 1:
        print(f"⚠️ 코드 버전이 섞였다: {sorted(revs)}. 조건이 다른 측정을 합치는 것이므로 "
              "결과를 하나의 수치로 인용하지 마라.")

    trials = [t for r in chosen for t in r.trials]
    n_ok = sum(t.success for t in trials)
    print(f"   합계 {n_ok}/{len(trials)} = {n_ok / len(trials) * 100:.1f}%")
    print()
    print(render_map(trials, base_xy, args.span, args.bins))
    print()
    print(summarise(trials, base_xy))

    print()
    if n_ok < MIN_SUCCESSES_FOR_SHAPE:
        print(
            f"⚠️ 성공 {n_ok}건. {MIN_SUCCESSES_FOR_SHAPE}건 미만이면 위 지도는 지도가 "
            "아니라 점 몇 개다. 모양에 대한 결론을 내지 마라 — 어느 칸이 비었는지는 "
            "정책이 못 해서가 아니라 그 칸을 안 뽑아서일 수 있다."
        )
    else:
        print(
            "성공이 특정 구역에 몰려 있으면 정책이 구역을 외운 것이고, 고르게 흩어져 "
            "있으면 능력이 전반적으로 부족한 것이다. 대응이 다르다 — 앞은 데이터 분포, "
            "뒤는 모델·목표 문제다."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
