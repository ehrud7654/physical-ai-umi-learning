"""What does the policy actually output at the observations it was trained on?
정책은 학습한 관측에서 실제로 무엇을 출력하는가?

`eval/trace_execution.py` section B already asks the network for an action at every
recorded observation, but it accumulates **absolute** error per joint and throws the
rest away. That is enough to say "the error is this big" and not enough to do
anything with it: the sign, the joint-to-joint covariance, the phase dependence and
the temporal correlation are all gone. A Gaussian fitted to the mean absolute error
is a different intervention from the residual the policy actually makes -- i.i.d.
noise averages out, a correlated bias pushes the whole trajectory.
`eval/trace_execution.py` 의 section B 는 이미 기록된 모든 관측에서 네트워크에
행동을 물어보지만, 관절별 **절대** 오차를 누적하고 나머지를 버린다. "오차가 이만큼
크다" 까지는 되고 그걸로 무언가를 하기에는 부족하다 — 부호, 관절 간 공분산, 위상
의존성, 시간 상관이 전부 사라진다. 평균절대오차에 맞춘 가우시안은 정책이 실제로
내는 잔차와 다른 개입이다. i.i.d. 노이즈는 평균으로 지워지고, 상관된 편향은 궤적
전체를 민다.

So this module stores the sequence itself. The stored `pred` is exactly what
`rollout` would apply if the policy saw the recorded observations, which makes the
open-loop replay of that sequence (condition B') a measurement with no residual
model, no block length and no scale factor to choose.
그래서 이 모듈은 시퀀스 자체를 저장한다. 저장된 `pred` 는 정책이 기록된 관측을
봤을 때 `rollout` 이 적용할 바로 그 값이고, 그 시퀀스를 개루프로 재생하는 것(조건 B')은
잔차 모형도 블록 길이도 배율도 고를 필요가 없는 계측이 된다.

⚠️ `BCPolicy.act` returns the action **after** clipping to the contract range, which
   is what the environment receives. `at_bound` marks components sitting exactly on
   ±1, which is a lower bound on clipping, not a count of it -- a prediction of
   exactly 1.0 is indistinguishable from one of 5.0 here.
⚠️ `BCPolicy.act` 는 계약 범위로 클립한 **뒤의** 행동을 돌려주고, 환경이 받는 것도
   그것이다. `at_bound` 는 성분이 정확히 ±1 에 앉은 것을 표시하며 클립 횟수의
   하한이다 — 여기서는 1.0 예측과 5.0 예측을 구분할 수 없다.

⚠️ Round 1 dumps every episode in the dataset, training episodes included. Train
   residuals are smaller than held-out ones, so this is the conservative direction
   for the question "do the policy's own on-demo errors break the trajectory": if
   even these break it the evidence is strong, and if they do not, held-out
   residuals still might. Splitting by `val_split: episode` needs the training
   Dataset's sample index and is deferred.
⚠️ 1차는 데이터셋의 모든 에피소드를 덤프한다. 학습에 쓴 편도 포함이다. 학습 잔차는
   홀드아웃 잔차보다 작으므로, "정책 자신의 시연상 오차가 궤적을 깨뜨리나" 라는
   질문에는 **보수적인 방향**이다 — 이것으로도 깨지면 근거가 강하고, 깨지지 않아도
   홀드아웃 잔차는 깨뜨릴 수 있다. `val_split: episode` 분리는 학습 Dataset 의 샘플
   인덱스가 필요해 뒤로 미뤘다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract.episode import Episode, read_episode
from policy.bc import BCPolicy
from sim.base import Observation

# Which conditions this module can assemble. B'_all is the policy's own sequence;
# the two attribution conditions replace one half with the recording, so that a
# verdict can be tied to the arm or to the gripper.
# 이 모듈이 조립할 수 있는 조건. B'_all 은 정책 자신의 시퀀스이고, 두 귀속 조건은
# 한쪽 절반을 기록으로 되돌려 판정을 팔 또는 그리퍼에 묶는다.
CONDITIONS: tuple[str, ...] = ("A", "B_all", "B_arm", "B_grip")

DUMP_VERSION = "1"


@dataclass(frozen=True)
class EpisodeDump:
    """One episode's demonstration track and the policy's answer to it.
    에피소드 하나의 시연 트랙과 그에 대한 정책의 답."""

    episode: str
    demo: np.ndarray  # (T, 6) float32 — recorded action, contract units
    pred: np.ndarray  # (T, 6) float32 — policy action at the recorded observation
    state: np.ndarray  # (T, 6) float32 — recorded state
    at_bound: np.ndarray  # (T, 6) bool — prediction component sits on ±1
    n_clipped: int  # policy's own counter over this episode
    object_xy: tuple[float, float] | None

    @property
    def residual(self) -> np.ndarray:
        """Signed `pred - demo` in contract units. Sign, covariance and order kept.
        계약 단위의 signed `pred - demo`. 부호·공분산·순서를 보존한다."""
        return self.pred - self.demo

    @property
    def n_steps(self) -> int:
        return int(self.demo.shape[0])


def heldout_episode_ids(
    n_episodes: int, val_fraction: float, seed: int
) -> list[int]:
    """Which episode indices `split_by_episode` holds out, without loading data.
    `split_by_episode` 가 홀드아웃하는 에피소드 인덱스. 데이터를 읽지 않는다.

    Reimplements the two lines of `policy.train_bc.split_by_episode` that decide
    the episode set, because that function needs a built Dataset -- which loads
    every image -- to return the same answer. Duplication is a hazard, so the
    contract is explicit: the permutation and the count must stay identical to
    that function. `split_by_episode` returns its `val_eps` list, so the two can
    be cross-checked directly whenever a Dataset is on hand.
    `policy.train_bc.split_by_episode` 에서 에피소드 집합을 정하는 두 줄을 다시
    구현한다. 그 함수는 같은 답을 내려면 Dataset 이 필요하고, Dataset 은 이미지를
    전부 읽는다. 중복은 위험하므로 계약을 명시한다 — 순열과 개수가 그 함수와
    **동일하게 유지되어야 한다.** `split_by_episode` 는 `val_eps` 를 반환하므로,
    Dataset 이 있는 자리에서는 두 값을 직접 대조할 수 있다.

    ⚠️ Episode index is the position in `sorted(glob("*.npz"))`, and the training
       Dataset skips episodes that fail `validate`. One rejected episode shifts
       every index after it. The caller must check `n_episodes` against the
       checkpoint's recorded count before trusting this.
    ⚠️ 에피소드 인덱스는 `sorted(glob("*.npz"))` 의 순서이고, 학습 Dataset 은
       `validate` 를 통과하지 못한 편을 건너뛴다. 한 편이 탈락하면 그 뒤 인덱스가
       전부 밀린다. 호출자는 이 값을 믿기 전에 체크포인트가 기록한 편 수와
       `n_episodes` 를 대조해야 한다.
    """
    if val_fraction <= 0.0 or n_episodes <= 1:
        return []
    rng = np.random.default_rng(seed)
    eps = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_fraction)))
    return sorted(int(e) for e in eps[:n_val])


def seed_from_ckpt_name(name: str) -> int | None:
    """The training seed a checkpoint filename claims, or None.
    체크포인트 파일명이 주장하는 학습 시드. 없으면 None.

    `--seed` on the command line overrides `train.seed` in the config but is not
    written back into `train_config`, so a checkpoint's metadata reports the
    config's seed (0) no matter which seed actually trained it. The filename is
    the only surviving record, which makes this a convention and not a fact --
    the caller prints what it inferred and takes an override.
    명령행 `--seed` 는 설정의 `train.seed` 를 덮어쓰지만 `train_config` 에 다시
    기록되지 않는다. 그래서 어떤 시드로 학습했든 체크포인트 메타는 설정값(0)을
    보고한다. 파일명이 유일하게 남은 기록이고, 그래서 이것은 사실이 아니라
    **관례**다 — 호출자는 추론한 값을 출력하고 재정의를 받는다.
    """
    import re

    m = re.search(r"_seed(\d+)", name)
    return int(m.group(1)) if m else None


def phase_bounds(cfg: dict[str, Any], n_steps: int) -> list[tuple[str, int, int]]:
    """Tick ranges of the expert's phases, derived from config -- never hardcoded.
    전문가 위상의 틱 구간. config 에서 유도하고 하드코딩하지 않는다.

    Returns an empty list when the configured timings do not add up to `n_steps`.
    A wrong phase label is worse than no label: it would attribute a failure to
    the wrong part of the motion and there would be nothing to catch it.
    설정된 타이밍의 합이 `n_steps` 와 맞지 않으면 빈 목록을 돌려준다. 틀린 위상
    라벨은 라벨이 없는 것보다 나쁘다 — 실패를 동작의 엉뚱한 부분에 귀속시키고,
    그걸 잡아줄 것이 아무것도 없다.
    """
    rate = float(cfg["control"]["rate_hz"])
    timing = cfg["grasp"]["timing"]
    order = (("approach", "approach_s"), ("descend", "descend_s"),
             ("close", "close_s"), ("lift", "lift_s"))
    spans: list[tuple[str, int, int]] = []
    start = 0
    for name, key in order:
        ticks = int(round(float(timing[key]) * rate))
        spans.append((name, start, start + ticks))
        start += ticks
    dwell = int(round(float(cfg["grasp"].get("dwell_s", 0.0)) * rate))
    if start + dwell != n_steps:
        return []
    return spans


def phase_of(spans: list[tuple[str, int, int]], t: int) -> str:
    """Phase name for a tick, or 'unknown' when the spans could not be derived.
    틱의 위상 이름. 구간을 유도하지 못했으면 'unknown'."""
    for name, lo, hi in spans:
        if lo <= t < hi:
            return name
    return "unknown"


def _obs_at(ep: Episode, t: int) -> Observation:
    """Rebuild the observation the collector recorded at tick `t`.
    수집기가 틱 t 에 기록한 관측을 그대로 되만든다."""
    return Observation(
        images={cam: arr[t] for cam, arr in ep.images.items()},
        state=ep.state[t].astype(np.float32),
        timestamp=float(ep.state_timestamp[t]),
    )


def dump_episode(policy: BCPolicy, npz: Path) -> EpisodeDump:
    """Ask the policy for an action at every recorded observation of one episode.
    에피소드 하나의 기록된 모든 관측에서 정책에 행동을 물어본다."""
    ep = read_episode(npz)
    policy.reset(seed=0)
    n = int(ep.action.shape[0])
    pred = np.empty((n, 6), dtype=np.float32)
    for t in range(n):
        pred[t] = policy.act(_obs_at(ep, t))
    xy = (ep.meta.notes or {}).get("object_init_xy")
    return EpisodeDump(
        episode=npz.stem,
        demo=ep.action.astype(np.float32),
        pred=pred,
        state=ep.state.astype(np.float32),
        at_bound=(np.abs(pred) >= 1.0),
        n_clipped=int(policy.n_clipped),
        object_xy=(float(xy[0]), float(xy[1])) if xy is not None else None,
    )


def write_dump(dump: EpisodeDump, out_dir: Path) -> Path:
    """Store one episode's dump. Arrays only -- readable without this module.
    에피소드 하나의 덤프를 저장한다. 배열만 넣어 이 모듈 없이도 읽힌다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dump.episode}.npz"
    np.savez_compressed(
        path,
        demo=dump.demo,
        pred=dump.pred,
        state=dump.state,
        at_bound=dump.at_bound,
        n_clipped=np.int64(dump.n_clipped),
        object_xy=(np.array(dump.object_xy, dtype=np.float64)
                   if dump.object_xy is not None else np.array([], dtype=np.float64)),
        dump_version=np.array(int(DUMP_VERSION), dtype=np.int64),
    )
    return path


def read_dump(path: Path) -> EpisodeDump:
    """Load one episode's dump back.
    에피소드 하나의 덤프를 되읽는다."""
    with np.load(path, allow_pickle=False) as z:
        version = int(z["dump_version"]) if "dump_version" in z else 0
        if version != int(DUMP_VERSION):
            raise ValueError(
                f"{path.name} 은 dump_version {version} 이고 이 코드는 "
                f"{DUMP_VERSION} 을 기대한다. 덤프를 다시 만들어라 — "
                "포맷이 다른 덤프를 섞으면 어느 잔차가 어느 정책의 것인지 알 수 없다"
            )
        xy = z["object_xy"]
        return EpisodeDump(
            episode=path.stem,
            demo=z["demo"],
            pred=z["pred"],
            state=z["state"],
            at_bound=z["at_bound"],
            n_clipped=int(z["n_clipped"]),
            object_xy=(float(xy[0]), float(xy[1])) if xy.size == 2 else None,
        )


def assemble(condition: str, dump: EpisodeDump, gripper_idx: int) -> np.ndarray:
    """Build the action sequence one condition replays open-loop.
    조건 하나가 개루프로 재생할 행동열을 만든다.

    `A` is the recording, `B_all` the policy's own sequence, and the two
    attribution conditions swap exactly one half. Splitting on `gripper_idx`
    rather than on the literal 5 is not tidiness: the gripper model is expected
    to change, and a hardcoded index would keep running and be wrong.
    `A` 는 기록, `B_all` 은 정책 자신의 시퀀스, 두 귀속 조건은 정확히 절반만
    바꾼다. 리터럴 5 가 아니라 `gripper_idx` 로 나누는 것은 깔끔함의 문제가
    아니다 — 그리퍼 모델은 바뀔 예정이고, 박아둔 인덱스는 계속 돌면서 틀린다.
    """
    if condition == "A":
        return dump.demo.copy()
    if condition == "B_all":
        return dump.pred.copy()
    arm = [j for j in range(dump.demo.shape[1]) if j != gripper_idx]
    out = dump.demo.copy()
    if condition == "B_arm":
        out[:, arm] = dump.pred[:, arm]
        return out
    if condition == "B_grip":
        out[:, gripper_idx] = dump.pred[:, gripper_idx]
        return out
    raise ValueError(f"모르는 조건: {condition!r} (가능한 값 {CONDITIONS})")


def residual_summary(
    dumps: list[EpisodeDump], spans_by_episode: dict[str, list[tuple[str, int, int]]]
) -> dict[str, Any]:
    """Signed residual statistics, per joint and per phase.
    관절별·위상별 signed 잔차 통계.

    Reports the signed mean beside the absolute mean on purpose. A joint whose
    absolute error is large but whose signed mean is near zero is noisy; one whose
    signed mean is close to its absolute mean carries a constant bias, and a
    constant bias is the thing training cannot average away.
    signed 평균을 절대 평균과 나란히 낸다. 절대 오차가 큰데 signed 평균이 0 근처인
    관절은 노이즈이고, signed 평균이 절대 평균에 가까운 관절은 상수 편향을 지닌다.
    상수 편향은 학습이 평균으로 지울 수 없는 바로 그것이다.
    """
    per_joint_signed = np.zeros(6, dtype=np.float64)
    per_joint_abs = np.zeros(6, dtype=np.float64)
    per_joint_delta = np.zeros(6, dtype=np.float64)
    total = 0
    phase_abs: dict[str, np.ndarray] = {}
    phase_n: dict[str, int] = {}
    for d in dumps:
        r = d.residual.astype(np.float64)
        per_joint_signed += r.sum(axis=0)
        per_joint_abs += np.abs(r).sum(axis=0)
        per_joint_delta += np.abs(d.demo.astype(np.float64)
                                  - d.state.astype(np.float64)).sum(axis=0)
        total += d.n_steps
        spans = spans_by_episode.get(d.episode, [])
        for t in range(d.n_steps):
            name = phase_of(spans, t)
            if name not in phase_abs:
                phase_abs[name] = np.zeros(6, dtype=np.float64)
                phase_n[name] = 0
            phase_abs[name] += np.abs(r[t])
            phase_n[name] += 1
    if total == 0:
        return {"n_steps": 0}
    signed = per_joint_signed / total
    absolute = per_joint_abs / total
    delta = per_joint_delta / total
    return {
        "n_episodes": len(dumps),
        "n_steps": total,
        "residual_signed_mean": [round(float(v), 6) for v in signed],
        "residual_abs_mean": [round(float(v), 6) for v in absolute],
        "tracking_delta_abs_mean": [round(float(v), 6) for v in delta],
        # |signed| / |abs| near 1 means a constant bias, near 0 means noise.
        # |signed| / |abs| 이 1 에 가까우면 상수 편향, 0 에 가까우면 노이즈다.
        "bias_fraction": [
            round(float(abs(s) / a), 4) if a > 0 else 0.0
            for s, a in zip(signed, absolute)
        ],
        "at_bound_frac": round(
            float(sum(int(d.at_bound.sum()) for d in dumps) / (total * 6)), 6
        ),
        "phase_abs_mean": {
            name: [round(float(v / phase_n[name]), 6) for v in phase_abs[name]]
            for name in sorted(phase_abs)
        },
        "phase_n": dict(sorted(phase_n.items())),
    }
