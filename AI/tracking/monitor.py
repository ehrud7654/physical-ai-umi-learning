"""Read-only views over EXP_LOG for notebook monitoring.
노트북 관제용 EXP_LOG 읽기 전용 뷰.

Nothing here writes. The experiment record is the source of truth; this module
only reshapes it so a human can compare runs at a glance. Long jobs are launched
with `launch()`, which detaches them from the kernel -- a Jupyter kernel dies
when the browser drops, and a 30-minute training run must not die with it.
여기서는 아무것도 쓰지 않는다. 정본은 실험 기록이고, 이 모듈은 사람이 한눈에
비교할 수 있게 모양만 바꾼다. 장시간 잡은 `launch()` 로 띄워 커널에서 떼어낸다 --
브라우저가 끊기면 Jupyter 커널이 죽는데, 30분짜리 학습이 같이 죽으면 안 된다.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any

AI_ROOT = Path(__file__).resolve().parents[1]
EXP_LOG = AI_ROOT / "EXP_LOG.jsonl"


def records(experiment: str | None = None, log: Path | None = None) -> list[dict[str, Any]]:
    """Every EXP_LOG line, optionally filtered by experiment name.
    EXP_LOG 전체 줄. 실험 이름으로 걸러낼 수 있다.

    `log` defaults to None, not to EXP_LOG. A default argument binds at
    definition time, so `log: Path = EXP_LOG` would freeze the path and make
    the module untestable -- reassigning `monitor.EXP_LOG` would have no
    effect. Found by a fixture test on 2026-09-07 that silently returned zero
    rows.
    `log` 기본값은 EXP_LOG 가 아니라 None 이다. 기본 인자는 정의 시점에
    묶이므로 `log: Path = EXP_LOG` 로 두면 경로가 고정돼 모듈을 테스트할 수
    없다 -- `monitor.EXP_LOG` 를 갈아끼워도 안 먹는다. 2026-09-07 픽스처
    테스트가 조용히 0줄을 반환해 발견했다."""
    log = EXP_LOG if log is None else log
    if not log.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if experiment is None or rec.get("experiment") == experiment:
            out.append(rec)
    return out


def runs_table() -> Any:
    """One row per repeat_runs execution: rate, spread, CI, val_loss.
    repeat_runs 실행 한 건당 한 줄. 성공률·편차·신뢰구간·val_loss.

    Read the CI column, not the mean. Two runs whose intervals overlap are not
    distinguishable at this n -- that is the whole point of printing it.
    평균이 아니라 CI 열을 봐라. 구간이 겹치는 두 실행은 이 n 으로는 구분되지
    않는다. CI 를 찍는 이유가 그것이다."""
    rows = []
    for rec in records("repeat_runs"):
        cond, res = rec.get("conditions", {}), rec.get("result", {})
        per = res.get("per_run", [])
        lo, hi = (res.get("ci95") or [None, None])[:2]
        rows.append({
            "ts": rec.get("ts", "")[:16],
            "data": str(cond.get("data", "")).split("/")[-1],
            "epochs": cond.get("epochs"),
            "n": res.get("pooled_n"),
            "mean%": None if res.get("mean") is None else round(res["mean"] * 100, 1),
            "per_run%": [round(p.get("rate", 0) * 100, 1) for p in per],
            "ci95%": "" if lo is None else f"{lo * 100:.1f}~{hi * 100:.1f}",
            "val_loss": [round(p.get("val_loss", float("nan")), 4) for p in per],
            "passed": res.get("passed"),
            "git": rec.get("git_rev", "")[:7],
        })
    return _frame(rows)


def _finite(values: list[Any]) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and abs(f) != float("inf"):   # NaN·inf 제외
            out.append(f)
    return out


def failure_shape(limit: int = 6) -> Any:
    """Per-policy failure shape from the newest rollout records.
    최근 롤아웃 기록에서 정책별 실패 모양.

    Success rate alone does not say why a policy fails. The closing distance
    does: measured replay tolerance is +-10mm (3/4), so a policy that closes
    further out than that cannot succeed no matter how the loss looks.
    성공률만으로는 왜 실패하는지 모른다. 닫는 순간 거리가 말해준다 -- 실측
    재생 허용오차가 ±10mm(3/4) 이므로, 그보다 멀리서 닫는 정책은 손실이
    어떻든 성공할 수 없다."""
    rows = []
    for rec in records("rollout_baselines")[-limit:]:
        cond, res = rec.get("conditions", {}), rec.get("result", {})
        tag = str(cond.get("policy_ckpt") or cond.get("data") or "").split("/")[-1]
        for policy, episodes in res.items():
            if not isinstance(episodes, list) or not episodes:
                continue
            n = len(episodes)
            ok = sum(1 for e in episodes if e.get("success"))
            contact = sum(1 for e in episodes if (e.get("first_contact_tick", -1) or -1) >= 0)
            close_xy = _finite([e.get("xy_at_close_mm") for e in episodes])
            near_xy = _finite([e.get("min_pinch_xy_mm") for e in episodes])
            near_3d = _finite([e.get("min_pinch_3d_mm") for e in episodes])
            ticks = _finite([e.get("close_tick") for e in episodes if (e.get("close_tick", -1) or -1) >= 0])
            rows.append({
                "ts": rec.get("ts", "")[:16],
                "ckpt": tag,
                "policy": policy,
                "n": n,
                "성공%": round(ok / n * 100, 1),
                "접촉%": round(contact / n * 100, 1),
                "닫는거리mm": None if not close_xy else round(median(close_xy), 1),
                "최근접xy_mm": None if not near_xy else round(median(near_xy), 1),
                "최근접3d_mm": None if not near_3d else round(median(near_3d), 1),
                "닫는틱": None if not ticks else round(median(ticks)),
            })
    return _frame(rows)


def _frame(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def launch(cmd: str, logname: str, cwd: Path = AI_ROOT) -> int:
    """Start a long job detached from the kernel, output to AI/out/<logname>.
    커널에서 떼어낸 장시간 잡을 띄운다. 출력은 AI/out/<logname>.

    PYTHONUNBUFFERED is set rather than passing -u, because the job spawns
    children (train_bc, eval_rollout) and only the environment variable
    reaches them. Measured 2026-09-07: with -u alone the log stayed empty
    for 18 minutes and looked hung.
    -u 대신 PYTHONUNBUFFERED 를 준다. 잡이 자식(train_bc, eval_rollout)을
    띄우는데 환경변수만 거기까지 전달되기 때문이다. 2026-09-07 실측: -u 만
    주면 로그가 18분간 비어 있어 멈춘 것처럼 보였다."""
    out_dir = cwd / "out"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / logname
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl", "PYTHONPATH": "."}
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            ["nohup", *cmd.split()], cwd=cwd, env=env,
            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"PID {proc.pid}  ->  out/{logname}")
    return proc.pid


def progress(logname: str = "repeat_v5.log", tail: int = 15, pattern: str = "sim_pick_v5*.pt") -> None:
    """What is running, on which GPU, and what the log says. Re-run freely.
    무엇이 어느 GPU 에서 도는지와 로그 꼬리. 반복 실행해도 된다."""
    ps = subprocess.run(
        "ps -ef | grep -E 'train_bc|repeat_runs|eval_rollout|collect_sim' | grep -v grep",
        shell=True, capture_output=True, text=True).stdout.strip()
    print("[프로세스]\n" + (ps or "  실행 중인 잡 없음"))

    gpu = subprocess.run(
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",
        shell=True, capture_output=True, text=True).stdout.strip()
    print("\n[GPU 점유]\n" + (gpu or "  없음"))

    ckpts = sorted((AI_ROOT / "checkpoints" / "bc").glob(pattern),
                   key=lambda p: p.stat().st_mtime)
    print("\n[체크포인트]")
    for p in ckpts:
        print(f"  {time.strftime('%H:%M', time.localtime(p.stat().st_mtime))}  {p.name}")
    if not ckpts:
        print("  아직 없음")

    log_path = AI_ROOT / "out" / logname
    print(f"\n[out/{logname} 마지막 {tail}줄]")
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines()[-tail:]:
            print("  " + line)
    else:
        print("  로그 없음")


# --- 조건 표류 감지 -----------------------------------------------------------
#
# A silent config change is the most expensive kind of mistake here: the run
# completes, the number looks plausible, and only much later does someone notice
# two variables moved at once. Measured 2026-09-07: v5 was launched without
# --epochs and silently trained 20 epochs while v2/v3/v4 had all used 30. The
# result was unusable and nothing warned.
# 조용한 설정 변경이 여기서 가장 비싼 실수다. 실행은 완주하고 숫자도 그럴듯해서,
# 두 변수가 함께 움직였다는 걸 한참 뒤에야 안다. 2026-09-07 실측: v5 를 --epochs
# 없이 띄워 20 epoch 으로 학습했는데 v2·v3·v4 는 전부 30 이었다. 결과는 못 쓰게
# 됐고 아무것도 경고하지 않았다.

VOLATILE_KEYS: tuple[str, ...] = ("data", "train_seeds")
"""Conditions expected to differ between runs. Everything else should not.
실행마다 달라지는 게 정상인 조건. 나머지는 달라지면 안 된다."""


def _show(value: Any) -> str:
    if value is None:
        return "미지정(config 기본값)"
    if value == "(없음)":
        return "(없음)"
    return str(value)


def condition_drift(
    current: dict[str, Any],
    experiment: str = "repeat_runs",
    log: Path | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The previous run of the same experiment, and what differs from it.
    같은 실험의 직전 실행과, 거기서 달라진 것.

    `data` and `train_seeds` are excluded -- those are supposed to change.
    `data` 와 `train_seeds` 는 제외한다. 달라지는 게 정상이다."""
    previous = None
    for rec in records(experiment, log):
        previous = rec
    if previous is None:
        return None, []

    prev_cond = previous.get("conditions") or {}
    diffs: list[dict[str, Any]] = []
    for key in sorted(set(prev_cond) | set(current)):
        if key in VOLATILE_KEYS:
            continue
        before, after = prev_cond.get(key, "(없음)"), current.get(key, "(없음)")
        if before != after:
            diffs.append({"key": key, "prev": before, "now": after})
    return previous, diffs


def print_drift(
    current: dict[str, Any],
    experiment: str = "repeat_runs",
    log: Path | None = None,
) -> list[dict[str, Any]]:
    """Print the drift block before a run starts. Never blocks; the human judges.
    실행 시작 전에 표류 블록을 찍는다. 막지 않는다 — 판단은 사람이 한다."""
    previous, diffs = condition_drift(current, experiment, log)
    if previous is None:
        print(f"조건 대조: `{experiment}` 의 직전 기록이 없다 — 대조 생략\n")
        return []

    when = str(previous.get("ts", ""))[:16]
    prev_data = str((previous.get("conditions") or {}).get("data", "")).split("/")[-1]
    head = f"직전 실행 {when} · {prev_data}"

    if not diffs:
        print(f"조건 대조: {head} 와 동일하다 (data 제외)\n")
        return []

    print(f"⚠️ 조건 대조: {head} 와 {len(diffs)}개가 다르다")
    for d in diffs:
        print(f"     {d['key']:16s} {_show(d['prev'])}  →  {_show(d['now'])}")
    print("     의도한 변경인지 확인해라. **두 개 이상 바뀌면 결과의 원인을 가를 수 없다.**\n")
    return diffs
