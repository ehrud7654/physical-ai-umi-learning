"""Regenerate the measured-numbers section of README.md from EXP_LOG.jsonl.
README.md 의 측정 수치 절을 EXP_LOG.jsonl 로부터 다시 생성한다.

Numbers typed by hand into a README go stale silently, and a stale number read
as current is exactly the failure this project's first principle forbids —
writing an expectation as if it were a result. So the numbers are not typed:
they are rendered from the last logged run of each experiment, together with the
conditions that produced them and the git revision the code was at.
손으로 적어넣은 수치는 조용히 낡는다. 낡은 수치를 현재 수치로 읽는 것이
이 프로젝트 절대원칙 1이 금지하는 실패 — 예상을 결과로 쓰는 것 — 그 자체다.
그래서 수치는 타이핑하지 않는다. 각 실험의 마지막 로그를 조건과 git 리비전과
함께 렌더링한다.

Only the block between the two markers is replaced. The narrative around it is
written by hand and never touched.
두 마커 사이 블록만 교체된다. 그 바깥의 서술은 손으로 쓰고 건드리지 않는다.

    python tools/update_readme.py           # README.md 갱신
    python tools/update_readme.py --check   # 낡았으면 종료코드 1 (CI 용)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paths import AI_ROOT, DEFAULT_EXP_LOG

BEGIN = "<!-- MEASURED:BEGIN — tools/update_readme.py 가 생성한다. 손으로 고치지 마라 -->"
END = "<!-- MEASURED:END -->"

README = AI_ROOT / "README.md"


def load_runs(log_path: Path) -> dict[str, dict[str, Any]]:
    """Last logged run per experiment name.
    실험 이름별 마지막 실행 기록."""
    latest: dict[str, dict[str, Any]] = {}
    if not log_path.exists():
        return latest
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = rec.get("experiment")
        if name:
            latest[name] = rec  # file is append-only, so last wins
    return latest


def _stamp(rec: dict[str, Any]) -> str:
    env = rec.get("env") or {}
    rev = rec.get("git_rev", "unknown")
    bits: list[str] = []
    if rev and rev != "unknown":
        bits.append(f"git `{rev}`" + (" **dirty**" if rec.get("git_dirty") else ""))
    else:
        # Saying nothing here would read as "no git info needed". It is missing.
        # 여기서 아무 말도 안 하면 "git 정보가 필요 없다"로 읽힌다. 없는 것이다.
        bits.append("git `없음` (계측이 git 체크아웃 밖에서 돌았다)")
    if rec.get("code_sha"):
        bits.append(f"code `{rec['code_sha']}`")
    bits += [f"MuJoCo {env.get('mujoco', '?')}", f"Python {env.get('python', '?')}"]
    sha = (rec.get("conditions") or {}).get("config_sha")
    if sha:
        bits.append(f"config `{sha}`")
    if rec.get("ts"):
        bits.append(str(rec["ts"]))
    return " · ".join(bits)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_grasp(rec: dict[str, Any]) -> list[str]:
    c, r = rec["conditions"], rec["result"]
    out = [
        "### 스크립트 파지 성공률",
        "",
        f"**{r['n_success']}/{r['n_trials']} = {_pct(r['success_rate'])}**",
        "",
        f"- 조건: 물체 xy 무작위 ±{c['jitter_m'] * 1000:.0f}mm, 시드 {c['seed']}, "
        f"제어 {c['control_rate_hz']}Hz, 물체 반치수 {c['object_half_size_m']}m, "
        f"close_cmd {c['close_cmd']}",
        f"- 성공 판정: {c['success_criterion']}",
    ]
    if r.get("failure_reasons"):
        for reason, n in sorted(r["failure_reasons"].items(), key=lambda kv: -kv[1]):
            out.append(f"- 실패 {n}회: {reason}")
    if r.get("lift_ik_not_converged"):
        out.append(f"- ⚠️ 들어올림 IK 미수렴 {r['lift_ik_not_converged']}회")
    out += ["", f"<sub>{_stamp(rec)}</sub>", ""]
    return out


def render_reach(rec: dict[str, Any]) -> list[str]:
    c, r = rec["conditions"], rec["result"]
    x0, x1, y0, y1 = r["largest_rect"]
    return [
        "### 작업공간 도달성",
        "",
        f"**{r['n_reachable']}/{r['n_total']} = {_pct(r['reachable_fraction'])}**",
        "",
        f"- 최대 연속 가능 영역: x[{x0:.3f}, {x1:.3f}] y[{y0:.3f}, {y1:.3f}] "
        f"= **{(x1 - x0) * 100:.0f}cm x {(y1 - y0) * 100:.0f}cm**",
        f"- 조건: x{c['x_range']} y{c['y_range']} 격자 {c['step_m'] * 100:.1f}cm, "
        f"파지 높이 {c['grasp_height_m']:.3f}m, wrist_roll {c.get('wrist_roll', '자유')}",
        f"- 판정: {c['criterion']}",
        "- ⚠️ 기구학만 본다. 충돌은 검사하지 않는다",
        "",
        f"<sub>{_stamp(rec)}</sub>",
        "",
    ]


def render_baselines(rec: dict[str, Any]) -> list[str]:
    c, r = rec["conditions"], rec["result"]
    rates: dict[str, float] = r["success_rates"]
    priv: dict[str, bool] = r.get("uses_privileged_state", {})
    out = [
        "### baseline 성공률과 게이트",
        "",
        f"지표는 **{c.get('metric', '롤아웃 성공률')}**",
        "",
        "| 정책 | 성공률 | 실물 배포 |",
        "|---|---|---|",
    ]
    for name, rate in rates.items():
        deployable = "**불가 — 특권정보**" if priv.get(name) else "가능"
        out.append(f"| {name} | {_pct(rate)} | {deployable} |")
    floor = max(rates.get("hold", 0.0), rates.get("zero", 0.0))
    out += [
        "",
        f"- 조건: {c['episodes']} 에피소드, 시드 {c['seeds'][0]}~{c['seeds'][1]}, "
        f"물체 xy ±{c['jitter_m'] * 1000:.0f}mm, 모든 정책이 동일 시드, "
        f"렌더 {'있음' if c.get('render') else '없음'}",
        "",
        "**게이트 판정** (기준은 결과 확인 전에 `eval/rollout.py` 의 `GATES` 에 확정):",
        "",
    ]
    if "replay" in rates:
        verdict = "통과" if rates["replay"] < 0.30 else "**실패 — 태스크 재설계 필요**"
        out.append(f"- `task_validity` replay {_pct(rates['replay'])} < 30% → {verdict}")
    if "scripted" in rates:
        verdict = "통과" if rates["scripted"] >= 0.80 else "**실패 — 태스크/씬 문제**"
        out.append(f"- `ceiling` scripted {_pct(rates['scripted'])} >= 80% → {verdict}")
    out += [
        f"- `floor`/`chance` → **학습 정책은 {_pct(floor + 0.20)} 를 넘어야 의미가 있다**",
        "",
        "⚠️ scripted 는 성능이 아니라 **상한선**이다. 물체의 정답 위치를 시뮬에서 직접 읽는다.",
        "",
    ]
    tol = r.get("replay_tolerance")
    if tol:
        state = "성공" if tol.get("at_recorded_condition") else "**실패 — 계측기 고장, 아래 수치 무효**"
        out += [
            "**replay 위치 허용오차** — 태스크가 요구하는 정밀도",
            "",
            f"원본 `{tol['episode']}`, 기록 위치 "
            f"({tol['recorded_xy'][0]:.4f}, {tol['recorded_xy'][1]:.4f}) → {state}",
            "",
            "| 물체 이동량 | 성공 (4방향) |",
            "|---|---|",
        ]
        for row in tol["rows"]:
            out.append(f"| ±{row['offset_m'] * 1000:.0f}mm | {row['success']}/{row['of']} |")
        out.append("")
    out += [f"<sub>{_stamp(rec)}</sub>", ""]
    return out


def render_dataset(rec: dict[str, Any]) -> list[str]:
    c, r = rec["conditions"], rec["result"]
    out = [
        "### 데이터 계약 왕복 검증",
        "",
        f"에피소드 **{c['n_episodes']}건**, 계약 위반 **{r['violations']}건**",
        "",
        f"- 데이터셋: `{c['dataset']}`",
        f"- 파지 성공 {r.get('grasp_success', '?')}/{c['n_episodes']}",
        f"- state 값 범위 실측 [{r['state_min']:.4f}, {r['state_max']:.4f}] (계약 [-1, 1])",
        f"- 용량 평균 {r['mb_per_episode']:.2f}MB/에피소드 → "
        f"100건 약 {r['mb_per_episode'] * 100 / 1000:.2f}GB, "
        f"1000건 약 {r['mb_per_episode'] * 1000 / 1000:.1f}GB "
        f"(10진 GB. BE 파트에 전달한 수치와 같은 단위)",
    ]
    if r.get("failure_kinds"):
        out.append(f"- 위반 종류: {r['failure_kinds']}")
    out += ["", f"<sub>{_stamp(rec)}</sub>", ""]
    return out


def render_gripper(rec: dict[str, Any]) -> list[str]:
    c, r = rec["conditions"], rec["result"]
    pin_w = r["pinch_point_with_pads"]
    pin_wo = r["pinch_point_without_pads"]
    grid_w = r["grid_with_pads"]
    grid_wo = r["grid_without_pads"]
    gap = r.get("gap_curve") or []
    out = [
        "### 그리퍼 접촉 형상 (MuJoCo 볼록껍질 문제)",
        "",
        "MuJoCo 는 메시를 **볼록껍질**로 충돌시킨다. 두 갈래 그리퍼의 껍질은 턱 사이 공간을 메우므로,",
        "공식 MJCF 그대로는 완전히 벌려도 물체가 턱에 들어가지 못한다. 이것이 파지 0/24 의 원인이었다.",
        "",
        "| 설정 | 파지점 관통 | 주변 격자 (24점) |",
        "|---|---|---|",
        f"| 공식 MJCF 그대로 | {pin_wo['penetration_m'] * 100:.2f}cm | "
        f"{grid_wo['blocked_points']}/{grid_wo['probe_points']}점 막힘, "
        f"최대 {grid_wo['max_penetration_m'] * 100:.2f}cm |",
        f"| 접촉패드 주입 (현재) | {pin_w['penetration_m'] * 100:.2f}cm | "
        f"{grid_w['blocked_points']}/{grid_w['probe_points']}점 막힘, "
        f"최대 {grid_w['max_penetration_m'] * 100:.2f}cm |",
        "",
        f"- 조건: 물체 반치수 {c['object_half_size_m']} m, open_cmd {c['open_cmd']}",
        f"- 격자: {c['probe_grid']}",
    ]
    if gap:
        out.append(
            f"- 턱 간격: 닫힘 {gap[0][1] * 100:.2f}cm ({gap[0][0]:.3f} rad) → "
            f"완전개방 {gap[-1][1] * 100:.2f}cm ({gap[-1][0]:.3f} rad), {len(gap)}점 실측"
        )
    out += [
        "- 패드는 `configs/so101.yaml` 의 `gripper_pads` 로 **로드 시점에 주입**한다. "
        "공식 MJCF 는 수정하지 않는다",
        "- ⚠️ 관통이 0 이 아니어도 파지는 성공한다. 이 수치는 껍질 형상 진단용이고, "
        "실제 판정은 파지 성공률이다",
        "",
        f"<sub>{_stamp(rec)}</sub>",
        "",
    ]
    return out


def render_train_bc(rec: dict[str, Any]) -> list[str]:
    c, r = rec["conditions"], rec["result"]
    out = [
        "### BC 학습 (project-task)",
        "",
        f"**성공률: 미측정.** 여기서 재지 않는다 — `tools/eval_rollout.py --policy-ckpt` 가 잰다.",
        "",
        f"- 학습 대상: `{c['trained_on']}`"
        + ("  ⚠️ **랜덤 텐서다. 루프 검증용이고 수치에 의미가 없다**"
           if c["trained_on"] == "random_tensors" else ""),
        f"- 에피소드 {c['n_episodes']} · 샘플 {c['n_samples']} · epochs {c['epochs']} · "
        f"batch {c['batch_size']} · lr {c['lr']} · loss {c['loss']} · seed {c['seed']}",
        f"- 파라미터 **{c['n_params']:,}** (~{c['n_params'] * 4 / 1024 / 1024:.1f}MB fp32) · "
        f"인코더 {c['encoder_mode']} · device {c['device']}",
        f"- 소요 {r.get('seconds', '?')}초 · 체크포인트 `{r.get('checkpoint', '?')}`",
    ]
    if r.get("best_val_loss") is not None:
        out.append(
            f"- best val_loss {r['best_val_loss']:.5f} — "
            "**학습이 망가지지 않았다는 확인일 뿐이다. 성과가 아니다**"
        )
    out += ["", f"<sub>{_stamp(rec)}</sub>", ""]
    return out


RENDERERS = {
    "train_bc": render_train_bc,
    "grasp_check": render_grasp,
    "reach_scan": render_reach,
    "rollout_baselines": render_baselines,
    "verify_dataset": render_dataset,
    "gripper_probe": render_gripper,
}

ORDER = ["grasp_check", "reach_scan", "rollout_baselines", "train_bc",
         "verify_dataset", "gripper_probe"]


def freshness(log: Path, runs: dict[str, dict[str, Any]]) -> list[str]:
    """State how fresh the log this block was rendered from actually is.
    이 블록을 렌더링한 로그가 실제로 얼마나 최신인지 밝힌다.

    `--check` compares the README against EXP_LOG and passes when they agree.
    That says nothing about whether EXP_LOG itself is current: on 2026-09-03 the
    check passed while the local log was 6 records behind the server's, because
    both sides were stale together. 🟢 The guard was guarding the wrong seam.
    So the block prints the newest record's revision and the current HEAD, and
    `--check` fails when they differ -- experiments have been run on code that
    this README has never seen.
    `--check` 는 README 와 EXP_LOG 를 대조해 일치하면 통과한다. 그건 EXP_LOG 자체가
    최신인지에 대해 아무 말도 하지 않는다. 2026-09-03 실측 🟢: 로컬 로그가 서버보다
    6건 뒤처진 상태에서 검사가 통과했다 — 양쪽이 같이 낡았기 때문이다.
    가드가 엉뚱한 이음매를 지키고 있었다. 그래서 최신 기록의 리비전과 현재 HEAD 를
    함께 찍고, 다르면 `--check` 를 실패시킨다. 이 README 가 본 적 없는 코드에서
    실험이 돌았다는 뜻이므로.
    """
    n = sum(1 for _ in log.open(encoding="utf-8")) if log.exists() else 0
    revs = {r.get("git_rev", "?") for r in runs.values()}
    head = _head_rev()
    stale = head is not None and head not in revs and revs != {"?"}
    lines = [
        f"로그 기록 수 **{n}** · 이 블록의 리비전 {sorted(revs)} · 현재 HEAD `{head or '?'}`",
        "",
    ]
    if stale:
        lines += [
            "> ⚠️ **이 수치들은 현재 코드에서 나온 것이 아니다.** 최신 기록의 리비전이",
            "> HEAD 와 다르다. 서버에서 돈 실험이 `EXP_LOG.jsonl` 로 커밋되지 않았거나,",
            "> 코드가 바뀐 뒤 재측정하지 않았다.",
            "",
        ]
    return lines


def _head_rev() -> str | None:
    """Current git revision, short. None if git is unavailable.
    현재 git 리비전(짧게). git 을 못 쓰면 None."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def build_block(runs: dict[str, dict[str, Any]], log: Path = DEFAULT_EXP_LOG) -> str:
    """Render the whole generated section.
    생성 구역 전체를 렌더링한다."""
    lines = [
        BEGIN,
        "",
        "모든 수치는 **시뮬**이며, 각 실험의 **가장 최근 로그**에서 자동 생성된다.",
        "직접 고치지 마라 — `python tools/update_readme.py` 로 다시 만든다.",
        "",
    ] + freshness(log, runs)
    missing = [n for n in ORDER if n not in runs]
    for name in ORDER:
        rec = runs.get(name)
        if rec is None:
            continue
        lines += RENDERERS[name](rec)
    if missing:
        lines += [
            "### 로그가 없는 실험",
            "",
            "아래는 EXP_LOG.jsonl 에 기록이 없다. **수치 없음 = 미측정**이다.",
            "",
        ] + [f"- `{n}`" for n in missing] + [""]
    extra = sorted(set(runs) - set(ORDER))
    if extra:
        lines += [
            "### 렌더러가 없는 실험 기록",
            "",
            "EXP_LOG 에는 있으나 이 스크립트가 표로 만드는 법을 모른다. "
            "`tools/update_readme.py` 에 렌더러를 추가하라.",
            "",
        ] + [f"- `{n}`" for n in extra] + [""]
    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_EXP_LOG)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--check", action="store_true",
                        help="갱신하지 않고, 낡았으면 종료코드 1")
    args = parser.parse_args()

    if not args.readme.exists():
        print(f"README 가 없다: {args.readme}")
        return 1
    text = args.readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"README 에 마커가 없다. 아래 두 줄을 넣어라:\n{BEGIN}\n{END}")
        return 1

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    block = build_block(load_runs(args.log), args.log)
    updated = head + block + tail

    if args.check:
        if updated != text:
            print("README 의 측정 수치가 EXP_LOG 와 다르다. tools/update_readme.py 를 돌려라.")
            return 1
        # Agreement is not freshness. See freshness() for why this second check exists.
        # 일치는 최신성이 아니다. 이 두 번째 검사가 왜 있는지는 freshness() 참조.
        if "이 수치들은 현재 코드에서 나온 것이 아니다" in updated:
            print("README 와 EXP_LOG 는 일치하지만 **EXP_LOG 가 현재 코드보다 낡았다.**")
            print("서버 EXP_LOG.jsonl 을 커밋하거나, 바뀐 코드로 재측정하라.")
            return 1
        print("README 측정 수치가 EXP_LOG 와 일치하고, EXP_LOG 가 현재 리비전이다.")
        return 0

    if updated == text:
        print("변경 없음 — 이미 최신이다.")
        return 0
    args.readme.write_text(updated, encoding="utf-8")
    print(f"갱신: {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
