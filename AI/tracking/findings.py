"""One always-current ledger of what every experiment actually found.
모든 실험이 실제로 무엇을 발견했는지 담은, 항상 최신인 원장 하나.

The data was never missing -- `EXP_LOG.jsonl` had it all along. What was missing
was it being **in front of the person deciding**. On 2026-09-07 a completed v4
run (98 episodes, 3 trainings, 0/300) was treated as "not yet collected", an
`--epochs 30` that every prior run had passed explicitly was dropped, and a
hypothesis was proposed that its own recorded result had already refuted. None of
those were memory failures that a better memory would fix; they were failures to
look. So this file is generated, never hand-written, and printed before a run
starts.
데이터가 없었던 적은 없다 -- `EXP_LOG.jsonl` 에 처음부터 다 있었다. 없었던 것은
그것이 **결정하는 사람 눈앞에** 있는 상태였다. 2026-09-07 에 완주한 v4(98편·학습
3회·0/300)를 "미수집"으로 취급했고, 이전 실행이 전부 명시했던 `--epochs 30` 을
빠뜨렸고, 자기 기록이 이미 반증한 가설을 제안했다. 셋 다 기억력으로 고칠 문제가
아니라 **보지 않은** 문제다. 그래서 이 파일은 손으로 쓰지 않고 생성하며,
실행 시작 전에 찍는다.

손으로 쓰는 해석(기각된 가설·살아있는 가설)은 `findings_notes.yaml` 에 따로 둔다.
생성물과 해석을 한 파일에 섞으면 재생성할 때마다 해석이 날아간다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paths import AI_ROOT, DEFAULT_EXP_LOG

# 로그 위치는 paths 가 정본이다. 여기서 다시 계산하면 파일이 옮겨질 때 조용히 갈라진다.
EXP_LOG = DEFAULT_EXP_LOG
NOTES = AI_ROOT / "findings_notes.yaml"
OUT = AI_ROOT / "FINDINGS.md"


def _records(log: Path | None = None) -> list[dict[str, Any]]:
    log = EXP_LOG if log is None else log
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _pct(x: Any, nd: int = 1) -> str:
    return "—" if x is None else f"{float(x) * 100:.{nd}f}%"


def policy_runs(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every repeat_runs execution, newest last. This is the score history.
    모든 repeat_runs 실행. 이것이 성적 이력이다."""
    rows = []
    for r in recs:
        if r.get("experiment") != "repeat_runs":
            continue
        c, x = r.get("conditions", {}) or {}, r.get("result", {}) or {}
        per = x.get("per_run", []) or []
        lo, hi = (x.get("ci95") or [None, None])[:2]
        rows.append({
            "ts": str(r.get("ts", ""))[:16].replace("T", " "),
            "data": str(c.get("data", "")).split("/")[-1],
            "tag": c.get("tag") or "",
            "epochs": c.get("epochs"),
            "noise": c.get("image_noise_gray"),
            "n": x.get("pooled_n"),
            "mean": x.get("mean"),
            "rates": [p.get("rate") for p in per],
            "vals": [p.get("val_loss") for p in per],
            "ci": (lo, hi),
            "passed": x.get("passed"),
        })
    return rows


def diagnostics(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diagnostic runs, flattened to one line each with their verdict.
    진단 실행을 한 줄씩, 판정과 함께."""
    rows = []
    for r in recs:
        exp = r.get("experiment")
        c, x = r.get("conditions", {}) or {}, r.get("result", {}) or {}
        ts = str(r.get("ts", ""))[:16].replace("T", " ")
        ck = str(c.get("policy_ckpt") or c.get("ckpt") or "").split("/")[-1]

        if exp == "trace_execution":
            a, b, cc = x.get("A", {}) or {}, x.get("B", {}) or {}, x.get("C", {}) or {}
            rows.append({"ts": ts, "kind": "trace", "ckpt": ck, "note": (
                f"A {'성공' if a.get('success') else '실패'} · "
                f"B 비율 {b.get('ratio', float('nan')):.4f} · "
                f"**C 폐루프 {'성공' if cc.get('success') else '실패'}** "
                f"(상승 {cc.get('lift_cm', 0):.3f}cm)")})
        elif exp == "diagnose_bc":
            v = x.get("verdicts", {}) or {}
            sc = {s.get("name"): s.get("l1") for s in (x.get("scores") or [])}
            rows.append({"ts": ts, "kind": "diagnose", "ckpt": ck, "note": (
                f"bc L1 {sc.get('bc', float('nan')):.6f} vs identity "
                f"{sc.get('identity', float('nan')):.6f} · "
                f"붕괴아님 {v.get('not_collapsed')} · 자명예측기초과 {v.get('beats_trivial')}")})
        elif exp == "image_sensitivity":
            rows.append({"ts": ts, "kind": "이미지민감도", "ckpt": ck, "note": (
                f"통과 {x.get('passed')} · 델타기준 비율 {x.get('delta_ratio', float('nan')):.3f} · "
                f"절대기준 {x.get('overall_ratio', float('nan')):.3f}")})
        elif exp == "check_determinism":
            rd, po, ro = x.get("render", {}) or {}, x.get("policy", {}) or {}, x.get("rollout", {}) or {}
            rows.append({"ts": ts, "kind": "결정론", "ckpt": ck, "note": (
                f"R 같은프로세스 {rd.get('same_process_ok')} · R 새env {rd.get('new_env_ok')} · "
                f"P {po.get('ok')} · E {ro.get('ok')}")})
        elif exp == "recovery_probe":
            su = x.get("summary", {}) or {}
            fb = {k.split("|")[1]: v for k, v in su.items() if k.startswith("scripted_fb|")}
            op = (su.get("scripted|none") or {}).get("rate")
            curve = " · ".join(f"{k} {v.get('rate', 0) * 100:.0f}%"
                               for k, v in sorted(fb.items(), key=lambda kv: kv[0]))
            rows.append({"ts": ts, "kind": "복구(G0)", "ckpt": "scripted_fb", "note": (
                f"G0-a {x.get('g0a_passed')} · G0-b {x.get('g0b_passed')} · "
                f"**DAgger {'착수 가능' if x.get('dagger_go') else '착수 불가'}** · "
                f"개루프 {op * 100:.0f}% · " if op is not None else "") + curve})
        elif exp == "collect_sim":
            rows.append({"ts": ts, "kind": "수집", "ckpt": str(c.get("skill_id", "")), "note": (
                f"{x.get('episodes_written')}편 저장 · 도달불가 {x.get('skipped_unreachable')} · "
                f"파지 {x.get('grasp_success')} · {x.get('mb_per_episode', 0):.2f}MB/편 · "
                f"지터 {c.get('jitter_m')}")})
    return rows


def _overlaps(a: tuple[Any, Any], b: tuple[Any, Any]) -> bool:
    """Do two 95% intervals overlap? Overlapping runs are not distinguishable.
    두 95% 구간이 겹치는가? 겹치는 두 실행은 이 n 으로 구분되지 않는다.

    Ranking by mean alone is what this file exists to prevent: a run whose mean
    is higher but whose interval overlaps the previous best has not been shown to
    be better. So "best" is always reported together with everything it cannot be
    told apart from.
    평균만으로 줄 세우는 것이 이 파일이 막으려는 실패다. 평균이 높아도 구간이
    겹치면 더 낫다는 것이 보인 것이 아니다. 그래서 "최고"는 언제나 그것과
    구분되지 않는 실행들과 함께 보고한다.
    """
    if a[0] is None or b[0] is None:
        return False
    return a[0] <= b[1] and b[0] <= a[1]


def _best(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Highest mean, and every run its interval overlaps.
    평균 최고 실행과, 그것과 구간이 겹치는 모든 실행."""
    best = max(rows, key=lambda r: (r["mean"] is not None, r["mean"] or 0))
    tied = [r for r in rows if r is not best and _overlaps(r["ci"], best["ci"])]
    return best, tied


def _notes() -> dict[str, Any]:
    if not NOTES.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(NOTES.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def brief(recs: list[dict[str, Any]] | None = None, last: int = 6) -> str:
    """The short form printed before a run starts. Score history only.
    실행 시작 전에 찍는 짧은 형태. 성적 이력만."""
    recs = _records() if recs is None else recs
    rows = policy_runs(recs)
    if not rows:
        return "정책 실험 기록 없음"
    best, tied = _best(rows)
    out = ["지금까지 정책 성적 (FINDINGS.md 전문)"]
    for r in rows[-last:]:
        star = " ★평균최고" if r is best else (" ≈최고와 구분안됨" if r in tied else "")
        cond = f"e{r['epochs']}"
        if r["noise"]:
            cond += f" σ{r['noise']}"
        if r["tag"]:
            cond += f" {r['tag']}"
        rates = "/".join(_pct(v, 0) for v in r["rates"])
        ci = f"{_pct(r['ci'][0])}~{_pct(r['ci'][1])}" if r["ci"][0] is not None else ""
        out.append(f"  {r['data']:<14} {cond:<16} 평균 {_pct(r['mean']):>6}  "
                   f"[{rates}]  CI {ci}{star}")
    if tied:
        out.append(f"  ⚠️ 최고({_pct(best['mean'])})와 구간이 겹치는 실행 {len(tied)}건 — "
                   "이 n 으로는 순위가 정해지지 않는다")
    out.append("  ⚠️ 구간이 겹치는 두 실행은 이 n 으로 구분되지 않는다")
    return "\n".join(out)


def render(recs: list[dict[str, Any]] | None = None) -> str:
    recs = _records() if recs is None else recs
    runs, diag, notes = policy_runs(recs), diagnostics(recs), _notes()
    L: list[str] = []
    L.append("# FINDINGS — 실험이 실제로 발견한 것\n")
    L.append("> **이 파일은 `tools/findings.py` 가 `EXP_LOG.jsonl` 에서 생성한다. 손으로 고치지 마라.**")
    L.append("> 해석·가설은 `findings_notes.yaml` 에 쓰면 여기에 합쳐진다.")
    L.append("> 실험을 시작하기 전에 이 파일을 읽는다. 기억으로 진행하지 않는다.\n")

    # 이 원장이 어느 로그에서 나왔는지 밝힌다. 비어 있음은 "실험이 없다"가 아니라
    # "이 로그에 없다"이며, 학습은 서버에서 돈다 — 로그가 커밋되기 전에는 여기가 뒤처진다.
    newest = max((str(r.get("ts", "")) for r in recs), default="")
    L.append(f"로그 기록 **{len(recs)}건** · 최신 기록 `{newest[:16].replace('T', ' ') or '없음'}`\n")
    if not runs:
        L.append("> ⚠️ **이 로그에 `repeat_runs` 기록이 없다.** 학습·평가는 서버에서 돌므로, "
                 "서버의 `EXP_LOG.jsonl` 을 커밋하기 전에는 이 원장이 성적을 못 본다. "
                 "기록 없음을 실험 없음으로 읽지 마라.\n")

    if runs:
        best, tied = _best(runs)
        L.append("## 현재 최고 기록\n")
        L.append(f"**{_pct(best['mean'])}** · `{best['data']}` "
                 f"epochs {best['epochs']}"
                 + (f" · 잡음 σ{best['noise']}" if best["noise"] else "")
                 + (f" · tag {best['tag']}" if best["tag"] else "")
                 + f" · n={best['n']} · CI {_pct(best['ci'][0])}~{_pct(best['ci'][1])}")
        L.append(f"\n배포 게이트 20% — **{'통과' if best['passed'] else '미통과'}**\n")
        if tied:
            L.append("⚠️ **이것을 '최고'라고 부를 수 없다.** 아래 실행들과 95% 구간이 겹친다 — "
                     "이 n 으로는 순위가 정해지지 않는다. 게이트를 옮기지 말고 n 을 올려라.\n")
            for t in tied:
                L.append(f"- `{t['data']}` {t['ts']} · 평균 {_pct(t['mean'])} · "
                         f"CI {_pct(t['ci'][0])}~{_pct(t['ci'][1])}")
            L.append("")

        L.append("## 정책 실험 이력 (repeat_runs)\n")
        L.append("| 시각 | 데이터 | 조건 | n | 평균 | per_run | 95% 구간 | val_loss |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in runs:
            cond = f"e{r['epochs']}"
            if r["noise"]:
                cond += f" σ{r['noise']}"
            if r["tag"]:
                cond += f" `{r['tag']}`"
            rates = " / ".join(_pct(v, 0) for v in r["rates"])
            vals = " / ".join("—" if v is None else f"{v:.4f}" for v in r["vals"])
            ci = f"{_pct(r['ci'][0])}~{_pct(r['ci'][1])}" if r["ci"][0] is not None else "—"
            L.append(f"| {r['ts']} | {r['data']} | {cond} | {r['n']} | "
                     f"**{_pct(r['mean'])}** | {rates} | {ci} | {vals} |")
        L.append("\n⚠️ **구간이 겹치는 두 실행은 이 n 으로 구분되지 않는다.** "
                 "평균이 높다고 개선된 것이 아니다.\n")

    if diag:
        L.append("## 진단 이력\n")
        L.append("| 시각 | 종류 | 대상 | 결과 |")
        L.append("| --- | --- | --- | --- |")
        for d in diag[-30:]:
            L.append(f"| {d['ts']} | {d['kind']} | {d['ckpt']} | {d['note']} |")
        L.append("")

    for key, title in (("facts", "확정된 사실 — 여기서 시작한다"),
                       ("rejected", "기각된 가설"), ("alive", "살아있는 가설 / 다음 후보"),
                       ("instruments", "계측기 상태")):
        items = notes.get(key) or []
        if not items:
            continue
        L.append(f"## {title}\n")
        for it in items:
            if isinstance(it, dict):
                head = it.get("name", "")
                L.append(f"- **{head}** — {it.get('what', '')}")
                for k, label in (("evidence", "근거"), ("revive", "되살릴 조건"),
                                 ("next", "다음")):
                    if it.get(k):
                        L.append(f"  - {label}: {it[k]}")
            else:
                L.append(f"- {it}")
        L.append("")
    return "\n".join(L) + "\n"


def write(path: Path | None = None) -> Path:
    path = OUT if path is None else path
    path.write_text(render(), encoding="utf-8")
    return path
