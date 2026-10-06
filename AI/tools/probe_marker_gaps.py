"""Locate gripper-marker detection gaps within episodes and compare two batches.
그리퍼 마커 결손이 에피소드의 어느 구간에서 나는지 재고 두 배치를 비교한다.

지표 (편마다)
  det_rate      두 마커 검출 프레임 비율
  max_gap       최장 연속 미검출 프레임
  gap_start     최장 결손이 시작하는 위치 (0=시작, 1=끝)
  dw_gap        최장 결손 직전 폭 − 직후 폭 [mm]  (작으면 결손 중 그리퍼가 안 움직임 → 보간 무해 후보)
  phase_miss    구간별 미검출 비율 — 앞 1/3 · 중간 1/3 · 뒤 1/3
  w_start       첫 20프레임 검출된 폭 중앙 [mm]  (기준 접근 폭 70.4 = 물리 80)

사용: ~/envs/handoff312/bin/python ~/probe_marker_gaps.py
"""
from __future__ import annotations

import csv
import glob
import os
import statistics as st
import sys

BATCHES = {
    "0923": os.path.expanduser("~/data/data2_pilotA/processed"),
    "0924": os.path.expanduser("~/data/data3_0924/processed"),
}


def longest_false(det: list[bool]) -> tuple[int, int]:
    """Return (length, start index) of the longest False run; (0, -1) if none.
    가장 긴 False 구간의 (길이, 시작 인덱스). 없으면 (0, -1)."""
    best, best_i, cur, cur_i = 0, -1, 0, 0
    for i, d in enumerate(det):
        if not d:
            if cur == 0:
                cur_i = i
            cur += 1
            if cur > best:
                best, best_i = cur, cur_i
        else:
            cur = 0
    return best, best_i


def probe(path: str) -> dict | None:
    """Per-episode gap metrics from gripper_width.csv; None if empty.
    gripper_width.csv 에서 편별 지표. 비어 있으면 None."""
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    if n == 0:
        return None
    det = [r["marker_detected"] == "True" for r in rows]
    raw = [float(r["gripper_width_mm"]) if d else None for r, d in zip(rows, det)]
    L, s = longest_false(det)
    before = next((raw[i] for i in range(s - 1, -1, -1) if raw[i] is not None), None) if L else None
    after = next((raw[i] for i in range(s + L, n) if raw[i] is not None), None) if L else None
    thirds = [det[: n // 3], det[n // 3 : 2 * n // 3], det[2 * n // 3 :]]
    w0 = [w for w in raw[:20] if w is not None]
    return {
        "n": n,
        "det_rate": sum(det) / n,
        "max_gap": L,
        "gap_start": (s / n) if L else None,
        "dw_gap": (abs(before - after) if before is not None and after is not None else None),
        "phase_miss": [1 - sum(t) / len(t) if t else None for t in thirds],
        "w_start": st.median(w0) if w0 else None,
    }


def selftest() -> None:
    """Known answers for longest_false, including the all-detected and all-missing rows.
    정답을 아는 행: 결손 없음 / 전부 결손 / 중간 결손."""
    cases = [([True] * 5, (0, -1)), ([False] * 4, (4, 0)), ([True, False, False, True, False], (2, 1))]
    bad = 0
    for det, want in cases:
        got = longest_false(det)
        bad += got != want
        print(f"  selftest {det} want {want} got {got} {'OK' if got == want else 'FAIL'}")
    if bad:
        sys.exit(f"자체검사 실패 {bad}건")
    print("  자체검사 3/3")


def q(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


def summarize(name: str, root: str) -> None:
    files = sorted(glob.glob(f"{root}/rec_*/gripper_width.csv"))
    res = [(os.path.basename(os.path.dirname(f)), probe(f)) for f in files]
    res = [(r, m) for r, m in res if m]
    print(f"\n[{name}] 편 {len(res)} / 파일 {len(files)}   ({root})")
    if not res:
        return
    det = [m["det_rate"] for _, m in res]
    gap = [m["max_gap"] for _, m in res]
    print(f"  검출률      중앙 {st.median(det):.3f}  q1 {q(det,.25):.3f}  q3 {q(det,.75):.3f}")
    print(f"  최장 결손   중앙 {st.median(gap)}  q3 {q(gap,.75)}  최대 {max(gap)}   >15: {sum(g>15 for g in gap)} / {len(gap)}")
    long = [m for _, m in res if m["max_gap"] > 15]
    if long:
        gs = [m["gap_start"] for m in long]
        bins = [sum(1 for x in gs if lo <= x < hi) for lo, hi in ((0, .33), (.33, .67), (.67, 1.01))]
        print(f"  >15 편의 최장 결손 시작 위치   앞 {bins[0]} · 중간 {bins[1]} · 뒤 {bins[2]}   (모수 {len(gs)})")
        dw = [m["dw_gap"] for m in long if m["dw_gap"] is not None]
        if dw:
            print(f"  >15 편의 결손 전후 폭 차 mm    중앙 {st.median(dw):.1f}  최대 {max(dw):.1f}   ≤3mm: {sum(x<=3 for x in dw)} / {len(dw)}")
    pm = list(zip(*[m["phase_miss"] for _, m in res]))
    print("  구간별 미검출 중앙   앞 {:.3f} · 중간 {:.3f} · 뒤 {:.3f}".format(*[st.median(p) for p in pm]))
    ws = [m["w_start"] for _, m in res if m["w_start"] is not None]
    print(f"  시작 폭 mm  중앙 {st.median(ws):.1f}  q1 {q(ws,.25):.1f}  q3 {q(ws,.75):.1f}   (기준 70.4)  미판정 {len(res)-len(ws)}")


if __name__ == "__main__":
    selftest()
    for k, v in BATCHES.items():
        summarize(k, v)
