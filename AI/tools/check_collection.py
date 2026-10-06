"""Judge a freshly built UMI dataset report against the 2026-09-21 baseline.
새로 빌드한 UMI 데이터셋 리포트를 2026-09-21 기준선과 대조해 판정한다.

Baseline / 기준선 — s22_pick_20260921_atlas_v1, 70편 전수 🟢
    접근 개구(편별 최대)  중앙 70.40mm  q1 69.64  q3 71.15
    파지 개구(편별 최소)  중앙 39.10mm  q1 38.65  q3 39.50
    편 길이              중앙  7.23s   5.50~10.00
    마커 검출률           중앙  0.86    최소 0.74
    outcome              success 70/70

Recorded width offset / 기록 오프셋 (독립 2점에서 일치, 2026-09-22) 🟢
    핸들 물리 80mm -> 기록 70.40mm   차 -9.6
    물체 물리 48mm -> 기록 39.10mm   차 -8.9
    => 기록값 = 물리 - 9mm. 그래서 물체 48mm 의 접촉 파지는 기록 39mm 근처여야 한다.

Every check prints its value, its threshold and the parameter set.
모든 검사가 값·기준·모수를 함께 찍는다. 미판정은 통과가 아니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

OFFSET_MM = 9.0          # 기록값 = 물리 - 9mm
APPROACH_MM = 70.40      # 기준선 접근 개구 중앙
APPROACH_TOL = 3.0
DURATION_RANGE = (5.0, 12.0)
APPROACH_IQR_MAX = 3.0
DETECT_MEDIAN_MIN = 0.80
DETECT_MIN_MIN = 0.70
GRASP_SLACK_MM = 1.0     # 기록 파지 중앙이 (물체-9)+1.0 이하여야 한다


def quantile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("빈 표본")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def judge(report: dict[str, Any], object_mm: float, minimum_episodes: int
          ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (rows, tally). Each row carries 값·기준·판정·모수."""
    eps = report.get("episode_reports")
    if not isinstance(eps, list) or not eps:
        raise ValueError("episode_reports 가 없다 - 이 파일은 zarr.report.json 이 아니다")
    n = len(eps)

    def col(key: str, mul: float = 1.0) -> list[float]:
        return [e[key] * mul for e in eps if isinstance(e.get(key), (int, float))]

    rows: list[dict[str, Any]] = []

    def add(name: str, got: Any, want: str, ok: bool | None, denom: int) -> None:
        rows.append({"name": name, "got": got, "want": want, "ok": ok, "n": denom})

    add("편수", n, f">= {minimum_episodes}", n >= minimum_episodes, n)

    outcomes = [e.get("outcome") for e in eps]
    good = sum(1 for o in outcomes if o == "success")
    add("outcome success", f"{good} / {n}", "전부 success", good == n, n)

    ap = col("gripper_width_max_m", 1000)
    if len(ap) != n:
        add("접근 개구 중앙", f"표본 {len(ap)} / {n}", "모든 편에 값", None, n)
    else:
        med = quantile(ap, 0.5)
        add("접근 개구 중앙", f"{med:.2f} mm",
            f"{APPROACH_MM:.2f} +-{APPROACH_TOL:.0f} mm",
            abs(med - APPROACH_MM) <= APPROACH_TOL, n)
        iqr = quantile(ap, 0.75) - quantile(ap, 0.25)
        add("접근 개구 IQR", f"{iqr:.2f} mm", f"<= {APPROACH_IQR_MAX:.1f} mm",
            iqr <= APPROACH_IQR_MAX, n)

    gr = col("gripper_width_min_m", 1000)
    target = object_mm - OFFSET_MM
    if len(gr) != n:
        add("파지 개구 중앙", f"표본 {len(gr)} / {n}", "모든 편에 값", None, n)
    else:
        med = quantile(gr, 0.5)
        add("파지 개구 중앙", f"{med:.2f} mm",
            f"<= {target + GRASP_SLACK_MM:.2f} mm (물체 {object_mm:.0f} - 오프셋 {OFFSET_MM:.0f} + 여유 {GRASP_SLACK_MM:.1f})",
            med <= target + GRASP_SLACK_MM, n)
        loose = sum(1 for v in gr if v > target + GRASP_SLACK_MM)
        add("덜 조인 편", f"{loose} / {n}", "<= 10%", loose <= max(1, n // 10), n)

    du = col("duration_s")
    if len(du) == n:
        med = quantile(du, 0.5)
        add("편 길이 중앙", f"{med:.2f} s",
            f"{DURATION_RANGE[0]:.1f} ~ {DURATION_RANGE[1]:.1f} s",
            DURATION_RANGE[0] <= med <= DURATION_RANGE[1], n)
    else:
        add("편 길이 중앙", f"표본 {len(du)} / {n}", "모든 편에 값", None, n)

    de = col("marker_detection_rate")
    if len(de) == n:
        add("마커 검출률 중앙", f"{quantile(de, 0.5):.3f}", f">= {DETECT_MEDIAN_MIN}",
            quantile(de, 0.5) >= DETECT_MEDIAN_MIN, n)
        add("마커 검출률 최소", f"{min(de):.3f}", f">= {DETECT_MIN_MIN}",
            min(de) >= DETECT_MIN_MIN, n)
    else:
        add("마커 검출률", f"표본 {len(de)} / {n}", "모든 편에 값", None, n)

    lost = [e.get("lost_frames_after_initialization") for e in eps]
    clean = sum(1 for v in lost if v == 0)
    add("추적 손실 0 인 편", f"{clean} / {n}", "전부", clean == n, n)

    tally = {"PASS": sum(1 for r in rows if r["ok"] is True),
             "FAIL": sum(1 for r in rows if r["ok"] is False),
             "미판정": sum(1 for r in rows if r["ok"] is None)}
    return rows, tally


def render(rows: list[dict[str, Any]], tally: dict[str, int]) -> str:
    out = []
    for r in rows:
        mark = "PASS" if r["ok"] is True else ("FAIL" if r["ok"] is False else "미판정")
        out.append(f"  [{mark:4s}] {r['name']:18s} {str(r['got']):>28s}   기준 {r['want']}")
    total = sum(tally.values())
    out.append(f"\n통과 {tally['PASS']} · 불합격 {tally['FAIL']} · 미판정 {tally['미판정']} / 전체 {total}")
    verdict = "PASS" if tally["FAIL"] == 0 and tally["미판정"] == 0 else "FAIL"
    out.append(f"판정: {verdict}   (미판정은 통과가 아니다)")
    return "\n".join(out)


def _episode(width_max: float, width_min: float, duration: float = 7.2,
             detect: float = 0.86, outcome: str = "success") -> dict[str, Any]:
    return {"gripper_width_max_m": width_max / 1000, "gripper_width_min_m": width_min / 1000,
            "duration_s": duration, "marker_detection_rate": detect,
            "lost_frames_after_initialization": 0, "outcome": outcome}


def _selftest() -> int:
    checks: list[tuple[str, bool]] = []
    goodeps = [_episode(70.4 + (i % 3) * 0.5, 38.8 + (i % 3) * 0.3) for i in range(20)]
    rows, tally = judge({"episode_reports": goodeps}, 48.0, 20)
    checks.append(("정상 20편 전부 통과", tally["FAIL"] == 0 and tally["미판정"] == 0))

    # 판별행 - 고의로 망가뜨린 입력을 잡아야 한다
    def fails(name: str, eps: list[dict[str, Any]], obj: float = 48.0, minimum: int = 20) -> None:
        _, t = judge({"episode_reports": eps}, obj, minimum)
        checks.append((name, t["FAIL"] > 0))

    fails("판별: 덜 조임(파지 44mm)", [_episode(70.4, 44.0) for _ in range(20)])
    fails("판별: 접근 개구 80mm", [_episode(80.0, 38.9) for _ in range(20)])
    fails("판별: 검출률 저하", [_episode(70.4, 38.9, detect=0.6) for _ in range(20)])
    fails("판별: 실패 편 포함",
          [_episode(70.4, 38.9) for _ in range(19)] + [_episode(70.4, 38.9, outcome="fail")])
    fails("판별: 편수 부족", [_episode(70.4, 38.9) for _ in range(5)])
    fails("판별: 편 길이 이상", [_episode(70.4, 38.9, duration=2.0) for _ in range(20)])
    fails("판별: 접근 분산 큼",
          [_episode(66.0 + (i % 2) * 9.0, 38.9) for i in range(20)])

    rows, _ = judge({"episode_reports": goodeps}, 41.0, 20)
    bad = [r for r in rows if r["name"] == "파지 개구 중앙"][0]
    checks.append(("물체 41mm 면 같은 데이터가 FAIL", bad["ok"] is False))

    try:
        judge({"foo": 1}, 48.0, 20); checks.append(("판별: 잘못된 파일", False))
    except ValueError:
        checks.append(("판별: 잘못된 파일", True))

    ok = sum(1 for _, p in checks if p)
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"자체검사 통과 {ok} / 전체 {len(checks)}")
    return 0 if ok == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="수집 데이터셋 20편 판정")
    ap.add_argument("report", nargs="?", help="*.zarr.report.json 경로")
    ap.add_argument("--object-mm", type=float, default=48.0, help="물체 실측 폭(파지하는 면)")
    ap.add_argument("--min-episodes", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or not args.report:
        return _selftest()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    print(f"리포트 {args.report}")
    print(f"물체 {args.object_mm:.0f}mm · 기록 오프셋 -{OFFSET_MM:.0f}mm "
          f"-> 기대 파지 기록값 {args.object_mm - OFFSET_MM:.1f}mm\n")
    rows, tally = judge(report, args.object_mm, args.min_episodes)
    print(render(rows, tally))
    return 0 if tally["FAIL"] == 0 and tally["미판정"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
