"""Post-filter a gate-built episode plan: drop SLAM jumps and long marker gaps.
게이트 plan 에서 SLAM 점프 편과 마커 결손이 긴 편을 뺀다.

규칙 (2026-09-24, AI 학습 담당자)
  - 게이트 PASS (make_episode_plan 이 이미 거름)
  - 프레임 간 최대 이동 <= 50mm (state==2 이고 is_lost=false 인 연속 프레임만)
    → gate_slam_batch 와 도경 validation 둘 다 이 점프를 못 잡는다 (0924 실측 2편)
  - gripper maximum_missing_run_frames <= 15 (빌더 configs/dataset.json 한계와 동일.
    넘으면 build 가 전체 중단된다 — 0921 실증)

사용: ~/envs/handoff312/bin/python ~/filter_plan_0924.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

B = os.path.expanduser("~/data/data3_0924")
SRC = f"{B}/out/plan_gate.json"
DST = f"{B}/out/plan_0924.json"
STEP_LIMIT_M = 0.050
MISSING_LIMIT = 15


def max_step_m(path: str) -> float | None:
    """Largest frame-to-frame move among tracked frames; None if unreadable.
    추적 프레임 간 최대 이동 [m]. 읽을 수 없으면 None."""
    if not os.path.exists(path):
        return None
    best, prev, rows = 0.0, None, 0
    with open(path) as f:
        for x in csv.DictReader(f):
            rows += 1
            if x["is_lost"].lower() == "true" or x["state"] != "2":
                prev = None
                continue
            q = (float(x["x"]), float(x["y"]), float(x["z"]))
            if prev is not None:
                best = max(best, math.dist(prev, q))
            prev = q
    return best if rows else None


def selftest() -> None:
    """Known-answer rows: a clean track, a jump, and an unreadable file.
    정답을 아는 입력 3종."""
    import tempfile

    d = tempfile.mkdtemp()
    head = "frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w\n"
    ok = head + "".join(f"{i},0,2,false,0,{i*0.01},0,0,0,0,0,1\n" for i in range(5))
    jump = head + "0,0,2,false,0,0,0,0,0,0,0,1\n1,0,2,false,0,0.08,0,0,0,0,0,1\n"
    lost = head + "0,0,2,false,0,0,0,0,0,0,0,1\n1,0,1,true,0,9,9,9,0,0,0,1\n2,0,2,false,0,9.001,9,9,0,0,0,1\n"
    cases = {"ok": (ok, 0.01), "jump": (jump, 0.08), "lost_reset": (lost, 0.0)}
    fails = 0
    for name, (txt, want) in cases.items():
        p = os.path.join(d, name + ".csv")
        open(p, "w").write(txt)
        got = max_step_m(p)
        good = got is not None and abs(got - want) < 1e-9
        fails += not good
        print(f"  selftest {name:10s} want {want:.3f} got {got}  {'OK' if good else 'FAIL'}")
    got = max_step_m(os.path.join(d, "missing.csv"))
    fails += got is not None
    print(f"  selftest missing    want None got {got}  {'OK' if got is None else 'FAIL'}")
    if fails:
        sys.exit(f"자체검사 실패 {fails}건 — 필터를 쓰지 않는다")
    print("  자체검사 4/4")


def main() -> None:
    selftest()
    plan = json.load(open(SRC))
    keep: list[dict] = []
    drop: dict[str, list] = {"jump": [], "missing_run": [], "unreadable": []}
    for e in plan["episodes"]:
        r = os.path.basename(e["session"].rstrip("/"))
        step = max_step_m(e["trajectory"])
        mr = json.load(open(f"{B}/processed/{r}/gripper_report.json")).get("maximum_missing_run_frames")
        if step is None:
            drop["unreadable"].append(r)
        elif step > STEP_LIMIT_M:
            drop["jump"].append((r, round(step * 1000, 1)))
        elif mr is None or mr > MISSING_LIMIT:
            drop["missing_run"].append((r, mr))
        else:
            keep.append(e)
    total = len(plan["episodes"])
    dropped = sum(len(v) for v in drop.values())
    if len(keep) + dropped != total:
        sys.exit(f"부분합 불일치 kept {len(keep)} + drop {dropped} != {total}")
    plan["episodes"] = keep
    plan.setdefault("provenance", {})["post_filter"] = {
        "rule": "gate PASS and max frame step <= 50mm (state==2, !is_lost) and gripper maximum_missing_run_frames <= 15",
        "gate_plan": total, "kept": len(keep), "dropped": drop,
    }
    json.dump(plan, open(DST, "w"), indent=1, ensure_ascii=False)
    print(f"게이트 plan {total} → kept {len(keep)}")
    for k, v in drop.items():
        print(f"  {k:12s} {len(v)}  {v[:6]}")
    print(f"→ {DST}")


if __name__ == "__main__":
    main()
