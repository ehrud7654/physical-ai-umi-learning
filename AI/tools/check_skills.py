"""Show the skill catalog and validate a registry against it.
스킬 목록을 보이고 레지스트리를 그것에 대조한다.

Run with no arguments to see what the five skills are and what a deployable
entry has to satisfy. Point it at a registry directory to find out whether
anything in there is actually deployable — which today nothing is, because no
policy has passed its gate.
인자 없이 실행하면 스킬 다섯이 무엇이고 배포 가능한 항목이 무엇을 만족해야 하는지
보여준다. 레지스트리 디렉터리를 주면 그 안의 것이 실제로 배포 가능한지 판정한다.
오늘 기준 배포 가능한 것은 없다. 게이트를 넘은 정책이 아직 없기 때문이다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.skills import (  # noqa: E402
    CATALOG,
    DESTINATIONS,
    GATE_MIN_N,
    GATE_MIN_RUNS,
    OBJECT_SIZE_MM,
    ROLLOUT_GATE,
    SKILL_IDS,
    VLM_CONFIDENCE_FLOOR,
    PolicyRef,
    PostAction,
    SkillEntry,
    load_registry,
    parse_vlm_decision,
    shared_policy_report,
    should_execute,
    validate_entry,
    write_entry,
)
from sim.mujoco.build_scene import load_config  # noqa: E402


def print_catalog() -> None:
    print(f"{'skill_id':16s} {'이름':14s} {'층':9s} {'VLM':4s} {'놓기 단계':30s} destination")
    print("-" * 100)
    for sid in SKILL_IDS:
        s = CATALOG[sid]
        print(
            f"{s.skill_id:16s} {s.display_name:14s} {s.tier:9s} "
            f"{'필요' if s.needs_vlm else '  · ':4s} "
            f"{s.place_step:30s} {','.join(s.destinations)}"
        )


def print_gates() -> None:
    print("배포 게이트 (결과 확인 전 확정):")
    print(f"  롤아웃 성공률 > {ROLLOUT_GATE:.2f}, 95% 신뢰구간 하한도 그 위여야 한다")
    print(f"  n >= {GATE_MIN_N}      — n=20 은 CI 반폭 ±18%p 로 게이트 폭과 같다 🟢 2026-09-01")
    print(f"  runs >= {GATE_MIN_RUNS}       — 동일 데이터 학습 2회에서 성공률 25%p 차이 🟢 2026-09-01")
    print(f"  대상물 {OBJECT_SIZE_MM[0]:.0f}~{OBJECT_SIZE_MM[1]:.0f}mm, 작업영역은 실측 도달 사각형 안")
    print(f"  VLM confidence < {VLM_CONFIDENCE_FLOOR:.2f} 이면 실행하지 않고 되묻는다 🟡 잠정값")


def draft_entries(cfg: dict) -> list[SkillEntry]:
    """Registry rows for the five skills as they stand today: no policy yet.
    오늘 시점의 다섯 스킬 레지스트리 행. 아직 정책이 없다.

    They are written deliberately incomplete. An entry with no gate must fail
    validation, and seeing it fail is the check that the guard is real.
    일부러 미완성으로 만든다. 게이트 없는 항목은 검증에 실패해야 하고, 실패하는
    것을 보는 것이 그 가드가 진짜인지 확인하는 방법이다.
    """
    rect = cfg["workspace"]["reachable_rect_m"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[SkillEntry] = []
    for sid in SKILL_IDS:
        spec = CATALOG[sid]
        out.append(
            SkillEntry(
                skill_id=sid,
                version="0.1.0-draft",
                tier=spec.tier,
                status="planned",
                policy=PolicyRef(
                    kind="learned",
                    ckpt_uri="",
                    trained_on="",
                    action_space="next_joint_position_and_gap",
                    gate=None,
                ),
                post_actions=[
                    PostAction(name=spec.place_step, destination=d, params={})
                    for d in spec.destinations
                ],
                object_size_mm=list(OBJECT_SIZE_MM),
                workspace_m={"x": list(rect["x"]), "y": list(rect["y"])},
                robot={"model": cfg["robot"]["name"], "dof": int(cfg["robot"]["dof"])},
                created_by="AI 학습 담당자(트랙B)",
                created_at=now,
                notes={
                    "plan": "A — 파지 정책 1개 공유. B(스킬별 독립 정책)는 확장 목표",
                    "blocked_by": "파지 정책이 게이트 미통과. 계약 skill 필드는 트랙 A 확인 대기",
                },
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=None, help="검증할 레지스트리 디렉터리")
    parser.add_argument("--emit-draft", type=Path, default=None, help="초안 항목 5개를 쓸 위치")
    args = parser.parse_args()

    cfg = load_config()

    print("스킬 5종 (D-AI-22)\n")
    print_catalog()
    print()
    print_gates()

    if args.emit_draft:
        entries = draft_entries(cfg)
        for e in entries:
            write_entry(e, args.emit_draft)
        print(f"\n초안 {len(entries)}건 기록: {args.emit_draft}")

    target = args.registry or args.emit_draft
    if target is None:
        print("\nVLM 출력 예시 (constrained decoding 대상은 skill_id·destination enum):")
        sample = {
            "skill_id": "sort_two",
            "target": {"description_ko": "긁힘 있는 부품", "bbox_norm": [0.31, 0.44, 0.39, 0.52]},
            "destination": "right_tray",
            "confidence": 0.87,
            "abstain": False,
        }
        d = parse_vlm_decision(sample)
        ok, why = should_execute(d)
        print(f"  {d.to_json()}")
        print(f"  실행 가능: {ok}{('  — ' + why) if why else ''}")
        print(f"\n허용 destination: {DESTINATIONS}")
        return 0

    entries = load_registry(target)
    if not entries:
        print(f"\n{target} 에 항목이 없다.")
        return 1

    print(f"\n레지스트리 검증 — {target}\n")
    n_bad = 0
    for e in entries:
        problems = validate_entry(e, cfg)
        mark = "배포 가능" if not problems else "배포 불가"
        print(f"  [{mark}] {e.skill_id} v{e.version}")
        for p in problems:
            print(f"      · {p}")
        n_bad += bool(problems)

    print()
    print(shared_policy_report(entries))
    print()
    print(f"배포 가능 {len(entries) - n_bad}/{len(entries)}")
    if n_bad:
        print(
            "⚠️ 배포 불가가 정상이다. 파지 정책이 아직 게이트를 넘지 못했고, "
            "레지스트리는 그 사실을 숨기지 않는다."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
