"""The five skills, as code rather than a slide.
스킬 5종. 슬라이드가 아니라 코드로 고정한다.

Why a registry and not a list of names.
왜 목록이 아니라 레지스트리인가.

A skill is what the platform stores, what the user clicks, what the VLM selects
and what the robot then runs. Four parties have to agree on the same identifier,
and one of them is a language model that will happily invent a sixth skill if
nothing constrains it. So the identifiers live in one tuple, `SKILL_IDS`, and
everything else — the VLM's output schema, the registry entries, the frontend's
list — is derived from it or validated against it.
스킬은 플랫폼이 저장하는 것이고, 사용자가 클릭하는 것이고, VLM 이 고르는 것이고,
로봇이 실행하는 것이다. 네 주체가 같은 식별자에 합의해야 하는데 그중 하나는
아무 제약이 없으면 여섯 번째 스킬을 기꺼이 지어내는 언어모델이다. 그래서 식별자는
`SKILL_IDS` 하나에만 두고, VLM 출력 스키마·레지스트리 항목·프런트 목록은 전부
거기서 파생되거나 거기에 대조된다.

Structure (D-AI-22): every skill is [grasp = learned policy] + [place = scripted].
구조 (D-AI-22): 모든 스킬은 [집기 = 학습 정책] + [놓기 = 스크립트] 다.

The split is not an architectural preference, it comes from a measurement. Replay
of a recorded trajectory survives a 5mm object displacement (4/4) and dies at 50mm
(0/4) — so the part of the task where the object position varies needs a policy,
and the part after the object is in the gripper does not. 🟢 2026-08-27
이 분할은 취향이 아니라 계측에서 나왔다. 기록된 궤적의 재생은 물체가 5mm 움직여도
살아남고(4/4) 50mm 에서는 죽는다(0/4). 즉 물체 위치가 변하는 구간에는 정책이
필요하고, 물체가 그리퍼에 들어온 뒤에는 필요 없다.

⚠️ 현재 채택안은 A — 파지 정책 **하나**를 다섯 스킬이 공유하고 놓기 단계만 다르다.
   B(스킬별 독립 정책 5개)는 파지 정책이 게이트를 넘은 뒤의 확장 목표다.
   `shared_policy_report()` 가 지금 어느 상태인지 항상 사실대로 말해준다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from contract.episode import CONTRACT_VERSION
from contract.ids import DESTINATIONS, SKILL_IDS, STATUSES, TIERS

# The only place a skill identifier is defined. Everything else derives from this.
# 스킬 식별자가 정의되는 유일한 자리. 나머지는 전부 여기서 파생된다.
POLICY_KINDS: tuple[str, ...] = ("learned", "scripted")
ACTION_SPACES: tuple[str, ...] = ("next_joint_position_and_gap",)

# Gates. Fixed before any skill was registered.
# 게이트. 스킬을 하나도 등록하기 전에 확정했다.
ROLLOUT_GATE = 0.20
"""A learned policy must exceed this rollout success rate to be deployable.
학습 정책이 배포 가능해지려면 이 롤아웃 성공률을 넘어야 한다."""

GATE_MIN_N = 100
"""n=20 gives a 95% CI half-width of about +-18%p, which is the width of the gate
itself — the instrument cannot resolve the decision. 🟢 2026-09-01
n=20 은 95% 신뢰구간 반폭이 약 ±18%p 로 게이트 폭과 같다. 계측기가 판정을
분해하지 못한다."""

GATE_MIN_RUNS = 3
"""Success rate varied by 25%p between two trainings on identical data and
settings, so a single run's number cannot stand for the checkpoint. 🟢 2026-09-01
동일한 데이터·설정의 두 학습에서 성공률이 25%p 갈렸다. 단일 실행의 수치는
체크포인트를 대표하지 못한다."""

VLM_CONFIDENCE_FLOOR = 0.70
"""Below this the VLM must abstain and ask rather than move the arm.
이 아래면 VLM 은 팔을 움직이지 말고 기권하고 되물어야 한다.

🟡 잠정값이다. M0(파인튜닝 없는 베이스 모델의 스킬 선택 정확도) 측정 후
   재조정하되, 재조정했다는 사실과 이유를 기록에 남긴다."""

OBJECT_SIZE_MM = (15.0, 25.0)
"""Gripper closes to about 1.7cm between fingers, so this is the whole product.
그리퍼 닫힘 시 손가락 간격이 약 1.7cm 다. 이 범위가 제품 전체다."""


class SkillError(ValueError):
    """A skill definition or a VLM decision violates the registry contract.
    스킬 정의 또는 VLM 결정이 레지스트리 계약을 위반했다."""


@dataclass(frozen=True)
class SkillSpec:
    """What a skill is, before any checkpoint exists for it.
    체크포인트가 생기기 전의 스킬 정의."""

    skill_id: str
    display_name: str
    description_ko: str
    place_step: str
    needs_vlm: bool
    destinations: tuple[str, ...]
    tier: str = "tutorial"


CATALOG: dict[str, SkillSpec] = {
    "pick_place": SkillSpec(
        skill_id="pick_place",
        display_name="집어 옮기기",
        description_ko="물체를 집어 지정한 좌표에 놓는다.",
        place_step="지정 좌표로 이동 후 개방",
        needs_vlm=False,
        destinations=("target_pose",),
    ),
    "sort_two": SkillSpec(
        skill_id="sort_two",
        display_name="두 곳으로 분류",
        description_ko="물체를 집고, VLM 판단에 따라 좌·우 트레이 중 하나에 놓는다.",
        place_step="분기된 트레이로 이동 후 개방",
        needs_vlm=True,
        destinations=("left_tray", "right_tray"),
    ),
    "align_fixture": SkillSpec(
        skill_id="align_fixture",
        display_name="지그에 정렬",
        description_ko="물체를 집어 지그의 고정 슬롯에 방향을 맞춰 놓는다.",
        place_step="지그 좌표로 이동, 자세 정렬 후 개방",
        needs_vlm=False,
        destinations=("fixture",),
    ),
    "present_inspect": SkillSpec(
        skill_id="present_inspect",
        display_name="검사 자세 제시",
        description_ko="물체를 집어 카메라 앞으로 들어 올리고 회전시켜 보여준다.",
        place_step="제시 자세로 이동, 회전, 원위치 복귀 후 개방",
        needs_vlm=False,
        destinations=("origin",),
    ),
    "line_up": SkillSpec(
        skill_id="line_up",
        display_name="순서대로 늘어놓기",
        description_ko="물체를 집어 계산된 간격의 다음 자리에 놓는다.",
        place_step="인덱스로 계산된 좌표로 이동 후 개방",
        needs_vlm=False,
        destinations=("target_pose",),
    ),
}

assert tuple(CATALOG) == SKILL_IDS, "CATALOG 와 SKILL_IDS 가 어긋났다"


@dataclass
class GateRecord:
    """The measurement that says a policy may be deployed.
    정책을 배포해도 된다고 말하는 계측 결과.

    Required, not optional. A checkpoint without one is a checkpoint nobody
    measured, and the deploy API must refuse it — otherwise "training finished"
    silently becomes "it works".
    선택이 아니라 필수다. 이게 없는 체크포인트는 아무도 재보지 않은 체크포인트이고
    배포 API 는 그것을 거부해야 한다. 그러지 않으면 "학습이 끝났다"가 조용히
    "된다"로 바뀐다.
    """

    metric: str
    n: int
    runs: int
    value: float
    ci95: list[float]
    value_range: list[float]
    measured_at: str = ""
    exp_log_git_rev: str = ""

    def problems(self) -> list[str]:
        """Every reason this gate record cannot authorise a deployment.
        이 게이트 기록이 배포를 승인할 수 없는 이유 전부."""
        out: list[str] = []
        if self.metric != "rollout_success":
            out.append(f"metric 은 'rollout_success' 여야 한다: {self.metric!r}")
        if self.n < GATE_MIN_N:
            out.append(f"n={self.n} < {GATE_MIN_N} — 이 표본으로는 게이트를 판정할 수 없다")
        if self.runs < GATE_MIN_RUNS:
            out.append(
                f"runs={self.runs} < {GATE_MIN_RUNS} — 학습 간 분산이 25%p 라 "
                "단일 실행으로 체크포인트를 대표할 수 없다"
            )
        if self.value <= ROLLOUT_GATE:
            out.append(f"성공률 {self.value:.3f} <= 게이트 {ROLLOUT_GATE:.2f}")
        if len(self.ci95) != 2 or self.ci95[0] > self.ci95[1]:
            out.append(f"ci95 형식이 잘못됐다: {self.ci95}")
        elif self.ci95[0] <= ROLLOUT_GATE:
            out.append(
                f"95% 신뢰구간 하한 {self.ci95[0]:.3f} 이 게이트 {ROLLOUT_GATE:.2f} 아래다 "
                "— 통과를 확정할 수 없다"
            )
        return out

    def passed(self) -> bool:
        return not self.problems()


@dataclass
class PolicyRef:
    """Where the behaviour comes from.
    행동이 어디에서 나오는가."""

    kind: str
    contract_version: str = CONTRACT_VERSION
    ckpt_uri: str = ""
    trained_on: str = ""
    action_space: str = "next_joint_position_and_gap"
    gate: GateRecord | None = None


@dataclass
class PostAction:
    """The scripted half of a skill — everything after the object is held.
    스킬의 스크립트 절반. 물체를 쥔 뒤의 전부."""

    name: str
    destination: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillEntry:
    """One row of the platform's skill registry.
    플랫폼 스킬 레지스트리의 한 행."""

    skill_id: str
    version: str
    tier: str
    status: str
    policy: PolicyRef
    post_actions: list[PostAction]
    object_size_mm: list[float]
    workspace_m: dict[str, list[float]]
    robot: dict[str, Any]
    created_by: str
    created_at: str
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> SkillSpec:
        return CATALOG[self.skill_id]


def _rect_contains(outer: dict[str, list[float]], inner: dict[str, list[float]]) -> bool:
    for axis in ("x", "y"):
        lo_o, hi_o = outer[axis]
        lo_i, hi_i = inner[axis]
        if lo_i < lo_o or hi_i > hi_o:
            return False
    return True


def validate_entry(entry: SkillEntry, cfg: dict[str, Any]) -> list[str]:
    """Return every reason this entry must not be deployed. Empty means deployable.
    이 항목을 배포하면 안 되는 이유를 전부 반환한다. 비어 있으면 배포 가능하다."""
    problems: list[str] = []

    if entry.skill_id not in SKILL_IDS:
        problems.append(f"알 수 없는 skill_id {entry.skill_id!r}. 허용: {SKILL_IDS}")
        return problems

    spec = entry.spec

    if entry.tier not in TIERS:
        problems.append(f"tier 는 {TIERS} 중 하나여야 한다: {entry.tier!r}")
    if entry.status not in STATUSES:
        problems.append(f"status 는 {STATUSES} 중 하나여야 한다: {entry.status!r}")
    elif entry.status != "deployed":
        problems.append(f"status={entry.status!r} — 아직 배포 단계가 아니다")
    if entry.tier == "demo" and not entry.notes.get("persona"):
        problems.append("demo 스킬에는 notes.persona 가 필요하다 — 누구를 겨냥한 동작인지 없이는 시연에 못 쓴다")

    if entry.policy.kind not in POLICY_KINDS:
        problems.append(f"policy.kind 는 {POLICY_KINDS} 중 하나여야 한다: {entry.policy.kind!r}")
    if entry.policy.contract_version != CONTRACT_VERSION:
        problems.append(
            f"contract_version {entry.policy.contract_version!r} != {CONTRACT_VERSION!r} "
            "— 다른 규격으로 학습된 정책이다"
        )
    if entry.policy.action_space not in ACTION_SPACES:
        problems.append(
            f"action_space {entry.policy.action_space!r} 이 {ACTION_SPACES} 에 없다 — "
            "계약 0.3은 state[t+1] 관절 위치 5개와 전체 gap만 허용한다"
        )
    if entry.policy.kind == "learned":
        if not entry.policy.ckpt_uri:
            problems.append("learned 정책인데 ckpt_uri 가 비어 있다")
        if entry.policy.gate is None:
            problems.append("learned 정책에는 gate 가 필수다 — 재보지 않은 체크포인트는 배포 불가")
        else:
            problems += [f"gate: {p}" for p in entry.policy.gate.problems()]

    if not entry.post_actions:
        problems.append("post_actions 가 비어 있다 — 물체를 쥔 뒤 무엇을 할지가 없다")
    for pa in entry.post_actions:
        if pa.destination not in DESTINATIONS:
            problems.append(f"알 수 없는 destination {pa.destination!r}. 허용: {DESTINATIONS}")
        elif pa.destination not in spec.destinations:
            problems.append(
                f"{entry.skill_id} 는 destination {spec.destinations} 만 쓴다: {pa.destination!r}"
            )

    lo, hi = OBJECT_SIZE_MM
    if len(entry.object_size_mm) != 2:
        problems.append(f"object_size_mm 은 [min, max] 두 값이어야 한다: {entry.object_size_mm}")
    elif entry.object_size_mm[0] < lo or entry.object_size_mm[1] > hi:
        problems.append(
            f"object_size_mm {entry.object_size_mm} 이 그리퍼 범위 [{lo}, {hi}] 를 벗어난다"
        )

    reach = cfg.get("workspace", {}).get("reachable_rect_m")
    if reach is None:
        problems.append("configs 에 workspace.reachable_rect_m 이 없다 — 작업영역을 검증할 수 없다")
    elif not _rect_contains(reach, entry.workspace_m):
        problems.append(
            f"workspace_m {entry.workspace_m} 이 실측 도달영역 {reach} 을 벗어난다"
        )

    dof = int(cfg.get("robot", {}).get("dof", -1))
    if entry.robot.get("dof") != dof:
        problems.append(f"robot.dof {entry.robot.get('dof')} 이 configs 의 {dof} 과 다르다")

    return problems


def shared_policy_report(entries: list[SkillEntry]) -> str:
    """Say plainly how many distinct learned policies these skills actually use.
    이 스킬들이 실제로 몇 개의 학습 정책을 쓰는지 사실대로 말한다.

    The product claim is "five behaviours learned". Under plan A that is one
    policy and five scripted endings, and saying otherwise in a demo would be a
    claim we cannot support. This function exists so the honest number is always
    one call away.
    제품 주장은 "다섯 행동을 학습"이다. 안 A 에서 그것은 정책 하나와 스크립트
    다섯이고, 시연에서 달리 말하면 뒷받침할 수 없는 주장이 된다. 정직한 숫자가
    항상 호출 한 번 거리에 있도록 이 함수를 둔다.
    """
    ckpts: dict[str, list[str]] = {}
    for e in entries:
        if e.policy.kind == "learned" and e.policy.ckpt_uri:
            ckpts.setdefault(e.policy.ckpt_uri, []).append(e.skill_id)

    by_tier: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for e in entries:
        by_tier[e.tier] = by_tier.get(e.tier, 0) + 1
        by_status[e.status] = by_status.get(e.status, 0) + 1

    lines = [
        f"스킬 {len(entries)}개 · 서로 다른 학습 정책 {len(ckpts)}개",
        "  층: " + ", ".join(f"{k} {v}" for k, v in sorted(by_tier.items())),
        "  상태: " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())),
    ]
    n_deployed = by_status.get("deployed", 0)
    if n_deployed < len(entries):
        lines.append(
            f"  → 실제로 동작하는 스킬은 {n_deployed}개다. "
            "계획된 개수를 구현된 개수처럼 말하지 마라."
        )
    for uri, ids in sorted(ckpts.items()):
        lines.append(f"  {uri}  ←  {', '.join(sorted(ids))}")
    if len(ckpts) == 1 and len(entries) > 1:
        lines.append(
            "  → 안 A 상태다. 발표에서 '5개를 학습시켰다'가 아니라 "
            "'파지 정책 1개를 5개 작업이 공유한다'로 말해야 한다."
        )
    elif len(ckpts) == len(entries) and len(entries) > 1:
        lines.append("  → 안 B 상태다. 스킬마다 독립 정책이다.")
    return "\n".join(lines)


# ---- VLM 출력 --------------------------------------------------------------


@dataclass
class VLMDecision:
    """What the VLM is allowed to return. Nothing else parses.
    VLM 이 반환해도 되는 것. 그 외에는 파싱되지 않는다."""

    skill_id: str
    target_description_ko: str
    target_bbox_norm: list[float]
    destination: str | None
    confidence: float
    abstain: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_vlm_decision(raw: str | dict[str, Any]) -> VLMDecision:
    """Parse and validate one VLM output, refusing anything outside the enums.
    VLM 출력 하나를 파싱·검증한다. enum 밖의 것은 거부한다.

    Refusing is the point. A free-generating model will produce a plausible sixth
    skill name, and a downstream `if` that falls through silently is how a robot
    ends up doing nothing while the UI says it is working.
    거부가 핵심이다. 자유 생성 모델은 그럴듯한 여섯 번째 스킬 이름을 만들어내고,
    조용히 빠져나가는 하류의 `if` 문이 바로 UI 는 동작 중이라고 표시하는데 로봇은
    아무것도 안 하는 상태를 만든다.
    """
    data = json.loads(raw) if isinstance(raw, str) else dict(raw)

    skill_id = data.get("skill_id")
    if skill_id not in SKILL_IDS:
        raise SkillError(f"VLM 이 알 수 없는 skill_id 를 냈다: {skill_id!r}. 허용: {SKILL_IDS}")

    dest = data.get("destination")
    if dest is not None:
        if dest not in DESTINATIONS:
            raise SkillError(f"알 수 없는 destination {dest!r}. 허용: {DESTINATIONS}")
        allowed = CATALOG[skill_id].destinations
        if dest not in allowed:
            raise SkillError(f"{skill_id} 는 {allowed} 만 쓴다: {dest!r}")
    elif CATALOG[skill_id].needs_vlm:
        raise SkillError(f"{skill_id} 는 destination 판단이 필요한데 null 이다")

    bbox = data.get("target", {}).get("bbox_norm") or []
    if len(bbox) != 4 or not all(0.0 <= float(v) <= 1.0 for v in bbox):
        raise SkillError(f"bbox_norm 은 [0,1] 범위의 4개 값이어야 한다: {bbox}")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise SkillError(f"bbox_norm 의 좌상단이 우하단보다 크거나 같다: {bbox}")

    conf = float(data.get("confidence", 0.0))
    if not 0.0 <= conf <= 1.0:
        raise SkillError(f"confidence 는 [0,1] 이어야 한다: {conf}")

    return VLMDecision(
        skill_id=skill_id,
        target_description_ko=str(data.get("target", {}).get("description_ko", "")),
        target_bbox_norm=[float(v) for v in bbox],
        destination=dest,
        confidence=conf,
        abstain=bool(data.get("abstain", False)),
    )


def should_execute(decision: VLMDecision) -> tuple[bool, str]:
    """Whether this decision may move the arm, and why not when it may not.
    이 결정이 팔을 움직여도 되는가, 안 된다면 왜 안 되는가."""
    if decision.abstain:
        return False, "VLM 이 기권했다. 사용자에게 되묻는다."
    if decision.confidence < VLM_CONFIDENCE_FLOOR:
        return False, (
            f"confidence {decision.confidence:.2f} < {VLM_CONFIDENCE_FLOOR:.2f} "
            "— 실행하지 않고 사용자에게 되묻는다."
        )
    return True, ""


# ---- 입출력 ----------------------------------------------------------------


def entry_from_dict(d: dict[str, Any]) -> SkillEntry:
    """Rebuild an entry from its stored JSON form.
    저장된 JSON 형태에서 항목을 복원한다."""
    pol = dict(d["policy"])
    gate = pol.pop("gate", None)
    return SkillEntry(
        skill_id=d["skill_id"],
        version=d["version"],
        tier=d.get("tier", "tutorial"),
        status=d.get("status", "planned"),
        policy=PolicyRef(**pol, gate=GateRecord(**gate) if gate else None),
        post_actions=[PostAction(**pa) for pa in d.get("post_actions", [])],
        object_size_mm=list(d.get("object_size_mm", [])),
        workspace_m={k: list(v) for k, v in d.get("workspace_m", {}).items()},
        robot=dict(d.get("robot", {})),
        created_by=d.get("created_by", ""),
        created_at=d.get("created_at", ""),
        notes=dict(d.get("notes", {})),
    )


def load_registry(path: Path) -> list[SkillEntry]:
    """Read every *.json under a registry directory.
    레지스트리 디렉터리의 모든 *.json 을 읽는다."""
    return [
        entry_from_dict(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(Path(path).glob("*.json"))
    ]


def write_entry(entry: SkillEntry, out_dir: Path) -> Path:
    """Write one entry, refusing to write an entry that cannot be deployed.
    항목 하나를 쓴다. 배포 불가한 항목은 쓰지 않는다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{entry.skill_id}.json"
    path.write_text(json.dumps(asdict(entry), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
