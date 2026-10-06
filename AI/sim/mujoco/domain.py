"""Domain randomisation, applied after compile so the official MJCF is untouched.
도메인 랜덤화. 컴파일 이후에 적용해서 공식 MJCF 는 건드리지 않는다.

Why this exists.
왜 있는가.

We have to decide whether a policy trained in simulation is worth anything as a
head start for the real UMI demonstrations (D-AI-18). That question cannot be
answered without the real arm, which is four weeks away. What CAN be measured
today is the weaker question it contains: when the domain moves, how fast does a
sim-trained policy fall over? If it collapses under a friction and lighting
change inside the same simulator, it will certainly collapse on hardware.
시뮬에서 학습한 정책이 실물 UMI 시연의 출발점으로 값어치가 있는지를 정해야
한다(D-AI-18). 그 질문은 실물 없이는 답할 수 없고 실물은 4주 뒤다. 오늘 잴 수
있는 것은 그 안에 든 약한 질문이다 — 도메인이 움직이면 시뮬 학습 정책은 얼마나
빨리 무너지는가. 같은 시뮬 안에서 마찰과 조명만 바꿔도 무너진다면 실물에서는
확실히 무너진다.

⚠️ 이 계측은 **한 방향으로만** 결론을 낸다. 여기서 무너지면 실물에서도 무너진다는
   것은 말할 수 있지만, 여기서 버틴다고 실물을 보장하지 않는다. 시뮬 내부 A→B 는
   실제 sim2real 갭의 하한이다.

What is perturbed, and what that stands in for:
무엇을 흔들고 그것이 무엇을 대신하는가:

  마찰      조립 오차·표면 상태·마모     물리
  질량      부품 편차·재질              물리
  조명      촬영 환경                   인식
  카메라 외인 마운트 조립 오차·재장착     인식
  색        조명 색온도·재질 편차        인식

Physics and perception are separated on purpose. A scripted policy reads
privileged state and never looks at an image, so its collapse isolates the
physics half; a vision policy's collapse includes both. The difference between
them is the perception cost.
물리와 인식을 일부러 나눴다. 스크립트 정책은 특권 정보를 읽고 이미지를 보지
않으므로 그 붕괴는 물리 쪽만 격리해서 보여준다. 시각 정책의 붕괴는 둘 다
포함한다. 둘의 차이가 인식이 치르는 비용이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class DomainSpec:
    """One domain condition. All values are half-widths of a uniform range.
    도메인 조건 하나. 모든 값은 균등분포의 반폭이다.

    Zero everywhere means "the nominal model", which is condition A. Condition A
    is not a no-op wrapper: it must go through the same code path as B so that a
    difference between them cannot come from the wrapper itself.
    전부 0 이면 "공칭 모델"이고 그것이 조건 A 다. 조건 A 는 빈 껍데기가 아니다 —
    B 와 같은 코드 경로를 지나야 둘의 차이가 래퍼 자체에서 나오지 않는다.
    """

    name: str
    friction_scale: float = 0.0   # 미끄럼 마찰 배율 ±비율
    mass_scale: float = 0.0       # 대상물 질량 배율 ±비율
    light_pos_m: float = 0.0      # 광원 위치 ±m
    light_diffuse: float = 0.0    # 광원 확산광 ±절대값
    cam_pos_m: float = 0.0        # 카메라 위치 ±m
    cam_rot_deg: float = 0.0      # 카메라 자세 ±도
    rgba_jitter: float = 0.0      # 대상물·테이블 색 ±절대값

    def is_nominal(self) -> bool:
        """True if this condition perturbs nothing.
        아무것도 흔들지 않는 조건인가."""
        return all(
            float(v) == 0.0
            for k, v in asdict(self).items()
            if k != "name"
        )


# Fixed before any transfer number was produced. These magnitudes are a claim
# about how much a real setup differs from a nominal one, and that claim is
# 🟡 설계 판단 — not measured. Widening them after seeing a result would make
# the gate meaningless.
# 어떤 전이 수치도 나오기 전에 확정했다. 이 크기는 실제 환경이 공칭과 얼마나
# 다른가에 대한 주장이고 그 주장은 🟡 설계 판단이다 — 계측된 값이 아니다.
# 결과를 보고 이 폭을 넓히면 게이트가 무의미해진다.
CONDITION_A = DomainSpec(name="A")
CONDITION_B = DomainSpec(
    name="B",
    friction_scale=0.30,
    mass_scale=0.20,
    light_pos_m=0.10,
    light_diffuse=0.15,
    cam_pos_m=0.005,
    cam_rot_deg=2.0,
    rgba_jitter=0.15,
)

PRESETS: dict[str, DomainSpec] = {"A": CONDITION_A, "B": CONDITION_B}


class DomainRandomizer:
    """Snapshot the nominal model once, then re-perturb it every reset.
    공칭 모델을 한 번 스냅샷하고, 리셋마다 거기서 다시 흔든다.

    Perturbing the already-perturbed model would compound across episodes and
    the condition would drift into something nobody chose. Restoring from the
    snapshot first is what makes episode N and episode N+1 the same condition.
    이미 흔들린 모델을 또 흔들면 에피소드를 거치며 누적되어, 아무도 고르지 않은
    조건으로 떠내려간다. 매번 스냅샷에서 복원해야 에피소드 N 과 N+1 이 같은
    조건이 된다.
    """

    def __init__(self, spec: DomainSpec, *, object_body: str = "target_object") -> None:
        self.spec = spec
        self._object_body = object_body
        self._nominal: dict[str, np.ndarray] = {}
        self._obj_body_id = -1
        self._n_lights = 0
        self._bound = False

    # ---- lifecycle ------------------------------------------------------

    def bind(self, model: mujoco.MjModel) -> None:
        """Record the nominal values this randomiser will perturb.
        이 랜덤화기가 흔들 값들의 공칭치를 기록한다."""
        self._nominal = {
            "geom_friction": model.geom_friction.copy(),
            "geom_rgba": model.geom_rgba.copy(),
            "body_mass": model.body_mass.copy(),
            "light_pos": model.light_pos.copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "cam_pos": model.cam_pos.copy(),
            "cam_quat": model.cam_quat.copy(),
        }
        self._obj_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._object_body
        )
        if self._obj_body_id < 0:
            raise ValueError(f"body {self._object_body!r} 이 모델에 없다")
        self._n_lights = int(model.nlight)
        self._bound = True

    def apply(
        self, model: mujoco.MjModel, data: mujoco.MjData, seed: int | None = None
    ) -> None:
        """Restore the nominal model, then perturb it for this episode.
        공칭 모델로 되돌린 뒤 이번 에피소드용으로 흔든다.

        The seed is offset from the caller's so the object placement, which uses
        the caller's seed directly, stays identical between condition A and B.
        Comparing A and B across different object placements would not be a
        comparison of conditions.
        시드는 호출자의 것에서 어긋나게 준다. 물체 배치는 호출자 시드를 그대로
        쓰므로 조건 A 와 B 에서 동일하게 유지된다. 물체 배치가 다른 A 와 B 를
        비교하는 것은 조건의 비교가 아니다.
        """
        if not self._bound:
            raise RuntimeError("bind(model) 을 먼저 호출해야 한다")

        for key, arr in self._nominal.items():
            getattr(model, key)[:] = arr

        if not self.spec.is_nominal():
            rng = np.random.default_rng(None if seed is None else seed + 10_000_000)
            self._perturb_physics(model, rng)
            self._perturb_appearance(model, rng)

        # Mass changes invalidate derived inertial constants; recompute them.
        # 질량을 바꾸면 파생 관성 상수가 무효가 된다. 다시 계산한다.
        mujoco.mj_setConst(model, data)

    # ---- perturbations --------------------------------------------------

    def _perturb_physics(self, model: mujoco.MjModel, rng: np.random.Generator) -> None:
        s = self.spec
        if s.friction_scale > 0.0:
            n = model.ngeom
            scale = rng.uniform(1.0 - s.friction_scale, 1.0 + s.friction_scale, size=n)
            model.geom_friction[:, 0] = self._nominal["geom_friction"][:, 0] * scale
        if s.mass_scale > 0.0:
            f = rng.uniform(1.0 - s.mass_scale, 1.0 + s.mass_scale)
            model.body_mass[self._obj_body_id] = (
                self._nominal["body_mass"][self._obj_body_id] * f
            )

    def _perturb_appearance(self, model: mujoco.MjModel, rng: np.random.Generator) -> None:
        s = self.spec
        if self._n_lights > 0:
            if s.light_pos_m > 0.0:
                model.light_pos[:] = self._nominal["light_pos"] + rng.uniform(
                    -s.light_pos_m, s.light_pos_m, size=self._nominal["light_pos"].shape
                )
            if s.light_diffuse > 0.0:
                model.light_diffuse[:] = np.clip(
                    self._nominal["light_diffuse"]
                    + rng.uniform(
                        -s.light_diffuse,
                        s.light_diffuse,
                        size=self._nominal["light_diffuse"].shape,
                    ),
                    0.0,
                    1.0,
                )
        if s.rgba_jitter > 0.0:
            rgba = self._nominal["geom_rgba"].copy()
            rgba[:, :3] = np.clip(
                rgba[:, :3]
                + rng.uniform(-s.rgba_jitter, s.rgba_jitter, size=rgba[:, :3].shape),
                0.0,
                1.0,
            )
            model.geom_rgba[:] = rgba
        if model.ncam > 0:
            if s.cam_pos_m > 0.0:
                model.cam_pos[:] = self._nominal["cam_pos"] + rng.uniform(
                    -s.cam_pos_m, s.cam_pos_m, size=self._nominal["cam_pos"].shape
                )
            if s.cam_rot_deg > 0.0:
                for i in range(model.ncam):
                    model.cam_quat[i] = self._random_tilt(
                        self._nominal["cam_quat"][i], s.cam_rot_deg, rng
                    )

    @staticmethod
    def _random_tilt(
        quat: np.ndarray, max_deg: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Rotate a quaternion by a small random amount about a random axis.
        쿼터니언을 임의 축 둘레로 조금 회전시킨다."""
        axis = rng.normal(size=3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return quat.copy()
        axis = axis / norm
        angle = float(rng.uniform(-np.deg2rad(max_deg), np.deg2rad(max_deg)))
        delta = np.zeros(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(delta, axis, angle)
        out = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(out, delta, np.asarray(quat, dtype=np.float64))
        return out

    # ---- reporting ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """What this randomiser actually does to this model, for the log.
        이 랜덤화기가 이 모델에 실제로 하는 일. 로그에 남길 것."""
        info: dict[str, Any] = {"spec": asdict(self.spec), "n_lights": self._n_lights}
        if self._n_lights == 0 and (
            self.spec.light_pos_m > 0.0 or self.spec.light_diffuse > 0.0
        ):
            info["warning"] = (
                "모델에 광원이 0개다 (MuJoCo 기본 헤드라이트 사용). "
                "조명 랜덤화는 실제로 아무 효과가 없다 — 인식 쪽 외란은 "
                "카메라 외인과 색만 반영된 것으로 읽어야 한다."
            )
        return info
