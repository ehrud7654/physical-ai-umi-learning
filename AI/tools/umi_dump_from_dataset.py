"""계약 데이터셋 → UMI raw. 진짜 렌더 이미지를 UMI 경로에 넣는 방법.
Contract dataset to UMI raw — how real rendered images get into the UMI path.

project-task 층 2. `state` 를 관절각으로 되돌리고 FK 로 EEF pose 를 만든다.
**이미지는 바이트 그대로 재사용한다** — 다시 렌더하지 않으므로 렌더 비결정성
(L37·L39)이 끼어들 여지가 없다.

## 어느 데이터셋을 읽는가 — 남의 것을 읽지 않는다

`datasets/sim_pick_*` 는 시뮬·정책 대화 단독 소유다. 읽기만 해도 **내 임계경로가
남이 움직이는 물건에 걸린다** — `sim_pick_v4` 가 실제로 그렇게 사라졌다.
그래서 `data/collect.py` 로 **내 이름의 데이터셋을 직접 수집해서** 그것을 읽는다.

## ⚠️ 왕복이 원본과 완전히 같아지지 않는다 — action 의 의미가 다르다

| | 시뮬 수집 | UMI 경로 |
|---|---|---|
| `state[t]` | 물리 적분 후 실제 관절각 (`qpos`) | `IK(pose[t])` — pose 가 그 qpos 의 FK 이므로 거의 같다 |
| `action[t]` | **`ctrl[t]` — 낸 명령** | **`q[t+1]` — 다음 스텝의 관절 배치** |

시뮬의 `ctrl` 은 목표 관절각이고 서보가 못 따라간 만큼 `state` 와 벌어져 있다.
UMI 시연에는 로봇이 없으므로 "명령"이 존재하지 않고, 다음 관절 배치가 그 대응물이다.

**이건 결함이 아니라 두 데이터 출처의 구조적 차이다** (LIMITS L38 이 `state` 에
대해 말한 것과 같은 종류). 왕복 차이를 재면 `state` 는 거의 0, `action` 은
뚜렷하게 다르게 나와야 한다. **0 으로 나오면 오히려 무언가 잘못된 것이다.**

    # [로컬]
    cd AI && python tools/umi_dump_from_dataset.py \\
        --data datasets/umi_src_sim_v1 --out out/umi_raw_from_sim_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from contract.episode import read_episode  # noqa: E402
from paths import DEFAULT_CONFIG, DEFAULT_SCENE, OUT_DIR  # noqa: E402
from sim.mujoco.build_scene import build_model, load_config  # noqa: E402
from umi.convert import denormalize_joints  # noqa: E402
from umi.raw import FRAME_ROBOT_BASE, RawEpisode, RawMeta, write_raw  # noqa: E402
from tools.umi_mujoco import (  # noqa: E402
    eef_pose_from_joints,
    gap_from_angle,
    joint_ranges,
)

SOURCE_TAG = "sim_fk"
"""`umi.raw.REAL_SOURCES` 에 없으므로 `calibration_id` 가 비어도 된다.
합성이 아니라 시뮬 실측 궤적이지만, 카메라도 hand-eye 도 실물이 아니다."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="계약 에피소드 디렉터리")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "umi_raw_from_sim_v1")
    ap.add_argument("--image-lag-ms", type=float, default=0.0,
                    help="RGB 스트림의 계통적 지연. 시뮬은 구조적으로 0 이다")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = ap.parse_args()

    srcs = sorted(args.data.glob("*.npz"))
    if not srcs:
        print(f"에피소드가 없다: {args.data}")
        return 1
    if args.out.exists() and sorted(args.out.glob("*.raw.npz")):
        print(f"이미 raw 가 있다: {args.out}\n  이름을 재사용하지 않는다. --out 에 다음 번호를 줘라")
        return 1

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    data = mujoco.MjData(model)
    ranges = joint_ranges(cfg)
    pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    curve = cfg["grasp"]["gap_curve"]
    lag = args.image_lag_ms / 1000.0

    total = 0
    print(f"{args.data} · {len(srcs)}편  →  {args.out}")
    print(f"이미지는 바이트 그대로 재사용한다 (재렌더 없음) · RGB 지연 {args.image_lag_ms}ms\n")

    for path in srcs:
        ep = read_episode(path)
        q = np.stack([denormalize_joints(row, ranges) for row in ep.state])  # (T, 6) rad
        poses = [eef_pose_from_joints(model, data, row, pinch) for row in q]
        eef_pos = np.stack([p for p, _ in poses]).astype(np.float64)
        eef_quat = np.stack([qq for _, qq in poses]).astype(np.float64)
        gaps = np.array([gap_from_angle(row[5], curve) for row in q], dtype=np.float64)

        # CHW → HWC. raw 스키마는 카메라가 낸 형태로 둔다.
        images = {
            cam: np.ascontiguousarray(arr.transpose(0, 2, 3, 1)).astype(np.uint8)
            for cam, arr in ep.images.items()
        }
        stamps = {
            cam: (np.asarray(ep.state_timestamp, dtype=np.float64) + lag).copy()
            for cam in images
        }

        raw = RawEpisode(
            meta=RawMeta(
                recording_id=ep.meta.episode_id,
                skill_id=ep.meta.skill_id,
                source=SOURCE_TAG,
                frame=FRAME_ROBOT_BASE,
                n_steps=ep.meta.n_steps,
                pose_rate_hz=float(ep.meta.control_rate_hz),
                cameras=sorted(images),
                calibration_id="",
                recorded_by="tools/umi_dump_from_dataset.py",
                notes={
                    **ep.meta.notes,
                    "task": ep.meta.task,
                    "success": bool(ep.meta.success),
                    "dumped_from_dataset": str(args.data),
                    "dumped_from_contract_version": ep.meta.contract_version,
                    "images_reused_verbatim": True,
                    # 원본의 action 은 ctrl(명령)이고 UMI 경로는 q[t+1] 이다.
                    # 왕복 차이를 읽을 때 이 필드를 근거로 삼는다.
                    "source_action_semantics": "sim_ctrl",
                },
            ),
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            gripper_gap_m=gaps,
            # 시뮬 FK 이므로 전 프레임 D. 실기록의 T·X 결측은 여기서 발생하지 않는다.
            gripper_status=np.full(ep.meta.n_steps, "D", dtype="<U1"),
            pose_timestamp=np.asarray(ep.state_timestamp, dtype=np.float64).copy(),
            images=images,
            image_timestamp=stamps,
        )
        out = write_raw(raw, args.out)
        total += out.stat().st_size
        print(f"  {out.name}  {ep.meta.n_steps}스텝  {out.stat().st_size / 1e6:5.2f}MB")

    print(f"\n{len(srcs)}편 · 합계 {total / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
