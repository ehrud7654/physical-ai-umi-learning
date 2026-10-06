"""Synthetic UMI raw recordings — the pass-through input, built without touching sim datasets.
합성 UMI raw 기록. 시뮬 데이터셋을 건드리지 않고 만드는 관통 입력.

project-task 층 1. 왜 자립 생성인가: `datasets/sim_pick_*` 는 시뮬·정책 대화
단독 소유이고 지금도 쓰이고 있다. 읽기만 해도 **내 임계경로가 남이 움직이는
물건에 걸린다** — `sim_pick_v4` 가 실제로 그렇게 사라졌다.

변환기가 이미지에 하는 일은 최근접 정렬 + CHW 전치뿐이고 기하는 안 만진다.
그래서 변환 정확성 검증에 실제 렌더 이미지가 필요하지 않다.

## 이미지에 프레임 인덱스를 픽셀로 심는다

off-by-one 과 카메라 뒤바뀜은 **조용히** 실패한다. 학습 손실은 잘 떨어지고
실물에서만 틀린다. 그래서 각 프레임의 좌상단에 프레임 번호와 카메라 코드를
비트로 인코딩한다. 변환이 끝난 뒤 픽셀에서 되읽어 **어느 프레임이 어느 스텝에
붙었는지 직접 대조**할 수 있다 — `tools/convert_umi.py --verify-image-index`.

배경은 무작위 노이즈가 아니라 기울기다. 노이즈는 압축이 전혀 안 돼서
에피소드당 27MB 가 그대로 디스크로 간다.

    # [로컬]
    cd AI && python tools/make_umi_synth.py --episodes 20 --steps 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from contract.episode import IMAGE_SHAPE  # noqa: E402
from paths import DEFAULT_CONFIG, DEFAULT_SCENE, OUT_DIR  # noqa: E402
from sim.mujoco.build_scene import build_model, load_config  # noqa: E402
from umi.raw import FRAME_ROBOT_BASE, RawEpisode, RawMeta, write_raw  # noqa: E402
from tools.umi_mujoco import (  # noqa: E402
    eef_pose_from_joints,
    gap_from_angle,
    joint_ranges,
    smooth_joint_trajectory,
)

HEIGHT, WIDTH = IMAGE_SHAPE[1], IMAGE_SHAPE[2]
INDEX_BITS = 12
CAMERA_BITS = 4
BIT_W, BIT_GAP = 3, 4
INDEX_ROWS = slice(2, 6)
CAMERA_ROWS = slice(8, 12)


def _write_bits(frame: np.ndarray, rows: slice, value: int, n_bits: int) -> None:
    for b in range(n_bits):
        x0 = 2 + b * BIT_GAP
        frame[rows, x0 : x0 + BIT_W, :] = 255 if (value >> b) & 1 else 0


def _read_bits(frame: np.ndarray, rows: slice, n_bits: int) -> int:
    out = 0
    for b in range(n_bits):
        x0 = 2 + b * BIT_GAP
        if float(frame[rows, x0 : x0 + BIT_W, :].mean()) > 127.0:
            out |= 1 << b
    return out


def make_frame(idx: int, cam_code: int) -> np.ndarray:
    """One HWC uint8 frame carrying its own frame number and camera code.
    자기 프레임 번호와 카메라 코드를 담은 HWC uint8 프레임 하나."""
    ramp_y = np.linspace(0, 255, HEIGHT, dtype=np.float64)[:, None]
    ramp_x = np.linspace(0, 255, WIDTH, dtype=np.float64)[None, :]
    base = (0.5 * ramp_y + 0.5 * ramp_x + 7.0 * idx) % 256.0
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = base.astype(np.uint8)
    frame[..., 1] = ((base + 85.0 + 11.0 * cam_code) % 256.0).astype(np.uint8)
    frame[..., 2] = ((base + 170.0) % 256.0).astype(np.uint8)
    _write_bits(frame, INDEX_ROWS, idx, INDEX_BITS)
    _write_bits(frame, CAMERA_ROWS, cam_code, CAMERA_BITS)
    return frame


def decode_frame(image: np.ndarray) -> tuple[int, int]:
    """(frame index, camera code) read back out of the pixels. Accepts CHW or HWC.
    픽셀에서 되읽은 (프레임 번호, 카메라 코드). CHW·HWC 둘 다 받는다."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    return _read_bits(arr, INDEX_ROWS, INDEX_BITS), _read_bits(arr, CAMERA_ROWS, CAMERA_BITS)


def gripper_profile(n: int, lo: float, hi: float) -> np.ndarray:
    """Open, then close, then hold — the shape a pick has, for the gripper channel only.
    열림 → 닫힘 → 유지. 그리퍼 채널만 파지 동작의 형태를 갖게 한다.

    The arm channels stay sinusoidal. This is not a demonstration trajectory and is
    not claimed to be one -- what it has to exercise is the gap-curve inversion
    across its measured range, and a monotone close does that.
    팔 채널은 사인 그대로다. 이건 시연 궤적이 아니고 그렇다고 주장하지도 않는다.
    이게 밟아야 하는 것은 gap 곡선 역변환의 실측 범위 전체이고, 단조 닫힘이 그걸 한다.
    """
    open_frac, close_frac = 0.55, 0.25
    n_open = int(n * open_frac)
    n_close = max(1, int(n * close_frac))
    n_hold = n - n_open - n_close
    return np.concatenate([
        np.full(n_open, hi * 0.9),
        np.linspace(hi * 0.9, lo + 0.02, n_close),
        np.full(max(0, n_hold), lo + 0.02),
    ])[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-rate-hz", type=float, default=30.0)
    ap.add_argument("--image-rate-hz", type=float, default=30.0,
                    help="pose 보다 느리게 주면 최근접 정렬만으로 남는 오차가 커진다")
    ap.add_argument("--image-lag-ms", type=float, default=8.0,
                    help="RGB 스트림의 계통적 지연. ARCore 는 pose 와 RGB 가 별도 스트림이다")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "umi_raw_synth_v1")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    args = ap.parse_args()

    existing = sorted(args.out.glob("*.raw.npz")) if args.out.exists() else []
    if existing:
        print(f"이미 {len(existing)}편이 있다: {args.out}\n"
              "  이름을 재사용하지 않는다. --out 으로 다음 번호를 줘라 "
              "(소유권 규칙: 기존 이름 재사용 금지)")
        return 1

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    data = mujoco.MjData(model)
    ranges = joint_ranges(cfg)
    pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
    curve = cfg["grasp"]["gap_curve"]
    cams = [k for k, v in cfg["cameras"].items() if isinstance(v, dict)]
    cam_code = {name: i for i, name in enumerate(sorted(cams))}

    rng = np.random.default_rng(args.seed)
    n = args.steps
    lag_s = args.image_lag_ms / 1000.0
    total_bytes = 0

    print(f"카메라 {sorted(cams)} · pose {args.pose_rate_hz}Hz · "
          f"image {args.image_rate_hz}Hz · 지연 {args.image_lag_ms}ms")
    print(f"출력 {args.out}\n")

    for e in range(args.episodes):
        q = smooth_joint_trajectory(ranges, n, rng)
        q[:, 5] = gripper_profile(n, ranges[5, 0], ranges[5, 1])

        poses = [eef_pose_from_joints(model, data, row, pinch) for row in q]
        eef_pos = np.stack([p for p, _ in poses]).astype(np.float64)
        eef_quat = np.stack([qq for _, qq in poses]).astype(np.float64)
        gaps = np.array([gap_from_angle(row[5], curve) for row in q], dtype=np.float64)

        t0 = 1_000_000.0 + e * 1000.0
        pose_ts = t0 + np.arange(n, dtype=np.float64) / args.pose_rate_hz
        duration = pose_ts[-1] - pose_ts[0]
        n_frames = max(2, int(round(duration * args.image_rate_hz)) + 1)
        img_ts = t0 + lag_s + np.arange(n_frames, dtype=np.float64) / args.image_rate_hz

        images, stamps = {}, {}
        for name in cams:
            code = cam_code[name]
            images[name] = np.stack([make_frame(i, code) for i in range(n_frames)])
            stamps[name] = img_ts.copy()

        raw = RawEpisode(
            meta=RawMeta(
                recording_id=f"umi_synth_{e:04d}",
                skill_id="pick_place",
                source="sim_synth",
                frame=FRAME_ROBOT_BASE,
                n_steps=n,
                pose_rate_hz=float(args.pose_rate_hz),
                cameras=sorted(cams),
                calibration_id="",  # 합성이라 카메라도 hand-eye 도 없다
                recorded_by="AI 학습 담당자(트랙B) tools/make_umi_synth.py",
                notes={
                    "task": "합성 관통 입력 (시연 궤적 아님)",
                    "success": True,
                    "synthetic": True,
                    "image_rate_hz": float(args.image_rate_hz),
                    "image_lag_ms": float(args.image_lag_ms),
                    "camera_codes": cam_code,
                    "index_encoded_in_pixels": True,
                    "gripper_status_note": "합성 — 전 프레임 D",
                },
            ),
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            gripper_gap_m=gaps,
            # 합성이므로 전 프레임 D(양쪽 직접검출). 실기록은 약 97% 만 D·M 이고
            # T·X 는 gap 결측이다 (SPEC) — 변환기가 그 프레임을 폐기한다.
            gripper_status=np.full(n, "D", dtype="<U1"),
            pose_timestamp=pose_ts,
            images=images,
            image_timestamp=stamps,
        )
        path = write_raw(raw, args.out)
        total_bytes += path.stat().st_size
        print(f"  {path.name}  {n}스텝 {n_frames}프레임  {path.stat().st_size / 1e6:5.2f}MB  "
              f"gap {gaps.min() * 100:.2f}~{gaps.max() * 100:.2f}cm")

    print(f"\n{args.episodes}편 · 합계 {total_bytes / 1e6:.1f}MB · "
          f"평균 {total_bytes / args.episodes / 1e6:.2f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
