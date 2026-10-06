"""raw → 계약 데이터셋 CLI. 이슈 31 과 127 이 같이 쓰는 진입점.
CLI turning UMI raw recordings into a contract dataset. Shared by project-task and -127.

This is the wiring layer, so it may import both `umi/` and `sim/`. The conversion
itself is in `umi/convert.py`, which imports neither.
배선 층이라 `umi/` 와 `sim/` 을 함께 임포트해도 된다. 변환 자체는 둘 중 어느
것도 임포트하지 않는 `umi/convert.py` 에 있다.

## 두 가지를 부산물로 낸다 — 별도 도구로 만들지 않는다

**수용률** (project-task) — 에피소드·스텝 단위 채택/폐기와 폐기 이유.
별도 도구는 데이터를 만든 파이프라인과 **다른 것**을 재게 된다.

**identity 기준선** — `mean|action - state|` 와 identity 예측기의 MSE.
모델 없이 나오는 데이터셋 속성이다. 절대 관절각 표현은 손실 대부분을 "팔이 지금
어디 있나"로 채우므로, 학습 손실은 이 값보다 **뚜렷하게** 낮아야 의미가 있다.

## 이미지 정렬을 픽셀로 대조한다

`--verify-image-index` 는 각 스텝의 이미지에서 프레임 번호와 카메라 코드를 되읽어,
**전체 프레임에 대한 무차별 최근접 탐색**과 대조한다. `umi.convert._nearest_frames`
를 다시 쓰지 않는다 — 같은 코드로 검사하면 그 코드의 off-by-one 을 못 잡는다.
합성 입력(`tools/make_umi_synth.py`)에서만 쓸 수 있다.

    # [로컬]
    cd AI && python tools/convert_umi.py --raw out/umi_raw_synth_v1 --out datasets/umi_synth_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from contract.episode import validate, write_dataset_index, write_episode  # noqa: E402
from paths import DATASET_DIR, DEFAULT_CONFIG, DEFAULT_SCENE, OUT_DIR  # noqa: E402
from sim.mujoco.build_scene import build_model, load_config  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402
from umi.convert import ConversionError, ConversionPolicy, convert  # noqa: E402
from umi.raw import read_raw  # noqa: E402
from umi.provenance import conversion_digest
from umi.trajectory import TrajectoryLimits
from tools.make_umi_synth import decode_frame  # noqa: E402
from tools.umi_mujoco import MujocoIK  # noqa: E402


def raw_bundle_digest(raw_dir: Path, raw_npzs: list[Path]) -> str:
    """Hash raw arrays and their metadata with stable relative file names."""
    root = raw_dir.resolve()
    files: list[Path] = []
    for npz_path in raw_npzs:
        meta_path = npz_path.with_suffix("").with_suffix(".raw.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"raw metadata missing: {meta_path}")
        files.extend((npz_path, meta_path))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.resolve().relative_to(root).as_posix()):
        relative = path.resolve().relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:12]


def verify_image_index(episode, raw, kept_span: tuple[int, int]) -> tuple[int, int]:
    """Compare the frame each step actually got with a brute-force nearest search.
    각 스텝이 실제로 받은 프레임을 무차별 최근접 탐색과 대조한다.

    Returns (frame index mismatches, camera code mismatches).
    반환 (프레임 번호 불일치 수, 카메라 코드 불일치 수)."""
    start, end = kept_span
    # The final raw state is only the target of the preceding action and has no
    # image-bearing training row of its own.
    pose_ts = np.asarray(raw.pose_timestamp[start:end - 1], dtype=np.float64)
    codes = raw.meta.notes.get("camera_codes", {})
    bad_idx = bad_cam = 0
    for cam, frames in episode.images.items():
        img_ts = np.asarray(raw.image_timestamp[cam], dtype=np.float64)
        for i, t in enumerate(pose_ts):
            expected = int(np.argmin(np.abs(img_ts - t)))
            got_idx, got_cam = decode_frame(frames[i])
            if got_idx != expected:
                bad_idx += 1
            if cam in codes and got_cam != int(codes[cam]):
                bad_cam += 1
    return bad_idx, bad_cam


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=OUT_DIR / "umi_raw_synth_v1")
    ap.add_argument("--out", type=Path, default=DATASET_DIR / "umi_synth_v1")
    ap.add_argument("--author", type=str, default="AI 학습 담당자(트랙B)")
    ap.add_argument("--clamp-gap", action="store_true",
                    help="실측 gap 곡선 밖의 간격을 양 끝으로 물린다. 기본은 스텝 폐기")
    ap.add_argument("--max-pos-error-mm", type=float, default=5.0)
    # 기본값을 여기 적지 않는다. MujocoIK 가 정본이고, CLI 에 같은 기본값을 또 두면
    # 바깥이 이긴다 — 실측 🟢 2026-09-08: MujocoIK 를 1e-7 로 고쳐놓고 이 CLI 기본값
    # 1e-5 가 그것을 덮어 IK 위치 오차가 0.0001mm 대신 0.0082mm 로 나왔다.
    ap.add_argument("--pos-tol-m", type=float, default=None,
                    help="솔버 정지 허용오차. 생략하면 MujocoIK 기본값")
    ap.add_argument("--axis-tol-deg", type=float, default=None)
    ap.add_argument("--verify-image-index", action="store_true", default=True)
    ap.add_argument("--no-verify-image-index", dest="verify_image_index", action="store_false")
    ap.add_argument("--log", action="store_true", help="EXP_LOG.jsonl 에 한 줄 append")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument('--trajectory-limits', type=Path,
                    help='JSON: max_step, max_velocity, max_acceleration; 5 rad channels + gap m')
    args = ap.parse_args()

    raws = sorted(args.raw.glob("*.raw.npz"))
    if not raws:
        print(f"raw 기록이 없다: {args.raw}")
        return 1
    raw_path = str(args.raw.resolve())
    raw_sha = raw_bundle_digest(args.raw, raws)
    tool_sha = conversion_digest()
    if args.out.exists() and sorted(args.out.glob("*.npz")):
        print(f"이미 에피소드가 있다: {args.out}\n"
              "  기존 이름을 재사용하지 않는다 — ckpt 까지 덮어쓴다. --out 에 다음 번호를 줘라")
        return 1

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    ik_kw = {}
    if args.pos_tol_m is not None:
        ik_kw["pos_tol_m"] = args.pos_tol_m
    if args.axis_tol_deg is not None:
        ik_kw["axis_tol_deg"] = args.axis_tol_deg
    ik = MujocoIK(model, cfg, **ik_kw)
    policy = ConversionPolicy(
        max_pos_error_m=args.max_pos_error_mm / 1000.0, clamp_gap=args.clamp_gap,
        trajectory_limits=(TrajectoryLimits(**json.loads(args.trajectory_limits.read_text(encoding='utf-8')))
                           if args.trajectory_limits else None)
    )
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    config_sha = file_digest(args.config)

    print(f"raw {args.raw} · {len(raws)}편  →  {args.out}")
    print(f"정규화 두 구현 최대차 {ik.norm_crosscheck_max:.3e} · "
          f"솔버 정지 pos {ik.pos_tol_m:.0e}m · roll 맞춤 {ik.match_roll}")
    print(f"위치 예산 {args.max_pos_error_mm}mm · gap 클램프 {args.clamp_gap}\n")

    all_sols, reports, failures = [], [], {}
    rejects: Counter[str] = Counter()
    bad_idx_total = bad_cam_total = 0
    n_index_checked = n_index_skipped = 0
    steps_in = steps_out = 0
    sync_max = 0.0
    ident_abs, ident_mse = [], []

    for path in raws:
        raw = read_raw(path)
        try:
            ep, rep = convert(
                raw, ik, ik.ranges, cfg["grasp"]["gap_curve"],
                policy=policy, control_rate_hz=raw.meta.pose_rate_hz,
                collected_by=args.author, config_sha=config_sha, git_rev=rev,
            )
        except ConversionError as exc:
            failures[path.name] = str(exc).splitlines()[0]
            print(f"  {path.name}  변환 실패 — {str(exc).splitlines()[0]}")
            continue

        problems = validate(ep)
        if problems:
            failures[path.name] = "; ".join(problems)
        else:
            write_episode(ep, args.out)

        # 픽셀에 프레임 번호가 심긴 입력에서만 유효하다. 진짜 렌더 이미지에는 그
        # 비트가 없어 쓰레기를 읽는다 — 실측 🟢 2026-09-08: 141스텝 x 2카메라에서
        # 280건이 "불일치" 로 잡혀 파이프라인이 끊겼다. 검사 실패가 아니라 적용
        # 불가였다. **계측기가 자기 적용 가능성을 알아야 한다.** 플래그에 맡기지 않는다.
        bi = bc = 0
        encoded = bool(raw.meta.notes.get("index_encoded_in_pixels", False))
        if args.verify_image_index and encoded:
            bi, bc = verify_image_index(ep, raw, rep.kept_span)
            bad_idx_total += bi
            bad_cam_total += bc
            n_index_checked += 1
        elif args.verify_image_index:
            n_index_skipped += 1

        reports.append(rep)
        rejects.update(rep.rejects)
        steps_in += rep.n_steps_in
        steps_out += rep.n_steps_out
        sync_max = max(sync_max, max(v["max_ms"] for v in rep.image_sync_offset_ms.values()))
        ident_abs.append(rep.identity_residual["mean_abs"])
        ident_mse.append(rep.identity_residual["identity_mse"])

        print(f"  {rep.recording_id}  {rep.n_steps_out}/{rep.n_steps_in}스텝  "
              f"IK위치중앙 {rep.residual.pos_error_median_mm:.4f}mm  "
              f"roll중앙 {rep.residual.roll_residual_median_deg:.3f}도  "
              f"동기max {max(v['max_ms'] for v in rep.image_sync_offset_ms.values()):.2f}ms  "
              f"{'계약OK' if not problems else '계약위반 ' + str(len(problems))}"
              + (f"  이미지정렬 불일치 {bi}/{bc}" if bi or bc else ""))

    if not reports:
        print("\n변환된 에피소드가 없다")
        return 1

    print(f"\n=== {len(reports)}편 · 스텝 {steps_out}/{steps_in} "
          f"(수용률 {steps_out / steps_in * 100:.1f}%) ===")
    print(f"폐기 이유 {dict(rejects) or '없음'}")
    pos = np.array([r.residual.pos_error_median_mm for r in reports])
    roll = np.array([r.residual.roll_residual_median_deg for r in reports])
    print(f"IK 위치 오차 중앙값의 중앙 {np.median(pos):.4f}mm · 최대 {pos.max():.4f}")
    print(f"roll 잔차 중앙값의 중앙 {np.median(roll):.4f}도 · 최대 {roll.max():.4f}")
    print(f"pose↔이미지 동기 오차 최대 {sync_max:.2f}ms  "
          f"(계약의 state/action 10ms 게이트는 이 오차를 보지 못한다)")
    print(f"identity 기준선  mean|action-state| {np.mean(ident_abs):.5f} · "
          f"MSE {np.mean(ident_mse):.3e}  (계약 정규화 단위)")
    # ⚠️ 2026-09-08 정정. 이전 문구는 "학습 손실이 이 MSE 보다 낮아야 한다" 였고
    #    **단위가 달라 비교 대상이 아니다.** train_bc 는 joint_delta_gripper_abs
    #    행동공간 + 타겟 표준화 단위로 학습하고, 자명한 예측기(항상 0) 손실을
    #    스스로 찍는다 (실측 예: 0.72245). 판정은 그 값을 쓴다.
    print("  이 값은 **데이터 속성**이다. 학습 손실과 직접 비교하지 마라 —")
    print("  train_bc 가 표준화 단위로 자명한 예측기 손실을 따로 찍는다. 판정은 그걸 쓴다")
    if n_index_checked:
        verdict = "통과" if bad_idx_total == 0 and bad_cam_total == 0 else "실패"
        print(f"이미지 정렬 픽셀 대조 {verdict} — {n_index_checked}편 "
              f"(프레임번호 불일치 {bad_idx_total} · 카메라코드 불일치 {bad_cam_total})")
    if n_index_skipped:
        print(f"이미지 정렬 픽셀 대조 **해당 없음** — {n_index_skipped}편 "
              "(픽셀에 프레임 번호가 심기지 않은 입력. 진짜 렌더 이미지가 그렇다)")
        print("  ⚠️ 이 입력에서는 프레임 정렬이 독립적으로 검증되지 않는다. "
              "동기 오차 수치만 남는다")
    print(f"계약 위반 에피소드 {len(failures)}/{len(raws)}")
    for name, why in failures.items():
        print(f"  {name}: {why}")

    if not failures:
        index = write_dataset_index(args.out, extra={
            "converted_by": args.author,
            "source": "umi_raw (synthetic)" if reports[0].recording_id.startswith("umi_synth") else "umi_raw",
            "step_acceptance_rate": round(steps_out / steps_in, 4),
            "ik_pos_error_median_mm": round(float(np.median(pos)), 5),
            "roll_residual_median_deg": round(float(np.median(roll)), 5),
            "image_sync_offset_max_ms": round(sync_max, 3),
            "identity_mse": round(float(np.mean(ident_mse)), 8),
            "image_index_verified_episodes": n_index_checked,
            "image_index_not_applicable_episodes": n_index_skipped,
            "solver_pos_tol_m": ik.pos_tol_m,
            "match_roll": ik.match_roll,
            "raw_path": raw_path,
            "raw_sha": raw_sha,
            "tool_sha": tool_sha,
        })
        print(f"인덱스: {index}")

    if args.log:
        log_run(
            experiment="umi_convert",
            author=args.author,
            issue="project-task",
            conditions={
                "raw_dir": str(args.raw), "out": str(args.out), "n_raw": len(raws),
                "config_sha": config_sha, "git_rev": rev,
                "raw_path": raw_path, "raw_sha": raw_sha, "tool_sha": tool_sha,
                "solver_pos_tol_m": ik.pos_tol_m, "match_roll": ik.match_roll,
                "python": ".".join(map(str, sys.version_info[:3])),
            },
            result={
                "episodes_ok": len(raws) - len(failures),
                "step_acceptance_rate": steps_out / steps_in,
                "ik_pos_error_median_mm": float(np.median(pos)),
                "roll_residual_median_deg": float(np.median(roll)),
                "image_sync_offset_max_ms": sync_max,
                "identity_mse": float(np.mean(ident_mse)),
                "image_index_mismatch": bad_idx_total,
                "contract_violations": len(failures),
                "rejects": dict(rejects),
            },
        )
        print("EXP_LOG 에 append 했다")

    return 1 if failures or bad_idx_total or bad_cam_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
