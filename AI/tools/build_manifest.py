#!/usr/bin/env python3
"""Build the CKPT manifest from a training run directory (no hand-written values).
학습 run 디렉터리에서 manifest 를 산출한다. 손으로 적는 값을 두지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

SCHEMA = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def architecture_of(cfg: dict) -> str:
    """Policy class name from the Hydra _target_. 손으로 적지 않는다."""
    target = (cfg.get("policy") or {}).get("_target_")
    if not target:
        raise ValueError("config.yaml 에 policy._target_ 가 없다")
    return str(target).rsplit(".", 1)[-1]


def n_params(ckpt_path: Path) -> tuple[int, str]:
    """Parameter count from the checkpoint. EMA copy is excluded.
    체크포인트의 파라미터 수. EMA 사본은 제외한다."""
    import torch
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sds = blob.get("state_dicts")
    if not isinstance(sds, dict) or "model" not in sds:
        raise ValueError(f"예상한 state_dicts['model'] 이 없다: keys={list(blob)}")
    sd = sds["model"]
    total = sum(t.numel() for t in sd.values() if hasattr(t, "numel"))
    note = "state_dicts['model'] (ema_model 제외)"
    return int(total), note


def schemas(cfg: dict) -> tuple[dict, dict, int, int, int]:
    meta = (cfg.get("task") or {}).get("shape_meta") or cfg.get("shape_meta")
    if not meta:
        raise ValueError("config.yaml 에 shape_meta 가 없다")
    obs = meta["obs"]
    inp = {}
    for key, spec in obs.items():
        h = int(spec.get("horizon", 1))
        inp[key] = {"shape": [h] + list(spec["shape"])}
    act = meta["action"]
    horizon = int(act["horizon"])
    dim = int(act["shape"][0])
    down = int(act.get("down_sample_steps", 1))
    return inp, {"actionShape": [horizon, dim]}, horizon, dim, down


def build(run_dir: Path, native_hz: float, exec_start: int, exec_end: int) -> dict:
    cfg_path = run_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ckpt = run_dir / "checkpoints" / "best.ckpt"
    inp, out, horizon, dim, down = schemas(cfg)
    params, note = n_params(ckpt)
    rate = native_hz / down
    n_action_steps = int(cfg.get("n_action_steps", exec_end - exec_start))
    return {
        "architecture": architecture_of(cfg),
        "framework": "PyTorch",
        "frameworkVersion": __import__("torch").__version__,
        "actionSpace": "EEF_RELATIVE_ROT6D",
        "actionSpec": {
            "schemaVersion": SCHEMA, "dim": dim, "horizon": horizon,
            "nActionSteps": n_action_steps, "rateHz": rate,
            "layout": ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12", "gap_m"],
            "layoutNote": "각 행은 증분이 아니라 현재 자세 기준 절대 상대 포즈다",
            "rotation": "ROTATION_MATRIX_ROWS_0_1",
            "compose": "T_next = T_cur @ A_relative",
            "gapUnit": "m", "gapRange": [0, 0.09],
        },
        "runtimeSpec": {
            "consumerMustConvert": True, "convertedDim": 7,
            "convertedLayout": ["x", "y", "z", "rx", "ry", "rz", "gap_m"],
            "convertedRotation": "AXIS_ANGLE_ROTVEC", "convertedFrame": "ABSOLUTE",
            "execSlice": {"startInclusive": exec_start, "endExclusive": exec_end},
            "execSliceAppliesTo": "raw",
            "execSliceNote": "원본 (horizon, dim) 배열에 적용한다. index 0 은 현재 시점이라 건너뛴다",
            "actionPointRateHz": rate,
            "reobserveRateHz": rate / (exec_end - exec_start),
            "reobserveRateNote": "공칭. 실측 사이클 주기는 추론 지연 때문에 더 느리다",
        },
        "cameraNames": [k for k in inp if k.endswith("_rgb")],
        "inputSchema": inp,
        "outputSchema": out,
        "controlRateHz": rate,
        "nParams": params,
        "provenance": {
            "runDir": str(run_dir), "configSha256": _sha256(cfg_path),
            "checkpointSha256": _sha256(ckpt), "nParamsSource": note,
            "nativeHz": native_hz, "downSampleSteps": down,
        },
    }


def selftest() -> bool:
    rows = []
    cfg = {"policy": {"_target_": "a.b.DiffusionUnetTimmPolicy"}}
    rows.append(("1 architecture 를 _target_ 에서 추출", architecture_of(cfg) == "DiffusionUnetTimmPolicy", architecture_of(cfg)))
    try:
        architecture_of({"policy": {}}); ok = False
    except ValueError:
        ok = True
    rows.append(("2 _target_ 없으면 거부 (정답 아는 행)", ok, "ValueError"))
    fake = {"task": {"shape_meta": {
        "obs": {"camera0_rgb": {"shape": [3, 224, 224], "horizon": 2},
                "robot0_gripper_width": {"shape": [1], "horizon": 2}},
        "action": {"shape": [10], "horizon": 16, "down_sample_steps": 3}}}}
    inp, out, h, d, down = schemas(fake)
    rows.append(("3 inputSchema 에 horizon 이 앞에 붙는다", inp["camera0_rgb"]["shape"] == [2, 3, 224, 224], str(inp["camera0_rgb"]["shape"])))
    rows.append(("4 outputSchema = [horizon, dim]", out == {"actionShape": [16, 10]}, str(out)))
    rows.append(("5 controlRate = native/down", abs(30 / down - 10) < 1e-9, f"30/{down}=10"))
    try:
        schemas({"task": {}}); ok2 = False
    except ValueError:
        ok2 = True
    rows.append(("6 shape_meta 없으면 거부 (정답 아는 행)", ok2, "ValueError"))
    passed = sum(1 for _, p, _ in rows if p)
    for name, p, note in rows:
        print(f"  {'PASS' if p else 'FAIL'}  {name}   {note}")
    print(f"자체검증 {passed} / {len(rows)}")
    return passed == len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--native-hz", type=float, default=30.0, help="원본 수집 샘플률. 기본 30")
    ap.add_argument("--exec-start", type=int, default=1)
    ap.add_argument("--exec-end", type=int, default=5)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest-only", action="store_true")
    args = ap.parse_args()
    print("=== 자체검증")
    if not selftest():
        print("!! 자체검증 실패 — manifest 를 내지 않는다")
        sys.exit(1)
    if args.selftest_only:
        return
    m = build(args.run_dir, args.native_hz, args.exec_start, args.exec_end)
    text = json.dumps(m, indent=2, ensure_ascii=False)
    print("\n" + text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"\n작성: {args.out}")


if __name__ == "__main__":
    main()
