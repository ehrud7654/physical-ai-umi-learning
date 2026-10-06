"""Run the complete hardware-free UMI regression suite on a local PC.

This broad suite remains local-only because it includes policy-level probes.
The shared host may run training and explicitly no-policy MuJoCo simulations,
but not learned-policy inference or this mixed regression suite.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


AI_ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    "check_umi_convert.py",
    "check_umi_provenance.py",
    "check_umi_action_decode.py",
    "check_umi_action_ik.py",
    "check_umi_policy_preflight.py",
    "check_umi_arm_preflight.py",
    "check_umi_chunk_runtime.py",
    "check_umi_real_fk.py",
    "check_umi_installation_limits.py",
    "check_umi_ros_adapter.py",
    "check_offline_real_handoff.py",
    "check_umi_handover_edges.py",
    "check_umi_ingestion.py",
    "check_gripper_gap.py",
    "check_arpose_delivery_adapter.py",
    "check_umi_delivery_inventory.py",
    "check_umi_camera_frames.py",
    "check_umi_relative_audit.py",
    "check_umi_real_placement.py",
    "check_umi_real_ik_diagnosis.py",
    "check_umi_axis_projection.py",
    "check_umi_trajectory_rate_limit.py",
    "check_umi_episode_physics.py",
    "check_umi_object_registration.py",
    "check_sim_gripper_contract.py",
    "check_relative_chunk_oracle_gate.py",
    "check_relative_chunk_registration_search.py",
    "check_ver1_collision_proxy.py",
    "check_ver1_side_workspace.py",
    "check_umi_action_roundtrip.py",
    "check_v10_localisation_failure_factors.py",
    "check_v10_lateral_sampling.py",
    "check_relative_chunk_spatial_encoder.py",
    "check_moving_h2_smooth_profile.py",
    "check_v10_heldout_visual_response.py",
)


def is_shared_gpu_server(hostname: str | None = None) -> bool:
    name = (hostname or platform.node()).lower()
    return name.startswith("jupyter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="검사 목록만 출력")
    args = parser.parse_args()
    if args.list:
        print("\n".join(CHECKS))
        return 0
    if is_shared_gpu_server():
        print(
            "거부: 정책 검사가 섞인 전체 회귀검사는 로컬 전용이다. "
            "공유 서버는 학습과 명시적인 no-policy MuJoCo만 허용한다.")
        return 2

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    failures: list[str] = []
    for index, script in enumerate(CHECKS, 1):
        print(f"\n[{index}/{len(CHECKS)}] {script}", flush=True)
        result = subprocess.run([sys.executable, str(AI_ROOT / "tools" / script)],
                                cwd=AI_ROOT, env=env, check=False)
        if result.returncode:
            failures.append(f"{script}({result.returncode})")

    print("\n=== UMI local regression ===")
    if failures:
        print("FAIL: " + ", ".join(failures))
        return 1
    print(f"PASS: {len(CHECKS)}/{len(CHECKS)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
