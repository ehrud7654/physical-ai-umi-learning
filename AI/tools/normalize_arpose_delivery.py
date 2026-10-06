"""Convert the delivered multi-recording arpose ZIP into canonical UMI bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track_a.convert.arpose_delivery import normalize_delivery
from tools.run_umi_regression import is_shared_gpu_server


def main() -> None:
    if is_shared_gpu_server():
        raise SystemExit("Shared GPU server is training-only")
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--gripper-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pre-stabilized", action="store_true", required=True,
                        help="explicit evidence that stabilization finished before recording")
    args = parser.parse_args()
    result = normalize_delivery(delivery_zip=args.delivery,
        gripper_root=args.gripper_root, output_dir=args.out,
        pre_stabilized=args.pre_stabilized)
    print(json.dumps({"episodes": len(result["episodes"]),
                      "rejected": len(result["rejected"]),
                      "target_schema": result["target_schema"],
                      "warmup_s": result["warmup_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
