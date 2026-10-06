#!/usr/bin/env bash
# Assemble a self-contained folder for a GPU server: trainer code + dataset + notebook, then zip it.
#   bash make_gpu_bundle.sh <dataset.zarr.zip> <out_dir>
set -euo pipefail
DATASET=$1; OUT=$2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT/data" "$OUT/03_umi_policy_trainer"
cp -r "$HERE"/{src,configs,patches,train_policy.py,requirements.txt,README.md} "$OUT/03_umi_policy_trainer/"
find "$OUT/03_umi_policy_trainer" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cp "$HERE/train_umi.ipynb" "$OUT/"
base="${DATASET%.zarr.zip}"
cp "$DATASET" "$base.zarr.report.json" "$OUT/data/"
[ -f "$base.episodes.json" ] && cp "$base.episodes.json" "$base.dataset_config.json" "$OUT/data/"
du -sh "$OUT"; ls -la "$OUT/data"
