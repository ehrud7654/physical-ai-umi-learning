#!/bin/sh
# Run ORB-SLAM3 + trajectory validation for many sessions. Skips sessions that already have output.
# usage: sudo sh run_orbslam_batch_jetson.sh RAW_ROOT PREPARED_ROOT OUTPUT_ROOT [SESSION ...]
set -eu

# docker -v requires absolute host paths; resolve whatever the caller passed.
RAW=$(cd "${1:?usage: run_orbslam_batch_jetson.sh RAW_ROOT PREPARED_ROOT OUTPUT_ROOT [SESSION ...]}" && pwd)
PREP=$(cd "${2:?}" && pwd)
mkdir -p "${3:?}"
OUT=$(cd "$3" && pwd)
shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -gt 0 ] || set -- $(ls "$PREP")

for name in "$@"; do
  if [ -f "$OUT/$name/camera_trajectory.csv" ]; then
    echo "skip $name (trajectory exists)"
    continue
  fi
  echo "== $name"
  start=$(date +%s)
  if sh "$HERE/run_orbslam_jetson.sh" "$RAW/$name" "$PREP/$name" "$OUT/$name" > "$OUT/$name.log" 2>&1; then
    python3 "$HERE/validate_orbslam_trajectory.py" "$OUT/$name/camera_trajectory.csv" \
      --output "$OUT/$name/trajectory_validation.json" || echo "VALIDATION FAILED $name"
  else
    echo "SLAM FAILED $name (see $OUT/$name.log)"
  fi
  echo "   $(( $(date +%s) - start ))s"
done
