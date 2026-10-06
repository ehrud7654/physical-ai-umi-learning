#!/bin/sh
# ORB-SLAM3 (UMI fork) in the ARM64 Docker image on the Jetson.
# Map reuse: ORB_SLAM_SAVE_MAP writes <OUTPUT_DIR>/<name>.osa; ORB_SLAM_LOAD_MAP mounts the map's folder.
# usage: run_orbslam_jetson.sh SESSION PREPARED_DIR OUTPUT_DIR
set -eu

SESSION=${1:?usage: run_orbslam_jetson.sh SESSION PREPARED_DIR OUTPUT_DIR}
INPUT=${2:?usage: run_orbslam_jetson.sh SESSION PREPARED_DIR OUTPUT_DIR}
OUTPUT=${3:?usage: run_orbslam_jetson.sh SESSION PREPARED_DIR OUTPUT_DIR}
IMAGE=${ORB_SLAM_IMAGE:-umi-orb-slam3:b741dca-s22}
INIT_TAG_ID=${INIT_TAG_ID:-13}
INIT_TAG_SIZE_M=${INIT_TAG_SIZE_M:-0.16}   # UMI 16 cm table marker; the ChArUco board's ID 13 is 0.018

test -f "$SESSION/video.mp4"
test -f "$INPUT/telemetry.json"
test -f "$INPUT/s22_mono_inertial.yaml"
mkdir -p "$OUTPUT"
SESSION=$(cd "$SESSION" && pwd); INPUT=$(cd "$INPUT" && pwd); OUTPUT=$(cd "$OUTPUT" && pwd)

MAP_MOUNT=""
MAP_ARGS=""
HERE=$(cd "$(dirname "$0")" && pwd)
MASK=${ORB_SLAM_MASK-$HERE/../configs/s22_slam_mask.png}   # ORB_SLAM_MASK="" disables the mask
if [ -n "$MASK" ] && [ -f "$MASK" ]; then
  MASK_DIR=$(cd "$(dirname "$MASK")" && pwd)
  MAP_MOUNT="$MAP_MOUNT -v $MASK_DIR:/mask:ro"
  MAP_ARGS="$MAP_ARGS --mask_img /mask/$(basename "$MASK")"
fi
if [ -n "${ORB_SLAM_LOAD_MAP:-}" ]; then
  test -f "$ORB_SLAM_LOAD_MAP"
  MAP_DIR=$(cd "$(dirname "$ORB_SLAM_LOAD_MAP")" && pwd)
  MAP_MOUNT="-v $MAP_DIR:/map:ro"
  MAP_ARGS="--load_map /map/$(basename "$ORB_SLAM_LOAD_MAP")"
fi
if [ -n "${ORB_SLAM_SAVE_MAP:-}" ]; then
  # The container can only write to /output, so the map is saved next to the trajectory.
  MAP_ARGS="$MAP_ARGS --save_map /output/$(basename "$ORB_SLAM_SAVE_MAP")"
fi

# shellcheck disable=SC2086
docker run --rm \
  -v "$SESSION:/session:ro" \
  -v "$INPUT:/input:ro" \
  -v "$OUTPUT:/output" \
  $MAP_MOUNT \
  "$IMAGE" \
  /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
  --vocabulary /ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --setting /input/s22_mono_inertial.yaml \
  --input_video /session/video.mp4 \
  --input_imu_json /input/telemetry.json \
  --output_trajectory_csv /output/camera_trajectory.csv \
  --output_trajectory_tum /output/camera_trajectory_tum.txt \
  --use_sensor_timestamps \
  --init_tag_id "$INIT_TAG_ID" \
  --init_tag_size "$INIT_TAG_SIZE_M" \
  $MAP_ARGS
