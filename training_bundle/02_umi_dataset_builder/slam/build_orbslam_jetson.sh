#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE=${1:-/path/to/umi-workspace/third_party/umi-orb-slam3}
DOCKERFILE="$SCRIPT_DIR/Dockerfile.umi-orb-slam3-arm64"
IMAGE=${ORB_SLAM_IMAGE:-umi-orb-slam3:b741dca-s22}
COMMIT=b741dca39015330ef4bcc3a85f89493503ade04b

test "$(git -c safe.directory="$SOURCE" -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -f "$SOURCE/Examples/Monocular-Inertial/gopro_slam.cc"
test -f "$DOCKERFILE"

if [ -n "${SUDO_USER:-}" ]; then
  sudo -u "$SUDO_USER" git -C "$SOURCE" submodule update --init --recursive
else
  git -C "$SOURCE" submodule update --init --recursive
fi
test -f "$SOURCE/Thirdparty/Pangolin/scripts/install_prerequisites.sh"

docker build -t "$IMAGE" -f "$DOCKERFILE" "$SOURCE"
