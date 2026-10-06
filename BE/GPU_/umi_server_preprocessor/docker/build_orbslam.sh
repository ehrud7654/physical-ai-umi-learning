#!/usr/bin/env bash
set -euo pipefail

root=${1:?usage: build_orbslam.sh ORB_SLAM3_DIR PATCH_DIR}
patches=${2:?}
jobs=${JOBS:-4}

apply_once() {
  if git -C "$root" apply --check "$1" 2>/dev/null; then
    git -C "$root" apply "$1"
  elif ! git -C "$root" apply --check -R "$1" 2>/dev/null; then
    echo "cannot apply $(basename "$1")" >&2
    exit 1
  fi
}

for patch in "$patches"/*.patch; do
  apply_once "$patch"
done
# Sophus enables -Werror for its test targets. GCC 13 reports a known Eigen SIMD
# array-bounds false positive, so keep the warning visible without failing the image build.
sed -i 's/-Werror/-Wno-error=array-bounds/g' "$root/Thirdparty/Sophus/CMakeLists.txt"
sed -i "s/^make -j.*$/make -j$jobs/; s/^mkdir build$/mkdir -p build/" "$root/build.sh"
(cd "$root" && bash build.sh)
test -x "$root/Examples/Monocular-Inertial/gopro_slam"
test -f "$root/Vocabulary/ORBvoc.txt"
