FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ORB_SLAM_COMMIT=b741dca39015330ef4bcc3a85f89493503ade04b
ARG JOBS=4

LABEL org.opencontainers.image.title="UMI ORB-SLAM Atlas Preprocessor" \
      org.opencontainers.image.version="2026.09.23-orbslam-v3" \
      org.opencontainers.image.revision="b741dca39015330ef4bcc3a85f89493503ade04b"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config ninja-build python3 python3-venv \
    libopencv-dev libeigen3-dev libboost-dev libboost-serialization-dev libssl-dev \
    libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev \
    libc++-dev libglew-dev libjpeg-dev libpng-dev libavcodec-dev libavutil-dev \
    libavformat-dev libswscale-dev libavdevice-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/umi
COPY requirements.txt ./
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY dataset_builder/slam /opt/umi/dataset_builder/slam
COPY docker/build_orbslam.sh /tmp/build_orbslam.sh
RUN git clone https://github.com/cheng-chi/ORB_SLAM3.git /opt/ORB_SLAM3 && \
    git -C /opt/ORB_SLAM3 checkout "$ORB_SLAM_COMMIT" && \
    git -C /opt/ORB_SLAM3 config submodule.Thirdparty/Pangolin.url https://github.com/stevenlovegrove/Pangolin.git && \
    git -C /opt/ORB_SLAM3 submodule update --init --recursive && \
    JOBS="$JOBS" bash /tmp/build_orbslam.sh /opt/ORB_SLAM3 /opt/umi/dataset_builder/slam && \
    rm -rf /opt/ORB_SLAM3/.git

COPY dataset_builder /opt/umi/dataset_builder
COPY track_a /opt/umi/track_a
COPY backend_contract.json SOURCE_PROVENANCE.txt /opt/umi/handoff/
COPY backend /opt/umi/handoff/backend
COPY app /opt/umi/app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/umi/app/src \
    ORB_SLAM_BIN=/opt/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
    ORB_SLAM_VOCAB=/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    LD_LIBRARY_PATH=/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib:/opt/ORB_SLAM3/Thirdparty/Pangolin/build \
    UMI_TRACK_A_ROOT=/opt/umi/track_a \
    UMI_BACKEND_CONTRACT=/opt/umi/handoff/backend_contract.json \
    UMI_JOBS_ROOT=/jobs \
    UMI_PROFILES_ROOT=/profiles

RUN useradd --create-home --uid 10001 umi && mkdir -p /jobs /profiles && chown -R umi:umi /jobs /profiles
USER umi
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"
CMD ["python", "/opt/umi/app/server.py"]
