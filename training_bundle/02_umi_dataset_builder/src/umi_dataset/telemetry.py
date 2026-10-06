from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path


def _read_xyz(path: Path, fields: tuple[str, str, str]):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    samples = [(int(row["timestamp_ns"]), [float(row[field]) for field in fields]) for row in rows]
    if len(samples) < 2 or any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
        raise ValueError(f"invalid timestamps in {path.name}")
    return samples


def _interpolate(samples, timestamp):
    times = [sample[0] for sample in samples]
    right = bisect.bisect_left(times, timestamp)
    if right == 0 or right == len(samples):
        raise ValueError("interpolation timestamp is outside source range")
    left = right - 1
    ratio = (timestamp - times[left]) / (times[right] - times[left])
    return [a + ratio * (b - a) for a, b in zip(samples[left][1], samples[right][1])]


def prepare_telemetry(session: Path, camera_imu: dict, output: Path) -> dict:
    if camera_imu.get("status") != "validated_repeatable":
        raise ValueError("camera-IMU calibration has not passed repeat validation")
    with (session / "frames.csv").open(encoding="utf-8") as stream:
        frame_times = [int(row["sensor_timestamp_ns"]) for row in csv.DictReader(stream)]
    accel = _read_xyz(session / "accelerometer.csv", ("x_m_s2", "y_m_s2", "z_m_s2"))
    gyro = _read_xyz(session / "gyroscope.csv", ("x_rad_s", "y_rad_s", "z_rad_s"))
    shift_ns = round(camera_imu["timeshift_cam_imu_s"] * 1e9)
    start, stop = frame_times[0], frame_times[-1]
    paired = []
    for timestamp, gyro_value in gyro:
        aligned = timestamp - shift_ns
        if start <= aligned <= stop and accel[0][0] < timestamp < accel[-1][0]:
            paired.append((aligned, _interpolate(accel, timestamp), gyro_value))
    if not paired:
        raise ValueError("no overlapping camera and IMU samples")
    origin = frame_times[0]
    payload = {"1": {"streams": {
        "ACCL": {"samples": [{"cts": (t - origin) / 1e6, "value": a} for t, a, _ in paired]},
        "GYRO": {"samples": [{"cts": (t - origin) / 1e6, "value": g} for t, _, g in paired]},
        "CORI": {"samples": [{"cts": (t - origin) / 1e6, "value": [0, 0, 0, 1]}
                              for t in frame_times]},
    }}}
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {
        "frames": len(frame_times),
        "imu_samples": len(paired),
        "duration_s": (frame_times[-1] - origin) / 1e9,
        "applied_imu_shift_ms": -shift_ns / 1e6,
    }
