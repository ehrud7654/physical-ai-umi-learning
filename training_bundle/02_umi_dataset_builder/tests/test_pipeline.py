import csv
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umi_dataset.camera_tcp import load_camera_tcp
from umi_dataset.replay_buffer import build_replay_buffer


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    camera_tcp, _ = load_camera_tcp(ROOT / "configs/s22_camera_tcp.json")
    assert camera_tcp.shape == (4, 4)
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        session = base / "rec_fixture"
        session.mkdir()
        frame_count = 30
        pts_us = np.arange(frame_count, dtype=np.int64) * 33333 + 1_000_000
        frame_ns = pts_us * 1000
        manifest = {
            "schema_version": 2, "app_version": "0.3", "status": "recorded",
            "outcome": "success", "image_orientation_contract": "upright_v1",
            "rotation_baked_into_pixels": True, "decoder_rotation_degrees": 0,
        }
        (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_csv(session / "frames.csv", ["frame_number", "sensor_timestamp_ns"],
                  ({"frame_number": i, "sensor_timestamp_ns": int(frame_ns[i])}
                   for i in range(frame_count)))
        write_csv(session / "encoded.csv", ["sample_index", "pts_us"],
                  ({"sample_index": i, "pts_us": int(pts_us[i])} for i in range(frame_count)))
        for name, fields in (("accelerometer.csv", ["timestamp_ns", "x_m_s2", "y_m_s2", "z_m_s2"]),
                             ("gyroscope.csv", ["timestamp_ns", "x_rad_s", "y_rad_s", "z_rad_s"])):
            write_csv(session / name, fields, ({field: (int(frame_ns[i]) if field == "timestamp_ns" else 0)
                                                for field in fields} for i in range(frame_count)))
        writer = cv2.VideoWriter(str(session / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        assert writer.isOpened()
        for i in range(frame_count):
            frame = np.full((240, 320, 3), (i * 7) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        trajectory = base / "camera_trajectory.csv"
        trajectory_fields = ["frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
                             "x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
        write_csv(trajectory, trajectory_fields, ({
            "frame_idx": i, "timestamp": (pts_us[i] - pts_us[0]) / 1e6,
            "state": 2, "is_lost": "false", "is_keyframe": "false",
            "x": i / 1000, "y": 0, "z": 0, "q_x": 0, "q_y": 0, "q_z": 0, "q_w": 1,
        } for i in range(frame_count)))
        gripper = base / "gripper_width.csv"
        write_csv(gripper, ["frame_index", "marker_detected", "gripper_width_mm"],
                  ({"frame_index": i, "marker_detected": "True", "gripper_width_mm": 20}
                   for i in range(frame_count)))
        plan = base / "episodes.json"
        plan.write_text(json.dumps({"schema_version": 1, "episodes": [{
            "session": str(session), "trajectory": str(trajectory), "gripper": str(gripper),
        }]}), encoding="utf-8")
        # The 1 s fixture is shorter than the production warm-up trim; test the trim separately below.
        dataset_config = json.loads((ROOT / "configs/dataset.json").read_text(encoding="utf-8"))
        assert dataset_config["episode_start_trim_s"] == 3.0
        untrimmed = base / "dataset_untrimmed.json"
        untrimmed.write_text(json.dumps({**dataset_config, "episode_start_trim_s": 0.0}), encoding="utf-8")
        output = base / "dataset.zarr.zip"
        report = build_replay_buffer(plan, output, untrimmed, ROOT / "configs/s22_camera_tcp.json")
        assert report["episodes"] == 1 and report["frames"] == frame_count
        assert report["world_frame"] == "slam_origin_of_each_recording"
        trimmed = base / "dataset_trim_half_second.json"
        trimmed.write_text(json.dumps({**dataset_config, "episode_start_trim_s": 0.5,
                                       "minimum_episode_frames": 10}), encoding="utf-8")
        report_trimmed = build_replay_buffer(plan, base / "dataset_trimmed.zarr.zip", trimmed,
                                             ROOT / "configs/s22_camera_tcp.json")
        expected_kept = int(((pts_us - pts_us[0]) / 1e6 >= 0.5).sum())
        assert report_trimmed["frames"] == expected_kept, (report_trimmed["frames"], expected_kept)
        assert report_trimmed["episode_reports"][0]["trimmed_frames"] == frame_count - expected_kept
        assert report_trimmed["episode_reports"][0]["episode_start_time_s"] >= 0.5
        with zarr.ZipStore(str(output), mode="r") as store:
            root = zarr.group(store=store)
            assert root["data/camera0_rgb"].shape == (frame_count, 224, 224, 3)
            assert root["data/robot0_gripper_width"].shape == (frame_count, 1)
            assert root["meta/episode_ends"][:].tolist() == [frame_count]
            assert np.isclose(root["data/robot0_eef_pos"][0, 2], camera_tcp[2, 3])
    print("PIPELINE_SELF_TEST_OK")


if __name__ == "__main__":
    main()
