"""Lossless arpose.episode/1 compatibility-adapter tests."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track_a.convert.arpose_delivery import (
    normalize_delivery, reviewed_segments_to_rows, seconds_segments_to_rows,
)


class Tests(unittest.TestCase):
    def test_seconds_segments_include_recorded_endpoint(self):
        stamps = [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]
        self.assertEqual(seconds_segments_to_rows(stamps, [[0, 2.0]]), [[0, 3]])
        self.assertEqual(seconds_segments_to_rows(stamps, [[1.0, 3.0]]), [[1, 4]])

    def test_normalizes_without_reencoding_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); delivery = root / "delivery.zip"
            gripper = root / "gripper" / "rec_a"; gripper.mkdir(parents=True)
            (gripper / "gripper.csv").write_text(
                "frame_index,gap_m,status\n0,0.03,D\n1,0.02,D\n", encoding="utf-8")
            meta = {
                "schema": "arpose.episode/1", "episode": "rec_a", "recorded_at_ms": 1,
                "contract": {"pose_frame": "T_world_camera", "translation_units": "m",
                             "quaternion_order": "xyzw", "image_size": [8, 6]},
                "session": {"user_id": "u", "skill_id": "pick_place_cup"},
                "device": "phone", "android": "16",
                "camera": {"modes": "ois=0 eis=0 ae_lock=true awb_lock=true",
                           "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1}},
                "summary": {"usable_segments": [[0, 1]], "frames_dropped": 0},
            }
            poses = ("index,timestamp_ns,tracking,x,y,z,qx,qy,qz,qw,image\n"
                     "0,1000000000,TRACKING,0,0,0,0,0,0,1,000000.jpg\n"
                     "1,2000000000,TRACKING,0,0,0,0,0,0,1,000001.jpg\n")
            images = {"000000.jpg": b"original-jpeg-zero", "000001.jpg": b"original-jpeg-one"}
            with zipfile.ZipFile(delivery, "w") as archive:
                prefix = "delivery/rec_a/"
                archive.writestr(prefix + "episode.json", json.dumps(meta))
                archive.writestr(prefix + "poses.csv", poses)
                archive.writestr(prefix + "imu.csv", "timestamp_ns,sensor,x,y,z\n")
                archive.writestr(prefix + "summary.txt", "ok")
                for name, data in images.items():
                    archive.writestr(prefix + "frames/" + name, data)
            out = root / "normalized"
            result = normalize_delivery(delivery_zip=delivery,
                gripper_root=root / "gripper", output_dir=out, pre_stabilized=True)
            self.assertEqual(len(result["episodes"]), 1)
            quality = json.loads((out / "rec_a.quality.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["warmup_s"], 0)
            self.assertTrue(quality["pre_stabilized"])
            self.assertEqual(quality["usable_segments"], [[0, 2]])
            with zipfile.ZipFile(out / "rec_a.zip") as normalized:
                canonical = json.loads(normalized.read("episode.json"))
                self.assertEqual(canonical["schema"], "umi_raw/0.1.0")
                self.assertEqual(canonical["skill_id"], "pick_place")
                self.assertEqual(canonical["camera"]["rotate_to_canonical_deg"], 180)
                self.assertEqual(len(canonical["source"]["delivery_sha256"]), 64)
                self.assertEqual(len(canonical["source"]["episode_json_sha256"]), 64)
                self.assertEqual(normalized.read("frames/000000.jpg"), images["000000.jpg"])
                frames = list(csv.DictReader(io.StringIO(normalized.read("frames.csv").decode())))
                self.assertEqual(frames[0]["image"], "frames/000000.jpg")

    def test_zero_warmup_requires_explicit_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                normalize_delivery(delivery_zip=Path(directory) / "none.zip",
                    gripper_root=Path(directory), output_dir=Path(directory) / "out",
                    pre_stabilized=False)

    def test_empty_tracking_segments_reject_episode_not_delivery(self):
        # The real rec_20260911_145737 has no usable segment after its 706.7 mm
        # tracking jump.  Batch conversion must quarantine it, not bless it or
        # discard all otherwise healthy recordings.
        self.assertIsNone(reviewed_segments_to_rows(
            [1_000_000_000, 2_000_000_000], []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
