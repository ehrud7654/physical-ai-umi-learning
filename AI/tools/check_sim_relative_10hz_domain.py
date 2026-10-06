"""Small tests for source-episode split and 10Hz visual comparison."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.preflight_sim_relative_10hz_domain import (
    _bbox_xywh, _compare_episode, first_index_per_source_row, preflight,
    source_episode_split,
)


class SimRelativeDomainTests(unittest.TestCase):
    def test_split_keeps_source_episode_whole(self) -> None:
        names = [f"episode_{index}" for index in range(7)]
        split = source_episode_split(names, seed=0)
        train = set(split["train_source_episodes"])
        holdout = set(split["holdout_source_episodes"])
        self.assertEqual(train | holdout, set(names))
        self.assertFalse(train & holdout)
        self.assertEqual(len(holdout), 2)
        self.assertEqual(split, source_episode_split(names[::-1], seed=0))

    def test_repeated_robot_time_rows_are_counted_once(self) -> None:
        self.assertEqual(first_index_per_source_row(
            np.asarray([7, 7, 8, 8, 8, 9])), [0, 2, 5])

    def test_identical_real_and_sim_yellow_object(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[40:170, 70:150] = [230, 170, 20]
        self.assertIsNotNone(_bbox_xywh(image))
        chw = image.transpose(2, 0, 1)
        real_images = np.stack([chw, chw])[None].repeat(2, axis=0)
        sim_images = np.stack([chw, chw])[None].repeat(3, axis=0)
        with tempfile.TemporaryDirectory() as root:
            real_path = Path(root) / "real.npz"
            sim_path = Path(root) / "sim.npz"
            np.savez_compressed(real_path, image=real_images)
            np.savez_compressed(sim_path, image=sim_images,
                                source_row=np.asarray([0, 0, 1]))
            result = _compare_episode(real_path, sim_path)
        self.assertEqual(result["candidate_rows"], 3)
        self.assertEqual(result["distinct_source_rows"], 2)
        self.assertEqual(result["paired"]["paired_source_rows"], 2)
        self.assertEqual(result["paired"]["bbox_center_l2_norm_median"], 0.0)
        self.assertEqual(result["problems"], [])

    def test_preflight_keeps_all_offsets_in_one_fold(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[40:170, 70:150] = [230, 170, 20]
        chw = image.transpose(2, 0, 1)
        real_images = np.stack([chw, chw])[None].repeat(2, axis=0)
        sim_images = np.stack([chw, chw])[None].repeat(2, axis=0)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            cases = []
            for episode in ("a", "b", "c"):
                np.savez_compressed(directory / f"{episode}.npz", image=real_images)
                candidate = directory / f"{episode}_candidate.npz"
                np.savez_compressed(candidate, image=sim_images,
                                    source_row=np.asarray([0, 1]))
                for offset in ([0.0, 0.0], [0.01, 0.0]):
                    cases.append({"episode": episode, "offset_xy_m": offset,
                                  "status": "CANDIDATE_CHECKED",
                                  "candidate": str(candidate)})
            for offset in ([0.0, 0.0], [0.01, 0.0]):
                cases.append({"episode": "short", "offset_xy_m": offset,
                              "status": "RUNNER_ERROR"})
            suite = directory / "suite.json"
            suite.write_text(json.dumps({
                "status": "COMPLETE_WITH_REJECTIONS_POLICY_FREE_10HZ_DIAGNOSTIC",
                "training_ready": False,
                "conditions": {"episodes": ["a", "b", "c", "short"],
                               "offsets_xy_m": [[0.0, 0.0], [0.01, 0.0]]},
                "cases": cases,
            }), encoding="utf-8")
            result = preflight(suite, directory)
        split = result["source_episode_split"]
        self.assertFalse(set(split["train_source_episodes"])
                         & set(split["holdout_source_episodes"]))
        self.assertEqual(sum(result["case_fold_counts"].values()), 6)
        self.assertEqual(result["ineligible_source_episodes"], ["short"])
        self.assertEqual(result["visual_comparison"]["paired_source_rows"], 6)
        self.assertFalse(result["training_ready"])


if __name__ == "__main__":
    unittest.main()
