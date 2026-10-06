"""Focused unit checks for provisional relative-chunk registration search."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sim.mujoco.build_scene import load_config
from tools.umi_mujoco import apply_kinematic_grasp

from search_relative_chunk_registration import (
    PROGRESS_SCHEMA,
    _approach_start_row,
    _cached_candidate,
    _canonical_replay_translation,
    _candidate_starts,
    _dynamic_rank,
    _apply_grip_preload,
    _load_progress,
    _planar_contact_offsets,
    _refined_starts,
    _require_source_registration,
    _save_candidate,
    _scene_gate,
    _static_gate_passed,
    _static_rank,
    _table_placement_metrics,
    _translated_episode_anchors,
    _workspace_audit,
)


class Tests(unittest.TestCase):
    def test_approach_start_uses_fixed_lookback_not_reachability(self) -> None:
        self.assertEqual(
            _approach_start_row(
                contact_row=12, episode_length=20, lookback_rows=8), 4)
        self.assertEqual(
            _approach_start_row(
                contact_row=3, episode_length=20, lookback_rows=8), 0)
        self.assertEqual(
            _approach_start_row(
                contact_row=12, episode_length=20, lookback_rows=None), 0)

    def test_grip_preload_changes_execution_copy_only_below_threshold(self) -> None:
        source = np.asarray([0.060, 0.055, 0.040, 0.001])
        executed = _apply_grip_preload(
            source, preload_m=0.002, activation_gap_m=0.055)
        np.testing.assert_allclose(executed, [0.060, 0.053, 0.038, 0.0])
        np.testing.assert_allclose(source, [0.060, 0.055, 0.040, 0.001])

    def test_diagnostic_candidate_can_be_rechecked_but_validated_cannot(self) -> None:
        _require_source_registration({
            "status": "DIAGNOSTIC_COLLISION_PROXY_CANDIDATE_NOT_PHYSICAL_VALIDATION"
        })
        with self.assertRaises(ValueError):
            _require_source_registration({"status": "VALIDATED"})

    def test_task_frame_beats_table_centrality_when_full_gates_tie(self) -> None:
        common = {
            "combined_accepted_fraction": 0.0,
            "scene_constraint_episode_fraction": 0.0,
            "first_approach_contact_body_depth_centered_episode_fraction": 0.0,
            "pad_centered_episode_fraction": 0.0,
            "contact_ik_available_episode_fraction": 0.0,
            "pad_centered_given_contact_ik_fraction": 0.0,
            "scene_component_fraction": 0.2,
            "oracle_geometry_accepted_fraction": 0.75,
            "anchor_accepted_fraction": 1.0,
            "start_table_contacts": 0,
            "start_self_contacts": 0,
        }
        side = {
            **common,
            "candidate_id": 1,
            "jaw_yaw_aligned_episode_fraction": 1.0,
            "approach_axis_aligned_episode_fraction": 1.0,
            "up_axis_aligned_episode_fraction": 1.0,
            "side_grasp_episode_fraction": 1.0,
            "object_table_margin_episode_fraction": 0.0,
            "mean_table_centre_distance_norm": 1.0,
            "minimum_table_edge_clearance_m": -0.01,
        }
        centred_top_down = {
            **common,
            "candidate_id": 2,
            "jaw_yaw_aligned_episode_fraction": 0.0,
            "approach_axis_aligned_episode_fraction": 1.0,
            "up_axis_aligned_episode_fraction": 0.0,
            "side_grasp_episode_fraction": 0.0,
            "object_table_margin_episode_fraction": 1.0,
            "mean_table_centre_distance_norm": 0.0,
            "minimum_table_edge_clearance_m": 0.2,
        }
        self.assertGreater(_static_rank(side), _static_rank(centred_top_down))

    def test_static_gate_requires_configured_episode_fraction(self) -> None:
        report = {
            "combined_accepted_fraction": 0.20,
            "scene_constraint_episode_fraction": 3 / 14,
        }
        self.assertFalse(_static_gate_passed(report, 0.8))
        report["scene_constraint_episode_fraction"] = 12 / 14
        self.assertTrue(_static_gate_passed(report, 0.8))

    def test_table_margin_uses_full_object_footprint(self) -> None:
        metrics = _table_placement_metrics(
            object_xyz=np.asarray([0.52639255, -0.00444458, 0.03835]),
            object_size_m=np.asarray([0.039, 0.039, 0.0767]),
            table_xy=np.asarray([0.15, 0.0]),
            table_half_xy=np.asarray([0.4, 0.4]),
            edge_margin_m=0.03,
        )
        self.assertFalse(metrics["object_on_table_with_margin"])
        self.assertAlmostEqual(
            metrics["minimum_table_edge_clearance_m"], 0.00410745,
            places=7,
        )

    def test_table_margin_accepts_central_object(self) -> None:
        metrics = _table_placement_metrics(
            object_xyz=np.asarray([0.30, 0.02, 0.03835]),
            object_size_m=np.asarray([0.039, 0.039, 0.0767]),
            table_xy=np.asarray([0.15, 0.0]),
            table_half_xy=np.asarray([0.4, 0.4]),
            edge_margin_m=0.03,
        )
        self.assertTrue(metrics["object_on_table_with_margin"])
        self.assertGreaterEqual(
            metrics["minimum_table_edge_clearance_m"], 0.03)

    def test_first_contact_centre_is_a_hard_scene_gate(self) -> None:
        accepted, fraction, failed = _scene_gate({
            "upright_body_side_grasp": True,
            "first_approach_contact_detected": True,
            "first_approach_contact_centered": False,
            "first_approach_contact_body_depth_centered": True,
        })
        self.assertFalse(accepted)
        self.assertEqual(failed, ["first_approach_contact_centered"])
        self.assertAlmostEqual(fraction, 0.75)

    def test_side_contact_centring_excludes_expected_approach_depth(self) -> None:
        approach, lateral = _planar_contact_offsets(
            np.asarray([0.020, 0.003, 0.006]),
            np.asarray([1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(approach, 0.020)
        self.assertAlmostEqual(lateral, 0.003)
        self.assertLess(lateral, 0.008)
        self.assertGreater(np.linalg.norm([0.020, 0.003]), 0.008)

    def test_planar_contact_offset_rotates_with_approach_axis(self) -> None:
        approach, lateral = _planar_contact_offsets(
            np.asarray([0.004, -0.020, 0.0]),
            np.asarray([0.0, -1.0, 0.0]),
        )
        self.assertAlmostEqual(approach, 0.020)
        self.assertAlmostEqual(lateral, 0.004)

    def test_workspace_audit_reports_common_translation_and_failures(self) -> None:
        rows = [{
            "object_xyz_m": [0.20, -0.05, 0.04],
            "contact_displacement_from_common_start_m": [0.10, 0.0, -0.01],
            "grasp_height_error_m": 0.002,
            "contact_target_gap_m": 0.038,
            "minimum_recorded_gap_m": 0.036,
            "first_approach_contact": {"detected": True},
            "scene_failure_reasons": ["contact_ik_available"],
        }, {
            "object_xyz_m": [0.30, 0.05, 0.04],
            "contact_displacement_from_common_start_m": [0.20, 0.10, -0.01],
            "grasp_height_error_m": 0.006,
            "contact_target_gap_m": 0.041,
            "minimum_recorded_gap_m": 0.040,
            "first_approach_contact": {"detected": False},
            "scene_failure_reasons": ["first_approach_contact_centered"],
        }]
        audit = _workspace_audit(
            episode_reports=rows,
            table_xy=np.asarray([0.25, 0.0]),
            table_half_xy=np.asarray([0.20, 0.20]),
            object_size_m=np.asarray([0.04, 0.04, 0.08]),
            edge_margin_m=0.03,
        )
        self.assertTrue(
            audit["common_xy_translation_for_table_margin_m"]["exists"])
        np.testing.assert_allclose(
            audit["inferred_object_xyz_m"]["span"], [0.1, 0.1, 0.0])
        self.assertEqual(
            audit["scene_failure_counts"]["contact_ik_available"], 1)
        self.assertEqual(
            audit["recorded_gap_contact_gate"]["contact_detected_episodes"], 1)

    def test_canonical_replay_translates_complete_trajectory(self) -> None:
        class Solution:
            converged = True
            position_error_m = 0.0
            axis_error_deg = 0.0
            roll_error_deg = 0.0

            def __init__(self, positions_rad: np.ndarray) -> None:
                self.positions_rad = positions_rad

        class Adapter:
            targets: list[np.ndarray] = []

            def __init__(self, ik: object, curve: object) -> None:
                del ik, curve

            def solve(self, target: np.ndarray, *, seed_rad: np.ndarray,
                      gripper_width_m: float) -> Solution:
                del gripper_width_m
                self.targets.append(target.copy())
                return Solution(np.asarray(seed_rad, dtype=float) + 0.01)

        poses = [np.eye(4), np.eye(4)]
        poses[1][:3, 3] = [0.1, -0.2, 0.3]
        translation = np.asarray([0.4, 0.05, -0.1])
        with mock.patch(
                "search_relative_chunk_registration.MujocoArmAdapter",
                Adapter):
            translated, arms = _translated_episode_anchors(
                poses=poses, gaps=np.asarray([0.06, 0.04]),
                translation_m=translation, ik=object(), curve=object(),
                ranges=np.asarray([[-1.0, 1.0]] * 5),
                seed_arm=np.zeros(5),
            )
        np.testing.assert_allclose(
            translated[0][:3, 3], poses[0][:3, 3] + translation)
        np.testing.assert_allclose(
            translated[1][:3, 3], poses[1][:3, 3] + translation)
        np.testing.assert_allclose(
            translated[1][:3, 3] - translated[0][:3, 3],
            poses[1][:3, 3] - poses[0][:3, 3])
        self.assertTrue(all(value is not None for value in arms))

    def test_canonical_replay_can_explicitly_project_side_frame(self) -> None:
        class Solution:
            converged = True
            position_error_m = 0.0
            axis_error_deg = 0.0
            roll_error_deg = 0.0

            def __init__(self, positions_rad: np.ndarray) -> None:
                self.positions_rad = positions_rad

        class Adapter:
            def __init__(self, ik: object, curve: object) -> None:
                del ik, curve

            def solve(self, target: np.ndarray, *, seed_rad: np.ndarray,
                      gripper_width_m: float) -> Solution:
                del target, gripper_width_m
                return Solution(np.asarray(seed_rad, dtype=float))

        poses = [np.eye(4), np.eye(4)]
        poses[1][:3, :3] = np.diag([1.0, -1.0, -1.0])
        side_frame = np.asarray([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        with mock.patch(
                "search_relative_chunk_registration.MujocoArmAdapter",
                Adapter):
            projected, arms = _translated_episode_anchors(
                poses=poses, gaps=np.asarray([0.06, 0.04]),
                translation_m=np.asarray([0.1, -0.2, 0.3]),
                ik=object(), curve=object(),
                ranges=np.asarray([[-1.0, 1.0]] * 5),
                seed_arm=np.zeros(5),
                rotation_world_eef=side_frame,
            )
        for pose in projected:
            np.testing.assert_allclose(pose[:3, :3], side_frame)
        self.assertTrue(all(value is not None for value in arms))

    def test_canonical_replay_rejects_reflection_as_side_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            _translated_episode_anchors(
                poses=[np.eye(4)], gaps=np.asarray([0.04]),
                translation_m=np.zeros(3), ik=object(), curve=object(),
                ranges=np.asarray([[-1.0, 1.0]] * 5),
                seed_arm=np.zeros(5),
                rotation_world_eef=np.diag([1.0, 1.0, -1.0]),
            )

    def test_canonical_alignment_separates_xy_only_from_3d(self) -> None:
        common = {
            "contact_position_m": np.asarray([0.30, -0.02, 0.12]),
            "desired_world_approach_axis": np.asarray([1.0, 0.0, 0.0]),
            "object_approach_offset_m": -0.02,
            "canonical_object_xyz_m": np.asarray([0.40, 0.0, 0.04]),
        }
        xy_only = _canonical_replay_translation(
            **common, align_contact_to_body_center_3d=False)
        aligned_3d = _canonical_replay_translation(
            **common, align_contact_to_body_center_3d=True)
        np.testing.assert_allclose(xy_only, [0.12, 0.02, 0.0])
        np.testing.assert_allclose(aligned_3d, [0.12, 0.02, -0.08])

    def test_handoff_overlay_replaces_legacy_pinch_frame(self) -> None:
        ai_root = Path(__file__).resolve().parents[1]
        cfg = load_config(ai_root / "configs" / "so101.yaml")
        payload = json.loads((
            ai_root / "configs" / "real"
            / "so101_ver1_phone_holder_kinematic.json"
        ).read_text(encoding="utf-8"))
        apply_kinematic_grasp(cfg, payload)
        np.testing.assert_allclose(
            cfg["grasp"]["pinch_offset_local"],
            [2.1e-7, -5.64e-7, -0.158118819], atol=1e-12)
        np.testing.assert_allclose(
            cfg["grasp"]["eef_frame_local"]["jaw_axis"],
            [-0.048656744, 0.998815559, -0.000003629], atol=1e-12)
        self.assertEqual(
            cfg["grasp"]["gap_dependent_pinch_midpoint"]["kind"],
            "symmetric_parallel_jaw")
        self.assertFalse(
            cfg["grasp"]["kinematic_overlay"]["dynamic_mujoco_ready"])

    def test_progress_round_trip_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "search.progress.json"
            payload = {"arguments": {"candidates": 2}}
            state = _load_progress(
                path=path, resume=False, signature="abc",
                signature_payload=payload)
            self.assertEqual(state["schema"], PROGRESS_SCHEMA)
            report = {"candidate_id": 3, "start_arm_rad": [0, 1, 2, 3, 4]}
            _save_candidate(
                state=state, progress_path=path, stage="screen",
                report=report)

            resumed = _load_progress(
                path=path, resume=True, signature="abc",
                signature_payload=payload)
            cached = _cached_candidate(
                state=resumed, stage="screen", candidate_id=3,
                arm=np.asarray(report["start_arm_rad"], dtype=float))
            self.assertEqual(cached, report)

    def test_resume_rejects_changed_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "search.progress.json"
            _load_progress(
                path=path, resume=False, signature="old",
                signature_payload={})
            with self.assertRaisesRegex(ValueError, "signature differs"):
                _load_progress(
                    path=path, resume=True, signature="new",
                    signature_payload={})

    def test_cached_candidate_rejects_changed_seed(self) -> None:
        state = {
            "stages": {"screen": {"0": {
                "candidate_id": 0,
                "start_arm_rad": [0, 0, 0, 0, 0],
            }}},
        }
        with self.assertRaisesRegex(ValueError, "different arm seed"):
            _cached_candidate(
                state=state, stage="screen", candidate_id=0,
                arm=np.ones(5))

    def test_dynamic_rank_penalises_simultaneous_tip(self) -> None:
        def report(tipped: int) -> dict:
            return {"dynamic_oracle": {
                "passed": False,
                "success_fraction": 1 / 3,
                "runner_errors": 0,
                "failure_counts": {
                    "ik_reject_after_contact": 1,
                    "object_tipped": tipped,
                    "object_displaced": tipped,
                    "static_scene_reject": 0,
                },
                "contacts": 3,
                "mean_max_lift_height_m": 0.05,
            }}
        self.assertGreater(
            _dynamic_rank(report(1), (0.0,)),
            _dynamic_rank(report(2), (1.0,)),
        )

    def test_source_registration_is_not_clipped_to_search_margin(self) -> None:
        ranges = np.asarray([[-2.0, 2.0]] * 5, dtype=float)
        source = np.asarray([0.0, 0.0, 0.0, -1.9, 1.9], dtype=float)
        candidates, targeted = _candidate_starts(
            source, ranges, count=1, seed=7, ik=None,
            initial_gap_m=0.06, targeted_start_rotation=np.eye(3))
        self.assertEqual(targeted, 0)
        np.testing.assert_array_equal(candidates[0], source)

    def test_cartesian_registration_candidates_preserve_source_rotation(self) -> None:
        class Solution:
            converged = True
            pos_error_m = 0.0
            axis_error_deg = 0.0
            roll_error_deg = 0.0

            def __init__(self, q_rad: np.ndarray) -> None:
                self.q_rad = q_rad

        class IK:
            cfg = {"grasp": {"gap_curve": [[0.0, 0.0], [1.0, 9.0]]}}

            def __init__(self) -> None:
                self.targets: list[np.ndarray] = []

            def forward_pose(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                del q_rad
                return np.asarray([0.30, 0.0, 0.06]), np.asarray([1.0, 0.0, 0.0, 0.0])

            def solve(self, xyz: np.ndarray, quat: np.ndarray, gap: float,
                      q_init: np.ndarray) -> Solution:
                del quat, gap, q_init
                self.targets.append(np.asarray(xyz, dtype=float))
                index = len(self.targets)
                return Solution(np.asarray([index * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))

        ik = IK()
        candidates, targeted = _candidate_starts(
            np.zeros(5), np.asarray([[-2.0, 2.0]] * 5), count=5,
            seed=7, ik=ik, initial_gap_m=0.06,
            targeted_start_rotation=np.eye(3))
        self.assertEqual(targeted, 3)
        self.assertEqual(len(candidates), 5)
        np.testing.assert_allclose(ik.targets[0], [0.26, 0.0, 0.06])
        np.testing.assert_allclose(ik.targets[1], [0.26, -0.04, 0.06])
        np.testing.assert_allclose(ik.targets[2], [0.26, 0.04, 0.06])

    def test_refinement_keeps_valid_near_limit_seed_nearby(self) -> None:
        ranges = np.asarray([[-2.0, 2.0]] * 5, dtype=float)
        source = np.asarray([0.0, 0.0, 0.0, -1.9, 1.9], dtype=float)
        refined = _refined_starts(
            seeds=[source], ranges=ranges, existing=[source], count=4,
            sigma_fraction=0.001, seed=8)
        self.assertEqual(len(refined), 4)
        self.assertTrue(all(np.linalg.norm(value - source) < 0.05
                            for value in refined))


if __name__ == "__main__":
    unittest.main(verbosity=2)
