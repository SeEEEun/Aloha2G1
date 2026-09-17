from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.evaluation.contracts import PHYSICAL_SUCCESS, SEMANTIC_SUCCESS, sha256_file
from tools.evaluation.metrics import (
    action_prediction_metrics,
    aggregate_episode_metrics,
    bimanual_relation_metrics,
    feasibility_metrics,
    handoff_ordering_metrics,
    path_efficiency_metrics,
    smoothness_metrics,
    chunk_smoothness_metrics,
    whole_hand_metrics,
    wrist_metrics,
    evaluate_task_sequence,
)
from tools.evaluation.paired_statistics import paired_statistics
from tools.evaluation.physical_success import detect_physical_success, validate_event_log
from tools.evaluation.semantic_sequence import semantic_phase_events
from tools.evaluation.tables import generate_tables
from tools.evaluation.io import evaluate_bundle
from tools.freeze_doll_handoff_rigid_proxy import freeze
from tools.calibrate_dex3_rigid_doll_grasp import summarize


ROOT = Path("/home/jbnu/aloha_g1_dataset")


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zeros = {side: np.zeros((10, 3)) for side in ("left", "right")}

    def test_perfect_cartesian_prediction_is_zero(self) -> None:
        wrist = wrist_metrics(self.zeros, self.zeros)
        hand = whole_hand_metrics(
            self.zeros,
            self.zeros,
            authoritative_definition=True,
            definition_provenance="authoritative-test-frame",
        )
        relation = bimanual_relation_metrics(self.zeros, self.zeros)
        self.assertEqual(wrist["combined"]["position_error_mm"]["max"], 0.0)
        self.assertEqual(hand["combined"]["position_error_mm"]["rmse"], 0.0)
        self.assertEqual(relation["error_mm"]["p95"], 0.0)

    def test_action_prediction_and_nrmse(self) -> None:
        truth = np.zeros((2, 4, 28))
        prediction = np.ones_like(truth) * 0.1
        result = action_prediction_metrics(truth, prediction, np.ones(28) * 2.0)
        self.assertAlmostEqual(result["OVERALL_28D_RMSE_rad"], 0.1)
        self.assertAlmostEqual(result["ARM_14D_RMSE_rad"], 0.1)
        self.assertAlmostEqual(result["DEX3_14D_RMSE_rad"], 0.1)
        self.assertAlmostEqual(result["FIRST_ACTION_RMSE_rad"], 0.1)
        self.assertAlmostEqual(result["FIRST4_RMSE_rad"], 0.1)
        self.assertAlmostEqual(result["FULL_CHUNK_RMSE_rad"], 0.1)
        self.assertAlmostEqual(result["NRMSE_percent"]["overall_28d"], 5.0)

    def test_chunk_padding_is_not_scored(self) -> None:
        truth = np.zeros((2, 5, 28))
        prediction = np.zeros_like(truth)
        prediction[1, 2:] = 100.0
        result = action_prediction_metrics(
            truth,
            prediction,
            np.ones(28),
            valid_lengths=np.array([5, 2]),
        )
        self.assertEqual(result["FULL_CHUNK_RMSE_rad"], 0.0)
        self.assertEqual(result["FIRST_ACTION_RMSE_rad"], 0.0)

    def test_path_efficiency_contract(self) -> None:
        reference = {
            "left": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "right": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        }
        twice = {
            "left": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "right": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        }
        result = path_efficiency_metrics(
            reference, twice, success=1, success_kind=SEMANTIC_SUCCESS
        )
        self.assertEqual(result["RPL"], 2.0)
        self.assertEqual(result["SWPE"], 0.5)
        self.assertEqual(
            path_efficiency_metrics(
                reference, twice, success=0, success_kind=SEMANTIC_SUCCESS
            )["SWPE"],
            0.0,
        )
        perfect = path_efficiency_metrics(
            reference, reference, success=1, success_kind=PHYSICAL_SUCCESS
        )
        self.assertEqual(perfect["RPL"], 1.0)
        self.assertEqual(perfect["SWPE"], 1.0)

    def test_ordered_sequence_and_handoff(self) -> None:
        phases = (
            "LEFT_APPROACH",
            "LEFT_GRASP",
            "LEFT_TRANSPORT",
            "RIGHT_APPROACH",
            "DUAL_CONTACT",
            "RIGHT_OWNED",
            "RIGHT_TRANSPORT",
            "RELEASE",
        )
        sequence = evaluate_task_sequence(
            {phase: index for index, phase in enumerate(phases)},
            success_kind=SEMANTIC_SUCCESS,
            authoritative=True,
        )
        self.assertEqual(sequence["SEMANTIC_TASK_SEQUENCE_SUCCESS"], 1)
        self.assertEqual(sequence["phase_completion_score"], 1.0)
        self.assertEqual(sequence["last_successfully_completed_phase"], "RELEASE")
        correct = handoff_ordering_metrics(10, 14, authoritative=True)
        reversed_order = handoff_ordering_metrics(14, 10, authoritative=True)
        self.assertEqual(correct["score"], 1)
        self.assertEqual(correct["acquisition_to_release_margin_frames"], 4)
        self.assertEqual(reversed_order["score"], 0)
        self.assertEqual(reversed_order["negative_margin"], 1)

    def test_smoothness_and_feasibility(self) -> None:
        result = smoothness_metrics(np.zeros((12, 28)))
        self.assertEqual(result["JOINT_JERK_RMS_rad_s3"], 0.0)
        self.assertEqual(result["DIRECTION_REVERSAL_RATE_per_joint_s"], 0.0)
        feasible = feasibility_metrics(
            {
                "hard_ik_failure_count": 0,
                "hard_collision_count": 0,
                "joint_limit_failure_count": 0,
                "branch_discontinuity_count": 0,
            },
            np.zeros(12),
        )
        self.assertEqual(feasible["FEASIBLE"], 1)
        self.assertEqual(feasible["projection_magnitude_mm"]["max"], 0.0)
        chunks = chunk_smoothness_metrics(np.zeros((2, 8, 28)))
        self.assertEqual(chunks["JOINT_JERK_RMS_rad_s3"], 0.0)
        self.assertFalse(chunks["derivatives_cross_chunk_boundaries"])

    def test_dataset_level_task_rate_and_pcs_summary(self) -> None:
        def row(identity: str, score: float, success: int, last: str) -> dict:
            return {
                "source_episode_id": identity,
                "metrics": {
                    "task_sequence": {
                        "status": "READY",
                        "success_kind": SEMANTIC_SUCCESS,
                        "phase_completion_score": score,
                        "task_sequence_success": success,
                        "last_successfully_completed_phase": last,
                    },
                    "handoff_ordering": {"status": "NA"},
                    "feasibility": {"status": "NA"},
                },
            }

        aggregate = aggregate_episode_metrics(
            [row("ep0", 1.0, 1, "RELEASE"), row("ep1", 0.5, 0, "RIGHT_APPROACH")]
        )
        self.assertEqual(
            aggregate["task_sequence"]["SEMANTIC_TASK_SEQUENCE_SUCCESS_RATE"], 0.5
        )
        self.assertEqual(aggregate["task_sequence"]["phase_completion_score"]["median"], 0.75)

    def test_explicit_semantic_contract_perfect_reference(self) -> None:
        path = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories/episode_000002.npz"
        config = json.loads(
            (ROOT / "configs/paper_semantic_task_sequence_v1.json").read_text(encoding="utf-8")
        )
        with np.load(path, allow_pickle=False) as archive:
            q = archive["replay_named_joint_qpos"]
            events = {
                str(name): int(frame)
                for name, frame in zip(archive["event_names"], archive["event_frames"], strict=True)
            }
            semantic = {
                key: archive[key]
                for key in ("left_hand_phase", "right_hand_phase", "ownership_state")
            }
        result = semantic_phase_events(
            q, q, event_frames=events, semantic_arrays=semantic, config=config
        )
        self.assertTrue(all(value is not None for value in result["canonical_phase_events_frame"].values()))
        self.assertFalse(result["physical_contact_or_ownership_inferred"])


class PairedStatisticsTests(unittest.TestCase):
    @staticmethod
    def _row(identity: str, value: float) -> dict:
        return {"source_episode_id": identity, "metrics": {"score": value}}

    def test_identical_pairs_have_zero_delta_and_ci(self) -> None:
        rows = [self._row("ep0", 1.0), self._row("ep1", 2.0)]
        result = paired_statistics(rows, rows, seed=20260826, resamples=200)
        metric = result["metrics"]["score"]
        self.assertEqual(metric["paired_B_minus_A"]["mean"], 0.0)
        self.assertEqual(metric["paired_bootstrap_95_percent_CI_of_mean_B_minus_A"], [0.0, 0.0])

    def test_unmatched_episode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmatched comparison is forbidden"):
            paired_statistics(
                [self._row("ep0", 1.0)], [self._row("ep1", 1.0)], resamples=10
            )


class PortableBundleTests(unittest.TestCase):
    def test_one_evaluator_scores_a_complete_synthetic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positions = np.column_stack((np.arange(10, dtype=float), np.zeros((10, 2))))
            arrays = root / "episode.npz"
            np.savez_compressed(
                arrays,
                reference_left_wrist_position_m=positions,
                reference_right_wrist_position_m=positions,
                candidate_left_wrist_position_m=positions,
                candidate_right_wrist_position_m=positions,
                reference_left_whole_hand_position_m=positions,
                reference_right_whole_hand_position_m=positions,
                candidate_left_whole_hand_position_m=positions,
                candidate_right_whole_hand_position_m=positions,
                reference_action_rad=np.zeros((10, 28)),
                candidate_action_rad=np.zeros((10, 28)),
                candidate_q_rad=np.zeros((10, 28)),
                feasibility_projection_m=np.zeros((10, 2)),
            )
            phases = (
                "LEFT_APPROACH",
                "LEFT_GRASP",
                "LEFT_TRANSPORT",
                "RIGHT_APPROACH",
                "DUAL_CONTACT",
                "RIGHT_OWNED",
                "RIGHT_TRANSPORT",
                "RELEASE",
            )
            bundle = {
                "schema_version": "paper_evaluation_bundle_v1",
                "evaluation_mode": "source_conditioned",
                "success_kind": SEMANTIC_SUCCESS,
                "episodes": [
                    {
                        "source_episode_id": "synthetic_ep0",
                        "arrays_path": str(arrays),
                        "annotations": {
                            "candidate_phase_events_frame": {
                                phase: index for index, phase in enumerate(phases)
                            },
                            "phase_events_authoritative": True,
                            "right_acquire_frame": 4,
                            "left_release_frame": 5,
                            "handoff_events_authoritative": True,
                        },
                        "feasibility": {
                            "hard_ik_failure_count": 0,
                            "hard_collision_count": 0,
                            "joint_limit_failure_count": 0,
                            "branch_discontinuity_count": 0,
                        },
                        "provenance": {
                            "whole_hand_frame_authoritative": True,
                            "whole_hand_frame_definition": "synthetic authoritative frame",
                        },
                    }
                ],
            }
            manifest = root / "bundle.json"
            manifest.write_text(json.dumps(bundle), encoding="utf-8")
            result = evaluate_bundle(manifest, np.ones(28))
            metrics = result["episodes"][0]["metrics"]
            self.assertEqual(metrics["wrist"]["combined"]["position_error_mm"]["max"], 0.0)
            self.assertEqual(metrics["action_prediction"]["OVERALL_28D_RMSE_rad"], 0.0)
            self.assertEqual(result["aggregate"]["task_sequence"]["TASK_SEQUENCE_SUCCESS_RATE"], 1.0)


class PhysicalDetectorTests(unittest.TestCase):
    def _successful_log(self, config: dict) -> dict[str, np.ndarray]:
        frames = 170
        com = np.repeat(
            np.asarray(config["object"]["initial_center_world_xyz_m"], dtype=float)[None],
            frames,
            axis=0,
        )
        com[30:110, 2] = 0.90
        transport = config["success_thresholds"]["right_transport_target_volume_world_m"]
        com[110:130] = (
            np.asarray(transport["lower_xyz_m"]) + np.asarray(transport["upper_xyz_m"])
        ) / 2.0
        release = config["success_thresholds"]["release_bin_interior_volume_world_m"]
        com[130:] = (
            np.asarray(release["lower_xyz_m"]) + np.asarray(release["upper_xyz_m"])
        ) / 2.0
        left = np.zeros(frames, dtype=bool)
        right = np.zeros(frames, dtype=bool)
        left[15:75] = True
        right[65:130] = True
        table = np.zeros(frames, dtype=bool)
        table[:25] = True
        left_pos = com + np.array([0.2, 0.0, 0.0])
        right_pos = com + np.array([0.2, 0.0, 0.0])
        left_pos[5:] = com[5:] + np.array([0.02, 0.0, 0.0])
        right_pos[55:] = com[55:] + np.array([0.02, 0.0, 0.0])
        orientation = np.zeros((frames, 4))
        orientation[:, 3] = 1.0
        return {
            "object_com_m": com,
            "object_orientation_xyzw": orientation,
            "object_linear_velocity_m_s": np.zeros((frames, 3)),
            "object_angular_velocity_rad_s": np.zeros((frames, 3)),
            "left_hand_object_contact": left,
            "right_hand_object_contact": right,
            "table_contact": table,
            "bin_contact": np.arange(frames) >= 135,
            "hand_joint_q_rad": np.zeros((frames, 14)),
            "arm_joint_q_rad": np.zeros((frames, 14)),
            "left_hand_position_m": left_pos,
            "right_hand_position_m": right_pos,
            "right_hand_open": np.arange(frames) >= 130,
        }

    def test_full_physical_success(self) -> None:
        config = json.loads(
            (ROOT / "configs/doll_handoff_rigid_proxy_v1.json").read_text(encoding="utf-8")
        )
        log = self._successful_log(config)
        self.assertEqual(validate_event_log(log)["status"], "PASS")
        result = detect_physical_success(log, config)
        self.assertEqual(result["FULL_PHYSICAL_SUCCESS"], 1)
        self.assertEqual(result["task_sequence"]["PHYSICAL_TASK_SUCCESS"], 1)
        self.assertEqual(result["task_sequence"]["phase_completion_score"], 1.0)


class TableAndFreezeTests(unittest.TestCase):
    def test_missing_table_inputs_are_na(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = generate_tables(root)
            self.assertEqual(manifest["tables"]["table4_isaac_physical_task_success"]["status"], "READY_TEMPLATE_RESULTS_NA")
            payload = json.loads(
                (root / "table4_isaac_physical_task_success.json").read_text(encoding="utf-8")
            )
            self.assertTrue(all(row["A mean ± std"] == "NA" for row in payload["rows"]))

    def test_freeze_requires_and_records_bilateral_policy_independent_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "candidate.json"
            config_path.write_bytes(
                (ROOT / "configs/doll_handoff_rigid_proxy_v1.json").read_bytes()
            )
            config_sha = sha256_file(config_path)
            primitive = root / "primitive.npz"
            event_log = root / "event_log.npz"
            primitive.write_bytes(b"fixed primitive")
            event_log.write_bytes(b"physics log")
            scene = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
            candidate = json.loads(config_path.read_text())["material_candidates"][1]
            result_paths = {}
            for side in ("left", "right"):
                result = {
                    "schema_version": "doll_handoff_rigid_proxy_grasp_calibration_v1",
                    "status": "PASS",
                    "side": side,
                    "material_candidate": "MEDIUM",
                    "LEFT_LIFT_PASS": side == "left",
                    "RIGHT_LIFT_PASS": side == "right",
                    "artifact_checks": {
                        "penetration_artifact_detected": False,
                        "magnetic_or_sticky_behavior_detected": False,
                        "teleportation_detected": False,
                        "object_constraint_attachment_detected": False,
                        "object_pose_writes_during_timed_loop": 0,
                    },
                    "runtime_material": {
                        "static_friction": candidate["static_friction"],
                        "dynamic_friction": candidate["dynamic_friction"],
                        "source_scene_saved_or_modified": False,
                    },
                    "config_sha256": config_sha,
                    "primitive": str(primitive),
                    "primitive_sha256": sha256_file(primitive),
                    "event_log": str(event_log),
                    "event_log_sha256": sha256_file(event_log),
                    "scene": str(scene),
                    "scene_sha256": sha256_file(scene),
                    "policy_or_checkpoint_used": False,
                    "policy_specific_logic": False,
                    "object_config_selected_from_policy_result": False,
                }
                path = root / f"{side}.json"
                path.write_text(json.dumps(result), encoding="utf-8")
                result_paths[side] = path
            output = root / "frozen.json"
            frozen, manifest = freeze(
                config_path,
                "MEDIUM",
                result_paths["left"],
                result_paths["right"],
                output,
            )
            self.assertTrue(frozen["freeze"]["frozen"])
            self.assertEqual(frozen["selected_material_candidate"], "MEDIUM")
            self.assertTrue(frozen["policy_evaluation_allowed"])
            self.assertFalse(manifest["policy_results_consulted"])

    def test_predeclared_material_rule_chooses_lowest_bilateral_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            passes = {
                ("left", "LOW"): True,
                ("right", "LOW"): False,
                ("left", "MEDIUM"): True,
                ("right", "MEDIUM"): True,
                ("left", "HIGH"): True,
                ("right", "HIGH"): True,
            }
            for material in ("LOW", "MEDIUM", "HIGH"):
                for side in ("left", "right"):
                    passed = passes[(side, material)]
                    row = {
                        "side": side,
                        "material_candidate": material,
                        "LEFT_LIFT_PASS": passed if side == "left" else False,
                        "RIGHT_LIFT_PASS": passed if side == "right" else False,
                    }
                    path = root / f"{material}_{side}.json"
                    path.write_text(json.dumps(row), encoding="utf-8")
                    paths.append(path)
            report = summarize(paths, root / "summary.json")
            self.assertEqual(report["recommended_candidate_under_predeclared_rule"], "MEDIUM")
            self.assertFalse(report["policy_results_consulted"])


if __name__ == "__main__":
    unittest.main()
