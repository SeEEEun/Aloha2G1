from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.doll_handoff_feasibility.common import (
    EXPECTED_FROZEN_HASHES,
    FROZEN_ROOT,
    OUTPUT_ROOT,
    load_json,
    stable_episode_id,
    verify_frozen_contract,
)
from tools.doll_handoff_feasibility.solver import GenericG1FeasibilityResolver


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
TARGET_KEYS = (
    "target_left_wrist_position_model",
    "target_right_wrist_position_model",
    "target_left_wrist_rotation_model",
    "target_right_wrist_rotation_model",
    "target_left_interaction_frame_position_world",
    "target_right_interaction_frame_position_world",
    "target_left_interaction_frame_position_task",
    "target_right_interaction_frame_position_task",
)
SEMANTIC_KEYS = (
    "ownership_state",
    "left_hand_phase",
    "right_hand_phase",
    "event_names",
    "event_frames",
)


def test_frozen_before_contract_is_still_authoritative() -> None:
    freeze = verify_frozen_contract()
    for key, value in EXPECTED_FROZEN_HASHES.items():
        assert freeze[key] == value
    assert freeze["frozen_episode_count"] == 50
    assert freeze["handoff_cartesian_residual_m"] == 0.0


def test_config_is_common_and_contains_no_episode_or_phase_solver_map() -> None:
    config = load_json(CONFIG)
    assert config["scope"] == "COMMON_BASELINE_AND_PROPOSED_TARGET_G1_BACKEND"
    assert config["task_independent"] is True
    assert config["episode_specific_parameters_allowed"] is False
    assert config["phase_specific_cartesian_parameters_allowed"] is False
    assert config["source_target_mutation_allowed"] is False
    assert config["ownership_timing_mutation_allowed"] is False
    for section in (
        "constrained_tracking",
        "orientation_policy",
        "collision_repair",
        "nearest_target_projection",
    ):
        encoded = json.dumps(config[section], sort_keys=True).lower()
        assert "episode" not in encoded
        assert "handoff" not in encoded
        assert "ownership" not in encoded


def test_source_targets_semantics_and_hands_are_byte_identical_for_all_50() -> None:
    for episode in range(50):
        stable = stable_episode_id(episode)
        before_path = FROZEN_ROOT / "proposed/trajectories" / f"{stable}.npz"
        after_path = OUTPUT_ROOT / "after/trajectories" / f"{stable}.npz"
        with np.load(before_path, allow_pickle=False) as before, np.load(
            after_path, allow_pickle=False
        ) as after:
            for key in (*TARGET_KEYS, *SEMANTIC_KEYS):
                assert np.array_equal(before[key], after[key]), (episode, key)
            for key in ("left_dex3_qpos", "right_dex3_qpos"):
                assert np.array_equal(before[key], after[key]), (episode, key)
            assert not bool(np.asarray(after["source_targets_modified"]).item())
            assert not bool(np.asarray(after["ownership_timing_modified"]).item())


def test_full50_gate_counts_and_semantics_are_explicit() -> None:
    aggregate = load_json(OUTPUT_ROOT / "full50/aggregate_summary.json")
    assert aggregate["classification_counts"] == {
        "CLEAN_PASS": 12,
        "USABLE_WITH_WARNING": 36,
        "HARD_FAIL": 2,
    }
    assert aggregate["physical_hard_ik_episode_count"] == 0
    assert aggregate["arm_torso_hard_collision_episode_count"] == 2
    assert aggregate["distal_hard_collision_episode_count"] == 0
    assert aggregate["joint_limit_violations"] == 0
    assert aggregate["branch_discontinuities"] == 0
    assert aggregate["handoff_ordering_valid"] == 50
    assert aggregate["ownership_transition_valid"] == 50
    assert aggregate["release_event_present"] == 50
    assert aggregate["source_bin_release_inside"] == 50
    assert aggregate["ownership_timing_change"] == 0
    assert aggregate["source_interaction_target_change"] == 0
    assert aggregate["source_target_hash"]["unchanged"] is True
    findings = load_json(OUTPUT_ROOT / "full50/failures.json")
    hard = [row for row in findings if row["classification"] == "HARD_FAIL"]
    assert [row["episode_index"] for row in hard] == [13, 36]
    assert all(row["reasons"] for row in findings)


def test_projection_and_branch_cap_are_mathematically_bounded() -> None:
    current = np.zeros(3)
    candidate = np.asarray([1.0, 0.0, 0.0])
    neighbors = (np.zeros(3), np.asarray([0.02, 0.0, 0.0]))
    result = GenericG1FeasibilityResolver._cap_branch_step(
        current, candidate, neighbors, 0.18
    )
    assert np.linalg.norm(result - neighbors[0]) <= 0.18 + 1e-12
    assert np.linalg.norm(result - neighbors[1]) <= 0.18 + 1e-12
    assert result[0] > 0.0


def test_freeze_manifest_records_review_not_dataset_packaging() -> None:
    manifest = load_json(OUTPUT_ROOT / "freeze_manifest.json")
    assert manifest["status"] == "GENERIC_G1_FEASIBILITY_REVIEW_FROZEN"
    assert manifest["integrity"]["all_source_targets_byte_equal"] is True
    assert manifest["integrity"]["all_interaction_semantics_byte_equal"] is True
    assert manifest["integrity"]["after_cartesian_target_array_set_sha256"] == (
        EXPECTED_FROZEN_HASHES["cartesian_target_array_set_sha256"]
    )
    assert manifest["retained_hard_failures"] == [13, 36]
    assert manifest["dataset_b_packaged"] is False
    assert manifest["dataset_a_packaged"] is False
    assert manifest["policy_training_started"] is False
    assert len(manifest["video_sha256"]) == 21
