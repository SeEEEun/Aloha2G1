"""Integrity and fairness gates for the fully generated feasibility-v3 audit."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from aloha_g1_feasibility_v3.common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    V2_ARM_ROOT,
    V2_INTEGRATED_ROOT,
    load_json,
)
from aloha_g1_arm_v2.audit import configure_g1  # noqa: E402
from aloha_g1_dataset_v1.core import G1Kinematics  # noqa: E402


OUT = DEFAULT_OUTPUT_ROOT


def _episodes(dataset: str) -> list[Path]:
    return sorted((OUT / dataset).glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))


def _all_metrics(dataset: str) -> list[dict]:
    return [load_json(path / "retargeting_metrics.json") for path in _episodes(dataset)]


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def test_v2_metric_reproduction_is_exact() -> None:
    audit = load_json(OUT / "audit/v2_reproduction.json")
    assert audit["exact_reproduction"] is True
    assert all(
        row["exact_json_reproduction"]
        for row in audit["comparisons"].values()
    )


def test_same_solver_class_and_parameters_for_a_b() -> None:
    frozen = load_json(OUT / "solver/frozen_feasibility_v3_config.json")
    expected = frozen["solver_parameters"]
    for dataset in ("dataset_a", "dataset_b"):
        for episode in _episodes(dataset):
            report = load_json(episode / "solver_report.json")
            assert report["solver_class"] == "SharedConstrainedFeasibilitySolver"
            assert report["solver_parameters"] == expected


def test_shared_slack_collision_limit_and_temporal_contract() -> None:
    fairness = load_json(OUT / "summary/fairness_audit.json")
    assert fairness["same_orientation_slack_bound"] is True
    assert fairness["same_collision_penalty"] is True
    assert fairness["same_joint_limit_constraints"] is True
    assert fairness["same_temporal_limits"] is True
    gate = load_json(OUT / "audit/acceptance_gate_contract.json")
    assert gate["gate_change_from_integrated_v2"] == "NONE"
    assert gate["temporal"]["changed"] is False
    assert gate["collision"]["changed"] is False
    assert gate["temporal"]["solver_numerical_interior"]["step_rad"] > 0
    assert (
        gate["temporal"]["solver_numerical_interior"][
            "acceleration_rad_s2"
        ]
        > 0
    )


def test_orientation_slack_used_is_bounded_and_excess_is_explicit() -> None:
    summary = load_json(OUT / "summary/orientation_slack_summary.json")
    for dataset in ("dataset_a", "dataset_b"):
        row = summary[dataset]
        assert row["max"] <= row["slack_bound_rad"] + 1e-7
        assert row["requested"]["max"] >= row["max"]
        assert row["unmet_excess"]["min"] >= 0.0


def test_all_100_outputs_are_finite_and_inside_hard_arm_limits() -> None:
    frozen_arm = load_json(OUT / "dependencies/frozen_common_arm_v2_config.json")
    g1 = G1Kinematics(frozen_arm["runtime_config"])
    configure_g1(g1, frozen_arm["runtime_config"])
    limits = np.asarray(g1.limits, dtype=np.float64)
    for dataset in ("dataset_a", "dataset_b"):
        assert len(_episodes(dataset)) == 50
        for episode in _episodes(dataset):
            with np.load(episode / "g1_arm_action.npz", allow_pickle=False) as payload:
                q = payload["action"].astype(np.float64)
            assert q.ndim == 2 and q.shape[1] == 14
            assert np.isfinite(q).all()
            assert np.all(q >= limits[:, 0] - 1e-9)
            assert np.all(q <= limits[:, 1] + 1e-9)


def test_deterministic_optimization_rerun() -> None:
    result = load_json(OUT / "tests/deterministic_rerun.json")
    assert result["pass"] is True
    assert all(result["checks"].values())


def test_passing_v2_task_frames_are_minimally_changed() -> None:
    exact = 0
    eligible = 0
    for dataset in ("dataset_a", "dataset_b"):
        for row in _all_metrics(dataset):
            eligible += int(row["v2_to_v3"]["v2_strict_task_success_frames"])
            exact += int(
                row["v2_to_v3"][
                    "v2_strict_task_success_frames_exactly_unchanged"
                ]
            )
    assert eligible > 0
    assert exact / eligible >= 0.95


def test_dataset_a_remains_wrist_level_and_binary_semantic() -> None:
    for episode in _episodes("dataset_a"):
        with np.load(episode / "g1_arm_action.npz", allow_pickle=False) as arm:
            representation = str(arm["representation"])
        with np.load(episode / "g1_hand_action.npz", allow_pickle=False) as hand:
            assert str(hand["mapper"]) == (
                "unchanged_binary_open_close_with_shared_temporal_feasibility"
            )
            phases = set(hand["left_phase"].astype(str)) | set(
                hand["right_phase"].astype(str)
            )
        assert "wrist-level" in representation
        assert phases <= {"OPEN", "CLOSE"}


def test_dataset_b_remains_interaction_aware_and_uses_frozen_hand() -> None:
    expected_phases = {"OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE"}
    seen: set[str] = set()
    for episode in _episodes("dataset_b"):
        with np.load(episode / "g1_arm_action.npz", allow_pickle=False) as arm:
            representation = str(arm["representation"])
        with np.load(episode / "g1_hand_action.npz", allow_pickle=False) as hand:
            assert str(hand["mapper"]) == "frozen_proposed_hand_v2_1"
            seen |= set(hand["left_phase"].astype(str))
            seen |= set(hand["right_phase"].astype(str))
        assert "bimanual" in representation and "pinch-frame" in representation
    assert seen == expected_phases


def test_hand_v2_1_and_common_mapping_dependencies_are_byte_identical() -> None:
    dependency = load_json(OUT / "dependencies/dependency_checksums.json")
    for key in ("common_arm_v2", "proposed_hand_v2_1"):
        assert dependency["frozen_copies"][key]["byte_identical"] is True
        assert dependency["frozen_copies"][key]["sha256"] == dependency["sources"][key]["sha256"]
    integrity = load_json(OUT / "summary/integrity.json")
    assert integrity["common_arm_v2_unchanged"] is True
    assert integrity["hand_v2_1_unchanged"] is True


def test_proposed_hand_actions_remain_byte_identical_to_integrated_v2() -> None:
    for episode_id in range(50):
        current = OUT / "dataset_b" / f"episode_{episode_id:06d}" / "g1_hand_action.npz"
        prior = V2_INTEGRATED_ROOT / "dataset_b" / f"episode_{episode_id:06d}" / "g1_hand_action.npz"
        with np.load(current, allow_pickle=False) as new, np.load(prior, allow_pickle=False) as old:
            assert np.array_equal(new["left_action"], old["left_action"])
            assert np.array_equal(new["right_action"], old["right_action"])
            assert np.array_equal(new["left_phase"], old["left_phase"])
            assert np.array_equal(new["right_phase"], old["right_phase"])


def test_frozen_target_representations_are_byte_identical_to_arm_v2() -> None:
    keys = (
        "target_left_wrist_position",
        "target_right_wrist_position",
        "target_left_wrist_rotation",
        "target_right_wrist_rotation",
        "target_left_task_tool_position",
        "target_right_task_tool_position",
    )
    for method, dataset in (("baseline", "dataset_a"), ("proposed", "dataset_b")):
        for episode_id in range(50):
            current = OUT / dataset / f"episode_{episode_id:06d}" / "g1_arm_action.npz"
            prior = V2_ARM_ROOT / method / f"episode_{episode_id:06d}" / "g1_arm_action.npz"
            with np.load(current, allow_pickle=False) as new, np.load(prior, allow_pickle=False) as old:
                for key in keys:
                    assert _array_hash(new[key]) == _array_hash(old[key])


def test_output_roots_and_manifests_are_separate() -> None:
    assert (OUT / "dataset_a").resolve() != (OUT / "dataset_b").resolve()
    for dataset, method in (("dataset_a", "baseline"), ("dataset_b", "proposed")):
        for episode in _episodes(dataset):
            manifest = load_json(episode / "manifest.json")
            assert manifest["dataset"] == dataset
            assert manifest["method"] == method
            assert manifest["a_b_output_separation"] is True


def test_source_hashes_and_split_are_unchanged() -> None:
    integrity = load_json(OUT / "summary/integrity.json")
    assert integrity["source_hashes_unchanged"] is True
    assert integrity["split_byte_identical"] is True
    frozen = load_json(OUT / "solver/frozen_feasibility_v3_config.json")
    assert len(frozen["calibration_episode_ids"]) == 40
    assert len(frozen["validation_episode_ids_locked_during_selection"]) == 10
    assert set(frozen["calibration_episode_ids"]).isdisjoint(
        frozen["validation_episode_ids_locked_during_selection"]
    )


def test_candidate_selection_never_used_locked_validation() -> None:
    frozen = load_json(OUT / "solver/frozen_feasibility_v3_config.json")
    assert frozen["validation_used_for_selection"] is False
    assert len(frozen["full_40_episode_calibration_finalists"]) >= 1
    for row in frozen["full_40_episode_calibration_finalists"]:
        assert row["validation_used"] is False
        assert len(row["calibration_episode_ids"]) == 40
    validation = load_json(OUT / "solver/validation_results.json")
    assert validation["solver_config_frozen_before_validation"] is True
    assert validation["retuned_after_validation"] is False


def test_no_episode_or_frame_specific_execution_logic() -> None:
    scan = load_json(OUT / "tests/anti_overfit_scan.json")
    assert scan["pass"] is True
    assert scan["hits"] == []
    paths = list((TOOLS / "aloha_g1_feasibility_v3").glob("*.py"))
    patterns = (
        re.compile(r"if\s+episode(?:_id)?\s*=="),
        re.compile(r"if\s+frame(?:_id|_index)?\s*=="),
    )
    assert not any(
        pattern.search(path.read_text(encoding="utf-8"))
        for path in paths
        for pattern in patterns
    )


def test_collision_gate_has_no_unknown_silent_exclusions() -> None:
    audit = load_json(OUT / "audit/collision_gate_semantics.json")
    assert audit["acceptance_gate_bug_found"] is False
    assert audit["gate_change"] == "NONE"
    assert audit["unknown_pair_count"] == 0
    assert audit["object_collision_geometry_in_gate"] is False


def test_training_and_real_execution_were_not_performed() -> None:
    for dataset in ("dataset_a", "dataset_b"):
        for episode in _episodes(dataset):
            manifest = load_json(episode / "manifest.json")
            assert manifest["training_executed"] is False
            assert manifest["physics_executed"] is False
            assert manifest["real_robot_commands"] is False


def test_every_episode_has_an_explicit_supported_status() -> None:
    supported = {
        "PASS",
        "FAIL_IK",
        "FAIL_COLLISION",
        "FAIL_TEMPORAL",
        "FAIL_LIMIT",
        "FAIL_DATA",
        "FAIL_OTHER",
    }
    for dataset in ("dataset_a", "dataset_b"):
        rows = _all_metrics(dataset)
        assert len(rows) == 50
        assert all(row["status"] in supported for row in rows)
