"""Integrity and anti-overfitting tests for frozen Common Collision-v4."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from aloha_g1_arm_v2.audit import configure_g1
from aloha_g1_collision_v4.solver import SharedCollisionWindowSolver, load_v3_episode
from aloha_g1_dataset_v1.core import G1Kinematics
from aloha_g1_hand_v2.collision_eval import CollisionClassifier, make_runtime


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/g1_dataset_collision_v4"
V3 = ROOT / "outputs/g1_dataset_feasibility_v3"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_50_outputs_are_separate_and_complete() -> None:
    for dataset in ("dataset_a", "dataset_b"):
        directories = sorted((OUT / dataset).glob("episode_*"))
        assert len(directories) == 50
        assert [path.name for path in directories] == [
            f"episode_{value:06d}" for value in range(50)
        ]
        assert all((path / "manifest.json").is_file() for path in directories)


def test_same_collision_solver_and_config_for_a_b() -> None:
    frozen = read(OUT / "calibration/frozen_global_collision_v4_config.json")
    expected = sha(OUT / "calibration/frozen_global_collision_v4_config.json")
    assert frozen["method_specific_parameters"] is False
    assert frozen["solver_class"] == "SharedCollisionWindowSolver"
    for dataset in ("dataset_a", "dataset_b"):
        for episode_id in range(50):
            manifest = read(OUT / dataset / f"episode_{episode_id:06d}" / "manifest.json")
            assert manifest["frozen_collision_v4_solver_sha256"] == expected


def test_one_global_clearance_and_temporal_rule() -> None:
    frozen = read(OUT / "calibration/frozen_global_collision_v4_config.json")
    fairness = read(OUT / "summary/fairness_audit.json")
    assert fairness["same_d_safe_m"] == frozen["parameters"]["d_safe_m"]
    assert fairness["same_temporal_window_rule"] is True
    assert frozen["parameters"]["window_padding_frames"] >= 1


def test_collision_semantics_are_byte_identical_to_v3() -> None:
    dependency = read(OUT / "dependencies/dependency_checksums.json")
    assert dependency["copies"]["frozen_collision_gate_semantics.json"]["byte_identical"]
    assert dependency["collision_semantics_sha256"] == sha(
        V3 / "audit/collision_gate_semantics.json"
    )


def test_task_and_internal_contacts_remain_allowed() -> None:
    semantics = read(OUT / "dependencies/frozen_collision_gate_semantics.json")
    classes = {row["classification"] for row in semantics["pairs"]}
    assert "KNOWN_ALLOWED_INTERNAL_CONTACT" in classes
    assert "DIAGNOSTIC_ONLY_CONTACT" in classes
    assert semantics["gate_change"] == "NONE"


def test_hard_joint_limits_and_finite_outputs() -> None:
    for dataset in ("dataset_a", "dataset_b"):
        for episode_id in range(50):
            directory = OUT / dataset / f"episode_{episode_id:06d}"
            metrics = read(directory / "retargeting_metrics.json")
            with np.load(directory / "g1_full_action.npz", allow_pickle=False) as payload:
                action = payload["action"]
            assert action.shape[1] == 28
            assert np.isfinite(action).all()
            assert metrics["joint_limit_violation_count"] == 0


def test_v3_collision_free_episodes_are_arm_byte_identical() -> None:
    for dataset in ("dataset_a", "dataset_b"):
        for episode_id in range(50):
            old_dir = V3 / dataset / f"episode_{episode_id:06d}"
            old_metrics = read(old_dir / "retargeting_metrics.json")
            if old_metrics["collision"]["prohibited_collision_frames"] != 0:
                continue
            with np.load(old_dir / "g1_arm_action.npz", allow_pickle=False) as old, np.load(
                OUT / dataset / f"episode_{episode_id:06d}" / "g1_arm_action.npz",
                allow_pickle=False,
            ) as new:
                assert np.array_equal(old["action"], new["action"])


def test_hand_actions_are_unchanged_for_both_methods() -> None:
    for dataset in ("dataset_a", "dataset_b"):
        for episode_id in range(50):
            with np.load(
                V3 / dataset / f"episode_{episode_id:06d}" / "g1_hand_action.npz",
                allow_pickle=False,
            ) as old, np.load(
                OUT / dataset / f"episode_{episode_id:06d}" / "g1_hand_action.npz",
                allow_pickle=False,
            ) as new:
                for key in ("left_action", "right_action", "left_phase", "right_phase"):
                    assert np.array_equal(old[key], new[key])


def test_upstream_configs_and_source_hashes_unchanged() -> None:
    integrity = read(OUT / "summary/integrity.json")
    assert integrity["source_hashes_unchanged"]
    assert integrity["common_arm_v2_unchanged"]
    assert integrity["feasibility_v3_config_unchanged"]
    assert integrity["hand_v2_1_unchanged"]
    assert integrity["dataset_a_definition_unchanged"]
    assert integrity["dataset_b_definition_unchanged"]


def test_a_and_b_representations_remain_distinct() -> None:
    fairness = read(OUT / "summary/fairness_audit.json")
    assert fairness["dataset_a_uses_proposed_objectives"] is False
    assert fairness["dataset_b_interaction_objectives_preserved"] is True
    with np.load(OUT / "dataset_a/episode_000000/g1_arm_action.npz", allow_pickle=False) as a, np.load(
        OUT / "dataset_b/episode_000000/g1_arm_action.npz", allow_pickle=False
    ) as b:
        assert "wrist" in str(a["representation"])
        assert "pinch-frame" in str(b["representation"])


def test_no_episode_or_frame_specific_solver_logic() -> None:
    paths = sorted((ROOT / "tools/aloha_g1_collision_v4").glob("*.py"))
    paths += [ROOT / "configs/aloha_g1_collision_v4.json"]
    forbidden = [
        re.compile(r"if\s+episode(?:_id)?\s*=="),
        re.compile(r"if\s+frame(?:_id|_index)?\s*=="),
        re.compile("ep" + "49", re.IGNORECASE),
        re.compile("manual" + r"[ _-]+offset", re.IGNORECASE),
    ]
    assert not [
        (path, pattern.pattern)
        for path in paths
        for pattern in forbidden
        if pattern.search(path.read_text(encoding="utf-8"))
    ]


def test_no_failed_episode_list_dependency() -> None:
    source = (ROOT / "tools/aloha_g1_collision_v4/solver.py").read_text(encoding="utf-8")
    for token in ("[0, 5, 7", "[4, 39, 42", "failed_episode_ids"):
        assert token not in source


def test_locked_validation_was_not_used_for_selection() -> None:
    frozen = read(OUT / "calibration/frozen_global_collision_v4_config.json")
    validation = read(OUT / "validation/locked_validation_results.json")
    assert frozen["validation_used_for_selection"] is False
    assert validation["configuration_frozen_before_validation"] is True
    assert validation["retuned_after_validation"] is False


def test_deterministic_identity_rerun_contract() -> None:
    # Collision-free frames take the explicit q_v4=q_v3 path.  Two independent
    # output hashes (arm payload and stored reference array) must agree exactly.
    directory = OUT / "dataset_a/episode_000001"
    with np.load(directory / "g1_arm_action.npz", allow_pickle=False) as action, np.load(
        directory / "collision_repair_diagnostics.npz", allow_pickle=False
    ) as diagnostics:
        assert np.array_equal(action["action"], diagnostics["q_v3"])
        assert np.array_equal(action["action"], diagnostics["q_v4"])


def test_deterministic_colliding_episode_rerun() -> None:
    audit = read(OUT / "audit/collision_events_v3.json")
    episode_id = min(audit["methods"]["baseline"]["affected_episode_ids"])
    arm = read(OUT / "dependencies/frozen_common_arm_v2_config.json")
    feasibility = read(OUT / "dependencies/frozen_feasibility_v3_config.json")
    frozen = read(OUT / "calibration/frozen_global_collision_v4_config.json")
    runtime_config = arm["runtime_config"]
    g1 = G1Kinematics(runtime_config)
    configure_g1(g1, runtime_config)
    runtime = make_runtime(runtime_config)
    classifier = CollisionClassifier(runtime, runtime_config)
    solver = SharedCollisionWindowSolver(
        runtime_config,
        feasibility,
        frozen["parameters"],
        g1,
        runtime,
        classifier,
    )
    episode = load_v3_episode("baseline", episode_id)
    first = solver.solve(episode)
    second = solver.solve(episode)
    assert np.array_equal(first.q, second.q)
    assert first.metadata["final_diagnostics"] == second.metadata["final_diagnostics"]


def test_anti_overfitting_audit_and_output_separation() -> None:
    audit = read(OUT / "summary/anti_overfitting_audit.json")
    integrity = read(OUT / "summary/integrity.json")
    assert audit["pass"]
    assert audit["one_global_d_safe"]
    assert audit["one_solver_class_for_a_b"]
    assert integrity["a_b_output_separate"]
