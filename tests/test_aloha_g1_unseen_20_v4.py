from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/g1_unseen_20_v4"
SOURCE = ROOT / "lerobot_magsafe_20_cam_high_v3_unseen_20260813"
OLD = ROOT / "outputs/g1_dataset_collision_v4"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def episode(dataset: str, episode_id: int) -> Path:
    return OUTPUT / dataset / f"episode_{episode_id:06d}"


def test_exact_raw_list_count_and_uniqueness() -> None:
    manifest = load(OUTPUT / "source/new_20_raw_manifest.json")
    names = [row["raw_directory_name"] for row in manifest["recordings"]]
    assert manifest["required_count"] == 20
    assert manifest["discovered_required_count"] == 20
    assert len(names) == len(set(names)) == 20


def test_no_unlisted_raw_recording_included() -> None:
    manifest = load(OUTPUT / "source/new_20_raw_manifest.json")
    expected = {
        line.strip().strip('",')
        for line in (ROOT / "tools/aloha_g1_unseen_20_v4/constants.py")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip().startswith('"GoPark_20260813_')
    }
    actual = {row["raw_directory_name"] for row in manifest["recordings"]}
    assert actual == expected
    assert "_invalid_no_parquet" not in actual
    assert all("202607" not in name for name in actual)


def test_every_source_file_hash_recorded() -> None:
    manifest = load(OUTPUT / "source/new_20_raw_manifest.json")
    index = OUTPUT / "source/raw_file_hashes.jsonl"
    assert manifest["all_source_hashes_recorded"]
    assert sum(1 for _ in index.open(encoding="utf-8")) == manifest["all_source_file_count"]
    assert all(len(row["content_tree_sha256"]) == 64 for row in manifest["recordings"])


def test_all_raw_sources_pass() -> None:
    manifest = load(OUTPUT / "source/new_20_raw_manifest.json")
    assert manifest["source_pass_count"] == 20
    assert manifest["source_fail_count"] == 0
    assert all(row["status"] == "SOURCE_PASS" for row in manifest["recordings"])


def test_raw_action_channel_contract() -> None:
    manifest = load(OUTPUT / "source/new_20_raw_manifest.json")
    for row in manifest["recordings"]:
        contract = row["action"]["verified_channel_contract"]
        assert contract == {
            "left_arm": [0, 1, 2, 3, 4, 5],
            "left_gripper": 6,
            "right_arm": [7, 8, 9, 10, 11, 12],
            "right_gripper": 13,
        }
        assert row["action"]["dimension"] == row["observation_state"]["dimension"] == 14


def test_layout_audit_has_exact_samples() -> None:
    value = load(OUTPUT / "source/layout_audit/layout_audit_manifest.json")
    assert value["sample_count"] == 60
    assert len(value["contact_sheets"]) == 4
    assert value["visual_review"] == "PASS_NO_GROSS_LAYOUT_OR_CAMERA_ANOMALY"
    assert value["object_relative_source_metadata"] == "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE"


def test_new_source_is_separate_lerobot_v3() -> None:
    info = load(SOURCE / "meta/info.json")
    assert SOURCE != ROOT / "lerobot_magsafe_50_cam_high_v3"
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 20
    assert info["total_frames"] == 18633
    assert info["fps"] == 30


def test_source_schema_matches_original_50() -> None:
    comparison = load(OUTPUT / "source/source_schema_comparison.json")
    assert comparison["status"] == "SCHEMA_EQUIVALENT"
    assert comparison["all_required_contracts_identical"]
    assert all(comparison["checks"].values())


def test_integrated_source_state_action_finite_and_14d() -> None:
    files = sorted((SOURCE / "data").glob("chunk-*/*.parquet"))
    assert files
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action"])
        for key in ("observation.state", "action"):
            array = np.asarray(table[key].combine_chunks().values.to_numpy()).reshape(-1, 14)
            assert array.shape[1] == 14
            assert np.isfinite(array).all()


def test_all_four_frozen_hashes_match() -> None:
    manifest = load(OUTPUT / "dependencies/dependency_checksums.json")
    expected = {
        "common_arm_v2": "48d9fe29503091fed1bdc4eeb359f349d4a188ad05c368d32c09cdecee60acab",
        "feasibility_v3": "167a2ade3ffe694fb958119d68d0ff83f3187f5983d5fe5221e284e8eaf09bd0",
        "proposed_hand_v2_1": "811eba1591131671d787bb714c86e75b643cd2a13cd87ecddbb486cdd912bbd3",
        "collision_v4": "b2ece14bfd673bac177ee833f94b0fb0805dd74c00e23c9da31a92c0cc046261",
    }
    assert manifest["status"] == "ALL_FROZEN_V4_HASHES_VERIFIED"
    assert {key: row["actual_sha256"] for key, row in manifest["dependencies"].items()} == expected
    assert all(row["verified"] for row in manifest["dependencies"].values())


def test_frozen_dependencies_unchanged_after_conversion() -> None:
    value = load(OUTPUT / "dependencies/post_conversion_integrity.json")
    assert value["all_unchanged"]
    assert value["original_source"]["unchanged"]
    assert value["unseen_source"]["matches_build_report"]


def test_exactly_20_separate_outputs_per_method() -> None:
    for dataset in ("dataset_a", "dataset_b"):
        directories = sorted((OUTPUT / dataset).glob("episode_*"))
        assert len(directories) == 20
        assert [path.name for path in directories] == [f"episode_{i:06d}" for i in range(20)]
    assert (OUTPUT / "dataset_a").resolve() != (OUTPUT / "dataset_b").resolve()


def test_all_output_shapes_and_finite() -> None:
    for dataset in ("dataset_a", "dataset_b"):
        for episode_id in range(20):
            root = episode(dataset, episode_id)
            metrics = load(root / "retargeting_metrics.json")
            with np.load(root / "g1_arm_action.npz", allow_pickle=False) as data:
                arm = data["action"]
            with np.load(root / "g1_hand_action.npz", allow_pickle=False) as data:
                hand = data["action"]
            with np.load(root / "g1_full_action.npz", allow_pickle=False) as data:
                full = data["action"]
            assert arm.shape == (metrics["frame_count"], 14)
            assert hand.shape == (metrics["frame_count"], 14)
            assert full.shape == (metrics["frame_count"], 28)
            assert np.array_equal(full, np.column_stack((arm, hand)))
            assert np.isfinite(full).all()


def test_same_frozen_solver_for_a_b() -> None:
    for episode_id in range(20):
        a = load(episode("dataset_a", episode_id) / "manifest.json")
        b = load(episode("dataset_b", episode_id) / "manifest.json")
        assert a["frozen_dependency_sha256"] == b["frozen_dependency_sha256"]
        assert a["implementation_sha256"] == b["implementation_sha256"]
        assert a["frozen_dependency_sha256"]["collision_v4"] == (
            "b2ece14bfd673bac177ee833f94b0fb0805dd74c00e23c9da31a92c0cc046261"
        )


def test_dataset_a_has_no_proposed_hand_mapper() -> None:
    for episode_id in range(20):
        with np.load(episode("dataset_a", episode_id) / "g1_hand_action.npz", allow_pickle=False) as data:
            assert str(data["mapper"]).startswith("unchanged_binary_open_close")
            assert str(data["frozen_hand_v2_1_sha256"]) == "NOT_APPLICABLE_DATASET_A"
        with np.load(episode("dataset_a", episode_id) / "g1_arm_action.npz", allow_pickle=False) as data:
            assert str(data["representation"]) == "trajectory-centric independent wrist-level 6D"


def test_dataset_b_uses_exact_frozen_hand() -> None:
    expected = "811eba1591131671d787bb714c86e75b643cd2a13cd87ecddbb486cdd912bbd3"
    for episode_id in range(20):
        with np.load(episode("dataset_b", episode_id) / "g1_hand_action.npz", allow_pickle=False) as data:
            assert str(data["mapper"]) == "frozen_proposed_hand_v2_1"
            assert str(data["frozen_hand_v2_1_sha256"]) == expected
            phases = set(data["left_phase"].astype(str)) | set(data["right_phase"].astype(str))
            assert phases <= {"OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE"}


def test_stable_old_new_source_identity() -> None:
    value = load(OUTPUT / "summary/source_identity_map.json")
    ids = [row["stable_source_id"] for row in value["sources"]]
    assert len(ids) == len(set(ids)) == 70
    assert ids[:50] == [f"old50:{i:03d}" for i in range(50)]
    assert ids[50:] == [f"new20:{i:03d}" for i in range(20)]


def test_matched_manifest_contains_only_dual_pass_sources() -> None:
    value = load(OUTPUT / "summary/matched_a_b_manifest.json")
    identity = {
        row["stable_source_id"]: row
        for row in load(OUTPUT / "summary/source_identity_map.json")["sources"]
    }
    assert value["entry_count"] == len(value["entries"])
    assert len({row["stable_source_id"] for row in value["entries"]}) == value["entry_count"]
    for row in value["entries"]:
        source = identity[row["stable_source_id"]]
        assert source["dataset_a_status"] == source["dataset_b_status"] == "PASS"
        assert row["dual_pass"]


def test_native_manifests_include_every_and_only_pass() -> None:
    for name in ("dataset_a", "dataset_b"):
        value = load(OUTPUT / f"summary/{name}_native_manifest.json")
        assert value["entry_count"] == len(value["entries"])
        assert all(row["status"] == "PASS" for row in value["entries"])
        assert len({row["stable_source_id"] for row in value["entries"]}) == value["entry_count"]


def test_no_arbitrary_50_cap() -> None:
    matched = load(OUTPUT / "summary/matched_a_b_manifest.json")
    readiness = load(OUTPUT / "summary/training_readiness.json")
    assert matched["entry_count"] == readiness["combined_matched"]
    assert matched["fairness_rule"].endswith("no arbitrary cap")


def test_anti_overfitting_scan_passes() -> None:
    value = load(OUTPUT / "tests/anti_overfitting_audit.json")
    assert value["pass"]
    assert not value["forbidden_logic_hits"]
    assert not value["new_raw_identity_execution_logic_hits"]
    assert value["no_candidate_search"] and value["no_new_validation_split"]


def test_deterministic_rerun_passes() -> None:
    value = load(OUTPUT / "tests/deterministic_rerun.json")
    assert value["pass"]
    assert all(row["array_equal"] and row["status_equal"] for row in value["methods"].values())


def test_no_training_packaging_physics_or_robot_execution() -> None:
    readiness = load(OUTPUT / "summary/training_readiness.json")
    assert not readiness["lerobot_packaging_performed"]
    assert not readiness["policy_training_performed"]
    for dataset in ("dataset_a", "dataset_b"):
        for episode_id in range(20):
            manifest = load(episode(dataset, episode_id) / "manifest.json")
            assert not manifest["training_executed"]
            assert not manifest["packaging_executed"]
            assert not manifest["physics_executed"]
            assert not manifest["real_robot_commands"]
