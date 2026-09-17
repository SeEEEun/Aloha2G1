from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tools.g1_policy_dataset_packaging_v1.audits import deterministic_tree_hash
from tools.g1_policy_dataset_packaging_v1.packager import (
    load_matched_pool,
    package_matched_pair,
    validate_matched_pair,
)
from tools.g1_policy_dataset_packaging_v1.training import config_fairness
from tools.g1_training_schema_v1.causal_alignment import action_chunk_indices
from tools.g1_training_schema_v1.constants import (
    ACTION_KEY,
    CANONICAL_JOINT_NAMES,
    CHUNK_SIZE,
    IMAGE_KEY,
    JOINT_SPECS,
    STATE_KEY,
    TASK_TEXT,
)
from tools.g1_training_schema_v1.source_audit import load_json, sha256_file
from tools.g1_training_schema_v1.validator import validate_feature_schema

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/g1_unseen_20_v4/summary/matched_a_b_manifest.json"
OUTPUT = ROOT / "outputs/g1_policy_dataset_packaging_v1"
DATASET_A = ROOT / "lerobot_g1_magsafe_matched51_baseline_a_v1"
DATASET_B = ROOT / "lerobot_g1_magsafe_matched51_proposed_b_v1"


@pytest.fixture(scope="module")
def pool():
    return load_matched_pool(MANIFEST, ROOT)


def _info(root: Path) -> dict:
    return load_json(root / "meta/info.json")


def _data(root: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    return {
        STATE_KEY: np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32),
        ACTION_KEY: np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32),
        **{key: np.asarray(table[key].to_numpy()) for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
    }


def _episodes(root: Path) -> list[dict]:
    return sorted(
        pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist(),
        key=lambda row: int(row["episode_index"]),
    )


def test_matched_identity_count_is_51(pool) -> None:
    assert len(pool.entries) == len(set(pool.stable_source_ids)) == 51


def test_a_b_matched_ids_identical(pool) -> None:
    a = load_json(DATASET_A / "meta/g1_packaging_manifest.json")
    b = load_json(DATASET_B / "meta/g1_packaging_manifest.json")
    assert a["stable_source_ids"] == b["stable_source_ids"] == pool.stable_source_ids


def test_a_b_episode_count_identical() -> None:
    assert _info(DATASET_A)["total_episodes"] == _info(DATASET_B)["total_episodes"] == 51


def test_a_b_corresponding_frame_counts_identical(pool) -> None:
    lengths_a = [int(row["length"]) for row in _episodes(DATASET_A)]
    lengths_b = [int(row["length"]) for row in _episodes(DATASET_B)]
    assert lengths_a == lengths_b == [entry.frame_count for entry in pool.entries]
    assert sum(lengths_a) == 50275


def test_a_b_rgb_references_byte_equivalent() -> None:
    a = load_json(DATASET_A / "meta/g1_packaging_manifest.json")["video_assets"]
    b = load_json(DATASET_B / "meta/g1_packaging_manifest.json")["video_assets"]
    assert len(a) == len(b) == 3
    for left, right in zip(a, b, strict=True):
        assert left["sha256"] == right["sha256"]
        path_a = DATASET_A / left["output_relative_path"]
        path_b = DATASET_B / right["output_relative_path"]
        assert sha256_file(path_a) == sha256_file(path_b) == left["sha256"]
        assert os.path.samefile(path_a, path_b)


def test_a_b_task_strings_identical() -> None:
    rows_a = pq.read_table(DATASET_A / "meta/tasks.parquet").to_pylist()
    rows_b = pq.read_table(DATASET_B / "meta/tasks.parquet").to_pylist()
    assert rows_a == rows_b == [{"__index_level_0__": TASK_TEXT, "task_index": 0}]


def test_a_b_schema_identical() -> None:
    assert _info(DATASET_A) == _info(DATASET_B)
    validate_feature_schema(_info(DATASET_A)["features"])


def test_state_and_action_dimensions_are_28() -> None:
    features = _info(DATASET_A)["features"]
    assert features[STATE_KEY]["shape"] == features[ACTION_KEY]["shape"] == [28]


def test_exact_joint_order() -> None:
    features = _info(DATASET_A)["features"]
    assert features[STATE_KEY]["names"] == features[ACTION_KEY]["names"] == list(CANONICAL_JOINT_NAMES)


def test_finite_values_and_same_row_contract() -> None:
    for root in (DATASET_A, DATASET_B):
        arrays = _data(root)
        assert np.isfinite(arrays[STATE_KEY]).all()
        assert np.isfinite(arrays[ACTION_KEY]).all()
        assert np.array_equal(arrays[STATE_KEY], arrays[ACTION_KEY])


def test_joint_limits() -> None:
    lower = np.asarray([joint.minimum for joint in JOINT_SPECS])
    upper = np.asarray([joint.maximum for joint in JOINT_SPECS])
    for root in (DATASET_A, DATASET_B):
        q = _data(root)[ACTION_KEY]
        assert np.all(q >= lower - 1e-5)
        assert np.all(q <= upper + 1e-5)


def test_timestamp_monotonicity() -> None:
    for root in (DATASET_A, DATASET_B):
        arrays = _data(root)
        for row in _episodes(root):
            mask = arrays["episode_index"] == int(row["episode_index"])
            timestamps = arrays["timestamp"][mask]
            assert timestamps[0] == 0.0
            assert np.all(np.diff(timestamps) > 0)
            assert np.allclose(timestamps, np.arange(len(timestamps)) / 30, atol=1e-5, rtol=0)


def test_no_episode_boundary_leakage_and_padding() -> None:
    for row in _episodes(DATASET_A):
        length = int(row["length"])
        for frame in (0, length // 2, length - 1):
            indices, padding = action_chunk_indices(frame, length, CHUNK_SIZE)
            assert indices.min() >= 0 and indices.max() < length
            assert np.array_equal(padding, frame + np.arange(CHUNK_SIZE) >= length)


def test_action_chunk_shape_and_final_mask_from_reader() -> None:
    readback = load_json(OUTPUT / "audit/lerobot_readback_audit.json")
    assert readback["status"] == "PASS" and readback["lerobot_version"] == "0.6.1"
    assert all(item["action_chunk_shape_a_b"] == [[50, 28], [50, 28]] for item in readback["sample_reports"])
    assert readback["sample_reports"][-1]["padding_true_count"] == 49


def test_normalization_only_matched_accepted_frames(pool) -> None:
    for name in ("normalization_a.json", "normalization_b.json"):
        value = load_json(OUTPUT / "summary" / name)
        assert value["fit_episode_count"] == 51
        assert value["fit_frame_count"] == 50275
        assert value["fit_stable_source_ids"] == pool.stable_source_ids
        assert value["failed_or_unmatched_episodes_used"] is False


def test_no_method_id_policy_feature() -> None:
    assert not ({"method", "method_id", "retargeting_method"} & set(_info(DATASET_A)["features"]))


def test_no_a_b_dataset_mixing() -> None:
    a = load_json(DATASET_A / "meta/g1_packaging_manifest.json")
    b = load_json(DATASET_B / "meta/g1_packaging_manifest.json")
    assert a["method_metadata_only"] == "dataset_a"
    assert b["method_metadata_only"] == "dataset_b"
    assert DATASET_A.resolve() != DATASET_B.resolve()


def test_actual_lerobot_reader_opened_both() -> None:
    readback = load_json(OUTPUT / "audit/lerobot_readback_audit.json")
    assert readback["dataset_length_a"] == readback["dataset_length_b"] == 50275
    assert readback["image_decode_pass"] and readback["a_b_rgb_decode_exact_equal"]


def test_packaged_values_equal_frozen_trajectories(pool) -> None:
    validate_matched_pair(pool, DATASET_A, DATASET_B)


def test_deterministic_packaging(pool) -> None:
    # Hardlink-based packaging must be repeated on the source filesystem.
    # TemporaryDirectory removes only this test-owned staging tree afterwards.
    with tempfile.TemporaryDirectory(prefix=".matched51-determinism-", dir=ROOT) as temporary:
        second_a = Path(temporary) / DATASET_A.name
        second_b = Path(temporary) / DATASET_B.name
        package_matched_pair(pool, second_a, second_b, "hardlink")
        assert deterministic_tree_hash(DATASET_A) == deterministic_tree_hash(second_a)
        assert deterministic_tree_hash(DATASET_B) == deterministic_tree_hash(second_b)


def test_policy_configs_identical_except_allowed_fields() -> None:
    result = config_fairness(
        OUTPUT / "training_configs/policy_a_config.json",
        OUTPUT / "training_configs/policy_b_config.json",
    )
    assert result["status"] == "PASS" and result["all_other_fields_identical"]


def test_same_pretrained_initialization() -> None:
    audit = load_json(OUTPUT / "audit/pretrained_initialization.json")
    assert audit["status"] == "PASS"
    assert audit["same_initialization"] and not audit["old_aloha_finetuned_20k_checkpoint_used"]


def test_same_primary_seed() -> None:
    a = load_json(OUTPUT / "training_configs/policy_a_config.json")
    b = load_json(OUTPUT / "training_configs/policy_b_config.json")
    assert a["seed"] == b["seed"] == 1000


def test_negative_swapped_dex3_columns_rejected() -> None:
    features = copy.deepcopy(_info(DATASET_A)["features"])
    names = features[ACTION_KEY]["names"]
    names[14], names[15] = names[15], names[14]
    with pytest.raises(ValueError, match="canonical|order"):
        validate_feature_schema(features)


def test_rgb_not_reencoded_and_hardlinked() -> None:
    for root in (DATASET_A, DATASET_B):
        manifest = load_json(root / "meta/g1_packaging_manifest.json")
        assert not manifest["images_reencoded"]
        assert manifest["video_storage_strategy"] == "hardlink"
        assert all(asset["same_inode_as_source"] for asset in manifest["video_assets"])


def test_source_visual_target_action_metadata() -> None:
    for root in (DATASET_A, DATASET_B):
        contract = load_json(root / "meta/g1_training_contract.json")
        assert contract["source_visual_embodiment"] == "ALOHA"
        assert contract["target_action_embodiment"] == "Unitree_G1_Dex3"
        assert "not measured real-G1" in contract["observation_state_semantic"]
