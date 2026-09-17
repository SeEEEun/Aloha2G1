from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.g1_training_schema_v1.causal_alignment import action_chunk_indices, validate_episode_timestamps
from tools.g1_training_schema_v1.constants import (
    ACTION_DIM,
    CANONICAL_JOINT_NAMES,
    CHUNK_SIZE,
    FPS,
    IMAGE_KEY,
    JOINT_SPECS,
    STATE_DIM,
    policy_features,
)
from tools.g1_training_schema_v1.episode_filter import (
    accepted_episode_ids,
    audit_episode_root,
    intersection_manifest,
    matched_episode_intersection,
)
from tools.g1_training_schema_v1.lerobot_writer import (
    inspect_packaging_inputs,
    package_dataset,
)
from tools.g1_training_schema_v1.normalization import (
    fit_accepted_normalization,
    fit_shared_diagnostic_normalization,
)
from tools.g1_training_schema_v1.source_audit import SourceDatasetIndex, build_source_audit
from tools.g1_training_schema_v1.state_adapter import adapt_target_qpos
from tools.g1_training_schema_v1.target_contract import (
    EpisodeDecision,
    discover_episode_dirs,
    inspect_episode,
    load_retargeted_trajectory,
)
from tools.g1_training_schema_v1.validator import (
    deterministic_tree_hash,
    validate_ab_schema_equality,
    validate_feature_schema,
    validate_packaged_dataset,
    validate_separate_output_roots,
    validate_source_not_copied_as_target,
    validate_target_episode,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lerobot_magsafe_50_cam_high_v3"
DRY_A = ROOT / "outputs/g1_dataset_retargeting_v1/baseline"
DRY_B = ROOT / "outputs/g1_dataset_retargeting_v1/proposed"


@pytest.fixture(scope="module")
def source() -> SourceDatasetIndex:
    return SourceDatasetIndex(SOURCE)


@pytest.fixture(scope="module")
def synthetic_pass_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("g1_schema_synthetic_pass")
    episode = root / "episode_000049"
    episode.mkdir()
    with np.load(DRY_A / "episode_000049/g1_full_action.npz", allow_pickle=False) as template:
        joint_names = np.asarray(template["joint_names"])
    frame_count = 990
    np.savez(
        episode / "g1_full_action.npz",
        action=np.zeros((frame_count, ACTION_DIM), dtype=np.float32),
        timestamps=np.arange(frame_count, dtype=np.float64) / FPS,
        fps=np.asarray(FPS, dtype=np.float64),
        joint_names=joint_names,
        fixture_marker=np.asarray("STRUCTURAL_SYNTHETIC_ONLY_NOT_ACCEPTED_TRAINING_DATA"),
    )
    shutil.copy2(DRY_A / "episode_000049/source_metadata.json", episode / "source_metadata.json")
    (episode / "validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "pass": True,
                "fixture_marker": "STRUCTURAL_TEST_ONLY_NOT_ACCEPTED_TRAINING_DATA",
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def packaged_root(
    tmp_path_factory: pytest.TempPathFactory, synthetic_pass_root: Path
) -> Path:
    output = tmp_path_factory.mktemp("g1_schema_package_parent") / "dataset_a"
    result = package_dataset("dataset_a", synthetic_pass_root, SOURCE, output)
    assert result["status"] == "PASS"
    return output


def _common_info(source: SourceDatasetIndex) -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1_fixed_base_dex3_retargeted",
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": policy_features(source.info["features"][IMAGE_KEY]),
    }


def _decision(episode_id: int, accepted: bool) -> EpisodeDecision:
    return EpisodeDecision(
        episode_id=episode_id,
        accepted=accepted,
        status="PASS" if accepted else "FAIL_IK",
        failure_category=None if accepted else "FAIL_IK",
        first_failed_frame=None if accepted else 3,
        source_hash=f"source-{episode_id}",
        retarget_output_reference=f"episode_{episode_id:06d}/g1_full_action.npz",
        retarget_output_sha256=f"retarget-{episode_id}",
        validation_path=f"episode_{episode_id:06d}/validation.json",
    )


def test_source_dataset_discovery(source: SourceDatasetIndex) -> None:
    assert len(source.episodes) == 50
    assert sorted(source.episodes) == list(range(50))
    assert sum(episode.length for episode in source.episodes.values()) == 50302


def test_source_feature_schema_reading() -> None:
    audit = build_source_audit(SOURCE)
    assert audit["dataset_version"] == "v3.0"
    assert audit["state_dimension"] == audit["action_dimension"] == 14
    assert audit["rgb_keys"] == [IMAGE_KEY]
    assert audit["fps"] == 30


def test_target_joint_order_unique_and_physical() -> None:
    assert len(JOINT_SPECS) == len(set(CANONICAL_JOINT_NAMES)) == 28
    assert [joint.index for joint in JOINT_SPECS] == list(range(28))
    assert CANONICAL_JOINT_NAMES[14:21] == (
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    )


def test_target_state_and_action_dimensions() -> None:
    trajectory = load_retargeted_trajectory(DRY_A / "episode_000000")
    adapted = adapt_target_qpos(trajectory)
    assert adapted.observation_state.shape == (976, STATE_DIM)
    assert adapted.action.shape == (976, ACTION_DIM)
    assert adapted.internal_state_semantic == "retargeted_target_state"


def test_a_b_schema_equality(source: SourceDatasetIndex) -> None:
    a = _common_info(source)
    b = copy.deepcopy(a)
    validate_ab_schema_equality(a, b)


def test_negative_a_b_mismatched_feature_dimensions_fails(source: SourceDatasetIndex) -> None:
    a = _common_info(source)
    b = copy.deepcopy(a)
    b["features"]["action"]["shape"] = [27]
    with pytest.raises(ValueError, match="dimension|schema mismatch|dimensions"):
        validate_ab_schema_equality(a, b)


def test_same_row_causal_alignment(source: SourceDatasetIndex) -> None:
    trajectory = load_retargeted_trajectory(DRY_A / "episode_000000")
    adapted = adapt_target_qpos(trajectory)
    arrays = source.read_episode_arrays(0)
    validate_target_episode(adapted.observation_state, adapted.action, adapted.timestamps, len(arrays["timestamp"]))
    assert np.array_equal(adapted.observation_state, adapted.action)


def test_timestamp_monotonicity_and_exact_grid(source: SourceDatasetIndex) -> None:
    arrays = source.read_episode_arrays(49)
    validate_episode_timestamps(arrays["timestamp"], len(arrays["timestamp"]))
    broken = arrays["timestamp"].copy()
    broken[20] = broken[19]
    with pytest.raises(ValueError, match="monotonic"):
        validate_episode_timestamps(broken, len(broken))


def test_frame_count_matching_rejects_mismatch(source: SourceDatasetIndex) -> None:
    trajectory = load_retargeted_trajectory(DRY_A / "episode_000000")
    adapted = adapt_target_qpos(trajectory)
    with pytest.raises(ValueError, match="state shape"):
        validate_target_episode(adapted.observation_state, adapted.action, adapted.timestamps, 975)


def test_episode_boundary_and_last_frame_handling() -> None:
    indices, padding = action_chunk_indices(8, 10, CHUNK_SIZE)
    assert indices[:2].tolist() == [8, 9]
    assert np.all(indices[2:] == 9)
    assert padding[:2].tolist() == [False, False]
    assert padding[2:].all()
    final_indices, final_padding = action_chunk_indices(9, 10, CHUNK_SIZE)
    assert np.all(final_indices == 9)
    assert not final_padding[0] and final_padding[1:].all()


def test_failed_episode_rejection_is_auditable() -> None:
    decision = inspect_episode(DRY_A / "episode_000000")
    assert not decision.accepted
    assert decision.failure_category == "FAIL_IK"
    assert decision.first_failed_frame == 845
    assert decision.source_hash and decision.retarget_output_sha256
    _, decisions = audit_episode_root(DRY_A)
    assert accepted_episode_ids(decisions) == []


def test_non_default_failure_policies_fail_fast() -> None:
    with pytest.raises(NotImplementedError, match="unsupported"):
        inspect_packaging_inputs(
            "dataset_a", DRY_A, SOURCE, failure_policy="include_with_mask"
        )


def test_matched_a_b_intersection_generation() -> None:
    a = {0: _decision(0, True), 1: _decision(1, True), 2: _decision(2, False)}
    b = {0: _decision(0, False), 1: _decision(1, True), 2: _decision(2, True)}
    assert matched_episode_intersection(a, b) == [1]
    manifest = intersection_manifest(a, b)
    assert manifest["matched_episode_ids"] == [1]
    assert manifest["counts"] == {"dataset_a": 2, "dataset_b": 2, "matched": 1}


def test_finite_state_action_validation(source: SourceDatasetIndex) -> None:
    trajectory = load_retargeted_trajectory(DRY_A / "episode_000000")
    adapted = adapt_target_qpos(trajectory)
    state = adapted.observation_state.copy()
    state[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_target_episode(state, adapted.action, adapted.timestamps, len(state))


def test_no_aloha_state_copied_as_g1_state(source: SourceDatasetIndex) -> None:
    source_arrays = source.read_episode_arrays(0)
    trajectory = load_retargeted_trajectory(DRY_A / "episode_000000")
    target = adapt_target_qpos(trajectory).observation_state
    validate_source_not_copied_as_target(source_arrays["observation.state"], target)
    assert source_arrays["observation.state"].shape[1] == 14
    assert target.shape[1] == 28


def test_no_method_id_in_policy_input(source: SourceDatasetIndex) -> None:
    features = policy_features(source.info["features"][IMAGE_KEY])
    validate_feature_schema(features)
    features["method"] = {"dtype": "int64", "shape": [1], "names": None}
    with pytest.raises(ValueError, match="feature key mismatch|method identity"):
        validate_feature_schema(features)


def test_source_rgb_reference_validity(source: SourceDatasetIndex) -> None:
    paths = {source.validate_rgb_reference(episode_id) for episode_id in source.episodes}
    assert len(paths) == 2
    assert all(path.suffix == ".mp4" and path.stat().st_size > 0 for path in paths)


def test_normalization_fit_only_on_accepted_train_data() -> None:
    state_a = np.arange(56, dtype=np.float32).reshape(2, 28)
    state_b = state_a + 2
    records = [
        {"accepted": True, "observation.state": state_a, "action": state_a},
        {"accepted": True, "observation.state": state_b, "action": state_b},
    ]
    stats = fit_accepted_normalization(records)
    assert stats["action"]["count"] == [4]
    assert np.allclose(stats["action"]["mean"], np.concatenate([state_a, state_b]).mean(axis=0))
    records.append({"accepted": False, "observation.state": state_a, "action": state_a})
    with pytest.raises(ValueError, match="non-accepted"):
        fit_accepted_normalization(records)


def test_shared_normalization_is_explicitly_diagnostic() -> None:
    values = np.zeros((2, 28), dtype=np.float32)
    record = {"accepted": True, "observation.state": values, "action": values}
    result = fit_shared_diagnostic_normalization([record], [record])
    assert result["mode"] == "SHARED_DIAGNOSTIC_NOT_DEFAULT"
    assert result["statistics"]["action"]["count"] == [4]


def test_packaged_dataset_structural_validation(packaged_root: Path) -> None:
    result = validate_packaged_dataset(packaged_root, SOURCE)
    assert result["status"] == "PASS"
    assert result["episodes"] == 1 and result["frames"] == 990
    manifest = json.loads((packaged_root / "meta/g1_packaging_manifest.json").read_text())
    assert manifest["method_metadata_only"] == "dataset_a"
    assert not manifest["method_is_policy_input"]
    assert not manifest["images_duplicated"]
    assert manifest["output_episode_mapping"] == [{"output_episode_id": 0, "source_episode_id": 49}]


def test_deterministic_packaging(
    tmp_path: Path, synthetic_pass_root: Path, packaged_root: Path
) -> None:
    second = tmp_path / "second_package"
    package_dataset("dataset_a", synthetic_pass_root, SOURCE, second)
    assert deterministic_tree_hash(packaged_root) == deterministic_tree_hash(second)


def test_a_b_separate_output_roots(tmp_path: Path) -> None:
    a = tmp_path / "dataset_a"
    b = tmp_path / "dataset_b"
    validate_separate_output_roots(a, b)
    with pytest.raises(ValueError, match="separate"):
        validate_separate_output_roots(a, a)


def test_dry_run_never_writes_dataset(tmp_path: Path) -> None:
    output = tmp_path / "must_not_exist"
    manifest = inspect_packaging_inputs("dataset_a", DRY_A, SOURCE)
    assert manifest["counts"] == {"discovered": 50, "accepted_native": 0, "selected": 0, "rejected": 50}
    assert not output.exists()


def test_dry_run_validates_selected_pass_inputs(synthetic_pass_root: Path) -> None:
    manifest = inspect_packaging_inputs("dataset_a", synthetic_pass_root, SOURCE)
    assert manifest["selected_episode_ids"] == [49]
    assert manifest["validated_selected_inputs"][0]["status"] == "PASS_INPUT_VALIDATION"
    assert manifest["validated_selected_inputs"][0]["frame_count"] == 990


def test_lerobot_061_load_and_chunk_mask(packaged_root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    deltas = {
        "observation.state": [0.0],
        "action": [index / FPS for index in range(CHUNK_SIZE)],
    }
    dataset = LeRobotDataset(
        repo_id="structural/g1_schema_v1_test",
        root=packaged_root,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="torchcodec",
    )
    item = dataset[len(dataset) - 1]
    assert tuple(item["observation.state"].shape) == (1, STATE_DIM)
    assert tuple(item["action"].shape) == (CHUNK_SIZE, ACTION_DIM)
    assert item["action_is_pad"].tolist() == [False] + [True] * (CHUNK_SIZE - 1)
    assert item["task"] == (
        "Remove the MagSafe accessory from the phone and place the phone on the MagSafe charger."
    )
