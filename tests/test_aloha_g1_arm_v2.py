from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "outputs/g1_dataset_retargeting_arm_v2"
INTEGRATED = ROOT / "outputs/g1_dataset_retargeting_integrated_v2"
V1 = ROOT / "outputs/g1_dataset_retargeting_v1"
HAND = ROOT / "outputs/g1_dataset_retargeting_hand_v2_1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_hand_dependency_is_byte_identical_and_ready() -> None:
    source_candidate = HAND / "config/proposed_hand_v2_1_candidate.json"
    source_readiness = HAND / "summary/integration_readiness.json"
    frozen_candidate = ARM / "dependencies/proposed_hand_v2_1_candidate.json"
    frozen_readiness = ARM / "dependencies/proposed_hand_v2_1_readiness.json"
    assert sha256(source_candidate) == sha256(frozen_candidate)
    assert sha256(source_readiness) == sha256(frozen_readiness)
    assert load(frozen_readiness)["ready_for_common_arm_rerun"] is True


def test_split_is_deterministic_40_10_and_disjoint() -> None:
    split = load(ARM / "split/calibration_validation_split.json")
    calibration = split["calibration_episode_ids"]
    validation = split["validation_episode_ids"]
    assert len(calibration) == 40
    assert len(validation) == 10
    assert set(calibration).isdisjoint(validation)
    assert sorted(calibration + validation) == list(range(50))
    recomputed = sorted(
        range(50),
        key=lambda value: hashlib.sha256(
            f"{split['salt']}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    assert validation == sorted(recomputed[:10])


def test_selected_mapping_is_one_shared_frozen_config() -> None:
    selected = ARM / "candidates/frozen_common_arm_v2_config.json"
    fairness = load(INTEGRATED / "summary/fairness_audit.json")
    assert fairness["selected_config_sha256_a"] == sha256(selected)
    assert fairness["selected_config_sha256_b"] == sha256(selected)
    assert fairness["identical_ik_and_config_verified"] is True


def test_mapping_rotation_and_tool_transforms_are_invertible() -> None:
    selected = load(ARM / "candidates/frozen_common_arm_v2_config.json")
    rotation = np.asarray(selected["global_mapping"]["axis_rotation"], dtype=float)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    for key in ("left_wrist_to_physical_pinch", "right_wrist_to_physical_pinch"):
        transform = np.asarray(selected["target_frames"][key], dtype=float)
        assert transform.shape == (4, 4)
        assert np.allclose(transform @ np.linalg.inv(transform), np.eye(4), atol=1e-12)


def test_arm_and_integrated_episode_counts_are_50_each() -> None:
    assert len(list((ARM / "baseline").glob("episode_*"))) == 50
    assert len(list((ARM / "proposed").glob("episode_*"))) == 50
    assert len(list((INTEGRATED / "dataset_a").glob("episode_*"))) == 50
    assert len(list((INTEGRATED / "dataset_b").glob("episode_*"))) == 50


def test_a_and_b_roots_are_separate() -> None:
    assert (INTEGRATED / "dataset_a").resolve() != (INTEGRATED / "dataset_b").resolve()
    for episode_id in range(50):
        assert (INTEGRATED / "dataset_a" / f"episode_{episode_id:06d}").is_dir()
        assert (INTEGRATED / "dataset_b" / f"episode_{episode_id:06d}").is_dir()


def test_all_actions_are_finite_with_expected_shape_and_frozen_arm() -> None:
    for dataset_name, method in (("dataset_a", "baseline"), ("dataset_b", "proposed")):
        for episode_id in range(50):
            integrated = INTEGRATED / dataset_name / f"episode_{episode_id:06d}"
            arm_folder = ARM / method / f"episode_{episode_id:06d}"
            with np.load(integrated / "g1_full_action.npz", allow_pickle=False) as payload:
                full = payload["action"]
            with np.load(integrated / "g1_arm_action.npz", allow_pickle=False) as payload:
                arm = payload["action"]
            with np.load(arm_folder / "g1_arm_action.npz", allow_pickle=False) as payload:
                frozen = payload["action"]
            assert full.ndim == 2 and full.shape[1] == 28
            assert arm.shape == frozen.shape == (len(full), 14)
            assert np.isfinite(full).all()
            assert np.array_equal(arm, frozen)


def test_dataset_a_uses_only_binary_mapper() -> None:
    for episode_id in range(50):
        folder = INTEGRATED / "dataset_a" / f"episode_{episode_id:06d}"
        manifest = load(folder / "manifest.json")
        assert manifest["hand_dependency_sha256"] is None
        with np.load(folder / "g1_hand_action.npz", allow_pickle=False) as payload:
            assert str(payload["mapper"]) == "unchanged_v1_binary_open_close"
            assert set(payload["left_phase"].astype(str)).issubset({"OPEN", "CLOSE"})
            assert set(payload["right_phase"].astype(str)).issubset({"OPEN", "CLOSE"})


def test_dataset_b_uses_exact_frozen_hand_candidate() -> None:
    expected = sha256(HAND / "config/proposed_hand_v2_1_candidate.json")
    valid = {"OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE"}
    for episode_id in range(50):
        folder = INTEGRATED / "dataset_b" / f"episode_{episode_id:06d}"
        manifest = load(folder / "manifest.json")
        assert manifest["hand_dependency_sha256"] == expected
        with np.load(folder / "g1_hand_action.npz", allow_pickle=False) as payload:
            assert str(payload["mapper"]) == "frozen_proposed_hand_v2_1"
            assert set(payload["left_phase"].astype(str)).issubset(valid)
            assert set(payload["right_phase"].astype(str)).issubset(valid)


def test_joint_limits_and_semantics_pass_per_episode_validation() -> None:
    for dataset_name in ("dataset_a", "dataset_b"):
        for episode_id in range(50):
            folder = INTEGRATED / dataset_name / f"episode_{episode_id:06d}"
            metrics = load(folder / "retargeting_metrics.json")
            assert metrics["finite_values"] is True
            assert metrics["joint_limit_violation_count"] == 0
            assert metrics["semantic"]["phase_completeness"] is True
            assert metrics["semantic"]["transition_validity"] is True


def test_source_hashes_are_unchanged() -> None:
    integrity = load(INTEGRATED / "summary/integrity.json")
    assert integrity["source_hashes_unchanged"] is True
    assert integrity["dataset_a_v1_unchanged"] is True
    assert integrity["hand_v2_1_unchanged"] is True


def test_deterministic_rerun_passed() -> None:
    report = load(ARM / "tests/deterministic_rerun.json")
    assert report["pass"] is True
    assert all(report["checks"].values())


def test_episode_metrics_tables_have_50_rows() -> None:
    for name in ("dataset_a_episode_metrics.csv", "dataset_b_episode_metrics.csv"):
        with (INTEGRATED / "summary" / name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 50
        assert [int(row["episode_id"]) for row in rows] == list(range(50))


def test_no_episode_or_frame_specific_execution_logic() -> None:
    sources = list((ROOT / "tools/aloha_g1_arm_v2").glob("*.py"))
    sources += [ROOT / "tools/run_common_arm_v2.py", ROOT / "configs/aloha_g1_arm_v2.json"]
    forbidden = [
        re.compile(r"if\s+episode(?:_id)?\s*=="),
        re.compile(r"if\s+frame(?:_id|_index)?\s*=="),
        re.compile("ep" + "49", re.IGNORECASE),
    ]
    hits = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(text):
                hits.append((path.name, pattern.pattern))
    assert hits == []


def test_readiness_and_fairness_artifacts_are_consistent() -> None:
    stage = load(ARM / "summary/stage_a_readiness.json")
    readiness = load(INTEGRATED / "summary/training_readiness.json")
    fairness = load(INTEGRATED / "summary/fairness_audit.json")
    assert stage["ready"] is True
    assert readiness["stage_a_ready"] is True
    assert fairness["identical_ik_and_config_verified"] is True
