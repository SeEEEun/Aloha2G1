"""Unit and contract tests for the episode-independent retargeting v1 pipeline."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from aloha_g1_dataset_v1.core import (  # noqa: E402
    DEFAULT_CONFIG,
    PHASE_NAMES,
    RetargetingPipeline,
    SourceDataset,
    inverse_transform,
    parse_action_channels,
    transform,
)


@pytest.fixture(scope="session")
def pipeline() -> RetargetingPipeline:
    return RetargetingPipeline()


def test_dataset_discovery_count_is_exactly_fifty() -> None:
    dataset = SourceDataset()
    assert dataset.episode_ids() == list(range(50))
    assert dataset.audit()["usable_episode_count"] == 50
    assert sum(dataset.audit()["episode_lengths"]) == 50_302


def test_action_channel_parsing(pipeline: RetargetingPipeline) -> None:
    action = pipeline.dataset.episode(0, max_frames=8).action
    parsed = parse_action_channels(action)
    assert parsed["left_arm"].shape == (8, 6)
    assert parsed["right_arm"].shape == (8, 6)
    np.testing.assert_array_equal(parsed["left_arm"], action[:, 0:6])
    np.testing.assert_array_equal(parsed["left_gripper"], action[:, 6])
    np.testing.assert_array_equal(parsed["right_arm"], action[:, 7:13])
    np.testing.assert_array_equal(parsed["right_gripper"], action[:, 13])


def test_aloha_fk_is_deterministic(pipeline: RetargetingPipeline) -> None:
    action = pipeline.dataset.episode(0, max_frames=12).action
    first = pipeline.source_kinematics.compute(action)
    second = pipeline.source_kinematics.compute(action)
    for key in ("qpos", "left_position", "right_position", "left_rotation", "right_rotation"):
        np.testing.assert_array_equal(first[key], second[key])


def test_source_target_transforms_are_invertible(pipeline: RetargetingPipeline) -> None:
    source = np.asarray(pipeline.config["source_frames"]["aloha_link6_to_tcp"])
    for value in (
        source,
        pipeline.g1.tool_local["left"],
        pipeline.g1.tool_local["right"],
        pipeline.g1.palm_local["left"],
        pipeline.g1.palm_local["right"],
        np.asarray(pipeline.config["tool_mapping"]["left_tool_transform"]),
        np.asarray(pipeline.config["tool_mapping"]["right_tool_transform"]),
    ):
        np.testing.assert_allclose(value @ inverse_transform(value), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(inverse_transform(value) @ value, np.eye(4), atol=1e-12)
        assert np.linalg.det(value[:3, :3]) == pytest.approx(1.0, abs=1e-12)


def test_workspace_transform_is_deterministic(pipeline: RetargetingPipeline) -> None:
    episode = pipeline.dataset.episode(0, max_frames=24)
    source_fk = pipeline.source_kinematics.compute(episode.action)
    first = pipeline.representation.build("proposed", source_fk)
    second = pipeline.representation.build("proposed", source_fk)
    for key in (
        "left_wrist_position", "right_wrist_position",
        "left_wrist_rotation", "right_wrist_rotation",
        "target_tool_midpoint", "target_tool_relative",
    ):
        np.testing.assert_array_equal(first[key], second[key])


def test_no_episode_id_dependency_in_config_or_executable_source() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["workspace_mapping"]["per_episode_anchor_tuning"] is False
    assert config["workspace_mapping"]["per_episode_scale_tuning"] is False
    serialized = json.dumps(config).lower()
    for forbidden in ("ep49", "frame-163", "frame-209", "correction_frame_ranges"):
        assert forbidden not in serialized

    paths = [
        ROOT / "tools/retarget_aloha_g1_dataset.py",
        ROOT / "tools/aloha_g1_dataset_v1/core.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for forbidden in ("ep49", "frame == 163", "frame == 209", "episode == 49"):
            assert forbidden not in lowered
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            names = {child.id for child in ast.walk(node.left) if isinstance(child, ast.Name)}
            constants = {
                child.value for comparator in node.comparators
                for child in ast.walk(comparator) if isinstance(child, ast.Constant)
            }
            assert not (names & {"episode", "episode_id", "frame"} and constants & {49, 163, 209})


def test_baseline_and_proposed_share_exact_ik_config(pipeline: RetargetingPipeline) -> None:
    assert pipeline.solver.config is pipeline.config["ik"]
    assert pipeline.config["ik"]["representation_specific_code_allowed"] is False
    assert pipeline.config["ik"]["bimanual_residual_weight"] == 0.0


def test_batch_exporter_shapes_and_finite_values(
    pipeline: RetargetingPipeline, tmp_path: Path
) -> None:
    result = pipeline.convert("baseline", 0, max_frames=36)
    directory = pipeline.export(result, tmp_path)
    with np.load(directory / "g1_arm_action.npz", allow_pickle=False) as arm:
        assert arm["action"].shape == (36, 14)
        assert np.isfinite(arm["action"]).all()
    with np.load(directory / "g1_hand_action.npz", allow_pickle=False) as hand:
        assert hand["action"].shape == (36, 14)
        assert np.isfinite(hand["action"]).all()
    with np.load(directory / "g1_full_action.npz", allow_pickle=False) as full:
        assert full["action"].shape == (36, 28)
        assert np.isfinite(full["action"]).all()
        assert not bool(full["real_robot_command_allowed"])
    for required in (
        "source_metadata.json", "retargeting_metrics.json", "validation.json", "manifest.json"
    ):
        assert (directory / required).is_file()


def test_joint_limits_and_semantic_phase_completeness(pipeline: RetargetingPipeline) -> None:
    result = pipeline.convert("proposed", 0, max_frames=120)
    q = result.solver["q"]
    assert np.all(q >= pipeline.g1.limits[:, 0] - 1e-12)
    assert np.all(q <= pipeline.g1.limits[:, 1] + 1e-12)
    valid = set(PHASE_NAMES.tolist())
    for side in ("left", "right"):
        labels = result.hand[f"{side}_phase"]
        assert labels.shape == (120,)
        assert not np.any(labels == "")
        assert set(labels.tolist()) <= valid
    assert result.hand["unknown_phase_count"] == 0


def test_comparison_uses_one_common_task_critical_mask(
    pipeline: RetargetingPipeline,
) -> None:
    baseline = pipeline.convert("baseline", 0, max_frames=120)
    proposed = pipeline.convert("proposed", 0, max_frames=120)
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            baseline.detected[side].phase, proposed.detected[side].phase
        )
        key = f"{side}_task_critical_frame_count"
        assert baseline.metrics["task_space"][key] == proposed.metrics["task_space"][key]
    assert (
        baseline.metrics["task_space"]["task_critical_mask_definition"]
        == proposed.metrics["task_space"]["task_critical_mask_definition"]
    )


@pytest.mark.parametrize("method", ("baseline", "proposed"))
def test_conversion_is_bit_reproducible(
    pipeline: RetargetingPipeline, method: str
) -> None:
    first = pipeline.convert(method, 0, max_frames=30)
    second = pipeline.convert(method, 0, max_frames=30)
    np.testing.assert_array_equal(first.solver["q"], second.solver["q"])
    np.testing.assert_array_equal(first.hand["left"], second.hand["left"])
    np.testing.assert_array_equal(first.hand["right"], second.hand["right"])
