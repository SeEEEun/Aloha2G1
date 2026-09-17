from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from aloha_g1_dataset_v1.core import SourceDataset, raw_contact_position  # noqa: E402
from aloha_g1_hand_v2.collision_eval import (  # noqa: E402
    COLLISION_CATEGORIES,
    classify_collision_pair,
    make_runtime,
)
from aloha_g1_hand_v2.contact_mapping import (  # noqa: E402
    map_source_contacts_to_frozen_g1_tool,
)
from aloha_g1_hand_v2.dex3_ik import Dex3InteractionIK  # noqa: E402
from aloha_g1_hand_v2.interaction_extraction import (  # noqa: E402
    AlohaContactExtractor,
    REPRESENTATION_FALLBACK,
    detect_first_complete_grasp,
)


V1 = ROOT / "outputs/g1_dataset_retargeting_v1"
V2 = ROOT / "outputs/g1_dataset_retargeting_hand_v2"


@pytest.fixture(scope="session")
def v1_config() -> dict:
    return json.loads((V1 / "config/aloha_g1_retargeting_v1.json").read_text())


@pytest.fixture(scope="session")
def v2_config() -> dict:
    return json.loads((ROOT / "configs/aloha_g1_hand_v2.json").read_text())


@pytest.fixture(scope="session")
def dataset() -> SourceDataset:
    return SourceDataset()


@pytest.fixture(scope="session")
def runtime(v1_config):
    return make_runtime(v1_config)


@pytest.fixture(scope="session")
def extractor(v1_config):
    return AlohaContactExtractor(v1_config["models"]["aloha_xml"])


@pytest.fixture(scope="session")
def development() -> dict:
    return json.loads((V2 / "summary/pipeline_results.json").read_text())["development"]


@pytest.fixture(scope="session")
def selected_episode(development, dataset):
    return dataset.episode(int(development["selection"]["selected_episode"]))


def test_01_collision_attribution_50ep_parses_and_matches_v1():
    data = json.loads((V2 / "collision_audit/collision_attribution_50ep.json").read_text())
    assert data["proposed"]["v1_exact_reconstruction"] is True
    assert data["proposed"]["v1_collision_frames"] == 4696
    assert len(data["proposed"]["episodes"]) == 50


def test_02_collision_categories_are_deterministic_and_complete():
    pairs = {
        ("left_shoulder_yaw_link", "torso_link"): "ARM_TORSO",
        ("left_wrist_yaw_link", "torso_link"): "WRIST_OR_PALM_TORSO",
        ("left_hand_thumb_2_link", "torso_link"): "THUMB_TORSO",
        ("left_hand_index_1_link", "torso_link"): "INDEX_TORSO",
        ("left_hand_middle_1_link", "torso_link"): "THIRD_TORSO",
        ("left_hand_thumb_2_link", "left_hand_index_1_link"): "THUMB_INDEX_SELF",
        ("left_hand_thumb_2_link", "left_wrist_yaw_link"): "THUMB_WRIST_SELF",
        ("left_hand_index_1_link", "right_hand_middle_1_link"): "HAND_HAND",
    }
    first = {pair: classify_collision_pair(pair) for pair in pairs}
    second = {pair: classify_collision_pair(pair) for pair in pairs}
    assert first == second == pairs
    data = json.loads((V2 / "collision_audit/collision_attribution_50ep.json").read_text())
    assert tuple(data["proposed"]["category_frame_incidence"].keys()) == COLLISION_CATEGORIES


def test_03_semantic_grasp_detection_is_deterministic(development):
    episode_id = int(development["selection"]["selected_episode"])
    with np.load(
        V1 / "proposed" / f"episode_{episode_id:06d}" / "g1_hand_action.npz",
        allow_pickle=False,
    ) as payload:
        labels = payload["left_phase"].copy()
    assert detect_first_complete_grasp(labels) == detect_first_complete_grasp(labels)
    assert detect_first_complete_grasp(labels) == development["event"]


def test_04_interaction_representation_schema(development):
    source = development["source_interaction"]
    required = {
        "schema_version",
        "representation_mode",
        "contact_A_tcp_m",
        "contact_B_tcp_m",
        "closing_axis_tcp",
        "approach_axis_tcp",
        "gripper_width_m",
        "geometry_approximation",
    }
    assert required <= set(source)
    assert source["schema_version"] == "aloha_source_interaction_v2"


def test_05_object_frame_requires_authoritative_provenance(extractor, selected_episode):
    action = selected_episode.action[0]
    with pytest.raises(ValueError, match="authoritative provenance"):
        extractor.extract(action, "left", authoritative_object_pose=np.eye(4))
    with pytest.raises(NotImplementedError, match="reserved"):
        extractor.extract(
            action,
            "left",
            authoritative_object_pose=np.eye(4),
            object_pose_provenance="authoritative-test-provenance",
        )


def test_06_tcp_fallback_is_explicitly_labeled(development):
    source = development["source_interaction"]
    assert source["representation_mode"] == REPRESENTATION_FALLBACK
    assert source["object_pose_used"] is False
    assert source["simulation_scene_used_as_source_annotation"] is False


def test_07_source_contact_targets_are_finite(development):
    source = development["source_interaction"]
    mapping = development["contact_mapping"]
    for key in ("contact_A_tcp_m", "contact_B_tcp_m", "closing_axis_tcp", "approach_axis_tcp"):
        assert np.isfinite(np.asarray(source[key])).all()
    for key in ("thumb_target_m", "index_target_m"):
        assert np.isfinite(np.asarray(mapping[key])).all()


def test_08_source_contact_geometry_is_deterministic(
    extractor, selected_episode, development
):
    frame = int(development["event"]["grasp_onset"])
    first = extractor.extract(selected_episode.action[frame], "left")
    second = extractor.extract(selected_episode.action[frame], "left")
    for key in ("contact_A_tcp_m", "contact_B_tcp_m", "closing_axis_tcp"):
        assert np.array_equal(first[key], second[key])
    assert first["gripper_width_m"] == second["gripper_width_m"]


def test_09_dex3_q_shape_is_seven():
    with np.load(V2 / "development/dex3_solution.npz", allow_pickle=False) as payload:
        assert payload["dex3_q"].shape == (7,)


def test_10_dex3_q_is_finite():
    with np.load(V2 / "development/dex3_solution.npz", allow_pickle=False) as payload:
        assert np.isfinite(payload["dex3_q"]).all()


def test_11_dex3_q_respects_joint_limits(runtime):
    with np.load(V2 / "development/dex3_solution.npz", allow_pickle=False) as payload:
        q = payload["dex3_q"]
    limits = runtime.hand_limits["left"]
    assert np.all(q >= limits[:, 0] - 1e-9)
    assert np.all(q <= limits[:, 1] + 1e-9)


def test_12_thumb_index_fk_is_deterministic(runtime, development):
    episode_id = int(development["selection"]["selected_episode"])
    frame = int(development["event"]["grasp_onset"])
    with np.load(
        V1 / "proposed" / f"episode_{episode_id:06d}" / "g1_arm_action.npz",
        allow_pickle=False,
    ) as arm, np.load(
        V1 / "proposed" / f"episode_{episode_id:06d}" / "g1_hand_action.npz",
        allow_pickle=False,
    ) as hand, np.load(V2 / "development/dex3_solution.npz", allow_pickle=False) as solution:
        args = (arm["action"][frame], solution["dex3_q"], hand["right_action"][frame])
        runtime.assign(*args)
        first = np.r_[raw_contact_position(runtime, "left_A"), raw_contact_position(runtime, "left_B")]
        runtime.assign(*args)
        second = np.r_[raw_contact_position(runtime, "left_A"), raw_contact_position(runtime, "left_B")]
    assert np.array_equal(first, second)


def test_13_contact_error_metric_is_deterministic(
    runtime, v1_config, v2_config, development
):
    solver = Dex3InteractionIK(runtime, v1_config, v2_config)
    episode_id = int(development["selection"]["selected_episode"])
    frame = int(development["event"]["grasp_onset"])
    with np.load(
        V1 / "proposed" / f"episode_{episode_id:06d}" / "g1_arm_action.npz",
        allow_pickle=False,
    ) as arm, np.load(
        V1 / "proposed" / f"episode_{episode_id:06d}" / "g1_hand_action.npz",
        allow_pickle=False,
    ) as hand, np.load(V2 / "development/dex3_solution.npz", allow_pickle=False) as solution:
        first = solver.evaluate(
            solution["dex3_q"], arm["action"][frame], hand["right_action"][frame], development["contact_mapping"]
        )
        second = solver.evaluate(
            solution["dex3_q"], arm["action"][frame], hand["right_action"][frame], development["contact_mapping"]
        )
    assert first["mean_task_finger_error_m"] == second["mean_task_finger_error_m"]


def test_14_dataset_a_checksum_is_unchanged():
    integrity = json.loads((V2 / "integrity/after_and_comparison.json").read_text())
    assert integrity["dataset_a_unchanged"] is True
    assert integrity["before"]["dataset_a_tree_sha256"] == integrity["after"]["dataset_a_tree_sha256"]


def test_15_all_g1_arm_trajectory_checksums_are_unchanged():
    integrity = json.loads((V2 / "integrity/after_and_comparison.json").read_text())
    assert integrity["g1_arm_trajectories_unchanged"] is True
    assert integrity["before"]["proposed_arm_files"] == integrity["after"]["proposed_arm_files"]


def test_16_development_arm_q_is_byte_identical():
    integrity = json.loads((V2 / "integrity/after_and_comparison.json").read_text())
    assert integrity["development_arm_frame_byte_identical"] is True


def test_17_no_episode_or_frame_specific_solver_logic():
    audit = json.loads((V2 / "tests/anti_overfitting_scan.json").read_text())
    assert audit["pass"] is True
    assert audit["hits"] == []


def test_18_no_episode_specific_contact_coordinates(v2_config):
    assert v2_config["contact_mapping"]["per_episode_transform_allowed"] is False
    assert v2_config["contact_mapping"]["per_frame_cartesian_residual_allowed"] is False
    source = (ROOT / "tools/aloha_g1_hand_v2/contact_mapping.py").read_text().lower()
    assert "manual_contact" not in source
    assert "hand_written_waypoint" not in source
    assert "ep49_offset" not in source


def test_19_same_ik_configuration_across_smoke_episodes():
    with (V2 / "smoke_test/episode_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert len({row["episode_id"] for row in rows}) == 3
    assert len({row["solver_config_sha256"] for row in rows}) == 1
    assert {row["per_episode_weight_tuning"] for row in rows} == {"False"}


def test_20_same_frame_conversion_is_bitwise_reproducible(development):
    assert development["bitwise_reproducible_q"] is True
    assert np.array_equal(
        development["interaction_ik"]["q"], development["repeat_solution_q"]
    )


def test_21_third_finger_is_neutral_and_non_task(development):
    metrics = development["interaction_ik"]["metrics"]
    assert metrics["third_finger_neutral_exact"] is True
    assert metrics["third_finger_policy"] == "ACTIVE_MODEL_OPEN_NEUTRAL_FIXED"
    assert metrics["third_finger_task_object_contact"] == "NOT_EVALUATED_SOURCE_OBJECT_POSE_UNAVAILABLE"


def test_22_source_width_equals_contact_point_distance(development):
    source = development["source_interaction"]
    actual = np.linalg.norm(
        np.asarray(source["contact_B_tcp_m"]) - np.asarray(source["contact_A_tcp_m"])
    )
    assert actual == pytest.approx(source["gripper_width_m"], abs=1e-12)


def test_23_contact_topology_is_thumb_index(runtime):
    assert runtime.contacts["left_A"].role == "A"
    assert "thumb" in runtime.contacts["left_A"].link
    assert runtime.contacts["left_B"].role == "B"
    assert "index" in runtime.contacts["left_B"].link


def test_24_cli_exposes_proposed_only_hand_mapper_modes():
    completed = subprocess.run(
        [sys.executable, "tools/repair_proposed_hand_v2.py", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0
    assert "--method {proposed}" in completed.stdout
    assert "--hand-mapper {interaction_ik,semantic_primitive}" in completed.stdout


def test_25_mapped_contact_targets_use_frozen_wrist(
    runtime, v1_config, extractor, selected_episode, development
):
    episode_id = int(development["selection"]["selected_episode"])
    frame = int(development["event"]["grasp_onset"])
    source = extractor.extract(selected_episode.action[frame], "left")
    with np.load(
        V1 / "proposed" / f"episode_{episode_id:06d}" / "g1_arm_action.npz",
        allow_pickle=False,
    ) as arm, np.load(
        V1 / "proposed" / f"episode_{episode_id:06d}" / "g1_hand_action.npz",
        allow_pickle=False,
    ) as hand:
        first = map_source_contacts_to_frozen_g1_tool(
            source, runtime, arm["action"][frame], hand["right_action"][frame], v1_config
        )
        second = map_source_contacts_to_frozen_g1_tool(
            source, runtime, arm["action"][frame], hand["right_action"][frame], v1_config
        )
    assert first["frozen_wrist_conditioning"] is True
    assert np.array_equal(first["thumb_target_m"], second["thumb_target_m"])
