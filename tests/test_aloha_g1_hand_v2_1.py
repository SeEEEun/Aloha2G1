"""Regression tests for the Proposed-only semantic hand v2.1 candidate."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from aloha_g1_hand_v2.collision_eval import make_runtime  # noqa: E402
from aloha_g1_hand_v2.common import load_v1_config  # noqa: E402
from aloha_g1_hand_v2_1.common import V2_1_ROOT, load_source_config  # noqa: E402
from aloha_g1_hand_v2_1.primitives import (  # noqa: E402
    PHASES,
    SemanticPrimitiveBuilder,
    tool_compatibility,
)


@pytest.fixture(scope="session")
def source_config() -> dict:
    return load_source_config()


@pytest.fixture(scope="session")
def v1_config() -> dict:
    return load_v1_config()


@pytest.fixture(scope="session")
def build(v1_config: dict, source_config: dict) -> tuple:
    runtime = make_runtime(v1_config)
    value = SemanticPrimitiveBuilder(runtime, v1_config, source_config).build()
    return runtime, value


@pytest.fixture(scope="session")
def artifact() -> dict:
    path = V2_1_ROOT / "config/proposed_hand_v2_1_candidate.json"
    assert path.is_file(), "run tools/build_proposed_hand_v2_1.py first"
    return json.loads(path.read_text(encoding="utf-8"))


def test_proposed_only_scope(source_config: dict) -> None:
    assert source_config["method"] == "proposed"
    assert source_config["baseline_dataset_a_mutable"] is False
    assert source_config["arm_trajectory_mutable"] is False
    assert source_config["wrist_trajectory_mutable"] is False


def test_exact_contact_ik_is_ablation_not_default(source_config: dict) -> None:
    assert source_config["diagnostic_exact_contact_ik"]["retained"] is True
    assert source_config["diagnostic_exact_contact_ik"]["default_mapper"] is False
    assert source_config["hand_mapper"] == "feasible_semantic_primitive"


def test_source_object_pose_is_not_fabricated(source_config: dict) -> None:
    geometry = source_config["object_class_geometry"]
    assert geometry["source_demo_object_pose_usage"] is False
    assert "TARGET_OBJECT_CLASS_GEOMETRY" in geometry["usage"]


def test_active_model_task_topology(build: tuple) -> None:
    runtime, value = build
    for side in ("left", "right"):
        assert len(value["sides"][side]["task_indices"]) == 5
        assert len(value["sides"][side]["third_indices"]) == 2
        assert set(value["sides"][side]["task_indices"]).isdisjoint(
            value["sides"][side]["third_indices"]
        )
        assert len(runtime.hand_joint_names[side]) == 7


def test_primitive_q_shape_and_finite(build: tuple) -> None:
    _, value = build
    for side in ("left", "right"):
        for phase in PHASES:
            q = np.asarray(value["sides"][side]["states"][phase])
            assert q.shape == (7,)
            assert np.isfinite(q).all()


def test_primitive_joint_limits(build: tuple) -> None:
    runtime, value = build
    for side in ("left", "right"):
        limits = runtime.hand_limits[side]
        for q in value["sides"][side]["states"].values():
            assert np.all(np.asarray(q) >= limits[:, 0] - 1e-9)
            assert np.all(np.asarray(q) <= limits[:, 1] + 1e-9)


def test_grasp_has_no_internal_self_collision(build: tuple) -> None:
    _, value = build
    for side in ("left", "right"):
        metrics = value["sides"][side]["grasp_optimization"]["metrics"]
        assert metrics["same_hand_internal_collision"] is False
        assert metrics["thumb_index_pathological_overlap"] is False


def test_object_class_surface_opening_is_compatible(build: tuple) -> None:
    _, value = build
    for side in ("left", "right"):
        metrics = value["sides"][side]["grasp_optimization"]["metrics"]
        assert metrics["object_class_opening_compatible"] is True
        assert metrics["surface_aperture_m"] > 0.0


def test_semantic_state_aliases_are_explicit(build: tuple) -> None:
    _, value = build
    for side in ("left", "right"):
        states = value["sides"][side]["states"]
        assert tuple(states) == PHASES
        assert np.array_equal(states["GRASP"], states["HOLD"])
        assert np.array_equal(states["OPEN"], states["RELEASE"])


def test_pregrasp_is_geometry_derived(build: tuple) -> None:
    _, value = build
    for side in ("left", "right"):
        row = value["sides"][side]["pregrasp_construction"]
        assert 0.0 < row["interpolation_alpha"] < 1.0
        assert row["clearance_m"] > 0.0
        assert row["candidate_count"] == 201


def test_primitive_construction_is_reproducible(
    v1_config: dict, source_config: dict, build: tuple
) -> None:
    _, first = build
    second_runtime = make_runtime(v1_config)
    second = SemanticPrimitiveBuilder(second_runtime, v1_config, source_config).build()
    for side in ("left", "right"):
        for phase in PHASES:
            assert np.array_equal(
                first["sides"][side]["states"][phase],
                second["sides"][side]["states"][phase],
            )


def test_wrist_to_pinch_transform_is_invertible(build: tuple, v1_config: dict) -> None:
    _, value = build
    compatibility = tool_compatibility(value, v1_config)
    for side in ("left", "right"):
        transform = np.asarray(
            compatibility["sides"][side]["candidate_wrist_to_pinch"]
        )
        assert transform.shape == (4, 4)
        assert np.allclose(transform @ np.linalg.inv(transform), np.eye(4), atol=1e-10)


def test_material_tool_change_is_not_silenced(build: tuple, v1_config: dict) -> None:
    _, value = build
    compatibility = tool_compatibility(value, v1_config)
    assert compatibility["tool_transform_changed_requires_arm_rerun"] is True
    assert compatibility["classification"] == "TOOL_TRANSFORM_CHANGED_REQUIRES_ARM_RERUN"


def test_candidate_has_one_global_third_neutral_per_side(artifact: dict) -> None:
    for side in ("left", "right"):
        neutral = np.asarray(artifact[f"{side}_third_safe_neutral"])
        assert neutral.shape == (2,)
        names = artifact["joint_order"][side]
        indices = [index for index, name in enumerate(names) if "middle" in name]
        assert len(indices) == 2
        for phase in PHASES:
            assert np.array_equal(
                np.asarray(artifact["states"][side][phase])[indices], neutral
            )
    assert artifact["third_finger_policy"]["per_episode_q"] is False
    assert artifact["third_finger_policy"]["per_phase_q"] is False


def test_global_third_neutrals_respect_limits(artifact: dict, build: tuple) -> None:
    runtime, value = build
    for side in ("left", "right"):
        indices = np.asarray(value["sides"][side]["third_indices"], dtype=int)
        limits = runtime.hand_limits[side][indices]
        q = np.asarray(artifact[f"{side}_third_safe_neutral"])
        assert np.all(q >= limits[:, 0])
        assert np.all(q <= limits[:, 1])


def test_third_search_replays_prior_count() -> None:
    audit = json.loads(
        (V2_1_ROOT / "audit/third_finger_collision_audit.json").read_text(
            encoding="utf-8"
        )
    )
    frame_audit = audit["search_frame_audit"]
    assert frame_audit["all_50_episodes_represented"] is True
    assert frame_audit["prior_parity"] is True
    assert frame_audit["current_v1_third_collision_frames_replayed"] == 4037


def test_50_episode_sweep_is_complete() -> None:
    aggregate = json.loads(
        (V2_1_ROOT / "metrics/aggregate_comparison.json").read_text(encoding="utf-8")
    )
    candidate = aggregate["variants"]["proposed_hand_v2_1"]
    assert candidate["episode_count"] == 50
    assert candidate["total_frames"] == 50302
    assert len(candidate["episodes"]) == 50


def test_candidate_collision_improves_without_arm_change() -> None:
    aggregate = json.loads(
        (V2_1_ROOT / "metrics/aggregate_comparison.json").read_text(encoding="utf-8")
    )
    current = aggregate["variants"]["current_proposed_v1"]["collision"]
    candidate = aggregate["variants"]["proposed_hand_v2_1"]["collision"]
    assert candidate["third_finger_related_frames"] < current["third_finger_related_frames"]
    assert candidate["comprehensive_hand_related_frames"] < current[
        "comprehensive_hand_related_frames"
    ]


def test_candidate_finite_and_joint_limits_50_episode() -> None:
    aggregate = json.loads(
        (V2_1_ROOT / "metrics/aggregate_comparison.json").read_text(encoding="utf-8")
    )
    kinematics = aggregate["variants"]["proposed_hand_v2_1"]["kinematics"]
    assert kinematics["finite"] is True
    assert kinematics["joint_limit_violation_count"] == 0


def test_semantic_phase_completeness_and_validity() -> None:
    semantic = json.loads(
        (V2_1_ROOT / "metrics/semantic_phase_validation.json").read_text(
            encoding="utf-8"
        )
    )["proposed_hand_v2_1"]
    assert semantic["unknown_phase_count"] == 0
    assert semantic["transition_count"] > 0
    assert semantic["phase_vocabulary"] == list(PHASES)


def test_dataset_a_and_frozen_arm_checksums_unchanged() -> None:
    integrity = json.loads(
        (V2_1_ROOT / "integrity/after_and_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert integrity["dataset_a_unchanged"] is True
    assert integrity["g1_arm_trajectories_unchanged"] is True


def test_same_config_is_used_for_all_episode_rows() -> None:
    aggregate = json.loads(
        (V2_1_ROOT / "metrics/aggregate_comparison.json").read_text(encoding="utf-8")
    )
    rows = aggregate["variants"]["proposed_hand_v2_1"]["episodes"]
    assert [row["episode_id"] for row in rows] == list(range(50))
    artifact = json.loads(
        (V2_1_ROOT / "config/proposed_hand_v2_1_candidate.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["third_finger_policy"]["per_episode_q"] is False


def test_required_renders_exist() -> None:
    for name in (
        "left_phone_states.png",
        "right_accessory_states.png",
        "third_neutral_comparison.png",
        "hand_v1_vs_v2_vs_v2_1.png",
    ):
        path = V2_1_ROOT / "renders" / name
        assert path.is_file()
        assert path.stat().st_size > 1000


def test_no_episode_or_frame_specific_solver_logic() -> None:
    paths = sorted((ROOT / "tools/aloha_g1_hand_v2_1").glob("*.py"))
    paths += [ROOT / "configs/aloha_g1_hand_v2_1.json"]
    patterns = (
        re.compile(r"if\s+episode(?:_id)?\s*==\s*49"),
        re.compile(r"if\s+frame\s*=="),
        re.compile(r"ep49[_-]?offset", re.IGNORECASE),
        re.compile(r"manual[_ -]?waypoint", re.IGNORECASE),
        re.compile(r"hand_q\s*\[\s*episode", re.IGNORECASE),
    )
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hits.extend((path.name, pattern.pattern) for pattern in patterns if pattern.search(text))
    assert hits == []


def test_candidate_marks_simulation_placeholder_status(artifact: dict) -> None:
    assert artifact["label_status"] == "SIMULATION_PLACEHOLDER_HAND_LABELS"
    assert artifact["authoritative_for_real_g1"] is False
    assert artifact["real_robot_command_allowed"] is False
