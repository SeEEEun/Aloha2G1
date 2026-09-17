from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/doll_handoff_retargeting/natural_arm_audit"
BEFORE = AUDIT / "final_before_resolver_off"
AFTER = AUDIT / "final_after_common_natural_arm_candidate"
EPISODES = (0, 24, 49)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(root: Path, episode: int) -> dict:
    return _json(
        root
        / "proposed/metrics"
        / f"doll_handoff_20260820_ep{episode:03d}.json"
    )


def test_natural_arm_is_common_and_has_no_task_specific_inputs() -> None:
    config = _json(ROOT / "configs/doll_handoff_retargeting/common_config.template.json")
    natural = config["natural_arm_redundancy"]
    assert natural["enabled"] is True
    assert natural["scope"] == "COMMON_BASELINE_AND_PROPOSED"
    assert natural["primary_target_mutation_allowed"] is False
    assert natural["episode_specific_parameters"] is False
    assert natural["phase_specific_parameters"] is False
    assert natural["task_specific_elbow_target"] is False


@pytest.mark.parametrize("episode", EPISODES)
def test_matched_smoke_targets_are_identical_and_after_is_usable(episode: int) -> None:
    before = _metrics(BEFORE, episode)
    after = _metrics(AFTER, episode)
    assert before["cartesian_target_sha256"] == after["cartesian_target_sha256"]
    assert after["natural_arm"]["primary_cartesian_targets_changed"] is False
    assert after["ik_success_rate"] >= 0.95
    assert after["joint_limit_violation_count"] == 0
    assert after["branch_discontinuity_count"] == 0
    assert (
        after["collisions"]["invalid_self_body_collision_frames"]
        <= before["collisions"]["invalid_self_body_collision_frames"]
    )
    assert (
        after["mean_ik_task_error_m"] - before["mean_ik_task_error_m"]
        <= 0.0001
    )
    assert after["semantics"]["right_grasp_before_left_release"] is True
    assert after["scene_diagnostics"]["right_final_release_xy_inside_bin_opening"] is True


def test_scene_hash_remains_authoritative() -> None:
    config = _json(ROOT / "configs/doll_handoff_retargeting/common_config.template.json")
    scene = Path(config["scene_config"])
    assert hashlib.sha256(scene.read_bytes()).hexdigest() == config["scene_config_sha256"]


def test_review_manifest_has_all_three_views() -> None:
    manifest = _json(AUDIT / "final_review/visual_review_manifest.json")
    assert manifest["primary_cartesian_targets_changed"] is False
    assert [row["episode_index"] for row in manifest["entries"]] == list(EPISODES)
    for row in manifest["entries"]:
        for camera in ("overview", "top", "side"):
            path = Path(row["outputs"][f"comparison_{camera}"])
            assert path.is_file() and path.stat().st_size > 0
