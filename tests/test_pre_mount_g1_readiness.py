from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def imported_names(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_frozen_policy_b_inputs_are_unchanged() -> None:
    assert sha256("outputs/doll_handoff_dataset_b_final/FINAL_DATASET_B_MANIFEST.json") == "693b8b5531c8adf1a7921452c7af6dbc89feb99f84aa70185ab33dca2cb11b44"
    assert sha256("outputs/doll_handoff_dataset_b_final/final_source_manifest.json") == "8fd073d78cfe2e27075954cd43910b1cac43ccf7f373ea528866a82d3cd04f76"
    assert sha256("outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2/checkpoints/020000/pretrained_model/model.safetensors") == "bfe3e2aa6529967a12831a6f0bb91104b704835b2f43733072489b9ea2b68395"


def test_official_upstreams_are_pinned_and_external() -> None:
    audit = read_json("docs/pre_mount_upstream_audit.json")
    expected = {
        "unitreerobotics/unitree_sdk2": "f29ee9f234851e9e79f75102c0f9e83008d8fdd1",
        "unitreerobotics/xr_teleoperate": "845b25a32f7febedf220e830952a7134897adb9d",
        "unitreerobotics/unitree_lerobot": "41c2805742de879ddab2d8d6beaeaf215f876395",
        "unitreerobotics/unitree_sim_isaaclab": "e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc",
        "realsenseai/librealsense": "e196cefa896e312d79c2df400c7623aa1e9c62ac",
    }
    assert {row["repository"]: row["commit_sha"] for row in audit["repositories"]} == expected
    assert all(row["license"] in {"Apache-2.0", "BSD-3-Clause"} for row in audit["repositories"])
    assert audit["upstream_worktrees_modified"] is False
    assert "external/third_party/" in (ROOT / ".gitignore").read_text()


def test_pending_camera_is_explicit_and_operationally_fail_closed() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from deployment_camera_config import load_camera_config

    template = read_json("configs/helmet_d455_mount_template.json")
    serialized = json.dumps(template)
    assert "CAMERA_NOT_MOUNTED" in serialized
    assert "EXTRINSIC_NOT_CALIBRATED" in serialized
    with pytest.raises(RuntimeError, match="FINAL_CAMERA_NOT_CALIBRATED"):
        load_camera_config(
            ROOT / "configs/camera_presets/HELMET_D455_FINAL_PENDING.json",
            purpose="test operational consumer",
        )


def test_movable_camera_parent_requires_locked_pose_or_timestamped_fk(tmp_path: Path) -> None:
    import sys
    from dataclasses import replace

    sys.path.insert(0, str(ROOT / "tools"))
    from deployment_camera_config import load_camera_config
    from helmet_d455.parent_pose import ParentLinkPoseAdapter

    camera = load_camera_config(
        ROOT / "configs/camera_presets/SOURCE_LIKE_CAM_HIGH_DIAGNOSTIC.json",
        purpose="parent adapter test",
    )
    locked_camera = replace(
        camera,
        mount={
            "status": "MEASURED",
            "parent_link": "head_link",
            "parent_link_can_move": True,
            "locked_parent_pose_id": "HEAD_DEPLOYMENT_LOCK",
            "parent_link_fk_provider": None,
            "parent_from_camera_matrix": np.eye(4).tolist(),
        },
    )
    with pytest.raises(RuntimeError, match="requires locked parent pose"):
        ParentLinkPoseAdapter(locked_camera, locked_parent_pose_id=None, fk_json=None)
    locked = ParentLinkPoseAdapter(
        locked_camera, locked_parent_pose_id="HEAD_DEPLOYMENT_LOCK", fk_json=None
    )
    assert locked.read(123)["mode"] == "LOCKED_PARENT_POSE"

    fk_camera = replace(
        camera,
        mount={
            "status": "MEASURED",
            "parent_link": "head_link",
            "parent_link_can_move": True,
            "locked_parent_pose_id": None,
            "parent_link_fk_provider": "read_only_g1_fk_bridge_v1",
            "parent_from_camera_matrix": np.eye(4).tolist(),
        },
    )
    with pytest.raises(RuntimeError, match="requires --parent-fk-json"):
        ParentLinkPoseAdapter(fk_camera, locked_parent_pose_id=None, fk_json=None)
    bridge = tmp_path / "fk.json"
    bridge.write_text(
        json.dumps(
            {
                "parent_link": "head_link",
                "host_monotonic_timestamp_ns": 1_000_000_000,
                "task_from_parent_matrix": np.eye(4).tolist(),
            }
        )
    )
    adapter = ParentLinkPoseAdapter(fk_camera, locked_parent_pose_id=None, fk_json=bridge)
    assert adapter.read(1_050_000_000)["timestamp_aligned_fk_used"] is True


def test_helmet_synthetic_capture_solve_validate_and_board_dimensions() -> None:
    board = read_json("configs/helmet_d455_charuco_board.json")
    assert board["squares_x"] == 7 and board["squares_y"] == 5
    assert board["square_length_m"] == 0.05
    assert board["marker_length_m"] == 0.0375
    assert board["printed_pattern_width_mm"] == 350.0
    assert board["printed_pattern_height_mm"] == 250.0
    assert board["opencv_detector_frame_conversion"]["opencv_from_canonical_matrix"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.25],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    validation = read_json("outputs/pre_mount_readiness/helmet_d455/synthetic_validation_v4.json")
    assert validation["status"] == "PASS_SYNTHETIC_OFFLINE_ONLY_NOT_FREEZABLE"
    assert validation["charuco"]["status"] == "PASS"
    assert validation["black_workspace_projection"]["status"] == "PASS"
    assert validation["freeze_permitted"] is False
    legacy_api_validation = read_json(
        "outputs/pre_mount_readiness/helmet_d455/synthetic_validation_v4_opencv45.json"
    )
    assert legacy_api_validation["status"] == "PASS_SYNTHETIC_OFFLINE_ONLY_NOT_FREEZABLE"
    printable = read_json("calibration/helmet_d455/printable/print_manifest.json")
    assert printable["spec_sha256"] == sha256("configs/helmet_d455_charuco_board.json")
    assert not list(ROOT.rglob("helmet_d455_final.json"))


def test_camera_config_is_required_by_all_renderer_and_rollout_paths() -> None:
    consumers = {
        "tools/render_doll_handoff_g1visual_dataset.py": ("--camera-config", "--dataset-variant"),
        "tools/run_policy_b_isaac_doll_handoff.py": ("--camera-config", "--policy-variant"),
        "tools/prepare_doll_handoff_g1visual_render_plan.py": ("--camera-config", "--dataset-variant"),
    }
    for relative, flags in consumers.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert all(flag in source for flag in flags)
    pending = read_json("configs/camera_presets/HELMET_D455_FINAL_PENDING.json")
    diagnostic = read_json("configs/camera_presets/SOURCE_LIKE_CAM_HIGH_DIAGNOSTIC.json")
    assert pending["render_allowed"] is False and pending["rollout_allowed"] is False
    assert diagnostic["final_deployment_use"] is False

    rollout_source = (ROOT / "tools/run_policy_b_isaac_doll_handoff.py").read_text(
        encoding="utf-8"
    )
    assert rollout_source.count("Camera(") == 1
    assert "set_world_poses_from_view" not in rollout_source

    smoke_root = ROOT / "outputs/pre_mount_readiness/isaac_camera_config_policy_b_smoke"
    camera_smoke = json.loads(
        (smoke_root / "camera/isaac_render_verification.json").read_text(encoding="utf-8")
    )
    stage_smoke = json.loads(
        (smoke_root / "stage0_inference/stage_report.json").read_text(encoding="utf-8")
    )
    assert camera_smoke["status"] == "PASS"
    assert all(camera_smoke["checks"].values())
    assert stage_smoke["status"] == "PASS"
    assert stage_smoke["policy_variant"] == "B"
    assert stage_smoke["action_shape"] == [50, 28]
    assert stage_smoke["camera_config_sha256"] == camera_smoke["camera_config"]["config_sha256"]


def test_final_view_ab_dry_plan_is_equal_and_did_not_render() -> None:
    report = read_json("outputs/pre_mount_readiness/final_view_ab/pipeline_dry_run.json")
    assert report["a_b_same_camera_config_sha256"]
    assert report["a_b_same_renderer"]
    assert report["a_b_same_frame_count"]
    assert report["a_b_same_task_schema_strategy_selection_rule"]
    assert report["final_50_episode_render_executed"] is False
    assert report["variants"]["B"]["source_gate"]["ready"] is True
    assert report["variants"]["A"]["source_gate"]["ready"] is False
    paired = read_json("outputs/pre_mount_readiness/final_view_ab/paired_builder_diagnostic_dry_run.json")
    assert paired["status"] == "PASS_DRY_RUN_NO_DATASET_CREATED"
    assert paired["paired_episodes"] == 100 and paired["paired_frames"] == 68956
    assert all(paired["supervision_identity"].values())
    diagnostic_plan = read_json("outputs/pre_mount_readiness/final_view_ab/b_diagnostic_render_plan/render_plan_manifest.json")
    pending_plan = read_json("outputs/pre_mount_readiness/final_view_ab/b_pending_helmet_render_plan/render_plan_manifest.json")
    assert pending_plan["status"] == "PREPARED_BLOCKED_PENDING_FINAL_CAMERA"
    assert diagnostic_plan["dataset_action_logical_sha256"] == pending_plan["dataset_action_logical_sha256"]
    assert diagnostic_plan["dataset_state_logical_sha256"] == pending_plan["dataset_state_logical_sha256"]


def test_baseline_a_render_metadata_fallback_uses_frozen_fk_only() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from doll_handoff_retargeting.models import G1Kinematics
    from prepare_doll_handoff_g1visual_render_plan import world_hand_transforms

    common = read_json("configs/doll_handoff_retargeting/common_config.template.json")
    scene = read_json("isaaclab_doll_handoff_scene/scene_layout.json")
    trajectory_path = ROOT / "outputs/dataset_a_final50_retargeting/baseline/trajectories/doll_handoff_20260820_ep000.npz"
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        assert "achieved_left_static_whole_hand_position_world" not in trajectory.files
        transforms = world_hand_transforms(trajectory, "left", G1Kinematics(common, scene))
    assert transforms.shape == (692, 4, 4)
    assert np.isfinite(transforms).all()
    assert np.allclose(transforms[:, 3], [0.0, 0.0, 0.0, 1.0])


def test_dataset_a_hard_fail_is_complete_and_has_no_silent_exclusion() -> None:
    report = read_json("outputs/pre_mount_readiness/dataset_a/dataset_a_validation.json")
    assert report["status"] == "HARD_FAIL_STOPPED_BEFORE_PACKAGING"
    assert report["episodes_attempted"] == 50
    assert len(report["hard_fails"]) == 22
    assert report["structural_classification_counts"] == {
        "PASS": 28,
        "WARNING": 0,
        "HARD_FAIL": 22,
    }
    assert report["source_episode_identity"] == "PASS_EXACT_FINAL_COMMON_50_ORDER"
    assert sum(row["validation_status"] == "FAIL_IK" for row in report["hard_fails"]) == 18
    assert sum(row["validation_status"] == "FAIL_COLLISION" for row in report["hard_fails"]) == 4
    assert report["silent_exclusions"] == 0
    assert report["dataset_created"] is False
    assert not (ROOT / "datasets/doll_handoff_trajectory_a_50").exists()


def test_policy_a_training_is_blocked_but_conditions_are_frozen_equal() -> None:
    gate = read_json("outputs/pre_mount_readiness/policy_a_source/training_gate.json")
    assert gate["status"] == "NOT_STARTED_BLOCKED_BY_DATASET_A_HARD_FAIL"
    assert gate["source_conditions_equal_to_original_policy_b"] is True
    assert all(gate["invariants"].values())
    assert gate["command_executed"] is False
    assert gate["checkpoint_created"] is False
    assert gate["nine_phase_probe"] == "NOT_RUN_NO_POLICY_A_CHECKPOINT"


def test_xr_bridge_simulation_and_equal_ab_configs() -> None:
    manifest = read_json("outputs/pre_mount_readiness/xr_bridge_sim_v2/xr_bridge_manifest.json")
    assert manifest["status"] == "PASS_SIMULATION_DRY_RUN"
    assert manifest["lerobot"]["status"] == "PASS"
    assert manifest["lerobot"]["episodes"] == 3 and manifest["lerobot"]["frames"] == 36
    split = read_json("outputs/pre_mount_readiness/xr_bridge_sim_v2/adaptation_reference_split.json")
    assert split["same_split_required_for_policy_a_and_policy_b"] is True
    shared = read_json("configs/xr_post_training/shared_equal_ab.json")
    a = read_json("configs/xr_post_training/policy_a_equal.json")
    b = read_json("configs/xr_post_training/policy_b_equal.json")
    assert shared["execution_allowed"] is False
    assert a["shared_config"] == b["shared_config"]
    assert a["override_fields_allowed"] == b["override_fields_allowed"] == ["policy_variant", "source_checkpoint", "output"]
    assert a["execution_allowed"] is False and b["execution_allowed"] is False


def test_dex3_recorder_is_read_only_and_captures_all_raw_channels() -> None:
    forbidden = {"ChannelPublisher", "HandCmd_", "MotionSwitcherClient", "SportClient"}
    assert not (imported_names("tools/record_dex3_contact_readonly.py") & forbidden)
    source = (ROOT / "tools/record_dex3_contact_readonly.py").read_text(encoding="utf-8")
    assert "rt/lf/dex3/left/state" in source and "rt/lf/dex3/right/state" in source
    manifest = read_json("outputs/pre_mount_readiness/dex3/synthetic_readonly.manifest.json")
    assert manifest["read_only"] is True and manifest["command_path"] == "ABSENT"
    assert manifest["press_sensor_records_per_hand"] == 9
    assert manifest["raw_values_per_press_sensor_record"] == 12
    with np.load(ROOT / "outputs/pre_mount_readiness/dex3/synthetic_readonly.npz") as archive:
        assert archive["left_q"].shape[1:] == (7,)
        assert archive["right_q"].shape[1:] == (7,)
        assert archive["left_press_raw"].shape[1:] == (9, 12)
        assert archive["right_press_raw"].shape[1:] == (9, 12)
    analysis = read_json("outputs/pre_mount_readiness/dex3/analysis/analysis.json")
    assert analysis["force_units"] == "NOT_ASSIGNED"
    assert analysis["contact_threshold"] == "NOT_ASSIGNED"


def test_contact_adapter_refuses_uncalibrated_template() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from contact_feedback_grasp_adapter import ContactFeedbackGraspAdapter

    adapter = ContactFeedbackGraspAdapter(ROOT / "configs/dex3_contact_calibration.template.json")
    with pytest.raises(RuntimeError, match="remains DISABLED"):
        adapter.activate()


def test_real_inference_harness_is_log_only_for_a_and_b() -> None:
    forbidden = {"ChannelPublisher", "HandCmd_", "MotionSwitcherClient", "SportClient"}
    assert not (imported_names("tools/run_real_g1_inference_only.py") & forbidden)
    for variant in ("a", "b"):
        root = f"outputs/pre_mount_readiness/real_g1_inference/simulation_policy_{variant}_v2"
        manifest = read_json(f"{root}/manifest.json")
        assert manifest["status"] == "INFERENCE_ONLY_NO_COMMAND_PUBLISHER"
        assert manifest["command_publisher"] == "ABSENT"
        assert manifest["robot_motion"] == "NOT_REQUESTED"
        assert manifest["camera_parent_pose"] == [
            {"mode": "SIMULATION_PENDING_CAMERA_NOT_OPERATIONAL"},
            {"mode": "SIMULATION_PENDING_CAMERA_NOT_OPERATIONAL"},
        ]
        with np.load(ROOT / root / "inference_log.npz") as archive:
            assert archive["raw_action_chunks_50x28"].shape == (2, 50, 28)
            assert archive["measured_state_28d"].shape == (2, 28)
    real_b = read_json(
        "outputs/pre_mount_readiness/real_g1_inference/simulation_policy_b_real_checkpoint/manifest.json"
    )
    assert real_b["checkpoint_model_sha256"] == "bfe3e2aa6529967a12831a6f0bb91104b704835b2f43733072489b9ea2b68395"
    assert "saved preprocessor/postprocessor" in real_b["dataset_normalization"]
    preflight = read_json(
        "outputs/pre_mount_readiness/real_g1_inference/simulation_policy_b_real_checkpoint/raw_action_preflight.json"
    )
    assert preflight["status"] == "PASS"
