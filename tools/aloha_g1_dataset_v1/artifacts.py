"""Auditable discovery and frame-semantics artifacts for retargeting v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core import ROOT, RetargetingPipeline, atomic_json, inverse_transform


def component_inventory() -> dict[str, Any]:
    """Record exactly which historical components were reused or isolated."""
    rows = [
        {
            "component": "authoritative LeRobot v3 source loading",
            "source_file": "reports/magsafe_lerobot_v3_build_report.json; reports/magsafe_lerobot_v3_manifest.csv",
            "function/class": "SourceDataset",
            "currently_reusable": True,
            "changes_required": "Read the integrated Parquet directly and enforce the 50-episode/hash contract.",
            "reason": "The original action column, not optimized_action, is the dataset-construction source.",
        },
        {
            "component": "stationary ALOHA FK",
            "source_file": "tools/validate_smolvla_in_stationary_aloha_mujoco.py",
            "function/class": "mapped_qpos; fk; load_validated_model",
            "currently_reusable": True,
            "changes_required": "None to the algorithm; invoked through SourceKinematics.",
            "reason": "It already validates joint order, gripper mimic joints, and deterministic TCP FK.",
        },
        {
            "component": "ALOHA TCP/tool frame",
            "source_file": "tools/replay_stationary_gopark.py",
            "function/class": "TCP_OFFSET_LOCAL; map_row_to_qpos",
            "currently_reusable": True,
            "changes_required": "Expose it explicitly in the v1 frame graph.",
            "reason": "The static [0.1487, 0, -0.00105] m transform is robot-model geometry.",
        },
        {
            "component": "ALOHA-to-G1 axis alignment",
            "source_file": "tools/search_g1_task_ready_anchor.py; generate_aloha_work_posture_dex3.py",
            "function/class": "make_align_rotation; ALIGN_RPY",
            "currently_reusable": True,
            "changes_required": "Freeze the pre-dataset calibration and prohibit per-episode changes.",
            "reason": "The rotation is a global base-axis calibration rather than an episode correction.",
        },
        {
            "component": "workspace scale and task-ready anchor",
            "source_file": "converted_runs/magsafe_20260723_162750/aloha_work_posture_dex3/posture_anchor_search_results.npz; converted_runs/magsafe_20260723_162750/dynamic_bimanual_spacing/g1_dynamic_bimanual_full_trajectory.npz",
            "function/class": "selected calibration row; task_arm[0]",
            "currently_reusable": True,
            "changes_required": "Use nominal-posture FK as the immutable anchor; remove literal scene/object offsets.",
            "reason": "This calibration predates and is outside the authoritative 50-demo source set.",
        },
        {
            "component": "G1 temporal IK",
            "source_file": "tools/retarget_episode49_optimized_action_to_g1.py",
            "function/class": "temporal_solve numerical weights and hard-limit projection",
            "currently_reusable": "PARTIAL",
            "changes_required": "Move the numerical objective into SharedTemporalIK and remove the fixed source, fixed length, target builder, and bimanual residual.",
            "reason": "Both v1 methods must use exactly one IK backend; legacy orchestration was episode-specific.",
        },
        {
            "component": "joint model/order/limits and wrist Jacobian",
            "source_file": "tools/validate_g1_targets_and_sparse_ik.py",
            "function/class": "validate_model; current_bimanual_state; arm limit helpers",
            "currently_reusable": True,
            "changes_required": "Use the actual wrist body for both methods instead of conflating it with the palm proxy.",
            "reason": "The model validation and name-based 14-arm ordering are authoritative.",
        },
        {
            "component": "velocity and acceleration regularization",
            "source_file": "tools/retarget_episode49_optimized_action_to_g1.py",
            "function/class": "temporal_solve wv/wa terms",
            "currently_reusable": True,
            "changes_required": "Apply unchanged in SharedTemporalIK to both representations.",
            "reason": "These are solver-quality controls and therefore must not differ by method.",
        },
        {
            "component": "branch continuity diagnostics",
            "source_file": "tools/retarget_episode49_optimized_action_to_g1.py",
            "function/class": "local-median joint-step branch check",
            "currently_reusable": True,
            "changes_required": "Generalize to arbitrary length and record the per-episode count.",
            "reason": "It has no task- or episode-dependent parameter.",
        },
        {
            "component": "bimanual midpoint and relative-vector representation",
            "source_file": "tools/retarget_episode49_consensus_relative_bimanual_to_g1.py",
            "function/class": "midpoint/relative-vector change mapping",
            "currently_reusable": True,
            "changes_required": "Apply it to original demonstrations with a nominal G1 pinch-frame anchor and no exact-spacing clamp.",
            "reason": "Changes in relations are general; absolute ALOHA spacing is morphology-specific.",
        },
        {
            "component": "semantic gripper phases",
            "source_file": "tools/aloha_magsafe_semantics/gripper_phase.py; configs/aloha_magsafe_semantic_detector_v1.json",
            "function/class": "detect_gripper_phases; GripperResult",
            "currently_reusable": True,
            "changes_required": "Run the same signal-derived state machine independently on every source trajectory.",
            "reason": "It uses time-scaled robust clustering and contains no frame-number rules.",
        },
        {
            "component": "predefined Dex3 primitives",
            "source_file": "configs/dex3_magsafe_grasp_primitives.sim.json; configs/dex3_abc_finger_mapping.sim.json",
            "function/class": "named OPEN/PREGRASP/GRASP/HOLD/RELEASE q vectors",
            "currently_reusable": "SIMULATION_ONLY",
            "changes_required": "Label every exported hand trajectory SIMULATION_PLACEHOLDER_HAND_LABELS.",
            "reason": "The primitives are deterministic and global but are not calibrated on real Dex3 hardware.",
        },
        {
            "component": "source/target EE-frame audit",
            "source_file": "outputs/scene_registered_retargeting/current_layout_ep49_trajbooster_frame_consistency_v18_1/complete_ee_frame_graph_v18_1.json; tools/build_episode49_trajbooster_frame_consistency_v18_1.py",
            "function/class": "explicit wrist/palm/tool transform graph",
            "currently_reusable": "CONCEPT_ONLY",
            "changes_required": "Rebuild only static robot-model transforms; discard scene-registered, object, and episode corrections.",
            "reason": "Frame typing is general, while the legacy scene registration is not.",
        },
        {
            "component": "G1 wrist/palm/thumb-index pinch-center FK",
            "source_file": "tools/aloha_g1_v15/kinematics.py",
            "function/class": "ActiveG1Dex3; wrist_pose; palm_pose; contact_pose",
            "currently_reusable": True,
            "changes_required": "Evaluate in the shared raw G1 base convention and verify static wrist-to-pinch transforms by FK.",
            "reason": "The active-model name mapping avoids ordering assumptions.",
        },
        {
            "component": "collision validation",
            "source_file": "tools/aloha_g1_v15/kinematics.py; generate_aloha_work_posture_dex3.py",
            "function/class": "penetrating_contacts; contacts",
            "currently_reusable": True,
            "changes_required": "Count penetrating cross-arm/torso contacts and exclude adjacent same-side hand-chain contacts.",
            "reason": "This is deterministic model-level self-collision validation, not PhysX task execution.",
        },
        {
            "component": "scene/object waypoint correction stack",
            "source_file": "tools/build_episode49_target_phase_anchored_v12.py; tools/finalize_episode49_physical_contact_anchored_v13.py and v14-v18 result builders",
            "function/class": "literal phase frames, per-stage residuals, scene-registered waypoints",
            "currently_reusable": False,
            "changes_required": "Isolated to archived legacy paths and never imported by v1.",
            "reason": "Those paths contain episode-inspected corrections forbidden in the dataset retargeter.",
        },
    ]
    return {
        "schema_version": "existing_component_inventory_v1",
        "new_pipeline": "tools/aloha_g1_dataset_v1/core.py",
        "legacy_files_modified": False,
        "components": rows,
    }


def trajbooster_scope() -> dict[str, Any]:
    return {
        "schema_version": "trajbooster_baseline_scope_v1",
        "method_name": "TrajBooster-style upper-body baseline",
        "not_a_full_reproduction": True,
        "paper": "https://arxiv.org/pdf/2509.11839",
        "official_repository": "https://github.com/OpenHelix-Team/OpenTrajBooster",
        "USED_IDEAS": [
            "dual-arm 6D end-effector trajectories as a cross-embodiment interface",
            "global source-to-target workspace mapping followed by target IK",
            "separation of arm/body retargeting from target-hand state mapping",
            "fixed target open/close hand states",
        ],
        "EXCLUDED_IDEAS": [
            "lower-body locomotion and height-varying whole-body manipulation",
            "Manager policy, Worker policy, and locomotion reinforcement learning",
            "Harmonized Online DAgger and all VLA/policy training",
            "body-height augmentation",
            "AgiBot-specific z-score, scale, clipping, and workspace constants",
            "real-robot execution",
        ],
        "ALOHA_G1_SPECIFIC_REPLACEMENTS": [
            "stationary ALOHA joint-target action -> validated ALOHA TCP FK",
            "pre-dataset ALOHA/G1 tabletop calibration -> fixed 0.42 similarity scale and -7 degree pitch alignment",
            "fixed-base Unitree G1/Dex3 MuJoCo model -> actual wrist-body temporal IK",
            "ALOHA gripper opening -> deterministic binary Dex3 OPEN/CLOSE labels",
        ],
        "FAIRNESS_NOTES": [
            "The baseline is an adaptation, not a claimed TrajBooster reproduction.",
            "Both methods share source data, timing, robot model, nominal posture, solver, limits, regularization, tolerances, collision checker, and sampling rate.",
            "The baseline is not intentionally weakened: it retains 6D wrist orientation, temporal IK, smoothing, and hard-limit projection.",
            "Method differences are target representation and hand semantics, not solver budget.",
        ],
    }


def frame_semantics(pipeline: RetargetingPipeline) -> dict[str, Any]:
    cfg = pipeline.config
    nominal_wrist = pipeline.g1.nominal_wrist_poses()
    nominal_pinch = {
        side: nominal_wrist[side] @ pipeline.g1.tool_local[side]
        for side in ("left", "right")
    }
    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        wrist_to_pinch = pipeline.g1.tool_local[side]
        sides[side] = {
            "source_dynamic_chain": [
                f"ALOHA base -> {side} arm terminal link (robot FK)",
                f"{side} arm terminal link -> {side} ALOHA TCP / gripper closing center (static SE(3))",
            ],
            "target_dynamic_chain": [
                f"G1 base -> {side} shoulder -> elbow -> wrist (robot FK)",
                f"{side} wrist -> palm (static diagnostic proxy)",
                f"{side} wrist -> thumb distal and index distal (Dex3 FK; phase-dependent)",
                f"thumb/index contacts -> physical pinch center and orthonormal pinch axes",
            ],
            "T_aloha_terminal_tcp": cfg["source_frames"]["aloha_link6_to_tcp"],
            "T_g1_wrist_palm": cfg["target_frames"][f"{side}_wrist_to_palm"],
            "T_g1_wrist_physical_pinch_static": wrist_to_pinch.tolist(),
            "T_g1_physical_pinch_wrist_static": inverse_transform(wrist_to_pinch).tolist(),
            "nominal_T_g1_base_wrist": nominal_wrist[side].tolist(),
            "nominal_T_g1_base_physical_pinch": nominal_pinch[side].tolist(),
            "physical_pinch_contact_labels": list(pipeline.g1.pinch_labels[side]),
            "static_transform_fk_max_abs_error": pipeline.tool_transform_fk_errors[side],
            "task_target_to_wrist_formula": "T_G1base_wrist_target(t) = T_G1base_task_tool_target(t) @ inverse(T_G1wrist_physical_pinch_static)",
        }
    return {
        "schema_version": "aloha_g1_frame_semantics_v1",
        "units": {"translation": "metre", "angle": "radian", "matrix": "4x4 homogeneous SE(3)"},
        "conventions": cfg["coordinate_conventions"],
        "source_frame_names": [
            "ALOHA base", "left/right arm terminal link", "left/right ALOHA TCP",
            "left/right gripper closing center / semantic grasp point",
        ],
        "target_frame_names": [
            "G1 base", "left/right shoulder", "left/right wrist", "left/right palm",
            "left/right thumb distal", "left/right index distal",
            "left/right physical thumb-index pinch center",
        ],
        "source_tcp_and_gripper_closing_center_coincident": True,
        "target_wrist_and_physical_grasp_frame_conflated": False,
        "physical_pinch_frame_construction": {
            "origin": "midpoint of the configured physical thumb-pad and index-pad contact centers",
            "positive_y_closing_axis": "unit vector from thumb contact to index contact",
            "positive_x_approach_axis": "G1 wrist local +x projected orthogonal to the closing axis",
            "positive_z_lateral_axis": "normalized cross(approach, closing)",
            "orthonormalization": "closing is recomputed as cross(lateral, approach)",
            "phase_dependence": "actual FK frame changes with Dex3 q; the static wrist-to-pinch transform is calibrated at the global GRASP primitive",
        },
        "static_source_tool_to_target_tool_axis_transforms": {
            "left": cfg["tool_mapping"]["left_tool_transform"],
            "right": cfg["tool_mapping"]["right_tool_transform"],
            "formula": cfg["orientation_mapping"]["formula"],
        },
        "left_right_differences": "Mirrored source/tool axis mappings, palm y offsets, pinch contact role labels, and wrist-to-pinch transforms are explicit.",
        "sides": sides,
        "baseline_IK_target_frame": "actual G1 wrist body",
        "proposed_representation_frame": "physical thumb-index pinch/task-tool frame",
        "proposed_IK_target_frame": "actual G1 wrist body after exact static SE(3) inversion",
        "hand_primitive_calibration_status": cfg["hand_mapping"]["label_calibration_status"],
    }


def write_discovery_artifacts(pipeline: RetargetingPipeline, output_root: str | Path) -> None:
    output_root = Path(output_root).resolve()
    discovery = output_root / "discovery"
    atomic_json(discovery / "source_dataset_audit.json", pipeline.dataset.audit())
    atomic_json(discovery / "existing_component_inventory.json", component_inventory())
    atomic_json(discovery / "trajbooster_baseline_scope.json", trajbooster_scope())
    atomic_json(discovery / "frame_semantics.json", frame_semantics(pipeline))
