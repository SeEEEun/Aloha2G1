#!/usr/bin/env python3
"""Create the complete authoritative physical scene freeze after PASS regression."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_direct_physical_eval35"
SCENE_OUT = OUT / "00_authoritative_physical_scene"
RUN = SCENE_OUT / "corrected_task_relative_scripted_regression_run"
PREFLIGHT = SCENE_OUT / "task_relative_scripted_validation/TASK_FRAME_NUMERICAL_PREFLIGHT.json"
AUDIT = SCENE_OUT / "AUTHORITATIVE_PHYSICAL_SCENE_AUDIT.json"
PROVENANCE = SCENE_OUT / "provenance_wrong_registration_freeze"
FREEZE_DIR = OUT / "00_freeze"
MANIFEST = FREEZE_DIR / "DIRECT_EVAL35_FREEZE_MANIFEST.json"
REPORT = FREEZE_DIR / "DIRECT_EVAL35_FREEZE_MANIFEST.md"
COMMANDS = OUT / "00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
EVAL35 = ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json"
ACT_A_CHECKPOINT = ROOT / "outputs/paper_core_ab/act_a40/train/checkpoints/100000/pretrained_model"
ACT_B_CHECKPOINT = ROOT / "outputs/paper_core_ab/act_b40/train/checkpoints/020000/pretrained_model"
STALE_GUARD_PROVENANCE = OUT / "provenance/invalidated_stale_dex3_guard_0001"
STALE_GUARD_FREEZE = STALE_GUARD_PROVENANCE / "00_freeze/DIRECT_EVAL35_FREEZE_MANIFEST.json"
STALE_GUARD_COMMANDS = STALE_GUARD_PROVENANCE / "DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.guard_0001.json"
AUTHORITATIVE_DEX3_GUARD_RAD = 5.0e-3
ARM_PROJECTOR_AUDIT = OUT / "00_preparation/COMMON_ARM_HARD_LIMIT_PROJECTOR_AUDIT.json"
ARM_PROJECTOR_REPORT = OUT / "00_preparation/COMMON_ARM_HARD_LIMIT_PROJECTOR_AUDIT.md"
ARM_PROJECTOR_PROVENANCE = OUT / "provenance/invalidated_no_common_arm_hard_limit_projector"
ARM_PROJECTOR_PREDECESSOR_FREEZE = ARM_PROJECTOR_PROVENANCE / "00_freeze/DIRECT_EVAL35_FREEZE_MANIFEST.json"
ARM_PROJECTOR_PREDECESSOR_RUN = ARM_PROJECTOR_PROVENANCE / "01_rollouts_invalid_arm_limit_attempt/act_a40/eval_00_doll_handoff_20260820_ep002/RUN_MANIFEST.json"

FILES = (
    ROOT / "configs/contact_eval_common_task_registration_v1.json",
    ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json",
    ROOT / "isaaclab_doll_handoff_scene/scene_layout.json",
    ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_scene.usda",
    ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda",
    ROOT / "isaaclab_doll_handoff_scene/generated/table_workspace.usda",
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json",
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_COMMON_EXECUTION_CONTROLLER.json",
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/PREDECLARED_TASK_SUCCESS_CRITERIA.md",
    ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json",
    ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/COMMON_EXECUTION_LAYER_FREEZE_MANIFEST.json",
    ROOT / "tools/common_execution_layer.py",
    ROOT / "tools/common_execution_isaac_runtime.py",
    ROOT / "tools/direct_physical_execution_layer.py",
    ROOT / "tools/direct_physical_execution_isaac_runtime.py",
    ROOT / "tools/run_direct_physical_execution_isaac.py",
    ROOT / "tools/run_direct_physical_eval35.py",
    ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py",
    ROOT / "tools/score_contact_constrained_full_task.py",
    ROOT / "tools/score_direct_physical_eval35_run.py",
    ROOT / "tools/prepare_direct_eval35_physical_commands.py",
    ROOT / "tools/audit_common_arm_hard_limit_projector_eval35.py",
    ROOT / "tools/validate_limit_safe_direct_execution_layer.py",
    ROOT / "tools/build_task_relative_scripted_validation.py",
    ROOT / "tools/audit_authoritative_physical_scene.py",
    ROOT / "tools/freeze_authoritative_physical_scene.py",
    COMMANDS,
    EVAL35,
    ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation/act_trajectories/act_a40/BATCH_MANIFEST.json",
    ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation/act_trajectories/act_b40/BATCH_MANIFEST.json",
    OUT / "00_preparation/act_trajectories/act_a40/BATCH_MANIFEST.json",
    OUT / "00_preparation/act_trajectories/act_b40/BATCH_MANIFEST.json",
    *tuple(sorted(path for path in ACT_A_CHECKPOINT.iterdir() if path.is_file())),
    *tuple(sorted(path for path in ACT_B_CHECKPOINT.iterdir() if path.is_file())),
    OUT / "00_preparation/LIMIT_SAFE_DEX3_VALIDATION.json",
    ARM_PROJECTOR_AUDIT,
    ARM_PROJECTOR_REPORT,
    OUT / "PREDECLARED_DIRECT_PHYSICAL_SUCCESS_CRITERIA.md",
    PREFLIGHT,
    Path(read_json_path := str(SCENE_OUT / "task_relative_scripted_validation/corrected_task_relative_scripted_regression.npz")),
    AUDIT,
    RUN / "event_log.npz",
    RUN / "robot_bin_contacts.npz",
    RUN / "trial_result.json",
    RUN / "CONTACT_CONSTRAINED_TASK_RESULT.json",
    RUN / "CONTACT_CONSTRAINED_TASK_RESULT.md",
    RUN / "engine.log",
    RUN / "scorer.log",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    if list((OUT / "01_rollouts").glob("act_*40/eval_*/RUN_MANIFEST.json")):
        raise RuntimeError("cannot freeze after EVAL35 rollout start")
    if (OUT / "ROLLOUTS_STARTED.json").exists():
        raise RuntimeError("EVAL35 rollout-start lock already exists")
    if not STALE_GUARD_FREEZE.is_file() or not STALE_GUARD_COMMANDS.is_file():
        raise RuntimeError("stale-guard predecessor was not preserved")
    stale_guard_commands = read_json(STALE_GUARD_COMMANDS)
    if (
        stale_guard_commands.get("dex3_hard_limit_guard_rad") != 1.0e-4
        or len(stale_guard_commands.get("records", [])) != 70
    ):
        raise RuntimeError("preserved stale-guard command provenance is invalid")
    if not ARM_PROJECTOR_PREDECESSOR_FREEZE.is_file() or not ARM_PROJECTOR_PREDECESSOR_RUN.is_file():
        raise RuntimeError("no-arm-projector predecessor was not preserved")
    if sha256_file(ARM_PROJECTOR_PREDECESSOR_FREEZE) != "3931eb5f1606102e12f4526fec3eca0db7728a8051a2c36da8955832d4e809b5":
        raise RuntimeError("preserved no-arm-projector freeze identity mismatch")
    projector_audit = read_json(ARM_PROJECTOR_AUDIT)
    if (
        projector_audit.get("status") != "PASS"
        or projector_audit.get("remaining_arm_hard_limit_violation_scalar_count") != 0
        or projector_audit.get("methods", {}).get("ACT-A40", {}).get("projected_scalar_count") != 1868
        or projector_audit.get("methods", {}).get("ACT-B40", {}).get("projected_scalar_count") != 0
    ):
        raise RuntimeError("common arm hard-limit projector offline audit failed")
    preflight = read_json(PREFLIGHT)
    audit = read_json(AUDIT)
    task = read_json(RUN / "CONTACT_CONSTRAINED_TASK_RESULT.json")
    trial = read_json(RUN / "trial_result.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("task-frame numerical preflight did not pass")
    if audit.get("status") != "PASS" or audit.get("safe_to_start_final_70_rollouts") is not True:
        raise RuntimeError("complete physical scene audit did not pass")
    if task.get("status") != "PASS" or task["outcomes"].get("FULL_TASK_SUCCESS") is not True:
        raise RuntimeError("corrected scripted physical task did not pass")
    if trial.get("executed_control_frames") != 3309 or trial.get("command_completed") is not True:
        raise RuntimeError("corrected scripted command did not complete 3309 frames")
    regression = audit["regression"]
    if (
        regression["commanded_dex3_hard_limit_violation_scalar_count"] != 0
        or regression["measured_dex3_hard_limit_violation_scalar_count"] != 0
        or regression["wrist_rescue_scalar_count"] != 0
        or regression["command_replay_difference_scalar_count"] != 0
        or not regression["hard_execution_valid"]
    ):
        raise RuntimeError("corrected scripted execution-integrity gate failed")

    # Preserve the invalid predecessor byte-for-byte before replacing the live
    # freeze path. Repeated invocation accepts only the already archived copy.
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    archived_manifest = PROVENANCE / "DIRECT_EVAL35_FREEZE_MANIFEST.WRONG_REGISTRATION.json"
    archived_report = PROVENANCE / "DIRECT_EVAL35_FREEZE_MANIFEST.WRONG_REGISTRATION.md"
    if not archived_manifest.exists():
        shutil.copy2(MANIFEST, archived_manifest)
    if REPORT.exists() and not archived_report.exists():
        shutil.copy2(REPORT, archived_report)
    invalidation = read_json(SCENE_OUT / "INVALIDATED_PRIOR_FREEZE_PROVENANCE.json")
    if sha256_file(archived_manifest) != invalidation["preserved_manifest_sha256"]:
        raise RuntimeError("archived predecessor freeze identity mismatch")

    rows: list[dict[str, Any]] = []
    for path in FILES:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    commands = read_json(COMMANDS)
    if len(commands.get("records", [])) != 70:
        raise RuntimeError("ACT-A/B physical command set is not 70")
    if commands.get("dex3_hard_limit_guard_rad") != AUTHORITATIVE_DEX3_GUARD_RAD:
        raise RuntimeError("ACT-A/B physical command guard is not authoritative")
    if sha256_file(COMMANDS) != "bba69a7c49def787aa3cb4ead57946736668c2551cd1fc4f5b688afc883b5468":
        raise RuntimeError("physical commands were regenerated or modified")
    method_counts = {
        method: sum(record.get("method") == method for record in commands["records"])
        for method in ("ACT-A40", "ACT-B40")
    }
    if method_counts != {"ACT-A40": 35, "ACT-B40": 35}:
        raise RuntimeError(f"ACT-A/B command counts are invalid: {method_counts}")
    for record in commands["records"]:
        path = Path(record["physical_command"]).resolve()
        actual = sha256_file(path)
        if actual != record["physical_command_sha256"]:
            raise RuntimeError(f"ACT-A/B command drift: {path}")
        rows.append({
            "path": str(path), "bytes": path.stat().st_size, "sha256": actual,
            "role": "frozen_act_ab_physical_command",
        })
    rows.sort(key=lambda row: row["path"])
    bundle = hashlib.sha256(
        ("\n".join(f"{row['path']}:{row['sha256']}" for row in rows) + "\n").encode("utf-8")
    ).hexdigest()
    registration = audit["registration"]
    environment = read_json(
        ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
    )
    eval35 = read_json(EVAL35)
    eval_entries = eval35.get("eval_entries", [])
    eval_ids = [entry.get("stable_episode_id") for entry in eval_entries]
    if len(eval_entries) != 35 or len(set(eval_ids)) != 35:
        raise RuntimeError("EVAL35 is not 35 unique episodes")
    checkpoint_rows = {}
    for method, checkpoint, expected_sha in (
        ("ACT-A40", ACT_A_CHECKPOINT, "7e9fe737c3fd8ad3919cf3887dad58a732f6651e84e1eab7c1af8267ee16912c"),
        ("ACT-B40", ACT_B_CHECKPOINT, "4c3c52a853cc242c6ba97fa6fa8d2dde96f6d65e737e291cfb99265f3c1b5198"),
    ):
        model = checkpoint / "model.safetensors"
        actual_sha = sha256_file(model)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{method} checkpoint model hash mismatch")
        checkpoint_rows[method] = {
            "path": str(checkpoint.resolve()),
            "model": str(model.resolve()),
            "model_sha256": actual_sha,
            "files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in sorted(checkpoint.iterdir())
                if path.is_file()
            ],
        }
    value = {
        "schema_version": "direct_physical_eval35_complete_authoritative_scene_freeze_v5",
        "status": "FROZEN_BEFORE_EVAL35",
        "evaluation_set": "EVAL35",
        "required_rollouts": 70,
        "eval35_rollouts_started_at_freeze": 0,
        "direct_execution_bundle_sha256": bundle,
        "complete_physical_scene_sha256": bundle,
        "files": rows,
        "eval35": {
            "manifest": str(EVAL35.resolve()),
            "manifest_sha256": sha256_file(EVAL35),
            "episode_count": len(eval_entries),
            "all_unique": len(set(eval_ids)) == len(eval_ids),
            "ordered_stable_episode_ids": eval_ids,
        },
        "act_checkpoints": checkpoint_rows,
        "common_arm_hard_limit_projector": {
            "status": "PASS",
            "algorithm": "nearest_valid_value_componentwise_np_clip",
            "scope": "all_14_g1_arm_and_wrist_command_channels",
            "same_for_act_a_and_act_b": True,
            "authoritative_limits": str((ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json").resolve()),
            "offline_audit": str(ARM_PROJECTOR_AUDIT.resolve()),
            "offline_audit_sha256": sha256_file(ARM_PROJECTOR_AUDIT),
            "act_a_projected_scalar_count": 1868,
            "act_a_maximum_correction_rad": projector_audit["methods"]["ACT-A40"]["maximum_correction_rad"],
            "act_b_projected_scalar_count": 0,
            "act_b_maximum_correction_rad": 0.0,
            "remaining_arm_hard_limit_violation_scalar_count": 0,
            "ik_used": False,
            "smoothing_used": False,
            "trajectory_regeneration_used": False,
            "arm_rescue_used": False,
            "wrist_rescue_used": False,
        },
        "authoritative_object_registration": {
            "config": str((ROOT / "configs/contact_eval_common_task_registration_v1.json").resolve()),
            "config_sha256": sha256_file(ROOT / "configs/contact_eval_common_task_registration_v1.json"),
            "position_xyz_m": registration["manifest_position_xyz_m"],
            "quaternion_xyzw": registration["manifest_quaternion_xyzw"],
            "runtime_position_before_frame_0_xyz_m": registration["runtime_position_before_frame_0_xyz_m"],
            "runtime_quaternion_before_frame_0_xyzw": registration["runtime_quaternion_before_frame_0_xyzw"],
            "translation_error_mm": registration["translation_error_mm"],
            "rotation_error_deg": registration["rotation_error_deg"],
            "identical_for_act_a_and_act_b": registration["act_a_and_act_b_exactly_same_initial_pose"],
        },
        "corrected_task_relative_scripted_validation": {
            "command": preflight["output_command"],
            "command_sha256": preflight["output_command_sha256"],
            "numerical_preflight": str(PREFLIGHT.resolve()),
            "numerical_preflight_sha256": sha256_file(PREFLIGHT),
            "phase_contract": preflight["phase_contract"],
            "dex3_commands_exactly_preserved": True,
            "frozen_handoff_and_transport_tail_exactly_preserved": True,
            "bin_relative_tail_exactly_preserved": True,
            "act_ab_trajectories_modified": False,
        },
        "doll_physics": environment["doll"],
        "table": audit["table"],
        "bin_150mm": audit["bin"],
        "contact_constrained_physics": audit["physics"],
        "dex3_limits": {
            "contract": str((ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json").resolve()),
            "contract_sha256": sha256_file(ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"),
            "common_safety_inset_rad": 0.005,
            "commanded_violations_regression": 0,
            "measured_violations_regression": 0,
        },
        "task_success_criteria": audit["success_semantics"],
        "correct_scene_scripted_regression": {
            "status": "PASS",
            "executed_control_frames": 3309,
            "outcomes": task["outcomes"],
            "release_classification": task["release_classification"],
            "object_pose_writes_after_initialization": task["provenance"]["object_pose_writes_during_timed_loop"],
            "doll_bin_tunneling": task["diagnostics"]["invalid_doll_wall_or_bottom_tunneling"],
            "robot_bin_tunneling": task["diagnostics"]["invalid_robot_wall_crossing"],
            "finite_states": not any(regression["nonfinite_scalar_count_by_state"].values()),
        },
        "preserved_invalid_predecessor": {
            "manifest": str(archived_manifest.resolve()),
            "manifest_sha256": sha256_file(archived_manifest),
            "reason": "wrong object registration",
        },
        "preserved_invalid_stale_guard_predecessor": {
            "manifest": str(STALE_GUARD_FREEZE.resolve()),
            "manifest_sha256": sha256_file(STALE_GUARD_FREEZE),
            "command_manifest": str(STALE_GUARD_COMMANDS.resolve()),
            "command_manifest_sha256": sha256_file(STALE_GUARD_COMMANDS),
            "stale_guard_rad": 1.0e-4,
            "reason": "prepared common Dex3 initial state used obsolete 0.0001 rad guard",
        },
        "preserved_invalid_no_arm_projector_predecessor": {
            "manifest": str(ARM_PROJECTOR_PREDECESSOR_FREEZE.resolve()),
            "manifest_sha256": sha256_file(ARM_PROJECTOR_PREDECESSOR_FREEZE),
            "invalid_run_manifest": str(ARM_PROJECTOR_PREDECESSOR_RUN.resolve()),
            "invalid_run_manifest_sha256": sha256_file(ARM_PROJECTOR_PREDECESSOR_RUN),
            "reason": "frozen ACT-A arm commands exceeded authoritative G1 hard limits before the common projector was authorized",
        },
        "graspability_classifier_used": False,
        "graspability_atlas_used": False,
        "wrist_distance_gate_used": False,
        "arm_rescue_allowed": False,
        "wrist_rescue_allowed": False,
        "dex3_common_primitive_allowed": True,
        "common_dex3_hard_limits_enforced": True,
        "common_execution_layer_modified": True,
        "common_execution_layer_modification": "authorized common nearest-bound arm hard-limit safety projector only",
        "post_rollout_start_tuning_allowed": False,
    }
    atomic_json(MANIFEST, value)
    REPORT.write_text(
        "# Complete authoritative physical EVAL35 freeze\n\n"
        "Status: **FROZEN BEFORE ANY OF 70 EVAL35 ROLLOUTS**\n\n"
        f"Complete physical scene SHA256: `{bundle}`\n\n"
        "- Authoritative object registration: PASS\n"
        "- Corrected task-relative scripted validation: PASS\n"
        "- Correct-scene 3309-frame physical regression: PASS\n"
        "- Doll physics / 150 mm bin / contact-constrained PhysX: FROZEN\n"
        "- Commanded / measured Dex3 violations: 0 / 0\n"
        "- ACT-A/B trajectories modified: NO\n"
        "- Dex3 preparation/runtime safety inset: 0.005 rad\n"
        "- Stale 0.0001-rad freeze preserved as invalidated provenance\n"
        "- Common A/B arm hard-limit projector: nearest valid componentwise value\n"
        "- Offline arm projector audit: 0 remaining violations across 70 archives\n"
        "- EVAL35 rollouts started: 0 / 70\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": value["status"],
        "complete_physical_scene_sha256": bundle,
        "manifest": str(MANIFEST),
        "archived_predecessor": str(archived_manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
