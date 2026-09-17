#!/usr/bin/env python3
"""Create the hard scientific freeze for episode-registered physical EVAL35."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
OUT = ROOT / "outputs/final_episode_registered_eval35"
FREEZE_DIR = OUT / "01_freeze"
REGISTRATION = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
REGISTRATION_CSV = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.csv"
REGISTRATION_MD = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.md"
QUALIFICATION = OUT / "00_qualification/FINAL_COMMON_DEX3_GRASP_QUALIFICATION.json"
QUALIFICATION_MD = OUT / "00_qualification/FINAL_COMMON_DEX3_GRASP_QUALIFICATION.md"
COMMANDS = ROOT / "outputs/final_direct_physical_eval35/00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
ARM_AUDIT = ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_ARM_HARD_LIMIT_PROJECTOR_AUDIT.json"
INTENT_JSON = ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_SOURCE_TASK_INTENT_EVAL35.json"
INTENT_NPZ = ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_SOURCE_TASK_INTENT_EVAL35.npz"
EVAL35 = ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
OLD_ENV = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
OLD_CONTROLLER = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_COMMON_EXECUTION_CONTROLLER.json"
MANIFEST = FREEZE_DIR / "FINAL_EVAL35_FREEZE_MANIFEST.json"
ARTICULATION_AUDIT = OUT / "00_forensic_audit/DEX3_ARTICULATION_FORENSIC_AUDIT.json"
ZERO_CONTACT_AUDIT = OUT / "00_forensic_audit/dex3_zero_contact_solver80/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_row(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved), "role": role}


def checkpoint(path: Path) -> dict[str, Any]:
    files = [file_row(value, "checkpoint_file") for value in sorted(path.iterdir()) if value.is_file()]
    model = path / "model.safetensors"
    return {"path": str(path.resolve()), "model": str(model.resolve()), "model_sha256": sha256_file(model), "files": files}


def main() -> int:
    existing_runs = list((OUT / "02_act_a_results/rollouts").glob("eval_*/RUN_MANIFEST.json")) + list((OUT / "03_act_b_results/rollouts").glob("eval_*/RUN_MANIFEST.json"))
    if existing_runs or (OUT / "FINAL_ROLLOUTS_STARTED.json").exists():
        raise RuntimeError("cannot create or revise hard freeze after final rollout start")
    qualification = read_json(QUALIFICATION)
    aggregate = qualification.get("aggregate", {})
    if not (
        qualification.get("status") == "READY_TO_FREEZE"
        and qualification.get("eval35_rollouts_started") == 0
        and aggregate.get("right_standalone_passes") == 3
        and aggregate.get("left_standalone_passes") == 3
        and aggregate.get("natural_release_passes") == 3
        and aggregate.get("scripted_full_task_passes") == 3
        and aggregate.get("commanded_hard_limit_violations") == 0
        and aggregate.get("measured_hard_limit_violations") == 0
        and aggregate.get("object_pose_writes_after_initialization") == 0
        and aggregate.get("arm_rescue") is False
        and aggregate.get("wrist_rescue") is False
    ):
        raise RuntimeError("persisted contact qualification is not complete")
    registration = read_json(REGISTRATION)
    if not (
        registration.get("status") == "PASS"
        and registration.get("EVAL35_count") == 35
        and registration.get("source_derived_count") == 35
        and registration.get("A_B_identical_object_pose_count") == 35
        and registration.get("one_global_canonical_object_pose") is False
        and registration.get("manual_episode_nudges") is False
        and registration.get("policy_output_derived_object_placement") is False
        and registration.get("physical_eval35_outcomes_read") is False
    ):
        raise RuntimeError("episode-conditioned registration is not source-only 35/35")
    commands = read_json(COMMANDS)
    records = list(commands.get("records", []))
    if len(records) != 70 or commands.get("dex3_hard_limit_guard_rad") != 0.005:
        raise RuntimeError("prepared commands are not exact guard-safe A/B EVAL35")
    methods = {method: sum(row["method"] == method for row in records) for method in ("ACT-A40", "ACT-B40")}
    if methods != {"ACT-A40": 35, "ACT-B40": 35}:
        raise RuntimeError(f"prepared method counts differ: {methods}")
    arm_audit = read_json(ARM_AUDIT)
    if arm_audit.get("status") != "PASS" or arm_audit.get("remaining_arm_hard_limit_violation_scalar_count") != 0:
        raise RuntimeError("common hard-limit projector audit did not pass")
    if arm_audit["projector"].get("same_instance_and_limits_for_a_b") is not True:
        raise RuntimeError("arm hard-limit projector is not common A/B")

    tests = subprocess.run(
        [str(ISAAC), "-m", "pytest", "-q", "tests/test_common_execution_layer.py", "tests/test_direct_physical_execution_layer.py"],
        cwd=ROOT, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        raise RuntimeError(f"execution tests failed:\n{tests.stdout}{tests.stderr}")
    patch = subprocess.run(
        [str(ISAAC), "tools/run_direct_physical_execution_isaac.py", "--validate-patch-only"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if patch.returncode != 0:
        raise RuntimeError(f"runtime instrumentation failed:\n{patch.stdout}{patch.stderr}")

    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    selected = qualification["selected_contact_model"]
    base_env = read_json(OLD_ENV)
    config = read_json(CONFIG)
    articulation_solver = config["articulation_solver"]
    zero_contact_audit = read_json(ZERO_CONTACT_AUDIT)
    if not (
        zero_contact_audit.get("status") == "PASS"
        and zero_contact_audit.get("summary", {}).get("mapping_pass_count") == 14
        and zero_contact_audit.get("summary", {}).get("sign_pass_count") == 14
        and zero_contact_audit.get("summary", {}).get("readback_pass_count") == 14
        and zero_contact_audit.get("summary", {}).get("runtime_hard_limit_pass_count") == 14
    ):
        raise RuntimeError("final 14-joint zero-contact articulation audit did not pass")
    environment = {
        **base_env,
        "schema_version": "episode_registered_final_physical_environment_v1",
        "status": "FROZEN",
        "registration": {
            "semantics": "EPISODE_CONDITIONED_SOURCE_DERIVED_TASK_REGISTRATION",
            "entries": 35, "one_global_canonical_pose": False,
            "matched_A_B_pose_equality": True,
            "manifest": str(REGISTRATION.resolve()), "manifest_sha256": sha256_file(REGISTRATION),
        },
        "doll": {
            **base_env["doll"],
            "collision_geometry": {
                "name": selected["name"],
                "dimensions_m": selected["collision_dimensions_m"],
                "collision_scale_percent": 81,
                "semantics": "qualified common midpoint rigid plush proxy within visual envelope",
            },
            "visual_dimensions_m": selected["visual_dimensions_m"],
            "additional_contact_tolerance_mm": selected["additional_contact_tolerance_mm"],
            "dynamic": True, "kinematic": False, "gravity_enabled": True,
            "attachment": False, "parenting": False, "object_following": False,
            "root_pose_writes_after_initialization": 0,
            "visual_minus_collision_half_extent_m": (
                (0.5 * (__import__("numpy").asarray(selected["visual_dimensions_m"]) - __import__("numpy").asarray(selected["collision_dimensions_m"]))).tolist()
            ),
        },
        "contact_constrained_physics": True,
        "physics": {
            **base_env["physics"],
            "solver": "TGS",
            "articulation_solver_position_iterations": int(articulation_solver["position_iterations"]),
            "articulation_solver_velocity_iterations": int(articulation_solver["velocity_iterations"]),
            "articulation_solver_scope": articulation_solver["scope"],
            "articulation_solver_selection_basis": articulation_solver["selection_basis"],
        },
    }
    atomic_json(FREEZE_DIR / "FINAL_PHYSICAL_ENVIRONMENT.json", environment)
    atomic_json(FREEZE_DIR / "FINAL_DOLL_CONTACT_MODEL.json", {
        "schema_version": "final_qualified_plush_contact_model_v1", "status": "FROZEN",
        "selection_source": str(QUALIFICATION.resolve()), "selection_source_sha256": sha256_file(QUALIFICATION),
        "name": selected["name"], "visual_dimensions_m": selected["visual_dimensions_m"],
        "collision_dimensions_m": selected["collision_dimensions_m"],
        "additional_contact_tolerance_mm": selected["additional_contact_tolerance_mm"],
        "mass_kg": environment["doll"]["mass_kg"], "material": environment["doll"]["material"],
        "linear_damping": environment["doll"]["linear_damping"], "angular_damping": environment["doll"]["angular_damping"],
        "restitution": environment["doll"]["restitution"], "no_adhesion_or_attachment": True,
        "supports_physical_grasp_and_natural_release": True,
    })
    atomic_json(FREEZE_DIR / "FINAL_EPISODE_REGISTRATION_MANIFEST.json", {
        "schema_version": "final_episode_registration_binding_v1", "status": "FROZEN",
        "registration_manifest": str(REGISTRATION.resolve()), "registration_manifest_sha256": sha256_file(REGISTRATION),
        "entries": 35, "source_derived": 35, "matched_A_B_identical": 35,
        "maximum_A_B_translation_difference_mm": 0.0, "maximum_A_B_rotation_difference_deg": 0.0,
        "one_global_canonical_pose": False, "manual_episode_nudges": False,
        "policy_output_derived_object_placement": False,
        "entry_sha256": [row["entry_sha256"] for row in registration["entries"]],
    })
    controller = read_json(OLD_CONTROLLER)
    atomic_json(FREEZE_DIR / "FINAL_COMMON_DEX3_CONTROLLER.json", {
        "schema_version": "final_common_dex3_mechanical_controller_v1", "status": "FROZEN",
        "common_for_A_B": True,
        "state_machine": qualification["state_machine"],
        "preshape_permanent_latch": False, "transient_contact_may_unlatch": True,
        "mechanical_grasp_confirmation": True, "lift_before_confirmed_grasp": False,
        "bounded_preload_max_rad": qualification["preload_max_rad"],
        "dex3_safety_inset_rad": 0.005,
        "source_intent": str(INTENT_JSON.resolve()), "source_intent_sha256": sha256_file(INTENT_JSON),
        "finger_drive": controller["finger_drive"],
        "arm_rescue": False, "wrist_rescue": False,
        "implementation": str((ROOT / "tools/direct_physical_execution_layer.py").resolve()),
        "implementation_sha256": sha256_file(ROOT / "tools/direct_physical_execution_layer.py"),
    })
    atomic_json(FREEZE_DIR / "FINAL_MECHANICAL_GRASP_SCORER.json", {
        "schema_version": "final_topology_neutral_mechanical_grasp_scorer_v1", "status": "FROZEN",
        "definition": [
            "actual PhysX contact", "meaningful opposing or enclosing support",
            "table support loss", "bounded hand-object relative motion", "physical retention",
            "no attachment or object following",
        ],
        "all_three_digits_mandatory": False,
        "accepted_topologies": ["thumb+index+middle", "thumb+index+palm", "thumb+middle+palm", "other stable opposing enclosure"],
        "contact_topology_recorded": True,
        "implementation": str((ROOT / "tools/score_episode_registered_physical_eval35_run.py").resolve()),
        "implementation_sha256": sha256_file(ROOT / "tools/score_episode_registered_physical_eval35_run.py"),
    })
    criteria = """# Final episode-registered EVAL35 task-success criteria

`FULL_TASK_SUCCESS` requires LEFT physical acquisition, no unintended loss,
physical LEFT-to-RIGHT handoff, RIGHT physical ownership, supported transport,
physical entry into the fixed 150 mm bin, and at least 1.0 s settled inside.

Robot/bin contact and moderate post-release motion are diagnostics unless they
invalidate physics. Release is classified separately as clean commanded release,
premature drop into bin, or premature drop outside bin. No definition may be
changed after this freeze.
"""
    atomic_text(FREEZE_DIR / "FINAL_TASK_SUCCESS_CRITERIA.md", criteria)

    checkpoint_paths = {
        "ACT-A40": ROOT / "outputs/paper_core_ab/act_a40/train/checkpoints/100000/pretrained_model",
        "ACT-B40": ROOT / "outputs/paper_core_ab/act_b40/train/checkpoints/020000/pretrained_model",
    }
    checkpoints = {method: checkpoint(path) for method, path in checkpoint_paths.items()}
    expected_models = {
        "ACT-A40": "7e9fe737c3fd8ad3919cf3887dad58a732f6651e84e1eab7c1af8267ee16912c",
        "ACT-B40": "4c3c52a853cc242c6ba97fa6fa8d2dde96f6d65e737e291cfb99265f3c1b5198",
    }
    if {method: row["model_sha256"] for method, row in checkpoints.items()} != expected_models:
        raise RuntimeError("selected checkpoint identities differ")
    if {method: {row["checkpoint_model_sha256"] for row in records if row["method"] == method} for method in checkpoints} != {method: {value} for method, value in expected_models.items()}:
        raise RuntimeError("prepared commands do not bind selected checkpoints")

    scientific_paths: list[tuple[Path, str]] = [
        (REGISTRATION, "episode_registration"), (REGISTRATION_CSV, "episode_registration"),
        (REGISTRATION_MD, "episode_registration"), (QUALIFICATION, "qualification"),
        (QUALIFICATION_MD, "qualification"), (COMMANDS, "command_manifest"),
        (ARM_AUDIT, "hard_limit_audit"), (INTENT_JSON, "common_task_intent"),
        (INTENT_NPZ, "common_task_intent"), (EVAL35, "eval35_membership"),
        (CONFIG, "physics_config"), (OLD_ENV, "physical_environment_provenance"),
        (OLD_CONTROLLER, "controller_provenance"),
        (ROOT / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json", "dex3_geometry"),
        (ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda", "scene"),
        (ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_scene.usda", "scene"),
        (ROOT / "isaaclab_doll_handoff_scene/generated/table_workspace.usda", "scene"),
        (ROOT / "isaaclab_doll_handoff_scene/scene_layout.json", "scene"),
        (ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json", "joint_limits"),
        (ROOT / "tools/policy_b_isaac_control_contract.py", "actuator_contract"),
        (ROOT / "tools/direct_physical_execution_layer.py", "execution_implementation"),
        (ROOT / "tools/direct_physical_execution_isaac_runtime.py", "execution_implementation"),
        (ROOT / "tools/run_direct_physical_execution_isaac.py", "execution_implementation"),
        (ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py", "physics_engine"),
        (ROOT / "tools/run_episode_registered_physical_eval35.py", "evaluation_runner"),
        (ROOT / "tools/score_episode_registered_physical_eval35_run.py", "task_scorer"),
        (ROOT / "tools/build_eval35_episode_object_registration.py", "registration_derivation"),
        (ROOT / "tools/audit_final_dex3_articulation_isaac.py", "articulation_audit_implementation"),
        (ARTICULATION_AUDIT, "articulation_root_cause_audit"),
        (ZERO_CONTACT_AUDIT, "articulation_zero_contact_qualification"),
        (ROOT / "tools/freeze_episode_registered_eval35.py", "freeze_implementation"),
        (FREEZE_DIR / "FINAL_EPISODE_REGISTRATION_MANIFEST.json", "freeze_component"),
        (FREEZE_DIR / "FINAL_DOLL_CONTACT_MODEL.json", "freeze_component"),
        (FREEZE_DIR / "FINAL_COMMON_DEX3_CONTROLLER.json", "freeze_component"),
        (FREEZE_DIR / "FINAL_MECHANICAL_GRASP_SCORER.json", "freeze_component"),
        (FREEZE_DIR / "FINAL_TASK_SUCCESS_CRITERIA.md", "freeze_component"),
        (FREEZE_DIR / "FINAL_PHYSICAL_ENVIRONMENT.json", "freeze_component"),
    ]
    for value in registration.get("authoritative_inputs", {}):
        path = Path(value)
        if path.is_file():
            scientific_paths.append((path, "registration_source"))
    rows_by_path: dict[str, dict[str, Any]] = {}
    for path, role in scientific_paths:
        row = file_row(path, role)
        rows_by_path[row["path"]] = row
    for record in records:
        path = Path(record["physical_command"])
        row = file_row(path, "frozen_physical_command")
        if row["sha256"] != record["physical_command_sha256"]:
            raise RuntimeError(f"command archive drift: {path}")
        rows_by_path[row["path"]] = row
    for checkpoint_value in checkpoints.values():
        for row in checkpoint_value["files"]:
            rows_by_path[row["path"]] = row
    rows = [rows_by_path[key] for key in sorted(rows_by_path)]
    bundle_sha = hashlib.sha256(("\n".join(f"{row['path']}:{row['sha256']}" for row in rows) + "\n").encode()).hexdigest()
    value = {
        "schema_version": "final_episode_registered_physical_eval35_freeze_v1",
        "status": "FROZEN_BEFORE_EVAL35", "evaluation_set": "EVAL35", "required_rollouts": 70,
        "scientific_bundle_sha256": bundle_sha, "files": rows, "frozen_dependency_count": len(rows),
        "checkpoints": checkpoints,
        "registration": {"entries": 35, "source_derived": 35, "matched_A_B_identical": 35, "one_global_canonical_pose": False},
        "qualification": {"status": "PASS", "right_standalone": "3/3", "left_standalone": "3/3", "natural_release": "3/3", "scripted_full_task": "3/3"},
        "contact_model": {"name": selected["name"], "collision_dimensions_m": selected["collision_dimensions_m"], "contact_tolerance_mm": selected["additional_contact_tolerance_mm"]},
        "common_articulation_solver": {
            "solver": "TGS",
            "position_iterations": int(articulation_solver["position_iterations"]),
            "velocity_iterations": int(articulation_solver["velocity_iterations"]),
            "scope": articulation_solver["scope"],
            "zero_contact_mapping_sign_readback_limits": "14/14 PASS",
        },
        "common_arm_hard_limit_projector": arm_audit["projector"],
        "common_arm_hard_limit_projector_results": arm_audit["methods"],
        "remaining_commanded_arm_hard_limit_violations": 0,
        "dex3_safety_inset_rad": 0.005,
        "graspability_classifier_used": False, "graspability_atlas_used": False,
        "arm_rescue_allowed": False, "wrist_rescue_allowed": False,
        "object_attachment": False, "object_following": False, "object_pose_writes_after_initialization": 0,
        "contact_constrained_physics": True, "bin_height_m": 0.150,
        "tests": tests.stdout.strip().splitlines()[-1], "runtime_instrumentation": json.loads(patch.stdout),
        "rollouts_started_at_freeze": 0, "post_freeze_scientific_tuning_allowed": False,
    }
    atomic_json(MANIFEST, value)
    manifest_sha = sha256_file(MANIFEST)
    atomic_text(FREEZE_DIR / "FINAL_EVAL35_FREEZE_MANIFEST.md", f"""# Final episode-registered physical EVAL35 freeze

Status: **FROZEN BEFORE 0/70 FINAL ROLLOUTS**

- Freeze manifest SHA256: `{manifest_sha}`
- Scientific bundle SHA256: `{bundle_sha}`
- Frozen dependencies: {len(rows)}
- Episode registrations: 35/35 source-derived; matched A/B equality 35/35
- Qualified contact model: `{selected['name']}`; {selected['collision_dimensions_m']} m; 0 mm added tolerance
- Qualification: RIGHT 3/3, LEFT 3/3, natural release 3/3, scripted full task 3/3
- Common arm projector: nearest authoritative bound only; remaining violations 0
- Common TGS articulation solver: {articulation_solver['position_iterations']} position / {articulation_solver['velocity_iterations']} velocity iterations; 14/14 zero-contact mapping/sign/readback/hard-limit audit PASS
- ARM rescue / WRIST rescue: NO / NO
- Contact-constrained PhysX; dynamic gravity-driven doll; no attachment/following/pose writes after initialization
""")
    print(json.dumps({"status": value["status"], "freeze_manifest_sha256": manifest_sha, "scientific_bundle_sha256": bundle_sha, "frozen_dependencies": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
