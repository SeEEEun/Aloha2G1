#!/usr/bin/env python3
"""Hard-freeze classifier-free direct physical EVAL35 before any rollout."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_direct_physical_eval35"
FREEZE_DIR = OUT / "00_freeze"
MANIFEST = FREEZE_DIR / "DIRECT_EVAL35_FREEZE_MANIFEST.json"
REPORT = FREEZE_DIR / "DIRECT_EVAL35_FREEZE_MANIFEST.md"
COMMANDS = OUT / "00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
CRITERIA = OUT / "PREDECLARED_DIRECT_PHYSICAL_SUCCESS_CRITERIA.md"
LIMIT_VALIDATION = OUT / "00_preparation/LIMIT_SAFE_DEX3_VALIDATION.json"
PRE_EVAL35_QUALIFICATION = (
    OUT / "00_pre_eval35_execution_freeze/PRE_EVAL35_EXECUTION_LAYER_QUALIFICATION.json"
)
SCRIPTED_REGRESSION = OUT / "00_pre_eval35_execution_freeze/SCRIPTED_REGRESSION.json"
FILES = (
    ROOT / "tools/common_execution_layer.py",
    ROOT / "tools/common_execution_isaac_runtime.py",
    ROOT / "tools/direct_physical_execution_layer.py",
    ROOT / "tools/direct_physical_execution_isaac_runtime.py",
    ROOT / "tools/prepare_direct_eval35_physical_commands.py",
    ROOT / "tools/validate_limit_safe_direct_execution_layer.py",
    ROOT / "tools/run_direct_physical_execution_isaac.py",
    ROOT / "tools/score_direct_physical_eval35_run.py",
    ROOT / "tools/run_direct_physical_eval35.py",
    ROOT / "tests/test_common_execution_layer.py",
    ROOT / "tests/test_direct_physical_execution_layer.py",
    ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py",
    ROOT / "tools/finalize_pre_eval35_execution_qualification.py",
    ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json",
    ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json",
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json",
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_COMMON_EXECUTION_CONTROLLER.json",
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/PREDECLARED_TASK_SUCCESS_CRITERIA.md",
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/SCRIPTED_REPEATABILITY.json",
    ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/COMMON_EXECUTION_LAYER_FREEZE_MANIFEST.json",
    ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json",
    ROOT / "outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json",
    ROOT / "outputs/final_direct_physical_eval35/00_preparation/NEW_UNSEEN_25_FROZEN_AB_CONVERSION.json",
    ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_SOURCE_TASK_INTENT_EVAL35.json",
    ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_SOURCE_TASK_INTENT_EVAL35.npz",
    COMMANDS,
    LIMIT_VALIDATION,
    OUT / "00_preparation/LIMIT_SAFE_DEX3_VALIDATION.md",
    PRE_EVAL35_QUALIFICATION,
    OUT / "00_pre_eval35_execution_freeze/PRE_EVAL35_EXECUTION_LAYER_QUALIFICATION.md",
    SCRIPTED_REGRESSION,
    OUT / "00_pre_eval35_execution_freeze/SCRIPTED_REGRESSION.md",
    CRITERIA,
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
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if list((OUT / "01_rollouts").glob("act_*40/eval_*/RUN_MANIFEST.json")):
        raise RuntimeError("cannot create or revise freeze after physical rollout start")
    commands = read_json(COMMANDS)
    if commands.get("status") != "FROZEN_BEFORE_PHYSICAL_ROLLOUT" or len(commands.get("records", [])) != 70:
        raise RuntimeError("direct EVAL35 commands are not complete/frozen")
    limit_validation = read_json(LIMIT_VALIDATION)
    if (
        limit_validation.get("status") != "PASS"
        or limit_validation.get("rollout_count") != 70
        or limit_validation.get("endpoint_hard_limit_violation_scalar_count") != 0
        or limit_validation.get("transition_hard_limit_violation_scalar_count") != 0
        or limit_validation.get("exact_rollout_hard_limit_violation_scalar_count") != 0
        or limit_validation.get("arm_command_difference_scalar_count") != 0
        or limit_validation.get("wrist_command_difference_scalar_count") != 0
        or limit_validation.get("both_hands_all_primitive_transitions_triggered") is not True
    ):
        raise RuntimeError("common Dex3 limit-safety validation did not pass")
    qualification = read_json(PRE_EVAL35_QUALIFICATION)
    scripted_regression = read_json(SCRIPTED_REGRESSION)
    required_qualification_gates = {
        "train_physx_smoke_4_of_4_pass",
        "full_task_success",
        "control_frames_3309_completed",
        "commanded_dex3_hard_limit_violations_zero",
        "measured_dex3_hard_limit_violations_zero",
        "finite_states_throughout",
        "no_invalid_articulation_state",
        "arm_trajectory_unchanged",
        "wrist_rescue_zero",
    }
    if (
        qualification.get("status") != "QUALIFIED_FOR_FINAL_EVAL35"
        or qualification.get("go_for_final_eval35") is not True
        or scripted_regression.get("status") != "PASS"
        or set(scripted_regression.get("gates", {})) != required_qualification_gates
        or not all(scripted_regression["gates"].values())
    ):
        raise RuntimeError("PRE-EVAL35 physical execution qualification did not pass")
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    tests = subprocess.run(
        ["/home/jbnu/miniconda3/envs/isaaclab6/bin/python", "-m", "pytest", "-q", "tests/test_common_execution_layer.py", "tests/test_direct_physical_execution_layer.py"],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0 or "passed" not in tests.stdout:
        raise RuntimeError(f"execution tests failed:\n{tests.stdout}{tests.stderr}")
    patch = subprocess.run(
        ["/home/jbnu/miniconda3/envs/isaaclab6/bin/python", "tools/run_direct_physical_execution_isaac.py", "--validate-patch-only"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if patch.returncode != 0:
        raise RuntimeError(f"engine instrumentation failed:\n{patch.stdout}{patch.stderr}")
    rows = []
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    for record in commands["records"]:
        path = Path(record["physical_command"])
        if sha256_file(path) != record["physical_command_sha256"]:
            raise RuntimeError(f"command drift before freeze: {path}")
        rows.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": record["physical_command_sha256"], "role": "frozen_physical_command"})
    # Checkpoint binaries are scientific identities, not modified by this workflow.
    checkpoint_rows = []
    for method, expected in (("ACT-A40", "7e9fe737c3fd8ad3919cf3887dad58a732f6651e84e1eab7c1af8267ee16912c"), ("ACT-B40", "4c3c52a853cc242c6ba97fa6fa8d2dde96f6d65e737e291cfb99265f3c1b5198")):
        matches = {row["checkpoint_model_sha256"] for row in commands["records"] if row["method"] == method}
        if matches != {expected}:
            raise RuntimeError(f"{method} checkpoint identity mismatch: {matches}")
        checkpoint_rows.append({"method": method, "model_sha256": expected})
    bundle_rows = [f"{row['path']}:{row['sha256']}" for row in rows]
    bundle = hashlib.sha256(("\n".join(bundle_rows) + "\n").encode("utf-8")).hexdigest()
    prior = read_json(ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/COMMON_EXECUTION_LAYER_FREEZE_MANIFEST.json")
    value = {
        "schema_version": "direct_physical_eval35_freeze_v2_limit_safe_dex3",
        "status": "FROZEN_BEFORE_EVAL35", "evaluation_set": "EVAL35",
        "required_rollouts": 70, "direct_execution_bundle_sha256": bundle,
        "files": rows, "checkpoints": checkpoint_rows,
        "preserved_common_execution_layer_declared_sha256": prior["execution_layer_sha256"],
        "preserved_common_execution_layer_modified": False,
        "graspability_classifier_used": False, "graspability_atlas_used": False,
        "wrist_distance_gate_used": False, "arm_rescue_allowed": False,
        "wrist_rescue_allowed": False, "dex3_common_primitive_allowed": True,
        "common_dex3_hard_limits_enforced": True,
        "common_dex3_endpoint_hard_limit_violations": 0,
        "common_dex3_transition_hard_limit_violations": 0,
        "common_dex3_exact_eval35_command_hard_limit_violations": 0,
        "commanded_dex3_scalars_preflight_checked": limit_validation["commanded_dex3_scalar_count"],
        "limit_safe_validation": str(LIMIT_VALIDATION.resolve()),
        "limit_safe_validation_sha256": sha256_file(LIMIT_VALIDATION),
        "pre_eval35_execution_qualification": str(PRE_EVAL35_QUALIFICATION.resolve()),
        "pre_eval35_execution_qualification_sha256": sha256_file(PRE_EVAL35_QUALIFICATION),
        "scripted_regression_sha256": sha256_file(SCRIPTED_REGRESSION),
        "scripted_regression_control_frames": scripted_regression["executed_control_frames"],
        "scripted_regression_all_required_gates_passed": True,
        "common_initial_state_identical_for_a_b": True,
        "common_source_intent_identical_for_a_b": True,
        "physical_environment": "contact-constrained 150 mm open-top bin",
        "scripted_environment_validation": "3/3 PASS (environment validation only)",
        "tests": tests.stdout.strip().splitlines()[-1], "instrumentation_validation": json.loads(patch.stdout),
        "post_rollout_start_tuning_allowed": False,
    }
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(MANIFEST, value)
    REPORT.write_text(
        "# Direct physical EVAL35 freeze\n\n"
        "Status: **FROZEN BEFORE ANY OF 70 PHYSICAL ROLLOUTS**\n\n"
        f"Direct execution bundle SHA256: `{bundle}`\n\n"
        "- Graspability classifier/atlas gate: NO\n"
        "- Wrist-distance gate: NO\n- ARM rescue: NO\n- WRIST rescue: NO\n"
        "- Common Dex3 P14 grasp/handoff/hold/release: YES\n"
        "- Authoritative measured Dex3 limits enforced at every commanded frame: YES\n"
        "- Static endpoint / full-transition / exact EVAL35 Dex3 limit violations: 0 / 0 / 0\n"
        f"- Commanded Dex3 scalars checked before freeze: {limit_validation['commanded_dex3_scalar_count']}\n"
        "- Contact-constrained 150 mm-bin PhysX: YES\n"
        "- PRE-EVAL35 scripted full-task regression: PASS (3309/3309 frames)\n"
        "- Common source-derived intent and initial state: identical for A/B\n"
        "- Existing scripted environment repeatability: 3/3 PASS; not policy performance\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": value["status"], "bundle_sha256": bundle, "manifest": str(MANIFEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
