#!/usr/bin/env python3
"""Freeze the 3/3 contact-constrained scripted environment and controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/final_contact_constrained_eval"
RUNS = OUTPUT / "02_scripted_validation"
FREEZE = OUTPUT / "03_freeze"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
COMMAND = (
    ROOT
    / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm"
    / "bin_calibrated_full_command.npz"
)
SCENE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
CONTROLLER = ROOT / "tools/policy_b_isaac_control_contract.py"
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
ENTRYPOINT = ROOT / "tools/run_contact_constrained_full_task_physics.py"
SCORER = ROOT / "tools/score_contact_constrained_full_task.py"
CRITERIA = OUTPUT / "PREDECLARED_TASK_SUCCESS_CRITERIA.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if FREEZE.exists():
        raise FileExistsError(f"refusing to overwrite freeze: {FREEZE}")
    FREEZE.mkdir(parents=True)
    results = [
        load(RUNS / f"run_{index:02d}/CONTACT_CONSTRAINED_TASK_RESULT.json")
        for index in (1, 2, 3)
    ]
    events = [RUNS / f"run_{index:02d}/event_log.npz" for index in (1, 2, 3)]
    contacts = [
        RUNS / f"run_{index:02d}/robot_bin_contacts.npz" for index in (1, 2, 3)
    ]
    if not all(result["status"] == "PASS" for result in results):
        raise RuntimeError("scripted task is not 3/3 PASS")
    if len({sha256(path) for path in events}) != 1 or len(
        {sha256(path) for path in contacts}
    ) != 1:
        raise RuntimeError("deterministic repeat traces are not byte-identical")
    config = load(CONFIG)
    environment = {
        "schema_version": "final_contact_constrained_physical_environment_v1",
        "status": "FROZEN",
        "bin": {
            "external_height_m": 0.150,
            "bottom_world_z_m": 0.795,
            "rim_world_z_m": 0.945,
            "rim_bevel_m": 0.003,
            "opening_dimensions_xy_m": [0.178, 0.153],
            "opening_center_world_xy_m": [0.7382120490074158, 0.09978766366839409],
            "wall_thickness_m": 0.006,
            "open_top": True,
            "collision_parts": [
                "Bottom",
                "FrontWallBeveled",
                "BackWallBeveled",
                "LeftWallBeveled",
                "RightWallBeveled",
            ],
            "original_sharp_wall_colliders_disabled": True,
        },
        "doll": {
            "visual_dimensions_m": config["object"]["visual_dimensions_m"],
            "collision_geometry": config["geometry_candidates"][0],
            "mass_kg": config["object"]["mass_kg"],
            "material": config["material"],
            "contact_offset_m": config["object"]["contact_offset_m"],
            "rest_offset_m": config["object"]["rest_offset_m"],
            "restitution": config["object"]["restitution"],
            "linear_damping": config["object"]["linear_damping"],
            "angular_damping": config["object"]["angular_damping"],
        },
        "physics": {
            "dt_s": config["timing"]["physics_dt_s"],
            "substeps_per_control_frame": config["timing"][
                "physics_substeps_per_control_frame"
            ],
            "control_fps_hz": config["timing"]["control_fps_hz"],
            "gravity_m_s2": config["simulation"]["gravity_m_s2"],
            "use_fabric": config["simulation"]["use_fabric"],
            "deterministic_seed": config["simulation"]["deterministic_seed"],
        },
        "penetration": {
            "normal_solver_tolerance_m": 0.003,
            "robot_wall_crossing_invalid": True,
            "doll_wall_or_bottom_crossing_invalid": True,
        },
        "real_robot": False,
    }
    controller = {
        "schema_version": "final_common_contact_constrained_execution_controller_v1",
        "status": "FROZEN",
        "execution_mode": "finite-gain articulation position targets + PhysX measured state",
        "reset_direct_state_write_only": True,
        "direct_state_writes_during_execution": False,
        "command": str(COMMAND),
        "command_sha256": sha256(COMMAND),
        "command_frames": 3309,
        "command_rate_hz": 30.0,
        "controller_contract": str(CONTROLLER),
        "controller_contract_sha256": sha256(CONTROLLER),
        "finger_drive": config["finger_drive"],
        "right_three_digit_gate": {
            "force_threshold_n": 0.015,
            "minimum_simultaneous_support_s": 0.5,
            "left_release_interlocked": True,
        },
        "common_for_act_a_and_act_b": True,
        "episode_independent": True,
        "object_relative_local_primitives": True,
        "scripted_validation_is_not_policy_performance": True,
        "prohibited_mechanisms": {
            "attachment": False,
            "magnet": False,
            "weld": False,
            "object_follow": False,
            "teleportation": False,
        },
    }
    environment_path = FREEZE / "FINAL_PHYSICAL_ENVIRONMENT.json"
    controller_path = FREEZE / "FINAL_COMMON_EXECUTION_CONTROLLER.json"
    dump(environment_path, environment)
    dump(controller_path, controller)
    criteria_copy = FREEZE / "PREDECLARED_TASK_SUCCESS_CRITERIA.md"
    shutil.copyfile(CRITERIA, criteria_copy)
    repeatability = {
        "schema_version": "contact_constrained_scripted_repeatability_v1",
        "status": "PASS",
        "successes": 3,
        "attempts": 3,
        "byte_identical_event_logs": True,
        "event_log_sha256": sha256(events[0]),
        "byte_identical_robot_bin_contact_logs": True,
        "robot_bin_contact_log_sha256": sha256(contacts[0]),
        "release_classification_all_runs": [
            result["release_classification"] for result in results
        ],
        "clean_commanded_release": False,
        "full_task_success_definition_satisfied": True,
        "run_results": [
            {
                "run": index,
                "result": str(RUNS / f"run_{index:02d}/CONTACT_CONSTRAINED_TASK_RESULT.json"),
                "result_sha256": sha256(
                    RUNS / f"run_{index:02d}/CONTACT_CONSTRAINED_TASK_RESULT.json"
                ),
            }
            for index in (1, 2, 3)
        ],
    }
    repeatability_path = FREEZE / "SCRIPTED_REPEATABILITY.json"
    dump(repeatability_path, repeatability)
    (FREEZE / "SCRIPTED_REPEATABILITY.md").write_text(
        "# Contact-constrained scripted repeatability\n\n"
        "Status: **PASS (3/3)**\n\n"
        f"Event logs are byte-identical: `{repeatability['event_log_sha256']}`.\n\n"
        "All runs satisfy grasp → no unintended pre-bin drop → handoff → RIGHT "
        "ownership → physical bin entry → 1 s settle. All three are explicitly "
        "classified `PREMATURE_DROP_INTO_BIN`, not clean commanded release.\n",
        encoding="utf-8",
    )

    authoritative = [
        environment_path,
        controller_path,
        criteria_copy,
        repeatability_path,
        CONFIG,
        COMMAND,
        SCENE,
        CONTROLLER,
        ENGINE,
        ENTRYPOINT,
        SCORER,
        *events,
        *contacts,
        *[
            RUNS / f"run_{index:02d}/CONTACT_CONSTRAINED_TASK_RESULT.json"
            for index in (1, 2, 3)
        ],
    ]
    manifest_path = FREEZE / "FREEZE_MANIFEST.json"
    manifest = {
        "schema_version": "final_contact_constrained_freeze_manifest_v1",
        "status": "FROZEN",
        "freeze_scope": "common 150 mm-bin G1 target-embodiment execution layer",
        "scripted_repeatability": "3/3 PASS",
        "post_freeze_tuning_allowed": False,
        "act_a_b_specific_parameters": False,
        "files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in authoritative
        ],
    }
    dump(manifest_path, manifest)
    (FREEZE / "FREEZE_MANIFEST.md").write_text(
        "# Final contact-constrained freeze\n\n"
        "Status: **FROZEN after scripted 3/3 PASS**\n\n"
        "The 150 mm bin, compressed-doll proxy, finite-gain G1/Dex3 controller, "
        "physical timing, contact/tunneling thresholds, and common command contract "
        "are immutable for ACT-A/B evaluation. No method- or episode-specific "
        "parameter exists in this freeze.\n\n"
        f"Manifest: `{manifest_path}`\n",
        encoding="utf-8",
    )
    (OUTPUT / "CURRENT_STATUS.md").write_text(
        "# Current status\n\n"
        "Highest completed gate: `CONTACT_CONSTRAINED_SCRIPTED_3_OF_3_AND_FREEZE_COMPLETE`\n\n"
        "- Scripted full task: 3/3 PASS; byte-identical physics/contact traces.\n"
        "- Release classification: PREMATURE_DROP_INTO_BIN (not clean release).\n"
        "- Final common environment/controller: FROZEN.\n"
        "- NEW_UNSEEN_2 integrity/task completeness: PASS; conversion now permitted.\n"
        "- Next gate: frozen Fair-A/Proposed-B conversion and EVAL10 physical evaluation.\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
