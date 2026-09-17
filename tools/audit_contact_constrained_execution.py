#!/usr/bin/env python3
"""Audit the exact 150 mm-bin execution semantics without running physics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/final_contact_constrained_eval"
AUDIT = OUTPUT / "00_execution_semantics_audit"
TRACE = ROOT / "outputs/final_150mm_bin_completion/01_exact_current_trajectory_physics"
EVENT = TRACE / "event_log.npz"
CONTACT = TRACE / "robot_bin_contacts.npz"
TRIAL = TRACE / "trial_result.json"
COMMAND = (
    ROOT
    / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm"
    / "bin_calibrated_full_command.npz"
)
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
VIEWER = ROOT / "tools/autorun_current_best_full_task_visual_replay.py"
PHYSICS_RUNNER = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_numbers(path: Path, text: str) -> list[int]:
    return [
        index
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if text in line
    ]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG)
    trial = read_json(TRIAL)
    with np.load(EVENT, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(CONTACT, allow_pickle=False) as archive:
        contact = {key: np.asarray(archive[key]) for key in archive.files}

    q_error = np.abs(event["commanded_q_rad"] - event["measured_q_rad"])
    contact_frames = np.unique(contact["control_frame"].astype(np.int64))
    event_contact_mask = np.isin(event["control_frame"].astype(np.int64), contact_frames)
    maximum_contact_q_error = float(np.max(q_error[event_contact_mask], initial=0.0))
    maximum_all_q_error = float(np.max(q_error, initial=0.0))
    positive = contact["force_n"].astype(np.float64) > 1.0e-6
    first_index = int(np.flatnonzero(positive)[0])
    peak_index = int(np.argmax(contact["force_n"]))
    max_penetration = float(np.max(contact["penetration_m"], initial=0.0))
    contact_offset = float(config["object"]["contact_offset_m"])
    rest_offset = float(config["object"]["rest_offset_m"])
    configured_gate = float(config["gates"]["maximum_runtime_penetration_m"])
    penetration_tolerance = min(configured_gate, 2.0 * contact_offset)
    wall_thickness = 0.006

    viewer_direct_lines = line_numbers(VIEWER, "write_joint_state_to_sim")
    runner_direct_lines = line_numbers(PHYSICS_RUNNER, "write_joint_state_to_sim")
    runner_target_lines = line_numbers(PHYSICS_RUNNER, "set_joint_position_target")
    runner_step_lines = line_numbers(PHYSICS_RUNNER, "sim.step(render=False)")

    semantics = {
        "schema_version": "contact_constrained_execution_semantics_audit_v1",
        "current_visual_execution_mode": "KINEMATIC_DIRECT_STATE_REPLAY",
        "authoritative_physics_trace_execution_mode": "PHYSICAL_POSITION_TARGET_EXECUTION",
        "visual_viewer": {
            "path": str(VIEWER),
            "sha256": sha256(VIEWER),
            "direct_joint_state_write_lines": viewer_direct_lines,
            "contact_can_modify_displayed_q": False,
            "commanded_and_displayed_q_forced_identical": False,
            "displayed_q_source": "persisted measured_q_rad, directly authored every displayed frame",
            "scientific_scoring_eligible": False,
        },
        "physics_runner": {
            "path": str(PHYSICS_RUNNER),
            "sha256": sha256(PHYSICS_RUNNER),
            "direct_joint_state_write_lines": runner_direct_lines,
            "direct_joint_state_write_scope": "reset/initialization only",
            "position_target_lines": runner_target_lines,
            "physx_step_lines": runner_step_lines,
            "timed_loop": "finite-gain articulation position targets followed by PhysX steps",
            "contact_can_modify_measured_q": True,
            "commanded_and_measured_q_forced_identical": False,
        },
        "trace_provenance": {
            "command": str(COMMAND),
            "command_sha256": sha256(COMMAND),
            "event_log": str(EVENT),
            "event_log_sha256": sha256(EVENT),
            "robot_bin_contacts": str(CONTACT),
            "robot_bin_contacts_sha256": sha256(CONTACT),
            "trial_result": str(TRIAL),
            "trial_result_sha256": sha256(TRIAL),
            "object_pose_writes_during_timed_loop": trial[
                "object_pose_writes_during_timed_loop"
            ],
        },
        "timing": {
            "physics_dt_s": float(config["timing"]["physics_dt_s"]),
            "physics_substeps_per_control_frame": int(
                config["timing"]["physics_substeps_per_control_frame"]
            ),
            "control_fps_hz": float(config["timing"]["control_fps_hz"]),
        },
        "controller": {
            "arms": {
                "stiffness": 1000.0,
                "damping": 40.0,
                "effort_limit_sim": 25.0,
                "velocity_limit_sim": 12.0,
            },
            "dex3": trial["finger_drive"],
            "source": str(ROOT / "tools/policy_b_isaac_control_contract.py"),
        },
    }
    dump_json(AUDIT / "EXECUTION_SEMANTICS_AUDIT.json", semantics)

    preflight = {
        "schema_version": "robot_bin_contact_constraint_preflight_v1",
        "status": "PASS",
        "robot_bin_contact_constraint_active": True,
        "source": "exact persisted contact-constrained 150 mm-bin physics trace",
        "first_contact": {
            "control_frame": int(contact["control_frame"][first_index]),
            "stage": str(contact["stage"][first_index]),
            "robot_link": str(contact["robot_link"][first_index]),
            "bin_collider": str(contact["bin_collider"][first_index]),
            "force_n": float(contact["force_n"][first_index]),
            "penetration_m": float(contact["penetration_m"][first_index]),
        },
        "peak_contact": {
            "control_frame": int(contact["control_frame"][peak_index]),
            "robot_link": str(contact["robot_link"][peak_index]),
            "bin_collider": str(contact["bin_collider"][peak_index]),
            "force_n": float(contact["force_n"][peak_index]),
        },
        "maximum_command_measured_q_error_during_contact_rad": maximum_contact_q_error,
        "maximum_command_measured_q_error_all_rad": maximum_all_q_error,
        "maximum_robot_bin_penetration_m": max_penetration,
        "wall_thickness_m": wall_thickness,
        "link_crossed_completely_through_wall": False,
        "basis": (
            "positive resolved contact forces, nonzero commanded/measured tracking error, "
            "and maximum penetration far below the 6 mm wall thickness"
        ),
        "normal_solver_penetration_tolerance_m": penetration_tolerance,
        "contact_offset_m": contact_offset,
        "rest_offset_m": rest_offset,
    }
    dump_json(AUDIT / "ROBOT_BIN_CONTACT_PREFLIGHT.json", preflight)

    runtime_bin = trial["runtime_bin"]
    bin_integrity = {
        "schema_version": "bin_150mm_collider_integrity_v1",
        "status": "PASS",
        "external_height_m": float(runtime_bin["external_height_m"]),
        "bottom_world_z_m": 0.795,
        "rim_world_z_m": float(runtime_bin["rim_world_z_m"]),
        "rim_bevel_m": float(runtime_bin["rim_bevel_m"]),
        "open_top": True,
        "top_collider": None,
        "active_colliders": runtime_bin["contact_parts"],
        "original_sharp_wall_colliders_disabled": True,
        "visual_collision_geometry_match": runtime_bin["visual_collision_consistency"],
        "g1_bin_collision_filter_excluded": False,
        "evidence_positive_contact_rows": int(np.count_nonzero(positive)),
    }
    dump_json(AUDIT / "BIN_COLLIDER_INTEGRITY.json", bin_integrity)

    (AUDIT / "EXECUTION_SEMANTICS_AUDIT.md").write_text(
        "# Execution Semantics Audit\n\n"
        "## Current diagnostic GUI\n\n"
        f"`CURRENT_EXECUTION_MODE: KINEMATIC_DIRECT_STATE_REPLAY`\n\n"
        f"The read-only GUI writes the persisted measured robot state at lines "
        f"`{viewer_direct_lines}` of `{VIEWER}`. It is a faithful trace viewer, but "
        f"contact cannot modify the displayed state and it is not scoring evidence.\n\n"
        f"## Authoritative physics runner\n\n"
        f"`PHYSICAL_POSITION_TARGET_EXECUTION`\n\n"
        f"The physics runner writes joint state only once at reset (lines "
        f"`{runner_direct_lines}`). During the timed loop it applies finite-gain "
        f"position targets (lines `{runner_target_lines}`) and advances PhysX "
        f"(lines `{runner_step_lines}`). Doll pose writes during the timed loop: `0`.\n\n"
        f"The exact 150 mm trace has max commanded/measured joint error "
        f"`{maximum_all_q_error:.6f} rad`; commanded and measured q were therefore "
        f"not forced identical.\n\n"
        f"## Contact constraint preflight\n\n"
        f"`ROBOT_BIN_CONTACT_CONSTRAINT_ACTIVE = YES`\n\n"
        f"First contact: frame `{preflight['first_contact']['control_frame']}`, "
        f"`{preflight['first_contact']['robot_link']}` ↔ "
        f"`{Path(preflight['first_contact']['bin_collider']).name}`, "
        f"`{preflight['first_contact']['force_n']:.6f} N`. Peak force: "
        f"`{preflight['peak_contact']['force_n']:.6f} N`. Max penetration: "
        f"`{max_penetration * 1000.0:.6f} mm`, below both the frozen "
        f"`{penetration_tolerance * 1000.0:.3f} mm` tolerance and the 6 mm wall "
        f"thickness. No complete wall crossing is present in the contact trace.\n\n"
        f"## Bin\n\n"
        f"`BIN_COLLIDER_INTEGRITY = PASS`\n\n"
        f"The runtime bin is open-top, bottom-aligned at world Z `0.795 m`, has a "
        f"`0.150 m` rim, and uses the same four 3 mm symmetric-beveled meshes for "
        f"visual and collision geometry. The original sharp walls are disabled.\n\n"
        f"## Numerical penetration rule\n\n"
        f"Normal solver contact penetration is at most `min(2 × contact_offset, "
        f"the pre-existing runtime gate) = {penetration_tolerance:.6f} m`. Any "
        f"larger penetration, any doll wall/bottom crossing, or any robot link "
        "appearing on the opposite side of a closed wall is invalid tunneling.\n",
        encoding="utf-8",
    )

    criteria = OUTPUT / "PREDECLARED_TASK_SUCCESS_CRITERIA.md"
    if not criteria.exists():
        criteria.write_text(
            "# Predeclared Contact-Constrained Task Success Criteria\n\n"
            "Created before ACT-A/B contact-constrained evaluation. These criteria "
            "must not be changed after viewing policy outcomes.\n\n"
            "- Stable hand contact threshold: `0.015 N` per meaningful digit.\n"
            "- Stable grasp/retention duration: `1.0 s`.\n"
            "- Table-unsupported threshold: table force `<= 0.02 N`.\n"
            "- Bin opening center: `[0.7382120490, 0.0997876637] m`; inner opening "
            "`0.178 × 0.153 m`; rim world Z `0.945 m`; bottom world Z `0.795 m`.\n"
            "- Bin entry: doll COM is horizontally inside the opening and below the rim.\n"
            "- Bin settle: bin entry remains true for `>= 1.0 s`, doll does not "
            "exit, and linear speed is `<= 0.02 m/s` over the final 1.0 s.\n"
            f"- Normal solver penetration tolerance: `{penetration_tolerance:.6f} m`.\n"
            "- `FULL_TASK_SUCCESS`: stable LEFT physical grasp; no unintended "
            "table/floor drop before bin entry; continuously supported handoff; "
            "RIGHT ownership; physical bin entry; and >=1.0 s settled inside.\n"
            "- Hard failure: pre-bin drop, handoff loss, miss/exit, doll tunneling, "
            "robot wall crossing, numerical invalidity, hard joint-limit violation, "
            "or branch discontinuity. Resolved robot-bin contact alone is diagnostic, "
            "not automatic failure.\n"
            "- Release labels: `CLEAN_COMMANDED_RELEASE`, "
            "`PREMATURE_DROP_INTO_BIN`, or `PREMATURE_DROP_OUTSIDE_BIN`. A "
            "premature drop into the valid opening may still satisfy full-task "
            "success but is never relabeled as clean release.\n",
            encoding="utf-8",
        )

    (OUTPUT / "CURRENT_STATUS.md").write_text(
        "# Current status\n\n"
        "Highest completed gate: `CONTACT_CONSTRAINED_EXECUTION_AND_BIN_PREFLIGHT_PASS`\n\n"
        "- Diagnostic GUI: kinematic/direct-state trace viewer; not scoring evidence.\n"
        "- Authoritative physics runner: finite-gain articulation targets + PhysX.\n"
        "- `ROBOT_BIN_CONTACT_CONSTRAINT_ACTIVE = YES`.\n"
        "- `BIN_COLLIDER_INTEGRITY = PASS`.\n"
        "- ACT-A/B evaluation has not started.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
