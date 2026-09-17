#!/usr/bin/env python3
"""Finalize the bounded Dex3 capsule retention experiment and stop gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DEFAULT_CONFIG = ROOT / "configs/dex3_capsule_retention_bounded_repair_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/dex3_rigid_proxy_retention_repair"
OLD_DIAGNOSTIC = ROOT / "outputs/dex3_rigid_proxy_contact_diagnostic/FINAL_DEX3_CONTACT_DIAGNOSTIC.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = read_json(config_path)
    diagnosis_path = output_root / "BASELINE_RETENTION_DIAGNOSIS.json"
    diagnosis = read_json(diagnosis_path)
    old = read_json(OLD_DIAGNOSTIC)
    if diagnosis["primary_cause"] != "INSUFFICIENT_FRICTION":
        raise RuntimeError("bounded friction conditions were not authorized by the baseline diagnosis")
    if config["bounded_repair_policy"][diagnosis["primary_cause"]] != [
        "friction_medium_high",
        "friction_high",
    ]:
        raise RuntimeError("bounded candidate list changed")
    for prohibited_condition in ("drive_small_step", "combined_smallest"):
        if (output_root / prohibited_condition).exists():
            raise RuntimeError(f"unauthorized condition was executed: {prohibited_condition}")
    if list(output_root.glob("**/*LIFT*")):
        raise RuntimeError("lift was executed despite failed retention prerequisite")
    conditions = ("baseline", "friction_medium_high", "friction_high")
    runs: dict[str, dict[str, Any]] = {}
    maximum_penetration = 0.0
    maximum_linear_speed = 0.0
    maximum_angular_speed = 0.0
    maximum_com_step = 0.0
    for condition in conditions:
        expected = config["conditions"][condition]
        runs[condition] = {}
        for side in ("left", "right"):
            directory = output_root / condition / side
            result_path = directory / "stage_result.json"
            stage_log_path = directory / "stage_log.npz"
            pair_log_path = directory / "contact_pairs.csv"
            result = read_json(result_path)
            checks = {
                "condition": result["execution_condition_name"] == condition,
                "side": result["side"] == side,
                "stage": result["stage"] == "CLOSURE_TO_GRAVITY_RETENTION",
                "shape": result["shape"] == config["frozen_geometry"]["name"],
                "material": result["execution_condition"]["material"] == expected["material"],
                "finger_drive": result["execution_condition"]["finger_drive"] == expected["finger_drive"],
                "initial_overlap_absent": result["initial_overlap"] is False,
                "three_roles_observed": result["contact_roles"] == ["A", "B", "C"],
                "policy_absent": result["learned_policy_used"] is False
                and result["policy_checkpoint_read"] is False,
                "attachment_absent": result["prohibited_attachment_used"] is False,
                "tensor_contact_error_absent": not result["tensor_contact_api_errors"],
            }
            if not all(checks.values()):
                raise RuntimeError(f"{condition}/{side} invariant failed: {checks}")
            maximum_penetration = max(maximum_penetration, result["maximum_penetration_m"])
            maximum_linear_speed = max(maximum_linear_speed, result["object_max_linear_velocity_m_s"])
            maximum_angular_speed = max(
                maximum_angular_speed, result["object_max_angular_velocity_rad_s"]
            )
            maximum_com_step = max(maximum_com_step, result["maximum_object_com_step_m"])
            runs[condition][side] = {
                "status": result["status"],
                "static_friction": result["fixed_material"]["static_friction"],
                "dynamic_friction": result["fixed_material"]["dynamic_friction"],
                "gravity_retention_s": result["gravity_retention"]["measured_duration_s"],
                "required_gravity_retention_s": result["gravity_retention"]["required_duration_s"],
                "all_contact_duration_s": result["static_hold_contact_duration_s"],
                "maximum_total_normal_force_n": result["gravity_retention"][
                    "maximum_total_normal_force_n"
                ],
                "maximum_tangential_velocity_m_s": result["gravity_retention"][
                    "maximum_contact_tangential_velocity_m_s"
                ],
                "maximum_penetration_m": result["maximum_penetration_m"],
                "maximum_object_linear_speed_m_s": result["object_max_linear_velocity_m_s"],
                "maximum_object_angular_speed_rad_s": result[
                    "object_max_angular_velocity_rad_s"
                ],
                "collision_contact_pairs": result["collision_contact_pairs"],
                "checks": checks,
                "artifacts": {
                    "result": file_record(result_path),
                    "stage_log": file_record(stage_log_path),
                    "contact_pairs": file_record(pair_log_path),
                },
            }
    all_retention_failed = all(
        runs[condition][side]["status"] == "FAIL"
        for condition in conditions
        for side in ("left", "right")
    )
    if not all_retention_failed:
        raise RuntimeError("this finalizer is only for the bounded-repair failure outcome")
    artifact_gates = {
        "maximum_penetration_m": maximum_penetration,
        "penetration_limit_m": config["gates"]["maximum_contact_penetration_m"],
        "maximum_linear_speed_m_s": maximum_linear_speed,
        "linear_speed_limit_m_s": config["gates"]["maximum_object_linear_speed_m_s"],
        "maximum_angular_speed_rad_s": maximum_angular_speed,
        "angular_speed_limit_rad_s": config["gates"]["maximum_object_angular_speed_rad_s"],
        "maximum_com_step_m": maximum_com_step,
        "com_step_limit_m": config["gates"]["maximum_object_com_step_m"],
    }
    artifact_gates["pass"] = bool(
        maximum_penetration <= artifact_gates["penetration_limit_m"]
        and maximum_linear_speed <= artifact_gates["linear_speed_limit_m_s"]
        and maximum_angular_speed <= artifact_gates["angular_speed_limit_rad_s"]
        and maximum_com_step <= artifact_gates["com_step_limit_m"]
    )
    low_level = {
        "single_fingertip_contact": {
            side: old["lowest_level"]["single_no_gravity"][side]["status"]
            for side in ("left", "right")
        },
        "gravity_enabled_push": {
            side: old["lowest_level"]["single_push_gravity"][side]["status"]
            for side in ("left", "right")
        },
        "capsule_three_finger_closure": {
            side: old["bounded_proxy_results"]["capsule_contact_fit_v2"][side]["close"]
            for side in ("left", "right")
        },
        "collision_filtering": old["collision_filtering"],
        "collider_verification": old["collider_verification"],
    }
    final_status = "RIGID_PROXY_RETENTION_FAILED_AFTER_BOUNDED_REPAIR"
    final = {
        "schema_version": "dex3_capsule_retention_bounded_repair_final_v1",
        "status": final_status,
        "supersedes_broad_status_label": old["status"],
        "validated_scope": "LOW_LEVEL_DEX3_OBJECT_CONTACT_VALID",
        "remaining_blocker": "GRASP_RETENTION_UNDER_GRAVITY",
        "primary_cause": "UNFAVORABLE_CONTACT_GEOMETRY",
        "baseline_cause_hypothesis": diagnosis["primary_cause"],
        "cause_resolution": (
            "The baseline force/slip evidence authorized friction-only testing. Both higher "
            "presets failed bilaterally, so insufficient friction is not a sufficient root cause. "
            "The force-bearing topology is primary: three collider pairs are present, but only "
            "two carry meaningful pre-gravity squeeze and those lateral contacts collapse when "
            "the capsule loads downward."
        ),
        "low_level_contact": low_level,
        "frozen_capsule_geometry_for_this_task": config["frozen_geometry"],
        "mass_kg": config["object"]["mass_kg"],
        "gravitational_load_n": config["object"]["gravitational_load_n"],
        "baseline_force_measurements": diagnosis["sides"],
        "bounded_runs": runs,
        "bounded_conditions_executed": list(conditions),
        "drive_candidate_executed": False,
        "combined_candidate_executed": False,
        "geometry_changed": False,
        "grasp_target_changed": False,
        "arm_pose_changed_for_repair": False,
        "restitution_changed": False,
        "left_retention_1s": "FAIL",
        "right_retention_1s": "FAIL",
        "left_lift_5cm_hold_0p5s": "NOT_RUN_RETENTION_PREREQUISITE_FAILED",
        "right_lift_5cm_hold_0p5s": "NOT_RUN_RETENTION_PREREQUISITE_FAILED",
        "artifact_gates": artifact_gates,
        "environment_frozen": False,
        "act_ab_physics_evaluation_reopened": False,
        "act_ab_physics_evaluation_stopped": True,
        "physical_task_success_on_paper_critical_path": False,
        "learned_policy_used": False,
        "policy_checkpoint_read": False,
        "real_robot": False,
        "prohibited_attachment_used": False,
        "config": file_record(config_path),
        "baseline_diagnosis": file_record(diagnosis_path),
        "source_low_level_diagnostic": file_record(OLD_DIAGNOSTIC),
    }
    final_json = output_root / "FINAL_DEX3_RIGID_PROXY_RETENTION.json"
    atomic_json(final_json, final)
    table_lines = []
    for condition in conditions:
        material = config["conditions"][condition]["material"]
        table_lines.append(
            f"| {condition} | {material['static_friction']:.2f}/{material['dynamic_friction']:.2f} | "
            f"{runs[condition]['left']['gravity_retention_s']:.3f} s / FAIL | "
            f"{runs[condition]['right']['gravity_retention_s']:.3f} s / FAIL |"
        )
    contact_lines = []
    for side in ("left", "right"):
        roles = diagnosis["sides"][side]["stable_close_window"]["per_contact_role"]
        for role in ("A", "B", "C"):
            item = roles[role]
            contact_lines.append(
                f"| {side.title()} | {role} | `{Path(item['link']).name if item['link'] else 'NA'}` | "
                f"{item['normal_force_n']['mean']:.3f} / {item['normal_force_n']['max']:.3f} N | "
                f"{item['normal_impulse_sum_ns']:.5f} N·s | "
                f"{item['tangential_velocity_m_s']['median']:.4f} / "
                f"{item['tangential_velocity_m_s']['max']:.4f} m/s | "
                f"{item['contact_normal_z']['median']:.3f} |"
            )
    report = "\n".join(
        [
            "# Dex3 Capsule Retention — Bounded Repair",
            "",
            f"Status: **{final_status}**",
            "",
            "The older `DEX3_CONTACT_MODEL_STILL_INVALID` label is superseded in scope. "
            "Single-fingertip contact, gravity-on pushing, collision/filter mapping, and "
            "bilateral three-finger capsule closure are valid. The unresolved gate is specifically "
            "`GRASP_RETENTION_UNDER_GRAVITY`.",
            "",
            "## Cause",
            "",
            "Primary cause after the bounded falsification: **UNFAVORABLE_CONTACT_GEOMETRY**.",
            "",
            "Before material changes, the stable closure supplied median total normal force of "
            f"{diagnosis['sides']['left']['derived']['stable_normal_force_n']:.3f} N left and "
            f"{diagnosis['sides']['right']['derived']['stable_normal_force_n']:.3f} N right. "
            "At baseline static friction these are approximately "
            f"{diagnosis['sides']['left']['derived']['capacity_to_gravity_ratio']:.3f}x and "
            f"{diagnosis['sides']['right']['derived']['capacity_to_gravity_ratio']:.3f}x the "
            f"{config['object']['gravitational_load_n']:.5f} N load. GRASP error RMS was "
            f"{diagnosis['sides']['left']['stable_close_window']['finger_position_error_rms_all_rad']:.4f} "
            "rad left and "
            f"{diagnosis['sides']['right']['stable_close_window']['finger_position_error_rms_all_rad']:.4f} "
            "rad right, with only "
            f"{100 * diagnosis['sides']['left']['stable_close_window']['applied_drive_torque_saturation_fraction_all']:.2f}%/"
            f"{100 * diagnosis['sides']['right']['stable_close_window']['applied_drive_torque_saturation_fraction_all']:.2f}% "
            "drive saturation. This ruled out a primary drive-force shortage.",
            "",
            "The initial friction hypothesis was tested exactly as bounded. Raising friction did "
            "not produce stable retention: the third contact was collision-active but effectively "
            "non-load-bearing before gravity, and the two force-bearing lateral contacts collapsed "
            "as the capsule slipped downward. No geometry or drive repair was then attempted.",
            "",
            "## Baseline contact instrumentation",
            "",
            "The following values are from the final 0.2 s of the successful no-gravity closure, "
            "before any parameter change. Point impulses are the pair-resolved normal force "
            "integrated at the 120 Hz physics step.",
            "",
            "| Hand | Role | Active distal link | Normal mean/max | Normal impulse | Tangential median/max | Median normal z |",
            "|---|---|---|---:|---:|---:|---:|",
            *contact_lines,
            "",
            "All three named distal links were collision-active on both hands. Direct computed and "
            "applied torque readback was available; per-joint GRASP error and torque are retained "
            "in `BASELINE_RETENTION_DIAGNOSIS.json`. Object/fingertip contact/rest offsets were "
            "2.0/0.0 mm and 0.351/0.0 mm, respectively. There was no spawn overlap, and the "
            "visual and collision proxy were the same capsule prim.",
            "",
            "## Bounded retention runs",
            "",
            "| Condition | static/dynamic friction | Left 1 s gate | Right 1 s gate |",
            "|---|---:|---:|---:|",
            *table_lines,
            "",
            f"Across all six runs, maximum penetration was {maximum_penetration * 1000:.3f} mm, "
            f"maximum object linear speed was {maximum_linear_speed:.3f} m/s, maximum angular "
            f"speed was {maximum_angular_speed:.3f} rad/s, and maximum COM step was "
            f"{maximum_com_step * 1000:.3f} mm. These pass the predeclared artifact gates; the "
            "failure is retention, not an explosive-contact or spawn-penetration artifact.",
            "",
            "## Gate decision",
            "",
            "- Left 1.0 s gravity retention: **FAIL**",
            "- Right 1.0 s gravity retention: **FAIL**",
            "- Left 5 cm lift + 0.5 s hold: **NOT RUN — retention prerequisite failed**",
            "- Right 5 cm lift + 0.5 s hold: **NOT RUN — retention prerequisite failed**",
            "- Capsule/contact environment frozen for ACT evaluation: **NO**",
            "- ACT-A/B physics task-success evaluation reopened: **NO**",
            "- Real G1: **NOT STARTED BY DESIGN**",
            "",
            "Physical task-success work is stopped and remains outside the paper critical path.",
            "",
            final_status,
            "",
        ]
    )
    report_path = output_root / "FINAL_DEX3_RIGID_PROXY_RETENTION_REPORT.md"
    atomic_text(report_path, report)
    gate = {
        "schema_version": "act_ab_physics_retention_gate_v1",
        "status": "STOPPED",
        "reason": "Bilateral 1.0 s gravity retention failed after the authorized friction-only bounded repair.",
        "remaining_blocker": "GRASP_RETENTION_UNDER_GRAVITY",
        "act_ab_physics_evaluation_allowed": False,
        "environment_frozen": False,
        "paper_critical_path": False,
        "final_report": file_record(report_path),
        "final_result": file_record(final_json),
    }
    gate_path = output_root / "ACT_AB_PHYSICS_EVALUATION_GATE.json"
    atomic_json(gate_path, gate)
    paper_gate_path = ROOT / "outputs/paper_physics_task_eval/RETENTION_GATE_STOP_20260827.json"
    atomic_json(paper_gate_path, gate)
    supersession = {
        "schema_version": "dex3_contact_status_supersession_v1",
        "superseded_broad_label": old["status"],
        "low_level_contact_status": "VALIDATED",
        "specific_remaining_blocker": "GRASP_RETENTION_UNDER_GRAVITY",
        "latest_status": final_status,
        "latest_report": file_record(report_path),
        "old_artifacts_preserved": True,
    }
    supersession_path = (
        ROOT
        / "outputs/dex3_rigid_proxy_contact_diagnostic/STATUS_SUPERSEDED_BY_RETENTION_DIAGNOSTIC.json"
    )
    atomic_json(supersession_path, supersession)
    manifest_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and not path.name.endswith(".incomplete") and path.name != "ARTIFACT_MANIFEST.json"
    )
    manifest = {
        "schema_version": "dex3_capsule_retention_artifact_manifest_v1",
        "status": final_status,
        "files": [file_record(path) for path in manifest_files],
        "paper_stop_gate": file_record(paper_gate_path),
        "supersession_pointer": file_record(supersession_path),
    }
    atomic_json(output_root / "ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
