#!/usr/bin/env python3
"""Finalize the policy-independent Dex3 rigid-proxy contact diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/dex3_rigid_proxy_contact_diagnostic"
CONFIG = ROOT / "configs/dex3_rigid_proxy_contact_diagnostic_v1.json"
STATIC = OUTPUT / "static_audit/STATIC_COLLIDER_AUDIT.json"


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


def result(relative: str) -> dict[str, Any]:
    path = OUTPUT / relative / "stage_result.json"
    row = read_json(path)
    row["result_path"] = str(path)
    row["result_sha256"] = sha256_file(path)
    return row


def skipped_lift(side: str, blocking: dict[str, Any]) -> dict[str, Any]:
    directory = OUTPUT / f"runtime/capsule_contact_fit_v2/{side}/05_vertical_lift_5cm"
    payload = {
        "schema_version": "dex3_contact_stage_skipped_v1",
        "status": "NOT_RUN_BLOCKED_BY_PRECEDING_STAGE",
        "stage": "VERTICAL_LIFT_5CM",
        "side": side,
        "shape": "capsule_contact_fit_v2",
        "blocking_stage": "STATIC_HOLD",
        "blocking_result": blocking["result_path"],
        "blocking_result_sha256": blocking["result_sha256"],
        "collision_contact_pairs": [],
        "contact_duration_s": None,
        "object_max_linear_velocity_m_s": None,
        "object_max_angular_velocity_rad_s": None,
        "maximum_penetration_m": None,
        "minimum_fingertip_point_to_proxy_signed_distance_m": None,
        "initial_overlap": None,
        "object_contact_offset_m": 0.002,
        "object_rest_offset_m": 0.0,
        "finger_drive_kp": 100.0,
        "finger_drive_kd": 4.0,
        "requested_lift_m": 0.05,
        "measured_lift_m": None,
        "reason": "The unchanged hierarchy forbids lift execution after static hold fails.",
        "learned_policy_used": False,
        "policy_checkpoint_read": False,
        "real_robot": False,
    }
    path = directory / "SKIPPED.json"
    atomic_json(path, payload)
    payload["result_path"] = str(path)
    payload["result_sha256"] = sha256_file(path)
    return payload


def main() -> int:
    config = read_json(CONFIG)
    static = read_json(STATIC)
    scene = Path(config["scene"])
    if sha256_file(scene) != static["scene_sha256"]:
        raise RuntimeError("source scene changed during session-layer diagnosis")
    paths = {
        "sphere_left_no_gravity": "runtime/sphere_75mm/left/01_single_no_gravity_tensor",
        "sphere_right_no_gravity": "runtime/sphere_75mm/right/01_single_no_gravity",
        "sphere_left_push": "runtime/sphere_75mm/left/02_single_push_gravity",
        "sphere_right_push": "runtime/sphere_75mm/right/02_single_push_gravity",
        "ellipsoid_v2_left_close": "runtime/ellipsoid_contact_fit_v2/left/03_fixed_close",
        "ellipsoid_v2_right_close": "runtime/ellipsoid_contact_fit_v2/right/03_fixed_close",
        "capsule_v1_left_close": "runtime/capsule_v1/left/03_fixed_close_retry",
        "capsule_v1_right_close": "runtime/capsule_v1/right/03_fixed_close",
        "capsule_v1_left_hold": "runtime/capsule_v1/left/04_static_hold_continuation",
        "capsule_v1_right_hold": "runtime/capsule_v1/right/04_static_hold_continuation",
        "capsule_v2_left_close": "runtime/capsule_contact_fit_v2/left/03_fixed_close",
        "capsule_v2_right_close": "runtime/capsule_contact_fit_v2/right/03_fixed_close",
        "capsule_v2_left_hold": "runtime/capsule_contact_fit_v2/left/04_static_hold",
        "capsule_v2_right_hold": "runtime/capsule_contact_fit_v2/right/04_static_hold",
    }
    results = {name: result(path) for name, path in paths.items()}
    offset_audit = result(
        "runtime/capsule_contact_fit_v2/left/offset_readback_audit"
    )
    if offset_audit["status"] != "PASS":
        raise RuntimeError("effective PhysX offset readback audit did not pass")
    required_pass = (
        "sphere_left_no_gravity",
        "sphere_right_no_gravity",
        "sphere_left_push",
        "sphere_right_push",
        "capsule_v1_left_close",
        "capsule_v1_right_close",
        "capsule_v2_left_close",
        "capsule_v2_right_close",
    )
    if not all(results[name]["status"] == "PASS" for name in required_pass):
        raise RuntimeError("expected lower-level pass evidence is incomplete")
    if not all(
        results[name]["status"] == "FAIL"
        for name in (
            "ellipsoid_v2_left_close",
            "ellipsoid_v2_right_close",
            "capsule_v1_left_hold",
            "capsule_v1_right_hold",
            "capsule_v2_left_hold",
            "capsule_v2_right_hold",
        )
    ):
        raise RuntimeError("expected bounded-candidate failure evidence is incomplete")
    lift = {
        side: skipped_lift(side, results[f"capsule_v2_{side}_hold"])
        for side in ("left", "right")
    }
    matrix_rows: list[dict[str, Any]] = []
    for name, row in results.items():
        matrix_rows.append(
            {
                "record": name,
                "shape": row["shape"],
                "side": row["side"],
                "stage": row["stage"],
                "status": row["status"],
                "contact_duration_s": row["contact_duration_s"],
                "contact_roles": "|".join(row["contact_roles"]),
                "collision_contact_pairs": "|".join(row["collision_contact_pairs"]),
                "max_object_linear_velocity_m_s": row["object_max_linear_velocity_m_s"],
                "max_object_angular_velocity_rad_s": row["object_max_angular_velocity_rad_s"],
                "maximum_penetration_m": row["maximum_penetration_m"],
                "minimum_fingertip_distance_to_proxy_m": row[
                    "minimum_fingertip_point_to_proxy_signed_distance_m"
                ],
                "initial_overlap": row["initial_overlap"],
                "object_contact_offset_m": row["object_contact_offset_m"],
                "object_rest_offset_m": row["object_rest_offset_m"],
                "finger_drive_kp": row["finger_drive"]["kp"],
                "finger_drive_kd": row["finger_drive"]["kd"],
                "result_path": row["result_path"],
                "result_sha256": row["result_sha256"],
            }
        )
    for side, row in lift.items():
        matrix_rows.append(
            {
                "record": f"capsule_v2_{side}_lift",
                "shape": row["shape"],
                "side": side,
                "stage": row["stage"],
                "status": row["status"],
                "contact_duration_s": None,
                "contact_roles": "",
                "collision_contact_pairs": "",
                "max_object_linear_velocity_m_s": None,
                "max_object_angular_velocity_rad_s": None,
                "maximum_penetration_m": None,
                "minimum_fingertip_distance_to_proxy_m": None,
                "initial_overlap": None,
                "object_contact_offset_m": row["object_contact_offset_m"],
                "object_rest_offset_m": row["object_rest_offset_m"],
                "finger_drive_kp": row["finger_drive_kp"],
                "finger_drive_kd": row["finger_drive_kd"],
                "result_path": row["result_path"],
                "result_sha256": row["result_sha256"],
            }
        )
    table = OUTPUT / "STAGE_MATRIX.csv"
    temporary = table.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    os.replace(temporary, table)
    sphere_audit = static["sphere_collision_envelope"]
    final_summary = {
        "schema_version": "dex3_rigid_proxy_contact_diagnostic_final_v1",
        "status": "DEX3_CONTACT_MODEL_STILL_INVALID",
        "policy_evaluation_stopped": True,
        "policy_evaluation_reopened": False,
        "learned_policy_used": False,
        "policy_checkpoint_read": False,
        "real_robot": False,
        "source_scene_modified": False,
        "source_scene": str(scene),
        "source_scene_sha256": sha256_file(scene),
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "static_audit": str(STATIC),
        "static_audit_sha256": sha256_file(STATIC),
        "stage_matrix": str(table),
        "stage_matrix_sha256": sha256_file(table),
        "collision_filtering": {
            "static": static["usd"]["filter_conclusion"],
            "runtime_exact_named_pairs_observed": True,
            "tensor_contact_api_errors": [],
        },
        "collider_verification": {
            "six_distal_colliders_enabled": static["usd"]["all_distal_colliders_enabled"],
            "distal_visual_collision_frames_match": static["usd"][
                "all_distal_visual_collision_frames_match"
            ],
            "distal_collision_local_transforms_identity": static["usd"][
                "all_distal_collision_local_transforms_identity"
            ],
            "object_visual_and_collision_same_prim_at_runtime": True,
            "object_local_pose_scale_identity_for_sphere_and_ellipsoid": True,
            "capsule_local_orientation_is_declared_major_axis": True,
        },
        "offsets_and_drive": {
            "object_contact_offset_m": 0.002,
            "object_rest_offset_m": 0.0,
            "finger_contact_offset_authored": False,
            "finger_rest_offset_authored": False,
            "finger_effective_contact_offsets_m": offset_audit[
                "selected_fingertip_offsets"
            ]["effective_contact_offsets_m"],
            "finger_effective_rest_offsets_m": offset_audit[
                "selected_fingertip_offsets"
            ]["effective_rest_offsets_m"],
            "object_rigid_body_effective_contact_offsets_m": offset_audit["runtime_proxy"][
                "effective_contact_offsets_m"
            ],
            "object_rigid_body_effective_rest_offsets_m": offset_audit["runtime_proxy"][
                "effective_rest_offsets_m"
            ],
            "effective_readback_audit": offset_audit["result_path"],
            "effective_readback_audit_sha256": offset_audit["result_sha256"],
            "finger_drive_kp": 100.0,
            "finger_drive_kd": 4.0,
        },
        "friction": {
            "static": config["fixed_material"]["static_friction"],
            "dynamic": config["fixed_material"]["dynamic_friction"],
            "sweep_or_tuning_performed": False,
        },
        "sphere_75mm_geometry": {
            "left_open_initial_penetration_m": sphere_audit["left"]["OPEN_RESET"][
                "maximum_penetration_m"
            ],
            "right_open_initial_penetration_m": sphere_audit["right"]["OPEN_RESET"][
                "maximum_penetration_m"
            ],
            "left_grasp_penetration_m": sphere_audit["left"]["FROZEN_GRASP"][
                "maximum_penetration_m"
            ],
            "right_grasp_penetration_m": sphere_audit["right"]["FROZEN_GRASP"][
                "maximum_penetration_m"
            ],
            "full_hand_status": "REJECTED_BEFORE_EXECUTION_FOR_SPAWN_AND_GRASP_PENETRATION",
        },
        "lowest_level": {
            "single_no_gravity": {
                side: {
                    "status": results[f"sphere_{side}_no_gravity"]["status"],
                    "contact_duration_s": results[f"sphere_{side}_no_gravity"][
                        "contact_duration_s"
                    ],
                    "maximum_penetration_m": results[f"sphere_{side}_no_gravity"][
                        "maximum_penetration_m"
                    ],
                }
                for side in ("left", "right")
            },
            "single_push_gravity": {
                side: {
                    "status": results[f"sphere_{side}_push"]["status"],
                    "contact_duration_s": results[f"sphere_{side}_push"]["contact_duration_s"],
                    "horizontal_displacement_m": results[f"sphere_{side}_push"][
                        "object_horizontal_displacement_m"
                    ],
                    "maximum_penetration_m": results[f"sphere_{side}_push"][
                        "maximum_penetration_m"
                    ],
                }
                for side in ("left", "right")
            },
        },
        "bounded_proxy_results": {
            "ellipsoid_contact_fit_v2": {
                side: {
                    "close": results[f"ellipsoid_v2_{side}_close"]["status"],
                    "contact_roles": results[f"ellipsoid_v2_{side}_close"]["contact_roles"],
                }
                for side in ("left", "right")
            },
            "capsule_v1": {
                side: {
                    "close": results[f"capsule_v1_{side}_close"]["status"],
                    "static_hold": results[f"capsule_v1_{side}_hold"]["status"],
                    "hold_contact_duration_s": results[f"capsule_v1_{side}_hold"][
                        "static_hold_contact_duration_s"
                    ],
                }
                for side in ("left", "right")
            },
            "capsule_contact_fit_v2": {
                side: {
                    "close": results[f"capsule_v2_{side}_close"]["status"],
                    "static_hold": results[f"capsule_v2_{side}_hold"]["status"],
                    "hold_contact_duration_s": results[f"capsule_v2_{side}_hold"][
                        "static_hold_contact_duration_s"
                    ],
                    "lift": lift[side]["status"],
                }
                for side in ("left", "right")
            },
        },
        "selected_proxy": None,
        "bilateral_static_hold_pass": False,
        "bilateral_5cm_lift_pass": False,
        "environment_frozen": False,
        "act_ab_physics_evaluation_allowed": False,
        "decision_reason": (
            "The contact engine and named distal colliders produce stable, low-penetration contacts, "
            "but no bounded simple convex proxy passes bilateral gravity-on static retention. The "
            "5 cm lift gate is therefore ineligible and the object/contact environment cannot be frozen."
        ),
    }
    summary_path = OUTPUT / "FINAL_DEX3_CONTACT_DIAGNOSTIC.json"
    atomic_json(summary_path, final_summary)
    report = f"""# Dex3 Rigid-Proxy Contact Diagnostic

Status: **DEX3_CONTACT_MODEL_STILL_INVALID**

ACT-A/B physics evaluation remained stopped. No learned policy, checkpoint, real robot, DDS,
magnet, joint attachment, weld, or hidden constraint was used.

## Root cause and low-level contact

- Collision filtering is not the blocker: exact named distal-collider/proxy pairs were observed
  through the error-free PhysX tensor contact view.
- All six distal collision meshes are enabled, unit-scale, and vertex-identical to their visual
  meshes. The diagnostic proxy uses one prim for both visible and collision geometry.
- The historical 75 mm sphere is geometrically invalid for the frozen grasp. OPEN reset penetration
  is {1000*sphere_audit['left']['OPEN_RESET']['maximum_penetration_m']:.3f} mm left and
  {1000*sphere_audit['right']['OPEN_RESET']['maximum_penetration_m']:.3f} mm right; frozen-GRASP
  penetration is {1000*sphere_audit['left']['FROZEN_GRASP']['maximum_penetration_m']:.3f} mm left
  and {1000*sphere_audit['right']['FROZEN_GRASP']['maximum_penetration_m']:.3f} mm right.
- Isolated sphere fingertip contact passes without gravity for both hands (contact duration
  {results['sphere_left_no_gravity']['contact_duration_s']:.3f} s left,
  {results['sphere_right_no_gravity']['contact_duration_s']:.3f} s right).
- Gravity-on single-fingertip pushes also pass (contact duration
  {results['sphere_left_push']['contact_duration_s']:.3f} s left,
  {results['sphere_right_push']['contact_duration_s']:.3f} s right).

## Bounded proxy review

- The contact-fit ellipsoid fails bilateral three-finger closure because the index chain never
  contacts on either side.
- Capsule-v1 passes bilateral three-finger closure, but gravity-on retention lasts only
  {results['capsule_v1_left_hold']['static_hold_contact_duration_s']:.3f} s left and
  {results['capsule_v1_right_hold']['static_hold_contact_duration_s']:.3f} s right before escape.
- The final +0.5 mm contact-fit capsule also passes bilateral closure, but retention lasts only
  {results['capsule_v2_left_hold']['static_hold_contact_duration_s']:.3f} s left and
  {results['capsule_v2_right_hold']['static_hold_contact_duration_s']:.3f} s right before escape.
- The 5 cm lift was not run because the preceding static-hold gate failed on both sides.

Friction remained fixed (static {config['fixed_material']['static_friction']}, dynamic
{config['fixed_material']['dynamic_friction']}); no friction sweep was performed. Object contact/rest
offsets were 0.002/0.0 m. The selected distal collider's effective PhysX contact offset was
{offset_audit['selected_fingertip_offsets']['effective_contact_offsets_m'][0]:.9f} m and its rest
offset was {offset_audit['selected_fingertip_offsets']['effective_rest_offsets_m'][0]:.1f} m. Dex3
drive kp/kd were 100/4. Source USD files were not modified.

Detailed stage matrix: `{table}`

The object/contact environment is **not frozen**, and ACT-A/B physics task evaluation remains blocked.
"""
    report_path = OUTPUT / "FINAL_DEX3_CONTACT_DIAGNOSTIC_REPORT.md"
    atomic_text(report_path, report)
    manifest = {
        "schema_version": "dex3_contact_diagnostic_manifest_v1",
        "status": "DEX3_CONTACT_MODEL_STILL_INVALID",
        "final_summary": str(summary_path),
        "final_summary_sha256": sha256_file(summary_path),
        "final_report": str(report_path),
        "final_report_sha256": sha256_file(report_path),
        "stage_matrix": str(table),
        "stage_matrix_sha256": sha256_file(table),
        "effective_offset_readback_audit": offset_audit["result_path"],
        "effective_offset_readback_audit_sha256": offset_audit["result_sha256"],
        "environment_frozen": False,
        "policy_evaluation_reopened": False,
        "learned_policy_used": False,
        "real_robot": False,
    }
    atomic_json(OUTPUT / "DIAGNOSTIC_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
