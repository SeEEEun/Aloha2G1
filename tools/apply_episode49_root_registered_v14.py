#!/usr/bin/env python3
"""Apply the selected v14 G1 total-forward root registration exactly once."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from pxr import Usd

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_root_registered_v14 as v14
from robot_model_preview_common import compose_stage

OUT = v14.OUT
APPROVED = ROOT / "configs/g1_root_forward_v14.approved.json"
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(payload, indent=2, default=v14.default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def transform(stage: Usd.Stage, path: str) -> np.ndarray:
    return v14.v12.active_transform(stage, path)


def main() -> int:
    selection = json.loads((OUT / "root_selection_report.json").read_text())
    if selection["status"] != "ROOT_CANDIDATE_SELECTED_PENDING_AUTHORITATIVE_APPLY":
        raise RuntimeError("root selection is not eligible for apply")
    selected = float(selection["selected_total_forward_offset_m"])
    if not (0.15 < selected <= 0.30):
        raise RuntimeError("selected root is outside user-authorized range")
    physical = selection["selected_physical_metrics"]
    fidelity_name = selection["selected_fidelity"]["selected_candidate"]
    fidelity = selection["selected_fidelity"]["candidates"][fidelity_name]
    if not physical["all_physical_gates_pass"]:
        raise RuntimeError("physical gates did not pass")
    if not fidelity["hard_fidelity_gate_pass"]:
        raise RuntimeError("ALOHA fidelity gates did not pass")

    preview_before = json.loads(v14.PREVIEW_CONFIG.read_text())
    registration_before = json.loads(v14.REGISTRATION.read_text())
    original_root = np.asarray(preview_before["g1"]["position_xyz_m"], float)
    forward = np.asarray(
        json.loads((OUT / "forward_direction_audit.json").read_text())["verified_forward_unit_vector"],
        float,
    )
    expected_root = original_root + selected * forward
    selected_root = np.asarray(selection["selected_root_xyz_m"], float)
    if np.max(np.abs(expected_root - selected_root)) > 1e-12:
        raise RuntimeError("selection root does not equal original + total_offset * forward")

    stage_before = Usd.Stage.Open(str(v14.ACTIVE_SCENE))
    if stage_before is None:
        raise RuntimeError(v14.ACTIVE_SCENE)
    paths = {
        "G1": "/World/G1",
        "phone": "/World/MagSafeScene/Phone",
        "accessory": "/World/MagSafeScene/Accessory",
        "charger": "/World/MagSafeScene/Charger",
        "pad": "/World/MagSafeScene/Charger/Visuals/PadFace",
    }
    transforms_before = {name: transform(stage_before, path) for name, path in paths.items()}
    hashes_before = {
        str(path.resolve()): v14.sha(path)
        for path in (v14.LAYOUT, v14.FIXED_SCENE, v14.ACTIVE_SCENE, v14.PREVIEW_CONFIG, v14.REGISTRATION)
    }

    approval = {
        "schema_version": 1,
        "status": "USER_AUTHORIZED_FORWARD_ROOT_REGISTRATION",
        "simulation_only": True,
        "authoritative_for_real_robot": False,
        "previous_total_forward_offset_m": v14.OLD_OFFSET,
        "selected_total_forward_offset_m": selected,
        "minimum_physical_forward_offset_m": selection["minimum_physical_and_fidelity_forward_offset_m"],
        "original_root_xyz_m": original_root,
        "new_exact_root_xyz_m": expected_root,
        "verified_forward_unit_vector": forward,
        "selection_reason": selection["selection_reason"],
        "physical_contact_metrics": {
            "action169_A_gap_mm": 1000 * physical["action169"]["left_A_gap_m"],
            "action169_B_gap_mm": 1000 * physical["action169"]["left_B_gap_m"],
            "action319_C_gap_mm": 1000 * physical["action319"]["right_C_ring_gap_m"],
            "action523_A_gap_mm": 1000 * physical["action523"]["left_A_gap_m"],
            "action523_B_gap_mm": 1000 * physical["action523"]["left_B_gap_m"],
            "phone_center_to_pad_mm": 1000 * physical["phone_center_to_pad_m"],
            "phone_normal_error_deg": physical["phone_normal_error_deg"],
        },
        "aloha_fidelity_metrics": fidelity["minimum_major_phase_fidelity"],
        "workspace_scale": v14.SCALE,
        "source_hashes": json.loads((OUT / "input_hash_audit.json").read_text())["immutable_input_sha256"],
        "backup": str(v14.BACKUP.resolve()),
        "no_dex3_trajectory": True,
        "no_physics": True,
        "no_hardware": True,
    }
    atomic_json(APPROVED, approval)

    # Preserve the original root pose.  Only attach explicit registration
    # metadata; compose_stage applies the approved total offset once.
    preview_after = dict(preview_before)
    preview_after["g1_root_registration"] = {
        "status": "USER_AUTHORIZED_FORWARD_ROOT_REGISTRATION",
        "total_forward_offset_m": selected,
        "original_root_xyz_m": original_root.tolist(),
        "applied_root_xyz_m": expected_root.tolist(),
        "forward_unit_vector": forward.tolist(),
        "approved_config": str(APPROVED.resolve()),
        "applied_exactly_once": True,
    }
    atomic_json(v14.PREVIEW_CONFIG, preview_after)

    registration_after = dict(registration_before)
    registration_after["registration_method"] = (
        f"current scene_layout geometry plus user-authorized final total +{selected:.3f} m "
        "G1 preview forward offset; simulation only"
    )
    registration_after["status"] = "USER_AUTHORIZED_FORWARD_ROOT_REGISTRATION"
    registration_after["evidence_sources"] = list(dict.fromkeys(
        list(registration_after.get("evidence_sources", [])) + [str(APPROVED.resolve())]
    ))
    registration_after["manual_adjustment_log"] = [{
        "parameter": "g1_root_forward_offset_m",
        "value_m": selected,
        "basis": selection["selection_reason"],
        "original_root_position_m": original_root.tolist(),
        "applied_root_position_m": expected_root.tolist(),
        "application_semantics": "total offset from original root; applied exactly once",
    }]
    t_scene_from_g1 = np.asarray(registration_after["T_scene_from_g1_base"], float)
    t_scene_from_g1[:3, 3] = expected_root
    registration_after["T_scene_from_g1_base"] = t_scene_from_g1.tolist()
    t_task_from_scene = np.asarray(registration_after["T_task_from_scene"], float)
    registration_after["T_task_from_g1_base"] = (t_task_from_scene @ t_scene_from_g1).tolist()
    for constraint in registration_after.get("constraints", []):
        if constraint.get("name") == "g1_root_forward_offset":
            constraint["value_m"] = selected
            constraint["source"] = str(APPROVED.resolve())
    atomic_json(v14.REGISTRATION, registration_after)

    # The pre-apply stage is open for immutable-object comparison, so USD will
    # not CreateNew the same identifier in this process.  Compose a sibling
    # layer and atomically replace the active layer; referenced assets remain
    # untouched.
    temporary_stage = v14.ACTIVE_SCENE.with_name(
        v14.ACTIVE_SCENE.stem + ".v14.incomplete" + v14.ACTIVE_SCENE.suffix
    )
    if temporary_stage.exists():
        temporary_stage.unlink()
    compose_stage(
        temporary_stage, "G1", G1_USD, "g1", forward_offset_m=selected,
    )
    os.replace(temporary_stage, v14.ACTIVE_SCENE)
    # Refresh the already-open USD identifier; otherwise the in-process layer
    # registry can return the pre-replacement +0.15 layer even though the file
    # on disk is the correct +0.199 composition.
    stage_before.Reload()
    stage_after = stage_before
    if stage_after is None:
        raise RuntimeError("failed to reopen composed v14 stage")
    transforms_after = {name: transform(stage_after, path) for name, path in paths.items()}
    root_error = float(np.max(np.abs(transforms_after["G1"][:3, 3] - expected_root)))
    object_errors = {
        name: float(np.max(np.abs(transforms_after[name] - transforms_before[name])))
        for name in ("phone", "accessory", "charger", "pad")
    }
    if root_error > 1e-6:
        raise RuntimeError(f"root apply mismatch: {root_error}")
    if max(object_errors.values()) > 1e-12:
        raise RuntimeError(f"immutable object moved: {object_errors}")
    if v14.sha(v14.LAYOUT) != hashes_before[str(v14.LAYOUT.resolve())]:
        raise RuntimeError("scene_layout changed")
    if v14.sha(v14.FIXED_SCENE) != hashes_before[str(v14.FIXED_SCENE.resolve())]:
        raise RuntimeError("fixed scene changed")

    hashes_after = {
        str(path.resolve()): v14.sha(path)
        for path in (v14.LAYOUT, v14.FIXED_SCENE, v14.ACTIVE_SCENE, v14.PREVIEW_CONFIG, v14.REGISTRATION, APPROVED)
    }
    snapshot_after = {
        "status": "ROOT_FORWARD_REGISTRATION_UPDATED",
        "selected_total_forward_offset_m": selected,
        "expected_root_xyz_m": expected_root,
        "active_composed_root_xyz_m": transforms_after["G1"][:3, 3],
        "root_max_error_m": root_error,
        "root_applied_exactly_once": True,
        "active_object_transform_max_errors_m": object_errors,
        "objects_unchanged": max(object_errors.values()) <= 1e-12,
        "layout_hash_unchanged": v14.sha(v14.LAYOUT) == hashes_before[str(v14.LAYOUT.resolve())],
        "fixed_scene_hash_unchanged": v14.sha(v14.FIXED_SCENE) == hashes_before[str(v14.FIXED_SCENE.resolve())],
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "preview_original_root_retained": preview_after["g1"]["position_xyz_m"] == preview_before["g1"]["position_xyz_m"],
    }
    v14.dump(OUT / "configs_snapshot_after.json", snapshot_after)

    # Promote the selected, still arm-only target artifacts to their required
    # v14 names.  The physical anchors will be recomputed once more from the
    # now-active composed root before temporal IK.
    shutil.copy2(OUT / "phase_residual_candidate_v14.npz", OUT / "phase_residual_v14.npz")
    shutil.copy2(OUT / "corrected_targets_candidate_v14.npz", OUT / "corrected_targets_v14.npz")
    selected_anchors = json.loads((OUT / "selected_physical_carrier_anchors_candidate.json").read_text())
    selected_anchors["active_root_verified_after_apply"] = True
    selected_anchors["active_root_max_error_m"] = root_error
    v14.dump(OUT / "selected_physical_carrier_anchors.json", selected_anchors)
    v14.dump(OUT / "global_task_registration_v14.json", {
        "method": "one fixed common rotation and translation after immutable workspace scale 0.42",
        "global_rotation": selection["selected_fidelity"]["global_rotation"],
        "global_translation_m": selection["selected_fidelity"]["global_translation_m"],
        "workspace_scale": v14.SCALE,
        "same_transform_both_hands_all_990_samples": True,
        "selected_root_total_forward_offset_m": selected,
        "selected_root_xyz_m": expected_root,
        "global_similarity_excluded_from_motion_deformation": True,
    })
    print(json.dumps(snapshot_after, indent=2, default=v14.default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
