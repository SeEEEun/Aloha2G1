#!/usr/bin/env python3
"""Re-evaluate v14 root candidates without repeating the physical FK sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path.insert(0, str(ROOT / "tools"))

import build_episode49_root_registered_v14 as v14


def main() -> int:
    out = v14.OUT
    fine_payload = json.loads((out / "root_sweep_fine.json").read_text())
    fine_rows = fine_payload["rows"]
    with np.load(v14.SOURCE, allow_pickle=False) as values:
        action = values["optimized_action"].copy()
        timestamps = values["timestamp"].copy()
    with np.load(v14.PHASE_LIBRARY, allow_pickle=False) as values:
        if not np.array_equal(values["optimized_action"], action):
            raise RuntimeError("immutable phase library mismatch")
        left_source_p = values["left_tcp_position"].copy()
        right_source_p = values["right_tcp_position"].copy()
        left_source_r = values["left_tcp_rotation"].copy()
        right_source_r = values["right_tcp_rotation"].copy()
    with np.load(v14.V12 / "globally_registered_base_targets.npz", allow_pickle=False) as values:
        base_model_l = values["base_aloha_derived_left_position_model"].copy()
        base_model_r = values["base_aloha_derived_right_position_model"].copy()
        old_global_r = values["global_registration_rotation"].copy()
        base_model_lr = np.einsum(
            "ji,tjk->tik", old_global_r, values["globally_registered_left_rotation"]
        )
        base_model_rr = np.einsum(
            "ji,tjk->tik", old_global_r, values["globally_registered_right_rotation"]
        )
        arm_names = values["arm_joint_names"].copy()
    source_progress = v14.v12.combined_source_progress(
        left_source_p, right_source_p, left_source_r, right_source_r
    )

    fidelity_by_root = {}
    for row in fine_rows:
        if not row["all_physical_gates_pass"]:
            continue
        fidelity = v14.fidelity_for_root(
            row, base_model_l, base_model_r, base_model_lr, base_model_rr, source_progress
        )
        row["fidelity"] = fidelity
        fidelity_by_root[f"{row['total_forward_offset_m']:.3f}"] = {
            key: value for key, value in fidelity.items()
            if key not in ("arrays", "base_l", "base_r", "base_lr", "base_rr")
        }
    v14.dump(out / "aloha_fidelity_by_root.json", fidelity_by_root)
    eligible = [
        row for row in fine_rows
        if row["all_physical_gates_pass"] and row["fidelity"]["hard_fidelity_gate_pass"]
    ]
    if not eligible:
        v14.dump(out / "root_selection_report.json", {
            "status": "BLOCKED_ALOHA_FIDELITY",
            "fidelity_by_root": fidelity_by_root,
            "authoritative_scene_modified": False,
        })
        print("BLOCKED_ALOHA_FIDELITY")
        return 3
    eligible.sort(key=lambda row: row["total_forward_offset_m"])
    minimum = eligible[0]
    minimum_margin = min(
        minimum["action169"]["physical_contact_margin_m"],
        minimum["action319"]["physical_contact_margin_m"],
        minimum["action523"]["physical_contact_margin_m"],
    )
    selected = minimum
    reason = "smallest total offset passing full physical, static-clearance, and ALOHA-fidelity gates"
    if minimum_margin < 0.002:
        desired = round(minimum["total_forward_offset_m"] + 0.005, 3)
        practical = next(
            (row for row in eligible if abs(row["total_forward_offset_m"] - desired) < 5e-7), None
        )
        if practical is not None:
            min_metrics = minimum["fidelity"]["candidates"][minimum["fidelity"]["selected_candidate"]]["minimum_major_phase_fidelity"]
            practical_metrics = practical["fidelity"]["candidates"][practical["fidelity"]["selected_candidate"]]["minimum_major_phase_fidelity"]
            # Correlation changes below 1e-4 are numerical/solver-level ties;
            # the practical root is accepted only if every hard gate remains.
            if practical_metrics["path_shape"] + 1e-4 >= min_metrics["path_shape"]:
                selected = practical
                reason = (
                    "minimum feasible +0.194 m root had <2 mm contact margin; "
                    "the required +5 mm practical candidate retained every fidelity/collision "
                    "gate with path correlation equal within 1e-4"
                )

    fidelity = selected["fidelity"]
    arrays = fidelity["arrays"]
    offset = float(selected["total_forward_offset_m"])
    root_xyz = np.asarray(selected["root_xyz_m"], float)
    v14.save_npz(
        out / "global_task_registration_candidate_v14.npz",
        global_rotation=fidelity["global_rotation"],
        global_translation=fidelity["global_translation_m"],
        base_left_position=fidelity["base_l"],
        base_right_position=fidelity["base_r"],
        base_left_rotation=fidelity["base_lr"],
        base_right_rotation=fidelity["base_rr"],
    )
    v14.save_npz(
        out / "phase_residual_candidate_v14.npz",
        residual_knots=v14.v12.KNOTS,
        left_knot_values=arrays["knots_l"],
        right_knot_values=arrays["knots_r"],
        left_translation_residual=arrays["residual_l"],
        right_translation_residual=arrays["residual_r"],
    )
    v14.save_npz(
        out / "corrected_targets_candidate_v14.npz",
        optimized_action=action,
        timestamps=timestamps,
        action_indices=np.arange(990),
        arm_joint_names=arm_names,
        base_left_position=fidelity["base_l"],
        base_right_position=fidelity["base_r"],
        corrected_left_position=arrays["corrected_l"],
        corrected_right_position=arrays["corrected_r"],
        corrected_left_rotation=fidelity["base_lr"],
        corrected_right_rotation=fidelity["base_rr"],
        global_registration_rotation=fidelity["global_rotation"],
        global_registration_translation=fidelity["global_translation_m"],
        left_translation_residual=arrays["residual_l"],
        right_translation_residual=arrays["residual_r"],
        selected_root_xyz=root_xyz,
        selected_total_forward_offset_m=np.array(offset),
        workspace_scale=np.array(v14.SCALE),
        method=np.array(v14.METHOD),
        diagnostic_only=np.array(True),
        real_robot_command_allowed=np.array(False),
    )
    selected_summary = {
        key: value for key, value in fidelity.items()
        if key not in ("arrays", "base_l", "base_r", "base_lr", "base_rr")
    }
    v14.dump(out / "root_selection_report.json", {
        "status": "ROOT_CANDIDATE_SELECTED_PENDING_AUTHORITATIVE_APPLY",
        "minimum_physical_and_fidelity_forward_offset_m": minimum["total_forward_offset_m"],
        "minimum_candidate_contact_margin_mm": 1000 * minimum_margin,
        "selected_total_forward_offset_m": offset,
        "selected_root_xyz_m": root_xyz,
        "verified_forward_direction": json.loads((out / "forward_direction_audit.json").read_text())["verified_forward_unit_vector"],
        "selection_reason": reason,
        "selected_physical_metrics": v14.stripped_row(selected),
        "selected_fidelity": selected_summary,
        "authoritative_scene_modified": False,
        "next_step_authorized": True,
    })
    v14.dump(out / "selected_physical_carrier_anchors_candidate.json", {
        "selected_total_forward_offset_m": offset,
        "root_xyz_m": root_xyz,
        "left_phone_action169": selected["action169"],
        "right_accessory_action319": selected["action319"],
        "left_charger_action523": selected["action523"],
        "phone_center_to_pad_m": 0.0,
        "phone_normal_error_deg": 0.0,
        "dex3_keyframe_values_diagnostic_only": True,
    })
    print(json.dumps({
        "status": "ROOT_CANDIDATE_SELECTED_PENDING_AUTHORITATIVE_APPLY",
        "minimum_offset_m": minimum["total_forward_offset_m"],
        "selected_offset_m": offset,
        "selected_root_xyz_m": root_xyz.tolist(),
        "reason": reason,
        "minimum_fidelity": selected_summary["candidates"][selected_summary["selected_candidate"]]["minimum_major_phase_fidelity"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
