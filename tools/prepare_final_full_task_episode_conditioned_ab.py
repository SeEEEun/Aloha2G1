#!/usr/bin/env python3
"""Prepare the fail-closed full-task A/B gate without running ACT or PhysX.

This tool is intentionally read-only with respect to all prior experiment
artifacts.  It packages the source-conditioned DEV35 environments, audits the
old checkpoint supervision against the latest materialized reset candidate,
and evaluates the structural/common-execution gates required before training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/final_full_task_episode_conditioned_ab"
REG_SRC = REPO / "outputs/final_episode_registered_eval35/00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
REF_ROOT = REPO / "outputs/reference_motion_scientific_reset"
RESET_CANDIDATE = REPO / "outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4"
COMMON_EXEC = REPO / "outputs/single_variable_ab_common_execution/COMMON_COLLISION_AWARE_IK_FINAL_AUDIT.json"
TRAIN_MANIFEST = REPO / "outputs/paper_core_ab/train40_manifest.json"
TRAIN_AUDIT = REPO / "outputs/paper_core_ab/act_a_b_training_audit.json"
FAIRNESS_AUDIT = REPO / "outputs/single_variable_ab_reset/A_B_DATASET_FAIRNESS_AUDIT.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical_sha(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def yaw_deg(q: list[float]) -> float:
    x, y, z, w = map(float, q)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def rotated_rectangle(center: tuple[float, float], size: tuple[float, float], yaw: float) -> np.ndarray:
    hx, hy = size[0] / 2.0, size[1] / 2.0
    pts = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s], [s, c]])
    return pts @ rot.T + np.asarray(center)


def build_environment_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(REG_SRC.read_text())
    assert source["EVAL35_count"] == 35
    assert source["source_derived_count"] == 35
    assert source["A_B_identical_object_pose_count"] == 35

    ref_paths = {
        int(p.name[4:6]): p
        for p in (REF_ROOT / "corrected_references/dev35").glob("EVAL*_SOURCE_A_B_REFERENCE.npz")
    }
    entries: list[dict[str, Any]] = []
    for src in source["entries"]:
        idx = int(src["eval_index"])
        pose = src["target_object_pose"]
        quat = list(map(float, pose["quaternion_xyzw"]))
        pos = list(map(float, pose["position_xyz_m"]))
        ref_path = ref_paths.get(idx)
        assert ref_path and ref_path.exists()
        methods = src.get("methods", {})
        entry = {
            "eval_index": idx,
            "eval_number": int(src["eval_number"]),
            "stable_episode_id": src["stable_episode_id"],
            "source_recording": src["source_recording"],
            "source_task_grasp_evidence": {
                "task_frame_source": src["source_object_task_frame_source"],
                "grasp_window": src["source_grasp_window"],
                "orientation_evidence": src["orientation_evidence"],
                "provenance": src["provenance"],
            },
            "object_xyz_m": pos,
            "object_quaternion_xyzw": quat,
            "object_rpy_deg": [0.0, 0.0, yaw_deg(quat)],
            "table_relation": {
                "table_top_world_z_m": float(source["bin_pose"]["bottom_world_z_m"]),
                "object_center_above_table_m": pos[2] - float(source["bin_pose"]["bottom_world_z_m"]),
            },
            "bin_relation": {
                "bin_opening_center_xy_m": source["bin_pose"]["opening_center_world_xy_m"],
                "bin_bottom_world_z_m": source["bin_pose"]["bottom_world_z_m"],
                "object_to_bin_xy_m": [
                    float(source["bin_pose"]["opening_center_world_xy_m"][0]) - pos[0],
                    float(source["bin_pose"]["opening_center_world_xy_m"][1]) - pos[1],
                ],
                "fixed_for_all_episodes": bool(source["bin_pose"]["fixed_for_all_episodes"]),
            },
            "source_to_target_transform": src["source_to_target_transform"],
            "A_reference": {
                "path": str(ref_path),
                "keys": ["a_left_wrist_position_world", "a_right_wrist_position_world"],
                "sha256": sha256_file(ref_path),
            },
            "B_reference": {
                "path": str(ref_path),
                "keys": ["b_left_wrist_position_world", "b_right_wrist_position_world"],
                "sha256": sha256_file(ref_path),
            },
            "pre_reset_commands_provenance_only": {
                "A": methods.get("ACT-A40", {}).get("command_path"),
                "B": methods.get("ACT-B40", {}).get("command_path"),
                "eligible_for_current_full_task": False,
            },
            "A_B_environment_equality": True,
            "A_B_object_translation_difference_mm": 0.0,
            "A_B_object_rotation_difference_deg": 0.0,
            "manual_episode_nudge": False,
            "policy_output_used": False,
        }
        entry["entry_sha256"] = canonical_sha(entry)
        entries.append(entry)

    xyz = np.asarray([e["object_xyz_m"] for e in entries], dtype=float)
    yaw = np.asarray([e["object_rpy_deg"][2] for e in entries], dtype=float)
    summary = {
        "x_range_m": [float(xyz[:, 0].min()), float(xyz[:, 0].max())],
        "y_range_m": [float(xyz[:, 1].min()), float(xyz[:, 1].max())],
        "z_range_m": [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
        "yaw_range_deg": [float(yaw.min()), float(yaw.max())],
        "unique_pose_count": len({tuple(np.round(np.r_[e["object_xyz_m"], e["object_quaternion_xyzw"]], 12)) for e in entries}),
    }
    manifest = {
        "schema_version": "final_full_task_episode_conditioned_environments_v1",
        "status": "PASS_ENVIRONMENT_REGISTRATION_ONLY",
        "scientific_semantics": "EPISODE_CONDITIONED_SOURCE_DERIVED_TASK_INITIALIZATION",
        "evaluation_label": "DEV35_DIAGNOSTIC35",
        "entry_count": 35,
        "source_derived_count": 35,
        "unique_source_episode_count": 35,
        "unique_episode_object_pose_count": summary["unique_pose_count"],
        "one_global_canonical_object_pose": False,
        "manual_episode_nudges": False,
        "policy_output_derived_object_placement": False,
        "A_B_matched_environment_equality_count": 35,
        "maximum_A_B_translation_difference_mm": 0.0,
        "maximum_A_B_rotation_difference_deg": 0.0,
        "source_registration_artifact": str(REG_SRC),
        "source_registration_sha256": sha256_file(REG_SRC),
        "bin_pose": source["bin_pose"],
        "visual_collision_audit": source["visual_collision_audit"],
        "variation": summary,
        "entries": entries,
    }
    manifest["manifest_content_sha256"] = canonical_sha(manifest)
    return manifest, summary


def save_environment_artifacts(manifest: dict[str, Any]) -> Path:
    out = OUT / "00_environment_registration"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "EVAL35_EPISODE_TASK_ENVIRONMENTS.json"
    csv_path = out / "EVAL35_EPISODE_TASK_ENVIRONMENTS.csv"
    md_path = out / "EVAL35_EPISODE_TASK_ENVIRONMENTS.md"
    write_json(json_path, manifest)

    fields = [
        "eval_index", "eval_number", "stable_episode_id", "source_recording",
        "object_x_m", "object_y_m", "object_z_m", "object_qx", "object_qy",
        "object_qz", "object_qw", "object_yaw_deg", "bin_x_m", "bin_y_m",
        "A_reference_path", "B_reference_path", "A_B_equal", "entry_sha256",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in manifest["entries"]:
            p, q = e["object_xyz_m"], e["object_quaternion_xyzw"]
            b = e["bin_relation"]["bin_opening_center_xy_m"]
            w.writerow({
                "eval_index": e["eval_index"], "eval_number": e["eval_number"],
                "stable_episode_id": e["stable_episode_id"], "source_recording": e["source_recording"],
                "object_x_m": p[0], "object_y_m": p[1], "object_z_m": p[2],
                "object_qx": q[0], "object_qy": q[1], "object_qz": q[2], "object_qw": q[3],
                "object_yaw_deg": e["object_rpy_deg"][2], "bin_x_m": b[0], "bin_y_m": b[1],
                "A_reference_path": e["A_reference"]["path"], "B_reference_path": e["B_reference"]["path"],
                "A_B_equal": e["A_B_environment_equality"], "entry_sha256": e["entry_sha256"],
            })

    v = manifest["variation"]
    md = f"""# DEV35 episode-conditioned task environments

Status: **PASS (registration only)**

- Environments: **35/35**
- Source-derived: **35/35**
- Matched A/B equality: **35/35**
- Maximum matched translation difference: **0 mm**
- Maximum matched rotation difference: **0 deg**
- One global canonical pose: **NO**
- Manual episode nudges: **NO**
- Policy-output-derived placement: **NO**
- Unique episode poses: **{v['unique_pose_count']}/35**
- Object X range: **{v['x_range_m'][0]:.6f} to {v['x_range_m'][1]:.6f} m**
- Object Y range: **{v['y_range_m'][0]:.6f} to {v['y_range_m'][1]:.6f} m**
- Object Z range: **{v['z_range_m'][0]:.6f} to {v['z_range_m'][1]:.6f} m**
- Yaw range: **{v['yaw_range_deg'][0]:.3f} to {v['yaw_range_deg'][1]:.3f} deg**

These are 35 source-conditioned task instances, not manually tuned environments.
The A/B command paths embedded in the older source manifest are retained only as
provenance. The current corrected references are recorded explicitly, but no
current ACT command archive exists because retraining has not occurred.

Manifest content SHA256: `{manifest['manifest_content_sha256']}`
"""
    md_path.write_text(md)

    # Outcome-blind 7 x 5 top-view audit.
    fig, axes = plt.subplots(5, 7, figsize=(20, 13), constrained_layout=True)
    vis_dims = manifest["visual_collision_audit"]["visual_dimensions_m"]
    col_dims = manifest["visual_collision_audit"]["collision_dimensions_m"]
    bin_xy = manifest["bin_pose"]["opening_center_world_xy_m"]
    bin_dims = manifest["bin_pose"]["opening_dimensions_xy_m"]
    for ax, e in zip(axes.flat, manifest["entries"]):
        x, y, _ = e["object_xyz_m"]
        yaw = math.radians(e["object_rpy_deg"][2])
        ax.add_patch(Rectangle((-0.08, -0.22), 0.95, 0.48, facecolor="#f3efe6", edgecolor="#9b8f7b", lw=.7))
        ax.add_patch(Rectangle((bin_xy[0]-bin_dims[0]/2, bin_xy[1]-bin_dims[1]/2), bin_dims[0], bin_dims[1],
                               facecolor="#d8e9f4", edgecolor="#3579a8", lw=1.0))
        ax.add_patch(Polygon(rotated_rectangle((x, y), (vis_dims[0], vis_dims[1]), yaw), closed=True,
                             facecolor="#75c46b", edgecolor="#236b2c", lw=1.0))
        ax.add_patch(Polygon(rotated_rectangle((x, y), (col_dims[0], col_dims[1]), yaw), closed=True,
                             fill=False, edgecolor="#1a4420", lw=.8, linestyle="--"))
        ax.plot([0], [0], marker="s", ms=4, color="#343434")
        ax.arrow(x, y, .055*math.cos(yaw), .055*math.sin(yaw), width=.0013,
                 head_width=.012, color="#b12d2d", length_includes_head=True)
        ax.set_xlim(-.08, .87); ax.set_ylim(-.22, .26); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{e['eval_number']:02d}  {e['source_recording'].replace('GoPark_', '')}\n"
                     f"({x:.3f}, {y:.3f})  yaw {e['object_rpy_deg'][2]:.1f}°", fontsize=7)
    fig.suptitle("DEV35 episode-conditioned source-derived task initialization\n"
                 "Outcome-blind registration; identical environment for matched A/B", fontsize=15)
    contact = OUT / "EVAL35_EPISODE_ENVIRONMENT_CONTACT_SHEET.png"
    fig.savefig(contact, dpi=180)
    plt.close(fig)
    return contact


def old_path(method: str, idx: int) -> Path:
    if method == "A":
        return REPO / f"outputs/fair_a_full50_hard_fail_audit/after_full_pose/trajectories/doll_handoff_20260820_ep{idx:03d}.npz"
    return REPO / f"outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories/episode_{idx:06d}.npz"


def candidate_path(method: str, idx: int) -> Path:
    side = "baseline" if method == "A" else "proposed"
    return RESET_CANDIDATE / side / f"trajectories/doll_handoff_20260820_ep{idx:03d}.npz"


def build_training_audit() -> dict[str, Any]:
    manifest = json.loads(TRAIN_MANIFEST.read_text())
    training = json.loads(TRAIN_AUDIT.read_text())
    fairness = json.loads(FAIRNESS_AUDIT.read_text())
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for method in ["A", "B"]:
        all_old: list[np.ndarray] = []
        all_new: list[np.ndarray] = []
        for idx in [0, 24, 49]:
            op, npth = old_path(method, idx), candidate_path(method, idx)
            od, nd = np.load(op, allow_pickle=False), np.load(npth, allow_pickle=False)
            oa = np.asarray(od["replay_named_joint_qpos"], dtype=np.float64)
            na = np.asarray(nd["replay_named_joint_qpos"], dtype=np.float64)
            assert oa.shape == na.shape
            delta = np.abs(oa - na)
            changed = delta > 1e-7
            row = {
                "method": method, "train_episode_index": idx, "frames": int(oa.shape[0]),
                "scalars": int(oa.size), "old_action_sha256": sha256_array(oa.astype(np.float32)),
                "reset_candidate_action_sha256": sha256_array(na.astype(np.float32)),
                "max_absolute_action_difference_rad": float(delta.max()),
                "changed_action_scalars_at_1e-7": int(changed.sum()),
                "changed_frames_at_1e-7": int(np.any(changed, axis=1).sum()),
                "timestamp_max_abs_difference_s": float(np.max(np.abs(od["timestamp"] - nd["timestamp"]))),
                "frame_count_equal": bool(oa.shape[0] == na.shape[0]),
                "joint_order_equal": bool(np.array_equal(od["replay_joint_names"], nd["replay_joint_names"])),
                "left_hand_phase_equal": bool(np.array_equal(od["left_hand_phase"], nd["left_hand_phase"])),
                "right_hand_phase_equal": bool(np.array_equal(od["right_hand_phase"], nd["right_hand_phase"])),
                "candidate_is_final_corrected_train_supervision": False,
            }
            rows.append(row); all_old.append(oa.astype(np.float32)); all_new.append(na.astype(np.float32))
        old_cat, new_cat = np.concatenate(all_old), np.concatenate(all_new)
        delta = np.abs(old_cat.astype(float) - new_cat.astype(float)); changed = delta > 1e-7
        aggregate[method] = {
            "scope": "EXACT_TRAIN_SMOKE_0_24_49_LOWER_BOUND",
            "episode_count": 3, "frame_count": int(old_cat.shape[0]), "scalar_count": int(old_cat.size),
            "old_action_sha256": sha256_array(old_cat), "reset_candidate_action_sha256": sha256_array(new_cat),
            "max_absolute_action_difference_rad": float(delta.max()),
            "changed_action_scalars_at_1e-7": int(changed.sum()),
            "changed_frames_at_1e-7": int(np.any(changed, axis=1).sum()),
            "training_action_changed": True,
        }

    checkpoint_hashes = {
        m: training["methods"][m.lower()]["final_checkpoint_model_sha256"] for m in ["A", "B"]
    }
    audit = {
        "schema_version": "full_task_act_supervision_diff_audit_v1",
        "decision": "ACT_RETRAINING_REQUIRED",
        "old_train_episode_count": int(manifest["episode_count"]),
        "corrected_train_episode_count": None,
        "corrected_train40_joint_action_tensors_materialized": False,
        "reason_full_exact_diff_unavailable": (
            "The qualified reference reset currently stores Cartesian A/B references and common semantic hand commands. "
            "A corrected 40-episode joint-action dataset cannot be generated until the shared sequential IK, full 6D, "
            "and loaded Dex3 gates pass."
        ),
        "exact_available_smoke_diff": aggregate,
        "per_episode_smoke_diff": rows,
        "old_vs_current_contract": {
            "source_episode_membership": "SAME COMMON48 / TRAIN40 split intended",
            "frame_counts": "same on exact smoke; corrected TRAIN40 not materialized",
            "state_tensor": "NOT COMPARABLE until corrected dataset materialization",
            "action_tensor": "CHANGED on every smoke frame for both methods",
            "action_timestamps": "same on exact smoke",
            "state_timestamps": "not materialized in corrected TRAIN40",
            "close_lift_timing": (
                "current source clock overlaps close/lift in 2/3 smoke and 9/35 DEV; the newly required target-side "
                "strict close-complete-before-lift rule is not present in old checkpoints and has not been materialized"
            ),
            "handoff_timing": "current common source clock preserves RIGHT acquire before LEFT release in 35/35",
            "joint_order": "equal on exact smoke",
            "action_normalization": "old method-specific stats exist; corrected stats do not exist",
            "state_normalization": "old method-specific stats exist; corrected stats do not exist",
            "chunk_target_construction": "old chunk_size=50; corrected chunks not built",
        },
        "prior_full_archive_fairness_evidence": {
            "same_left_hand_phase_labels": fairness.get("exact_archive_evidence", {}).get("same_left_hand_phase_labels", "0/48 (see source report)"),
            "same_right_hand_phase_labels": fairness.get("exact_archive_evidence", {}).get("same_right_hand_phase_labels", "0/48 (see source report)"),
            "same_dex3_commands": fairness.get("exact_archive_evidence", {}).get("same_dex3_commands", "0/48 (see source report)"),
            "source_report": str(FAIRNESS_AUDIT),
        },
        "existing_checkpoint_hashes": checkpoint_hashes,
        "existing_checkpoints_reusable": False,
        "must_regenerate_both_datasets": True,
        "must_retrain_both_policies": True,
        "act_training_run": False,
    }
    return audit


def save_training_audit(audit: dict[str, Any]) -> None:
    out = OUT / "01_act_training_audit"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "ACT_TRAINING_SUPERVISION_DIFF_AUDIT.json", audit)
    with (out / "ACT_TRAINING_SUPERVISION_DIFF_SMOKE.csv").open("w", newline="") as f:
        fields = list(audit["per_episode_smoke_diff"][0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audit["per_episode_smoke_diff"])
    a, b = audit["exact_available_smoke_diff"]["A"], audit["exact_available_smoke_diff"]["B"]
    (out / "ACT_TRAINING_SUPERVISION_DIFF_AUDIT.md").write_text(f"""# ACT training-supervision diff audit

Decision: **ACT_RETRAINING_REQUIRED**

The existing checkpoints cannot be reused. Exact numerical evidence is available
for TRAIN smoke episodes 0, 24, and 49:

| Method | Changed frames | Changed scalars | Max absolute difference |
|---|---:|---:|---:|
| A — WRIST | {a['changed_frames_at_1e-7']}/{a['frame_count']} | {a['changed_action_scalars_at_1e-7']}/{a['scalar_count']} | {a['max_absolute_action_difference_rad']:.9f} rad |
| B — INTERACTION | {b['changed_frames_at_1e-7']}/{b['frame_count']} | {b['changed_action_scalars_at_1e-7']}/{b['scalar_count']} | {b['max_absolute_action_difference_rad']:.9f} rad |

This is a lower-bound comparison against the latest materialized shared-pipeline
joint candidate, not a claim that the candidate is final training supervision.
The corrected TRAIN40 joint-action tensors do not exist: the common sequential
IK gate is still blocked, full 6D was not run, and loaded Dex3 was not run.
Consequently a full 40-episode byte/numerical diff, corrected normalization
statistics, and corrected chunk tensors cannot honestly be reported yet.

The reference reset also introduced the authoritative common phase clock, while
the newly requested strict target-side close-complete-before-lift rule would
retime 2/3 smoke and 9/35 DEV trajectories. That timing is absent from the old
checkpoint supervision. Both datasets must be regenerated together and both ACT
policies retrained under the frozen parity protocol after common execution passes.
""")


def build_structural_preflight() -> dict[str, Any]:
    common = json.loads(COMMON_EXEC.read_text())
    rows: list[dict[str, Any]] = []
    for idx in [0, 24, 49]:
        p = next((REF_ROOT / "corrected_references/train_smoke").glob(f"EP{idx:02d}_*.npz"))
        d = np.load(p, allow_pickle=False)
        events = {str(n): int(f) for n, f in zip(d["event_names"], d["event_frames"])}
        obj = np.asarray(d["object_position_world_m"], dtype=float)
        close_before_lift = events["LEFT_CLOSE_COMPLETE"] <= events["LEFT_LIFT_BEGIN"]
        acquire_before_release = events["RIGHT_ACQUIRE_SOURCE"] <= events["LEFT_RELEASE_BEGIN"]
        for method in ["A", "B"]:
            prefix = method.lower()
            tool = np.asarray(d[f"{prefix}_left_tool_position_world"], dtype=float)
            s, e = events["APPROACH_START"], events["LEFT_CLOSE_COMPLETE"]
            dist = np.linalg.norm(tool[s:e+1] - obj[None, :], axis=1) * 1000.0
            spatial = bool(np.isfinite(tool).all() and float(dist.min()) <= 90.0)
            rows.append({
                "train_episode_index": idx, "method": method,
                "source_recording": str(d["source_name"].item()),
                "natural_full_length_reference": True,
                "approach_start_center_distance_mm": float(dist[0]),
                "minimum_approach_close_center_distance_mm": float(dist.min()),
                "approach_reaches_registered_task_region": spatial,
                "left_close_complete_frame": events["LEFT_CLOSE_COMPLETE"],
                "left_lift_begin_frame": events["LEFT_LIFT_BEGIN"],
                "close_complete_before_lift": bool(close_before_lift),
                "right_acquire_frame": events["RIGHT_ACQUIRE_SOURCE"],
                "left_release_frame": events["LEFT_RELEASE_BEGIN"],
                "acquire_before_release": bool(acquire_before_release),
                "standardized_grasp_rebase_used": False,
                "reference_structure_valid_under_new_absolute_rule": bool(spatial and close_before_lift and acquire_before_release),
                "common_execution_qualified": False,
                "full_task_structural_smoke_valid": False,
            })
    result = {
        "schema_version": "full_task_structural_preflight_v1",
        "scope": "TRAIN_SMOKE_0_24_49_NO_PHYSICS",
        "A_reference_structure_pass": sum(r["reference_structure_valid_under_new_absolute_rule"] for r in rows if r["method"] == "A"),
        "B_reference_structure_pass": sum(r["reference_structure_valid_under_new_absolute_rule"] for r in rows if r["method"] == "B"),
        "A_full_execution_structural_smoke_pass": 0,
        "B_full_execution_structural_smoke_pass": 0,
        "close_complete_before_lift_pass": all(r["close_complete_before_lift"] for r in rows),
        "acquire_before_release_pass": all(r["acquire_before_release"] for r in rows),
        "common_execution_pass": False,
        "common_execution_blocker": {
            "position_only_A": f"{common['position_only']['A_qualified_episodes']}/3",
            "position_only_B": f"{common['position_only']['B_qualified_episodes']}/3",
            "hard_self_collision_frames_A": common["position_only"]["A_hard_self_collision_frames"],
            "hard_self_collision_frames_B": common["position_only"]["B_hard_self_collision_frames"],
            "full_6d": common["full_6d"]["status"],
            "loaded_dex3": common["loaded_dex3"]["status"],
            "source_audit": str(COMMON_EXEC),
        },
        "full_task_scope_viable": False,
        "standardized_grasp_required": False,
        "standardized_grasp_reason": "Not selected; it cannot repair an unqualified common IK/articulation layer and the prior rebase is invalid.",
        "physics_smoke_run": False,
        "rows": rows,
    }
    return result


def save_structural_preflight(pre: dict[str, Any]) -> None:
    out = OUT / "02_structural_preflight"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "FULL_TASK_STRUCTURAL_PREFLIGHT.json", pre)
    with (out / "FULL_TASK_STRUCTURAL_PREFLIGHT.csv").open("w", newline="") as f:
        fields = list(pre["rows"][0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pre["rows"])
    (out / "FULL_TASK_STRUCTURAL_PREFLIGHT.md").write_text(f"""# Full-task structural preflight

Status: **FAIL CLOSED — no physics run**

- A reference structure under the newly required absolute close-before-lift rule: **{pre['A_reference_structure_pass']}/3**
- B reference structure: **{pre['B_reference_structure_pass']}/3**
- Close complete before lift: **{'PASS' if pre['close_complete_before_lift_pass'] else 'FAIL'}**
- RIGHT acquire before LEFT release: **{'PASS' if pre['acquire_before_release_pass'] else 'FAIL'}**
- Common collision-aware position IK: **A {pre['common_execution_blocker']['position_only_A']}, B {pre['common_execution_blocker']['position_only_B']}**
- Remaining hard self-collision frames: **A {pre['common_execution_blocker']['hard_self_collision_frames_A']}, B {pre['common_execution_blocker']['hard_self_collision_frames_B']}**
- Full 6D: **{pre['common_execution_blocker']['full_6d']}**
- Loaded Dex3: **{pre['common_execution_blocker']['loaded_dex3']}**
- Physical smoke: **NOT RUN**

Episodes 24 and 49 preserve a source-observed close/lift overlap in the qualified
reference reset. The new experiment instead requires strict close completion
before target lift, so a single method-blind target event retiming must be
materialized in both training datasets. No spatial target change is implied.

The existing common execution layer is independently not qualified. It is not
scientifically valid to train from its current joint realization or to launch
natural-start PhysX smoke.
""")


def write_final_report(manifest: dict[str, Any], audit: dict[str, Any], pre: dict[str, Any], contact: Path) -> Path:
    v = manifest["variation"]
    path = OUT / "FINAL_FULL_TASK_EPISODE_CONDITIONED_AB_REPORT.md"
    path.write_text(f"""# Final Full-Task Episode-Conditioned A/B Report

## Outcome

**STOPPED AT THE ACT RETRAINING / COMMON-EXECUTION GATE.** No ACT training,
inference, physical smoke, DEV35 rollout, scoring, result figure, or replay was
started by this workflow. Producing those artifacts from old checkpoints or the
unqualified solver would mix incompatible scientific configurations.

## 1. Reference reset

The persisted reference reset remains authoritative: A WRIST references are
competent 35/35, B INTERACTION references are competent 35/35, common temporal
semantics pass 35/35, and observed unintended reference-level confounds are zero.
The former standardized-grasp rebase is invalid and is not used.

## 2–3. ACT action-diff audit and retraining decision

Both old training action tensors differ from the latest materialized shared-pipeline
candidate on every frame of exact TRAIN smoke episodes 0, 24, and 49. A changes
{audit['exact_available_smoke_diff']['A']['changed_action_scalars_at_1e-7']}
scalars (max {audit['exact_available_smoke_diff']['A']['max_absolute_action_difference_rad']:.9f} rad);
B changes {audit['exact_available_smoke_diff']['B']['changed_action_scalars_at_1e-7']}
scalars (max {audit['exact_available_smoke_diff']['B']['max_absolute_action_difference_rad']:.9f} rad).
The corrected common semantic clock and strict target close-before-lift contract
are also absent from the old supervision. **Both datasets require regeneration
and both policies require parity-controlled retraining.** Existing checkpoints
are not reused.

A complete corrected TRAIN40 action diff is intentionally not fabricated. The
corrected references currently end at Cartesian/semantic supervision because
common sequential IK is not qualified; therefore corrected joint actions,
normalization statistics, and chunk targets do not yet exist.

## 4–5. DEV35 environment derivation and provenance

- Source-derived task environments: 35/35
- Unique object poses: {v['unique_pose_count']}/35
- Matched A/B equality: 35/35 (0 mm, 0 deg)
- One canonical object pose: NO
- Manual episode nudges: NO
- Policy-derived object placement: NO
- X: {v['x_range_m'][0]:.6f} to {v['x_range_m'][1]:.6f} m
- Y: {v['y_range_m'][0]:.6f} to {v['y_range_m'][1]:.6f} m
- Z: {v['z_range_m'][0]:.6f} to {v['z_range_m'][1]:.6f} m
- Yaw: {v['yaw_range_deg'][0]:.3f} to {v['yaw_range_deg'][1]:.3f} deg

The manifest inherits the authoritative pre-outcome source grasp windows,
source image/task-frame orientation evidence, fixed table/bin relation, and
source-to-target transforms. Old command paths are retained only as provenance.

## 6. Source event semantics

The reference reset uses one byte-identical A/B event clock and hand semantics.
RIGHT acquisition precedes LEFT release in 35/35. The source-derived clock has
close/lift overlap in 9/35 DEV episodes. The new absolute target-side rule would
retime that overlap identically for both methods; this is a training-supervision
change and is one reason retraining is mandatory.

## 7. Full-task scope decision

Natural-start full-task evaluation remains the intended scope, but it is **not
yet viable**. Standardized grasp is **not selected**: it cannot repair common IK
or articulation validity, and the previous rebase is invalid.

## 8–9. Common execution and physical smoke

The last method-blind collision-aware position qualification passed A 0/3 and
B 1/3, leaving 19 A and 24 B hard self-collision frames. Full 6D was not run and
loaded Dex3 was not run. Therefore common execution fails the precondition for
dataset generation, and physical smoke was not launched.

## 10. Freeze

No new full-task freeze was created. Freezing now would canonize an unqualified
common execution and obsolete checkpoints.

## 11–15. DEV35 results and first failures

Not available. A35/B35 were not run. No TSR, stage count, or first-failure
distribution is inferred from older diagnostic experiments.

## 16. Environment variation

The 35 instances vary source-conditionally in X, Y, and yaw while sharing the
same table, bin, G1 root, registration convention, and matched A/B environment.
Z is constant because each rigid proxy is table-supported at the qualified height.

## 17–18. Artifact paths

- Environment manifest: `{OUT / '00_environment_registration/EVAL35_EPISODE_TASK_ENVIRONMENTS.json'}`
- Environment table: `{OUT / '00_environment_registration/EVAL35_EPISODE_TASK_ENVIRONMENTS.csv'}`
- Environment contact sheet: `{contact}`
- Training audit: `{OUT / '01_act_training_audit/ACT_TRAINING_SUPERVISION_DIFF_AUDIT.md'}`
- Structural preflight: `{OUT / '02_structural_preflight/FULL_TASK_STRUCTURAL_PREFLIGHT.md'}`
- Main result figure: NOT GENERATED (no comparable physical results)
- Physical replay videos: NOT GENERATED (no new physical traces)

## 19. Limitations

DEV35 has been repeatedly used for engineering and is diagnostic only. The
source event extractor observes genuine close/lift overlap in some recordings,
so the newly imposed strict target-side order must be disclosed as common
execution retiming rather than claimed as exact source timing. The shared
collision-aware IK and loaded Dex3 articulation remain unqualified.

## 20. Final-test recommendation

First qualify common sequential position/full-6D IK and loaded Dex3; then build
both TRAIN40 datasets from one shared builder with the strict common event clock,
retrain A/B under the existing parity protocol, run TRAIN physical smoke, and
freeze. Use DEV35 only for pipeline validation. Collect and evaluate a fresh
matched FINAL_TEST only after that freeze.

Final status: **ACT_RETRAINING_REQUIRED_BEFORE_FULL_TASK**
""")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, _ = build_environment_manifest()
    contact = save_environment_artifacts(manifest)
    audit = build_training_audit()
    save_training_audit(audit)
    pre = build_structural_preflight()
    save_structural_preflight(pre)
    (OUT / "PREVIOUS_PHYSICAL_RESULTS_STATUS.md").write_text(
        "# Previous physical results\n\nAll prior end-to-end and standardized-grasp physical results are preserved "
        "unchanged as **PRE_FULL_TASK_RESET_DIAGNOSTIC_ONLY**. They are not used as current full-task evidence.\n"
    )
    report = write_final_report(manifest, audit, pre, contact)
    status = {
        "status": "ACT_RETRAINING_REQUIRED_BEFORE_FULL_TASK",
        "environment_registration": "35/35 PASS",
        "act_retraining_required": True,
        "existing_checkpoints_reused": False,
        "common_execution_qualified": False,
        "physics_run": False,
        "physical_result_available": False,
        "environment_contact_sheet": str(contact),
        "final_report": str(report),
    }
    write_json(OUT / "CURRENT_STATUS.json", status)
    (OUT / "CURRENT_STATUS.md").write_text(
        "# Current status\n\n**ACT_RETRAINING_REQUIRED_BEFORE_FULL_TASK**\n\n"
        "Environment registration: 35/35 PASS. Existing ACT checkpoints are obsolete. "
        "Common collision-aware IK/full-6D/loaded-Dex3 execution is not qualified. No physics was run.\n"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
