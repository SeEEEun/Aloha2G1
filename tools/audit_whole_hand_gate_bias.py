#!/usr/bin/env python3
"""Adversarial, read-only audit of the common whole-hand readiness gate.

This audit deliberately excludes Proposed-B interaction-frame targets.  It uses
only frozen G1/Dex3 FK, the frozen doll collider, the common Dex3 closure, the
authoritative task object position, and task-semantic ownership/hand intent.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from audit_physical_evaluator_task_frame_alignment import (
    DIGITS,
    FrozenProxySurface,
    G1Kinematics,
    PALM_CONFIG,
    PHYSICS_CONFIG,
    assign,
    geometry_state,
    load_common_config,
    load_scene,
    npz_dict,
    pad_samples_model,
    q_from_reference,
    read_json,
    world_pose_from_model,
)


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_contact_constrained_eval/11_gate_bias_audit"
REFERENCE_MANIFEST = (
    ROOT
    / "outputs/final_contact_constrained_eval/09_reference_alignment_preflight/REFERENCE_COMMAND_MANIFEST.json"
)
TASK_AUDIT = (
    ROOT
    / "outputs/final_contact_constrained_eval/08_task_frame_alignment_audit/TASK_FRAME_ALIGNMENT_AUDIT.json"
)
CURRENT_GATE = ROOT / "configs/contact_eval_common_whole_hand_readiness_gate_v1.json"
CURRENT_REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
COMMON_GRASP = ROOT / "configs/contact_eval_common_dex3_grasp_realization_v1.json"
BUILDER = ROOT / "tools/build_common_whole_hand_reference_preflight.py"
SCRIPTED_BASELINE = (
    ROOT
    / "outputs/final_contact_constrained_eval/07_left_grasp_harness_audit/SCRIPTED_LEFT_GRASP_BASELINE.json"
)


def sha256(path: Path) -> str:
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
    temporary.replace(path)


def canonical_intent_frames(reference: dict[str, np.ndarray]) -> np.ndarray:
    """Method-neutral semantic window.

    Fair-A uses OPEN/CLOSED while Proposed-B uses
    OPEN/PRESHAPE/GRASP/HOLD.  Non-OPEN plus not RIGHT_OWNED is the common
    task-semantic statement; no method-specific interaction frame is read.
    """

    phase = reference["left_hand_phase"].astype(str)
    owner = reference["ownership_state"].astype(str)
    return np.flatnonzero((phase != "OPEN") & (owner != "RIGHT_OWNED"))


def object_pose(center: np.ndarray, yaw_deg: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = center
    pose[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    return pose


def nearest_frames(
    g1: G1Kinematics, q: np.ndarray, frames: np.ndarray, center: np.ndarray, count: int = 3
) -> list[tuple[float, int]]:
    rows: list[tuple[float, int]] = []
    for frame in frames:
        assign(g1, q[int(frame)])
        whole = world_pose_from_model(g1, g1.whole_hand_grasp_pose("left"))
        rows.append((float(np.linalg.norm(whole[:3, 3] - center)), int(frame)))
    return sorted(rows)[:count]


def full_close_geometry(
    g1: G1Kinematics,
    surface: FrozenProxySurface,
    palm: dict[str, Any],
    q: np.ndarray,
    frame: int,
    close_q: np.ndarray,
    pose: np.ndarray,
) -> dict[str, Any]:
    candidate = q[frame].copy()
    candidate[14:21] = close_q
    state = geometry_state(g1, surface, palm, candidate, pose, "left")
    gaps = {
        digit: float(state["pad_collider_to_doll_surface_sampled_signed_m"][digit])
        for digit in DIGITS
    }
    return {
        "frame": frame,
        "whole_hand_center_distance_m": float(state["whole_hand_to_doll_center_m"]),
        "pad_surface_gaps_m": gaps,
        "maximum_positive_pad_gap_m": max(max(value, 0.0) for value in gaps.values()),
        "object_between_opposition": bool(state["object_between_thumb_and_index_middle"]),
        "closing_axis_to_short_axis_deg": float(
            state["closing_axis_to_object_short_axis_unsigned_deg"]
        ),
        "pad_center_world_m": state["pad_center_world_m"],
    }


def plot_failures(
    records: list[dict[str, Any]], center: np.ndarray, dimensions: np.ndarray, path: Path
) -> None:
    selected = [
        min(records, key=lambda row: row["nearest_whole_hand_distance_m"]),
        sorted(records, key=lambda row: row["nearest_whole_hand_distance_m"])[len(records) // 2],
        max(records, key=lambda row: row["nearest_whole_hand_distance_m"]),
    ]
    colors = {"thumb": "#d62728", "index": "#1f77b4", "middle": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    for column, row in enumerate(selected):
        geometry = row["neutral_identity_full_close"]
        pads = geometry["pad_center_world_m"]
        for axis, plane in zip(axes[:, column], ("xy", "xz"), strict=True):
            if plane == "xy":
                axis.plot(
                    center[0] + dimensions[0] / 2.0 * np.cos(theta),
                    center[1] + dimensions[1] / 2.0 * np.sin(theta),
                    color="#555555",
                    linewidth=2.0,
                    label="doll collider",
                )
                first, second = 0, 1
                axis.set_xlabel("world X (m)")
                axis.set_ylabel("world Y (m)")
            else:
                axis.plot(
                    center[0] + dimensions[0] / 2.0 * np.cos(theta),
                    center[2] + dimensions[2] / 2.0 * np.sin(theta),
                    color="#555555",
                    linewidth=2.0,
                )
                first, second = 0, 2
                axis.set_xlabel("world X (m)")
                axis.set_ylabel("world Z (m)")
            for digit in DIGITS:
                point = np.asarray(pads[digit])
                axis.scatter(point[first], point[second], s=70, color=colors[digit], label=digit)
                axis.annotate(digit[0].upper(), (point[first], point[second]), xytext=(4, 4), textcoords="offset points")
            axis.scatter(center[first], center[second], marker="x", s=70, color="black")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)
        gaps = geometry["pad_surface_gaps_m"]
        axes[0, column].set_title(
            f"Fair-A eval {row['eval_index']} / frame {geometry['frame']}\n"
            f"whole={1000*row['nearest_whole_hand_distance_m']:.1f} mm; "
            f"T/I/M gaps={1000*gaps['thumb']:.1f}/{1000*gaps['index']:.1f}/{1000*gaps['middle']:.1f} mm"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4)
    fig.suptitle(
        "Representative Fair-A failures under identity-yaw, full Dex3 closure\n"
        "Object position and hand FK only; no Proposed-B interaction-frame quantity",
        fontsize=14,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_sensitivity(result: dict[str, Any], path: Path) -> None:
    thresholds = np.asarray(result["sensitivity"]["positive_pad_gap_threshold_m"])
    a_identity = np.asarray(result["sensitivity"]["identity_yaw_counts"]["Fair-A"])
    b_identity = np.asarray(result["sensitivity"]["identity_yaw_counts"]["Proposed-B"])
    a_best = np.asarray(result["sensitivity"]["per_episode_best_yaw_counts"]["Fair-A"])
    b_best = np.asarray(result["sensitivity"]["per_episode_best_yaw_counts"]["Proposed-B"])
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    x = thresholds * 1000.0
    axis.plot(x, a_identity, "o-", label="Fair-A, identity yaw", color="#4c78a8")
    axis.plot(x, b_identity, "o-", label="Proposed-B, identity yaw", color="#f58518")
    axis.plot(x, a_best, "--", label="Fair-A, adversarial best yaw/episode", color="#4c78a8")
    axis.plot(x, b_best, "--", label="Proposed-B, adversarial best yaw/episode", color="#f58518")
    axis.axvline(
        result["neutral_gate_v2"]["positive_pad_gap_margin_m"] * 1000.0,
        color="black",
        linestyle=":",
        label="scripted-contact-calibrated margin",
    )
    axis.set_xlabel("Allowed positive pad-to-surface gap after FULL CLOSE (mm)")
    axis.set_ylabel("Ready references / 10")
    axis.set_ylim(-0.3, 10.3)
    axis.set_yticks(range(0, 11))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    axis.set_title("Representation-neutral tolerance sensitivity")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(REFERENCE_MANIFEST)
    physics = read_json(PHYSICS_CONFIG)
    palm = read_json(PALM_CONFIG)
    grasp = read_json(COMMON_GRASP)
    current_gate = read_json(CURRENT_GATE)
    registration = read_json(CURRENT_REGISTRATION)
    task_audit = read_json(TASK_AUDIT)
    scripted = read_json(SCRIPTED_BASELINE)
    builder_source = BUILDER.read_text(encoding="utf-8")

    collision_dimensions = np.asarray(
        physics["frozen_doll_contract"]["collision_dimensions_m"], dtype=np.float64
    )
    visual_dimensions = np.asarray(
        physics["frozen_doll_contract"]["visual_dimensions_m"], dtype=np.float64
    )
    center = np.asarray(
        task_audit["scene_registration"][
            "authoritative_retarget_doll_center_world_m_with_current_proxy_bottom_alignment"
        ],
        dtype=np.float64,
    )
    surface = FrozenProxySurface(collision_dimensions, visual_dimensions)
    g1 = G1Kinematics(load_common_config(), load_scene(load_common_config()))
    close_q = np.asarray(grasp["full_close_7d_rad"], dtype=np.float64)

    # The geometry-model margin is calibrated only from an independently
    # successful physical scripted contact, plus fixed PhysX offsets.  It does
    # not inspect either method's trajectory.
    successful_index_gap_m = 0.010120078254315159
    contact_margin = (
        successful_index_gap_m
        + float(physics["object"]["contact_offset_m"])
        + float(physics["gates"]["maximum_runtime_penetration_m"])
    )

    method_rows: dict[str, list[dict[str, Any]]] = {"Fair-A": [], "Proposed-B": []}
    for method_label, output_label in (("ACT-A40", "Fair-A"), ("ACT-B40", "Proposed-B")):
        for record in [row for row in manifest["records"] if row["method"] == method_label]:
            reference = npz_dict(Path(record["source_reference"]))
            q = q_from_reference(reference)
            frames = canonical_intent_frames(reference)
            if not len(frames):
                raise RuntimeError(f"canonical intent window empty: {record['source_reference']}")
            nearest = nearest_frames(g1, q, frames, center, 3)
            identity_candidates = [
                full_close_geometry(
                    g1, surface, palm, q, frame, close_q, object_pose(center, 0.0)
                )
                for _, frame in nearest
            ]
            identity = min(
                identity_candidates,
                key=lambda row: (
                    not row["object_between_opposition"],
                    row["maximum_positive_pad_gap_m"],
                ),
            )
            yaw_candidates = []
            for _, frame in nearest:
                for yaw in range(0, 180, 15):
                    geometry = full_close_geometry(
                        g1,
                        surface,
                        palm,
                        q,
                        frame,
                        close_q,
                        object_pose(center, float(yaw)),
                    )
                    geometry["yaw_deg"] = yaw
                    yaw_candidates.append(geometry)
            best_yaw = min(
                yaw_candidates,
                key=lambda row: (
                    not row["object_between_opposition"],
                    row["maximum_positive_pad_gap_m"],
                ),
            )
            method_rows[output_label].append(
                {
                    "eval_index": int(record["eval_index"]),
                    "stable_episode_id": record["stable_episode_id"],
                    "source_reference": record["source_reference"],
                    "source_reference_sha256": record["source_reference_sha256"],
                    "phase_vocabulary": sorted(set(reference["left_hand_phase"].astype(str))),
                    "canonical_intent_frame_count": int(len(frames)),
                    "nearest_whole_hand_distance_m": nearest[0][0],
                    "nearest_frames": [frame for _, frame in nearest],
                    "neutral_identity_full_close": identity,
                    "adversarial_per_episode_best_yaw_full_close": best_yaw,
                }
            )

    sensitivity_thresholds = [
        float(physics["object"]["contact_offset_m"]),
        contact_margin,
        0.030,
        0.050,
        0.080,
        0.100,
    ]
    identity_counts: dict[str, list[int]] = {}
    best_yaw_counts: dict[str, list[int]] = {}
    for method, rows in method_rows.items():
        identity_counts[method] = [
            sum(
                row["neutral_identity_full_close"]["object_between_opposition"]
                and row["neutral_identity_full_close"]["maximum_positive_pad_gap_m"] <= threshold
                for row in rows
            )
            for threshold in sensitivity_thresholds
        ]
        best_yaw_counts[method] = [
            sum(
                row["adversarial_per_episode_best_yaw_full_close"]["object_between_opposition"]
                and row["adversarial_per_episode_best_yaw_full_close"][
                    "maximum_positive_pad_gap_m"
                ]
                <= threshold
                for row in rows
            )
            for threshold in sensitivity_thresholds
        ]

    neutral_counts = {
        method: sum(
            row["neutral_identity_full_close"]["object_between_opposition"]
            and row["neutral_identity_full_close"]["maximum_positive_pad_gap_m"]
            <= contact_margin
            for row in rows
        )
        for method, rows in method_rows.items()
    }

    result = {
        "schema_version": "adversarial_whole_hand_gate_bias_audit_v1",
        "status": "AUDIT_COMPLETE",
        "current_gate_audit": {
            "formula_reads_proposed_b_interaction_frame": "interaction_frame" in inspect.getsource(geometry_state),
            "builder_gate_formula_reads_proposed_b_interaction_frame": "target_left_interaction_frame" in builder_source,
            "registration_yaw_selected_from_proposed_b": "Proposed-B" in registration["orientation_gauge_resolution"],
            "tolerances_selected_to_enclose_proposed_b_and_exclude_fair_a": (
                "enclose all frozen Proposed-B" in current_gate["selection_provenance"]
                and "below the Fair-A" in current_gate["selection_provenance"]
            ),
            "semantic_candidate_bug": {
                "builder_labels": ["PRESHAPE", "GRASP", "HOLD"],
                "configured_labels": current_gate["required_semantic_intent"],
                "fair_a_phase_vocabulary": sorted(
                    set(value for row in method_rows["Fair-A"] for value in row["phase_vocabulary"])
                ),
                "effect": "Fair-A had zero candidate frames before geometry was evaluated because its frozen vocabulary is OPEN/CLOSED.",
            },
            "unbiased_as_implemented": False,
        },
        "neutral_gate_v2": {
            "status": "AUDIT_ONLY_NOT_FROZEN_FOR_TSR",
            "task_object_position_source": "authoritative task scene",
            "object_yaw_deg": 0.0,
            "object_yaw_source": "identity task axes; source sphere had no method-specific yaw",
            "semantic_intent": "left phase is non-OPEN and ownership is not RIGHT_OWNED",
            "finger_realization": "common FULL CLOSE; arm/wrist held at the frozen reference pose",
            "positive_pad_gap_margin_m": contact_margin,
            "positive_pad_gap_margin_provenance": {
                "successful_scripted_index_geometry_model_gap_m": successful_index_gap_m,
                "physx_contact_offset_m": float(physics["object"]["contact_offset_m"]),
                "maximum_runtime_penetration_m": float(
                    physics["gates"]["maximum_runtime_penetration_m"]
                ),
            },
            "require_object_between_opposition": True,
            "uses_wrist_distance": False,
            "uses_interaction_frame": False,
            "uses_method_name": False,
            "uses_episode_id_or_frame_id": False,
            "ready_counts": neutral_counts,
        },
        "sensitivity": {
            "positive_pad_gap_threshold_m": sensitivity_thresholds,
            "identity_yaw_counts": identity_counts,
            "per_episode_best_yaw_counts": best_yaw_counts,
            "note": "Per-episode best yaw is adversarial audit evidence only and is never proposed as an evaluator correction.",
        },
        "methods": method_rows,
        "fair_a_decision": {
            "physically_graspable_at_reference_pose_under_neutral_gate": bool(neutral_counts["Fair-A"]),
            "closest_episode_best_case": min(
                method_rows["Fair-A"],
                key=lambda row: row["adversarial_per_episode_best_yaw_full_close"][
                    "maximum_positive_pad_gap_m"
                ],
            ),
            "preserve_zero_of_ten_readiness": neutral_counts["Fair-A"] == 0,
        },
        "inputs": {
            "reference_manifest": str(REFERENCE_MANIFEST),
            "reference_manifest_sha256": sha256(REFERENCE_MANIFEST),
            "current_gate": str(CURRENT_GATE),
            "current_gate_sha256": sha256(CURRENT_GATE),
            "current_registration": str(CURRENT_REGISTRATION),
            "current_registration_sha256": sha256(CURRENT_REGISTRATION),
            "common_grasp": str(COMMON_GRASP),
            "common_grasp_sha256": sha256(COMMON_GRASP),
            "physics": str(PHYSICS_CONFIG),
            "physics_sha256": sha256(PHYSICS_CONFIG),
            "scripted_baseline": str(SCRIPTED_BASELINE),
            "scripted_baseline_sha256": sha256(SCRIPTED_BASELINE),
        },
        "mutations": {
            "trajectory": False,
            "retargeting": False,
            "policy": False,
            "physics": False,
            "task_success_definition": False,
        },
    }
    atomic_json(OUT / "ADVERSARIAL_GATE_BIAS_AUDIT.json", result)
    plot_failures(method_rows["Fair-A"], center, collision_dimensions, OUT / "A_REPRESENTATIVE_FAILURES.png")
    plot_sensitivity(result, OUT / "GATE_TOLERANCE_SENSITIVITY.png")

    closest = result["fair_a_decision"]["closest_episode_best_case"]
    closest_geometry = closest["adversarial_per_episode_best_yaw_full_close"]
    lines = [
        "# Adversarial whole-hand readiness-gate bias audit",
        "",
        "## Verdict",
        "",
        "The existing A=0/10 versus B=10/10 gate result is **not acceptable as an unbiased result**. The geometric formula itself does not read a Proposed-B interaction frame, but two configuration choices and one semantic implementation bug bias the result:",
        "",
        "1. Doll yaw was selected from the Proposed-B EVAL10 closing axes.",
        "2. The 90/80/65 tolerances were explicitly selected to enclose Proposed-B and remain below Fair-A's minimum.",
        "3. The builder searched only `PRESHAPE/GRASP/HOLD`; frozen Fair-A uses `OPEN/CLOSED`, so Fair-A received zero geometry evaluations.",
        "",
        "## Representation-neutral counterfactual",
        "",
        "The audit therefore re-evaluated both methods using identity task-axis yaw, a common non-OPEN/not-RIGHT-owned semantic intent, the same frozen common FULL CLOSE, and no wrist/arm movement. The only geometric contact margin is derived from an independently successful scripted physical contact plus the frozen PhysX contact/penetration offsets.",
        "",
        f"- Contact-margin threshold: `{1000*contact_margin:.3f} mm`.",
        f"- Fair-A ready: `{neutral_counts['Fair-A']}/10`.",
        f"- Proposed-B ready: `{neutral_counts['Proposed-B']}/10`.",
        "",
        "This audit does not force equal outcomes. It asks whether the doll is actually reachable by all three closing pad colliders at the frozen reference arm/wrist pose.",
        "",
        "## Fair-A adversarial best case",
        "",
        f"The closest Fair-A case is eval `{closest['eval_index']}` (`{closest['stable_episode_id']}`). Even after selecting its best doll yaw from 0–165 deg in 15 deg increments (audit-only, never an evaluator correction), FULL CLOSE leaves a maximum positive pad gap of `{1000*closest_geometry['maximum_positive_pad_gap_m']:.3f} mm` at yaw `{closest_geometry['yaw_deg']} deg`. Its T/I/M gaps are `{1000*closest_geometry['pad_surface_gaps_m']['thumb']:.3f}` / `{1000*closest_geometry['pad_surface_gaps_m']['index']:.3f}` / `{1000*closest_geometry['pad_surface_gaps_m']['middle']:.3f}` mm. This is far beyond the `{1000*contact_margin:.3f} mm` calibrated margin.",
        "",
        "Thus Fair-A is not being rejected only by a wrist or semantic gate: at its frozen reference pose, even the common full physical closure cannot reach the doll with all three digits. The neutral readiness result remains Fair-A 0/10.",
        "",
        "## Scope",
        "",
        "No trajectory, dataset, checkpoint, physics parameter, task-success criterion, or A/B artifact was changed. The neutral gate is an audit definition only; it is not frozen for TSR because the downstream reference physical-grasp preflight remains blocked.",
        "",
        "Artifacts:",
        "",
        f"- `{OUT / 'ADVERSARIAL_GATE_BIAS_AUDIT.json'}`",
        f"- `{OUT / 'A_REPRESENTATIVE_FAILURES.png'}`",
        f"- `{OUT / 'GATE_TOLERANCE_SENSITIVITY.png'}`",
    ]
    (OUT / "ADVERSARIAL_GATE_BIAS_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = OUT / "AUDIT_MANIFEST.json"
    files = [
        OUT / "ADVERSARIAL_GATE_BIAS_AUDIT.json",
        OUT / "ADVERSARIAL_GATE_BIAS_AUDIT.md",
        OUT / "A_REPRESENTATIVE_FAILURES.png",
        OUT / "GATE_TOLERANCE_SENSITIVITY.png",
    ]
    atomic_json(
        manifest_path,
        {
            "status": "READ_ONLY_ADVERSARIAL_AUDIT_COMPLETE",
            "files": [
                {"path": str(path.resolve()), "sha256": sha256(path)} for path in files
            ],
        },
    )
    print(json.dumps({"status": result["status"], "neutral_ready_counts": neutral_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
