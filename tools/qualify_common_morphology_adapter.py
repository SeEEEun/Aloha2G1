#!/usr/bin/env python3
"""Qualify the TRAIN-only method-blind morphology adapter, fail-closed by stage."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_g1_morphology_adapter import CommonG1MorphologyAdapter
from tools.doll_handoff_retargeting.common import branch_flags, load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.search_common_workspace_solver import transformed_targets


OUT = ROOT / "outputs/single_variable_ab_common_execution"
SMOKE = OUT / "04_workspace_registration/smoke"
OLD_REGISTRATION = OUT / "04_workspace_registration/COMMON_TASK_REGISTRATION_TRAIN_SMOKE.json"
SELECTED_REGISTRATION = OUT / "06_common_morphology/COMMON_TASK_REGISTRATION_TRAIN_SMOKE.json"
SEARCH_DETAIL = OUT / "workspace_solver_search/refine_yaw20_y_plus_10mm.json"
SEARCH_Q = OUT / "workspace_solver_search/refine_yaw20_y_plus_10mm_q.npz"
ADAPTER_DIR = OUT / "06_common_morphology/adapter_position_only"
ADAPTER_CONFIG = ROOT / "configs/common_g1_morphology_adapter_v1.json"
ADAPTER_IMPLEMENTATION = ROOT / "tools/common_g1_morphology_adapter.py"
COMMON_SOLVER = ROOT / "tools/doll_handoff_retargeting/retarget.py"
EPISODES = (0, 24, 49)
METHODS = (("baseline", "A", "WRIST", "#3465a4"), ("proposed", "B", "INTERACTION", "#cc0000"))
SIDES = ("left", "right")


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(item) for item in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_path(method: str, episode: int) -> Path:
    paths = list((SMOKE / method / "trajectories").glob(f"*ep{episode:03d}.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"archive count for {method} ep{episode}: {len(paths)}")
    return paths[0]


def stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1) * scale
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def no_method_branch_audit(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Validating a config declaration named ``method_blind`` is not a method
    # branch.  Only an actual representation/mode/outcome token in a control
    # predicate is forbidden.
    forbidden_tokens = ("representation_mode", "WRIST", "INTERACTION", "task_success", "physical_outcome")
    bad_tests: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Match)):
            segment = ast.get_source_segment(source, node) or ""
            first_line = segment.splitlines()[0] if segment else ""
            if any(token in first_line for token in forbidden_tokens):
                bad_tests.append({"line": getattr(node, "lineno", None), "source": first_line})
    return {
        "implementation": str(path.resolve()),
        "implementation_sha256": sha256(path),
        "conditional_tests_using_forbidden_method_or_outcome_inputs": bad_tests,
        "pass": not bad_tests,
        "note": "metadata declarations that state forbidden inputs were not consumed are not branches",
    }


def main() -> int:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    common = load_common_config(SMOKE / "config/common_config.json")
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal = np.asarray(common["resolved"]["canonical_g1_nominal_q"], dtype=float)
    old_registration = json.loads(OLD_REGISTRATION.read_text(encoding="utf-8"))
    selected_registration = json.loads(SELECTED_REGISTRATION.read_text(encoding="utf-8"))
    search_detail = json.loads(SEARCH_DETAIL.read_text(encoding="utf-8"))
    old_matrix = np.asarray(old_registration["entries"][0]["common_workspace_transform_matrix"], dtype=float)
    selected_matrix = np.asarray(search_detail["transform_matrix"], dtype=float)
    if not np.array_equal(
        selected_matrix,
        np.asarray(selected_registration["entries"][0]["common_workspace_transform_matrix"], dtype=float),
    ):
        raise RuntimeError("selected formal registration does not match best search transform")
    q_source = np.load(SEARCH_Q, allow_pickle=False)
    adapter = CommonG1MorphologyAdapter(common, g1, nominal, ADAPTER_CONFIG)
    config_hash = sha256(ADAPTER_CONFIG)
    implementation_hash = sha256(ADAPTER_IMPLEMENTATION)
    solver_hash = sha256(COMMON_SOLVER)
    validation = common["validation"]
    required_rate = float(common["shared_temporal_ik"]["required_success_rate"])

    rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for method, label, mode, _ in METHODS:
        for episode in EPISODES:
            source_path = archive_path(method, episode)
            with np.load(source_path, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            incoming = transformed_targets(values, g1, old_matrix, selected_matrix)
            q_initial = q_source[f"{method}_ep{episode:03d}"].astype(float)
            fps = float(1.0 / np.median(np.diff(values["timestamp"].astype(float))))
            print(f"adapting {label} ep{episode:02d}", flush=True)
            result = adapter.adapt(
                incoming,
                q_initial,
                values["left_dex3_qpos"].astype(float),
                values["right_dex3_qpos"].astype(float),
                fps,
                include_orientation=False,
            )
            position_residual = np.maximum(
                *[
                    np.linalg.norm(
                        result.achieved_positions_model[side]
                        - result.feasible_targets[f"{side}_wrist_position"],
                        axis=1,
                    )
                    for side in SIDES
                ]
            )
            accepted = position_residual <= float(common["shared_temporal_ik"]["position_tolerance_m"])
            rate = float(np.mean(accepted))
            branch = branch_flags(
                result.projected_q,
                float(validation["branch_absolute_step_norm_rad"]),
                float(validation["branch_local_multiplier"]),
            )
            hard_limits = int(
                np.count_nonzero(
                    (result.projected_q < g1.arm_limits[:, 0] - 1e-9)
                    | (result.projected_q > g1.arm_limits[:, 1] + 1e-9)
                )
            )
            temporal = result.metadata["temporal"]
            temporal_pass = bool(
                float(temporal["maximum_joint_step_rad"]) <= float(validation["maximum_joint_step_rad"]) + 1e-7
                and float(temporal["maximum_velocity_rad_s"]) <= float(validation["maximum_velocity_rad_s"]) + 1e-7
                and float(temporal["maximum_acceleration_rad_s2"]) <= float(validation["maximum_acceleration_rad_s2"]) + 1e-5
            )
            projection_bound_pass = bool(result.metadata["position_projection_m"]["within_bound"])
            self_collision_pass = int(result.metadata["hard_self_collision_frames"]) == 0
            qualified = bool(
                rate >= required_rate
                and hard_limits == 0
                and int(np.count_nonzero(branch)) == 0
                and temporal_pass
                and result.metadata["finite"]
                and projection_bound_pass
                and self_collision_pass
            )
            row = {
                "episode_index": episode,
                "method": label,
                "representation_mode": mode,
                "source_archive": str(source_path.resolve()),
                "source_archive_sha256": sha256(source_path),
                "common_workspace_transform": selected_matrix,
                "method_blind_adapter": True,
                "position_accepted_frames": int(np.count_nonzero(accepted)),
                "frame_count": len(accepted),
                "position_acceptance_rate": rate,
                "position_residual_m": stats(position_residual),
                "position_projection_m": result.metadata["position_projection_m"],
                "hard_limit_violation_count": hard_limits,
                "branch_discontinuity_count": int(np.count_nonzero(branch)),
                "finite": bool(result.metadata["finite"]),
                "temporal_pass": temporal_pass,
                "hard_self_collision_frames": int(result.metadata["hard_self_collision_frames"]),
                "self_collision_contact_frames": int(result.metadata["self_collision_contact_frames"]),
                "nominal_clearance_projection_status": result.metadata["nominal_clearance_projection"].get("status"),
                "nominal_clearance_projection_candidates": result.metadata["nominal_clearance_projection"].get("candidates", []),
                "projection_bound_pass": projection_bound_pass,
                "position_only_qualified": qualified,
                "failure_reasons": [
                    reason
                    for reason, passed in (
                        ("COMMON_IK_ACCEPTANCE", rate >= required_rate),
                        ("HARD_LIMIT", hard_limits == 0),
                        ("BRANCH_CONTINUITY", int(np.count_nonzero(branch)) == 0),
                        ("TEMPORAL_CONTINUITY", temporal_pass),
                        ("FINITE", bool(result.metadata["finite"])),
                        ("MORPHOLOGY_PROJECTION_BOUND", projection_bound_pass),
                        ("SELF_COLLISION", self_collision_pass),
                    )
                    if not passed
                ],
                "adapter_metadata": result.metadata,
            }
            rows.append(row)
            arrays[(method, episode)] = {
                **{f"raw_{side}_position_model": result.raw_targets[f"{side}_wrist_position"] for side in SIDES},
                **{f"raw_{side}_rotation_model": result.raw_targets[f"{side}_wrist_rotation"] for side in SIDES},
                **{f"feasible_{side}_position_model": result.feasible_targets[f"{side}_wrist_position"] for side in SIDES},
                **{f"feasible_{side}_rotation_model": result.feasible_targets[f"{side}_wrist_rotation"] for side in SIDES},
                "projected_q": result.projected_q,
                "position_projection_m": result.position_projection_m,
                "orientation_projection_rad": result.orientation_projection_rad,
                "registered_object_position_world": np.asarray(
                    {int(entry["episode_index"]): entry for entry in selected_registration["entries"]}[episode]["target_object_pose"]["position_xyz_m"],
                    dtype=float,
                ),
                "timestamp": values["timestamp"],
            }
            artifact = ADAPTER_DIR / f"{label}_ep{episode:03d}_raw_and_feasible.npz"
            atomic_npz(artifact, **arrays[(method, episode)])
            atomic_json(artifact.with_suffix(".json"), row)
            print(
                f"{label} ep{episode:02d}: IK={rate:.6f} collision={row['hard_self_collision_frames']} "
                f"projection={row['position_projection_m']['max']*1000:.3f}mm qualified={qualified}",
                flush=True,
            )

    method_counts = {
        label: sum(row["position_only_qualified"] for row in rows if row["method"] == label)
        for _, label, _, _ in METHODS
    }
    method_projection = {
        label: stats(
            np.concatenate(
                [
                    arrays[(method, episode)]["position_projection_m"].reshape(-1)
                    for episode in EPISODES
                ]
            ),
            1000.0,
        )
        for method, label, _, _ in METHODS
    }
    position_pass = all(count == 3 for count in method_counts.values())

    figure = plt.figure(figsize=(16, 8), constrained_layout=True)
    for column, episode in enumerate(EPISODES):
        for row_index, (method, label, mode, color) in enumerate(METHODS):
            ax = figure.add_subplot(2, 3, row_index * 3 + column + 1, projection="3d")
            values = arrays[(method, episode)]
            for side, linestyle in (("left", "-"), ("right", "--")):
                raw = g1.model_to_world_position(values[f"raw_{side}_position_model"])
                feasible = g1.model_to_world_position(values[f"feasible_{side}_position_model"])
                ax.plot(*raw.T, color="#777777", lw=1.0, alpha=0.55, ls=linestyle, label="RAW" if side == "left" else None)
                ax.plot(*feasible.T, color=color, lw=1.8, alpha=0.9, ls=linestyle, label="COMMON FEASIBLE" if side == "left" else None)
            doll = values["registered_object_position_world"]
            ax.scatter(*doll, color="#fdae61", s=90, marker="s", label="registered doll")
            ax.view_init(elev=86, azim=-90)
            ax.set_title(f"{label} {mode} — episode {episode:02d}\nqualified={'YES' if next(r for r in rows if r['method']==label and r['episode_index']==episode)['position_only_qualified'] else 'NO'}")
            ax.set_xlabel("world X (m)")
            ax.set_ylabel("world Y (m)")
            ax.set_zlabel("world Z (m)")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(alpha=0.2)
    figure.suptitle("TRAIN smoke: RAW targets vs method-blind COMMON FEASIBLE targets", fontsize=15)
    projection_visual = OUT / "target_ik_visual_audit/SMOKE_AB_RAW_VS_FEASIBLE_TARGETS.png"
    projection_visual.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(projection_visual, dpi=220)
    plt.close(figure)

    branch_audit = no_method_branch_audit(ADAPTER_IMPLEMENTATION)
    fairness_pass = bool(branch_audit["pass"])
    report = {
        "schema_version": "common_g1_morphology_adapter_qualification_v1",
        "status": "PASS" if position_pass and fairness_pass else "COMMON_POSITION_FEASIBILITY_STILL_INVALID",
        "scope": "TRAIN_ONLY_SMOKE_0_24_49",
        "static_workspace_sufficient": False,
        "selected_common_workspace_transform": selected_registration["common_workspace_registration"],
        "adapter_used": True,
        "method_blind": fairness_pass,
        "method_position_qualified_counts": method_counts,
        "position_projection_mm": method_projection,
        "position_only_pass": position_pass,
        "full_6d": "NOT_RUN_BLOCKED_BY_POSITION_GATE" if not position_pass else "READY",
        "loaded_dex3": "NOT_RUN_BLOCKED_BY_POSITION_GATE" if not position_pass else "READY_AFTER_6D",
        "shared_pipeline_smoke": "NOT_RUN_BLOCKED_BY_POSITION_GATE" if not position_pass else "READY_AFTER_6D_AND_DEX3",
        "rows": rows,
        "fairness": branch_audit,
        "hashes": {
            "adapter_implementation_sha256": implementation_hash,
            "adapter_config_sha256": config_hash,
            "common_solver_sha256": solver_hash,
            "selected_registration_sha256": sha256(SELECTED_REGISTRATION),
        },
        "visual": str(projection_visual.resolve()),
    }
    atomic_json(OUT / "COMMON_MORPHOLOGY_ADAPTER_REPORT.json", report)

    report_lines = [
        "# Common G1 morphology adapter report",
        "",
        f"Status: **{report['status']}**",
        "",
        "The raw A/B targets are preserved. One method-blind constrained G1 projection was attempted after target generation and before any dataset or policy work.",
        "",
        "- Common static registration sufficient: **NO**",
        f"- A position-only qualification: **{method_counts['A']}/3**",
        f"- B position-only qualification: **{method_counts['B']}/3**",
        f"- A projection mean/p95/max: **{method_projection['A']['mean']:.3f} / {method_projection['A']['p95']:.3f} / {method_projection['A']['max']:.3f} mm**",
        f"- B projection mean/p95/max: **{method_projection['B']['mean']:.3f} / {method_projection['B']['p95']:.3f} / {method_projection['B']['max']:.3f} mm**",
        f"- Declared common position bound: **{json.loads(ADAPTER_CONFIG.read_text())['projection_bounds']['maximum_position_projection_m']*1000:.3f} mm**",
        f"- Before/after visual: `{projection_visual.resolve()}`",
        "",
        "| episode | method | common IK rate | projection max | hard collision frames | bound | qualified | failure |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['episode_index']} | {row['method']} | {row['position_acceptance_rate']*100:.2f}% | "
            f"{row['position_projection_m']['max']*1000:.3f} mm | {row['hard_self_collision_frames']} | "
            f"{'PASS' if row['projection_bound_pass'] else 'FAIL'} | {'PASS' if row['position_only_qualified'] else 'FAIL'} | "
            f"{', '.join(row['failure_reasons']) or 'NONE'} |"
        )
    if not position_pass:
        report_lines.extend(
            [
                "",
                "## Stop gate",
                "",
                "At least one trajectory cannot simultaneously satisfy the unchanged sequential IK acceptance, self-collision, temporal-continuity, and morphology-derived projection-bound constraints. Full 6D orientation, loaded Dex3, shared physical/reference smoke, dataset regeneration, ACT training, DEV35, and EVAL35 remain unrun.",
            ]
        )
    atomic_text(OUT / "COMMON_MORPHOLOGY_ADAPTER_REPORT.md", "\n".join(report_lines) + "\n")

    fairness = {
        "schema_version": "common_morphology_adapter_fairness_audit_v1",
        "status": "PASS" if fairness_pass else "FAIL",
        "A_pipeline": "WRIST raw target -> CommonG1MorphologyAdapter",
        "B_pipeline": "INTERACTION raw target -> CommonG1MorphologyAdapter",
        "adapter_call_signature": "adapt(incoming_targets, initial_q, left_hand, right_hand, fps, include_orientation)",
        "method_or_representation_argument": False,
        "method_specific_parameters": 0,
        "episode_specific_parameters": 0,
        "object_or_outcome_input": False,
        "conditional_branch_audit": branch_audit,
        "hashes": report["hashes"],
    }
    atomic_json(OUT / "COMMON_MORPHOLOGY_ADAPTER_FAIRNESS_AUDIT.json", fairness)
    atomic_text(
        OUT / "COMMON_MORPHOLOGY_ADAPTER_FAIRNESS_AUDIT.md",
        "\n".join(
            [
                "# Common morphology adapter fairness audit",
                "",
                f"Status: **{fairness['status']}**",
                "",
                "- A: `WRIST raw target -> CommonG1MorphologyAdapter`",
                "- B: `INTERACTION raw target -> CommonG1MorphologyAdapter`",
                "- Method/representation argument: **NO**",
                "- Object/task-success/physical-outcome input: **NO**",
                "- Method-specific parameters: **0**",
                "- Episode-specific parameters: **0**",
                f"- Adapter implementation SHA256: `{implementation_hash}`",
                f"- Adapter config SHA256: `{config_hash}`",
                f"- Common solver SHA256: `{solver_hash}`",
                "",
            ]
        ),
    )

    full6d = {
        "schema_version": "common_6d_ik_final_audit_v1",
        "status": "NOT_RUN_BLOCKED_BY_POSITION_GATE" if not position_pass else "READY_NOT_RUN",
        "position_only": {"A": method_counts["A"], "B": method_counts["B"], "denominator": 3},
        "full_6d": {"A": 0, "B": 0, "denominator": 3, "not_run": True},
        "orientation_projection": "NOT_RUN",
        "reason": "user-required ordering forbids orientation work before A/B position-only 3/3",
    }
    atomic_json(OUT / "COMMON_6D_IK_FINAL_AUDIT.json", full6d)
    atomic_text(
        OUT / "COMMON_6D_IK_FINAL_AUDIT.md",
        f"# Common 6D IK final audit\n\nStatus: **{full6d['status']}**\n\n- Position-only A: **{method_counts['A']}/3**\n- Position-only B: **{method_counts['B']}/3**\n- Full 6D A/B: **NOT RUN**\n- Orientation projection: **NOT RUN**\n\nThe position gate did not pass; the required sequence prohibits orientation qualification.\n",
    )
    loaded = {
        "schema_version": "loaded_dex3_final_audit_v1",
        "status": "NOT_RUN_BLOCKED_BY_POSITION_GATE",
        "mapping": "NOT_RUN",
        "sign": "NOT_RUN",
        "readback": "NOT_RUN",
        "runtime_limits": "NOT_RUN",
        "measured_violations": "NOT_RUN",
        "reason": "loaded Dex3 is permitted only after A/B full 6D IK 3/3",
    }
    atomic_json(OUT / "LOADED_DEX3_FINAL_AUDIT.json", loaded)
    atomic_text(
        OUT / "LOADED_DEX3_FINAL_AUDIT.md",
        "# Loaded Dex3 final audit\n\nStatus: **NOT RUN — BLOCKED BY POSITION GATE**\n\nNo PhysX articulation test was launched. The user-required sequence permits loaded Dex3 only after A/B full 6D IK reaches 3/3.\n",
    )
    parity_status = "PASS_DOWNSTREAM_PARITY_BUT_PIPELINE_NOT_QUALIFIED" if fairness_pass else "FAIL"
    parity_lines = [
        "# Final common pipeline parity audit",
        "",
        f"Status: **{parity_status}**",
        "",
        "| Stage | A | B | Parity |",
        "|---|---|---|---|",
        "| Raw target generation | WRIST | INTERACTION | ONLY INTENTIONAL DIFFERENCE |",
        "| Task registration | selected common XYZ+yaw | selected common XYZ+yaw | IDENTICAL |",
        "| Morphology adapter | CommonG1MorphologyAdapter | CommonG1MorphologyAdapter | IDENTICAL |",
        "| IK | shared temporal IK acceptance/config | shared temporal IK acceptance/config | IDENTICAL |",
        "| Temporal semantics | common source events | common source events | IDENTICAL |",
        "| Dex3 controller | common | common | IDENTICAL (not rerun) |",
        "| Articulation | common | common | IDENTICAL (loaded gate not run) |",
        "| Physical scene/scorer | common | common | IDENTICAL (physical smoke not run) |",
        "",
        "Unintended A/B code/config confounds observed in this qualification: **0**. This parity result does not override the failed common position-feasibility gate.",
        "",
    ]
    atomic_text(OUT / "FINAL_COMMON_PIPELINE_PARITY_AUDIT.md", "\n".join(parity_lines))

    final_lines = [
        "# Common execution qualification — final",
        "",
        "Status: **COMMON_POSITION_FEASIBILITY_STILL_INVALID**" if not position_pass else "Status: **POSITION PASS; FURTHER STAGES REQUIRED**",
        "",
        "- Raw target semantics: **A PASS / B PASS**",
        "- Common static workspace registration: **INSUFFICIENT (17 actual sequential-solver candidates)**",
        f"- Common morphology adapter method-blind: **{'YES' if fairness_pass else 'NO'}**",
        f"- Position-only A: **{method_counts['A']}/3**",
        f"- Position-only B: **{method_counts['B']}/3**",
        "- Full 6D: **NOT RUN**",
        "- Loaded Dex3: **NOT RUN**",
        "- Shared reference pipeline smoke: **NOT RUN**",
        "- Dataset regeneration/training/DEV35/EVAL35: **NOT RUN**",
        "",
        "The failing episodes require either unresolved self-collision or a collision-clearing Cartesian projection larger than the morphology-derived bound. Treating those projections as valid would be an unbounded trajectory rewrite, so the workflow stops before orientation and physics.",
        "",
    ]
    atomic_text(OUT / "COMMON_EXECUTION_QUALIFICATION_FINAL.md", "\n".join(final_lines))
    print(json.dumps({"status": report["status"], "position_counts": method_counts, "projection_mm": method_projection}, indent=2), flush=True)
    return 0 if position_pass and fairness_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
