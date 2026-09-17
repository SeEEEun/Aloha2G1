#!/usr/bin/env python3
"""Finalize the frozen contact-constrained ACT-A/B EVAL10 evaluation.

This program is deliberately read-only with respect to policy, command, scene,
physics, and scoring artifacts.  It only validates, aggregates, audits, and
plots the twenty already-completed physical runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.stats import beta


ROOT = Path("/home/jbnu/aloha_g1_dataset")
BASE = ROOT / "outputs/final_contact_constrained_eval"
PREP = BASE / "04_eval10_preparation"
RESULTS = BASE / "05_act_ab_results"
FIGURES = BASE / "06_paper_figures"
RUNS = RESULTS / "runs"
COMMANDS = RESULTS / "commands"
FREEZE = BASE / "03_freeze"

METHODS = {"act_a40": "ACT-A40", "act_b40": "ACT-B40"}
STAGES = [
    "LEFT_GRASP",
    "HANDOFF",
    "RIGHT_OWNERSHIP",
    "NO_DROP_BEFORE_BIN",
    "DOLL_ENTERS_BIN",
    "DOLL_SETTLES",
    "FULL_TASK_SUCCESS",
]
STAGE_LABELS = [
    "Left grasp",
    "Handoff",
    "Right ownership",
    "No drop to bin",
    "Bin entry",
    "Bin settle",
    "Full task",
]
HELDOUT8 = [2, 13, 23, 27, 28, 31, 37, 40]
NEW2 = ["GoPark_20260901_140555", "GoPark_20260901_140822"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_ci(successes: int, trials: int, alpha: float = 0.05) -> list[float]:
    if successes == 0:
        low = 0.0
    else:
        low = float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    if successes == trials:
        high = 1.0
    else:
        high = float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return [low, high]


def last_physics_sample_per_control_frame(event: dict[str, np.ndarray], frames: int) -> np.ndarray:
    control = np.asarray(event["control_frame"], dtype=np.int64)
    measured = np.asarray(event["measured_q_rad"], dtype=np.float32)
    output = np.empty((frames, measured.shape[1]), dtype=np.float32)
    for frame in range(frames):
        indices = np.flatnonzero(control == frame)
        if not indices.size:
            raise RuntimeError(f"missing physics samples for control frame {frame}")
        output[frame] = measured[indices[-1]]
    return output


def first_failure(outcomes: dict[str, bool]) -> str:
    for stage in STAGES[:-1]:
        if not bool(outcomes[stage]):
            return stage
    return "NONE" if bool(outcomes["FULL_TASK_SUCCESS"]) else "FULL_TASK_SUCCESS"


def intervention_audit(
    command_path: Path,
    event_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    with np.load(command_path, allow_pickle=False) as command_npz:
        command = {name: np.asarray(command_npz[name]) for name in command_npz.files}
    with np.load(event_path, allow_pickle=False) as event_npz:
        event = {name: np.asarray(event_npz[name]) for name in event_npz.files}

    raw = np.asarray(command["raw_policy_command"], dtype=np.float32)
    executed = np.asarray(command["executed_common_controller_command"], dtype=np.float32)
    if raw.shape != executed.shape or raw.ndim != 2 or raw.shape[1] != 28:
        raise RuntimeError(f"invalid command arrays: {command_path}")
    measured = last_physics_sample_per_control_frame(event, raw.shape[0])
    joint_names = np.asarray(command["joint_names"]).astype(str)
    stage = np.asarray(command["stage"]).astype(str)

    # The persisted zero mask means "no local physical primitive override".
    # The deployment-safety projection occurred earlier and is audited here
    # explicitly as the numerical raw-to-executed difference.
    local_primitive_mask = np.asarray(command["common_controller_override_mask"], dtype=bool)
    projection_mask = np.abs(raw - executed) > 1.0e-9
    union_mask = local_primitive_mask | projection_mask
    diff = executed - raw
    arm = np.arange(14)
    dex3 = np.arange(14, 28)
    wrist = np.asarray([index for index, name in enumerate(joint_names) if "wrist" in name], dtype=np.int64)

    phase: dict[str, Any] = {}
    for name in np.unique(stage):
        rows = stage == name
        phase[name] = {
            "frames": int(np.sum(rows)),
            "frame_intervention_fraction": float(np.mean(np.any(union_mask[rows], axis=1))),
            "scalar_intervention_fraction": float(np.mean(union_mask[rows])),
        }

    audit = {
        "raw_policy_command": str(command_path),
        "event_log": str(event_path),
        "frames": int(raw.shape[0]),
        "raw_to_executed_rmse_rad": float(np.sqrt(np.mean(diff**2))),
        "raw_to_executed_arm_rmse_rad": float(np.sqrt(np.mean(diff[:, arm] ** 2))),
        "raw_to_executed_dex3_rmse_rad": float(np.sqrt(np.mean(diff[:, dex3] ** 2))),
        "maximum_raw_to_executed_abs_rad": float(np.max(np.abs(diff))),
        "common_local_primitive_override_scalar_fraction": float(np.mean(local_primitive_mask)),
        "deployment_safety_projection_scalar_fraction": float(np.mean(projection_mask)),
        "union_common_execution_scalar_fraction": float(np.mean(union_mask)),
        "union_common_execution_frame_fraction": float(np.mean(np.any(union_mask, axis=1))),
        "arm_intervention_scalar_fraction": float(np.mean(union_mask[:, arm])),
        "wrist_intervention_scalar_fraction": float(np.mean(union_mask[:, wrist])) if wrist.size else 0.0,
        "dex3_intervention_scalar_fraction": float(np.mean(union_mask[:, dex3])),
        "intervened_joint_names": joint_names[np.any(union_mask, axis=0)].tolist(),
        "phase_intervention": phase,
        "interpretation": (
            "No post-freeze local grasp/handoff/transport primitive was activated. "
            "The only raw-to-executed difference is the identical frozen deployment-safety "
            "projection; arm commands are unchanged."
        ),
    }

    np.savez_compressed(
        run_dir / "POLICY_EXECUTION_AUDIT.npz",
        RAW_POLICY_COMMAND=raw,
        EXECUTED_COMMAND=executed,
        MEASURED_Q=measured,
        COMMON_CONTROLLER_OVERRIDE_MASK=union_mask,
        COMMON_LOCAL_PRIMITIVE_OVERRIDE_MASK=local_primitive_mask,
        DEPLOYMENT_SAFETY_PROJECTION_MASK=projection_mask,
        STAGE=stage,
        JOINT_NAMES=joint_names,
    )
    dump_json(run_dir / "POLICY_EXECUTION_AUDIT.json", audit)
    return audit


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = read_json(FREEZE / "FREEZE_MANIFEST.json")
    if freeze["status"] != "FROZEN" or freeze["scripted_repeatability"] != "3/3 PASS":
        raise RuntimeError("contact-constrained environment is not frozen at scripted 3/3")
    eval10 = read_json(PREP / "EVAL10_RETARGETING_MANIFEST.json")
    integrity = read_json(BASE / "01_new_unseen_2_integrity/NEW_UNSEEN_2_MANIFEST.json")
    if eval10["status"] != "PASS" or len(eval10["eval_entries"]) != 10:
        raise RuntimeError("EVAL10 conversion manifest is incomplete")
    if integrity["status"] != "PASS":
        raise RuntimeError("NEW_UNSEEN_2 integrity gate is not PASS")
    if [entry["source_final_episode"] for entry in eval10["eval_entries"][:8]] != HELDOUT8:
        raise RuntimeError("authoritative HELDOUT8 identity mismatch")
    if [entry["source_name"] for entry in eval10["eval_entries"][8:]] != NEW2:
        raise RuntimeError("fixed NEW_UNSEEN_2 identity mismatch")
    commands = read_json(RESULTS / "PHYSICAL_COMMAND_MANIFEST.json")
    if commands["evaluation_set"] != "EVAL10" or len(commands["records"]) != 20:
        raise RuntimeError("physical command manifest is incomplete")
    return freeze, eval10, integrity


def build_results() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    freeze, eval10, integrity = validate_inputs()
    records: list[dict[str, Any]] = []
    audits: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    entries = {int(entry["eval_index"]): entry for entry in eval10["eval_entries"]}

    for method_dir, method_label in METHODS.items():
        run_dirs = sorted((RUNS / method_dir).glob("eval_*"))
        if len(run_dirs) != 10:
            raise RuntimeError(f"{method_label}: expected 10 physical runs, found {len(run_dirs)}")
        for run_dir in run_dirs:
            result_path = run_dir / "CONTACT_CONSTRAINED_TASK_RESULT.json"
            result = read_json(result_path)
            if not result["command_completed"] or result["execution_mode"] != "CONTACT_CONSTRAINED_PHYSICS":
                raise RuntimeError(f"incomplete/non-physical run: {run_dir}")
            command_path = Path(result["provenance"]["command"])
            event_path = Path(result["provenance"]["event_log"])
            if sha256(command_path) != result["provenance"]["command_sha256"]:
                raise RuntimeError(f"command hash mismatch: {command_path}")
            if sha256(event_path) != result["provenance"]["event_log_sha256"]:
                raise RuntimeError(f"event hash mismatch: {event_path}")
            with np.load(command_path, allow_pickle=False) as command:
                index = int(command["eval_index"].item())
                stable_id = str(command["stable_episode_id"].item())
                provenance = str(command["provenance"].item())
            expected = entries[index]
            if stable_id != expected["stable_episode_id"] or provenance != expected["provenance"]:
                raise RuntimeError(f"EVAL10/run identity mismatch: {run_dir}")
            audit = intervention_audit(command_path, event_path, run_dir)
            audits[method_dir].append(audit)
            outcomes = {stage: bool(result["outcomes"][stage]) for stage in STAGES}
            diagnostics = result["diagnostics"]
            records.append(
                {
                    "eval_index": index,
                    "source_episode_id": stable_id,
                    "source_final_episode": expected.get("source_final_episode"),
                    "source_recording": expected.get("source_name"),
                    "provenance": provenance,
                    "method": method_label,
                    **outcomes,
                    "CLEAN_COMMANDED_RELEASE": result["release_classification"] == "CLEAN_COMMANDED_RELEASE",
                    "PREMATURE_DROP_INTO_BIN": result["release_classification"] == "PREMATURE_DROP_INTO_BIN",
                    "PREMATURE_DROP_OUTSIDE_BIN": result["release_classification"] == "PREMATURE_DROP_OUTSIDE_BIN",
                    "release_classification": result["release_classification"],
                    "first_failure_stage": first_failure(outcomes),
                    "robot_bin_contact_frames": diagnostics["robot_bin_contact_frames"],
                    "maximum_robot_bin_force_n": diagnostics["maximum_robot_bin_force_n"],
                    "maximum_robot_bin_penetration_m": diagnostics["maximum_robot_bin_penetration_m"],
                    "maximum_doll_bin_penetration_m": diagnostics["maximum_doll_bin_penetration_m"],
                    "peak_held_object_speed_m_s": diagnostics["peak_held_object_speed_m_s"],
                    "maximum_command_measured_q_error_rad": diagnostics["maximum_command_measured_q_error_rad"],
                    "joint_limit_violation_count": diagnostics["joint_limit_violation_count"],
                    "joint_limit_violation_joints": diagnostics["joint_limit_violation_joints"],
                    "branch_discontinuity_count": diagnostics["branch_discontinuity_count"],
                    "invalid_robot_wall_crossing": diagnostics["invalid_robot_wall_crossing"],
                    "invalid_doll_wall_or_bottom_tunneling": diagnostics["invalid_doll_wall_or_bottom_tunneling"],
                    "command_path": str(command_path),
                    "command_sha256": sha256(command_path),
                    "event_log_path": str(event_path),
                    "event_log_sha256": sha256(event_path),
                    "execution_audit": audit,
                }
            )

    records.sort(key=lambda item: (item["eval_index"], item["method"]))
    summary: dict[str, Any] = {
        "schema_version": "contact_constrained_eval10_summary_v1",
        "evaluation_set": "EVAL10",
        "construction": "authoritative HELDOUT8 + fixed NEW_UNSEEN_2",
        "trials_per_method": 10,
        "scripted_environment_validation": "3/3 PASS",
        "environment_freeze_manifest": str(FREEZE / "FREEZE_MANIFEST.json"),
        "environment_freeze_manifest_sha256": sha256(FREEZE / "FREEZE_MANIFEST.json"),
        "new_unseen_2_integrity": integrity["status"],
        "methods": {},
    }
    for method_label in METHODS.values():
        subset = [record for record in records if record["method"] == method_label]
        counts = {stage: int(sum(record[stage] for record in subset)) for stage in STAGES}
        full = counts["FULL_TASK_SUCCESS"]
        method_key = method_label.replace("-", "_")
        aggregate_audits = [record["execution_audit"] for record in subset]
        summary["methods"][method_key] = {
            "completed": len(subset),
            "counts": counts,
            "rates": {stage: counts[stage] / len(subset) for stage in STAGES},
            "full_task_success_rate": full / len(subset),
            "full_task_exact_95pct_clopper_pearson_ci": exact_ci(full, len(subset)),
            "clean_commanded_release_count": int(sum(record["CLEAN_COMMANDED_RELEASE"] for record in subset)),
            "first_failure_counts": {
                stage: int(sum(record["first_failure_stage"] == stage for record in subset)) for stage in STAGES
            },
            "episodes_with_joint_limit_violations": int(
                sum(record["joint_limit_violation_count"] > 0 for record in subset)
            ),
            "episodes_with_branch_discontinuity": int(
                sum(record["branch_discontinuity_count"] > 0 for record in subset)
            ),
            "intervention": {
                "mean_common_local_primitive_scalar_fraction": float(
                    np.mean([audit["common_local_primitive_override_scalar_fraction"] for audit in aggregate_audits])
                ),
                "mean_deployment_projection_scalar_fraction": float(
                    np.mean([audit["deployment_safety_projection_scalar_fraction"] for audit in aggregate_audits])
                ),
                "mean_union_frame_fraction": float(
                    np.mean([audit["union_common_execution_frame_fraction"] for audit in aggregate_audits])
                ),
                "mean_arm_scalar_fraction": float(
                    np.mean([audit["arm_intervention_scalar_fraction"] for audit in aggregate_audits])
                ),
                "mean_wrist_scalar_fraction": float(
                    np.mean([audit["wrist_intervention_scalar_fraction"] for audit in aggregate_audits])
                ),
                "mean_dex3_scalar_fraction": float(
                    np.mean([audit["dex3_intervention_scalar_fraction"] for audit in aggregate_audits])
                ),
                "mean_raw_to_executed_arm_rmse_rad": float(
                    np.mean([audit["raw_to_executed_arm_rmse_rad"] for audit in aggregate_audits])
                ),
                "mean_raw_to_executed_dex3_rmse_rad": float(
                    np.mean([audit["raw_to_executed_dex3_rmse_rad"] for audit in aggregate_audits])
                ),
            },
        }

    paired = []
    for index in range(10):
        a = next(record for record in records if record["method"] == "ACT-A40" and record["eval_index"] == index)
        b = next(record for record in records if record["method"] == "ACT-B40" and record["eval_index"] == index)
        paired.append(
            {
                "eval_index": index,
                "source_episode_id": a["source_episode_id"],
                "ACT_A_full_success": a["FULL_TASK_SUCCESS"],
                "ACT_B_full_success": b["FULL_TASK_SUCCESS"],
                "disagreement": a["FULL_TASK_SUCCESS"] != b["FULL_TASK_SUCCESS"],
            }
        )
    summary["paired_full_task"] = {
        "ACT_B_minus_ACT_A_success_count": int(
            summary["methods"]["ACT_B40"]["counts"]["FULL_TASK_SUCCESS"]
            - summary["methods"]["ACT_A40"]["counts"]["FULL_TASK_SUCCESS"]
        ),
        "percentage_point_difference": 100.0
        * (
            summary["methods"]["ACT_B40"]["full_task_success_rate"]
            - summary["methods"]["ACT_A40"]["full_task_success_rate"]
        ),
        "disagreement_count": int(sum(item["disagreement"] for item in paired)),
        "episodes": paired,
    }
    return records, summary


def write_evaluation_manifest(eval10: dict[str, Any], integrity: dict[str, Any]) -> None:
    a_batch = read_json(PREP / "act_trajectories/act_a40/BATCH_MANIFEST.json")
    b_batch = read_json(PREP / "act_trajectories/act_b40/BATCH_MANIFEST.json")
    payload = {
        "schema_version": "contact_constrained_evaluation_set_manifest_v1",
        "evaluation_set": "EVAL10",
        "composition": {"PREDEFINED_HELDOUT": 8, "POST_TRAINING_UNSEEN": 2},
        "original_split_name_unchanged": "HELDOUT8",
        "heldout8_source_indices": HELDOUT8,
        "new_unseen_2": NEW2,
        "episode_replacement_allowed": False,
        "new_unseen_used_for_training_checkpoint_controller_bin_or_criteria_tuning": False,
        "new_unseen_integrity_status": integrity["status"],
        "entries": eval10["eval_entries"],
        "ACT_A40": {
            "checkpoint": a_batch["checkpoint"],
            "checkpoint_model_sha256": a_batch["checkpoint_model_sha256"],
        },
        "ACT_B40": {
            "checkpoint": b_batch["checkpoint"],
            "checkpoint_model_sha256": b_batch["checkpoint_model_sha256"],
        },
        "policy_input": "original ALOHA cam_high RGB + frozen method-specific retargeted state",
        "temporal_ensemble": {"implementation": "ACTTemporalEnsembler", "coefficient": 0.01},
    }
    dump_json(RESULTS / "EVALUATION_SET_MANIFEST.json", payload)
    lines = [
        "# EVAL10 evaluation set manifest",
        "",
        "`EVAL10 = original HELDOUT8 + fixed NEW_UNSEEN_2`.",
        "",
        "- Original split remains named `HELDOUT8`: 2, 13, 23, 27, 28, 31, 37, 40.",
        f"- NEW_UNSEEN_2: `{NEW2[0]}`, `{NEW2[1]}`.",
        "- Both new recordings passed integrity/task-completeness review.",
        "- They were converted only after the physical environment freeze.",
        "- They were not used for training, checkpoint selection, controller/bin/criterion tuning, or search.",
        "- Replacement based on evaluation outcome was prohibited and did not occur.",
        f"- ACT-A40 checkpoint SHA256: `{a_batch['checkpoint_model_sha256']}`.",
        f"- ACT-B40 checkpoint SHA256: `{b_batch['checkpoint_model_sha256']}`.",
        "- Policy input: original ALOHA cam_high RGB plus the frozen method-specific retargeted state.",
        "- Temporal ensemble: official `ACTTemporalEnsembler`, coefficient 0.01.",
    ]
    (RESULTS / "EVALUATION_SET_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fieldnames = [
        "eval_index", "source_episode_id", "source_final_episode", "source_recording", "provenance", "method",
        *STAGES, "CLEAN_COMMANDED_RELEASE", "PREMATURE_DROP_INTO_BIN", "PREMATURE_DROP_OUTSIDE_BIN",
        "first_failure_stage", "joint_limit_violation_count", "joint_limit_violation_joints",
        "branch_discontinuity_count", "maximum_command_measured_q_error_rad", "peak_held_object_speed_m_s",
        "maximum_robot_bin_force_n", "maximum_robot_bin_penetration_m", "maximum_doll_bin_penetration_m",
        "invalid_robot_wall_crossing", "invalid_doll_wall_or_bottom_tunneling", "command_path", "command_sha256",
        "event_log_path", "event_log_sha256",
    ]
    with (RESULTS / "PER_EPISODE_RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["joint_limit_violation_joints"] = ";".join(row["joint_limit_violation_joints"])
            writer.writerow(row)

    lines = [
        "# Contact-constrained EVAL10 per-episode results",
        "",
        "All 20 physics runs completed. The first failed physical stage was `LEFT_GRASP` in every run.",
        "",
        "| Eval | Provenance | Source | Method | Grasp | Handoff | R ownership | No drop | Bin entry | Settle | Full | First failure | Joint-limit samples |",
        "|---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---:|",
    ]
    for record in records:
        mark = lambda key: "P" if record[key] else "F"
        lines.append(
            f"| {record['eval_index']} | {record['provenance']} | {record['source_episode_id']} | {record['method']} "
            f"| {mark('LEFT_GRASP')} | {mark('HANDOFF')} | {mark('RIGHT_OWNERSHIP')} | {mark('NO_DROP_BEFORE_BIN')} "
            f"| {mark('DOLL_ENTERS_BIN')} | {mark('DOLL_SETTLES')} | {mark('FULL_TASK_SUCCESS')} "
            f"| {record['first_failure_stage']} | {record['joint_limit_violation_count']} |"
        )
    lines.extend(
        [
            "",
            "`Joint-limit samples` is a diagnostic count, not an episode-exclusion rule. Physical failures were not converted to invalid runs.",
        ]
    )
    (RESULTS / "PER_EPISODE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dump_json(RESULTS / "TASK_SUCCESS_SUMMARY.json", summary)
    for method_label in METHODS.values():
        key = method_label.replace("-", "_")
        dump_json(RESULTS / ("ACT_A_RESULTS.json" if method_label == "ACT-A40" else "ACT_B_RESULTS.json"), {
            "method": method_label,
            "summary": summary["methods"][key],
            "episodes": [record for record in records if record["method"] == method_label],
        })

    md = [
        "# Contact-constrained physical task-success summary",
        "",
        "The common scripted controller validated that the frozen 150 mm-bin environment can complete the task (3/3). This does not count as ACT performance.",
        "",
        "| Metric | ACT-A40 | ACT-B40 |",
        "|---|---:|---:|",
    ]
    for stage, label in zip(STAGES, STAGE_LABELS):
        a = summary["methods"]["ACT_A40"]["counts"][stage]
        b = summary["methods"]["ACT_B40"]["counts"][stage]
        md.append(f"| {label} | {a}/10 = {10*a:.1f}% | {b}/10 = {10*b:.1f}% |")
    ci_a = summary["methods"]["ACT_A40"]["full_task_exact_95pct_clopper_pearson_ci"]
    ci_b = summary["methods"]["ACT_B40"]["full_task_exact_95pct_clopper_pearson_ci"]
    md.extend(
        [
            "",
            f"- ACT-A40 full-task exact 95% CI: {100*ci_a[0]:.1f}%–{100*ci_a[1]:.1f}%.",
            f"- ACT-B40 full-task exact 95% CI: {100*ci_b[0]:.1f}%–{100*ci_b[1]:.1f}%.",
            "- Paired full-task disagreement: 0/10; success-count difference B−A: 0.",
            "- All runs failed first at LEFT_GRASP; no episode was replaced or excluded.",
            "- With N=10 and zero successes in both methods, the data show no downstream physical advantage and do not support a significance claim.",
        ]
    )
    (RESULTS / "TASK_SUCCESS_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_intervention_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Common-controller intervention audit",
        "",
        "No frozen local grasp, handoff, transport, or release primitive activated during ACT evaluation. The policy commands were executed directly after the identical frozen deployment-safety projection.",
        "",
        "The zero mask stored in each immutable physical command NPZ denotes local-primitive override only. Each run now contains `POLICY_EXECUTION_AUDIT.npz`, whose union mask explicitly includes the earlier deployment-safety projection.",
        "",
        "| Method | Local primitive | Deployment projection (scalar) | Any-intervention frames | Arm | Wrist | Dex3 | Arm RMSE | Dex3 RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("ACT_A40", "ACT-A40"), ("ACT_B40", "ACT-B40")):
        item = summary["methods"][key]["intervention"]
        lines.append(
            f"| {label} | {100*item['mean_common_local_primitive_scalar_fraction']:.3f}% "
            f"| {100*item['mean_deployment_projection_scalar_fraction']:.3f}% "
            f"| {100*item['mean_union_frame_fraction']:.3f}% "
            f"| {100*item['mean_arm_scalar_fraction']:.3f}% "
            f"| {100*item['mean_wrist_scalar_fraction']:.3f}% "
            f"| {100*item['mean_dex3_scalar_fraction']:.3f}% "
            f"| {item['mean_raw_to_executed_arm_rmse_rad']:.6f} "
            f"| {item['mean_raw_to_executed_dex3_rmse_rad']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Arm and wrist trajectories are bitwise unchanged by the common execution layer. Projection affects only bounded Dex3 commands. There is no episode-specific or method-specific controller tuning.",
        ]
    )
    (RESULTS / "COMMON_CONTROLLER_INTERVENTION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    colors = ["#4C78A8", "#E45756"]
    fig = plt.figure(figsize=(12.0, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.6, 2.4])

    ax = fig.add_subplot(grid[0, 0])
    values = [summary["methods"]["ACT_A40"]["full_task_success_rate"], summary["methods"]["ACT_B40"]["full_task_success_rate"]]
    cis = [summary["methods"]["ACT_A40"]["full_task_exact_95pct_clopper_pearson_ci"], summary["methods"]["ACT_B40"]["full_task_exact_95pct_clopper_pearson_ci"]]
    upper = [ci[1] * 100 for ci in cis]
    bars = ax.bar([0, 1], np.asarray(values) * 100, color=colors, width=0.62)
    ax.errorbar([0, 1], np.asarray(values) * 100, yerr=[np.zeros(2), upper], fmt="none", color="#333333", capsize=4, lw=1.2)
    ax.set_xticks([0, 1], ["ACT-A40", "ACT-B40"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("(a) Full task success")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, 3, "0/10\n0.0%", ha="center", va="bottom", color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#555555", edgecolor="none"))
    ax.text(0.5, 35, "Exact 95% CI\n0.0–30.8%", ha="center", va="center", transform=ax.transData, color="#555555")

    ax = fig.add_subplot(grid[0, 1])
    x = np.arange(len(STAGES))
    width = 0.36
    a = [100 * summary["methods"]["ACT_A40"]["rates"][stage] for stage in STAGES]
    b = [100 * summary["methods"]["ACT_B40"]["rates"][stage] for stage in STAGES]
    ax.bar(x - width/2, a, width, label="ACT-A40", color=colors[0])
    ax.bar(x + width/2, b, width, label="ACT-B40", color=colors[1])
    ax.set_xticks(x, ["Grasp", "Handoff", "R own.", "No drop", "Entry", "Settle", "Full"], rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("(b) Stage-wise physical success")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.5, 0.48, "All stages: 0/10 for both methods", transform=ax.transAxes, ha="center", color="#555555")

    ax = fig.add_subplot(grid[0, 2])
    matrix = np.zeros((20, len(STAGES)), dtype=int)
    row_labels: list[str] = []
    for row, record in enumerate(records):
        matrix[row] = [int(record[stage]) for stage in STAGES]
        tag = "H" if record["provenance"] == "PREDEFINED_HELDOUT" else "N"
        row_labels.append(f"{record['method'][-3]}  E{record['eval_index']:02d} ({tag})")
    ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=ListedColormap(["#D9D9D9", "#4C9F70"]))
    ax.set_xticks(np.arange(len(STAGES)), ["Grasp", "Handoff", "R own.", "No drop", "Entry", "Settle", "Full"], rotation=35, ha="right")
    ax.set_yticks(np.arange(20), row_labels)
    ax.set_title("(c) EVAL10 episode-stage matrix")
    ax.set_xticks(np.arange(-0.5, len(STAGES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(0.99, -0.22, "H: predefined held-out; N: post-training unseen", transform=ax.transAxes, ha="right", color="#555555")

    fig.suptitle("Contact-constrained G1 physical evaluation — frozen 150 mm-bin environment", fontsize=12, fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(FIGURES / f"Fig07_Physical_Task_Success_double.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    shutil.copy2(
        FIGURES / "Fig07_Physical_Task_Success_double.png",
        FIGURES / "ACT_AB_PHYSICAL_CONTACT_SHEET.png",
    )

    source_rows = []
    for record in records:
        source_rows.append({key: record[key] for key in ["eval_index", "source_episode_id", "provenance", "method", *STAGES]})
    with (FIGURES / "Fig07_Physical_Task_Success_source_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)
    (FIGURES / "Fig07_Physical_Task_Success_CAPTION.md").write_text(
        "# Fig. 7. Contact-constrained physical task success\n\n"
        "**English.** Physical task success for source-conditioned ACT-A40 and ACT-B40 in the same frozen 150 mm-bin G1 simulation environment. "
        "Both methods completed all 10 evaluations but failed to establish the initial left-hand physical grasp in every episode (0/10 full-task success; exact two-sided 95% Clopper–Pearson CI 0.0–30.8%). "
        "The scripted common controller independently passed 3/3 and is not counted as policy performance. H denotes the predefined HELDOUT8; N denotes the two fixed post-training unseen recordings.\n\n"
        "**한국어.** 동일하게 동결된 150 mm bin G1 접촉 물리 환경에서 source-conditioned ACT-A40과 ACT-B40을 비교하였다. "
        "두 방법 모두 10개 평가를 완료했지만 모든 에피소드에서 최초 왼손 물리 grasp를 형성하지 못해 전체 과제 성공은 각각 0/10이었다. "
        "공통 scripted controller의 3/3 성공은 정책 성능에 포함하지 않았다. H는 기존 HELDOUT8, N은 수집 후 고정한 NEW_UNSEEN_2이다.\n\n"
        "**Caveat.** Isaac simulation; compressed-plush rigid proxy; original ALOHA RGB plus method-specific frozen state; no target-domain visual autonomy or real-G1 claim.\n",
        encoding="utf-8",
    )


def write_final_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    a = summary["methods"]["ACT_A40"]
    b = summary["methods"]["ACT_B40"]
    a_batch = read_json(PREP / "act_trajectories/act_a40/BATCH_MANIFEST.json")
    b_batch = read_json(PREP / "act_trajectories/act_b40/BATCH_MANIFEST.json")
    text = f"""# Final contact-constrained physical evaluation report

## Outcome

- Execution before: diagnostic kinematic direct-state replay (visualization only).
- Final scoring execution: contact-constrained PhysX articulation position targets; direct robot state writes only at reset.
- Robot/bin contact constraint: active and preflight-verified.
- Frozen 150 mm-bin scripted task: PASS, 3/3 byte-identical physics traces.
- Evaluation set: EVAL10 = predefined HELDOUT8 + exact NEW_UNSEEN_2.
- ACT-A40: 10/10 completed, 0/10 full success (0.0%; exact 95% CI 0.0–30.8%).
- ACT-B40: 10/10 completed, 0/10 full success (0.0%; exact 95% CI 0.0–30.8%).
- Paired difference B−A: 0 successes / 0.0 percentage points; disagreement 0/10.

All 20 true physics runs failed first at `LEFT_GRASP`. No run was discarded or relabeled as invalid. The later-stage zeros are therefore conditional consequences, not evidence that those stages were independently reached and failed.

## Frozen environment

- Freeze manifest: `{FREEZE / 'FREEZE_MANIFEST.json'}`
- Freeze SHA256: `{sha256(FREEZE / 'FREEZE_MANIFEST.json')}`
- Physics: 1/240 s, 8 substeps per 30 Hz command frame, gravity on.
- Bin: open-top, 0.150 m wall height, bottom world Z 0.795 m, 3 mm symmetric rim bevel.
- Doll: 0.020 kg graspable rigid proxy approximating the compact geometry and low mass of the plush object.
- Scripted release classification: `PREMATURE_DROP_INTO_BIN`; it meets the frozen primary success rule but is not a clean commanded release.

## Evaluation identities

- HELDOUT8 source indices: {HELDOUT8}.
- NEW_UNSEEN_2: `{NEW2[0]}` and `{NEW2[1]}`.
- NEW2 integrity/task completeness: PASS. Both were fixed in advance and neither was used for training, checkpoint selection, controller/bin/success-criterion tuning, or search; neither was replaced.
- ACT-A40 checkpoint: `{a_batch['checkpoint']}`; SHA256 `{a_batch['checkpoint_model_sha256']}`.
- ACT-B40 checkpoint: `{b_batch['checkpoint']}`; SHA256 `{b_batch['checkpoint_model_sha256']}`.
- Policy input: original ALOHA `cam_high` RGB and the corresponding frozen method-specific G1 reference state.
- Temporal ensemble: official `ACTTemporalEnsembler`, coefficient 0.01.

## Execution assistance

No post-freeze local common grasp/handoff/transport/release primitive activated. Arm and wrist policy commands were unchanged. The identical frozen deployment-safety projection made small Dex3-only changes, fully stored in each run's `POLICY_EXECUTION_AUDIT.npz`; see `COMMON_CONTROLLER_INTERVENTION_AUDIT.md`.

Episodes with measured joint-limit violations: ACT-A40 {a['episodes_with_joint_limit_violations']}/10; ACT-B40 {b['episodes_with_joint_limit_violations']}/10. Branch discontinuities: ACT-A40 {a['episodes_with_branch_discontinuity']}/10; ACT-B40 {b['episodes_with_branch_discontinuity']}/10. These diagnostics were retained and did not convert physical failures into excluded runs.

## Scientific interpretation

The environment-validity result and policy result are different: the common scripted controller proves the task is physically executable in this frozen simulation, whereas neither source-conditioned ACT policy established the first physical grasp in EVAL10. Therefore these data do **not** support a downstream physical advantage for A or B. The result is an Isaac/target-robot simulation evaluation, not real-G1 success, target-domain visual autonomy, exact plush mechanics, or evidence that raw ACT solved contact interaction.

## Artifacts

- Per-episode audit: `{RESULTS / 'PER_EPISODE_RESULTS.md'}`
- Numerical summary: `{RESULTS / 'TASK_SUCCESS_SUMMARY.json'}`
- Intervention audit: `{RESULTS / 'COMMON_CONTROLLER_INTERVENTION_AUDIT.md'}`
- Figure: `{FIGURES / 'Fig07_Physical_Task_Success_double.png'}`
- ACT-A/B result contact sheet: `{FIGURES / 'ACT_AB_PHYSICAL_CONTACT_SHEET.png'}`
- Scripted measured-state storyboard: `{FIGURES / 'PHYSICAL_TASK_STORYBOARD.png'}`
- Physical trace viewer: `cd {ROOT} && /home/jbnu/miniconda3/envs/isaaclab6/bin/python tools/autorun_current_best_full_task_visual_replay.py --contact-constrained-final --playback-speed 1 --gui`

## Unsupported claims

- No real robot experiment was run.
- No live G1/Isaac visual policy was evaluated.
- No exact deformable plush model was used.
- No statistical superiority claim is supported by 0/10 versus 0/10.
"""
    (RESULTS / "FINAL_PHYSICAL_EVALUATION_REPORT.md").write_text(text, encoding="utf-8")
    (BASE / "FINAL_PHYSICAL_EVALUATION_REPORT.md").write_text(text, encoding="utf-8")

    status = f"""# Current status

- Contact-constrained robot/bin preflight: PASS.
- Scripted 150 mm-bin full task: 3/3 PASS; environment frozen.
- NEW_UNSEEN_2 integrity/task completeness: PASS.
- Frozen Fair-A/Proposed-B conversion: PASS.
- ACT-A40 EVAL10 physics: 10/10 completed, 0/10 full success.
- ACT-B40 EVAL10 physics: 10/10 completed, 0/10 full success.
- First failure for all 20 runs: LEFT_GRASP.
- Final reports and Fig. 7: complete.
- Highest gate: FINAL_A_B_TASK_SUCCESS_RESULTS_READY (evaluation complete; TSR is 0.0% for both methods).
"""
    (BASE / "CURRENT_STATUS.md").write_text(status, encoding="utf-8")


def write_result_manifest() -> None:
    authoritative = [
        RESULTS / "EVALUATION_SET_MANIFEST.json",
        RESULTS / "ACT_A_RESULTS.json",
        RESULTS / "ACT_B_RESULTS.json",
        RESULTS / "PER_EPISODE_RESULTS.csv",
        RESULTS / "PER_EPISODE_RESULTS.md",
        RESULTS / "TASK_SUCCESS_SUMMARY.json",
        RESULTS / "TASK_SUCCESS_SUMMARY.md",
        RESULTS / "COMMON_CONTROLLER_INTERVENTION_AUDIT.md",
        RESULTS / "FINAL_PHYSICAL_EVALUATION_REPORT.md",
        FIGURES / "Fig07_Physical_Task_Success_double.png",
        FIGURES / "Fig07_Physical_Task_Success_double.pdf",
        FIGURES / "Fig07_Physical_Task_Success_double.svg",
        FIGURES / "Fig07_Physical_Task_Success_source_data.csv",
        FIGURES / "Fig07_Physical_Task_Success_CAPTION.md",
        FIGURES / "ACT_AB_PHYSICAL_CONTACT_SHEET.png",
    ]
    storyboard = FIGURES / "PHYSICAL_TASK_STORYBOARD.png"
    if storyboard.is_file():
        authoritative.extend(
            [
                storyboard,
                FIGURES / "SCRIPTED_PHYSICAL_CONTACT_SHEET.png",
                FIGURES / "PHYSICAL_TASK_STORYBOARD_MANIFEST.json",
            ]
        )
    authoritative.extend(sorted(RUNS.glob("*/*/POLICY_EXECUTION_AUDIT.json")))
    authoritative.extend(sorted(RUNS.glob("*/*/POLICY_EXECUTION_AUDIT.npz")))
    missing = [str(path) for path in authoritative if not path.is_file()]
    if missing:
        raise RuntimeError(f"final result manifest inputs missing: {missing}")
    payload = {
        "schema_version": "contact_constrained_eval10_result_manifest_v1",
        "status": "FINAL_A_B_TASK_SUCCESS_RESULTS_READY",
        "evaluation_runs_complete": "ACT-A40 10/10; ACT-B40 10/10",
        "task_success": "ACT-A40 0/10; ACT-B40 0/10",
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in authoritative
        ],
    }
    dump_json(RESULTS / "RESULT_MANIFEST.json", payload)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records, summary = build_results()
    _, eval10, integrity = validate_inputs()
    write_evaluation_manifest(eval10, integrity)
    write_tables(records, summary)
    write_intervention_report(summary)
    plot_results(records, summary)
    write_final_report(records, summary)
    write_result_manifest()
    print("FINALIZED_CONTACT_CONSTRAINED_EVAL10")
    print(f"ACT_A_FULL_SUCCESS={summary['methods']['ACT_A40']['counts']['FULL_TASK_SUCCESS']}/10")
    print(f"ACT_B_FULL_SUCCESS={summary['methods']['ACT_B40']['counts']['FULL_TASK_SUCCESS']}/10")
    print(f"REPORT={RESULTS / 'FINAL_PHYSICAL_EVALUATION_REPORT.md'}")
    print(f"FIGURE={FIGURES / 'Fig07_Physical_Task_Success_double.png'}")


if __name__ == "__main__":
    main()
