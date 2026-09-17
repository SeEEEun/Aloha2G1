#!/usr/bin/env python3
"""Finalize the paired-rehearsal Policy-B G1-visual experiment.

This is an analysis-only tool.  It reads the already-completed fixed phase
probes and checkpoints, verifies the projection-only checkpoint provenance,
and writes the dual-domain gate and handoff-approach evidence.  It never
modifies either training dataset or any checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
import torch

import probe_policy_b_dataset_phases as base
import probe_policy_b_g1visual_phases as g1probe


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/policy_b_g1visual/rehearsal/final_analysis"
PROBE_ROOT = ROOT / "outputs/policy_b_g1visual/rehearsal/probes"
DUAL_RESULTS = PROBE_ROOT / "dual_domain_checkpoint_results.json"
QA = ROOT / "outputs/policy_b_g1visual/root_cause_diagnostic/phase_visual_qa/validation.json"
ABC = ROOT / "outputs/policy_b_g1visual/root_cause_diagnostic/abc_comparison/comparison.json"
ORIGINAL_MODEL = (
    ROOT
    / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2"
    / "checkpoints/020000/pretrained_model/model.safetensors"
)
R2_MODEL = (
    ROOT
    / "outputs/policy_b_g1visual/rehearsal/training"
    / "R2_paired_visual_connector_001500/checkpoints/001500/pretrained_model/model.safetensors"
)
R2_AUDIT = ROOT / "outputs/policy_b_g1visual/rehearsal/training/R2_trainable_parameter_audit.json"
EXPECTED_CONNECTOR = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
R1_PREFIXES = (
    "model.vlm_with_expert.lm_expert.",
    "model.state_proj.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)

CONDITIONS = {
    "A_ORIGINAL_POLICY_ALOHA_RGB": (
        ROOT / "outputs/policy_b_offline_phase_probe/probe_predictions.npz",
        "policy_prediction_A",
    ),
    "B_ORIGINAL_POLICY_G1_RGB": (
        ROOT
        / "outputs/policy_b_g1visual/root_cause_diagnostic"
        / "old_policy_on_g1visual/probe_predictions.npz",
        "policy_prediction",
    ),
    "C_G1_ONLY_5K_POLICY_G1_RGB": (
        ROOT / "outputs/policy_b_g1visual/offline_phase_probe/probe_predictions.npz",
        "policy_prediction",
    ),
    "D_R2_1500_REHEARSAL_POLICY_G1_RGB": (
        PROBE_ROOT / "R2/001500/G1VISUAL/probe_predictions.npz",
        "policy_prediction",
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_probe(path: Path, prediction_key: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "prediction": np.asarray(data[prediction_key], dtype=np.float32),
        "target": np.asarray(data["authoritative_target"], dtype=np.float32),
        "episode": np.asarray(data["episode_index"], dtype=np.int64),
        "frame": np.asarray(data["frame_index"], dtype=np.int64),
        "phase": np.asarray(data["phase"]).astype("U40"),
        "joints": np.asarray(data["joint_names"]).astype("U64"),
    }


def compare_r2_tensors() -> dict[str, Any]:
    """Verify that the R2 checkpoint changed only the selected connector tensor."""
    with safe_open(ORIGINAL_MODEL, framework="pt", device="cpu") as original, safe_open(
        R2_MODEL, framework="pt", device="cpu"
    ) as adapted:
        original_keys = set(original.keys())
        adapted_keys = set(adapted.keys())
        if original_keys != adapted_keys:
            raise RuntimeError("R2 state-dict key set differs from original Policy B")
        changed: list[dict[str, Any]] = []
        unchanged_count = 0
        for name in sorted(original_keys):
            before = original.get_tensor(name)
            after = adapted.get_tensor(name)
            if before.shape != after.shape or before.dtype != after.dtype:
                raise RuntimeError(f"R2 tensor schema changed: {name}")
            if torch.equal(before, after):
                unchanged_count += 1
                continue
            delta = (after.float() - before.float()).reshape(-1)
            changed.append(
                {
                    "name": name,
                    "shape": list(before.shape),
                    "scalar_count": int(before.numel()),
                    "maximum_absolute_delta": float(delta.abs().max()),
                    "root_mean_square_delta": float(torch.sqrt(torch.mean(delta.square()))),
                }
            )
    return {
        "status": "PASS" if [row["name"] for row in changed] == [EXPECTED_CONNECTOR] else "FAIL",
        "original_model": str(ORIGINAL_MODEL),
        "original_model_sha256": sha256_file(ORIGINAL_MODEL),
        "adapted_model": str(R2_MODEL),
        "adapted_model_sha256": sha256_file(R2_MODEL),
        "state_dict_tensor_count": len(original_keys),
        "unchanged_tensor_count": unchanged_count,
        "changed_tensors": changed,
        "expected_only_changed_tensor": EXPECTED_CONNECTOR,
    }


def enumerate_r1_trainables() -> dict[str, Any]:
    """Resolve the R1 requires-grad selectors to exact saved tensor names."""
    with safe_open(ORIGINAL_MODEL, framework="pt", device="cpu") as model:
        names = [
            name
            for name in sorted(model.keys())
            if any(name.startswith(prefix) for prefix in R1_PREFIXES)
            and "lm_head" not in name
        ]
        scalars = int(sum(math.prod(model.get_slice(name).get_shape()) for name in names))
    audit = {
        "status": "PASS" if scalars == 99_880_992 else "FAIL",
        "source_implementation": {
            "expert_requires_grad": "/home/jbnu/lerobot-smolvla/src/lerobot/policies/smolvla/smolvlm_with_expert.py:152",
            "state_projection_requires_grad": "/home/jbnu/lerobot-smolvla/src/lerobot/policies/smolvla/modeling_smolvla.py:539",
        },
        "resolved_prefixes": list(R1_PREFIXES),
        "excluded_name_substring": "lm_head",
        "trainable_tensor_count": len(names),
        "trainable_scalar_count": scalars,
        "trainable_tensor_names": names,
    }
    atomic_json(OUT / "R1_trainable_parameter_audit.json", audit)
    atomic_text(OUT / "R1_trainable_parameter_names.txt", "\n".join(names) + "\n")
    return audit


def final_logged_loss(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(
        r"loss:([0-9.]+).*?losses_after_forward:([0-9.]+).*?"
        r"losses_after_in_ep_bound:([0-9.]+).*?losses_after_rm_padding:([0-9.]+)",
        text,
    )
    if not rows:
        raise RuntimeError(f"no finite training-loss record in {path}")
    values = [float(value) for value in rows[-1]]
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite final training loss in {path}")
    return {
        "logged_loss": values[0],
        "after_forward": values[1],
        "after_episode_boundary_mask": values[2],
        "after_padding_mask": values[3],
    }


def phase_count(result: dict[str, Any], phase: str) -> int:
    groups = result["relevant_evidence"][phase]
    return min(int(value["episode_pass_count"]) for value in groups.values())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qa = read_json(QA)
    abc = read_json(ABC)
    dual = read_json(DUAL_RESULTS)
    r2_declared = read_json(R2_AUDIT)
    if qa["status"] != "PASS" or qa["manual_visual_review"]["status"] != "PASS":
        raise RuntimeError("G1-visual phase-image QA is not PASS")
    if abc["root_cause_classification"] != "VISUAL_GAP_DOMINANT":
        raise RuntimeError("root-cause prerequisite changed")

    abc_phase_counts: dict[str, dict[str, int]] = {}
    for condition, phase_results in abc["phase_results"].items():
        abc_phase_counts[condition] = {}
        for phase, row in phase_results.items():
            abc_phase_counts[condition][phase] = min(
                int(group["pass_count"]) for group in row["evidence"].values()
            )

    checkpoint_rows = dual["results"]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in checkpoint_rows:
        grouped.setdefault((row["experiment"], int(row["step"])), {})[row["domain"]] = row

    # Gate: preserve all nine ALOHA phase groups and strictly improve on the 6/9
    # untouched-policy G1 baseline.  No checkpoint meets it.
    checkpoint_summary: list[dict[str, Any]] = []
    for (experiment, step), domains in sorted(grouped.items()):
        aloha, g1 = domains["ALOHA"], domains["G1VISUAL"]
        checkpoint_summary.append(
            {
                "experiment": experiment,
                "step": step,
                "checkpoint": aloha["checkpoint"],
                "model_sha256": aloha["model_sha256"],
                "aloha_phase_score": int(aloha["phase_score"]),
                "g1visual_phase_score": int(g1["phase_score"]),
                "aloha_arm_rmse_rad": float(aloha["arm_full_chunk_rmse_rad"]),
                "g1visual_arm_rmse_rad": float(g1["arm_full_chunk_rmse_rad"]),
                "aloha_dex3_rmse_rad": float(aloha["dex3_full_chunk_rmse_rad"]),
                "g1visual_dex3_rmse_rad": float(g1["dex3_full_chunk_rmse_rad"]),
                "aloha_handoff_pass_count": phase_count(aloha, "handoff_approach"),
                "g1visual_handoff_pass_count": phase_count(g1, "handoff_approach"),
                "dual_domain_gate": bool(
                    aloha["phase_score"] == 9 and g1["phase_score"] > 6
                ),
            }
        )
    if any(row["dual_domain_gate"] for row in checkpoint_summary):
        raise RuntimeError("analysis assumption invalid: a rehearsal checkpoint now meets the gate")

    # R2/1500 is retained only as the best Pareto analysis artifact.  It has the
    # lowest original-domain arm and hand RMSE among rehearsal checkpoints and
    # retains the untouched G1 baseline's 2/6 handoff count, but is not approved.
    selected = next(
        row
        for row in checkpoint_summary
        if row["experiment"] == "R2" and row["step"] == 1500
    )

    probes = {name: load_probe(*source) for name, source in CONDITIONS.items()}
    reference = probes["A_ORIGINAL_POLICY_ALOHA_RGB"]
    for condition, probe in probes.items():
        for key in ("target", "episode", "frame", "phase", "joints"):
            if not np.array_equal(probe[key], reference[key]):
                raise RuntimeError(f"{condition}: probe identity mismatch in {key}")

    # Consolidate the requested first-action, prefix-4, phase-detection, and
    # four-group motion metrics for all baselines and every saved checkpoint.
    all_probes = dict(probes)
    for row in checkpoint_rows:
        condition = f"{row['experiment']}_{int(row['step']):06d}_{row['domain']}"
        all_probes[condition] = load_probe(
            Path(row["probe_output"]) / "probe_predictions.npz", "policy_prediction"
        )
    for condition, probe in all_probes.items():
        for key in ("target", "episode", "frame", "phase", "joints"):
            if not np.array_equal(probe[key], reference[key]):
                raise RuntimeError(f"{condition}: consolidated probe mismatch in {key}")
    all_metric_rows: list[dict[str, Any]] = []
    for condition, probe in all_probes.items():
        for phase, phase_label in base.PHASE_LABELS.items():
            sample_indices = np.flatnonzero(probe["phase"] == phase)
            sample_passes: list[bool] = []
            for sample_index in sample_indices:
                relevant_passes = []
                for relevant_group in g1probe.RELEVANT_GROUPS[phase]:
                    metric = base.group_metrics(
                        probe["prediction"][sample_index],
                        probe["target"][sample_index],
                        base.GROUPS[relevant_group],
                    )
                    relevant_passes.append(
                        metric["trajectory_motion_cosine"] is not None
                        and metric["trajectory_motion_cosine"]
                        >= g1probe.ESTABLISHED_COSINE_GATE
                        and metric["target_direction_projection_progress"] is not None
                        and metric["target_direction_projection_progress"]
                        >= g1probe.ESTABLISHED_DIRECTION_PROGRESS_GATE
                    )
                sample_passes.append(all(relevant_passes))
            for group, joint_indices in base.GROUPS.items():
                metrics = [
                    base.group_metrics(
                        probe["prediction"][sample_index],
                        probe["target"][sample_index],
                        joint_indices,
                    )
                    for sample_index in sample_indices
                ]
                all_metric_rows.append(
                    {
                        "condition": condition,
                        "phase": phase,
                        "phase_label": phase_label,
                        "joint_group": group,
                        "phase_behavior_pass_count": int(sum(sample_passes)),
                        "phase_behavior_total": len(sample_passes),
                        "phase_behavior_present": all(sample_passes),
                        "first_action_rmse_rad_mean": float(
                            np.mean([metric["first_action_rmse_rad"] for metric in metrics])
                        ),
                        "prefix4_rmse_rad_mean": float(
                            np.mean([metric["prefix4_rmse_rad"] for metric in metrics])
                        ),
                        "full_chunk_rmse_rad_mean": float(
                            np.mean([metric["full_chunk_rmse_rad"] for metric in metrics])
                        ),
                        "predicted_motion_magnitude_rad_rms_mean": float(
                            np.mean(
                                [
                                    metric["predicted_motion_magnitude_rad_rms"]
                                    for metric in metrics
                                ]
                            )
                        ),
                        "target_motion_magnitude_rad_rms_mean": float(
                            np.mean(
                                [metric["target_motion_magnitude_rad_rms"] for metric in metrics]
                            )
                        ),
                    }
                )
    all_metrics_path = OUT / "all_conditions_phase_group_metrics.csv"
    with all_metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_metric_rows[0]))
        writer.writeheader()
        writer.writerows(all_metric_rows)

    right = base.GROUPS["right_arm"]
    handoff_indices = np.flatnonzero(reference["phase"] == "handoff_approach")
    if len(handoff_indices) != 6:
        raise RuntimeError("expected six matched handoff-approach probes")
    handoff_rows: list[dict[str, Any]] = []
    handoff_summary: dict[str, Any] = {}
    plot_dir = OUT / "handoff_right_arm_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "A_ORIGINAL_POLICY_ALOHA_RGB": "#3776d6",
        "B_ORIGINAL_POLICY_G1_RGB": "#e58b25",
        "C_G1_ONLY_5K_POLICY_G1_RGB": "#4d9e64",
        "D_R2_1500_REHEARSAL_POLICY_G1_RGB": "#a154b5",
    }
    for sample_index in handoff_indices:
        target = reference["target"][sample_index]
        episode = int(reference["episode"][sample_index])
        frame = int(reference["frame"][sample_index])
        fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
        for local_joint, axis in enumerate(axes.flat[:7]):
            joint_index = int(right[local_joint])
            axis.plot(target[:, joint_index], color="black", linewidth=2.2, label="target")
            for condition, probe in probes.items():
                axis.plot(
                    probe["prediction"][sample_index, :, joint_index],
                    color=colors[condition],
                    alpha=0.9,
                    label=condition.split("_")[0],
                )
            axis.axvspan(0, 3, color="#dddddd", alpha=0.35)
            axis.set_title(str(reference["joints"][joint_index]))
            axis.set_xlabel("action-chunk row")
            axis.set_ylabel("absolute joint target [rad]")
            axis.grid(alpha=0.2)
        axes.flat[7].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=5)
        fig.suptitle(f"Handoff approach | episode {episode:02d} | frame {frame:04d}")
        figure_path = plot_dir / f"episode_{episode:02d}_frame_{frame:04d}.png"
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)

        target_prefix_displacement = float(np.linalg.norm(target[3, right] - target[0, right]))
        for condition, probe in probes.items():
            prediction = probe["prediction"][sample_index]
            metrics = base.group_metrics(prediction, target, right)
            cosine = metrics["trajectory_motion_cosine"]
            progress = metrics["target_direction_projection_progress"]
            behavior = bool(
                cosine is not None
                and cosine >= g1probe.ESTABLISHED_COSINE_GATE
                and progress is not None
                and progress >= g1probe.ESTABLISHED_DIRECTION_PROGRESS_GATE
            )
            handoff_rows.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "condition": condition,
                    "target_right_arm_prefix4_net_displacement_rad_l2": target_prefix_displacement,
                    "predicted_right_arm_prefix4_net_displacement_rad_l2": float(
                        np.linalg.norm(prediction[3, right] - prediction[0, right])
                    ),
                    "right_arm_first_action_rmse_rad": metrics["first_action_rmse_rad"],
                    "right_arm_prefix4_rmse_rad": metrics["prefix4_rmse_rad"],
                    "right_arm_full_chunk_rmse_rad": metrics["full_chunk_rmse_rad"],
                    "right_arm_predicted_motion_magnitude_rad_rms": metrics[
                        "predicted_motion_magnitude_rad_rms"
                    ],
                    "right_arm_target_motion_magnitude_rad_rms": metrics[
                        "target_motion_magnitude_rad_rms"
                    ],
                    "trajectory_motion_cosine": cosine,
                    "target_direction_projection_progress": progress,
                    "phase_behavior_present": behavior,
                    "trajectory_plot": str(figure_path),
                }
            )

    for condition in CONDITIONS:
        rows = [row for row in handoff_rows if row["condition"] == condition]
        handoff_summary[condition] = {
            "phase_behavior_pass_count": int(sum(row["phase_behavior_present"] for row in rows)),
            "phase_behavior_total": 6,
            "target_right_arm_prefix4_net_displacement_rad_l2_mean": float(
                np.mean([row["target_right_arm_prefix4_net_displacement_rad_l2"] for row in rows])
            ),
            "predicted_right_arm_prefix4_net_displacement_rad_l2_mean": float(
                np.mean([row["predicted_right_arm_prefix4_net_displacement_rad_l2"] for row in rows])
            ),
            "right_arm_prefix4_rmse_rad_mean": float(
                np.mean([row["right_arm_prefix4_rmse_rad"] for row in rows])
            ),
            "trajectory_motion_cosine_mean": float(
                np.mean([row["trajectory_motion_cosine"] for row in rows])
            ),
            "target_direction_projection_progress_mean": float(
                np.mean([row["target_direction_projection_progress"] for row in rows])
            ),
        }

    with (OUT / "handoff_approach_right_arm.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(handoff_rows[0]))
        writer.writeheader()
        writer.writerows(handoff_rows)

    with (OUT / "dual_domain_checkpoint_gate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoint_summary[0]))
        writer.writeheader()
        writer.writerows(checkpoint_summary)

    tensor_audit = compare_r2_tensors()
    r1_parameter_audit = enumerate_r1_trainables()
    if (
        tensor_audit["status"] != "PASS"
        or r2_declared["status"] != "PASS"
        or r1_parameter_audit["status"] != "PASS"
    ):
        raise RuntimeError("rehearsal trainable/checkpoint provenance audit failed")

    training = {
        "R1": {
            "steps": 1500,
            "final_loss": final_logged_loss(
                ROOT / "outputs/policy_b_g1visual/rehearsal/training/R1_console.log"
            ),
            "trainable_components": [
                "action expert (lm_expert, excluding lm_head)",
                "state projection",
                "action input projection",
                "action output projection",
                "action-time input/output MLP",
            ],
            "trainable_scalar_count": 99_880_992,
            "parameter_audit": r1_parameter_audit,
        },
        "R2": {
            "steps": 1500,
            "final_loss": final_logged_loss(
                ROOT / "outputs/policy_b_g1visual/rehearsal/training/R2_console.log"
            ),
            "trainable_components": [EXPECTED_CONNECTOR],
            "trainable_scalar_count": 11_796_480,
            "tensor_provenance": tensor_audit,
        },
    }

    result = {
        "status": "COMPLETE",
        "root_cause_classification": "VISUAL_GAP_DOMINANT",
        "root_cause_detail": (
            "The untouched policy drops from 9/9 with ALOHA RGB to 6/9 with matched G1 RGB while "
            "state/action/task stay exact and the phase-image QA passes. The prior G1-only run and both "
            "paired runs recover no phase group; adaptation-induced handoff degradation is secondary."
        ),
        "g1_visual_dataset_qa": "PASS",
        "old_policy_original_rgb_phase_score": "9/9",
        "old_policy_g1visual_rgb_phase_score": "6/9",
        "previous_g1_only_g1visual_phase_score": "6/9",
        "baseline_phase_pass_counts_out_of_6": abc_phase_counts,
        "checkpoint_gate": {
            "definition": "ALOHA phase score 9/9 AND G1-visual phase score greater than 6/9",
            "passed": False,
            "checkpoint_results": checkpoint_summary,
            "approved_checkpoint": None,
            "best_pareto_analysis_artifact": selected,
            "selection_reason": (
                "R2/1500 has the lowest ALOHA arm and Dex3 RMSE among rehearsal checkpoints and "
                "retains 2/6 G1 handoff cases, but ALOHA is only 8/9 and G1 remains 6/9."
            ),
        },
        "handoff_approach": {
            "diagnosis": (
                "The target and original ALOHA observation elicit the right-arm approach in 6/6 cases; "
                "matched G1 imagery reduces this to 2/6 before adaptation, 1/6 after G1-only adaptation, "
                "and 2/6 after projection-only rehearsal. With semantic render QA passing, this is "
                "primarily failure to recognize/condition on the G1-rendered handoff phase, expressed as "
                "suppression or misdirection of the target-aligned right-arm component; local adaptation "
                "forgetting is secondary."
            ),
            "conditions": handoff_summary,
        },
        "training": training,
        "all_conditions_phase_group_metrics": str(all_metrics_path),
        "closed_loop": {
            "status": "NOT_RUN",
            "reason": "No rehearsal checkpoint satisfied the required dual-domain retention gate.",
        },
        "real_g1": "NOT_STARTED_BY_DESIGN",
        "final_deployment_camera": "NOT_YET_DECIDED",
        "final_outcome": "G1_VISUAL_ADAPTATION_STILL_FAILED",
    }
    atomic_json(OUT / "final_analysis.json", result)

    old_g1_counts = abc_phase_counts["B_ORIGINAL_POLICY_G1_RGB"]
    lines = [
        "# Policy-B G1-visual paired-rehearsal final report",
        "",
        "## OLD POLICY ON ORIGINAL RGB",
        "",
        "Phase score: **9/9** (all nine groups 6/6).",
        "",
        "## OLD POLICY ON G1 RGB",
        "",
        "Phase score: **6/9**.",
        "",
        "| Phase | Pass count |",
        "|---|---:|",
    ]
    for phase, label in base.PHASE_LABELS.items():
        lines.append(f"| {label} | {old_g1_counts[phase]}/6 |")
    lines.extend(
        [
        "",
        "## PREVIOUS G1-ONLY ADAPTATION",
        "",
        "Phase score: **6/9** (initial 4/6, left transport 5/6, handoff approach 1/6).",
        "",
        "## G1 VISUAL DATASET QA",
        "",
        "**PASS.** Fifty-four exactly matched episode/frame/state/action rows were checked across all "
        "nine phases. Dataset arrays and timestamps match; the rendered pose equals the stored state; "
        "the ownership reconstruction is continuous; no camera drift, object teleport, temporal offset, "
        "right-arm visibility failure, or hand/object contradiction was found.",
        "",
        "## ROOT-CAUSE CLASSIFICATION",
        "",
        "**VISUAL_GAP_DOMINANT.** The untouched model falls from 9/9 to 6/9 when only the matched RGB "
        "domain changes. The render QA passes. G1-only adaptation did not recover a phase group and "
        "locally degraded handoff from 2/6 to 1/6, so forgetting is present but secondary.",
        "",
        "## NEW REHEARSAL EXPERIMENTS",
        "",
        "R1 training setup: 50/50 paired ALOHA/G1 frames; 1,500 steps; AdamW; batch 16; "
        "LR 1e-5 to 1e-6; AMP; action expert, state/action projections, and action-time MLP trainable "
        "(99,880,992 scalars). ALOHA phase score: **8/9**. G1 phase score: **6/9**.",
        "",
        "R2 training setup: same paired data/schedule; only the resolved SmolVLM visual connector "
        "projection trainable (11,796,480 scalars; all 499 other checkpoint tensors bitwise unchanged). "
        "ALOHA phase score: **8/9**. G1 phase score: **6/9**.",
        "",
        "## Decision",
        "",
        "No rehearsal checkpoint passed the dual-domain retention gate, so closed-loop Isaac was not run.",
        "The root-cause classification remains **VISUAL_GAP_DOMINANT**, while these two bounded "
        "adaptation strategies failed to improve the strict G1 phase score.",
        "",
        "## Dual-domain checkpoint gate",
        "",
        "| Checkpoint | ALOHA phases | G1 phases | ALOHA arm RMSE | G1 arm RMSE | ALOHA Dex3 RMSE | G1 Dex3 RMSE | A/G1 handoff | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in checkpoint_summary:
        lines.append(
            f"| {row['experiment']} {row['step']} | {row['aloha_phase_score']}/9 | "
            f"{row['g1visual_phase_score']}/9 | {row['aloha_arm_rmse_rad']:.6f} | "
            f"{row['g1visual_arm_rmse_rad']:.6f} | {row['aloha_dex3_rmse_rad']:.6f} | "
            f"{row['g1visual_dex3_rmse_rad']:.6f} | {row['aloha_handoff_pass_count']}/6 / "
            f"{row['g1visual_handoff_pass_count']}/6 | FAIL |"
        )
    lines.extend(
        [
            "",
            "R2/1500 is retained as an analysis-only Pareto artifact; it is not rollout-approved. "
            "The original frozen Policy B remains the only checkpoint with complete ALOHA-domain "
            "phase retention.",
            "",
            "## Handoff approach",
            "",
            "| Condition | Right-arm behavior | Mean target prefix-4 displacement | Mean predicted prefix-4 displacement | Prefix-4 RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, row in handoff_summary.items():
        lines.append(
            f"| {condition} | {row['phase_behavior_pass_count']}/6 | "
            f"{row['target_right_arm_prefix4_net_displacement_rad_l2_mean']:.6f} | "
            f"{row['predicted_right_arm_prefix4_net_displacement_rad_l2_mean']:.6f} | "
            f"{row['right_arm_prefix4_rmse_rad_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The evidence is consistent with G1-image phase recognition failure expressed as suppression "
            "or misdirection of target-aligned right-arm motion, not object/hand semantic inconsistency. "
            "The G1-only adaptation also introduced secondary local handoff forgetting.",
            "",
            "Selected checkpoint: **NONE APPROVED**. R2/1500 (SHA256 "
            f"`{selected['model_sha256']}`) is retained only as the best Pareto analysis artifact because "
            "it has the lowest rehearsal-checkpoint ALOHA arm/Dex3 RMSE and preserves 2/6 G1 handoff "
            "cases; it still fails both phase gates.",
            "",
            "## Training provenance",
            "",
            f"- R1: 1,500 steps, 99,880,992 trainable scalars; final logged loss {training['R1']['final_loss']['logged_loss']:.3f}.",
            f"- R2: 1,500 steps, 11,796,480 trainable scalars; final logged loss {training['R2']['final_loss']['logged_loss']:.3f}.",
            f"- R2 tensor audit: {tensor_audit['status']}; exactly one changed tensor: `{EXPECTED_CONNECTOR}`.",
            "- Complete per-phase/per-group first-action RMSE, prefix-4 RMSE, phase detections, and "
            "predicted/target motion magnitudes for every baseline/checkpoint/domain are in "
            "`final_analysis/all_conditions_phase_group_metrics.csv`.",
            "- Dataset inputs and all frozen supervision/camera/task artifacts were read-only.",
            "",
            "## Execution disposition",
            "",
            "- CLOSED LOOP: **NOT_RUN** (dual-domain checkpoint gate failed).",
            "- Real G1: **NOT_STARTED_BY_DESIGN**.",
            "- Final deployment camera: **NOT_YET_DECIDED**.",
            "",
            "Final outcome: **G1_VISUAL_ADAPTATION_STILL_FAILED**.",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    atomic_text(OUT / "report.md", report_text)
    atomic_text(OUT.parent / "final_report.md", report_text)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "root_cause": result["root_cause_classification"],
                "approved_checkpoint": None,
                "pareto_artifact": selected["checkpoint"],
                "closed_loop": "NOT_RUN",
                "outcome": result["final_outcome"],
                "output": str(OUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
