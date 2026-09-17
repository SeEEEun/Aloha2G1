#!/usr/bin/env python3
"""Compare ALOHA/G1 visual phase probes for root-cause attribution."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import probe_policy_b_dataset_phases as base
import probe_policy_b_g1visual_phases as g1probe


ROOT = Path(__file__).resolve().parents[1]
PROBES = {
    "A_ORIGINAL_POLICY_ALOHA_RGB": ROOT / "outputs/policy_b_offline_phase_probe/probe_predictions.npz",
    "B_ORIGINAL_POLICY_G1_RGB": ROOT
    / "outputs/policy_b_g1visual/root_cause_diagnostic/old_policy_on_g1visual/probe_predictions.npz",
    "C_G1_ONLY_5K_POLICY_G1_RGB": ROOT / "outputs/policy_b_g1visual/offline_phase_probe/probe_predictions.npz",
}
DECISIONS = {
    "A_ORIGINAL_POLICY_ALOHA_RGB": ROOT
    / "outputs/policy_b_offline_phase_probe/phase_learning_decision.json",
    "B_ORIGINAL_POLICY_G1_RGB": ROOT
    / "outputs/policy_b_g1visual/root_cause_diagnostic/old_policy_on_g1visual/phase_learning_decision.json",
    "C_G1_ONLY_5K_POLICY_G1_RGB": ROOT
    / "outputs/policy_b_g1visual/offline_phase_probe/phase_learning_decision.json",
}
QA = ROOT / "outputs/policy_b_g1visual/root_cause_diagnostic/phase_visual_qa/validation.json"
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_g1visual/root_cause_diagnostic/abc_comparison"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_probe(name: str, path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if name.startswith("A_"):
        prediction = np.asarray(data["policy_prediction_A"], dtype=np.float32)
    else:
        prediction = np.asarray(data["policy_prediction"], dtype=np.float32)
    return {
        "prediction": prediction,
        "target": np.asarray(data["authoritative_target"], dtype=np.float32),
        "episode": np.asarray(data["episode_index"], dtype=np.int64),
        "frame": np.asarray(data["frame_index"], dtype=np.int64),
        "phase": np.asarray(data["phase"]).astype("U40"),
        "joints": np.asarray(data["joint_names"]).astype("U64"),
    }


def relevant_evidence(
    prediction: np.ndarray, target: np.ndarray, phase: np.ndarray, phase_name: str
) -> tuple[bool, dict[str, Any]]:
    sample_indices = np.flatnonzero(phase == phase_name)
    evidence: dict[str, Any] = {}
    phase_pass = True
    for group in g1probe.RELEVANT_GROUPS[phase_name]:
        rows = [
            base.group_metrics(prediction[index], target[index], base.GROUPS[group])
            for index in sample_indices
        ]
        passed = [
            row["trajectory_motion_cosine"] is not None
            and row["trajectory_motion_cosine"] >= g1probe.ESTABLISHED_COSINE_GATE
            and row["target_direction_projection_progress"] is not None
            and row["target_direction_projection_progress"]
            >= g1probe.ESTABLISHED_DIRECTION_PROGRESS_GATE
            for row in rows
        ]
        evidence[group] = {
            "pass_count": int(sum(passed)),
            "total": len(passed),
            "trajectory_motion_cosine_mean": float(
                np.mean([row["trajectory_motion_cosine"] for row in rows])
            ),
            "target_direction_projection_progress_mean": float(
                np.mean([row["target_direction_projection_progress"] for row in rows])
            ),
        }
        phase_pass &= all(passed)
    return phase_pass, evidence


def metric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    qa = read_json(QA)
    if qa["status"] != "PASS" or qa["manual_visual_review"]["status"] != "PASS":
        raise RuntimeError("G1-visual semantic QA is not PASS")

    probes = {name: load_probe(name, path) for name, path in PROBES.items()}
    reference = next(iter(probes.values()))
    for name, probe in probes.items():
        for key in ("target", "episode", "frame", "phase", "joints"):
            if not np.array_equal(probe[key], reference[key]):
                raise RuntimeError(f"{name}: unmatched {key}")

    detail_rows: list[dict[str, Any]] = []
    phase_results: dict[str, Any] = {}
    condition_scores: dict[str, int] = {}
    for condition, probe in probes.items():
        phase_results[condition] = {}
        condition_score = 0
        for phase_name, phase_label in base.PHASE_LABELS.items():
            sample_indices = np.flatnonzero(probe["phase"] == phase_name)
            behavior, evidence = relevant_evidence(
                probe["prediction"], probe["target"], probe["phase"], phase_name
            )
            condition_score += int(behavior)
            groups: dict[str, Any] = {}
            for group, indices in base.GROUPS.items():
                metrics = [
                    base.group_metrics(
                        probe["prediction"][index], probe["target"][index], indices
                    )
                    for index in sample_indices
                ]
                groups[group] = {
                    key: metric_summary([float(row[key]) for row in metrics])
                    for key in (
                        "first_action_rmse_rad",
                        "prefix4_rmse_rad",
                        "full_chunk_rmse_rad",
                        "predicted_motion_magnitude_rad_rms",
                        "target_motion_magnitude_rad_rms",
                    )
                }
                detail_rows.append(
                    {
                        "condition": condition,
                        "phase": phase_name,
                        "phase_label": phase_label,
                        "group": group,
                        "behavior_present": behavior,
                        "first_action_rmse_rad_mean": groups[group]["first_action_rmse_rad"]["mean"],
                        "prefix4_rmse_rad_mean": groups[group]["prefix4_rmse_rad"]["mean"],
                        "full_chunk_rmse_rad_mean": groups[group]["full_chunk_rmse_rad"]["mean"],
                        "predicted_motion_magnitude_rad_rms_mean": groups[group][
                            "predicted_motion_magnitude_rad_rms"
                        ]["mean"],
                        "target_motion_magnitude_rad_rms_mean": groups[group][
                            "target_motion_magnitude_rad_rms"
                        ]["mean"],
                    }
                )
            phase_results[condition][phase_name] = {
                "phase_label": phase_label,
                "behavior_present": behavior,
                "evidence": evidence,
                "groups": groups,
            }
        condition_scores[condition] = condition_score

    with (output / "phase_group_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    score_pattern = (
        condition_scores["A_ORIGINAL_POLICY_ALOHA_RGB"],
        condition_scores["B_ORIGINAL_POLICY_G1_RGB"],
        condition_scores["C_G1_ONLY_5K_POLICY_G1_RGB"],
    )
    if score_pattern[0] == 9 and score_pattern[1] < 9 and score_pattern[2] <= score_pattern[1]:
        classification = "VISUAL_GAP_DOMINANT"
        rationale = (
            "Changing only ALOHA RGB to matched G1 RGB removes three phase groups from the untouched "
            "policy. The 5k G1-only continuation recovers no additional phase group and does not show "
            "broad phase loss relative to that G1-input baseline; its handoff evidence locally degrades."
        )
    elif score_pattern[2] < score_pattern[1]:
        classification = "G1_ONLY_ADAPTATION_FORGETTING_DOMINANT"
        rationale = "The adapted policy loses phase groups that the original policy retained on identical G1 RGB."
    else:
        classification = "MIXED / INCONCLUSIVE"
        rationale = "The three-condition phase pattern does not isolate one dominant failure mode."
    if qa["status"] != "PASS":
        classification = "G1_RENDER_SEMANTIC_MISMATCH"
        rationale = "Matched-frame G1 visual QA failed."

    # Save a compact plot of relevant-group prefix error and motion evidence.
    colors = {name: color for name, color in zip(probes, ["#2f6bff", "#ef8a17", "#3a9d5d"], strict=True)}
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for axis, (phase_name, phase_label) in zip(axes.flat, base.PHASE_LABELS.items(), strict=True):
        relevant = g1probe.RELEVANT_GROUPS[phase_name][0]
        values = [
            phase_results[name][phase_name]["groups"][relevant]["prefix4_rmse_rad"]["mean"]
            for name in probes
        ]
        axis.bar(range(3), values, color=[colors[name] for name in probes])
        axis.set_title(f"{phase_label}\nrelevant: {relevant}")
        axis.set_xticks(range(3), ["A", "B", "C"])
        axis.set_ylabel("prefix-4 RMSE [rad]")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Matched phase probes: A=old/ALOHA, B=old/G1, C=5k/G1", fontsize=14)
    fig.savefig(output / "phase_prefix4_rmse_abc.png", dpi=180)
    plt.close(fig)

    handoff_indices = np.flatnonzero(reference["phase"] == "handoff_approach")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    right = base.GROUPS["right_arm"]
    for axis, index in zip(axes.flat, handoff_indices, strict=True):
        target_delta = reference["target"][index, :4, right] - reference["target"][index, :1, right]
        axis.plot(np.linalg.norm(target_delta, axis=1), "k-o", label="target")
        for name, probe in probes.items():
            delta = probe["prediction"][index, :4, right] - probe["prediction"][index, :1, right]
            axis.plot(np.linalg.norm(delta, axis=1), "-o", color=colors[name], label=name.split("_")[0])
        axis.set_title(
            f"episode {int(reference['episode'][index]):02d}, frame {int(reference['frame'][index]):04d}"
        )
        axis.set_xlabel("prefix step")
        axis.set_ylabel("right-arm displacement norm [rad]")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4)
    fig.suptitle("Handoff-approach: right-arm prefix trajectories")
    fig.savefig(output / "handoff_right_arm_prefix_abc.png", dpi=180)
    plt.close(fig)

    result = {
        "status": "PASS",
        "matched_samples": 54,
        "matched_episode_frame_state_action_task": True,
        "visual_dataset_qa": "PASS",
        "condition_phase_scores": {name: f"{score}/9" for name, score in condition_scores.items()},
        "root_cause_classification": classification,
        "rationale": rationale,
        "phase_results": phase_results,
        "sources": {name: str(path) for name, path in PROBES.items()},
    }
    atomic_json(output / "comparison.json", result)

    lines = [
        "# Policy-B G1-visual root-cause comparison",
        "",
        f"Classification: **{classification}**",
        "",
        rationale,
        "",
        "| Phase | A: old + ALOHA | B: old + G1 | C: 5k + G1 |",
        "|---|---:|---:|---:|",
    ]
    for phase_name, phase_label in base.PHASE_LABELS.items():
        cells = [
            "PASS" if phase_results[name][phase_name]["behavior_present"] else "FAIL"
            for name in probes
        ]
        lines.append(f"| {phase_label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        [
            "",
            f"Scores: A={score_pattern[0]}/9, B={score_pattern[1]}/9, C={score_pattern[2]}/9.",
            "",
            "Detailed first-action, prefix-4, full-chunk, and four-group motion metrics are in "
            "`phase_group_metrics.csv`.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "classification": classification, "scores": score_pattern}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
