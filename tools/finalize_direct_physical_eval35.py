#!/usr/bin/env python3
"""Aggregate complete direct ACT-A/B EVAL35, statistics, figure, table, report."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.stats import beta, binomtest


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_direct_physical_eval35"
RUNS = OUT / "01_rollouts"
RESULTS = OUT / "02_results"
FIGURES = OUT / "03_paper_figures"
FREEZE = OUT / "00_freeze/DIRECT_EVAL35_FREEZE_MANIFEST.json"
COMMANDS = OUT / "00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
EVAL35 = ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json"
METRICS = (
    "LEFT_GRASP_SUCCESS", "HANDOFF_SUCCESS", "RIGHT_OWNERSHIP_SUCCESS",
    "NO_DROP_TO_BIN", "BIN_ENTRY_SUCCESS", "BIN_SETTLE_SUCCESS",
    "FULL_TASK_SUCCESS",
)
LABELS = ("Left grasp", "Handoff", "Right ownership", "No-drop-to-bin", "Bin entry", "Bin settle", "Full task")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ci_exact(success: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if success == 0 else float(beta.ppf(alpha / 2, success, total - success + 1))
    upper = 1.0 if success == total else float(beta.ppf(1 - alpha / 2, success + 1, total - success))
    return lower, upper


def main() -> int:
    freeze = read_json(FREEZE)
    commands = read_json(COMMANDS)
    identity = read_json(EVAL35)
    records = {(row["method"], int(row["eval_index"])): row for row in commands["records"]}
    results: dict[str, list[dict[str, Any]]] = {"ACT-A40": [], "ACT-B40": []}
    for method in results:
        letter = method[4].lower()
        for index in range(35):
            command = records[(method, index)]
            path = RUNS / f"act_{letter}40/eval_{index:02d}_{command['stable_episode_id']}/RUN_MANIFEST.json"
            if not path.is_file():
                raise RuntimeError(f"missing rollout result: {path}")
            row = read_json(path)
            if row.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL"}:
                raise RuntimeError(f"non-aggregable rollout {path}: {row.get('status')}")
            if row["fairness_audit"]["arm_common_override_scalar_count"] != 0 or row["fairness_audit"]["wrist_common_override_scalar_count"] != 0:
                raise RuntimeError("COMMON_EXECUTION_LAYER_INVALID")
            results[method].append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    counts = {method: {metric: sum(bool(row["outcomes"][metric]) for row in rows) for metric in METRICS} for method, rows in results.items()}
    failures = {method: Counter(row["first_failure_stage"] or "NONE" for row in rows) for method, rows in results.items()}
    a_binary = np.asarray([row["outcomes"]["FULL_TASK_SUCCESS"] for row in results["ACT-A40"]], dtype=bool)
    b_binary = np.asarray([row["outcomes"]["FULL_TASK_SUCCESS"] for row in results["ACT-B40"]], dtype=bool)
    paired = {
        "A_fail_B_fail": int(np.count_nonzero(~a_binary & ~b_binary)),
        "A_success_B_fail": int(np.count_nonzero(a_binary & ~b_binary)),
        "A_fail_B_success": int(np.count_nonzero(~a_binary & b_binary)),
        "A_success_B_success": int(np.count_nonzero(a_binary & b_binary)),
    }
    discord = paired["A_success_B_fail"] + paired["A_fail_B_success"]
    p_value = 1.0 if discord == 0 else float(binomtest(paired["A_fail_B_success"], discord, 0.5).pvalue)
    rng = np.random.default_rng(20260902)
    draws = rng.integers(0, 35, size=(100000, 35))
    differences = 100.0 * np.mean(b_binary[draws].astype(float) - a_binary[draws].astype(float), axis=1)
    diff_ci = np.quantile(differences, [0.025, 0.975]).tolist()
    summary = {
        "schema_version": "final_direct_physical_eval35_summary_v1",
        "evaluation_set": "EVAL35", "completed_rollouts": 70,
        "freeze_manifest": str(FREEZE), "direct_execution_bundle_sha256": freeze["direct_execution_bundle_sha256"],
        "no_classifier_or_atlas_gate": True, "arm_rescue": False, "wrist_rescue": False,
        "counts": counts,
        "percent": {method: {metric: 100.0 * count / 35.0 for metric, count in values.items()} for method, values in counts.items()},
        "full_task_95pct_clopper_pearson": {method: ci_exact(values["FULL_TASK_SUCCESS"], 35) for method, values in counts.items()},
        "B_minus_A_percentage_points": 100.0 * (counts["ACT-B40"]["FULL_TASK_SUCCESS"] - counts["ACT-A40"]["FULL_TASK_SUCCESS"]) / 35.0,
        "paired_outcome": paired, "mcnemar_exact_two_sided_p": p_value,
        "paired_bootstrap": {"samples": 100000, "seed": 20260902, "difference_95pct_ci_percentage_points": diff_ci},
        "first_failure_distribution": {method: dict(value) for method, value in failures.items()},
        "controller_intervention": {
            method: {
                "arm_fraction_mean": float(np.mean([row["fairness_audit"]["arm_intervention_fraction"] for row in rows])),
                "wrist_fraction_mean": float(np.mean([row["fairness_audit"]["wrist_intervention_fraction"] for row in rows])),
                "dex3_fraction_mean": float(np.mean([row["fairness_audit"]["dex3_intervention_fraction"] for row in rows])),
            }
            for method, rows in results.items()
        },
    }
    (RESULTS / "TASK_SUCCESS_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (RESULTS / "PER_EPISODE_PHYSICAL_RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("eval_index", "stable_episode_id", "provenance", *(f"A_{metric}" for metric in METRICS), "A_first_failure", *(f"B_{metric}" for metric in METRICS), "B_first_failure"))
        writer.writeheader()
        for index in range(35):
            a, b = results["ACT-A40"][index], results["ACT-B40"][index]
            writer.writerow({"eval_index": index, "stable_episode_id": a["stable_episode_id"], "provenance": a["provenance"], **{f"A_{metric}": int(a["outcomes"][metric]) for metric in METRICS}, "A_first_failure": a["first_failure_stage"] or "NONE", **{f"B_{metric}": int(b["outcomes"][metric]) for metric in METRICS}, "B_first_failure": b["first_failure_stage"] or "NONE"})
    table_lines = ["# Final physical task success — EVAL35", "", "| Metric | ACT-A | ACT-B |", "|---|---:|---:|"]
    for metric, label in zip(METRICS, LABELS, strict=True):
        ac, bc = counts["ACT-A40"][metric], counts["ACT-B40"][metric]
        suffix_a = f" ({100*ac/35:.1f}%)" if metric == "FULL_TASK_SUCCESS" else ""
        suffix_b = f" ({100*bc/35:.1f}%)" if metric == "FULL_TASK_SUCCESS" else ""
        table_lines.append(f"| {label} | {ac}/35{suffix_a} | {bc}/35{suffix_b} |")
    table_lines.extend(["", f"Full-task B − A: **{summary['B_minus_A_percentage_points']:.1f} percentage points**."])
    table_path = RESULTS / "TABLE_FINAL_PHYSICAL_TASK_SUCCESS_EVAL35.md"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    colors = {"ACT-A40": "#4C78A8", "ACT-B40": "#F58518"}
    fig = plt.figure(figsize=(14.2, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.45, 0.8, 1.25))
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(METRICS)); width = 0.36
    for offset, method in ((-width/2, "ACT-A40"), (width/2, "ACT-B40")):
        values = np.asarray([counts[method][metric] for metric in METRICS])
        bars = ax.bar(x + offset, 100 * values / 35, width, color=colors[method], label=method.replace("40", ""))
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{value}/35", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, ["Grasp", "Handoff", "Ownership", "No-drop", "Bin entry", "Settle", "Full"], rotation=28, ha="right")
    ax.set_ylim(0, 108); ax.set_ylabel("Success rate (%)"); ax.set_title("(a) Stage-wise physical success", loc="left", fontweight="bold"); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    ax = fig.add_subplot(grid[0, 1])
    vals = [counts["ACT-A40"]["FULL_TASK_SUCCESS"], counts["ACT-B40"]["FULL_TASK_SUCCESS"]]
    cis = [ci_exact(value, 35) for value in vals]
    rates = np.asarray(vals) / 35
    errors = np.asarray([[rates[i]-cis[i][0] for i in range(2)], [cis[i][1]-rates[i] for i in range(2)]])
    ax.bar([0,1], 100*rates, color=[colors["ACT-A40"], colors["ACT-B40"]], width=.62)
    ax.errorbar([0,1], 100*rates, yerr=100*errors, fmt="none", ecolor="black", capsize=4, lw=1.2)
    for i, value in enumerate(vals): ax.text(i, 100*rates[i]+3, f"{value}/35\n{100*rates[i]:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks([0,1], ["ACT-A", "ACT-B"]); ax.set_ylim(0,108); ax.set_title("(b) Full Task Success", loc="left", fontweight="bold"); ax.grid(axis="y", alpha=.2)
    ax.text(.5, .02, f"B − A = {summary['B_minus_A_percentage_points']:.1f} pp", transform=ax.transAxes, ha="center", fontsize=9)
    ax = fig.add_subplot(grid[0, 2])
    stage_code = {"LEFT_GRASP":1, "HANDOFF":2, "RIGHT_OWNERSHIP":3, "TRANSPORT":4, "BIN_ENTRY":5, "SETTLE":6, None:7}
    matrix = np.asarray([[stage_code.get(results[method][i]["first_failure_stage"], 0) if not results[method][i]["outcomes"]["FULL_TASK_SUCCESS"] else 7 for method in ("ACT-A40","ACT-B40")] for i in range(35)])
    cmap = ListedColormap(["#ffffff", "#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850", "#2166ac"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=7)
    ax.set_xticks([0,1], ["ACT-A", "ACT-B"]); ax.set_yticks(np.arange(35), [str(i) for i in range(35)], fontsize=6)
    ax.axhline(7.5, color="black", lw=.8); ax.axhline(9.5, color="black", lw=.8, ls="--")
    ax.set_ylabel("Matched EVAL35 episode"); ax.set_title("(c) First failure / success matrix", loc="left", fontweight="bold")
    fig.suptitle("Direct contact-constrained physical task evaluation (no readiness gate)", fontsize=14, fontweight="bold")
    stem = FIGURES / "Fig08_ACT_AB_Physical_Task_Success_EVAL35_double"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    report = OUT / "FINAL_ACT_AB_PHYSICAL_TASK_SUCCESS_EVAL35.md"
    a, b = counts["ACT-A40"]["FULL_TASK_SUCCESS"], counts["ACT-B40"]["FULL_TASK_SUCCESS"]
    report.write_text(
        "# Final ACT-A40 vs ACT-B40 direct physical EVAL35\n\n"
        f"ACT-A Full Task Success: **{a}/35 = {100*a/35:.1f}%**  \n"
        f"ACT-B Full Task Success: **{b}/35 = {100*b/35:.1f}%**  \n"
        f"B − A: **{summary['B_minus_A_percentage_points']:.1f} percentage points**\n\n"
        "The 70 complete rollouts used contact-constrained PhysX, the same frozen 150 mm bin, doll, common source-derived intent, and Dex3 finger primitives. No graspability classifier, atlas gate, wrist-distance gate, ARM rescue, or WRIST rescue was used. Method-specific ACT arm/wrist trajectories were preserved.\n\n"
        f"Direct execution bundle SHA256: `{freeze['direct_execution_bundle_sha256']}`\n\n"
        "## Stage results\n\n" + "\n".join(table_lines[2:]) + "\n\n"
        f"Paired outcomes: `{paired}`. Exact paired discordance p = `{p_value:.6g}`; paired bootstrap 95% CI for B−A = `{diff_ci[0]:.1f}` to `{diff_ci[1]:.1f}` percentage points (100,000 samples, seed 20260902). N=35 remains modest, so no universal or real-robot claim is made.\n\n"
        f"First failures A: `{dict(failures['ACT-A40'])}`  \nFirst failures B: `{dict(failures['ACT-B40'])}`\n\n"
        f"Figure: `{stem.with_suffix('.png')}`  \nTable: `{table_path}`  \nPer-episode CSV: `{RESULTS / 'PER_EPISODE_PHYSICAL_RESULTS.csv'}`\n\n"
        "This is a source-conditioned, common-controller-assisted G1 simulation result; it is not real-G1 success or target-domain visual autonomy.\n",
        encoding="utf-8",
    )
    print(json.dumps({"ACT_A_full": a, "ACT_B_full": b, "difference_pp": summary["B_minus_A_percentage_points"], "figure": str(stem.with_suffix('.png')), "table": str(table_path), "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
