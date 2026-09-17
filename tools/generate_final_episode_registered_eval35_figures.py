#!/usr/bin/env python3
"""Generate publication figures from verified final EVAL35 numeric artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_episode_registered_eval35"
RESULTS = OUT / "04_results/FINAL_NUMERIC_RESULTS.json"
FORENSIC = OUT / "00_forensic_audit/ACT_A_LEFT_GRASP_FORENSIC_REPORT.json"
PAPER = OUT / "05_paper_artifacts"
MAIN_BASE = PAPER / "Fig17_ACT_AB_Physical_Task_Success_EVAL35_double"
AUDIT_BASE = PAPER / "Fig17b_Physical_Grasp_Failure_and_Contact_Audit"


def atomic_save(fig: plt.Figure, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
    fig.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def main() -> int:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    a, b = data["ACT_A"], data["ACT_B"]
    stages = (
        ("LEFT_GRASP_SUCCESS", "Left\ngrasp"),
        ("HANDOFF_SUCCESS", "Handoff"),
        ("RIGHT_OWNERSHIP_SUCCESS", "Right\nownership"),
        ("NO_DROP_TO_BIN", "No-drop"),
        ("BIN_ENTRY_SUCCESS", "Bin\nentry"),
        ("BIN_SETTLE_SUCCESS", "Bin\nsettle"),
        ("FULL_TASK_SUCCESS", "Full\ntask"),
    )
    colors = ("#4978B8", "#E18B3B")
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.22), height_ratios=(1.0, 1.0))
    ax_stage = fig.add_subplot(grid[0, 0])
    ax_tsr = fig.add_subplot(grid[0, 1])
    ax_fail = fig.add_subplot(grid[1, 0])
    ax_matrix = fig.add_subplot(grid[1, 1])

    x = np.arange(len(stages))
    width = 0.37
    ac = np.asarray([a["stage_counts"][key] for key, _ in stages])
    bc = np.asarray([b["stage_counts"][key] for key, _ in stages])
    for offset, values, label, color in ((-width / 2, ac, "ACT-A", colors[0]), (width / 2, bc, "ACT-B", colors[1])):
        bars = ax_stage.bar(x + offset, 100 * values / 35, width, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            ax_stage.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{value}/35", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax_stage.set_xticks(x, [label for _, label in stages], fontsize=7)
    ax_stage.set_ylim(0, 108)
    ax_stage.set_ylabel("Cumulative success (%)")
    ax_stage.set_title("(a) Stage-wise physical success", loc="left", fontweight="bold")
    ax_stage.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax_stage.grid(axis="y", alpha=0.2)

    full = [a["stage_counts"]["FULL_TASK_SUCCESS"], b["stage_counts"]["FULL_TASK_SUCCESS"]]
    cis = [a["stage_binomial_95_percent_ci_percent"]["FULL_TASK_SUCCESS"], b["stage_binomial_95_percent_ci_percent"]["FULL_TASK_SUCCESS"]]
    values = np.asarray(full) * 100 / 35
    yerr = np.asarray([[values[i] - cis[i][0] for i in range(2)], [cis[i][1] - values[i] for i in range(2)]])
    bars = ax_tsr.bar([0, 1], values, color=colors, width=0.58, yerr=yerr, capsize=4)
    for bar, count, value in zip(bars, full, values, strict=True):
        ax_tsr.text(bar.get_x() + bar.get_width() / 2, max(2.0, value + 2.5), f"{count}/35\n{value:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    diff = data["paired"]["B_minus_A_percentage_points"]
    ci = data["paired"]["paired_bootstrap_95_percent_CI_percentage_points"]
    ax_tsr.text(0.5, 0.96, f"B−A = {diff:+.1f} pp\npaired 95% CI [{ci[0]:.1f}, {ci[1]:.1f}] pp", transform=ax_tsr.transAxes, ha="center", va="top", fontsize=8.5)
    ax_tsr.set_xticks([0, 1], ["ACT-A", "ACT-B"])
    ax_tsr.set_ylim(0, 108)
    ax_tsr.set_ylabel("Full Task Success Rate (%)")
    ax_tsr.set_title("(b) Full Task Success Rate", loc="left", fontweight="bold")
    ax_tsr.grid(axis="y", alpha=0.2)

    failure_keys = ("LEFT_GRASP", "HANDOFF", "RIGHT_OWNERSHIP", "TRANSPORT", "BIN_ENTRY", "SETTLE", "NONE_SUCCESS")
    failure_labels = ("Grasp", "Handoff", "Ownership", "Transport", "Bin", "Settle", "Success")
    af = np.asarray([a["first_failure_counts"][key] for key in failure_keys])
    bf = np.asarray([b["first_failure_counts"][key] for key in failure_keys])
    ax_fail.barh(np.arange(7) + 0.19, af, 0.38, color=colors[0], label="ACT-A")
    ax_fail.barh(np.arange(7) - 0.19, bf, 0.38, color=colors[1], label="ACT-B")
    for row, (av, bv) in enumerate(zip(af, bf, strict=True)):
        if av: ax_fail.text(av + .25, row + .19, str(av), va="center", fontsize=7)
        if bv: ax_fail.text(bv + .25, row - .19, str(bv), va="center", fontsize=7)
    ax_fail.set_yticks(np.arange(7), failure_labels)
    ax_fail.invert_yaxis()
    ax_fail.set_xlim(0, 37)
    ax_fail.set_xlabel("Episodes")
    ax_fail.set_title("(c) First failure stage", loc="left", fontweight="bold")
    ax_fail.grid(axis="x", alpha=0.2)

    runs = data["runs"]
    matrix_keys = ("LEFT_GRASP_SUCCESS", "HANDOFF_SUCCESS", "RIGHT_OWNERSHIP_SUCCESS", "BIN_SETTLE_SUCCESS", "FULL_TASK_SUCCESS")
    matrix = np.asarray([[int(run["outcomes"][key]) for key in matrix_keys] for method in ("ACT_A", "ACT_B") for run in runs[method]], dtype=int)
    ax_matrix.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#E7E7E7", "#3A8B68"]), vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_matrix.axhline(34.5, color="black", lw=1.2)
    ax_matrix.set_xticks(np.arange(5), ["G", "H", "R", "B", "F"])
    ax_matrix.set_yticks([0, 8, 17, 26, 34, 35, 43, 52, 61, 69], ["A01", "A09", "A18", "A27", "A35", "B01", "B09", "B18", "B27", "B35"], fontsize=7)
    ax_matrix.set_title("(d) Matched EVAL35 progression", loc="left", fontweight="bold")
    ax_matrix.set_xlabel("G grasp · H handoff · R ownership · B bin settle · F full")
    for spine in ax_matrix.spines.values(): spine.set_visible(False)

    for suffix in ("png", "pdf", "svg"):
        atomic_save(fig, MAIN_BASE.with_suffix("." + suffix), dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    forensic = json.loads(FORENSIC.read_text(encoding="utf-8"))
    records = forensic["rows"]
    labels = ("palm", "thumb", "index", "middle")
    distance = 1000.0 * np.asarray([[row["minimum_collision_surface_distance_m"][key] for key in labels] for row in records])
    force = np.asarray([[row["contact_force_max_n"][key] for key in labels] for row in records])
    lift = 1000.0 * np.asarray([row["maximum_doll_com_lift_m"] for row in records])
    diagnostic_path = OUT / "07_paper_visuals/ACT_A_GRASP_FAILURE_DIAGNOSTICS.png"
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    diagnostic = plt.imread(diagnostic_path)
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.4, 1.0))
    ax_image = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    ax_image.imshow(diagnostic)
    ax_image.set_axis_off()
    ax_image.set_title("(a) Deterministic ACT-A trace closeups: episodes 01, 09, 18, 27, 35", loc="left", fontweight="bold")
    axes[0].boxplot([distance[:, i] for i in range(4)], tick_labels=[x.title() for x in labels], showfliers=True)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Minimum signed surface distance (mm)")
    axes[0].set_title("(b) Closest approach", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].boxplot([force[:, i] for i in range(4)], tick_labels=[x.title() for x in labels], showfliers=True)
    axes[1].set_ylabel("Maximum contact force (N)")
    axes[1].set_title("(c) Physical contact", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].hist(lift, bins=max(5, min(12, len(np.unique(lift)))), color=colors[0], edgecolor="white")
    axes[2].axvline(50, color="#B13A3A", ls="--", lw=1, label="50 mm qualification")
    axes[2].set_xlabel("Maximum doll COM lift (mm)")
    axes[2].set_ylabel("Episodes")
    axes[2].set_title("(d) Retention/lift", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("ACT-A pre-fix A35 forensic evidence (preserved provenance): geometric grasp misses, not scorer false negatives", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        atomic_save(fig, AUDIT_BASE.with_suffix("." + suffix), dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"main_figure": str(MAIN_BASE.with_suffix('.png')), "audit_figure": str(AUDIT_BASE.with_suffix('.png'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
