#!/usr/bin/env python3
"""Generate the four-panel standardized-grasp DEV35 paper figure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/standardized_grasp_ab_dev35"
RESULTS = OUT / "05_results/FINAL_STANDARDIZED_GRASP_NUMERIC_RESULTS.json"
PAPER = OUT / "06_paper_artifacts"
BASE = PAPER / "FigXX_Standardized_Grasp_AB_Physical_Comparison_double"
COLORS = ("#4477AA", "#EE8833")
STAGES = (
    ("LIFT_SUCCESS", "Lift"),
    ("LEFT_RETENTION_SUCCESS", "Left\nretention"),
    ("HANDOFF_SUCCESS", "Handoff"),
    ("RIGHT_OWNERSHIP_SUCCESS", "Right\nownership"),
    ("RIGHT_TRANSPORT_RETENTION", "Transport"),
    ("BIN_ENTRY_SUCCESS", "Bin\nentry"),
    ("BIN_SETTLE_SUCCESS", "Settle"),
    ("POST_GRASP_FULL_TASK_SUCCESS", "Post-grasp\nfull task"),
)


def atomic_save(fig: plt.Figure, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
    fig.savefig(temporary, **kwargs)
    os.replace(temporary, path)


def box(ax: plt.Axes, center: tuple[float, float], text: str, color: str, width: float, height: float = .22, fontsize: float = 6.5) -> None:
    x, y = center
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#333333", linewidth=.8)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="white" if color != "#E9ECEF" else "#222222", fontweight="bold")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color="#555555", linewidth=.9))


def main() -> int:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    a, b = data["A"], data["B"]
    fig = plt.figure(figsize=(7.25, 6.55), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.02, 1.18), height_ratios=(1.0, 1.0))
    ax_design = fig.add_subplot(grid[0, 0])
    ax_stage = fig.add_subplot(grid[0, 1])
    ax_tsr = fig.add_subplot(grid[1, 0])
    ax_matrix = fig.add_subplot(grid[1, 1])

    # (a) Controlled design, intentionally explicit about what is standardized.
    ax_design.set_xlim(0, 1); ax_design.set_ylim(0, 1); ax_design.axis("off")
    box(ax_design, (.5, .84), "SAME VERIFIED PHYSICAL\nLEFT-GRASP INITIALIZATION", "#4C956C", .78, fontsize=6.7)
    arrow(ax_design, (.42, .72), (.28, .59)); arrow(ax_design, (.58, .72), (.72, .59))
    box(ax_design, (.25, .46), "A · WRIST-CENTRIC\nPOST-GRASP\nTARGET", COLORS[0], .45, fontsize=5.5)
    box(ax_design, (.75, .46), "B · INTERACTION-CENTRIC\nPOST-GRASP\nTARGET", COLORS[1], .45, fontsize=5.0)
    arrow(ax_design, (.28, .33), (.42, .21)); arrow(ax_design, (.72, .33), (.58, .21))
    box(ax_design, (.5, .13), "SAME IK · DEX3 · PHYSICS · SCORER", "#E9ECEF", .82, fontsize=6.2)
    ax_design.set_title("(a) Controlled design", loc="left", fontweight="bold", fontsize=9)

    # (b) Cumulative physical stages.
    y = np.arange(len(STAGES)); height = .36
    ac = np.asarray([a["stage_counts"][key] for key, _ in STAGES]); bc = np.asarray([b["stage_counts"][key] for key, _ in STAGES])
    for offset, values, label, color in ((-height / 2, ac, "A · Wrist", COLORS[0]), (height / 2, bc, "B · Interaction", COLORS[1])):
        bars = ax_stage.barh(y + offset, 100.0 * values / 35.0, height, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            ax_stage.text(max(1.2, bar.get_width() + 1.2), bar.get_y() + bar.get_height() / 2, f"{value}/35", ha="left", va="center", fontsize=6.4)
    ax_stage.set_yticks(y, [label.replace("\n", " ") for _, label in STAGES], fontsize=6.5); ax_stage.invert_yaxis()
    ax_stage.set_xlim(0, 108); ax_stage.set_xlabel("Cumulative success (%)", fontsize=8)
    ax_stage.set_title("(b) Post-grasp stage success", loc="left", fontweight="bold", fontsize=9)
    ax_stage.legend(frameon=False, fontsize=6.6, ncol=2, loc="lower right")
    ax_stage.grid(axis="x", alpha=.18); ax_stage.tick_params(axis="x", labelsize=7)

    # (c) Dominant primary metric with exact confidence intervals.
    key = "POST_GRASP_FULL_TASK_SUCCESS"
    counts = [a["stage_counts"][key], b["stage_counts"][key]]
    values = np.asarray(counts, dtype=float) * 100.0 / 35.0
    cis = [a["stage_clopper_pearson_95_percent_ci_percent"][key], b["stage_clopper_pearson_95_percent_ci_percent"][key]]
    errors = np.asarray([[values[i] - cis[i][0] for i in range(2)], [cis[i][1] - values[i] for i in range(2)]])
    bars = ax_tsr.bar([0, 1], values, color=COLORS, width=.58, yerr=errors, capsize=5)
    for bar, count, value in zip(bars, counts, values, strict=True):
        ax_tsr.text(bar.get_x() + bar.get_width() / 2, max(12.0, value + 4.0), f"{count}/35\n{value:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    paired = data["paired"]; effect_ci = paired["paired_bootstrap_95_percent_ci_percentage_points"]
    ax_tsr.text(.5, .96, f"B−A = {paired['B_minus_A_percentage_points']:+.1f} pp\npaired 95% CI [{effect_ci[0]:.1f}, {effect_ci[1]:.1f}] pp", transform=ax_tsr.transAxes, ha="center", va="top", fontsize=9)
    ax_tsr.set_xticks([0, 1], ["A · Wrist", "B · Interaction"]); ax_tsr.set_ylim(0, 109)
    ax_tsr.set_ylabel("Post-grasp task success (%)")
    ax_tsr.set_title("(c) Post-grasp task completion", loc="left", fontweight="bold", fontsize=9)
    ax_tsr.grid(axis="y", alpha=.18)

    # (d) One row per matched episode and method. Color encodes deepest passed stage.
    runs = data["runs"]
    matrix = np.zeros((2, 35), dtype=int)
    for method_index, method in enumerate(("A", "B")):
        for index, run in enumerate(runs[method]):
            matrix[method_index, index] = sum(int(run["outcomes"][stage]) for stage, _ in STAGES[:-1])
            if run["outcomes"][key]: matrix[method_index, index] = 7
    palette = ListedColormap(["#E7E7E7", "#D7EAF3", "#C2DDEB", "#A9D1DC", "#8DC3CB", "#70B5B7", "#52A18F", "#2B7A55"])
    ax_matrix.imshow(matrix, cmap=palette, norm=BoundaryNorm(np.arange(-.5, 8.5), 8), aspect="auto", interpolation="nearest")
    ax_matrix.set_yticks([0, 1], ["A · Wrist", "B · Interaction"], fontsize=7)
    ticks = [0, 8, 17, 26, 34]
    ax_matrix.set_xticks(ticks, ["01", "09", "18", "27", "35"], fontsize=7)
    for row in range(2):
        for column in range(35):
            if matrix[row, column] > 0: ax_matrix.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=4.8, color="#15372A")
    ax_matrix.set_title("(d) Matched 35-episode progression", loc="left", fontweight="bold", fontsize=10)
    ax_matrix.set_xlabel("Matched DEV35 episode", fontsize=8)
    for boundary in range(1, 35): ax_matrix.axvline(boundary - .5, color="white", linewidth=.25, alpha=.7)
    ax_matrix.axhline(.5, color="white", linewidth=.8)
    legend = "0 fail before lift   1 lift   2 left retention   3 handoff   4 ownership   5 transport   6 bin entry   7 settle/full"
    ax_matrix.text(.5, -.36, legend, transform=ax_matrix.transAxes, ha="center", va="top", fontsize=6.0, wrap=True)
    for spine in ax_matrix.spines.values(): spine.set_visible(False)

    fig.suptitle("DEV35 standardized-grasp physical comparison (not end-to-end grasp acquisition)", fontsize=10.0, fontweight="bold")
    for suffix in ("png", "pdf", "svg"):
        atomic_save(fig, BASE.with_suffix("." + suffix), dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"png": str(BASE.with_suffix('.png')), "pdf": str(BASE.with_suffix('.pdf')), "svg": str(BASE.with_suffix('.svg'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
