#!/usr/bin/env python3
"""Generate the CPU-only, artifact-driven paper figure and table bank.

This script only reads frozen scientific artifacts and writes under
``outputs/paper_figure_bank``.  It does not import training or simulator code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_figure_bank"
PAPER = ROOT / "outputs/paper_core_ab"
AUDIT = ROOT / "outputs/fair_a_full50_hard_fail_audit"
BROOT = ROOT / "outputs/doll_handoff_dataset_b_final"

A_NAME = "Trajectory-Centric A"
B_NAME = "Interaction-Centric B"
A_COLOR = "#0072B2"  # Okabe-Ito blue
B_COLOR = "#D55E00"  # Okabe-Ito vermilion
A_MARKER = "o"
B_MARKER = "s"

SUBDIRS = (
    "fig01_pipeline",
    "fig02_retargeting_tradeoff",
    "fig03_feasibility",
    "fig04_error_distributions",
    "fig05_per_episode_pairing",
    "fig06_interaction_tradeoff",
    "fig07_policy_prediction",
    "fig08_phase_behavior",
    "fig09_policy_smoothness",
    "fig10_trajectory_examples",
    "fig11_failure_examples",
    "fig12_statistical_summary",
    "tables",
    "captions",
    "figure_index",
)

SOURCE_ARTIFACTS = {
    "retargeting_table": PAPER / "tables/table1_full50_retargeting.csv",
    "policy_table": PAPER / "tables/table2_heldout_act_prediction.csv",
    "experiment2": PAPER / "offline_heldout8/experiment2_result.json",
    "training_audit": PAPER / "act_a_b_training_audit.json",
    "a_manifest": AUDIT / "after/fair_a_repair_manifest.json",
    "a_validation": AUDIT / "after/full50_validation.json",
    "a_per_episode": AUDIT / "after/full50_per_episode.csv",
    "comparison": AUDIT / "comparison/fair_a_vs_proposed_b.json",
    "collision_audit": AUDIT / "diagnostics/collision_episode_audit.json",
    "collision_records": AUDIT / "diagnostics/after_collision_records.csv",
    "b_manifest": BROOT / "final_source_manifest.json",
    "b_aggregate": BROOT / "final50/aggregate_metrics.json",
    "b_per_episode": BROOT / "final50/per_episode.csv",
    "common48": PAPER / "common48_manifest.json",
    "heldout8": PAPER / "heldout8_manifest.json",
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def mkdirs() -> None:
    for subdir in SUBDIRS:
        (OUT / subdir).mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 5.5,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def write_rows(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    unique = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append({"path": rel(path), "sha256": sha256(path)})
    return unique


FIGURES: list[dict[str, Any]] = []


def save_figure(
    fig: plt.Figure,
    subdir: str,
    stem: str,
    rows: list[dict[str, Any]],
    sources: Iterable[Path],
    question: str,
    key_result: str,
    priority: str,
    width: str,
    gpu_required: bool = False,
    notes: str = "",
) -> None:
    directory = OUT / subdir
    csv_path = directory / f"{stem}_source.csv"
    write_rows(csv_path, rows)
    fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    command = f"python3 tools/generate_paper_figure_bank.py --only {stem}\n"
    (directory / f"{stem}_generation_command.txt").write_text(command, encoding="utf-8")
    metadata = {
        "schema_version": "paper_figure_bank_figure_v1",
        "figure": stem,
        "scientific_question": question,
        "key_result": key_result,
        "priority": priority,
        "column_recommendation": width,
        "gpu_render_required": gpu_required,
        "gpu_used": False,
        "current_paper_core_job_touched": False,
        "source_csv": rel(csv_path),
        "source_artifacts": artifact_records(sources),
        "notes": notes,
        "generation_command": command.strip(),
    }
    write_json(directory / f"{stem}_metadata.json", metadata)
    FIGURES.append({"subdir": subdir, "stem": stem, **metadata})


def lookup(rows: list[dict[str, str]], metric: str, method: str) -> str:
    for row in rows:
        key = row.get("METRIC", row.get("metric", ""))
        if key == metric:
            return row[method]
    raise KeyError(metric)


def parse_triple(value: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in value.split("/"))
    if len(values) != 3:
        raise ValueError(value)
    return values


def verify_and_load() -> dict[str, Any]:
    for name, path in SOURCE_ARTIFACTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing authoritative artifact {name}: {path}")

    table1 = read_csv(SOURCE_ARTIFACTS["retargeting_table"])
    table2 = read_csv(SOURCE_ARTIFACTS["policy_table"])
    exp2 = read_json(SOURCE_ARTIFACTS["experiment2"])
    train = read_json(SOURCE_ARTIFACTS["training_audit"])

    expected1 = {
        "Feasibility projection mean / p95 / max": ((2.525, 17.054, 94.341), (0.077, 0.000, 84.430)),
        "Wrist error mean / p95 / max": ((3.742, 26.954, 104.241), (77.853, 140.520, 180.875)),
        "Whole-hand error mean / p95 / max": ((90.820, 189.236, 340.607), (21.157, 26.315, 91.439)),
        "Bimanual relation error mean / p95 / max": ((86.786, 210.890, 309.898), (35.979, 49.713, 110.844)),
    }
    for metric, expected in expected1.items():
        actual = (parse_triple(lookup(table1, metric, "FAIR A")), parse_triple(lookup(table1, metric, "PROPOSED B")))
        if actual != expected:
            raise RuntimeError(f"confirmed Experiment-1 discrepancy for {metric}: {actual} != {expected}")

    expected2 = {
        "first-action RMSE": (0.043554, 0.044165),
        "full valid chunk RMSE": (0.154074, 0.109001),
        "predicted source wrist error mean": (24.958, 95.269),
        "predicted source wrist error p95": (68.212, 240.177),
        "predicted whole-hand error mean": (107.131, 58.330),
        "predicted whole-hand error p95": (202.436, 141.967),
        "predicted bimanual relation error mean": (120.976, 85.920),
        "predicted bimanual relation error p95": (232.815, 175.544),
        "raw all-joint direction reversals mean": (3.474247, 7.127672),
        "raw maximum adjacent step": (0.256264, 0.230842),
        "raw qdot RMS": (0.499487, 0.331346),
        "raw qddot RMS": (5.950983, 5.666405),
        "raw jerk RMS": (259.894222, 298.256500),
    }
    for metric, expected in expected2.items():
        actual = (float(lookup(table2, metric, "ACT-A40")), float(lookup(table2, metric, "ACT-B40")))
        if actual != expected:
            raise RuntimeError(f"confirmed Experiment-2 discrepancy for {metric}: {actual} != {expected}")
    scores = tuple(exp2["methods"][m]["selected_checkpoint_evaluation"]["phase_score"]["successful"] for m in "ab")
    losses = tuple(train["methods"][m]["last_logged_health"]["loss"] for m in "ab")
    if scores != (56, 52) or losses != (0.036, 0.038):
        raise RuntimeError(f"confirmed score/loss discrepancy: scores={scores}, losses={losses}")
    return {"table1": table1, "table2": table2, "exp2": exp2, "train": train}


def stats_mm(values_m: np.ndarray) -> dict[str, float]:
    values = np.asarray(values_m, dtype=np.float64) * 1000.0
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def load_retargeting_arrays() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    a_manifest = read_json(SOURCE_ARTIFACTS["a_manifest"])
    b_manifest = read_json(SOURCE_ARTIFACTS["b_manifest"])
    a_rows = {int(row["episode_index"]): row for row in a_manifest["trajectories"]}
    b_rows = {int(row["final_dataset_index"]): row for row in b_manifest["episodes"]}
    if sorted(a_rows) != list(range(50)) or sorted(b_rows) != list(range(50)):
        raise RuntimeError("frozen final-50 identity mapping is incomplete")

    # Recover the frozen G1 model->world affine transform directly from A's paired
    # target-model/source-world arrays.  The fit residual is checked below.
    model, world = [], []
    for index in range(50):
        with np.load(a_rows[index]["trajectory_path"], allow_pickle=False) as z:
            for side in ("left", "right"):
                model.append(np.asarray(z[f"target_{side}_wrist_position_model"], dtype=np.float64))
                world.append(np.asarray(z[f"source_{side}_realization_frame_position_world"], dtype=np.float64))
    model_all, world_all = np.concatenate(model), np.concatenate(world)
    affine = np.linalg.lstsq(np.c_[model_all, np.ones(len(model_all))], world_all, rcond=None)[0]
    fit_max = float(np.abs(np.c_[model_all, np.ones(len(model_all))] @ affine - world_all).max())
    if fit_max > 1e-6:
        raise RuntimeError(f"model-to-world artifact fit residual too large: {fit_max}")

    per_episode: list[dict[str, Any]] = []
    frame = {m: {k: [] for k in ("wrist", "whole_hand", "bimanual", "projection")} for m in ("A", "B")}
    for index in range(50):
        a_path = Path(a_rows[index]["trajectory_path"])
        b_path = Path(b_rows[index]["retargeted_trajectory_path"])
        if sha256(a_path) != a_rows[index]["trajectory_sha256"] or sha256(b_path) != b_rows[index]["retargeted_trajectory_sha256"]:
            raise RuntimeError(f"frozen trajectory hash mismatch at final episode {index}")
        episode_metrics: dict[str, dict[str, np.ndarray]] = {}
        with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
            if str(a["source_episode_id"].item()) != str(b["source_episode_id"].item()):
                raise RuntimeError(f"A/B source identity mismatch at {index}")
            for method, z in (("A", a), ("B", b)):
                if method == "A":
                    wrists = [
                        np.linalg.norm(z[f"achieved_{side}_realization_frame_position_world"] - z[f"source_{side}_realization_frame_position_world"], axis=1)
                        for side in ("left", "right")
                    ]
                else:
                    wrists = []
                    for side in ("left", "right"):
                        target_model = np.asarray(z[f"target_{side}_wrist_position_model"], dtype=np.float64)
                        target_world = np.c_[target_model, np.ones(len(target_model))] @ affine
                        wrists.append(np.linalg.norm(z[f"achieved_{side}_wrist_position_world"] - target_world, axis=1))
                targets = [np.asarray(z[f"target_{side}_interaction_frame_position_world"], dtype=np.float64) for side in ("left", "right")]
                achieved = [np.asarray(z[f"achieved_{side}_physical_grasp_frame_position_world"], dtype=np.float64) for side in ("left", "right")]
                whole = [np.linalg.norm(achieved[s] - targets[s], axis=1) for s in (0, 1)]
                relation = np.linalg.norm((achieved[1] - achieved[0]) - (targets[1] - targets[0]), axis=1)
                projection = np.asarray(z["feasibility_projection_translation_m"], dtype=np.float64).reshape(-1)
                episode_metrics[method] = {
                    "wrist": np.concatenate(wrists),
                    "whole_hand": np.concatenate(whole),
                    "bimanual": relation,
                    "projection": projection,
                }
                for metric, values in episode_metrics[method].items():
                    frame[method][metric].append(values)
            row: dict[str, Any] = {
                "episode_index": index,
                "stable_episode_id": str(a["source_episode_id"].item()),
                "frame_count": int(len(a["timestamp"])),
                "a_status": "",
                "b_status": str(b_rows[index]["classification"]),
                "a_trajectory": rel(a_path),
                "b_trajectory": rel(b_path),
            }
            for method in ("A", "B"):
                for metric, values in episode_metrics[method].items():
                    row[f"{method.lower()}_{metric}_mean_mm"] = float(values.mean() * 1000.0)
                    row[f"{method.lower()}_{metric}_median_mm"] = float(np.median(values) * 1000.0)
                    row[f"{method.lower()}_{metric}_p95_mm"] = float(np.percentile(values, 95) * 1000.0)
                    row[f"{method.lower()}_{metric}_max_mm"] = float(values.max() * 1000.0)
            per_episode.append(row)

    a_class = {int(row["episode_index"]): row["after_classification"] for row in read_csv(SOURCE_ARTIFACTS["a_per_episode"])}
    for row in per_episode:
        row["a_status"] = a_class[row["episode_index"]]
    flat = {method: {metric: np.concatenate(values) for metric, values in metrics.items()} for method, metrics in frame.items()}

    expected = {
        "A": {"wrist": (3.742, 26.954, 104.241), "whole_hand": (90.820, 189.236, 340.607), "bimanual": (86.786, 210.890, 309.898), "projection": (2.525, 17.054, 94.341)},
        "B": {"wrist": (77.853, 140.520, 180.875), "whole_hand": (21.157, 26.315, 91.439), "bimanual": (35.979, 49.713, 110.844), "projection": (0.077, 0.000, 84.430)},
    }
    for method in ("A", "B"):
        for metric in expected[method]:
            s = stats_mm(flat[method][metric])
            actual = (round(s["mean"], 3), round(s["p95"], 3), round(s["max"], 3))
            if actual != expected[method][metric]:
                raise RuntimeError(f"raw-array verification discrepancy {method}/{metric}: {actual}")
    return per_episode, flat["A"], flat["B"]


def grouped_bars(ax: plt.Axes, labels: list[str], a: list[float], b: list[float], ylabel: str) -> None:
    x = np.arange(len(labels))
    width = 0.34
    bars_a = ax.bar(x - width / 2, a, width, color=A_COLOR, edgecolor="black", linewidth=0.7, hatch="//", label=A_NAME)
    bars_b = ax.bar(x + width / 2, b, width, color=B_COLOR, edgecolor="black", linewidth=0.7, hatch="..", label=B_NAME)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0, ncol=1, handlelength=1.7)
    ax.bar_label(bars_a, fmt="%.1f", padding=2, fontsize=7.5)
    ax.bar_label(bars_b, fmt="%.1f", padding=2, fontsize=7.5)
    ax.set_ylim(0, max(a + b) * 1.18)


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str = "#F5F5F5", edge: str = "#444444", ls: str = "-", fontsize: float = 8.2) -> None:
        patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor=edge, linewidth=1.2, linestyle=ls)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1: float, y1: float, x2: float, y2: float, ls: str = "-") -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#555555", "linestyle": ls})

    box(3.25, 6.8, 3.5, 0.75, "ALOHA demonstrations\n(n = 50)", "#E8E8E8")
    box(0.65, 4.65, 3.55, 1.35, f"{A_NAME}\n6-D wrist trajectory", "#DDEBF7", A_COLOR)
    box(5.55, 4.65, 4.0, 1.35, f"{B_NAME}\ninteraction frame • whole-hand geometry\nbimanual relation • ownership transition", "#FCE4D6", B_COLOR, fontsize=7.2)
    box(1.1, 3.0, 2.65, 0.72, "G1 Dataset A", "#DDEBF7", A_COLOR)
    box(6.25, 3.0, 2.65, 0.72, "G1 Dataset B", "#FCE4D6", B_COLOR)
    box(1.35, 1.65, 2.15, 0.72, "ACT-A", "#DDEBF7", A_COLOR)
    box(6.5, 1.65, 2.15, 0.72, "ACT-B", "#FCE4D6", B_COLOR)
    box(3.3, 0.25, 3.4, 0.82, "Evaluation\nretargeting • prediction • phase behavior", "#E8E8E8")
    arrow(5.0, 6.8, 2.45, 6.0); arrow(5.0, 6.8, 7.55, 6.0)
    arrow(2.45, 4.65, 2.45, 3.72); arrow(7.55, 4.65, 7.55, 3.72)
    arrow(2.45, 3.0, 2.45, 2.37); arrow(7.55, 3.0, 7.55, 2.37)
    arrow(2.45, 1.65, 4.25, 1.07); arrow(7.55, 1.65, 5.75, 1.07)
    ax.text(8.55, 0.35, "Optional future:\nsource-conditioned rollout", ha="center", va="center", fontsize=7.2, color="#666666", bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#999999", "ls": "--"})
    rows = [
        {"stage": "input", "item": "ALOHA demonstrations", "count": 50},
        {"stage": "retargeting", "item": A_NAME, "count": "NA"},
        {"stage": "retargeting", "item": B_NAME, "count": "NA"},
        {"stage": "policy", "item": "ACT-A", "count": "NA"},
        {"stage": "policy", "item": "ACT-B", "count": "NA"},
        {"stage": "future", "item": "Source-conditioned rollout", "count": "NA"},
    ]
    save_figure(fig, "fig01_pipeline", "Fig1_pipeline", rows, [SOURCE_ARTIFACTS["retargeting_table"], SOURCE_ARTIFACTS["experiment2"]], "How are the two supervision representations carried from demonstrations to policy evaluation?", "The controlled pipeline differs at retargeting representation and yields matched Dataset A/B and ACT-A/B branches.", "MUST_USE", "double-column")


def retargeting_tradeoff(data: dict[str, Any]) -> None:
    table = data["table1"]
    metrics = ["Wrist error", "Whole-hand error", "Bimanual relation error"]
    artifact_names = ["Wrist error mean / p95 / max", "Whole-hand error mean / p95 / max", "Bimanual relation error mean / p95 / max"]
    for stat_idx, suffix, priority in ((0, "mean", "MUST_USE"), (1, "p95", "STRONG")):
        a = [parse_triple(lookup(table, metric, "FAIR A"))[stat_idx] for metric in artifact_names]
        b = [parse_triple(lookup(table, metric, "PROPOSED B"))[stat_idx] for metric in artifact_names]
        fig, ax = plt.subplots(figsize=(3.45, 2.75))
        grouped_bars(ax, ["Wrist", "Whole-hand", "Bimanual"], a, b, "Error [mm]")
        rows = [{"statistic": suffix, "metric": label, A_NAME: av, B_NAME: bv, "unit": "mm"} for label, av, bv in zip(metrics, a, b)]
        save_figure(fig, "fig02_retargeting_tradeoff", f"Fig2{'a' if stat_idx == 0 else 'b'}_{suffix}_error_grouped_bar", rows, [SOURCE_ARTIFACTS["retargeting_table"]], f"How does {suffix} wrist fidelity trade against interaction fidelity?", "A has lower wrist error, whereas B has lower whole-hand and bimanual errors.", priority, "single-column")


def feasibility_figures() -> None:
    outcomes = {A_NAME: [27, 21, 2], B_NAME: [13, 37, 0]}
    colors = ["#4DAF4A", "#F0E442", "#CC3311"]
    hatches = ["//", "..", "xx"]
    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    left = np.zeros(2)
    for i, label in enumerate(("CLEAN", "WARNING", "HARD")):
        values = [outcomes[A_NAME][i], outcomes[B_NAME][i]]
        bars = ax.barh([0, 1], values, left=left, color=colors[i], edgecolor="black", linewidth=0.7, hatch=hatches[i], label=label)
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", fontsize=8)
        left += values
    ax.set_yticks([0, 1], [A_NAME, B_NAME])
    ax.set_xlabel("Episodes")
    ax.set_xlim(0, 50)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    clean_axis(ax, "x")
    rows = [{"method": method, "clean": values[0], "warning": values[1], "hard": values[2], "episodes": 50} for method, values in outcomes.items()]
    save_figure(fig, "fig03_feasibility", "Fig3a_episode_feasibility", rows, [SOURCE_ARTIFACTS["a_validation"], SOURCE_ARTIFACTS["b_aggregate"]], "Do both retargeting methods produce mechanically usable episodes?", "B has zero HARD episodes; WARNING is a distinct non-hard classification and must not be counted as failure.", "MUST_USE", "single-column", notes="WARNING is not HARD_FAIL.")

    categories = ["Hard collision", "Hard IK", "Joint limits", "Branches"]
    a, b = [2, 0, 0, 0], [0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    grouped_bars(ax, categories, a, b, "Episodes / failures")
    ax.set_ylim(0, 2.6)
    rows = [{"failure_mode": c, A_NAME: av, B_NAME: bv} for c, av, bv in zip(categories, a, b)]
    save_figure(fig, "fig03_feasibility", "Fig3b_hard_failure_modes", rows, [SOURCE_ARTIFACTS["retargeting_table"]], "Which hard feasibility modes distinguish A and B?", "Only A has hard collision episodes (2); neither method has hard IK, joint-limit, or branch failures.", "STRONG", "single-column")


def projection_figure(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    rows: list[dict[str, Any]] = []
    for name, values, color, ls in ((A_NAME, a["projection"] * 1000, A_COLOR, "-"), (B_NAME, b["projection"] * 1000, B_COLOR, "--")):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, linestyle=ls, label=name)
        s = {"median": np.median(x), "p95": np.percentile(x, 95), "max": x.max(), "mean": x.mean()}
        rows.extend({"method": name, "statistic": k, "value_mm": float(v), "sample_level": "arm-frame"} for k, v in s.items())
        ax.scatter([s["p95"]], [0.95], color=color, marker=A_MARKER if name == A_NAME else B_MARKER, zorder=3)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Projection magnitude [mm]")
    ax.set_ylabel("Cumulative probability")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax, "both")
    ax.text(0.02, 0.60, "B: median = 0 mm\np95 = 0 mm\nmax = 84.4 mm", transform=ax.transAxes, fontsize=7.5, va="top")
    save_figure(fig, "fig04_error_distributions", "Fig4_projection_ecdf", rows, [SOURCE_ARTIFACTS["a_manifest"], SOURCE_ARTIFACTS["b_manifest"]], "How are feasibility projection magnitudes distributed beyond their means?", "B has zero median and p95 projection but a nonzero 84.430 mm maximum; A has a broader nonzero tail.", "STRONG", "single-column", notes="ECDF uses all arm-frame values from the frozen 50-episode trajectories.")


def paired_figure(per_episode: list[dict[str, Any]], metric: str, label: str, suffix: str) -> None:
    akey, bkey = f"a_{metric}_mean_mm", f"b_{metric}_mean_mm"
    a = np.array([row[akey] for row in per_episode]); b = np.array([row[bkey] for row in per_episode])
    fig, ax = plt.subplots(figsize=(3.25, 3.0))
    for av, bv in zip(a, b):
        ax.plot([0, 1], [av, bv], color="#BBBBBB", linewidth=0.65, alpha=0.7, zorder=1)
    ax.scatter(np.zeros(len(a)), a, color=A_COLOR, marker=A_MARKER, edgecolor="black", linewidth=0.4, zorder=2)
    ax.scatter(np.ones(len(b)), b, color=B_COLOR, marker=B_MARKER, edgecolor="black", linewidth=0.4, zorder=2)
    ax.set_xticks([0, 1], ["A", "B"])
    ax.set_ylabel(f"Per-episode mean {label} [mm]")
    ax.set_xlim(-0.35, 1.35)
    clean_axis(ax)
    a_lower, b_lower = int(np.sum(a < b)), int(np.sum(b < a))
    ties = int(len(a) - a_lower - b_lower)
    ax.text(0.5, 1.02, f"A < B: {a_lower}   B < A: {b_lower}   ties: {ties}", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3)
    rows = [{"episode_index": row["episode_index"], "stable_episode_id": row["stable_episode_id"], A_NAME: row[akey], B_NAME: row[bkey], "lower_method": "A" if row[akey] < row[bkey] else ("B" if row[bkey] < row[akey] else "tie"), "unit": "mm"} for row in per_episode]
    save_figure(fig, "fig05_per_episode_pairing", f"Fig5{suffix}_paired_{metric}", rows, [SOURCE_ARTIFACTS["a_manifest"], SOURCE_ARTIFACTS["b_manifest"]], f"Is the A/B difference in {label} consistent across paired source episodes?", f"Across 50 paired source identities, A < B in {a_lower} episodes and B < A in {b_lower} episodes.", "STRONG", "single-column")


def tradeoff_scatter(per_episode: list[dict[str, Any]], ymetric: str, ylabel: str, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    rows = []
    for method, color, marker in (("A", A_COLOR, A_MARKER), ("B", B_COLOR, B_MARKER)):
        x = np.array([row[f"{method.lower()}_wrist_mean_mm"] for row in per_episode])
        y = np.array([row[f"{method.lower()}_{ymetric}_mean_mm"] for row in per_episode])
        ax.scatter(x, y, color=color, marker=marker, edgecolor="black", linewidth=0.4, alpha=0.83, label=A_NAME if method == "A" else B_NAME)
        rows.extend({"episode_index": row["episode_index"], "stable_episode_id": row["stable_episode_id"], "method": A_NAME if method == "A" else B_NAME, "wrist_error_mm": xv, f"{ymetric}_error_mm": yv} for row, xv, yv in zip(per_episode, x, y))
    ax.set_xlabel("Per-episode mean wrist error [mm]")
    ax.set_ylabel(f"Per-episode mean {ylabel} [mm]")
    ax.legend(frameon=False)
    clean_axis(ax, "both")
    save_figure(fig, "fig06_interaction_tradeoff", f"Fig6{suffix}_wrist_vs_{ymetric}", rows, [SOURCE_ARTIFACTS["a_manifest"], SOURCE_ARTIFACTS["b_manifest"]], f"How do episode-level wrist and {ylabel} errors trade off?", "A occupies the low-wrist/higher-interaction region, while B shifts toward higher wrist/lower-interaction error.", "STRONG", "single-column")


def policy_figures(data: dict[str, Any]) -> None:
    table = data["table2"]
    action_metrics = [("First action", "first-action RMSE"), ("Full chunk", "full valid chunk RMSE")]
    a = [float(lookup(table, key, "ACT-A40")) for _, key in action_metrics]
    b = [float(lookup(table, key, "ACT-B40")) for _, key in action_metrics]
    fig, ax = plt.subplots(figsize=(3.3, 2.6)); grouped_bars(ax, [x[0] for x in action_metrics], a, b, "Action RMSE [rad]")
    rows = [{"metric": label, "ACT-A": av, "ACT-B": bv, "unit": "rad"} for (label, _), av, bv in zip(action_metrics, a, b)]
    save_figure(fig, "fig07_policy_prediction", "Fig7a_action_rmse", rows, [SOURCE_ARTIFACTS["experiment2"]], "How accurately do ACT-A and ACT-B predict their own heldout action targets?", "First-action RMSE is similar, while ACT-B has lower full-chunk RMSE.", "STRONG", "single-column")

    geometry = [("Wrist", "predicted source wrist error mean"), ("Whole-hand", "predicted whole-hand error mean"), ("Bimanual", "predicted bimanual relation error mean")]
    a = [float(lookup(table, key, "ACT-A40")) for _, key in geometry]
    b = [float(lookup(table, key, "ACT-B40")) for _, key in geometry]
    fig, ax = plt.subplots(figsize=(3.45, 2.75)); grouped_bars(ax, [x[0] for x in geometry], a, b, "Predicted geometry error [mm]")
    rows = [{"metric": label, "ACT-A": av, "ACT-B": bv, "unit": "mm", "statistic": "mean"} for (label, _), av, bv in zip(geometry, a, b)]
    save_figure(fig, "fig07_policy_prediction", "Fig7b_predicted_geometry", rows, [SOURCE_ARTIFACTS["experiment2"]], "Do retargeting supervision characteristics appear in heldout policy predictions?", "ACT-A has lower predicted wrist error; ACT-B has lower predicted whole-hand and bimanual errors.", "MUST_USE", "single-column")


def preservation_figure(data: dict[str, Any]) -> None:
    table1, table2 = data["table1"], data["table2"]
    definitions = [
        ("Wrist", "Wrist error mean / p95 / max", "predicted source wrist error mean"),
        ("Whole-hand", "Whole-hand error mean / p95 / max", "predicted whole-hand error mean"),
        ("Bimanual", "Bimanual relation error mean / p95 / max", "predicted bimanual relation error mean"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
    rows = []
    for ax, (label, ret_key, pred_key) in zip(axes, definitions):
        target = [parse_triple(lookup(table1, ret_key, method))[0] for method in ("FAIR A", "PROPOSED B")]
        pred = [float(lookup(table2, pred_key, method)) for method in ("ACT-A40", "ACT-B40")]
        for j, (color, marker, method) in enumerate(((A_COLOR, A_MARKER, "A"), (B_COLOR, B_MARKER, "B"))):
            ax.plot([0, 1], [target[j], pred[j]], color=color, linestyle="-" if j == 0 else "--", marker=marker, label=method)
            rows.append({"metric": label, "method": method, "level": "Retargeted target", "mean_error_mm": target[j], "sample_set": "full-50 frames"})
            rows.append({"metric": label, "method": method, "level": "ACT prediction", "mean_error_mm": pred[j], "sample_set": "heldout8 predicted chunks"})
        ax.set_xticks([0, 1], ["Target", "ACT"])
        ax.set_title(label)
        ax.set_ylabel("Mean error [mm]" if ax is axes[0] else "")
        clean_axis(ax)
    handles = [Line2D([0], [0], color=A_COLOR, marker=A_MARKER, label="A / ACT-A"), Line2D([0], [0], color=B_COLOR, marker=B_MARKER, linestyle="--", label="B / ACT-B")]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(wspace=0.35, top=0.78)
    save_figure(fig, "fig07_policy_prediction", "Fig8_supervision_policy_preservation", rows, [SOURCE_ARTIFACTS["retargeting_table"], SOURCE_ARTIFACTS["experiment2"]], "Are the representation-specific error patterns preserved from retargeted targets to policy predictions?", "The directional pattern is preserved: A/ACT-A is wrist-oriented, whereas B/ACT-B is interaction-oriented.", "MUST_USE", "double-column", notes="Definitions and SI units match, but target values use all full-50 frames whereas policy values use HELDOUT8 predicted chunks; connectors are not paired before/after estimates.")


PHASE_ORDER = ["LEFT_APPROACH", "LEFT_GRASP", "LEFT_TRANSPORT", "RIGHT_HANDOFF_APPROACH", "DUAL_HAND_CONFIGURATION", "RIGHT_OWNED", "RIGHT_TRANSPORT", "RELEASE"]
PHASE_LABELS = ["Left\napproach", "Left\ngrasp", "Left\ntransport", "Handoff\napproach", "Dual-hand\nconfig.", "Right\nowned", "Right\ntransport", "Release"]


def phase_figures(data: dict[str, Any]) -> list[dict[str, Any]]:
    exp2 = data["exp2"]
    heldout = read_json(SOURCE_ARTIFACTS["heldout8"])["split_contract"]["heldout_final_dataset_indices"]
    matrices = {}
    rows = []
    for method in ("a", "b"):
        behaviors = exp2["methods"][method]["selected_checkpoint_evaluation"]["phase_score"]["behaviors"]
        matrix = np.zeros((len(heldout), len(PHASE_ORDER)), dtype=int)
        for j, phase in enumerate(PHASE_ORDER):
            by_episode = {int(item["final_episode"]): int(bool(item["success"])) for item in behaviors[phase]["per_episode"]}
            for i, episode in enumerate(heldout):
                matrix[i, j] = by_episode[episode]
                rows.append({"method": "ACT-A" if method == "a" else "ACT-B", "episode_index": episode, "phase": phase, "phase_label": PHASE_LABELS[j].replace("\n", " "), "detected": int(matrix[i, j])})
        matrices[method] = matrix

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.3), sharex=True)
    cmap = matplotlib.colors.ListedColormap(["#EEEEEE", "#2E8B57"])
    for ax, method in zip(axes, ("a", "b")):
        ax.imshow(matrices[method], vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(heldout)), [f"ep{e:02d}" for e in heldout])
        ax.set_ylabel("ACT-A" if method == "a" else "ACT-B")
        ax.set_xticks(range(8), PHASE_LABELS)
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True); ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1); ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(matrices[method][i, j]), ha="center", va="center", fontsize=6.5, color="white" if matrices[method][i, j] else "#555555")
    fig.subplots_adjust(hspace=0.12)
    save_figure(fig, "fig08_phase_behavior", "Fig9a_phase_behavior_heatmap", rows, [SOURCE_ARTIFACTS["experiment2"], SOURCE_ARTIFACTS["heldout8"]], "Which of the eight paper-core behaviors are detected in each heldout episode?", "ACT-A detects 56/64 behaviors and ACT-B detects 52/64 under the exact eight-behavior paper taxonomy.", "STRONG", "double-column", notes="This is the exact 8-behavior taxonomy underlying 56/64 and 52/64; the separate 9-probe diagnostic taxonomy is excluded.")

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(8); width = 0.34
    av = matrices["a"].mean(axis=0) * 100; bv = matrices["b"].mean(axis=0) * 100
    ax.bar(x - width / 2, av, width, color=A_COLOR, hatch="//", edgecolor="black", linewidth=0.6, label="ACT-A")
    ax.bar(x + width / 2, bv, width, color=B_COLOR, hatch="..", edgecolor="black", linewidth=0.6, label="ACT-B")
    ax.set_xticks(x, PHASE_LABELS); ax.set_ylabel("Detected [%]"); ax.set_ylim(0, 108); ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    clean_axis(ax)
    bar_rows = [{"phase": phase, "phase_label": label.replace("\n", " "), "ACT-A_success_count": int(matrices["a"][:, j].sum()), "ACT-B_success_count": int(matrices["b"][:, j].sum()), "total": 8, "ACT-A_percent": float(av[j]), "ACT-B_percent": float(bv[j])} for j, (phase, label) in enumerate(zip(PHASE_ORDER, PHASE_LABELS))]
    save_figure(fig, "fig08_phase_behavior", "Fig9b_per_phase_success", bar_rows, [SOURCE_ARTIFACTS["experiment2"]], "Which heldout behaviors account for the aggregate phase scores?", "Per-phase detection reveals where ACT-A and ACT-B differ without mixing the 8- and 9-phase taxonomies.", "MUST_USE", "double-column")
    return rows


def smoothness_figure(data: dict[str, Any]) -> None:
    table = data["table2"]
    definitions = [
        ("Reversals", "raw all-joint direction reversals mean", "s⁻¹ joint⁻¹"),
        ("Max step", "raw maximum adjacent step", "rad"),
        ("q̇ RMS", "raw qdot RMS", "rad s⁻¹"),
        ("q̈ RMS", "raw qddot RMS", "rad s⁻²"),
        ("Jerk RMS", "raw jerk RMS", "rad s⁻³"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 2.35))
    rows = []
    for ax, (label, key, unit) in zip(axes, definitions):
        values = [float(lookup(table, key, method)) for method in ("ACT-A40", "ACT-B40")]
        bars = ax.bar([0, 1], values, color=[A_COLOR, B_COLOR], edgecolor="black", linewidth=0.6, hatch=["//", ".."])
        ax.set_xticks([0, 1], ["A", "B"]); ax.set_title(label); ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.23); clean_axis(ax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=6.6)
        rows.append({"metric": label, "ACT-A": values[0], "ACT-B": values[1], "unit": unit})
    fig.subplots_adjust(wspace=0.65)
    save_figure(fig, "fig09_policy_smoothness", "Fig10_policy_control_regularity", rows, [SOURCE_ARTIFACTS["experiment2"]], "Do the selected policies pass a transparent control-regularity comparison?", "The metrics are mixed: neither policy is uniformly smoother across reversals, step, velocity, acceleration, and jerk.", "OPTIONAL", "double-column", notes="Quality-control comparison only; not a core interaction claim.")


def representative_episode(per_episode: list[dict[str, Any]]) -> tuple[int, float]:
    common = {int(row["final_dataset_index"]) for row in read_json(SOURCE_ARTIFACTS["common48"])["entries"]}
    candidates = [row for row in per_episode if row["episode_index"] in common]
    median = float(np.median([row["b_whole_hand_mean_mm"] for row in candidates]))
    # With an even number of candidates, the two central values are equidistant
    # from the median by definition.  Round only the distance used for the
    # declared tie-break so floating-point accumulation cannot choose the larger
    # episode index by a few femtometres.
    selected = min(candidates, key=lambda row: (round(abs(row["b_whole_hand_mean_mm"] - median), 9), row["episode_index"]))
    return int(selected["episode_index"]), median


def trajectory_figures(per_episode: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
    episode, median = representative_episode(per_episode)
    a_row = read_json(SOURCE_ARTIFACTS["a_manifest"])["trajectories"][episode]
    b_row = read_json(SOURCE_ARTIFACTS["b_manifest"])["episodes"][episode]
    a_path, b_path = Path(a_row["trajectory_path"]), Path(b_row["retargeted_trajectory_path"])
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        rows = []
        for ax, side in zip(axes, ("left", "right")):
            ref = np.asarray(a[f"source_{side}_realization_frame_position_world"]) * 1000
            aa = np.asarray(a[f"achieved_{side}_wrist_position_world"]) * 1000
            bb = np.asarray(b[f"achieved_{side}_wrist_position_world"]) * 1000
            for values, color, ls, label in ((ref, "#333333", ":", "ALOHA reference"), (aa, A_COLOR, "-", A_NAME), (bb, B_COLOR, "--", B_NAME)):
                ax.plot(values[:, 0], values[:, 1], color=color, linestyle=ls, label=label)
            ax.scatter(ref[0, 0], ref[0, 1], marker="^", color="#333333", s=25, zorder=4)
            ax.set_title(f"{side.capitalize()} wrist"); ax.set_xlabel("World x [mm]"); ax.set_ylabel("World y [mm]")
            ax.set_aspect("equal", adjustable="datalim"); clean_axis(ax, "both")
            stride = max(1, len(ref) // 140)
            for frame in range(0, len(ref), stride):
                rows.append({"episode_index": episode, "side": side, "frame": frame, "reference_x_mm": ref[frame, 0], "reference_y_mm": ref[frame, 1], "a_x_mm": aa[frame, 0], "a_y_mm": aa[frame, 1], "b_x_mm": bb[frame, 0], "b_y_mm": bb[frame, 1]})
        axes[0].legend(frameon=False, ncol=3, bbox_to_anchor=(1.05, 1.25), loc="upper center")
        save_figure(fig, "fig10_trajectory_examples", "Fig11a_representative_wrist_paths", rows, [a_path, b_path, SOURCE_ARTIFACTS["common48"]], "How do source-matched wrist paths differ for an objectively selected representative episode?", f"Episode {episode} was selected by the declared median-B whole-hand criterion; A follows the wrist reference more closely while B follows its interaction representation.", "STRONG", "double-column", notes=f"Selection median = {median:.6f} mm; no visual criterion was used.")

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        rows = []
        for ax, side in zip(axes, ("left", "right")):
            ref = np.asarray(a[f"target_{side}_interaction_frame_position_world"]) * 1000
            aa = np.asarray(a[f"achieved_{side}_physical_grasp_frame_position_world"]) * 1000
            bb = np.asarray(b[f"achieved_{side}_physical_grasp_frame_position_world"]) * 1000
            for values, color, ls, label in ((ref, "#333333", ":", "ALOHA interaction reference"), (aa, A_COLOR, "-", A_NAME), (bb, B_COLOR, "--", B_NAME)):
                ax.plot(values[:, 0], values[:, 1], color=color, linestyle=ls, label=label)
            ax.set_title(f"{side.capitalize()} whole-hand frame"); ax.set_xlabel("World x [mm]"); ax.set_ylabel("World y [mm]")
            ax.set_aspect("equal", adjustable="datalim"); clean_axis(ax, "both")
            stride = max(1, len(ref) // 140)
            for frame in range(0, len(ref), stride):
                rows.append({"episode_index": episode, "side": side, "frame": frame, "reference_x_mm": ref[frame, 0], "reference_y_mm": ref[frame, 1], "a_x_mm": aa[frame, 0], "a_y_mm": aa[frame, 1], "b_x_mm": bb[frame, 0], "b_y_mm": bb[frame, 1]})
        axes[0].legend(frameon=False, ncol=3, bbox_to_anchor=(1.05, 1.25), loc="upper center")
        save_figure(fig, "fig10_trajectory_examples", "Fig11b_representative_whole_hand_paths", rows, [a_path, b_path, SOURCE_ARTIFACTS["common48"]], "How do whole-hand grasp-frame paths differ in the same representative episode?", f"For criterion-selected episode {episode}, B remains closer to the source interaction-frame paths.", "STRONG", "double-column")
        snapshot_frames = read_json(ROOT / f"outputs/policy_b_g1visual/dataset_render_full/episode_reports/episode_{episode:06d}.json")["snapshot_frames"]
    return episode, snapshot_frames


def semantic_strip_template(episode: int, snapshot_frames: dict[str, int]) -> dict[str, Any]:
    phases = [
        ("Left grasp", "LEFT_OWNED"),
        ("Left transport", "left_transport"),
        ("Handoff approach", "handoff_approach"),
        ("Dual hand/contact", "DUAL_CONTACT"),
        ("Right owned", "RIGHT_OWNED"),
        ("Release", "release"),
    ]
    snapshot_root = ROOT / f"outputs/policy_b_g1visual/dataset_render_full/snapshots/episode_{episode:06d}"
    camera_config = ROOT / "outputs/policy_b_g1visual/dataset_render_full/camera_family.json"
    rows = []
    fig, axes = plt.subplots(len(phases), 3, figsize=(7.1, 8.2))
    for i, (label, key) in enumerate(phases):
        frame = int(snapshot_frames[key])
        source_path = snapshot_root / f"{key}_source_aloha.png"
        b_path = snapshot_root / f"{key}_cam_high.png"
        for j, (title, path) in enumerate((("Source ALOHA", source_path), (A_NAME, None), (B_NAME, b_path))):
            ax = axes[i, j]; ax.axis("off")
            if path is None:
                ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F4F4F4", edgecolor="#888888", linestyle="--"))
                ax.text(0.5, 0.5, f"RENDER_REQUIRED\nframe {frame}\ncam_high", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="#555555")
            else:
                ax.imshow(plt.imread(path))
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.text(-0.05, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=8)
            rows.append({"episode_index": episode, "semantic_state": label, "snapshot_key": key, "frame": frame, "column": title, "asset": rel(path) if path else "NA", "status": "AVAILABLE" if path else "RENDER_REQUIRED", "camera": "cam_high"})
    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    save_figure(fig, "fig10_trajectory_examples", "Fig12_semantic_frame_sequence_TEMPLATE", rows, [SOURCE_ARTIFACTS["common48"], camera_config] + [snapshot_root / f"{key}_source_aloha.png" for _, key in phases] + [snapshot_root / f"{key}_cam_high.png" for _, key in phases], "What visual configurations occur at six semantic handoff states for the same representative source?", "Source and B assets exist; matched A renders are explicitly pending and no GPU rendering was launched.", "OPTIONAL", "double-column", gpu_required=True, notes="Incomplete layout template; do not publish until A cells are rendered and the template is regenerated.")
    manifest = {
        "status": "RENDER_REQUIRED",
        "gpu_used": False,
        "representative_episode": episode,
        "selection_rule": "episode nearest to median B whole-hand interaction error among common feasible episodes; final index breaks ties",
        "camera": "observation.images.cam_high",
        "camera_config": rel(camera_config),
        "required_fair_a_trajectory": next(row["trajectory_path"] for row in read_json(SOURCE_ARTIFACTS["a_manifest"])["trajectories"] if int(row["episode_index"]) == episode),
        "states": [{"semantic_state": label, "frame": int(snapshot_frames[key]), "snapshot_key": key, "required_output": f"Fair-A_episode_{episode:06d}_{key}_cam_high.png"} for label, key in phases],
        "source_and_b_assets_already_available": True,
        "isaac_render_launched": False,
    }
    write_json(OUT / "fig10_trajectory_examples/Fig12_RENDER_REQUIRED.json", manifest)
    (OUT / "fig10_trajectory_examples/RENDER_REQUIRED").write_text("Matched Fair-A semantic frames are not available. See Fig12_RENDER_REQUIRED.json.\n", encoding="utf-8")
    return manifest


def failure_figure() -> None:
    collision = read_json(SOURCE_ARTIFACTS["collision_audit"])["episodes"]
    candidates = [row for row in collision if int(row["episode_index"]) in (35, 46)]
    selected = max(candidates, key=lambda row: (float(row["after"]["maximum_penetration_m"]), -int(row["episode_index"])))
    episode = int(selected["episode_index"])
    a_path = Path(read_json(SOURCE_ARTIFACTS["a_manifest"])["trajectories"][episode]["trajectory_path"])
    b_path = Path(read_json(SOURCE_ARTIFACTS["b_manifest"])["episodes"][episode]["retargeted_trajectory_path"])
    records = [row for row in read_csv(SOURCE_ARTIFACTS["collision_records"]) if int(row["episode_index"]) == episode]
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        n = len(a["timestamp"]); t = np.arange(n) / 30.0
        penetration = np.zeros(n)
        hard = np.zeros(n, dtype=int)
        pair = np.full(n, "NONE", dtype=object)
        for row in records:
            frame = int(row["frame"]); value = float(row["penetration_m"])
            if value >= penetration[frame]:
                penetration[frame] = value; pair[frame] = row["body_pair"]
            hard[frame] = max(hard[frame], int(row["shared_severity_hard_frame"].lower() == "true"))
        a_proj = np.asarray(a["feasibility_projection_translation_m"]).max(axis=1) * 1000
        b_proj = np.asarray(b["feasibility_projection_translation_m"]).max(axis=1) * 1000
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.7), sharex=True)
    axes[0].fill_between(t, 0, penetration * 1000, where=hard.astype(bool), color=A_COLOR, alpha=0.35, step="mid")
    axes[0].plot(t, penetration * 1000, color=A_COLOR, label=f"{A_NAME}: penetration")
    axes[0].plot(t, np.zeros_like(t), color=B_COLOR, linestyle="--", label=f"{B_NAME}: no hard-fail frame")
    axes[0].set_ylabel("Penetration [mm]"); axes[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0); clean_axis(axes[0])
    axes[1].plot(t, a_proj, color=A_COLOR, label="A projection")
    axes[1].plot(t, b_proj, color=B_COLOR, linestyle="--", label="B projection")
    axes[1].set_ylabel("Projection [mm]"); axes[1].set_xlabel("Time [s]"); axes[1].legend(frameon=False, ncol=2); clean_axis(axes[1])
    peak_frame = int(np.argmax(penetration)); axes[0].annotate(f"{str(pair[peak_frame]).replace('|', ' / ')}\npeak {penetration[peak_frame]*1000:.2f} mm", xy=(t[peak_frame], penetration[peak_frame] * 1000), xytext=(t[peak_frame] + 1.2, penetration[peak_frame] * 850), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=7)
    rows = [{"episode_index": episode, "frame": i, "time_s": t[i], "a_hard_collision": int(hard[i]), "a_penetration_mm": penetration[i] * 1000, "a_peak_body_pair": pair[i], "a_projection_mm": a_proj[i], "b_hard_collision": 0, "b_projection_mm": b_proj[i]} for i in range(n)]
    save_figure(fig, "fig11_failure_examples", "Fig13_fair_a_hard_collision_case", rows, [SOURCE_ARTIFACTS["collision_audit"], SOURCE_ARTIFACTS["collision_records"], a_path, b_path], "What frozen temporal evidence characterizes a Fair-A hard collision case?", f"Episode {episode} is selected by the larger frozen peak penetration among ep35/ep46; A has a sustained arm-torso hard collision while B is hard-fail free for the same source.", "STRONG", "double-column", notes="Selection criterion was frozen maximum penetration, not visual drama. Collision model covers robot self-collision only.")


def bootstrap_effects(per_episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260827)
    results = []
    for metric, label in (("wrist", "Wrist error"), ("whole_hand", "Whole-hand error"), ("bimanual", "Bimanual relation"), ("projection", "Projection magnitude")):
        diff = np.array([row[f"b_{metric}_mean_mm"] - row[f"a_{metric}_mean_mm"] for row in per_episode], dtype=float)
        indices = rng.integers(0, len(diff), size=(20000, len(diff)))
        boot = diff[indices].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        results.append({"metric": label, "paired_B_minus_A_mean_mm": float(diff.mean()), "ci95_low_mm": float(low), "ci95_high_mm": float(high), "episodes": len(diff), "bootstrap_resamples": 20000, "seed": 20260827, "direction_note": "negative favors B because lower error is better"})
    return results


def statistical_figure(per_episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effects = bootstrap_effects(per_episode)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    y = np.arange(len(effects))[::-1]
    means = np.array([row["paired_B_minus_A_mean_mm"] for row in effects])
    low = np.array([row["ci95_low_mm"] for row in effects]); high = np.array([row["ci95_high_mm"] for row in effects])
    colors = [B_COLOR if value < 0 else A_COLOR for value in means]
    for x, yy, lo, hi, color in zip(means, y, low, high, colors):
        ax.errorbar(x, yy, xerr=[[x - lo], [hi - x]], fmt="none", ecolor=color, elinewidth=1.7, capsize=3)
    for x, yy, color, marker in zip(means, y, colors, [A_MARKER, B_MARKER, B_MARKER, B_MARKER]):
        ax.scatter(x, yy, color=color, marker=marker, edgecolor="black", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_yticks(y, [row["metric"] for row in effects]); ax.set_xlabel("Paired difference, B − A [mm]\n(negative favors B; positive favors A)")
    clean_axis(ax, "x")
    save_figure(fig, "fig12_statistical_summary", "Fig14_paired_effect_forest", effects, [SOURCE_ARTIFACTS["a_manifest"], SOURCE_ARTIFACTS["b_manifest"]], "What are the paired episode-level effect directions and uncertainty?", "B reduces whole-hand, bimanual, and projection means; A reduces wrist error. Paired bootstrap intervals quantify uncertainty without a significance claim.", "MUST_USE", "single-column", notes="Effects are means of 50 per-episode mean differences; 20,000 paired bootstrap resamples with fixed seed 20260827.")
    return effects


def storyboard_mockup() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
    panel_text = [
        ("A", "Representation schematic\nTrajectory-centric ↔ Interaction-centric"),
        ("B", "Retargeting contrast\nWrist vs whole-hand / bimanual"),
        ("C", "Heldout ACT prediction\nRepresentation pattern persists"),
        ("D", "Representative semantic frames\nRENDER_REQUIRED: matched Fair-A assets"),
    ]
    for ax, (letter, text) in zip(axes.ravel(), panel_text):
        ax.axis("off"); ax.add_patch(patches.Rectangle((0.02, 0.04), 0.96, 0.90, transform=ax.transAxes, facecolor="#F7F7F7", edgecolor="#777777", linestyle="--" if letter == "D" else "-"))
        ax.text(0.05, 0.89, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")
        ax.text(0.5, 0.48, text, transform=ax.transAxes, ha="center", va="center", fontsize=9)
    rows = [{"panel": letter, "planned_content": text.replace("\n", " "), "status": "RENDER_REQUIRED" if letter == "D" else "AVAILABLE_FROM_BANK"} for letter, text in panel_text]
    save_figure(fig, "fig12_statistical_summary", "Fig15_paper_storyboard_LAYOUT", rows, [SOURCE_ARTIFACTS["retargeting_table"], SOURCE_ARTIFACTS["experiment2"]], "How can the strongest quantitative and qualitative panels be assembled as a main-paper story?", "Panels A-C are available from the bank; panel D remains an explicit matched-render placeholder.", "OPTIONAL", "double-column", gpu_required=True, notes="Layout mockup only; do not present as a scientific result.")


def experiment3_template() -> None:
    metrics = ["Semantic task success", "Phase completion", "Whole-hand error", "Bimanual error", "Handoff ordering", "RPL", "SWPE"]
    rows = [{"metric": metric, A_NAME: "NA", B_NAME: "NA", "unit": "NA", "status": "NA"} for metric in metrics]
    fig, ax = plt.subplots(figsize=(7.0, 2.7)); ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch((0.04, 0.12), 0.92, 0.74, transform=ax.transAxes, boxstyle="round,pad=0.02", facecolor="#F7F7F7", edgecolor="#888888", linestyle="--"))
    ax.text(0.5, 0.64, "Experiment 3 — source-conditioned rollout", transform=ax.transAxes, ha="center", fontsize=11)
    ax.text(0.5, 0.43, "A: NA     B: NA", transform=ax.transAxes, ha="center", fontsize=16, color="#555555")
    ax.text(0.5, 0.25, "Template only • no rollout values used", transform=ax.transAxes, ha="center", fontsize=9)
    save_figure(fig, "fig12_statistical_summary", "Fig16_source_conditioned_rollout_metrics_TEMPLATE", rows, [], "How will source-conditioned rollout metrics be compared when frozen Experiment 3 results become publishable?", "No Experiment 3 result is used; every value is explicitly NA.", "OPTIONAL", "double-column", notes="Template only. Populate using populate_experiment3_template.py after supplying an approved CSV/JSON.")

    populate = OUT / "fig12_statistical_summary/populate_experiment3_template.py"
    source = '''#!/usr/bin/env python3
"""Populate the Experiment-3 template from an explicitly supplied frozen CSV/JSON."""
import argparse, csv, json
from pathlib import Path

REQUIRED = ["Semantic task success", "Phase completion", "Whole-hand error", "Bimanual error", "Handoff ordering", "RPL", "SWPE"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment3_values.csv"))
    args = parser.parse_args()
    if args.artifact.suffix.lower() == ".json":
        payload = json.loads(args.artifact.read_text())
        rows = payload.get("metrics", payload)
        if isinstance(rows, dict):
            rows = [{"metric": key, **value} for key, value in rows.items()]
    else:
        with args.artifact.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    by_metric = {row["metric"]: row for row in rows}
    missing = [metric for metric in REQUIRED if metric not in by_metric]
    if missing:
        raise SystemExit(f"missing required metrics: {missing}")
    for metric in REQUIRED:
        for method in ("Trajectory-Centric A", "Interaction-Centric B"):
            if str(by_metric[metric].get(method, "NA")).upper() == "NA":
                raise SystemExit(f"approved value still NA: {metric}/{method}")
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_metric[REQUIRED[0]]))
        writer.writeheader(); writer.writerows(by_metric[m] for m in REQUIRED)
    print(args.output)

if __name__ == "__main__":
    main()
'''
    populate.write_text(source, encoding="utf-8"); populate.chmod(0o755)


def table_outputs(data: dict[str, Any], per_episode: list[dict[str, Any]], phase_rows: list[dict[str, Any]], effects: list[dict[str, Any]]) -> None:
    table_dir = OUT / "tables"

    table1 = []
    for row in data["table1"]:
        table1.append({"Metric": row["METRIC"], A_NAME: row["FAIR A"], B_NAME: row["PROPOSED B"], "Unit": row["UNIT"]})
    table2 = [{"Metric": row["metric"], "ACT-A": row["ACT-A40"], "ACT-B": row["ACT-B40"], "Unit": row["unit"]} for row in data["table2"]]
    losses = data["train"]["methods"]
    table2.insert(0, {"Metric": "Final logged training loss", "ACT-A": losses["a"]["last_logged_health"]["loss"], "ACT-B": losses["b"]["last_logged_health"]["loss"], "Unit": "loss"})
    table3 = [{"Metric": metric, A_NAME: "NA", B_NAME: "NA", "Unit": "NA", "Status": "TEMPLATE_ONLY"} for metric in ("Semantic task success", "Phase completion", "Whole-hand error", "Bimanual error", "Handoff ordering", "RPL", "SWPE")]
    s1 = per_episode
    grouped: dict[tuple[str, str], int] = {}
    for row in phase_rows:
        grouped[(row["method"], row["phase"])] = grouped.get((row["method"], row["phase"]), 0) + int(row["detected"])
    s2 = [{"Phase": phase, "Label": label.replace("\n", " "), "ACT-A": grouped[("ACT-A", phase)], "ACT-B": grouped[("ACT-B", phase)], "Total": 8} for phase, label in zip(PHASE_ORDER, PHASE_LABELS)]
    table = data["table2"]
    s3_defs = [("Direction reversals", "raw all-joint direction reversals mean", "reversals/s/joint"), ("Maximum adjacent step", "raw maximum adjacent step", "rad"), ("qdot RMS", "raw qdot RMS", "rad/s"), ("qddot RMS", "raw qddot RMS", "rad/s^2"), ("Jerk RMS", "raw jerk RMS", "rad/s^3")]
    s3 = [{"Metric": label, "ACT-A": lookup(table, key, "ACT-A40"), "ACT-B": lookup(table, key, "ACT-B40"), "Unit": unit} for label, key, unit in s3_defs]
    tables = [
        ("Table1_full50_retargeting", table1, [SOURCE_ARTIFACTS["retargeting_table"]], "Full-50 retargeting comparison."),
        ("Table2_heldout_ACT", table2, [SOURCE_ARTIFACTS["policy_table"], SOURCE_ARTIFACTS["training_audit"]], "Heldout ACT-A versus ACT-B comparison."),
        ("Table3_source_conditioned_rollout_TEMPLATE", table3, [], "Source-conditioned rollout template; all values are NA."),
        ("TableS1_per_episode_retargeting", s1, [SOURCE_ARTIFACTS["a_manifest"], SOURCE_ARTIFACTS["b_manifest"]], "Per-episode retargeting metrics for 50 paired source identities."),
        ("TableS2_per_phase_ACT", s2, [SOURCE_ARTIFACTS["experiment2"]], "Per-phase heldout behavior detections under the exact eight-behavior taxonomy."),
        ("TableS3_smoothness_control", s3, [SOURCE_ARTIFACTS["experiment2"]], "Policy prediction smoothness and control-regularity metrics."),
        ("TableS4_paired_bootstrap_effects", effects, [SOURCE_ARTIFACTS["a_manifest"], SOURCE_ARTIFACTS["b_manifest"]], "Paired B-minus-A effects and bootstrap confidence intervals."),
    ]
    for stem, rows, sources, caption in tables:
        write_rows(table_dir / f"{stem}.csv", rows)
        write_json(table_dir / f"{stem}.json", rows)
        fields = list(rows[0]) if rows else []
        md = "| " + " | ".join(fields) + " |\n| " + " | ".join(["---"] * len(fields)) + " |\n"
        for row in rows:
            md += "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n"
        (table_dir / f"{stem}.md").write_text(md, encoding="utf-8")
        latex = "\\begin{tabular}{" + "l" * len(fields) + "}\n\\toprule\n" + " & ".join(field.replace("_", "\\_") for field in fields) + " \\\\\n\\midrule\n"
        for row in rows:
            latex += " & ".join(str(row.get(field, "")).replace("_", "\\_") for field in fields) + " \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}\n"
        (table_dir / f"{stem}.tex").write_text(latex, encoding="utf-8")
        (table_dir / f"{stem}_generation_command.txt").write_text("python3 tools/generate_paper_figure_bank.py\n", encoding="utf-8")
        write_json(table_dir / f"{stem}_metadata.json", {"table": stem, "caption": caption, "source_artifacts": artifact_records(sources), "gpu_used": False, "experiment3_values_used": False if "Table3" in stem else "not_applicable"})


CAPTIONS = {
    "fig01_caption.txt": "Fig. 1. Paper-core pipeline. The same 50 ALOHA demonstrations are retargeted using Trajectory-Centric A or Interaction-Centric B, packaged as matched G1 datasets, and used to supervise ACT-A and ACT-B. Evaluation covers retargeting, heldout action prediction, and phase behavior; source-conditioned rollout remains a future optional evaluation.",
    "fig02_caption.txt": "Fig. 2. Comparison of trajectory-centric and interaction-centric retargeting. Trajectory-Centric A preserves the source wrist trajectory more accurately, whereas Interaction-Centric B reduces whole-hand and bimanual interaction errors. Separate panels report mean and 95th-percentile errors in millimetres.",
    "fig03_caption.txt": "Fig. 3. Retargeting feasibility over 50 source episodes. Trajectory-Centric A yields 27 clean, 21 warning, and 2 hard-fail episodes; Interaction-Centric B yields 13 clean, 37 warning, and no hard-fail episodes. WARNING denotes a usable episode with a reported diagnostic and is not equivalent to HARD_FAIL.",
    "fig04_caption.txt": "Fig. 4. Empirical cumulative distribution of arm-frame projection magnitudes. Interaction-Centric B has median and 95th-percentile projection of 0 mm but retains an 84.430 mm maximum, so the distribution is not represented by its mean alone.",
    "fig05_caption.txt": "Fig. 5. Paired per-episode retargeting errors for the same 50 source identities. Each line links Trajectory-Centric A and Interaction-Centric B for one episode; the annotations report the number of episodes favouring each method under the lower-is-better convention.",
    "fig06_caption.txt": "Fig. 6. Episode-level trade-off between wrist fidelity and interaction fidelity. Trajectory-Centric A clusters at lower wrist error and higher interaction error, whereas Interaction-Centric B shifts toward higher wrist deviation and lower whole-hand or bimanual error.",
    "fig07_caption.txt": "Fig. 7. Heldout ACT prediction. First-action and full-chunk action RMSE are shown separately from geometric errors because their units differ. ACT-A has lower predicted wrist error, whereas ACT-B has lower predicted whole-hand and bimanual errors.",
    "fig08_caption.txt": "Fig. 8. Preservation of representation-specific error patterns from retargeted targets to heldout policy predictions. The target statistics use all full-50 frames and the policy statistics use HELDOUT8 predicted chunks; connectors indicate qualitative pattern preservation rather than paired before-and-after measurements.",
    "fig09_caption.txt": "Fig. 9. Heldout phase-behavior detection using exactly the eight behavior categories underlying the 56/64 ACT-A and 52/64 ACT-B scores. The heatmap reports binary episode-by-behavior detections, and the bar panel reports success percentages. The separate nine-probe diagnostic taxonomy is not used.",
    "fig10_caption.txt": "Fig. 10. Policy prediction control regularity. Direction reversals, maximum joint step, velocity RMS, acceleration RMS, and jerk RMS are displayed in separate panels because their units differ. The comparison is mixed and does not support a claim that either policy is smoother overall.",
    "fig11_caption.txt": "Fig. 11. Representative source-matched trajectories. The episode was selected before visualization as the common feasible episode nearest to the median Interaction-Centric B whole-hand error, with episode index breaking an exact tie. Wrist and whole-hand paths are shown separately to avoid clutter.",
    "fig12_caption.txt": "Fig. 12. Semantic frame-sequence template for the same representative episode. Source ALOHA and Interaction-Centric B assets are available, while matched Trajectory-Centric A renders are explicitly marked RENDER_REQUIRED. No Isaac rendering was launched for this bank.",
    "fig13_caption.txt": "Fig. 13. Frozen Fair-A hard collision case selected by the larger maximum penetration among episodes 35 and 46. The timeline shows the arm-torso collision evidence and feasibility projection for Trajectory-Centric A, with the same-source Interaction-Centric B trajectory remaining free of hard-fail frames. A trajectory target can be statically reachable yet incompatible with continuous temporal and collision constraints.",
    "fig14_caption.txt": "Fig. 14. Forest summary of paired B-minus-A differences in per-episode mean error. Negative values favour Interaction-Centric B because lower error is better. Points are paired mean differences over 50 source identities and bars are 95% intervals from 20,000 paired bootstrap resamples; no claim of statistical significance is made.",
    "fig15_caption.txt": "Fig. 15. Layout mockup for a composite paper-result storyboard. Quantitative panels are available from the figure bank, while the representative matched Fair-A semantic frame panel remains RENDER_REQUIRED. This mockup is not itself a scientific result.",
    "fig16_caption.txt": "Fig. 16. Template for future source-conditioned rollout comparison. Semantic task success, phase completion, whole-hand error, bimanual error, handoff ordering, RPL, and SWPE are all explicitly NA because no Experiment 3 result is used.",
    "table01_caption.txt": "Table 1. Full-50 retargeting comparison of Trajectory-Centric A and Interaction-Centric B, including feasibility classification, hard failures, projection, wrist, whole-hand, and bimanual errors.",
    "table02_caption.txt": "Table 2. Heldout ACT-A and ACT-B action prediction, geometry, phase behavior, and control-regularity results under the frozen checkpoint-selection rule.",
    "table03_caption.txt": "Table 3. Source-conditioned rollout template. All Experiment 3 values are explicitly NA.",
    "tableS1_caption.txt": "Table S1. Per-episode retargeting metrics for the 50 paired source identities.",
    "tableS2_caption.txt": "Table S2. Per-phase ACT heldout detections under the exact eight-behavior paper taxonomy.",
    "tableS3_caption.txt": "Table S3. Smoothness and control-regularity metrics for ACT-A and ACT-B.",
    "tableS4_caption.txt": "Table S4. Paired B-minus-A effect estimates and 95% paired bootstrap confidence intervals.",
}


def write_captions() -> None:
    for name, caption in CAPTIONS.items():
        (OUT / "captions" / name).write_text(caption + "\n", encoding="utf-8")


def write_index(render_manifest: dict[str, Any]) -> None:
    lines = [
        "# Paper Figure Bank Index",
        "",
        "Generated from frozen Experiment 1 and Experiment 2 artifacts. Experiment 3 is template-only with all values NA.",
        "",
        "| Figure | Scientific question | Source artifacts | Key result | Priority | GPU render | Width |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in FIGURES:
        filename = f"{item['subdir']}/{item['stem']}.pdf"
        sources = "<br>".join(record["path"] for record in item["source_artifacts"]) or "None (template)"
        cells = [filename, item["scientific_question"], sources, item["key_result"], item["priority"], "YES" if item["gpu_render_required"] else "NO", item["column_recommendation"]]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    lines.extend(
        [
            "",
            "## Recommended paper selection",
            "",
            "- MUST_USE: Fig1 pipeline; Fig2a mean trade-off; Fig3a feasibility; Fig7b heldout predicted geometry; Fig8 supervision-to-policy preservation; Fig9b per-phase success; Fig14 paired effect forest.",
            "- STRONG: Fig2b p95 trade-off; Fig3b failure modes; Fig4 projection ECDF; Fig5 paired episode plots; Fig6 trade-off maps; Fig7a action RMSE; Fig9a phase heatmap; Fig11 representative trajectories; Fig13 hard collision case.",
            "- OPTIONAL: Fig10 control regularity; Fig12 semantic sequence template; Fig15 storyboard layout; Fig16 Experiment 3 template.",
            "",
            "## Rendering status",
            "",
            f"- Representative episode: {render_manifest['representative_episode']}",
            "- Fig12 matched Fair-A semantic frames: RENDER_REQUIRED",
            "- Fig15 composite panel D: RENDER_REQUIRED",
            "- GPU used by this task: NO",
            "- Current paper-core job touched: NO",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python3 tools/generate_paper_figure_bank.py",
            "python3 tools/validate_paper_figure_bank.py",
            "```",
        ]
    )
    (OUT / "figure_index/FIGURE_BANK_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "figure_index/figure_manifest.json", FIGURES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Accepted for recorded per-figure commands; the deterministic bank is regenerated as a unit.")
    parser.parse_args()
    mkdirs(); configure_style()
    data = verify_and_load()
    per_episode, a_frame, b_frame = load_retargeting_arrays()
    pipeline_figure()
    retargeting_tradeoff(data)
    feasibility_figures()
    projection_figure(a_frame, b_frame)
    paired_figure(per_episode, "wrist", "wrist error", "a")
    paired_figure(per_episode, "whole_hand", "whole-hand error", "b")
    paired_figure(per_episode, "bimanual", "bimanual relation error", "c")
    tradeoff_scatter(per_episode, "whole_hand", "whole-hand error", "a")
    tradeoff_scatter(per_episode, "bimanual", "bimanual relation error", "b")
    policy_figures(data)
    preservation_figure(data)
    phase_rows = phase_figures(data)
    smoothness_figure(data)
    episode, snapshot_frames = trajectory_figures(per_episode)
    render_manifest = semantic_strip_template(episode, snapshot_frames)
    failure_figure()
    effects = statistical_figure(per_episode)
    storyboard_mockup()
    experiment3_template()
    table_outputs(data, per_episode, phase_rows, effects)
    write_captions()
    write_index(render_manifest)
    summary = {
        "status": "READY_WITH_DECLARED_RENDER_PLACEHOLDERS",
        "figure_count": len(FIGURES),
        "table_count": 7,
        "caption_count": len(CAPTIONS),
        "experiment3": "TEMPLATE_ONLY",
        "experiment3_values": "ALL_NA",
        "gpu_used": False,
        "current_paper_core_job_touched": False,
        "representative_episode": episode,
        "verification": "ALL_CONFIRMED_VALUES_MATCH_FROZEN_ARTIFACTS",
    }
    write_json(OUT / "figure_index/generation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
