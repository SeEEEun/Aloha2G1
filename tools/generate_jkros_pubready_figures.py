#!/usr/bin/env python3
"""Generate the JKROS publication-ready figure package from frozen artifacts.

This is deliberately a CPU-only, read-only scientific analysis.  It imports the
artifact loaders from ``generate_paper_figure_bank.py`` and writes only beneath
``outputs/paper_figure_bank_pubready``.  Training, simulation, checkpoints, and
Dataset A/B are never imported or modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import generate_paper_figure_bank as frozen  # noqa: E402


OUT = ROOT / "outputs/paper_figure_bank_pubready"
OLD_BANK = ROOT / "outputs/paper_figure_bank"

DIRS = (
    "png_all",
    "pdf_all",
    "svg_all",
    "must_use",
    "strong",
    "optional",
    "single_column",
    "double_column",
    "source_data",
    "captions",
    "figure_manifest",
)

A_NAME = "Trajectory-Centric A"
B_NAME = "Interaction-Centric B"
A_SHORT = "A"
B_SHORT = "B"
A_COLOR = "#3F5F8F"       # muted blue
B_COLOR = "#A9553B"       # muted vermilion
A_LIGHT = "#DCE3ED"
B_LIGHT = "#EDDCD6"
INK = "#252525"
MID = "#777777"
GRID = "#D9D9D9"
GREEN = "#6D8C73"
AMBER = "#C3A64B"
RED = "#9B4A42"

SINGLE_MM = 88.0
DOUBLE_MM = 178.0
MM_TO_IN = 1.0 / 25.4
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_N = 20_000

PHASE_ORDER = frozen.PHASE_ORDER
PHASE_LABELS = [
    "Left approach",
    "Left grasp",
    "Left transport",
    "Handoff approach",
    "Dual-hand configuration",
    "Right owned",
    "Right transport",
    "Release",
]


@dataclass(frozen=True)
class FigureSpec:
    number: int
    stem: str
    priority: str
    recommendation: str
    question: str
    definition: str
    result: str
    sample_description: str
    caption: str
    render_required: bool = False


FIGURE_RECORDS: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def ensure_dirs() -> None:
    for name in DIRS:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.9,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.15,
            "lines.markersize": 4.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def width_inches(variant: str) -> float:
    if variant == "single":
        return SINGLE_MM * MM_TO_IN
    return DOUBLE_MM * MM_TO_IN


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=8.0)


def method_handles(policy: bool = False) -> list[Line2D]:
    return [
        Line2D([0], [0], color=A_COLOR, marker="o", linestyle="-", markerfacecolor="white", markeredgewidth=0.9, label="ACT-A" if policy else A_NAME),
        Line2D([0], [0], color=B_COLOR, marker="s", linestyle="--", markerfacecolor="white", markeredgewidth=0.9, label="ACT-B" if policy else B_NAME),
    ]


def union_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    fields = union_fields(records)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_formats(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")


def copy_formats(base: Path, directory: Path, name: str) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        shutil.copy2(base.with_suffix(suffix), directory / f"{name}{suffix}")


def source_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    result = []
    for path in dict.fromkeys(Path(p).resolve() for p in paths):
        result.append({"path": rel(path), "sha256": sha256(path)})
    return result


def save_master(
    spec: FigureSpec,
    fig: plt.Figure,
    rows: list[dict[str, Any]],
    sources: list[Path],
    variant_builder: Callable[[str], tuple[plt.Figure, list[dict[str, Any]], list[Path]]] | None = None,
) -> None:
    base = OUT / "figure_manifest" / spec.stem
    save_formats(fig, base)
    plt.close(fig)
    write_csv(OUT / "source_data" / f"{spec.stem}.csv", rows)
    (OUT / "captions" / f"{spec.stem}_caption.txt").write_text(spec.caption.strip() + "\n", encoding="utf-8")

    for suffix, collection in ((".png", "png_all"), (".pdf", "pdf_all"), (".svg", "svg_all")):
        shutil.copy2(base.with_suffix(suffix), OUT / collection / f"{spec.stem}{suffix}")
    copy_formats(base, OUT / spec.priority.lower(), spec.stem)

    variants: dict[str, dict[str, str]] = {}
    if spec.priority == "MUST_USE":
        if variant_builder is None:
            raise RuntimeError(f"MUST_USE figure lacks variant builder: {spec.stem}")
        for variant in ("single", "double"):
            vfig, _, _ = variant_builder(variant)
            vbase = OUT / f"{variant}_column" / f"{spec.stem}_{variant}"
            save_formats(vfig, vbase)
            plt.close(vfig)
            variants[variant] = {
                "png": rel(vbase.with_suffix(".png")),
                "pdf": rel(vbase.with_suffix(".pdf")),
                "svg": rel(vbase.with_suffix(".svg")),
                "width_mm": SINGLE_MM if variant == "single" else DOUBLE_MM,
            }

    metadata = {
        "schema_version": "jkros_pubready_figure_v1",
        "figure_number": spec.number,
        "filename": f"{spec.stem}.png",
        "scientific_question": spec.question,
        "underlying_artifacts": source_records(sources),
        "number_of_episodes_or_frames": spec.sample_description,
        "metric_definition": spec.definition,
        "main_numerical_result": spec.result,
        "priority": spec.priority,
        "column_recommendation": spec.recommendation,
        "new_rendering_required": spec.render_required,
        "caption_path": rel(OUT / "captions" / f"{spec.stem}_caption.txt"),
        "source_csv": rel(OUT / "source_data" / f"{spec.stem}.csv"),
        "master_files": {
            "png": rel(OUT / "png_all" / f"{spec.stem}.png"),
            "pdf": rel(OUT / "pdf_all" / f"{spec.stem}.pdf"),
            "svg": rel(OUT / "svg_all" / f"{spec.stem}.svg"),
        },
        "publication_variants": variants,
        "generation_command": "python3 tools/generate_jkros_pubready_figures.py",
        "bootstrap": {"resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED} if "bootstrap" in spec.definition.lower() or "CI" in spec.caption else None,
        "gpu_used": False,
        "scientific_results_modified": False,
        "experiment3_values_used": False if spec.number == 18 else "not_applicable",
    }
    write_json(OUT / "figure_manifest" / f"{spec.stem}_metadata.json", metadata)
    (OUT / "figure_manifest" / f"{spec.stem}_generation_command.txt").write_text(metadata["generation_command"] + "\n", encoding="utf-8")
    FIGURE_RECORDS.append(metadata)


def t1_triple(data: dict[str, Any], key: str, method: str) -> tuple[float, float, float]:
    return frozen.parse_triple(frozen.lookup(data["table1"], key, method))


def t2_value(data: dict[str, Any], key: str, method: str) -> float:
    return float(frozen.lookup(data["table2"], key, method))


def weighted_bootstrap_ci(values: np.ndarray, weights: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    n = len(values)
    # Chunking keeps peak memory small while retaining exactly 20,000 resamples.
    boot = np.empty(BOOTSTRAP_N, dtype=float)
    for start in range(0, BOOTSTRAP_N, 1000):
        stop = min(start + 1000, BOOTSTRAP_N)
        idx = rng.integers(0, n, size=(stop - start, n))
        selected_w = weights[idx]
        boot[start:stop] = np.sum(values[idx] * selected_w, axis=1) / np.sum(selected_w, axis=1)
    return tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))


def median_bootstrap_ci(values: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    n = len(values)
    boot = np.empty(BOOTSTRAP_N, dtype=float)
    for start in range(0, BOOTSTRAP_N, 1000):
        stop = min(start + 1000, BOOTSTRAP_N)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = np.median(values[idx], axis=1)
    return tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))


def jitter(n: int, width: float = 0.08) -> np.ndarray:
    # Deterministic symmetric jitter; episode order remains recoverable.
    return np.linspace(-width, width, n)


def phase_matrices(data: dict[str, Any]) -> tuple[list[int], dict[str, np.ndarray], list[dict[str, Any]]]:
    heldout = frozen.read_json(frozen.SOURCE_ARTIFACTS["heldout8"])["split_contract"]["heldout_final_dataset_indices"]
    matrices: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for method in ("a", "b"):
        behaviors = data["exp2"]["methods"][method]["selected_checkpoint_evaluation"]["phase_score"]["behaviors"]
        matrix = np.zeros((len(heldout), len(PHASE_ORDER)), dtype=int)
        for j, phase in enumerate(PHASE_ORDER):
            per_ep = {int(x["final_episode"]): int(bool(x["success"])) for x in behaviors[phase]["per_episode"]}
            for i, episode in enumerate(heldout):
                matrix[i, j] = per_ep[episode]
                rows.append(
                    {
                        "method": "ACT-A" if method == "a" else "ACT-B",
                        "episode_index": episode,
                        "phase": phase,
                        "phase_label": PHASE_LABELS[j],
                        "detected": int(matrix[i, j]),
                    }
                )
        matrices[method] = matrix
    if int(matrices["a"].sum()) != 56 or int(matrices["b"].sum()) != 52:
        raise RuntimeError("paper-core 8-phase totals changed")
    return heldout, matrices, rows


def main_context() -> dict[str, Any]:
    data = frozen.verify_and_load()
    per_episode, a_frame, b_frame = frozen.load_retargeting_arrays()
    heldout, matrices, phase_rows = phase_matrices(data)
    effects = frozen.bootstrap_effects(per_episode)
    expected = {
        "Wrist error": (74.1150187, 71.4980931, 76.6152623),
        "Whole-hand error": (-69.6779192, -73.1546151, -66.5314876),
        "Bimanual relation": (-50.8185309, -55.6821121, -46.2170664),
        "Projection magnitude": (-2.4498773, -4.5820630, -0.7517673),
    }
    for row in effects:
        actual = (row["paired_B_minus_A_mean_mm"], row["ci95_low_mm"], row["ci95_high_mm"])
        if not np.allclose(actual, expected[row["metric"]], rtol=0.0, atol=5e-7):
            raise RuntimeError(f"paired effect discrepancy for {row['metric']}: {actual}")
    episode, median = frozen.representative_episode(per_episode)
    if episode != 23:
        raise RuntimeError(f"representative episode selection changed: {episode}")
    return {
        "data": data,
        "per_episode": per_episode,
        "a_frame": a_frame,
        "b_frame": b_frame,
        "heldout": heldout,
        "phase_matrices": matrices,
        "phase_rows": phase_rows,
        "effects": effects,
        "representative_episode": episode,
        "representative_median": median,
    }


def build_pipeline(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    width = width_inches(variant)
    height = 3.95 if variant == "single" else 2.28
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    rows = [
        {"stage": "Input", "branch": "shared", "description": "ALOHA demonstrations", "sample_count": 50},
        {"stage": "Retargeting", "branch": "A", "description": "6-D wrist trajectory", "sample_count": 50},
        {"stage": "Retargeting", "branch": "B", "description": "interaction frame; whole-hand geometry; bimanual relation; ownership transition", "sample_count": 50},
        {"stage": "Dataset", "branch": "A", "description": "G1 Dataset A", "sample_count": 50},
        {"stage": "Dataset", "branch": "B", "description": "G1 Dataset B", "sample_count": 50},
        {"stage": "Policy", "branch": "A", "description": "ACT-A", "sample_count": 40},
        {"stage": "Policy", "branch": "B", "description": "ACT-B", "sample_count": 40},
        {"stage": "Evaluation", "branch": "shared", "description": "retargeting; heldout prediction; phase behavior", "sample_count": 8},
    ]

    def rect(x: float, y: float, w: float, h: float, text: str, edge: str = INK, fill: str = "white", fs: float = 6.8) -> None:
        ax.add_patch(patches.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=fill, edgecolor=edge, linewidth=0.85))
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=fs)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": MID, "mutation_scale": 7})

    if variant == "single":
        rect(0.15, 0.88, 0.70, 0.08, "ALOHA demonstrations (n = 50)", fill="#F4F4F4", fs=7.0)
        rect(0.03, 0.62, 0.44, 0.17, f"{A_NAME}\n6-D wrist trajectory", edge=A_COLOR, fill=A_LIGHT, fs=6.5)
        rect(0.53, 0.62, 0.44, 0.17, f"{B_NAME}\ninteraction frame; whole-hand\nbimanual; ownership", edge=B_COLOR, fill=B_LIGHT, fs=6.05)
        rect(0.08, 0.46, 0.34, 0.075, "G1 Dataset A", edge=A_COLOR, fill=A_LIGHT)
        rect(0.58, 0.46, 0.34, 0.075, "G1 Dataset B", edge=B_COLOR, fill=B_LIGHT)
        rect(0.12, 0.31, 0.26, 0.075, "ACT-A", edge=A_COLOR, fill=A_LIGHT)
        rect(0.62, 0.31, 0.26, 0.075, "ACT-B", edge=B_COLOR, fill=B_LIGHT)
        rect(0.10, 0.08, 0.80, 0.12, "Evaluation\nretargeting metrics · heldout prediction · phase behavior", fill="#F4F4F4", fs=6.5)
        arrow(0.5, 0.88, 0.25, 0.79); arrow(0.5, 0.88, 0.75, 0.79)
        arrow(0.25, 0.62, 0.25, 0.535); arrow(0.75, 0.62, 0.75, 0.535)
        arrow(0.25, 0.46, 0.25, 0.385); arrow(0.75, 0.46, 0.75, 0.385)
        arrow(0.25, 0.31, 0.39, 0.20); arrow(0.75, 0.31, 0.61, 0.20)
    else:
        rect(0.36, 0.84, 0.28, 0.11, "ALOHA demonstrations (n = 50)", fill="#F4F4F4", fs=7.2)
        rect(0.04, 0.54, 0.39, 0.19, f"{A_NAME}\n6-D wrist trajectory", edge=A_COLOR, fill=A_LIGHT, fs=7.0)
        rect(0.57, 0.54, 0.39, 0.19, f"{B_NAME}\ninteraction frame · whole-hand geometry\nbimanual relation · ownership transition", edge=B_COLOR, fill=B_LIGHT, fs=6.6)
        rect(0.12, 0.36, 0.23, 0.09, "G1 Dataset A", edge=A_COLOR, fill=A_LIGHT)
        rect(0.65, 0.36, 0.23, 0.09, "G1 Dataset B", edge=B_COLOR, fill=B_LIGHT)
        rect(0.15, 0.20, 0.17, 0.085, "ACT-A", edge=A_COLOR, fill=A_LIGHT)
        rect(0.68, 0.20, 0.17, 0.085, "ACT-B", edge=B_COLOR, fill=B_LIGHT)
        rect(0.27, 0.02, 0.46, 0.11, "Evaluation: retargeting · heldout prediction · phase behavior", fill="#F4F4F4", fs=6.3)
        arrow(0.5, 0.84, 0.235, 0.73); arrow(0.5, 0.84, 0.765, 0.73)
        arrow(0.235, 0.54, 0.235, 0.45); arrow(0.765, 0.54, 0.765, 0.45)
        arrow(0.235, 0.36, 0.235, 0.285); arrow(0.765, 0.36, 0.765, 0.285)
        arrow(0.235, 0.20, 0.405, 0.13); arrow(0.765, 0.20, 0.595, 0.13)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["retargeting_table"], frozen.SOURCE_ARTIFACTS["experiment2"]]


def build_core_tradeoff(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    per = ctx["per_episode"]
    af, bf = ctx["a_frame"], ctx["b_frame"]
    defs = [("wrist", "Wrist error"), ("whole_hand", "Whole-hand interaction error"), ("bimanual", "Bimanual relation error")]
    if variant == "single":
        fig, axes = plt.subplots(3, 1, figsize=(width_inches(variant), 6.0))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(width_inches(variant), 2.45))
    rows: list[dict[str, Any]] = []
    weights = np.array([r["frame_count"] for r in per], dtype=float)
    for j, (ax, (metric, label)) in enumerate(zip(np.ravel(axes), defs)):
        vals = {}
        means = {}
        cis = {}
        for method, frame, color, marker in (("A", af, A_COLOR, "o"), ("B", bf, B_COLOR, "s")):
            values = np.array([r[f"{method.lower()}_{metric}_mean_mm"] for r in per], dtype=float)
            mean = float(np.asarray(frame[metric]).mean() * 1000.0)
            ci = weighted_bootstrap_ci(values, weights, seed_offset=20 * j + (0 if method == "A" else 1))
            x = 0 if method == "A" else 1
            ax.scatter(x + jitter(len(values)), values, s=9, facecolor="white", edgecolor=color, linewidth=0.45, alpha=0.58, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.45, markersize=5.6, capsize=2.6, elinewidth=1.25, zorder=4)
            vals[method], means[method], cis[method] = values, mean, ci
            rows.extend(
                {"record_type": "episode", "metric": label, "episode_index": r["episode_index"], "stable_episode_id": r["stable_episode_id"], "method": method, "episode_mean_mm": float(v)}
                for r, v in zip(per, values)
            )
            rows.append({"record_type": "aggregate", "metric": label, "method": method, "frame_weighted_mean_mm": mean, "cluster_bootstrap_ci95_low_mm": ci[0], "cluster_bootstrap_ci95_high_mm": ci[1], "episodes": 50, "resamples": BOOTSTRAP_N})
        ax.set_xticks([0, 1], ["A", "B"])
        ax.set_xlim(-0.32, 1.32)
        ax.set_ylabel("Error [mm]")
        ax.set_title(label, pad=4)
        ax.set_ylim(bottom=0)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + j), x=-0.17 if variant == "double" else -0.12)
        ax.text(0.5, 0.97, f"{means['A']:.1f}  vs  {means['B']:.1f} mm", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
    fig.legend(handles=method_handles(), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.12 if variant == "single" else 0.075, right=0.985, bottom=0.07 if variant == "single" else 0.18, top=0.91 if variant == "single" else 0.80, hspace=0.62 if variant == "single" else 0.0, wspace=0.36 if variant == "double" else 0.0)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["a_manifest"], frozen.SOURCE_ARTIFACTS["b_manifest"], frozen.SOURCE_ARTIFACTS["retargeting_table"]]


def build_paired_effects(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    per = ctx["per_episode"]
    effects = {r["metric"]: r for r in ctx["effects"]}
    defs = [("wrist", "Wrist error", "Wrist error"), ("whole_hand", "Whole-hand error", "Whole-hand error"), ("bimanual", "Bimanual relation", "Bimanual relation")]
    if variant == "single":
        fig, axes = plt.subplots(3, 1, figsize=(width_inches(variant), 6.1))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(width_inches(variant), 2.55))
    rows: list[dict[str, Any]] = []
    for j, (ax, (metric, title, effect_key)) in enumerate(zip(np.ravel(axes), defs)):
        av = np.array([r[f"a_{metric}_mean_mm"] for r in per])
        bv = np.array([r[f"b_{metric}_mean_mm"] for r in per])
        for a, b in zip(av, bv):
            ax.plot([0, 1], [a, b], color="#B9B9B9", lw=0.45, alpha=0.65, zorder=1)
        ax.scatter(np.zeros(len(av)), av, s=9, facecolor="white", edgecolor=A_COLOR, linewidth=0.5, alpha=0.75, zorder=2)
        ax.scatter(np.ones(len(bv)), bv, s=9, marker="s", facecolor="white", edgecolor=B_COLOR, linewidth=0.5, alpha=0.75, zorder=2)
        for x, values, color, marker, offset in ((0, av, A_COLOR, "o", 100 + j * 2), (1, bv, B_COLOR, "s", 101 + j * 2)):
            median = float(np.median(values)); lo, hi = median_bootstrap_ci(values, offset)
            ax.errorbar(x, median, yerr=[[median - lo], [hi - median]], fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.45, markersize=5.5, capsize=2.5, elinewidth=1.2, zorder=4)
            rows.append({"record_type": "median", "metric": title, "method": "A" if x == 0 else "B", "median_mm": median, "bootstrap_ci95_low_mm": lo, "bootstrap_ci95_high_mm": hi, "episodes": 50})
        effect = effects[effect_key]
        ax.text(0.5, 0.98, f"B − A = {effect['paired_B_minus_A_mean_mm']:+.1f} mm", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
        ax.set_xticks([0, 1], ["A", "B"]); ax.set_xlim(-0.30, 1.30); ax.set_ylim(bottom=0)
        ax.set_title(title, pad=4); ax.set_ylabel("Episode mean [mm]"); clean_axis(ax)
        panel_label(ax, chr(ord("a") + j), x=-0.17 if variant == "double" else -0.12)
        rows.extend({"record_type": "paired_episode", "metric": title, "episode_index": r["episode_index"], "stable_episode_id": r["stable_episode_id"], "a_mean_mm": float(a), "b_mean_mm": float(b), "b_minus_a_mm": float(b - a)} for r, a, b in zip(per, av, bv))
    fig.subplots_adjust(left=0.13 if variant == "single" else 0.075, right=0.985, bottom=0.07 if variant == "single" else 0.16, top=0.94 if variant == "single" else 0.88, hspace=0.60 if variant == "single" else 0.0, wspace=0.37 if variant == "double" else 0.0)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["a_manifest"], frozen.SOURCE_ARTIFACTS["b_manifest"]]


def build_effect_forest(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    effects = ctx["effects"]
    fig, ax = plt.subplots(figsize=(width_inches(variant), 2.48 if variant == "single" else 2.35))
    y = np.arange(len(effects))[::-1]
    for yy, row in zip(y, effects):
        mean = row["paired_B_minus_A_mean_mm"]; lo = row["ci95_low_mm"]; hi = row["ci95_high_mm"]
        color = A_COLOR if mean > 0 else B_COLOR
        marker = "o" if mean > 0 else "s"
        ax.errorbar(mean, yy, xerr=[[mean - lo], [hi - mean]], fmt=marker, color=color, markerfacecolor="white", markeredgewidth=0.9, markersize=5.2, elinewidth=1.35, capsize=2.6, zorder=3)
        ax.annotate(f"{mean:+.1f}", xy=(mean, yy), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=5.8, color=color)
    ax.axvline(0, color=INK, lw=0.75)
    ax.set_xlim(-82, 84)
    ax.set_ylim(-0.65, len(effects) - 0.35)
    ax.set_yticks(y, ["Wrist", "Whole-hand", "Bimanual\nrelation", "Projection\nmagnitude"])
    ax.set_xlabel("Paired difference, B − A [mm]\n← lower error for B     lower error for A →")
    clean_axis(ax, "x")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=A_COLOR, markerfacecolor="white", linestyle="none", label="A lower"), Line2D([0], [0], marker="s", color=B_COLOR, markerfacecolor="white", linestyle="none", label="B lower")], frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2)
    fig.subplots_adjust(left=0.30 if variant == "single" else 0.18, right=0.97, bottom=0.27, top=0.82)
    return fig, effects, [frozen.SOURCE_ARTIFACTS["a_manifest"], frozen.SOURCE_ARTIFACTS["b_manifest"]]


def build_tradeoff_scatter(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    per = ctx["per_episode"]
    if variant == "single":
        fig, axes = plt.subplots(2, 1, figsize=(width_inches(variant), 5.2))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(width_inches(variant), 2.8))
    rows: list[dict[str, Any]] = []
    for j, (ax, metric, ylabel) in enumerate(zip(np.ravel(axes), ("whole_hand", "bimanual"), ("Whole-hand interaction error [mm]", "Bimanual relation error [mm]"))):
        for method, color, marker, mname in (("a", A_COLOR, "o", A_NAME), ("b", B_COLOR, "s", B_NAME)):
            x = np.array([r[f"{method}_wrist_mean_mm"] for r in per])
            y = np.array([r[f"{method}_{metric}_mean_mm"] for r in per])
            ax.scatter(x, y, s=12, marker=marker, facecolor="white", edgecolor=color, linewidth=0.6, alpha=0.78, label=mname)
            xm, ym = float(np.median(x)), float(np.median(y))
            xq = np.percentile(x, [25, 75]); yq = np.percentile(y, [25, 75])
            ax.errorbar(xm, ym, xerr=[[xm - xq[0]], [xq[1] - xm]], yerr=[[ym - yq[0]], [yq[1] - ym]], fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.5, markersize=6.0, elinewidth=1.35, capsize=0, zorder=4)
            rows.extend({"episode_index": r["episode_index"], "stable_episode_id": r["stable_episode_id"], "method": mname, "wrist_mean_mm": float(xv), f"{metric}_mean_mm": float(yv)} for r, xv, yv in zip(per, x, y))
            rows.append({"episode_index": "SUMMARY", "method": mname, "wrist_median_mm": xm, f"{metric}_median_mm": ym, "wrist_iqr_low_mm": xq[0], "wrist_iqr_high_mm": xq[1], f"{metric}_iqr_low_mm": yq[0], f"{metric}_iqr_high_mm": yq[1]})
        ax.set_xlabel("Wrist error [mm]"); ax.set_ylabel(ylabel); clean_axis(ax, "both")
        panel_label(ax, chr(ord("a") + j), x=-0.14)
    axes_flat = np.ravel(axes)
    fig.legend(handles=method_handles(), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2)
    fig.subplots_adjust(left=0.15 if variant == "single" else 0.085, right=0.985, bottom=0.10 if variant == "single" else 0.17, top=0.90 if variant == "single" else 0.77, hspace=0.48 if variant == "single" else 0.0, wspace=0.30 if variant == "double" else 0.0)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["a_manifest"], frozen.SOURCE_ARTIFACTS["b_manifest"]]


def build_feasibility(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    per = ctx["per_episode"]
    normalize = {"CLEAN": "CLEAN", "CLEAN_PASS": "CLEAN", "USABLE_WITH_WARNING": "WARNING", "WARNING": "WARNING", "HARD_FAIL": "HARD", "HARD": "HARD"}
    counts: dict[str, dict[str, int]] = {"A": {k: 0 for k in ("CLEAN", "WARNING", "HARD")}, "B": {k: 0 for k in ("CLEAN", "WARNING", "HARD")}}
    for r in per:
        counts["A"][normalize[r["a_status"]]] += 1
        counts["B"][normalize[r["b_status"]]] += 1
    if counts != {"A": {"CLEAN": 27, "WARNING": 21, "HARD": 2}, "B": {"CLEAN": 13, "WARNING": 37, "HARD": 0}}:
        raise RuntimeError(f"feasibility counts changed: {counts}")
    if variant == "single":
        fig, axes = plt.subplots(2, 1, figsize=(width_inches(variant), 3.8), gridspec_kw={"height_ratios": [1.0, 1.25]})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(width_inches(variant), 2.35), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = np.ravel(axes)[0]
    left = np.zeros(2)
    for status, color, hatch in (("CLEAN", GREEN, "//"), ("WARNING", AMBER, ".."), ("HARD", RED, "xx")):
        values = [counts["A"][status], counts["B"][status]]
        bars = ax.barh([0, 1], values, left=left, color=color, edgecolor=INK, linewidth=0.55, hatch=hatch, height=0.55, label=status)
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", fontsize=6.8, color="white" if status == "HARD" else INK)
        left += np.array(values)
    ax.set_yticks([0, 1], [A_NAME, B_NAME]); ax.invert_yaxis(); ax.set_xlim(0, 50); ax.set_xlabel("Episodes")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), handlelength=1.6)
    clean_axis(ax, "x"); panel_label(ax, "a", x=-0.15)

    ax = np.ravel(axes)[1]
    modes = ["Hard collision", "Hard IK", "Joint limit", "Branch"]
    av = np.array([2, 0, 0, 0]); bv = np.array([0, 0, 0, 0])
    y = np.arange(4)[::-1]
    for yy, a, b in zip(y, av, bv):
        ax.plot([a, b], [yy + 0.08, yy - 0.08], color="#BBBBBB", lw=0.7)
        ax.scatter(a, yy + 0.08, color=A_COLOR, marker="o", facecolor="white", zorder=3)
        ax.scatter(b, yy - 0.08, color=B_COLOR, marker="s", facecolor="white", zorder=3)
    ax.set_yticks(y, modes); ax.set_xlim(-0.15, 2.35); ax.set_xticks([0, 1, 2]); ax.set_xlabel("Hard-failure count")
    clean_axis(ax, "x"); panel_label(ax, "b", x=-0.12)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=A_COLOR, markerfacecolor="white", linestyle="none", label="A"), Line2D([0], [0], marker="s", color=B_COLOR, markerfacecolor="white", linestyle="none", label="B")], frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.text(0.5, 0.012, "WARNING ≠ HARD_FAIL; B has 0/50 hard episodes.", ha="center", va="bottom", fontsize=6.4)
    fig.subplots_adjust(left=0.32 if variant == "single" else 0.20, right=0.98, bottom=0.12 if variant == "single" else 0.20, top=0.88 if variant == "single" else 0.81, hspace=0.70 if variant == "single" else 0.0, wspace=0.48 if variant == "double" else 0.0)
    rows = [{"method": method, **values, "episodes": 50} for method, values in counts.items()]
    rows.extend({"method": "A", "failure_mode": mode, "count": int(a)} for mode, a in zip(modes, av))
    rows.extend({"method": "B", "failure_mode": mode, "count": int(b)} for mode, b in zip(modes, bv))
    return fig, rows, [frozen.SOURCE_ARTIFACTS["a_per_episode"], frozen.SOURCE_ARTIFACTS["b_manifest"], frozen.SOURCE_ARTIFACTS["retargeting_table"]]


def prediction_probe_values(ctx: dict[str, Any], method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    probes = ctx["data"]["exp2"]["methods"][method]["common_source_geometry"]["per_probe"]
    return (
        np.array([float(p[metric]["mean"]) for p in probes], dtype=float),
        np.array([int(p["valid_frames"]) for p in probes], dtype=float),
    )


def build_supervision_policy(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    data = ctx["data"]; per = ctx["per_episode"]; af = ctx["a_frame"]; bf = ctx["b_frame"]
    defs = [
        ("wrist", "wrist_error_mm", "Wrist", "predicted source wrist error mean"),
        ("whole_hand", "whole_hand_error_mm", "Whole-hand", "predicted whole-hand error mean"),
        ("bimanual", "bimanual_relation_error_mm", "Bimanual", "predicted bimanual relation error mean"),
    ]
    if variant == "single":
        fig, axes = plt.subplots(3, 2, figsize=(width_inches(variant), 6.0), sharey="row")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(width_inches(variant), 3.10), sharey="col")
    rows: list[dict[str, Any]] = []
    weights = np.array([r["frame_count"] for r in per], dtype=float)
    for col, (metric, probe_key, title, policy_key) in enumerate(defs):
        ax = axes[col, 0] if variant == "single" else axes[0, col]
        for method, frame, color, marker in (("A", af, A_COLOR, "o"), ("B", bf, B_COLOR, "s")):
            values = np.array([r[f"{method.lower()}_{metric}_mean_mm"] for r in per])
            mean = float(frame[metric].mean() * 1000.0)
            lo, hi = weighted_bootstrap_ci(values, weights, 300 + col * 10 + (method == "B"))
            x = 0 if method == "A" else 1
            ax.scatter(x + jitter(len(values), 0.07), values, s=6.5, facecolor="white", edgecolor=color, linewidth=0.35, alpha=0.48)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.4, markersize=5.0, capsize=2.0, elinewidth=1.1, zorder=4)
            rows.extend({"level": "Retargeted supervision", "metric": title, "method": method, "episode_index": r["episode_index"], "sample_mean_mm": float(v)} for r, v in zip(per, values))
            rows.append({"level": "Retargeted supervision", "metric": title, "method": method, "record_type": "aggregate", "mean_mm": mean, "cluster_bootstrap_ci95_low_mm": lo, "cluster_bootstrap_ci95_high_mm": hi, "samples": 50})
        if variant == "double":
            ax.set_title(title, pad=3.5)
        ax.set_xticks([0, 1], ["A", "B"]); ax.set_xlim(-0.30, 1.30); ax.set_ylim(bottom=0)
        ax.set_ylabel((f"{title}\nmean error [mm]" if variant == "single" else ("Mean error [mm]" if col == 0 else "")))
        clean_axis(ax)
        if col == 0 and variant == "double":
            ax.text(0.03, 0.96, "Retargeted supervision", transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color=MID)

        ax = axes[col, 1] if variant == "single" else axes[1, col]
        for method, color, marker, policy_name, table_method in (("a", A_COLOR, "o", "ACT-A", "ACT-A40"), ("b", B_COLOR, "s", "ACT-B", "ACT-B40")):
            values, probe_weights = prediction_probe_values(ctx, method, probe_key)
            mean = t2_value(data, policy_key, table_method)
            lo, hi = weighted_bootstrap_ci(values, probe_weights, 400 + col * 10 + (method == "b"))
            x = 0 if method == "a" else 1
            ax.scatter(x + jitter(len(values), 0.07), values, s=6.0, facecolor="white", edgecolor=color, linewidth=0.35, alpha=0.45)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.4, markersize=5.0, capsize=2.0, elinewidth=1.1, zorder=4)
            probes = data["exp2"]["methods"][method]["common_source_geometry"]["per_probe"]
            rows.extend({"level": "Learned ACT prediction", "metric": title, "method": policy_name, "episode_index": p["final_episode"], "phase": p["phase"], "probe_frame": p["frame"], "valid_frames": p["valid_frames"], "sample_mean_mm": float(v)} for p, v in zip(probes, values))
            rows.append({"level": "Learned ACT prediction", "metric": title, "method": policy_name, "record_type": "aggregate", "mean_mm": mean, "cluster_bootstrap_ci95_low_mm": lo, "cluster_bootstrap_ci95_high_mm": hi, "samples": 72})
        ax.set_xticks([0, 1], ["ACT-A", "ACT-B"]); ax.set_xlim(-0.30, 1.30); ax.set_ylim(bottom=0)
        ax.set_ylabel("Mean error [mm]" if col == 0 and variant == "double" else "")
        clean_axis(ax)
        if col == 0 and variant == "double":
            ax.text(0.03, 0.96, "Learned ACT prediction", transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color=MID)
        panel_label(axes[col, 0] if variant == "single" else axes[0, col], chr(ord("a") + col), x=-0.24 if variant == "single" else -0.18)
    if variant == "single":
        axes[0, 0].set_title("Retargeted", pad=4)
        axes[0, 1].set_title("ACT prediction", pad=4)
    fig.legend(handles=method_handles(), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.27 if variant == "single" else 0.095, right=0.985, bottom=0.06 if variant == "single" else 0.13, top=0.92 if variant == "single" else 0.84, hspace=0.48 if variant == "single" else 0.42, wspace=0.30 if variant == "single" else 0.32)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["retargeting_table"], frozen.SOURCE_ARTIFACTS["experiment2"], frozen.SOURCE_ARTIFACTS["a_manifest"], frozen.SOURCE_ARTIFACTS["b_manifest"]]


def build_action_rmse(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    data = ctx["data"]
    metrics = [("First action", "first-action RMSE"), ("Full valid chunk", "full valid chunk RMSE")]
    av = [t2_value(data, key, "ACT-A40") for _, key in metrics]
    bv = [t2_value(data, key, "ACT-B40") for _, key in metrics]
    fig, ax = plt.subplots(figsize=(width_inches(variant), 2.15 if variant == "single" else 2.05))
    y = np.array([1, 0])
    for yy, a, b in zip(y, av, bv):
        ax.plot([a, b], [yy, yy], color="#AAAAAA", lw=0.85, zorder=1)
        ax.scatter(a, yy, marker="o", facecolor="white", edgecolor=A_COLOR, linewidth=1.0, zorder=3)
        ax.scatter(b, yy, marker="s", facecolor="white", edgecolor=B_COLOR, linewidth=1.0, zorder=3)
        ax.text(a, yy + 0.16, f"{a:.4f}", color=A_COLOR, ha="center", va="bottom", fontsize=6.2)
        ax.text(b, yy - 0.16, f"{b:.4f}", color=B_COLOR, ha="center", va="top", fontsize=6.2)
    ax.set_yticks(y, [m[0] for m in metrics]); ax.set_xlim(0.025, 0.17); ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Action RMSE [rad]"); clean_axis(ax, "x")
    ax.legend(handles=method_handles(policy=True), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.subplots_adjust(left=0.31 if variant == "single" else 0.18, right=0.97, bottom=0.24, top=0.80)
    rows = [{"metric": label, "ACT-A_RMSE_rad": a, "ACT-B_RMSE_rad": b, "uncertainty": "NA (aggregate artifact only)", "scored_joint_frames_per_method": 100800} for (label, _), a, b in zip(metrics, av, bv)]
    return fig, rows, [frozen.SOURCE_ARTIFACTS["experiment2"], frozen.SOURCE_ARTIFACTS["policy_table"]]


def build_phase_success(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    mats = ctx["phase_matrices"]
    av = mats["a"].mean(axis=0) * 100.0; bv = mats["b"].mean(axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(width_inches(variant), 3.80 if variant == "single" else 2.85))
    y = np.arange(8)[::-1]
    for yy, a, b in zip(y, av, bv):
        ax.plot([a, b], [yy, yy], color="#B5B5B5", lw=0.85, zorder=1)
        ax.scatter(a, yy, marker="o", facecolor="white", edgecolor=A_COLOR, linewidth=0.9, zorder=3)
        ax.scatter(b, yy, marker="s", facecolor="white", edgecolor=B_COLOR, linewidth=0.9, zorder=3)
    ax.set_yticks(y, PHASE_LABELS); ax.set_xlim(-3, 104); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Behavior detected [% of 8 heldout episodes]")
    clean_axis(ax, "x")
    fig.legend(handles=method_handles(policy=True), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    fig.text(0.5, 0.855, "Overall: ACT-A 56/64; ACT-B 52/64", ha="center", va="center", fontsize=6.4)
    ax.set_xlabel("Phase detection [%]")
    fig.subplots_adjust(left=0.43 if variant == "single" else 0.27, right=0.96, bottom=0.16, top=0.79)
    rows = []
    for j, (phase, label) in enumerate(zip(PHASE_ORDER, PHASE_LABELS)):
        rows.append({"phase": phase, "phase_label": label, "ACT-A_success_count": int(mats["a"][:, j].sum()), "ACT-B_success_count": int(mats["b"][:, j].sum()), "heldout_episodes": 8, "ACT-A_percent": float(av[j]), "ACT-B_percent": float(bv[j])})
    return fig, rows, [frozen.SOURCE_ARTIFACTS["experiment2"], frozen.SOURCE_ARTIFACTS["heldout8"]]


def ecdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def build_error_ecdf(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    defs = [("wrist", "Wrist error"), ("whole_hand", "Whole-hand error"), ("bimanual", "Bimanual relation error")]
    if variant == "single":
        fig, axes = plt.subplots(3, 1, figsize=(width_inches(variant), 5.9))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(width_inches(variant), 2.35))
    rows: list[dict[str, Any]] = []
    for j, (ax, (metric, title)) in enumerate(zip(np.ravel(axes), defs)):
        for method, frame, color, ls, marker in (("A", ctx["a_frame"], A_COLOR, "-", "o"), ("B", ctx["b_frame"], B_COLOR, "--", "s")):
            values = np.asarray(frame[metric]) * 1000.0
            x, y = ecdf_xy(values)
            ax.step(x, y, where="post", color=color, ls=ls)
            median = float(np.median(values)); p95 = float(np.percentile(values, 95))
            ax.scatter([median, p95], [0.5, 0.95], marker=marker, facecolor="white", edgecolor=color, linewidth=0.75, s=15, zorder=3)
            rows.extend({"metric": title, "method": method, "rank": i + 1, "sample_count": len(x), "error_mm": float(v), "ecdf": float(c)} for i, (v, c) in enumerate(zip(x, y)))
            rows.append({"metric": title, "method": method, "record_type": "summary", "median_mm": median, "p95_mm": p95, "max_mm": float(values.max()), "mean_mm": float(values.mean())})
        ax.set_title(title, pad=3); ax.set_xlabel("Error [mm]"); ax.set_ylabel("Empirical CDF" if j == 0 or variant == "single" else "")
        ax.set_ylim(0, 1.01); ax.set_xlim(left=0); clean_axis(ax, "both"); panel_label(ax, chr(ord("a") + j), x=-0.17)
    fig.legend(handles=method_handles(), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    fig.subplots_adjust(left=0.14 if variant == "single" else 0.075, right=0.985, bottom=0.08 if variant == "single" else 0.19, top=0.91 if variant == "single" else 0.76, hspace=0.58 if variant == "single" else 0.0, wspace=0.35 if variant == "double" else 0.0)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["a_manifest"], frozen.SOURCE_ARTIFACTS["b_manifest"]]


def build_projection_distribution(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    if variant == "single":
        fig, axes = plt.subplots(2, 1, figsize=(width_inches(variant), 4.15))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(width_inches(variant), 2.35))
    rows: list[dict[str, Any]] = []
    for method, frame, color, ls, marker in (("A", ctx["a_frame"], A_COLOR, "-", "o"), ("B", ctx["b_frame"], B_COLOR, "--", "s")):
        values = np.asarray(frame["projection"]) * 1000.0
        x, y = ecdf_xy(values)
        axes_flat = np.ravel(axes)
        axes_flat[0].step(x, y, where="post", color=color, ls=ls)
        survival = (len(x) - np.arange(len(x))) / len(x)
        axes_flat[1].step(x, survival, where="post", color=color, ls=ls)
        stats = {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}
        axes_flat[0].scatter([stats["median"], stats["p95"]], [0.5, 0.95], marker=marker, facecolor="white", edgecolor=color, linewidth=0.8, s=15, zorder=3)
        axes_flat[1].scatter([stats["max"]], [1.0 / len(values)], marker=marker, facecolor="white", edgecolor=color, linewidth=0.8, s=15, zorder=3)
        rows.extend({"method": method, "rank": i + 1, "sample_count": len(x), "projection_mm": float(v), "ecdf": float(c), "tail_probability": float(s)} for i, (v, c, s) in enumerate(zip(x, y, survival)))
        rows.append({"method": method, "record_type": "summary", **{f"{k}_mm": v for k, v in stats.items()}})
    axes_flat = np.ravel(axes)
    axes_flat[0].set_xlabel("Projection magnitude [mm]"); axes_flat[0].set_ylabel("Empirical CDF"); axes_flat[0].set_xlim(0, 100); axes_flat[0].set_ylim(0, 1.01); clean_axis(axes_flat[0], "both"); panel_label(axes_flat[0], "a")
    axes_flat[1].set_xlabel("Projection magnitude [mm]"); axes_flat[1].set_ylabel("Tail probability, P(X ≥ x)"); axes_flat[1].set_xlim(0, 100); axes_flat[1].set_yscale("log"); axes_flat[1].set_ylim(5e-6, 1.2); clean_axis(axes_flat[1], "both"); panel_label(axes_flat[1], "b")
    fig.legend(handles=method_handles(), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    fig.subplots_adjust(left=0.15 if variant == "single" else 0.085, right=0.98, bottom=0.10 if variant == "single" else 0.19, top=0.91 if variant == "single" else 0.76, hspace=0.56 if variant == "single" else 0.0, wspace=0.32 if variant == "double" else 0.0)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["a_manifest"], frozen.SOURCE_ARTIFACTS["b_manifest"], frozen.SOURCE_ARTIFACTS["retargeting_table"]]


def representative_assets(ctx: dict[str, Any]) -> tuple[Path, Path, dict[str, int]]:
    episode = ctx["representative_episode"]
    a_row = frozen.read_json(frozen.SOURCE_ARTIFACTS["a_manifest"])["trajectories"][episode]
    b_row = frozen.read_json(frozen.SOURCE_ARTIFACTS["b_manifest"])["episodes"][episode]
    report = frozen.read_json(ROOT / f"outputs/policy_b_g1visual/dataset_render_full/episode_reports/episode_{episode:06d}.json")
    return Path(a_row["trajectory_path"]), Path(b_row["retargeted_trajectory_path"]), report["snapshot_frames"]


def build_representative_trajectory(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    episode = ctx["representative_episode"]
    a_path, b_path, snapshots = representative_assets(ctx)
    if variant == "single":
        fig, axes = plt.subplots(3, 1, figsize=(width_inches(variant), 7.25))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(width_inches(variant), 2.85))
    rows: list[dict[str, Any]] = []
    events = [("Left grasp", snapshots["LEFT_OWNED"]), ("Handoff", snapshots["DUAL_CONTACT"]), ("Right owned", snapshots["RIGHT_OWNED"]), ("Release", snapshots["release"])]
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        n = len(a["timestamp"]); t = np.arange(n) / 30.0
        for panel, kind in enumerate(("wrist", "whole")):
            ax = np.ravel(axes)[panel]
            for side in ("left", "right"):
                if kind == "wrist":
                    ref = np.asarray(a[f"source_{side}_realization_frame_position_world"]) * 1000.0
                    aa = np.asarray(a[f"achieved_{side}_wrist_position_world"]) * 1000.0
                    bb = np.asarray(b[f"achieved_{side}_wrist_position_world"]) * 1000.0
                else:
                    ref = np.asarray(a[f"target_{side}_interaction_frame_position_world"]) * 1000.0
                    aa = np.asarray(a[f"achieved_{side}_physical_grasp_frame_position_world"]) * 1000.0
                    bb = np.asarray(b[f"achieved_{side}_physical_grasp_frame_position_world"]) * 1000.0
                for method, values, color, ls in (("Source reference", ref, INK, ":"), ("A", aa, A_COLOR, "-"), ("B", bb, B_COLOR, "--")):
                    ax.plot(values[:, 0], values[:, 1], color=color, ls=ls, lw=1.05 if side == "left" else 0.8, alpha=1.0 if side == "left" else 0.72)
                    rows.extend({"episode_index": episode, "panel": kind, "side": side, "method": method, "frame": i, "time_s": i / 30.0, "world_x_mm": float(v[0]), "world_y_mm": float(v[1])} for i, v in enumerate(values))
                ax.text(ref[-1, 0], ref[-1, 1], " L" if side == "left" else " R", fontsize=6.2, color=INK)
            ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("World x [mm]"); ax.set_ylabel("World y [mm]")
            ax.set_title("Wrist paths (top view)" if kind == "wrist" else "Whole-hand paths (top view)", pad=3)
            clean_axis(ax, "both"); panel_label(ax, chr(ord("a") + panel), x=-0.14, y=1.07)

        ax = np.ravel(axes)[2]
        target_l = np.asarray(a["target_left_interaction_frame_position_world"])
        target_r = np.asarray(a["target_right_interaction_frame_position_world"])
        a_l = np.asarray(a["achieved_left_physical_grasp_frame_position_world"])
        a_r = np.asarray(a["achieved_right_physical_grasp_frame_position_world"])
        b_l = np.asarray(b["achieved_left_physical_grasp_frame_position_world"])
        b_r = np.asarray(b["achieved_right_physical_grasp_frame_position_world"])
        curves = {
            "Source reference": np.linalg.norm(target_r - target_l, axis=1) * 1000.0,
            "A": np.linalg.norm(a_r - a_l, axis=1) * 1000.0,
            "B": np.linalg.norm(b_r - b_l, axis=1) * 1000.0,
        }
        for method, color, ls in (("Source reference", INK, ":"), ("A", A_COLOR, "-"), ("B", B_COLOR, "--")):
            ax.plot(t, curves[method], color=color, ls=ls, label=method)
            rows.extend({"episode_index": episode, "panel": "relative_displacement", "method": method, "frame": i, "time_s": float(tt), "relative_displacement_mm": float(v)} for i, (tt, v) in enumerate(zip(t, curves[method])))
        for k, (label, frame) in enumerate(events):
            if frame < n:
                ax.axvline(frame / 30.0, color="#A6A6A6", lw=0.55, ls=(0, (2, 2)))
                ax.text(frame / 30.0, 0.98, str(k + 1), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.8, color=MID, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
                rows.append({"episode_index": episode, "panel": "semantic_event", "semantic_event": label, "frame": frame, "time_s": frame / 30.0})
        ax.text(0.02, 0.02, "1 grasp · 2 handoff · 3 right-owned · 4 release", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.0, color=MID)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Right–left displacement [mm]"); ax.set_title("Bimanual relative displacement", pad=3); clean_axis(ax); panel_label(ax, "c", x=-0.14, y=1.07)
        if variant == "single":
            handles = [Line2D([0], [0], color=INK, ls=":", label="Source"), Line2D([0], [0], color=A_COLOR, ls="-", label="A"), Line2D([0], [0], color=B_COLOR, ls="--", label="B")]
        else:
            handles = [Line2D([0], [0], color=INK, ls=":", label="Source reference"), Line2D([0], [0], color=A_COLOR, ls="-", label=A_NAME), Line2D([0], [0], color=B_COLOR, ls="--", label=B_NAME)]
        fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.16 if variant == "single" else 0.075, right=0.985, bottom=0.07 if variant == "single" else 0.17, top=0.93 if variant == "single" else 0.81, hspace=0.50 if variant == "single" else 0.0, wspace=0.35 if variant == "double" else 0.0)
    return fig, rows, [a_path, b_path, frozen.SOURCE_ARTIFACTS["common48"]]


def failure_data(ctx: dict[str, Any]) -> dict[str, Any]:
    collision = frozen.read_json(frozen.SOURCE_ARTIFACTS["collision_audit"])["episodes"]
    candidates = [r for r in collision if int(r["episode_index"]) in (35, 46)]
    selected = max(candidates, key=lambda r: (float(r["after"]["maximum_penetration_m"]), -int(r["episode_index"])))
    episode = int(selected["episode_index"])
    if episode != 46:
        raise RuntimeError(f"frozen collision selection changed: {episode}")
    a_path = Path(frozen.read_json(frozen.SOURCE_ARTIFACTS["a_manifest"])["trajectories"][episode]["trajectory_path"])
    b_entry = frozen.read_json(frozen.SOURCE_ARTIFACTS["b_manifest"])["episodes"][episode]
    b_path = Path(b_entry["retargeted_trajectory_path"])
    records = [r for r in frozen.read_csv(frozen.SOURCE_ARTIFACTS["collision_records"]) if int(r["episode_index"]) == episode]
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        n = len(a["timestamp"]); t = np.arange(n) / 30.0
        penetration = np.zeros(n); hard = np.zeros(n, dtype=int); pair = np.full(n, "NONE", dtype=object)
        for row in records:
            frame = int(row["frame"]); value = float(row["penetration_m"])
            if value >= penetration[frame]:
                penetration[frame] = value; pair[frame] = row["body_pair"]
            hard[frame] = max(hard[frame], int(row["shared_severity_hard_frame"].lower() == "true"))
        a_projection = np.asarray(a["feasibility_projection_translation_m"]).reshape(n, -1).max(axis=1) * 1000.0
        b_projection = np.asarray(b["feasibility_projection_translation_m"]).reshape(n, -1).max(axis=1) * 1000.0
    peak_frame = int(np.argmax(penetration))
    report_path = ROOT / f"outputs/policy_b_g1visual/dataset_render_full/episode_reports/episode_{episode:06d}.json"
    snapshots = frozen.read_json(report_path)["snapshot_frames"]
    nearest_key, nearest_frame = min(snapshots.items(), key=lambda item: (abs(int(item[1]) - peak_frame), item[0]))
    image_path = ROOT / f"outputs/policy_b_g1visual/dataset_render_full/snapshots/episode_{episode:06d}/{nearest_key}_cam_high.png"
    return {
        "episode": episode,
        "a_path": a_path,
        "b_path": b_path,
        "t": t,
        "penetration": penetration * 1000.0,
        "hard": hard,
        "pair": pair,
        "a_projection": a_projection,
        "b_projection": b_projection,
        "peak_frame": peak_frame,
        "nearest_key": nearest_key,
        "nearest_frame": int(nearest_frame),
        "image_path": image_path,
        "report_path": report_path,
        "b_status": b_entry["classification"],
        "selected_record": selected,
    }


def build_failure_case(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    d = failure_data(ctx)
    if variant == "single":
        fig = plt.figure(figsize=(width_inches(variant), 7.05))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.4, 0.85])
        axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    else:
        fig = plt.figure(figsize=(width_inches(variant), 4.10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 0.90], height_ratios=[1, 1])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    t = d["t"]
    axes[0].plot(t, d["a_projection"], color=A_COLOR, label="A")
    axes[0].plot(t, d["b_projection"], color=B_COLOR, ls="--", label="B")
    axes[0].set_ylabel("Projection [mm]"); axes[0].set_xlabel("Time [s]" if variant == "single" else "")
    axes[0].legend(handles=method_handles(), frameon=False, ncol=2, loc="upper right"); clean_axis(axes[0]); panel_label(axes[0], "a")

    axes[1].plot(t, d["penetration"], color=A_COLOR)
    axes[1].fill_between(t, 0, d["penetration"], where=d["hard"].astype(bool), color=A_LIGHT, step="mid")
    peak = d["peak_frame"]
    axes[1].scatter([t[peak]], [d["penetration"][peak]], color=A_COLOR, marker="o", zorder=3)
    axes[1].annotate(f"peak {d['penetration'][peak]:.2f} mm\nframe {peak}", xy=(t[peak], d["penetration"][peak]), xytext=(t[peak] + 1.0, d["penetration"][peak] * 0.70), arrowprops={"arrowstyle": "->", "lw": 0.65, "color": MID}, fontsize=6.2)
    axes[1].set_ylabel("Penetration [mm]"); axes[1].set_xlabel("Time [s]"); clean_axis(axes[1]); panel_label(axes[1], "b")

    axes[2].imshow(plt.imread(d["image_path"])); axes[2].axis("off"); panel_label(axes[2], "c", x=-0.07, y=1.01)
    axes[2].text(0.02, 0.03, f"Same-source B · {d['nearest_key'].replace('_', ' ').title()}\nframe {d['nearest_frame']} (nearest frozen snapshot)", transform=axes[2].transAxes, ha="left", va="bottom", fontsize=5.8, color="white", bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 2})
    axes[2].text(0.98, 0.97, f"A collision frame {peak}\nmatched render required", transform=axes[2].transAxes, ha="right", va="top", fontsize=5.8, color=INK, bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": MID, "linestyle": "--", "pad": 2})

    axes[3].axis("off"); panel_label(axes[3], "d", x=-0.07, y=1.01)
    pair = str(d["pair"][peak]).replace("|", " / ")
    status_text = (
        f"Same source episode {d['episode']}\n\n"
        f"A: HARD_FAIL (collision)\n"
        f"  {int(d['hard'].sum())} hard frames\n"
        f"  {pair}\n\n"
        f"B: {d['b_status'].replace('_', ' ')}\n"
        f"  0 hard-fail frames"
    )
    axes[3].text(0.03, 0.96, status_text, transform=axes[3].transAxes, ha="left", va="top", fontsize=6.9, linespacing=1.25)
    axes[3].add_patch(patches.Rectangle((0, 0), 1, 1, transform=axes[3].transAxes, fill=False, edgecolor="#B8B8B8", lw=0.6))
    if variant == "single":
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.06, top=0.98, hspace=0.48)
    else:
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.96, hspace=0.37, wspace=0.24)
    rows = [
        {
            "episode_index": d["episode"], "frame": i, "time_s": float(t[i]), "a_projection_mm": float(d["a_projection"][i]), "b_projection_mm": float(d["b_projection"][i]), "a_collision_penetration_mm": float(d["penetration"][i]), "a_hard_collision": int(d["hard"][i]), "a_body_pair_at_max": str(d["pair"][i]), "b_hard_collision": 0,
        }
        for i in range(len(t))
    ]
    rows.append({"episode_index": d["episode"], "record_type": "selection", "selection_rule": "larger frozen maximum penetration among ep35 and ep46", "selected_peak_frame": peak, "selected_peak_penetration_mm": float(d["penetration"][peak]), "available_b_snapshot": rel(d["image_path"]), "matched_a_snapshot": "RENDER_REQUIRED", "b_status": d["b_status"]})
    return fig, rows, [frozen.SOURCE_ARTIFACTS["collision_audit"], frozen.SOURCE_ARTIFACTS["collision_records"], d["a_path"], d["b_path"], d["image_path"], d["report_path"]]


def build_smoothness(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    data = ctx["data"]
    defs = [
        ("Direction reversals", "raw all-joint direction reversals mean", "s⁻¹ joint⁻¹"),
        ("Maximum step", "raw maximum adjacent step", "rad"),
        ("q̇ RMS", "raw qdot RMS", "rad s⁻¹"),
        ("q̈ RMS", "raw qddot RMS", "rad s⁻²"),
        ("Jerk RMS", "raw jerk RMS", "rad s⁻³"),
    ]
    if variant == "single":
        fig, axes = plt.subplots(5, 1, figsize=(width_inches(variant), 7.5))
    else:
        fig, axes = plt.subplots(1, 5, figsize=(width_inches(variant), 2.25))
    rows: list[dict[str, Any]] = []
    for ax, (label, key, unit) in zip(np.ravel(axes), defs):
        a = t2_value(data, key, "ACT-A40"); b = t2_value(data, key, "ACT-B40")
        ax.plot([0, 1], [a, b], color="#AAAAAA", lw=0.8)
        ax.scatter(0, a, marker="o", facecolor="white", edgecolor=A_COLOR, linewidth=0.9, zorder=3)
        ax.scatter(1, b, marker="s", facecolor="white", edgecolor=B_COLOR, linewidth=0.9, zorder=3)
        ax.set_xticks([0, 1], ["ACT-A", "ACT-B"]); ax.set_xlim(-0.35, 1.35); ax.set_ylim(0, max(a, b) * 1.22)
        ax.set_title(label, pad=3); ax.set_ylabel(unit); clean_axis(ax)
        ax.text(0, a, f" {a:.3g}", color=A_COLOR, ha="left", va="bottom", fontsize=5.8)
        ax.text(1, b, f" {b:.3g}", color=B_COLOR, ha="left", va="bottom", fontsize=5.8)
        rows.append({"metric": label, "ACT-A": a, "ACT-B": b, "unit": unit})
    fig.subplots_adjust(left=0.19 if variant == "single" else 0.075, right=0.985, bottom=0.05 if variant == "single" else 0.20, top=0.97 if variant == "single" else 0.84, hspace=0.72 if variant == "single" else 0.0, wspace=0.68 if variant == "double" else 0.0)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["experiment2"], frozen.SOURCE_ARTIFACTS["policy_table"]]


def build_numbers_glance(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    data = ctx["data"]
    rows = [
        {"Metric": "Wrist error [mm]", "A": t1_triple(data, "Wrist error mean / p95 / max", "FAIR A")[0], "B": t1_triple(data, "Wrist error mean / p95 / max", "PROPOSED B")[0]},
        {"Metric": "Whole-hand error [mm]", "A": t1_triple(data, "Whole-hand error mean / p95 / max", "FAIR A")[0], "B": t1_triple(data, "Whole-hand error mean / p95 / max", "PROPOSED B")[0]},
        {"Metric": "Bimanual error [mm]", "A": t1_triple(data, "Bimanual relation error mean / p95 / max", "FAIR A")[0], "B": t1_triple(data, "Bimanual relation error mean / p95 / max", "PROPOSED B")[0]},
        {"Metric": "Hard episodes [count/50]", "A": 2, "B": 0},
    ]
    fig, ax = plt.subplots(figsize=(width_inches(variant), 2.15 if variant == "single" else 1.75)); ax.axis("off")
    cell_text = [[r["Metric"], f"{r['A']:.3f}" if isinstance(r["A"], float) else str(r["A"]), f"{r['B']:.3f}" if isinstance(r["B"], float) else str(r["B"])] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=["Metric", "A", "B"], cellLoc="center", colLoc="center", loc="center", colWidths=[0.55, 0.225, 0.225])
    table.auto_set_font_size(False); table.set_fontsize(7.0); table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("white"); cell.set_edgecolor("#888888"); cell.set_linewidth(0.35)
        if r == 0:
            cell.set_text_props(weight="bold", color=A_COLOR if c == 1 else (B_COLOR if c == 2 else INK)); cell.set_linewidth(0.65)
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    ax.text(0.5, 0.04, "Mean errors; lower is better.  Full-50 frozen retargeting set.", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.98)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["retargeting_table"]]


def draw_story_pipeline(ax: plt.Axes) -> None:
    ax.axis("off")
    boxes = [(0.31, 0.79, 0.38, 0.14, "ALOHA n=50", "#F4F4F4", INK), (0.03, 0.43, 0.42, 0.22, "Trajectory-Centric A\n6-D wrist", A_LIGHT, A_COLOR), (0.55, 0.43, 0.42, 0.22, "Interaction-Centric B\nwhole-hand · bimanual", B_LIGHT, B_COLOR), (0.12, 0.11, 0.27, 0.14, "Dataset A → ACT-A", A_LIGHT, A_COLOR), (0.61, 0.11, 0.27, 0.14, "Dataset B → ACT-B", B_LIGHT, B_COLOR)]
    for x, y, w, h, text, fc, ec in boxes:
        ax.add_patch(patches.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=fc, edgecolor=ec, lw=0.65)); ax.text(x+w/2, y+h/2, text, transform=ax.transAxes, ha="center", va="center", fontsize=5.5)
    for start, end in [((0.5,0.79),(0.24,0.65)),((0.5,0.79),(0.76,0.65)),((0.24,0.43),(0.255,0.25)),((0.76,0.43),(0.745,0.25))]:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle":"-|>","lw":0.5,"color":MID,"mutation_scale":5})


def build_storyboard(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    # The composite is intentionally double-column in spirit; the single form is
    # provided for review but not recommended for publication.
    fig, axes = plt.subplots(2, 2, figsize=(width_inches(variant), 5.35 if variant == "single" else 4.35))
    rows: list[dict[str, Any]] = []
    draw_story_pipeline(axes[0, 0]); panel_label(axes[0, 0], "a", x=-0.04, y=1.00)

    metrics = [("Wrist", "wrist"), ("Whole-hand", "whole_hand"), ("Bimanual", "bimanual")]
    ax = axes[0, 1]
    y = np.arange(3)[::-1]
    for yy, (label, metric) in zip(y, metrics):
        a = float(ctx["a_frame"][metric].mean()*1000); b = float(ctx["b_frame"][metric].mean()*1000)
        ax.plot([a, b], [yy, yy], color="#AAAAAA", lw=0.7); ax.scatter(a, yy, marker="o", facecolor="white", edgecolor=A_COLOR); ax.scatter(b, yy, marker="s", facecolor="white", edgecolor=B_COLOR)
        rows.append({"panel": "b", "metric": label, "A_mean_mm": a, "B_mean_mm": b})
    ax.set_yticks(y, [m[0] for m in metrics]); ax.set_xlabel("Retargeting error [mm]"); clean_axis(ax, "x"); panel_label(ax, "b")

    ax = axes[1, 0]
    x = np.arange(3)
    data = ctx["data"]
    for method, color, marker, table_method in (("ACT-A", A_COLOR, "o", "ACT-A40"), ("ACT-B", B_COLOR, "s", "ACT-B40")):
        values = [t2_value(data, key, table_method) for key in ("predicted source wrist error mean", "predicted whole-hand error mean", "predicted bimanual relation error mean")]
        ax.plot(x, values, color=color, ls="-" if method == "ACT-A" else "--", marker=marker, markerfacecolor="white", label=method)
        rows.extend({"panel": "c", "metric": label, "method": method, "mean_mm": value} for (label, _), value in zip(metrics, values))
    ax.set_xticks(x, ["Wrist", "Whole-hand", "Bimanual"]); ax.set_ylabel("Predicted error [mm]"); clean_axis(ax); panel_label(ax, "c"); ax.legend(frameon=False, ncol=2, loc="upper right")

    ax = axes[1, 1]
    a_path, b_path, _ = representative_assets(ctx)
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        for side in ("left", "right"):
            ref = np.asarray(a[f"target_{side}_interaction_frame_position_world"])*1000
            aa = np.asarray(a[f"achieved_{side}_physical_grasp_frame_position_world"])*1000
            bb = np.asarray(b[f"achieved_{side}_physical_grasp_frame_position_world"])*1000
            for values, color, ls in ((ref, INK, ":"),(aa,A_COLOR,"-"),(bb,B_COLOR,"--")):
                ax.plot(values[:,0], values[:,1], color=color, ls=ls, lw=0.8)
    ax.set_xlabel("World x [mm]"); ax.set_ylabel("World y [mm]"); ax.set_aspect("equal", adjustable="datalim"); clean_axis(ax,"both"); panel_label(ax,"d"); ax.set_title(f"Criterion-selected episode {ctx['representative_episode']}",fontsize=6.4)
    fig.subplots_adjust(left=0.13 if variant == "single" else 0.09, right=0.98, bottom=0.09, top=0.97, hspace=0.40, wspace=0.36)
    return fig, rows, [frozen.SOURCE_ARTIFACTS["retargeting_table"], frozen.SOURCE_ARTIFACTS["experiment2"], a_path, b_path]


def build_phase_heatmap(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    heldout = ctx["heldout"]; mats = ctx["phase_matrices"]
    fig, axes = plt.subplots(2, 1, figsize=(width_inches(variant), 3.25 if variant == "single" else 2.80), sharex=True)
    cmap = mpl_colors.ListedColormap(["#F2F2F2", "#607C68"])
    for ax, method in zip(axes, ("a", "b")):
        ax.imshow(mats[method], cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(8), [f"ep{e:02d}" for e in heldout]); ax.set_ylabel("ACT-A" if method == "a" else "ACT-B")
        ax.set_xticks(np.arange(8), [x.replace(" ", "\n", 1) for x in PHASE_LABELS])
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True); ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.55); ax.tick_params(which="minor", left=False, bottom=False)
    fig.subplots_adjust(left=0.18 if variant == "single" else 0.10, right=0.99, bottom=0.25, top=0.98, hspace=0.10)
    return fig, ctx["phase_rows"], [frozen.SOURCE_ARTIFACTS["experiment2"], frozen.SOURCE_ARTIFACTS["heldout8"]]


EXPERIMENT3_METRICS = ["Phase completion", "Semantic task success", "Whole-hand error", "Bimanual error", "HOA", "RPL", "SWPE"]


def build_experiment3_template(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    rows = [{"metric": metric, A_NAME: "NA", B_NAME: "NA", "status": "TEMPLATE_ONLY"} for metric in EXPERIMENT3_METRICS]
    fig, ax = plt.subplots(figsize=(width_inches(variant), 2.75 if variant == "single" else 2.20)); ax.axis("off")
    table = ax.table(cellText=[[r["metric"], "NA", "NA"] for r in rows], colLabels=["Source-conditioned rollout metric", "A", "B"], cellLoc="center", colLoc="center", bbox=[0.0, 0.12, 1.0, 0.85], colWidths=[0.62, 0.19, 0.19])
    table.auto_set_font_size(False); table.set_fontsize(6.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("white"); cell.set_edgecolor("#8A8A8A"); cell.set_linewidth(0.35)
        if r == 0: cell.set_text_props(weight="bold"); cell.set_linewidth(0.65)
        if c == 0 and r > 0: cell.set_text_props(ha="left")
    ax.text(0.5, 0.02, "Experiment 3 template only — no rollout values are available or used.", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.98)
    return fig, rows, []


def make_specs(ctx: dict[str, Any]) -> list[tuple[FigureSpec, Callable[..., Any], str]]:
    af, bf = ctx["a_frame"], ctx["b_frame"]
    n_wrist = (len(af["wrist"]), len(bf["wrist"]))
    n_whole = (len(af["whole_hand"]), len(bf["whole_hand"]))
    n_bimanual = (len(af["bimanual"]), len(bf["bimanual"]))
    n_projection = (len(af["projection"]), len(bf["projection"]))
    a_path, _, _ = representative_assets(ctx)
    with np.load(a_path, allow_pickle=False) as z:
        representative_frames = len(z["timestamp"])
    failure_frames = len(failure_data(ctx)["t"])
    specs: list[tuple[FigureSpec, Callable[..., Any], str]] = [
        (
            FigureSpec(
                1, "01_pipeline", "MUST_USE", "double-column",
                "How does the controlled A/B representation comparison propagate from demonstrations to heldout evaluation?",
                "Matched paper-core pipeline; both branches originate from the same 50 demonstrations and differ at the retargeting representation.",
                "Fifty source demonstrations produce matched Dataset A/B branches and ACT-A/B policies evaluated on the same heldout protocol.",
                "50 demonstrations; 40 training episodes per policy; 8 heldout episodes.",
                "Fig. 1. Paper-core A/B pipeline. The same 50 ALOHA demonstrations are retargeted using Trajectory-Centric A, based on a 6-D wrist trajectory, or Interaction-Centric B, based on interaction frames, whole-hand geometry, bimanual relations, and ownership transitions. The matched datasets supervise ACT-A and ACT-B, respectively; retargeting, heldout prediction, and phase behavior are evaluated without including unavailable source-conditioned rollout results.",
            ), build_pipeline, "double",
        ),
        (
            FigureSpec(
                2, "02_core_tradeoff", "MUST_USE", "double-column",
                "Does the representation choice exchange wrist fidelity for whole-hand and bimanual interaction fidelity?",
                "Frame-weighted mean Euclidean position/relation errors; points are 50 episode means and error bars are 95% episode-cluster bootstrap CIs (20,000 resamples).",
                "A/B means are 3.742/77.853 mm (wrist), 90.820/21.157 mm (whole-hand), and 86.786/35.979 mm (bimanual).",
                f"50 paired episodes; A/B frame samples: wrist {n_wrist[0]}/{n_wrist[1]}, whole-hand {n_whole[0]}/{n_whole[1]}, bimanual {n_bimanual[0]}/{n_bimanual[1]}.",
                "Fig. 2. Core retargeting trade-off over 50 paired source episodes. Small symbols show per-episode mean errors, filled symbols show frame-weighted means, and error bars denote 95% episode-cluster bootstrap confidence intervals from 20,000 resamples. Trajectory-Centric A yields lower wrist error (3.742 versus 77.853 mm), whereas Interaction-Centric B yields lower whole-hand (21.157 versus 90.820 mm) and bimanual relation error (35.979 versus 86.786 mm). Lower error is better; no claim of statistical significance is made.",
            ), build_core_tradeoff, "double",
        ),
        (
            FigureSpec(
                3, "03_paired_episode_effects", "STRONG", "double-column",
                "Are the retargeting differences consistent across the same 50 source identities?",
                "Connected per-episode mean errors; large symbols are medians with 95% episode bootstrap CIs, and annotations report paired mean B−A effects.",
                "B−A paired means are +74.115 mm (wrist), −69.678 mm (whole-hand), and −50.819 mm (bimanual).",
                "50 paired source episodes.",
                "Fig. 3. Source-paired retargeting effects for 50 episodes. Each thin line connects the two methods for the same source identity; large symbols show the median and error bars show its 95% episode-bootstrap confidence interval from 20,000 resamples. The annotated paired mean B−A effects are +74.115 mm for wrist error, −69.678 mm for whole-hand error, and −50.819 mm for bimanual relation error, revealing the representation-dependent trade-off across episodes rather than only in aggregate bars.",
            ), build_paired_effects, "double",
        ),
        (
            FigureSpec(
                4, "04_effect_forest", "MUST_USE", "single-column",
                "What are the direction, magnitude, and uncertainty of the paired A/B retargeting effects?",
                "Mean of 50 paired episode-level B−A differences with 95% paired bootstrap CIs from 20,000 resamples; lower error is better.",
                "B−A [95% CI]: wrist +74.115 [71.498, 76.615], whole-hand −69.678 [−73.155, −66.531], bimanual −50.819 [−55.682, −46.217], projection −2.450 [−4.582, −0.752] mm.",
                "50 paired source episodes; 20,000 paired bootstrap resamples.",
                "Fig. 4. Forest summary of paired B−A differences in per-episode mean error over 50 source identities. Points are paired mean differences and horizontal bars are 95% confidence intervals from 20,000 paired bootstrap resamples. Positive wrist difference indicates lower error for Trajectory-Centric A, while negative whole-hand, bimanual, and projection differences indicate lower error for Interaction-Centric B. Lower error is better; the intervals quantify estimation uncertainty without an additional significance test.",
            ), build_effect_forest, "single",
        ),
        (
            FigureSpec(
                5, "05_tradeoff_scatter", "STRONG", "double-column",
                "Where do individual episodes lie in the wrist-versus-interaction error space?",
                "Episode-level mean errors; large symbols are method medians and crosshairs span the interquartile ranges.",
                "A occupies the low-wrist/higher-interaction region, while B shifts toward higher wrist/lower whole-hand and bimanual error.",
                "50 episodes per method in each panel.",
                "Fig. 5. Episode-level wrist-versus-interaction trade-off for the 50 matched source episodes. Open symbols show episode means; filled symbols denote method medians and crosshairs span the corresponding interquartile ranges. Trajectory-Centric A occupies the lower-wrist, higher-interaction-error region, whereas Interaction-Centric B shifts toward higher wrist deviation with lower whole-hand and bimanual relation error. No regression model is imposed.",
            ), build_tradeoff_scatter, "double",
        ),
        (
            FigureSpec(
                6, "06_feasibility", "MUST_USE", "double-column",
                "How do usability classifications and hard-failure modes differ between A and B?",
                "Full-50 episode classification under the frozen validation criteria; WARNING is distinct from HARD_FAIL.",
                "A: 27 CLEAN/21 WARNING/2 HARD; B: 13 CLEAN/37 WARNING/0 HARD. Only A has hard collision episodes (2).",
                "50 episodes per method.",
                "Fig. 6. Retargeting feasibility over 50 source episodes per method. The horizontal composition shows 27 CLEAN, 21 WARNING, and 2 HARD episodes for Trajectory-Centric A, compared with 13 CLEAN, 37 WARNING, and no HARD episodes for Interaction-Centric B. The failure panel identifies two hard collision episodes for A and zero hard collision, IK, joint-limit, or branch failures for B. WARNING is a usable diagnostic category and is not equivalent to HARD_FAIL; the primary feasibility observation is the elimination of hard failures in B.",
            ), build_feasibility, "double",
        ),
        (
            FigureSpec(
                7, "07_supervision_to_policy", "MUST_USE", "double-column",
                "Does the representation-specific geometric pattern persist from retargeted supervision to heldout ACT prediction?",
                "Top row: full-50 retargeting errors over episode clusters. Bottom row: 72 HELDOUT8 prediction probes. Filled symbols are frame-weighted means; error bars are 95% sample-cluster bootstrap CIs.",
                "A/ACT-A remains wrist-oriented (3.742/24.958 mm), whereas B/ACT-B remains lower in whole-hand (21.157/58.330 mm) and bimanual error (35.979/85.920 mm).",
                "Retargeting: 50 episodes; policy prediction: 72 probes from 8 heldout episodes.",
                "Fig. 7. Preservation of representation-dependent geometry from retargeted supervision (top) to learned ACT prediction (bottom). Small symbols show episode or prediction-probe means, filled symbols show the frozen frame-weighted aggregate, and error bars are 95% sample-cluster bootstrap confidence intervals from 20,000 resamples. Retargeting uses 50 episodes, while policy evaluation uses 72 probes from eight heldout episodes; the rows therefore represent distinct sample sets rather than paired before-and-after observations. The wrist-oriented A/ACT-A and interaction-oriented B/ACT-B patterns persist after learning.",
            ), build_supervision_policy, "double",
        ),
        (
            FigureSpec(
                8, "08_action_rmse", "STRONG", "single-column",
                "How do ACT-A and ACT-B compare in immediate and full-chunk heldout action prediction?",
                "Frozen aggregate RMSE over 100,800 scored joint-frames per method; no probe-level RMSE distribution is stored, so no uncertainty bars are shown.",
                "First-action RMSE is 0.043554/0.044165 rad for ACT-A/B; full-chunk RMSE is 0.154074/0.109001 rad.",
                "72 probes from 8 heldout episodes; 100,800 scored joint-frames per method.",
                "Fig. 8. Heldout action prediction accuracy for ACT-A and ACT-B. Values are frozen aggregate root-mean-square errors over 100,800 scored joint-frames per method (72 probes from eight heldout episodes). First-action RMSE is similar (0.043554 versus 0.044165 rad), whereas ACT-B has lower full-valid-chunk RMSE (0.109001 versus 0.154074 rad). Probe-level RMSE values are not stored in the authoritative artifact, so uncertainty bars are not inferred.",
            ), build_action_rmse, "single",
        ),
        (
            FigureSpec(
                9, "09_phase_success", "MUST_USE", "single-column",
                "Which exact paper-core behaviors contribute to the heldout 56/64 and 52/64 scores?",
                "Behavior detection percentage across eight heldout episodes for each of the exact eight canonical paper phases.",
                "ACT-A detects 56/64 phase behaviors and ACT-B detects 52/64; per-phase counts expose where those totals differ.",
                "8 heldout episodes × 8 phases = 64 binary evaluations per method.",
                "Fig. 9. Heldout phase-behavior detection under the exact eight-category paper-core taxonomy. Each point is the percentage of eight heldout episodes in which the corresponding behavior was detected; ACT-A totals 56/64 and ACT-B totals 52/64 detections. The temporal ordering is left approach, left grasp, left transport, handoff approach, dual-hand configuration, right owned, right transport, and release. The separate nine-probe diagnostic taxonomy is not used, and the figure does not imply that ACT-B has the higher aggregate phase score.",
            ), build_phase_success, "single",
        ),
        (
            FigureSpec(
                10, "10_error_ecdf", "STRONG", "double-column",
                "Do frame-level retargeting error distributions support the mean-level trade-off?",
                "Empirical CDFs over all frozen frame/hand samples; open symbols mark median and p95.",
                "The full distributions retain A's wrist advantage and B's whole-hand and bimanual advantages, including their tails.",
                f"A/B samples: wrist {n_wrist[0]}/{n_wrist[1]}, whole-hand {n_whole[0]}/{n_whole[1]}, bimanual {n_bimanual[0]}/{n_bimanual[1]}.",
                "Fig. 10. Frame-level empirical cumulative distributions of wrist, whole-hand, and bimanual relation errors over the frozen 50-episode retargeting sets. Open symbols mark each distribution's median and 95th percentile. The distributions corroborate the aggregate trade-off: Trajectory-Centric A concentrates wrist errors near zero, whereas Interaction-Centric B shifts whole-hand and bimanual errors downward. All errors are in millimetres; no synthetic samples or fitted distributions are used.",
            ), build_error_ecdf, "double",
        ),
        (
            FigureSpec(
                11, "11_projection_ecdf", "STRONG", "double-column",
                "How is feasibility projection distributed, including the rare upper tail?",
                "Empirical CDF and log-scale complementary empirical CDF over all arm-frame projection magnitudes.",
                "A mean/p95/max is 2.525/17.054/94.341 mm; B mean/median/p95/max is 0.077/0/0/84.430 mm.",
                f"A/B arm-frame samples: {n_projection[0]}/{n_projection[1]}.",
                "Fig. 11. Distribution of feasibility projection magnitude over all arm-frame samples from 50 episodes per method. Panel (a) shows the empirical CDF, and panel (b) uses a logarithmic tail-probability axis to retain rare large values. Interaction-Centric B has median and 95th-percentile projection of 0 mm and mean 0.077 mm, but its 84.430 mm maximum remains visible; Trajectory-Centric A has mean/p95/max of 2.525/17.054/94.341 mm. Thus the near-zero central mass does not conceal the upper tail.",
            ), build_projection_distribution, "double",
        ),
        (
            FigureSpec(
                12, "12_representative_trajectory", "MUST_USE", "double-column",
                "How do wrist and whole-hand paths differ in a predeclared representative episode?",
                "Top-view Cartesian paths and right-minus-left interaction-frame displacement; episode selected nearest the median B whole-hand error among common feasible episodes, with lower final index breaking the exact central tie.",
                f"Episode {ctx['representative_episode']} is the criterion-selected representative; A follows source wrist paths, while B more closely follows interaction-frame paths.",
                f"One source-matched episode ({ctx['representative_episode']}), {representative_frames} frames at 30 Hz.",
                f"Fig. 12. Representative source-matched trajectories for episode {ctx['representative_episode']} ({representative_frames} frames at 30 Hz). The episode was selected before visualization as the common feasible episode nearest to the median Interaction-Centric B whole-hand error, with lower final index breaking the exact central tie; no visual criterion was used. Panels show left/right wrist paths, whole-hand interaction-frame paths, and bimanual relative displacement with semantic event markers. Trajectory-Centric A follows the source wrist reference more closely, whereas Interaction-Centric B follows the interaction-frame geometry more closely.",
            ), build_representative_trajectory, "double",
        ),
        (
            FigureSpec(
                13, "13_failure_case", "STRONG", "double-column",
                "What temporal collision evidence explains a frozen hard failure of Trajectory-Centric A?",
                "Episode selected by the larger authoritative maximum penetration among ep35/ep46; timelines use framewise projection and frozen self-collision penetration records.",
                "Episode 46 has 102 A hard-collision frames and 8.413 mm peak arm–torso penetration; same-source B is usable with warning and has no hard-fail frame.",
                f"One predeclared failure episode (ep46), {failure_frames} trajectory frames; 102 hard-collision frames.",
                "Fig. 13. Frozen hard-collision case for source episode 46, selected by the larger authoritative peak penetration among the two final Fair-A hard episodes (35 and 46), not by visual appearance. Projection and penetration are plotted over the complete trajectory; A exhibits 102 hard collision frames and 8.413 mm peak penetration between the left shoulder-roll and torso links, whereas same-source B remains hard-fail free and is classified usable with warning. The available B snapshot is the nearest frozen semantic frame; the matched A collision snapshot is explicitly marked as requiring rendering. The evidence shows that a wrist-level target may be statically reachable yet infeasible under continuous temporal and self-collision constraints.",
                render_required=True,
            ), build_failure_case, "double",
        ),
        (
            FigureSpec(
                14, "14_control_regularity", "OPTIONAL", "double-column",
                "How do selected ACT-A/B predictions compare across control-regularity metrics?",
                "Five frozen aggregate metrics shown in separate unit-preserving panels; no normalization or composite score.",
                "The comparison is mixed: neither ACT-A nor ACT-B is uniformly smoother across reversals, maximum step, qdot, qddot, and jerk.",
                "72 raw prediction chunks from 8 heldout episodes per method.",
                "Fig. 14. Control-regularity metrics for the selected ACT-A and ACT-B predictions, computed from 72 heldout raw chunks per method. Direction reversals, maximum adjacent joint step, velocity RMS, acceleration RMS, and jerk RMS retain their original units in separate panels. The result is mixed—ACT-B is lower in maximum step, velocity RMS, and acceleration RMS, while ACT-A is lower in reversals and jerk RMS—so the figure does not claim that either policy is smoother overall.",
            ), build_smoothness, "double",
        ),
        (
            FigureSpec(
                15, "15_storyboard", "OPTIONAL", "double-column",
                "Can the representation, retargeting, policy, and trajectory evidence be assembled into one compact paper story?",
                "Composite redraw of artifact-driven pipeline, aggregate errors, heldout predictions, and criterion-selected trajectory; no raster screenshot panels.",
                "The composite links the representation choice to its retargeting and learned-policy signatures without using Experiment 3.",
                "50 retargeting episodes; 72 policy probes; one criterion-selected representative episode.",
                "Fig. 15. Double-column paper-storyboard candidate combining the controlled representation pipeline, the wrist-versus-interaction retargeting contrast, heldout ACT geometric prediction, and the criterion-selected representative trajectory. Quantitative panels use the same frozen Experiment 1 and Experiment 2 artifacts as the standalone figures. The composite is intended for layout review and does not introduce additional estimates or unavailable Experiment 3 results.",
            ), build_storyboard, "double",
        ),
        (
            FigureSpec(
                16, "16_numbers_at_glance", "OPTIONAL", "single-column",
                "What are the core full-50 retargeting results in a compact visual summary?",
                "Artifact-loaded full-frame mean errors and hard-episode counts; lower is better for error metrics.",
                "A: 3.742/90.820/86.786 mm and 2 hard episodes; B: 77.853/21.157/35.979 mm and 0 hard episodes.",
                "50 episodes per method.",
                "Fig. 16. Numbers-at-a-glance summary of the frozen full-50 retargeting result. Entries are mean wrist, whole-hand, and bimanual errors in millimetres plus hard-episode counts. Trajectory-Centric A has lower wrist error, while Interaction-Centric B has lower interaction errors and no hard episodes. This compact table-like figure is intended for presentation or overview use rather than replacing the distributional paper figures.",
            ), build_numbers_glance, "single",
        ),
        (
            FigureSpec(
                17, "17_optional_phase_heatmap", "OPTIONAL", "double-column",
                "Which episode-by-phase detections underlie the compact phase-success figure?",
                "Binary 8-heldout-episode × 8-paper-phase matrix for each selected policy.",
                "The matrices sum to 56/64 for ACT-A and 52/64 for ACT-B.",
                "8 heldout episodes × 8 phases per method.",
                "Fig. 17. Supplementary episode-by-phase behavior matrices for ACT-A and ACT-B. Rows are the eight heldout source episodes and columns are exactly the eight paper-core behaviors used to compute 56/64 and 52/64. Green cells indicate detected behavior and gray cells indicate no detection. This compact diagnostic is supplementary; the paired phase-success plot is preferred for the main paper.",
            ), build_phase_heatmap, "double",
        ),
        (
            FigureSpec(
                18, "18_experiment3_template", "OPTIONAL", "double-column",
                "How will source-conditioned rollout metrics be inserted when approved Experiment 3 results become available?",
                "NA-only template for phase completion, semantic task success, whole-hand error, bimanual error, HOA, RPL, and SWPE.",
                "All A/B values are explicitly NA; no Experiment 3 scientific result is used.",
                "No available Experiment 3 sample.",
                "Fig. 18. Template for future source-conditioned rollout evaluation. Phase completion, semantic task success, whole-hand error, bimanual error, handoff-order accuracy (HOA), RPL, and SWPE are listed with all A/B values explicitly marked NA. The template contains no inferred, interpolated, or fabricated Experiment 3 result and should be populated only from an approved frozen artifact.",
            ), build_experiment3_template, "double",
        ),
    ]
    return specs


def centralize_legacy() -> list[Path]:
    originals = sorted(OLD_BANK.glob("fig*/*.png"), key=lambda p: (p.parent.name, p.name))
    legacy_pngs: list[Path] = []
    records = []
    for index, png in enumerate(originals, start=1):
        clean_stem = f"Legacy{index:02d}_{png.stem}"
        for suffix, directory in ((".png", "png_all"), (".pdf", "pdf_all"), (".svg", "svg_all")):
            src = png.with_suffix(suffix)
            if not src.is_file():
                raise FileNotFoundError(f"legacy publication format missing: {src}")
            dst = OUT / directory / f"{clean_stem}{suffix}"
            shutil.copy2(src, dst)
            if suffix == ".png":
                legacy_pngs.append(dst)
        records.append({"central_stem": clean_stem, "original_png": rel(png), "original_sha256": sha256(png), "central_png": rel(OUT / "png_all" / f"{clean_stem}.png"), "central_sha256": sha256(OUT / "png_all" / f"{clean_stem}.png")})
    write_json(OUT / "figure_manifest/LEGACY_ASSET_MAP.json", records)
    return legacy_pngs


def load_contact_font(size: int) -> ImageFont.ImageFont:
    for path in (Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_contact_sheet(paths: list[Path], destination: Path, columns: int, cell_width: int = 410, thumb_height: int = 270) -> None:
    label_height = 42
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet); font = load_contact_font(17)
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            tile = im.convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            tile.thumbnail((cell_width - 20, thumb_height - 16), resampling.LANCZOS)
            col, row = i % columns, i // columns
            x0 = col * cell_width + (cell_width - tile.width) // 2
            y0 = row * (thumb_height + label_height) + (thumb_height - tile.height) // 2
            sheet.paste(tile, (x0, y0))
            label = path.name
            box = draw.textbbox((0, 0), label, font=font)
            tx = col * cell_width + max(6, (cell_width - (box[2] - box[0])) // 2)
            ty = row * (thumb_height + label_height) + thumb_height + 7
            draw.text((tx, ty), label, fill=INK, font=font)
        draw.rectangle((col * cell_width, row * (thumb_height + label_height), (col + 1) * cell_width - 1, (row + 1) * (thumb_height + label_height) - 1), outline="#DDDDDD", width=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, dpi=(300, 300), optimize=True)


def write_manifest(legacy_pngs: list[Path]) -> None:
    lines = [
        "# Final JKROS Figure Selection",
        "",
        "All numerical figures are regenerated from frozen CSV/JSON/NPZ artifacts. Original files under `outputs/paper_figure_bank/` remain unchanged. Experiment 3 is template-only with every unavailable value set to NA.",
        "",
        "## Consistent method encoding",
        "",
        "- Trajectory-Centric A: muted blue, circle marker, solid line, diagonal hatch where applicable.",
        "- Interaction-Centric B: muted vermilion, square marker, dashed line, dotted hatch where applicable.",
        "",
        "## Figure manifest",
        "",
        "| File | Scientific question | Underlying artifact | Samples | Metric definition | Main numerical result | Priority | Width | New rendering | Caption |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in FIGURE_RECORDS:
        artifacts = "<br>".join(a["path"] for a in r["underlying_artifacts"]) or "None (NA-only template)"
        cells = [r["filename"], r["scientific_question"], artifacts, r["number_of_episodes_or_frames"], r["metric_definition"], r["main_numerical_result"], r["priority"], r["column_recommendation"], "YES" if r["new_rendering_required"] else "NO", r["caption_path"]]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines.extend(
        [
            "",
            "## Recommended JKROS body set",
            "",
            "1. `01_pipeline` — controlled A/B paper pipeline.",
            "2. `02_core_tradeoff` — central wrist-versus-interaction claim.",
            "3. `04_effect_forest` — paired magnitude and uncertainty.",
            "4. `06_feasibility` — warning-aware hard-failure comparison.",
            "5. `07_supervision_to_policy` — representation pattern persists downstream.",
            "6. `09_phase_success` — compact replacement for the large binary heatmap.",
            "7. `12_representative_trajectory` — predeclared, non-cherry-picked trajectory example.",
            "",
            "## Strong alternatives / supplement",
            "",
            "- `03_paired_episode_effects`, `05_tradeoff_scatter`, `08_action_rmse`, `10_error_ecdf`, `11_projection_ecdf`, and `13_failure_case`.",
            "- `17_optional_phase_heatmap` is retained only as a compact supplementary diagnostic.",
            "",
            "## Rendering and integrity status",
            "",
            "- Every MUST_USE figure is complete in 88 mm and 178 mm PNG (600 dpi), PDF, and SVG variants.",
            "- Figure 13 contains an available same-source B snapshot and an explicit placeholder for a matched A collision-frame render; generating that optional qualitative cell requires Isaac/GPU rendering.",
            "- The original semantic frame strip remains available as a centralized legacy candidate and still requires matched Fair-A renders.",
            "- GPU used: NO.",
            "- Training, checkpoints, Dataset A/B, Isaac, and the paper-core job touched: NO.",
            "- Experiment 3: TEMPLATE ONLY; all values NA.",
            "",
            "## Centralized legacy collection",
            "",
            f"{len(legacy_pngs)} original PNG candidates were copied byte-for-byte into `png_all/` with `LegacyXX_` prefixes; corresponding PDF and SVG files are in `pdf_all/` and `svg_all/`. Source-to-copy hashes are recorded in `LEGACY_ASSET_MAP.json`.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python3 tools/generate_jkros_pubready_figures.py",
            "python3 tools/validate_jkros_pubready_figures.py",
            "```",
        ]
    )
    (OUT / "figure_manifest/FINAL_FIGURE_SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "figure_manifest/figure_manifest.json", FIGURE_RECORDS)


def main() -> None:
    ensure_dirs(); configure_style()
    ctx = main_context()
    specs = make_specs(ctx)
    for spec, builder, master_variant in specs:
        fig, rows, sources = builder(ctx, master_variant)
        variant_builder = (lambda v, b=builder: b(ctx, v)) if spec.priority == "MUST_USE" else None
        save_master(spec, fig, rows, sources, variant_builder)
        print(f"generated {spec.stem}", flush=True)

    legacy_pngs = centralize_legacy()
    new_pngs = [OUT / "png_all" / f"{r['filename']}" for r in FIGURE_RECORDS]
    all_candidates = new_pngs + legacy_pngs
    must_pngs = [OUT / "png_all" / r["filename"] for r in FIGURE_RECORDS if r["priority"] == "MUST_USE"]
    all_sheet = OUT / "png_all/ALL_PNG_CONTACT_SHEET.png"
    must_sheet = OUT / "png_all/MUST_USE_CONTACT_SHEET.png"
    make_contact_sheet(all_candidates, all_sheet, columns=5)
    make_contact_sheet(must_pngs, must_sheet, columns=3, cell_width=480, thumb_height=315)
    shutil.copy2(all_sheet, OUT / "png_all/CONTACT_SHEET_ALL_FIGURES.png")
    shutil.copy2(all_sheet, OUT / "ALL_PNG_CONTACT_SHEET.png")
    shutil.copy2(must_sheet, OUT / "MUST_USE_CONTACT_SHEET.png")
    shutil.copy2(must_sheet, OUT / "must_use/MUST_USE_CONTACT_SHEET.png")
    write_manifest(legacy_pngs)

    priorities = {p: sum(r["priority"] == p for r in FIGURE_RECORDS) for p in ("MUST_USE", "STRONG", "OPTIONAL")}
    summary = {
        "status": "READY_WITH_DECLARED_OPTIONAL_RENDER_PLACEHOLDER",
        "publication_master_figures": len(FIGURE_RECORDS),
        "legacy_figures_centralized": len(legacy_pngs),
        "browseable_png_candidates": len(all_candidates),
        "must_use_variants": 2 * priorities["MUST_USE"],
        "priority_counts": priorities,
        "captions": len(FIGURE_RECORDS),
        "source_csv_files": len(FIGURE_RECORDS),
        "representative_episode": ctx["representative_episode"],
        "failure_episode": 46,
        "experiment3": "TEMPLATE_ONLY_ALL_NA",
        "gpu_used": False,
        "isaac_run": False,
        "training_run": False,
        "dataset_ab_modified": False,
        "paper_core_job_touched": False,
        "scientific_verification": "ALL_CONFIRMED_VALUES_MATCH_FROZEN_ARTIFACTS",
        "central_png_directory": rel(OUT / "png_all"),
        "all_contact_sheet": rel(all_sheet),
        "must_use_contact_sheet": rel(must_sheet),
    }
    write_json(OUT / "figure_manifest/generation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
