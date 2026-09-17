#!/usr/bin/env python3
"""Generate exactly the six final JKROS paper figures from frozen artifacts.

The script performs read-only scientific analysis and writes only beneath
``outputs/paper_final_figures``.  It does not run retargeting, training, policy
inference, or simulation.  Fig. 6 consumes the separately audited matched-still
render output produced by ``render_jkros_final_matched_stills.py``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import generate_paper_figure_bank as frozen  # noqa: E402
import generate_jkros_pubready_figures as pubready  # noqa: E402


OUT = ROOT / "outputs/paper_final_figures"
FOLDERS = {
    1: "Fig01_Method_Overview",
    2: "Fig02_Retargeting_Tradeoff",
    3: "Fig03_Paired_Statistics",
    4: "Fig04_Feasibility",
    5: "Fig05_Supervision_to_Policy",
    6: "Fig06_Representative_Motion",
}
STEMS = {
    1: "Fig01_Method_Overview",
    2: "Fig02_Retargeting_Tradeoff",
    3: "Fig03_Paired_Statistical_Effect",
    4: "Fig04_Feasibility",
    5: "Fig05_Supervision_to_Policy",
    6: "Fig06_Representative_Motion",
}
SUPPORT_DIRS = ("contact_sheet", "captions", "manifest")

A_NAME = "Trajectory-Centric A"
B_NAME = "Interaction-Centric B"
A_COLOR = "#3F5F8F"
B_COLOR = "#A9553B"
A_LIGHT = "#DCE3ED"
B_LIGHT = "#EDDCD6"
INK = "#252525"
MID = "#717171"
GRID = "#D9D9D9"
GREEN = "#6D8C73"
AMBER = "#C3A64B"
RED = "#9B4A42"
SINGLE_MM = 88.0
DOUBLE_MM = 178.0
MM_TO_IN = 1.0 / 25.4
BOOTSTRAP_N = 20_000
BOOTSTRAP_SEED = 20260827

FOREST_SOURCE = OUT.parent / "paper_figure_bank_pubready/source_data/04_effect_forest.csv"
RENDER_REPORT = OUT / FOLDERS[6] / "render_assets/matched_render_report.json"
GENERATION_COMMAND = "python3 tools/generate_jkros_final_figures.py"
RENDER_COMMAND = (
    "/home/jbnu/miniconda3/bin/conda run -n isaaclab6 --no-capture-output "
    "/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p "
    "tools/render_jkros_final_matched_stills.py --episode 23 --headless"
)


@dataclass(frozen=True)
class FigureInfo:
    number: int
    question: str
    result: str
    sample_count: str
    metric_definition: str
    paper_location: str
    recommended_variant: str
    caption: str
    gpu_render_used: bool = False


FIGURE_METADATA: list[dict[str, Any]] = []


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    fields: list[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    result = []
    for path in dict.fromkeys(Path(item).resolve() for item in paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({"path": rel(path), "sha256": sha256(path)})
    return result


def ensure_dirs() -> None:
    for folder in FOLDERS.values():
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    for folder in SUPPORT_DIRS:
        (OUT / folder).mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.8,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.1,
            "lines.markersize": 4.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def width_inches(variant: str) -> float:
    return (SINGLE_MM if variant == "single" else DOUBLE_MM) * MM_TO_IN


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.48, alpha=0.78)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
    )


def method_handles(policy_combined: bool = False) -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=A_COLOR, marker="o", linestyle="-", markerfacecolor="white",
            markeredgewidth=0.9, label="A / ACT-A" if policy_combined else A_NAME,
        ),
        Line2D(
            [0], [0], color=B_COLOR, marker="s", linestyle="--", markerfacecolor="white",
            markeredgewidth=0.9, label="B / ACT-B" if policy_combined else B_NAME,
        ),
    ]


def save_formats(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")


def save_final_figure(
    info: FigureInfo,
    builders: dict[str, Callable[[], tuple[plt.Figure, list[dict[str, Any]], list[Path]]]],
) -> None:
    folder = OUT / FOLDERS[info.number]
    stem = STEMS[info.number]
    if info.recommended_variant not in builders:
        raise RuntimeError(f"recommended variant missing for Fig. {info.number}")

    combined_rows: list[dict[str, Any]] = []
    all_sources: list[Path] = []
    variants: dict[str, dict[str, Any]] = {}
    for variant, builder in builders.items():
        fig, rows, sources = builder()
        base = folder / f"{stem}_{variant}"
        save_formats(fig, base)
        plt.close(fig)
        if not combined_rows:
            combined_rows = rows
        all_sources.extend(sources)
        variants[variant] = {
            "png": rel(base.with_suffix(".png")),
            "pdf": rel(base.with_suffix(".pdf")),
            "svg": rel(base.with_suffix(".svg")),
            "final_width_mm": SINGLE_MM if variant == "single" else DOUBLE_MM,
        }

    recommended_base = folder / f"{stem}_{info.recommended_variant}"
    main_base = folder / stem
    for suffix in (".png", ".pdf", ".svg"):
        shutil.copy2(recommended_base.with_suffix(suffix), main_base.with_suffix(suffix))

    source_csv = folder / f"{stem}_source_data.csv"
    write_csv(source_csv, combined_rows)
    local_caption = folder / f"{stem}_caption.txt"
    global_caption = OUT / "captions" / f"Fig{info.number:02d}_caption.txt"
    local_caption.write_text(info.caption.strip() + "\n", encoding="utf-8")
    shutil.copy2(local_caption, global_caption)
    command_path = folder / f"{stem}_generation_command.txt"
    command_path.write_text(GENERATION_COMMAND + "\n", encoding="utf-8")

    with Image.open(main_base.with_suffix(".png")) as image:
        main_png = {"pixels": list(image.size), "dpi": list(image.info.get("dpi", (0, 0)))}
    metadata = {
        "schema_version": "jkros_final_figure_v1",
        "figure_number": info.number,
        "figure_stem": stem,
        "scientific_question": info.question,
        "sample_count": info.sample_count,
        "metric_definition": info.metric_definition,
        "primary_numerical_result": info.result,
        "recommended_paper_location": info.paper_location,
        "recommended_variant": info.recommended_variant,
        "main_files": {
            "png": rel(main_base.with_suffix(".png")),
            "pdf": rel(main_base.with_suffix(".pdf")),
            "svg": rel(main_base.with_suffix(".svg")),
        },
        "publication_variants": variants,
        "source_data_csv": rel(source_csv),
        "source_artifacts": source_records(all_sources),
        "caption_path": rel(global_caption),
        "generation_command": GENERATION_COMMAND,
        "gpu_render_used": info.gpu_render_used,
        "scientific_results_modified": False,
        "experiment3_used": False,
        "main_png_properties": main_png,
        "bootstrap": {
            "resamples": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "paired_source_episode_resampling": True,
        } if info.number in (2, 3, 5) else None,
    }
    if info.number == 6:
        render_report = json.loads(RENDER_REPORT.read_text(encoding="utf-8"))
        metadata["render_generation_command"] = render_report["generation_command"]
        metadata["gpu_render_wall_time_s"] = render_report["gpu_render_wall_time_s"]
        metadata["render_report"] = rel(RENDER_REPORT)
    write_json(folder / f"{stem}_metadata.json", metadata)
    FIGURE_METADATA.append({**metadata, "caption": info.caption})


def paired_weighted_bootstrap_cis(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    seed_offset: int,
) -> dict[str, tuple[float, float]]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(a) != len(b) or len(a) != len(weights):
        raise ValueError("paired bootstrap arrays differ in length")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    boot_a = np.empty(BOOTSTRAP_N, dtype=float)
    boot_b = np.empty(BOOTSTRAP_N, dtype=float)
    for start in range(0, BOOTSTRAP_N, 1000):
        stop = min(BOOTSTRAP_N, start + 1000)
        idx = rng.integers(0, len(a), size=(stop - start, len(a)))
        selected_weights = weights[idx]
        denominator = selected_weights.sum(axis=1)
        boot_a[start:stop] = (a[idx] * selected_weights).sum(axis=1) / denominator
        boot_b[start:stop] = (b[idx] * selected_weights).sum(axis=1) / denominator
    return {
        "A": tuple(float(x) for x in np.percentile(boot_a, [2.5, 97.5])),
        "B": tuple(float(x) for x in np.percentile(boot_b, [2.5, 97.5])),
    }


def jitter(n: int, width: float = 0.075) -> np.ndarray:
    return np.linspace(-width, width, n)


def build_method_overview(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    width = width_inches(variant)
    height = 4.75 if variant == "single" else 2.85
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, edge: str, fill: str, fs: float = 6.7) -> None:
        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=0.85,
                transform=ax.transAxes,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, transform=ax.transAxes)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": MID, "lw": 0.75, "mutation_scale": 7},
        )

    if variant == "double":
        box(0.37, 0.89, 0.26, 0.075, "ALOHA demonstrations (n = 50)", INK, "#F5F5F5", 7.0)
        box(0.035, 0.66, 0.41, 0.145, f"{A_NAME}\n6-D wrist transfer", A_COLOR, A_LIGHT, 6.8)
        box(0.555, 0.66, 0.41, 0.145, f"{B_NAME}\ninteraction frame · whole-hand geometry\nbimanual relation · ownership transition", B_COLOR, B_LIGHT, 6.25)
        ax.add_patch(patches.Rectangle((0.035, 0.46), 0.93, 0.105, facecolor="#F3F3F3", edgecolor=INK, linewidth=0.85, transform=ax.transAxes))
        ax.text(0.5, 0.525, "SHARED G1 REALIZATION", ha="center", va="center", fontsize=6.7, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.485, "temporal IK  ·  generic feasibility", ha="center", va="center", fontsize=6.5, transform=ax.transAxes)
        ax.plot([0.5, 0.5], [0.46, 0.565], color="#B0B0B0", lw=0.55, ls=(0, (2, 2)), transform=ax.transAxes)
        box(0.12, 0.315, 0.25, 0.075, "G1 Dataset A", A_COLOR, A_LIGHT)
        box(0.63, 0.315, 0.25, 0.075, "G1 Dataset B", B_COLOR, B_LIGHT)
        box(0.16, 0.19, 0.17, 0.068, "ACT-A", A_COLOR, A_LIGHT)
        box(0.67, 0.19, 0.17, 0.068, "ACT-B", B_COLOR, B_LIGHT)
        box(0.25, 0.025, 0.50, 0.095, "COMMON EVALUATION\nwrist fidelity · whole-hand interaction · bimanual relation · held-out prediction", INK, "#F5F5F5", 6.15)
        arrow(0.5, 0.89, 0.24, 0.805); arrow(0.5, 0.89, 0.76, 0.805)
        arrow(0.24, 0.66, 0.24, 0.565); arrow(0.76, 0.66, 0.76, 0.565)
        arrow(0.24, 0.46, 0.245, 0.39); arrow(0.76, 0.46, 0.755, 0.39)
        arrow(0.245, 0.315, 0.245, 0.258); arrow(0.755, 0.315, 0.755, 0.258)
        arrow(0.245, 0.19, 0.40, 0.12); arrow(0.755, 0.19, 0.60, 0.12)
    else:
        box(0.14, 0.91, 0.72, 0.055, "ALOHA demonstrations (n = 50)", INK, "#F5F5F5", 6.8)
        box(0.025, 0.70, 0.455, 0.13, f"{A_NAME}\n6-D wrist transfer", A_COLOR, A_LIGHT, 6.25)
        box(0.52, 0.70, 0.455, 0.13, f"{B_NAME}\ninteraction frame · whole-hand\nbimanual relation\nownership transition", B_COLOR, B_LIGHT, 5.25)
        ax.add_patch(patches.Rectangle((0.025, 0.53), 0.95, 0.095, facecolor="#F3F3F3", edgecolor=INK, linewidth=0.85, transform=ax.transAxes))
        ax.text(0.5, 0.587, "SHARED G1 REALIZATION", ha="center", va="center", fontsize=6.5, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.552, "temporal IK · generic feasibility", ha="center", va="center", fontsize=6.1, transform=ax.transAxes)
        box(0.06, 0.395, 0.37, 0.065, "G1 Dataset A", A_COLOR, A_LIGHT, 6.3)
        box(0.57, 0.395, 0.37, 0.065, "G1 Dataset B", B_COLOR, B_LIGHT, 6.3)
        box(0.12, 0.275, 0.25, 0.062, "ACT-A", A_COLOR, A_LIGHT, 6.3)
        box(0.63, 0.275, 0.25, 0.062, "ACT-B", B_COLOR, B_LIGHT, 6.3)
        box(0.08, 0.065, 0.84, 0.115, "COMMON EVALUATION\nwrist fidelity · whole-hand interaction\nbimanual relation · held-out prediction", INK, "#F5F5F5", 5.9)
        arrow(0.5, 0.91, 0.25, 0.83); arrow(0.5, 0.91, 0.75, 0.83)
        arrow(0.25, 0.70, 0.25, 0.625); arrow(0.75, 0.70, 0.75, 0.625)
        arrow(0.25, 0.53, 0.245, 0.46); arrow(0.75, 0.53, 0.755, 0.46)
        arrow(0.245, 0.395, 0.245, 0.337); arrow(0.755, 0.395, 0.755, 0.337)
        arrow(0.245, 0.275, 0.40, 0.18); arrow(0.755, 0.275, 0.60, 0.18)

    rows = [
        {"order": 1, "branch": "shared", "stage": "source", "description": "ALOHA demonstrations", "episodes": 50},
        {"order": 2, "branch": "A", "stage": "representation", "description": "6-D wrist transfer"},
        {"order": 2, "branch": "B", "stage": "representation", "description": "interaction frame; whole-hand geometry; bimanual relation; ownership transition"},
        {"order": 3, "branch": "shared", "stage": "G1 realization", "description": "temporal IK; generic feasibility"},
        {"order": 4, "branch": "A", "stage": "dataset", "description": "G1 Dataset A"},
        {"order": 4, "branch": "B", "stage": "dataset", "description": "G1 Dataset B"},
        {"order": 5, "branch": "A", "stage": "policy", "description": "ACT-A"},
        {"order": 5, "branch": "B", "stage": "policy", "description": "ACT-B"},
        {"order": 6, "branch": "shared", "stage": "evaluation", "description": "wrist fidelity; whole-hand interaction; bimanual relation; held-out prediction"},
    ]
    return fig, rows, [
        frozen.SOURCE_ARTIFACTS["retargeting_table"],
        frozen.SOURCE_ARTIFACTS["experiment2"],
        frozen.SOURCE_ARTIFACTS["a_manifest"],
        frozen.SOURCE_ARTIFACTS["b_manifest"],
    ]


def build_tradeoff(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    per = ctx["per_episode"]
    frame = {"A": ctx["a_frame"], "B": ctx["b_frame"]}
    definitions = (
        ("wrist", "Wrist trajectory error"),
        ("whole_hand", "Whole-hand interaction error"),
        ("bimanual", "Bimanual relation error"),
    )
    if variant == "double":
        fig, axes = plt.subplots(1, 3, figsize=(width_inches(variant), 2.42))
    else:
        fig, axes = plt.subplots(3, 1, figsize=(width_inches(variant), 5.75))
    weights = np.asarray([row["frame_count"] for row in per], dtype=float)
    rows: list[dict[str, Any]] = []
    for index, (ax, (metric, title)) in enumerate(zip(np.ravel(axes), definitions)):
        values = {
            method: np.asarray([row[f"{method.lower()}_{metric}_mean_mm"] for row in per], dtype=float)
            for method in ("A", "B")
        }
        cis = paired_weighted_bootstrap_cis(values["A"], values["B"], weights, 10 + index)
        means = {method: float(np.mean(frame[method][metric]) * 1000.0) for method in ("A", "B")}
        for method, x, color, marker in (("A", 0, A_COLOR, "o"), ("B", 1, B_COLOR, "s")):
            ax.scatter(
                x + jitter(50), values[method], s=9.0, marker=marker, facecolor="white",
                edgecolor=color, linewidth=0.42, alpha=0.58, zorder=2,
            )
            low, high = cis[method]
            ax.errorbar(
                x, means[method], yerr=[[means[method] - low], [high - means[method]]],
                fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK,
                markeredgewidth=0.45, markersize=6.0, capsize=2.5, elinewidth=1.25, zorder=4,
            )
            rows.extend(
                {
                    "record_type": "episode_mean",
                    "metric": title,
                    "episode_index": row["episode_index"],
                    "stable_episode_id": row["stable_episode_id"],
                    "method": method,
                    "episode_mean_mm": float(value),
                }
                for row, value in zip(per, values[method])
            )
            rows.append(
                {
                    "record_type": "frame_weighted_summary",
                    "metric": title,
                    "method": method,
                    "mean_mm": means[method],
                    "paired_cluster_bootstrap_ci95_low_mm": low,
                    "paired_cluster_bootstrap_ci95_high_mm": high,
                    "episodes": 50,
                    "bootstrap_resamples": BOOTSTRAP_N,
                    "seed": BOOTSTRAP_SEED + 10 + index,
                }
            )
        ax.set_xticks([0, 1], ["A", "B"])
        ax.set_xlim(-0.30, 1.30)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Error [mm]")
        ax.set_title(title, pad=4)
        ax.text(0.5, 0.97, f"{means['A']:.1f} vs {means['B']:.1f} mm", transform=ax.transAxes, ha="center", va="top", fontsize=6.3)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + index), x=-0.18 if variant == "double" else -0.13)
    fig.legend(handles=method_handles(), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    if variant == "double":
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.79, wspace=0.38)
    else:
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.065, top=0.92, hspace=0.60)
    return fig, rows, [
        frozen.SOURCE_ARTIFACTS["a_manifest"],
        frozen.SOURCE_ARTIFACTS["b_manifest"],
        frozen.SOURCE_ARTIFACTS["retargeting_table"],
    ]


def load_forest_effects(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(FOREST_SOURCE)
    parsed = [
        {
            "metric": row["metric"],
            "paired_B_minus_A_mean_mm": float(row["paired_B_minus_A_mean_mm"]),
            "ci95_low_mm": float(row["ci95_low_mm"]),
            "ci95_high_mm": float(row["ci95_high_mm"]),
            "episodes": int(row["episodes"]),
            "bootstrap_resamples": int(row["bootstrap_resamples"]),
            "seed": int(row["seed"]),
            "direction_note": row["direction_note"],
        }
        for row in rows
    ]
    if len(parsed) != 4:
        raise RuntimeError("stored paired-effect source must contain four metrics")
    for stored, verified in zip(parsed, ctx["effects"]):
        if stored["metric"] != verified["metric"] or not np.allclose(
            [stored["paired_B_minus_A_mean_mm"], stored["ci95_low_mm"], stored["ci95_high_mm"]],
            [verified["paired_B_minus_A_mean_mm"], verified["ci95_low_mm"], verified["ci95_high_mm"]],
            rtol=0.0,
            atol=5e-12,
        ):
            raise RuntimeError(f"stored paired effect differs from frozen-array verification: {stored['metric']}")
    return parsed


def build_forest(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    effects = load_forest_effects(ctx)
    height = 2.68 if variant == "single" else 2.32
    fig, ax = plt.subplots(figsize=(width_inches(variant), height))
    y = np.arange(4)[::-1]
    for yy, row in zip(y, effects):
        mean = row["paired_B_minus_A_mean_mm"]
        low = row["ci95_low_mm"]
        high = row["ci95_high_mm"]
        ax.errorbar(
            mean, yy, xerr=[[mean - low], [high - mean]], fmt="D", color=INK,
            markerfacecolor="white", markeredgewidth=0.85, markersize=4.6,
            elinewidth=1.25, capsize=2.4, zorder=3,
        )
        # Place labels toward the zero line so the compact single-column
        # variant cannot clip the two large effects at either axis boundary.
        offset = -6 if mean >= 0 else 6
        ax.annotate(
            f"{mean:+.1f} [{low:+.1f}, {high:+.1f}]", xy=(mean, yy),
            xytext=(offset, 6), textcoords="offset points",
            ha="right" if mean >= 0 else "left", va="bottom", fontsize=5.8,
        )
    ax.axvline(0, color=MID, linewidth=0.75)
    ax.set_xlim(-88, 88)
    ax.set_ylim(-0.7, 3.55)
    ax.set_yticks(y, ["Wrist", "Whole-hand", "Bimanual relation", "Projection magnitude"])
    ax.set_xlabel("Paired difference, B − A [mm]")
    if variant == "single":
        ax.text(0.5, -0.27, "B − A < 0: lower error for B", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=MID)
        ax.text(0.5, -0.35, "B − A > 0: lower error for A", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=MID)
    else:
        ax.text(0.5, -0.27, "B − A < 0: lower error for B     B − A > 0: lower error for A", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=MID)
    clean_axis(ax, "x")
    fig.subplots_adjust(left=0.32 if variant == "single" else 0.19, right=0.975, bottom=0.37 if variant == "single" else 0.30, top=0.92)
    return fig, effects, [
        FOREST_SOURCE,
        frozen.SOURCE_ARTIFACTS["a_manifest"],
        frozen.SOURCE_ARTIFACTS["b_manifest"],
    ]


def build_feasibility(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    table = ctx["data"]["table1"]
    a_counts = tuple(int(value.strip()) for value in frozen.lookup(table, "Clean / warning / hard", "FAIR A").split("/"))
    b_counts = tuple(int(value.strip()) for value in frozen.lookup(table, "Clean / warning / hard", "PROPOSED B").split("/"))
    counts = {"A": dict(zip(("CLEAN", "WARNING", "HARD"), a_counts)), "B": dict(zip(("CLEAN", "WARNING", "HARD"), b_counts))}
    if counts != {"A": {"CLEAN": 27, "WARNING": 21, "HARD": 2}, "B": {"CLEAN": 13, "WARNING": 37, "HARD": 0}}:
        raise RuntimeError(f"frozen feasibility outcome discrepancy: {counts}")
    failure_metrics = (
        ("Hard collision", "Hard collision episodes"),
        ("Hard IK", "Hard IK episodes"),
        ("Joint limit", "Joint-limit failures"),
        ("Branch", "Branch failures"),
    )
    failure_counts = {
        method: [int(frozen.lookup(table, key, "FAIR A" if method == "A" else "PROPOSED B")) for _, key in failure_metrics]
        for method in ("A", "B")
    }
    if variant == "double":
        fig, axes = plt.subplots(1, 2, figsize=(width_inches(variant), 2.32), gridspec_kw={"width_ratios": [1.30, 1.0]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(width_inches(variant), 3.92), gridspec_kw={"height_ratios": [1.0, 1.15]})
    axes = np.ravel(axes)
    ax = axes[0]
    left = np.zeros(2)
    for status, color, hatch in (("CLEAN", GREEN, "//"), ("WARNING", AMBER, ".."), ("HARD", RED, "xx")):
        values = [counts["A"][status], counts["B"][status]]
        bars = ax.barh([0, 1], values, left=left, height=0.54, color=color, edgecolor=INK, linewidth=0.55, hatch=hatch, label=status)
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", fontsize=6.8, color="white" if status == "HARD" else INK)
        left += np.asarray(values)
    ax.set_yticks([0, 1], [A_NAME, B_NAME])
    ax.invert_yaxis()
    ax.set_xlim(0, 50)
    ax.set_xlabel("Episodes")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), handlelength=1.5)
    clean_axis(ax, "x")
    panel_label(ax, "a", x=-0.16)

    ax = axes[1]
    labels = [label for label, _ in failure_metrics]
    y = np.arange(4)[::-1]
    for yy, a, b in zip(y, failure_counts["A"], failure_counts["B"]):
        ax.plot([a, b], [yy + 0.09, yy - 0.09], color="#B9B9B9", lw=0.65, zorder=1)
        ax.scatter(a, yy + 0.09, marker="o", facecolor="white", edgecolor=A_COLOR, linewidth=0.9, zorder=3)
        ax.scatter(b, yy - 0.09, marker="s", facecolor="white", edgecolor=B_COLOR, linewidth=0.9, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.15, 2.35)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Hard-failure episodes")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=A_COLOR, markerfacecolor="white", linestyle="none", label="A"),
            Line2D([0], [0], marker="s", color=B_COLOR, markerfacecolor="white", linestyle="none", label="B"),
        ],
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18),
    )
    clean_axis(ax, "x")
    panel_label(ax, "b", x=-0.14)
    if variant == "double":
        fig.text(0.5, 0.012, "WARNING ≠ HARD_FAIL", ha="center", va="bottom", fontsize=6.3, color=MID)
        fig.subplots_adjust(left=0.205, right=0.985, bottom=0.20, top=0.80, wspace=0.47)
    else:
        fig.text(0.5, 0.025, "WARNING ≠ HARD_FAIL", ha="center", va="bottom", fontsize=6.1, color=MID)
        fig.subplots_adjust(left=0.34, right=0.975, bottom=0.20, top=0.89, hspace=0.70)
    rows = [
        {"record_type": "outcome", "method": method, **values, "episodes": 50}
        for method, values in counts.items()
    ]
    rows.extend(
        {"record_type": "hard_failure_mode", "method": method, "failure_mode": label, "count": count, "episodes": 50}
        for method in ("A", "B")
        for label, count in zip(labels, failure_counts[method])
    )
    return fig, rows, [
        frozen.SOURCE_ARTIFACTS["retargeting_table"],
        frozen.SOURCE_ARTIFACTS["a_per_episode"],
        frozen.SOURCE_ARTIFACTS["b_manifest"],
    ]


def heldout_episode_values(ctx: dict[str, Any], method: str, metric: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    probes = ctx["data"]["exp2"]["methods"][method]["common_source_geometry"]["per_probe"]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        grouped[int(probe["final_episode"])].append(probe)
    episodes = sorted(grouped)
    if episodes != sorted(ctx["heldout"]) or any(len(grouped[episode]) != 9 for episode in episodes):
        raise RuntimeError("HELDOUT8 probe-to-episode grouping changed")
    values, weights = [], []
    for episode in episodes:
        episode_weights = np.asarray([int(probe["valid_frames"]) for probe in grouped[episode]], dtype=float)
        episode_values = np.asarray([float(probe[metric]["mean"]) for probe in grouped[episode]], dtype=float)
        values.append(float(np.average(episode_values, weights=episode_weights)))
        weights.append(float(episode_weights.sum()))
    return episodes, np.asarray(values), np.asarray(weights)


def build_supervision_policy(ctx: dict[str, Any], variant: str) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    definitions = (
        ("wrist", "wrist_error_mm", "Wrist", "predicted source wrist error mean"),
        ("whole_hand", "whole_hand_error_mm", "Whole-hand", "predicted whole-hand error mean"),
        ("bimanual", "bimanual_relation_error_mm", "Bimanual", "predicted bimanual relation error mean"),
    )
    if variant == "double":
        fig, axes = plt.subplots(2, 3, figsize=(width_inches(variant), 3.15), sharey="col")
    else:
        fig, axes = plt.subplots(3, 2, figsize=(width_inches(variant), 6.05), sharey="row")
    per = ctx["per_episode"]
    top_weights = np.asarray([row["frame_count"] for row in per], dtype=float)
    rows: list[dict[str, Any]] = []
    for column, (metric, probe_metric, title, table_metric) in enumerate(definitions):
        top_ax = axes[0, column] if variant == "double" else axes[column, 0]
        top_values = {
            method: np.asarray([row[f"{method.lower()}_{metric}_mean_mm"] for row in per], dtype=float)
            for method in ("A", "B")
        }
        top_cis = paired_weighted_bootstrap_cis(top_values["A"], top_values["B"], top_weights, 100 + column)
        top_means = {
            "A": float(ctx["a_frame"][metric].mean() * 1000.0),
            "B": float(ctx["b_frame"][metric].mean() * 1000.0),
        }
        for method, x, color, marker in (("A", 0, A_COLOR, "o"), ("B", 1, B_COLOR, "s")):
            top_ax.scatter(x + jitter(50, 0.068), top_values[method], s=6.2, marker=marker, facecolor="white", edgecolor=color, linewidth=0.34, alpha=0.48)
            low, high = top_cis[method]
            top_ax.errorbar(x, top_means[method], yerr=[[top_means[method] - low], [high - top_means[method]]], fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.4, markersize=5.2, capsize=2.0, elinewidth=1.05, zorder=4)
            rows.extend(
                {"level": "Retargeted supervision", "record_type": "episode_mean", "metric": title, "method": method, "episode_index": row["episode_index"], "sample_mean_mm": float(value)}
                for row, value in zip(per, top_values[method])
            )
            rows.append({"level": "Retargeted supervision", "record_type": "frame_weighted_summary", "metric": title, "method": method, "mean_mm": top_means[method], "paired_cluster_bootstrap_ci95_low_mm": low, "paired_cluster_bootstrap_ci95_high_mm": high, "episodes": 50, "resamples": BOOTSTRAP_N})
        top_ax.set_xticks([0, 1], ["A", "B"])
        top_ax.set_xlim(-0.30, 1.30)
        top_ax.set_ylim(bottom=0)
        top_ax.set_title(title, pad=4 if variant == "double" else 3)
        if variant == "double" and column == 0:
            top_ax.set_ylabel("Mean error [mm]")
        elif variant == "single":
            top_ax.set_ylabel("Mean error [mm]")
        clean_axis(top_ax)
        panel_label(top_ax, chr(ord("a") + column), x=-0.22 if variant == "double" else -0.25)

        bottom_ax = axes[1, column] if variant == "double" else axes[column, 1]
        bottom_samples: dict[str, tuple[list[int], np.ndarray, np.ndarray]] = {
            method: heldout_episode_values(ctx, method.lower(), probe_metric) for method in ("A", "B")
        }
        if bottom_samples["A"][0] != bottom_samples["B"][0]:
            raise RuntimeError("ACT-A/ACT-B held-out episode identities differ")
        bottom_cis = paired_weighted_bootstrap_cis(
            bottom_samples["A"][1], bottom_samples["B"][1], bottom_samples["A"][2], 200 + column
        )
        bottom_means = {
            "A": pubready.t2_value(ctx["data"], table_metric, "ACT-A40"),
            "B": pubready.t2_value(ctx["data"], table_metric, "ACT-B40"),
        }
        for method, x, color, marker, label in (("A", 0, A_COLOR, "o", "ACT-A"), ("B", 1, B_COLOR, "s", "ACT-B")):
            episodes, values, weights = bottom_samples[method]
            aggregate = float(np.average(values, weights=weights))
            if not np.isclose(aggregate, bottom_means[method], rtol=0.0, atol=5e-4):
                raise RuntimeError(f"held-out episode aggregation differs from frozen table: {title}/{method}")
            bottom_ax.scatter(x + jitter(8, 0.055), values, s=10.0, marker=marker, facecolor="white", edgecolor=color, linewidth=0.5, alpha=0.70)
            low, high = bottom_cis[method]
            bottom_ax.errorbar(x, bottom_means[method], yerr=[[bottom_means[method] - low], [high - bottom_means[method]]], fmt=marker, color=color, markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.4, markersize=5.2, capsize=2.0, elinewidth=1.05, zorder=4)
            rows.extend(
                {"level": "Held-out ACT prediction", "record_type": "heldout_episode_mean", "metric": title, "method": label, "episode_index": episode, "sample_mean_mm": float(value), "probe_count": 9, "valid_predicted_frames": int(weight)}
                for episode, value, weight in zip(episodes, values, weights)
            )
            rows.append({"level": "Held-out ACT prediction", "record_type": "frame_weighted_summary", "metric": title, "method": label, "mean_mm": bottom_means[method], "paired_cluster_bootstrap_ci95_low_mm": low, "paired_cluster_bootstrap_ci95_high_mm": high, "heldout_episodes": 8, "probes_per_episode": 9, "resamples": BOOTSTRAP_N})
        bottom_ax.set_xticks([0, 1], ["ACT-A", "ACT-B"])
        bottom_ax.set_xlim(-0.30, 1.30)
        if variant == "double" and column == 0:
            bottom_ax.set_ylabel("Mean error [mm]")
        elif variant == "single":
            bottom_ax.set_ylabel("Mean error [mm]")
        clean_axis(bottom_ax)
        panel_label(bottom_ax, chr(ord("d") + column), x=-0.22 if variant == "double" else -0.25)

        upper = max(
            float(np.max(top_values["A"])), float(np.max(top_values["B"])),
            top_cis["A"][1], top_cis["B"][1],
            float(np.max(bottom_samples["A"][1])), float(np.max(bottom_samples["B"][1])),
            bottom_cis["A"][1], bottom_cis["B"][1],
        ) * 1.08
        top_ax.set_ylim(0, upper)
        bottom_ax.set_ylim(0, upper)

    if variant == "double":
        fig.text(0.030, 0.69, "RETARGETED\nSUPERVISION\n(n = 50)", rotation=90, ha="center", va="center", fontsize=6.0, fontweight="bold", color=MID)
        fig.text(0.030, 0.27, "HELD-OUT ACT\nPREDICTION\n(n = 8)", rotation=90, ha="center", va="center", fontsize=6.0, fontweight="bold", color=MID)
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.82, hspace=0.47, wspace=0.33)
    else:
        axes[0, 0].text(0.5, 1.18, "RETARGETED\nSUPERVISION\n(n = 50)", transform=axes[0, 0].transAxes, ha="center", va="bottom", fontsize=5.7, fontweight="bold", color=MID)
        axes[0, 1].text(0.5, 1.18, "HELD-OUT ACT\nPREDICTION\n(n = 8)", transform=axes[0, 1].transAxes, ha="center", va="bottom", fontsize=5.7, fontweight="bold", color=MID)
        fig.subplots_adjust(left=0.24, right=0.98, bottom=0.055, top=0.89, hspace=0.58, wspace=0.42)
    fig.legend(handles=method_handles(policy_combined=True), frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    return fig, rows, [
        frozen.SOURCE_ARTIFACTS["retargeting_table"],
        frozen.SOURCE_ARTIFACTS["policy_table"],
        frozen.SOURCE_ARTIFACTS["experiment2"],
        frozen.SOURCE_ARTIFACTS["heldout8"],
        frozen.SOURCE_ARTIFACTS["a_manifest"],
        frozen.SOURCE_ARTIFACTS["b_manifest"],
    ]


def build_representative_motion(ctx: dict[str, Any]) -> tuple[plt.Figure, list[dict[str, Any]], list[Path]]:
    if ctx["representative_episode"] != 23:
        raise RuntimeError("predeclared representative episode changed")
    if not RENDER_REPORT.is_file():
        raise FileNotFoundError(f"matched render report unavailable: {RENDER_REPORT}")
    report = json.loads(RENDER_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "MATCHED_STILLS_COMPLETE" or int(report.get("episode_index", -1)) != 23:
        raise RuntimeError("matched episode-23 still render is incomplete")
    if report.get("videos_written") or report.get("training_or_policy_inference_run"):
        raise RuntimeError("Fig. 6 render violated still-only scientific scope")
    if not report["object_visualization"]["identical_pose_for_A_and_B_at_each_source_frame"]:
        raise RuntimeError("A/B object visualization is not matched")

    a_path = Path(report["trajectory_sources"]["A"]["path"])
    b_path = Path(report["trajectory_sources"]["B"]["path"])
    if sha256(a_path) != report["trajectory_sources"]["A"]["sha256"] or sha256(b_path) != report["trajectory_sources"]["B"]["sha256"]:
        raise RuntimeError("Fig. 6 trajectory hash mismatch after rendering")
    semantic = report["semantic_states"]
    expected = [("LEFT_OWNED", 182), ("DUAL_CONTACT", 299), ("RIGHT_OWNED", 320), ("release", 402)]
    if [(row["snapshot_key"], int(row["source_frame"])) for row in semantic] != expected:
        raise RuntimeError("Fig. 6 semantic-frame contract changed")

    source_by_key = report["source_assets"]
    render_by = {(row["method"], row["snapshot_key"]): row for row in report["rendered_assets"]}
    image_paths: list[Path] = []
    for row in semantic:
        key = row["snapshot_key"]
        image_paths.append(Path(source_by_key[key]["output_path"]))
        for method in ("A", "B"):
            image_paths.append(Path(render_by[(method, key)]["output_path"]))
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    fig = plt.figure(figsize=(width_inches("double"), 8.05))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.42], hspace=0.18)
    top = outer[0].subgridspec(1, 2, wspace=0.31)
    ax_path = fig.add_subplot(top[0, 0])
    ax_bimanual = fig.add_subplot(top[0, 1])
    rows: list[dict[str, Any]] = []
    event_frames = [int(row["source_frame"]) for row in semantic]

    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        if len(a["timestamp"]) != len(b["timestamp"]) or len(a["timestamp"]) != 693:
            raise RuntimeError("representative A/B trajectory length changed")
        time_s = np.asarray(a["timestamp"], dtype=float)
        method_specs = (
            ("Source reference", INK, ":"),
            ("Trajectory-Centric A", A_COLOR, "-"),
            ("Interaction-Centric B", B_COLOR, "--"),
        )
        for side, hand_marker in (("left", "o"), ("right", "^")):
            values_by_method = {
                "Source reference": np.asarray(a[f"target_{side}_interaction_frame_position_world"], dtype=float) * 1000.0,
                "Trajectory-Centric A": np.asarray(a[f"achieved_{side}_physical_grasp_frame_position_world"], dtype=float) * 1000.0,
                "Interaction-Centric B": np.asarray(b[f"achieved_{side}_physical_grasp_frame_position_world"], dtype=float) * 1000.0,
            }
            for method, color, line_style in method_specs:
                values = values_by_method[method]
                ax_path.plot(values[:, 0], values[:, 1], color=color, linestyle=line_style, linewidth=1.05 if side == "left" else 0.82, alpha=1.0 if side == "left" else 0.74)
                ax_path.scatter(values[event_frames, 0], values[event_frames, 1], s=10, marker=hand_marker, facecolor="white", edgecolor=color, linewidth=0.55, zorder=3)
                rows.extend(
                    {"panel": "whole_hand_top_view", "episode_index": 23, "frame": frame, "time_s": float(time_s[frame]), "side": side, "method": method, "world_x_mm": float(value[0]), "world_y_mm": float(value[1])}
                    for frame, value in enumerate(values)
                )
        ax_path.set_aspect("equal", adjustable="datalim")
        ax_path.set_xlabel("World x [mm]")
        ax_path.set_ylabel("World y [mm]")
        ax_path.set_title("Whole-hand paths (top view)", pad=4)
        clean_axis(ax_path, "both")
        panel_label(ax_path, "a", x=-0.14)

        target_l = np.asarray(a["target_left_interaction_frame_position_world"], dtype=float)
        target_r = np.asarray(a["target_right_interaction_frame_position_world"], dtype=float)
        a_l = np.asarray(a["achieved_left_physical_grasp_frame_position_world"], dtype=float)
        a_r = np.asarray(a["achieved_right_physical_grasp_frame_position_world"], dtype=float)
        b_l = np.asarray(b["achieved_left_physical_grasp_frame_position_world"], dtype=float)
        b_r = np.asarray(b["achieved_right_physical_grasp_frame_position_world"], dtype=float)
        curves = {
            "Source reference": np.linalg.norm(target_r - target_l, axis=1) * 1000.0,
            "Trajectory-Centric A": np.linalg.norm(a_r - a_l, axis=1) * 1000.0,
            "Interaction-Centric B": np.linalg.norm(b_r - b_l, axis=1) * 1000.0,
        }
        for method, color, line_style in method_specs:
            ax_bimanual.plot(time_s, curves[method], color=color, linestyle=line_style)
            rows.extend(
                {"panel": "bimanual_relative_displacement", "episode_index": 23, "frame": frame, "time_s": float(time_s[frame]), "method": method, "relative_displacement_mm": float(value)}
                for frame, value in enumerate(curves[method])
            )
        for number, state in enumerate(semantic, 1):
            frame = int(state["source_frame"])
            ax_bimanual.axvline(time_s[frame], color="#A7A7A7", linewidth=0.55, linestyle=(0, (2, 2)))
            ax_bimanual.text(time_s[frame], 0.98, str(number), transform=ax_bimanual.get_xaxis_transform(), ha="center", va="top", fontsize=5.8, color=MID, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35})
            rows.append({"panel": "semantic_event", "episode_index": 23, "event_number": number, "semantic_state": state["label"], "snapshot_key": state["snapshot_key"], "frame": frame, "time_s": float(time_s[frame])})
        ax_bimanual.text(0.02, 0.02, "1 grasp · 2 handoff · 3 right-owned · 4 release", transform=ax_bimanual.transAxes, ha="left", va="bottom", fontsize=5.2, color=MID)
        ax_bimanual.set_xlabel("Time [s]")
        ax_bimanual.set_ylabel("Right–left displacement [mm]")
        ax_bimanual.set_title("Bimanual relative displacement", pad=4)
        clean_axis(ax_bimanual)
        panel_label(ax_bimanual, "b", x=-0.14)

    method_legend = [
        Line2D([0], [0], color=INK, linestyle=":", label="Source reference"),
        Line2D([0], [0], color=A_COLOR, linestyle="-", label=A_NAME),
        Line2D([0], [0], color=B_COLOR, linestyle="--", label=B_NAME),
    ]
    fig.legend(handles=method_legend, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 0.993), columnspacing=1.5, handlelength=2.0)
    ax_path.text(0.02, 0.02, "○ left hand  ·  △ right hand", transform=ax_path.transAxes, ha="left", va="bottom", fontsize=5.4, color=MID, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})

    strip = outer[1].subgridspec(4, 3, hspace=0.035, wspace=0.025)
    columns = ("Source ALOHA", A_NAME, B_NAME)
    display_crop = (0, 50, 640, 450)
    strip_axes: list[list[plt.Axes]] = []
    for row_index, state in enumerate(semantic):
        key = state["snapshot_key"]
        assets = (
            Path(source_by_key[key]["output_path"]),
            Path(render_by[("A", key)]["output_path"]),
            Path(render_by[("B", key)]["output_path"]),
        )
        current: list[plt.Axes] = []
        for column_index, (title, path) in enumerate(zip(columns, assets)):
            ax = fig.add_subplot(strip[row_index, column_index])
            current.append(ax)
            image = np.asarray(Image.open(path).convert("RGB"))
            if image.shape != (480, 640, 3):
                raise RuntimeError(f"unexpected semantic still resolution: {path} {image.shape}")
            cropped = image[display_crop[1]:display_crop[3], display_crop[0]:display_crop[2]]
            ax.imshow(cropped)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.45); spine.set_color("#777777")
            if row_index == 0:
                ax.set_title(title, fontsize=7.0, pad=3)
            if column_index == 0:
                display_label = str(state["label"]).replace(" / ", " /\n")
                ax.text(-0.035, 0.5, f"{display_label}\nf = {int(state['source_frame'])}", transform=ax.transAxes, ha="right", va="center", fontsize=6.1)
            rows.append({"panel": "semantic_render_strip", "episode_index": 23, "semantic_state": state["label"], "snapshot_key": key, "frame": int(state["source_frame"]), "column": title, "asset_path": rel(path), "asset_sha256": sha256(path), "display_crop_left_px": display_crop[0], "display_crop_top_px": display_crop[1], "display_crop_right_px": display_crop[2], "display_crop_bottom_px": display_crop[3]})
        strip_axes.append(current)
    panel_label(strip_axes[0][0], "c", x=-0.24, y=1.08)
    fig.subplots_adjust(left=0.135, right=0.992, bottom=0.018, top=0.91)
    sources = [
        a_path,
        b_path,
        RENDER_REPORT,
        Path(report["camera_config_path"]),
        Path(report["object_visualization"]["plan_path"]),
        frozen.SOURCE_ARTIFACTS["common48"],
        *image_paths,
    ]
    return fig, rows, sources


def create_contact_sheet() -> Path:
    main_paths = [OUT / FOLDERS[number] / f"{STEMS[number]}.png" for number in range(1, 7)]
    if any(not path.is_file() for path in main_paths):
        raise FileNotFoundError("one or more of the six final PNGs is missing")
    width, height = 2200, 3300
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 48)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 31)
    except OSError:
        title_font = ImageFont.load_default(); label_font = ImageFont.load_default()
    draw.text((width // 2, 45), "Final JKROS paper figures", fill=INK, font=title_font, anchor="ma")
    margin_x, top, gap_x, gap_y = 70, 135, 55, 55
    cell_w = (width - 2 * margin_x - gap_x) // 2
    cell_h = (height - top - 80 - 2 * gap_y) // 3
    for index, path in enumerate(main_paths):
        row, column = divmod(index, 2)
        x0 = margin_x + column * (cell_w + gap_x)
        y0 = top + row * (cell_h + gap_y)
        label = f"Fig. {index + 1} — {STEMS[index + 1].replace('Fig%02d_' % (index + 1), '').replace('_', ' ')}"
        draw.text((x0 + cell_w // 2, y0 + 5), label, fill=INK, font=label_font, anchor="ma")
        with Image.open(path) as source:
            image = source.convert("RGB")
            available = (cell_w - 24, cell_h - 70)
            scale = min(available[0] / image.width, available[1] / image.height)
            resampling = getattr(Image, "Resampling", Image)
            resized = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                resampling.LANCZOS,
            )
        px = x0 + (cell_w - resized.width) // 2
        py = y0 + 58 + (cell_h - 70 - resized.height) // 2
        canvas.paste(resized, (px, py))
        draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline="#B8B8B8", width=2)
    path = OUT / "contact_sheet/FINAL_SIX_FIGURES_CONTACT_SHEET.png"
    canvas.save(path, dpi=(300, 300))
    return path


def write_manifest(contact_sheet: Path) -> None:
    lines = [
        "# Final Six-Figure Manifest",
        "",
        "This package contains exactly the six main figures selected for the JKROS paper. No Experiment-3, physical-execution, XR, D455, or real-G1 claim is shown.",
        "",
    ]
    for metadata in FIGURE_METADATA:
        number = metadata["figure_number"]
        lines.extend(
            [
                f"## Fig. {number}",
                "",
                f"- Path: `{metadata['main_files']['png']}`",
                f"- Source artifact(s): " + "; ".join(f"`{row['path']}`" for row in metadata["source_artifacts"]),
                f"- Sample count: {metadata['sample_count']}",
                f"- Scientific question: {metadata['scientific_question']}",
                f"- Primary numerical result: {metadata['primary_numerical_result']}",
                f"- Recommended paper location: {metadata['recommended_paper_location']}",
                f"- Column format: {metadata['recommended_variant']}-column",
                f"- Caption: {metadata['caption']}",
                f"- GPU render used: {'YES' if metadata['gpu_render_used'] else 'NO'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Package audit",
            "",
            f"- Contact sheet: `{rel(contact_sheet)}`",
            "- Main figure count: 6",
            "- Scientific result mutation: NO",
            "- Training/policy inference: NO",
            "- Isaac use: one bounded still-only render for Fig. 6",
            "- Experiment 3 used: NO",
            "- Physical task success claimed: NO",
            "",
        ]
    )
    (OUT / "manifest/FINAL_SIX_FIGURE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "manifest/final_six_figure_manifest.json", {"figures": FIGURE_METADATA, "contact_sheet": rel(contact_sheet), "main_figure_count": 6})
    captions = ["# Final figure captions", ""]
    for metadata in FIGURE_METADATA:
        captions.extend([f"## Fig. {metadata['figure_number']}", "", metadata["caption"], ""])
    (OUT / "captions/ALL_FINAL_CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    configure_style()
    ctx = pubready.main_context()
    if ctx["representative_episode"] != 23:
        raise RuntimeError("frozen representative selection no longer resolves to episode 23")

    figures = [
        (
            FigureInfo(
                1,
                "What exactly is compared?",
                "The A/B branches differ in representation but share temporal IK and generic G1 feasibility realization before matched dataset and ACT evaluation.",
                "50 source ALOHA demonstrations; HELDOUT8 for downstream prediction",
                "Controlled experimental pipeline; no scalar uncertainty.",
                "Methods",
                "double",
                "Fig. 1. Controlled comparison from 50 ALOHA demonstrations to G1 supervision and held-out ACT evaluation. Trajectory-Centric A transfers a 6-D wrist trajectory, whereas Interaction-Centric B represents interaction frames, whole-hand geometry, bimanual relation, and ownership transition. Both branches use the same temporal inverse-kinematics and generic-feasibility realization before producing G1 Dataset A/B and ACT-A/B. The common evaluation measures wrist fidelity, whole-hand interaction, bimanual relation, and held-out policy prediction; no physical-robot or closed-loop rollout result is included.",
            ),
            {"single": lambda: build_method_overview(ctx, "single"), "double": lambda: build_method_overview(ctx, "double")},
        ),
        (
            FigureInfo(
                2,
                "Does accurate wrist transfer preserve manipulation interaction?",
                "Mean A/B errors are 3.742/77.853 mm (wrist), 90.820/21.157 mm (whole hand), and 86.786/35.979 mm (bimanual).",
                "50 paired source demonstrations",
                "Per-episode mean Euclidean errors; filled symbols are frame-weighted means; 95% paired episode-cluster bootstrap intervals use 20,000 resamples.",
                "Results—primary retargeting result",
                "double",
                "Fig. 2. Retargeting fidelity over the same 50 source demonstrations. Small open symbols denote episode-level mean errors, large filled symbols denote frame-weighted means, and vertical bars denote 95% paired episode-cluster bootstrap intervals from 20,000 resamples. Lower is better in all panels. Trajectory-Centric A has lower wrist-trajectory error (3.742 versus 77.853 mm), whereas Interaction-Centric B has lower whole-hand interaction error (21.157 versus 90.820 mm) and bimanual relation error (35.979 versus 86.786 mm).",
            ),
            {"single": lambda: build_tradeoff(ctx, "single"), "double": lambda: build_tradeoff(ctx, "double")},
        ),
        (
            FigureInfo(
                3,
                "Is the A/B difference consistent over the 50 matched demonstrations?",
                "B − A is +74.115 mm for wrist, −69.678 mm for whole hand, −50.819 mm for bimanual relation, and −2.450 mm for projection.",
                "50 paired source demonstrations",
                "Mean of paired episode-level B-minus-A differences with stored 20,000-resample paired-bootstrap 95% confidence intervals.",
                "Results—paired statistical evidence",
                "single",
                "Fig. 3. Paired effect estimates for the same 50 source demonstrations. Points denote paired mean differences (Interaction-Centric B minus Trajectory-Centric A), and bars denote 95% bootstrap confidence intervals from 20,000 paired resamples. Because lower error is better, negative values indicate lower error for B and positive values indicate lower error for A. The effects are +74.115 mm for wrist error, −69.678 mm for whole-hand error, −50.819 mm for bimanual relation error, and −2.450 mm for feasibility projection magnitude. The intervals quantify paired uncertainty and are not presented as a significance test.",
            ),
            {"single": lambda: build_forest(ctx, "single"), "double": lambda: build_forest(ctx, "double")},
        ),
        (
            FigureInfo(
                4,
                "Can the retargeted trajectories be mechanically realized on G1?",
                "A has 27 CLEAN, 21 WARNING, and 2 HARD episodes; B has 13 CLEAN, 37 WARNING, and 0 HARD episodes. Both A hard failures are collisions.",
                "50 episodes per method",
                "Frozen episode outcome classification after the shared temporal IK and generic feasibility realization; WARNING is distinct from HARD_FAIL.",
                "Results—feasibility",
                "double",
                "Fig. 4. G1 feasibility outcomes after the shared temporal inverse-kinematics and generic-feasibility realization (50 episodes per method). (a) Trajectory-Centric A yields 27 CLEAN, 21 WARNING, and 2 HARD episodes; Interaction-Centric B yields 13 CLEAN, 37 WARNING, and 0 HARD episodes. WARNING is not a hard failure. (b) Both A hard failures are collision episodes; neither method has a hard IK, joint-limit, or branch failure. The CLEAN count is therefore not used as a standalone performance measure.",
            ),
            {"single": lambda: build_feasibility(ctx, "single"), "double": lambda: build_feasibility(ctx, "double")},
        ),
        (
            FigureInfo(
                5,
                "Does the representation characteristic survive policy learning?",
                "A/ACT-A retains lower wrist error, while B/ACT-B retains lower whole-hand and bimanual errors on the corresponding evaluation sets.",
                "Retargeting n=50; held-out ACT prediction n=8 episodes (9 probes per episode)",
                "Top: full-50 retargeting episode means. Bottom: per-held-out-episode aggregation of predicted-chunk geometry errors. Filled symbols are frame-weighted means; 95% paired episode-cluster bootstrap intervals use 20,000 resamples within each row.",
                "Results—downstream policy",
                "double",
                "Fig. 5. Representation-dependent geometry from retargeted supervision to learned-policy prediction. The top row shows full-50 retargeting results (n = 50 episodes), and the bottom row shows held-out ACT predictions aggregated to the eight held-out source episodes (nine phase probes per episode). Small open symbols denote episode means, large filled symbols denote frame-weighted means, and bars denote 95% paired episode-cluster bootstrap intervals from 20,000 resamples within each row. A/ACT-A retains lower wrist error (3.742/24.958 mm versus 77.853/95.269 mm), whereas B/ACT-B retains lower whole-hand error (21.157/58.330 mm versus 90.820/107.131 mm) and bimanual error (35.979/85.920 mm versus 86.786/120.976 mm). The two rows use different evaluation sets and are not paired before-and-after samples.",
            ),
            {"single": lambda: build_supervision_policy(ctx, "single"), "double": lambda: build_supervision_policy(ctx, "double")},
        ),
        (
            FigureInfo(
                6,
                "What does the A/B retargeting difference look like on the robot?",
                "Episode 23 is the frozen criterion-selected representative; semantic source frames are 182, 299, 320, and 402.",
                "One predeclared representative episode (693 frames); four matched semantic stills per column",
                "Top-view whole-hand paths and right-minus-left displacement in millimetres; matched stills use frozen qpos at the same source semantic frames. No summary marker or confidence interval is used.",
                "Results—representative qualitative geometry",
                "double",
                "Fig. 6. Criterion-selected representative retargeting example. Episode 23 (693 frames) is the source episode closest to the median Interaction-Centric-B whole-hand error within the common feasible set; no visual criterion was used. (a) Source, Trajectory-Centric-A, and Interaction-Centric-B whole-hand paths in the world top view; circles and triangles identify left and right hands. (b) Bimanual right-minus-left displacement in millimetres, with source-derived semantic events at frames 182 (left grasp), 299 (handoff/dual contact), 320 (right owned), and 402 (release). (c) Matched source RGB and Isaac G1 stills at those exact frames. The A/B renders use the exact frozen retargeted joint configurations, the same scene and camera, and an identical frozen phase-consistent object visualization. The strip is qualitative and does not demonstrate grasp physics, task success, policy rollout, or real-robot execution.",
                gpu_render_used=True,
            ),
            {"double": lambda: build_representative_motion(ctx)},
        ),
    ]

    for info, builders in figures:
        save_final_figure(info, builders)
    if len(FIGURE_METADATA) != 6:
        raise RuntimeError(f"final main-figure count must be six, not {len(FIGURE_METADATA)}")
    contact_sheet = create_contact_sheet()
    write_manifest(contact_sheet)
    print(json.dumps({
        "status": "JKROS_FINAL_SIX_FIGURES_READY",
        "figure_count": len(FIGURE_METADATA),
        "representative_episode": ctx["representative_episode"],
        "contact_sheet": str(contact_sheet),
        "output": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
