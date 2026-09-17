#!/usr/bin/env python3
"""Populate the JKROS Experiment-3 figure only from an approved frozen artifact.

Until a complete artifact is supplied, the publication package retains explicit
NA values.  This utility performs no inference and does not search for partial
rollout outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/paper_figure_bank_pubready"
A_NAME = "Trajectory-Centric A"
B_NAME = "Interaction-Centric B"
REQUIRED = ["Phase completion", "Semantic task success", "Whole-hand error", "Bimanual error", "HOA", "RPL", "SWPE"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("metrics", payload) if isinstance(payload, dict) else payload
        if isinstance(rows, dict):
            rows = [{"metric": metric, **values} for metric, values in rows.items()]
        return list(rows)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help="Approved frozen Experiment-3 CSV or JSON")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = load_rows(args.artifact)
    by_metric = {str(row["metric"]): row for row in rows}
    missing = [metric for metric in REQUIRED if metric not in by_metric]
    if missing:
        raise SystemExit(f"missing required metrics: {missing}")
    for metric in REQUIRED:
        for method in (A_NAME, B_NAME):
            value = str(by_metric[metric].get(method, "NA")).strip()
            if not value or value.upper() == "NA":
                raise SystemExit(f"approved value is unavailable: {metric} / {method}")

    output = args.output_root
    output.joinpath("source_data").mkdir(parents=True, exist_ok=True)
    fields = ["metric", A_NAME, B_NAME, "unit"]
    csv_path = output / "source_data/18_experiment3_POPULATED.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(by_metric[m] for m in REQUIRED)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.2, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(178 / 25.4, 2.35)); ax.axis("off")
    cells = [[m, str(by_metric[m][A_NAME]), str(by_metric[m][B_NAME]), str(by_metric[m].get("unit", ""))] for m in REQUIRED]
    table = ax.table(cellText=cells, colLabels=["Metric", "A", "B", "Unit"], cellLoc="center", colLoc="center", loc="center", colWidths=[0.50, 0.19, 0.19, 0.12])
    table.auto_set_font_size(False); table.set_fontsize(6.8); table.scale(1.0, 1.18)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white"); cell.set_edgecolor("#888888"); cell.set_linewidth(0.65 if row == 0 else 0.35)
        if row == 0: cell.set_text_props(weight="bold")
        if col == 0 and row > 0: cell.set_text_props(ha="left")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)
    base = output / "figure_manifest/18_experiment3_POPULATED"
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
