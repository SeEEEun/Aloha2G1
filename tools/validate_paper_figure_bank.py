#!/usr/bin/env python3
"""Validate completeness and provenance of outputs/paper_figure_bank."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_figure_bank"
REQUIRED_DIRS = [
    "fig01_pipeline", "fig02_retargeting_tradeoff", "fig03_feasibility",
    "fig04_error_distributions", "fig05_per_episode_pairing",
    "fig06_interaction_tradeoff", "fig07_policy_prediction",
    "fig08_phase_behavior", "fig09_policy_smoothness",
    "fig10_trajectory_examples", "fig11_failure_examples",
    "fig12_statistical_summary", "tables", "captions", "figure_index",
]


def main() -> None:
    errors: list[str] = []
    for name in REQUIRED_DIRS:
        if not (OUT / name).is_dir():
            errors.append(f"missing directory: {name}")

    manifest_path = OUT / "figure_index/figure_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing figure manifest")
        figures = []
    else:
        figures = json.loads(manifest_path.read_text())
    for item in figures:
        directory = OUT / item["subdir"]
        stem = item["stem"]
        for suffix in (".png", ".pdf", ".svg", "_source.csv", "_metadata.json", "_generation_command.txt"):
            path = directory / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing/empty figure asset: {path.relative_to(ROOT)}")
        png = directory / f"{stem}.png"
        if png.is_file():
            with Image.open(png) as image:
                dpi = image.info.get("dpi", (0, 0))
                if min(dpi) < 590:
                    errors.append(f"PNG below 600-dpi tolerance: {png.relative_to(ROOT)} dpi={dpi}")
        if item.get("gpu_used") is not False or item.get("current_paper_core_job_touched") is not False:
            errors.append(f"unsafe metadata flags: {stem}")

    tables = sorted((OUT / "tables").glob("Table*.csv"))
    if len(tables) != 7:
        errors.append(f"expected 7 table CSVs, found {len(tables)}")
    for table in tables:
        stem = table.stem
        for suffix in (".json", ".md", ".tex", "_metadata.json", "_generation_command.txt"):
            path = table.with_name(stem + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing/empty table asset: {path.relative_to(ROOT)}")

    captions = sorted((OUT / "captions").glob("*.txt"))
    if len(captions) != 23:
        errors.append(f"expected 23 captions, found {len(captions)}")

    template = OUT / "fig12_statistical_summary/Fig16_source_conditioned_rollout_metrics_TEMPLATE_source.csv"
    if template.is_file():
        with template.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["Trajectory-Centric A"] != "NA" or row["Interaction-Centric B"] != "NA":
                    errors.append("Experiment 3 template contains a non-NA value")
    else:
        errors.append("missing Experiment 3 template source CSV")

    summary_path = OUT / "figure_index/generation_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("gpu_used") is not False or summary.get("current_paper_core_job_touched") is not False:
            errors.append("unsafe generation summary flags")
        if summary.get("verification") != "ALL_CONFIRMED_VALUES_MATCH_FROZEN_ARTIFACTS":
            errors.append("artifact verification flag missing")
        if summary.get("figure_count") != len(figures):
            errors.append("summary/manifest figure count mismatch")
    else:
        errors.append("missing generation summary")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "figure_count": len(figures),
        "table_count": len(tables),
        "caption_count": len(captions),
        "errors": errors,
        "gpu_used": False,
        "current_paper_core_job_touched": False,
    }
    (OUT / "figure_index/validation_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
