"""Automatic CSV, Markdown, and JSON generators for four paper tables."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping


TABLE_SPECS: dict[str, dict[str, Any]] = {
    "table1_retargeting_quality": {
        "title": "TABLE 1 — RETARGETING QUALITY",
        "optional": False,
        "rows": [
            ("Feasible episodes", "feasibility.FEASIBLE", "episode fraction"),
            ("Hard collision episodes", "feasibility.hard_collision_episode", "episode fraction"),
            ("Wrist error", "wrist.combined.position_error_mm.mean", "mm"),
            ("Whole-hand error", "whole_hand.combined.position_error_mm.mean", "mm"),
            ("Bimanual relation error", "bimanual_relation.error_mm.mean", "mm"),
            ("Projection magnitude", "feasibility.projection_magnitude_mm.mean", "mm"),
        ],
    },
    "table2_downstream_act_policy": {
        "title": "TABLE 2 — DOWNSTREAM ACT POLICY",
        "optional": False,
        "rows": [
            ("28D action RMSE", "action_prediction.OVERALL_28D_RMSE_rad", "rad"),
            ("Arm RMSE", "action_prediction.ARM_14D_RMSE_rad", "rad"),
            ("Dex3 RMSE", "action_prediction.DEX3_14D_RMSE_rad", "rad"),
            ("Phase Completion", "task_sequence.phase_completion_score", "fraction"),
            ("Handoff Ordering", "handoff_ordering.score", "episode fraction"),
            ("Jerk RMS", "smoothness.JOINT_JERK_RMS_rad_s3", "rad/s^3"),
        ],
    },
    "table3_source_conditioned_rollout": {
        "title": "TABLE 3 — FULL SOURCE-CONDITIONED ROLLOUT",
        "optional": False,
        "rows": [
            ("Task-sequence success", "task_sequence.task_sequence_success", "episode fraction"),
            ("Phase completion", "task_sequence.phase_completion_score", "fraction"),
            ("Whole-hand error", "whole_hand.combined.position_error_mm.mean", "mm"),
            ("Bimanual relation error", "bimanual_relation.error_mm.mean", "mm"),
            ("RPL", "path_efficiency.RPL", "ratio"),
            ("SWPE", "path_efficiency.SWPE", "ratio"),
            ("Jerk RMS", "smoothness.JOINT_JERK_RMS_rad_s3", "rad/s^3"),
        ],
    },
    "table4_isaac_physical_task_success": {
        "title": "TABLE 4 — ISAAC PHYSICAL TASK SUCCESS (OPTIONAL)",
        "optional": True,
        "rows": [
            ("Left lift success", "physical_success.LEFT_LIFT_SUCCESS", "episode fraction"),
            ("Handoff success", "physical_success.HANDOFF_SUCCESS", "episode fraction"),
            ("Right transport success", "physical_success.RIGHT_TRANSPORT_SUCCESS", "episode fraction"),
            ("Release success", "physical_success.RELEASE_SUCCESS", "episode fraction"),
            ("Full physical success", "physical_success.FULL_PHYSICAL_SUCCESS", "episode fraction"),
            ("PCS", "task_sequence.phase_completion_score", "fraction"),
            ("SWPE", "path_efficiency.SWPE", "ratio"),
        ],
    },
}


def _format_mean_std(summary: Mapping[str, Any]) -> str:
    return f"{float(summary['mean']):.6g} ± {float(summary['std']):.6g}"


def _format_median_iqr(summary: Mapping[str, Any]) -> str:
    return (
        f"{float(summary['median']):.6g} "
        f"[{float(summary['q25']):.6g}, {float(summary['q75']):.6g}]"
    )


def _row(label: str, path: str, unit: str, report: Mapping[str, Any] | None) -> dict[str, Any]:
    metric = None if report is None else report.get("metrics", {}).get(path)
    if not isinstance(metric, Mapping) or metric.get("status") != "READY":
        return {
            "Metric": label,
            "Unit": unit,
            "A mean ± std": "NA",
            "B mean ± std": "NA",
            "A median [IQR]": "NA",
            "B median [IQR]": "NA",
            "Paired B-A": "NA",
            "Paired bootstrap 95% CI": "NA",
            "N pairs": "NA",
        }
    ci = metric["paired_bootstrap_95_percent_CI_of_mean_B_minus_A"]
    return {
        "Metric": label,
        "Unit": unit,
        "A mean ± std": _format_mean_std(metric["A"]),
        "B mean ± std": _format_mean_std(metric["B"]),
        "A median [IQR]": _format_median_iqr(metric["A"]),
        "B median [IQR]": _format_median_iqr(metric["B"]),
        "Paired B-A": f"{float(metric['paired_B_minus_A']['mean']):.6g}",
        "Paired bootstrap 95% CI": f"[{float(ci[0]):.6g}, {float(ci[1]):.6g}]",
        "N pairs": int(metric["paired_episode_count"]),
    }


def _markdown(title: str, rows: list[dict[str, Any]], optional: bool) -> str:
    keys = list(rows[0])
    lines = [f"# {title}", ""]
    if optional:
        lines.extend(["This table remains unpopulated until physical Isaac evaluation exists.", ""])
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join("---" for _ in keys) + " |")
    lines.extend("| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def generate_tables(
    output_dir: Path,
    paired_reports: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Generate all formats; absent results are literal ``NA``, never fake zeros."""

    output_dir = Path(output_dir)
    reports = dict(paired_reports or {})
    manifest: dict[str, Any] = {
        "schema_version": "paper_table_manifest_v1",
        "status": "READY",
        "missing_results_rendered_as": "NA",
        "tables": {},
    }
    for name, specification in TABLE_SPECS.items():
        report = reports.get(name)
        rows = [_row(*row, report) for row in specification["rows"]]
        csv_path = output_dir / f"{name}.csv"
        csv_temp = csv_path.with_suffix(".csv.incomplete")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(csv_temp, csv_path)
        md_path = output_dir / f"{name}.md"
        _atomic_text(
            md_path,
            _markdown(specification["title"], rows, bool(specification["optional"])),
        )
        json_path = output_dir / f"{name}.json"
        payload = {
            "schema_version": "paper_table_v1",
            "title": specification["title"],
            "optional": bool(specification["optional"]),
            "status": "READY_WITH_RESULTS" if report is not None else "READY_TEMPLATE_RESULTS_NA",
            "rows": rows,
        }
        _atomic_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        manifest["tables"][name] = {
            "csv": str(csv_path.resolve()),
            "markdown": str(md_path.resolve()),
            "json": str(json_path.resolve()),
            "status": payload["status"],
        }
    return manifest
