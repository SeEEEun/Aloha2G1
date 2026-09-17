#!/usr/bin/env python3
"""Fail-closed validation for the final standardized-grasp result package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/standardized_grasp_ab_dev35"


def read(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"); os.replace(temporary, path)


def probe(path: Path) -> dict[str, Any]:
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)], text=True))["streams"][0]


def main() -> int:
    numeric = read(OUT / "05_results/FINAL_STANDARDIZED_GRASP_NUMERIC_RESULTS.json")
    freeze = OUT / "02_freeze/STANDARDIZED_GRASP_FINAL_FREEZE.json"
    figures = [OUT / f"06_paper_artifacts/FigXX_Standardized_Grasp_AB_Physical_Comparison_double.{suffix}" for suffix in ("png", "pdf", "svg")]
    videos = [OUT / "07_physical_replays" / name for name in (
        "A_POST_GRASP_DEV35_TOP_35SPLIT.mp4", "A_POST_GRASP_DEV35_OVERVIEW_35SPLIT.mp4",
        "B_POST_GRASP_DEV35_TOP_35SPLIT.mp4", "B_POST_GRASP_DEV35_OVERVIEW_35SPLIT.mp4",
        "AB_POST_GRASP_MATCHED_PHYSICAL_REVIEW.mp4",
    )]
    runs = {
        "A": sorted((OUT / "03_a_results/rollouts").glob("eval_*/RUN_MANIFEST.json")),
        "B": sorted((OUT / "04_b_results/rollouts").glob("eval_*/RUN_MANIFEST.json")),
    }
    checks = {
        "freeze_exists": freeze.is_file(), "result_status": numeric.get("status") == "FINAL_COMPARABLE_35_PLUS_35",
        "A_35_valid": len(runs["A"]) == 35, "B_35_valid": len(runs["B"]) == 35,
        "70_result_rows": numeric.get("integrity", {}).get("valid_rows") == 70,
        "35_matched_pairs": numeric.get("integrity", {}).get("matched_pairs") == 35,
        "A_standardized_35": numeric["A"]["standardized_initial_grasp_count"] == 35,
        "B_standardized_35": numeric["B"]["standardized_initial_grasp_count"] == 35,
        "figures_exist": all(path.is_file() and path.stat().st_size > 0 for path in figures),
        "table_exists": (OUT / "06_paper_artifacts/TABLE_STANDARDIZED_GRASP_AB_PHYSICAL_RESULTS.md").is_file(),
        "report_exists": (OUT / "FINAL_STANDARDIZED_GRASP_AB_PHYSICAL_REPORT.md").is_file(),
        "videos_exist": all(path.is_file() and path.stat().st_size > 0 for path in videos),
        "no_arm_rescue": numeric["integrity"]["arm_rescue"] is False,
        "no_wrist_rescue": numeric["integrity"]["wrist_rescue"] is False,
        "zero_object_pose_writes": numeric["integrity"]["object_pose_writes_after_initialization"] == 0,
    }
    video_probes = {path.name: probe(path) for path in videos if path.is_file()}
    checks["video_codec_resolution_fps"] = len(video_probes) == 5 and all(
        value["codec_name"] == "h264" and int(value["width"]) == 3840 and int(value["height"]) == 2160 and value["avg_frame_rate"] == "30/1"
        for value in video_probes.values()
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {"schema_version": "standardized_grasp_ab_package_verification_v1", "status": status, "checks": checks, "video_probes": video_probes}
    atomic_json(OUT / "FINAL_PACKAGE_VERIFICATION.json", report)
    print(json.dumps(report, indent=2)); return 0 if status == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
