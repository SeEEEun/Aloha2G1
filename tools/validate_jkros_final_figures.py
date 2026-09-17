#!/usr/bin/env python3
"""Validate the exact-six JKROS figure package and its frozen values."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_final_figures"
FOLDERS = {
    1: ("Fig01_Method_Overview", "Fig01_Method_Overview", "double"),
    2: ("Fig02_Retargeting_Tradeoff", "Fig02_Retargeting_Tradeoff", "double"),
    3: ("Fig03_Paired_Statistics", "Fig03_Paired_Statistical_Effect", "single"),
    4: ("Fig04_Feasibility", "Fig04_Feasibility", "double"),
    5: ("Fig05_Supervision_to_Policy", "Fig05_Supervision_to_Policy", "double"),
    6: ("Fig06_Representative_Motion", "Fig06_Representative_Motion", "double"),
}
EXPECTED_EFFECTS = {
    "Wrist error": (74.11501871520937, 71.49809312506959, 76.61526225730262),
    "Whole-hand error": (-69.67791922449261, -73.15461511625703, -66.53148757846465),
    "Bimanual relation": (-50.81853092092401, -55.682112117841825, -46.21706637913662),
    "Projection magnitude": (-2.4498772634494976, -4.582062966296461, -0.7517672625772608),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    checks: list[dict[str, object]] = []

    def json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def check(name: str, condition: bool, detail: object = "") -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": json_safe(detail)})
        if not condition:
            raise RuntimeError(f"validation failed: {name}: {detail}")

    directories = sorted(path.name for path in OUT.glob("Fig[0-9][0-9]_*" ) if path.is_dir())
    check("exact six main figure directories", directories == sorted(item[0] for item in FOLDERS.values()), directories)
    check("no video in final package", not any(OUT.rglob("*.mp4")))
    check("no Experiment-3 file in final package", not any("experiment3" in path.name.lower() for path in OUT.rglob("*")))

    for number, (folder_name, stem, recommended) in FOLDERS.items():
        folder = OUT / folder_name
        required = [
            folder / f"{stem}.png",
            folder / f"{stem}.pdf",
            folder / f"{stem}.svg",
            folder / f"{stem}_source_data.csv",
            folder / f"{stem}_metadata.json",
            folder / f"{stem}_generation_command.txt",
            folder / f"{stem}_caption.txt",
            OUT / "captions" / f"Fig{number:02d}_caption.txt",
        ]
        check(f"Fig{number:02d} required files", all(path.is_file() and path.stat().st_size > 0 for path in required), [str(path) for path in required])
        for suffix in (".png", ".pdf", ".svg"):
            check(
                f"Fig{number:02d} recommended variant equals main {suffix}",
                sha256(folder / f"{stem}{suffix}") == sha256(folder / f"{stem}_{recommended}{suffix}"),
            )
        with Image.open(folder / f"{stem}.png") as image:
            dpi = image.info.get("dpi", (0, 0))
            expected_width = 88.0 if recommended == "single" else 178.0
            actual_width_mm = image.width / float(dpi[0]) * 25.4 if dpi[0] else 0.0
            check(f"Fig{number:02d} PNG 600 dpi", min(dpi) >= 599.0, dpi)
            check(f"Fig{number:02d} final width", abs(actual_width_mm - expected_width) < 0.2, actual_width_mm)
            check(f"Fig{number:02d} PNG nonblank", float(np.asarray(image.convert("L")).std()) > 5.0)
        ET.parse(folder / f"{stem}.svg")
        pdf_info = subprocess.run(["pdfinfo", str(folder / f"{stem}.pdf")], check=True, capture_output=True, text=True).stdout
        check(f"Fig{number:02d} one-page PDF", "Pages:           1" in pdf_info, pdf_info.splitlines())
        metadata = json.loads((folder / f"{stem}_metadata.json").read_text(encoding="utf-8"))
        check(f"Fig{number:02d} metadata number", metadata["figure_number"] == number)
        check(f"Fig{number:02d} no result mutation", metadata["scientific_results_modified"] is False)
        check(f"Fig{number:02d} no Experiment-3", metadata["experiment3_used"] is False)
        for source in metadata["source_artifacts"]:
            path = ROOT / source["path"]
            check(f"Fig{number:02d} source hash {path.name}", path.is_file() and sha256(path) == source["sha256"])
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in [folder / f"{stem}.svg", folder / f"{stem}_caption.txt"])
        check(f"Fig{number:02d} no placeholder text", "RENDER_REQUIRED" not in text and "PLACEHOLDER" not in text.upper())

    fig2 = rows(OUT / "Fig02_Retargeting_Tradeoff/Fig02_Retargeting_Tradeoff_source_data.csv")
    fig2_summary = {(row["metric"], row["method"]): float(row["mean_mm"]) for row in fig2 if row["record_type"] == "frame_weighted_summary"}
    expected_means = {
        ("Wrist trajectory error", "A"): 3.742,
        ("Wrist trajectory error", "B"): 77.853,
        ("Whole-hand interaction error", "A"): 90.820,
        ("Whole-hand interaction error", "B"): 21.157,
        ("Bimanual relation error", "A"): 86.786,
        ("Bimanual relation error", "B"): 35.979,
    }
    check("Fig02 all 50 episode points per metric/method", all(sum(row["record_type"] == "episode_mean" and row["metric"] == metric and row["method"] == method for row in fig2) == 50 for metric, method in expected_means))
    check("Fig02 authoritative means", all(abs(fig2_summary[key] - value) < 5e-4 for key, value in expected_means.items()), fig2_summary)

    fig3 = rows(OUT / "Fig03_Paired_Statistics/Fig03_Paired_Statistical_Effect_source_data.csv")
    check("Fig03 four stored effects", len(fig3) == 4)
    for row in fig3:
        actual = tuple(float(row[key]) for key in ("paired_B_minus_A_mean_mm", "ci95_low_mm", "ci95_high_mm"))
        check(f"Fig03 authoritative effect {row['metric']}", np.allclose(actual, EXPECTED_EFFECTS[row["metric"]], rtol=0.0, atol=5e-12), actual)
        check(f"Fig03 bootstrap contract {row['metric']}", int(row["episodes"]) == 50 and int(row["bootstrap_resamples"]) == 20_000 and int(row["seed"]) == 20260827)

    fig4 = rows(OUT / "Fig04_Feasibility/Fig04_Feasibility_source_data.csv")
    outcomes = {row["method"]: (int(row["CLEAN"]), int(row["WARNING"]), int(row["HARD"])) for row in fig4 if row["record_type"] == "outcome"}
    check("Fig04 authoritative outcomes", outcomes == {"A": (27, 21, 2), "B": (13, 37, 0)}, outcomes)
    failures = {(row["method"], row["failure_mode"]): int(row["count"]) for row in fig4 if row["record_type"] == "hard_failure_mode"}
    check("Fig04 authoritative failure modes", failures == {("A", "Hard collision"): 2, ("A", "Hard IK"): 0, ("A", "Joint limit"): 0, ("A", "Branch"): 0, ("B", "Hard collision"): 0, ("B", "Hard IK"): 0, ("B", "Joint limit"): 0, ("B", "Branch"): 0}, failures)

    fig5 = rows(OUT / "Fig05_Supervision_to_Policy/Fig05_Supervision_to_Policy_source_data.csv")
    check("Fig05 retargeting episode sample count", sum(row["level"] == "Retargeted supervision" and row["record_type"] == "episode_mean" for row in fig5) == 300)
    check("Fig05 held-out episode sample count", sum(row["level"] == "Held-out ACT prediction" and row["record_type"] == "heldout_episode_mean" for row in fig5) == 48)
    heldout_ids = sorted({int(row["episode_index"]) for row in fig5 if row["level"] == "Held-out ACT prediction" and row["record_type"] == "heldout_episode_mean"})
    check("Fig05 HELDOUT8 identities", heldout_ids == [2, 13, 23, 27, 28, 31, 37, 40], heldout_ids)
    summaries = {(row["level"], row["metric"], row["method"]): float(row["mean_mm"]) for row in fig5 if row["record_type"] == "frame_weighted_summary"}
    expected_policy = {
        ("Held-out ACT prediction", "Wrist", "ACT-A"): 24.958,
        ("Held-out ACT prediction", "Wrist", "ACT-B"): 95.269,
        ("Held-out ACT prediction", "Whole-hand", "ACT-A"): 107.131,
        ("Held-out ACT prediction", "Whole-hand", "ACT-B"): 58.330,
        ("Held-out ACT prediction", "Bimanual", "ACT-A"): 120.976,
        ("Held-out ACT prediction", "Bimanual", "ACT-B"): 85.920,
    }
    check("Fig05 authoritative policy means", all(abs(summaries[key] - value) < 5e-4 for key, value in expected_policy.items()), {str(key): summaries[key] for key in expected_policy})

    report_path = OUT / "Fig06_Representative_Motion/render_assets/matched_render_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    check("Fig06 render status", report["status"] == "MATCHED_STILLS_COMPLETE")
    check("Fig06 representative episode", report["episode_index"] == 23)
    check("Fig06 semantic frames", [row["source_frame"] for row in report["semantic_states"]] == [182, 299, 320, 402])
    check("Fig06 still count", len(report["rendered_assets"]) == 8)
    check("Fig06 source RGB count", len(report["source_assets"]) == 4)
    check("Fig06 zero q readback", max(report["maximum_named_joint_qpos_readback_error_rad"].values()) <= 2e-6)
    check("Fig06 identical object visualization", report["object_visualization"]["identical_pose_for_A_and_B_at_each_source_frame"] is True)
    check("Fig06 no video/simulation advance", report["videos_written"] is False and report["simulation_time_advanced"] is False)
    for asset in report["rendered_assets"]:
        path = Path(asset["output_path"])
        with Image.open(path) as image:
            check(f"Fig06 render resolution {path.name}/{asset['method']}", image.size == (640, 480))
            check(f"Fig06 render nonblank {path.name}/{asset['method']}", float(np.asarray(image.convert("L")).std()) > 10.0)
        check(f"Fig06 render hash {path.name}/{asset['method']}", sha256(path) == asset["sha256"])

    contact = OUT / "contact_sheet/FINAL_SIX_FIGURES_CONTACT_SHEET.png"
    with Image.open(contact) as image:
        check("contact sheet dimensions", image.size == (2200, 3300), image.size)
        check("contact sheet nonblank", float(np.asarray(image.convert("L")).std()) > 5.0)
    manifest = json.loads((OUT / "manifest/final_six_figure_manifest.json").read_text(encoding="utf-8"))
    check("manifest figure count", manifest["main_figure_count"] == 6 and len(manifest["figures"]) == 6)

    report_out = {
        "schema_version": "jkros_final_figure_validation_v1",
        "status": "PASS",
        "checks_passed": len(checks),
        "checks_failed": 0,
        "checks": checks,
    }
    (OUT / "manifest/validation_report.json").write_text(json.dumps(report_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks_passed": len(checks), "report": str(OUT / "manifest/validation_report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
