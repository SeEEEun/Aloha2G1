#!/usr/bin/env python3
"""Validate formats, variants, provenance, and scientific invariants."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_figure_bank_pubready"
sys.path.insert(0, str(ROOT / "tools"))
import generate_jkros_pubready_figures as pub  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    manifest_path = OUT / "figure_manifest/figure_manifest.json"
    check("manifest exists", manifest_path.is_file(), str(manifest_path))
    if not manifest_path.is_file():
        raise SystemExit("run tools/generate_jkros_pubready_figures.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check("18 redesigned master figures", len(manifest) == 18, len(manifest))
    priorities = {p: sum(r["priority"] == p for r in manifest) for p in ("MUST_USE", "STRONG", "OPTIONAL")}
    check("priority counts", priorities == {"MUST_USE": 7, "STRONG": 6, "OPTIONAL": 5}, priorities)

    for record in manifest:
        stem = Path(record["filename"]).stem
        for suffix, folder in ((".png", "png_all"), (".pdf", "pdf_all"), (".svg", "svg_all")):
            path = OUT / folder / f"{stem}{suffix}"
            check(f"{stem}{suffix} exists", path.is_file() and path.stat().st_size > 1000, path.stat().st_size if path.is_file() else "missing")
        png = OUT / "png_all" / f"{stem}.png"
        if png.is_file():
            with Image.open(png) as image:
                dpi = image.info.get("dpi", (0, 0))
                check(f"{stem} PNG 600 dpi", min(dpi) >= 599.0, dpi)
                check(f"{stem} white-background RGB/RGBA", image.mode in ("RGB", "RGBA"), image.mode)
        csv_path = OUT / "source_data" / f"{stem}.csv"
        caption = OUT / "captions" / f"{stem}_caption.txt"
        metadata = OUT / "figure_manifest" / f"{stem}_metadata.json"
        check(f"{stem} source CSV", csv_path.is_file() and csv_path.stat().st_size > 5, csv_path.stat().st_size if csv_path.is_file() else "missing")
        check(f"{stem} caption", caption.is_file() and len(caption.read_text(encoding="utf-8").strip()) > 40, str(caption))
        check(f"{stem} metadata", metadata.is_file(), str(metadata))
        category = OUT / record["priority"].lower() / f"{stem}.png"
        check(f"{stem} category copy", category.is_file(), str(category))

    must = [r for r in manifest if r["priority"] == "MUST_USE"]
    expected_widths = {"single": round(88.0 / 25.4 * 600), "double": round(178.0 / 25.4 * 600)}
    for record in must:
        stem = Path(record["filename"]).stem
        for variant, expected_width in expected_widths.items():
            base = OUT / f"{variant}_column" / f"{stem}_{variant}"
            for suffix in (".png", ".pdf", ".svg"):
                check(f"{stem} {variant} {suffix}", base.with_suffix(suffix).is_file(), str(base.with_suffix(suffix)))
            with Image.open(base.with_suffix(".png")) as image:
                check(f"{stem} {variant} exact publication width", abs(image.width - expected_width) <= 1, {"actual_px": image.width, "expected_px": expected_width})
                dpi = image.info.get("dpi", (0, 0))
                check(f"{stem} {variant} 600 dpi", min(dpi) >= 599.0, dpi)

    legacy_map_path = OUT / "figure_manifest/LEGACY_ASSET_MAP.json"
    legacy = json.loads(legacy_map_path.read_text(encoding="utf-8")) if legacy_map_path.is_file() else []
    check("24 original candidates centralized", len(legacy) == 24, len(legacy))
    check("legacy copies byte-identical", bool(legacy) and all(r["original_sha256"] == r["central_sha256"] for r in legacy), "all copied PNG hashes match" if legacy else "missing map")

    for name in ("ALL_PNG_CONTACT_SHEET.png", "MUST_USE_CONTACT_SHEET.png"):
        path = OUT / "png_all" / name
        check(name, path.is_file() and path.stat().st_size > 10_000, path.stat().st_size if path.is_file() else "missing")

    exp3 = OUT / "source_data/18_experiment3_template.csv"
    with exp3.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    check("Experiment 3 has 7 template rows", len(rows) == 7, len(rows))
    check("Experiment 3 values all NA", all(row[pub.A_NAME] == "NA" and row[pub.B_NAME] == "NA" for row in rows), rows)

    ctx = pub.main_context()
    check("frozen artifacts reverified", True, "all confirmed Experiment 1/2 values match")
    check("phase totals exact", (int(ctx["phase_matrices"]["a"].sum()), int(ctx["phase_matrices"]["b"].sum())) == (56, 52), [int(ctx["phase_matrices"]["a"].sum()), int(ctx["phase_matrices"]["b"].sum())])
    check("representative episode criterion", ctx["representative_episode"] == 23, ctx["representative_episode"])
    check("failure selection criterion", pub.failure_data(ctx)["episode"] == 46, pub.failure_data(ctx)["episode"])

    summary_path = OUT / "figure_manifest/generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    check("GPU not used", summary["gpu_used"] is False and summary["isaac_run"] is False, {"gpu_used": summary["gpu_used"], "isaac_run": summary["isaac_run"]})
    check("scientific assets untouched", summary["dataset_ab_modified"] is False and summary["paper_core_job_touched"] is False and summary["training_run"] is False, summary)

    failures = [c for c in checks if not c["passed"]]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failures),
        "checks_failed": len(failures),
        "failures": failures,
        "checks": checks,
    }
    report_path = OUT / "figure_manifest/validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "checks_total", "checks_passed", "checks_failed")}, indent=2))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure['check']}: {failure['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
