#!/usr/bin/env python3
"""Generate the exact-scale A3 ChArUco print used by the helmet D455 kit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
from PIL import Image
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from helmet_d455.calibration import atomic_json, board_from_spec, load_json, sha256_file


def _draw_board(board: object, size: tuple[int, int]) -> object:
    if hasattr(board, "generateImage"):
        return board.generateImage(size, marginSize=0, borderBits=1)
    return board.draw(size, marginSize=0, borderBits=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/helmet_d455_charuco_board.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("calibration/helmet_d455/printable"))
    parser.add_argument("--pixels-per-mm", type=int, default=10)
    args = parser.parse_args()
    spec = load_json(args.spec)
    if args.pixels_per_mm < 4:
        raise ValueError("pixels-per-mm must be at least 4")

    pattern_w = float(spec["printed_pattern_width_mm"])
    pattern_h = float(spec["printed_pattern_height_mm"])
    page = spec["page"]
    width_px = round(pattern_w * args.pixels_per_mm)
    height_px = round(pattern_h * args.pixels_per_mm)
    board = board_from_spec(spec)
    raster = _draw_board(board, (width_px, height_px))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = spec["name"]
    png = args.output_dir / f"{stem}_{args.pixels_per_mm}PX_PER_MM.png"
    pdf = args.output_dir / f"{stem}_A3_PRINT_AT_100_PERCENT.pdf"
    Image.fromarray(raster).save(png, dpi=(args.pixels_per_mm * 25.4,) * 2)

    canvas = Canvas(
        str(pdf),
        pagesize=(float(page["width_mm"]) * mm, float(page["height_mm"]) * mm),
        pageCompression=0,
    )
    # PDF origin is lower-left; the specification's offset is from top-left.
    x = float(page["pattern_offset_left_mm"]) * mm
    y = (float(page["height_mm"]) - float(page["pattern_offset_top_mm"]) - pattern_h) * mm
    canvas.drawImage(str(png), x, y, width=pattern_w * mm, height=pattern_h * mm, mask=None)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(8 * mm, 7 * mm, f"{stem}; pattern {pattern_w:.1f} x {pattern_h:.1f} mm; PRINT AT 100%; NO FIT-TO-PAGE")
    canvas.save()

    report_path = args.output_dir / "print_manifest.json"
    report = {
        "schema_version": "helmet_d455_charuco_print_manifest_v1",
        "status": "PRINTABLE_GENERATED_NOT_PHYSICALLY_VERIFIED",
        "spec": str(args.spec.resolve()),
        "spec_sha256": sha256_file(args.spec),
        "png": str(png.resolve()),
        "png_sha256": sha256_file(png),
        "pdf": str(pdf.resolve()),
        "pdf_sha256": sha256_file(pdf),
        "page_size_mm": [float(page["width_mm"]), float(page["height_mm"])],
        "pattern_size_mm": [pattern_w, pattern_h],
        "pattern_offset_top_left_mm": [float(page["pattern_offset_left_mm"]), float(page["pattern_offset_top_mm"])],
        "print_scale_percent": 100.0,
        "physical_measurement_required_before_use": True,
        "allowed_pattern_dimension_error_mm": 0.5,
    }
    atomic_json(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
