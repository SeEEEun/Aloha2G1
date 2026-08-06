#!/usr/bin/env python3
"""Install the measured G1 MagSafe layout without changing G1 root/camera/lighting.

Default target:
  /home/jbnu/aloha_g1_dataset/isaaclab_magsafe_fixed_scene

Changes only:
  phone.bottom_left_xy[1]  = 0.150
  phone.bottom_right_xy[1] = 0.150
  charger.center_xy[1]     = 0.210
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_SCENE_DIR = Path("/home/jbnu/aloha_g1_dataset/isaaclab_magsafe_fixed_scene")
PHONE_Y_M = 0.150
CHARGER_Y_M = 0.210

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data

def validate_base(data: dict[str, Any]) -> None:
    required = ("coordinate_frame", "table", "phone", "accessory", "charger", "render")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    axis = data["coordinate_frame"].get("y_axis", "")
    if "front/operator side to back/charger side" not in axis:
        raise ValueError(
            "Unexpected Y-axis convention. Refusing to apply measured distances."
        )

    table = data["table"]
    if abs(float(table["size_x"]) - 0.835) > 1e-9 or abs(float(table["size_y"]) - 0.720) > 1e-9:
        raise ValueError("Unexpected table size. Refusing to modify a different scene.")

    for key in ("bottom_left_xy", "bottom_right_xy"):
        value = data["phone"].get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Invalid phone.{key}: {value}")

    center = data["charger"].get("center_xy")
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError(f"Invalid charger.center_xy: {center}")

def make_measured(data: dict[str, Any]) -> dict[str, Any]:
    # JSON round-trip gives a clean deep copy.
    measured = json.loads(json.dumps(data))

    # Preserve X. Change only front/back Y.
    measured["phone"]["bottom_left_xy"][1] = PHONE_Y_M
    measured["phone"]["bottom_right_xy"][1] = PHONE_Y_M
    measured["charger"]["center_xy"][1] = CHARGER_Y_M

    measured["phone"]["notes"] = (
        "G1 measured provisional layout: phone initial pose is 0.150 m from "
        "the table front/operator edge. X position, dimensions, orientation, "
        "and height convention are unchanged."
    )
    measured["charger"]["notes"] = (
        "G1 measured provisional layout: charger center is 0.210 m from "
        "the table front/operator edge. X position, dimensions, orientation, "
        "and height convention are unchanged."
    )
    return measured

def summary(data: dict[str, Any]) -> str:
    p0 = data["phone"]["bottom_left_xy"]
    p1 = data["phone"]["bottom_right_xy"]
    c = data["charger"]["center_xy"]
    size = data["charger"]["base_size_xy"]
    phone_depth = float(data["phone"]["size_landscape_xyz"][1])
    charger_front = float(c[1]) - float(size[1]) / 2.0
    phone_back = float(p0[1]) + phone_depth / 2.0
    return "\n".join([
        f"phone bottom-left XY : {p0}",
        f"phone bottom-right XY: {p1}",
        f"phone center XY      : [{(float(p0[0])+float(p1[0]))/2:.6f}, {float(p0[1]):.6f}]",
        f"charger center XY    : {c}",
        f"phone↔charger ΔY     : {float(c[1])-float(p0[1]):.6f} m",
        f"charger base front Y : {charger_front:.6f} m",
        f"phone back face Y    : {phone_back:.6f} m",
        f"planar Y clearance   : {charger_front-phone_back:.6f} m",
    ])

def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--apply", action="store_true", help="Back up and overwrite scene_layout.json")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed changes only")
    args = parser.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    src = scene_dir / "scene_layout.json"
    measured_copy = scene_dir / "scene_layout.g1_measured.json"

    if not src.is_file():
        print(f"ERROR: missing {src}", file=sys.stderr)
        return 2

    original = load_json(src)
    validate_base(original)
    measured = make_measured(original)

    print("SIMULATION-ONLY G1 MEASURED MAGSAFE LAYOUT")
    print("No G1 root, camera, lighting, X coordinate, Z coordinate, or robot command is modified.")
    print(f"source : {src}")
    print(f"source SHA-256: {sha256(src)}")
    print("\n[BEFORE]")
    print(summary(original))
    print("\n[AFTER]")
    print(summary(measured))

    if not args.apply:
        print("\nDRY RUN: no files changed. Re-run with --apply to install.")
        return 0

    backup_dir = scene_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"scene_layout.before_g1_measured.{stamp}.json"
    shutil.copy2(src, backup)

    atomic_write_json(measured_copy, measured)
    atomic_write_json(src, measured)

    print("\nAPPLIED")
    print(f"backup        : {backup}")
    print(f"measured copy : {measured_copy}")
    print(f"active layout : {src}")
    print(f"active SHA-256: {sha256(src)}")
    print("Status: MEASURED_NOT_APPROVED (simulation layout only)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
