#!/usr/bin/env python3
"""Rebuild the authoritative MagSafe USD assets from the active scene_layout.json.

Run through Isaac Lab:
  /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p rebuild_g1_measured_scene.py
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
LAYOUT = ROOT / "scene_layout.json"
GENERATED = ROOT / "generated"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    if not LAYOUT.is_file():
        print(f"ERROR: missing {LAYOUT}", file=sys.stderr)
        return 2

    data = json.loads(LAYOUT.read_text(encoding="utf-8"))
    phone_y0 = float(data["phone"]["bottom_left_xy"][1])
    phone_y1 = float(data["phone"]["bottom_right_xy"][1])
    charger_y = float(data["charger"]["center_xy"][1])

    if abs(phone_y0 - 0.070) > 1e-9 or abs(phone_y1 - 0.070) > 1e-9:
        print(f"ERROR: phone Y is not the measured 0.070 m: {phone_y0}, {phone_y1}", file=sys.stderr)
        return 3
    if abs(charger_y - 0.210) > 1e-9:
        print(f"ERROR: charger Y is not the measured 0.210 m: {charger_y}", file=sys.stderr)
        return 4

    from magsafe_scene_builder import build_all_assets

    print("SIMULATION-ONLY AUTHORITATIVE MAGSAFE SCENE REBUILD")
    print(f"layout: {LAYOUT}")
    print(f"layout SHA-256: {sha256(LAYOUT)}")
    print(f"generated: {GENERATED}")
    result = build_all_assets(LAYOUT, GENERATED)
    print(f"build result: {result}")
    print("PASS: measured layout assets rebuilt")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
