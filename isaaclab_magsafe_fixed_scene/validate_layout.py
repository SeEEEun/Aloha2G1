"""Validate the fixed table-relative layout without launching Isaac Sim."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LAYOUT = json.loads((ROOT / "scene_layout.json").read_text(encoding="utf-8"))


def main() -> None:
    table = LAYOUT["table"]
    phone = LAYOUT["phone"]
    accessory = LAYOUT["accessory"]
    charger = LAYOUT["charger"]

    sx, sy = table["size_x"], table["size_y"]
    z_table = table["surface_height"]
    p0, p1 = phone["bottom_left_xy"], phone["bottom_right_xy"]
    phone_length, phone_depth, phone_height = phone["size_landscape_xyz"]
    phone_center = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0, z_table + phone_height / 2.0)
    measured_edge = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    accessory_center = (
        phone_center[0],
        phone_center[1] + phone_depth / 2.0 + accessory["phone_back_clearance"] + accessory["main_depth"] / 2.0,
        phone_center[2],
    )
    mount_height = charger["mount_plate"]["size_xyz"][2] if charger["mount_plate"]["enabled"] else 0.0
    charger_origin = (charger["center_xy"][0], charger["center_xy"][1], z_table + mount_height)

    print("Coordinate convention: +X left→right, +Y front→back, +Z up")
    print(f"Table bounds: X=[0,{sx:.3f}] m, Y=[0,{sy:.3f}] m, top Z={z_table:.3f} m")
    print(f"Measured phone lower-edge length: {measured_edge:.4f} m")
    print(f"Exact phone model length:         {phone_length:.4f} m")
    print(f"Phone center world:              {phone_center}")
    print(f"Accessory main center world:     {accessory_center}")
    print(f"Charger base-bottom world:       {charger_origin}")

    errors: list[str] = []
    if not (0.0 <= phone_center[0] <= sx and 0.0 <= phone_center[1] <= sy):
        errors.append("Phone center is outside the table.")
    if not (0.0 <= charger_origin[0] <= sx and 0.0 <= charger_origin[1] <= sy):
        errors.append("Charger center is outside the table.")
    if abs(measured_edge - phone_length) > 0.002:
        errors.append("Measured phone lower-edge points differ from the exact phone length by more than 2 mm.")

    if errors:
        print("\nLayout validation FAILED:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nLayout validation OK.")


if __name__ == "__main__":
    main()
