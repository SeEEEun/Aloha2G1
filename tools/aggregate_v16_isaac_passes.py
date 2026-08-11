#!/usr/bin/env python3
"""Aggregate memory-safe one-camera Isaac v16 replay audits."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
PASSES = ("overview", "side", "left_close", "right_close", "charger_close")


def main() -> int:
    rows = []
    for name in PASSES:
        path = OUT / f"isaaclab_kinematic_validation_{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    first = rows[0]
    videos = {}
    for row in rows:
        videos.update(row.get("videos", {}))
    result = dict(first)
    result.update({
        "status": (
            "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS"
            if all(row.get("status") == "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS" for row in rows)
            else "BLOCKED_ISAACLAB_DEX3_REPLAY"
        ),
        "camera_pass": "memory_safe_sequential_5_camera",
        "camera_passes": list(PASSES),
        "maximum_requested_readback_error_rad": max(
            float(row["maximum_requested_readback_error_rad"]) for row in rows
        ),
        "maximum_numerical_contact_proxy_vs_Isaac_error_m": max(
            float(row["maximum_numerical_contact_proxy_vs_Isaac_error_m"]) for row in rows
        ),
        "actual_task_finger_links_move": all(
            bool(row.get("actual_task_finger_links_move")) for row in rows
        ),
        "physics_steps": max(int(row.get("physics_steps", -1)) for row in rows),
        "videos": videos,
        "pass_audits": {
            name: str((OUT / f"isaaclab_kinematic_validation_{name}.json").resolve())
            for name in PASSES
        },
    })
    target = OUT / "isaaclab_kinematic_validation.json"
    temporary = target.with_suffix(".json.incomplete")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps({"status": result["status"], "videos": len(videos)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
