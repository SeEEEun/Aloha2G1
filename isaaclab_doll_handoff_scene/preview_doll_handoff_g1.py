#!/usr/bin/env python3
"""Static fixed-base G1/Dex3 Doll-Handoff preview with direct pelvis placement."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "doll_handoff_g1_model_preview.usda"

parser = argparse.ArgumentParser(description="Static fixed-base G1 Doll-Handoff scene preview")
parser.add_argument(
    "--camera", choices=("overview", "legacy_overview", "front", "side", "top"), default="overview"
)
parser.add_argument("--screenshot", type=Path, default=None)
parser.add_argument("--hold-seconds", type=float, default=None, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from preview_common import compose_robot_preview, g1_geometry_report, load_layout, print_task_report, run_viewer


def main() -> None:
    layout = load_layout()
    robot_usd = Path(layout["g1"]["asset_usd"])
    stage = compose_robot_preview(OUTPUT, "g1", "G1")
    geometry = g1_geometry_report(stage)
    print(f"[PREVIEW] output={OUTPUT}", flush=True)
    print("[PREVIEW] task=doll_handoff objects=doll,trash_bin", flush=True)
    print("[PREVIEW] G1 asset +X faces world +Y toward the table", flush=True)
    print(f"G1_PELVIS_TO_TABLE_FRONT_GAP_M = {geometry['pelvis_to_table_front_gap_m']:.6f}", flush=True)
    print(f"TARGET = {geometry['target_m']:.3f}", flush=True)
    print(f"ERROR = {geometry['error_m']:.6f}", flush=True)
    print_task_report(stage, "/World/G1", robot_usd)
    run_viewer(simulation_app, OUTPUT, args_cli.camera, args_cli.hold_seconds, args_cli.screenshot)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
