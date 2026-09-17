#!/usr/bin/env python3
"""Static, recording-independent Stationary ALOHA Doll-Handoff preview."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "doll_handoff_aloha_model_preview.usda"

parser = argparse.ArgumentParser(description="Static Stationary ALOHA Doll-Handoff scene preview")
parser.add_argument(
    "--camera", choices=("overview", "legacy_overview", "front", "side", "top"), default="overview"
)
parser.add_argument("--screenshot", type=Path, default=None)
parser.add_argument("--hold-seconds", type=float, default=None, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from preview_common import compose_robot_preview, load_layout, print_task_report, run_viewer


def main() -> None:
    layout = load_layout()
    robot_usd = Path(layout["aloha"]["asset_usd"])
    stage = compose_robot_preview(OUTPUT, "aloha", "StationaryALOHA")
    print(f"[PREVIEW] output={OUTPUT}", flush=True)
    print("[PREVIEW] pose=USD_AUTHORED_STATIC_DEFAULT recording_dependency=NONE", flush=True)
    print("[PREVIEW] imported ALOHA tabletop/camera fixture suppressed; task table is authoritative", flush=True)
    print_task_report(stage, "/World/StationaryALOHA", robot_usd)
    run_viewer(simulation_app, OUTPUT, args_cli.camera, args_cli.hold_seconds, args_cli.screenshot)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
