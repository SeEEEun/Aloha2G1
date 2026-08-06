"""GUI preview: unchanged MagSafe magnetic v2 scene plus fixed-base G1/Dex3."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
OUTPUT = ROOT / "generated" / "magsafe_g1_model_preview.usda"

parser = argparse.ArgumentParser(description="Static/default-pose G1 model preview.")
parser.add_argument("--camera", choices=("overview", "front", "side", "top"), default="overview")
parser.add_argument("--hold-seconds", type=float, default=None, help=argparse.SUPPRESS)
# Final total offset for this static environment. This is applied once to the
# original root pose; it is not an increment on top of an earlier preview.
parser.add_argument("--root-forward-offset-m", type=float, default=0.15)
parser.add_argument("--root-lateral-offset-m", type=float, default=0.0)
parser.add_argument("--root-z-offset-m", type=float, default=0.0)
parser.add_argument("--root-yaw-offset-deg", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robot_model_preview_common import compose_stage, print_stage_report, run_viewer


def main() -> None:
    stage = compose_stage(
        OUTPUT, "G1", G1_USD, "g1",
        forward_offset_m=args_cli.root_forward_offset_m,
        lateral_offset_m=args_cli.root_lateral_offset_m,
        z_offset_m=args_cli.root_z_offset_m,
        yaw_offset_deg=args_cli.root_yaw_offset_deg,
    )
    print(f"[PREVIEW] output={OUTPUT}", flush=True)
    print("[PREVIEW] G1 authored forward axis=+X; world forward axis=+Y")
    print_stage_report(stage, "/World/G1", G1_USD)
    run_viewer(simulation_app, OUTPUT, args_cli.camera, args_cli.hold_seconds)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
