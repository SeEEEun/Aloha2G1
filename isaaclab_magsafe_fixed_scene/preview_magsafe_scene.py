"""Build and preview the fixed-layout MagSafe scene in Isaac Lab.

Usage from an Isaac Lab checkout:

    ./isaaclab.sh -p /absolute/path/preview_magsafe_scene.py --rebuild

The script creates modular USDA assets under ``generated/`` and then references
``magsafe_fixed_scene.usda`` into a robot-free Isaac Lab stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


_THIS_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Build and preview the fixed MagSafe task scene.")
parser.add_argument(
    "--layout",
    type=Path,
    default=_THIS_DIR / "scene_layout.json",
    help="Path to the fixed scene layout JSON.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=_THIS_DIR / "generated",
    help="Directory for generated USDA assets.",
)
parser.add_argument("--rebuild", action="store_true", help="Rebuild all USDA assets before previewing.")
parser.add_argument("--export-only", action="store_true", help="Build assets and exit without opening the preview loop.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports below require a running Isaac Sim application.
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

from magsafe_scene_builder import build_all_assets, load_layout


def _build_if_needed() -> dict[str, Path]:
    output_dir = args_cli.output_dir.expanduser().resolve()
    scene_path = output_dir / "magsafe_fixed_scene.usda"
    required = [
        output_dir / "table_optical.usda",
        output_dir / "phone_landscape.usda",
        output_dir / "magsafe_poppinger_1062886.usda",
        output_dir / "charger_stand.usda",
        scene_path,
    ]
    if args_cli.rebuild or not all(path.exists() for path in required):
        print(f"[INFO] Building MagSafe scene assets in: {output_dir}")
        paths = build_all_assets(args_cli.layout, output_dir)
        for name, path in paths.items():
            print(f"[INFO]   {name:10s}: {path}")
        return paths
    return {
        "table": required[0],
        "phone": required[1],
        "accessory": required[2],
        "charger": required[3],
        "scene": required[4],
    }


def main() -> None:
    paths = _build_if_needed()
    if args_cli.export_only:
        print(f"[INFO] Export complete: {paths['scene']}")
        return

    layout = load_layout(args_cli.layout)
    render_cfg = layout["render"]

    sim_cfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
    sim = SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg(
        size=(6.0, 6.0),
        color=(0.12, 0.13, 0.14),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
    )
    ground_cfg.func("/World/Ground", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(
        intensity=float(render_cfg["dome_intensity"]),
        color=(0.78, 0.82, 0.88),
    )
    dome_cfg.func("/World/Lights/Dome", dome_cfg)

    key_cfg = sim_utils.DistantLightCfg(
        intensity=float(render_cfg["key_light_intensity"]),
        color=(1.0, 0.96, 0.90),
        angle=0.45,
    )
    key_cfg.func(
        "/World/Lights/Key",
        key_cfg,
        translation=(0.0, -1.0, 3.0),
        orientation=(0.9239, 0.2209, -0.2209, 0.2209),
    )

    fill_cfg = sim_utils.SphereLightCfg(
        radius=0.35,
        intensity=float(render_cfg["fill_light_intensity"]),
        color=(0.78, 0.87, 1.0),
    )
    fill_cfg.func("/World/Lights/Fill", fill_cfg, translation=(0.15, 0.25, 1.75))

    scene_cfg = sim_utils.UsdFileCfg(usd_path=str(paths["scene"]))
    scene_cfg.func("/World/MagSafeScene", scene_cfg)

    sim.set_camera_view(render_cfg["camera_eye"], render_cfg["camera_target"])
    sim.reset()
    print("[INFO] Fixed MagSafe scene loaded.")
    print("[INFO] Coordinate convention: +X table-left→right, +Y front→back, +Z up.")
    print(f"[INFO] Composite USD: {paths['scene']}")
    print("[INFO] No robot is included. Close the viewer or press Ctrl+C to exit.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
