#!/usr/bin/env python3
"""Isaac/PhysX Doll-Handoff settle, hollow-bin, and diagnostic-drop test."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parent
SCENE_USD = ROOT / "generated" / "doll_handoff_scene.usda"
REPORT_PATH = ROOT / "outputs" / "doll_handoff_scene" / "physics_validation.json"

parser = argparse.ArgumentParser(description="Doll-Handoff physics sanity validation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _position(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim during physics validation: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    value = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return tuple(float(value[index]) for index in range(3))


def _finite(position: tuple[float, float, float]) -> bool:
    return all(math.isfinite(value) for value in position)


def _make_runtime_stage(path: Path, layout: dict) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    environment = UsdGeom.Xform.Define(stage, "/World/Environment").GetPrim()
    environment.GetReferences().AddReference(str(SCENE_USD))

    physics = layout["physics_validation"]
    bin_cfg = layout["bin"]
    table_z = float(layout["table"]["surface_height_m"])
    bin_height = float(bin_cfg["outer_dimensions_xyz_m"][2])
    bin_x, bin_y = map(float, bin_cfg["center_world_xy_m"])
    diameter = float(physics["diagnostic_sphere_diameter_m"])
    diagnostic = UsdGeom.Sphere.Define(stage, "/World/DiagnosticDrop")
    diagnostic.CreateRadiusAttr(diameter / 2.0)
    diagnostic.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(bin_x, bin_y, table_z + bin_height + float(physics["diagnostic_start_above_bin_rim_m"]))
    )
    diagnostic.CreateDisplayColorAttr([Gf.Vec3f(*map(float, physics["diagnostic_color_rgb"]))])
    UsdPhysics.CollisionAPI.Apply(diagnostic.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(diagnostic.GetPrim())
    UsdPhysics.MassAPI.Apply(diagnostic.GetPrim()).CreateMassAttr(float(physics["diagnostic_sphere_mass_kg"]))
    stage.GetRootLayer().Save()


def main() -> int:
    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext

    if not SCENE_USD.is_file():
        raise FileNotFoundError(f"Build the scene before physics validation: {SCENE_USD}")
    layout = json.loads((ROOT / "scene_layout.json").read_text(encoding="utf-8"))
    physics = layout["physics_validation"]
    dt = float(physics["dt_s"])
    steps = int(physics["steps"])
    settle_window = int(physics["settle_window_steps"])

    with tempfile.TemporaryDirectory(prefix="doll_handoff_physics_") as temp_dir:
        runtime_path = Path(temp_dir) / "runtime.usda"
        _make_runtime_stage(runtime_path, layout)
        context = omni.usd.get_context()
        if not context.open_stage(str(runtime_path)):
            raise RuntimeError(f"Could not open physics validation stage: {runtime_path}")
        gravity = float(physics["gravity_m_s2"])
        # USD transform readback is part of this validator, so disable Fabric's
        # USD-bypassing transform cache for the short diagnostic run.
        sim = SimulationContext(
            SimulationCfg(
                dt=dt,
                device="cpu",
                gravity=(0.0, 0.0, -gravity),
                use_fabric=False,
            )
        )
        stage = context.get_stage()

        doll_path = "/World/Environment/Doll"
        bin_root_path = "/World/Environment/TrashBin"
        diagnostic_path = "/World/DiagnosticDrop"
        bin_parts = [f"{bin_root_path}/{name}" for name in layout["bin"]["collision_parts"]]
        # Capture the authored start before reset performs its initialization step.
        doll_initial = _position(stage, doll_path)
        diagnostic_initial = _position(stage, diagnostic_path)
        sim.reset()
        bin_before = {path: _position(stage, path) for path in bin_parts}
        doll_z_window: list[float] = []
        diagnostic_z_min = diagnostic_initial[2]
        crossed_opening = False
        table_z = float(layout["table"]["surface_height_m"])
        bin_height = float(layout["bin"]["outer_dimensions_xyz_m"][2])

        for index in range(steps):
            sim.step(render=False)
            doll_position = _position(stage, doll_path)
            diagnostic_position = _position(stage, diagnostic_path)
            diagnostic_z_min = min(diagnostic_z_min, diagnostic_position[2])
            if diagnostic_position[2] < table_z + bin_height:
                crossed_opening = True
            if index >= steps - settle_window:
                doll_z_window.append(doll_position[2])

        doll_final = _position(stage, doll_path)
        diagnostic_final = _position(stage, diagnostic_path)
        bin_after = {path: _position(stage, path) for path in bin_parts}

    doll_radius = float(layout["doll"]["diameter_m"]) / 2.0
    expected_doll_z = table_z + doll_radius
    doll_finite = _finite(doll_final)
    doll_on_table = abs(doll_final[2] - expected_doll_z) <= float(physics["doll_height_tolerance_m"])
    doll_settled = bool(doll_z_window) and max(doll_z_window) - min(doll_z_window) <= float(physics["settle_motion_tolerance_m"])
    doll_fell = doll_final[2] < doll_initial[2]

    bin_stable = all(
        all(abs(before[index] - bin_after[path][index]) <= 1e-9 for index in range(3))
        for path, before in bin_before.items()
    )
    bin_cfg = layout["bin"]
    bin_x, bin_y = map(float, bin_cfg["center_world_xy_m"])
    opening_x, opening_y = map(float, bin_cfg["opening_dimensions_xy_m"])
    diagnostic_radius = float(physics["diagnostic_sphere_diameter_m"]) / 2.0
    inside_opening_xy = (
        abs(diagnostic_final[0] - bin_x) + diagnostic_radius < opening_x / 2.0
        and abs(diagnostic_final[1] - bin_y) + diagnostic_radius < opening_y / 2.0
    )
    bottom_top = table_z + float(bin_cfg["bottom_thickness_m"])
    diagnostic_inside_height = (
        diagnostic_final[2] >= bottom_top + diagnostic_radius - 0.006
        and diagnostic_final[2] < table_z + float(bin_cfg["outer_dimensions_xyz_m"][2])
    )
    diagnostic_finite = _finite(diagnostic_final)
    drop_pass = crossed_opening and inside_opening_xy and diagnostic_inside_height and diagnostic_finite

    structural_report = json.loads(
        (ROOT / "outputs" / "doll_handoff_scene" / "static_validation.json").read_text(encoding="utf-8")
    )
    opening_unobstructed = (
        structural_report["bin"]["top_collider"] is None
        and structural_report["bin"]["interior_filler_collider"] is None
        and len(structural_report["bin"]["colliders"]) == 5
    )
    checks = {
        "doll_finite": doll_finite,
        "doll_falls_to_table": doll_fell and doll_on_table,
        "doll_settled": doll_settled,
        "bin_stable": bin_stable,
        "opening_unobstructed": opening_unobstructed,
        "drop_crossed_opening": crossed_opening,
        "drop_landed_inside": drop_pass,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "doll": {
            "initial_world_xyz_m": doll_initial,
            "final_world_xyz_m": doll_final,
            "expected_settled_center_z_m": expected_doll_z,
            "settle_window_z_span_m": max(doll_z_window) - min(doll_z_window),
        },
        "bin": {
            "part_positions_before": bin_before,
            "part_positions_after": bin_after,
            "opening_dimensions_xy_m": [opening_x, opening_y],
        },
        "diagnostic_drop": {
            "persisted_in_normal_scene": False,
            "initial_world_xyz_m": diagnostic_initial,
            "final_world_xyz_m": diagnostic_final,
            "minimum_center_z_m": diagnostic_z_min,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: DOLL_HANDOFF_PHYSICS_VALIDATION")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(f"Doll-Handoff physics sanity failed: {checks}")
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
