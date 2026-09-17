"""Shared composition and viewer helpers for Doll-Handoff robot previews."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

from pxr import Gf, Usd, UsdGeom


ROOT = Path(__file__).resolve().parent
LAYOUT_PATH = ROOT / "scene_layout.json"
SCENE_USD = ROOT / "generated" / "doll_handoff_scene.usda"


def load_layout() -> dict[str, Any]:
    return json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))


def _root_pose(robot_cfg: dict[str, Any]) -> tuple[list[float], list[float]]:
    position = [float(value) for value in robot_cfg["root_position_world_xyz_m"]]
    orientation = [float(value) for value in robot_cfg["root_orientation_world_wxyz"]]
    if len(position) != 3 or len(orientation) != 4:
        raise ValueError("Robot root pose must contain XYZ and WXYZ values")
    norm = math.sqrt(sum(value * value for value in orientation))
    if not math.isclose(norm, 1.0, abs_tol=1e-8):
        raise ValueError(f"Robot root quaternion is not normalized: norm={norm}")
    return position, orientation


def compose_robot_preview(output: Path, robot_key: str, robot_prim_name: str) -> Usd.Stage:
    layout = load_layout()
    if not SCENE_USD.is_file():
        raise FileNotFoundError(f"Build the Doll-Handoff scene first: {SCENE_USD}")
    robot_cfg = layout[robot_key]
    robot_usd = Path(robot_cfg["asset_usd"])
    if not robot_usd.is_file():
        raise FileNotFoundError(f"Robot asset is missing: {robot_usd}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world.SetCustomDataByKey("task", "doll_handoff")
    stage.SetDefaultPrim(world)

    environment = UsdGeom.Xform.Define(stage, "/World/DollHandoffEnvironment").GetPrim()
    environment.GetReferences().AddReference(SCENE_USD.name)
    robot = UsdGeom.Xform.Define(stage, f"/World/{robot_prim_name}")
    robot_asset = UsdGeom.Xform.Define(stage, f"/World/{robot_prim_name}/Asset").GetPrim()
    robot_asset.GetReferences().AddReference(str(robot_usd))
    position, orientation = _root_pose(robot_cfg)
    robot.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
    robot.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))
    )
    robot.GetPrim().SetCustomDataByKey("task", "doll_handoff")
    robot.GetPrim().SetCustomDataByKey("static_preview", True)

    if robot_key == "aloha":
        _suppress_imported_aloha_fixture(stage, f"/World/{robot_prim_name}/Asset")
    stage.GetRootLayer().Save()
    return stage


def _suppress_imported_aloha_fixture(stage: Usd.Stage, asset_path: str) -> None:
    base = f"{asset_path}/Geometry/tabletop_link"
    fixture_names = [
        "tabletop_link",
        "tabletop",
        "frame_link",
        "cam_high_mount_link",
        "cam_low_mount_link",
        "cam_high_color_frame",
        "cam_low_color_frame",
        "Box",
        *(f"Box_{index}" for index in range(1, 13)),
    ]
    for name in fixture_names:
        stage.OverridePrim(f"{base}/{name}").SetActive(False)


def point_to_segment_distance_xy(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    px, py = map(float, point[:2])
    ax, ay = map(float, a[:2])
    bx, by = map(float, b[:2])
    vx, vy = bx - ax, by - ay
    denominator = vx * vx + vy * vy
    if denominator <= 0.0:
        raise ValueError("Table-front segment has zero length")
    amount = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denominator))
    closest_x, closest_y = ax + amount * vx, ay + amount * vy
    return math.hypot(px - closest_x, py - closest_y)


def g1_geometry_report(stage: Usd.Stage, robot_prim_name: str = "G1") -> dict[str, Any]:
    layout = load_layout()
    pelvis_path = f"/World/{robot_prim_name}/Asset/{layout['g1']['pelvis_path_below_asset']}"
    pelvis = stage.GetPrimAtPath(pelvis_path)
    if not pelvis:
        raise RuntimeError(f"G1 pelvis prim is missing: {pelvis_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    center = cache.GetLocalToWorldTransform(pelvis).ExtractTranslation()
    center_xyz = [float(center[index]) for index in range(3)]
    front_a, front_b = layout["table"]["front_edge_world_xy_m"]
    actual = point_to_segment_distance_xy(center_xyz, front_a, front_b)
    target = float(layout["g1"]["pelvis_to_table_front_target_m"])
    error = actual - target

    root = stage.GetPrimAtPath(f"/World/{robot_prim_name}")
    root_matrix = cache.GetLocalToWorldTransform(root)
    forward = root_matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized()
    report = {
        "scene": "doll_handoff",
        "pelvis_path": pelvis_path,
        "pelvis_center_world_xyz_m": center_xyz,
        "table_front_edge_world_xy_m": [front_a, front_b],
        "pelvis_to_table_front_gap_m": actual,
        "target_m": target,
        "error_m": error,
        "g1_forward_world_xyz": [float(forward[index]) for index in range(3)],
        "facing_table": float(forward[1]) > 0.999,
    }
    tolerance = float(layout["g1"]["pelvis_to_table_front_tolerance_m"])
    if abs(error) > tolerance:
        raise RuntimeError(f"G1 pelvis/table-front gap error exceeds tolerance: {report}")
    if not report["facing_table"]:
        raise RuntimeError(f"G1 is not facing the table: {report}")
    return report


def print_task_report(stage: Usd.Stage, robot_path: str, robot_usd: Path) -> None:
    layout = load_layout()
    doll = layout["doll"]["center_world_xy_m"]
    bin_center = layout["bin"]["center_world_xy_m"]
    referenced = Usd.Stage.Open(str(robot_usd))
    if referenced is None:
        raise RuntimeError(f"Could not inspect robot asset: {robot_usd}")
    robot = stage.GetPrimAtPath(robot_path)
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(robot).ComputeAlignedRange()
    print("[DOLL_HANDOFF] scene=doll_handoff", flush=True)
    print(f"[DOLL_HANDOFF] doll_center_world_xy_m={doll}", flush=True)
    print(f"[DOLL_HANDOFF] trash_bin_center_world_xy_m={bin_center}", flush=True)
    print(f"[PREVIEW] robot_prim={robot_path} asset_root_prim={referenced.GetDefaultPrim().GetPath()}", flush=True)
    print(f"[PREVIEW] world_bound_min={tuple(bounds.GetMin())}", flush=True)
    print(f"[PREVIEW] world_bound_max={tuple(bounds.GetMax())}", flush=True)
    print("[PREVIEW] trajectory=OFF episode=OFF joint_control=OFF IK=OFF retargeting=OFF", flush=True)


def run_viewer(
    simulation_app,
    stage_path: Path,
    camera_name: str,
    hold_seconds: float | None,
    screenshot: Path | None,
) -> None:
    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext

    layout = load_layout()
    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"Could not open preview stage: {stage_path}")
    sim = SimulationContext(SimulationCfg(device="cpu"))
    # IsaacLab 3 beta initializes visualizers lazily at reset.  A static model
    # preview must not reset/step physics, so initialize only the visualizer.
    sim.initialize_visualizers()
    preset = layout["camera"]["presets"][camera_name]
    eye = tuple(map(float, preset["eye_world_xyz_m"]))
    target = tuple(map(float, preset["target_world_xyz_m"]))
    sim.set_camera_view(eye, target)
    print(f"[PREVIEW] camera={camera_name} eye={eye} target={target}", flush=True)

    for _ in range(30):
        if simulation_app.is_running():
            simulation_app.update()

    if screenshot is not None:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        screenshot = screenshot.expanduser().resolve()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active Kit viewport is available for screenshot capture")
        capture_viewport_to_file(viewport, file_path=str(screenshot))
        deadline = time.monotonic() + 30.0
        while simulation_app.is_running() and time.monotonic() < deadline and not screenshot.is_file():
            simulation_app.update()
        if not screenshot.is_file() or screenshot.stat().st_size == 0:
            raise RuntimeError(f"Viewport screenshot was not written: {screenshot}")
        print(f"[PREVIEW] screenshot={screenshot}", flush=True)

    if hold_seconds is None:
        while simulation_app.is_running():
            simulation_app.update()
    else:
        deadline = time.monotonic() + max(float(hold_seconds), 0.0)
        while simulation_app.is_running() and time.monotonic() < deadline:
            simulation_app.update()
