"""Shared stage composition for static/default-pose robot model previews."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parent
MAGSAFE_USD = ROOT / "generated" / "magsafe_fixed_scene.usda"
POSE_CONFIG = ROOT / "magsafe_robot_preview_config.json"

CAMERAS = {
    "overview": ((1.55, -1.65, 1.55), (0.42, 0.18, 0.75)),
    "front": ((0.42, -2.15, 1.05), (0.42, 0.20, 0.78)),
    "side": ((1.90, 0.20, 1.05), (0.35, 0.20, 0.78)),
    "top": ((0.42, 0.20, 2.70), (0.42, 0.20, 0.0)),
}


def load_pose(key: str) -> tuple[list[float], list[float]]:
    data = json.loads(POSE_CONFIG.read_text())
    pose = data[key]
    position = [float(value) for value in pose["position_xyz_m"]]
    orientation = [float(value) for value in pose["orientation_wxyz"]]
    if len(position) != 3 or len(orientation) != 4:
        raise ValueError(f"Invalid pose entry for {key!r} in {POSE_CONFIG}")
    return position, orientation


def task_geometry(stage: Usd.Stage, root_position: list[float]) -> dict[str, object]:
    """Read authoritative task-frame poses and derive robot approach axes."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def world_position(path: str) -> list[float]:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"Required authoritative task frame is missing: {path}")
        p = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        return [float(p[0]), float(p[1]), float(p[2])]

    phone = world_position("/World/MagSafeScene/Frames/PhoneInitialCenter")
    accessory = world_position("/World/MagSafeScene/Frames/AccessoryInitialCenter")
    charger = world_position("/World/MagSafeScene/Frames/ChargerBaseCenter")
    task_center = [0.5 * (phone[i] + accessory[i]) for i in range(3)]
    dx, dy = task_center[0] - root_position[0], task_center[1] - root_position[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        raise RuntimeError("G1 root and task center have identical horizontal positions")
    forward = [dx / distance, dy / distance, 0.0]
    lateral = [-forward[1], forward[0], 0.0]
    return {
        "phone_center_world": phone, "accessory_center_world": accessory,
        "charger_center_world": charger, "task_center_world": task_center,
        "table_forward_direction": forward, "table_lateral_direction": lateral,
        "root_to_task_center_horizontal_distance_m": distance,
    }


def offset_root_pose(
    stage: Usd.Stage, position: list[float], orientation: list[float], *,
    forward_offset_m: float = 0.0, lateral_offset_m: float = 0.0,
    z_offset_m: float = 0.0, yaw_offset_deg: float = 0.0,
) -> tuple[list[float], list[float], dict[str, object]]:
    geometry = task_geometry(stage, position)
    forward = geometry["table_forward_direction"]
    lateral = geometry["table_lateral_direction"]
    shifted = [
        position[i] + forward_offset_m * forward[i] + lateral_offset_m * lateral[i]
        for i in range(3)
    ]
    shifted[2] += z_offset_m
    q = Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))
    yaw = Gf.Quatd(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(yaw_offset_deg)).GetQuat())
    out = yaw * q
    imag = out.GetImaginary()
    shifted_q = [float(out.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]
    geometry.update({"original_root_position": list(position), "applied_root_position": shifted,
                     "root_forward_offset_m": forward_offset_m,
                     "root_lateral_offset_m": lateral_offset_m,
                     "root_z_offset_m": z_offset_m, "root_yaw_offset_deg": yaw_offset_deg})
    return shifted, shifted_q, geometry


def compose_stage(
    output: Path, robot_prim_name: str, robot_usd: Path, pose_key: str,
    sublayers: tuple[Path, ...] = (),
    *, forward_offset_m: float = 0.0, lateral_offset_m: float = 0.0,
    z_offset_m: float = 0.0, yaw_offset_deg: float = 0.0,
) -> Usd.Stage:
    """Create a two-reference stage without editing either referenced asset."""
    if not MAGSAFE_USD.is_file():
        raise FileNotFoundError(MAGSAFE_USD)
    if not robot_usd.is_file():
        raise FileNotFoundError(robot_usd)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [str(path) for path in sublayers]
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    magsafe = UsdGeom.Xform.Define(stage, "/World/MagSafeScene").GetPrim()
    magsafe.GetReferences().AddReference(str(MAGSAFE_USD))
    robot_xform = UsdGeom.Xform.Define(stage, f"/World/{robot_prim_name}")
    robot_asset = UsdGeom.Xform.Define(stage, f"/World/{robot_prim_name}/Asset").GetPrim()
    robot_asset.GetReferences().AddReference(str(robot_usd))
    position, q = load_pose(pose_key)
    position, q, geometry = offset_root_pose(
        stage, position, q, forward_offset_m=forward_offset_m,
        lateral_offset_m=lateral_offset_m, z_offset_m=z_offset_m,
        yaw_offset_deg=yaw_offset_deg,
    )
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    robot_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3]))
    )
    robot_xform.GetPrim().SetCustomDataByKey("preview_pose_source", str(POSE_CONFIG))
    robot_xform.GetPrim().SetCustomDataByKey("preview_control_applied", False)
    robot_xform.GetPrim().SetCustomDataByKey("workspace_calibration", json.dumps(geometry))
    print("[G1_WORKSPACE_GEOMETRY] " + json.dumps(geometry, sort_keys=True), flush=True)
    stage.GetRootLayer().Save()
    return stage


def print_stage_report(stage: Usd.Stage, robot_path: str, robot_usd: Path) -> None:
    robot = stage.GetPrimAtPath(robot_path)
    referenced = Usd.Stage.Open(str(robot_usd))
    default_path = referenced.GetDefaultPrim().GetPath()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_range = cache.ComputeWorldBound(robot).ComputeAlignedRange()
    print(f"[PREVIEW] robot_prim={robot_path} asset_root_prim={default_path}", flush=True)
    print(f"[PREVIEW] world_bound_min={tuple(world_range.GetMin())}")
    print(f"[PREVIEW] world_bound_max={tuple(world_range.GetMax())}")
    print("[PREVIEW] trajectory=OFF episode=OFF npz=OFF joint_control=OFF IK=OFF retargeting=OFF")


def suppress_stationary_aloha_fixture(stage: Usd.Stage) -> None:
    """Hide the MJCF's own table fixture in this composition, leaving both arms intact."""
    base = "/World/StationaryALOHA/Asset/Geometry/tabletop_link"
    fixture_names = [
        "tabletop_link",
        "tabletop",
        "frame_link",
        "cam_high_mount_link",
        "cam_low_mount_link",
        "cam_high_color_frame",
        "cam_low_color_frame",
        *(f"Box_{index}" for index in range(1, 13)),
        "Box",
    ]
    for name in fixture_names:
        stage.OverridePrim(f"{base}/{name}").SetActive(False)
    stage.GetRootLayer().Save()


def run_viewer(
    simulation_app,
    stage_path: Path,
    camera_name: str,
    hold_seconds: float | None,
    initialize_scene=None,
) -> None:
    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"Could not open preview stage: {stage_path}")
    sim = SimulationContext(SimulationCfg(device="cpu"))
    if initialize_scene is not None:
        initialize_scene(sim)
    eye, target = CAMERAS[camera_name]
    sim.set_camera_view(eye, target)
    print(f"[PREVIEW] camera={camera_name} eye={eye} target={target}")
    # The physics timeline is intentionally never started or stepped. This keeps
    # every articulation at its authored default pose and applies no robot control.
    if hold_seconds is None:
        while simulation_app.is_running():
            simulation_app.update()
    else:
        import time

        deadline = time.monotonic() + max(hold_seconds, 0.0)
        while simulation_app.is_running() and time.monotonic() < deadline:
            simulation_app.update()
