#!/usr/bin/env python3
"""Build the clean, task-only Doll-Handoff USD scene.

This builder copies only the known-good table geometry from the protected scene,
removes its legacy custom metadata, then authors doll/bin/lighting/cameras locally.
It never opens any recording and never edits the source scene.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parent
LAYOUT_PATH = ROOT / "scene_layout.json"
GENERATED = ROOT / "generated"
TABLE_ASSET = GENERATED / "table_workspace.usda"
SCENE_ASSET = GENERATED / "doll_handoff_scene.usda"
BUILD_REPORT = GENERATED / "doll_handoff_build_report.json"


def _load_layout() -> dict[str, Any]:
    return json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _material(stage: Usd.Stage, path: str, color: Sequence[float], roughness: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _set_translate(prim: Usd.Prim, xyz: Sequence[float]) -> None:
    UsdGeom.Xformable(prim).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*map(float, xyz))
    )


def _cube(
    stage: Usd.Stage,
    path: str,
    size_xyz: Sequence[float],
    center_xyz: Sequence[float],
    material: UsdShade.Material,
    *,
    collision: bool,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_translate(cube.GetPrim(), center_xyz)
    cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size_xyz)))
    _bind(cube.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _sphere(
    stage: Usd.Stage,
    path: str,
    radius: float,
    material: UsdShade.Material,
    *,
    center_xyz: Sequence[float] | None = None,
    collision: bool = False,
) -> UsdGeom.Sphere:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    if center_xyz is not None:
        _set_translate(sphere.GetPrim(), center_xyz)
    _bind(sphere.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    return sphere


def _prepare_authoritative_table(layout: dict[str, Any]) -> dict[str, str]:
    source = (ROOT / layout["table"]["source"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Authoritative table asset is missing: {source}")
    actual_hash = _sha256(source)
    expected_hash = str(layout["table"]["source_asset_sha256"])
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Protected table asset does not match the audited source hash: "
            f"expected={expected_hash} actual={actual_hash} path={source}"
        )

    shutil.copy2(source, TABLE_ASSET)
    table_stage = Usd.Stage.Open(str(TABLE_ASSET))
    if table_stage is None:
        raise RuntimeError(f"Could not open copied table asset: {TABLE_ASSET}")
    for prim in table_stage.TraverseAll():
        prim.SetCustomData({})
    table_stage.GetRootLayer().Save()
    _validate_table_geometry(table_stage, layout)
    return {
        "protected_source": str(source),
        "protected_source_sha256": actual_hash,
        "local_sanitized_copy": str(TABLE_ASSET),
        "local_sanitized_sha256": _sha256(TABLE_ASSET),
    }


def _world_center_and_size(stage: Usd.Stage, path: str) -> tuple[list[float], list[float]]:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing authoritative table prim: {path}")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(prim).ComputeAlignedRange()
    lower, upper = bounds.GetMin(), bounds.GetMax()
    center = [0.5 * float(lower[i] + upper[i]) for i in range(3)]
    size = [float(upper[i] - lower[i]) for i in range(3)]
    return center, size


def _assert_vector(label: str, actual: Sequence[float], expected: Sequence[float], tolerance: float = 1e-6) -> None:
    if len(actual) != len(expected) or any(abs(float(a) - float(e)) > tolerance for a, e in zip(actual, expected)):
        raise RuntimeError(f"{label} mismatch: expected={list(expected)} actual={list(actual)}")


def _validate_table_geometry(stage: Usd.Stage, layout: dict[str, Any]) -> None:
    table = layout["table"]
    frame = layout["black_frame"]
    top_center, top_size = _world_center_and_size(stage, "/OpticalTable/Visuals/Top")
    sx, sy = map(float, table["size_xy_m"])
    top_t = float(table["top_thickness_m"])
    surface_z = float(table["surface_height_m"])
    _assert_vector("table size", top_size, [sx, sy, top_t])
    _assert_vector("table center", top_center, [sx / 2.0, sy / 2.0, surface_z - top_t / 2.0])
    for name, values in frame["rails"].items():
        center, size = _world_center_and_size(stage, f"/OpticalTable/Visuals/Frame{name.capitalize()}")
        _assert_vector(f"black frame {name} size", size, values["size_xyz_m"])
        _assert_vector(f"black frame {name} center", center, values["center_xyz_m"])


def _author_camera(stage: Usd.Stage, path: str, values: dict[str, Any], camera_cfg: dict[str, Any]) -> None:
    eye = Gf.Vec3d(*map(float, values["eye_world_xyz_m"]))
    target = Gf.Vec3d(*map(float, values["target_world_xyz_m"]))
    view = Gf.Matrix4d(1.0).SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
    camera = UsdGeom.Camera.Define(stage, path)
    camera.MakeMatrixXform().Set(view.GetInverse())
    camera.CreateFocalLengthAttr(float(camera_cfg["focal_length_mm"]))
    camera.CreateClippingRangeAttr(Gf.Vec2f(*map(float, camera_cfg["clipping_range_m"])))


def _build_scene(layout: dict[str, Any]) -> None:
    stage = Usd.Stage.CreateNew(str(SCENE_ASSET))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/DollHandoffScene").GetPrim()
    root.SetCustomDataByKey("task", "doll_handoff")
    root.SetCustomDataByKey("layout", "scene_layout.json")
    stage.SetDefaultPrim(root)

    table = UsdGeom.Xform.Define(stage, "/DollHandoffScene/Table").GetPrim()
    table.GetReferences().AddReference(TABLE_ASSET.name)

    looks = "/DollHandoffScene/Looks"
    green = _material(stage, f"{looks}/DollGreen", layout["doll"]["body_color_rgb"], 0.82)
    black = _material(stage, f"{looks}/DollEyes", layout["doll"]["eye_color_rgb"], 0.62)
    off_white = _material(stage, f"{looks}/BinOffWhite", layout["bin"]["color_rgb"], 0.72)
    ground_mat = _material(stage, f"{looks}/Ground", layout["ground"]["color_rgb"], 0.9)

    ground_cfg = layout["ground"]
    ground_sx, ground_sy = map(float, ground_cfg["size_xy_m"])
    ground_t = float(ground_cfg["thickness_m"])
    ground_z = float(ground_cfg["surface_height_m"])
    _cube(
        stage,
        "/DollHandoffScene/Ground",
        [ground_sx, ground_sy, ground_t],
        [float(layout["table"]["size_xy_m"][0]) / 2.0, 0.0, ground_z - ground_t / 2.0],
        ground_mat,
        collision=True,
    )

    table_z = float(layout["table"]["surface_height_m"])
    task_frame = UsdGeom.Xform.Define(stage, "/DollHandoffScene/TaskFrame").GetPrim()
    _set_translate(task_frame, layout["task_frame"]["origin_world_xyz_m"])
    task_frame.SetCustomDataByKey("axes", "+X right, +Y back, +Z up")

    doll_cfg = layout["doll"]
    diameter = float(doll_cfg["diameter_m"])
    radius = diameter / 2.0
    doll_x, doll_y = map(float, doll_cfg["center_world_xy_m"])
    doll_z = table_z + radius + float(doll_cfg["initial_table_clearance_m"])
    doll = UsdGeom.Xform.Define(stage, "/DollHandoffScene/Doll").GetPrim()
    _set_translate(doll, [doll_x, doll_y, doll_z])
    doll.SetCustomDataByKey("dimensions_status", str(doll_cfg["status"]))
    UsdPhysics.RigidBodyAPI.Apply(doll)
    UsdPhysics.MassAPI.Apply(doll).CreateMassAttr(float(doll_cfg["mass_kg"]))
    _sphere(stage, "/DollHandoffScene/Doll/Body", radius, green, collision=True)
    for index, center in enumerate(doll_cfg["nub_centers_local_xyz_m"]):
        _sphere(
            stage,
            f"/DollHandoffScene/Doll/Nub{index + 1}",
            float(doll_cfg["nub_radius_m"]),
            green,
            center_xyz=center,
        )
    for index, center in enumerate(doll_cfg["eye_centers_local_xyz_m"]):
        _sphere(
            stage,
            f"/DollHandoffScene/Doll/Eye{index + 1}",
            float(doll_cfg["eye_radius_m"]),
            black,
            center_xyz=center,
        )

    bin_cfg = layout["bin"]
    outer_x, outer_y, outer_z = map(float, bin_cfg["outer_dimensions_xyz_m"])
    wall_t = float(bin_cfg["wall_thickness_m"])
    bottom_t = float(bin_cfg["bottom_thickness_m"])
    bin_x, bin_y = map(float, bin_cfg["center_world_xy_m"])
    bin_root = UsdGeom.Xform.Define(stage, "/DollHandoffScene/TrashBin").GetPrim()
    _set_translate(bin_root, [bin_x, bin_y, table_z])
    bin_root.SetCustomDataByKey("container", "open_top_five_part_static_bin")
    bin_root.SetCustomDataByKey("dimensions_status", str(bin_cfg["status"]))
    wall_height = outer_z - bottom_t
    wall_center_z = bottom_t + wall_height / 2.0
    _cube(stage, "/DollHandoffScene/TrashBin/Bottom", [outer_x, outer_y, bottom_t], [0.0, 0.0, bottom_t / 2.0], off_white, collision=True)
    _cube(stage, "/DollHandoffScene/TrashBin/FrontWall", [outer_x, wall_t, wall_height], [0.0, -(outer_y - wall_t) / 2.0, wall_center_z], off_white, collision=True)
    _cube(stage, "/DollHandoffScene/TrashBin/BackWall", [outer_x, wall_t, wall_height], [0.0, (outer_y - wall_t) / 2.0, wall_center_z], off_white, collision=True)
    _cube(stage, "/DollHandoffScene/TrashBin/LeftWall", [wall_t, outer_y - 2.0 * wall_t, wall_height], [-(outer_x - wall_t) / 2.0, 0.0, wall_center_z], off_white, collision=True)
    _cube(stage, "/DollHandoffScene/TrashBin/RightWall", [wall_t, outer_y - 2.0 * wall_t, wall_height], [(outer_x - wall_t) / 2.0, 0.0, wall_center_z], off_white, collision=True)

    frames = {
        "TableFrontEdgeCenter": [float(layout["table"]["size_xy_m"][0]) / 2.0, 0.0, table_z],
        "DollInitialCenter": [doll_x, doll_y, doll_z],
        "TrashBinCenter": [bin_x, bin_y, table_z + outer_z / 2.0],
        "TrashBinOpeningCenter": [bin_x, bin_y, table_z + outer_z],
    }
    for name, xyz in frames.items():
        prim = UsdGeom.Xform.Define(stage, f"/DollHandoffScene/Frames/{name}").GetPrim()
        _set_translate(prim, xyz)

    light_cfg = layout["lighting"]
    dome = UsdLux.DomeLight.Define(stage, "/DollHandoffScene/Lights/Dome")
    dome.CreateIntensityAttr(float(light_cfg["dome"]["intensity"]))
    dome.CreateColorAttr(Gf.Vec3f(*map(float, light_cfg["dome"]["color_rgb"])))
    for name in ("key", "fill"):
        values = light_cfg[name]
        light = UsdLux.SphereLight.Define(stage, f"/DollHandoffScene/Lights/{name.capitalize()}")
        light.CreateIntensityAttr(float(values["intensity"]))
        light.CreateRadiusAttr(float(values["radius_m"]))
        light.CreateColorAttr(Gf.Vec3f(*map(float, values["color_rgb"])))
        _set_translate(light.GetPrim(), values["position_world_xyz_m"])

    camera_cfg = layout["camera"]
    for name, values in camera_cfg["presets"].items():
        _author_camera(stage, f"/DollHandoffScene/Cameras/{name}", values, camera_cfg)

    stage.GetRootLayer().Save()


def main() -> int:
    GENERATED.mkdir(parents=True, exist_ok=True)
    layout = _load_layout()
    if layout.get("task") != "doll_handoff":
        raise RuntimeError(f"Unexpected task in {LAYOUT_PATH}: {layout.get('task')!r}")
    table_report = _prepare_authoritative_table(layout)
    _build_scene(layout)
    report = {
        "status": "PASS",
        "task": "doll_handoff",
        "layout": str(LAYOUT_PATH),
        "scene_usd": str(SCENE_ASSET),
        "scene_usd_sha256": _sha256(SCENE_ASSET),
        "table": table_report,
        "recording_dependency": None,
    }
    BUILD_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: DOLL_HANDOFF_SCENE_BUILT")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
