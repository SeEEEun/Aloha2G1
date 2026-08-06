"""Author a robot-free dynamic layer without modifying the fixed MagSafe scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdPhysics

ROOT = Path(__file__).resolve().parent


def _apply_dynamic(prim: Usd.Prim, mass: float, enable_ccd: bool) -> None:
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    # PhysxSchema is loaded only after Kit starts in this pip installation.
    # Author the same registered schemas/attributes directly so export-only
    # operation also works under the conda Python interpreter.
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(bool(enable_ccd))
    prim.CreateAttribute("physxRigidBody:maxLinearVelocity", Sdf.ValueTypeNames.Float).Set(8.0)
    prim.CreateAttribute("physxRigidBody:maxAngularVelocity", Sdf.ValueTypeNames.Float).Set(25.0)
    prim.AddAppliedSchema("PhysxContactReportAPI")
    prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.0)


def _make_fixed_joint(stage: Usd.Stage, break_force: float = 5.0, break_torque: float = 0.25) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, "/MagSafeScene/MagneticJoints/AccessoryPhone")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/MagSafeScene/Phone")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/MagSafeScene/Accessory")])
    # Anchors are expressed in each body's local frame. The 0.5 mm center gap
    # is retained exactly; no body is teleported at attachment.
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.004175, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, -0.00175, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateBreakForceAttr(float(break_force))
    joint.CreateBreakTorqueAttr(float(break_torque))
    joint.GetPrim().SetCustomDataByKey("magsafe:mode", "breakable_joint")


def build_magnetic_scene(
    fixed_scene: str | Path = ROOT / "generated/magsafe_fixed_scene.usda",
    output_path: str | Path = ROOT / "generated/magsafe_magnetic_scene.usda",
    config_path: str | Path = ROOT / "magnet_config.json",
) -> Path:
    fixed = Path(fixed_scene).resolve()
    output = Path(output_path).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if output == fixed:
        raise ValueError("Magnetic output must not overwrite magsafe_fixed_scene.usda")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    root = stage.DefinePrim("/MagSafeScene", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(fixed.name)
    root.SetCustomDataByKey("magsafe:class", "magsafe_magnetic_scene")
    root.SetCustomDataByKey("magsafe:fixed_source", fixed.name)
    root.SetCustomDataByKey("magsafe:robot_included", False)
    root.SetCustomDataByKey("magsafe:parameter_status", "DEBUG_INITIAL_GUESS")

    _apply_dynamic(stage.OverridePrim("/MagSafeScene/Phone"), 0.177, config["safety"]["enable_ccd"])
    _apply_dynamic(
        stage.OverridePrim("/MagSafeScene/Accessory"),
        float(config["accessory"]["mass_kg"]),
        config["safety"]["enable_ccd"],
    )
    # Keep the existing proxy mesh and dimensions, but explicitly select the
    # dynamic-body-compatible approximation that PhysX otherwise chooses with
    # a runtime warning.
    phone_collider = stage.OverridePrim("/MagSafeScene/Phone/Colliders/Main")
    UsdPhysics.MeshCollisionAPI.Apply(phone_collider).CreateApproximationAttr("convexHull")
    for path in (
        "/MagSafeScene/Phone/Colliders/Main",
        "/MagSafeScene/Charger/Colliders/Pad",
    ):
        collider = stage.OverridePrim(path)
        collider.AddAppliedSchema("PhysxCollisionAPI")
        collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
        collider.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    # In the supplied open-stand pose the support ring passes through the
    # tabletop. Keeping those segments active on a newly dynamic body produces
    # a large depenetration impulse. Preserve every proxy prim and its geometry,
    # but disable only this already-penetrating sub-proxy in the magnetic layer.
    for index in range(12):
        support = stage.OverridePrim(f"/MagSafeScene/Accessory/Colliders/SupportRing/Segment_{index:02d}")
        UsdPhysics.CollisionAPI.Apply(support).CreateCollisionEnabledAttr(False)
    _make_fixed_joint(
        stage,
        config["accessory_phone"]["break_force_n"],
        config["accessory_phone"]["break_torque_nm"],
    )
    if int(config.get("metadata", {}).get("version", 1)) >= 2:
        _author_v2(stage, config)
    stage.GetRootLayer().Save()
    return output


def _author_v2(stage: Usd.Stage, config: dict) -> None:
    """Add portrait target, non-magnetic tags, and a stable support-foot proxy."""
    from pxr import UsdGeom

    # Explicit magnetic/non-magnetic debug metadata. Controller code still uses
    # only the two configured frame pairs and never searches collision prims.
    for path in (
        "/MagSafeScene/Table",
        "/MagSafeScene/Charger",
        "/MagSafeScene/Charger/Visuals/Base",
        "/MagSafeScene/Charger/Colliders/Base",
        "/MagSafeScene/Charger/Visuals/Support_0",
        "/MagSafeScene/Charger/Visuals/Support_1",
        "/MagSafeScene/ChargerMountPlate",
    ):
        stage.OverridePrim(path).SetCustomDataByKey("magsafe:magnetic_material", "non_magnetic")
    stage.OverridePrim("/MagSafeScene/Charger/Frames/PhoneTargetCenter").SetCustomDataByKey(
        "magsafe:magnetic_material", "magnetic_target_only"
    )
    stage.OverridePrim("/MagSafeScene/Accessory/Frames/MagneticCenter").SetCustomDataByKey(
        "magsafe:magnetic_material", "magnetic_ring"
    )

    # Lean the inherited phone/accessory assembly about the phone bottom edge.
    # The angle is deliberately small because the supplied accessory asset
    # already encodes the open support ring as part of one rigid body.
    lean = math.radians(float(config["initial_assembly"]["lean_degrees_about_world_x"]))
    rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(lean))
    quat = rot.GetQuat()
    pivot = Gf.Vec3d(0.525, 0.255, 0.795)
    phone_old = Gf.Vec3d(0.525, 0.255, 0.83075)
    acc_old = Gf.Vec3d(0.525, 0.261425, 0.83075)
    for path, old in (
        ("/MagSafeScene/Phone", phone_old),
        ("/MagSafeScene/Accessory", acc_old),
    ):
        new = pivot + rot.TransformDir(old - pivot)
        body_xf = UsdGeom.Xformable(stage.OverridePrim(path))
        body_xf.ClearXformOpOrder()
        body_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(new)
        body_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)

    # Move the target a small distance out of the physical pad face. The source
    # is the phone's MagSafeCenter, 0.2 mm outside its back surface.
    tilt = math.radians(15.0)
    pad_n = Gf.Vec3d(0.0, -math.cos(tilt), math.sin(tilt))
    old = Gf.Vec3d(0.0, 0.005846519, 0.13261811)
    clearance = float(config["charger_target"]["surface_clearance_m"])
    target = old + pad_n * clearance
    target_prim = stage.OverridePrim("/MagSafeScene/Charger/Frames/PhoneTargetCenter")
    xf = UsdGeom.Xformable(target_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(target)
    q = config["charger_target"]["target_rotation_wxyz"]
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))
    )
    target_prim.SetCustomDataByKey("magsafe:target_orientation", "portrait_full_rotation")

    # Keep the detailed support-ring proxy prims but replace their unstable
    # table interaction with one convex foot proxy at the lower ring patch.
    size = config["accessory"]["support_foot_proxy_size_xyz_m"]
    center = config["accessory"]["support_foot_proxy_center_local_xyz_m"]
    foot = UsdGeom.Cube.Define(stage, "/MagSafeScene/Accessory/Colliders/SupportFootProxy")
    foot.CreateSizeAttr(2.0)
    foot_xf = UsdGeom.Xformable(foot)
    foot_xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    foot_xf.AddScaleOp().Set(Gf.Vec3f(float(size[0]) / 2, float(size[1]) / 2, float(size[2]) / 2))
    UsdPhysics.CollisionAPI.Apply(foot.GetPrim()).CreateCollisionEnabledAttr(True)
    foot.GetPrim().SetCustomDataByKey("magsafe:class", "accessory_support_contact_proxy")
    foot.GetPrim().SetCustomDataByKey("magsafe:magnetic_material", "non_magnetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-scene", type=Path, default=ROOT / "generated/magsafe_fixed_scene.usda")
    parser.add_argument("--output", type=Path, default=ROOT / "generated/magsafe_magnetic_scene.usda")
    parser.add_argument("--config", type=Path, default=ROOT / "magnet_config.json")
    args = parser.parse_args()
    print(build_magnetic_scene(args.fixed_scene, args.output, args.config))
