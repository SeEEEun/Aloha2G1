#!/usr/bin/env python3
"""Render the static photo-calibrated LEFT Dex3 fingertip pinch in Isaac Lab.

This is a zero-physics-step, hand-calibration renderer.  The articulation is
placed at one fixed documented arm/wrist pose, only the calibrated left Dex3 q
is applied, and the authoritative phone asset is registered to the two distal
pad frames for visual review.  It never edits v17.2 or sends hardware commands.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SCENE = ROOT / "isaaclab_magsafe_fixed_scene"
DEFAULT_OUT = ROOT / (
    "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
)
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
PHONE_USD = SCENE / "generated/phone_landscape.usda"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--calibration", type=Path, default=DEFAULT_OUT / "left_phone_fingertip_pinch_calibration.npz")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
parser.add_argument("--width", type=int, default=1200)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--pause", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.gui:
    args.headless = False
    args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def dump(path: Path, payload) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=convert, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    import carb
    import cv2
    import omni.usd
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Usd, UsdGeom, UsdLux
    from scipy.spatial.transform import Rotation
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.calibration.resolve(), allow_pickle=False) as archive:
        left_names = archive["left_dex3_joint_names"].astype(str).tolist()
        left_q = archive["left_dex3_q"].astype(np.float32)
        phone_center_wrist = archive["phone_center_wrist"].astype(np.float64)
        phone_rotation_wrist = archive["phone_rotation_wrist"].astype(np.float64)
        thumb_wrist = archive["thumb_pad_wrist"].astype(np.float64)
        index_wrist = archive["index_pad_wrist"].astype(np.float64)
        third_wrist = archive["third_pad_wrist"].astype(np.float64)

    fixed_arm = {
        "left_shoulder_pitch_joint": -0.70,
        "left_shoulder_roll_joint": 0.40,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 1.10,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
    }
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", True)
    settings.set_float_array(
        "/rtx/post/backgroundZeroAlpha/backgroundDefaultColor", [0.97, 0.97, 0.97, 1.0]
    )
    settings.set_float("/rtx/post/tonemap/filmIso", 160.0)
    settings.set_float("/rtx/post/tonemap/exposureTime", 1.0 / 60.0)
    settings.set_float("/rtx/post/tonemap/fNumber", 8.0)
    settings.set_bool("/rtx/post/histogram/enabled", False)

    stage_path = out / "left_phone_pinch_static_review.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.Xform.Define(stage, "/World/G1")
    asset = UsdGeom.Xform.Define(stage, "/World/G1/Asset").GetPrim()
    asset.GetReferences().AddReference(str(G1_USD))
    phone = UsdGeom.Xform.Define(stage, "/World/DiagnosticPhone")
    phone.GetPrim().GetReferences().AddReference(str(PHONE_USD))
    markers = UsdGeom.Xform.Define(stage, "/World/IdentityMarkers")
    markers.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/PaperWhiteDome")
    dome.CreateIntensityAttr(1350.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/PaperWhiteKey")
    key.CreateIntensityAttr(3200.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.985, 0.965))
    key.CreateAngleAttr(3.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 24.0, 18.0))
    fill = UsdLux.SphereLight.Define(stage, "/World/Lights/PaperWhiteFill")
    fill.CreateIntensityAttr(900.0)
    fill.CreateColorAttr(Gf.Vec3f(0.94, 0.97, 1.0))
    fill.CreateRadiusAttr(0.35)
    UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(0.5, 0.15, 1.25))
    stage.GetRootLayer().Save()
    if not omni.usd.get_context().open_stage(str(stage_path)):
        raise RuntimeError(stage_path)
    live_stage = omni.usd.get_context().get_stage()

    sim = SimulationContext(SimulationCfg(device="cuda:0", use_fabric=False))
    controlled = list(fixed_arm) + left_names
    robot = Articulation(ArticulationCfg(
        prim_path="/World/G1/Asset/root_joint", spawn=None,
        actuators={
            "static_review": ImplicitActuatorCfg(
                joint_names_expr=controlled,
                effort_limit_sim=25.0, velocity_limit_sim=12.0,
                stiffness=100.0, damping=5.0,
            )
        },
    ))
    views = (
        "front_oblique", "back_oblique", "top", "side",
        "fingertip_closeup", "palm_side", "third_finger_check", "identity_overlay",
    )
    cameras = {
        name: Camera(CameraCfg(
            prim_path=f"/World/Camera_{name}", update_period=0,
            height=args.height, width=args.width, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=50.0 if name != "fingertip_closeup" else 65.0,
                clipping_range=(0.02, 20.0),
            ),
        )) for name in views
    }
    print("[PINCH-RENDER] cameras constructed", flush=True)
    sim.reset()
    print("[PINCH-RENDER] simulation reset", flush=True)
    runtime_names = list(robot.data.joint_names)
    print(f"[PINCH-RENDER] runtime joint names {len(runtime_names)}", flush=True)
    missing = [name for name in controlled if name not in runtime_names]
    if missing:
        raise RuntimeError(f"missing controlled joints {missing}")
    ids = [runtime_names.index(name) for name in controlled]
    print(f"[PINCH-RENDER] controlled ids {ids}", flush=True)
    values = np.r_[list(fixed_arm.values()), left_q].astype(np.float32)
    print(f"[PINCH-RENDER] requested values {values.tolist()} device={robot.device}", flush=True)
    requested = torch.as_tensor(values[None], dtype=torch.float32, device=robot.device)
    print("[PINCH-RENDER] requested tensor created", flush=True)
    target_full = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    id_tensor = torch.as_tensor(ids, dtype=torch.long, device=robot.device)
    target_full.index_copy_(1, id_tensor, requested)
    velocity_full = torch.zeros_like(target_full)
    print("[PINCH-RENDER] full articulation state tensor created", flush=True)
    robot.write_joint_state_to_sim(target_full, velocity_full)
    print("[PINCH-RENDER] full joint state written", flush=True)
    sim.physics_manager.get_physics_sim_view().update_articulations_kinematic()
    print("[PINCH-RENDER] kinematic articulation update", flush=True)
    sim.forward()
    print("[PINCH-RENDER] sim forward", flush=True)
    robot.update(sim.get_physics_dt())
    print("[PINCH-RENDER] robot update", flush=True)
    torch.cuda.synchronize()
    print("[PINCH-RENDER] articulation state applied", flush=True)
    actual = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().copy()
    if float(np.max(np.abs(actual - values))) > 1e-5:
        raise RuntimeError("static articulation readback mismatch")

    body_names = list(robot.data.body_names)
    wrist_id = body_names.index("left_wrist_yaw_link")
    wrist_p = robot.data.body_pos_w.torch[0, wrist_id].detach().cpu().numpy().astype(float)
    wrist_q = robot.data.body_quat_w.torch[0, wrist_id].detach().cpu().numpy().astype(float)
    wrist_r = Rotation.from_quat(wrist_q).as_matrix()
    phone_p = wrist_p + wrist_r @ phone_center_wrist
    phone_r = wrist_r @ phone_rotation_wrist

    phone_op = UsdGeom.Xformable(live_stage.GetPrimAtPath("/World/DiagnosticPhone")).MakeMatrixXform()
    phone_matrix = np.eye(4)
    phone_matrix[:3, :3] = phone_r
    phone_matrix[:3, 3] = phone_p
    phone_op.Set(Gf.Matrix4d(*phone_matrix.T.reshape(-1).tolist()))

    asset_prim = live_stage.GetPrimAtPath("/World/G1/Asset")
    world_from_asset = np.asarray(
        UsdGeom.Xformable(asset_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=float,
    ).T
    asset_from_world = np.linalg.inv(world_from_asset)
    body_ops = {}
    for body_name in body_names:
        prim = live_stage.GetPrimAtPath(f"/World/G1/Asset/{body_name}")
        if not prim.IsValid():
            raise RuntimeError(f"missing body prim {body_name}")
        body_ops[body_name] = UsdGeom.Xformable(prim).MakeMatrixXform()
    body_positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy().astype(float)
    body_quaternions = robot.data.body_quat_w.torch[0].detach().cpu().numpy().astype(float)
    for body_id, body_name in enumerate(body_names):
        world_from_body = np.eye(4)
        world_from_body[:3, :3] = Rotation.from_quat(body_quaternions[body_id]).as_matrix()
        world_from_body[:3, 3] = body_positions[body_id]
        body_ops[body_name].Set(Gf.Matrix4d(*(
            asset_from_world @ world_from_body
        ).T.reshape(-1).tolist()))
    print("[PINCH-RENDER] actual articulation transforms synchronized", flush=True)

    pad_world = {
        "thumb": wrist_p + wrist_r @ thumb_wrist,
        "index": wrist_p + wrist_r @ index_wrist,
        "third": wrist_p + wrist_r @ third_wrist,
    }
    focus = 0.5 * (pad_world["thumb"] + pad_world["index"])
    # Camera offsets are visual-only and derived from the solved hand focus.
    long_axis, thickness_axis, short_axis = phone_r.T
    palm_view = wrist_p - focus
    palm_view /= np.linalg.norm(palm_view)
    offsets = {
        "front_oblique": -0.30 * thickness_axis + 0.14 * short_axis + 0.08 * long_axis,
        "back_oblique": 0.30 * thickness_axis + 0.14 * short_axis + 0.08 * long_axis,
        "top": np.array([0.02, 0.02, 0.42]),
        "side": 0.38 * long_axis + 0.06 * thickness_axis,
        "fingertip_closeup": -0.22 * long_axis + 0.10 * thickness_axis + 0.05 * short_axis,
        "palm_side": -0.28 * palm_view + 0.16 * long_axis + 0.08 * short_axis,
        "third_finger_check": 0.29 * thickness_axis - 0.14 * short_axis,
        "identity_overlay": -0.25 * long_axis + 0.07 * short_axis,
    }
    for name, camera in cameras.items():
        target_view = focus if name != "third_finger_check" else 0.5 * (focus + pad_world["third"])
        eye = target_view + offsets[name]
        camera.set_world_poses_from_view(
            np.asarray([eye], np.float32), np.asarray([target_view], np.float32)
        )
    if args.gui:
        sim.set_camera_view((focus + offsets["front_oblique"]).tolist(), focus.tolist())

    overlay_view = offsets["identity_overlay"] / np.linalg.norm(offsets["identity_overlay"])
    marker_positions = {name: point + 0.012 * overlay_view for name, point in pad_world.items()}

    def sphere(path: str, position: np.ndarray, color: tuple[float, float, float]):
        primitive = UsdGeom.Sphere.Define(live_stage, path)
        primitive.CreateRadiusAttr(0.007)
        primitive.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdGeom.Xformable(primitive).AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))

    sphere("/World/IdentityMarkers/Thumb", marker_positions["thumb"], (0.0, 0.85, 1.0))
    sphere("/World/IdentityMarkers/Index", marker_positions["index"], (1.0, 0.15, 0.75))
    sphere("/World/IdentityMarkers/Third", marker_positions["third"], (1.0, 0.75, 0.0))
    line = UsdGeom.BasisCurves.Define(live_stage, "/World/IdentityMarkers/PinchAxis")
    line.CreateTypeAttr(UsdGeom.Tokens.linear)
    line.CreateCurveVertexCountsAttr([2])
    line.CreatePointsAttr([Gf.Vec3f(*marker_positions["thumb"]), Gf.Vec3f(*marker_positions["index"])])
    line.CreateWidthsAttr([0.005, 0.005])
    line.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.3)])

    output_names = {
        "front_oblique": "left_phone_pinch_front_oblique.png",
        "back_oblique": "left_phone_pinch_back_oblique.png",
        "top": "left_phone_pinch_top.png",
        "side": "left_phone_pinch_side.png",
        "fingertip_closeup": "left_phone_pinch_fingertip_closeup.png",
        "palm_side": "left_phone_pinch_palm_side.png",
        "third_finger_check": "left_phone_pinch_third_finger_check.png",
        "identity_overlay": "left_phone_pinch_identity_overlay.png",
    }
    render_records = {}
    for name in views:
        markers.GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if name == "identity_overlay" else UsdGeom.Tokens.invisible
        )
        sim.forward()
        sim.render()
        sim.render_context.reset_transform_cadence()
        cameras[name].update(sim.get_physics_dt(), force_recompute=True)
        rgb = cameras[name].data.output["rgb"].torch[0, ..., :3].detach().cpu().numpy().copy()
        image = Image.fromarray(rgb)
        if name == "identity_overlay":
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            draw.rounded_rectangle((18, 18, 505, 154), radius=12, fill=(247, 247, 247), outline=(40, 40, 40), width=2)
            draw.text((34, 30), "CYAN: THUMB distal contact", fill=(0, 95, 120), font=font)
            draw.text((34, 58), "MAGENTA: INDEX distal contact", fill=(150, 0, 100), font=font)
            draw.text((34, 86), "YELLOW: THIRD — NON-TASK", fill=(130, 85, 0), font=font)
            draw.text((34, 114), "GREEN: thumb→index pinch axis", fill=(0, 105, 35), font=font)
        path = out / output_names[name]
        image.save(path)
        print(f"[PINCH-RENDER] wrote {path.name}", flush=True)
        render_records[name] = {
            "path": str(path),
            "camera_eye_m": (focus + offsets[name]).tolist(),
            "camera_target_m": focus.tolist(),
            "pixel_size": [args.width, args.height],
        }

    # Make the identity proof unambiguous in 2-D without inventing locations:
    # threshold the rendered cyan/magenta marker pixels, join their centroids,
    # and add the separately rendered third-finger view as a labeled inset.
    identity_path = out / "left_phone_pinch_identity_overlay.png"
    identity_image = Image.open(identity_path).convert("RGB")
    marker_rgb = np.asarray(identity_image)
    cyan_mask = (
        (marker_rgb[..., 0] < 110) & (marker_rgb[..., 1] > 135) & (marker_rgb[..., 2] > 135)
    )
    magenta_mask = (
        (marker_rgb[..., 0] > 130) & (marker_rgb[..., 1] < 140) & (marker_rgb[..., 2] > 100)
    )
    marker_centroids = {}
    for label, mask in (("thumb", cyan_mask), ("index", magenta_mask)):
        yy, xx = np.nonzero(mask)
        if len(xx) < 20:
            raise RuntimeError(f"rendered {label} contact marker not visible")
        marker_centroids[label] = (int(np.mean(xx)), int(np.mean(yy)))
    identity_draw = ImageDraw.Draw(identity_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    identity_draw.line(
        [marker_centroids["thumb"], marker_centroids["index"]],
        fill=(0, 210, 55), width=7,
    )
    identity_draw.text(
        (marker_centroids["thumb"][0] + 10, marker_centroids["thumb"][1] + 12),
        "THUMB", fill=(0, 95, 120), font=font,
    )
    identity_draw.text(
        (marker_centroids["index"][0] - 90, marker_centroids["index"][1] - 36),
        "INDEX", fill=(150, 0, 100), font=font,
    )
    third_image = Image.open(out / "left_phone_pinch_third_finger_check.png").convert("RGB")
    third_image.thumbnail((410, 300), Image.Resampling.LANCZOS)
    inset_x, inset_y = args.width - third_image.width - 24, args.height - third_image.height - 24
    identity_image.paste(third_image, (inset_x, inset_y))
    identity_draw = ImageDraw.Draw(identity_image)
    identity_draw.rectangle(
        (inset_x - 3, inset_y - 34, inset_x + third_image.width + 3, inset_y + third_image.height + 3),
        outline=(235, 175, 0), width=5,
    )
    identity_draw.rectangle(
        (inset_x - 1, inset_y - 32, inset_x + third_image.width + 1, inset_y),
        fill=(250, 247, 225),
    )
    identity_draw.text(
        (inset_x + 8, inset_y - 29), "THIRD — NON-TASK (separate view)",
        fill=(130, 85, 0), font=font,
    )
    identity_image.save(identity_path)
    render_records["identity_overlay"]["rendered_marker_centroids_xy"] = marker_centroids
    render_records["identity_overlay"]["pinch_axis_2d_drawn_from_rendered_markers"] = True
    render_records["identity_overlay"]["third_finger_inset_source"] = str(
        out / "left_phone_pinch_third_finger_check.png"
    )

    # Pair all six photo views with different Isaac views. Cropping removes the
    # browser chrome only; the full content rectangles are retained.
    photo_dir = out / "photo_references"
    photo_views = [
        photo_dir / f"real_dex3_left_phone_pinch_{i:02d}_content.png" for i in range(1, 7)
    ]
    isaac_views = [
        out / "left_phone_pinch_front_oblique.png",
        out / "left_phone_pinch_back_oblique.png",
        out / "left_phone_pinch_side.png",
        out / "left_phone_pinch_fingertip_closeup.png",
        out / "left_phone_pinch_third_finger_check.png",
        out / "left_phone_pinch_palm_side.png",
    ]
    panel_w, panel_h, title_h = 520, 390, 42
    sheet = Image.new("RGB", (panel_w * 2, (panel_h + title_h) * 6 + 54), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 16), "REAL DEX3 PHOTO (qualitative)  |  ISAAC DEX3 GEOMETRIC CALIBRATION", fill=(20, 20, 20))
    for row, (photo_path, isaac_path) in enumerate(zip(photo_views, isaac_views)):
        y = 54 + row * (panel_h + title_h)
        for col, (path, label) in enumerate(((photo_path, f"REAL VIEW {row+1}"), (isaac_path, "ISAAC CORRESPONDING VIEW"))):
            image = Image.open(path).convert("RGB")
            image.thumbnail((panel_w - 20, panel_h - 10), Image.Resampling.LANCZOS)
            x = col * panel_w + (panel_w - image.width) // 2
            sheet.paste(image, (x, y + title_h))
            draw.text((col * panel_w + 14, y + 10), label, fill=(20, 20, 20))
    comparison = out / "real_photo_vs_isaac_phone_pinch.png"
    sheet.save(comparison)

    dump(out / "isaac_static_render_audit.json", {
        "status": "ISAAC_STATIC_MULTI_VIEW_RENDER_PASS",
        "physics_steps": int(sim.get_physics_step_count()),
        "kinematic_static_calibration": True,
        "actual_articulation_readback_max_error_rad": float(np.max(np.abs(actual - values))),
        "arm_wrist_joint_values_changed_by_optimizer": False,
        "fixed_arm_joint_values_rad": fixed_arm,
        "left_dex3_joint_names": left_names,
        "left_dex3_joint_values_rad": left_q,
        "phone_asset": str(PHONE_USD),
        "phone_pose_is_static_diagnostic_only": True,
        "paper_white": True,
        "body_mesh_sync": "actual articulation body states mirrored to temporary review-stage prim transforms after sim.forward",
        "render_records": render_records,
        "comparison_sheet": str(comparison),
        "right_hand_primitive_modified": False,
        "hardware_commands": False,
    })
    live_stage.GetRootLayer().Save()
    if args.gui:
        while simulation_app.is_running() and args.pause:
            sim.render()
            time.sleep(0.01)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
