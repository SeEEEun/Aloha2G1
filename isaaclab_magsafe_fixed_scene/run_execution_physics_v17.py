#!/usr/bin/env python3
"""True-physics Episode-49 G1+Dex3 execution trial for v17.

Only the initial robot articulation state is written directly.  Every timed
sample is executed through Isaac actuator position targets.  Phone/accessory
poses are never commanded, and semantic events never attach or detach bodies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
SCENE = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17/final_arm_dex3_trajectory.npz"
DEFAULT_OUTPUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17"
G1_USD = Path("/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd")

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--trial", choices=(
    "phone_grasp", "phone_rotation", "accessory_removal", "bimanual_transport",
    "charger_placement", "accessory_release", "full_task", "full_task_diagnostic",
), required=True)
parser.add_argument("--speed", type=float, choices=(0.25, 0.5, 1.0), default=0.25)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--no-video", action="store_true")
parser.add_argument(
    "--gui", action="store_true",
    help="Open the Isaac Lab Kit GUI. Physics remains actuator-driven.",
)
parser.add_argument(
    "--interactive-review", action="store_true",
    help=(
        "Run the selected true-physics trial in the GUI through the repaired "
        "PhysX-to-Fabric render path. Implies --gui and --render-parity."
    ),
)
parser.add_argument(
    "--interactive-only", action="store_true",
    help=(
        "Visible Kit viewport review without MP4 encoding, segmentation/mask "
        "audits, contact sheets, or offline report regeneration. True PhysX, "
        "actual articulation readback, Fabric synchronization, and live viewport "
        "rendering remain enabled. Implies --interactive-review."
    ),
)
parser.add_argument(
    "--camera",
    choices=("overview", "side", "top", "closeup", "phone", "accessory", "charger"),
    default="overview",
    help="Initial freely orbitable GUI viewport camera.",
)
parser.add_argument(
    "--render-preset", choices=("default", "paper-white"), default="default",
    help="Visual-only lighting/background preset; never changes physics.",
)
parser.add_argument(
    "--loop", action="store_true",
    help="GUI review only: reset by restarting the process and replay the frozen trial.",
)
parser.add_argument(
    "--pause-at-end", action="store_true",
    help="GUI review only: hold the final actual PhysX state for orbit/pan/zoom inspection.",
)
parser.add_argument(
    "--render-parity",
    action="store_true",
    help=(
        "Enable actual-state/link/USD/render parity instrumentation and the native "
        "PhysX-to-USD RTX synchronization fix. This never writes joint/link poses."
    ),
)
parser.add_argument(
    "--artifact-prefix", default="v17",
    help="Output naming prefix only; it does not change physics or trajectory data.",
)
parser.add_argument(
    "--diagnostic-video-prefix",
    default="v17_1_FULL_TRAJECTORY_DIAGNOSTIC",
    help=(
        "Output filename prefix for full_task_diagnostic videos only. "
        "This is a render-artifact label and never changes trajectory or physics."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.interactive_only:
    args.interactive_review = True
if args.interactive_review:
    args.gui = True
    args.render_parity = True
if args.gui:
    # AppLauncher resolves headless mode from explicit visualizer intent.  Merely
    # assigning headless=False (the former implementation) is insufficient: when
    # no Kit visualizer is selected AppLauncher deliberately forces headless mode.
    # Match the known-good preview_magsafe_g1_model.py --viz kit launch path.
    if getattr(args, "headless_explicit", False):
        parser.error("interactive GUI review cannot be combined with --headless")
    if getattr(args, "visualizer_explicit", False):
        requested = set(args.visualizer or [])
        if "kit" not in requested:
            parser.error("interactive GUI review requires --viz kit (not --viz none/non-Kit)")
    args.visualizer = ["kit"]
    args.visualizer_explicit = True
    args.visualizer_disable_all = False
    args.headless = False
    args.headless_explicit = False
    # Camera sensors and a visible viewport are independent.  Keep camera support
    # available for the normal video path, but Kit visualizer selection above is
    # what actually creates the desktop window.
    args.enable_cameras = True
    if args.interactive_only:
        # The visible Kit viewport has its own RTX render delegate and does not
        # need Isaac camera sensors.  Select the same lightweight UI experience
        # as the working preview even if a legacy copy-paste command includes
        # --enable_cameras; this affects rendering resources only, never physics.
        args.enable_cameras = False
if (args.loop or args.pause_at_end) and not args.gui:
    parser.error("--loop and --pause-at-end require --gui")
if args.loop and args.pause_at_end:
    parser.error("choose either --loop or --pause-at-end")
if args.render_parity and args.no_video and not args.interactive_only:
    parser.error("--render-parity requires camera capture; do not combine it with --no-video")
if args.gui and args.no_video and not args.interactive_only:
    parser.error("interactive GUI review must initialize rendering; do not use --no-video")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload: dict) -> None:
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=default, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def checksum(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def load_input() -> dict:
    with np.load(args.input.resolve(), allow_pickle=False) as archive:
        required = (
            "arm_qpos", "left_dex3_qpos", "right_dex3_qpos", "arm_joint_names",
            "left_dex3_joint_names", "right_dex3_joint_names", "source_timestamps",
            "optimized_action", "g1_root", "workspace_scale", "physics_applied",
            "simulation_only", "real_robot_command_allowed",
        )
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise RuntimeError(f"v17 input missing {missing}")
        data = {name: archive[name].copy() for name in required}
    if data["arm_qpos"].shape != (990, 14):
        raise RuntimeError("v17 arm shape mismatch")
    if data["left_dex3_qpos"].shape != (990, 7) or data["right_dex3_qpos"].shape != (990, 7):
        raise RuntimeError("v17 Dex3 shape mismatch")
    if bool(data["physics_applied"]) or not bool(data["simulation_only"]) or bool(data["real_robot_command_allowed"]):
        raise RuntimeError("v17 simulation safety metadata failed")
    if not all(np.isfinite(data[name]).all() for name in ("arm_qpos", "left_dex3_qpos", "right_dex3_qpos")):
        raise RuntimeError("v17 q contains NaN/Inf")
    return data


trajectory = load_input()
launcher = AppLauncher(args)
simulation_app = launcher.app


def main() -> int:
    import carb
    import cv2
    import omni.usd
    import torch
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils

    sys.path[:0] = [str(ROOT / "tools"), str(SCENE)]
    from aloha_g1_v15.semantic_input import load_human_reviewed_development_timeline
    from magsafe_magnet_controller import BodyState, MagnetState, MagneticPair, load_config, quat_rotate
    from physical_contact_monitor import enable_contact_reporting
    from robot_model_preview_common import compose_stage
    from aloha_g1_v15.kinematics import ActiveG1Dex3

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # This installed PhysX GPU backend does not push articulation link xforms
    # into USD when Fabric is disabled.  The former true-physics renderer mixed
    # use_fabric=False with RTX's Fabric transform reader, so actual tensor
    # states advanced while the visual links remained at their authored pose.
    # For positive-step true physics, use the native PhysX->Fabric->RTX path.
    # This changes only render synchronization: no articulation, joint, link,
    # or object pose is written by this fix.
    settings = carb.settings.get_settings()
    if args.interactive_only:
        # `--viz kit` is needed by AppLauncher to select the visible, non-headless
        # Kit experience.  The working preview then pumps the default Kit viewport
        # directly; it does not construct a second SimulationContext-owned
        # KitVisualizer.  Keep those responsibilities separate here as well.
        settings.set_string("/isaaclab/visualizer/types", "")
        settings.set_bool("/isaaclab/visualizer/explicit", True)
        settings.set_bool("/isaaclab/visualizer/disable_all", True)
    if args.render_parity and (not args.no_video or args.gui):
        settings.set_bool(
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate", True
        )
    paper_white_settings = {
        "background_rgb": [0.97, 0.97, 0.97],
        "background_mechanism": "RTX backgroundZeroAlpha composite; render-only",
        "dome": {"intensity": 1350.0, "color_rgb": [1.0, 1.0, 1.0]},
        "key": {
            "type": "DistantLight", "intensity": 3200.0,
            "exposure": 0.0, "color_rgb": [1.0, 0.985, 0.965],
            "angle_deg": 3.0, "rotation_xyz_deg": [-42.0, 24.0, 18.0],
        },
        "fill": {
            "type": "SphereLight", "intensity": 850.0,
            "exposure": 0.0, "color_rgb": [0.94, 0.97, 1.0],
            "radius_m": 0.45, "translation_xyz_m": [0.25, -0.10, 1.65],
        },
        "tone_mapping": {
            "operator": "UNCHANGED",
            "film_iso": 160.0,
            "exposure_time_s": 1.0 / 60.0,
            "f_number": 8.0,
            "automatic_histogram": False,
        },
        "physical_effect": "NONE",
    }
    if args.render_preset == "paper-white":
        settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
        settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", True)
        settings.set_float_array(
            "/rtx/post/backgroundZeroAlpha/backgroundDefaultColor",
            paper_white_settings["background_rgb"] + [1.0],
        )
        settings.set_float("/rtx/post/tonemap/filmIso", 160.0)
        settings.set_float("/rtx/post/tonemap/exposureTime", 1.0 / 60.0)
        settings.set_float("/rtx/post/tonemap/fNumber", 8.0)
        settings.set_bool("/rtx/post/histogram/enabled", False)
    source_path = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
    phase_path = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12/aloha_phase_motion_library.npz"
    timeline_path = ROOT / "configs/episode49_task_timeline.approved.json"
    alignment_path = ROOT / "configs/episode49_action_observation_alignment.approved.json"
    layout_path = SCENE / "scene_layout.json"
    model_path = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
    mapping_path = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
    palm_path = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"
    magnet_path = SCENE / "magnet_config_v2.json"
    with np.load(phase_path, allow_pickle=False) as archive:
        timeline = load_human_reviewed_development_timeline(
            timeline_path, alignment_path,
            trajectory["optimized_action"], trajectory["source_timestamps"],
            archive["left_tcp_position"], archive["right_tcp_position"],
            archive["left_tcp_rotation"], archive["right_tcp_rotation"],
            trajectory_path=source_path, fk_model_path=model_path, task_geometry_path=layout_path,
        )
    endpoints = {
        "phone_grasp": "phone_rotation_to_portrait_start",
        "phone_rotation": "phone_portrait_reached",
        "accessory_removal": "accessory_removed",
        "bimanual_transport": "phone_move_to_charger_start",
        "charger_placement": "left_phone_release_complete",
        "accessory_release": "right_accessory_release_complete",
    }
    full_trajectory_trial = args.trial in ("full_task", "full_task_diagnostic")
    end_frame = (
        timeline.end_index
        if full_trajectory_trial
        else int(timeline.event(endpoints[args.trial]).action_index)
    )
    event_by_index: dict[int, list[str]] = {}
    for name, record in timeline.events.items():
        if record.action_index is not None:
            event_by_index.setdefault(int(record.action_index), []).append(name)

    root_offset = float(json.loads((ROOT / "configs/g1_root_forward_v14.approved.json").read_text())["selected_total_forward_offset_m"])
    numerical_runtime = ActiveG1Dex3(
        model_path, mapping_path, palm_path, np.asarray(trajectory["g1_root"], dtype=np.float64)
    )
    stage_path = out / f"{args.artifact_prefix}_physics_{args.trial}_{str(args.speed).replace('.', 'p')}x.usda"
    stage = compose_stage(stage_path, "G1", G1_USD, "g1", forward_offset_m=root_offset)

    authored_lights_before = [
        {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
        for prim in stage.Traverse() if prim.GetTypeName().endswith("Light")
    ]
    if args.render_preset == "paper-white":
        dome = UsdLux.DomeLight.Define(stage, "/World/V17ReviewLights/PaperWhiteDome")
        dome.CreateIntensityAttr(paper_white_settings["dome"]["intensity"])
        dome.CreateColorAttr(Gf.Vec3f(*paper_white_settings["dome"]["color_rgb"]))
        key = UsdLux.DistantLight.Define(stage, "/World/V17ReviewLights/PaperWhiteKey")
        key.CreateIntensityAttr(paper_white_settings["key"]["intensity"])
        key.CreateExposureAttr(paper_white_settings["key"]["exposure"])
        key.CreateColorAttr(Gf.Vec3f(*paper_white_settings["key"]["color_rgb"]))
        key.CreateAngleAttr(paper_white_settings["key"]["angle_deg"])
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(
            Gf.Vec3f(*paper_white_settings["key"]["rotation_xyz_deg"])
        )
        fill = UsdLux.SphereLight.Define(stage, "/World/V17ReviewLights/PaperWhiteFill")
        fill.CreateIntensityAttr(paper_white_settings["fill"]["intensity"])
        fill.CreateExposureAttr(paper_white_settings["fill"]["exposure"])
        fill.CreateColorAttr(Gf.Vec3f(*paper_white_settings["fill"]["color_rgb"]))
        fill.CreateRadiusAttr(paper_white_settings["fill"]["radius_m"])
        UsdGeom.Xformable(fill).AddTranslateOp().Set(
            Gf.Vec3d(*paper_white_settings["fill"]["translation_xyz_m"])
        )
        dump(out / "render_preset_paper_white.json", {
            "schema": "v17_1_paper_white_render_preset_v1",
            "status": "PAPER_WHITE_RENDER_PRESET_CONFIGURED",
            "preset": "PAPER_WHITE",
            "authored_lights_before": authored_lights_before,
            "added_render_only_lights": paper_white_settings,
            "render_only_prim_paths": [
                "/World/V17ReviewLights/PaperWhiteDome",
                "/World/V17ReviewLights/PaperWhiteKey",
                "/World/V17ReviewLights/PaperWhiteFill",
            ],
            "visual_ground_collider_added": False,
            "physics_configuration_changed": False,
            "trajectory_sha256": sha256_file(args.input.resolve()),
        })

    cfg = load_config(magnet_path)

    def apply_dynamic(path: str, mass: float) -> None:
        prim = stage.GetPrimAtPath(path)
        rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(bool(cfg["safety"]["enable_ccd"]))
        prim.CreateAttribute("physxRigidBody:maxLinearVelocity", Sdf.ValueTypeNames.Float).Set(8.0)
        prim.CreateAttribute("physxRigidBody:maxAngularVelocity", Sdf.ValueTypeNames.Float).Set(25.0)
        prim.AddAppliedSchema("PhysxContactReportAPI")
        prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.0)

    phone_path = "/World/MagSafeScene/Phone"
    accessory_path = "/World/MagSafeScene/Accessory"
    apply_dynamic(phone_path, 0.177)
    apply_dynamic(accessory_path, float(cfg["accessory"]["mass_kg"]))
    phone_collider = stage.GetPrimAtPath(f"{phone_path}/Colliders/Main")
    UsdPhysics.MeshCollisionAPI.Apply(phone_collider).CreateApproximationAttr("convexHull")
    for path in (f"{phone_path}/Colliders/Main", "/World/MagSafeScene/Charger/Colliders/Pad"):
        collider = stage.GetPrimAtPath(path)
        collider.AddAppliedSchema("PhysxCollisionAPI")
        collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
        collider.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    for index in range(12):
        support = stage.GetPrimAtPath(f"{accessory_path}/Colliders/SupportRing/Segment_{index:02d}")
        UsdPhysics.CollisionAPI.Apply(support).CreateCollisionEnabledAttr(False)
    size = cfg["accessory"]["support_foot_proxy_size_xyz_m"]
    center = cfg["accessory"]["support_foot_proxy_center_local_xyz_m"]
    foot = UsdGeom.Cube.Define(stage, f"{accessory_path}/Colliders/SupportFootProxy")
    foot.CreateSizeAttr(2.0)
    foot_xf = UsdGeom.Xformable(foot)
    foot_xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    foot_xf.AddScaleOp().Set(Gf.Vec3f(float(size[0]) / 2, float(size[1]) / 2, float(size[2]) / 2))
    UsdPhysics.CollisionAPI.Apply(foot.GetPrim()).CreateCollisionEnabledAttr(True)
    joint_path = "/World/MagSafeScene/MagneticJoints/AccessoryPhone"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(phone_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(accessory_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.004175, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, -0.00175, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateBreakForceAttr(float(cfg["accessory_phone"]["break_force_n"]))
    joint.CreateBreakTorqueAttr(float(cfg["accessory_phone"]["break_torque_nm"]))
    joint.GetPrim().SetCustomDataByKey("magsafe:mode", "breakable_joint")
    enable_contact_reporting(stage, [
        "/World/G1/Asset", phone_path, accessory_path, "/World/MagSafeScene/Table",
    ])
    stage.GetRootLayer().Save()
    if not omni.usd.get_context().open_stage(str(stage_path)):
        raise RuntimeError(stage_path)
    live_stage = omni.usd.get_context().get_stage()
    sim = SimulationContext(SimulationCfg(device="cuda:0", use_fabric=bool(args.render_parity)))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/G1/Asset/root_joint", spawn=None,
        actuators={
            # The arm trajectory was solved about the zero waist/lower-body
            # standing reference.  In true physics those uncommanded joints
            # must still be held by the fixed-base posture controller;
            # otherwise gravity sags the torso and invalidates arm FK before
            # any task command is evaluated.  These are constant support
            # targets, not an additional behavior trajectory.
            "fixed_base_legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"(left|right)_(hip|knee|ankle)_.*_joint",
                    r"(left|right)_knee_joint",
                ],
                effort_limit_sim=139.0, velocity_limit_sim=32.0,
                stiffness=200.0, damping=10.0,
            ),
            "fixed_base_waist": ImplicitActuatorCfg(
                joint_names_expr=[r"waist_.*_joint"],
                effort_limit_sim=35.0, velocity_limit_sim=30.0,
                stiffness=1000.0, damping=40.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_(shoulder|wrist)_.*_joint", r"(left|right)_elbow_joint"],
                effort_limit_sim=25.0, velocity_limit_sim=12.0, stiffness=1000.0, damping=40.0,
            ),
            "dex3": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_hand_.*_joint"],
                effort_limit_sim=2.5, velocity_limit_sim=12.0, stiffness=100.0, damping=4.0,
            ),
        },
    ))
    phone = RigidObject(RigidObjectCfg(prim_path=phone_path, spawn=None))
    accessory = RigidObject(RigidObjectCfg(prim_path=accessory_path, spawn=None))
    mapping = json.loads(mapping_path.read_text())
    contact_links = {
        "left_thumb": mapping["left"]["A"]["distal_link"],
        "left_index": mapping["left"]["B"]["distal_link"],
        "left_third": mapping["left"]["C"]["distal_link"],
        "right_index": mapping["right"]["A"]["distal_link"],
        "right_thumb": mapping["right"]["B"]["distal_link"],
        "right_third": mapping["right"]["C"]["distal_link"],
    }
    filters = [phone_path, accessory_path, "/World/MagSafeScene/Table/Colliders/Top"]
    sensors = {
        name: ContactSensor(ContactSensorCfg(
            prim_path=f"/World/G1/Asset/{link}", update_period=0.0,
            filter_prim_paths_expr=filters, track_contact_points=True,
            max_contact_data_count_per_prim=32, force_threshold=0.0,
        )) for name, link in contact_links.items()
    }
    # A task-contact sensor is insufficient for diagnosing an approach that
    # nudges an object with a non-task digit or the palm.  Keep an additional
    # all-link view so every physical G1/object pair is attributable.  This is
    # observation only: it never changes collision filtering or object state.
    all_robot_contact = ContactSensor(ContactSensorCfg(
        prim_path="/World/G1/Asset/.*_link", update_period=0.0,
        filter_prim_paths_expr=filters, track_contact_points=True,
        track_friction_forces=True, max_contact_data_count_per_prim=64,
        force_threshold=0.0,
    ))
    camera_poses = {
        "overview": ((1.15, -1.22, 1.38), (0.49, 0.02, 0.98)),
        "side": ((1.62, -0.10, 1.34), (0.45, 0.00, 1.01)),
        "top": ((0.52, 0.05, 1.72), (0.47, 0.08, 0.87)),
        "phone": ((0.79, -0.31, 1.04), (0.52, 0.07, 0.87)),
        "accessory": ((0.78, -0.27, 1.15), (0.50, 0.08, 0.98)),
        "charger": ((0.73, -0.04, 1.14), (0.42, 0.21, 0.94)),
        "release": ((0.88, -0.15, 1.08), (0.50, 0.12, 0.84)),
    }
    gui_camera_name = args.camera
    if gui_camera_name == "closeup":
        gui_camera_name = "phone" if args.trial in ("phone_grasp", "phone_rotation") else (
            "accessory" if args.trial in ("accessory_removal", "bimanual_transport") else "charger"
        )
    if args.trial == "full_task_diagnostic":
        camera_names = ["overview", "side", "top"]
        if args.artifact_prefix == "v18":
            # One uninterrupted execution supplies both full views and all
            # close-ups; these are never stage-specific trajectories/runs.
            camera_names += ["phone", "accessory", "charger"]
    elif args.trial == "full_task":
        camera_names = list(camera_poses)
    elif args.trial in ("phone_grasp", "phone_rotation"):
        camera_names = ["overview", "phone"]
    elif args.trial in ("accessory_removal", "bimanual_transport"):
        camera_names = ["overview", "accessory"]
    elif args.trial == "charger_placement":
        camera_names = ["overview", "charger"]
    else:
        camera_names = ["overview", "release"]
    cameras = {}
    if not args.no_video and not args.interactive_only:
        camera_data_types = ["rgb"]
        if args.render_parity:
            camera_data_types.append("instance_id_segmentation_fast")
        cameras = {
            name: Camera(CameraCfg(
                prim_path=f"/World/V17PhysicsCamera_{name}", update_period=0,
                height=args.height, width=args.width, data_types=camera_data_types,
                colorize_instance_id_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.05, 20.0)),
            )) for name in camera_names
        }
    sim.reset()
    names = list(robot.data.joint_names)
    wanted = (
        trajectory["arm_joint_names"].astype(str).tolist()
        + trajectory["left_dex3_joint_names"].astype(str).tolist()
        + trajectory["right_dex3_joint_names"].astype(str).tolist()
    )
    missing = [name for name in wanted if name not in names]
    ids = [names.index(name) for name in wanted if name in names]
    if missing or len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError(f"Isaac name mapping failed missing={missing}")
    for name, camera in cameras.items():
        eye, target = camera_poses[name]
        camera.set_world_poses_from_view(np.asarray([eye], np.float32), np.asarray([target], np.float32))
    if args.gui:
        sim.set_camera_view(*camera_poses[gui_camera_name])
        if args.interactive_only:
            # SimulationContext.set_camera_view targets configured visualizers.
            # Interactive-only intentionally uses the default Kit viewport, so
            # point that viewport's actual perspective camera explicitly.
            from isaacsim.core.rendering_manager import ViewportManager

            eye, look_at = camera_poses[gui_camera_name]
            ViewportManager.set_camera_view(
                "/OmniverseKit_Persp", eye=list(eye), target=list(look_at)
            )
    desired = np.c_[trajectory["arm_qpos"], trajectory["left_dex3_qpos"], trajectory["right_dex3_qpos"]]
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    target[0, ids] = torch.as_tensor(desired[0], device=robot.device, dtype=torch.float32)
    # Initial condition only; timed execution below uses actuator targets.
    robot.write_joint_state_to_sim(target, zero)
    sim.forward()
    if args.gui:
        # Initialize and validate the *interactive* viewport before execution.
        # The default viewport is pumped exactly like the working preview.
        sim.render()
        sim.render_context.reset_transform_cadence()
        if args.interactive_only:
            simulation_app.update()
        try:
            from omni.kit.viewport.utility import get_active_viewport_window

            viewport_window = get_active_viewport_window()
        except Exception as exc:
            raise RuntimeError(f"visible Kit viewport lookup failed: {exc}") from exc
        if viewport_window is None:
            raise RuntimeError(
                "interactive review requested but no active Kit viewport exists; "
                "offscreen IsaacRtxRenderer is not accepted"
            )
        print("[GUI] VISIBLE KIT WINDOW READY", flush=True)
    dt = float(sim.get_physics_dt())
    settle_steps = max(1, int(round(args.settle_seconds / dt)))
    for _ in range(settle_steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt); phone.update(dt); accessory.update(dt)
        for sensor in sensors.values(): sensor.update(dt)
        all_robot_contact.update(dt)
    initial_phone_pose = phone.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
    initial_accessory_pose = accessory.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
    initial_phone_velocity = phone.data.root_vel_w.torch[0].detach().cpu().numpy().copy()
    initial_accessory_velocity = accessory.data.root_vel_w.torch[0].detach().cpu().numpy().copy()

    cache = UsdGeom.XformCache()
    charger_target_prim = live_stage.GetPrimAtPath("/World/MagSafeScene/Charger/Frames/PhoneTargetCenter")
    charger_target_matrix = np.asarray(cache.GetLocalToWorldTransform(charger_target_prim), dtype=float).T
    pad_prim = live_stage.GetPrimAtPath("/World/MagSafeScene/Charger/Visuals/PadFace")
    pad_matrix = np.asarray(cache.GetLocalToWorldTransform(pad_prim), dtype=float).T
    pad_outward = pad_matrix[:3, 2] / np.linalg.norm(pad_matrix[:3, 2])
    charger_state = BodyState(
        charger_target_matrix[:3, 3].copy(), np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3), np.zeros(3),
    )
    charger_pair = MagneticPair(
        "phone_charger", cfg["phone_charger"],
        np.array([0.0, 0.004175, 0.0]), np.array([0.0, 1.0, 0.0]),
        np.zeros(3), pad_outward,
        0.177, cfg["safety"]["max_phone_acceleration_mps2"],
        initial_state=MagnetState.DETACHED,
        target_orientation_world=np.asarray(cfg["charger_target"]["target_rotation_wxyz"], float),
    )

    def body_state(obj) -> BodyState:
        pose = obj.data.root_pose_w.torch[0].detach().cpu().numpy()
        velocity = obj.data.root_vel_w.torch[0].detach().cpu().numpy()
        q_xyzw = pose[3:7]
        return BodyState(
            pose[:3].copy(), np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]),
            velocity[:3].copy(), velocity[3:].copy(),
        )

    def wrench(obj, force: np.ndarray, torque: np.ndarray) -> None:
        force_tensor = torch.tensor(force.reshape(1, 1, 3), device=obj.device, dtype=torch.float32)
        torque_tensor = torch.tensor(torque.reshape(1, 1, 3), device=obj.device, dtype=torch.float32)
        obj.instantaneous_wrench_composer.set_forces_and_torques_index(
            force_tensor, torque_tensor, is_global=True
        )

    def relative_pose(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ra = Rotation.from_quat(a[3:7])
        rb = Rotation.from_quat(b[3:7])
        return ra.inv().apply(b[:3] - a[:3]), (ra.inv() * rb).as_quat()

    initial_relative_p, initial_relative_q = relative_pose(initial_phone_pose, initial_accessory_pose)
    accessory_detached = False
    accessory_detach_frame = None
    charger_attach_frame = None
    actual_q, command_q, effort_q, velocity_q = [], [], [], []
    full_actual_q, full_velocity_q = [], []
    phone_pose_rows, accessory_pose_rows = [], []
    phone_velocity_rows, accessory_velocity_rows = [], []
    contact_rows = []
    all_robot_contact_rows = []
    magnet_rows = []
    wrist_rows = []
    sim_time = 0.0
    max_tracking = 0.0
    physics_steps = settle_steps

    video_paths = {}
    writers = {}
    if cameras:
        if args.render_parity:
            # Preserve all historical v17.1 filenames.  A caller-provided
            # diagnostic prefix is already the complete artifact family name.
            white = (
                "_WHITE"
                if args.render_preset == "paper-white"
                and args.diagnostic_video_prefix == "v17_1_FULL_TRAJECTORY_DIAGNOSTIC"
                else ""
            )
            if args.trial == "full_task_diagnostic":
                mapping_video = {
                    "overview": f"{args.diagnostic_video_prefix}{white}_overview.mp4",
                    "side": f"{args.diagnostic_video_prefix}{white}_side.mp4",
                    "top": f"{args.diagnostic_video_prefix}{white}_top.mp4",
                }
                if args.artifact_prefix == "v18":
                    mapping_video.update({
                        "phone": "v18_TRUE_PHYSICS_PHONE_GRASP_ROTATION_closeup.mp4",
                        "accessory": "v18_TRUE_PHYSICS_ACCESSORY_REMOVAL_closeup.mp4",
                        "charger": "v18_TRUE_PHYSICS_CHARGER_CAPTURE_closeup.mp4",
                    })
            else:
                mapping_video = {
                    "overview": f"v17_1_{args.trial}_physics_RENDERFIX{white}_overview.mp4",
                    "phone": f"v17_1_{args.trial}_physics_RENDERFIX{white}_closeup.mp4",
                    "side": f"v17_1_{args.trial}_physics_RENDERFIX_side.mp4",
                    "top": f"v17_1_{args.trial}_physics_RENDERFIX_top.mp4",
                    "accessory": f"v17_1_{args.trial}_physics_RENDERFIX_accessory.mp4",
                    "charger": f"v17_1_{args.trial}_physics_RENDERFIX_charger.mp4",
                    "release": f"v17_1_{args.trial}_physics_RENDERFIX_release.mp4",
                }
        elif args.artifact_prefix == "v17_1":
            mapping_video = {
                "overview": (
                    "v17_1_fulltask_physics_overview.mp4" if args.trial == "full_task"
                    else f"v17_1_{args.trial}_physics_overview.mp4"
                ),
                "side": "v17_1_fulltask_physics_side.mp4",
                "top": "v17_1_fulltask_physics_top.mp4",
                "phone": (
                    "v17_1_phone_grasp_physics_closeup.mp4" if args.trial == "phone_grasp"
                    else "v17_1_phone_rotation_physics.mp4"
                ),
                "accessory": "v17_1_accessory_removal_physics.mp4",
                "charger": "v17_1_charger_placement_physics.mp4",
                "release": "v17_1_accessory_release_physics.mp4",
            }
        else:
            mapping_video = {
                "overview": "v17_physics_fulltask_overview.mp4" if args.trial == "full_task" else f"v17_physics_{args.trial}_overview.mp4",
                "side": "v17_physics_fulltask_side.mp4",
                "top": "v17_physics_fulltask_top.mp4",
                "phone": "v17_physics_phone_grasp_closeup.mp4" if args.trial == "phone_grasp" else "v17_physics_phone_rotation_closeup.mp4",
                "accessory": "v17_physics_accessory_removal_closeup.mp4",
                "charger": "v17_physics_charger_placement_closeup.mp4",
                "release": "v17_physics_accessory_release_closeup.mp4",
            }
        for name in camera_names:
            output = out / mapping_video[name]
            raw = out / f".{output.stem}.{args.trial}.{str(args.speed).replace('.', 'p')}x.raw.mp4"
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (args.width, args.height))
            if not writer.isOpened(): raise RuntimeError(raw)
            writers[name] = (writer, raw, output)

    body_names = list(robot.data.body_names)
    if args.render_parity:
        print(f"[RENDER_PARITY] articulation ready bodies={len(body_names)}", flush=True)
    wrist_ids = {side: body_names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")}
    contact_body_ids = {name: body_names.index(link) for name, link in contact_links.items()}
    diagnostic_body_ids = {
        name: body_names.index(name) for name in (
            "left_hand_palm_link", "left_hand_camera_base_link",
            "left_hand_index_0_link", "left_hand_middle_0_link",
            "pelvis", "torso_link",
        ) if name in body_names
    }
    parity_links = {
        "left_shoulder": "left_shoulder_pitch_link",
        "left_elbow": "left_elbow_link",
        "left_wrist": "left_wrist_yaw_link",
        "left_palm": "left_hand_palm_link",
        "left_thumb_distal": contact_links["left_thumb"],
        "left_index_distal": contact_links["left_index"],
        "left_third_distal": contact_links["left_third"],
        "right_shoulder": "right_shoulder_pitch_link",
        "right_elbow": "right_elbow_link",
        "right_wrist": "right_wrist_yaw_link",
        "right_palm": "right_hand_palm_link",
        "right_thumb_distal": contact_links["right_thumb"],
        "right_index_distal": contact_links["right_index"],
        "right_third_distal": contact_links["right_third"],
    }
    missing_parity_bodies = [name for name in parity_links.values() if name not in body_names]
    if missing_parity_bodies:
        raise RuntimeError(f"parity link mapping failed missing={missing_parity_bodies}")
    parity_body_ids = {label: body_names.index(name) for label, name in parity_links.items()}
    numerical_body_ids = {
        label: (
            None if label in {"left_palm", "right_palm"}
            else numerical_runtime.body_id(name)
        )
        for label, name in parity_links.items()
    }
    if args.render_parity:
        print(f"[RENDER_PARITY] link mappings ready count={len(parity_links)}", flush=True)

    def isaac_link_states() -> dict[str, dict[str, np.ndarray]]:
        positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
        quaternions = robot.data.body_quat_w.torch[0].detach().cpu().numpy()
        return {
            label: {
                "position_m": positions[body_id].astype(np.float64).copy(),
                "quaternion_xyzw": quaternions[body_id].astype(np.float64).copy(),
            }
            for label, body_id in parity_body_ids.items()
        }

    def numerical_link_states(q_value: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        numerical_runtime.assign(q_value[:14], q_value[14:21], q_value[21:28])
        result = {}
        for label, body_id in numerical_body_ids.items():
            if body_id is None:
                palm = numerical_runtime.palm_pose(label.split("_", 1)[0])
                result[label] = {
                    "position_m": palm[:3, 3].copy(),
                    "rotation": palm[:3, :3].copy(),
                }
                continue
            position = numerical_runtime.model_to_scene_position(
                numerical_runtime.data.xpos[body_id]
            )
            rotation = numerical_runtime.model_to_scene_rotation(
                numerical_runtime.data.xmat[body_id].reshape(3, 3)
            )
            result[label] = {
                "position_m": position.copy(),
                "rotation": rotation.copy(),
            }
        return result

    def usd_link_states() -> dict[str, dict[str, np.ndarray]]:
        # Read-only audit of the transforms consumed by RTX.  No prim xform is
        # authored here; PhysX remains the sole producer of articulation state.
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        result = {}
        for label, name in parity_links.items():
            prim = live_stage.GetPrimAtPath(f"/World/G1/Asset/{name}")
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
            result[label] = {
                "position_m": matrix[:3, 3].copy(),
                "rotation": matrix[:3, :3].copy(),
            }
        return result

    def serialize_link_states(value: dict[str, dict[str, np.ndarray]]) -> dict:
        return {
            label: {key: np.asarray(item).tolist() for key, item in row.items()}
            for label, row in value.items()
        }

    steps_per_frame = max(1, int(round((1.0 / 30.0) / (args.speed * dt))))
    parity_frames = sorted(set(
        int(round(fraction * end_frame))
        for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    ))
    parity_samples = []
    frame_trace_rows = []
    parity_rgb: dict[tuple[str, int], np.ndarray] = {}
    parity_masks: dict[tuple[str, int], np.ndarray] = {}
    first_render_rgb: dict[str, np.ndarray] = {}
    first_render_mask: dict[str, np.ndarray] = {}
    render_motion_rows: dict[str, list[dict]] = {name: [] for name in cameras}
    previous_target = None
    previous_actual = None
    captured_video_frame = 0
    if args.render_parity:
        print(
            f"[RENDER_PARITY] timed physics start frames={end_frame + 1} "
            f"substeps={steps_per_frame}", flush=True,
        )
    if args.gui:
        print("[GUI] LIVE V18 TRUE-PHYSICS REVIEW START", flush=True)
    for frame in range(end_frame + 1):
        if args.render_parity and frame % 50 == 0:
            print(f"[RENDER_PARITY] frame={frame}/{end_frame}", flush=True)
        previous = desired[max(frame - 1, 0)]
        current = desired[frame]
        for substep in range(steps_per_frame):
            alpha = float(substep + 1) / steps_per_frame
            command = (1.0 - alpha) * previous + alpha * current
            target[0, ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            phone_state = body_state(phone)
            correct_face = float(np.dot(quat_rotate(phone_state.quaternion_wxyz, np.array([0.0, 1.0, 0.0])), -pad_outward)) > 0.0
            charger_result = charger_pair.update(sim_time, phone_state, charger_state, penetration=0.0, correct_face=correct_face)
            wrench(phone, charger_result.force, charger_result.torque)
            phone.write_data_to_sim(); accessory.write_data_to_sim()
            sim.step(render=False)
            physics_steps += 1; sim_time += dt
            robot.update(dt); phone.update(dt); accessory.update(dt)
            for sensor in sensors.values(): sensor.update(dt)
            all_robot_contact.update(dt)
        actual = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().copy()
        command = current.copy()
        isaac_state = isaac_link_states() if args.render_parity else {}
        numerical_state = numerical_link_states(actual) if args.render_parity and frame in parity_frames else {}
        actual_q.append(actual); command_q.append(command)
        effort_q.append(robot.data.applied_torque.torch[0, ids].detach().cpu().numpy().copy())
        velocity_q.append(robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().copy())
        full_actual_q.append(robot.data.joint_pos.torch[0].detach().cpu().numpy().copy())
        full_velocity_q.append(robot.data.joint_vel.torch[0].detach().cpu().numpy().copy())
        max_tracking = max(max_tracking, float(np.max(np.abs(actual - command))))
        phone_pose_value = phone.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
        accessory_pose_value = accessory.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
        phone_pose_rows.append(phone_pose_value); accessory_pose_rows.append(accessory_pose_value)
        phone_velocity_rows.append(phone.data.root_vel_w.torch[0].detach().cpu().numpy().copy())
        accessory_velocity_rows.append(accessory.data.root_vel_w.torch[0].detach().cpu().numpy().copy())
        relative_p, relative_q = relative_pose(phone_pose_value, accessory_pose_value)
        relative_rotation = Rotation.from_quat(relative_q).magnitude()
        if not accessory_detached and (
            np.linalg.norm(relative_p - initial_relative_p) > 0.003
            or math.degrees(relative_rotation) > 5.0
        ):
            accessory_detached = True
            accessory_detach_frame = frame
        forces = {}
        for name, sensor in sensors.items():
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                forces[name] = [0.0, 0.0, 0.0]
            else:
                value = matrix.torch[0, 0].detach().cpu().numpy()
                forces[name] = np.linalg.norm(value, axis=1).tolist()
        contact_rows.append(forces)
        all_matrix = all_robot_contact.data.force_matrix_w
        if all_matrix is None:
            all_robot_contact_rows.append([])
        else:
            all_value = all_matrix.torch[0].detach().cpu().numpy()
            owner_paths = list(all_robot_contact.body_physx_view.prim_paths[: all_robot_contact.num_sensors])
            frame_contacts = []
            for owner_index, owner_path in enumerate(owner_paths):
                for filter_index, filter_path in enumerate(filters):
                    force_n = float(np.linalg.norm(all_value[owner_index, filter_index]))
                    if force_n > 1e-6:
                        frame_contacts.append({
                            "owner": owner_path, "other": filter_path,
                            "force_n": force_n,
                        })
            all_robot_contact_rows.append(frame_contacts)
        wrist_rows.append({
            side: robot.data.body_pos_w.torch[0, wrist_ids[side]].detach().cpu().numpy().copy()
            for side in ("left", "right")
        } | {
            name: robot.data.body_pos_w.torch[0, body_id].detach().cpu().numpy().copy()
            for name, body_id in contact_body_ids.items()
        } | {
            name: robot.data.body_pos_w.torch[0, body_id].detach().cpu().numpy().copy()
            for name, body_id in diagnostic_body_ids.items()
        })
        if charger_result.attach_event and charger_attach_frame is None:
            charger_attach_frame = frame
        magnet_rows.append({
            "state": charger_result.state.value, "distance_m": charger_result.distance,
            "angle_deg": charger_result.angle_deg, "force_n": float(np.linalg.norm(charger_result.force)),
            "torque_nm": float(np.linalg.norm(charger_result.torque)),
        })
        captured_this_frame = None
        overview_camera_checksum = None
        overview_robot_mask_centroid_yx = [None, None]
        usd_stage_state = {}
        if cameras or args.gui:
            if args.render_parity:
                # Push the post-step PhysX articulation state into Fabric.
                # This is the installed backend's native render synchronization
                # API, not a joint/link pose write.
                sim.forward()
            sim.render()
            if args.interactive_only:
                # Match preview_magsafe_g1_model.py: process the desktop Kit UI
                # event/render loop after Fabric received the final PhysX state.
                simulation_app.update()
            if args.render_parity:
                # A physics step has already advanced the articulation.  Reset
                # only RTX's transform de-duplication cadence before sensor
                # capture; unlike the v12 kinematic workaround, no link prim
                # transform is authored here.
                sim.render_context.reset_transform_cadence()
            phase = str(timeline.sample_arrays["global_task_phase"][frame])
            event = ",".join(event_by_index.get(frame, [])) or "-"
            for name, camera in cameras.items():
                camera.update(dt, force_recompute=True)
                image_rgb = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3].copy()
                if args.render_parity:
                    segmentation = (
                        camera.data.output["instance_id_segmentation_fast"].torch[0]
                        .detach().cpu().numpy().squeeze().astype(np.int32).copy()
                    )
                    if segmentation.ndim == 3:
                        segmentation = segmentation[..., 0]
                    labels = camera.data.info.get("instance_id_segmentation_fast", {}).get(
                        "idToLabels", {}
                    )
                    present_ids = set(np.unique(segmentation).astype(int).tolist())
                    robot_ids = []
                    for identifier, label in labels.items():
                        if "/World/G1/" not in str(label):
                            continue
                        candidates = identifier if isinstance(identifier, tuple) else (identifier,)
                        robot_ids.extend(
                            value for value in (int(candidate) for candidate in candidates)
                            if value in present_ids and value != 0
                        )
                    robot_ids = sorted(set(robot_ids))
                    robot_mask = np.isin(segmentation, robot_ids)
                    if name not in first_render_rgb:
                        first_render_rgb[name] = image_rgb.copy()
                        first_render_mask[name] = robot_mask.copy()
                    union = robot_mask | first_render_mask[name]
                    rgb_delta = np.abs(
                        image_rgb.astype(np.float32) - first_render_rgb[name].astype(np.float32)
                    )
                    centroid = (
                        np.mean(np.argwhere(robot_mask), axis=0).tolist()
                        if np.any(robot_mask) else [None, None]
                    )
                    first_centroid = (
                        np.mean(np.argwhere(first_render_mask[name]), axis=0)
                        if np.any(first_render_mask[name]) else np.array([np.nan, np.nan])
                    )
                    centroid_delta = (
                        float(np.linalg.norm(np.asarray(centroid, dtype=float) - first_centroid))
                        if centroid[0] is not None else None
                    )
                    coordinates = np.argwhere(robot_mask)
                    bbox = (
                        [
                            int(coordinates[:, 1].min()), int(coordinates[:, 0].min()),
                            int(coordinates[:, 1].max()), int(coordinates[:, 0].max()),
                        ]
                        if len(coordinates) else None
                    )
                    render_motion_rows[name].append({
                        "frame": frame,
                        "robot_pixel_count": int(np.count_nonzero(robot_mask)),
                        "mask_xor_pixels_from_frame0": int(np.count_nonzero(
                            robot_mask ^ first_render_mask[name]
                        )),
                        "mask_identical_to_frame0": bool(np.array_equal(
                            robot_mask, first_render_mask[name]
                        )),
                        "robot_mask_centroid_yx": centroid,
                        "robot_mask_centroid_displacement_px": centroid_delta,
                        "robot_mask_bbox_xyxy": bbox,
                        "mean_rgb_difference_in_robot_union": float(
                            np.mean(rgb_delta[union]) if np.any(union) else 0.0
                        ),
                        "robot_mask_sha256": hashlib.sha256(robot_mask.tobytes()).hexdigest(),
                        "raw_rgb_sha256": hashlib.sha256(image_rgb.tobytes()).hexdigest(),
                    })
                    if name == "overview":
                        overview_camera_checksum = render_motion_rows[name][-1]["raw_rgb_sha256"]
                        overview_robot_mask_centroid_yx = centroid
                    if frame in parity_frames:
                        parity_rgb[(name, frame)] = image_rgb.copy()
                        parity_masks[(name, frame)] = robot_mask.copy()
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (0, 0), (args.width, 78), (0, 0, 0), -1)
                if args.trial == "full_task_diagnostic":
                    cv2.putText(image, f"FULL TRAJECTORY DIAGNOSTIC | TRUE PHYSICS | {args.speed:.2f}x | action {frame}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .39, (60, 235, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f"phase {phase[:58]} | event {event[:50]}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .38, (100, 255, 120), 1, cv2.LINE_AA)
                    cv2.putText(image, "STAGE FAILURE DOES NOT STOP PLAYBACK | NOT A FULL-TASK SUCCESS CLAIM", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, .34, (80, 170, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(image, f"TRUE ISAAC PHYSICS | {args.trial} | {args.speed:.2f}x | action {frame}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (60, 235, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f"phase {phase[:58]} | event {event[:50]}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .38, (100, 255, 120), 1, cv2.LINE_AA)
                    cv2.putText(image, f"phone/accessory PHYSICS ONLY | no kinematic object follow | charger {charger_result.state.value}", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, .36, (80, 170, 255), 1, cv2.LINE_AA)
                writers[name][0].write(image)
            if cameras:
                captured_this_frame = captured_video_frame
                captured_video_frame += 1
            if args.render_parity:
                # USD remains a useful diagnostic, but Fabric is the declared
                # RTX transform source in true-physics render-parity mode.
                usd_stage_state = usd_link_states()

        if args.render_parity:
            target_delta = 0.0 if previous_target is None else float(
                np.max(np.abs(command - previous_target))
            )
            actual_delta = 0.0 if previous_actual is None else float(
                np.max(np.abs(actual - previous_actual))
            )
            frame_trace_rows.append({
                "rendered_frame": captured_this_frame,
                "action_index": frame,
                "physics_step": physics_steps,
                "target_q_checksum": checksum(command),
                "actual_q_checksum": checksum(actual),
                "left_wrist_xyz": json.dumps(wrist_rows[-1]["left"].tolist()),
                "right_wrist_xyz": json.dumps(wrist_rows[-1]["right"].tolist()),
                "left_thumb_xyz": json.dumps(wrist_rows[-1]["left_thumb"].tolist()),
                "left_index_xyz": json.dumps(wrist_rows[-1]["left_index"].tolist()),
                "right_thumb_xyz": json.dumps(wrist_rows[-1]["right_thumb"].tolist()),
                "right_index_xyz": json.dumps(wrist_rows[-1]["right_index"].tolist()),
                "right_third_xyz": json.dumps(wrist_rows[-1]["right_third"].tolist()),
                "camera_frame_checksum": overview_camera_checksum,
                "robot_mask_centroid_x": overview_robot_mask_centroid_yx[1],
                "robot_mask_centroid_y": overview_robot_mask_centroid_yx[0],
                "trajectory_index": frame,
                "target_q_sha256": checksum(command),
                "target_q_delta_max_rad": target_delta,
                "physics_step_after_execution": physics_steps,
                "actual_q_sha256": checksum(actual),
                "actual_q_delta_max_rad": actual_delta,
                "target_actual_max_error_rad": float(np.max(np.abs(command - actual))),
                "captured_video_frame": captured_this_frame,
                "capture_after_physics_step": bool(captured_this_frame is not None),
            })
            if frame in parity_frames:
                link_metrics = {}
                for label in parity_links:
                    isaac_position = np.asarray(isaac_state[label]["position_m"])
                    numerical_position = np.asarray(numerical_state[label]["position_m"])
                    usd_position = np.asarray(usd_stage_state[label]["position_m"])
                    link_metrics[label] = {
                        "numerical_actual_q_vs_isaac_position_error_m": float(
                            np.linalg.norm(numerical_position - isaac_position)
                        ),
                        "isaac_vs_usd_stage_diagnostic_position_error_m": float(
                            np.linalg.norm(isaac_position - usd_position)
                        ),
                    }
                parity_samples.append({
                    "trajectory_index": frame,
                    "captured_video_frame": captured_this_frame,
                    "physics_step": physics_steps,
                    "q_target": command.copy(),
                    "q_actual": actual.copy(),
                    "target_actual_max_error_rad": float(np.max(np.abs(command - actual))),
                    "numerical_link_state_from_actual_q": serialize_link_states(numerical_state),
                    "isaac_link_state": serialize_link_states(isaac_state),
                    "usd_stage_link_state_diagnostic_not_render_source": serialize_link_states(
                        usd_stage_state
                    ),
                    "link_position_parity": link_metrics,
                })
            previous_target = command.copy()
            previous_actual = actual.copy()

    for name, (writer, raw, output) in writers.items():
        writer.release()
        os.replace(raw, output)
        video_paths[name] = str(output.resolve())

    actual_q = np.asarray(actual_q); command_q = np.asarray(command_q)
    effort_q = np.asarray(effort_q); velocity_q = np.asarray(velocity_q)
    full_actual_q = np.asarray(full_actual_q); full_velocity_q = np.asarray(full_velocity_q)
    phone_pose_rows = np.asarray(phone_pose_rows); accessory_pose_rows = np.asarray(accessory_pose_rows)
    phone_velocity_rows = np.asarray(phone_velocity_rows); accessory_velocity_rows = np.asarray(accessory_velocity_rows)
    parity_result = None
    if args.render_parity and not args.interactive_only:
        preset_suffix = "_paper_white" if args.render_preset == "paper-white" else ""
        trace_name = (
            f"target_actual_frame_trace{preset_suffix}.csv"
            if args.trial == "phone_grasp"
            else f"target_actual_frame_trace_{args.trial}{preset_suffix}.csv"
        )
        with (out / trace_name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(frame_trace_rows[0]))
            writer.writeheader()
            writer.writerows(frame_trace_rows)
        save_npz(
            out / f"render_parity_keyframes_{args.trial}{preset_suffix}.npz",
            parity_frames=np.asarray(parity_frames, dtype=np.int64),
            **{
                f"rgb_{name}_{frame}": image
                for (name, frame), image in parity_rgb.items()
            },
            **{
                f"mask_{name}_{frame}": mask
                for (name, frame), mask in parity_masks.items()
            },
        )
        link_displacements = {}
        for label in parity_links:
            positions = np.asarray([
                row["isaac_link_state"][label]["position_m"] for row in parity_samples
            ])
            link_displacements[label] = {
                "key_sample_max_displacement_from_start_m": float(
                    np.max(np.linalg.norm(positions - positions[0], axis=1))
                ),
                "start_position_m": positions[0],
                "end_position_m": positions[-1],
            }
        camera_motion = {}
        for name, rows in render_motion_rows.items():
            key_rows = [rows[index] for index in parity_frames]
            camera_motion[name] = {
                "keyframes": key_rows,
                "robot_masks_identical_at_all_keyframes": bool(all(
                    row["mask_identical_to_frame0"] for row in key_rows
                )),
                "maximum_keyframe_mask_xor_pixels": int(max(
                    row["mask_xor_pixels_from_frame0"] for row in key_rows
                )),
                "maximum_keyframe_robot_centroid_displacement_px": float(max(
                    row["robot_mask_centroid_displacement_px"] or 0.0 for row in key_rows
                )),
                "maximum_keyframe_mean_rgb_difference_in_robot_union": float(max(
                    row["mean_rgb_difference_in_robot_union"] for row in key_rows
                )),
                "robot_mask_nonempty_all_frames": bool(all(
                    row["robot_pixel_count"] > 0 for row in rows
                )),
            }
        numerical_isaac_errors = [
            values["numerical_actual_q_vs_isaac_position_error_m"]
            for sample in parity_samples for values in sample["link_position_parity"].values()
        ]
        isaac_usd_diagnostic_errors = [
            values["isaac_vs_usd_stage_diagnostic_position_error_m"]
            for sample in parity_samples for values in sample["link_position_parity"].values()
        ]
        # Full-body review cameras must contain the robot at every sample.
        # Stage-local close-ups intentionally lose the robot before/after their
        # phase, so requiring a non-empty robot mask for all 990 frames creates
        # a false static-render failure even when their stage frames and every
        # full-body view visibly move.  Keep those two contracts separate.
        full_body_camera_names = [
            name for name in ("overview", "side", "top") if name in camera_motion
        ]
        stage_local_camera_names = [
            name for name in camera_motion if name not in full_body_camera_names
        ]
        full_body_motion_pass = bool(full_body_camera_names and all(
            not camera_motion[name]["robot_masks_identical_at_all_keyframes"]
            and camera_motion[name]["maximum_keyframe_mask_xor_pixels"] > 100
            and camera_motion[name]["robot_mask_nonempty_all_frames"]
            for name in full_body_camera_names
        ))
        stage_local_motion_pass = bool(all(
            not camera_motion[name]["robot_masks_identical_at_all_keyframes"]
            and camera_motion[name]["maximum_keyframe_mask_xor_pixels"] > 100
            for name in stage_local_camera_names
        ))
        rendered_motion_pass = bool(full_body_motion_pass and stage_local_motion_pass)
        parity_result = {
            "status": (
                "COMMAND_ACTUAL_RENDER_PARITY_PASS"
                if rendered_motion_pass else "BLOCKED_RENDERED_MESH_MOTION"
            ),
            "trial": args.trial,
            "trajectory_sha256": sha256_file(args.input.resolve()),
            "samples": parity_samples,
            "target_motion_max_peak_to_peak_rad": float(np.max(np.ptp(command_q, axis=0))),
            "actual_motion_max_peak_to_peak_rad": float(np.max(np.ptp(actual_q, axis=0))),
            "target_actual_rmse_rad": float(np.sqrt(np.mean((command_q - actual_q) ** 2))),
            "target_actual_max_error_rad": float(np.max(np.abs(command_q - actual_q))),
            "link_displacements": link_displacements,
            "maximum_numerical_actual_q_vs_isaac_link_position_error_m": float(max(numerical_isaac_errors)),
            "maximum_isaac_vs_usd_stage_diagnostic_position_error_m": float(
                max(isaac_usd_diagnostic_errors)
            ),
            "rendered_motion": camera_motion,
            "command_target_advancement_pass": bool(np.max(np.ptp(command_q, axis=0)) > 0.01),
            "actual_articulation_motion_pass": bool(np.max(np.ptp(actual_q, axis=0)) > 0.01),
            "link_transform_motion_pass": bool(
                link_displacements["left_wrist"]["key_sample_max_displacement_from_start_m"] > 0.02
                and link_displacements["right_wrist"]["key_sample_max_displacement_from_start_m"] > 0.02
            ),
            "rendered_mesh_motion_pass": rendered_motion_pass,
            "rendered_mesh_motion_scope": {
                "full_body_cameras": full_body_camera_names,
                "full_body_motion_pass": full_body_motion_pass,
                "full_body_robot_mask_nonempty_all_frames_required": True,
                "stage_local_cameras": stage_local_camera_names,
                "stage_local_motion_pass": stage_local_motion_pass,
                "stage_local_robot_mask_nonempty_all_frames_required": False,
                "reason": (
                    "A task close-up may legitimately exclude the robot outside its "
                    "semantic phase; static-render regression is anchored by overview, "
                    "side, and top while close-ups must still show non-identical masks."
                ),
            },
            "captured_frame_count": captured_video_frame,
            "target_actual_trace_path": str((out / trace_name).resolve()),
            "keyframe_npz_path": str((
                out / f"render_parity_keyframes_{args.trial}{preset_suffix}.npz"
            ).resolve()),
            "render_sync": {
                "transform_source": "FABRIC_ACTUAL_PHYSX_ARTICULATION_STATE",
                "use_fabric": True,
                "rtx_read_transforms_from_fabric": True,
                "explicit_sim_forward_after_physics_state_readback": True,
                "render_context_transform_cadence_reset_before_camera_update": True,
                "explicit_link_transform_writes": 0,
                "kinematic_joint_writes_during_timed_run": 0,
                "physics_actuator_execution_preserved": True,
            },
        }
        dump(out / f"render_parity_{args.trial}{preset_suffix}.json", parity_result)
    elif args.render_parity:
        # Interactive-only deliberately omits offscreen cameras and mask/video
        # auditing.  Preserve numerical command/actual/link evidence while the
        # visible OS-window validation is performed by the GUI review audit.
        link_displacements = {}
        for label in parity_links:
            positions = np.asarray([
                row["isaac_link_state"][label]["position_m"] for row in parity_samples
            ])
            link_displacements[label] = {
                "key_sample_max_displacement_from_start_m": float(
                    np.max(np.linalg.norm(positions - positions[0], axis=1))
                ),
                "start_position_m": positions[0],
                "end_position_m": positions[-1],
            }
        parity_result = {
            "status": "LIVE_KIT_VIEWPORT_SYNC_ACTIVE",
            "trial": args.trial,
            "trajectory_sha256": sha256_file(args.input.resolve()),
            "target_motion_max_peak_to_peak_rad": float(np.max(np.ptp(command_q, axis=0))),
            "actual_motion_max_peak_to_peak_rad": float(np.max(np.ptp(actual_q, axis=0))),
            "target_actual_rmse_rad": float(np.sqrt(np.mean((command_q - actual_q) ** 2))),
            "target_actual_max_error_rad": float(np.max(np.abs(command_q - actual_q))),
            "link_displacements": link_displacements,
            "offscreen_camera_capture_skipped": True,
            "rendered_mesh_validation": "VISIBLE_OS_KIT_VIEWPORT_AUDIT",
            "render_sync": {
                "transform_source": "FABRIC_ACTUAL_PHYSX_ARTICULATION_STATE",
                "use_fabric": True,
                "rtx_read_transforms_from_fabric": True,
                "sim_forward_after_final_physics_substep": True,
                "kit_visualizer_render_each_action_sample": True,
                "explicit_link_transform_writes": 0,
                "kinematic_joint_writes_during_timed_run": 0,
                "physics_actuator_execution_preserved": True,
            },
        }
        dump(out / f"live_viewport_sync_{args.trial}.json", parity_result)
    sensor_order = (
        "left_thumb", "left_index", "left_third",
        "right_thumb", "right_index", "right_third",
    )
    phone_forces = np.asarray([[row[name][0] for name in sensor_order] for row in contact_rows])
    accessory_forces = np.asarray([[row[name][1] for name in sensor_order] for row in contact_rows])
    table_forces = np.asarray([[row[name][2] for name in sensor_order] for row in contact_rows])
    phone_rotation = Rotation.from_quat(phone_pose_rows[:, 3:7])
    phone_long = phone_rotation.apply(np.array([1.0, 0.0, 0.0]))
    portrait_error = np.degrees(np.arccos(np.clip(np.abs(phone_long[:, 2]), 0.0, 1.0)))
    target_rotation = Rotation.from_quat(np.asarray(cfg["charger_target"]["target_rotation_wxyz"])[[1, 2, 3, 0]])
    charger_orientation_error = np.degrees((target_rotation.inv() * phone_rotation).magnitude())
    charger_center_error = np.linalg.norm(phone_pose_rows[:, :3] - charger_target_matrix[:3, 3], axis=1)
    phone_motion = np.linalg.norm(phone_pose_rows[:, :3] - initial_phone_pose[:3], axis=1)
    accessory_motion = np.linalg.norm(accessory_pose_rows[:, :3] - initial_accessory_pose[:3], axis=1)
    end_slice = slice(max(0, len(phone_pose_rows) - min(15, len(phone_pose_rows))), len(phone_pose_rows))
    simultaneous_ab = (phone_forces[:, 0] > 0.05) & (phone_forces[:, 1] > 0.05)
    simultaneous_right_thumb_index = (
        (accessory_forces[:, 3] > 0.05) & (accessory_forces[:, 4] > 0.05)
    )

    def longest_true_run(mask: np.ndarray) -> int:
        best = current = 0
        for value in np.asarray(mask, dtype=bool):
            current = current + 1 if value else 0
            best = max(best, current)
        return int(best)

    left_grasp_index = int(timeline.event("left_phone_grasp_start").action_index)
    left_wrist_positions = np.asarray([row["left"] for row in wrist_rows], dtype=np.float64)
    wrist_displacement_after_grasp = float(np.linalg.norm(
        left_wrist_positions[-1] - left_wrist_positions[left_grasp_index]
    ))
    phone_displacement_after_grasp = float(np.linalg.norm(
        phone_pose_rows[-1, :3] - phone_pose_rows[left_grasp_index, :3]
    ))
    phone_hand_relative = phone_pose_rows[:, :3] - left_wrist_positions
    phone_hand_translation_slip = float(np.linalg.norm(
        phone_hand_relative[-1] - phone_hand_relative[left_grasp_index]
    ))
    phone_wrist_motion_ratio = phone_displacement_after_grasp / max(wrist_displacement_after_grasp, 1e-8)
    simultaneous_ab_run = longest_true_run(simultaneous_ab)
    final_ab_contact = bool(np.any(simultaneous_ab[end_slice]))
    left_phone_contact = bool(simultaneous_ab_run >= 3)
    right_accessory_contact = bool(longest_true_run(simultaneous_right_thumb_index) >= 3)
    phone_retained = bool(
        left_phone_contact
        and final_ab_contact
        and phone_displacement_after_grasp >= max(0.005, 0.40 * wrist_displacement_after_grasp)
        and phone_hand_translation_slip <= 0.020
    )
    phone_lift_above_settle = float(phone_pose_rows[-1, 2] - initial_phone_pose[2])
    phone_rotation_pass = bool(phone_retained and portrait_error[-1] <= 10.0)
    accessory_removed = bool(accessory_detached and np.max(accessory_motion) > 0.003)
    accessory_retained = bool(accessory_removed and right_accessory_contact and accessory_pose_rows[-1, 2] > 0.795)
    charger_pass = bool(
        charger_pair.state == MagnetState.ATTACHED
        and charger_center_error[-1] <= 0.005
        and charger_orientation_error[-1] <= 5.0
    )
    accessory_speed_final = float(np.mean(np.linalg.norm(accessory_velocity_rows[end_slice, :3], axis=1)))
    accessory_release_pass = bool(
        accessory_removed and accessory_speed_final < 0.08
        and accessory_pose_rows[-1, 2] >= 0.79
        and float(np.mean(accessory_forces[end_slice, 2])) < 0.05
    )
    stage_passes = {
        "phone_grasp": phone_retained,
        "phone_rotation": phone_rotation_pass,
        "accessory_removal": accessory_retained,
        "bimanual_transport": bool(phone_retained and accessory_retained),
        "charger_placement": charger_pass,
        "accessory_release": accessory_release_pass,
    }
    pair_maxima: dict[str, float] = {}
    first_pair_frame: dict[str, int] = {}
    for frame, rows in enumerate(all_robot_contact_rows):
        for row in rows:
            key = f"{row['owner']} <-> {row['other']}"
            pair_maxima[key] = max(pair_maxima.get(key, 0.0), float(row["force_n"]))
            first_pair_frame.setdefault(key, frame)
    diagnostic_complete = bool(
        args.trial == "full_task_diagnostic"
        and len(command_q) == desired.shape[0]
        and len(actual_q) == desired.shape[0]
    )
    trial_pass = (
        False if args.trial == "full_task_diagnostic"
        else all(stage_passes.values()) if args.trial == "full_task"
        else stage_passes[args.trial]
    )
    result = {
        "status": (
            "FULL_TRAJECTORY_DIAGNOSTIC_COMPLETE" if diagnostic_complete
            else f"SIM_FULL_TASK_PASS_{int(args.speed * 100):03d}X" if trial_pass and args.trial == "full_task"
            else f"{args.trial.upper()}_PHYSICS_PASS" if trial_pass
            else f"BLOCKED_{args.trial.upper()}_PHYSICS"
        ),
        "trial": args.trial, "speed_scale": args.speed, "pass": trial_pass,
        "diagnostic_complete": diagnostic_complete,
        "full_task_success_claim": False if args.trial == "full_task_diagnostic" else trial_pass,
        "stage_failure_stops_playback": False if args.trial == "full_task_diagnostic" else None,
        "artifact_prefix": args.artifact_prefix,
        "end_event": "trajectory_end" if full_trajectory_trial else endpoints[args.trial],
        "end_action_index": end_frame, "source_frames_executed": end_frame + 1,
        "physics_steps": physics_steps, "physics_dt": dt, "steps_per_source_frame": steps_per_frame,
        "joint_mapping": {"requested": 28, "mapped": len(ids), "missing": missing, "duplicates": len(ids) - len(set(ids))},
        "tracking": {
            "maximum_absolute_error_rad": float(np.max(np.abs(actual_q - command_q))),
            "rmse_rad": float(np.sqrt(np.mean((actual_q - command_q) ** 2))),
            "maximum_velocity_rad_s": float(np.max(np.abs(velocity_q))),
            "maximum_effort": float(np.max(np.abs(effort_q))),
            "maximum_support_joint_deviation_rad": float(np.max(np.abs(full_actual_q[:, [
                index for index, name in enumerate(names) if name not in wanted
            ]]))),
        },
        "stage_passes_observed": stage_passes,
        "object_metrics": {
            "phone_max_displacement_m": float(np.max(phone_motion)),
            "phone_final_xyz_m": phone_pose_rows[-1, :3],
            "phone_portrait_error_final_deg": float(portrait_error[-1]),
            "left_thumb_phone_force_max_n": float(np.max(phone_forces[:, 0])),
            "left_index_phone_force_max_n": float(np.max(phone_forces[:, 1])),
            "left_third_phone_force_max_n": float(np.max(phone_forces[:, 2])),
            "left_thumb_index_simultaneous_contact_longest_run_samples": simultaneous_ab_run,
            "left_thumb_index_simultaneous_contact_at_stage_end": final_ab_contact,
            "phone_displacement_after_grasp_m": phone_displacement_after_grasp,
            "left_wrist_displacement_after_grasp_m": wrist_displacement_after_grasp,
            "phone_to_wrist_motion_ratio": phone_wrist_motion_ratio,
            "phone_hand_translation_slip_m": phone_hand_translation_slip,
            "phone_lift_above_settle_m": phone_lift_above_settle,
            "phone_lifted_clear_of_table_3mm": bool(phone_lift_above_settle >= 0.003),
            "accessory_max_displacement_m": float(np.max(accessory_motion)),
            "accessory_final_xyz_m": accessory_pose_rows[-1, :3],
            "right_thumb_accessory_force_max_n": float(np.max(accessory_forces[:, 3])),
            "right_index_accessory_force_max_n": float(np.max(accessory_forces[:, 4])),
            "right_third_accessory_force_max_n": float(np.max(accessory_forces[:, 5])),
            "right_thumb_index_simultaneous_contact_longest_run_samples": longest_true_run(simultaneous_right_thumb_index),
            "accessory_detached": accessory_detached,
            "accessory_detach_action_index": accessory_detach_frame,
            "charger_state_final": charger_pair.state.value,
            "charger_attach_action_index": charger_attach_frame,
            "charger_center_error_final_mm": float(charger_center_error[-1] * 1000.0),
            "charger_orientation_error_final_deg": float(charger_orientation_error[-1]),
            "accessory_final_linear_speed_m_s": accessory_speed_final,
        },
        "all_robot_object_contact_pairs": [
            {
                "pair": key,
                "maximum_force_n": value,
                "first_action_index": first_pair_frame[key],
            }
            for key, value in sorted(pair_maxima.items(), key=lambda item: -item[1])
        ],
        "settling": {
            "phone_pose_after_settle": initial_phone_pose,
            "accessory_pose_after_settle": initial_accessory_pose,
            "phone_velocity_after_settle": initial_phone_velocity,
            "accessory_velocity_after_settle": initial_accessory_velocity,
        },
        "integrity": {
            "gravity_enabled": True, "collision_enabled": True,
            "object_pose_commands": 0, "semantic_scripted_attach_detach": 0,
            "kinematic_object_follow": False, "direct_joint_writes_during_timed_run": 0,
            "initial_robot_state_write_count": 1, "actuator_target_steps": physics_steps - settle_steps,
            "fixed_base_posture_joints_held_at_nominal": 15,
            "render_parity_instrumentation": bool(args.render_parity),
            "rtx_read_transforms_from_fabric": True if args.render_parity else None,
            "fabric_enabled_for_render_sync": True if args.render_parity else None,
            "render_preset": args.render_preset,
            "interactive_review": bool(args.interactive_review),
            "interactive_only": bool(args.interactive_only),
            "gui_viewport": bool(args.gui),
            "explicit_link_transform_writes": 0,
            "dds": False, "publisher": False, "real_robot_command": False,
        },
        "video_paths": video_paths,
        "execution_render_parity": parity_result,
        "input_sha256": sha256_file(args.input.resolve()),
        "stage_sha256": sha256_file(stage_path),
    }
    tag = (
        f"{args.trial}_{str(args.speed).replace('.', 'p')}x"
        + ("_paper_white" if args.render_preset == "paper-white" else "")
    )
    dump(out / f"physics_trial_{tag}.json", result)
    save_npz(
        out / f"physics_trial_{tag}.npz",
        commanded_q=command_q, actual_q=actual_q, applied_effort=effort_q,
        actual_velocity=velocity_q, phone_pose_xyzw=phone_pose_rows,
        full_runtime_joint_names=np.asarray(names), full_actual_q=full_actual_q,
        full_actual_velocity=full_velocity_q,
        accessory_pose_xyzw=accessory_pose_rows, phone_velocity=phone_velocity_rows,
        accessory_velocity=accessory_velocity_rows, phone_contact_force_n=phone_forces,
        accessory_contact_force_n=accessory_forces, table_contact_force_n=table_forces,
        wrist_and_contact_link_positions=np.asarray(wrist_rows, dtype=object),
        all_robot_object_contact_rows=np.asarray(all_robot_contact_rows, dtype=object),
        magnet_diagnostics=np.asarray(magnet_rows, dtype=object),
        action_indices=np.arange(end_frame + 1), speed_scale=np.asarray(args.speed),
        physics_steps=np.asarray(physics_steps), object_pose_scripted=np.asarray(False),
    )
    if args.trial == "full_task_diagnostic":
        dump(out / "full_task_diagnostic_result.json", result)
    print(json.dumps(result, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), flush=True)
    if args.gui and args.pause_at_end:
        print("[GUI] PAUSED AT FINAL STATE", flush=True)
        print(
            "[GUI REVIEW] Trial complete. Final actual PhysX state is held; "
            "orbit/pan/zoom freely and close Isaac Sim when finished.", flush=True,
        )
        while simulation_app.is_running():
            # No physics step and no state write: only propagate the already
            # simulated final articulation through the repaired Fabric path.
            sim.forward()
            sim.render()
            sim.render_context.reset_transform_cadence()
            simulation_app.update()
    if args.gui and args.loop and simulation_app.is_running():
        print(
            "[GUI REVIEW] Restarting Kit to reset the authoritative physics "
            "scene and replay the identical frozen trial.", flush=True,
        )
        os.execv(sys.executable, [sys.executable, *sys.argv])
    return 0 if trial_pass or diagnostic_complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
