#!/usr/bin/env python3
"""Human-teach only the local RIGHT-owned -> transport-grasp transition.

This GUI restores the independently verified post-handoff keyframe for local
debugging.  It exposes only object-relative RIGHT wrist offsets and the seven
RIGHT Dex3 targets.  It never edits doll physics, controller gains, A/B data,
or the qualified transport-to-bin trajectory.  Recorded teaching is stored as
an object-relative local primitive, never as episode/world coordinates.

State restoration in this tool is diagnostic-only and is explicitly forbidden
from final continuous validation.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gui", action="store_true", help="Required: open the visible Isaac Kit GUI.")
parser.add_argument(
    "--output-root",
    type=Path,
    default=ROOT / "outputs/final_methodology_preserving_completion/05_gui_local_primitive",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    parser.error("this human-teaching diagnostic requires --gui")
if getattr(args, "headless_explicit", False):
    parser.error("--gui cannot be combined with --headless")
args.visualizer = ["kit"]
args.visualizer_explicit = True
args.visualizer_disable_all = False
args.headless = False
args.headless_explicit = False
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

import carb
import carb.input
import carb.settings
import omni.appwindow
import omni.ui as ui
import omni.usd
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.evaluation.contracts import AUTHORITATIVE_REFERENCES, authoritative_joint_ranges
from tools.policy_b_isaac_control_contract import CONTROLLER_CONTRACT


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
KEYFRAME = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/success_first_common_execution"
    / "post_handoff_keyframe/POST_HANDOFF_KEYFRAME.npz"
)
KEYFRAME_MANIFEST = KEYFRAME.with_name("POST_HANDOFF_KEYFRAME_MANIFEST.json")
G04_ROOT = (
    ROOT
    / "outputs/final_methodology_preserving_completion/03_fallback_transport_grasp"
    / "candidates/G04_R10_CRADLE_GATE_QUALIFIED"
)
G04_COMMAND = G04_ROOT / "right_only_r6_command.npz"
G04_EVENT = G04_ROOT / "physics_r6/event_log.npz"
G04_QUALIFICATION = G04_ROOT / "RIGHT_ONLY_TRANSPORT_QUALIFICATION.json"
EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    KEYFRAME: "298f68726b077ee3b8760fdfa178c5f3b81a1d2c84a6ce808395e594a536464f",
    KEYFRAME_MANIFEST: "94c4ab35951f9c3e8262bc323bbf8537ca92861dc1d3c6aa9d56282c1470d4bb",
    G04_COMMAND: "fab93d5440140451c1ec24e625a7c2e1c2d2c41a799baf15bf8d74f947202acd",
    G04_EVENT: "b819988b9b6aab86d44d049d2c5499b6bb484fe872cdba91bd14c991482d4b41",
    G04_QUALIFICATION: "fcc71333256e319397474ac2660eceb9acba73163896270bc82b3d85c713d755",
}
DOLL = "/World/DollHandoffEnvironment/Doll"
SOURCE_BODY = f"{DOLL}/Body"
PROXY = f"{DOLL}/MeasuredProxyV3Collider"
VISUAL = f"{DOLL}/MeasuredProxyV3Visual"
G1 = "/World/G1/Asset"
TABLE = "/World/DollHandoffEnvironment/Table/Colliders/Top"
CONTROL_FPS = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return result


def pose_from_xyzw(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    return transform(Rotation.from_quat(quaternion).as_matrix(), position)


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def rounded_oval_mesh(dimensions: np.ndarray) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    radii = np.asarray(dimensions, dtype=np.float64) / 2.0
    latitudes, longitudes = 12, 24
    points: list[np.ndarray] = [np.asarray([0.0, 0.0, radii[2]])]
    for latitude in range(1, latitudes):
        theta = math.pi * latitude / latitudes
        for longitude in range(longitudes):
            phi = 2.0 * math.pi * longitude / longitudes
            z = radii[2] * math.cos(theta)
            if latitude >= latitudes - 2:
                z = -radii[2]
            points.append(
                np.asarray(
                    [
                        radii[0] * math.sin(theta) * math.cos(phi),
                        radii[1] * math.sin(theta) * math.sin(phi),
                        z,
                    ]
                )
            )
    points.append(np.asarray([0.0, 0.0, -radii[2]]))
    north, south = 0, len(points) - 1
    faces: list[list[int]] = []
    for longitude in range(longitudes):
        faces.append([north, 1 + longitude, 1 + (longitude + 1) % longitudes])
    for latitude in range(latitudes - 2):
        first = 1 + latitude * longitudes
        second = first + longitudes
        for longitude in range(longitudes):
            following = (longitude + 1) % longitudes
            faces.append([first + longitude, second + longitude, second + following, first + following])
    last = 1 + (latitudes - 2) * longitudes
    for longitude in range(longitudes):
        faces.append([last + longitude, south, last + (longitude + 1) % longitudes])
    return (
        [Gf.Vec3f(*map(float, point)) for point in points],
        [len(face) for face in faces],
        [index for face in faces for index in face],
    )


def apply_frozen_proxy(stage: Usd.Stage, config: dict[str, Any]) -> dict[str, Any]:
    geometry = config["geometry_candidates"][0]
    source = stage.GetPrimAtPath(SOURCE_BODY)
    if not source.IsValid():
        raise RuntimeError("source doll collider is missing")
    UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Set(False)
    for child in stage.GetPrimAtPath(DOLL).GetChildren():
        if child.GetPath() in {Sdf.Path(PROXY), Sdf.Path(VISUAL)}:
            continue
        if child.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(child).MakeInvisible()
    visual_dimensions = np.asarray(config["object"]["visual_dimensions_m"], dtype=np.float64)
    collision_dimensions = np.asarray(geometry["dimensions_m"], dtype=np.float64)
    for path, dimensions, visible in (
        (VISUAL, visual_dimensions, True),
        (PROXY, collision_dimensions, False),
    ):
        mesh = UsdGeom.Mesh.Define(stage, path)
        points, counts, indices = rounded_oval_mesh(dimensions)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        if visible:
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.72, 0.34)])
        else:
            offset = float((collision_dimensions[2] - visual_dimensions[2]) / 2.0)
            UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, offset))
            UsdGeom.Imageable(mesh).MakeInvisible()
    prim = stage.GetPrimAtPath(PROXY)
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateContactOffsetAttr(float(config["object"]["contact_offset_m"]))
    physx_collision.CreateRestOffsetAttr(float(config["object"]["rest_offset_m"]))
    material = UsdShade.Material.Define(stage, "/World/DollGraspableProxyV2Material")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(config["material"]["static_friction"]))
    material_api.CreateDynamicFrictionAttr(float(config["material"]["dynamic_friction"]))
    material_api.CreateRestitutionAttr(float(config["object"]["restitution"]))
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(str(config["material"]["friction_combine_mode"]))
    physx_material.CreateRestitutionCombineModeAttr().Set(str(config["material"]["restitution_combine_mode"]))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    root = stage.GetPrimAtPath(DOLL)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(config["object"]["mass_kg"]))
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rigid.CreateLinearDampingAttr(float(config["object"]["linear_damping"]))
    rigid.CreateAngularDampingAttr(float(config["object"]["angular_damping"]))
    rigid.CreateMaxDepenetrationVelocityAttr(float(config["object"]["max_depenetration_velocity_m_s"]))
    return {
        "visual_dimensions_m": visual_dimensions.tolist(),
        "collision_dimensions_m": collision_dimensions.tolist(),
        "mass_kg": float(config["object"]["mass_kg"]),
        "static_friction": float(config["material"]["static_friction"]),
        "dynamic_friction": float(config["material"]["dynamic_friction"]),
    }


def build_actuators(config: dict[str, Any]) -> dict[str, ImplicitActuatorCfg]:
    values = copy.deepcopy(CONTROLLER_CONTRACT["actuators"])
    drive = config["finger_drive"]
    values["dex3"].update(
        effort_limit_sim=float(drive["effort_limit_sim"]),
        velocity_limit_sim=float(drive["velocity_limit_sim"]),
        stiffness=float(drive["kp"]),
        damping=float(drive["kd"]),
    )
    return {name: ImplicitActuatorCfg(**spec) for name, spec in values.items()}


def contact_rows(sensor: ContactSensor, dt: float) -> list[dict[str, Any]]:
    forces, points, normals, separations, counts, starts = sensor.contact_view.get_contact_data(dt)
    forces_np = numpy(forces).reshape(-1)
    points_np = numpy(points).reshape(-1, 3)
    normals_np = numpy(normals).reshape(-1, 3)
    separations_np = numpy(separations).reshape(-1)
    owners = list(sensor.body_physx_view.prim_paths[: sensor.num_sensors])
    counts_np = numpy(counts).reshape(len(owners), -1).astype(np.int64)
    starts_np = numpy(starts).reshape(len(owners), -1).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for owner_index, owner in enumerate(owners):
        for filter_index in range(counts_np.shape[1]):
            start, count = starts_np[owner_index, filter_index], counts_np[owner_index, filter_index]
            for index in range(int(start), int(start + count)):
                rows.append(
                    {
                        "owner": owner,
                        "force_n": float(abs(forces_np[index])),
                        "point": points_np[index].astype(np.float64),
                        "normal": normals_np[index].astype(np.float64),
                        "separation_m": float(separations_np[index]),
                    }
                )
    return rows


def filtered_force(sensor: ContactSensor) -> float:
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        return 0.0
    value = numpy(matrix)
    return float(np.max(np.linalg.norm(value.reshape(-1, 3), axis=1))) if value.size else 0.0


def main() -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable dependency changed: {path}: {actual}")
    config = read_json(CONFIG)
    key_manifest = read_json(KEYFRAME_MANIFEST)
    qualification = read_json(G04_QUALIFICATION)
    if key_manifest.get("status") != "POST_HANDOFF_KEYFRAME_VERIFIED":
        raise RuntimeError("post-handoff keyframe is no longer verified")
    if qualification.get("status") != "PASS" or qualification["repeatability"]["successes"] != 3:
        raise RuntimeError("G04 transport endpoint is no longer 3/3 qualified")
    with np.load(KEYFRAME, allow_pickle=False) as archive:
        keyframe = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(G04_COMMAND, allow_pickle=False) as archive:
        g04_command = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(G04_EVENT, allow_pickle=False) as archive:
        g04_event = {key: np.asarray(archive[key]) for key in archive.files}
    names, _ranges = authoritative_joint_ranges()
    if keyframe["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("keyframe named 28D order changed")
    if g04_command["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("G04 named 28D order changed")
    joint_contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray(
        [float(row["minimum"]) for row in joint_contract["joint_specs"]],
        dtype=np.float64,
    )
    upper = np.asarray(
        [float(row["maximum"]) for row in joint_contract["joint_specs"]],
        dtype=np.float64,
    )
    lookup = {name: index for index, name in enumerate(names)}
    right_arm_indices = np.asarray([lookup[f"right_{name}_joint"] for name in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"
    )])
    right_hand_names = [f"right_hand_{joint}_joint" for joint in config["joint_order_7d"]]
    right_hand_indices = np.asarray([lookup[name] for name in right_hand_names])

    common = load_common_config()
    scene = load_scene(common)
    kinematics = G1Kinematics(common, scene)

    # The transport reference is measured at the end of its independently
    # qualified high stabilization, before horizontal travel begins.
    high = g04_event["stage"].astype(str) == "RIGHT_HIGH_STABILIZATION"
    target_row = int(np.flatnonzero(high)[-1])
    target_object_world = pose_from_xyzw(
        g04_event["object_position_world_m"][target_row],
        g04_event["object_quaternion_xyzw"][target_row],
    )
    target_measured_q = g04_event["measured_q_rad"][target_row].astype(np.float64)

    def wrist_world(q28: np.ndarray) -> np.ndarray:
        kinematics.assign(q28[:14], q28[14:21], q28[21:28])
        wrist = kinematics.wrist_pose("right")
        return transform(
            kinematics.model_to_world_rotation(wrist[:3, :3]),
            kinematics.model_to_world_position(wrist[:3, 3]),
        )

    target_object_to_wrist = inverse_pose(target_object_world) @ wrist_world(target_measured_q)
    target_hand = np.asarray(g04_command["candidate_right_hand_model_order_7d_rad"], dtype=np.float64)

    settings = carb.settings.get_settings()
    settings.set_string("/isaaclab/visualizer/types", "")
    settings.set_bool("/isaaclab/visualizer/explicit", True)
    settings.set_bool("/isaaclab/visualizer/disable_all", True)
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    if not omni.usd.get_context().open_stage(config["source_scene"]):
        raise RuntimeError("failed to open frozen Doll-Handoff scene")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    proxy_manifest = apply_frozen_proxy(stage, config)
    dt = float(config["timing"]["physics_dt_s"])
    substeps = int(config["timing"]["physics_substeps_per_control_frame"])
    if not np.isclose(dt * substeps, 1.0 / CONTROL_FPS):
        raise RuntimeError("frozen physics/control timing changed")
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=str(config["simulation"]["device"]),
            gravity=(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"])),
            use_fabric=bool(config["simulation"]["use_fabric"]),
        )
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path=f"{G1}/root_joint",
            spawn=None,
            actuators=build_actuators(config),
        )
    )
    doll = RigidObject(RigidObjectCfg(prim_path=DOLL, spawn=None))
    whole_hand = read_json(Path(config["whole_hand_geometry"]))
    digit_sensors: dict[str, ContactSensor] = {}
    for digit in ("thumb", "index", "middle"):
        role = next(role for role in ("A", "B", "C") if whole_hand["right"][role]["digit_chain"] == digit)
        digit_sensors[digit] = ContactSensor(
            ContactSensorCfg(
                prim_path=f"{G1}/right_hand_{digit}_.*_link",
                update_period=0.0,
                filter_prim_paths_expr=[DOLL],
                track_contact_points=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
    table_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=DOLL,
            update_period=0.0,
            filter_prim_paths_expr=[TABLE],
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            force_threshold=0.0,
        )
    )
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    joint_ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(joint_ids) != 28:
        raise RuntimeError(f"Isaac named joint map failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    target_velocity = torch.zeros_like(target)

    camera = read_json(ROOT / "isaaclab_doll_handoff_scene/scene_layout.json")["camera"]["presets"]["overview"]
    sim.set_camera_view(camera["eye_world_xyz_m"], camera["target_world_xyz_m"])
    from isaacsim.core.rendering_manager import ViewportManager
    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=camera["eye_world_xyz_m"],
        target=camera["target_world_xyz_m"],
    )

    actions: deque[str] = deque()
    playing = False
    quitting = False
    replay_rows: dict[str, np.ndarray] | None = None
    replay_frame = 0
    verification_repeats_remaining = 0
    verification_results: list[dict[str, Any]] = []
    records: dict[str, list[Any]] = {}
    latest = {
        "forces": np.zeros(3, dtype=np.float64),
        "contacts": np.zeros(3, dtype=np.int64),
        "table_force": 0.0,
        "object_pose": pose_from_xyzw(
            keyframe["object_position_world_m"], keyframe["object_quaternion_xyzw"]
        ),
        "actual_q": keyframe["measured_q_rad"].astype(np.float64),
        "command_q": keyframe["commanded_q_rad"].astype(np.float64),
        "safety": "PASS",
        "object_speed": float(np.linalg.norm(keyframe["object_linear_velocity_m_s"])),
    }
    baseline_object_to_wrist = inverse_pose(latest["object_pose"]) @ wrist_world(latest["command_q"])

    def enqueue(value: str) -> None:
        actions.append(value)

    def make_slider(label: str, value: float, minimum: float, maximum: float, step: float) -> ui.SimpleFloatModel:
        with ui.HStack(height=24, spacing=4):
            ui.Label(label, width=160)
            model = ui.SimpleFloatModel(value)
            ui.FloatSlider(model=model, min=minimum, max=maximum, step=step, width=240)
            ui.FloatField(model=model, width=80)
        return model

    window_title = "Local RIGHT Ownership -> Transport Grasp Teaching"
    control_window = ui.Window(window_title, width=610, height=900, visible=True, dock_preference=ui.DockPreference.RIGHT_TOP)
    with control_window.frame:
        with ui.VStack(spacing=4):
            ui.Label("DIAGNOSTIC STATE RESTORATION — NOT FINAL TASK VALIDATION", height=24)
            ui.Label("Only object-relative RIGHT wrist and RIGHT Dex3 targets are editable.", height=22)
            with ui.HStack(height=32, spacing=4):
                ui.Button("Play/Pause [Space]", clicked_fn=lambda: enqueue("TOGGLE"))
                ui.Button("Single Step [N]", clicked_fn=lambda: enqueue("STEP"))
                ui.Button("Reset Local [R]", clicked_fn=lambda: enqueue("RESET"))
                ui.Button("Quit [Q]", clicked_fn=lambda: enqueue("QUIT"))
            with ui.HStack(height=32, spacing=4):
                ui.Button("Replay Recording [P]", clicked_fn=lambda: enqueue("REPLAY"))
                ui.Button("Save Teaching [S]", clicked_fn=lambda: enqueue("SAVE"))
                ui.Button("Verify Replay 3x [V]", clicked_fn=lambda: enqueue("VERIFY"))
                ui.Button("Clear Recording [C]", clicked_fn=lambda: enqueue("CLEAR"))
            ui.Separator()
            ui.Label("RIGHT wrist delta relative to doll / local entry", height=22)
            wrist_models = {
                "dx": make_slider("delta X [m]", 0.0, -0.08, 0.08, 0.001),
                "dy": make_slider("delta Y [m]", 0.0, -0.08, 0.08, 0.001),
                "dz": make_slider("delta Z [m]", 0.0, -0.08, 0.08, 0.001),
                "roll": make_slider("roll [deg]", 0.0, -60.0, 60.0, 0.5),
                "pitch": make_slider("pitch [deg]", 0.0, -60.0, 60.0, 0.5),
                "yaw": make_slider("yaw [deg]", 0.0, -90.0, 90.0, 0.5),
            }
            ui.Separator()
            ui.Label("RIGHT Dex3 absolute targets [rad]", height=22)
            hand_models = {
                name: make_slider(name, float(latest["command_q"][index]), float(lower[index]), float(upper[index]), 0.005)
                for name, index in zip(right_hand_names, right_hand_indices, strict=True)
            }
            ui.Separator()
            diagnostics_label = ui.Label("Initializing", word_wrap=True, height=190)
            target_label = ui.Label("Target transport grasp", word_wrap=True, height=100)
            status_label = ui.Label("PAUSED AT VERIFIED POST-HANDOFF KEYFRAME", word_wrap=True, height=70)

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_map = {
        "SPACE": "TOGGLE", "N": "STEP", "R": "RESET", "P": "REPLAY",
        "S": "SAVE", "V": "VERIFY", "C": "CLEAR", "Q": "QUIT",
    }

    def keyboard_callback(event: object, *_unused: object) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            action = keyboard_map.get(event.input.name)
            if action:
                enqueue(action)
        return True

    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_callback)

    def pump_viewport() -> None:
        prior = settings.get("/app/player/playSimulations")
        settings.set_bool("/app/player/playSimulations", False)
        try:
            simulation_app.update()
        finally:
            settings.set_bool("/app/player/playSimulations", True if prior is None else bool(prior))

    def render_and_pump() -> None:
        sim.forward()
        sim.render()
        sim.render_context.reset_transform_cadence()
        pump_viewport()

    def expose_window() -> None:
        control_window.visible = True
        workspace_window = ui.Workspace.get_window(window_title)
        property_window = ui.Workspace.get_window("Property")
        if workspace_window is None:
            raise RuntimeError("local teaching panel is absent from Kit workspace")
        if property_window is not None and workspace_window != property_window:
            workspace_window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
        workspace_window.visible = True
        workspace_window.focus()

    def clear_records() -> None:
        nonlocal records
        records = {key: [] for key in (
            "normalized_placeholder", "commanded_q_rad", "measured_q_rad",
            "object_position_world_m", "object_quaternion_xyzw",
            "object_relative_right_wrist_se3", "right_dex3_target_7d_rad",
            "thumb_force_n", "index_force_n", "middle_force_n", "table_force_n",
            "object_speed_m_s", "safety_pass",
        )}

    def reset_local(*, clear: bool = True) -> None:
        nonlocal playing, replay_frame, target
        sim.reset()
        target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
        target[0, joint_ids] = torch.as_tensor(keyframe["measured_q_rad"], device=robot.device, dtype=torch.float32)
        target_velocity.zero_()
        target_velocity[0, joint_ids] = torch.as_tensor(keyframe["measured_qd_rad_s"], device=robot.device, dtype=torch.float32)
        robot.write_joint_state_to_sim(target, target_velocity)
        target[0, joint_ids] = torch.as_tensor(keyframe["commanded_q_rad"], device=robot.device, dtype=torch.float32)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        doll_pose = torch.as_tensor(
            np.r_[keyframe["object_position_world_m"], keyframe["object_quaternion_xyzw"]][None],
            device=doll.device,
            dtype=torch.float32,
        )
        doll_velocity = torch.as_tensor(
            np.r_[keyframe["object_linear_velocity_m_s"], keyframe["object_angular_velocity_rad_s"]][None],
            device=doll.device,
            dtype=torch.float32,
        )
        doll.write_root_pose_to_sim_index(root_pose=doll_pose)
        doll.write_root_velocity_to_sim_index(root_velocity=doll_velocity)
        robot.update(0.0)
        doll.update(0.0)
        for sensor in digit_sensors.values():
            sensor.update(0.0, force_recompute=True)
        table_sensor.update(0.0, force_recompute=True)
        latest["command_q"] = keyframe["commanded_q_rad"].astype(np.float64).copy()
        latest["actual_q"] = keyframe["measured_q_rad"].astype(np.float64).copy()
        latest["object_pose"] = pose_from_xyzw(keyframe["object_position_world_m"], keyframe["object_quaternion_xyzw"])
        latest["safety"] = "PASS"
        for model in wrist_models.values():
            model.set_value(0.0)
        for name, index in zip(right_hand_names, right_hand_indices, strict=True):
            hand_models[name].set_value(float(latest["command_q"][index]))
        if clear:
            clear_records()
        replay_frame = 0
        playing = False
        render_and_pump()
        status_label.text = "PAUSED AT VERIFIED POST-HANDOFF KEYFRAME"

    def object_relative_slider_target(object_pose: np.ndarray) -> np.ndarray:
        value = baseline_object_to_wrist.copy()
        value[:3, 3] += np.asarray([wrist_models[k].as_float for k in ("dx", "dy", "dz")])
        delta_rotation = Rotation.from_euler(
            "xyz", [wrist_models[k].as_float for k in ("roll", "pitch", "yaw")], degrees=True
        ).as_matrix()
        value[:3, :3] = delta_rotation @ baseline_object_to_wrist[:3, :3]
        return object_pose @ value

    def solve_right_arm(target_world: np.ndarray, seed_q28: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        desired_position = kinematics.world_to_model_position(target_world[:3, 3])
        desired_rotation = kinematics.world_to_model_rotation(target_world[:3, :3])
        fixed_arms = seed_q28[:14].copy()
        seed = fixed_arms[7:14].copy()

        def residual(right: np.ndarray) -> np.ndarray:
            arms = fixed_arms.copy()
            arms[7:14] = right
            state = kinematics.wrist_state(arms)
            return np.r_[
                8.0 * (desired_position - state["right_position"]),
                0.8 * rotation_error(state["right_rotation"], desired_rotation),
                0.01 * (right - seed),
            ]

        bounds = (lower[right_arm_indices] + 1.0e-6, upper[right_arm_indices] - 1.0e-6)
        solution = least_squares(
            residual, seed, bounds=bounds, max_nfev=120,
            xtol=1.0e-9, ftol=1.0e-9, gtol=1.0e-9,
        )
        arms = fixed_arms.copy()
        arms[7:14] = solution.x
        state = kinematics.wrist_state(arms)
        return solution.x, {
            "position_error_m": float(np.linalg.norm(desired_position - state["right_position"])),
            "orientation_error_rad": float(np.linalg.norm(rotation_error(state["right_rotation"], desired_rotation))),
        }

    def safety_check(command: np.ndarray) -> tuple[bool, str]:
        if not np.isfinite(command).all():
            return False, "NONFINITE_COMMAND"
        if np.any(command < lower - 1.0e-9) or np.any(command > upper + 1.0e-9):
            return False, "JOINT_LIMIT"
        geometry = kinematics.trajectory_geometry(
            command[None, :14], command[None, 14:21], command[None, 21:28], 1.0e-5
        )
        collisions = {
            category: int(np.count_nonzero(flags))
            for category, flags in geometry["collision_flags"].items()
        }
        failed = [category for category, count in collisions.items() if count]
        return (not failed), ("PASS" if not failed else "COLLISION:" + ",".join(failed))

    def append_record() -> None:
        q = latest["actual_q"]
        object_pose = latest["object_pose"]
        object_to_wrist = inverse_pose(object_pose) @ wrist_world(q)
        records["normalized_placeholder"].append(0.0)
        records["commanded_q_rad"].append(latest["command_q"].copy())
        records["measured_q_rad"].append(q.copy())
        records["object_position_world_m"].append(object_pose[:3, 3].copy())
        records["object_quaternion_xyzw"].append(Rotation.from_matrix(object_pose[:3, :3]).as_quat())
        records["object_relative_right_wrist_se3"].append(object_to_wrist)
        records["right_dex3_target_7d_rad"].append(latest["command_q"][right_hand_indices].copy())
        records["thumb_force_n"].append(float(latest["forces"][0]))
        records["index_force_n"].append(float(latest["forces"][1]))
        records["middle_force_n"].append(float(latest["forces"][2]))
        records["table_force_n"].append(float(latest["table_force"]))
        records["object_speed_m_s"].append(float(latest["object_speed"]))
        records["safety_pass"].append(latest["safety"] == "PASS")

    def requested_command() -> tuple[np.ndarray, dict[str, float]]:
        if replay_rows is not None and replay_frame < len(replay_rows["right_dex3_target_7d_rad"]):
            object_to_wrist = replay_rows["object_relative_right_wrist_se3"][replay_frame]
            target_world = latest["object_pose"] @ object_to_wrist
            desired_hand = replay_rows["right_dex3_target_7d_rad"][replay_frame]
        else:
            target_world = object_relative_slider_target(latest["object_pose"])
            desired_hand = np.asarray([hand_models[name].as_float for name in right_hand_names])
        right_arm, errors = solve_right_arm(target_world, latest["command_q"])
        command = latest["command_q"].copy()
        command[right_arm_indices] = right_arm
        command[right_hand_indices] = desired_hand
        # Interactive target changes are rate-limited at the unchanged 30 Hz;
        # this is not a hidden trajectory optimizer and every resulting command
        # is recorded exactly.
        arm_delta = np.clip(command[right_arm_indices] - latest["command_q"][right_arm_indices], -0.015, 0.015)
        hand_delta = np.clip(command[right_hand_indices] - latest["command_q"][right_hand_indices], -0.025, 0.025)
        command[right_arm_indices] = latest["command_q"][right_arm_indices] + arm_delta
        command[right_hand_indices] = latest["command_q"][right_hand_indices] + hand_delta
        return command, errors

    def control_step() -> None:
        nonlocal replay_frame, playing
        command, errors = requested_command()
        safe, reason = safety_check(command)
        if not safe or errors["position_error_m"] > 0.005 or errors["orientation_error_rad"] > 0.08:
            playing = False
            latest["safety"] = reason if not safe else "IK_RESIDUAL"
            status_label.text = (
                f"COMMAND REJECTED: {latest['safety']} | IK pos={errors['position_error_m']:.4f} m "
                f"rot={errors['orientation_error_rad']:.4f} rad"
            )
            return
        target[0, joint_ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        for _ in range(substeps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            doll.update(dt)
            for sensor in digit_sensors.values():
                sensor.update(dt, force_recompute=True)
            table_sensor.update(dt, force_recompute=True)
        latest["command_q"] = command
        latest["actual_q"] = numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64)
        pose = numpy(doll.data.root_pose_w)[0].astype(np.float64)
        velocity = numpy(doll.data.root_vel_w)[0].astype(np.float64)
        latest["object_pose"] = pose_from_xyzw(pose[:3], pose[3:7])
        latest["object_speed"] = float(np.linalg.norm(velocity[:3]))
        force_values, contact_values = [], []
        for digit in ("thumb", "index", "middle"):
            rows = contact_rows(digit_sensors[digit], dt)
            force_values.append(float(sum(abs(row["force_n"]) for row in rows)))
            contact_values.append(len(rows))
        latest["forces"] = np.asarray(force_values)
        latest["contacts"] = np.asarray(contact_values)
        latest["table_force"] = filtered_force(table_sensor)
        latest["safety"] = "PASS"
        if latest["object_speed"] > float(config["gates"]["maximum_object_linear_speed_m_s"]):
            latest["safety"] = "OBJECT_SPEED_GATE"
            playing = False
            status_label.text = f"PAUSED: object speed {latest['object_speed']:.3f} m/s exceeds 1.0 m/s"
        append_record()
        if replay_rows is not None:
            replay_frame += 1
            if replay_frame >= len(replay_rows["right_dex3_target_7d_rad"]):
                playing = False
                status_label.text = "AUTONOMOUS LOCAL REPLAY COMPLETE — inspect gate, then save/verify"

    def current_gate() -> dict[str, Any]:
        count = len(records.get("table_force_n", []))
        hold = min(count, int(CONTROL_FPS))
        if hold == 0:
            return {"status": "NO_RECORDED_CONTROL_FRAMES"}
        forces = np.column_stack([
            records["thumb_force_n"], records["index_force_n"], records["middle_force_n"]
        ])
        threshold = float(config["gates"]["meaningful_digit_force_n"])
        table_threshold = float(config["gates"]["maximum_table_force_for_elevated_n"])
        endpoint_pose = np.asarray(records["object_relative_right_wrist_se3"][-1])
        endpoint_position_error = float(np.linalg.norm(endpoint_pose[:3, 3] - target_object_to_wrist[:3, 3]))
        endpoint_orientation_error = float(np.linalg.norm(rotation_error(endpoint_pose[:3, :3], target_object_to_wrist[:3, :3])))
        endpoint_hand_error = float(np.max(np.abs(np.asarray(records["right_dex3_target_7d_rad"][-1]) - target_hand)))
        checks = {
            "last_one_second_three_digit_support": bool(np.all(forces[-hold:] >= threshold)),
            "last_one_second_table_unsupported": bool(np.all(np.asarray(records["table_force_n"][-hold:]) <= table_threshold)),
            "all_recorded_safety_pass": bool(np.all(records["safety_pass"])),
            "object_speed_gate": bool(np.max(records["object_speed_m_s"]) <= float(config["gates"]["maximum_object_linear_speed_m_s"])),
            "transport_wrist_endpoint": endpoint_position_error <= 0.005 and endpoint_orientation_error <= 0.08,
            "transport_dex3_endpoint": endpoint_hand_error <= 0.03,
            "retention_duration": hold == int(CONTROL_FPS),
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "frames": count,
            "duration_s": count / CONTROL_FPS,
            "endpoint_position_error_m": endpoint_position_error,
            "endpoint_orientation_error_rad": endpoint_orientation_error,
            "endpoint_hand_error_rad": endpoint_hand_error,
            "maximum_object_speed_m_s": float(np.max(records["object_speed_m_s"])),
            "minimum_last_second_force_n_thumb_index_middle": np.min(forces[-hold:], axis=0).tolist(),
            "maximum_last_second_table_force_n": float(np.max(records["table_force_n"][-hold:])),
        }

    def save_teaching() -> Path | None:
        count = len(records.get("commanded_q_rad", []))
        if count < 2:
            status_label.text = "Nothing to save: run at least two control frames."
            return None
        arrays = {key: np.asarray(value) for key, value in records.items()}
        arrays["normalized_phase_time"] = np.linspace(0.0, 1.0, count)
        arrays.pop("normalized_placeholder")
        arrays["joint_names"] = np.asarray(names)
        arrays["right_hand_joint_names"] = np.asarray(right_hand_names)
        arrays["control_fps_hz"] = np.asarray(CONTROL_FPS)
        arrays["state_restoration_debug_only"] = np.asarray(True)
        arrays["object_relative_primitive"] = np.asarray(True)
        arrays["episode_id_encoded"] = np.asarray(False)
        arrays["source_frame_encoded"] = np.asarray(False)
        arrays["attachment_used"] = np.asarray(False)
        teaching_path = output_root / "latest_human_teaching.npz"
        atomic_npz(teaching_path, arrays)
        gate = current_gate()
        manifest = {
            "schema_version": "gui_local_handoff_transport_teaching_v1",
            "status": "TEACHING_LOCAL_GATE_PASS" if gate["status"] == "PASS" else "UNVALIDATED_HUMAN_TEACHING",
            "teaching": str(teaching_path),
            "teaching_sha256": sha256_file(teaching_path),
            "gate": gate,
            "source_keyframe": str(KEYFRAME),
            "source_keyframe_sha256": EXPECTED[KEYFRAME],
            "target_transport_command": str(G04_COMMAND),
            "target_transport_command_sha256": EXPECTED[G04_COMMAND],
            "object_relative": True,
            "episode_independent": True,
            "method_independent": True,
            "state_restoration_scope": "GUI_DIAGNOSTIC_ONLY",
            "permitted_in_final_continuous_validation": False,
            "doll_physics_changed": False,
            "controller_gains_changed": False,
            "prohibited_mechanisms_used": False,
        }
        atomic_json(output_root / "latest_human_teaching_manifest.json", manifest)
        status_label.text = f"SAVED: {manifest['status']} | {teaching_path}"
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return teaching_path

    def load_replay(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if not bool(data["object_relative_primitive"].item()):
            raise RuntimeError("teaching is not object-relative")
        return data

    def update_labels() -> None:
        q = latest["actual_q"]
        object_pose = latest["object_pose"]
        current_relative = inverse_pose(object_pose) @ wrist_world(q)
        current_rpy = Rotation.from_matrix(current_relative[:3, :3]).as_euler("xyz", degrees=True)
        target_rpy = Rotation.from_matrix(target_object_to_wrist[:3, :3]).as_euler("xyz", degrees=True)
        meaningful = latest["forces"] >= float(config["gates"]["meaningful_digit_force_n"])
        diagnostics_label.text = (
            "CONTACT thumb/index/middle: " + " / ".join("ON" if value else "off" for value in meaningful) + "\n"
            f"FORCE [N]: {latest['forces'][0]:.3f} / {latest['forces'][1]:.3f} / {latest['forces'][2]:.3f}\n"
            f"CONTACT PAIRS: {latest['contacts'].tolist()} | TABLE FORCE: {latest['table_force']:.4f} N\n"
            f"OBJECT COM [m]: {object_pose[0,3]:+.4f}, {object_pose[1,3]:+.4f}, {object_pose[2,3]:+.4f}\n"
            f"OBJECT QUAT xyzw: {Rotation.from_matrix(object_pose[:3,:3]).as_quat().round(4).tolist()}\n"
            f"RIGHT WRIST obj-pos [m]: {current_relative[:3,3].round(4).tolist()}\n"
            f"RIGHT WRIST obj-rpy [deg]: {current_rpy.round(2).tolist()}\n"
            f"RIGHT Dex3 q [rad]: {q[right_hand_indices].round(4).tolist()}\n"
            f"OBJECT SPEED: {latest['object_speed']:.3f} m/s | SAFETY: {latest['safety']}"
        )
        target_label.text = (
            f"TARGET wrist obj-pos [m]: {target_object_to_wrist[:3,3].round(4).tolist()}\n"
            f"TARGET wrist obj-rpy [deg]: {target_rpy.round(2).tolist()}\n"
            f"TARGET Dex3 [rad]: {target_hand.round(4).tolist()}"
        )

    clear_records()
    try:
        reset_local(clear=True)
        for _ in range(4):
            render_and_pump()
        expose_window()
        setup_manifest = {
            "schema_version": "gui_local_primitive_setup_v1",
            "status": "GUI_LOCAL_PRIMITIVE_READY_FOR_HUMAN_TEACHING",
            "tool": str(Path(__file__).resolve()),
            "source_keyframe": str(KEYFRAME),
            "target_transport_grasp": str(G04_COMMAND),
            "immutable_sha256": {str(path): digest for path, digest in EXPECTED.items()},
            "proxy": proxy_manifest,
            "editable_scope": ["object-relative RIGHT wrist xyz/rpy", "RIGHT Dex3 7D"],
            "state_restoration_scope": "GUI_DIAGNOSTIC_ONLY",
            "final_validation_may_restore_state": False,
            "prohibited_mechanisms_used": False,
        }
        atomic_json(output_root / "GUI_LOCAL_PRIMITIVE_SETUP.json", setup_manifest)
        print("GUI_LOCAL_PRIMITIVE_READY_FOR_HUMAN_TEACHING", flush=True)
        print(f"TEACHING_OUTPUT_ROOT={output_root}", flush=True)
        while simulation_app.is_running() and not quitting:
            loop_start = time.monotonic()
            single_step = False
            while actions:
                action = actions.popleft()
                if action == "TOGGLE":
                    playing = not playing
                    status_label.text = "PLAYING/RECORDING" if playing else "PAUSED"
                elif action == "STEP":
                    single_step = True
                elif action == "RESET":
                    reset_local(clear=True)
                elif action == "CLEAR":
                    clear_records()
                    status_label.text = "RECORDING CLEARED; PHYSICS STATE PRESERVED"
                elif action == "SAVE":
                    save_teaching()
                elif action == "REPLAY":
                    teaching = output_root / "latest_human_teaching.npz"
                    if teaching.is_file():
                        replay_rows = load_replay(teaching)
                        reset_local(clear=True)
                        replay_frame = 0
                        playing = True
                        status_label.text = "AUTONOMOUS REPLAY OF OBJECT-RELATIVE TEACHING"
                    else:
                        status_label.text = "No saved teaching exists. Press S first."
                elif action == "VERIFY":
                    teaching = output_root / "latest_human_teaching.npz"
                    manifest = output_root / "latest_human_teaching_manifest.json"
                    if teaching.is_file() and manifest.is_file() and read_json(manifest)["status"] == "TEACHING_LOCAL_GATE_PASS":
                        replay_rows = load_replay(teaching)
                        verification_results.clear()
                        verification_repeats_remaining = 3
                        reset_local(clear=True)
                        replay_frame = 0
                        playing = True
                        status_label.text = "AUTONOMOUS QUALIFICATION REPLAY 1/3"
                    else:
                        status_label.text = "Teaching must pass the local gate before 3x verification."
                elif action == "QUIT":
                    quitting = True
            if playing or single_step:
                control_step()
            if verification_repeats_remaining and not playing:
                gate = current_gate()
                verification_results.append(gate)
                verification_repeats_remaining -= 1
                if gate.get("status") != "PASS":
                    verification_repeats_remaining = 0
                    status_label.text = "AUTONOMOUS QUALIFICATION FAILED; teaching not accepted"
                elif verification_repeats_remaining:
                    repeat_number = 4 - verification_repeats_remaining
                    reset_local(clear=True)
                    replay_frame = 0
                    playing = True
                    status_label.text = f"AUTONOMOUS QUALIFICATION REPLAY {repeat_number}/3"
                else:
                    result = {
                        "schema_version": "autonomous_local_primitive_qualification_v1",
                        "status": "AUTONOMOUS_LOCAL_PRIMITIVE_3_OF_3_PASS",
                        "passes": 3,
                        "attempts": 3,
                        "results": verification_results,
                        "teaching_sha256": sha256_file(output_root / "latest_human_teaching.npz"),
                        "object_relative": True,
                        "state_restoration_scope": "LOCAL_QUALIFICATION_ONLY",
                        "final_continuous_validation_may_restore_state": False,
                    }
                    atomic_json(output_root / "AUTONOMOUS_LOCAL_PRIMITIVE_QUALIFICATION.json", result)
                    status_label.text = "AUTONOMOUS_LOCAL_PRIMITIVE_3_OF_3_PASS"
                    print("AUTONOMOUS_LOCAL_PRIMITIVE_3_OF_3_PASS", flush=True)
            render_and_pump()
            update_labels()
            remaining = (1.0 / CONTROL_FPS) - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
        control_window.visible = False
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
