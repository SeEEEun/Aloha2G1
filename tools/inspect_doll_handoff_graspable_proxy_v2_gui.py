#!/usr/bin/env python3
"""Interactive teacher-forced source-video ACT trajectory viewer in Isaac GUI.

Press F once to precompute (or hash-validate and reuse) one complete held-out
ACT-A40/ACT-B40 trajectory, reset the scene, and replay its exact joint poses
kinematically at 30 Hz. Policy input is original ALOHA cam_high RGB[t] plus the selected
retargeting method's frozen observation.state[t]. Isaac RGB and measured qpos
are never policy inputs. This is a visual motion diagnostic, not an object or
task-success evaluation, and it contains no real-robot transport.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gui", action="store_true", help="Required: launch the visible Kit GUI.")
parser.add_argument(
    "--autorun-full-act",
    action="store_true",
    help="Automatically replay frozen ACT-B40 held-out source episode 13 after Kit initialization.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    parser.error("this visual diagnostic requires --gui")
if getattr(args, "headless_explicit", False):
    parser.error("--gui cannot be combined with --headless")
if getattr(args, "visualizer_explicit", False) and "kit" not in set(args.visualizer or []):
    parser.error("--gui requires the Kit visualizer")
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
from pxr import UsdPhysics
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.paper_core_source_rollout_common import SafetyAudit, frozen_interfaces
from tools.policy_b_isaac_control_contract import (
    CONTROL_FPS,
    PHYSICS_DT,
    build_implicit_actuators,
)


HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EXPERIMENT2 = ROOT / "outputs/paper_core_ab/offline_heldout8/experiment2_result.json"
SCENE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
PRECOMPUTE = ROOT / "tools/precompute_teacher_forced_source_video_act.py"
POLICY_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
CACHE_ROOT = ROOT / "outputs/paper_core_ab/teacher_forced_source_video_gui_cache"
EXPECTED_HELDOUT_SHA256 = "a86181b049d0f521d1167c2b58bc15f3a7cb6ad87ee9a1f634ef02c04adcc710"
EXPECTED_EXPERIMENT2_SHA256 = "c3e0c24611997c5a61dcc8f1686dfd9b5a8691e8b9bf2db22f7a0013c62ee3ae"
EXPECTED_SCENE_SHA256 = "770f574db096819dd661b86d2bafdbd99964de7b77718e2013efeb95d1840aaf"
TABLE = "/World/DollHandoffEnvironment/Table/Colliders/Top"
G1 = "/World/G1/Asset"
DOLL = "/World/DollHandoffEnvironment/Doll"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected={expected} actual={actual} path={path}")


def cache_dir_for(
    method: str,
    heldout_episode: int,
    heldout: dict[str, Any],
    experiment2: dict[str, Any],
) -> Path:
    entry = heldout["entries"][heldout_episode]
    selection = experiment2["methods"][method]["checkpoint_selection"]
    return (
        CACHE_ROOT
        / f"act_{method}40"
        / f"heldout_{heldout_episode:02d}_source_{int(entry['final_dataset_index']):02d}"
        / (
            f"checkpoint_{int(selection['selected_step']):06d}_"
            f"{selection['selected_model_sha256'][:12]}_e1_k001"
        )
    )


def load_cache(
    path: Path,
    method: str,
    heldout_episode: int,
    heldout: dict[str, Any],
    experiment2: dict[str, Any],
    names: list[str],
    safety: SafetyAudit,
) -> dict[str, Any]:
    manifest_path = path / "cache_manifest.json"
    trajectory_path = path / "full_trajectory.npz"
    if not manifest_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(path)
    manifest = read_json(manifest_path)
    entry = heldout["entries"][heldout_episode]
    selection = experiment2["methods"][method]["checkpoint_selection"]
    expected_method = f"ACT-{method.upper()}40"
    checks = {
        "status": manifest.get("status") == "PASS",
        "experiment": manifest.get("experiment")
        == "TEACHER_FORCED_SOURCE_VIDEO_POLICY_VISUALIZATION",
        "method": manifest.get("method") == expected_method,
        "heldout_episode": manifest.get("heldout_output_episode") == heldout_episode,
        "source_episode": manifest.get("source_final_episode")
        == int(entry["final_dataset_index"]),
        "frames": manifest.get("frames") == int(entry["frames"]),
        "checkpoint": manifest.get("checkpoint_model_sha256")
        == selection["selected_model_sha256"],
        "source_video": manifest.get("source_video_sha256")
        == entry["source_rgb_identity"]["canonical_video_sha256"],
        "trajectory_hash": manifest.get("trajectory_sha256") == sha256_file(trajectory_path),
        "precompute_tool_hash": manifest.get("precompute_tool_sha256") == sha256_file(PRECOMPUTE),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "cached full trajectory failed validation: "
            + ", ".join(key for key, passed in checks.items() if not passed)
        )
    require_sha(
        Path(manifest["source_video"]),
        entry["source_rgb_identity"]["canonical_video_sha256"],
        "cached original ALOHA source video",
    )
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    frames = int(entry["frames"])
    required_shapes = {
        "raw_act_chunks": (frames, 50, 28),
        "raw_temporal_ensemble_action": (frames, 28),
        "deployment_safe_action": (frames, 28),
        "method_specific_reference_state": (frames, 28),
        "source_frame_index": (frames,),
        "source_timestamp_seconds": (frames,),
        "source_rgb_sha256": (frames,),
        "joint_names": (28,),
    }
    for key, shape in required_shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            actual = arrays[key].shape if key in arrays else None
            raise RuntimeError(f"cached array {key} has wrong shape: {actual}")
    if arrays["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("cached named joint order differs from frozen 28D interface")
    if not np.array_equal(arrays["source_frame_index"], np.arange(frames)):
        raise RuntimeError("cached source frame clock is not sequential")
    if not np.allclose(
        arrays["source_timestamp_seconds"],
        np.arange(frames) / CONTROL_FPS,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("cached source timestamps differ from the original 30-Hz clock")
    for key in (
        "raw_act_chunks",
        "raw_temporal_ensemble_action",
        "deployment_safe_action",
        "method_specific_reference_state",
    ):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"cached array {key} contains NaN/Inf")
        expected = manifest["array_sha256"].get(key)
        if expected is not None and sha256_array(arrays[key]) != expected:
            raise RuntimeError(f"cached array {key} hash changed")

    commands = arrays["deployment_safe_action"].astype(np.float64)
    states = arrays["method_specific_reference_state"].astype(np.float64)
    initial_velocity = (
        (states[1] - states[0]) * CONTROL_FPS
        if frames > 1
        else np.zeros(28, dtype=np.float64)
    )
    command_history: list[np.ndarray] = []
    first_abort = manifest["safety_preflight"].get("first_abort")
    for frame, command in enumerate(commands):
        audit = safety.command(command_history, states[frame], initial_velocity, command)
        if audit["status"] != "PASS" and first_abort is None:
            first_abort = {
                "frame": frame,
                "source_time_seconds": frame / CONTROL_FPS,
                "failed_checks": [key for key, passed in audit["checks"].items() if not passed],
                "audit": audit,
                "source": "UNCHANGED_ISAAC_ENVIRONMENT_SAFETY_AUDIT",
            }
        command_history.append(command.copy())
    return {
        "manifest": manifest,
        "arrays": arrays,
        "trajectory_path": trajectory_path,
        "method": method,
        "method_label": expected_method,
        "heldout_episode": heldout_episode,
        "entry": entry,
        "first_abort": first_abort,
    }


def main() -> int:
    require_sha(HELDOUT, EXPECTED_HELDOUT_SHA256, "frozen HELDOUT8 manifest")
    require_sha(EXPERIMENT2, EXPECTED_EXPERIMENT2_SHA256, "frozen Experiment-2 result")
    require_sha(SCENE, EXPECTED_SCENE_SHA256, "Doll-Handoff source scene")
    heldout = read_json(HELDOUT)
    experiment2 = read_json(EXPERIMENT2)
    if heldout.get("status") != "PASS" or heldout.get("episode_count") != 8:
        raise RuntimeError("HELDOUT8 is not frozen/PASS")
    if int(heldout.get("representative_episode")) != 13:
        raise RuntimeError("predeclared representative source episode changed")
    representative_slot = next(
        index
        for index, entry in enumerate(heldout["entries"])
        if int(entry["final_dataset_index"]) == int(heldout["representative_episode"])
    )
    if representative_slot != 1:
        raise RuntimeError("representative held-out slot changed")
    for method in ("a", "b"):
        selection = experiment2["methods"][method]["checkpoint_selection"]
        require_sha(
            Path(selection["selected_checkpoint"]) / "model.safetensors",
            selection["selected_model_sha256"],
            f"selected ACT-{method.upper()} checkpoint",
        )

    names, lower, upper, _projector = frozen_interfaces()
    safety = SafetyAudit(names, lower, upper)
    settings = carb.settings.get_settings()
    settings.set_string("/isaaclab/visualizer/types", "")
    settings.set_bool("/isaaclab/visualizer/explicit", True)
    settings.set_bool("/isaaclab/visualizer/disable_all", True)
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    if not omni.usd.get_context().open_stage(str(SCENE)):
        raise RuntimeError(f"failed to open Doll-Handoff scene: {SCENE}")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())

    # Existing paper-core visual-doll mode: retain authored doll/bin context,
    # remove doll collision response, and hold it kinematically.
    doll_prim = stage.GetPrimAtPath(DOLL)
    if not doll_prim.IsValid():
        raise RuntimeError("Doll-Handoff scene is missing the visual Doll prim")
    rigid = UsdPhysics.RigidBodyAPI.Get(stage, doll_prim.GetPath())
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateRigidBodyEnabledAttr(True)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(DOLL):
            collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
            if collision:
                collision.CreateCollisionEnabledAttr(False)

    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, device="cuda:0", use_fabric=True))
    robot = Articulation(
        ArticulationCfg(
            prim_path=f"{G1}/root_joint",
            spawn=None,
            actuators=build_implicit_actuators(ImplicitActuatorCfg),
        )
    )
    table_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=f"{G1}/.*_link",
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
    if missing or len(joint_ids) != 28 or len(set(joint_ids)) != 28:
        raise RuntimeError(f"Isaac named joint mapping failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero_joint_velocity = torch.zeros_like(target)
    steps_per_control = int(round((1.0 / CONTROL_FPS) / PHYSICS_DT))
    if steps_per_control != 4:
        raise RuntimeError("frozen 30-Hz controller is not four 120-Hz physics steps")

    layout = read_json(SCENE_LAYOUT)
    camera = layout["camera"]["presets"]["overview"]
    eye = list(map(float, camera["eye_world_xyz_m"]))
    look_at = list(map(float, camera["target_world_xyz_m"]))
    sim.set_camera_view(eye, look_at)
    from isaacsim.core.rendering_manager import ViewportManager
    from omni.kit.viewport.utility import get_active_viewport_window

    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye, target=look_at)

    actions: deque[str] = deque()
    active: dict[str, Any] | None = None
    source_capture: cv2.VideoCapture | None = None
    pending_source_rgb: np.ndarray | None = None
    displayed_source_frame = 0
    action_frame = 0
    playing = False
    quitting = False
    live_abort: dict[str, Any] | None = None
    pose_history: list[np.ndarray] = []
    kinematic_velocity_history: list[np.ndarray] = []
    precompute_process: subprocess.Popen[str] | None = None
    precompute_log_stream: Any = None
    precompute_log_path: Path | None = None
    precompute_request: tuple[str, int, Path] | None = None
    autorun_contract_active = False

    def enqueue(action: str) -> None:
        actions.append(action)

    image_provider = ui.ByteImageProvider()
    blank = np.zeros((480, 640, 4), dtype=np.uint8)
    blank[..., 3] = 255
    image_provider.set_bytes_data(blank.flatten().data, [640, 480])
    episode_labels = [
        (
            f"source ep {int(entry['final_dataset_index']):02d}"
            + ("  [REPRESENTATIVE]" if int(entry["final_dataset_index"]) == 13 else "")
        )
        for entry in heldout["entries"]
    ]
    control_window_title = "Full Source-Video-Conditioned ACT Run"
    control_window = ui.Window(
        control_window_title,
        width=440,
        height=720,
        visible=True,
        dock_preference=ui.DockPreference.RIGHT_TOP,
    )
    with control_window.frame:
        with ui.VStack(spacing=6):
            ui.Label("PHYSICAL OBJECT SUCCESS NOT EVALUATED", height=24)
            ui.Label("Teacher-forced input; exact kinematic action-trajectory replay", height=22)
            with ui.HStack(height=28, spacing=5):
                ui.Label("Method", width=75)
                method_combo = ui.ComboBox(1, "ACT-A", "ACT-B", width=330)
            with ui.HStack(height=28, spacing=5):
                ui.Label("Episode", width=75)
                episode_combo = ui.ComboBox(representative_slot, *episode_labels, width=330)
            ui.Button(
                "RUN FULL ACT TRAJECTORY",
                height=42,
                clicked_fn=lambda: enqueue("FULL"),
            )
            with ui.HStack(height=32, spacing=4):
                ui.Button("Space  Pause/Resume", clicked_fn=lambda: enqueue("TOGGLE"))
                ui.Button("R  Replay", clicked_fn=lambda: enqueue("REPLAY"))
                ui.Button("X  Reset", clicked_fn=lambda: enqueue("RESET"))
                ui.Button("Q  Quit", clicked_fn=lambda: enqueue("QUIT"))
            overlay_label = ui.Label("Initializing", word_wrap=True, height=120)
            ui.Label("Original ALOHA cam_high RGB currently associated with the action", height=22)
            ui.ImageWithProvider(
                image_provider,
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                width=416,
                height=312,
                pixel_aligned=True,
            )
            status_label = ui.Label("READY — press F", word_wrap=True, height=70)
            ui.Label("Main viewport camera remains freely orbitable / pannable / zoomable.", height=22)

    method_combo.model.add_item_changed_fn(lambda _model, _item: enqueue("SELECTION"))
    episode_combo.model.add_item_changed_fn(lambda _model, _item: enqueue("SELECTION"))
    keyboard_map = {
        "F": "FULL",
        "SPACE": "TOGGLE",
        "R": "REPLAY",
        "X": "RESET",
        "Q": "QUIT",
    }

    def keyboard_callback(event: object, *_unused: object) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            action = keyboard_map.get(event.input.name)
            if action is not None:
                enqueue(action)
        return True

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_callback)

    def selected_identity() -> tuple[str, int]:
        method = ("a", "b")[method_combo.model.get_item_value_model().as_int]
        episode = int(episode_combo.model.get_item_value_model().as_int)
        return method, episode

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

    def expose_control_window() -> None:
        """Override any persisted off-screen/floating Kit window placement."""

        control_window.visible = True
        workspace_window = ui.Workspace.get_window(control_window_title)
        property_window = ui.Workspace.get_window("Property")
        if workspace_window is None:
            raise RuntimeError("custom ACT control window is absent from the Kit workspace")
        if property_window is not None and workspace_window != property_window:
            workspace_window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
        workspace_window.visible = True
        workspace_window.focus()
        print("[FULL ACT GUI] custom control/source-preview panel forced visible", flush=True)

    def set_source_preview(rgb: np.ndarray) -> None:
        rgba = np.empty((480, 640, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        image_provider.set_bytes_data(rgba.flatten().data, [640, 480])

    def close_source() -> None:
        nonlocal source_capture, pending_source_rgb
        if source_capture is not None:
            source_capture.release()
        source_capture = None
        pending_source_rgb = None

    def read_source_frame(frame: int) -> np.ndarray:
        nonlocal pending_source_rgb
        if active is None or source_capture is None:
            raise RuntimeError("source video is not open")
        if frame == 0 and pending_source_rgb is not None:
            rgb = pending_source_rgb
            pending_source_rgb = None
        else:
            ok, bgr = source_capture.read()
            if not ok:
                raise RuntimeError(f"source video ended at frame {frame}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"invalid source RGB at frame {frame}")
        expected = str(active["arrays"]["source_rgb_sha256"][frame])
        if sha256_array(rgb) != expected:
            raise RuntimeError(f"decoded source RGB identity mismatch at frame {frame}")
        return rgb

    def measured_state() -> np.ndarray:
        value = robot.data.joint_pos.torch[0, joint_ids].detach().cpu().numpy().astype(np.float64)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("non-finite/malformed measured 28D state")
        return value

    def measured_velocity() -> np.ndarray:
        value = robot.data.joint_vel.torch[0, joint_ids].detach().cpu().numpy().astype(np.float64)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("non-finite/malformed measured 28D velocity")
        return value

    def table_contact_force_n() -> float:
        matrix = table_sensor.data.force_matrix_w
        if matrix is None:
            return 0.0
        value = matrix.torch.detach().cpu().numpy()
        return float(np.max(np.linalg.norm(value.reshape(-1, 3), axis=1))) if value.size else 0.0

    def reset_full(autoplay: bool) -> None:
        nonlocal action_frame, displayed_source_frame, playing, live_abort
        nonlocal pending_source_rgb, source_capture, pose_history
        nonlocal kinematic_velocity_history, target
        if active is None:
            status_label.text = "No full trajectory loaded. Press F to generate/load one."
            return
        sim.reset()
        states = active["arrays"]["method_specific_reference_state"]
        target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
        target[0, joint_ids] = torch.as_tensor(states[0], device=robot.device, dtype=torch.float32)
        robot.write_joint_state_to_sim(target, zero_joint_velocity)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        robot.update(0.0)
        table_sensor.update(0.0)
        close_source()
        source_capture = cv2.VideoCapture(str(active["manifest"]["source_video"]))
        if not source_capture.isOpened():
            raise RuntimeError("could not open the frozen source video for GUI preview")
        ok, bgr = source_capture.read()
        if not ok:
            raise RuntimeError("could not decode source frame 0")
        pending_source_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if sha256_array(pending_source_rgb) != str(active["arrays"]["source_rgb_sha256"][0]):
            raise RuntimeError("source frame-0 hash mismatch on replay reset")
        set_source_preview(pending_source_rgb)
        action_frame = 0
        displayed_source_frame = 0
        playing = autoplay
        live_abort = None
        pose_history = [measured_state()]
        kinematic_velocity_history = [np.zeros(28, dtype=np.float64)]
        render_and_pump()
        status_label.text = "PLAYING FULL EPISODE" if autoplay else "RESET AT SOURCE FRAME 0 — PAUSED"
        print(
            f"[FULL ACT GUI] {'PLAY' if autoplay else 'RESET/PAUSE'} "
            f"{active['method_label']} source_ep={active['entry']['final_dataset_index']} "
            f"frames={active['manifest']['frames']} cache={active['trajectory_path']}",
            flush=True,
        )

    def activate_cache(method: str, episode: int, cache_path: Path) -> None:
        nonlocal active
        status_label.text = "Validating cache and unchanged kinematic safety audit..."
        render_and_pump()
        active = load_cache(cache_path, method, episode, heldout, experiment2, names, safety)
        if active["first_abort"] is None:
            print("[FULL ACT GUI] complete trajectory safety preflight PASS", flush=True)
        else:
            print(f"[FULL ACT GUI] safety abort predeclared: {active['first_abort']}", flush=True)
        if autorun_contract_active:
            print("FULL_ACT_TRAJECTORY_READY", flush=True)
        reset_full(autoplay=True)
        if autorun_contract_active:
            print("FULL_ACT_REPLAY_START", flush=True)

    def begin_full(
        identity_override: tuple[str, int] | None = None,
        *,
        autorun: bool = False,
    ) -> None:
        nonlocal playing, precompute_process, precompute_log_stream
        nonlocal precompute_log_path, precompute_request, autorun_contract_active
        if playing:
            status_label.text = "FULL EPISODE ALREADY PLAYING — use R for an explicit replay."
            return
        if precompute_process is not None:
            status_label.text = "ACT trajectory precompute is already running."
            return
        playing = False
        if autorun:
            autorun_contract_active = True
        method, episode = identity_override or selected_identity()
        path = cache_dir_for(method, episode, heldout, experiment2)
        try:
            activate_cache(method, episode, path)
            status_label.text = "CACHE VALIDATED — PLAYING FULL EPISODE"
            return
        except FileNotFoundError:
            pass
        path.mkdir(parents=True, exist_ok=True)
        precompute_log_path = path / "gui_precompute.log"
        precompute_log_stream = precompute_log_path.open("w", encoding="utf-8")
        command = [
            str(POLICY_PYTHON),
            str(PRECOMPUTE),
            "--method",
            method,
            "--heldout-episode",
            str(episode),
            "--output",
            str(path),
        ]
        precompute_process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=precompute_log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        precompute_request = (method, episode, path)
        entry = heldout["entries"][episode]
        status_label.text = (
            f"PRECOMPUTING ACT-{method.upper()}40: source episode "
            f"{entry['final_dataset_index']} — playback will start automatically"
        )
        print("[FULL ACT GUI] precompute launched: " + " ".join(command), flush=True)

    def poll_precompute() -> None:
        nonlocal precompute_process, precompute_log_stream, precompute_request
        if precompute_process is None or precompute_request is None:
            return
        method, episode, path = precompute_request
        return_code = precompute_process.poll()
        if return_code is None:
            progress_path = path / "progress.json"
            if progress_path.is_file():
                try:
                    progress = read_json(progress_path)
                    status_label.text = (
                        f"{progress.get('status', 'PRECOMPUTING')} | "
                        f"{progress.get('frame', 0)} / {progress.get('frames', '?')} | "
                        "automatic replay follows"
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            return
        precompute_log_stream.close()
        precompute_log_stream = None
        precompute_process = None
        precompute_request = None
        if return_code != 0:
            tail = ""
            if precompute_log_path is not None and precompute_log_path.is_file():
                tail = "\n".join(precompute_log_path.read_text(encoding="utf-8").splitlines()[-8:])
            status_label.text = f"PRECOMPUTE FAILED (exit {return_code}); see {precompute_log_path}"
            print(f"[FULL ACT GUI] precompute failed exit={return_code}\n{tail}", flush=True)
            return
        try:
            activate_cache(method, episode, path)
            status_label.text = "PRECOMPUTE COMPLETE — PLAYING FULL EPISODE"
        except Exception as exc:
            status_label.text = f"CACHE LOAD FAILED: {exc}"
            print(f"[FULL ACT GUI] cache activation failed: {exc}", flush=True)

    def update_overlay() -> None:
        if active is None:
            method, episode = selected_identity()
            source_episode = heldout["entries"][episode]["final_dataset_index"]
            overlay_label.text = (
                f"METHOD: ACT-{method.upper()}\n"
                f"SOURCE EPISODE: {source_episode}\n"
                "SOURCE FRAME: 0 / ?\nSOURCE TIME: 0.000 s\n"
                "POLICY: ACT\nMODE: TEACHER_FORCED_SOURCE_VIDEO / KINEMATIC REPLAY\n"
                "CURRENT 28D ACTION FRAME: not loaded"
            )
            return
        total = int(active["manifest"]["frames"])
        overlay_label.text = (
            f"METHOD: ACT-{active['method'].upper()}\n"
            f"SOURCE EPISODE: {active['entry']['final_dataset_index']}\n"
            f"SOURCE FRAME: {displayed_source_frame} / {total - 1}\n"
            f"SOURCE TIME: {displayed_source_frame / CONTROL_FPS:.3f} s\n"
            "POLICY: ACT\nMODE: TEACHER_FORCED_SOURCE_VIDEO / KINEMATIC REPLAY\n"
            f"CURRENT 28D ACTION FRAME: {min(action_frame, total - 1)}"
        )

    try:
        # Stage, articulation, and viewport must all survive several complete
        # Kit updates before an autorun trigger is accepted.
        for _ in range(4):
            render_and_pump()
        if get_active_viewport_window() is None:
            raise RuntimeError("visible Kit viewport is unavailable")
        expose_control_window()
        render_and_pump()
        print("[GUI] VISIBLE KIT WINDOW READY", flush=True)
        print(
            "[FULL ACT GUI] default=ACT-B40 heldout_slot=1 source_episode=13 "
            "mode=TEACHER_FORCED_SOURCE_VIDEO_POLICY_VISUALIZATION",
            flush=True,
        )
        print(
            "[FULL ACT GUI] F=full run SPACE=pause/resume R=replay X=reset Q=quit; "
            "physical object success is not evaluated",
            flush=True,
        )
        if args.autorun_full_act:
            print("FULL_ACT_TRIGGER_RECEIVED", flush=True)
            actions.append("AUTORUN")
        while simulation_app.is_running() and not quitting:
            loop_start = time.monotonic()
            poll_precompute()
            while actions:
                action = actions.popleft()
                if action == "FULL":
                    begin_full()
                elif action == "AUTORUN":
                    begin_full(("b", representative_slot), autorun=True)
                elif action == "TOGGLE":
                    if active is not None and live_abort is None:
                        playing = not playing
                        status_label.text = "PLAYING" if playing else "PAUSED"
                elif action == "REPLAY":
                    if active is not None:
                        reset_full(autoplay=True)
                    else:
                        begin_full()
                elif action == "RESET":
                    if active is not None:
                        reset_full(autoplay=False)
                elif action == "SELECTION":
                    if not playing and precompute_process is None:
                        status_label.text = "Selection changed — press F for that complete trajectory."
                elif action == "QUIT":
                    quitting = True

            if playing and active is not None:
                total = int(active["manifest"]["frames"])
                if action_frame >= total:
                    playing = False
                else:
                    rgb = read_source_frame(action_frame)
                    displayed_source_frame = action_frame
                    set_source_preview(rgb)
                    declared_abort = active["first_abort"]
                    if declared_abort is not None and action_frame >= int(declared_abort["frame"]):
                        playing = False
                        live_abort = declared_abort
                        reasons = ", ".join(declared_abort["failed_checks"])
                        status_label.text = f"SAFETY ABORT BEFORE FRAME {action_frame}: {reasons}"
                        print(f"[FULL ACT GUI] SAFETY ABORT frame={action_frame} reasons={reasons}", flush=True)
                    else:
                        command = active["arrays"]["deployment_safe_action"][action_frame]
                        previous_pose = pose_history[-1]
                        command_velocity = (command - previous_pose) * CONTROL_FPS
                        target[0, joint_ids] = torch.as_tensor(
                            command, device=robot.device, dtype=torch.float32
                        )
                        kinematic_velocity = torch.zeros_like(target)
                        kinematic_velocity[0, joint_ids] = torch.as_tensor(
                            command_velocity, device=robot.device, dtype=torch.float32
                        )
                        # This GUI is an exact action-trajectory visualizer. Direct pose
                        # writes are the repository's established Isaac kinematic-render
                        # convention and avoid adding implicit-drive overshoot to an
                        # already-preflighted policy trajectory. No action is altered.
                        robot.write_joint_state_to_sim(target, kinematic_velocity)
                        sim.forward()
                        robot.update(0.0)
                        readback = measured_state()
                        readback_error = float(np.max(np.abs(readback - command)))
                        pose_history.append(readback)
                        kinematic_velocity_history.append(command_velocity.copy())
                        measured_audit = safety.measured(
                            pose_history, kinematic_velocity_history
                        )
                        action_frame += 1
                        if measured_audit["status"] != "PASS" or readback_error > 1e-5:
                            playing = False
                            reasons = [
                                key for key, passed in measured_audit["checks"].items() if not passed
                            ]
                            if readback_error > 1e-5:
                                reasons.append(f"KINEMATIC_READBACK_ERROR_{readback_error:.9f}RAD")
                            live_abort = {
                                "frame": displayed_source_frame,
                                "failed_checks": reasons,
                                "measured_audit": measured_audit,
                                "kinematic_readback_error_rad": readback_error,
                            }
                            status_label.text = (
                                f"SAFETY ABORT AT FRAME {displayed_source_frame}: "
                                + ", ".join(reasons)
                            )
                            print(f"[FULL ACT GUI] LIVE SAFETY ABORT {live_abort}", flush=True)
                        else:
                            if autorun_contract_active:
                                print(
                                    f"FULL_ACT_FRAME {displayed_source_frame}/{total}",
                                    flush=True,
                                )
                            if action_frame >= total:
                                playing = False
                                status_label.text = (
                                    "FULL SOURCE EPISODE COMPLETE — final physical pose held; "
                                    "physical object success not evaluated"
                                )
                                print(
                                    f"[FULL ACT GUI] COMPLETE {active['method_label']} "
                                    f"source_ep={active['entry']['final_dataset_index']} frames={total}",
                                    flush=True,
                                )
                                if autorun_contract_active:
                                    print("FULL_ACT_REPLAY_DONE", flush=True)
                                    autorun_contract_active = False
            render_and_pump()
            update_overlay()
            remaining = (1.0 / CONTROL_FPS) - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        playing = False
        close_source()
        if precompute_process is not None and precompute_process.poll() is None:
            precompute_process.terminate()
            try:
                precompute_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                precompute_process.kill()
                precompute_process.wait(timeout=5.0)
        if precompute_log_stream is not None:
            precompute_log_stream.close()
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
        control_window.visible = False
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
