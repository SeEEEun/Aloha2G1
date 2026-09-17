#!/usr/bin/env python3
"""Run one source-video-conditioned ACT-A40 or ACT-B40 rollout in Isaac.

This is explicitly not an onboard-G1 visual policy.  At logical time ``t`` the
policy receives frozen source ALOHA cam_high RGB[t] plus the current measured
Isaac 28D state.  The source clock advances at its original 30 Hz regardless of
policy progress.  Execution is the frozen official ACT-E1 configuration and
the frozen common policy-independent Dex3 deployment projection only.

The visual doll and bin are kinematic context.  Physical grasp, handoff, and
drop success are neither required nor claimed.  No real-robot transport exists.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np

from isaaclab.app import AppLauncher

from deployment_camera_config import load_camera_config
from paper_core_source_rollout_common import (
    ACTE1Bridge,
    COMMON_PROJECTION_FREEZE,
    COMMON_PROJECTION_SHA256,
    EXECUTION_CONFIG,
    EXECUTION_CONFIG_SHA256,
    INITIAL_CONTRACT_SHA256,
    ROOT,
    SCENE_LAYOUT,
    SafetyAudit,
    SourceVideo,
    RolloutVideoRecorder,
    atomic_json,
    atomic_npz,
    common_initial_condition,
    frozen_interfaces,
    look_at_ros_camera_quaternion_xyzw,
    read_json,
    rollout_dynamics,
    sha256_array,
    sha256_file,
    method_consistent_initial_condition,
)
from policy_b_isaac_control_contract import (
    CONTROL_FPS,
    PHYSICS_DT,
    build_implicit_actuators,
)


SCENE_STAGE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
HELDOUT_MANIFEST = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EXPERIMENT2 = ROOT / "outputs/paper_core_ab/offline_heldout8/experiment2_result.json"
DEFAULT_CAMERA = ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=("a", "b"), required=True)
parser.add_argument("--heldout-episode", type=int, choices=range(8), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
parser.add_argument("--no-video", action="store_true")
parser.add_argument(
    "--initialization-mode",
    choices=("common_ab_v1", "method_consistent_v1"),
    default="common_ab_v1",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video
launcher = AppLauncher(args)
simulation_app = launcher.app


def method_record() -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    if not EXPERIMENT2.is_file():
        raise FileNotFoundError("Experiment-2 selection must be complete before Experiment 3")
    result = read_json(EXPERIMENT2)
    if result.get("status") != "PASS":
        raise RuntimeError("Experiment-2 selected-checkpoint result is not PASS")
    row = result["methods"][args.method]
    selection = row["checkpoint_selection"]
    checkpoint = Path(selection["selected_checkpoint"]).resolve()
    model_sha = str(selection["selected_model_sha256"])
    if sha256_file(checkpoint / "model.safetensors") != model_sha:
        raise RuntimeError("selected paper ACT checkpoint hash changed")
    manifest = read_json(HELDOUT_MANIFEST)
    if manifest.get("status") != "PASS" or manifest["episode_count"] != 8:
        raise RuntimeError("HELDOUT8 manifest is not frozen and valid")
    entry = manifest["entries"][args.heldout_episode]
    return row, entry, checkpoint, model_sha


def run() -> None:
    import carb
    import omni.usd
    import torch
    from pxr import UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rollout output: {output}")
    output.mkdir(parents=True)
    method_result, entry, checkpoint, model_sha = method_record()
    source_episode = int(entry["final_dataset_index"])
    expected_frames = int(entry["frames"])
    source_video_path = Path(entry["source_rgb_identity"]["canonical_video_path"])
    expected_video_sha = str(entry["source_rgb_identity"]["canonical_video_sha256"])
    if sha256_file(source_video_path) != expected_video_sha:
        raise RuntimeError("frozen source ALOHA video hash changed")

    names, lower, upper, projector = frozen_interfaces()
    if args.initialization_mode == "method_consistent_v1":
        experiment_name = "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT"
        init_q, initial_record = method_consistent_initial_condition(
            names,
            args.method,
            args.heldout_episode,
            source_episode,
            str(entry["stable_episode_id"]),
            args.settle_seconds,
        )
        excess = np.maximum(np.maximum(lower - init_q, init_q - upper), 0.0)
        arm_excess = float(np.max(excess[:14]))
        dex3_excess = float(np.max(excess[14:]))
        dex3_cap = float(
            initial_record["preexisting_measured_dex3_excursion_cap_rad"]
        )
        if arm_excess > 1e-9 or dex3_excess > dex3_cap:
            raise RuntimeError("method-consistent exact state[0] exceeds the frozen reset allowance")
    else:
        experiment_name = "SOURCE-VIDEO-CONDITIONED G1 POLICY ROLLOUT"
        init_q, initial_record = common_initial_condition(names, args.settle_seconds)
        if np.any(init_q < lower) or np.any(init_q > upper):
            raise RuntimeError("common A/B initial pose violates hard limits")
    atomic_json(output / "initial_condition.json", initial_record)
    atomic_json(
        output / "experiment_contract.json",
        {
            "schema_version": "paper_core_source_conditioned_rollout_contract_v1",
            "experiment_name": experiment_name,
            "method": f"ACT-{args.method.upper()}40",
            "heldout_output_episode": args.heldout_episode,
            "source_final_episode": source_episode,
            "stable_episode_id": entry["stable_episode_id"],
            "source_recording_id": entry["original_source_recording_id"],
            "source_video": str(source_video_path),
            "source_video_sha256": expected_video_sha,
            "source_clock": "original frame t at t/30; never paused, jumped, or policy-progress aligned",
            "input": "source ALOHA RGB[t] + current measured Isaac G1/Dex3 28D state",
            "g1_onboard_rgb_used": False,
            "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": model_sha,
            "checkpoint_step": method_result["checkpoint_selection"]["selected_step"],
            "execution_config": str(EXECUTION_CONFIG),
            "execution_config_sha256": EXECUTION_CONFIG_SHA256,
            "execution": "installed official ACTTemporalEnsembler coefficient 0.01",
            "initialization_mode": args.initialization_mode,
            "initial_state_contract": initial_record["contract"],
            "initial_state_contract_sha256": initial_record["contract_sha256"],
            "method_consistent_state0": args.initialization_mode
            == "method_consistent_v1",
            "common_projection": str(COMMON_PROJECTION_FREEZE),
            "common_projection_sha256": COMMON_PROJECTION_SHA256,
            "object_mode": "kinematic visual doll; collision response disabled",
            "physical_manipulation_success_required": False,
            "physical_manipulation_success_claimed": False,
            "real_hardware_transport": False,
            "dds_command_publisher": False,
        },
    )
    safety = SafetyAudit(names, lower, upper)
    initial_collision = safety.collision(init_q[None])
    if initial_collision["invalid_hard_self_collision_incidence"]:
        raise RuntimeError("selected initial pose has a hard self collision")
    if not SCENE_STAGE.is_file():
        raise FileNotFoundError(SCENE_STAGE)

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"failed to open scene stage: {SCENE_STAGE}")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    sys.path.insert(0, str(ROOT / "isaaclab_magsafe_fixed_scene"))
    from physical_contact_monitor import enable_contact_reporting

    enable_contact_reporting(
        stage,
        [
            "/World/G1/Asset",
            "/World/DollHandoffEnvironment/Doll",
            "/World/DollHandoffEnvironment/Table",
            "/World/DollHandoffEnvironment/TrashBin",
        ],
    )
    doll_prim = stage.GetPrimAtPath("/World/DollHandoffEnvironment/Doll")
    if not doll_prim.IsValid():
        raise RuntimeError("diagnostic stage is missing the Doll prim")
    rigid = UsdPhysics.RigidBodyAPI.Get(stage, doll_prim.GetPath())
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateRigidBodyEnabledAttr(True)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/DollHandoffEnvironment/Doll") or path.startswith(
            "/World/DollHandoffEnvironment/TrashBin"
        ):
            collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
            if collision:
                collision.CreateCollisionEnabledAttr(False)

    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, device="cuda:0", use_fabric=True))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint",
            spawn=None,
            actuators=build_implicit_actuators(ImplicitActuatorCfg),
        )
    )
    table_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/G1/Asset/.*_link",
            update_period=0.0,
            filter_prim_paths_expr=["/World/DollHandoffEnvironment/Table/Colliders/Top"],
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            force_threshold=0.0,
        )
    )

    cameras: dict[str, Any] = {}
    if not args.no_video:
        camera_config = load_camera_config(
            args.camera_config, purpose="paper-core source-conditioned comparison recording"
        )

        def camera_spawn() -> Any:
            return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                camera_config.intrinsic_matrix.reshape(-1).tolist(),
                width=640,
                height=480,
                clipping_range=camera_config.clipping_range_m,
                lock_camera=True,
            )

        cameras = {
            name: Camera(
                CameraCfg(
                    prim_path=f"/World/PaperCore{name.title().replace('_', '')}Camera",
                    update_period=0.0,
                    width=640,
                    height=480,
                    data_types=["rgb"],
                    spawn=camera_spawn(),
                )
            )
            for name in ("overview", "three_quarter")
        }

    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError(f"Isaac named joint mapping failed: {missing}")
    if cameras:
        layout = read_json(SCENE_LAYOUT)
        presets = {"overview": "overview", "three_quarter": "legacy_overview"}
        for name, preset_name in presets.items():
            preset = layout["camera"]["presets"][preset_name]
            eye = np.asarray(preset["eye_world_xyz_m"], dtype=np.float32)
            target_point = np.asarray(preset["target_world_xyz_m"], dtype=np.float32)
            quaternion = look_at_ros_camera_quaternion_xyzw(eye, target_point).astype(np.float32)
            cameras[name].set_world_poses(eye[None], quaternion[None], convention="ros")

    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    target[0, ids] = torch.as_tensor(init_q, device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(target, zero)
    sim.forward()
    robot.update(0.0)
    table_sensor.update(0.0)
    immediate = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
    reset_tolerance = float(initial_record["tolerances"]["immediate_reset_max_abs_from_nominal_rad"])
    if float(np.max(np.abs(immediate - init_q))) > reset_tolerance:
        raise RuntimeError("initial-state immediate-reset tolerance failed")
    steps_per_control = int(round((1.0 / CONTROL_FPS) / PHYSICS_DT))

    def measured_state() -> np.ndarray:
        value = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("malformed/non-finite Isaac measured state")
        return value

    def measured_velocity() -> np.ndarray:
        value = robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("malformed/non-finite Isaac measured velocity")
        return value

    def table_contact_force_n() -> float:
        matrix = table_sensor.data.force_matrix_w
        if matrix is None:
            return 0.0
        value = matrix.torch.detach().cpu().numpy()
        return float(np.max(np.linalg.norm(value.reshape(-1, 3), axis=1))) if value.size else 0.0

    def capture() -> dict[str, np.ndarray]:
        if not cameras:
            return {}
        sim.forward()
        sim.render()
        sim.render_context.reset_transform_cadence()
        images = {}
        for name, camera in cameras.items():
            camera.update(PHYSICS_DT, force_recompute=True)
            image = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3]
            images[name] = image.astype(np.uint8) if image.dtype != np.uint8 else image.copy()
        return images

    for _ in range(max(1, int(round(args.settle_seconds / PHYSICS_DT)))):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT)
        table_sensor.update(PHYSICS_DT)
    settled = measured_state()
    settle_tolerance = float(initial_record["tolerances"]["post_settle_max_abs_from_nominal_rad"])
    if float(np.max(np.abs(settled - init_q))) > settle_tolerance:
        raise RuntimeError("initial-state post-settle tolerance failed")
    atomic_json(
        output / "initial_state_contract_verification.json",
        {
            "status": "PASS",
            "contract_sha256": initial_record["contract_sha256"],
            "initialization_mode": args.initialization_mode,
            "joint_names": names,
            "immediate_reset_max_abs_from_nominal_rad": float(np.max(np.abs(immediate - init_q))),
            "immediate_reset_tolerance_rad": reset_tolerance,
            "settled_max_abs_from_nominal_rad": float(np.max(np.abs(settled - init_q))),
            "settled_tolerance_rad": settle_tolerance,
            "policy_specific_tuned_pose": False,
            "retargeting_method_specific": args.initialization_mode
            == "method_consistent_v1",
            "episode_specific": args.initialization_mode == "method_consistent_v1",
            "exact_frozen_state0_unmodified": args.initialization_mode
            == "method_consistent_v1",
        },
    )

    source = SourceVideo(source_video_path, expected_frames)
    source_provenance = source.provenance()
    atomic_json(output / "source_video_provenance.json", source_provenance)
    recorder = RolloutVideoRecorder(output, not args.no_video, args.method, source_episode)
    bridge: ACTE1Bridge | None = None
    commands: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    normalized_actions: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    normalized_chunks: list[np.ndarray] = []
    hard_actions: list[np.ndarray] = []
    deployment_actions: list[np.ndarray] = []
    measured: list[np.ndarray] = [measured_state()]
    velocities: list[np.ndarray] = [measured_velocity()]
    source_frames: list[int] = []
    source_timestamps: list[float] = []
    source_rgb_hashes: list[str] = []
    table_forces: list[float] = []
    inference_records: list[dict[str, Any]] = []
    safety_records: list[dict[str, Any]] = []
    projection_records: list[dict[str, Any]] = []
    first_chunk_audit: dict[str, Any] | None = None
    frame0_reference_audit: dict[str, Any] | None = None
    frame0_eligible: bool | None = None
    safety_abort: dict[str, Any] | None = None
    start_time = time.monotonic()
    try:
        bridge = ACTE1Bridge(checkpoint, model_sha, args.method, output)
        for frame in range(expected_frames):
            source_rgb = source.read()
            current = measured[-1]
            current_velocity = velocities[-1]
            inference = bridge.infer(source_rgb, current)
            raw_action = inference["raw_ensembled_action"]
            raw_chunk = inference["raw_chunk"]
            projection = projector.project(raw_action.astype(np.float32)[None], inference_index=frame)
            hard_action = projection.hard_limit_projected_action[0].astype(np.float64)
            deployment_action = projection.deployment_safe_action[0].astype(np.float64)
            projection_records.extend(projection.records)

            raw_actions.append(raw_action.copy())
            normalized_actions.append(inference["normalized_action"].copy())
            raw_chunks.append(raw_chunk.copy())
            normalized_chunks.append(inference["normalized_chunk"].copy())
            hard_actions.append(hard_action.copy())
            deployment_actions.append(deployment_action.copy())
            source_frames.append(frame)
            source_timestamps.append(frame / CONTROL_FPS)
            source_rgb_hashes.append(sha256_array(source_rgb))
            if frame == 0:
                chunk_projection = projector.project(
                    raw_chunk,
                    inference_index=0,
                    global_row_offset=0,
                )
                projection_records.extend(chunk_projection.records)
                eligibility_reference = (
                    init_q
                    if args.initialization_mode == "method_consistent_v1"
                    else current
                )
                first_chunk_audit = safety.first_raw_chunk(
                    eligibility_reference, chunk_projection.deployment_safe_action
                )
                if args.initialization_mode == "method_consistent_v1":
                    frame0_reference_audit = safety.command(
                        [], init_q, np.zeros(28, dtype=np.float64), deployment_action
                    )

            command_audit = safety.command(
                commands, current, current_velocity, deployment_action
            )
            failed_checks = [
                name for name, passed in command_audit["checks"].items() if not passed
            ]
            if frame == 0 and frame0_reference_audit is not None:
                failed_checks.extend(
                    f"frozen_method_state0_{name}"
                    for name, passed in frame0_reference_audit["checks"].items()
                    if not passed
                )
            if frame == 0 and first_chunk_audit is not None and first_chunk_audit["status"] != "PASS":
                failed_checks.extend(
                    f"first_query_chunk_{name}"
                    for name, passed in first_chunk_audit["checks"].items()
                    if not passed
                )
            safety_row = {
                "frame": frame,
                "source_timestamp_seconds": frame / CONTROL_FPS,
                "source_rgb_sha256": source_rgb_hashes[-1],
                "inference": inference["record"],
                "raw_ensembled_action_sha256": sha256_array(raw_action.astype(np.float32)),
                "deployment_action_sha256": sha256_array(deployment_action.astype(np.float32)),
                "projection_summary": projection.summary,
                "command_audit": command_audit,
                "frozen_method_state0_command_audit": frame0_reference_audit
                if frame == 0
                else None,
                "first_query_chunk_audit": first_chunk_audit if frame == 0 else None,
            }
            if failed_checks:
                if frame == 0:
                    frame0_eligible = False
                safety_abort = {
                    "frame": frame,
                    "source_timestamp_seconds": frame / CONTROL_FPS,
                    "when": "before_command",
                    "failed_checks": failed_checks,
                }
                safety_records.append(safety_row)
                break

            if frame == 0:
                frame0_eligible = True

            target[0, ids] = torch.as_tensor(
                deployment_action, device=robot.device, dtype=torch.float32
            )
            for _ in range(steps_per_control):
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(PHYSICS_DT)
                table_sensor.update(PHYSICS_DT)
            commands.append(deployment_action.copy())
            measured.append(measured_state())
            velocities.append(measured_velocity())
            measured_audit = safety.measured(measured, velocities)
            table_force = table_contact_force_n()
            table_forces.append(table_force)
            table_contact = table_force > 0.05
            safety_row["measured_audit"] = measured_audit
            safety_row["table_contact"] = {
                "maximum_force_n": table_force,
                "threshold_n": 0.05,
                "contact": table_contact,
            }
            safety_records.append(safety_row)
            inference_records.append(
                {
                    "frame": frame,
                    "source_timestamp_seconds": frame / CONTROL_FPS,
                    **inference["record"],
                }
            )
            if cameras:
                recorder.add(capture(), frame)
            if measured_audit["status"] != "PASS" or table_contact:
                safety_abort = {
                    "frame": frame,
                    "source_timestamp_seconds": frame / CONTROL_FPS,
                    "when": "after_command_measured_state",
                    "failed_checks": [
                        *[
                            name
                            for name, passed in measured_audit["checks"].items()
                            if not passed
                        ],
                        *(["table_contact"] if table_contact else []),
                    ],
                }
                break
            if frame % 50 == 0:
                print(
                    f"paper ACT-{args.method.upper()} source ep{source_episode:02d}: "
                    f"frame {frame}/{expected_frames}",
                    flush=True,
                )
    finally:
        if bridge is not None:
            bridge.close()
        source.close()
        recorder.close()

    commands_array = np.asarray(commands, dtype=np.float32).reshape(-1, 28)
    raw_actions_array = np.asarray(raw_actions, dtype=np.float32).reshape(-1, 28)
    normalized_actions_array = np.asarray(normalized_actions, dtype=np.float32).reshape(-1, 28)
    raw_chunks_array = np.asarray(raw_chunks, dtype=np.float32).reshape(-1, 50, 28)
    normalized_chunks_array = np.asarray(normalized_chunks, dtype=np.float32).reshape(-1, 50, 28)
    hard_actions_array = np.asarray(hard_actions, dtype=np.float32).reshape(-1, 28)
    deployment_actions_array = np.asarray(deployment_actions, dtype=np.float32).reshape(-1, 28)
    measured_array = np.asarray(measured, dtype=np.float32).reshape(-1, 28)
    velocity_array = np.asarray(velocities, dtype=np.float32).reshape(-1, 28)
    atomic_npz(
        output / "rollout_arrays.npz",
        joint_names=np.asarray(names),
        source_frame_index=np.asarray(source_frames, dtype=np.int64),
        source_timestamp_seconds=np.asarray(source_timestamps, dtype=np.float64),
        source_rgb_sha256=np.asarray(source_rgb_hashes),
        normalized_raw_query_chunk=normalized_chunks_array,
        raw_act_query_chunk=raw_chunks_array,
        normalized_official_e1_action=normalized_actions_array,
        raw_act_e1_action=raw_actions_array,
        hard_limit_projected_action=hard_actions_array,
        deployment_projected_action=deployment_actions_array,
        commanded_action=commands_array,
        measured_state=measured_array,
        measured_velocity=velocity_array,
        table_contact_force_n=np.asarray(table_forces, dtype=np.float32),
    )
    atomic_json(output / "inference_records.json", inference_records)
    atomic_json(output / "executed_prefix_safety_records.json", safety_records)
    atomic_json(output / "projection_records.json", projection_records)
    if safety_abort is None and len(commands) == expected_frames:
        status = "PASS"
    elif (
        args.initialization_mode == "method_consistent_v1"
        and safety_abort is not None
        and safety_abort["frame"] == 0
        and safety_abort["when"] == "before_command"
    ):
        status = "FRAME0_INELIGIBLE"
    else:
        status = "SAFETY_ABORT"
    report = {
        "schema_version": "paper_core_source_conditioned_isaac_rollout_v2",
        "status": status,
        "experiment_name": experiment_name,
        "method": f"ACT-{args.method.upper()}40",
        "heldout_output_episode": args.heldout_episode,
        "source_final_episode": source_episode,
        "stable_episode_id": entry["stable_episode_id"],
        "requested_frames": expected_frames,
        "executed_frames": len(commands),
        "duration_seconds_at_30_hz": len(commands) / CONTROL_FPS,
        "wall_seconds": time.monotonic() - start_time,
        "source_video": source_provenance,
        "source_video_frames_consumed": len(source_frames),
        "source_clock_advanced_independently_of_policy_progress": True,
        "source_clock_paused_or_jumped": False,
        "current_measured_isaac_state_used": True,
        "g1_onboard_rgb_used": False,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": model_sha,
        "official_act_execution": {
            "selection": "ACT_E1_TEMPORAL_ENSEMBLE",
            "temporal_ensembler": "lerobot.policies.act.modeling_act.ACTTemporalEnsembler",
            "coefficient": 0.01,
            "chunk_size": 50,
            "n_action_steps": 1,
            "control_fps": CONTROL_FPS,
            "worker_ready": bridge.ready if bridge is not None else None,
            "custom_averaging": False,
            "custom_smoothing": False,
        },
        "raw_predictions_preserved": True,
        "initialization": {
            "mode": args.initialization_mode,
            "contract": initial_record["contract"],
            "contract_sha256": initial_record["contract_sha256"],
            "exact_initial_q_float32_sha256": sha256_array(
                np.asarray(init_q, dtype=np.float32)
            ),
            "method_consistent": args.initialization_mode
            == "method_consistent_v1",
            "episode_specific": args.initialization_mode
            == "method_consistent_v1",
            "initial_state_projection_applied": False
            if args.initialization_mode == "method_consistent_v1"
            else bool(
                read_json(initial_record["contract"])["projection"]["summary"][
                    "total_modified_scalar_count"
                ]
            ),
            "policy_specific_tuning": False,
            "frozen_state0_gate_reference": args.initialization_mode
            == "method_consistent_v1",
        },
        "frame0_eligible": frame0_eligible,
        "frame0_frozen_method_state_command_audit": frame0_reference_audit,
        "common_deployment_projection": {
            "sha256": COMMON_PROJECTION_SHA256,
            "policy_independent": True,
            "act_specific_clamp": False,
            "projection_record_count": len(projection_records),
        },
        "object_mode": "KINEMATIC_VISUAL_DOLL_AND_BIN",
        "physical_doll_grasp_scored": False,
        "safety_abort": safety_abort,
        "first_raw_chunk_audit": first_chunk_audit,
        "all_command_hard_limit_checks_passed": all(
            row["command_audit"]["checks"]["command_hard_limits"] for row in safety_records
        ),
        "all_executed_prefix_collision_checks_passed": all(
            row["command_audit"]["checks"]["executed_prefix_self_collision"]
            for row in safety_records
        ),
        "all_branch_checks_passed": all(
            row["command_audit"]["checks"]["branch"] for row in safety_records
        ),
        "all_velocity_checks_passed": all(
            row["command_audit"]["checks"]["velocity"] for row in safety_records
        ),
        "all_acceleration_checks_passed": all(
            row["command_audit"]["checks"]["acceleration"] for row in safety_records
        ),
        "table_contact": {
            "occurrence_count": int(np.count_nonzero(np.asarray(table_forces) > 0.05)),
            "maximum_force_n": float(max(table_forces, default=0.0)),
            "threshold_n": 0.05,
        },
        "command_dynamics": rollout_dynamics(commands_array),
        "measured_dynamics": rollout_dynamics(measured_array),
        "videos": recorder.records(),
        "real_command_publisher_present": False,
        "real_hardware_transport": False,
        "claims": {
            "generated_g1_motion_only": True,
            "source_video_conditioned": True,
            "physical_manipulation_success": False,
            "autonomous_onboard_g1_visual_deployment": False,
            "sim_to_real": False,
        },
    }
    atomic_json(output / "rollout_report.json", report)
    print(read_json(output / "rollout_report.json"))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
