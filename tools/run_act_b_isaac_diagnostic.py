#!/usr/bin/env python3
"""Bounded Isaac-only ACT-B diagnostic using official ACT execution semantics.

The runner has no SmolVLA, RTC, crossfade, Ruckig, low-pass, custom averaging,
DDS, or real-hardware command path.  Every 30-Hz command is the direct output
of official ACTPolicy.select_action followed only by the already-frozen common
Dex3 deployment projection.  Raw and projected arrays remain separate, and all
commands are fail-closed by hard-limit, velocity, acceleration, branch, contact,
and self-collision checks before being sent to Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np

from isaaclab.app import AppLauncher

from common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
from deployment_camera_config import apply_configured_distortion, load_camera_config
from policy_b_isaac_control_contract import (
    CONTROL_FPS,
    PHYSICS_DT,
    build_implicit_actuators,
)


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SCENE_STAGE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
ACTION_FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
PRIOR_INITIAL_CONDITION = (
    ROOT / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout/initial_condition.json"
)
COMMON_PROJECTION_FREEZE = (
    ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
)
INITIAL_CONTRACT = (
    ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout/COMMON_G1_POLICY_INITIAL_STATE_V1.json"
)
INITIAL_CONTRACT_SHA256 = "e1b64c4b009d8b10c47995a5866ad9f3f0d427c967975fb45b6c702ea443e0b1"
COMMON_CONFIG = ROOT / "configs/doll_handoff_retargeting/common_config.template.json"
FEASIBILITY_CONFIG = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
POLICY_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
POLICY_WORKER = ROOT / "tools/act_b_inference_worker.py"
DEFAULT_CAMERA = ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"
AUTHKEY = b"act-b-isaac-local-v1"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--checkpoint-sha256", required=True)
parser.add_argument("--execution", choices=("e0", "e1"), required=True)
parser.add_argument("--stage", choices=("object-free", "kinematic-doll"), required=True)
parser.add_argument("--frames", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--no-video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.frames <= 688:
    parser.error("--frames must be in 1..688")
DEPLOYMENT_CAMERA = load_camera_config(
    args.camera_config, purpose="ACT-B bounded Isaac diagnostic"
)
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def look_at_ros_camera_quaternion_xyzw(
    eye_world_xyz: np.ndarray, target_world_xyz: np.ndarray
) -> np.ndarray:
    eye = np.asarray(eye_world_xyz, dtype=np.float64)
    target = np.asarray(target_world_xyz, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.column_stack((right, down, forward))
    # Stable matrix -> xyzw quaternion conversion.
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale]
            )
    return quaternion / np.linalg.norm(quaternion)


def frozen_interfaces() -> tuple[list[str], np.ndarray, np.ndarray, NamedJointDeploymentSafetyProjector]:
    freeze = read_json(ACTION_FREEZE)
    names = list(freeze["joint_names"])
    specs = freeze["joint_specs"]
    if len(names) != 28 or [row["joint_name"] for row in specs] != names:
        raise RuntimeError("frozen Dataset-B interface changed")
    projector = NamedJointDeploymentSafetyProjector.from_path(COMMON_PROJECTION_FREEZE)
    if projector.names != names:
        raise RuntimeError("frozen common projection named order differs from Dataset B")
    return names, projector.hard_lower, projector.hard_upper, projector


def initial_condition(names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    actual_sha = sha256_file(INITIAL_CONTRACT)
    if actual_sha != INITIAL_CONTRACT_SHA256:
        raise RuntimeError(f"common initial-state contract changed: {actual_sha}")
    contract = read_json(INITIAL_CONTRACT)
    if contract.get("status") != "COMMON_G1_POLICY_INITIAL_STATE_V1" or contract["joint_order"] != names:
        raise RuntimeError("invalid COMMON_G1_POLICY_INITIAL_STATE_V1 named interface")
    if float(contract["settle_duration_seconds"]) != float(args.settle_seconds):
        raise RuntimeError("rollout settle duration differs from frozen common A/B contract")
    q = np.asarray(contract["full_28d_initial_q_rad"], dtype=np.float64)
    return q, {
        "status": contract["status"],
        "contract": str(INITIAL_CONTRACT),
        "contract_sha256": actual_sha,
        "joint_names": names,
        "q_rad": q,
        "settle_duration_seconds": contract["settle_duration_seconds"],
        "reset_procedure": contract["reset_procedure"],
        "tolerances": contract["tolerances"],
        "common_for_act_a_and_act_b": True,
        "policy_specific_start_pose": False,
    }


class ACTBridge:
    def __init__(self, checkpoint: Path, model_sha: str, execution: str, output: Path):
        bridge = output / "inference_bridge"
        bridge.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:10]
        self.socket = Path(f"/tmp/actb_{os.getpid()}_{token}.sock")
        self.ready_path = bridge / "ready.json"
        self.log_path = bridge / "worker.log"
        for path in (self.socket, self.ready_path):
            if path.exists() or path.is_socket():
                path.unlink()
        self.log_stream = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(POLICY_PYTHON),
                str(POLICY_WORKER),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                model_sha,
                "--socket",
                str(self.socket),
                "--ready",
                str(self.ready_path),
                "--execution",
                execution,
            ],
            cwd=ROOT,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 90.0
        while not self.ready_path.is_file():
            if self.process.poll() is not None:
                self.log_stream.flush()
                raise RuntimeError(f"ACT worker exited; inspect {self.log_path}")
            if time.monotonic() >= deadline:
                self.process.terminate()
                raise TimeoutError("ACT worker was not ready in 90 seconds")
            time.sleep(0.1)
        self.ready = read_json(self.ready_path)
        self.connection = Client(str(self.socket), family="AF_UNIX", authkey=AUTHKEY)

    def infer(
        self, rgb: np.ndarray, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        self.connection.send(
            {
                "command": "infer",
                "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
                "state": np.ascontiguousarray(state, dtype=np.float32),
            }
        )
        response = self.connection.recv()
        if response.get("status") != "PASS":
            raise RuntimeError(f"ACT worker inference failed: {response}")
        action = np.asarray(response.pop("action"), dtype=np.float64)
        normalized_action = np.asarray(response.pop("normalized_action"), dtype=np.float32)
        normalized_chunk_value = response.pop("normalized_chunk", None)
        physical_chunk_value = response.pop("physical_chunk", None)
        normalized_chunk = (
            np.asarray(normalized_chunk_value, dtype=np.float32)
            if normalized_chunk_value is not None
            else None
        )
        physical_chunk = (
            np.asarray(physical_chunk_value, dtype=np.float32)
            if physical_chunk_value is not None
            else None
        )
        if action.shape != (28,) or not np.isfinite(action).all():
            raise RuntimeError(f"malformed ACT worker action {action.shape}")
        if normalized_action.shape != (28,) or not np.isfinite(normalized_action).all():
            raise RuntimeError("malformed normalized ACT action")
        if (normalized_chunk is None) != (physical_chunk is None):
            raise RuntimeError("ACT worker returned only one raw chunk representation")
        if physical_chunk is not None and (
            physical_chunk.shape != (50, 28)
            or normalized_chunk.shape != (50, 28)
            or not np.isfinite(physical_chunk).all()
            or not np.isfinite(normalized_chunk).all()
        ):
            raise RuntimeError("malformed ACT raw policy query chunk")
        return action, normalized_action, normalized_chunk, physical_chunk, response

    def close(self) -> None:
        try:
            if hasattr(self, "connection"):
                self.connection.send({"command": "shutdown"})
                self.connection.recv()
                self.connection.close()
        finally:
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=10.0)
            self.log_stream.close()


class SafetyAudit:
    def __init__(self, names: list[str], lower: np.ndarray, upper: np.ndarray):
        sys.path.insert(0, str(ROOT / "tools"))
        from doll_handoff_retargeting.models import G1Kinematics

        self.names = names
        self.lower = lower
        self.upper = upper
        self.thresholds = read_json(FEASIBILITY_CONFIG)["unchanged_acceptance"]
        self.g1 = G1Kinematics(read_json(COMMON_CONFIG), read_json(SCENE_LAYOUT))
        self.left = [names.index(name) for name in self.g1.hand_joint_names["left"]]
        self.right = [names.index(name) for name in self.g1.hand_joint_names["right"]]

    def collision(self, q: np.ndarray) -> dict[str, Any]:
        values = np.asarray(q, dtype=np.float64).reshape(-1, 28)
        geometry = self.g1.trajectory_geometry(
            values[:, :14],
            values[:, self.left],
            values[:, self.right],
            float(self.thresholds["collision_penetration_tolerance_m"]),
        )
        counts = {
            key: int(np.count_nonzero(value))
            for key, value in geometry["collision_flags"].items()
        }
        invalid = sum(
            counts.get(key, 0)
            for key in ("ARM_TORSO", "CROSS_ARM", "WRIST_OR_PALM_TORSO", "OTHER")
        )
        return {
            "frame_counts": counts,
            "invalid_hard_self_collision_incidence": invalid,
            "distal_hand_hand_reported_separately": counts.get("DISTAL_HAND_HAND", 0),
            "records": geometry["collision_records"],
        }

    def command(
        self,
        command_history: list[np.ndarray],
        measured: np.ndarray,
        measured_velocity: np.ndarray,
        proposed: np.ndarray,
    ) -> dict[str, Any]:
        previous = command_history[-1] if command_history else measured
        delta = proposed - previous
        velocity = delta * CONTROL_FPS
        if len(command_history) >= 2:
            previous_velocity = (command_history[-1] - command_history[-2]) * CONTROL_FPS
        else:
            previous_velocity = measured_velocity
        acceleration = (velocity - previous_velocity) * CONTROL_FPS
        # The frozen branch detector is a trajectory-adjacency test.  At frame 0
        # the current->first-command transition is governed by the independent
        # max-step/velocity/acceleration gates.  Applying the 14D branch norm to
        # that transition was the audited implementation error.
        history = np.asarray([*command_history, proposed], dtype=np.float64)
        arm_norms = np.linalg.norm(np.diff(history[:, :14], axis=0), axis=1)
        latest_arm_norm = float(arm_norms[-1]) if len(arm_norms) else None
        prior_norms = arm_norms[:-1] if len(arm_norms) > 1 else np.empty(0)
        local = float(np.median(prior_norms[-10:])) if len(prior_norms) else 0.0
        branch_threshold = max(
            float(self.thresholds["branch_absolute_step_norm_rad"]),
            float(self.thresholds["branch_local_multiplier"]) * max(local, 1e-6),
        )
        branch_applicable = len(command_history) > 0
        branch_pass = (not branch_applicable) or bool(latest_arm_norm <= branch_threshold)
        violation = (proposed < self.lower - 1e-9) | (proposed > self.upper + 1e-9)
        collision = self.collision(np.stack((previous, proposed)))
        checks = {
            "finite": bool(np.isfinite(proposed).all()),
            "command_hard_limits": int(np.count_nonzero(violation)) == 0,
            "adjacent_step": float(np.max(np.abs(delta))) <= float(self.thresholds["maximum_joint_step_rad"]),
            "velocity": float(np.max(np.abs(velocity))) <= float(self.thresholds["maximum_velocity_rad_s"]),
            "acceleration": float(np.max(np.abs(acceleration))) <= float(self.thresholds["maximum_acceleration_rad_s2"]),
            "branch": branch_pass,
            "executed_prefix_self_collision": collision["invalid_hard_self_collision_incidence"] == 0,
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "command_hard_limit_violation_count": int(np.count_nonzero(violation)),
            "command_hard_limit_violation_joints": [self.names[index] for index in np.flatnonzero(violation)],
            "maximum_joint_step_rad": float(np.max(np.abs(delta))),
            "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
            "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
            "arm_step_l2_rad": latest_arm_norm,
            "branch_threshold_rad": branch_threshold,
            "branch_gate_applicable": branch_applicable,
            "frame0_current_to_first_command_uses_step_velocity_acceleration_not_branch": not branch_applicable,
            "collision": collision,
            "thresholds": self.thresholds,
        }

    def chunk(self, current: np.ndarray, chunk: np.ndarray) -> dict[str, Any]:
        values = np.asarray(chunk, dtype=np.float64)
        if values.shape != (50, 28):
            raise ValueError(f"ACT query chunk must be [50,28], received {values.shape}")
        steps = np.diff(values, axis=0)
        velocity = steps * CONTROL_FPS
        acceleration = np.diff(values, n=2, axis=0) * CONTROL_FPS**2
        arm_norms = np.linalg.norm(np.diff(values[:, :14], axis=0), axis=1)
        branch_flags = np.zeros(len(values), dtype=bool)
        absolute = float(self.thresholds["branch_absolute_step_norm_rad"])
        multiplier = float(self.thresholds["branch_local_multiplier"])
        for index in range(1, len(values)):
            local = float(
                np.median(arm_norms[max(0, index - 10) : min(len(arm_norms), index + 9)])
            )
            branch_flags[index] = arm_norms[index - 1] > max(
                absolute, multiplier * max(local, 1e-6)
            )
        violation = (values < self.lower[None] - 1e-9) | (values > self.upper[None] + 1e-9)
        collision = self.collision(values)
        first_delta = values[0] - np.asarray(current, dtype=np.float64)
        checks = {
            "finite": bool(np.isfinite(values).all()),
            "command_hard_limits": int(np.count_nonzero(violation)) == 0,
            "first_command_delta": float(np.max(np.abs(first_delta)))
            <= float(self.thresholds["maximum_joint_step_rad"]),
            "adjacent_step": float(np.max(np.abs(steps)))
            <= float(self.thresholds["maximum_joint_step_rad"]),
            "velocity": float(np.max(np.abs(velocity)))
            <= float(self.thresholds["maximum_velocity_rad_s"]),
            "acceleration": float(np.max(np.abs(acceleration)))
            <= float(self.thresholds["maximum_acceleration_rad_s2"]),
            "branch": int(np.count_nonzero(branch_flags)) == 0,
            "hard_self_collision": collision["invalid_hard_self_collision_incidence"] == 0,
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "hard_limit_violation_count": int(np.count_nonzero(violation)),
            "first_command_delta_max_abs_rad": float(np.max(np.abs(first_delta))),
            "first_command_delta_arm_l2_rad_diagnostic_only": float(np.linalg.norm(first_delta[:14])),
            "maximum_adjacent_step_rad": float(np.max(np.abs(steps))),
            "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
            "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
            "maximum_internal_arm_l2_step_rad": float(np.max(arm_norms)),
            "branch_discontinuity_count": int(np.count_nonzero(branch_flags)),
            "collision": collision,
            "thresholds": self.thresholds,
        }

    def measured(
        self,
        measured_history: list[np.ndarray],
        velocity_history: list[np.ndarray],
    ) -> dict[str, Any]:
        value = measured_history[-1]
        velocity = velocity_history[-1]
        acceleration = (
            (velocity_history[-1] - velocity_history[-2]) * CONTROL_FPS
            if len(velocity_history) >= 2
            else np.zeros(28)
        )
        lower_excess = np.maximum(self.lower - value, 0.0)
        upper_excess = np.maximum(value - self.upper, 0.0)
        excess = np.maximum(lower_excess, upper_excess)
        collision = self.collision(value[None])
        # The prior Isaac audit already isolated a microscopic Dex3 PhysX
        # boundary excursion.  Preserve and log it without reopening margin research.
        arm_excess = float(np.max(excess[:14]))
        dex3_excess = float(np.max(excess[14:]))
        microscopic_dex3_only = arm_excess <= 1e-9 and 0.0 < dex3_excess <= 1e-4
        checks = {
            "finite": bool(np.isfinite(value).all() and np.isfinite(velocity).all()),
            "measured_arm_hard_limits": arm_excess <= 1e-9,
            "measured_dex3_hard_limits_or_precharacterized_microscopic_excursion": dex3_excess <= 1e-4,
            "measured_velocity": float(np.max(np.abs(velocity))) <= float(self.thresholds["maximum_velocity_rad_s"]),
            "measured_acceleration": float(np.max(np.abs(acceleration))) <= float(self.thresholds["maximum_acceleration_rad_s2"]),
            "measured_self_collision": collision["invalid_hard_self_collision_incidence"] == 0,
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "maximum_arm_hard_limit_excess_rad": arm_excess,
            "maximum_dex3_hard_limit_excess_rad": dex3_excess,
            "precharacterized_microscopic_dex3_excursion_logged_separately": microscopic_dex3_only,
            "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
            "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
            "collision": collision,
        }


class VideoRecorder:
    def __init__(self, output: Path, enabled: bool):
        self.enabled = enabled
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.paths: dict[str, Path] = {}
        self.samples: dict[str, list[np.ndarray]] = {name: [] for name in ("source_like", "overview", "top", "side")}
        if not enabled:
            return
        for name in self.samples:
            path = output / f"{name}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_FPS, (640, 480))
            if not writer.isOpened():
                raise RuntimeError(f"could not open video writer {path}")
            self.paths[name] = path
            self.writers[name] = writer

    def add(self, images: dict[str, np.ndarray], frame: int, execution: str, stage: str) -> None:
        if not self.enabled:
            return
        for name, writer in self.writers.items():
            bgr = cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (0, 0), (640, 34), (0, 0, 0), -1)
            cv2.putText(
                bgr,
                f"ACT-{execution.upper()} | {stage} | frame {frame:04d} | official ACT + frozen common Dex3 adapter",
                (7, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (80, 235, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(bgr)
            if frame in {0, 30, 90, 150, 240, 360, 500, 650}:
                self.samples[name].append(bgr.copy())

    def close(self, output: Path) -> None:
        for writer in self.writers.values():
            writer.release()
        if not self.enabled or not self.samples["overview"]:
            return
        panels = self.samples["overview"][:6]
        while len(panels) < 6:
            panels.append(np.zeros_like(panels[0]))
        sheet = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:6])))
        cv2.imwrite(str(output / "contact_sheet.png"), sheet)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values), dtype=np.float64))))


def rollout_dynamics(q: np.ndarray) -> dict[str, float]:
    values = np.asarray(q, dtype=np.float64)
    if len(values) < 4:
        return {}
    qdot = np.diff(values, axis=0) * CONTROL_FPS
    qddot = np.diff(qdot, axis=0) * CONTROL_FPS
    jerk = np.diff(qddot, axis=0) * CONTROL_FPS
    return {
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(values, axis=0)))),
        "qdot_rms_rad_s": rms(qdot),
        "qdot_max_abs_rad_s": float(np.max(np.abs(qdot))),
        "qddot_rms_rad_s2": rms(qddot),
        "qddot_max_abs_rad_s2": float(np.max(np.abs(qddot))),
        "jerk_rms_rad_s3": rms(jerk),
        "jerk_max_abs_rad_s3": float(np.max(np.abs(jerk))),
    }


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
        raise FileExistsError(f"refusing to overwrite Isaac diagnostic output: {output}")
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    actual_model_sha = sha256_file(checkpoint / "model.safetensors")
    if actual_model_sha != args.checkpoint_sha256:
        raise RuntimeError("selected ACT checkpoint hash changed")
    names, lower, upper, projector = frozen_interfaces()
    init_q, init_record = initial_condition(names)
    if np.any(init_q < lower) or np.any(init_q > upper):
        raise RuntimeError("frozen common initial pose violates hard limits")
    atomic_json(output / "initial_condition.json", init_record)
    atomic_json(
        output / "common_deployment_projection_provenance.json",
        {
            "freeze_manifest": str(COMMON_PROJECTION_FREEZE),
            "freeze_manifest_sha256": sha256_file(COMMON_PROJECTION_FREEZE),
            "implementation": projector.config["implementation"],
            "applicable_policies": projector.config["applicable_policies"],
            "act_specific_clamp": False,
            "raw_hard_projected_and_deployment_safe_arrays_separate": True,
        },
    )
    safety = SafetyAudit(names, lower, upper)
    initial_collision = safety.collision(init_q[None])
    if initial_collision["invalid_hard_self_collision_incidence"]:
        raise RuntimeError("common initial pose has a hard self collision")
    if not SCENE_STAGE.is_file():
        raise FileNotFoundError(SCENE_STAGE)

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"failed to open {SCENE_STAGE}")
    stage = omni.usd.get_context().get_stage()
    # All diagnostic scene edits live only in the USD session layer.
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
    collision_disabled_roots = ["/World/DollHandoffEnvironment/Doll"]
    if args.stage == "object-free":
        # Established object-free semantics retain the expected visual context
        # while disabling Doll/bin collision response.
        collision_disabled_roots.append("/World/DollHandoffEnvironment/TrashBin")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(path.startswith(root) for root in collision_disabled_roots):
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

    def camera_spawn() -> Any:
        return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            DEPLOYMENT_CAMERA.intrinsic_matrix.reshape(-1).tolist(),
            width=DEPLOYMENT_CAMERA.width,
            height=DEPLOYMENT_CAMERA.height,
            clipping_range=DEPLOYMENT_CAMERA.clipping_range_m,
            lock_camera=True,
        )

    cameras = {
        key: Camera(
            CameraCfg(
                prim_path=f"/World/{prim}",
                update_period=0.0,
                width=DEPLOYMENT_CAMERA.width,
                height=DEPLOYMENT_CAMERA.height,
                data_types=["rgb"],
                spawn=camera_spawn(),
            )
        )
        for key, prim in {
            "source_like": "ACTFrame0PolicyCamera",
            "overview": "ACTExecutionOverviewCamera",
            "top": "ACTExecutionTopCamera",
            "side": "ACTExecutionSideCamera",
        }.items()
    }
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError(f"Isaac named joint mapping failed: {missing}")
    cameras["source_like"].set_world_poses(
        DEPLOYMENT_CAMERA.position_world_xyz_m.astype(np.float32)[None],
        DEPLOYMENT_CAMERA.orientation_world_xyzw_ros_camera.astype(np.float32)[None],
        convention="ros",
    )
    layout = read_json(SCENE_LAYOUT)
    for name in ("overview", "top", "side"):
        preset = layout["camera"]["presets"][name]
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
    immediate_reset_state = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
    reset_tolerance = float(init_record["tolerances"]["immediate_reset_max_abs_from_nominal_rad"])
    if float(np.max(np.abs(immediate_reset_state - init_q))) > reset_tolerance:
        raise RuntimeError("COMMON_G1_POLICY_INITIAL_STATE_V1 immediate-reset tolerance failed")
    steps_per_control = int(round((1.0 / CONTROL_FPS) / PHYSICS_DT))

    def measured_state() -> np.ndarray:
        value = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("Isaac measured state malformed")
        return value

    def measured_velocity() -> np.ndarray:
        value = robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("Isaac measured velocity malformed")
        return value

    def capture() -> dict[str, np.ndarray]:
        sim.forward()
        sim.render()
        sim.render_context.reset_transform_cadence()
        result = {}
        for name, camera in cameras.items():
            camera.update(PHYSICS_DT, force_recompute=True)
            image = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            result[name] = (
                apply_configured_distortion(image.copy(), DEPLOYMENT_CAMERA)
                if name == "source_like"
                else image.copy()
            )
        return result

    for _ in range(max(1, int(round(args.settle_seconds / PHYSICS_DT)))):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT)
        table_sensor.update(PHYSICS_DT)

    settled_contract_state = measured_state()
    settle_tolerance = float(init_record["tolerances"]["post_settle_max_abs_from_nominal_rad"])
    if float(np.max(np.abs(settled_contract_state - init_q))) > settle_tolerance:
        raise RuntimeError("COMMON_G1_POLICY_INITIAL_STATE_V1 post-settle tolerance failed")
    atomic_json(
        output / "initial_state_contract_verification.json",
        {
            "status": "PASS",
            "contract": str(INITIAL_CONTRACT),
            "contract_sha256": INITIAL_CONTRACT_SHA256,
            "joint_names": names,
            "immediate_reset_max_abs_from_nominal_rad": float(np.max(np.abs(immediate_reset_state - init_q))),
            "immediate_reset_tolerance_rad": reset_tolerance,
            "settled_max_abs_from_nominal_rad": float(np.max(np.abs(settled_contract_state - init_q))),
            "settled_tolerance_rad": settle_tolerance,
            "post_settle_resnap": False,
            "policy_specific_pose": False,
        },
    )

    def table_contact_force_n() -> float:
        matrix = table_sensor.data.force_matrix_w
        if matrix is None:
            return 0.0
        value = matrix.torch.detach().cpu().numpy()
        return float(np.max(np.linalg.norm(value.reshape(-1, 3), axis=1))) if value.size else 0.0

    recorder = VideoRecorder(output, not args.no_video)
    bridge: ACTBridge | None = None
    commands: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    normalized_actions: list[np.ndarray] = []
    hard_projected_actions: list[np.ndarray] = []
    deployment_projected_actions: list[np.ndarray] = []
    query_frames: list[int] = []
    normalized_query_chunks: list[np.ndarray] = []
    raw_query_chunks: list[np.ndarray] = []
    hard_projected_query_chunks: list[np.ndarray] = []
    deployment_projected_query_chunks: list[np.ndarray] = []
    projection_records: list[dict[str, Any]] = []
    table_forces: list[float] = []
    measured: list[np.ndarray] = [measured_state()]
    velocities: list[np.ndarray] = [measured_velocity()]
    inference_records = []
    safety_records = []
    safety_abort: dict[str, Any] | None = None
    start_time = time.monotonic()
    try:
        bridge = ACTBridge(checkpoint, actual_model_sha, args.execution, output)
        images = capture()
        for frame in range(args.frames):
            current = measured[-1]
            current_velocity = velocities[-1]
            raw_action, normalized_action, normalized_chunk, raw_chunk, inference = bridge.infer(
                images["source_like"], current
            )
            action_projection = projector.project(
                raw_action.astype(np.float32)[None], inference_index=int(inference["call_index"])
            )
            hard_action = action_projection.hard_limit_projected_action[0].astype(np.float64)
            deployment_action = action_projection.deployment_safe_action[0].astype(np.float64)
            raw_actions.append(raw_action.copy())
            normalized_actions.append(normalized_action.copy())
            hard_projected_actions.append(hard_action.copy())
            deployment_projected_actions.append(deployment_action.copy())
            projection_records.extend(action_projection.records)

            chunk_audit = None
            chunk_gate_applicable = False
            if raw_chunk is not None:
                assert normalized_chunk is not None
                chunk_projection = projector.project(
                    raw_chunk,
                    inference_index=int(inference["policy_query_count"]) - 1,
                    global_row_offset=(int(inference["policy_query_count"]) - 1) * 50,
                )
                query_frames.append(frame)
                normalized_query_chunks.append(normalized_chunk.copy())
                raw_query_chunks.append(raw_chunk.copy())
                hard_projected_query_chunks.append(
                    chunk_projection.hard_limit_projected_action.astype(np.float32)
                )
                deployment_projected_query_chunks.append(
                    chunk_projection.deployment_safe_action.astype(np.float32)
                )
                projection_records.extend(chunk_projection.records)
                chunk_audit = safety.chunk(
                    current, chunk_projection.deployment_safe_action
                )
                # E0 will execute its query chunk as the official 50-action queue.
                # At E1 frame 0, the official ensemble output equals raw row 0;
                # later raw chunks are preserved diagnostics, not direct commands.
                chunk_gate_applicable = args.execution == "e0" or frame == 0

            command_audit = safety.command(
                commands, current, current_velocity, deployment_action
            )
            record = {
                "frame": frame,
                "inference": inference,
                "raw_action_sha256": hashlib.sha256(np.ascontiguousarray(raw_action.astype(np.float32)).tobytes()).hexdigest(),
                "hard_projected_action_sha256": hashlib.sha256(np.ascontiguousarray(hard_action.astype(np.float32)).tobytes()).hexdigest(),
                "deployment_projected_action_sha256": hashlib.sha256(np.ascontiguousarray(deployment_action.astype(np.float32)).tobytes()).hexdigest(),
                "action_projection_summary": action_projection.summary,
                "query_chunk_audit": chunk_audit,
                "query_chunk_gate_applicable": chunk_gate_applicable,
                "command_audit": command_audit,
            }
            failed_checks = [key for key, value in command_audit["checks"].items() if not value]
            if chunk_gate_applicable and chunk_audit is not None and chunk_audit["status"] != "PASS":
                failed_checks.extend(
                    f"query_chunk_{key}"
                    for key, value in chunk_audit["checks"].items()
                    if not value
                )
            if failed_checks:
                safety_abort = {
                    "frame": frame,
                    "when": "before_command",
                    "failed_checks": failed_checks,
                }
                safety_records.append(record)
                break
            target[0, ids] = torch.as_tensor(deployment_action, device=robot.device, dtype=torch.float32)
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
            record["measured_audit"] = measured_audit
            record["table_contact"] = {
                "maximum_force_n": table_force,
                "threshold_n": 0.05,
                "contact": table_contact,
                "threshold_provenance": "existing Policy-B Isaac TaskSemantics.FORCE_THRESHOLD_N",
            }
            safety_records.append(record)
            inference_records.append({"frame": frame, **inference})
            images = capture()
            recorder.add(images, frame, args.execution, args.stage)
            if measured_audit["status"] != "PASS" or table_contact:
                safety_abort = {
                    "frame": frame,
                    "when": "after_command_measured_state",
                    "failed_checks": [
                        *[key for key, value in measured_audit["checks"].items() if not value],
                        *(["table_contact"] if table_contact else []),
                    ],
                }
                break
            if frame % 50 == 0:
                print(f"ACT Isaac {args.execution} {args.stage}: frame {frame}/{args.frames}", flush=True)
    finally:
        if bridge is not None:
            bridge.close()
        recorder.close(output)

    commands_array = np.asarray(commands, dtype=np.float32).reshape(-1, 28)
    raw_actions_array = np.asarray(raw_actions, dtype=np.float32).reshape(-1, 28)
    normalized_actions_array = np.asarray(normalized_actions, dtype=np.float32).reshape(-1, 28)
    hard_actions_array = np.asarray(hard_projected_actions, dtype=np.float32).reshape(-1, 28)
    deployment_actions_array = np.asarray(deployment_projected_actions, dtype=np.float32).reshape(-1, 28)
    measured_array = np.asarray(measured, dtype=np.float32).reshape(-1, 28)
    velocity_array = np.asarray(velocities, dtype=np.float32).reshape(-1, 28)
    normalized_chunks_array = np.asarray(normalized_query_chunks, dtype=np.float32).reshape(-1, 50, 28)
    raw_chunks_array = np.asarray(raw_query_chunks, dtype=np.float32).reshape(-1, 50, 28)
    hard_chunks_array = np.asarray(hard_projected_query_chunks, dtype=np.float32).reshape(-1, 50, 28)
    deployment_chunks_array = np.asarray(deployment_projected_query_chunks, dtype=np.float32).reshape(-1, 50, 28)
    atomic_npz(
        output / "rollout_arrays.npz",
        joint_names=np.asarray(names),
        inference_attempt_frames=np.arange(len(raw_actions_array), dtype=np.int64),
        normalized_official_action=normalized_actions_array,
        raw_act_action=raw_actions_array,
        hard_limit_projected_action=hard_actions_array,
        deployment_projected_action=deployment_actions_array,
        commanded_action=commands_array,
        measured_state=measured_array,
        measured_velocity=velocity_array,
        table_contact_force_n=np.asarray(table_forces, dtype=np.float32),
        policy_query_frames=np.asarray(query_frames, dtype=np.int64),
        normalized_raw_query_chunk=normalized_chunks_array,
        raw_act_query_chunk=raw_chunks_array,
        hard_limit_projected_query_chunk=hard_chunks_array,
        deployment_projected_query_chunk=deployment_chunks_array,
    )
    atomic_json(output / "inference_records.json", inference_records)
    atomic_json(output / "executed_prefix_safety_records.json", safety_records)
    atomic_json(output / "projection_records.json", projection_records)
    video_records = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in recorder.paths.items()
    }
    report = {
        "schema_version": "act_b_isaac_diagnostic_v2",
        "status": "PASS" if safety_abort is None and len(commands) == args.frames else "SAFETY_ABORT",
        "stage": args.stage,
        "execution": args.execution,
        "requested_frames": args.frames,
        "executed_frames": len(commands),
        "duration_seconds_at_30_hz": len(commands) / CONTROL_FPS,
        "wall_seconds": time.monotonic() - start_time,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": actual_model_sha,
        "worker_ready": bridge.ready if bridge is not None else None,
        "controlled_input": "current diagnostic RGB plus current measured Isaac 28D state",
        "command_semantics": "official ACTPolicy.select_action at 30 Hz followed only by frozen policy-independent Dex3 hard/simulation-safe projection",
        "raw_act_predictions_preserved_separately": True,
        "raw_query_chunk_count": len(raw_query_chunks),
        "raw_query_chunk_shape": [50, 28],
        "common_deployment_projection": {
            "freeze_manifest": str(COMMON_PROJECTION_FREEZE),
            "freeze_manifest_sha256": sha256_file(COMMON_PROJECTION_FREEZE),
            "policy_independent": True,
            "act_specific_clamp": False,
            "arms_bitwise_preserved": True,
            "projection_record_count": len(projection_records),
        },
        "initial_state_contract": {
            "path": str(INITIAL_CONTRACT),
            "sha256": INITIAL_CONTRACT_SHA256,
            "status": "PASS",
            "post_settle_resnap": False,
        },
        "same_initial_g1_pose": True,
        "object_free": args.stage == "object-free",
        "kinematic_doll": args.stage == "kinematic-doll",
        "physical_doll_grasp_scored": False,
        "smolvla_rtc_imported": False,
        "execution_adapter": "FROZEN_COMMON_POLICY_INDEPENDENT_DEX3_DEPLOYMENT_PROJECTION_ONLY",
        "ruckig": False,
        "crossfade": False,
        "low_pass": False,
        "custom_temporal_averaging": False,
        "real_command_publisher_present": False,
        "real_hardware_transport": False,
        "safety_abort": safety_abort,
        "all_command_hard_limit_checks_passed": all(row["command_audit"]["checks"]["command_hard_limits"] for row in safety_records),
        "all_executed_prefix_collision_checks_passed": all(row["command_audit"]["checks"]["executed_prefix_self_collision"] for row in safety_records),
        "all_branch_checks_passed": all(row["command_audit"]["checks"]["branch"] for row in safety_records),
        "all_applicable_raw_query_chunk_checks_passed": all(
            (not row.get("query_chunk_gate_applicable"))
            or row.get("query_chunk_audit", {}).get("status") == "PASS"
            for row in safety_records
        ),
        "all_velocity_checks_passed": all(row["command_audit"]["checks"]["velocity"] for row in safety_records),
        "all_acceleration_checks_passed": all(row["command_audit"]["checks"]["acceleration"] for row in safety_records),
        "precharacterized_microscopic_dex3_excursion_records": int(
            sum(
                bool(row.get("measured_audit", {}).get("precharacterized_microscopic_dex3_excursion_logged_separately"))
                for row in safety_records
            )
        ),
        "table_contact": {
            "occurrence_count": int(np.count_nonzero(np.asarray(table_forces) > 0.05)),
            "maximum_force_n": float(max(table_forces, default=0.0)),
            "threshold_n": 0.05,
            "all_checks_passed": not any(value > 0.05 for value in table_forces),
        },
        "command_dynamics": rollout_dynamics(commands_array),
        "measured_dynamics": rollout_dynamics(measured_array),
        "videos": video_records,
        "behavior_like_classification": {
            "LEFT_APPROACH_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
            "LEFT_GRASP_MOTION_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
            "LEFT_TRANSPORT_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
            "RIGHT_HANDOFF_APPROACH_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
            "BIMANUAL_HANDOFF_POSTURE_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
            "LEFT_RELEASE_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
            "RIGHT_TRANSPORT_TO_BIN_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
            "RIGHT_RELEASE_LIKE": "DEFERRED_TO_VIDEO_AND_TRAJECTORY_REVIEW",
        },
    }
    atomic_json(output / "isaac_diagnostic_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
