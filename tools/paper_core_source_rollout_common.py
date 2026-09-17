"""Shared contracts for paper-core source-conditioned Isaac rollouts.

The module is simulation-only.  It centralizes named interfaces, the frozen
common A/B initial state, official ACT-E1 inference IPC, and the unchanged
command/measured-state safety gates used by the prior ACT-B Isaac audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np

try:
    from tools.common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
    from tools.policy_b_isaac_control_contract import CONTROL_FPS
except ModuleNotFoundError:  # Direct ``python tools/<runner>.py`` execution.
    from common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
    from policy_b_isaac_control_contract import CONTROL_FPS


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ACTION_FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
COMMON_PROJECTION_FREEZE = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
COMMON_PROJECTION_SHA256 = "05078d0038ab6defaa8a0f56b1f38b752cc92892856996fecc05b75fcce27cf2"
INITIAL_CONTRACT = ROOT / "outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json"
INITIAL_CONTRACT_SHA256 = "a96e915d9eaa098d42913f8f69f80d659c580018d46e0b468b20e7e760e72f9e"
METHOD_CONSISTENT_ROOT = ROOT / "outputs/paper_core_ab/method_consistent_source_video_rollout"
METHOD_CONSISTENT_CONTRACT = METHOD_CONSISTENT_ROOT / "METHOD_CONSISTENT_INITIALIZATION_CONTRACT.json"
METHOD_CONSISTENT_FREEZE = METHOD_CONSISTENT_ROOT / "METHOD_CONSISTENT_INITIALIZATION_CONTRACT.sha256.json"
EXECUTION_CONFIG = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout/SELECTED_ACT_EXECUTION_CONFIG.json"
EXECUTION_CONFIG_SHA256 = "656f6f474f6981c5b0c0417895b91ad924d171031bb66b26e4640b54bc863b64"
COMMON_CONFIG = ROOT / "configs/doll_handoff_retargeting/common_config.template.json"
FEASIBILITY_CONFIG = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
POLICY_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
POLICY_WORKER = ROOT / "tools/paper_core_act_e1_worker.py"
AUTHKEY = b"paper-core-act-ab-source-v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def frozen_interfaces() -> tuple[list[str], np.ndarray, np.ndarray, NamedJointDeploymentSafetyProjector]:
    if sha256_file(COMMON_PROJECTION_FREEZE) != COMMON_PROJECTION_SHA256:
        raise RuntimeError("frozen common deployment projection changed")
    freeze = read_json(ACTION_FREEZE)
    names = list(freeze["joint_names"])
    specs = freeze["joint_specs"]
    if len(names) != 28 or [row["joint_name"] for row in specs] != names:
        raise RuntimeError("frozen Dataset-B named interface changed")
    projector = NamedJointDeploymentSafetyProjector.from_path(COMMON_PROJECTION_FREEZE)
    if projector.names != names:
        raise RuntimeError("frozen common projection named order differs")
    return names, projector.hard_lower, projector.hard_upper, projector


def common_initial_condition(names: list[str], settle_seconds: float) -> tuple[np.ndarray, dict[str, Any]]:
    if sha256_file(INITIAL_CONTRACT) != INITIAL_CONTRACT_SHA256:
        raise RuntimeError("frozen common A/B initial-state contract changed")
    if sha256_file(EXECUTION_CONFIG) != EXECUTION_CONFIG_SHA256:
        raise RuntimeError("frozen selected ACT execution config changed")
    execution = read_json(EXECUTION_CONFIG)
    official = execution["official_lerobot_execution"]
    if (
        execution["selection"] != "ACT_E1_TEMPORAL_ENSEMBLE"
        or official["chunk_size"] != 50
        or official["n_action_steps"] != 1
        or official["temporal_ensemble_coeff"] != 0.01
        or official["control_fps"] != CONTROL_FPS
        or official["custom_smoothing"]
    ):
        raise RuntimeError("selected execution semantics no longer match paper contract")
    contract = read_json(INITIAL_CONTRACT)
    if contract.get("status") != "COMMON_G1_POLICY_INITIAL_STATE_AB_V1" or contract["joint_order"] != names:
        raise RuntimeError("invalid common A/B named initial-state contract")
    if (
        contract.get("scope") != ["ACT-A40", "ACT-B40"]
        or not contract.get("frozen_before_paper_model_prediction")
        or contract.get("selection_used_policy_predictions_or_performance")
        or contract.get("heldout_state_used")
        or contract["projection"]["manifest_sha256"] != COMMON_PROJECTION_SHA256
    ):
        raise RuntimeError("common A/B initial-state fairness provenance changed")
    if float(contract["settle_duration_seconds"]) != float(settle_seconds):
        raise RuntimeError("settle duration differs from common A/B contract")
    q = np.asarray(contract["full_28d_initial_q_rad"], dtype=np.float64)
    return q, {
        "status": contract["status"],
        "contract": str(INITIAL_CONTRACT),
        "contract_sha256": INITIAL_CONTRACT_SHA256,
        "joint_names": names,
        "q_rad": q,
        "settle_duration_seconds": float(settle_seconds),
        "reset_procedure": contract["reset_procedure"],
        "tolerances": contract["tolerances"],
        "common_for_act_a_and_act_b": True,
        "policy_specific_start_pose": False,
        "selected_execution_config": str(EXECUTION_CONFIG),
        "selected_execution_config_sha256": EXECUTION_CONFIG_SHA256,
    }


def method_consistent_initial_condition(
    names: list[str],
    method: str,
    heldout_output_episode: int,
    source_final_episode: int,
    stable_episode_id: str,
    settle_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact frozen method/episode HELDOUT8 ``state[0]``.

    The state is never projected or selected using policy output.  The
    companion freeze manifest makes the contract immutable after its
    pre-inference construction.
    """

    if method not in ("a", "b") or heldout_output_episode not in range(8):
        raise ValueError("invalid method-consistent initialization identity")
    if not METHOD_CONSISTENT_CONTRACT.is_file() or not METHOD_CONSISTENT_FREEZE.is_file():
        raise FileNotFoundError("method-consistent initialization contract is not frozen")
    freeze = read_json(METHOD_CONSISTENT_FREEZE)
    contract_sha = sha256_file(METHOD_CONSISTENT_CONTRACT)
    if freeze.get("status") != "FROZEN" or freeze.get("contract_sha256") != contract_sha:
        raise RuntimeError("method-consistent initialization contract hash changed")
    if sha256_file(EXECUTION_CONFIG) != EXECUTION_CONFIG_SHA256:
        raise RuntimeError("frozen selected ACT execution config changed")
    execution = read_json(EXECUTION_CONFIG)
    official = execution["official_lerobot_execution"]
    if (
        execution["selection"] != "ACT_E1_TEMPORAL_ENSEMBLE"
        or official["chunk_size"] != 50
        or official["n_action_steps"] != 1
        or official["temporal_ensemble_coeff"] != 0.01
        or official["control_fps"] != CONTROL_FPS
        or official["custom_smoothing"]
    ):
        raise RuntimeError("selected execution semantics no longer match paper contract")
    contract = read_json(METHOD_CONSISTENT_CONTRACT)
    if (
        contract.get("status")
        != "METHOD_CONSISTENT_INITIALIZATION_FROZEN_BEFORE_INFERENCE"
        or contract.get("experiment_name") != "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT"
        or contract.get("joint_order") != names
        or contract["rule"].get("policy_prediction_or_performance_used")
        or contract["rule"].get("initial_q_modified_or_projected")
    ):
        raise RuntimeError("invalid method-consistent initialization contract")
    if float(contract["settle_duration_seconds"]) != float(settle_seconds):
        raise RuntimeError("settle duration differs from method-consistent contract")
    entry = contract["entries"][heldout_output_episode]
    if (
        int(entry["heldout_output_episode"]) != heldout_output_episode
        or int(entry["source_final_episode"]) != source_final_episode
        or entry["stable_episode_id"] != stable_episode_id
    ):
        raise RuntimeError("method-consistent episode/source identity mismatch")
    row = entry["methods"][method]
    q_float32 = np.asarray(row["initial_q_rad"], dtype=np.float32)
    q = q_float32.astype(np.float64)
    if q.shape != (28,) or not np.isfinite(q).all():
        raise RuntimeError("malformed method-consistent initial state")
    if sha256_array(q_float32) != row["initial_q_float32_sha256"]:
        raise RuntimeError("method-consistent initial-state array hash changed")
    if row.get("initial_state_projection_applied") or not row.get(
        "state0_equals_action0_bit_exact"
    ):
        raise RuntimeError("method-consistent state[0] provenance changed")
    return q, {
        "status": "METHOD_CONSISTENT_INITIAL_STATE",
        "contract": str(METHOD_CONSISTENT_CONTRACT),
        "contract_sha256": contract_sha,
        "freeze_manifest": str(METHOD_CONSISTENT_FREEZE),
        "freeze_manifest_sha256": sha256_file(METHOD_CONSISTENT_FREEZE),
        "joint_names": names,
        "q_rad": q,
        "q_float32_sha256": row["initial_q_float32_sha256"],
        "settle_duration_seconds": float(settle_seconds),
        "tolerances": contract["tolerances"],
        "method": method,
        "method_label": row["method"],
        "heldout_output_episode": heldout_output_episode,
        "source_final_episode": source_final_episode,
        "stable_episode_id": stable_episode_id,
        "dataset": row["dataset"],
        "global_dataset_row": row["global_dataset_row"],
        "state0_equals_action0_bit_exact": True,
        "initial_state_projection_applied": False,
        "hard_limit_violation_count_at_exact_state0": row["hard_limit_violation_count"],
        "maximum_arm_hard_limit_excursion_rad": row[
            "maximum_arm_hard_limit_excursion_rad"
        ],
        "maximum_dex3_hard_limit_excursion_rad": row[
            "maximum_dex3_hard_limit_excursion_rad"
        ],
        "preexisting_measured_dex3_excursion_cap_rad": row[
            "preexisting_measured_dex3_excursion_cap_rad"
        ],
        "microscopic_dex3_excursion_logged_not_commanded": row[
            "microscopic_dex3_excursion_logged_not_commanded"
        ],
        "common_for_act_a_and_act_b": False,
        "retargeting_method_specific": True,
        "episode_specific": True,
        "policy_specific_tuned_pose": False,
        "selected_execution_config": str(EXECUTION_CONFIG),
        "selected_execution_config_sha256": EXECUTION_CONFIG_SHA256,
    }


class SourceVideo:
    """Strict sequential source clock: one original video frame per 30-Hz command."""

    def __init__(self, path: Path, expected_frames: int):
        self.path = path.resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open source video: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(round(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.width = int(round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height = int(round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not math.isclose(self.fps, CONTROL_FPS, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"source video is not 30 Hz: {self.fps}")
        if self.frame_count != expected_frames or (self.height, self.width) != (480, 640):
            raise RuntimeError(
                f"source video contract mismatch frames={self.frame_count}/{expected_frames} "
                f"shape={(self.height, self.width)}"
            )
        self.next_index = 0

    def read(self) -> np.ndarray:
        ok, bgr = self.capture.read()
        if not ok:
            raise RuntimeError(f"source video ended at frame {self.next_index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"malformed source RGB at {self.next_index}: {rgb.shape}")
        self.next_index += 1
        return rgb

    def close(self) -> None:
        self.capture.release()

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "height": self.height,
            "width": self.width,
            "decoder": "OpenCV VideoCapture sequential decode",
            "clock": "frame t at original timestamp t/30; no pausing, progress alignment, or phase jumping",
        }


class ACTE1Bridge:
    def __init__(self, checkpoint: Path, model_sha: str, method: str, output: Path):
        bridge = output / "inference_bridge"
        bridge.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:10]
        self.socket = Path(f"/tmp/paper_act_{method}_{os.getpid()}_{token}.sock")
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
                "--method",
                method,
                "--socket",
                str(self.socket),
                "--ready",
                str(self.ready_path),
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

    def infer(self, rgb: np.ndarray, state: np.ndarray) -> dict[str, Any]:
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
        result = {
            "normalized_chunk": np.asarray(response.pop("normalized_chunk"), dtype=np.float32),
            "raw_chunk": np.asarray(response.pop("raw_chunk"), dtype=np.float32),
            "normalized_action": np.asarray(response.pop("normalized_action"), dtype=np.float32),
            "raw_ensembled_action": np.asarray(response.pop("raw_ensembled_action"), dtype=np.float64),
            "record": response,
        }
        if (
            result["normalized_chunk"].shape != (50, 28)
            or result["raw_chunk"].shape != (50, 28)
            or result["normalized_action"].shape != (28,)
            or result["raw_ensembled_action"].shape != (28,)
            or not all(np.isfinite(value).all() for key, value in result.items() if key != "record")
        ):
            raise RuntimeError("malformed/non-finite ACT worker response")
        return result

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
    """Unchanged ACT-B command, measured-state, branch, limit, and collision gates."""

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
            "command_hard_limit_violation_joints": [
                self.names[index] for index in np.flatnonzero(violation)
            ],
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

    def first_raw_chunk(self, current: np.ndarray, chunk: np.ndarray) -> dict[str, Any]:
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


class RolloutVideoRecorder:
    def __init__(self, output: Path, enabled: bool, method: str, source_episode: int):
        self.enabled = enabled
        self.method = method.upper()
        self.source_episode = source_episode
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.paths: dict[str, Path] = {}
        if not enabled:
            return
        for name in ("overview", "three_quarter"):
            path = output / f"{name}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_FPS, (640, 480)
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open rollout video writer: {path}")
            self.paths[name] = path
            self.writers[name] = writer

    def add(self, images: dict[str, np.ndarray], frame: int) -> None:
        if not self.enabled:
            return
        for name, writer in self.writers.items():
            bgr = cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (0, 0), (640, 34), (0, 0, 0), -1)
            cv2.putText(
                bgr,
                f"ACT-{self.method} | source ep {self.source_episode:02d} | t={frame / CONTROL_FPS:05.2f}s | ACT-E1",
                (7, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (80, 235, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(bgr)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()

    def records(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
            for name, path in self.paths.items()
        }
