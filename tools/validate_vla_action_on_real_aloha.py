#!/usr/bin/env python3
"""Fail-closed Stationary ALOHA source-action validator and hardware adapter.

The default, inspect, dry-run, and static hardware-preflight paths never create
``ManipulatorRobot`` and never open a follower-controller connection.  The real
backend deliberately imports and constructs ``ManipulatorRobot`` only inside
``connect()``, after a reviewed safety configuration and explicit operator
acknowledgements have produced a connection authorization.

LeRobot ``capture_observation()`` does not expose controller timestamps.  State
samples therefore carry a host ``time.monotonic_ns()`` timestamp only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import select
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aloha_source_validation_common import minimum_jerk_transition, motion_metrics, sha


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DEFAULT_INPUT = (
    ROOT
    / "evaluation/smolvla_episode49_temporal_consensus/"
    "episode_000049_temporal_consensus.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs/aloha_real_validation/backend_integration"
DEFAULT_INSPECTION_OUTPUT = ROOT / "outputs/aloha_real_validation/hardware_inspection"
DEFAULT_SOFTWARE_SAFETY_OUTPUT = (
    ROOT / "outputs/aloha_real_validation/software_brake_watchdog"
)
DEFAULT_CHARACTERIZATION_OUTPUT = (
    ROOT / "outputs/aloha_real_validation/real_hardware_characterization"
)
DEFAULT_STAGE_A_OUTPUT = ROOT / "outputs/aloha_real_validation/stage_a_inspection"
DEFAULT_AUTHORITATIVE_STAGE_A_RESULT = (
    DEFAULT_STAGE_A_OUTPUT / "stage_a_20260810_160936"
)
DEFAULT_STAGE_B_PREPARATION_OUTPUT = (
    ROOT / "outputs/aloha_real_validation/stage_b_preparation"
)
DEFAULT_STAGE_B_TRACKING_OUTPUT = (
    ROOT / "outputs/aloha_real_validation/stage_b_tracking"
)
DEFAULT_CHARACTERIZATION_CONFIG = (
    ROOT / "configs/aloha_tracking_characterization.reviewed.json"
)
DEFAULT_SAFETY = ROOT / "configs/aloha_source_validation_safety.reviewed.json"
SAFETY_TEMPLATE = ROOT / "configs/aloha_source_validation_safety.template.json"
DEFAULT_STOP_VERIFICATION = ROOT / "configs/aloha_hardware_stop_verification.json"
JOINTS = [*(f"left_joint_{i}" for i in range(7)), *(f"right_joint_{i}" for i in range(7))]
UNITS = ["rad"] * 6 + ["m"] + ["rad"] * 6 + ["m"]
FOLLOWER_ORDER = ("left", "right")
SOURCE_FPS = 30.0
HARDWARE_CONFIRMATION = "I APPROVE STATIONARY ALOHA CONNECT AND SHORT REPLAY"
SHUTDOWN_CONFIRMATION = "I APPROVE VENDOR HOME SLEEP SHUTDOWN MOTION"
INSPECTION_CONFIRMATION = (
    "I APPROVE STATIONARY ALOHA CONNECT FOR READ-ONLY HARDWARE INSPECTION"
)
CHARACTERIZATION_CONFIRMATION = (
    "I APPROVE LOW-LEVEL IDLE-BRAKE CONNECT AND TINY TRACKING CHARACTERIZATION"
)
STAGE_A_CONFIRMATION = "I APPROVE READ-ONLY IDLE-BRAKE STAGE A INSPECTION"
LOW_LEVEL_NO_HOME_PATH_VERIFIED = True
MODE_IDLE_SEMANTICS_VERIFIED = True
NO_POSITION_TARGET_ALLOWED = True
VERIFIED_SOURCE_GRIPPER_RANGE = {
    "left": (-0.00017970101907849312, 0.033793918788433075),
    "right": (-0.00014547863975167274, 0.02084464393556118),
}
STAGE_B_RECOMMENDED_SIDE = "left"
STAGE_B_RECOMMENDED_JOINT = 0
STAGE_B_RECOMMENDED_AMPLITUDE_RAD = 0.005
STAGE_B_RECOMMENDED_DURATION_SEC = 2.0
STAGE_B_RATE_HZ = 30.0
POWER_LOSS_ARM_BEHAVIORS = {
    "REMAINS_SUPPORTED",
    "SLOWLY_SAGS",
    "DROPS_SIGNIFICANTLY",
}


class Blocked(RuntimeError):
    """A fail-closed validation or authorization result."""


class InspectionFailure(Blocked):
    """A categorized inspection-only hardware failure."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class GripperPolicy(str, Enum):
    REJECT = "REJECT"
    CLAMP_TO_CONTROLLER_LIMIT = "CLAMP_TO_CONTROLLER_LIMIT"
    CALIBRATED_OFFSET = "CALIBRATED_OFFSET"


class HardwareRiskClass(str, Enum):
    READ_ONLY_IDLE_INSPECTION = "READ_ONLY_IDLE_INSPECTION"
    MOTION = "MOTION"


RISK_CLASS_A = HardwareRiskClass.READ_ONLY_IDLE_INSPECTION
RISK_CLASS_MOTION = HardwareRiskClass.MOTION


@dataclass(frozen=True)
class StateSample:
    state14: np.ndarray
    host_monotonic_timestamp_ns: int
    timestamp_source: str = "HOST_MONOTONIC_ONLY"


@dataclass(frozen=True)
class HardwareAuthorization:
    connection_allowed: bool
    replay_allowed: bool
    normal_shutdown_allowed: bool
    reasons: tuple[str, ...]
    risk_class: HardwareRiskClass = RISK_CLASS_MOTION


@dataclass(frozen=True)
class InspectionAuthorization:
    inspection_allowed: bool
    normal_shutdown_allowed: bool
    reasons: tuple[str, ...]
    states: dict[str, bool]


@dataclass(frozen=True)
class StageAInspectionAuthorization:
    """Authorization for low-level idle/read/limit inspection only."""

    connection_allowed: bool
    close_transport_allowed: bool
    reasons: tuple[str, ...]
    states: dict[str, bool]
    risk_class: HardwareRiskClass = RISK_CLASS_A
    stage: str = "A"
    motion_allowed: bool = False


@dataclass(frozen=True)
class CharacterizationAuthorization:
    """Strict authorization for generated Stage B/C/D motion only."""

    connection_allowed: bool
    motion_allowed: bool
    close_transport_allowed: bool
    stage: str
    reasons: tuple[str, ...]
    states: dict[str, bool]
    risk_class: HardwareRiskClass = RISK_CLASS_MOTION


@dataclass(frozen=True)
class WatchdogConfig:
    """Runtime thresholds; ``None`` means unapproved/needs characterization."""

    max_tracking_error_arm_rad: float | None = None
    max_tracking_error_gripper_m: float | None = None
    tracking_error_duration_sec: float | None = None
    max_command_step_arm_rad: float | None = None
    max_command_step_gripper_m: float | None = None
    max_command_velocity_arm_rad_s: float | None = None
    max_command_velocity_gripper_m_s: float | None = None
    max_state_age_sec: float | None = None
    max_state_read_duration_sec: float | None = None
    max_command_call_duration_sec: float | None = None
    max_loop_overrun_sec: float | None = None
    controller_position_tolerance: float | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "WatchdogConfig":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise Blocked(f"unknown watchdog config fields: {sorted(unknown)}")
        config = cls(**value)
        for name, raw in value.items():
            if raw is not None and (not np.isfinite(float(raw)) or float(raw) <= 0):
                raise Blocked(f"watchdog threshold {name} must be positive or null")
        return config

    def unresolved_fields(self) -> list[str]:
        return [
            name
            for name in self.__dataclass_fields__
            if getattr(self, name) is None
        ]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def load_stop_verification(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Load a human-created physical power-cut record and validate it fail-closed.

    A CLI acknowledgement is deliberately not an input to this function.  The
    record describes observations that cannot be established from software.
    """
    path = Path(path)
    if not path.exists():
        return {}, [f"BLOCKED_NO_VERIFIED_OPERATOR_STOP: record missing: {path}"]
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"BLOCKED_NO_VERIFIED_OPERATOR_STOP: invalid record: {error}"]
    if not isinstance(record, dict):
        return {}, ["BLOCKED_NO_VERIFIED_OPERATOR_STOP: record must be a JSON object"]

    reasons: list[str] = []

    def require_true(field: str) -> None:
        if record.get(field) is not True:
            reasons.append(f"physical stop field is not true: {field}")

    def require_text(field: str) -> None:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"physical stop field is not a non-empty string: {field}")

    require_true("verified")
    require_true("cuts_left_follower")
    require_true("cuts_right_follower")
    require_true("single_operator_action")
    require_true("reachable_outside_workspace")
    require_true("area_below_both_arms_clear")
    require_true("power_loss_behavior_accepted")
    for field in ("mechanism", "location", "verified_by", "verification_date", "notes"):
        require_text(field)

    behavior = record.get("power_loss_arm_behavior")
    if behavior not in POWER_LOSS_ARM_BEHAVIORS:
        reasons.append(
            "physical stop power_loss_arm_behavior must be one of "
            + ", ".join(sorted(POWER_LOSS_ARM_BEHAVIORS))
        )
    elif behavior == "DROPS_SIGNIFICANTLY":
        reasons.append("physical stop is unacceptable: arms drop significantly on power loss")

    if reasons:
        reasons.insert(0, "BLOCKED_NO_VERIFIED_OPERATOR_STOP")
    return record, reasons


def verified_operator_stop(record: dict[str, Any], reasons: list[str]) -> bool:
    return not reasons and record.get("verified") is True


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_action14(action14: Any, *, label: str = "action14") -> np.ndarray:
    array = np.asarray(action14)
    if not np.issubdtype(array.dtype, np.number):
        raise Blocked(f"{label} must have numeric dtype")
    if array.shape != (14,):
        raise Blocked(f"{label} must have shape (14,), got {array.shape}")
    if not np.isfinite(array).all():
        raise Blocked(f"{label} contains NaN/Inf")
    return np.asarray(array, dtype=np.float64).copy()


def to_cpu_float32_tensor(action14: Any) -> Any:
    import torch

    validated = validate_action14(action14)
    tensor = torch.as_tensor(validated, dtype=torch.float32, device="cpu")
    if tensor.shape != (14,) or tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise Blocked("hardware action conversion contract failed")
    if not bool(torch.isfinite(tensor).all()):
        raise Blocked("hardware action tensor contains NaN/Inf")
    return tensor


def load_source_action(path: Path = DEFAULT_INPUT) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    file_hash_before = sha(path)
    with np.load(path, allow_pickle=False) as archive:
        if "optimized_action" not in archive:
            raise Blocked("optimized_action missing")
        source = np.asarray(archive["optimized_action"]).copy()
        fps = float(archive["fps"]) if "fps" in archive else SOURCE_FPS
        keys = list(archive.files)
    if source.shape != (990, 14):
        raise Blocked(f"optimized_action must have shape (990,14), got {source.shape}")
    if not np.issubdtype(source.dtype, np.number) or not np.isfinite(source).all():
        raise Blocked("optimized_action must be finite numeric data")
    if fps != SOURCE_FPS:
        raise Blocked(f"source fps must remain 30 Hz, got {fps}")
    integrity = {
        "status": "VERIFIED_OFFLINE",
        "path": str(path.resolve()),
        "npz_sha256_before": file_hash_before,
        "npz_sha256_after": sha(path),
        "array_sha256_before": array_sha256(source),
        "shape_before": list(source.shape),
        "dtype_before": str(source.dtype),
        "keys": keys,
    }
    return source, integrity


def assert_source_unchanged(path: Path, source: np.ndarray, integrity: dict[str, Any]) -> None:
    if sha(path) != integrity["npz_sha256_before"]:
        raise Blocked("BLOCKED_SOURCE_ACTION_MUTATION: source NPZ hash changed")
    if array_sha256(source) != integrity["array_sha256_before"]:
        raise Blocked("BLOCKED_SOURCE_ACTION_MUTATION: in-memory source changed")
    if list(source.shape) != integrity["shape_before"]:
        raise Blocked("BLOCKED_SOURCE_ACTION_MUTATION: source shape changed")
    integrity.update(
        {
            "npz_sha256_after": sha(path),
            "array_sha256_after": array_sha256(source),
            "shape_after": list(source.shape),
            "source_action_unchanged": True,
        }
    )


def _config_summary(config: Any) -> dict[str, Any]:
    followers = getattr(config, "follower_arms", None)
    if not isinstance(followers, dict):
        raise Blocked("Stationary ALOHA config has no follower_arms dictionary")
    order = tuple(followers.keys())
    if order != FOLLOWER_ORDER:
        raise Blocked(f"BLOCKED_FOLLOWER_ORDER: expected {FOLLOWER_ORDER}, got {order}")
    devices = {}
    for side, arm in followers.items():
        devices[side] = {
            "ip": getattr(arm, "ip", None),
            "model": getattr(arm, "model", None),
            "min_time_to_move_multiplier": getattr(arm, "min_time_to_move_multiplier", None),
        }
    return {
        "status": "VERIFIED_FROM_SOURCE_CODE",
        "config_source": "create_robot_config('trossen_ai_stationary')",
        "follower_order": list(order),
        "followers": devices,
        "leaders_removed": getattr(config, "leader_arms", None) == {},
        "cameras_removed": getattr(config, "cameras", None) == {},
    }


def serialize_controller_limits(raw: Any) -> dict[str, float]:
    fields = (
        "position_min",
        "position_max",
        "position_tolerance",
        "velocity_max",
        "velocity_tolerance",
        "effort_max",
        "effort_tolerance",
    )
    result: dict[str, float] = {}
    for field in fields:
        if not hasattr(raw, field):
            raise Blocked(f"controller limit object missing {field}")
        value = float(getattr(raw, field))
        if not np.isfinite(value):
            raise Blocked(f"controller limit {field} is non-finite")
        result[field] = value
    if result["position_min"] > result["position_max"]:
        raise Blocked("controller position_min exceeds position_max")
    return result


def validate_controller_limits(limits: Any) -> dict[str, list[dict[str, float]]]:
    if not isinstance(limits, dict) or tuple(limits.keys()) != FOLLOWER_ORDER:
        raise Blocked("controller limits must be ordered left then right")
    validated: dict[str, list[dict[str, float]]] = {}
    for side in FOLLOWER_ORDER:
        if len(limits[side]) != 7:
            raise Blocked(f"{side} controller limits must contain 7 joints")
        validated[side] = []
        for item in limits[side]:
            if isinstance(item, dict):
                position_min = float(item["position_min"])
                position_max = float(item["position_max"])
                if not np.isfinite([position_min, position_max]).all() or position_min > position_max:
                    raise Blocked(f"invalid {side} controller position limit")
                validated[side].append({**item, "position_min": position_min, "position_max": position_max})
            else:
                validated[side].append(serialize_controller_limits(item))
    return validated


def assert_controller_limits_match_review(
    actual: dict[str, list[dict[str, float]]],
    reviewed: dict[str, list[dict[str, float]]],
    *,
    absolute_tolerance: float = 1e-9,
) -> None:
    actual_checked = validate_controller_limits(actual)
    reviewed_checked = validate_controller_limits(reviewed)
    for side in FOLLOWER_ORDER:
        for joint in range(7):
            for field in ("position_min", "position_max"):
                observed = actual_checked[side][joint][field]
                approved = reviewed_checked[side][joint][field]
                if not np.isclose(observed, approved, rtol=0.0, atol=absolute_tolerance):
                    raise Blocked(
                        f"controller limit differs from reviewed value: {side} joint_{joint} "
                        f"{field} actual={observed} reviewed={approved}"
                    )


def _read_json_file(path: Path, expected_type: type) -> Any:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Blocked(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, expected_type):
        raise Blocked(
            f"{path} must contain {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def load_authoritative_stage_a_limits(
    directory: Path = DEFAULT_AUTHORITATIVE_STAGE_A_RESULT,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, Any]]:
    """Load controller-reported limits from the user-designated real Stage A run.

    This is a filesystem-only operation.  It never imports a vendor driver and
    never opens a controller connection.  The provenance checks prevent a
    candidate/URDF range from being silently substituted for the measured
    controller values.
    """
    directory = Path(directory)
    required = {
        "run_manifest": directory / "run_manifest.json",
        "stage_a_result": directory / "stage_a_result.json",
        "position_command_audit": directory / "position_command_audit.json",
        "persistent_config_snapshot": directory / "persistent_config_snapshot.json",
        "state14": directory / "state14.json",
        "left_modes": directory / "left_modes.json",
        "right_modes": directory / "right_modes.json",
        "left_limits": directory / "controller_limits_left.json",
        "right_limits": directory / "controller_limits_right.json",
        "gripper_limits": directory / "gripper_limits.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise Blocked(
            "authoritative Stage A result is incomplete; missing: " + ", ".join(missing)
        )

    manifest = _read_json_file(required["run_manifest"], dict)
    stage_result = _read_json_file(required["stage_a_result"], dict)
    command_audit = _read_json_file(required["position_command_audit"], dict)
    config_snapshot = _read_json_file(required["persistent_config_snapshot"], dict)
    state_record = _read_json_file(required["state14"], dict)
    mode_records = {
        side: _read_json_file(required[f"{side}_modes"], dict)
        for side in FOLLOWER_ORDER
    }
    gripper_record = _read_json_file(required["gripper_limits"], dict)
    left = _read_json_file(required["left_limits"], list)
    right = _read_json_file(required["right_limits"], list)

    if manifest.get("status") != "STAGE_A_INSPECTION_COMPLETE":
        raise Blocked("authoritative Stage A manifest is not complete")
    if stage_result.get("status") != "STAGE_A_INSPECTION_COMPLETE" or stage_result.get(
        "failure"
    ) is not None:
        raise Blocked("authoritative Stage A result did not complete successfully")
    invariant_fields = {
        "position_commands": 0,
        "goal_position_commands": 0,
        "send_action_calls": 0,
        "optimized_action_loaded": False,
        "trajectory_loop_entered": False,
        "high_level_connect_called": False,
    }
    for field, expected in invariant_fields.items():
        if command_audit.get(field) != expected:
            raise Blocked(
                f"authoritative Stage A position-command invariant failed: {field}"
            )
    if command_audit.get("invariant_confirmed") is not True:
        raise Blocked("authoritative Stage A command invariant was not confirmed")
    if stage_result.get("brake_abort", {}).get("status") != "SOFTWARE_BRAKE_ABORT_CONFIRMED":
        raise Blocked("authoritative Stage A did not confirm two-sided idle brake")
    if stage_result.get("transport_close", {}).get("status") != "NO_HOME_TRANSPORT_CLOSED":
        raise Blocked("authoritative Stage A did not confirm no-home transport close")
    for side, modes in mode_records.items():
        for phase in ("after_configure", "after_reads", "brake_abort_readback"):
            if modes.get(phase) != ["idle"] * 7:
                raise Blocked(
                    f"authoritative Stage A {side} {phase} is not 7/7 idle"
                )
    if tuple(config_snapshot.get("follower_order", ())) != FOLLOWER_ORDER:
        raise Blocked("BLOCKED_FOLLOWER_ORDER in authoritative Stage A result")

    limits = validate_inspection_limits({"left": left, "right": right})
    state = validate_action14(state_record.get("state14"), label="Stage A state14")
    sides = gripper_record.get("sides")
    if (
        gripper_record.get("authority") != "CONTROLLER_REPORTED_JOINT_INDEX_6"
        or not isinstance(sides, dict)
    ):
        raise Blocked("Stage A gripper limits lack controller-reported authority")
    for side in FOLLOWER_ORDER:
        recorded = sides.get(side)
        if not isinstance(recorded, dict) or recorded.get("joint_index") != 6:
            raise Blocked(f"Stage A {side} gripper limit record is invalid")
        for field in ("position_min", "position_max"):
            if not np.isclose(
                float(recorded[field]),
                float(limits[side][6][field]),
                rtol=0.0,
                atol=0.0,
            ):
                raise Blocked(f"Stage A {side} gripper {field} disagrees with joint index 6")

    source_comparison_path = directory / "controller_limit_source_comparison.json"
    source_comparison = (
        _read_json_file(source_comparison_path, dict)
        if source_comparison_path.is_file()
        else None
    )
    if source_comparison is not None and source_comparison.get("arm_violation_count") != 0:
        raise Blocked("authoritative Stage A source audit contains arm limit violations")

    authority = {
        "status": "AUTHORITATIVE_REAL_STAGE_A_CONTROLLER_LIMITS_LOADED",
        "authority": "CONTROLLER_REPORTED_LIMITS_FROM_USER_DESIGNATED_REAL_STAGE_A",
        "directory": str(directory.resolve()),
        "manifest_status": manifest["status"],
        "risk_class": manifest.get("risk_class"),
        "follower_order": list(FOLLOWER_ORDER),
        "followers": config_snapshot.get("followers"),
        "state14": state.tolist(),
        "state_timestamp": {
            "host_monotonic_timestamp_ns": state_record.get(
                "host_monotonic_timestamp_ns"
            ),
            "controller_synchronized": False,
        },
        "arm_violation_count": (
            None
            if source_comparison is None
            else int(source_comparison["arm_violation_count"])
        ),
        "gripper_feedback_is_command_authority": False,
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha(path)}
            for name, path in required.items()
        },
    }
    return limits, authority


class StageAHardwareCommandAdapter:
    """Transient gripper-only saturation against real Stage A controller limits."""

    ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    GRIPPER_CHANNELS = {"left": 6, "right": 13}

    def __init__(
        self,
        controller_limits: dict[str, list[dict[str, float]]],
        authority: dict[str, Any],
    ):
        self.controller_limits = validate_controller_limits(controller_limits)
        if authority.get("status") != "AUTHORITATIVE_REAL_STAGE_A_CONTROLLER_LIMITS_LOADED":
            raise Blocked("hardware adapter requires authoritative real Stage A provenance")
        self.authority = authority

    @classmethod
    def from_stage_a_directory(
        cls, directory: Path = DEFAULT_AUTHORITATIVE_STAGE_A_RESULT
    ) -> "StageAHardwareCommandAdapter":
        limits, authority = load_authoritative_stage_a_limits(directory)
        return cls(limits, authority)

    def adapt(
        self, source_action14: Any, *, source_frame: int | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raw = np.asarray(source_action14)
        validate_action14(raw, label="hardware adapter source action")
        command = (
            raw.copy()
            if np.issubdtype(raw.dtype, np.floating)
            else raw.astype(np.float64, copy=True)
        )
        source_before = command.copy()

        for side_index, side in enumerate(FOLLOWER_ORDER):
            base = side_index * 7
            for joint in range(6):
                value = float(source_before[base + joint])
                lower = float(self.controller_limits[side][joint]["position_min"])
                upper = float(self.controller_limits[side][joint]["position_max"])
                if not lower <= value <= upper:
                    raise Blocked(
                        f"{side} arm joint_{joint}={value} is outside authoritative "
                        f"Stage A controller range [{lower}, {upper}]"
                    )

        channel_records: dict[str, Any] = {}
        for side in FOLLOWER_ORDER:
            index = self.GRIPPER_CHANNELS[side]
            lower = float(self.controller_limits[side][6]["position_min"])
            upper = float(self.controller_limits[side][6]["position_max"])
            source_value = float(source_before[index])
            command_value = float(np.clip(source_value, lower, upper))
            command[index] = command_value
            if source_value < lower:
                reason = "BELOW_CONTROLLER_MINIMUM"
            elif source_value > upper:
                reason = "ABOVE_CONTROLLER_MAXIMUM"
            else:
                reason = "IN_RANGE_UNCHANGED"
            channel_records[side] = {
                "source_index": index,
                "source_before": source_value,
                "command_after_saturation": command_value,
                "delta": command_value - source_value,
                "reason": reason,
                "controller_min": lower,
                "controller_max": upper,
                "unit": "m",
            }

        if not np.array_equal(
            source_before[self.ARM_INDICES], command[self.ARM_INDICES]
        ):
            raise Blocked("hardware command adapter modified an arm channel")
        record = {
            "status": "HARDWARE_COMMAND_LIMIT_ADAPTED",
            "source_frame": source_frame,
            "policy": "GRIPPER_ONLY_SATURATION_TO_REAL_STAGE_A_CONTROLLER_LIMITS",
            "source_before": source_before.tolist(),
            "command_after_saturation": command.tolist(),
            "delta": (command.astype(np.float64) - source_before.astype(np.float64)).tolist(),
            "channels": channel_records,
            "arm_channels_modified": 0,
            "source_object_modified": False,
        }
        return command, record


def audit_gripper_hardware_saturation(
    source: np.ndarray,
    adapter: StageAHardwareCommandAdapter,
) -> tuple[dict[str, Any], np.ndarray]:
    """Audit the complete source without mutating or persisting an adapted trajectory."""
    source_array = np.asarray(source)
    if source_array.shape != (990, 14) or not np.isfinite(source_array).all():
        raise Blocked("gripper saturation audit requires finite (990,14) source")
    source_hash = array_sha256(source_array)
    commands = source_array.copy()
    for side, index in adapter.GRIPPER_CHANNELS.items():
        lower = float(adapter.controller_limits[side][6]["position_min"])
        upper = float(adapter.controller_limits[side][6]["position_max"])
        commands[:, index] = np.clip(commands[:, index], lower, upper)
    if array_sha256(source_array) != source_hash:
        raise Blocked("BLOCKED_SOURCE_ACTION_MUTATION during saturation audit")
    arm_equal = np.array_equal(
        source_array[:, adapter.ARM_INDICES], commands[:, adapter.ARM_INDICES]
    )
    if not arm_equal:
        raise Blocked("hardware saturation audit modified an arm channel")

    sides: dict[str, Any] = {}
    for side, index in adapter.GRIPPER_CHANNELS.items():
        source_values = source_array[:, index].astype(np.float64)
        command_values = commands[:, index].astype(np.float64)
        correction = command_values - source_values
        saturated = np.flatnonzero(correction != 0.0)
        sides[side] = {
            "source_index": index,
            "unit": "m",
            "source_min": float(source_values.min()),
            "source_max": float(source_values.max()),
            "hardware_command_min": float(command_values.min()),
            "hardware_command_max": float(command_values.max()),
            "controller_min": float(adapter.controller_limits[side][6]["position_min"]),
            "controller_max": float(adapter.controller_limits[side][6]["position_max"]),
            "saturated_frame_count": int(saturated.size),
            "maximum_saturation_correction": (
                0.0 if not saturated.size else float(np.abs(correction[saturated]).max())
            ),
            "mean_correction_over_saturated_frames": (
                0.0 if not saturated.size else float(np.abs(correction[saturated]).mean())
            ),
            "first_saturated_frame": None if not saturated.size else int(saturated[0]),
            "last_saturated_frame": None if not saturated.size else int(saturated[-1]),
            "below_minimum_frame_count": int(
                np.count_nonzero(source_values < adapter.controller_limits[side][6]["position_min"])
            ),
            "above_maximum_frame_count": int(
                np.count_nonzero(source_values > adapter.controller_limits[side][6]["position_max"])
            ),
        }
    audit = {
        "status": "HARDWARE_GRIPPER_SATURATION_AUDIT_COMPLETE",
        "terminology": "hardware command saturation / hardware limit adaptation",
        "source_shape": list(source_array.shape),
        "source_dtype": str(source_array.dtype),
        "source_array_sha256_before": source_hash,
        "source_array_sha256_after": array_sha256(source_array),
        "source_action_modified": False,
        "adapted_trajectory_persisted": False,
        "arm_channels_byte_identical": arm_equal,
        "arm_channel_modification_count": int(
            np.count_nonzero(
                source_array[:, adapter.ARM_INDICES]
                != commands[:, adapter.ARM_INDICES]
            )
        ),
        "gripper_event_ordering_unchanged": True,
        "gripper_temporal_indices_unchanged": True,
        "frame_index_mapping": "IDENTITY_0_TO_989",
        "sides": sides,
        "authority": adapter.authority,
    }
    return audit, commands


def prepare_action_for_limits(
    action14: Any,
    controller_limits: dict[str, list[dict[str, float]]] | None,
    gripper_policy: GripperPolicy = GripperPolicy.REJECT,
    calibrated_offsets: dict[str, float] | None = None,
) -> np.ndarray:
    prepared = validate_action14(action14).copy()
    if controller_limits is None:
        raise Blocked("NEEDS_USER_CONFIRMATION: controller limits are unavailable")
    limits = validate_controller_limits(controller_limits)
    policy = GripperPolicy(gripper_policy)
    if policy is GripperPolicy.CALIBRATED_OFFSET:
        if calibrated_offsets is None or any(side not in calibrated_offsets for side in FOLLOWER_ORDER):
            raise Blocked("CALIBRATED_OFFSET requires reviewed left/right offsets")
        prepared[6] += float(calibrated_offsets["left"])
        prepared[13] += float(calibrated_offsets["right"])

    for side_index, side in enumerate(FOLLOWER_ORDER):
        base = side_index * 7
        for joint in range(7):
            index = base + joint
            lower = limits[side][joint]["position_min"]
            upper = limits[side][joint]["position_max"]
            if lower <= prepared[index] <= upper:
                continue
            is_gripper = joint == 6
            if is_gripper and policy is GripperPolicy.CLAMP_TO_CONTROLLER_LIMIT:
                prepared[index] = np.clip(prepared[index], lower, upper)
                continue
            kind = "gripper" if is_gripper else "arm"
            raise Blocked(
                f"{side} {kind} channel {index}={prepared[index]:.9g} outside "
                f"controller range [{lower:.9g}, {upper:.9g}] under {policy.value}"
            )
    return prepared


class ALOHAHardwareBackend:
    """Thin, lazy adapter over the verified LeRobot ``ManipulatorRobot`` path."""

    def __init__(self, config_factory: Callable[[str], Any] | None = None):
        self._config_factory = config_factory
        self.config: Any | None = None
        self.config_summary: dict[str, Any] | None = None
        self.robot: Any | None = None
        self.command_loop_stopped = True
        self.last_brake_result: dict[str, Any] | None = None

    def build_config(self) -> dict[str, Any]:
        if self._config_factory is None:
            from trossen_ai_data_collection_ui.utils.utils import create_robot_config

            factory = create_robot_config
        else:
            factory = self._config_factory
        config = factory("trossen_ai_stationary")
        config.leader_arms = {}
        config.cameras = {}
        summary = _config_summary(config)
        self.config = config
        self.config_summary = summary
        return summary

    def connect(self, authorization: HardwareAuthorization) -> None:
        if not authorization.connection_allowed:
            raise Blocked("HARDWARE EXECUTION REFUSED: " + "; ".join(authorization.reasons))
        self._construct_and_connect(command_loop_active=True)

    def connect_for_inspection(self, authorization: InspectionAuthorization) -> None:
        if not authorization.inspection_allowed:
            raise Blocked("HARDWARE INSPECTION REFUSED: " + "; ".join(authorization.reasons))
        self._construct_and_connect(command_loop_active=False)

    def _construct_and_connect(self, *, command_loop_active: bool) -> None:
        if self.robot is not None:
            raise Blocked("hardware backend already constructed")
        if self.config is None:
            self.build_config()
        # Import and construction are intentionally delayed until every connection gate passes.
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

        self.robot = ManipulatorRobot(self.config)
        self.robot.connect()  # Existing vendor path: configures modes and moves followers home.
        self.command_loop_stopped = not command_loop_active

    def is_connected(self) -> bool:
        return bool(self.robot is not None and getattr(self.robot, "is_connected", False))

    def read_state(self) -> StateSample:
        if not self.is_connected():
            raise Blocked("hardware backend is not connected")
        observation = self.robot.capture_observation()
        raw = observation["observation.state"]
        if hasattr(raw, "detach") and hasattr(raw, "cpu"):
            raw = raw.detach().cpu().numpy()
        state = validate_action14(raw, label="observation.state")
        return StateSample(state, time.monotonic_ns())

    def read_controller_limits(self) -> dict[str, list[dict[str, float]]]:
        if not self.is_connected():
            raise Blocked("hardware backend is not connected")
        limits: dict[str, list[dict[str, float]]] = {}
        if tuple(self.robot.follower_arms.keys()) != FOLLOWER_ORDER:
            raise Blocked("BLOCKED_FOLLOWER_ORDER at connected robot")
        for side in FOLLOWER_ORDER:
            raw = self.robot.follower_arms[side].driver.get_joint_limits()
            limits[side] = [serialize_controller_limits(item) for item in raw]
        return validate_controller_limits(limits)

    def prepare_action(
        self,
        action14: Any,
        controller_limits: dict[str, list[dict[str, float]]],
        gripper_policy: GripperPolicy = GripperPolicy.REJECT,
        calibrated_offsets: dict[str, float] | None = None,
    ) -> np.ndarray:
        return prepare_action_for_limits(action14, controller_limits, gripper_policy, calibrated_offsets)

    def send_action(
        self,
        action14: Any,
        authorization: HardwareAuthorization,
        controller_limits: dict[str, list[dict[str, float]]],
        gripper_policy: GripperPolicy = GripperPolicy.REJECT,
        calibrated_offsets: dict[str, float] | None = None,
    ) -> np.ndarray:
        if not authorization.replay_allowed:
            raise Blocked("HARDWARE EXECUTION REFUSED: replay authorization is false")
        if not self.is_connected() or self.command_loop_stopped:
            raise Blocked("hardware command loop is not active")
        prepared = self.prepare_action(action14, controller_limits, gripper_policy, calibrated_offsets)
        tensor = to_cpu_float32_tensor(prepared)
        sent = self.robot.send_action(tensor)
        if hasattr(sent, "detach") and hasattr(sent, "cpu"):
            sent = sent.detach().cpu().numpy()
        return validate_action14(sent, label="sent action")

    def stop_command_loop(self) -> None:
        """Stop issuing trajectory commands; deliberately does not disconnect or move."""
        self.command_loop_stopped = True

    def brake_abort(self, reason: str) -> dict[str, Any]:
        """Request low-level idle through the connected LeRobot followers."""
        self.stop_command_loop()
        result: dict[str, Any] = {
            "status": "SOFTWARE_BRAKE_ABORT_REQUESTED",
            "reason": str(reason),
            "requested_host_monotonic_ns": time.monotonic_ns(),
            "completed_host_monotonic_ns": None,
            "sides": {},
            "disconnect_called": False,
            "home_called": False,
            "sleep_called": False,
            "physical_estop": False,
            "warning": (
                "BRAKE_ABORT IS SOFTWARE-COMMANDED, NETWORK/CONTROLLER-DEPENDENT, "
                "AND IS NOT A PHYSICAL E-STOP"
            ),
        }
        try:
            import trossen_arm as trossen

            idle_mode = trossen.Mode.idle
        except Exception as error:
            idle_mode = None
            import_error = f"{type(error).__name__}: {error}"
        else:
            import_error = None
        followers = (
            getattr(self.robot, "follower_arms", {}) if self.robot is not None else {}
        )
        confirmed = 0
        for side in FOLLOWER_ORDER:
            side_result = {
                "idle_requested": False,
                "idle_confirmed": False,
                "api_readback": None,
                "error": None,
            }
            result["sides"][side] = side_result
            try:
                if import_error is not None:
                    raise RuntimeError(import_error)
                wrapper = followers[side]
                driver = wrapper.driver
                side_result["idle_requested"] = True
                driver.set_all_modes(idle_mode)
                modes = list(driver.get_modes())
                side_result["api_readback"] = [_mode_name(mode) for mode in modes]
                if len(modes) != 7 or any(mode != idle_mode for mode in modes):
                    raise Blocked(f"{side} did not return seven idle modes")
                side_result["idle_confirmed"] = True
                confirmed += 1
            except Exception as error:
                side_result["error"] = f"{type(error).__name__}: {error}"
        result["completed_host_monotonic_ns"] = time.monotonic_ns()
        result["status"] = (
            "SOFTWARE_BRAKE_ABORT_CONFIRMED"
            if confirmed == 2
            else "SOFTWARE_BRAKE_ABORT_PARTIAL"
            if confirmed == 1
            else "SOFTWARE_BRAKE_ABORT_FAILED"
        )
        self.last_brake_result = result
        return result

    def normal_shutdown(self, authorization: HardwareAuthorization) -> None:
        if not authorization.normal_shutdown_allowed:
            raise Blocked("vendor disconnect motion is not approved")
        if not self.is_connected():
            raise Blocked("hardware backend is not connected")
        self.robot.disconnect()  # Existing vendor path moves home, then sleep.

    def normal_shutdown_for_inspection(self, authorization: InspectionAuthorization) -> None:
        if not authorization.normal_shutdown_allowed:
            raise Blocked("inspection vendor disconnect motion is not approved")
        if not self.is_connected():
            raise Blocked("hardware backend is not connected")
        self.robot.disconnect()  # Existing vendor path moves home, then sleep.


def _mode_name(mode: Any) -> str:
    name = getattr(mode, "name", None)
    return str(name if name is not None else mode)


class StageADriverGuard:
    """Allow Stage-A read/idle calls while rejecting position APIs before delegation."""

    POSITION_METHODS = {
        "set_all_positions",
        "set_arm_positions",
        "set_gripper_position",
        "set_joint_position",
        "set_joint_positions",
    }
    FORBIDDEN_METHODS = {*POSITION_METHODS, "send_action", "write"}

    def __init__(self, driver: Any, audit: dict[str, Any], side: str):
        self._driver = driver
        self._audit = audit
        self._side = side

    @property
    def raw_driver(self) -> Any:
        return self._driver

    def __getattr__(self, name: str) -> Any:
        if name not in self.FORBIDDEN_METHODS:
            return getattr(self._driver, name)

        def reject(*args: Any, **kwargs: Any) -> None:
            del kwargs
            if name in self.POSITION_METHODS:
                self._audit["position_commands"] += 1
            elif name == "send_action":
                self._audit["send_action_calls"] += 1
            elif name == "write":
                data_name = args[0] if args else None
                if data_name == "Goal_Position":
                    self._audit["goal_position_commands"] += 1
                else:
                    self._audit["other_write_attempts"] += 1
            self._audit["violations"].append(
                {"side": self._side, "method": name, "args_count": len(args)}
            )
            raise Blocked(
                f"BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION: {self._side}.{name}"
            )

        return reject


class StationaryAlohaLowLevelBackend:
    """Direct installed ``trossen_arm`` path with no home/sleep position target.

    The installed binary's ``configure()`` requests ``Mode.idle`` for all seven
    joints and starts its input/output daemon.  It does not call
    ``set_all_positions``.  A brake-mode transition can still have a mechanical
    effect, so this is deliberately described as *no-home*, not proven
    physically motion-free.

    Production imports and driver construction remain lazy.  Unit tests inject
    driver/mode/model factories and therefore never touch Ethernet hardware.
    """

    CONNECTION_CLASSIFICATION = "LOW_LEVEL_NO_HOME_IDLE_BRAKE_VERIFIED_MOTION_INCONCLUSIVE"

    def __init__(
        self,
        config_factory: Callable[[str], Any] | None = None,
        driver_factory: Callable[[str], Any] | None = None,
        *,
        idle_mode: Any | None = None,
        position_mode: Any | None = None,
        model_mapping: dict[str, tuple[Any, Any] | list[Any]] | None = None,
    ):
        self._config_factory = config_factory
        self._driver_factory = driver_factory
        self._idle_mode = idle_mode
        self._position_mode = position_mode
        self._model_mapping = model_mapping
        self.config: Any | None = None
        self.config_summary: dict[str, Any] | None = None
        self.drivers: dict[str, Any] = {}
        self.configured_sides: list[str] = []
        self.command_loop_stopped = True
        self.motion_sides: tuple[str, ...] = ()
        self.motion_joint_indices: tuple[int, ...] = ()
        self.last_brake_result: dict[str, Any] | None = None
        self.brake_history: list[dict[str, Any]] = []
        self.transport_close_history: list[dict[str, Any]] = []
        self.active_risk_class: HardwareRiskClass | None = None
        self._stage_a_audit: dict[str, Any] = self._new_stage_a_audit()

    @staticmethod
    def _new_stage_a_audit() -> dict[str, Any]:
        return {
            "position_commands": 0,
            "goal_position_commands": 0,
            "send_action_calls": 0,
            "other_write_attempts": 0,
            "trajectory_loop_entered": False,
            "optimized_action_loaded": False,
            "high_level_connect_called": False,
            "manipulator_robot_constructed": False,
            "violations": [],
        }

    def begin_stage_a_inspection(self) -> None:
        if self.drivers or self.configured_sides:
            raise Blocked("Stage A guard must be activated before driver construction")
        self.active_risk_class = RISK_CLASS_A
        self._stage_a_audit = self._new_stage_a_audit()

    def stage_a_position_command_audit(self) -> dict[str, Any]:
        audit = {
            **self._stage_a_audit,
            "violations": list(self._stage_a_audit["violations"]),
        }
        invariant_ok = (
            audit["position_commands"] == 0
            and audit["goal_position_commands"] == 0
            and audit["send_action_calls"] == 0
            and audit["other_write_attempts"] == 0
            and audit["trajectory_loop_entered"] is False
            and audit["optimized_action_loaded"] is False
            and audit["high_level_connect_called"] is False
            and audit["manipulator_robot_constructed"] is False
            and not audit["violations"]
        )
        audit.update(
            {
                "status": (
                    "STAGE_A_NO_POSITION_COMMAND_INVARIANT_CONFIRMED"
                    if invariant_ok
                    else "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION"
                ),
                "invariant_confirmed": invariant_ok,
                "position_target_reached_hardware": False,
            }
        )
        return audit

    def assert_stage_a_position_command_invariant(self) -> dict[str, Any]:
        audit = self.stage_a_position_command_audit()
        if not audit["invariant_confirmed"]:
            raise Blocked("BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION")
        return audit

    def build_config(self) -> dict[str, Any]:
        if self._config_factory is None:
            from trossen_ai_data_collection_ui.utils.utils import create_robot_config

            factory = create_robot_config
        else:
            factory = self._config_factory
        config = factory("trossen_ai_stationary")
        config.leader_arms = {}
        config.cameras = {}
        summary = _config_summary(config)
        self.config = config
        self.config_summary = summary
        return summary

    def _resolve_vendor(self) -> None:
        if (
            self._driver_factory is not None
            and self._idle_mode is not None
            and self._position_mode is not None
            and self._model_mapping is not None
        ):
            return
        import trossen_arm as trossen
        from lerobot.common.robot_devices.motors.trossen_arm_driver import (
            TROSSEN_ARM_MODELS,
        )

        self._driver_factory = lambda _side: trossen.TrossenArmDriver()
        self._idle_mode = trossen.Mode.idle
        self._position_mode = trossen.Mode.position
        self._model_mapping = TROSSEN_ARM_MODELS

    def _assert_idle_modes(self, side: str) -> list[str]:
        modes = list(self.drivers[side].get_modes())
        if len(modes) != 7:
            raise Blocked(f"{side} mode readback must contain 7 joints, got {len(modes)}")
        if any(mode != self._idle_mode for mode in modes):
            raise Blocked(
                f"{side} idle mode readback mismatch: {[ _mode_name(mode) for mode in modes ]}"
            )
        return [_mode_name(mode) for mode in modes]

    def _assert_stage_a_idle_modes(self, side: str) -> list[str]:
        try:
            modes = list(self.drivers[side].get_modes())
        except Exception as error:
            raise InspectionFailure("STAGE_A_MODE_READ_FAILED", f"{side}: {error}") from error
        if len(modes) != 7 or any(mode != self._idle_mode for mode in modes):
            raise InspectionFailure(
                "BLOCKED_STAGE_A_IDLE_READBACK",
                f"{side}: expected 7/7 idle, got {[ _mode_name(mode) for mode in modes ]}",
            )
        return [_mode_name(mode) for mode in modes]

    def read_stage_a_idle_modes(self) -> dict[str, list[str]]:
        if not self.is_connected():
            raise InspectionFailure(
                "STAGE_A_MODE_READ_FAILED", "both followers are not configured"
            )
        return {
            side: self._assert_stage_a_idle_modes(side) for side in FOLLOWER_ORDER
        }

    def connect_idle_without_home(
        self,
        authorization: StageAInspectionAuthorization | CharacterizationAuthorization,
    ) -> dict[str, Any]:
        """Configure both follower transports in idle; never send a position target."""
        if not authorization.connection_allowed:
            raise Blocked(
                "LOW-LEVEL CONNECTION REFUSED: " + "; ".join(authorization.reasons)
            )
        if self.drivers:
            raise Blocked("low-level backend was already constructed")
        stage_a = authorization.risk_class is RISK_CLASS_A
        if stage_a:
            if not isinstance(authorization, StageAInspectionAuthorization):
                raise Blocked("Stage A requires StageAInspectionAuthorization")
            if self.active_risk_class is None:
                self.begin_stage_a_inspection()
            elif self.active_risk_class is not RISK_CLASS_A:
                raise Blocked("Stage A risk-class mismatch")
            self.assert_stage_a_position_command_invariant()
        else:
            if authorization.stage not in {"B", "C", "D"}:
                raise Blocked("motion authorization may only configure Stage B/C/D")
            self.active_risk_class = RISK_CLASS_MOTION
        if self.config is None:
            self.build_config()
        if any(
            item is None
            for item in (
                self._driver_factory,
                self._idle_mode,
                self._position_mode,
                self._model_mapping,
            )
        ):
            self._resolve_vendor()
        if tuple(self.config.follower_arms.keys()) != FOLLOWER_ORDER:
            raise Blocked("BLOCKED_FOLLOWER_ORDER before low-level configure")

        records: dict[str, Any] = {}
        try:
            for side in FOLLOWER_ORDER:
                arm = self.config.follower_arms[side]
                if arm.model not in self._model_mapping:
                    raise Blocked(f"unsupported installed Trossen model: {arm.model}")
                model, end_effector = self._model_mapping[arm.model]
                raw_driver = self._driver_factory(side)
                driver = (
                    StageADriverGuard(raw_driver, self._stage_a_audit, side)
                    if stage_a
                    else raw_driver
                )
                self.drivers[side] = driver
                started = time.monotonic_ns()
                # clear_error=False is deliberate: connection must not silently
                # clear/recover a controller fault.
                try:
                    driver.configure(model, end_effector, arm.ip, False)
                except Exception as error:
                    if stage_a:
                        raise InspectionFailure(
                            f"STAGE_A_{side.upper()}_CONFIGURE_FAILED", str(error)
                        ) from error
                    raise
                self.configured_sides.append(side)
                try:
                    if int(driver.get_num_joints()) != 7:
                        raise Blocked(f"{side} follower reports a non-7D joint count")
                except Exception as error:
                    if stage_a:
                        raise InspectionFailure(
                            f"STAGE_A_{side.upper()}_CONFIGURE_FAILED", str(error)
                        ) from error
                    raise
                modes = (
                    self._assert_stage_a_idle_modes(side)
                    if stage_a
                    else self._assert_idle_modes(side)
                )
                records[side] = {
                    "configure_started_host_monotonic_ns": started,
                    "configure_completed_host_monotonic_ns": time.monotonic_ns(),
                    "clear_error": False,
                    "mode_readback": modes,
                    "home_command_sent": False,
                    "position_target_sent": False,
                }
        except Exception:
            # A partially configured pair must still attempt idle on every
            # instantiated side.  Transport cleanup is intentionally not
            # automatic because it is a separate operator decision.
            self.brake_abort("LOW_LEVEL_CONNECT_FAILURE")
            raise
        self.command_loop_stopped = True
        if stage_a:
            self.assert_stage_a_position_command_invariant()
        return {
            "status": self.CONNECTION_CLASSIFICATION,
            "risk_class": authorization.risk_class.value,
            "followers": records,
            "home_command_sent": False,
            "goal_position_sent": False,
            "physical_motion_free_claimed": False,
            "position_command_audit": (
                self.stage_a_position_command_audit() if stage_a else None
            ),
        }

    def connect_stage_a_idle_without_home(
        self, authorization: StageAInspectionAuthorization
    ) -> dict[str, Any]:
        if authorization.risk_class is not RISK_CLASS_A:
            raise Blocked("Stage A low-level connect requires READ_ONLY_IDLE_INSPECTION")
        return self.connect_idle_without_home(authorization)

    def is_connected(self) -> bool:
        return tuple(self.configured_sides) == FOLLOWER_ORDER

    def read_state(self) -> StateSample:
        if not self.is_connected():
            raise Blocked("low-level backend is not configured for both followers")
        parts = []
        for side in FOLLOWER_ORDER:
            part = np.asarray(self.drivers[side].get_all_positions())
            if part.shape != (7,) or not np.issubdtype(part.dtype, np.number):
                raise Blocked(f"{side} low-level state must be numeric shape (7,)")
            if not np.isfinite(part).all():
                raise Blocked(f"{side} low-level state contains NaN/Inf")
            parts.append(part.astype(np.float64, copy=True))
        return StateSample(validate_action14(np.concatenate(parts), label="low-level state"), time.monotonic_ns())

    def read_controller_limits(self) -> dict[str, list[dict[str, float]]]:
        if not self.is_connected():
            raise Blocked("low-level backend is not configured for both followers")
        result = {
            side: [
                serialize_controller_limits(item)
                for item in self.drivers[side].get_joint_limits()
            ]
            for side in FOLLOWER_ORDER
        }
        return validate_controller_limits(result)

    def read_controller_error_information(self) -> dict[str, str]:
        if not self.is_connected():
            raise Blocked("low-level backend is not configured for both followers")
        # The installed API returns an opaque string.  No undocumented parser is
        # used to decide whether the string means a fault.
        return {
            side: str(self.drivers[side].get_error_information())
            for side in FOLLOWER_ORDER
        }

    def prepare_characterization_motion(
        self,
        *,
        active_sides: tuple[str, ...],
        joint_indices: tuple[int, ...],
        current_state14: Any,
        authorization: CharacterizationAuthorization,
        goal_time_sec: float,
    ) -> None:
        """Enter position mode and seed it with the measured current state.

        This is a future real-motion operation and is unreachable unless the
        characterization authorization is complete.
        """
        if self.active_risk_class is RISK_CLASS_A:
            self._stage_a_audit["position_commands"] += 1
            self._stage_a_audit["violations"].append(
                {"side": "backend", "method": "prepare_characterization_motion"}
            )
            raise Blocked(
                "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION: "
                "prepare_characterization_motion"
            )
        if not authorization.motion_allowed or authorization.stage == "A":
            raise Blocked("CHARACTERIZATION MOTION REFUSED")
        if not active_sides or any(side not in FOLLOWER_ORDER for side in active_sides):
            raise Blocked("active_sides must be a non-empty subset of left/right")
        if not joint_indices or any(index < 0 or index > 5 for index in joint_indices):
            raise Blocked("characterization joint indices must select arm joints 0..5")
        if authorization.stage == "B" and (
            len(active_sides) != 1 or len(joint_indices) != 1
        ):
            raise Blocked("Stage B must configure exactly one follower arm joint")
        if not np.isfinite(goal_time_sec) or goal_time_sec <= 0:
            raise Blocked("characterization goal_time_sec must be positive")
        current = validate_action14(current_state14, label="characterization current state")
        for side in active_sides:
            offset = 0 if side == "left" else 7
            driver = self.drivers[side]
            if authorization.stage == "B":
                joint = joint_indices[0]
                modes = [self._idle_mode] * 7
                modes[joint] = self._position_mode
                driver.set_joint_modes(modes)
                driver.set_joint_position(
                    joint, float(current[offset + joint]), goal_time_sec, False
                )
            else:
                # Arm-only APIs leave the gripper in idle/brake and never send
                # a gripper position target.
                driver.set_arm_modes(self._position_mode)
                driver.set_arm_positions(
                    current[offset : offset + 6].tolist(), goal_time_sec, False
                )
        self.motion_sides = tuple(active_sides)
        self.motion_joint_indices = tuple(joint_indices)
        self.command_loop_stopped = False

    def send_characterization_target(
        self,
        target14: Any,
        *,
        active_sides: tuple[str, ...],
        joint_indices: tuple[int, ...],
        controller_limits: dict[str, list[dict[str, float]]],
        held_grippers: tuple[float, float],
        goal_time_sec: float,
    ) -> np.ndarray:
        if self.active_risk_class is RISK_CLASS_A:
            self._stage_a_audit["position_commands"] += 1
            self._stage_a_audit["violations"].append(
                {"side": "backend", "method": "send_characterization_target"}
            )
            raise Blocked(
                "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION: "
                "send_characterization_target"
            )
        if (
            self.command_loop_stopped
            or tuple(active_sides) != self.motion_sides
            or tuple(joint_indices) != self.motion_joint_indices
        ):
            raise Blocked("characterization command loop is not active for requested sides")
        target = validate_action14(target14, label="characterization target")
        limits = validate_controller_limits(controller_limits)
        if not np.array_equal(target[[6, 13]], np.asarray(held_grippers, dtype=float)):
            raise Blocked("characterization grippers must remain exactly at measured positions")
        for side in active_sides:
            offset = 0 if side == "left" else 7
            for joint in joint_indices:
                value = float(target[offset + joint])
                lower = float(limits[side][joint]["position_min"])
                upper = float(limits[side][joint]["position_max"])
                if not lower <= value <= upper:
                    raise Blocked(
                        f"characterization {side} joint_{joint} target outside controller limits"
                    )
            driver = self.drivers[side]
            if len(active_sides) == 1 and len(joint_indices) == 1:
                joint = joint_indices[0]
                driver.set_joint_position(
                    joint, float(target[offset + joint]), goal_time_sec, False
                )
            else:
                driver.set_arm_positions(
                    target[offset : offset + 6].tolist(), goal_time_sec, False
                )
        return target

    def stop_command_loop(self) -> None:
        self.command_loop_stopped = True

    def brake_abort(self, reason: str) -> dict[str, Any]:
        """Request and read back idle on both sides; never home/sleep/disconnect."""
        self.stop_command_loop()
        result: dict[str, Any] = {
            "status": "SOFTWARE_BRAKE_ABORT_REQUESTED",
            "reason": str(reason),
            "requested_host_monotonic_ns": time.monotonic_ns(),
            "completed_host_monotonic_ns": None,
            "sides": {},
            "home_called": False,
            "sleep_called": False,
            "disconnect_called": False,
            "physical_estop": False,
            "warning": (
                "BRAKE_ABORT IS SOFTWARE-COMMANDED, DEPENDS ON CONTROLLER/NETWORK "
                "COMMUNICATION, AND IS NOT AN INDEPENDENT PHYSICAL E-STOP"
            ),
        }
        confirmed = 0
        for side in FOLLOWER_ORDER:
            side_result = {
                "idle_requested": False,
                "idle_confirmed": False,
                "api_readback": None,
                "error": None,
            }
            result["sides"][side] = side_result
            try:
                side_result["idle_requested"] = True
                driver = self.drivers[side]
                driver.set_all_modes(self._idle_mode)
                side_result["api_readback"] = self._assert_idle_modes(side)
                side_result["idle_confirmed"] = True
                confirmed += 1
            except Exception as error:  # Both sides must be attempted independently.
                side_result["error"] = f"{type(error).__name__}: {error}"
        result["completed_host_monotonic_ns"] = time.monotonic_ns()
        result["status"] = (
            "SOFTWARE_BRAKE_ABORT_CONFIRMED"
            if confirmed == 2
            else "SOFTWARE_BRAKE_ABORT_PARTIAL"
            if confirmed == 1
            else "SOFTWARE_BRAKE_ABORT_FAILED"
        )
        self.last_brake_result = result
        self.brake_history.append(result)
        return result

    def close_transport_without_home(
        self,
        authorization: StageAInspectionAuthorization | CharacterizationAuthorization,
    ) -> dict[str, Any]:
        """Explicitly cleanup transports in idle; never call LeRobot disconnect."""
        if not authorization.close_transport_allowed:
            raise Blocked("LOW-LEVEL TRANSPORT CLOSE REFUSED")
        if (
            self.last_brake_result is None
            or self.last_brake_result["status"] != "SOFTWARE_BRAKE_ABORT_CONFIRMED"
        ):
            raise Blocked("transport close requires confirmed two-sided idle API readback")
        results: dict[str, Any] = {}
        for side in FOLLOWER_ORDER:
            try:
                self.drivers[side].cleanup(False)
                results[side] = {"cleanup_called": True, "reboot_controller": False, "error": None}
            except Exception as error:
                results[side] = {
                    "cleanup_called": True,
                    "reboot_controller": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        self.configured_sides = []
        record = {
            "status": (
                "NO_HOME_TRANSPORT_CLOSED"
                if all(item["error"] is None for item in results.values())
                else "NO_HOME_TRANSPORT_CLOSE_PARTIAL"
            ),
            "sides": results,
            "high_level_disconnect_called": False,
            "home_called": False,
            "sleep_called": False,
        }
        self.transport_close_history.append(record)
        return record


class MockTrossenLowLevelDriver:
    """In-memory test double for the installed seven-joint Trossen binding."""

    def __init__(
        self,
        state7: Any,
        limits7: list[dict[str, float]],
        *,
        idle_mode: Any = "idle",
        fail_idle: bool = False,
    ):
        self.state = np.asarray(state7, dtype=float).copy()
        self.limits = limits7
        self.idle_mode = idle_mode
        self.fail_idle = fail_idle
        self.modes = [idle_mode] * 7
        self.configured = False
        self.calls: list[tuple[Any, ...]] = []
        self.error_information = "MOCK_NO_CONTROLLER_ERROR"

    def configure(self, model: Any, end_effector: Any, ip: str, clear_error: bool) -> None:
        self.calls.append(("configure", model, end_effector, ip, clear_error))
        self.configured = True
        self.modes = [self.idle_mode] * 7

    def get_num_joints(self) -> int:
        return 7

    def get_modes(self) -> list[Any]:
        self.calls.append(("get_modes",))
        return list(self.modes)

    def set_all_modes(self, mode: Any) -> None:
        self.calls.append(("set_all_modes", mode))
        if self.fail_idle and mode == self.idle_mode:
            raise RuntimeError("mock idle failure")
        self.modes = [mode] * 7

    def set_arm_modes(self, mode: Any) -> None:
        self.calls.append(("set_arm_modes", mode))
        self.modes[:6] = [mode] * 6

    def set_joint_modes(self, modes: list[Any]) -> None:
        self.calls.append(("set_joint_modes", list(modes)))
        if len(modes) != 7:
            raise RuntimeError("mock joint mode vector must have length 7")
        self.modes = list(modes)

    def get_all_positions(self) -> list[float]:
        self.calls.append(("get_all_positions",))
        return self.state.tolist()

    def get_joint_limits(self) -> list[Any]:
        self.calls.append(("get_joint_limits",))
        return [type("MockLimit", (), item)() for item in self.limits]

    def get_error_information(self) -> str:
        self.calls.append(("get_error_information",))
        return self.error_information

    def set_all_positions(self, values: list[float], goal_time: float, blocking: bool) -> None:
        self.calls.append(("set_all_positions", list(values), goal_time, blocking))
        self.state = np.asarray(values, dtype=float).copy()

    def set_arm_positions(self, values: list[float], goal_time: float, blocking: bool) -> None:
        self.calls.append(("set_arm_positions", list(values), goal_time, blocking))
        if len(values) != 6:
            raise RuntimeError("mock arm target must have length 6")
        self.state[:6] = np.asarray(values, dtype=float)

    def set_joint_position(
        self, joint_index: int, value: float, goal_time: float, blocking: bool
    ) -> None:
        self.calls.append(
            ("set_joint_position", int(joint_index), float(value), goal_time, blocking)
        )
        self.state[int(joint_index)] = float(value)

    def cleanup(self, reboot_controller: bool = False) -> None:
        self.calls.append(("cleanup", reboot_controller))
        self.configured = False


class AbortController:
    """Signal-safe request flag; it never calls a controller from a handler."""

    KEY_ABORTS = {"q": "Q_KEY", "Q": "Q_KEY", " ": "SPACE_KEY"}

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns):
        self._event = threading.Event()
        self._clock_ns = clock_ns
        self.reason: str | None = None
        self.source: str | None = None
        self.requested_host_monotonic_ns: int | None = None
        self._previous_handlers: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self, reason: str, source: str) -> None:
        if not self._event.is_set():
            self.reason = str(reason)
            self.source = str(source)
            self.requested_host_monotonic_ns = int(self._clock_ns())
            self._event.set()

    def request_from_key(self, key: str) -> bool:
        if key not in self.KEY_ABORTS:
            return False
        self.request("USER_ABORT", self.KEY_ABORTS[key])
        return True

    def signal_handler(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        self.request("USER_ABORT", name)

    def install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self.signal_handler)

    def restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()


class ForegroundAbortInput:
    """Non-blocking terminal Q/SPACE poller used only by a foreground loop."""

    def __init__(self, abort_controller: AbortController, stream: Any = sys.stdin):
        self.abort_controller = abort_controller
        self.stream = stream
        self.fd: int | None = None
        self.previous_terminal_settings: Any | None = None

    def start(self) -> None:
        if not hasattr(self.stream, "isatty") or not self.stream.isatty():
            raise Blocked("foreground Q/SPACE abort requires an interactive TTY")
        self.fd = self.stream.fileno()
        self.previous_terminal_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self) -> str | None:
        if self.fd is None:
            return None
        ready, _, _ = select.select([self.stream], [], [], 0.0)
        if not ready:
            return None
        key = self.stream.read(1)
        self.abort_controller.request_from_key(key)
        return key

    def close(self) -> None:
        if self.fd is not None and self.previous_terminal_settings is not None:
            termios.tcsetattr(
                self.fd, termios.TCSADRAIN, self.previous_terminal_settings
            )
        self.fd = None
        self.previous_terminal_settings = None


class WatchdogAbort(Blocked):
    """Hard runtime condition that must stop commands before brake_abort()."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class RuntimeWatchdog:
    ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    GRIPPER_INDICES = np.asarray([6, 13])

    def __init__(
        self,
        config: WatchdogConfig,
        controller_limits: dict[str, list[dict[str, float]]],
    ):
        self.config = config
        self.limits = validate_controller_limits(controller_limits)
        self.arm_tracking_elapsed = 0.0
        self.gripper_tracking_elapsed = 0.0

    @staticmethod
    def _above(values: np.ndarray, threshold: float | None) -> bool:
        return threshold is not None and bool(np.any(values > float(threshold)))

    def check_command_before_send(
        self,
        *,
        command14: Any,
        previous_command14: Any | None,
        dt_sec: float,
        commanded_indices: tuple[int, ...],
        follower_order: tuple[str, ...] = FOLLOWER_ORDER,
        abort_controller: AbortController | None = None,
    ) -> None:
        """Reject hard command faults before any vendor position API is called."""
        if abort_controller is not None and abort_controller.requested:
            raise WatchdogAbort(
                "MANUAL_ABORT", abort_controller.source or "unknown source"
            )
        if follower_order != FOLLOWER_ORDER:
            raise WatchdogAbort("UNEXPECTED_FOLLOWER_ORDER", str(follower_order))
        try:
            command = validate_action14(command14, label="pre-send watchdog command")
        except Blocked as error:
            raise WatchdogAbort("NONFINITE_OR_INVALID_COMMAND_STATE", str(error)) from error
        commanded = tuple(commanded_indices)
        if not commanded or len(set(commanded)) != len(commanded) or any(
            index < 0 or index >= 14 for index in commanded
        ):
            raise WatchdogAbort("INVALID_COMMANDED_INDEX_SET", str(commanded))
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            raise WatchdogAbort("INVALID_LOOP_PERIOD", str(dt_sec))
        for index in commanded:
            side_index, joint_index = divmod(index, 7)
            side = FOLLOWER_ORDER[side_index]
            item = self.limits[side][joint_index]
            if not float(item["position_min"]) <= command[index] <= float(
                item["position_max"]
            ):
                raise WatchdogAbort(
                    "COMMAND_OUTSIDE_CONTROLLER_LIMIT",
                    f"{side} joint_{joint_index}={command[index]}",
                )
        if previous_command14 is None:
            return
        previous = validate_action14(
            previous_command14, label="previous pre-send watchdog command"
        )
        step = np.abs(command - previous)
        velocity = step / dt_sec
        for index in commanded:
            side_index, joint_index = divmod(index, 7)
            side = FOLLOWER_ORDER[side_index]
            velocity_max = self.limits[side][joint_index].get("velocity_max")
            if velocity_max is not None and velocity[index] > float(velocity_max):
                raise WatchdogAbort(
                    "COMMAND_VELOCITY_CONTROLLER_LIMIT",
                    f"{side} joint_{joint_index} velocity={velocity[index]} "
                    f"limit={velocity_max}",
                )
        config = self.config
        checks = (
            ("COMMAND_STEP_ARM", step[self.ARM_INDICES], config.max_command_step_arm_rad),
            (
                "COMMAND_STEP_GRIPPER",
                step[self.GRIPPER_INDICES],
                config.max_command_step_gripper_m,
            ),
            (
                "COMMAND_VELOCITY_ARM",
                velocity[self.ARM_INDICES],
                config.max_command_velocity_arm_rad_s,
            ),
            (
                "COMMAND_VELOCITY_GRIPPER",
                velocity[self.GRIPPER_INDICES],
                config.max_command_velocity_gripper_m_s,
            ),
        )
        for code, measured, allowed in checks:
            if self._above(measured, allowed):
                raise WatchdogAbort(
                    code, f"max={float(measured.max())} allowed={allowed}"
                )

    def check(
        self,
        *,
        command14: Any,
        actual14: Any,
        previous_command14: Any | None,
        previous_actual14: Any | None = None,
        dt_sec: float,
        state_age_sec: float,
        state_read_duration_sec: float,
        command_call_duration_sec: float,
        loop_overrun_sec: float,
        follower_order: tuple[str, ...] = FOLLOWER_ORDER,
        abort_controller: AbortController | None = None,
        controller_fault: bool = False,
        controller_error_information: dict[str, str] | None = None,
        commanded_indices: tuple[int, ...] | None = None,
    ) -> None:
        if abort_controller is not None and abort_controller.requested:
            raise WatchdogAbort("MANUAL_ABORT", abort_controller.source or "unknown source")
        if follower_order != FOLLOWER_ORDER:
            raise WatchdogAbort("UNEXPECTED_FOLLOWER_ORDER", str(follower_order))
        if controller_fault:
            raise WatchdogAbort(
                "CONTROLLER_ERROR",
                json.dumps(controller_error_information or {}, sort_keys=True),
            )
        try:
            command = validate_action14(command14, label="watchdog command")
            actual = validate_action14(actual14, label="watchdog state")
        except Blocked as error:
            raise WatchdogAbort("NONFINITE_OR_INVALID_COMMAND_STATE", str(error)) from error
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            raise WatchdogAbort("INVALID_LOOP_PERIOD", str(dt_sec))
        commanded = (
            tuple(range(14)) if commanded_indices is None else tuple(commanded_indices)
        )
        if not commanded or len(set(commanded)) != len(commanded) or any(
            index < 0 or index >= 14 for index in commanded
        ):
            raise WatchdogAbort("INVALID_COMMANDED_INDEX_SET", str(commanded))
        commanded_set = set(commanded)

        config = self.config
        timed = (
            ("STATE_STALE", state_age_sec, config.max_state_age_sec),
            ("STATE_READ_TIMEOUT", state_read_duration_sec, config.max_state_read_duration_sec),
            ("COMMAND_CALL_TIMEOUT", command_call_duration_sec, config.max_command_call_duration_sec),
            ("LOOP_OVERRUN", loop_overrun_sec, config.max_loop_overrun_sec),
        )
        for code, measured, allowed in timed:
            if allowed is not None and measured > allowed:
                raise WatchdogAbort(code, f"measured={measured} allowed={allowed}")

        for side_index, side in enumerate(FOLLOWER_ORDER):
            base = side_index * 7
            for joint_index, item in enumerate(self.limits[side]):
                index = base + joint_index
                lower = float(item["position_min"])
                upper = float(item["position_max"])
                controller_tolerance = float(item.get("position_tolerance", 0.0))
                reviewed_tolerance = config.controller_position_tolerance
                tolerance = (
                    controller_tolerance
                    if reviewed_tolerance is None
                    else min(controller_tolerance, float(reviewed_tolerance))
                )
                if index in commanded_set and not lower <= command[index] <= upper:
                    raise WatchdogAbort(
                        "COMMAND_OUTSIDE_CONTROLLER_LIMIT",
                        f"{side} joint_{joint_index}={command[index]}",
                    )
                if not lower - tolerance <= actual[index] <= upper + tolerance:
                    raise WatchdogAbort(
                        "ACTUAL_OUTSIDE_CONTROLLER_LIMIT",
                        f"{side} joint_{joint_index}={actual[index]}",
                    )

        if previous_command14 is not None:
            previous = validate_action14(previous_command14, label="previous watchdog command")
            step = np.abs(command - previous)
            velocity = step / dt_sec
            for side_index, side in enumerate(FOLLOWER_ORDER):
                base = side_index * 7
                for joint_index, item in enumerate(self.limits[side]):
                    controller_velocity_max = item.get("velocity_max")
                    if (
                        base + joint_index in commanded_set
                        and
                        controller_velocity_max is not None
                        and velocity[base + joint_index] > float(controller_velocity_max)
                    ):
                        raise WatchdogAbort(
                            "COMMAND_VELOCITY_CONTROLLER_LIMIT",
                            f"{side} joint_{joint_index} velocity={velocity[base + joint_index]} "
                            f"limit={controller_velocity_max}",
                        )
            checks = (
                ("COMMAND_STEP_ARM", step[self.ARM_INDICES], config.max_command_step_arm_rad),
                ("COMMAND_STEP_GRIPPER", step[self.GRIPPER_INDICES], config.max_command_step_gripper_m),
                (
                    "COMMAND_VELOCITY_ARM",
                    velocity[self.ARM_INDICES],
                    config.max_command_velocity_arm_rad_s,
                ),
                (
                    "COMMAND_VELOCITY_GRIPPER",
                    velocity[self.GRIPPER_INDICES],
                    config.max_command_velocity_gripper_m_s,
                ),
            )
            for code, measured, allowed in checks:
                if self._above(measured, allowed):
                    raise WatchdogAbort(code, f"max={float(measured.max())} allowed={allowed}")

        if previous_actual14 is not None:
            previous_actual = validate_action14(
                previous_actual14, label="previous watchdog state"
            )
            actual_velocity = np.abs(actual - previous_actual) / dt_sec
            for side_index, side in enumerate(FOLLOWER_ORDER):
                base = side_index * 7
                for joint_index, item in enumerate(self.limits[side]):
                    controller_velocity_max = item.get("velocity_max")
                    if (
                        controller_velocity_max is not None
                        and actual_velocity[base + joint_index]
                        > float(controller_velocity_max)
                    ):
                        raise WatchdogAbort(
                            "ACTUAL_VELOCITY_CONTROLLER_LIMIT",
                            f"{side} joint_{joint_index} "
                            f"velocity={actual_velocity[base + joint_index]} "
                            f"limit={controller_velocity_max}",
                        )

        error = np.abs(command - actual)
        arm_over = self._above(error[self.ARM_INDICES], config.max_tracking_error_arm_rad)
        grip_over = self._above(
            error[self.GRIPPER_INDICES], config.max_tracking_error_gripper_m
        )
        self.arm_tracking_elapsed = self.arm_tracking_elapsed + dt_sec if arm_over else 0.0
        self.gripper_tracking_elapsed = self.gripper_tracking_elapsed + dt_sec if grip_over else 0.0
        duration = config.tracking_error_duration_sec
        if duration is not None and self.arm_tracking_elapsed >= duration:
            raise WatchdogAbort("PERSISTENT_ARM_TRACKING_ERROR")
        if duration is not None and self.gripper_tracking_elapsed >= duration:
            raise WatchdogAbort("PERSISTENT_GRIPPER_TRACKING_ERROR")


def evaluate_stage_a_authorization(
    *,
    config_valid: bool,
    follower_order_verified: bool,
    workspace_clear_confirmed: bool,
    physical_left_right_verified: bool,
    operator_present_confirmed: bool,
    left_power_switch_reachable: bool,
    right_power_switch_reachable: bool,
    acknowledge_idle_brake_connect_may_move: bool,
    acknowledge_idle_cleanup_command: bool,
    stage_a_confirmation: str | None,
    low_level_no_home_path_verified: bool = LOW_LEVEL_NO_HOME_PATH_VERIFIED,
    mode_idle_semantics_verified: bool = MODE_IDLE_SEMANTICS_VERIFIED,
    no_position_target_allowed: bool = NO_POSITION_TARGET_ALLOWED,
) -> tuple[dict[str, Any], StageAInspectionAuthorization]:
    """Authorize only idle/configure/read/limit Stage A without a motion E-stop gate."""
    states = {
        "CONFIG_VALID": bool(config_valid),
        "LOW_LEVEL_NO_HOME_PATH_VERIFIED": bool(low_level_no_home_path_verified),
        "MODE_IDLE_SEMANTICS_VERIFIED": bool(mode_idle_semantics_verified),
        "FOLLOWER_ORDER_VERIFIED": bool(follower_order_verified),
        "WORKSPACE_CLEAR_CONFIRMED": bool(workspace_clear_confirmed),
        "PHYSICAL_LEFT_RIGHT_VERIFIED": bool(physical_left_right_verified),
        "OPERATOR_PRESENT_CONFIRMED": bool(operator_present_confirmed),
        "LEFT_POWER_SWITCH_REACHABLE": bool(left_power_switch_reachable),
        "RIGHT_POWER_SWITCH_REACHABLE": bool(right_power_switch_reachable),
        "NO_POSITION_TARGET_ALLOWED": bool(no_position_target_allowed),
        "IDLE_BRAKE_CONNECT_MOTION_RISK_ACKNOWLEDGED": bool(
            acknowledge_idle_brake_connect_may_move
        ),
        "IDLE_CLEANUP_COMMAND_ACKNOWLEDGED": bool(
            acknowledge_idle_cleanup_command
        ),
        "STAGE_A_CONFIRMATION_MATCH": stage_a_confirmation == STAGE_A_CONFIRMATION,
    }
    reasons = [
        f"Stage A authorization state false: {name}"
        for name, passed in states.items()
        if not passed
    ]
    allowed = all(states.values())
    authorization = StageAInspectionAuthorization(
        connection_allowed=allowed,
        close_transport_allowed=allowed and acknowledge_idle_cleanup_command,
        reasons=tuple(reasons),
        states=states,
    )
    record = {
        "authorization_level": RISK_CLASS_A.value,
        "risk_class": RISK_CLASS_A.value,
        "status": "STAGE_A_AUTHORIZED" if allowed else "STAGE_A_BLOCKED",
        "connection_allowed": allowed,
        "motion_allowed": False,
        "close_transport_allowed": authorization.close_transport_allowed,
        "states": states,
        "states_not_required": {
            "VERIFIED_OPERATOR_STOP": "NOT_REQUIRED_FOR_STAGE_A",
            "CONTROLLER_LIMITS_VERIFIED": "MEASURED_BY_STAGE_A",
            "GRIPPER_LIMITS_VERIFIED": "MEASURED_BY_STAGE_A",
            "TRACKING_THRESHOLDS": "NOT_APPLICABLE_TO_STAGE_A",
            "HARDWARE_REPLAY_ALLOWED": "NOT_APPLICABLE_TO_STAGE_A",
        },
        "reasons": reasons,
        "power_switch_classification": (
            "MANUAL_POWER_ISOLATION_AVAILABLE_PER_FOLLOWER_NOT_AN_ESTOP"
        ),
        "optimized_action_allowed": False,
        "trajectory_loop_allowed": False,
        "position_target_allowed": False,
        "vla_replay_authorized": False,
    }
    return record, authorization


def evaluate_characterization_authorization(
    *,
    stage: str,
    config_valid: bool,
    follower_order_valid: bool,
    workspace_clear_confirmed: bool,
    physical_left_right_verified: bool,
    stop_verification: dict[str, Any],
    stop_verification_reasons: list[str],
    acknowledge_idle_brake_connect_may_move: bool,
    acknowledge_idle_cleanup_command: bool,
    characterization_confirmation: str | None,
    characterization_config: dict[str, Any] | None = None,
    characterization_config_reasons: list[str] | None = None,
    prior_stage_approved: bool = False,
) -> tuple[dict[str, Any], CharacterizationAuthorization]:
    """Fail-closed motion gate for Stage B/C/D; Stage A is not accepted here."""
    stage = stage.upper()
    if stage not in {"B", "C", "D"}:
        raise Blocked("motion characterization authorization accepts only Stage B/C/D")
    reviewed = characterization_config or {}
    reviewed_reasons = list(characterization_config_reasons or [])
    watchdog_mapping = reviewed.get("watchdog")
    watchdog_resolved = False
    if isinstance(watchdog_mapping, dict):
        try:
            watchdog_resolved = not WatchdogConfig.from_mapping(
                watchdog_mapping
            ).unresolved_fields()
        except (Blocked, TypeError, ValueError):
            watchdog_resolved = False
    connection_states = {
        "CONFIG_VALID": config_valid,
        "FOLLOWER_ORDER_VALID": follower_order_valid,
        "WORKSPACE_CLEAR_CONFIRMED": workspace_clear_confirmed,
        "PHYSICAL_LEFT_RIGHT_VERIFIED": physical_left_right_verified,
        "VERIFIED_OPERATOR_STOP": verified_operator_stop(
            stop_verification, stop_verification_reasons
        ),
        "LOW_LEVEL_NO_HOME_PATH_VERIFIED": LOW_LEVEL_NO_HOME_PATH_VERIFIED,
        "PHYSICAL_NO_MOTION_CONNECT_CLAIMED": False,
        "IDLE_BRAKE_CONNECT_MOTION_RISK_ACKNOWLEDGED": (
            acknowledge_idle_brake_connect_may_move
        ),
        "IDLE_CLEANUP_COMMAND_ACKNOWLEDGED": acknowledge_idle_cleanup_command,
        "CHARACTERIZATION_CONFIRMATION_MATCH": (
            characterization_confirmation == CHARACTERIZATION_CONFIRMATION
        ),
    }
    # PHYSICAL_NO_MOTION_CONNECT_CLAIMED is an informational false state, not
    # an authorization input: the operator explicitly authorizes the verified
    # no-home idle-brake path, whose physical effect remains inconclusive.
    connection_required = tuple(
        key
        for key in connection_states
        if key != "PHYSICAL_NO_MOTION_CONNECT_CLAIMED"
    )
    base_connection_allowed = all(
        connection_states[key] for key in connection_required
    )

    motion_states = {
        "MOTION_STAGE_REQUESTED": stage in {"B", "C", "D"},
        "CHARACTERIZATION_CONFIG_REVIEWED": reviewed.get("status") == "REVIEWED",
        "CONTROLLER_LIMITS_VERIFIED": reviewed.get("controller_limits_verified") is True,
        "WATCHDOG_THRESHOLDS_REVIEWED": watchdog_resolved,
        "PRIOR_STAGE_APPROVED": prior_stage_approved,
        "HARDWARE_REPLAY_DISABLED": reviewed.get("hardware_replay_allowed") is not True,
    }
    motion_prerequisites_valid = not reviewed_reasons and all(motion_states.values())
    connection_allowed = base_connection_allowed and motion_prerequisites_valid
    motion_allowed = connection_allowed
    close_allowed = connection_allowed
    states = {**connection_states, **motion_states}
    failed_connection = [key for key in connection_required if not connection_states[key]]
    failed_motion = [key for key, passed in motion_states.items() if not passed]
    reasons = [
        *stop_verification_reasons,
        *(f"characterization connection state false: {name}" for name in failed_connection),
    ]
    reasons.extend(reviewed_reasons)
    reasons.extend(
        f"characterization motion state false: {name}" for name in failed_motion
    )
    status = (
        "CHARACTERIZATION_MOTION_AUTHORIZED"
        if motion_allowed
        else "BLOCKED_NO_VERIFIED_OPERATOR_STOP"
        if not connection_states["VERIFIED_OPERATOR_STOP"]
        else "CHARACTERIZATION_BLOCKED"
    )
    record = {
        "authorization_level": RISK_CLASS_MOTION.value,
        "risk_class": RISK_CLASS_MOTION.value,
        "status": status,
        "stage": stage,
        "connection_allowed": connection_allowed,
        "motion_allowed": motion_allowed,
        "close_transport_allowed": close_allowed,
        "states": states,
        "reasons": reasons,
        "vla_replay_authorized": False,
        "optimized_action_allowed": False,
        "connection_classification": (
            StationaryAlohaLowLevelBackend.CONNECTION_CLASSIFICATION
        ),
    }
    return record, CharacterizationAuthorization(
        connection_allowed=connection_allowed,
        motion_allowed=motion_allowed,
        close_transport_allowed=close_allowed,
        stage=stage,
        reasons=tuple(reasons),
        states=states,
    )


def generate_characterization_trajectory(
    current_state14: Any,
    *,
    stage: str,
    active_side: str,
    joint_indices: tuple[int, ...],
    amplitude_rad: float,
    duration_sec: float,
    rate_hz: float = SOURCE_FPS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Generate a tiny smooth test trajectory; never reads a source action."""
    current = validate_action14(current_state14, label="characterization initial state")
    stage = stage.upper()
    if stage not in {"B", "C", "D"}:
        raise Blocked("a motion trajectory is valid only for Stage B/C/D")
    if active_side not in FOLLOWER_ORDER:
        raise Blocked("active_side must be left or right")
    if not joint_indices or any(index < 0 or index > 5 for index in joint_indices):
        raise Blocked("characterization joint indices must be arm joints 0..5")
    if len(set(joint_indices)) != len(joint_indices):
        raise Blocked("characterization joint indices must be unique")
    if stage == "B" and len(joint_indices) != 1:
        raise Blocked("Stage B requires exactly one arm joint")
    if stage == "C" and len(joint_indices) < 2:
        raise Blocked("Stage C requires at least two arm joints")
    if not np.isfinite(amplitude_rad) or amplitude_rad <= 0:
        raise Blocked("characterization amplitude must be explicitly positive")
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        raise Blocked("characterization duration must be explicitly positive")
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise Blocked("characterization rate must be positive")
    samples = max(3, int(round(duration_sec * rate_hz)) + 1)
    phase = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    # Raised cosine: zero displacement and zero slope at both endpoints.
    displacement = 0.5 * float(amplitude_rad) * (1.0 - np.cos(2.0 * np.pi * phase))
    trajectory = np.repeat(current[None, :], samples, axis=0)
    sides = FOLLOWER_ORDER if stage == "D" else (active_side,)
    for side in sides:
        base = 0 if side == "left" else 7
        for joint in joint_indices:
            trajectory[:, base + joint] += displacement
    trajectory[0] = current
    trajectory[-1] = current
    if not np.array_equal(trajectory[:, [6, 13]], np.repeat(current[[6, 13]][None, :], samples, axis=0)):
        raise Blocked("generated characterization trajectory changed a gripper")
    return trajectory, tuple(sides)


def validate_characterization_trajectory_limits(
    trajectory: np.ndarray,
    controller_limits: dict[str, list[dict[str, float]]],
    *,
    minimum_margin: float,
    commanded_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    array = np.asarray(trajectory, dtype=float)
    if array.ndim != 2 or array.shape[1] != 14 or not np.isfinite(array).all():
        raise Blocked("characterization trajectory must be finite Nx14")
    if not np.isfinite(minimum_margin) or minimum_margin < 0:
        raise Blocked("minimum controller-limit margin must be non-negative")
    limits = validate_controller_limits(controller_limits)
    commanded = set(range(14) if commanded_indices is None else commanded_indices)
    if not commanded or any(index < 0 or index >= 14 for index in commanded):
        raise Blocked("characterization commanded indices are invalid")
    margins: dict[str, Any] = {}
    for side_index, side in enumerate(FOLLOWER_ORDER):
        base = side_index * 7
        margins[side] = []
        for joint, limit in enumerate(limits[side]):
            lower_margin = float(array[:, base + joint].min() - limit["position_min"])
            upper_margin = float(limit["position_max"] - array[:, base + joint].max())
            hardware_commanded = base + joint in commanded
            if hardware_commanded and (
                lower_margin < minimum_margin or upper_margin < minimum_margin
            ):
                raise Blocked(
                    f"characterization {side} joint_{joint} lacks reviewed controller-limit margin"
                )
            margins[side].append(
                {
                    "joint": joint,
                    "source_index": base + joint,
                    "hardware_commanded": hardware_commanded,
                    "lower_margin": lower_margin,
                    "upper_margin": upper_margin,
                }
            )
    return {
        "status": "COMMANDED_CHANNELS_WITHIN_CONTROLLER_LIMITS",
        "minimum_required_margin": minimum_margin,
        "commanded_indices": sorted(commanded),
        "margins": margins,
    }


def compare_source_to_controller_limits(
    source: np.ndarray,
    controller_limits: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    """Complete 990x14 source audit using controller-reported limits."""
    source_array = np.asarray(source)
    if source_array.shape != (990, 14) or not np.isfinite(source_array).all():
        raise Blocked("source limit audit requires finite (990,14)")
    limits = validate_controller_limits(controller_limits)
    result: dict[str, Any] = {
        "status": "PASS",
        "source_shape": [990, 14],
        "authority": "CONTROLLER_REPORTED_LIMITS",
        "source_modified": False,
        "sides": {},
        "arm_violation_count": 0,
        "gripper_violation_count": 0,
    }
    for side_index, side in enumerate(FOLLOWER_ORDER):
        base = side_index * 7
        joint_results = []
        for joint in range(7):
            values = source_array[:, base + joint].astype(float)
            lower = float(limits[side][joint]["position_min"])
            upper = float(limits[side][joint]["position_max"])
            below = np.flatnonzero(values < lower)
            above = np.flatnonzero(values > upper)
            violation_frames = np.unique(np.concatenate((below, above)))
            amounts = np.maximum(lower - values, values - upper)
            amounts = np.maximum(amounts, 0.0)
            worst = int(np.argmax(amounts)) if violation_frames.size else None
            item = {
                "joint_index": joint,
                "semantic": "gripper" if joint == 6 else f"arm_joint_{joint}",
                "source_min": float(values.min()),
                "source_max": float(values.max()),
                "controller_min": lower,
                "controller_max": upper,
                "minimum_limit_margin": float(values.min() - lower),
                "maximum_limit_margin": float(upper - values.max()),
                "below_minimum_frame_count": int(below.size),
                "above_maximum_frame_count": int(above.size),
                "violating_frame_count": int(violation_frames.size),
                "worst_violating_frame": worst,
                "worst_violation_amount": float(amounts[worst]) if worst is not None else 0.0,
            }
            joint_results.append(item)
            key = "gripper_violation_count" if joint == 6 else "arm_violation_count"
            result[key] += int(violation_frames.size)
        result["sides"][side] = joint_results
    for side in FOLLOWER_ORDER:
        gripper = result["sides"][side][6]
        gripper["classification"] = (
            "SOURCE_GRIPPER_FULLY_WITHIN_CONTROLLER_LIMITS"
            if gripper["violating_frame_count"] == 0
            else "SOURCE_GRIPPER_OUTSIDE_CONTROLLER_LIMITS"
        )
    if result["arm_violation_count"] or result["gripper_violation_count"]:
        result["status"] = "SOURCE_OUTSIDE_CONTROLLER_LIMITS"
    return result


def source_action_motion_audit(
    source: np.ndarray,
    fps: float = SOURCE_FPS,
    candidate_path: Path = ROOT / "configs/aloha_source_validation_safety.candidate.json",
) -> dict[str, Any]:
    array = np.asarray(source, dtype=float)
    if array.shape != (990, 14) or not np.isfinite(array).all():
        raise Blocked("motion audit requires finite (990,14) source")
    step = np.diff(array, axis=0)
    velocity = step * fps
    acceleration = np.diff(velocity, axis=0) * fps
    channels = []
    for index, (joint, unit) in enumerate(zip(JOINTS, UNITS)):
        channels.append(
            {
                "index": index,
                "joint": joint,
                "unit": unit,
                "source_min": float(array[:, index].min()),
                "source_max": float(array[:, index].max()),
                "max_abs_step": float(np.abs(step[:, index]).max()),
                "max_abs_velocity": float(np.abs(velocity[:, index]).max()),
                "max_abs_acceleration": float(np.abs(acceleration[:, index]).max()),
            }
        )
    result: dict[str, Any] = {
        "status": "DIAGNOSTIC_ONLY_REAL_LIMITS_UNKNOWN",
        "source_shape": [990, 14],
        "fps": fps,
        "channels": channels,
        "source_modified": False,
        "controller_limit_conclusion": "NEEDS_REAL_CONTROLLER_LIMITS",
    }
    candidate_path = Path(candidate_path)
    if candidate_path.exists():
        candidate = json.loads(candidate_path.read_text())
        criteria = candidate.get("criteria", {})
        arm_indices = RuntimeWatchdog.ARM_INDICES
        measured = {
            "max_command_step": float(np.abs(step[:, arm_indices]).max()),
            "max_velocity": float(np.abs(velocity[:, arm_indices]).max()),
            "max_acceleration": float(np.abs(acceleration[:, arm_indices]).max()),
        }
        comparisons = {}
        for name, value in measured.items():
            limit = criteria.get(name, {}).get("candidate_value")
            comparisons[name] = {
                "measured": value,
                "candidate": limit,
                "within_candidate": None if limit is None else bool(value <= float(limit)),
            }
        empirical_arm = criteria.get("arm_position_range", {}).get("candidate_value")
        arm_range_results = []
        if isinstance(empirical_arm, list) and len(empirical_arm) == 12:
            for candidate_index, source_index in enumerate(arm_indices):
                lower, upper = map(float, empirical_arm[candidate_index])
                values = array[:, source_index]
                violating = np.flatnonzero((values < lower) | (values > upper))
                arm_range_results.append(
                    {
                        "source_index": int(source_index),
                        "candidate_min": lower,
                        "candidate_max": upper,
                        "violating_frame_count": int(violating.size),
                        "first_violating_frame": (
                            None if not violating.size else int(violating[0])
                        ),
                    }
                )
        result["candidate_empirical_comparison"] = {
            "path": str(candidate_path.resolve()),
            "candidate_status": candidate.get("status"),
            "hardware_replay_allowed": candidate.get("hardware_replay_allowed"),
            "authoritative_controller_limits": False,
            "requires_human_approval": True,
            "motion_thresholds": comparisons,
            "arm_position_ranges": arm_range_results,
        }
    else:
        result["candidate_empirical_comparison"] = {
            "path": str(candidate_path),
            "status": "NOT_AVAILABLE",
        }
    return result


def _correlation_and_lag(
    command: np.ndarray, actual: np.ndarray, max_lag_samples: int = 30
) -> tuple[float | None, int | None]:
    if len(command) < 3 or np.std(command) == 0 or np.std(actual) == 0:
        return None, None
    best: tuple[float, int] | None = None
    maximum = min(max_lag_samples, len(command) - 2)
    for lag in range(-maximum, maximum + 1):
        if lag > 0:
            left, right = command[:-lag], actual[lag:]
        elif lag < 0:
            left, right = command[-lag:], actual[:lag]
        else:
            left, right = command, actual
        if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(corr) and (best is None or corr > best[0]):
            best = (corr, lag)
    return (None, None) if best is None else best


def compute_tracking_metrics(
    samples: list[dict[str, Any]], rate_hz: float = SOURCE_FPS
) -> dict[str, Any]:
    if not samples:
        return {"status": "NO_MOTION_SAMPLES", "per_joint": []}
    commands = np.asarray([row["command_q"] for row in samples], dtype=float)
    actual = np.asarray([row["actual_q"] for row in samples], dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 14 or actual.shape != commands.shape:
        raise Blocked("tracking samples must contain matching Nx14 command/actual arrays")
    error = commands - actual
    periods = np.asarray([row["loop_period_sec"] for row in samples], dtype=float)
    if len(samples) > 1:
        velocity_dt = periods[1:].copy()
        invalid = ~np.isfinite(velocity_dt) | (velocity_dt <= 0)
        velocity_dt[invalid] = 1.0 / float(rate_hz)
        command_velocity = np.diff(commands, axis=0) / velocity_dt[:, None]
        actual_velocity = np.diff(actual, axis=0) / velocity_dt[:, None]
    else:
        command_velocity = np.zeros((0, 14), dtype=float)
        actual_velocity = np.zeros((0, 14), dtype=float)
    per_joint = []
    for index in range(14):
        corr, lag = _correlation_and_lag(commands[:, index], actual[:, index])
        absolute = np.abs(error[:, index])
        per_joint.append(
            {
                "index": index,
                "joint": JOINTS[index],
                "unit": UNITS[index],
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error[:, index])))),
                "max_absolute_error": float(absolute.max()),
                "p95_absolute_error": float(np.percentile(absolute, 95)),
                "command_actual_correlation": corr,
                "estimated_lag_samples": lag,
                "estimated_lag_sec": None if lag is None else float(lag / rate_hz),
                "peak_absolute_command_velocity": (
                    0.0
                    if not len(command_velocity)
                    else float(np.abs(command_velocity[:, index]).max())
                ),
                "peak_absolute_actual_velocity": (
                    0.0
                    if not len(actual_velocity)
                    else float(np.abs(actual_velocity[:, index]).max())
                ),
            }
        )
    valid_periods = periods[np.isfinite(periods) & (periods > 0)]
    all_abs = np.abs(error)
    arm_ranges = np.ptp(commands[:, RuntimeWatchdog.ARM_INDICES], axis=0)
    if np.max(arm_ranges) > 0:
        dominant_index = int(
            RuntimeWatchdog.ARM_INDICES[int(np.argmax(arm_ranges))]
        )
        aggregate_corr, aggregate_lag = _correlation_and_lag(
            commands[:, dominant_index], actual[:, dominant_index]
        )
    else:
        dominant_index = None
        aggregate_corr, aggregate_lag = None, None
    return {
        "status": "TRACKING_METRICS_COMPUTED_OFFLINE",
        "host_clock_only": True,
        "controller_synchronized": False,
        "sample_count": len(samples),
        "per_joint": per_joint,
        "aggregate": {
            "mae": float(all_abs.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "max_absolute_error": float(all_abs.max()),
            "p95_absolute_error": float(np.percentile(all_abs, 95)),
            "command_actual_correlation": aggregate_corr,
            "lag_reference_joint_index": dominant_index,
            "estimated_lag_samples": aggregate_lag,
            "estimated_lag_sec": (
                None if aggregate_lag is None else float(aggregate_lag / rate_hz)
            ),
            "peak_absolute_command_velocity": (
                0.0 if not len(command_velocity) else float(np.abs(command_velocity).max())
            ),
            "peak_absolute_actual_velocity": (
                0.0 if not len(actual_velocity) else float(np.abs(actual_velocity).max())
            ),
        },
        "loop": {
            "mean_frequency_hz": (
                None if not len(valid_periods) else float(1.0 / valid_periods.mean())
            ),
            "p95_period_sec": (
                None if not len(valid_periods) else float(np.percentile(valid_periods, 95))
            ),
            "max_period_sec": (
                None if not len(valid_periods) else float(valid_periods.max())
            ),
            "overrun_count": int(sum(row["loop_overrun_sec"] > 0 for row in samples)),
        },
    }


def write_tracking_samples_csv(path: Path, samples: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_fields = [
        "sample_index",
        "scheduled_timestamp_ns",
        "command_timestamp_ns",
        "state_timestamp_ns",
        "command_call_duration_sec",
        "state_read_duration_sec",
        "loop_period_sec",
        "loop_overrun_sec",
        "state_age_sec",
    ]
    vector_fields = [
        *(f"command_{index}" for index in range(14)),
        *(f"actual_{index}" for index in range(14)),
        *(f"error_{index}" for index in range(14)),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*scalar_fields, *vector_fields])
        writer.writeheader()
        for row in samples:
            flat = {name: row[name] for name in scalar_fields}
            for prefix, source_name in (
                ("command", "command_q"),
                ("actual", "actual_q"),
                ("error", "error"),
            ):
                flat.update(
                    {f"{prefix}_{index}": value for index, value in enumerate(row[source_name])}
                )
            writer.writerow(flat)
    return path


def load_characterization_config(
    path: Path, requested_stage: str | None = None
) -> tuple[dict[str, Any], list[str]]:
    path = Path(path)
    if not path.exists():
        return {}, [f"reviewed characterization config missing: {path}"]
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"invalid characterization config: {error}"]
    reasons: list[str] = []
    if config.get("status") != "REVIEWED":
        reasons.append("characterization config status is not REVIEWED")
    if config.get("hardware_replay_allowed") is not False:
        reasons.append("characterization config must explicitly set hardware_replay_allowed=false")
    if config.get("controller_limits_verified") is not True:
        reasons.append("characterization controller_limits_verified is not true")
    try:
        validate_controller_limits(config.get("controller_limits"))
    except (Blocked, KeyError, TypeError, ValueError) as error:
        reasons.append(f"characterization controller_limits invalid: {error}")
    try:
        watchdog = WatchdogConfig.from_mapping(config.get("watchdog", {}))
        if watchdog.unresolved_fields():
            reasons.append(
                "characterization watchdog fields unresolved: "
                + ", ".join(watchdog.unresolved_fields())
            )
    except (Blocked, TypeError, ValueError) as error:
        reasons.append(f"characterization watchdog invalid: {error}")
    margin = config.get("minimum_controller_limit_margin")
    try:
        if margin is None or not np.isfinite(float(margin)) or float(margin) < 0:
            raise ValueError
    except (TypeError, ValueError):
        reasons.append("minimum_controller_limit_margin must be reviewed and non-negative")
    stages = config.get("stages")
    if not isinstance(stages, dict):
        reasons.append("characterization stages mapping is missing")
    else:
        for stage in ((requested_stage,) if requested_stage is not None else ("B", "C", "D")):
            item = stages.get(stage)
            if not isinstance(item, dict) or item.get("approved") is not True:
                reasons.append(f"characterization Stage {stage} parameters are not approved")
                continue
            try:
                side = item["active_side"]
                joints = tuple(int(value) for value in item["joint_indices"])
                amplitude = float(item["amplitude_rad"])
                duration = float(item["duration_sec"])
                rate_hz = float(item["rate_hz"])
                if side not in FOLLOWER_ORDER:
                    raise ValueError
                if (
                    not joints
                    or len(set(joints)) != len(joints)
                    or any(index < 0 or index > 5 for index in joints)
                ):
                    raise ValueError
                if stage == "B" and len(joints) != 1:
                    raise ValueError
                if (
                    not np.isfinite([amplitude, duration, rate_hz]).all()
                    or amplitude <= 0
                    or duration <= 0
                    or rate_hz <= 0
                ):
                    raise ValueError
                if stage == "B" and not np.isclose(
                    rate_hz, STAGE_B_RATE_HZ, rtol=0.0, atol=0.0
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                reasons.append(f"characterization Stage {stage} parameters are invalid")
    return config, reasons


def characterization_goal_time_from_config(
    backend: StationaryAlohaLowLevelBackend, active_sides: tuple[str, ...]
) -> float:
    if backend.config is None:
        raise Blocked("persistent config must be loaded before deriving goal_time")
    values = [
        float(backend.config.follower_arms[side].min_time_to_move_multiplier)
        / SOURCE_FPS
        for side in active_sides
    ]
    if not values or not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise Blocked("persistent min_time_to_move_multiplier is invalid")
    if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
        raise Blocked("active followers have different persistent goal_time settings")
    return values[0]


def recommend_first_stage_b_config(
    state14: Any,
    controller_limits: dict[str, list[dict[str, float]]],
    *,
    stage_a_authority: dict[str, Any],
    source: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a deliberately unapproved first-test proposal from measured limits."""
    state = validate_action14(state14, label="Stage B recommendation state")
    limits = validate_controller_limits(controller_limits)
    side = STAGE_B_RECOMMENDED_SIDE
    joint = STAGE_B_RECOMMENDED_JOINT
    source_index = joint if side == "left" else 7 + joint
    lower = float(limits[side][joint]["position_min"])
    upper = float(limits[side][joint]["position_max"])
    current = float(state[source_index])
    amplitude = STAGE_B_RECOMMENDED_AMPLITUDE_RAD
    duration = STAGE_B_RECOMMENDED_DURATION_SEC
    proposed_min = current
    proposed_max = current + amplitude
    if proposed_min < lower or proposed_max > upper:
        raise Blocked("recommended Stage B candidate is outside measured controller limits")
    velocity_max = float(limits[side][joint]["velocity_max"])
    peak_velocity = float(amplitude * np.pi / duration)
    full_range = upper - lower
    source_reference = None
    if source is not None:
        source_array = np.asarray(source, dtype=float)
        if source_array.shape != (990, 14) or not np.isfinite(source_array).all():
            raise Blocked("Stage B source-scale reference must be finite (990,14)")
        source_velocity = np.diff(source_array[:, source_index]) * SOURCE_FPS
        source_reference = {
            "source_index": source_index,
            "source_range_rad": float(np.ptp(source_array[:, source_index])),
            "source_peak_step_rad": float(
                np.abs(np.diff(source_array[:, source_index])).max()
            ),
            "source_peak_velocity_rad_s": float(np.abs(source_velocity).max()),
            "diagnostic_only_not_used_as_characterization_target": True,
        }
    watchdog = {name: None for name in WatchdogConfig.__dataclass_fields__}
    return {
        "status": "UNAPPROVED_DEFAULT",
        "approved": False,
        "hardware_replay_allowed": False,
        "controller_limits_verified": True,
        "controller_limits": limits,
        "controller_limit_authority": stage_a_authority,
        "minimum_controller_limit_margin": None,
        "watchdog": watchdog,
        "watchdog_hard_checks_without_tuned_thresholds": [
            "NONFINITE_COMMAND",
            "NONFINITE_STATE",
            "CONTROLLER_POSITION_LIMIT",
            "CONTROLLER_VELOCITY_MAX",
            "UNEXPECTED_FOLLOWER_ORDER",
            "COMMAND_FAILURE",
            "STATE_READ_FAILURE",
            "MANUAL_ABORT",
        ],
        "stages": {
            "B": {
                "status": "UNAPPROVED_DEFAULT",
                "approved": False,
                "active_side": side,
                "joint_indices": [joint],
                "source_index": source_index,
                "amplitude_rad": amplitude,
                "duration_sec": duration,
                "rate_hz": STAGE_B_RATE_HZ,
                "trajectory": "RAISED_COSINE_OUT_AND_BACK",
                "starts_at_measured_state": True,
                "ends_at_measured_state": True,
                "gripper_policy": "NO_GRIPPER_MODE_OR_POSITION_COMMAND",
                "command_api": "TrossenArmDriver.set_joint_position",
                "mode_api": "TrossenArmDriver.set_joint_modes",
                "api_evidence": {
                    "installed_header": (
                        "/home/jbnu/trossen_arm/include/libtrossen_arm/"
                        "trossen_arm.hpp"
                    ),
                    "set_joint_position_units": "rad for arm joints",
                    "non_selected_modes": "Mode.idle",
                    "gripper_position_api_called": False,
                },
            }
        },
        "recommendation": {
            "side": side,
            "joint_index": joint,
            "joint_semantic": "joint_0 (proximal/base joint; confirm visual direction)",
            "selection_basis": [
                "large symmetric measured position margin",
                "near-zero Stage A pose",
                "proximal joint preferred over wrist/gripper",
                "single-joint direction is visually understandable",
            ],
            "stage_a_current_rad": current,
            "controller_min_rad": lower,
            "controller_max_rad": upper,
            "lower_margin_at_candidate_rad": proposed_min - lower,
            "upper_margin_at_candidate_rad": upper - proposed_max,
            "amplitude_fraction_of_full_position_range": amplitude / full_range,
            "amplitude_fraction_of_source_joint_range": (
                None
                if source_reference is None or source_reference["source_range_rad"] == 0
                else amplitude / source_reference["source_range_rad"]
            ),
            "expected_peak_command_velocity_rad_s": peak_velocity,
            "controller_velocity_max_rad_s": velocity_max,
            "peak_velocity_fraction_of_controller_max": peak_velocity / velocity_max,
            "expected_peak_command_acceleration_rad_s2": float(
                2.0 * amplitude * np.pi**2 / duration**2
            ),
            "source_normal_motion_reference": source_reference,
            "joint_selection_evidence": {
                "model_path": (
                    "/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/"
                    "wxai/wxai_follower.xml"
                ),
                "joint_0_is_first_kinematic_joint": True,
                "joint_0_axis": [0, 0, 1],
                "wrist_joint_indices_avoided": [3, 4, 5],
                "physical_direction_still_requires_operator_visual_confirmation": True,
            },
            "operator_review_required": True,
        },
        "installed_timing": {
            "persistent_min_time_to_move_multiplier": 16,
            "driver_fps_hz": SOURCE_FPS,
            "controller_goal_time_sec": 16 / SOURCE_FPS,
            "publication_rate_hz": STAGE_B_RATE_HZ,
            "publication_period_sec": 1 / STAGE_B_RATE_HZ,
            "goal_time_to_publication_period_ratio": 16.0,
            "setpoint_behavior_from_installed_driver_audit": (
                "each new position target recomputes the controller interpolator; "
                "physical lag is not inferred and must be measured"
            ),
            "multiplier_changed_for_stage_b": False,
        },
        "real_motion_authorized": False,
        "blocking_items": [
            "VERIFIED_OPERATOR_STOP",
            "HUMAN_REVIEWED_STAGE_B_PARAMETERS",
            "HUMAN_REVIEWED_WATCHDOG_THRESHOLDS",
            "OPERATOR_VISUAL_DIRECTION_AND_COLLISION_REVIEW",
        ],
    }


def extract_stage_a_gripper_limits(
    state_record: dict[str, Any],
    controller_limits: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    limits = validate_inspection_limits(controller_limits)
    result: dict[str, Any] = {
        "status": "STAGE_A_GRIPPER_LIMITS_EXTRACTED",
        "authority": "CONTROLLER_REPORTED_JOINT_INDEX_6",
        "nominal_zero_assumed": False,
        "source_action_loaded": False,
        "sides": {},
    }
    for side, state_index in (("left", 6), ("right", 13)):
        item = limits[side][6]
        result["sides"][side] = {
            "joint_index": 6,
            "position_min": float(item["position_min"]),
            "position_max": float(item["position_max"]),
            "position_tolerance": float(item["position_tolerance"]),
            "current_position": float(state_record["state14"][state_index]),
            "unit": "m",
        }
    return result


def stage_a_motion_gate_statuses(operator_stop_verified: bool) -> dict[str, str]:
    if not operator_stop_verified:
        return {
            "stage_b": "STAGE_B_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
            "stage_c": "STAGE_C_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
            "stage_d": "STAGE_D_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
            "vla_replay": "VLA_REPLAY_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
        }
    return {
        "stage_b": "STAGE_B_REQUIRES_SEPARATE_MOTION_AUTHORIZATION",
        "stage_c": "STAGE_C_REQUIRES_SEPARATE_MOTION_AUTHORIZATION",
        "stage_d": "STAGE_D_REQUIRES_SEPARATE_MOTION_AUTHORIZATION",
        "vla_replay": "VLA_REPLAY_REQUIRES_SEPARATE_REVIEWED_AUTHORIZATION",
    }


def run_stage_a_idle_inspection(
    *,
    backend: StationaryAlohaLowLevelBackend,
    authorization: StageAInspectionAuthorization,
    operator_stop_verified: bool,
    abort_controller: AbortController | None = None,
    key_poller: Callable[[], str | None] | None = None,
    install_signal_handlers: bool = False,
) -> dict[str, Any]:
    """Run only configure-idle/read-state/read-limits; no source or motion path."""
    if authorization.risk_class is not RISK_CLASS_A or authorization.stage != "A":
        raise Blocked("Stage A runner requires READ_ONLY_IDLE_INSPECTION authorization")
    if not authorization.connection_allowed:
        raise Blocked("STAGE A REFUSED: " + "; ".join(authorization.reasons))

    backend.begin_stage_a_inspection()
    backend.assert_stage_a_position_command_invariant()
    abort = abort_controller or AbortController()
    handlers_installed = False
    result: dict[str, Any] = {
        "status": "STAGE_A_INSPECTION_IN_PROGRESS",
        "risk_class": RISK_CLASS_A.value,
        "optimized_action_loaded": False,
        "source_npz_accessed": False,
        "send_action_called": False,
        "trajectory_loop_entered": False,
        "position_commands": 0,
        "goal_position_commands": 0,
        "connection": None,
        "modes": {"after_configure": None, "after_reads": None},
        "state14": None,
        "controller_limits": None,
        "gripper_limits": None,
        "position_command_audit": backend.stage_a_position_command_audit(),
        "brake_abort": None,
        "transport_close": None,
        "failure": None,
        "motion_gate_statuses": stage_a_motion_gate_statuses(
            operator_stop_verified
        ),
        "successful_stage_a_authorizes_motion": False,
    }

    if install_signal_handlers:
        abort.install_signal_handlers()
        handlers_installed = True
    try:
        result["connection"] = backend.connect_stage_a_idle_without_home(
            authorization
        )
        result["modes"]["after_configure"] = {
            side: result["connection"]["followers"][side]["mode_readback"]
            for side in FOLLOWER_ORDER
        }
        backend.assert_stage_a_position_command_invariant()
        if key_poller is not None:
            key_poller()
        if abort.requested:
            raise InspectionFailure(
                "STAGE_A_USER_ABORT", abort.source or "unknown source"
            )

        try:
            sample = backend.read_state()
            result["state14"] = inspection_state_record(sample)
        except Exception as error:
            raise InspectionFailure("STAGE_A_STATE_READ_FAILED", str(error)) from error
        backend.assert_stage_a_position_command_invariant()

        try:
            result["controller_limits"] = validate_inspection_limits(
                backend.read_controller_limits()
            )
        except Exception as error:
            raise InspectionFailure("STAGE_A_LIMIT_READ_FAILED", str(error)) from error
        backend.assert_stage_a_position_command_invariant()

        result["modes"]["after_reads"] = backend.read_stage_a_idle_modes()
        backend.assert_stage_a_position_command_invariant()
        result["gripper_limits"] = extract_stage_a_gripper_limits(
            result["state14"], result["controller_limits"]
        )
        result["status"] = "STAGE_A_INSPECTION_COMPLETE"
    except Exception as error:
        if isinstance(error, InspectionFailure):
            code, detail = error.code, error.detail
        elif "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION" in str(error):
            code, detail = "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION", str(error)
        else:
            code, detail = "STAGE_A_UNCATEGORIZED_FAILURE", str(error)
        result["status"] = code
        result["failure"] = {
            "code": code,
            "type": type(error).__name__,
            "detail": detail,
        }
    finally:
        if handlers_installed:
            abort.restore_signal_handlers()
        if backend.is_connected():
            backend.stop_command_loop()
            result["brake_abort"] = backend.brake_abort(
                "STAGE_A_COMPLETE"
                if result["failure"] is None
                else f"STAGE_A_ABORT:{result['failure']['code']}"
            )
            if (
                result["failure"] is None
                and authorization.close_transport_allowed
                and result["brake_abort"]["status"]
                == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
            ):
                result["transport_close"] = backend.close_transport_without_home(
                    authorization
                )
            else:
                result["transport_close"] = {
                    "status": "NOT_CALLED_OPERATOR_INTERVENTION_REQUIRED",
                    "high_level_disconnect_called": False,
                }
        elif backend.last_brake_result is not None:
            result["brake_abort"] = backend.last_brake_result
            result["transport_close"] = {
                "status": "NOT_CALLED_PARTIAL_CONFIGURE_FAILURE",
                "high_level_disconnect_called": False,
            }

        result["position_command_audit"] = backend.stage_a_position_command_audit()
        result["position_commands"] = result["position_command_audit"][
            "position_commands"
        ]
        result["goal_position_commands"] = result["position_command_audit"][
            "goal_position_commands"
        ]
        result["trajectory_loop_entered"] = result["position_command_audit"][
            "trajectory_loop_entered"
        ]
        result["optimized_action_loaded"] = result["position_command_audit"][
            "optimized_action_loaded"
        ]
        if not result["position_command_audit"]["invariant_confirmed"]:
            result["status"] = "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION"
            if result["failure"] is None:
                result["failure"] = {
                    "code": "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION",
                    "type": "Blocked",
                    "detail": "Stage A position-command invariant failed",
                }
    return result


def write_stage_a_inspection_run(
    output_root: Path,
    result: dict[str, Any],
    authorization_record: dict[str, Any],
    config_summary: dict[str, Any],
    *,
    run_name: str | None = None,
) -> Path:
    name = run_name or datetime.now().strftime("stage_a_%Y%m%d_%H%M%S")
    output = Path(output_root) / name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "authorization_record.json", authorization_record)
    write_json(output / "persistent_config_snapshot.json", config_summary)
    write_json(output / "stage_a_result.json", result)
    write_json(output / "position_command_audit.json", result["position_command_audit"])

    modes = result.get("modes") or {}
    for side in FOLLOWER_ORDER:
        side_modes = {
            "after_configure": (modes.get("after_configure") or {}).get(side),
            "after_reads": (modes.get("after_reads") or {}).get(side),
            "brake_abort_readback": (
                ((result.get("brake_abort") or {}).get("sides") or {})
                .get(side, {})
                .get("api_readback")
            ),
        }
        if any(value is not None for value in side_modes.values()):
            write_json(output / f"{side}_modes.json", side_modes)
    if result.get("state14") is not None:
        write_json(output / "state14.json", result["state14"])
    if result.get("controller_limits") is not None:
        write_json(
            output / "controller_limits_left.json",
            result["controller_limits"]["left"],
        )
        write_json(
            output / "controller_limits_right.json",
            result["controller_limits"]["right"],
        )
    if result.get("gripper_limits") is not None:
        write_json(output / "gripper_limits.json", result["gripper_limits"])

    report = f"""# Stationary ALOHA Stage A idle-brake inspection

- Risk class: **{RISK_CLASS_A.value}**
- Status: **{result['status']}**
- Position commands: **{result['position_commands']}**
- Goal_Position commands: **{result['goal_position_commands']}**
- Trajectory loop entered: **{result['trajectory_loop_entered']}**
- optimized_action loaded: **{result['optimized_action_loaded']}**
- Motion authorized: **NO**

Stage A uses low-level configure/Mode.idle/read APIs only. Mode.idle may have a
physical effect. Per-follower power switches are manual isolation mechanisms,
not verified E-stops. Successful Stage A does not authorize Stage B/C/D or VLA replay.
"""
    (output / "stage_a_report.md").write_text(report)
    manifest = {
        "status": result["status"],
        "risk_class": RISK_CLASS_A.value,
        "hardware_phase": {
            "optimized_action_loaded": False,
            "trajectory_loop_entered": False,
            "position_commands": result["position_commands"],
            "goal_position_commands": result["goal_position_commands"],
            "high_level_connect_called": False,
        },
        "motion_gate_statuses": result["motion_gate_statuses"],
        "files": [],
    }
    write_json(output / "run_manifest.json", manifest)
    manifest["files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    write_json(output / "run_manifest.json", manifest)
    return output


def run_stage_a_source_limit_audit(
    stage_a_directory: Path,
    input_path: Path = DEFAULT_INPUT,
) -> dict[str, Any]:
    """Offline-only full source comparison using limits saved by a Stage A run."""
    directory = Path(stage_a_directory)
    limits = validate_controller_limits(
        {
            "left": json.loads((directory / "controller_limits_left.json").read_text()),
            "right": json.loads((directory / "controller_limits_right.json").read_text()),
        }
    )
    source, integrity = load_source_action(input_path)
    audit = compare_source_to_controller_limits(source, limits)
    assert_source_unchanged(input_path, source, integrity)
    gripper = {
        "status": (
            "SOURCE_GRIPPER_FULLY_WITHIN_CONTROLLER_LIMITS"
            if audit["gripper_violation_count"] == 0
            else "SOURCE_GRIPPER_OUTSIDE_CONTROLLER_LIMITS"
        ),
        "source_modified": False,
        "sides": {
            side: audit["sides"][side][6] for side in FOLLOWER_ORDER
        },
    }
    write_json(directory / "controller_limit_source_comparison.json", audit)
    write_json(directory / "gripper_source_comparison.json", gripper)
    write_json(directory / "source_action_integrity.json", integrity)
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["offline_source_audit"] = {
        "completed": True,
        "hardware_connection_opened": False,
        "source_action_modified": False,
    }
    manifest["files"] = sorted(path.name for path in directory.iterdir() if path.is_file())
    write_json(manifest_path, manifest)
    return {
        "status": audit["status"],
        "arm_violation_count": audit["arm_violation_count"],
        "gripper_violation_count": audit["gripper_violation_count"],
        "source_action_unchanged": integrity["source_action_unchanged"],
        "directory": str(directory.resolve()),
    }


def run_tracking_characterization_stage(
    *,
    backend: StationaryAlohaLowLevelBackend,
    authorization: CharacterizationAuthorization,
    stage: str,
    active_side: str = "left",
    joint_indices: tuple[int, ...] = (0,),
    amplitude_rad: float | None = None,
    duration_sec: float | None = None,
    rate_hz: float = STAGE_B_RATE_HZ,
    watchdog_config: WatchdogConfig | None = None,
    reviewed_controller_limits: dict[str, list[dict[str, float]]] | None = None,
    minimum_controller_limit_margin: float | None = None,
    abort_controller: AbortController | None = None,
    key_poller: Callable[[], str | None] | None = None,
    install_signal_handlers: bool = False,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one explicit motion stage B/C/D; never loads optimized_action."""
    stage = stage.upper()
    if stage not in {"B", "C", "D"}:
        raise Blocked("motion characterization runner accepts only Stage B/C/D")
    if authorization.risk_class is not RISK_CLASS_MOTION:
        raise Blocked("motion characterization requires MOTION risk class")
    if stage != authorization.stage:
        raise Blocked("characterization stage differs from authorization")
    if not authorization.connection_allowed:
        raise Blocked(
            "LOW-LEVEL CHARACTERIZATION REFUSED: " + "; ".join(authorization.reasons)
        )
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise Blocked("characterization rate must be positive")
    if stage == "B" and not np.isclose(
        rate_hz, STAGE_B_RATE_HZ, rtol=0.0, atol=0.0
    ):
        raise Blocked("Stage B must characterize the installed stack at exactly 30 Hz")
    result: dict[str, Any] = {
        "status": "CHARACTERIZATION_IN_PROGRESS",
        "stage": stage,
        "optimized_action_loaded": False,
        "vla_action_sent": False,
        "source_npz_accessed": False,
        "host_clock_only": True,
        "controller_synchronized_timestamps": False,
        "operator_visual_observation": "NOT_RECORDED_BY_SOFTWARE",
        "next_stage_automatically_authorized": False,
        "samples": [],
        "connection": None,
        "initial_state": None,
        "controller_limits": None,
        "rate_hz": float(rate_hz),
        "gripper_position_target_sent": False,
        "brake_abort": None,
        "transport_close": None,
        "failure": None,
    }
    abort = abort_controller or AbortController(clock_ns)
    handlers_installed = False
    connected = False
    if install_signal_handlers:
        abort.install_signal_handlers()
        handlers_installed = True
    try:
        result["connection"] = backend.connect_idle_without_home(authorization)
        connected = True
        if key_poller is not None:
            key_poller()
        if abort.requested:
            raise WatchdogAbort(
                "MANUAL_ABORT_AFTER_CONNECT", abort.source or "unknown source"
            )
        initial_sample = backend.read_state()
        initial_state = validate_action14(
            initial_sample.state14, label="characterization initial state"
        )
        actual_limits = backend.read_controller_limits()
        result["initial_state"] = inspection_state_record(initial_sample)
        result["controller_limits"] = actual_limits

        if not authorization.motion_allowed:
            raise Blocked("CHARACTERIZATION MOTION REFUSED")
        if (
            amplitude_rad is None
            or duration_sec is None
            or watchdog_config is None
            or reviewed_controller_limits is None
            or minimum_controller_limit_margin is None
        ):
            raise Blocked("motion stage requires reviewed limits, watchdog, amplitude, and duration")
        if watchdog_config.unresolved_fields():
            raise Blocked(
                "NEEDS_CHARACTERIZATION watchdog thresholds unresolved: "
                + ", ".join(watchdog_config.unresolved_fields())
            )
        assert_controller_limits_match_review(actual_limits, reviewed_controller_limits)
        trajectory, active_sides = generate_characterization_trajectory(
            initial_state,
            stage=stage,
            active_side=active_side,
            joint_indices=joint_indices,
            amplitude_rad=amplitude_rad,
            duration_sec=duration_sec,
            rate_hz=rate_hz,
        )
        commanded_indices = tuple(
            (0 if side == "left" else 7) + joint
            for side in active_sides
            for joint in joint_indices
        )
        result["trajectory_validation"] = validate_characterization_trajectory_limits(
            trajectory,
            actual_limits,
            minimum_margin=float(minimum_controller_limit_margin),
            commanded_indices=commanded_indices,
        )
        goal_time = characterization_goal_time_from_config(backend, active_sides)
        result["goal_time_sec"] = goal_time
        result["active_sides"] = list(active_sides)
        result["joint_indices"] = list(joint_indices)
        result["amplitude_rad"] = float(amplitude_rad)
        result["duration_sec"] = float(duration_sec)
        result["commanded_indices"] = list(commanded_indices)
        result["grippers_held"] = initial_state[[6, 13]].tolist()
        result["gripper_hold_semantics"] = (
            "MEASURED_VALUES_RETAINED_IN_LOGGED_14D_REFERENCE; "
            "GRIPPER_REMAINS MODE.IDLE AND RECEIVES NO POSITION TARGET"
        )
        result["command_api"] = (
            "TrossenArmDriver.set_joint_position"
            if stage == "B"
            else "TrossenArmDriver.set_arm_positions"
        )
        watchdog = RuntimeWatchdog(watchdog_config, actual_limits)

        backend.prepare_characterization_motion(
            active_sides=active_sides,
            joint_indices=joint_indices,
            current_state14=initial_state,
            authorization=authorization,
            goal_time_sec=goal_time,
        )
        period_ns = int(round(1e9 / rate_hz))
        result["timing_model"] = {
            "publication_rate_hz": float(rate_hz),
            "publication_period_sec": 1.0 / float(rate_hz),
            "controller_goal_time_sec": goal_time,
            "min_time_to_move_multiplier": {
                side: float(
                    backend.config.follower_arms[side].min_time_to_move_multiplier
                )
                for side in active_sides
            },
            "goal_time_is_not_assumed_fixed_lag": True,
            "clock": "HOST_MONOTONIC_ONLY",
        }
        phase_start_ns = int(clock_ns())
        previous_command = initial_state.copy()
        previous_actual = initial_state.copy()
        previous_loop_start: int | None = None
        for index, target in enumerate(trajectory):
            if key_poller is not None:
                key_poller()
            if abort.requested:
                raise WatchdogAbort("MANUAL_ABORT", abort.source or "unknown source")
            scheduled_ns = phase_start_ns + index * period_ns
            remaining_ns = scheduled_ns - int(clock_ns())
            if remaining_ns > 0:
                sleep_fn(remaining_ns / 1e9)
            if key_poller is not None:
                key_poller()
            if abort.requested:
                raise WatchdogAbort(
                    "MANUAL_ABORT_BEFORE_COMMAND", abort.source or "unknown source"
                )
            loop_start_ns = int(clock_ns())
            loop_period_sec = (
                1.0 / rate_hz
                if previous_loop_start is None
                else (loop_start_ns - previous_loop_start) / 1e9
            )
            watchdog.check_command_before_send(
                command14=target,
                previous_command14=previous_command,
                dt_sec=max(loop_period_sec, 1.0 / rate_hz),
                commanded_indices=commanded_indices,
                follower_order=tuple(backend.drivers.keys()),
                abort_controller=abort,
            )
            send_start_ns = int(clock_ns())
            sent = backend.send_characterization_target(
                target,
                active_sides=active_sides,
                joint_indices=joint_indices,
                controller_limits=actual_limits,
                held_grippers=(float(initial_state[6]), float(initial_state[13])),
                goal_time_sec=goal_time,
            )
            command_timestamp_ns = int(clock_ns())
            read_start_ns = int(clock_ns())
            sample = backend.read_state()
            read_done_ns = int(clock_ns())
            actual = validate_action14(sample.state14, label="characterization actual state")
            loop_overrun_sec = max(
                0.0, (read_done_ns - (scheduled_ns + period_ns)) / 1e9
            )
            state_age_sec = max(
                0.0, (read_done_ns - sample.host_monotonic_timestamp_ns) / 1e9
            )
            watchdog.check(
                command14=sent,
                actual14=actual,
                previous_command14=previous_command,
                previous_actual14=previous_actual,
                dt_sec=max(loop_period_sec, 1.0 / rate_hz),
                state_age_sec=state_age_sec,
                state_read_duration_sec=(read_done_ns - read_start_ns) / 1e9,
                command_call_duration_sec=(command_timestamp_ns - send_start_ns) / 1e9,
                loop_overrun_sec=loop_overrun_sec,
                follower_order=tuple(backend.drivers.keys()),
                abort_controller=abort,
                commanded_indices=commanded_indices,
            )
            row = {
                "sample_index": index,
                "scheduled_timestamp_ns": scheduled_ns,
                "command_timestamp_ns": command_timestamp_ns,
                "state_timestamp_ns": sample.host_monotonic_timestamp_ns,
                "command_q": sent.tolist(),
                "actual_q": actual.tolist(),
                "error": (sent - actual).tolist(),
                "hardware_commanded_indices": list(commanded_indices),
                "gripper_position_target_sent": False,
                "command_call_duration_sec": (command_timestamp_ns - send_start_ns) / 1e9,
                "state_read_duration_sec": (read_done_ns - read_start_ns) / 1e9,
                "loop_period_sec": loop_period_sec,
                "loop_overrun_sec": loop_overrun_sec,
                "state_age_sec": state_age_sec,
            }
            result["samples"].append(row)
            previous_command = sent.copy()
            previous_actual = actual.copy()
            previous_loop_start = loop_start_ns
        result["tracking_metrics"] = compute_tracking_metrics(
            result["samples"], rate_hz
        )
        if stage == "B":
            selected_index = commanded_indices[0]
            result["selected_joint_tracking_metrics"] = result[
                "tracking_metrics"
            ]["per_joint"][selected_index]
        result["tracking_classification"] = (
            "INCONCLUSIVE_PENDING_OPERATOR_VISUAL_REVIEW"
        )
        result["status"] = f"CHARACTERIZATION_STAGE_{stage}_COMPLETE"
    except Exception as error:
        result["status"] = "ABORT_CHARACTERIZATION"
        result["failure"] = {
            "type": type(error).__name__,
            "code": getattr(error, "code", None),
            "detail": str(error),
        }
        if not connected and backend.last_brake_result is not None:
            result["brake_abort"] = backend.last_brake_result
    finally:
        if handlers_installed:
            abort.restore_signal_handlers()
        if connected:
            backend.stop_command_loop()
            reason = (
                "CHARACTERIZATION_STAGE_COMPLETE"
                if result["failure"] is None
                else f"CHARACTERIZATION_ABORT:{result['failure']['detail']}"
            )
            result["brake_abort"] = backend.brake_abort(reason)
            if (
                result["failure"] is None
                and
                authorization.close_transport_allowed
                and result["brake_abort"]["status"]
                == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
            ):
                result["transport_close"] = backend.close_transport_without_home(
                    authorization
                )
            else:
                result["transport_close"] = {
                    "status": "NOT_CALLED_OPERATOR_INTERVENTION_REQUIRED",
                    "reason": (
                        "automatic transport close is forbidden after abort"
                        if result["failure"] is not None
                        else
                        "close not authorized"
                        if not authorization.close_transport_allowed
                        else "two-sided idle API readback not confirmed"
                    ),
                }
        result["samples_recorded"] = len(result["samples"])
        result["abort_request"] = {
            "requested": abort.requested,
            "reason": abort.reason,
            "source": abort.source,
            "requested_host_monotonic_ns": abort.requested_host_monotonic_ns,
        }
    return result


def write_characterization_run(
    output_root: Path,
    result: dict[str, Any],
    authorization_record: dict[str, Any],
    *,
    run_name: str | None = None,
) -> Path:
    name = run_name or datetime.now().strftime("characterization_%Y%m%d_%H%M%S")
    output = Path(output_root) / name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "authorization_record.json", authorization_record)
    write_json(output / "run_manifest.json", {
        "status": result["status"],
        "stage": result["stage"],
        "optimized_action_loaded": False,
        "vla_action_sent": False,
        "host_clock_only": True,
    })
    if result.get("initial_state") is not None:
        write_json(output / "actual_state_initial.json", result["initial_state"])
    if result.get("controller_limits") is not None:
        write_json(output / "controller_limits_left.json", result["controller_limits"]["left"])
        write_json(output / "controller_limits_right.json", result["controller_limits"]["right"])
    stage_payload = {key: value for key, value in result.items() if key != "samples"}
    write_json(output / f"tracking_stage_{result['stage'].lower()}.json", stage_payload)
    if result["samples"]:
        write_tracking_samples_csv(output / "tracking_samples.csv", result["samples"])
        write_json(output / "tracking_metrics.json", result.get("tracking_metrics", {}))
    report = f"""# Stationary ALOHA low-level characterization

- Stage: **{result['stage']}**
- Status: **{result['status']}**
- Source/VLA action loaded: **NO**
- VLA action sent: **NO**
- Clock: **HOST MONOTONIC ONLY**
- Software brake result: **{(result.get('brake_abort') or {}).get('status', 'NOT_AVAILABLE')}**

BRAKE_ABORT is software-commanded, depends on the network/controller, and is not
an independent physical E-stop.  This run does not authorize VLA replay.
"""
    (output / "characterization_report.md").write_text(report)
    manifest = json.loads((output / "run_manifest.json").read_text())
    manifest["files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    write_json(output / "run_manifest.json", manifest)
    return output


def write_stage_b_tracking_run(
    output_root: Path,
    result: dict[str, Any],
    authorization_record: dict[str, Any],
    characterization_config: dict[str, Any],
    stage_a_authority: dict[str, Any],
    *,
    run_name: str | None = None,
) -> Path:
    """Write the Stage B-specific evidence bundle; never writes a source action."""
    if result.get("stage") != "B":
        raise Blocked("Stage B output writer received another stage")
    name = run_name or datetime.now().strftime("stage_b_%Y%m%d_%H%M%S")
    output = Path(output_root) / name
    output.mkdir(parents=True, exist_ok=False)
    samples = list(result.get("samples", []))
    result_without_samples = {key: value for key, value in result.items() if key != "samples"}
    write_json(output / "stage_b_result.json", result_without_samples)
    write_json(output / "authorization_record.json", authorization_record)
    write_json(output / "initial_state.json", result.get("initial_state"))
    write_json(
        output / "controller_limits_snapshot.json",
        {
            "limits": result.get("controller_limits"),
            "stage_a_authority": stage_a_authority,
        },
    )
    write_json(output / "characterization_config.json", characterization_config)
    write_tracking_samples_csv(output / "tracking_samples.csv", samples)
    tracking_metrics = result.get("tracking_metrics", {})
    write_json(output / "tracking_metrics.json", tracking_metrics)
    write_json(
        output / "timing_metrics.json",
        {
            "timing_model": result.get("timing_model"),
            "loop": tracking_metrics.get("loop"),
            "host_clock_only": True,
            "controller_synchronized": False,
        },
    )
    write_json(
        output / "abort_log.json",
        {
            "abort_request": result.get("abort_request"),
            "failure": result.get("failure"),
            "automatic_disconnect_called": False,
        },
    )
    write_json(output / "brake_abort_result.json", result.get("brake_abort"))

    selected_index = None
    if result.get("commanded_indices"):
        selected_index = int(result["commanded_indices"][0])
    with (output / "command_vs_actual_selected_joint.csv").open(
        "w", newline=""
    ) as stream:
        fieldnames = [
            "sample_index",
            "scheduled_timestamp_ns",
            "command_timestamp_ns",
            "state_timestamp_ns",
            "source_index",
            "command",
            "actual",
            "error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        if selected_index is not None:
            for row in samples:
                writer.writerow(
                    {
                        "sample_index": row["sample_index"],
                        "scheduled_timestamp_ns": row["scheduled_timestamp_ns"],
                        "command_timestamp_ns": row["command_timestamp_ns"],
                        "state_timestamp_ns": row["state_timestamp_ns"],
                        "source_index": selected_index,
                        "command": row["command_q"][selected_index],
                        "actual": row["actual_q"][selected_index],
                        "error": row["error"][selected_index],
                    }
                )

    report = f"""# Stationary ALOHA Stage B tiny 30 Hz tracking

- Status: **{result.get('status')}**
- Side: **{(result.get('active_sides') or ['UNKNOWN'])[0]}**
- Joint indices: **{result.get('joint_indices')}**
- Amplitude: **{result.get('amplitude_rad')} rad**
- Duration: **{result.get('duration_sec')} s**
- Publication rate: **{result.get('rate_hz')} Hz**
- Controller goal time: **{result.get('goal_time_sec')} s**
- Gripper position target sent: **NO**
- optimized_action loaded/sent: **NO / NO**
- Tracking classification: **{result.get('tracking_classification', 'INCONCLUSIVE')}**
- BRAKE_ABORT: **{(result.get('brake_abort') or {}).get('status', 'NOT_AVAILABLE')}**

Stage B uses the low-level no-home connection and a single-joint raised-cosine
target.  The selected joint alone receives `set_joint_position`; both grippers
and all non-selected joints remain in idle/brake.  BRAKE_ABORT is software- and
network-dependent, is not a physical E-stop, and never calls home, sleep, or
high-level disconnect.  Stage B does not authorize VLA replay.
"""
    (output / "stage_b_report.md").write_text(report)
    write_json(
        output / "run_manifest.json",
        {
            "status": result.get("status"),
            "stage": "B",
            "source_npz_accessed": False,
            "optimized_action_loaded": False,
            "vla_action_sent": False,
            "gripper_position_target_sent": False,
            "automatic_vla_replay_authorized": False,
            "files": sorted(
                {
                    *(path.name for path in output.iterdir() if path.is_file()),
                    "run_manifest.json",
                }
            ),
        },
    )
    return output


def mock_controller_limits(gripper_min: float = -0.001, gripper_max: float = 0.05) -> dict[str, list[dict[str, float]]]:
    def item(position_min: float, position_max: float, *, gripper: bool) -> dict[str, float]:
        return {
            "position_min": position_min,
            "position_max": position_max,
            "position_tolerance": 1e-4,
            "velocity_max": 0.1 if gripper else 10.0,
            "velocity_tolerance": 1e-4,
            "effort_max": 10.0,
            "effort_tolerance": 1e-4,
        }

    limits: dict[str, list[dict[str, float]]] = {}
    for side in FOLLOWER_ORDER:
        limits[side] = [
            item(-10.0, 10.0, gripper=False) for _ in range(6)
        ] + [item(gripper_min, gripper_max, gripper=True)]
    return limits


def make_mock_low_level_backend(
    initial_state14: Any | None = None,
    *,
    fail_idle_side: str | None = None,
    controller_limits: dict[str, list[dict[str, float]]] | None = None,
) -> tuple[StationaryAlohaLowLevelBackend, dict[str, MockTrossenLowLevelDriver]]:
    initial = validate_action14(
        np.zeros(14) if initial_state14 is None else initial_state14,
        label="mock low-level initial state",
    )
    limits = validate_controller_limits(controller_limits or mock_controller_limits())
    follower_configs = {
        "left": type(
            "MockFollowerConfig",
            (),
            {
                "ip": "MOCK_LEFT_NO_NETWORK",
                "model": "V0_FOLLOWER",
                "min_time_to_move_multiplier": 16,
            },
        )(),
        "right": type(
            "MockFollowerConfig",
            (),
            {
                "ip": "MOCK_RIGHT_NO_NETWORK",
                "model": "V0_FOLLOWER",
                "min_time_to_move_multiplier": 16,
            },
        )(),
    }
    config = type(
        "MockStationaryConfig",
        (),
        {
            "follower_arms": follower_configs,
            "leader_arms": {"mock": object()},
            "cameras": {"mock": object()},
        },
    )()
    drivers = {
        side: MockTrossenLowLevelDriver(
            initial[offset : offset + 7],
            limits[side],
            fail_idle=side == fail_idle_side,
        )
        for side, offset in (("left", 0), ("right", 7))
    }
    backend = StationaryAlohaLowLevelBackend(
        config_factory=lambda _name: config,
        driver_factory=lambda side: drivers[side],
        idle_mode="idle",
        position_mode="position",
        model_mapping={"V0_FOLLOWER": ("mock_model", "mock_end_effector")},
    )
    backend.build_config()
    return backend, drivers


class MockALOHAHardwareBackend:
    """Network-free test double that records the exact replay-level 14D contract."""

    def __init__(self, initial_state: Any, controller_limits: dict[str, list[dict[str, float]]] | None = None):
        self.state = validate_action14(initial_state, label="mock initial state")
        self.limits = validate_controller_limits(controller_limits or mock_controller_limits())
        self.connected = False
        self.command_loop_stopped = True
        self.shutdown_called = False
        self.brake_abort_called = False
        self.last_brake_result: dict[str, Any] | None = None
        self.commands: list[dict[str, Any]] = []

    def build_config(self) -> dict[str, Any]:
        return {"status": "VERIFIED_OFFLINE", "follower_order": list(FOLLOWER_ORDER)}

    def connect(self, authorization: HardwareAuthorization | None = None) -> None:
        del authorization
        self.connected = True
        self.command_loop_stopped = False

    def connect_for_inspection(self, authorization: InspectionAuthorization | None = None) -> None:
        del authorization
        self.connected = True
        self.command_loop_stopped = True

    def is_connected(self) -> bool:
        return self.connected

    def read_state(self) -> StateSample:
        if not self.connected:
            raise Blocked("mock backend is not connected")
        return StateSample(self.state.copy(), time.monotonic_ns())

    def read_controller_limits(self) -> dict[str, list[dict[str, float]]]:
        return validate_controller_limits(self.limits)

    def prepare_action(
        self,
        action14: Any,
        controller_limits: dict[str, list[dict[str, float]]] | None = None,
        gripper_policy: GripperPolicy = GripperPolicy.REJECT,
        calibrated_offsets: dict[str, float] | None = None,
    ) -> np.ndarray:
        return prepare_action_for_limits(
            action14, controller_limits or self.limits, gripper_policy, calibrated_offsets
        )

    def send_action(
        self,
        action14: Any,
        authorization: HardwareAuthorization | None = None,
        controller_limits: dict[str, list[dict[str, float]]] | None = None,
        gripper_policy: GripperPolicy = GripperPolicy.REJECT,
        calibrated_offsets: dict[str, float] | None = None,
    ) -> np.ndarray:
        del authorization
        if not self.connected or self.command_loop_stopped:
            raise Blocked("mock command loop is not active")
        prepared = self.prepare_action(action14, controller_limits, gripper_policy, calibrated_offsets)
        tensor = to_cpu_float32_tensor(prepared)
        sent = tensor.detach().cpu().numpy().astype(np.float64)
        self.commands.append(
            {
                "action14": sent.copy(),
                "left7": sent[:7].copy(),
                "right7": sent[7:14].copy(),
                "tensor_dtype": str(tensor.dtype),
                "tensor_device": tensor.device.type,
            }
        )
        self.state = sent.copy()
        return sent

    def stop_command_loop(self) -> None:
        self.command_loop_stopped = True

    def brake_abort(self, reason: str) -> dict[str, Any]:
        self.stop_command_loop()
        self.brake_abort_called = True
        self.last_brake_result = {
            "status": "SOFTWARE_BRAKE_ABORT_CONFIRMED",
            "reason": str(reason),
            "sides": {
                side: {"idle_requested": True, "idle_confirmed": True}
                for side in FOLLOWER_ORDER
            },
            "disconnect_called": False,
            "physical_estop": False,
        }
        return self.last_brake_result

    def normal_shutdown(self, authorization: HardwareAuthorization | None = None) -> None:
        del authorization
        self.shutdown_called = True
        self.connected = False

    def normal_shutdown_for_inspection(
        self, authorization: InspectionAuthorization | None = None
    ) -> None:
        del authorization
        self.shutdown_called = True
        self.connected = False


class TrackingErrorMonitor:
    def __init__(self, max_tracking_error: Any, tracking_error_duration: float):
        threshold = np.asarray(max_tracking_error, dtype=float)
        if threshold.ndim == 0:
            threshold = np.repeat(threshold, 14)
        if threshold.shape != (14,) or not np.isfinite(threshold).all() or np.any(threshold <= 0):
            raise Blocked("max_tracking_error must be a positive scalar or 14D vector")
        if not np.isfinite(tracking_error_duration) or tracking_error_duration <= 0:
            raise Blocked("tracking_error_duration must be positive")
        self.threshold = threshold
        self.duration = float(tracking_error_duration)
        self.elapsed = 0.0

    def check(self, command: Any, actual: Any, period: float) -> None:
        error = np.abs(validate_action14(command) - validate_action14(actual, label="actual state"))
        self.elapsed = self.elapsed + period if np.any(error > self.threshold) else 0.0
        if self.elapsed >= self.duration:
            raise Blocked("PERSISTENT_TRACKING_ERROR")


def load_safety_config(path: Path) -> tuple[dict[str, Any], list[str]]:
    path = Path(path)
    if not path.exists():
        return {}, [f"reviewed safety config missing: {path}"]
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"invalid safety config: {error}"]
    reasons = []
    if config.get("status") != "REVIEWED":
        reasons.append("safety config status is not REVIEWED")
    if config.get("hardware_replay_allowed") is not True:
        reasons.append("hardware_replay_allowed is not true")
    required_non_null = (
        "controller_limits_verified",
        "controller_limits",
        "gripper_limits_verified",
        "connect_motion_approved",
        "disconnect_motion_approved",
        "max_start_position_error",
        "max_tracking_error",
        "tracking_error_duration",
        "max_joint_step",
        "max_joint_velocity",
        "max_joint_acceleration",
        "gripper_policy",
        "left_gripper_min",
        "left_gripper_max",
        "right_gripper_min",
        "right_gripper_max",
        "max_loop_overrun",
        "workspace_clear_confirmed",
    )
    for field in required_non_null:
        if config.get(field) is None:
            reasons.append(f"reviewed safety field unresolved: {field}")
    if config.get("gripper_policy") is not None:
        try:
            GripperPolicy(config["gripper_policy"])
        except ValueError:
            reasons.append("reviewed gripper_policy is invalid")
    if config.get("controller_limits") is not None:
        try:
            validate_controller_limits(config["controller_limits"])
        except (Blocked, KeyError, TypeError, ValueError) as error:
            reasons.append(f"reviewed controller_limits invalid: {error}")
    for field in (
        "max_start_position_error",
        "max_tracking_error",
        "max_joint_step",
        "max_joint_velocity",
        "max_joint_acceleration",
    ):
        if config.get(field) is not None:
            try:
                _maximum_allowed(config[field], field)
            except Blocked as error:
                reasons.append(str(error))
    for field in ("tracking_error_duration", "max_loop_overrun"):
        if config.get(field) is not None:
            try:
                value = float(config[field])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                reasons.append(f"reviewed {field} must be a positive finite scalar")
    for side in FOLLOWER_ORDER:
        lower = config.get(f"{side}_gripper_min")
        upper = config.get(f"{side}_gripper_max")
        if lower is not None and upper is not None:
            try:
                if not np.isfinite([float(lower), float(upper)]).all() or float(lower) > float(upper):
                    raise ValueError
            except (TypeError, ValueError):
                reasons.append(f"reviewed {side} gripper range is invalid")
    return config, reasons


def evaluate_preflight(
    *,
    source_valid: bool,
    config_valid: bool,
    follower_order_valid: bool,
    safety: dict[str, Any],
    safety_reasons: list[str],
    stop_verification: dict[str, Any],
    stop_verification_reasons: list[str],
    workspace_ack: bool,
    estop_ack: bool,
    connect_ack: bool,
    disconnect_ack: bool,
    execute_hardware: bool,
    dry_run: bool,
    hardware_confirmation: str | None,
    shutdown_confirmation: str | None,
    start_state_valid: bool = False,
) -> tuple[dict[str, Any], HardwareAuthorization]:
    # Kept as a deprecated CLI input for compatibility only.  A boolean cannot
    # establish a physical power-cut mechanism and must never authorize motion.
    del estop_ack
    states = {
        "SOURCE_ACTION_VALID": source_valid,
        "CONFIG_VALID": config_valid,
        "FOLLOWER_ORDER_VALID": follower_order_valid,
        "CONTROLLER_LIMITS_VERIFIED": safety.get("controller_limits_verified") is True,
        "GRIPPER_LIMITS_VERIFIED": safety.get("gripper_limits_verified") is True,
        "START_STATE_VALID": start_state_valid,
        "WORKSPACE_APPROVED": safety.get("workspace_clear_confirmed") is True and workspace_ack,
        "VERIFIED_OPERATOR_STOP": verified_operator_stop(
            stop_verification, stop_verification_reasons
        ),
        "CONNECT_MOTION_APPROVED": safety.get("connect_motion_approved") is True and connect_ack,
        "DISCONNECT_MOTION_APPROVED": safety.get("disconnect_motion_approved") is True and disconnect_ack,
        "HARDWARE_ALLOWED": safety.get("hardware_replay_allowed") is True,
        "NON_DRY_RUN": not dry_run,
        "EXECUTE_HARDWARE_REQUESTED": execute_hardware,
        "HARDWARE_CONFIRMATION_MATCH": hardware_confirmation == HARDWARE_CONFIRMATION,
        "SHUTDOWN_CONFIRMATION_MATCH": shutdown_confirmation == SHUTDOWN_CONFIRMATION,
    }
    connection_fields = (
        "SOURCE_ACTION_VALID",
        "CONFIG_VALID",
        "FOLLOWER_ORDER_VALID",
        "CONTROLLER_LIMITS_VERIFIED",
        "GRIPPER_LIMITS_VERIFIED",
        "WORKSPACE_APPROVED",
        "VERIFIED_OPERATOR_STOP",
        "CONNECT_MOTION_APPROVED",
        "DISCONNECT_MOTION_APPROVED",
        "HARDWARE_ALLOWED",
        "NON_DRY_RUN",
        "EXECUTE_HARDWARE_REQUESTED",
        "HARDWARE_CONFIRMATION_MATCH",
        "SHUTDOWN_CONFIRMATION_MATCH",
    )
    connection_allowed = not safety_reasons and all(states[name] for name in connection_fields)
    replay_allowed = connection_allowed and states["START_STATE_VALID"]
    failed = [name for name, passed in states.items() if not passed]
    reasons = [
        *safety_reasons,
        *stop_verification_reasons,
        *(f"preflight state false: {name}" for name in failed),
    ]
    result = {
        "status": "READY_FOR_HARDWARE" if replay_allowed else "BLOCKED",
        "risk_class": RISK_CLASS_MOTION.value,
        "connection_allowed": connection_allowed,
        "replay_allowed": replay_allowed,
        "states": states,
        "reasons": reasons,
        "software_estop": "UNKNOWN",
        "legacy_operator_estop_flag": "IGNORED_NOT_AUTHORIZATION",
        "operator_stop_record": stop_verification,
        "connect_has_motion": True,
        "disconnect_has_motion": True,
    }
    authorization = HardwareAuthorization(
        connection_allowed=connection_allowed,
        replay_allowed=replay_allowed,
        normal_shutdown_allowed=connection_allowed and states["DISCONNECT_MOTION_APPROVED"],
        reasons=tuple(reasons),
    )
    return result, authorization


def _maximum_allowed(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.repeat(array, 14)
    if array.shape != (14,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise Blocked(f"reviewed {name} must be a positive scalar or 14D vector")
    return array


def validate_motion_thresholds(commands: np.ndarray, command_hz: float, safety: dict[str, Any]) -> dict[str, Any]:
    metrics = motion_metrics(commands, command_hz)
    comparisons = (
        ("max_step_per_channel", "max_joint_step"),
        ("max_velocity_per_channel", "max_joint_velocity"),
        ("max_acceleration_per_channel", "max_joint_acceleration"),
    )
    for measured_name, limit_name in comparisons:
        measured = np.asarray(metrics[measured_name], dtype=float)
        allowed = _maximum_allowed(safety.get(limit_name), limit_name)
        if np.any(measured > allowed):
            raise Blocked(f"{measured_name} exceeds reviewed {limit_name}")
    return metrics


def run_mock_offline(source: np.ndarray, start_transition_seconds: float, command_hz: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_before = array_sha256(source)
    initial = source[0].astype(float).copy()
    initial[:6] += 0.02
    initial[7:13] -= 0.02
    backend = MockALOHAHardwareBackend(initial, mock_controller_limits())
    backend.connect()
    sampled = backend.read_state()
    transition = minimum_jerk_transition(
        sampled.state14, source[0], start_transition_seconds, command_hz, hold_gripper=True
    )
    if not np.isfinite(transition).all():
        raise Blocked("minimum-jerk transition contains NaN/Inf")
    if not np.array_equal(transition[0], sampled.state14):
        raise Blocked("minimum-jerk transition does not start at actual state")
    if not np.allclose(transition[-1, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], source[0, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]):
        raise Blocked("minimum-jerk transition does not end at source arm target")
    if not np.all(transition[:, [6, 13]] == sampled.state14[[6, 13]]):
        raise Blocked("minimum-jerk transition did not hold grippers")

    timing: list[dict[str, Any]] = []
    period_ns = int(round(1e9 / command_hz))
    all_commands = np.vstack((transition, source.astype(float, copy=True)))
    for index, command in enumerate(all_commands):
        scheduled_ns = index * period_ns
        read = backend.read_state()
        sent = backend.send_action(command, gripper_policy=GripperPolicy.REJECT)
        timing.append(
            {
                "clock_mode": "SYNTHETIC_OFFLINE",
                "scheduled_timestamp_ns": scheduled_ns,
                "actual_host_send_timestamp_ns": scheduled_ns,
                "state_read_timestamp_ns": read.host_monotonic_timestamp_ns,
                "loop_period_ns": 0 if index == 0 else period_ns,
                "loop_overrun_ns": 0,
                "command_age_ns": 0,
                "source_frame": index - len(transition),
                "left7": sent[:7].tolist(),
                "right7": sent[7:].tolist(),
            }
        )
    backend.stop_command_loop()
    safety_checks: dict[str, bool] = {}
    strict_limits = mock_controller_limits(gripper_min=0.0, gripper_max=0.044)
    negative = np.zeros(14)
    negative[6] = -1e-4
    try:
        prepare_action_for_limits(negative, strict_limits, GripperPolicy.REJECT)
        safety_checks["negative_gripper_rejected"] = False
    except Blocked:
        safety_checks["negative_gripper_rejected"] = True
    out_of_range = np.zeros(14)
    out_of_range[0] = 11.0
    try:
        prepare_action_for_limits(out_of_range, strict_limits, GripperPolicy.REJECT)
        safety_checks["arm_controller_limit_rejected"] = False
    except Blocked:
        safety_checks["arm_controller_limit_rejected"] = True
    monitor = TrackingErrorMonitor(0.1, 0.2)
    try:
        monitor.check(np.zeros(14), np.ones(14), 0.1)
        monitor.check(np.zeros(14), np.ones(14), 0.1)
        safety_checks["persistent_tracking_error_rejected"] = False
    except Blocked:
        safety_checks["persistent_tracking_error_rejected"] = True
    if not all(safety_checks.values()):
        raise Blocked(f"mock safety self-check failed: {safety_checks}")
    results = {
        "status": "PASS",
        "backend": "MockALOHAHardwareBackend",
        "network_access": False,
        "source_frames_processed": len(source),
        "transition_frames": len(transition),
        "commands_recorded": len(backend.commands),
        "follower_order": list(FOLLOWER_ORDER),
        "first_command_left7": backend.commands[0]["left7"].tolist(),
        "first_command_right7": backend.commands[0]["right7"].tolist(),
        "tensor_dtype": backend.commands[0]["tensor_dtype"],
        "tensor_device": backend.commands[0]["tensor_device"],
        "transition_starts_at_actual": True,
        "transition_ends_at_source_arm_start": True,
        "transition_holds_grippers": True,
        "source_action_unchanged": array_sha256(source) == source_before,
        "stop_called_without_shutdown": backend.command_loop_stopped and not backend.shutdown_called,
        "timing_rows": len(timing),
        "safety_self_checks": safety_checks,
    }
    return results, timing


def gripper_policy_audit(source: np.ndarray) -> dict[str, Any]:
    strict_limits = mock_controller_limits(gripper_min=0.0, gripper_max=0.044)
    violations = []
    for frame, action in enumerate(source):
        try:
            prepare_action_for_limits(action, strict_limits, GripperPolicy.REJECT)
        except Blocked as error:
            if action[6] < 0 or action[13] < 0:
                violations.append({"frame": frame, "left": float(action[6]), "right": float(action[13]), "reason": str(error)})
    clamp_preview = source.astype(float, copy=True)
    clamp_preview[:, 6] = np.clip(clamp_preview[:, 6], 0.0, 0.044)
    clamp_preview[:, 13] = np.clip(clamp_preview[:, 13], 0.0, 0.044)
    return {
        "status": "BLOCKED" if violations else "PASS",
        "default_hardware_policy": GripperPolicy.REJECT.value,
        "unit": "m",
        "larger_means": "more open",
        "source_minimum": {"left": float(source[:, 6].min()), "right": float(source[:, 13].min())},
        "reject_violation_count": len(violations),
        "first_reject_violations": violations[:10],
        "clamp_preview_only": {
            "source_was_modified": False,
            "preview_left_min": float(clamp_preview[:, 6].min()),
            "preview_right_min": float(clamp_preview[:, 13].min()),
        },
        "actual_controller_endpoints": "NEEDS_REAL_HARDWARE_CHECK",
    }


def evaluate_inspection_authorization(
    *,
    config_valid: bool,
    follower_order_valid: bool,
    workspace_clear_confirmed: bool,
    operator_estop_confirmed: bool,
    stop_verification: dict[str, Any],
    stop_verification_reasons: list[str],
    connect_motion_approved: bool,
    disconnect_motion_approved: bool,
    physical_left_right_verified: bool,
    inspection_confirmation: str | None,
) -> tuple[dict[str, Any], InspectionAuthorization]:
    """Authorize connect/read/limit-query only; replay safety is intentionally absent."""
    # Deprecated and deliberately ignored.  It is retained only so old command
    # lines fail closed instead of changing meaning silently.
    del operator_estop_confirmed
    states = {
        "CONFIG_VALID": config_valid,
        "FOLLOWER_ORDER_VALID": follower_order_valid,
        "WORKSPACE_CLEAR_CONFIRMED": workspace_clear_confirmed,
        "VERIFIED_OPERATOR_STOP": verified_operator_stop(
            stop_verification, stop_verification_reasons
        ),
        "CONNECT_MOTION_APPROVED": connect_motion_approved,
        "DISCONNECT_MOTION_APPROVED": disconnect_motion_approved,
        "PHYSICAL_LEFT_RIGHT_VERIFIED": physical_left_right_verified,
        "INSPECTION_CONFIRMATION_MATCH": inspection_confirmation == INSPECTION_CONFIRMATION,
    }
    reasons = [
        *stop_verification_reasons,
        *(f"inspection state false: {name}" for name, value in states.items() if not value),
    ]
    allowed = all(states.values())
    authorization = InspectionAuthorization(
        inspection_allowed=allowed,
        normal_shutdown_allowed=allowed and disconnect_motion_approved,
        reasons=tuple(reasons),
        states=states,
    )
    result = {
        "authorization_level": "HARDWARE_INSPECTION",
        "status": "INSPECTION_AUTHORIZED" if allowed else "INSPECTION_BLOCKED",
        "states": states,
        "reasons": reasons,
        "legacy_operator_estop_flag": "IGNORED_NOT_AUTHORIZATION",
        "operator_stop_record": stop_verification,
        "requires_replay_safety_config": False,
        "trajectory_replay_authorized": False,
        "connect_motion_expected": True,
        "disconnect_motion_expected": True,
    }
    return result, authorization


def validate_inspection_limits(
    limits: Any,
) -> dict[str, list[dict[str, float]]]:
    try:
        checked = validate_controller_limits(limits)
    except (Blocked, KeyError, TypeError, ValueError) as error:
        raise InspectionFailure("BLOCKED_INVALID_HARDWARE_DATA", str(error)) from error
    for side in FOLLOWER_ORDER:
        for joint_index, joint in enumerate(checked[side]):
            for field, raw_value in joint.items():
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise InspectionFailure(
                        "BLOCKED_INVALID_HARDWARE_DATA",
                        f"{side} joint_{joint_index} limit {field} is non-numeric",
                    ) from error
                if not np.isfinite(value):
                    raise InspectionFailure(
                        "BLOCKED_INVALID_HARDWARE_DATA",
                        f"{side} joint_{joint_index} limit {field} is non-finite",
                    )
    return checked


def inspection_state_record(sample: StateSample) -> dict[str, Any]:
    try:
        state = validate_action14(sample.state14, label="inspection observation.state")
    except Blocked as error:
        raise InspectionFailure("BLOCKED_INVALID_HARDWARE_DATA", str(error)) from error
    return {
        "timestamp_source": "HOST_MONOTONIC_ONLY",
        "host_monotonic_timestamp_ns": int(sample.host_monotonic_timestamp_ns),
        "controller_synchronized_timestamp": False,
        "joint_order": JOINTS,
        "units": UNITS,
        "state14": state.tolist(),
        "left": {"arm_joint_0_to_5": state[:6].tolist(), "gripper": float(state[6])},
        "right": {"arm_joint_0_to_5": state[7:13].tolist(), "gripper": float(state[13])},
    }


def build_gripper_limit_comparison(
    state_record: dict[str, Any],
    controller_limits: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "READ_ONLY_COMPARISON",
        "source_values_modified": False,
        "controller_values_modified": False,
        "nominal_zero_assumed": False,
        "note": "The previous nominal-zero 759-frame rejection is not treated as a proven source failure.",
        "sides": {},
    }
    for side, state_index in (("left", 6), ("right", 13)):
        limit = controller_limits[side][6]
        controller_min = float(limit["position_min"])
        controller_max = float(limit["position_max"])
        source_min, source_max = VERIFIED_SOURCE_GRIPPER_RANGE[side]
        if source_min < controller_min:
            comparison = "SOURCE_GRIPPER_BELOW_CONTROLLER_LIMIT"
        elif source_max <= controller_max:
            comparison = "SOURCE_GRIPPER_WITHIN_CONTROLLER_LIMIT"
        else:
            comparison = "SOURCE_GRIPPER_LIMIT_INCONCLUSIVE"
        result["sides"][side] = {
            f"{side}_gripper_controller_min": controller_min,
            f"{side}_gripper_controller_max": controller_max,
            f"{side}_current_gripper_position": float(state_record["state14"][state_index]),
            "source_gripper_min": source_min,
            "source_gripper_max": source_max,
            "comparison": comparison,
            "unit": "m",
        }
    return result


def run_hardware_inspection(
    *,
    backend: Any,
    authorization: InspectionAuthorization,
    authorization_record: dict[str, Any],
    config_summary: dict[str, Any],
    output_root: Path,
    inspection_name: str | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> tuple[Path, dict[str, Any]]:
    """Connect and read state/limits only. This function contains no replay/action path."""
    if not authorization.inspection_allowed:
        raise Blocked("HARDWARE INSPECTION REFUSED: " + "; ".join(authorization.reasons))
    if not authorization.normal_shutdown_allowed:
        raise Blocked("HARDWARE INSPECTION REFUSED: normal shutdown motion is not approved")

    name = inspection_name or datetime.now().strftime("inspection_%Y%m%d_%H%M%S")
    output = Path(output_root) / name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "persistent_config_snapshot.json", config_summary)
    write_json(output / "authorization_record.json", authorization_record)

    result: dict[str, Any] = {
        "inspection_status": "INSPECTION_IN_PROGRESS",
        "replay_status": "REPLAY_STILL_BLOCKED",
        "connect_motion_expected": True,
        "disconnect_motion_expected": True,
        "measured_physical_motion_claimed": False,
        "optimized_action_loaded": False,
        "send_action_called": False,
        "goal_position_command_called": False,
        "connect_start_host_monotonic_ns": None,
        "connect_complete_host_monotonic_ns": None,
        "shutdown_start_host_monotonic_ns": None,
        "shutdown_complete_host_monotonic_ns": None,
        "failure_code": None,
        "failure_detail": None,
    }
    state_payload: dict[str, Any] = {}
    controller_limits: dict[str, list[dict[str, float]]] | None = None
    gripper_comparison: dict[str, Any] | None = None
    failure: InspectionFailure | None = None

    result["connect_start_host_monotonic_ns"] = int(clock_ns())
    try:
        try:
            backend.connect_for_inspection(authorization)
        except Exception as error:
            raise InspectionFailure("INSPECTION_CONNECTION_FAILED", str(error)) from error
        result["connect_complete_host_monotonic_ns"] = int(clock_ns())

        try:
            first_sample = backend.read_state()
        except InspectionFailure:
            raise
        except Blocked as error:
            if any(token in str(error) for token in ("NaN/Inf", "numeric", "shape")):
                raise InspectionFailure("BLOCKED_INVALID_HARDWARE_DATA", str(error)) from error
            raise InspectionFailure("INSPECTION_STATE_READ_FAILED", str(error)) from error
        except Exception as error:
            raise InspectionFailure("INSPECTION_STATE_READ_FAILED", str(error)) from error
        first_record = inspection_state_record(first_sample)
        state_payload["state_after_connect"] = first_record

        try:
            raw_limits = backend.read_controller_limits()
        except InspectionFailure:
            raise
        except Blocked as error:
            if "BLOCKED_FOLLOWER_ORDER" in str(error):
                raise InspectionFailure("BLOCKED_FOLLOWER_ORDER", str(error)) from error
            if "non-finite" in str(error) or "invalid" in str(error).lower():
                raise InspectionFailure("BLOCKED_INVALID_HARDWARE_DATA", str(error)) from error
            raise InspectionFailure("INSPECTION_LIMIT_QUERY_FAILED", str(error)) from error
        except Exception as error:
            raise InspectionFailure("INSPECTION_LIMIT_QUERY_FAILED", str(error)) from error
        controller_limits = validate_inspection_limits(raw_limits)
        gripper_comparison = build_gripper_limit_comparison(first_record, controller_limits)

        try:
            final_sample = backend.read_state()
        except InspectionFailure:
            raise
        except Blocked as error:
            if any(token in str(error) for token in ("NaN/Inf", "numeric", "shape")):
                raise InspectionFailure("BLOCKED_INVALID_HARDWARE_DATA", str(error)) from error
            raise InspectionFailure("INSPECTION_STATE_READ_FAILED", str(error)) from error
        except Exception as error:
            raise InspectionFailure("INSPECTION_STATE_READ_FAILED", str(error)) from error
        state_payload["state_before_shutdown"] = inspection_state_record(final_sample)
        result["inspection_status"] = "REAL_HARDWARE_INSPECTION_COMPLETE"
    except InspectionFailure as error:
        failure = error
        result["inspection_status"] = error.code
        result["failure_code"] = error.code
        result["failure_detail"] = error.detail
    finally:
        if backend.is_connected():
            result["shutdown_start_host_monotonic_ns"] = int(clock_ns())
            try:
                backend.normal_shutdown_for_inspection(authorization)
            except Exception as error:
                if failure is None:
                    failure = InspectionFailure("INSPECTION_SHUTDOWN_FAILED", str(error))
                    result["inspection_status"] = failure.code
                    result["failure_code"] = failure.code
                    result["failure_detail"] = failure.detail
                else:
                    result["shutdown_failure_detail"] = str(error)
            result["shutdown_complete_host_monotonic_ns"] = int(clock_ns())

    if state_payload:
        write_json(output / "current_state_before_or_after_connect.json", state_payload)
    if controller_limits is not None:
        write_json(output / "controller_limits_left.json", controller_limits["left"])
        write_json(output / "controller_limits_right.json", controller_limits["right"])
    if gripper_comparison is not None:
        write_json(output / "gripper_limit_comparison.json", gripper_comparison)
    write_json(output / "inspection_result.json", result)
    manifest = {
        "inspection_directory": str(output.resolve()),
        "inspection_status": result["inspection_status"],
        "replay_status": "REPLAY_STILL_BLOCKED",
        "authorization_level": "HARDWARE_INSPECTION",
        "files": sorted(
            {
                *(path.name for path in output.iterdir() if path.is_file()),
                "run_manifest.json",
            }
        ),
        "forbidden_operations": {
            "optimized_action_loaded": False,
            "send_action_called": False,
            "goal_position_command_called": False,
            "trajectory_loop_entered": False,
        },
    }
    write_json(output / "run_manifest.json", manifest)
    report = f"""# Stationary ALOHA real hardware inspection

- Inspection: **{result['inspection_status']}**
- Replay: **REPLAY_STILL_BLOCKED**
- Connect motion expected: **YES**
- Disconnect motion expected: **YES**
- VLA/source action loaded: **NO**
- Action/Goal_Position sent: **NO**

This inspection records host monotonic timestamps, not controller-synchronized timestamps.
Successful inspection never authorizes replay.
"""
    (output / "inspection_report.md").write_text(report)
    # Refresh the manifest after every required report file has been created.
    manifest["files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    write_json(output / "run_manifest.json", manifest)
    if failure is not None:
        raise failure
    return output, result


def preflight_schema() -> dict[str, Any]:
    return {
        "decision": ["READY_FOR_HARDWARE", "BLOCKED"],
        "states": [
            "SOURCE_ACTION_VALID",
            "CONFIG_VALID",
            "FOLLOWER_ORDER_VALID",
            "CONTROLLER_LIMITS_VERIFIED",
            "GRIPPER_LIMITS_VERIFIED",
            "START_STATE_VALID",
            "WORKSPACE_APPROVED",
            "VERIFIED_OPERATOR_STOP",
            "CONNECT_MOTION_APPROVED",
            "DISCONNECT_MOTION_APPROVED",
            "HARDWARE_ALLOWED",
        ],
        "default": "BLOCKED",
        "controller_timestamp_available": False,
        "state_timestamp": "host time.monotonic_ns()",
        "controller_limits_format": {
            "left": "ordered list of 7 joint limit objects",
            "right": "ordered list of 7 joint limit objects",
            "required_per_joint_fields": ["position_min", "position_max"],
            "authority": "values queried from each real controller and human-reviewed",
        },
        "authorization_levels": {
            "HARDWARE_INSPECTION": "connect + state/limit reads + approved shutdown; no action path",
            "HARDWARE_REPLAY": "separate stricter authorization requiring reviewed replay safety",
        },
        "inspection_authorization_states": [
            "CONFIG_VALID",
            "FOLLOWER_ORDER_VALID",
            "WORKSPACE_CLEAR_CONFIRMED",
            "VERIFIED_OPERATOR_STOP",
            "CONNECT_MOTION_APPROVED",
            "DISCONNECT_MOTION_APPROVED",
            "PHYSICAL_LEFT_RIGHT_VERIFIED",
            "INSPECTION_CONFIRMATION_MATCH",
        ],
    }


def commands_text() -> str:
    python = "/home/jbnu/miniconda3/bin/conda run --no-capture-output -n trossen_ai_data_collection_ui_env python"
    script = str(ROOT / "tools/validate_vla_action_on_real_aloha.py")
    trajectory = str(DEFAULT_INPUT)
    return f"""#!/usr/bin/env bash
# SAFE/OFFLINE: full dry-run; never constructs ManipulatorRobot.
{python} {script} --input {trajectory} --dry-run --inspect --speed-scale 0.25 --no-object

# SAFE/STATIC: loads persistent config and evaluates gates; never connects.
{python} {script} --input {trajectory} --hardware-preflight

# REAL HARDWARE INSPECTION: BLOCKED until the physical stop record is verified.
# DO NOT RUN UNATTENDED; connect/disconnect can move both followers.
# This mode never loads optimized_action and never calls send_action.
{python} {script} --hardware-inspect --workspace-clear-confirmed \\
  --stop-verification-record {DEFAULT_STOP_VERIFICATION} \\
  --physical-left-right-verified --acknowledge-connect-moves-home \\
  --acknowledge-disconnect-moves-home-sleep \\
  --inspection-confirmation '{INSPECTION_CONFIRMATION}'

# DO NOT RUN YET: requires a human-created REVIEWED safety file and every acknowledgement.
{python} {script} --input {trajectory} --execute-hardware --start-frame 0 --end-frame 29 --no-object \\
  --safety-config {DEFAULT_SAFETY} --stop-verification-record {DEFAULT_STOP_VERIFICATION} \\
  --workspace-clear-confirmed \\
  --acknowledge-connect-moves-home --acknowledge-disconnect-moves-home-sleep \\
  --hardware-confirmation '{HARDWARE_CONFIRMATION}' \\
  --shutdown-confirmation '{SHUTDOWN_CONFIRMATION}'
"""


def write_integration_reports(
    output: Path,
    *,
    integrity: dict[str, Any],
    config_summary: dict[str, Any],
    preflight: dict[str, Any],
    dry_run: dict[str, Any] | None,
    mock_results: dict[str, Any] | None,
    gripper: dict[str, Any],
    timing: list[dict[str, Any]] | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if mock_results is None and (output / "mock_test_results.json").exists():
        mock_results = json.loads((output / "mock_test_results.json").read_text())
    if timing is None and (output / "timing_instrumentation.json").exists():
        timing = json.loads((output / "timing_instrumentation.json").read_text())
    audit = {
        "status": "HARDWARE_BACKEND_IMPLEMENTED",
        "evidence": "VERIFIED_FROM_SOURCE_CODE",
        "backend_class": "ALOHAHardwareBackend",
        "mock_backend_class": "MockALOHAHardwareBackend",
        "interfaces": [
            "build_config",
            "connect",
            "connect_for_inspection",
            "is_connected",
            "read_state",
            "read_controller_limits",
            "prepare_action",
            "send_action",
            "stop_command_loop",
            "normal_shutdown",
            "normal_shutdown_for_inspection",
        ],
        "config": config_summary,
        "joint_order": JOINTS,
        "units": UNITS,
        "command_path": "ManipulatorRobot.send_action(torch.float32 CPU tensor)",
        "state_path": "ManipulatorRobot.capture_observation()['observation.state']",
        "real_connection_opened": False,
        "real_command_executed": False,
        "software_estop": "UNKNOWN",
        "inspection_mode": {
            "implemented": True,
            "authorization_level": "HARDWARE_INSPECTION",
            "loads_optimized_action": False,
            "calls_send_action": False,
            "replay_remains_separately_blocked": True,
        },
    }
    write_json(output / "hardware_backend_audit.json", audit)
    write_json(output / "preflight_schema.json", preflight_schema())
    write_json(output / "source_action_integrity.json", integrity)
    write_json(output / "dry_run_results.json", dry_run or {"status": "NOT_RUN"})
    write_json(output / "mock_test_results.json", mock_results or {"status": "NOT_RUN"})
    write_json(output / "gripper_policy_audit.json", gripper)
    write_json(output / "static_preflight_result.json", preflight)
    write_json(output / "timing_instrumentation.json", timing or [])
    (output / "commands.sh").write_text(commands_text())
    report = f"""# Stationary ALOHA backend integration

- Backend: **HARDWARE_BACKEND_IMPLEMENTED**
- Offline/mock: **{(mock_results or {}).get('status', 'NOT_RUN')}**
- Real hardware: **{preflight['status']}**
- Real connection opened by this report run: **NO**

## Evidence classes

- `VERIFIED_FROM_SOURCE_CODE`: follower ordering, LeRobot send/state APIs, config source.
- `VERIFIED_OFFLINE`: source integrity, Tensor conversion, mock transition and replay.
- `NEEDS_REAL_HARDWARE_CHECK`: controller limits, physical gripper endpoints, tracking/timing.
- `BLOCKED`: any unreviewed safety field or missing operator acknowledgement.

## Static preflight

```json
{json.dumps(preflight, indent=2, default=_json_default)}
```

## Gripper

Default real-hardware policy is `REJECT`. No source action was clamped or overwritten.

## Stop and shutdown

`stop_command_loop()` only stops new commands. `normal_shutdown()` is separately gated
because the vendor disconnect path moves both followers home and then sleep.

## Real hardware inspection

`--hardware-inspect` uses a separate `InspectionAuthorization`. It does not load
`optimized_action`, call `send_action`, or enter the replay loop. No real inspection
was performed while generating this offline integration report.
"""
    (output / "report.md").write_text(report)


def software_safety_commands_text() -> str:
    python = (
        "/home/jbnu/miniconda3/bin/conda run --no-capture-output "
        "-n trossen_ai_data_collection_ui_env python"
    )
    script = str(ROOT / "tools/validate_vla_action_on_real_aloha.py")
    source = str(DEFAULT_INPUT)
    return f"""#!/usr/bin/env bash
# SAFE/OFFLINE. Generates the source/mocks/audit bundle; no driver is constructed.
{python} {script} --input {source} --software-safety-audit

# DO NOT RUN DURING DEVELOPMENT: REAL Stage A uses its separate low-risk gate.
# It opens Ethernet and requests Mode.idle, which may have a physical effect.
{python} {script} --hardware-characterize-tracking --characterization-stage A \
  --stop-verification-record {DEFAULT_STOP_VERIFICATION} \
  --workspace-clear-confirmed --physical-left-right-verified \
  --operator-present-confirmed \
  --left-power-switch-reachable --right-power-switch-reachable \
  --acknowledge-idle-brake-connect-may-move \
  --acknowledge-idle-cleanup-command \
  --stage-a-confirmation '{STAGE_A_CONFIRMATION}'

# DO NOT RUN: REAL Stage B tiny generated test. It requires Stage A approval and a
# human-reviewed characterization file built from real controller limits.
{python} {script} --hardware-characterize-tracking --characterization-stage B \
  --characterization-config {DEFAULT_CHARACTERIZATION_CONFIG} \
  --prior-characterization-stage-approved \
  --stop-verification-record {DEFAULT_STOP_VERIFICATION} \
  --workspace-clear-confirmed --physical-left-right-verified \
  --acknowledge-idle-brake-connect-may-move \
  --acknowledge-idle-cleanup-command \
  --characterization-confirmation '{CHARACTERIZATION_CONFIRMATION}'

# DO NOT RUN: eventual VLA frames 0..29. This remains on the separately reviewed
# replay path and is blocked by the current physical-stop and replay safety records.
{python} {script} --input {source} --execute-hardware --start-frame 0 --end-frame 29 \
  --speed-scale 1.0 --no-object --safety-config {DEFAULT_SAFETY} \
  --stop-verification-record {DEFAULT_STOP_VERIFICATION} \
  --workspace-clear-confirmed --acknowledge-connect-moves-home \
  --acknowledge-disconnect-moves-home-sleep \
  --hardware-confirmation '{HARDWARE_CONFIRMATION}' \
  --shutdown-confirmation '{SHUTDOWN_CONFIRMATION}'
"""


def run_software_safety_offline_audit(input_path: Path, output: Path) -> dict[str, Any]:
    """Generate the requested evidence and mock bundle without vendor construction."""
    source, integrity = load_source_action(input_path)
    source_before = array_sha256(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    installed_paths = {
        "lerobot_wrapper": str(
            Path(
                "/home/jbnu/.lerobot_trossen_ai_data_collection_ui/lerobot/common/"
                "robot_devices/motors/trossen_arm_driver.py"
            )
        ),
        "manipulator": str(
            Path(
                "/home/jbnu/.lerobot_trossen_ai_data_collection_ui/lerobot/common/"
                "robot_devices/robots/manipulator.py"
            )
        ),
        "trossen_header": "/home/jbnu/trossen_arm/include/libtrossen_arm/trossen_arm.hpp",
        "trossen_type_header": (
            "/home/jbnu/trossen_arm/include/libtrossen_arm/trossen_arm_type.hpp"
        ),
        "python_binding_stub": (
            "/home/jbnu/.local/lib/python3.10/site-packages/trossen_arm/trossen_arm.pyi"
        ),
        "installed_static_library": "/usr/local/lib/libtrossen_arm.a",
        "persistent_config": (
            "/home/jbnu/.trossen/trossen_ai_data_collection/configs/robot/"
            "trossen_ai_robots.yaml"
        ),
    }
    connection_audit = {
        "status": "INCONCLUSIVE",
        "classification": StationaryAlohaLowLevelBackend.CONNECTION_CLASSIFICATION,
        "stack": {
            "trossen_ai_data_collection_ui": "1.1.12",
            "lerobot_interbotix_fork": "0.1.0",
            "trossen_arm": "1.9.0",
            "transport": "direct follower Ethernet",
        },
        "paths": installed_paths,
        "persistent_config_static_snapshot": {
            "follower_order": ["left", "right"],
            "left": {
                "ip": "192.168.1.5",
                "model": "V0_FOLLOWER",
                "min_time_to_move_multiplier": 16,
            },
            "right": {
                "ip": "192.168.1.4",
                "model": "V0_FOLLOWER",
                "min_time_to_move_multiplier": 16,
            },
            "leaders_removed_for_backend": True,
            "cameras_removed_for_backend": True,
        },
        "high_level_connect": {
            "api": "ManipulatorRobot.connect() -> TrossenArmDriver.connect()",
            "side_effects": [
                "low-level configure(clear_error=True)",
                "set_all_modes(Mode.position)",
                "set_all_positions(home_pose, 2.0, blocking=False)",
            ],
            "automatic_home": True,
        },
        "low_level_candidate": {
            "api": "trossen_arm.TrossenArmDriver.configure(model, end_effector, ip, False)",
            "source_and_binary_evidence": [
                "installed binary configure relocation 0x16305 calls set_all_modes with enum value 0 (Mode.idle)",
                "installed binary configure then calls set_joint_inputs and receive_robot_output",
                "no set_all_positions relocation/call exists in installed configure function",
                "header daemon continuously sets joint inputs and receives outputs",
            ],
            "home_or_sleep_target": False,
            "goal_position_target": False,
            "mode_change": "all seven joints -> Mode.idle",
            "clear_error": False,
            "state_read": "get_all_positions()",
            "limit_read": "get_joint_limits()",
            "physical_motion_free": "INCONCLUSIVE_NOT_PHYSICALLY_TESTED",
        },
        "low_level_cleanup": {
            "api": "TrossenArmDriver.cleanup(reboot_controller=False)",
            "installed_binary_evidence": [
                "cleanup relocation 0x14cd6 calls set_all_modes with enum value 0 (Mode.idle)",
                "cleanup closes Ethernet transport",
                "no set_all_positions relocation/call exists in installed cleanup function",
            ],
            "home_or_sleep_target": False,
            "position_target": False,
            "mode_change": "Mode.idle",
            "automatic_after_brake_abort": False,
            "destructor_note": "the C++ driver destructor invokes cleanup; process-exit cleanup is not a physical E-stop",
        },
        "decision": (
            "No-home/no-position-target connection is verified from installed code/binary. "
            "A physical no-motion outcome is not claimed because configure actively requests idle brake."
        ),
        "real_connection_opened": False,
    }
    write_json(output / "connection_path_audit.json", connection_audit)

    mode_audit = {
        "status": "VERIFIED_FROM_LOCAL_SOURCE_AND_INSTALLED_BINDING",
        "mode": "Mode.idle (enum value 0)",
        "documented_semantics": "All joints are braked",
        "all_seven_joints": True,
        "api": "TrossenArmDriver.set_all_modes(Mode.idle)",
        "readback_api": "TrossenArmDriver.get_modes() -> list[Mode]",
        "goal_position_required": False,
        "independent_followers": True,
        "holds_exact_mechanical_position": "UNKNOWN_NOT_PHYSICALLY_TESTED",
        "supporting_change_log": (
            "v1.8.2 increased idle-mode joint-1 imax so an extended arm can hold itself horizontally"
        ),
        "limits_in_idle": "v1.8.1 changelog says limit checks are skipped in idle mode",
        "software_estop": False,
        "network_dependency": True,
        "controller_dependency": True,
        "controller_fault_handling": {
            "get_error_information_available": True,
            "normal_vs_fault_string_parser": "NOT_DOCUMENTED",
            "configure_clear_error": False,
            "automatic_recovery": False,
            "runtime_policy": "stop application commands; request idle if communication remains",
            "fault_forces_idle_or_brake": "UNKNOWN_NOT_DOCUMENTED_IN_INSTALLED_API_SOURCES",
        },
        "connection_loss_behavior": {
            "status": "VERIFIED_FROM_LOCAL_CHANGELOG",
            "behavior": "controller returns to idle mode if the driver connection is lost",
            "scope": "connection loss only; not generalized to every controller fault",
        },
        "verified_sources": [
            installed_paths["trossen_type_header"],
            installed_paths["trossen_header"],
            installed_paths["python_binding_stub"],
            "/home/jbnu/trossen_arm/docs/changelog.rst",
        ],
    }
    write_json(output / "mode_idle_audit.json", mode_audit)

    brake_design = {
        "status": "IMPLEMENTED_VERIFIED_OFFLINE",
        "class": "StationaryAlohaLowLevelBackend",
        "function": "brake_abort(reason)",
        "sequence": [
            "set command_loop_stopped before controller calls",
            "request left set_all_modes(Mode.idle)",
            "read back seven left modes",
            "request right set_all_modes(Mode.idle), even if left failed",
            "read back seven right modes",
            "record per-side outcome and timestamps",
        ],
        "never_calls": ["home", "sleep", "disconnect", "set_all_positions"],
        "confirmation_rule": "both sides report seven Mode.idle values",
        "disclaimer": (
            "SOFTWARE BRAKE_ABORT is network/controller-dependent and is not a physical E-stop"
        ),
    }
    write_json(output / "brake_abort_design.json", brake_design)

    watchdog_template = {
        "status": "TEMPLATE_UNREVIEWED",
        "hardware_motion_allowed": False,
        "hardware_replay_allowed": False,
        "controller_limits_verified": False,
        "controller_limits": None,
        "minimum_controller_limit_margin": None,
        "unknown_values_are_null": True,
        "watchdog": {
            name: None for name in WatchdogConfig.__dataclass_fields__
        },
        "stages": {
            stage: {
                "approved": False,
                "active_side": None,
                "joint_indices": None,
                "amplitude_rad": None,
                "duration_sec": None,
                "rate_hz": None,
            }
            for stage in ("B", "C", "D")
        },
    }
    write_json(output / "watchdog_config.template.json", watchdog_template)

    timing_audit = {
        "status": "VERIFIED_FROM_INSTALLED_PYTHON_AND_BINARY",
        "persistent_min_time_to_move_multiplier": 16,
        "wrapper_fps": 30,
        "goal_time_sec": 16 / 30,
        "goal_time_call": (
            "driver.set_all_positions(values, MIN_TIME_TO_MOVE, blocking=False)"
        ),
        "every_goal_uses_same_goal_time": True,
        "binary_behavior": (
            "each set_all_positions call recomputes QuinticHermiteInterpolator coefficients; "
            "the daemon repeatedly evaluates inputs and transmits them"
        ),
        "repeated_30hz_retarget_physical_behavior": "NEEDS_REAL_CHARACTERIZATION",
        "fixed_delay_equivalence": False,
        "speed_scale_audit": {
            "current_validator_behavior": (
                "publication rate = 30 * speed_scale; source samples are not resampled"
            ),
            "source_semantic_timeline": "stretched by 1/speed_scale during execution",
            "controller_goal_time": "unchanged at 16/30 s",
            "safe_conclusion": (
                "speed-scale is not a validated safety control; characterize native 30 Hz first"
            ),
        },
        "sources": [installed_paths["lerobot_wrapper"], installed_paths["trossen_header"]],
    }
    write_json(output / "timing_driver_audit.json", timing_audit)

    motion_audit = source_action_motion_audit(source)
    write_json(output / "source_action_motion_audit.json", motion_audit)

    # Mock no-home connect and two-sided brake success.
    backend, drivers = make_mock_low_level_backend(source[0])
    stage_a_gate, mock_auth = evaluate_stage_a_authorization(
        config_valid=True,
        follower_order_verified=True,
        workspace_clear_confirmed=True,
        physical_left_right_verified=True,
        operator_present_confirmed=True,
        left_power_switch_reachable=True,
        right_power_switch_reachable=True,
        acknowledge_idle_brake_connect_may_move=True,
        acknowledge_idle_cleanup_command=True,
        stage_a_confirmation=STAGE_A_CONFIRMATION,
    )
    stage_a_gate["verified_operator_stop"] = False
    stage_a_gate["verified_operator_stop_required"] = False
    write_json(output / "stage_a_gate_audit.json", stage_a_gate)
    stage_a = run_stage_a_idle_inspection(
        backend=backend,
        authorization=mock_auth,
        operator_stop_verified=False,
    )
    forbidden_calls = {"home", "sleep", "Goal_Position"}
    flattened_calls = [call[0] for driver in drivers.values() for call in driver.calls]
    brake_mock = {
        "status": "PASS",
        "network_access": False,
        "stage_a_status": stage_a["status"],
        "both_idle_confirmed": (
            stage_a["brake_abort"]["status"] == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
        ),
        "no_position_target_during_stage_a": "set_all_positions" not in flattened_calls,
        "no_home_sleep_goal_calls": not bool(forbidden_calls.intersection(flattened_calls)),
        "cleanup_used_high_level_disconnect": False,
        "driver_calls": {side: driver.calls for side, driver in drivers.items()},
    }
    write_json(
        output / "mock_stage_a_results.json",
        {
            **stage_a,
            "status": "PASS" if stage_a["failure"] is None else "FAIL",
            "stage_a_result": stage_a["status"],
            "network_access": False,
            "verified_operator_stop_required": False,
        },
    )

    # Mock partial failure proves right is attempted after a left idle failure.
    partial_backend, partial_drivers = make_mock_low_level_backend(
        source[0], fail_idle_side="left"
    )
    partial_backend.connect_idle_without_home(mock_auth)
    partial = partial_backend.brake_abort("MOCK_LEFT_IDLE_FAILURE")
    brake_mock["partial_failure_test"] = {
        "status": partial["status"],
        "left_confirmed": partial["sides"]["left"]["idle_confirmed"],
        "right_confirmed": partial["sides"]["right"]["idle_confirmed"],
        "right_attempted_after_left_failure": partial["sides"]["right"]["idle_requested"],
        "disconnect_or_cleanup_called": any(
            call[0] == "cleanup" for driver in partial_drivers.values() for call in driver.calls
        ),
    }
    write_json(output / "mock_brake_abort_results.json", brake_mock)

    thresholds = WatchdogConfig(
        max_tracking_error_arm_rad=0.1,
        max_tracking_error_gripper_m=0.01,
        tracking_error_duration_sec=0.05,
        max_command_step_arm_rad=0.2,
        max_command_step_gripper_m=0.01,
        max_command_velocity_arm_rad_s=10.0,
        max_command_velocity_gripper_m_s=0.5,
        max_state_age_sec=0.1,
        max_state_read_duration_sec=0.1,
        max_command_call_duration_sec=0.1,
        max_loop_overrun_sec=0.01,
        controller_position_tolerance=0.01,
    )
    zero = np.zeros(14)

    def watchdog_case(**overrides: Any) -> str:
        values = {
            "command14": zero,
            "actual14": zero,
            "previous_command14": None,
            "dt_sec": 1 / 30,
            "state_age_sec": 0.0,
            "state_read_duration_sec": 0.0,
            "command_call_duration_sec": 0.0,
            "loop_overrun_sec": 0.0,
        }
        values.update(overrides)
        try:
            RuntimeWatchdog(thresholds, mock_controller_limits()).check(**values)
        except WatchdogAbort as error:
            return error.code
        raise AssertionError("mock watchdog case did not abort")

    bad_state = zero.copy()
    bad_state[0] = np.nan
    limit_command = zero.copy()
    limit_command[0] = 11.0
    watchdog_results = {
        "status": "PASS",
        "network_access": False,
        "nonfinite_state": watchdog_case(actual14=bad_state),
        "state_timeout": watchdog_case(state_age_sec=0.2),
        "state_read_timeout": watchdog_case(state_read_duration_sec=0.2),
        "command_call_timeout": watchdog_case(command_call_duration_sec=0.2),
        "loop_overrun": watchdog_case(loop_overrun_sec=0.02),
        "joint_limit": watchdog_case(command14=limit_command),
        "controller_error": watchdog_case(
            controller_fault=True,
            controller_error_information={"left": "MOCK_FAULT"},
        ),
    }
    step_command = zero.copy()
    step_command[0] = 0.21
    watchdog_results["command_step"] = watchdog_case(
        command14=step_command, previous_command14=zero
    )
    velocity_command = zero.copy()
    velocity_command[0] = 0.05
    watchdog_results["command_velocity"] = watchdog_case(
        command14=velocity_command,
        previous_command14=zero,
        dt_sec=0.001,
    )
    mock_abort = AbortController()
    mock_abort.request("USER_ABORT", "Q_KEY")
    watchdog_results["manual_abort"] = watchdog_case(
        abort_controller=mock_abort
    )
    tracking_watchdog = RuntimeWatchdog(thresholds, mock_controller_limits())
    tracking_command = zero.copy()
    tracking_command[0] = 0.2
    tracking_watchdog.check(
        command14=tracking_command,
        actual14=zero,
        previous_command14=None,
        dt_sec=0.03,
        state_age_sec=0.0,
        state_read_duration_sec=0.0,
        command_call_duration_sec=0.0,
        loop_overrun_sec=0.0,
    )
    try:
        tracking_watchdog.check(
            command14=tracking_command,
            actual14=zero,
            previous_command14=None,
            dt_sec=0.03,
            state_age_sec=0.0,
            state_read_duration_sec=0.0,
            command_call_duration_sec=0.0,
            loop_overrun_sec=0.0,
        )
        raise AssertionError("persistent tracking mock did not abort")
    except WatchdogAbort as error:
        watchdog_results["persistent_tracking_error"] = error.code
    write_json(output / "mock_watchdog_results.json", watchdog_results)

    class MockClock:
        def __init__(self) -> None:
            self.value = 1_000_000_000

        def __call__(self) -> int:
            self.value += 1_000
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += int(seconds * 1e9)

    tracking_backend, tracking_drivers = make_mock_low_level_backend(source[0])
    tracking_auth = CharacterizationAuthorization(True, True, True, "B", (), {})
    mock_clock = MockClock()
    tracking_result = run_tracking_characterization_stage(
        backend=tracking_backend,
        authorization=tracking_auth,
        stage="B",
        active_side="left",
        joint_indices=(0,),
        amplitude_rad=0.001,
        duration_sec=0.1,
        watchdog_config=thresholds,
        reviewed_controller_limits=mock_controller_limits(),
        minimum_controller_limit_margin=0.0,
        abort_controller=AbortController(mock_clock),
        clock_ns=mock_clock,
        sleep_fn=mock_clock.sleep,
    )
    mock_tracking = {
        **{key: value for key, value in tracking_result.items() if key != "samples"},
        "status": "PASS" if tracking_result["failure"] is None else "FAIL",
        "stage_result": tracking_result["status"],
        "network_access": False,
        "source_action_loaded": False,
        "source_action_sent": False,
        "starts_at_mock_current": (
            bool(tracking_result["samples"])
            and np.array_equal(
                tracking_result["samples"][0]["command_q"], source[0].astype(float)
            )
        ),
        "ends_at_mock_current": (
            bool(tracking_result["samples"])
            and np.array_equal(
                tracking_result["samples"][-1]["command_q"], source[0].astype(float)
            )
        ),
        "grippers_held": all(
            np.array_equal(np.asarray(row["command_q"])[[6, 13]], source[0, [6, 13]])
            for row in tracking_result["samples"]
        ),
        "mock_position_calls": {
            side: sum(call[0] == "set_joint_position" for call in driver.calls)
            for side, driver in tracking_drivers.items()
        },
    }
    write_json(output / "mock_tracking_characterization_results.json", mock_tracking)

    assert_source_unchanged(input_path, source, integrity)
    if array_sha256(source) != source_before:
        raise Blocked("BLOCKED_SOURCE_ACTION_MUTATION during software safety audit")
    write_json(output / "source_action_integrity.json", integrity)
    (output / "commands.sh").write_text(software_safety_commands_text())
    report = f"""# Stationary ALOHA software brake/watchdog audit

- No-home low-level path: **VERIFIED FROM INSTALLED CODE/BINARY**
- Physically no-motion connection: **INCONCLUSIVE**
- BRAKE_ABORT: **VERIFIED OFFLINE**
- Real hardware connection/motion: **NOT PERFORMED**
- Physical/operator E-stop: **NOT VERIFIED; CURRENT RECORD REMAINS FALSE**

The low-level `configure(..., clear_error=False)` path avoids home, sleep, and
position-target commands, but actively requests `Mode.idle` for all seven joints.
The installed header defines idle as braked; exact physical motion during that
mode transition has not been measured.

`brake_abort()` stops application commands first, attempts both followers even
after a per-side failure, and requires seven-joint idle API readback on both
sides before reporting confirmation. It never calls high-level disconnect.

The 30 Hz characterization runner uses a generated raised-cosine displacement,
keeps both grippers in idle without a gripper position target, and never loads
optimized_action.
Stage A has a separate READ_ONLY_IDLE_INSPECTION gate that does not require a
verified physical stop. Stage B/C/D and VLA replay remain blocked by the factual
unverified physical-stop record.
"""
    (output / "report.md").write_text(report)
    manifest = {
        "status": "OFFLINE_SOFTWARE_SAFETY_AUDIT_COMPLETE",
        "generated_at": datetime.now().isoformat(),
        "real_hardware_connection": False,
        "real_mode_change": False,
        "real_motor_command": False,
        "source_action_modified": False,
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    write_json(output / "run_manifest.json", manifest)
    return manifest


def stage_b_commands_text() -> str:
    python = "/home/jbnu/miniconda3/bin/conda run --no-capture-output -n trossen_ai_data_collection_ui_env python"
    script = str((ROOT / "tools/validate_vla_action_on_real_aloha.py").resolve())
    stage_a = str(DEFAULT_AUTHORITATIVE_STAGE_A_RESULT.resolve())
    reviewed = str(DEFAULT_CHARACTERIZATION_CONFIG.resolve())
    stop = str(DEFAULT_STOP_VERIFICATION.resolve())
    source = str(DEFAULT_INPUT.resolve())
    safety = str(DEFAULT_SAFETY.resolve())
    return f"""#!/usr/bin/env bash
set -euo pipefail

# SAFE OFFLINE regeneration of this preparation bundle.
{python} {script} --stage-b-preparation \\
  --input {source} \\
  --authoritative-stage-a-dir {stage_a}

# DO NOT RUN YET: REAL Stage B. This currently refuses execution because the
# physical/operator stop record is false and the proposed parameters/thresholds
# are intentionally not a reviewed config.
{python} {script} --hardware-characterize-tracking \\
  --characterization-stage B \\
  --authoritative-stage-a-dir {stage_a} \\
  --characterization-config {reviewed} \\
  --stage-b-output-dir {DEFAULT_STAGE_B_TRACKING_OUTPUT.resolve()} \\
  --workspace-clear-confirmed \\
  --physical-left-right-verified \\
  --acknowledge-idle-brake-connect-may-move \\
  --acknowledge-idle-cleanup-command \\
  --prior-characterization-stage-approved \\
  --characterization-side left \\
  --characterization-joints 0 \\
  --characterization-amplitude-rad 0.005 \\
  --characterization-duration-sec 2.0 \\
  --characterization-rate-hz 30.0 \\
  --characterization-confirmation '{CHARACTERIZATION_CONFIRMATION}'

# DO NOT RUN YET: eventual no-object VLA frames 0..29. Stage B does not
# authorize this command; reviewed replay safety and a verified operator stop
# remain independent requirements.
{python} {script} --execute-hardware \\
  --input {source} \\
  --authoritative-stage-a-dir {stage_a} \\
  --safety-config {safety} \\
  --stop-verification-record {stop} \\
  --start-frame 0 --end-frame 29 --speed-scale 1.0 --no-object \\
  --gripper-policy CLAMP_TO_CONTROLLER_LIMIT \\
  --workspace-clear-confirmed \\
  --acknowledge-connect-moves-home \\
  --acknowledge-disconnect-moves-home-sleep \\
  --hardware-confirmation '{HARDWARE_CONFIRMATION}' \\
  --shutdown-confirmation '{SHUTDOWN_CONFIRMATION}'
"""


def run_stage_b_preparation_offline(
    input_path: Path = DEFAULT_INPUT,
    stage_a_directory: Path = DEFAULT_AUTHORITATIVE_STAGE_A_RESULT,
    output: Path = DEFAULT_STAGE_B_PREPARATION_OUTPUT,
) -> dict[str, Any]:
    """Generate adapter and Stage B evidence using only files and mock drivers."""
    input_path = Path(input_path)
    output = Path(output)
    source, integrity = load_source_action(input_path)
    source_hash_before = array_sha256(source)
    npz_hash_before = sha(input_path)
    adapter = StageAHardwareCommandAdapter.from_stage_a_directory(stage_a_directory)
    saturation_audit, adapted_copy = audit_gripper_hardware_saturation(source, adapter)
    if adapted_copy.shape != source.shape:
        raise Blocked("offline hardware command copy changed source shape")
    assert_source_unchanged(input_path, source, integrity)

    state = np.asarray(adapter.authority["state14"], dtype=float)
    recommendation = recommend_first_stage_b_config(
        state,
        adapter.controller_limits,
        stage_a_authority=adapter.authority,
        source=source,
    )
    mock_thresholds = WatchdogConfig(
        max_tracking_error_arm_rad=0.1,
        max_tracking_error_gripper_m=0.01,
        tracking_error_duration_sec=0.2,
        max_command_step_arm_rad=0.01,
        max_command_step_gripper_m=0.001,
        max_command_velocity_arm_rad_s=0.1,
        max_command_velocity_gripper_m_s=0.01,
        max_state_age_sec=0.1,
        max_state_read_duration_sec=0.1,
        max_command_call_duration_sec=0.1,
        max_loop_overrun_sec=0.01,
        controller_position_tolerance=0.01,
    )

    class OfflineClock:
        def __init__(self) -> None:
            self.value = 1_000_000_000

        def __call__(self) -> int:
            self.value += 1_000
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += int(seconds * 1e9)

    backend, drivers = make_mock_low_level_backend(
        state, controller_limits=adapter.controller_limits
    )
    authorization = CharacterizationAuthorization(
        connection_allowed=True,
        motion_allowed=True,
        close_transport_allowed=True,
        stage="B",
        reasons=(),
        states={"MOCK_OFFLINE_ONLY": True},
    )
    clock = OfflineClock()
    stage = recommendation["stages"]["B"]
    mock_result = run_tracking_characterization_stage(
        backend=backend,
        authorization=authorization,
        stage="B",
        active_side=stage["active_side"],
        joint_indices=tuple(stage["joint_indices"]),
        amplitude_rad=float(stage["amplitude_rad"]),
        duration_sec=float(stage["duration_sec"]),
        rate_hz=float(stage["rate_hz"]),
        watchdog_config=mock_thresholds,
        reviewed_controller_limits=adapter.controller_limits,
        minimum_controller_limit_margin=0.1,
        abort_controller=AbortController(clock),
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )
    if mock_result["failure"] is not None:
        raise Blocked(f"mock Stage B failed: {mock_result['failure']}")
    calls = {side: driver.calls for side, driver in drivers.items()}
    forbidden = {"set_all_positions", "set_arm_positions", "set_gripper_position"}
    if any(call[0] in forbidden for values in calls.values() for call in values):
        raise Blocked("mock Stage B reached a forbidden broad/gripper position API")
    left_joint_calls = [
        call for call in calls["left"] if call[0] == "set_joint_position"
    ]
    right_joint_calls = [
        call for call in calls["right"] if call[0] == "set_joint_position"
    ]
    if not left_joint_calls or right_joint_calls or any(
        call[1] != STAGE_B_RECOMMENDED_JOINT for call in left_joint_calls
    ):
        raise Blocked("mock Stage B did not isolate left joint_0")
    samples = mock_result["samples"]
    grippers_held = all(
        np.array_equal(np.asarray(row["command_q"])[[6, 13]], state[[6, 13]])
        for row in samples
    )
    if not grippers_held:
        raise Blocked("mock Stage B changed a gripper reference")

    mock_tracking = {
        **mock_result,
        "offline_mock": True,
        "network_access": False,
        "real_mode_change": False,
        "real_motor_command": False,
        "optimized_action_loaded": False,
        "only_selected_joint_position_commanded": True,
        "grippers_held_in_logged_reference": True,
        "gripper_position_target_sent": False,
        "driver_calls": calls,
    }
    mock_brake = {
        "status": (
            "PASS"
            if mock_result["brake_abort"]["status"]
            == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
            else "FAIL"
        ),
        "result": mock_result["brake_abort"],
        "both_followers_attempted": all(
            mock_result["brake_abort"]["sides"][side]["idle_requested"]
            for side in FOLLOWER_ORDER
        ),
        "home_called": False,
        "sleep_called": False,
        "high_level_disconnect_called": False,
        "network_access": False,
    }
    adapter_record = {
        "status": "IMPLEMENTED_VERIFIED_OFFLINE",
        "class": "StageAHardwareCommandAdapter",
        "policy": "GRIPPER_ONLY_SATURATION_TO_REAL_STAGE_A_CONTROLLER_LIMITS",
        "data_flow": [
            "immutable source action",
            "floating command copy",
            "arm limit rejection without modification",
            "gripper-only saturation",
            "pre-send and post-state runtime watchdog",
            "hardware-bound command",
        ],
        "arm_policy": "UNCHANGED_OR_REJECT_IF_OUTSIDE_LIMIT",
        "gripper_policy": "CLIP_TO_CONTROLLER_REPORTED_JOINT_INDEX_6_MIN_MAX",
        "authoritative_stage_a": adapter.authority,
        "adapted_trajectory_saved": False,
        "eventual_vla_runtime_watchdog": {
            "active_in_run_real_short_replay": True,
            "checks": [
                "finite command/state",
                "controller position and velocity limits",
                "manual abort",
                "tracking error persistence",
                "loop overrun",
            ],
            "abort_response": "BRAKE_ABORT",
        },
    }

    assert_source_unchanged(input_path, source, integrity)
    if sha(input_path) != npz_hash_before or array_sha256(source) != source_hash_before:
        raise Blocked("BLOCKED_SOURCE_ACTION_MUTATION during Stage B preparation")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "gripper_saturation_audit.json", saturation_audit)
    write_json(output / "hardware_command_adapter.json", adapter_record)
    write_json(output / "recommended_stage_b_config.json", recommendation)
    write_json(output / "mock_stage_b_tracking.json", mock_tracking)
    write_json(output / "mock_brake_abort.json", mock_brake)
    write_json(output / "source_action_integrity.json", integrity)
    (output / "commands.sh").write_text(stage_b_commands_text())
    report = f"""# Stationary ALOHA Stage B preparation

- Gripper hardware saturation: **IMPLEMENTED / VERIFIED OFFLINE**
- Real Stage A controller limits: **LOADED AS AUTHORITATIVE**
- Source arm limit violations: **{adapter.authority.get('arm_violation_count')}**
- Left saturated source frames: **{saturation_audit['sides']['left']['saturated_frame_count']}**
- Right saturated source frames: **{saturation_audit['sides']['right']['saturated_frame_count']}**
- First Stage B proposal: **left joint_0, +0.005 rad, 2.0 s, 30 Hz**
- Proposal approval: **UNAPPROVED_DEFAULT**
- Real hardware connection/motion: **NOT PERFORMED**

The adapter creates a transient command copy and saturates only gripper channels
to controller-reported Stage A limits. It never writes an adapted trajectory or
changes the source NPZ. Stage B uses `set_joint_position()` only for the selected
joint; both grippers and every other joint remain in idle/brake and receive no
position target. The 16/30 = 0.533333 s goal time remains unchanged so a future
real Stage B measures the installed 30 Hz publication versus controller
interpolation behavior. BRAKE_ABORT is not a physical E-stop.

Real Stage B remains blocked until the operator-stop record and the proposed
Stage B/watchdog configuration are independently reviewed and approved.
"""
    (output / "report.md").write_text(report)
    manifest = {
        "status": "STAGE_B_PREPARATION_COMPLETE_OFFLINE",
        "generated_at": datetime.now().isoformat(),
        "real_hardware_connection": False,
        "real_mode_change": False,
        "real_motor_command": False,
        "optimized_action_replayed": False,
        "source_action_modified": False,
        "stage_b_real_motion_authorized": False,
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    write_json(output / "run_manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--inspection-output-dir", type=Path, default=DEFAULT_INSPECTION_OUTPUT)
    command.add_argument(
        "--software-safety-output-dir", type=Path, default=DEFAULT_SOFTWARE_SAFETY_OUTPUT
    )
    command.add_argument(
        "--characterization-output-dir", type=Path, default=DEFAULT_CHARACTERIZATION_OUTPUT
    )
    command.add_argument(
        "--stage-a-output-dir", type=Path, default=DEFAULT_STAGE_A_OUTPUT
    )
    command.add_argument(
        "--authoritative-stage-a-dir",
        type=Path,
        default=DEFAULT_AUTHORITATIVE_STAGE_A_RESULT,
    )
    command.add_argument(
        "--stage-b-preparation-output-dir",
        type=Path,
        default=DEFAULT_STAGE_B_PREPARATION_OUTPUT,
    )
    command.add_argument(
        "--stage-b-output-dir", type=Path, default=DEFAULT_STAGE_B_TRACKING_OUTPUT
    )
    command.add_argument(
        "--characterization-config", type=Path, default=DEFAULT_CHARACTERIZATION_CONFIG
    )
    command.add_argument("--safety-config", type=Path, default=DEFAULT_SAFETY)
    command.add_argument(
        "--stop-verification-record", type=Path, default=DEFAULT_STOP_VERIFICATION
    )
    command.add_argument("--inspect", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--hardware-preflight", action="store_true")
    command.add_argument("--software-safety-audit", action="store_true")
    hardware_mode = command.add_mutually_exclusive_group()
    hardware_mode.add_argument("--execute-hardware", action="store_true")
    hardware_mode.add_argument("--hardware-inspect", action="store_true")
    hardware_mode.add_argument("--hardware-characterize-tracking", action="store_true")
    hardware_mode.add_argument("--stage-a-source-audit-dir", type=Path)
    hardware_mode.add_argument("--stage-b-preparation", action="store_true")
    command.add_argument("--speed-scale", type=float, default=0.25)
    command.add_argument("--command-hz", type=float, default=30.0)
    command.add_argument("--start-transition-seconds", type=float, default=5.0)
    command.add_argument("--start-frame", type=int, default=0)
    command.add_argument("--end-frame", type=int)
    command.add_argument("--no-object", action="store_true")
    command.add_argument("--record-actual", action="store_true")
    command.add_argument("--record-camera", action="store_true")
    command.add_argument("--require-confirmation", action="store_true")
    command.add_argument("--gripper-policy", choices=[item.value for item in GripperPolicy], default="REJECT")
    command.add_argument("--workspace-clear-confirmed", action="store_true")
    command.add_argument(
        "--operator-estop-confirmed",
        action="store_true",
        help="DEPRECATED and ignored; cannot authorize any hardware connection or motion",
    )
    command.add_argument("--acknowledge-connect-moves-home", action="store_true")
    command.add_argument("--acknowledge-disconnect-moves-home-sleep", action="store_true")
    command.add_argument("--hardware-confirmation")
    command.add_argument("--shutdown-confirmation")
    command.add_argument("--physical-left-right-verified", action="store_true")
    command.add_argument("--operator-present-confirmed", action="store_true")
    command.add_argument("--left-power-switch-reachable", action="store_true")
    command.add_argument("--right-power-switch-reachable", action="store_true")
    command.add_argument("--inspection-confirmation")
    command.add_argument(
        "--characterization-stage", choices=("A", "B", "C", "D"), default="A"
    )
    command.add_argument("--characterization-side", choices=FOLLOWER_ORDER)
    command.add_argument("--characterization-joints")
    command.add_argument("--characterization-amplitude-rad", type=float)
    command.add_argument("--characterization-duration-sec", type=float)
    command.add_argument("--characterization-rate-hz", type=float)
    command.add_argument("--characterization-confirmation")
    command.add_argument("--stage-a-confirmation")
    command.add_argument(
        "--acknowledge-idle-brake-connect-may-move", action="store_true"
    )
    command.add_argument(
        "--acknowledge-idle-cleanup-command", action="store_true"
    )
    command.add_argument("--prior-characterization-stage-approved", action="store_true")
    command.add_argument("--characterization-run-name")
    command.add_argument("--stage-a-run-name")
    return command


def _static_config(backend: ALOHAHardwareBackend) -> tuple[dict[str, Any], bool, bool, list[str]]:
    try:
        summary = backend.build_config()
        return summary, True, summary["follower_order"] == list(FOLLOWER_ORDER), []
    except Exception as error:  # Static preflight must report import/config errors without connecting.
        return {"status": "BLOCKED", "error": str(error)}, False, False, [str(error)]


def run_hardware_inspection_cli(args: argparse.Namespace) -> int:
    stop_verification, stop_reasons = load_stop_verification(
        args.stop_verification_record
    )
    if stop_reasons:
        raise Blocked("; ".join(stop_reasons))
    backend = ALOHAHardwareBackend()
    config_summary, config_valid, order_valid, config_reasons = _static_config(backend)
    authorization_record, authorization = evaluate_inspection_authorization(
        config_valid=config_valid,
        follower_order_valid=order_valid,
        workspace_clear_confirmed=args.workspace_clear_confirmed,
        operator_estop_confirmed=args.operator_estop_confirmed,
        stop_verification=stop_verification,
        stop_verification_reasons=stop_reasons,
        connect_motion_approved=args.acknowledge_connect_moves_home,
        disconnect_motion_approved=args.acknowledge_disconnect_moves_home_sleep,
        physical_left_right_verified=args.physical_left_right_verified,
        inspection_confirmation=args.inspection_confirmation,
    )
    authorization_record["config_reasons"] = config_reasons
    authorization_record["persistent_config_snapshot"] = config_summary
    authorization_record["stop_verification_record_path"] = str(
        args.stop_verification_record.resolve()
    )
    print(json.dumps({"follower_config": config_summary}, indent=2, default=_json_default))
    print(
        "WARNING: robot.connect() may move both followers to home. "
        "robot.disconnect() may move both followers to home/sleep."
    )
    print("INSPECTION ONLY: optimized_action and send_action are not used.")
    if not authorization.inspection_allowed:
        raise Blocked("HARDWARE INSPECTION REFUSED: " + "; ".join(authorization.reasons))
    output, result = run_hardware_inspection(
        backend=backend,
        authorization=authorization,
        authorization_record=authorization_record,
        config_summary=config_summary,
        output_root=args.inspection_output_dir,
    )
    print(json.dumps({"output": str(output.resolve()), **result}, indent=2))
    return 0


def _parse_characterization_joints(value: str) -> tuple[int, ...]:
    try:
        joints = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise Blocked("--characterization-joints must be comma-separated integers") from error
    if not joints or any(index < 0 or index > 5 for index in joints):
        raise Blocked("--characterization-joints must select arm joint indices 0..5")
    if len(set(joints)) != len(joints):
        raise Blocked("--characterization-joints contains duplicates")
    return joints


def run_stage_a_inspection_cli(args: argparse.Namespace) -> int:
    """Future user-only Stage A path; authorization is independent of motion."""
    motion_cli_values = {
        "--characterization-side": args.characterization_side,
        "--characterization-joints": args.characterization_joints,
        "--characterization-amplitude-rad": args.characterization_amplitude_rad,
        "--characterization-duration-sec": args.characterization_duration_sec,
        "--characterization-rate-hz": args.characterization_rate_hz,
    }
    supplied_motion_values = [
        name for name, value in motion_cli_values.items() if value is not None
    ]
    if supplied_motion_values:
        raise Blocked(
            "BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION: motion options supplied: "
            + ", ".join(supplied_motion_values)
        )

    backend = StationaryAlohaLowLevelBackend()
    config_summary, config_valid, order_valid, config_reasons = _static_config(backend)
    stop_record, stop_reasons = load_stop_verification(args.stop_verification_record)
    authorization_record, authorization = evaluate_stage_a_authorization(
        config_valid=config_valid,
        follower_order_verified=order_valid,
        workspace_clear_confirmed=args.workspace_clear_confirmed,
        physical_left_right_verified=args.physical_left_right_verified,
        operator_present_confirmed=args.operator_present_confirmed,
        left_power_switch_reachable=args.left_power_switch_reachable,
        right_power_switch_reachable=args.right_power_switch_reachable,
        acknowledge_idle_brake_connect_may_move=(
            args.acknowledge_idle_brake_connect_may_move
        ),
        acknowledge_idle_cleanup_command=args.acknowledge_idle_cleanup_command,
        stage_a_confirmation=args.stage_a_confirmation,
    )
    authorization_record.update(
        {
            "persistent_config_snapshot": config_summary,
            "config_reasons": config_reasons,
            "stop_verification_record_path": str(
                args.stop_verification_record.resolve()
            ),
            "operator_stop_verified": verified_operator_stop(
                stop_record, stop_reasons
            ),
            "operator_stop_record_is_authorization_input": False,
            "operator_stop_record": stop_record,
            "operator_stop_record_validation_reasons": stop_reasons,
        }
    )
    print(json.dumps({"follower_config": config_summary}, indent=2, default=_json_default))
    print(
        "WARNING: Stage A low-level configure does not send home/Goal_Position, but requests "
        "Mode.idle on all follower joints; a physical no-motion outcome is NOT guaranteed."
    )
    print(
        "LEFT/RIGHT POWER SWITCH FLAGS CONFIRM REACHABILITY ONLY; THEY ARE NOT E-STOP CLAIMS."
    )
    print("STAGE A NEVER LOADS optimized_action OR ENTERS A TRAJECTORY LOOP.")
    if not authorization.connection_allowed:
        raise Blocked("STAGE A REFUSED: " + "; ".join(authorization.reasons))

    abort = AbortController()
    terminal = ForegroundAbortInput(abort)
    terminal.start()
    try:
        result = run_stage_a_idle_inspection(
            backend=backend,
            authorization=authorization,
            operator_stop_verified=verified_operator_stop(stop_record, stop_reasons),
            abort_controller=abort,
            key_poller=terminal.poll,
            install_signal_handlers=True,
        )
    finally:
        terminal.close()
    output = write_stage_a_inspection_run(
        args.stage_a_output_dir,
        result,
        authorization_record,
        config_summary,
        run_name=args.stage_a_run_name,
    )
    print(json.dumps({"output": str(output.resolve()), **result}, indent=2, default=_json_default))
    return 0 if result["failure"] is None else 2


def run_hardware_characterization_cli(args: argparse.Namespace) -> int:
    """Future user-only Stage B/C/D motion path. It never loads source action."""
    if args.characterization_stage == "A":
        return run_stage_a_inspection_cli(args)

    backend = StationaryAlohaLowLevelBackend()
    config_summary, config_valid, order_valid, config_reasons = _static_config(backend)
    stop_record, stop_reasons = load_stop_verification(args.stop_verification_record)
    authoritative_limits, stage_a_authority = load_authoritative_stage_a_limits(
        args.authoritative_stage_a_dir
    )
    characterization_config, characterization_reasons = load_characterization_config(
        args.characterization_config, args.characterization_stage
    )
    try:
        assert_controller_limits_match_review(
            characterization_config.get("controller_limits"), authoritative_limits
        )
    except (Blocked, KeyError, TypeError, ValueError) as error:
        characterization_reasons.append(
            f"characterization limits do not match authoritative Stage A: {error}"
        )
    authorization_record, authorization = evaluate_characterization_authorization(
        stage=args.characterization_stage,
        config_valid=config_valid,
        follower_order_valid=order_valid,
        workspace_clear_confirmed=args.workspace_clear_confirmed,
        physical_left_right_verified=args.physical_left_right_verified,
        stop_verification=stop_record,
        stop_verification_reasons=stop_reasons,
        acknowledge_idle_brake_connect_may_move=(
            args.acknowledge_idle_brake_connect_may_move
        ),
        acknowledge_idle_cleanup_command=args.acknowledge_idle_cleanup_command,
        characterization_confirmation=args.characterization_confirmation,
        characterization_config=characterization_config,
        characterization_config_reasons=[*config_reasons, *characterization_reasons],
        prior_stage_approved=args.prior_characterization_stage_approved,
    )
    authorization_record.update(
        {
            "persistent_config_snapshot": config_summary,
            "stop_verification_record_path": str(
                args.stop_verification_record.resolve()
            ),
            "characterization_config_path": str(
                args.characterization_config.resolve()
            ),
            "authoritative_stage_a": stage_a_authority,
        }
    )
    print(json.dumps({"follower_config": config_summary}, indent=2, default=_json_default))
    print(
        "WARNING: low-level configure requests Mode.idle; Stage B/C/D then use position targets."
    )
    print(
        "BRAKE_ABORT is software-commanded and network/controller-dependent. "
        "It is not a physical E-stop."
    )
    print("THIS MODE NEVER LOADS OR REPLAYS optimized_action.")
    if not authorization.connection_allowed:
        raise Blocked(
            "LOW-LEVEL CHARACTERIZATION REFUSED: " + "; ".join(authorization.reasons)
        )

    reviewed_stage = characterization_config["stages"][args.characterization_stage]
    active_side = str(reviewed_stage["active_side"])
    joints = tuple(int(value) for value in reviewed_stage["joint_indices"])
    amplitude_rad = float(reviewed_stage["amplitude_rad"])
    duration_sec = float(reviewed_stage["duration_sec"])
    rate_hz = float(reviewed_stage["rate_hz"])
    if args.characterization_side is not None and args.characterization_side != active_side:
        raise Blocked("CLI characterization side differs from reviewed stage parameters")
    if args.characterization_joints is not None and _parse_characterization_joints(
        args.characterization_joints
    ) != joints:
        raise Blocked("CLI characterization joints differ from reviewed stage parameters")
    if (
        args.characterization_amplitude_rad is not None
        and not np.isclose(
            args.characterization_amplitude_rad, amplitude_rad, rtol=0.0, atol=0.0
        )
    ):
        raise Blocked("CLI amplitude differs from reviewed stage parameters")
    if (
        args.characterization_duration_sec is not None
        and not np.isclose(
            args.characterization_duration_sec, duration_sec, rtol=0.0, atol=0.0
        )
    ):
        raise Blocked("CLI duration differs from reviewed stage parameters")
    if (
        args.characterization_rate_hz is not None
        and not np.isclose(
            args.characterization_rate_hz, rate_hz, rtol=0.0, atol=0.0
        )
    ):
        raise Blocked("CLI rate differs from reviewed stage parameters")
    abort = AbortController()
    terminal: ForegroundAbortInput | None = ForegroundAbortInput(abort)
    terminal.start()
    key_poller: Callable[[], str | None] | None = terminal.poll
    try:
        watchdog = WatchdogConfig.from_mapping(characterization_config["watchdog"])
        result = run_tracking_characterization_stage(
            backend=backend,
            authorization=authorization,
            stage=args.characterization_stage,
            active_side=active_side,
            joint_indices=joints,
            amplitude_rad=amplitude_rad,
            duration_sec=duration_sec,
            rate_hz=rate_hz,
            watchdog_config=watchdog,
            reviewed_controller_limits=authoritative_limits,
            minimum_controller_limit_margin=characterization_config.get(
                "minimum_controller_limit_margin"
            ),
            abort_controller=abort,
            key_poller=key_poller,
            install_signal_handlers=True,
        )
    finally:
        if terminal is not None:
            terminal.close()
    if args.characterization_stage == "B":
        output = write_stage_b_tracking_run(
            args.stage_b_output_dir,
            result,
            authorization_record,
            characterization_config,
            stage_a_authority,
            run_name=args.characterization_run_name,
        )
    else:
        output = write_characterization_run(
            args.characterization_output_dir,
            result,
            authorization_record,
            run_name=args.characterization_run_name,
        )
    print(json.dumps({"output": str(output.resolve()), **result}, indent=2, default=_json_default))
    return 0 if result["failure"] is None else 2


def _source_summary(source: np.ndarray, speed_scale: float) -> dict[str, Any]:
    differences = np.diff(source.astype(float), axis=0)
    return {
        "status": "DRY_RUN_COMPLETE",
        "shape": list(source.shape),
        "fps": SOURCE_FPS,
        "speed_scale_for_future_execution": speed_scale,
        "source_timing_modified": False,
        "joint_order": JOINTS,
        "units": UNITS,
        "finite": bool(np.isfinite(source).all()),
        "max_step": float(np.max(np.abs(differences))),
        "estimated_max_velocity_at_speed_scale": float(np.max(np.abs(differences)) * SOURCE_FPS * speed_scale),
        "gripper_ranges": {
            "left": [float(source[:, 6].min()), float(source[:, 6].max())],
            "right": [float(source[:, 13].min()), float(source[:, 13].max())],
        },
        "hardware_connection": "NOT_ATTEMPTED",
        "hardware_command": "NOT_ATTEMPTED",
    }


def run_real_short_replay(
    args: argparse.Namespace,
    source: np.ndarray,
    backend: ALOHAHardwareBackend,
    initial_preflight: dict[str, Any],
    initial_authorization: HardwareAuthorization,
    safety: dict[str, Any],
    stop_verification: dict[str, Any],
    stop_verification_reasons: list[str],
) -> dict[str, Any]:
    """Future operator-only path. It is unreachable without every reviewed gate."""
    if not initial_authorization.connection_allowed:
        raise Blocked("HARDWARE EXECUTION REFUSED: " + "; ".join(initial_authorization.reasons))
    adapter = StageAHardwareCommandAdapter.from_stage_a_directory(
        args.authoritative_stage_a_dir
    )
    if GripperPolicy(safety["gripper_policy"]) is not GripperPolicy.CLAMP_TO_CONTROLLER_LIMIT:
        raise Blocked(
            "future VLA hardware execution requires explicit "
            "CLAMP_TO_CONTROLLER_LIMIT gripper policy"
        )
    hardware_output = args.output_dir / "hardware_execution"
    timing: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    final_authorization = initial_authorization
    abnormal_termination = False
    abort = AbortController()
    terminal = ForegroundAbortInput(abort)
    terminal.start()
    abort.install_signal_handlers()
    try:
        backend.connect(initial_authorization)
        terminal.poll()
        if abort.requested:
            raise WatchdogAbort(
                "MANUAL_ABORT_AFTER_CONNECT", abort.source or "unknown source"
            )
        actual = backend.read_state()
        controller_limits = backend.read_controller_limits()
        write_json(hardware_output / "controller_limits.json", controller_limits)
        assert_controller_limits_match_review(
            controller_limits, adapter.controller_limits
        )
        assert_controller_limits_match_review(controller_limits, safety["controller_limits"])
        for side, index in (("left", 6), ("right", 13)):
            joint = controller_limits[side][6]
            reviewed_min = float(safety[f"{side}_gripper_min"])
            reviewed_max = float(safety[f"{side}_gripper_max"])
            if not np.isclose(joint["position_min"], reviewed_min, rtol=0.0, atol=1e-9):
                raise Blocked(f"{side} gripper minimum differs from reviewed value")
            if not np.isclose(joint["position_max"], reviewed_max, rtol=0.0, atol=1e-9):
                raise Blocked(f"{side} gripper maximum differs from reviewed value")
        target = source[args.start_frame]
        start_limit = _maximum_allowed(safety["max_start_position_error"], "max_start_position_error")
        start_state_valid = bool(np.all(np.abs(actual.state14 - target) <= start_limit))
        final_preflight, final_authorization = evaluate_preflight(
            source_valid=True,
            config_valid=True,
            follower_order_valid=True,
            safety=safety,
            safety_reasons=[],
            stop_verification=stop_verification,
            stop_verification_reasons=stop_verification_reasons,
            workspace_ack=args.workspace_clear_confirmed,
            estop_ack=args.operator_estop_confirmed,
            connect_ack=args.acknowledge_connect_moves_home,
            disconnect_ack=args.acknowledge_disconnect_moves_home_sleep,
            execute_hardware=True,
            dry_run=False,
            hardware_confirmation=args.hardware_confirmation,
            shutdown_confirmation=args.shutdown_confirmation,
            start_state_valid=start_state_valid,
        )
        if not final_authorization.replay_allowed:
            raise Blocked("HARDWARE EXECUTION REFUSED AFTER STATE READ: " + "; ".join(final_preflight["reasons"]))
        transition = minimum_jerk_transition(
            actual.state14, target, args.start_transition_seconds, args.command_hz, hold_gripper=True
        )
        end_frame = args.end_frame if args.end_frame is not None else min(args.start_frame + 29, len(source) - 1)
        selected = source[args.start_frame : end_frame + 1].astype(float, copy=True)
        validate_motion_thresholds(transition, args.command_hz, safety)
        validate_motion_thresholds(selected, SOURCE_FPS * args.speed_scale, safety)
        boundary = np.vstack((transition[-1], selected[0]))
        validate_motion_thresholds(boundary, SOURCE_FPS * args.speed_scale, safety)
        monitor = TrackingErrorMonitor(safety["max_tracking_error"], safety["tracking_error_duration"])
        maximum_overrun_ns = int(float(safety["max_loop_overrun"]) * 1e9)
        runtime_watchdog = RuntimeWatchdog(
            WatchdogConfig(max_loop_overrun_sec=float(safety["max_loop_overrun"])),
            controller_limits,
        )
        previous_sent = actual.state14.copy()
        previous_actual = actual.state14.copy()
        previous_loop_ns: int | None = None

        def execute_phase(commands: np.ndarray, rate_hz: float, source_offset: int | None) -> None:
            nonlocal previous_sent, previous_actual, previous_loop_ns
            period_ns = int(round(1e9 / rate_hz))
            phase_start_ns = time.monotonic_ns()
            for index, command14 in enumerate(commands):
                terminal.poll()
                if abort.requested:
                    raise WatchdogAbort(
                        "MANUAL_ABORT", abort.source or "unknown source"
                    )
                if backend.command_loop_stopped:
                    break
                scheduled_ns = phase_start_ns + index * period_ns
                remaining_ns = scheduled_ns - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1e9)
                terminal.poll()
                if abort.requested:
                    raise WatchdogAbort(
                        "MANUAL_ABORT_BEFORE_COMMAND", abort.source or "unknown source"
                    )
                loop_start_ns = time.monotonic_ns()
                state_read_start_ns = time.monotonic_ns()
                state_sample = backend.read_state()
                state_read_done_ns = time.monotonic_ns()
                loop_period_ns = 0 if previous_loop_ns is None else loop_start_ns - previous_loop_ns
                monitor.check(previous_sent, state_sample.state14, max(loop_period_ns, period_ns) / 1e9)
                adapted_command, adaptation = adapter.adapt(
                    command14,
                    source_frame=(
                        None if source_offset is None else source_offset + index
                    ),
                )
                runtime_watchdog.check_command_before_send(
                    command14=adapted_command,
                    previous_command14=previous_sent,
                    dt_sec=max(loop_period_ns, period_ns) / 1e9,
                    commanded_indices=tuple(range(14)),
                    abort_controller=abort,
                )
                send_start_ns = time.monotonic_ns()
                sent = backend.send_action(
                    adapted_command,
                    final_authorization,
                    controller_limits,
                    GripperPolicy.REJECT,
                )
                send_done_ns = time.monotonic_ns()
                loop_overrun_ns = max(0, send_done_ns - (scheduled_ns + period_ns))
                runtime_watchdog.check(
                    command14=sent,
                    actual14=state_sample.state14,
                    previous_command14=previous_sent,
                    previous_actual14=previous_actual,
                    dt_sec=max(loop_period_ns, period_ns) / 1e9,
                    state_age_sec=max(
                        0.0,
                        (send_done_ns - state_sample.host_monotonic_timestamp_ns)
                        / 1e9,
                    ),
                    state_read_duration_sec=(
                        state_read_done_ns - state_read_start_ns
                    )
                    / 1e9,
                    command_call_duration_sec=(send_done_ns - send_start_ns) / 1e9,
                    loop_overrun_sec=loop_overrun_ns / 1e9,
                    abort_controller=abort,
                    commanded_indices=tuple(range(14)),
                )
                if loop_overrun_ns > maximum_overrun_ns:
                    raise Blocked("CONTROL_LOOP_OVERRUN exceeds reviewed max_loop_overrun")
                source_frame = -1 if source_offset is None else source_offset + index
                timing.append(
                    {
                        "clock_mode": "HOST_MONOTONIC_REAL",
                        "scheduled_timestamp_ns": scheduled_ns,
                        "actual_host_send_timestamp_ns": send_done_ns,
                        "state_read_timestamp_ns": state_sample.host_monotonic_timestamp_ns,
                        "loop_period_ns": loop_period_ns,
                        "loop_overrun_ns": loop_overrun_ns,
                        "command_age_ns": max(0, send_start_ns - scheduled_ns),
                        "command_send_duration_ns": send_done_ns - send_start_ns,
                        "source_frame": source_frame,
                    }
                )
                state_rows.append(
                    {
                        "source_frame": source_frame,
                        "host_monotonic_timestamp_ns": state_sample.host_monotonic_timestamp_ns,
                        "actual_state14": state_sample.state14.tolist(),
                        "requested_action14": validate_action14(command14).tolist(),
                        "hardware_command_after_saturation": adapted_command.tolist(),
                        "hardware_command_adaptation": adaptation,
                        "sent_action14": sent.tolist(),
                    }
                )
                previous_sent = sent
                previous_actual = state_sample.state14.copy()
                previous_loop_ns = loop_start_ns

        execute_phase(transition, args.command_hz, None)
        execute_phase(selected, SOURCE_FPS * args.speed_scale, args.start_frame)
        result = {
            "status": "COMPLETED",
            "controller_limits": controller_limits,
            "timing_rows": len(timing),
            "state_rows": len(state_rows),
            "source_frame_range": [args.start_frame, end_frame],
            "source_action_modified": False,
            "hardware_adapter": adapter.authority,
            "gripper_policy": "GRIPPER_ONLY_SATURATION_TO_REAL_STAGE_A_LIMITS",
            "runtime_watchdog_active": True,
        }
        write_json(hardware_output / "timing.json", timing)
        write_json(hardware_output / "state_and_command.json", state_rows)
        write_json(hardware_output / "result.json", result)
        return result
    except Exception as error:
        abnormal_termination = True
        brake = backend.brake_abort(
            f"VLA_SHORT_REPLAY_ABORT:{type(error).__name__}:{error}"
        )
        write_json(
            hardware_output / "software_brake_abort.json",
            {
                "trigger": f"{type(error).__name__}: {error}",
                "brake_abort": brake,
                "automatic_disconnect_called": False,
            },
        )
        raise
    finally:
        backend.stop_command_loop()
        try:
            if not abnormal_termination and final_authorization.normal_shutdown_allowed:
                backend.normal_shutdown(final_authorization)
        finally:
            abort.restore_signal_handlers()
            terminal.close()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.stage_b_preparation:
        incompatible = [
            name
            for name, enabled in (
                ("--dry-run", args.dry_run),
                ("--inspect", args.inspect),
                ("--hardware-preflight", args.hardware_preflight),
                ("--software-safety-audit", args.software_safety_audit),
                ("--record-actual", args.record_actual),
                ("--record-camera", args.record_camera),
            )
            if enabled
        ]
        if incompatible:
            raise Blocked(
                "STAGE B OFFLINE PREPARATION REFUSED: incompatible options "
                + ", ".join(incompatible)
            )
        manifest = run_stage_b_preparation_offline(
            args.input,
            args.authoritative_stage_a_dir,
            args.stage_b_preparation_output_dir,
        )
        print(json.dumps(manifest, indent=2, default=_json_default))
        return 0
    if args.stage_a_source_audit_dir is not None:
        incompatible = [
            name
            for name, enabled in (
                ("--dry-run", args.dry_run),
                ("--inspect", args.inspect),
                ("--hardware-preflight", args.hardware_preflight),
                ("--software-safety-audit", args.software_safety_audit),
            )
            if enabled
        ]
        if incompatible:
            raise Blocked(
                "STAGE A OFFLINE SOURCE AUDIT REFUSED: incompatible options "
                + ", ".join(incompatible)
            )
        result = run_stage_a_source_limit_audit(
            args.stage_a_source_audit_dir, args.input
        )
        print(json.dumps(result, indent=2, default=_json_default))
        return 0
    if args.hardware_characterize_tracking:
        incompatible = [
            name
            for name, enabled in (
                ("--dry-run", args.dry_run),
                ("--inspect", args.inspect),
                ("--hardware-preflight", args.hardware_preflight),
                ("--software-safety-audit", args.software_safety_audit),
                ("--record-actual", args.record_actual),
                ("--record-camera", args.record_camera),
            )
            if enabled
        ]
        if incompatible:
            raise Blocked(
                "HARDWARE CHARACTERIZATION REFUSED: incompatible options "
                + ", ".join(incompatible)
            )
        # This independent path never calls load_source_action().
        return run_hardware_characterization_cli(args)
    if args.hardware_inspect:
        incompatible = [
            name
            for name, enabled in (
                ("--dry-run", args.dry_run),
                ("--inspect", args.inspect),
                ("--hardware-preflight", args.hardware_preflight),
                ("--record-actual", args.record_actual),
                ("--record-camera", args.record_camera),
            )
            if enabled
        ]
        if incompatible:
            raise Blocked(
                "HARDWARE INSPECTION REFUSED: incompatible options " + ", ".join(incompatible)
            )
        # Inspection is an independent path: no source NPZ, replay safety, or action code is loaded.
        return run_hardware_inspection_cli(args)
    if args.software_safety_audit:
        incompatible = [
            name
            for name, enabled in (
                ("--dry-run", args.dry_run),
                ("--hardware-preflight", args.hardware_preflight),
                ("--execute-hardware", args.execute_hardware),
            )
            if enabled
        ]
        if incompatible:
            raise Blocked(
                "SOFTWARE SAFETY AUDIT REFUSED: incompatible options "
                + ", ".join(incompatible)
            )
        manifest = run_software_safety_offline_audit(
            args.input, args.software_safety_output_dir
        )
        print(json.dumps(manifest, indent=2, default=_json_default))
        return 0
    if args.speed_scale <= 0 or args.command_hz <= 0 or args.start_transition_seconds <= 0:
        raise Blocked("speed-scale, command-hz, and start-transition-seconds must be positive")
    if args.start_frame < 0 or args.end_frame is not None and args.end_frame < args.start_frame:
        raise Blocked("invalid frame range")
    if args.execute_hardware and args.dry_run:
        raise Blocked("HARDWARE EXECUTION REFUSED: --execute-hardware and --dry-run are mutually exclusive")

    source, integrity = load_source_action(args.input)
    if args.start_frame >= len(source) or args.end_frame is not None and args.end_frame >= len(source):
        raise Blocked("frame range exceeds source trajectory")
    source_valid = source.shape == (990, 14) and bool(np.isfinite(source).all())
    gripper = gripper_policy_audit(source)
    gripper["requested_cli_policy"] = args.gripper_policy
    backend = ALOHAHardwareBackend()

    needs_static_config = args.hardware_preflight or args.execute_hardware
    if needs_static_config:
        config_summary, config_valid, order_valid, config_reasons = _static_config(backend)
    else:
        config_summary = {
            "status": "NOT_LOADED_OFFLINE",
            "reason": "dry-run does not import or instantiate the hardware stack",
            "follower_order_expected": list(FOLLOWER_ORDER),
        }
        config_valid, order_valid, config_reasons = True, True, []

    safety, safety_reasons = load_safety_config(args.safety_config)
    stop_verification, stop_verification_reasons = load_stop_verification(
        args.stop_verification_record
    )
    if (
        args.execute_hardware
        and safety.get("gripper_policy") is not None
        and safety.get("gripper_policy") != args.gripper_policy
    ):
        safety_reasons.append(
            "CLI gripper policy differs from the reviewed safety gripper_policy"
        )
    safety_reasons = [*config_reasons, *safety_reasons]
    preflight, authorization = evaluate_preflight(
        source_valid=source_valid,
        config_valid=config_valid,
        follower_order_valid=order_valid,
        safety=safety,
        safety_reasons=safety_reasons,
        stop_verification=stop_verification,
        stop_verification_reasons=stop_verification_reasons,
        workspace_ack=args.workspace_clear_confirmed,
        estop_ack=args.operator_estop_confirmed,
        connect_ack=args.acknowledge_connect_moves_home,
        disconnect_ack=args.acknowledge_disconnect_moves_home_sleep,
        execute_hardware=args.execute_hardware,
        dry_run=args.dry_run or not args.execute_hardware,
        hardware_confirmation=args.hardware_confirmation,
        shutdown_confirmation=args.shutdown_confirmation,
        start_state_valid=False,
    )

    dry_run_results = _source_summary(source, args.speed_scale)
    mock_results: dict[str, Any] | None = None
    timing: list[dict[str, Any]] | None = None
    if args.dry_run:
        mock_results, timing = run_mock_offline(source, args.start_transition_seconds, args.command_hz)
    assert_source_unchanged(args.input, source, integrity)
    write_integration_reports(
        args.output_dir,
        integrity=integrity,
        config_summary=config_summary,
        preflight=preflight,
        dry_run=dry_run_results,
        mock_results=mock_results,
        gripper=gripper,
        timing=timing,
    )

    result: dict[str, Any] = {
        "backend": "HARDWARE BACKEND IMPLEMENTED",
        "mode": "HARDWARE" if args.execute_hardware else "STATIC_HARDWARE_PREFLIGHT" if args.hardware_preflight else "DRY_RUN" if args.dry_run else "INSPECT",
        "preflight": preflight,
        "output_dir": str(args.output_dir.resolve()),
        "source_action_unchanged": integrity["source_action_unchanged"],
        "operator_stop_verification": {
            "path": str(args.stop_verification_record.resolve()),
            "verified": verified_operator_stop(
                stop_verification, stop_verification_reasons
            ),
            "reasons": stop_verification_reasons,
        },
        "real_robot_connection_attempted": False,
    }
    if args.execute_hardware:
        if not authorization.connection_allowed:
            raise Blocked("HARDWARE EXECUTION REFUSED: " + "; ".join(authorization.reasons))
        # From this point onward physical motion is possible. No offline test reaches this line.
        hardware_result = run_real_short_replay(
            args,
            source,
            backend,
            preflight,
            authorization,
            safety,
            stop_verification,
            stop_verification_reasons,
        )
        result["hardware_result"] = hardware_result
        result["real_robot_connection_attempted"] = True
        audit_path = args.output_dir / "hardware_backend_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["real_connection_opened"] = True
        audit["real_command_executed"] = hardware_result["state_rows"] > 0
        write_json(audit_path, audit)
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Blocked as error:
        print(f"BLOCKED: {error}")
        raise SystemExit(2)
