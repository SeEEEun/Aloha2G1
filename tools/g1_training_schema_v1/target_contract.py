from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CANONICAL_JOINT_NAMES, FPS, STATE_DIM
from .source_audit import sha256_file

EPISODE_RE = re.compile(r"episode_(\d{6})$")
FAILURE_FRAME_PATTERNS = (
    re.compile(r"first failed frame\s*=\s*(\d+)"),
    re.compile(r"first frame\s*=\s*(\d+)"),
    re.compile(r"frame\s*=\s*(\d+)"),
)


@dataclass(frozen=True)
class RetargetedTrajectory:
    episode_id: int
    q: np.ndarray
    timestamps: np.ndarray
    fps: float
    input_joint_names: tuple[str, ...]
    reorder_indices: tuple[int, ...]
    trajectory_path: Path
    trajectory_sha256: str


@dataclass(frozen=True)
class EpisodeDecision:
    episode_id: int
    accepted: bool
    status: str
    failure_category: str | None
    first_failed_frame: int | None
    source_hash: str | None
    retarget_output_reference: str
    retarget_output_sha256: str | None
    validation_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "accepted": self.accepted,
            "status": self.status,
            "failure_category": self.failure_category,
            "first_failed_frame": self.first_failed_frame,
            "source_hash": self.source_hash,
            "retarget_output_reference": self.retarget_output_reference,
            "retarget_output_sha256": self.retarget_output_sha256,
            "validation_path": self.validation_path,
        }


def discover_episode_dirs(root: str | Path) -> dict[int, Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"retarget input root does not exist: {root}")
    result: dict[int, Path] = {}
    for path in sorted(root.iterdir()):
        match = EPISODE_RE.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        episode_id = int(match.group(1))
        if episode_id in result:
            raise ValueError(f"duplicate retarget episode {episode_id}")
        result[episode_id] = path
    if not result:
        raise ValueError(f"no episode_XXXXXX directories under {root}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _first_failure_frame(validation: dict[str, Any]) -> int | None:
    causal = validation.get("first_causal_failure") or {}
    if not isinstance(causal, dict):
        return None
    for key in ("first_failed_frame", "first_frame", "frame"):
        if key in causal and causal[key] is not None:
            return int(causal[key])
    reason = str(causal.get("reason", ""))
    for pattern in FAILURE_FRAME_PATTERNS:
        match = pattern.search(reason)
        if match:
            return int(match.group(1))
    return None


def inspect_episode(episode_dir: str | Path) -> EpisodeDecision:
    episode_dir = Path(episode_dir).resolve()
    match = EPISODE_RE.fullmatch(episode_dir.name)
    if not match:
        raise ValueError(f"invalid episode directory name: {episode_dir.name}")
    episode_id = int(match.group(1))
    validation_path = episode_dir / "validation.json"
    validation = _read_json(validation_path)
    status = str(validation.get("status", "MISSING_VALIDATION"))
    accepted = bool(validation.get("pass", False)) and status == "PASS"
    source_metadata = _read_json(episode_dir / "source_metadata.json")
    source_hash = source_metadata.get("source_action_sha256")
    if not source_hash:
        raise ValueError(f"{episode_dir}: source_metadata.json lacks required source_action_sha256")
    trajectory_path = episode_dir / "g1_full_action.npz"
    return EpisodeDecision(
        episode_id=episode_id,
        accepted=accepted,
        status=status,
        failure_category=None if accepted else status,
        first_failed_frame=None if accepted else _first_failure_frame(validation),
        source_hash=str(source_hash),
        retarget_output_reference=str(trajectory_path),
        retarget_output_sha256=sha256_file(trajectory_path) if trajectory_path.is_file() else None,
        validation_path=str(validation_path),
    )


def load_retargeted_trajectory(episode_dir: str | Path) -> RetargetedTrajectory:
    episode_dir = Path(episode_dir).resolve()
    match = EPISODE_RE.fullmatch(episode_dir.name)
    if not match:
        raise ValueError(f"invalid episode directory name: {episode_dir.name}")
    path = episode_dir / "g1_full_action.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing retarget trajectory: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"action", "timestamps", "fps", "joint_names"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        action = np.asarray(archive["action"], dtype=np.float32)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        fps = float(np.asarray(archive["fps"]).item())
        input_names = tuple(str(name) for name in np.asarray(archive["joint_names"]).tolist())
    if action.ndim != 2 or action.shape[1] != STATE_DIM:
        raise ValueError(f"{path}: expected [T,{STATE_DIM}] action, got {action.shape}")
    if len(input_names) != STATE_DIM or len(set(input_names)) != STATE_DIM:
        raise ValueError(f"{path}: joint_names must contain {STATE_DIM} unique entries")
    if set(input_names) != set(CANONICAL_JOINT_NAMES):
        missing = sorted(set(CANONICAL_JOINT_NAMES) - set(input_names))
        extra = sorted(set(input_names) - set(CANONICAL_JOINT_NAMES))
        raise ValueError(f"{path}: joint-name contract mismatch; missing={missing}, extra={extra}")
    if timestamps.shape != (action.shape[0],):
        raise ValueError(f"{path}: timestamp shape {timestamps.shape} != ({action.shape[0]},)")
    if not np.isfinite(action).all() or not np.isfinite(timestamps).all():
        raise ValueError(f"{path}: non-finite state/action or timestamp")
    if fps != FPS:
        raise ValueError(f"{path}: fps {fps} != {FPS}")
    expected_ts = np.arange(action.shape[0], dtype=np.float64) / FPS
    if not np.allclose(timestamps, expected_ts, rtol=0.0, atol=1e-5):
        raise ValueError(f"{path}: timestamps are not frame_index/fps")
    reorder = tuple(input_names.index(name) for name in CANONICAL_JOINT_NAMES)
    q = np.ascontiguousarray(action[:, reorder], dtype=np.float32)
    return RetargetedTrajectory(
        episode_id=int(match.group(1)),
        q=q,
        timestamps=timestamps,
        fps=fps,
        input_joint_names=input_names,
        reorder_indices=reorder,
        trajectory_path=path,
        trajectory_sha256=sha256_file(path),
    )


def build_target_contract() -> dict[str, Any]:
    return {
        "audit_status": "PASS",
        "policy_scope": "Unitree G1 fixed-base tabletop manipulation",
        "learned_control_dimension": 28,
        "learned_groups": {"left_arm": 7, "right_arm": 7, "left_Dex3": 7, "right_Dex3": 7},
        "excluded_from_policy": [
            "floating-base qpos",
            "left/right leg joints",
            "waist joints",
            "walking/base velocity/body-height commands",
            "TrajBooster Manager/Worker and lower-body RL",
        ],
        "compatibility_constants": (
            "A downstream execution adapter must hold base, legs, and waist at its independently verified "
            "fixed-base stand targets; these constants are not stored as learned features."
        ),
        "full_model_qpos_dimension": 50,
        "full_model_qpos_is_policy_action": False,
        "arm_transport": "absolute G1 arm motor q targets, motors 15..28",
        "hand_transport": "absolute left/right Dex3 q targets, DDS motor indices 0..6 per hand",
        "integrated_export_note": (
            "Integrated Arm-v2 exporter emits semantic joint_names; the packager performs a lossless "
            "name-based column permutation into physical Dex3 DDS order."
        ),
        "real_recording_contract": {
            "schema": "g1_behavior_v1",
            "actual_arrays": [
                "actual_arm_qpos",
                "actual_left_dex3_qpos",
                "actual_right_dex3_qpos",
            ],
            "name_arrays": [
                "arm_joint_names",
                "left_dex3_joint_names",
                "right_dex3_joint_names",
            ],
            "policy_adapter": (
                "concatenate the three actual qpos arrays after exact-name remapping into target_joint_order.json; "
                "never use full floating-base qpos directly"
            ),
        },
        "sources_of_truth": [
            "/home/jbnu/jaeyoung/unitree/unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py",
            "/home/jbnu/jaeyoung/unitree/unitree_sdk2/example/g1/dex3/g1_dex3_example.cpp",
            "/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml",
            "/home/jbnu/aloha_g1_dataset/tools/record_g1_behavior_readonly.py",
            "/home/jbnu/aloha_g1_dataset/tools/g1_behavior_schema.py",
            "/home/jbnu/aloha_g1_dataset/tools/record_g1_dex3_magsafe_primitives.py",
            "/home/jbnu/aloha_g1_dataset/tools/aloha_g1_arm_v2/integrated.py",
        ],
    }
