"""Common offline ALOHA -> G1 dataset retargeting implementation.

The baseline and proposed methods deliberately share the source loader, source
FK, G1 model, wrist IK, numerical tolerances, temporal regularization,
collision classifier, validation, and exporter.  They differ only in the
representation builder and hand mapper.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import pyarrow.dataset as _pyarrow_dataset  # noqa: F401
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
DEFAULT_DATASET = ROOT / "lerobot_magsafe_50_cam_high_v3"
DEFAULT_OUTPUT = ROOT / "outputs/g1_dataset_retargeting_v1"
DEFAULT_CONFIG = DEFAULT_OUTPUT / "config/aloha_g1_retargeting_v1.json"

# The audited ALOHA validator only uses pandas in standalone dataset/plotting
# entry points.  Its FK routines do not, so permit the retargeter to run in the
# project's Isaac environment where pandas is intentionally absent.
try:  # pragma: no cover - environment-dependent compatibility
    import pandas as _pandas  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

# These imports are existing, audited project implementations.
import validate_smolvla_in_stationary_aloha_mujoco as aloha_fk  # noqa: E402
import validate_g1_targets_and_sparse_ik as g1_ik  # noqa: E402
from aloha_g1_v15.kinematics import ActiveG1Dex3  # noqa: E402
from aloha_magsafe_semantics.gripper_phase import (  # noqa: E402
    PHASE_NAMES,
    GripperResult,
    detect_gripper_phases,
)


METHODS = ("baseline", "proposed")
FINAL_STATUSES = (
    "PASS",
    "FAIL_IK",
    "FAIL_JOINT_LIMIT",
    "FAIL_COLLISION",
    "FAIL_FRAME_MAPPING",
    "FAIL_DATA",
    "FAIL_HAND_MAPPING",
    "FAIL_OTHER",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def implementation_fingerprint() -> tuple[str, dict[str, str]]:
    paths = [
        ROOT / "tools/aloha_g1_dataset_v1/__init__.py",
        ROOT / "tools/aloha_g1_dataset_v1/core.py",
        ROOT / "tools/aloha_g1_dataset_v1/artifacts.py",
        ROOT / "tools/retarget_aloha_g1_dataset.py",
    ]
    file_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    digest = hashlib.sha256()
    for name, value in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), file_hashes


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **value)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    names = fieldnames or (list(rows[0]) if rows else [])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "aloha_g1_retargeting_v1":
        raise ValueError(f"unexpected retargeter config schema: {path}")
    if not value.get("offline_only") or value.get("real_robot_command_allowed"):
        raise ValueError("v1 configuration must be offline-only")
    return value


def require_hash(path: str | Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected={expected}, actual={actual}, path={path}")


def fixed_list_numpy(column: Any, width: int) -> np.ndarray:
    chunks = column.combine_chunks()
    values = chunks.values.to_numpy(zero_copy_only=False)
    return np.asarray(values).reshape(len(chunks), width)


def scalar_numpy(column: Any, dtype: Any) -> np.ndarray:
    return np.asarray(column.combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def parse_action_channels(action: np.ndarray) -> dict[str, np.ndarray]:
    """Parse the verified stationary-ALOHA 14-D follower target convention."""
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 14 or not np.isfinite(action).all():
        raise ValueError(f"ALOHA action must be finite [T,14], got {action.shape}")
    return {
        "left_arm": action[:, 0:6],
        "left_gripper": action[:, 6],
        "right_arm": action[:, 7:13],
        "right_gripper": action[:, 13],
    }


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = np.asarray(rotation, dtype=np.float64)
    if translation is not None:
        value[:3, 3] = np.asarray(translation, dtype=np.float64)
    return value


def inverse_transform(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def rotation_error_vector(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(target) @ np.asarray(current).T).as_rotvec()


def rotation_errors(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("tij,tkj->tik", target, current)
    return np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=1)


def stats(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(value) == 0:
        return {key: 0.0 for key in ("mean", "median", "max", "p95", "p99")}
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "max": float(np.max(value)),
        "p95": float(np.percentile(value, 95)),
        "p99": float(np.percentile(value, 99)),
    }


def phase_runs(labels: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
    labels = np.asarray(labels).astype(str)
    result: list[dict[str, Any]] = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            result.append({
                "phase": str(labels[start]),
                "start_frame": int(start),
                "end_frame": int(end - 1),
                "start_time_sec": float(timestamps[start]),
                "end_time_sec": float(timestamps[end - 1]),
                "frames": int(end - start),
            })
            start = end
    return result


@dataclass(frozen=True)
class SourceEpisode:
    episode_id: int
    action: np.ndarray
    state: np.ndarray
    timestamps: np.ndarray
    frame_index: np.ndarray
    task_index: np.ndarray
    task: str
    nominal_fps: float
    source_folder: str
    source_parquet: str
    image_reference: dict[str, Any]

    @property
    def fps(self) -> float:
        return float(self.nominal_fps)


class SourceDataset:
    """Validated in-memory view of the authoritative integrated dataset."""

    def __init__(self, root: str | Path = DEFAULT_DATASET, config: dict[str, Any] | None = None):
        self.root = Path(root).resolve()
        self.config = config or load_config()
        info_path = self.root / "meta/info.json"
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        expected = self.config["source_dataset"]
        if self.root == Path(expected["root"]).resolve():
            require_hash(info_path, expected["info_sha256"], "dataset info")
        if self.info.get("codebase_version") != "v3.0":
            raise RuntimeError("authoritative dataset is not LeRobot v3.0")
        if int(self.info.get("total_episodes", -1)) != int(expected["expected_episodes"]):
            raise RuntimeError(
                "authoritative dataset episode count differs from immutable configuration"
            )
        if float(self.info.get("fps", 0.0)) != float(expected["fps"]):
            raise RuntimeError("dataset FPS mismatch")

        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if data_files != [self.root / "data/chunk-000/file-000.parquet"]:
            raise RuntimeError(f"unexpected authoritative data shards: {data_files}")
        if self.root == Path(expected["root"]).resolve():
            require_hash(data_files[0], expected["data_sha256"], "dataset data")
        table = pq.read_table(
            data_files[0],
            columns=[
                "observation.state", "action", "timestamp", "frame_index",
                "episode_index", "index", "task_index",
            ],
        )
        self.state = fixed_list_numpy(table["observation.state"], 14).astype(np.float64)
        self.action = fixed_list_numpy(table["action"], 14).astype(np.float64)
        self.timestamp = scalar_numpy(table["timestamp"], np.float64)
        self.frame_index = scalar_numpy(table["frame_index"], np.int64)
        self.episode_index = scalar_numpy(table["episode_index"], np.int64)
        self.global_index = scalar_numpy(table["index"], np.int64)
        self.task_index = scalar_numpy(table["task_index"], np.int64)
        if self.action.shape != (int(self.info["total_frames"]), 14):
            raise RuntimeError(f"dataset action shape mismatch: {self.action.shape}")
        if not all(np.isfinite(value).all() for value in (self.action, self.state, self.timestamp)):
            raise RuntimeError("dataset contains NaN or infinity")
        actual_ids = np.unique(self.episode_index)
        if not np.array_equal(actual_ids, np.arange(int(expected["expected_episodes"]))):
            raise RuntimeError(
                "episode IDs are not contiguous from zero: " + str(actual_ids.tolist())
            )

        tasks_table = pq.read_table(self.root / "meta/tasks.parquet")
        task_rows = tasks_table.to_pylist()
        self.tasks = {int(row["task_index"]): str(row["__index_level_0__"]) for row in task_rows}
        episode_meta_path = self.root / "meta/episodes/chunk-000/file-000.parquet"
        self.episode_meta = {
            int(row["episode_index"]): row for row in pq.read_table(episode_meta_path).to_pylist()
        }
        manifest_path = ROOT / "reports/magsafe_lerobot_v3_manifest.csv"
        self.source_manifest: dict[int, dict[str, str]] = {}
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                self.source_manifest[int(row["output_episode_index"])] = row
        self._validate_all_episode_boundaries()

    def _validate_all_episode_boundaries(self) -> None:
        for episode_id in range(int(self.info["total_episodes"])):
            mask = self.episode_index == episode_id
            indices = np.flatnonzero(mask)
            if len(indices) == 0 or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                raise RuntimeError(f"episode {episode_id} is absent or non-contiguous")
            frames = self.frame_index[mask]
            timestamps = self.timestamp[mask]
            expected_length = int(self.episode_meta[episode_id]["length"])
            if len(frames) != expected_length or not np.array_equal(frames, np.arange(expected_length)):
                raise RuntimeError(f"episode {episode_id} frame contract mismatch")
            if len(timestamps) > 1 and not np.allclose(
                np.diff(timestamps), 1.0 / float(self.info["fps"]), atol=1e-5, rtol=0.0
            ):
                raise RuntimeError(f"episode {episode_id} timestamp cadence mismatch")

    def episode_ids(self) -> list[int]:
        return list(range(int(self.info["total_episodes"])))

    def episode(self, episode_id: int, max_frames: int | None = None) -> SourceEpisode:
        if episode_id not in self.episode_meta:
            raise KeyError(f"episode not found: {episode_id}")
        indices = np.flatnonzero(self.episode_index == episode_id)
        if max_frames is not None:
            indices = indices[:max_frames]
        source = self.source_manifest[episode_id]
        meta = self.episode_meta[episode_id]
        task_id = int(self.task_index[indices[0]])
        image_key = "observation.images.cam_high"
        image_reference = {
            "key": image_key,
            "dataset_root": str(self.root),
            "video_path_template": self.info["video_path"],
            "chunk_index": int(meta[f"videos/{image_key}/chunk_index"]),
            "file_index": int(meta[f"videos/{image_key}/file_index"]),
            "from_timestamp": float(meta[f"videos/{image_key}/from_timestamp"]),
            "to_timestamp": float(meta[f"videos/{image_key}/to_timestamp"]),
            "assets_duplicated": False,
        }
        return SourceEpisode(
            episode_id=episode_id,
            action=self.action[indices].copy(),
            state=self.state[indices].copy(),
            timestamps=self.timestamp[indices].copy(),
            frame_index=self.frame_index[indices].copy(),
            task_index=self.task_index[indices].copy(),
            task=self.tasks[task_id],
            nominal_fps=float(self.info["fps"]),
            source_folder=source["source_folder"],
            source_parquet=source["source_parquet"],
            image_reference=image_reference,
        )

    def audit(self) -> dict[str, Any]:
        features = self.info["features"]
        feature_keys = sorted(features)
        lengths = [int(self.episode_meta[index]["length"]) for index in self.episode_ids()]
        object_keys = [
            key for key in feature_keys
            if any(word in key.lower() for word in ("phone", "accessory", "charger", "object", "task_frame"))
        ]
        return {
            "status": "AUTHORITATIVE_50_EPISODE_DATASET_VERIFIED",
            "dataset_root": str(self.root),
            "dataset_format": "LeRobot",
            "dataset_format_version": self.info["codebase_version"],
            "robot_type": self.info["robot_type"],
            "number_of_episodes": int(self.info["total_episodes"]),
            "usable_episode_count": len(self.episode_ids()),
            "exactly_50_usable_demonstrations": len(self.episode_ids()) == 50,
            "total_frames": int(self.info["total_frames"]),
            "fps": float(self.info["fps"]),
            "image_keys": [key for key, value in features.items() if value["dtype"] == "video"],
            "observation_state": {
                "key": "observation.state",
                "dimension": int(features["observation.state"]["shape"][0]),
                "dtype": features["observation.state"]["dtype"],
                "names": features["observation.state"]["names"],
            },
            "action": {
                "key": "action",
                "dimension": int(features["action"]["shape"][0]),
                "dtype": features["action"]["dtype"],
                "names": features["action"]["names"],
                "semantics": "direct follower joint-position target",
                "semantics_evidence": str(
                    ROOT / "evaluation/mujoco_stationary_aloha_validation/action_semantics_report.json"
                ),
            },
            "language_task_metadata": {
                "total_tasks": int(self.info["total_tasks"]),
                "task_index_to_instruction": self.tasks,
                "episode_task_storage": "meta/episodes/... column 'tasks' and frame-level task_index",
            },
            "episode_lengths": lengths,
            "episode_length_min": min(lengths),
            "episode_length_max": max(lengths),
            "episode_length_mean": float(np.mean(lengths)),
            "gripper_channels": {
                "left": {
                    "action_index": 6,
                    "feature_name": features["action"]["names"][6],
                    "source_actuator": "follower_left_gripper",
                    "source_joint": "follower_left_left_carriage_joint",
                    "range_m": [0.0, 0.044],
                    "increasing_is_open": True,
                },
                "right": {
                    "action_index": 13,
                    "feature_name": features["action"]["names"][13],
                    "source_actuator": "follower_right_gripper",
                    "source_joint": "follower_right_left_carriage_joint",
                    "range_m": [0.0, 0.044],
                    "increasing_is_open": True,
                },
                "verification_source": str(
                    ROOT / "evaluation/mujoco_stationary_aloha_validation/mujoco_model_mapping.json"
                ),
            },
            "arm_channels": {
                "left": "action[0:6]",
                "right": "action[7:13]",
            },
            "all_feature_keys": feature_keys,
            "object_pose_feature_keys": object_keys,
            "object_relative_metadata_status": (
                "AVAILABLE" if object_keys else "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE"
            ),
            "authoritative_object_pose_pairing_audit": {
                "locations_checked": [
                    str(self.root / "meta/info.json"),
                    str(self.root / "meta/tasks.parquet"),
                    str(self.root / "meta/episodes/chunk-000/file-000.parquet"),
                    str(ROOT / "reports/magsafe_lerobot_v3_build_report.json"),
                    str(ROOT / "reports/magsafe_lerobot_v3_manifest.csv"),
                    "all integrated Parquet feature columns",
                    "original source recording metadata referenced by the manifest",
                ],
                "phone_pose": "NOT_AVAILABLE",
                "accessory_pose": "NOT_AVAILABLE",
                "charger_pose": "NOT_AVAILABLE",
                "task_frame_pose": "NOT_AVAILABLE",
                "result": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
            },
            "simulation_scene_used_as_source_object_measurement": False,
            "build_report": str(ROOT / "reports/magsafe_lerobot_v3_build_report.json"),
            "source_manifest": str(ROOT / "reports/magsafe_lerobot_v3_manifest.csv"),
            "primary_source_is_temporal_consensus": False,
        }


class SourceKinematics:
    """Authoritative stationary ALOHA FK and explicit TCP semantics."""

    def __init__(self, config: dict[str, Any]):
        path = Path(config["models"]["aloha_xml"])
        require_hash(path, config["models"]["aloha_xml_sha256"], "ALOHA model")
        self.model, _ = aloha_fk.load_validated_model(path)

    def compute(self, action: np.ndarray) -> dict[str, np.ndarray | int]:
        action = np.asarray(action, dtype=np.float64)
        parse_action_channels(action)
        qpos, clipped = aloha_fk.mapped_qpos(action)
        value = aloha_fk.fk(self.model, qpos)
        left_rotation = Rotation.from_quat(value["left_quaternion_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
        right_rotation = Rotation.from_quat(value["right_quaternion_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
        return {
            "qpos": qpos,
            "gripper_clipped_frames": int(clipped),
            "left_position": value["left_position_m"],
            "right_position": value["right_position_m"],
            "left_rotation": left_rotation,
            "right_rotation": right_rotation,
        }


def raw_wrist_pose(runtime: ActiveG1Dex3, side: str) -> np.ndarray:
    body = runtime.body_ids[f"{side}_wrist_yaw_link"]
    return transform(runtime.data.xmat[body].reshape(3, 3), runtime.data.xpos[body])


def raw_contact_position(runtime: ActiveG1Dex3, label: str) -> np.ndarray:
    spec = runtime.contacts[label]
    body = runtime.body_ids[spec.link]
    rotation = runtime.data.xmat[body].reshape(3, 3)
    return np.asarray(runtime.data.xpos[body]) + rotation @ spec.local_position


def normalize(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length > np.finfo(np.float64).eps:
        return value / length
    fallback_value = np.asarray(fallback, dtype=np.float64)
    return fallback_value / np.linalg.norm(fallback_value)


def physical_pinch_frame(
    runtime: ActiveG1Dex3, side: str, labels: tuple[str, str]
) -> np.ndarray:
    wrist = raw_wrist_pose(runtime, side)
    thumb = raw_contact_position(runtime, labels[0])
    index = raw_contact_position(runtime, labels[1])
    closing = normalize(index - thumb, (0.0, 1.0, 0.0))
    approach = wrist[:3, 0] - closing * float(np.dot(wrist[:3, 0], closing))
    approach = normalize(approach, (1.0, 0.0, 0.0))
    lateral = normalize(np.cross(approach, closing), (0.0, 0.0, 1.0))
    closing = normalize(np.cross(lateral, approach), (0.0, 1.0, 0.0))
    return transform(np.column_stack((approach, closing, lateral)), 0.5 * (thumb + index))


class HandMapper:
    """Baseline binary and proposed five-state mappings on one Dex3 model."""

    def __init__(self, config: dict[str, Any], runtime: ActiveG1Dex3):
        self.config = config
        self.runtime = runtime
        hand_cfg = config["hand_mapping"]
        require_hash(hand_cfg["primitive_source"], hand_cfg["primitive_source_sha256"], "Dex3 primitives")
        require_hash(hand_cfg["dex3_mapping_source"], hand_cfg["dex3_mapping_source_sha256"], "Dex3 mapping")
        require_hash(hand_cfg["palm_source"], hand_cfg["palm_source_sha256"], "Dex3 palm calibration")
        require_hash(hand_cfg["semantic_detector_source"], hand_cfg["semantic_detector_source_sha256"], "semantic detector")
        self.primitive_data = json.loads(Path(hand_cfg["primitive_source"]).read_text(encoding="utf-8"))
        detector = json.loads(Path(hand_cfg["semantic_detector_source"]).read_text(encoding="utf-8"))
        self.detector_config = detector["gripper"]
        self.names = {side: np.asarray(runtime.hand_joint_names[side]) for side in ("left", "right")}
        self.limits = runtime.hand_limits
        self.proposed_primitives = self._load_proposed_primitives()
        self.baseline_primitives = self._baseline_primitives()

    def _primitive_by_name(self, key: str, side: str) -> np.ndarray:
        row = self.primitive_data["primitives"][key]
        lookup = dict(zip(row["joint_names"], row["qpos"]))
        missing = [name for name in self.names[side] if name not in lookup]
        if missing:
            raise RuntimeError(f"primitive {key} missing joints {missing}")
        return np.asarray([lookup[name] for name in self.names[side]], dtype=np.float64)

    def _load_proposed_primitives(self) -> dict[str, dict[str, np.ndarray]]:
        mapping = self.config["hand_mapping"]["proposed"]
        result: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
        for side in ("left", "right"):
            for phase, key in mapping[f"{side}_phase_to_primitive"].items():
                q = self._primitive_by_name(key, side)
                limits = self.limits[side]
                if np.any(q < limits[:, 0] - 1e-9) or np.any(q > limits[:, 1] + 1e-9):
                    raise RuntimeError(f"primitive {key} violates active-model limits")
                result[side][phase] = q
        return result

    def _baseline_primitives(self) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, np.ndarray]] = {}
        generic = self.config["hand_mapping"]["baseline"]["close_joint_values_by_name"]
        for side in ("left", "right"):
            open_q = self.runtime.open_hand_q[side].copy()
            close_q = np.asarray([generic[name] for name in self.names[side]], dtype=np.float64)
            margin = 0.08 * (self.limits[side][:, 1] - self.limits[side][:, 0])
            close_q = np.clip(close_q, self.limits[side][:, 0] + margin, self.limits[side][:, 1] - margin)
            result[side] = {"OPEN": open_q, "CLOSE": close_q}
        return result

    def detect(self, episode: SourceEpisode) -> dict[str, GripperResult]:
        parsed = parse_action_channels(episode.action)
        return {
            "left": detect_gripper_phases(parsed["left_gripper"], episode.timestamps, self.detector_config),
            "right": detect_gripper_phases(parsed["right_gripper"], episode.timestamps, self.detector_config),
        }

    @staticmethod
    def _smooth_state_commands(
        labels: np.ndarray,
        primitives: dict[str, np.ndarray],
        frames: int,
    ) -> np.ndarray:
        labels = np.asarray(labels).astype(str)
        output = np.empty((len(labels), len(next(iter(primitives.values())))), dtype=np.float64)
        current = primitives[str(labels[0])].copy()
        output[0] = current
        active_label = str(labels[0])
        start_q = current.copy()
        transition_age = frames
        for index in range(1, len(labels)):
            label = str(labels[index])
            if label != active_label:
                active_label = label
                start_q = current.copy()
                transition_age = 0
            transition_age += 1
            u = min(1.0, transition_age / max(frames, 1))
            smooth = u * u * (3.0 - 2.0 * u)
            current = start_q + smooth * (primitives[active_label] - start_q)
            output[index] = current
        return output

    def map(
        self, method: str, episode: SourceEpisode, detected: dict[str, GripperResult]
    ) -> dict[str, Any]:
        transition_frames = max(
            1, int(round(float(self.config["hand_mapping"]["transition_duration_sec"]) * episode.fps))
        )
        labels: dict[str, np.ndarray] = {}
        commands: dict[str, np.ndarray] = {}
        if method == "baseline":
            rules = self.config["hand_mapping"]["baseline"]
            for side in ("left", "right"):
                normalized = detected[side].normalized_open
                state = "OPEN" if normalized[0] >= 0.5 else "CLOSE"
                side_labels = np.empty(len(normalized), dtype="U12")
                for index, value in enumerate(normalized):
                    if state == "OPEN" and value <= float(rules["close_enter_normalized_open"]):
                        state = "CLOSE"
                    elif state == "CLOSE" and value >= float(rules["open_enter_normalized_open"]):
                        state = "OPEN"
                    side_labels[index] = state
                labels[side] = side_labels
                commands[side] = self._smooth_state_commands(
                    side_labels, self.baseline_primitives[side], transition_frames
                )
        elif method == "proposed":
            for side in ("left", "right"):
                labels[side] = detected[side].phase.copy()
                commands[side] = self._smooth_state_commands(
                    labels[side], self.proposed_primitives[side], transition_frames
                )
        else:
            raise ValueError(method)
        valid = {"baseline": {"OPEN", "CLOSE"}, "proposed": set(PHASE_NAMES.tolist())}[method]
        unknown = int(sum(np.count_nonzero(~np.isin(labels[side], list(valid))) for side in labels))
        limit_violations = int(sum(
            np.count_nonzero(
                (commands[side] < self.limits[side][:, 0] - 1e-9)
                | (commands[side] > self.limits[side][:, 1] + 1e-9)
            ) for side in commands
        ))
        return {
            "left": commands["left"],
            "right": commands["right"],
            "left_phase": labels["left"],
            "right_phase": labels["right"],
            "transition_frames": transition_frames,
            "unknown_phase_count": unknown,
            "joint_limit_violation_count": limit_violations,
            "label_validity": bool(unknown == 0),
        }


class G1Kinematics:
    """One active G1 model for wrist IK, hand FK, and collision checks."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        model_path = Path(config["models"]["g1_xml"])
        require_hash(model_path, config["models"]["g1_xml_sha256"], "G1 model")
        hand_cfg = config["hand_mapping"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.runtime = ActiveG1Dex3(
                model_path,
                hand_cfg["dex3_mapping_source"],
                hand_cfg["palm_source"],
                np.asarray([0.0, 0.0, 0.7922728583]),
            )
        self.info = self.runtime.info
        self.model = self.runtime.model
        self.data = mujoco.MjData(self.model)
        self.nominal_q = np.asarray(config["nominal_g1_arm_q"], dtype=np.float64)
        self.limits = np.asarray(self.info["joint_limits"], dtype=np.float64)
        if self.nominal_q.shape != (14,) or np.any(self.nominal_q < self.limits[:, 0]) or np.any(self.nominal_q > self.limits[:, 1]):
            raise RuntimeError("configured nominal G1 posture is invalid")
        self.wrist_ids = {
            side: int(self.runtime.body_ids[f"{side}_wrist_yaw_link"])
            for side in ("left", "right")
        }
        self.arm_dofs = np.asarray(self.info["arm_dof_ids"], dtype=np.int64)
        self.tool_local = {
            side: np.asarray(config["target_frames"][f"{side}_wrist_to_physical_pinch"], dtype=np.float64)
            for side in ("left", "right")
        }
        self.palm_local = {
            side: np.asarray(config["target_frames"][f"{side}_wrist_to_palm"], dtype=np.float64)
            for side in ("left", "right")
        }
        self.pinch_labels = {
            "left": tuple(config["target_frames"]["left_physical_pinch_contacts"]),
            "right": tuple(config["target_frames"]["right_physical_pinch_contacts"]),
        }

    def assign_arm(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.info["stand_qpos"]
        self.data.qpos[self.info["arm_qpos_ids"]] = np.asarray(q, dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def wrist_state(self, q: np.ndarray) -> dict[str, np.ndarray]:
        self.assign_arm(q)
        output: dict[str, np.ndarray] = {}
        for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
            body = self.wrist_ids[side]
            position = np.asarray(self.data.xpos[body], dtype=np.float64).copy()
            rotation = np.asarray(self.data.xmat[body], dtype=np.float64).reshape(3, 3).copy()
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body)
            dofs = self.arm_dofs[block]
            output[f"{side}_position"] = position
            output[f"{side}_rotation"] = rotation
            output[f"{side}_jacobian"] = np.vstack((jacp[:, dofs], jacr[:, dofs]))
        return output

    def nominal_wrist_poses(self) -> dict[str, np.ndarray]:
        state = self.wrist_state(self.nominal_q)
        return {
            side: transform(state[f"{side}_rotation"], state[f"{side}_position"])
            for side in ("left", "right")
        }

    def verify_tool_transforms(self, hand_mapper: HandMapper) -> dict[str, float]:
        errors: dict[str, float] = {}
        for side in ("left", "right"):
            left = (
                hand_mapper.proposed_primitives["left"]["GRASP"]
                if side == "left" else self.runtime.open_hand_q["left"]
            )
            right = (
                hand_mapper.proposed_primitives["right"]["GRASP"]
                if side == "right" else self.runtime.open_hand_q["right"]
            )
            self.runtime.assign(self.nominal_q, left, right)
            wrist = raw_wrist_pose(self.runtime, side)
            pinch = physical_pinch_frame(self.runtime, side, self.pinch_labels[side])
            derived = inverse_transform(wrist) @ pinch
            error = float(np.max(np.abs(derived - self.tool_local[side])))
            errors[side] = error
            if error > 1e-9:
                raise RuntimeError(f"configured {side} wrist-to-pinch transform fails FK parity: {error}")
        return errors

    def evaluate_wrist(self, q: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        count = len(q)
        result = {
            f"{side}_{kind}": np.empty((count, 3, 3) if kind == "rotation" else (count, 3))
            for side in ("left", "right") for kind in ("position", "rotation")
        }
        for index, row in enumerate(q):
            state = self.wrist_state(row)
            for side in ("left", "right"):
                result[f"{side}_position"][index] = state[f"{side}_position"]
                result[f"{side}_rotation"][index] = state[f"{side}_rotation"]
        for side in ("left", "right"):
            result[f"{side}_position_error"] = np.linalg.norm(
                result[f"{side}_position"] - targets[f"{side}_wrist_position"], axis=1
            )
            result[f"{side}_orientation_error"] = rotation_errors(
                result[f"{side}_rotation"], targets[f"{side}_wrist_rotation"]
            )
        return result

    def full_geometry(
        self,
        arm: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> dict[str, Any]:
        count = len(arm)
        result: dict[str, Any] = {}
        for side in ("left", "right"):
            for frame_name in ("wrist", "palm", "pinch"):
                result[f"{side}_{frame_name}_position"] = np.empty((count, 3))
                result[f"{side}_{frame_name}_rotation"] = np.empty((count, 3, 3))
        collision = np.zeros(count, dtype=bool)
        pairs: set[str] = set()
        tolerance = float(self.config["validation"]["collision_penetration_tolerance_m"])
        allowlist = set(self.config["validation"]["static_model_contact_allowlist"])
        for index in range(count):
            self.runtime.assign(arm[index], left_hand[index], right_hand[index])
            for side in ("left", "right"):
                wrist = raw_wrist_pose(self.runtime, side)
                palm = wrist @ self.palm_local[side]
                pinch = physical_pinch_frame(self.runtime, side, self.pinch_labels[side])
                for name, pose in (("wrist", wrist), ("palm", palm), ("pinch", pinch)):
                    result[f"{side}_{name}_position"][index] = pose[:3, 3]
                    result[f"{side}_{name}_rotation"][index] = pose[:3, :3]
            for contact in self.runtime.data.contact:
                if float(contact.dist) >= -tolerance:
                    continue
                bodies = []
                for geom in (contact.geom1, contact.geom2):
                    body = int(self.model.geom_bodyid[int(geom)])
                    bodies.append(
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                        or f"body_{body}"
                    )
                pair = "|".join(sorted(bodies))
                same_side_hand_chain = (
                    (bodies[0].startswith("left_hand") and bodies[1].startswith("left_"))
                    or (bodies[1].startswith("left_hand") and bodies[0].startswith("left_"))
                    or (bodies[0].startswith("right_hand") and bodies[1].startswith("right_"))
                    or (bodies[1].startswith("right_hand") and bodies[0].startswith("right_"))
                )
                if pair in allowlist or bodies[0] == bodies[1] or same_side_hand_chain:
                    continue
                relevant = any(
                    word in "|".join(bodies)
                    for word in ("shoulder", "elbow", "wrist", "hand", "torso")
                )
                if relevant:
                    collision[index] = True
                    pairs.add(pair)
        result["prohibited_collision_flag"] = collision
        result["prohibited_collision_pairs"] = sorted(pairs)
        return result


class RepresentationBuilder:
    """Build wrist targets while retaining one common diagnostic task-tool path."""

    def __init__(self, config: dict[str, Any], g1: G1Kinematics):
        self.config = config
        self.g1 = g1
        self.axis = np.asarray(config["coordinate_conventions"]["source_to_target_axis_rotation"], dtype=np.float64)
        self.scale = float(config["workspace_mapping"]["uniform_scale"])
        self.calibration = {
            side: np.asarray(
                config["tool_mapping"][f"{side}_tool_transform"],
                dtype=np.float64,
            )[:3, :3] for side in ("left", "right")
        }
        for side in ("left", "right"):
            legacy_axes = np.asarray(
                config["orientation_mapping"][f"{side}_source_tool_to_target_tool_axes"]
            )
            if not np.array_equal(self.calibration[side], legacy_axes):
                raise RuntimeError(f"{side} static tool-transform axis provenance mismatch")

    def mapped_rotation_delta(self, source: np.ndarray, side: str) -> np.ndarray:
        local_delta = np.einsum("ij,tjk->tik", source[0].T, source)
        calibration = self.calibration[side]
        return np.einsum("ij,tjk,kl->til", calibration.T, local_delta, calibration)

    def common_task_tool(self, source_fk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        nominal_wrist = self.g1.nominal_wrist_poses()
        nominal_tool = {
            side: nominal_wrist[side] @ self.g1.tool_local[side] for side in ("left", "right")
        }
        source_left = np.asarray(source_fk["left_position"])
        source_right = np.asarray(source_fk["right_position"])
        source_midpoint = 0.5 * (source_left + source_right)
        source_relative = source_right - source_left
        midpoint_delta = (source_midpoint - source_midpoint[0]) @ self.axis.T
        relative_delta = (source_relative - source_relative[0]) @ self.axis.T
        target_midpoint0 = 0.5 * (nominal_tool["left"][:3, 3] + nominal_tool["right"][:3, 3])
        target_relative0 = nominal_tool["right"][:3, 3] - nominal_tool["left"][:3, 3]
        target_midpoint = target_midpoint0 + self.scale * midpoint_delta
        target_relative = target_relative0 + self.scale * relative_delta
        result: dict[str, np.ndarray] = {
            "source_midpoint": source_midpoint,
            "source_relative": source_relative,
            "target_tool_midpoint": target_midpoint,
            "target_tool_relative": target_relative,
            "left_tool_position": target_midpoint - 0.5 * target_relative,
            "right_tool_position": target_midpoint + 0.5 * target_relative,
        }
        for side in ("left", "right"):
            delta = self.mapped_rotation_delta(np.asarray(source_fk[f"{side}_rotation"]), side)
            result[f"{side}_tool_rotation"] = np.einsum(
                "ij,tjk->tik", nominal_tool[side][:3, :3], delta
            )
        return result

    def build(self, method: str, source_fk: dict[str, np.ndarray]) -> dict[str, np.ndarray | str | float]:
        nominal_wrist = self.g1.nominal_wrist_poses()
        task_tool = self.common_task_tool(source_fk)
        target: dict[str, Any] = dict(task_tool)
        if method == "baseline":
            for side in ("left", "right"):
                source_position = np.asarray(source_fk[f"{side}_position"])
                delta_position = (source_position - source_position[0]) @ self.axis.T
                delta_rotation = self.mapped_rotation_delta(
                    np.asarray(source_fk[f"{side}_rotation"]), side
                )
                target[f"{side}_wrist_position"] = (
                    nominal_wrist[side][:3, 3] + self.scale * delta_position
                )
                target[f"{side}_wrist_rotation"] = np.einsum(
                    "ij,tjk->tik", nominal_wrist[side][:3, :3], delta_rotation
                )
            target["representation"] = "trajectory-centric independent wrist-level 6D"
        elif method == "proposed":
            for side in ("left", "right"):
                tool_rotation = np.asarray(target[f"{side}_tool_rotation"])
                tool_position = np.asarray(target[f"{side}_tool_position"])
                local = self.g1.tool_local[side]
                wrist_rotation = np.einsum("tij,jk->tik", tool_rotation, local[:3, :3].T)
                wrist_position = tool_position - np.einsum(
                    "tij,j->ti", wrist_rotation, local[:3, 3]
                )
                target[f"{side}_wrist_rotation"] = wrist_rotation
                target[f"{side}_wrist_position"] = wrist_position
            target["representation"] = "first-frame-relative bimanual task-tool/pinch-frame SE(3)"
        else:
            raise ValueError(method)
        target["method"] = method
        target["scale"] = self.scale
        target["frame_mapping_finite"] = bool(all(
            np.isfinite(value).all() for value in target.values() if isinstance(value, np.ndarray)
        ))
        target["bimanual_reconstruction_max_error_m"] = float(max(
            np.max(np.abs(
                0.5 * (target["left_tool_position"] + target["right_tool_position"])
                - target["target_tool_midpoint"]
            )),
            np.max(np.abs(
                (target["right_tool_position"] - target["left_tool_position"])
                - target["target_tool_relative"]
            )),
        ))
        return target


class SharedTemporalIK:
    """Shared causal temporal DLS plus joint smoothing and target reprojection.

    Each frame minimizes the same objective for both methods:

      wp ||FK_wrist(q)-T_wrist|| + wo ||Log(R*R(q)^T)||
      + wv ||q-q[t-1]|| + wa ||q-2q[t-1]+q[t-2]||
      + wn ||q-q_nominal||,

    with damped normal equations, bounded updates, and hard joint-limit
    projection.  No bimanual relation residual is present in the solver.
    """

    def __init__(self, config: dict[str, Any], g1: G1Kinematics):
        self.config = config["ik"]
        self.g1 = g1

    def _system(
        self,
        q: np.ndarray,
        targets: dict[str, np.ndarray],
        index: int,
        previous: np.ndarray,
        previous2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        state = self.g1.wrist_state(q)
        wp = float(self.config["position_weight"])
        wo = float(self.config["orientation_weight"])
        rows: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
            jacobian = np.zeros((6, 14), dtype=np.float64)
            jacobian[:, block] = state[f"{side}_jacobian"]
            p_error = targets[f"{side}_wrist_position"][index] - state[f"{side}_position"]
            o_error = rotation_error_vector(
                state[f"{side}_rotation"], targets[f"{side}_wrist_rotation"][index]
            )
            rows.extend((wp * jacobian[:3], wo * jacobian[3:]))
            errors.extend((wp * p_error, wo * o_error))
            position_errors.append(float(np.linalg.norm(p_error)))
            orientation_errors.append(float(np.linalg.norm(o_error)))
        identity = np.eye(14, dtype=np.float64)
        wv = float(self.config["velocity_regularization_weight"])
        wa = float(self.config["acceleration_regularization_weight"])
        wn = float(self.config["nominal_regularization_weight"])
        rows.extend((wv * identity, wa * identity, wn * identity))
        errors.extend((
            wv * (previous - q),
            wa * (2.0 * previous - previous2 - q),
            wn * (self.g1.nominal_q - q),
        ))
        return (
            np.vstack(rows),
            np.concatenate(errors),
            (position_errors[0], position_errors[1], orientation_errors[0], orientation_errors[1]),
        )

    def _solve_frame(
        self,
        targets: dict[str, np.ndarray],
        index: int,
        initial: np.ndarray,
        previous: np.ndarray,
        previous2: np.ndarray,
        iterations: int,
    ) -> tuple[np.ndarray, int]:
        q = np.asarray(initial, dtype=np.float64).copy()
        limits = self.g1.limits
        damping = float(self.config["damping"])
        max_update = float(self.config["max_joint_update_rad"])
        max_frame_step = float(self.config["max_frame_joint_step_rad"])
        frame_lower = np.maximum(limits[:, 0], previous - max_frame_step)
        frame_upper = np.minimum(limits[:, 1], previous + max_frame_step)
        q = np.clip(q, frame_lower, frame_upper)
        pos_tol = min(float(self.config["position_tolerance_m"]), 5e-4)
        ori_tol = float(self.config["orientation_tolerance_rad"])
        best = q.copy()
        best_key = (math.inf, math.inf, math.inf)
        used = 0
        for used in range(1, iterations + 1):
            jacobian, error, residuals = self._system(q, targets, index, previous, previous2)
            position_max = max(residuals[:2])
            orientation_max = max(residuals[2:])
            key = (
                0.0 if position_max <= float(self.config["position_tolerance_m"]) else 1.0,
                position_max,
                orientation_max,
            )
            if key < best_key:
                best_key = key
                best = q.copy()
            if position_max <= pos_tol and orientation_max <= ori_tol:
                break
            normal = jacobian.T @ jacobian + (damping * damping) * np.eye(14)
            right = jacobian.T @ error
            try:
                delta = np.linalg.solve(normal, right)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(normal, right, rcond=None)[0]
            q = np.clip(
                q + np.clip(delta, -max_update, max_update), frame_lower, frame_upper
            )
        _, _, residuals = self._system(q, targets, index, previous, previous2)
        position_max = max(residuals[:2])
        orientation_max = max(residuals[2:])
        key = (
            0.0 if position_max <= float(self.config["position_tolerance_m"]) else 1.0,
            position_max,
            orientation_max,
        )
        if key < best_key:
            best = q.copy()
        return best, used

    def solve(self, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        count = len(targets["left_wrist_position"])
        raw = np.empty((count, 14), dtype=np.float64)
        initial_iterations = np.zeros(count, dtype=np.int64)
        previous = self.g1.nominal_q.copy()
        previous2 = previous.copy()
        for index in range(count):
            maximum = int(
                self.config["max_iterations_initial_frame"]
                if index == 0 else self.config["max_iterations_per_frame"]
            )
            raw[index], initial_iterations[index] = self._solve_frame(
                targets, index, previous, previous, previous2, maximum
            )
            previous2, previous = previous, raw[index].copy()

        window = min(int(self.config["temporal_smoothing_window"]), count if count % 2 else count - 1)
        polyorder = int(self.config["temporal_smoothing_polyorder"])
        if window >= max(polyorder + 2, 3):
            smoothed = savgol_filter(raw, window, polyorder, axis=0, mode="interp")
        else:
            smoothed = raw.copy()
        smoothed = np.clip(smoothed, self.g1.limits[:, 0], self.g1.limits[:, 1])

        final = np.empty_like(raw)
        reprojection_iterations = np.zeros(count, dtype=np.int64)
        previous = smoothed[0].copy()
        previous2 = previous.copy()
        for index in range(count):
            final[index], reprojection_iterations[index] = self._solve_frame(
                targets,
                index,
                smoothed[index],
                previous,
                previous2,
                int(self.config["max_iterations_reprojection"]),
            )
            previous2, previous = previous, final[index].copy()
        return {
            "q": final,
            "raw_q": raw,
            "smoothed_q": smoothed,
            "initial_iterations": initial_iterations,
            "reprojection_iterations": reprojection_iterations,
        }


@dataclass
class ConversionResult:
    method: str
    episode: SourceEpisode
    source_fk: dict[str, Any]
    targets: dict[str, Any]
    solver: dict[str, np.ndarray]
    wrist: dict[str, np.ndarray]
    hand: dict[str, Any]
    detected: dict[str, GripperResult]
    geometry: dict[str, Any]
    metrics: dict[str, Any]
    validation: dict[str, Any]


def branch_flags(q: np.ndarray, absolute: float, multiplier: float) -> np.ndarray:
    norms = np.linalg.norm(np.diff(q, axis=0), axis=1)
    flags = np.zeros(len(q), dtype=bool)
    for index in range(1, len(q)):
        local = float(np.median(norms[max(0, index - 10):min(len(norms), index + 9)]))
        flags[index] = norms[index - 1] > max(absolute, multiplier * max(local, 1e-5))
    return flags


def validate_conversion(
    config: dict[str, Any],
    method: str,
    episode: SourceEpisode,
    source_fk: dict[str, Any],
    targets: dict[str, Any],
    solver: dict[str, np.ndarray],
    wrist: dict[str, np.ndarray],
    hand: dict[str, Any],
    detected: dict[str, GripperResult],
    geometry: dict[str, Any],
    g1: G1Kinematics,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm = solver["q"]
    full_action = np.column_stack((arm, hand["left"], hand["right"]))
    fps = float(episode.fps)
    ik_cfg = config["ik"]
    validation_cfg = config["validation"]
    left_pos_ok = wrist["left_position_error"] <= float(ik_cfg["position_tolerance_m"])
    right_pos_ok = wrist["right_position_error"] <= float(ik_cfg["position_tolerance_m"])
    left_ori_ok = wrist["left_orientation_error"] <= float(ik_cfg["orientation_tolerance_rad"])
    right_ori_ok = wrist["right_orientation_error"] <= float(ik_cfg["orientation_tolerance_rad"])
    ik_success = left_pos_ok & right_pos_ok & left_ori_ok & right_ori_ok
    arm_violations = (
        (arm < g1.limits[:, 0] - 1e-9) | (arm > g1.limits[:, 1] + 1e-9)
    )
    finite = bool(all(np.isfinite(value).all() for value in (
        episode.action, arm, hand["left"], hand["right"], full_action,
        wrist["left_position"], wrist["right_position"],
    )))
    step = np.abs(np.diff(arm, axis=0))
    velocity = step * fps
    acceleration = np.abs(np.diff(arm, n=2, axis=0)) * fps * fps
    branches = branch_flags(
        arm,
        float(validation_cfg["branch_absolute_step_norm_rad"]),
        float(validation_cfg["branch_local_multiplier"]),
    )

    actual_tool = {
        side: geometry[f"{side}_pinch_position"] for side in ("left", "right")
    }
    target_tool = {
        side: np.asarray(targets[f"{side}_tool_position"]) for side in ("left", "right")
    }
    pinch_error = {
        side: np.linalg.norm(actual_tool[side] - target_tool[side], axis=1)
        for side in ("left", "right")
    }
    pinch_orientation_error = {
        side: rotation_errors(
            geometry[f"{side}_pinch_rotation"], np.asarray(targets[f"{side}_tool_rotation"])
        ) for side in ("left", "right")
    }
    actual_midpoint = 0.5 * (actual_tool["left"] + actual_tool["right"])
    actual_relative = actual_tool["right"] - actual_tool["left"]
    target_midpoint = np.asarray(targets["target_tool_midpoint"])
    target_relative = np.asarray(targets["target_tool_relative"])
    midpoint_error = np.linalg.norm(actual_midpoint - target_midpoint, axis=1)
    relative_error = np.linalg.norm(actual_relative - target_relative, axis=1)
    actual_distance = np.linalg.norm(actual_relative, axis=1)
    target_distance = np.linalg.norm(target_relative, axis=1)
    distance_change_error = np.abs(
        (actual_distance - actual_distance[0]) - (target_distance - target_distance[0])
    )
    # A comparison metric must use the same timestamps for both methods.  The
    # common semantic detector supplies that mask; method-specific target hand
    # labels are assessed separately below.
    task_masks = {
        side: np.isin(detected[side].phase, ["GRASP", "HOLD"])
        for side in ("left", "right")
    }
    ik_component_success = {
        "left_position": left_pos_ok,
        "right_position": right_pos_ok,
        "left_orientation": left_ori_ok,
        "right_orientation": right_ori_ok,
    }
    failed_ik_indices = np.flatnonzero(~ik_success)
    first_failed_ik_frame = int(failed_ik_indices[0]) if len(failed_ik_indices) else None
    first_failed_ik_components = (
        [key for key, value in ik_component_success.items() if not bool(value[first_failed_ik_frame])]
        if first_failed_ik_frame is not None else []
    )

    gripper_events = {
        side: detected[side].transitions for side in ("left", "right")
    }
    target_runs = {
        side: phase_runs(hand[f"{side}_phase"], episode.timestamps)
        for side in ("left", "right")
    }
    collision_count = int(np.count_nonzero(geometry["prohibited_collision_flag"]))
    metrics = {
        "method": method,
        "method_report_name": (
            "TrajBooster-style upper-body baseline"
            if method == "baseline" else "Interaction-aware proposed converter"
        ),
        "episode_id": int(episode.episode_id),
        "status": "PENDING_CLASSIFICATION",
        "frame_count": int(len(episode.action)),
        "fps": fps,
        "duration_sec": float(episode.timestamps[-1] - episode.timestamps[0]),
        "finite_values": finite,
        "g1_joint_shape": list(full_action.shape),
        "g1_arm_shape": list(arm.shape),
        "g1_hand_shape": list(np.column_stack((hand["left"], hand["right"])).shape),
        "joint_limit_violation_count": int(np.count_nonzero(arm_violations)) + int(hand["joint_limit_violation_count"]),
        "arm_joint_limit_violation_count": int(np.count_nonzero(arm_violations)),
        "hand_joint_limit_violation_count": int(hand["joint_limit_violation_count"]),
        "ik_success_rate": float(np.mean(ik_success)),
        "ik_failed_frame_count": int(np.count_nonzero(~ik_success)),
        "ik_component_success_rate": {
            key: float(np.mean(value)) for key, value in ik_component_success.items()
        },
        "ik_component_failed_frame_count": {
            key: int(np.count_nonzero(~value)) for key, value in ik_component_success.items()
        },
        "first_failed_ik_frame": first_failed_ik_frame,
        "first_failed_ik_components": first_failed_ik_components,
        "maximum_joint_step_rad": float(np.max(step, initial=0.0)),
        "joint_step_rad": stats(step),
        "joint_velocity_rad_s": stats(velocity),
        "joint_acceleration_rad_s2": stats(acceleration),
        "branch_discontinuity_count": int(np.count_nonzero(branches)),
        "prohibited_arm_self_collision_count": collision_count,
        "prohibited_collision_pairs": geometry["prohibited_collision_pairs"],
        "task_space": {
            "source_to_target_mapping_reconstruction_error_m": float(targets["bimanual_reconstruction_max_error_m"]),
            "left_wrist_target_error_m": stats(wrist["left_position_error"]),
            "right_wrist_target_error_m": stats(wrist["right_position_error"]),
            "left_wrist_orientation_error_rad": stats(wrist["left_orientation_error"]),
            "right_wrist_orientation_error_rad": stats(wrist["right_orientation_error"]),
            "left_task_tool_target_error_m": stats(pinch_error["left"]),
            "right_task_tool_target_error_m": stats(pinch_error["right"]),
            "left_physical_pinch_center_error_m": stats(pinch_error["left"]),
            "right_physical_pinch_center_error_m": stats(pinch_error["right"]),
            "left_physical_pinch_orientation_error_rad": stats(pinch_orientation_error["left"]),
            "right_physical_pinch_orientation_error_rad": stats(pinch_orientation_error["right"]),
            "left_task_critical_pinch_error_m": stats(pinch_error["left"][task_masks["left"]]),
            "right_task_critical_pinch_error_m": stats(pinch_error["right"][task_masks["right"]]),
            "task_critical_mask_definition": "common source semantic detector phases GRASP or HOLD",
            "left_task_critical_frame_count": int(np.count_nonzero(task_masks["left"])),
            "right_task_critical_frame_count": int(np.count_nonzero(task_masks["right"])),
            "task_tool_target_role": (
                "diagnostic_only_not_optimized" if method == "baseline" else "primary_optimized_representation"
            ),
        },
        "bimanual": {
            "midpoint_trajectory_error_m": stats(midpoint_error),
            "relative_vector_error_m": stats(relative_error),
            "inter_hand_distance_change_error_m": stats(distance_change_error),
            "target_inter_hand_distance_m": stats(target_distance),
            "actual_inter_hand_distance_m": stats(actual_distance),
            "spacing_clamp": None,
        },
        "hand": {
            "gripper_event_times": gripper_events,
            "target_hand_phase_runs": target_runs,
            "transition_counts": {
                side: max(0, len(target_runs[side]) - 1) for side in ("left", "right")
            },
            "unknown_phase_count": int(hand["unknown_phase_count"]),
            "hand_label_validity": bool(hand["label_validity"]),
            "label_calibration_status": config["hand_mapping"]["label_calibration_status"],
        },
        "solver": {
            "backend": ik_cfg["backend"],
            "initial_iteration_mean": float(np.mean(solver["initial_iterations"])),
            "reprojection_iteration_mean": float(np.mean(solver["reprojection_iterations"])),
            "objective_weights": {
                key: ik_cfg[key] for key in (
                    "position_weight", "orientation_weight",
                    "velocity_regularization_weight", "acceleration_regularization_weight",
                    "nominal_regularization_weight", "bimanual_residual_weight",
                )
            },
        },
        "source_fk_gripper_clipped_frames": int(source_fk["gripper_clipped_frames"]),
        "object_relative_metadata_status": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
    }

    checks = {
        "data": finite and episode.action.shape == (len(episode.timestamps), 14),
        "frame_mapping": bool(targets["frame_mapping_finite"]) and float(targets["bimanual_reconstruction_max_error_m"]) <= 1e-10,
        "hand_mapping": bool(hand["label_validity"]) and int(hand["joint_limit_violation_count"]) == 0,
        "ik": float(np.mean(ik_success)) >= float(ik_cfg["required_success_rate"]),
        "joint_limits": int(np.count_nonzero(arm_violations)) == 0,
        "collision": collision_count <= int(validation_cfg["prohibited_collision_frames_allowed"]),
        "temporal": bool(
            int(np.count_nonzero(branches)) == 0
            and float(np.max(step, initial=0.0)) <= float(validation_cfg["maximum_joint_step_rad"])
            and float(np.max(velocity, initial=0.0)) <= float(validation_cfg["maximum_velocity_rad_s"])
            and float(np.max(acceleration, initial=0.0)) <= float(validation_cfg["maximum_acceleration_rad_s2"])
        ),
    }
    collision_indices = np.flatnonzero(geometry["prohibited_collision_flag"])
    first_collision_frame = int(collision_indices[0]) if len(collision_indices) else None
    failure_reasons = {
        "data": "non-finite value or source/output shape mismatch",
        "frame_mapping": "frame transform or bimanual reconstruction invalid",
        "hand_mapping": "unknown phase or invalid Dex3 label",
        "ik": (
            f"shared wrist IK success rate {float(np.mean(ik_success)):.6f} is below "
            f"{float(ik_cfg['required_success_rate']):.6f}; first failed frame="
            f"{first_failed_ik_frame}, components={first_failed_ik_components}"
        ),
        "joint_limits": "G1 arm joint-limit projection gate failed",
        "collision": (
            f"prohibited G1 self-collision detected at first frame={first_collision_frame}; "
            f"pairs={geometry['prohibited_collision_pairs']}"
        ),
        "temporal": (
            "temporal gate failed: "
            f"branches={int(np.count_nonzero(branches))}, "
            f"max_step={float(np.max(step, initial=0.0)):.6f} rad, "
            f"max_velocity={float(np.max(velocity, initial=0.0)):.6f} rad/s, "
            f"max_acceleration={float(np.max(acceleration, initial=0.0)):.6f} rad/s^2"
        ),
    }
    classification = (
        ("FAIL_DATA", "data"),
        ("FAIL_FRAME_MAPPING", "frame_mapping"),
        ("FAIL_HAND_MAPPING", "hand_mapping"),
        ("FAIL_IK", "ik"),
        ("FAIL_JOINT_LIMIT", "joint_limits"),
        ("FAIL_COLLISION", "collision"),
        ("FAIL_OTHER", "temporal"),
    )
    status = "PASS"
    first_failure = None
    for candidate, key in classification:
        if not checks[key]:
            status = candidate
            first_failure = {"gate": key, "reason": failure_reasons[key]}
            break
    metrics["status"] = status
    validation = {
        "status": status,
        "pass": status == "PASS",
        "first_causal_failure": first_failure,
        "checks": checks,
        "thresholds": {
            "position_tolerance_m": ik_cfg["position_tolerance_m"],
            "orientation_tolerance_rad": ik_cfg["orientation_tolerance_rad"],
            "required_ik_success_rate": ik_cfg["required_success_rate"],
            "maximum_joint_step_rad": validation_cfg["maximum_joint_step_rad"],
            "maximum_velocity_rad_s": validation_cfg["maximum_velocity_rad_s"],
            "maximum_acceleration_rad_s2": validation_cfg["maximum_acceleration_rad_s2"],
            "prohibited_collision_frames_allowed": validation_cfg["prohibited_collision_frames_allowed"],
        },
        "real_robot_validation": "NOT_PERFORMED",
        "physics_validation": "NOT_PERFORMED",
        "vla_training": "NOT_PERFORMED",
    }
    return metrics, validation


class RetargetingPipeline:
    """Single common converter with baseline/proposed representation hooks."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        dataset_root: str | Path | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        self.dataset = SourceDataset(dataset_root or self.config["source_dataset"]["root"], self.config)
        self.source_kinematics = SourceKinematics(self.config)
        self.g1 = G1Kinematics(self.config)
        self.hand_mapper = HandMapper(self.config, self.g1.runtime)
        self.tool_transform_fk_errors = self.g1.verify_tool_transforms(self.hand_mapper)
        self.representation = RepresentationBuilder(self.config, self.g1)
        self.solver = SharedTemporalIK(self.config, self.g1)
        self.config_sha256 = sha256_file(self.config_path)
        self.implementation_sha256, self.implementation_files_sha256 = implementation_fingerprint()

    def convert(
        self, method: str, episode_id: int, max_frames: int | None = None
    ) -> ConversionResult:
        if method not in METHODS:
            raise ValueError(f"unknown method {method}; expected {METHODS}")
        episode = self.dataset.episode(episode_id, max_frames=max_frames)
        source_fk = self.source_kinematics.compute(episode.action)
        targets = self.representation.build(method, source_fk)
        detected = self.hand_mapper.detect(episode)
        hand = self.hand_mapper.map(method, episode, detected)
        solved = self.solver.solve(targets)
        wrist = self.g1.evaluate_wrist(solved["q"], targets)
        geometry = self.g1.full_geometry(solved["q"], hand["left"], hand["right"])
        metrics, validation = validate_conversion(
            self.config, method, episode, source_fk, targets, solved, wrist,
            hand, detected, geometry, self.g1,
        )
        return ConversionResult(
            method=method,
            episode=episode,
            source_fk=source_fk,
            targets=targets,
            solver=solved,
            wrist=wrist,
            hand=hand,
            detected=detected,
            geometry=geometry,
            metrics=metrics,
            validation=validation,
        )

    def source_metadata(self, result: ConversionResult) -> dict[str, Any]:
        episode = result.episode
        return {
            "authoritative_dataset_root": str(self.dataset.root),
            "dataset_format": "LeRobot v3.0",
            "episode_id": int(episode.episode_id),
            "frame_count": int(len(episode.action)),
            "fps": float(episode.fps),
            "duration_sec": float(episode.timestamps[-1] - episode.timestamps[0]),
            "motion_source_key": "action",
            "observation_state_key": "observation.state",
            "action_shape": list(episode.action.shape),
            "observation_state_shape": list(episode.state.shape),
            "language_instruction": episode.task,
            "task_index": int(episode.task_index[0]),
            "source_folder": episode.source_folder,
            "source_parquet": episode.source_parquet,
            "integrated_data_shard": str(self.dataset.root / "data/chunk-000/file-000.parquet"),
            "image_reference": episode.image_reference,
            "images_duplicated": False,
            "source_action_sha256": array_sha256(episode.action.astype(np.float32)),
            "source_state_sha256": array_sha256(episode.state.astype(np.float32)),
            "object_relative_metadata_status": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
        }

    def export(self, result: ConversionResult, output_root: str | Path = DEFAULT_OUTPUT) -> Path:
        output_root = Path(output_root).resolve()
        directory = output_root / result.method / f"episode_{result.episode.episode_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        arm = result.solver["q"].astype(np.float32)
        left = result.hand["left"].astype(np.float32)
        right = result.hand["right"].astype(np.float32)
        hand = np.column_stack((left, right)).astype(np.float32)
        full = np.column_stack((arm, hand)).astype(np.float32)
        timestamps = result.episode.timestamps.astype(np.float64)
        arm_names = np.asarray(self.g1.info["joint_names"]).astype("U64")
        left_names = self.hand_mapper.names["left"].astype("U64")
        right_names = self.hand_mapper.names["right"].astype("U64")
        full_names = np.concatenate((arm_names, left_names, right_names))
        atomic_json(directory / "source_metadata.json", self.source_metadata(result))
        atomic_npz(
            directory / "g1_arm_action.npz",
            action=arm,
            timestamps=timestamps,
            fps=np.asarray(result.episode.fps),
            joint_names=arm_names,
            target_left_wrist_position=np.asarray(result.targets["left_wrist_position"]),
            target_right_wrist_position=np.asarray(result.targets["right_wrist_position"]),
            target_left_wrist_rotation=np.asarray(result.targets["left_wrist_rotation"]),
            target_right_wrist_rotation=np.asarray(result.targets["right_wrist_rotation"]),
            achieved_left_wrist_position=result.wrist["left_position"],
            achieved_right_wrist_position=result.wrist["right_position"],
            target_left_task_tool_position=np.asarray(result.targets["left_tool_position"]),
            target_right_task_tool_position=np.asarray(result.targets["right_tool_position"]),
            achieved_left_physical_pinch_position=result.geometry["left_pinch_position"],
            achieved_right_physical_pinch_position=result.geometry["right_pinch_position"],
            method=np.asarray(result.method),
            representation=np.asarray(result.targets["representation"]),
            config_sha256=np.asarray(self.config_sha256),
            implementation_sha256=np.asarray(self.implementation_sha256),
            real_robot_command_allowed=np.asarray(False),
        )
        atomic_npz(
            directory / "g1_hand_action.npz",
            action=hand,
            left_action=left,
            right_action=right,
            timestamps=timestamps,
            fps=np.asarray(result.episode.fps),
            left_joint_names=left_names,
            right_joint_names=right_names,
            left_phase=result.hand["left_phase"].astype("U12"),
            right_phase=result.hand["right_phase"].astype("U12"),
            label_calibration_status=np.asarray(
                self.config["hand_mapping"]["label_calibration_status"]
            ),
            real_robot_command_allowed=np.asarray(False),
        )
        atomic_npz(
            directory / "g1_full_action.npz",
            action=full,
            timestamps=timestamps,
            fps=np.asarray(result.episode.fps),
            joint_names=full_names,
            arm_dimension=np.asarray(14),
            hand_dimension=np.asarray(14),
            method=np.asarray(result.method),
            real_robot_command_allowed=np.asarray(False),
        )
        atomic_json(directory / "retargeting_metrics.json", result.metrics)
        atomic_json(directory / "validation.json", result.validation)
        files = [
            "source_metadata.json", "g1_arm_action.npz", "g1_hand_action.npz",
            "g1_full_action.npz", "retargeting_metrics.json", "validation.json",
        ]
        manifest = {
            "schema_version": "g1_retargeted_episode_v1",
            "method": result.method,
            "episode_id": int(result.episode.episode_id),
            "status": result.validation["status"],
            "config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "implementation_sha256": self.implementation_sha256,
            "implementation_files_sha256": self.implementation_files_sha256,
            "source_dataset": str(self.dataset.root),
            "frame_count": int(len(result.episode.action)),
            "fps": float(result.episode.fps),
            "files": {name: sha256_file(directory / name) for name in files},
            "offline_kinematic_only": True,
            "physics_executed": False,
            "training_executed": False,
            "real_robot_commands": False,
        }
        atomic_json(directory / "manifest.json", manifest)
        return directory


def flatten_episode_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": metrics["episode_id"],
        "status": metrics["status"],
        "frame_count": metrics["frame_count"],
        "fps": metrics["fps"],
        "duration_sec": metrics["duration_sec"],
        "finite_values": metrics["finite_values"],
        "joint_limit_violation_count": metrics["joint_limit_violation_count"],
        "ik_success_rate": metrics["ik_success_rate"],
        "maximum_joint_step_rad": metrics["maximum_joint_step_rad"],
        "velocity_mean_rad_s": metrics["joint_velocity_rad_s"]["mean"],
        "velocity_max_rad_s": metrics["joint_velocity_rad_s"]["max"],
        "acceleration_mean_rad_s2": metrics["joint_acceleration_rad_s2"]["mean"],
        "acceleration_max_rad_s2": metrics["joint_acceleration_rad_s2"]["max"],
        "branch_discontinuity_count": metrics["branch_discontinuity_count"],
        "prohibited_collision_count": metrics["prohibited_arm_self_collision_count"],
        "left_wrist_error_mean_m": metrics["task_space"]["left_wrist_target_error_m"]["mean"],
        "right_wrist_error_mean_m": metrics["task_space"]["right_wrist_target_error_m"]["mean"],
        "left_pinch_error_mean_m": metrics["task_space"]["left_physical_pinch_center_error_m"]["mean"],
        "right_pinch_error_mean_m": metrics["task_space"]["right_physical_pinch_center_error_m"]["mean"],
        "physical_pinch_error_mean_m": 0.5 * (
            metrics["task_space"]["left_physical_pinch_center_error_m"]["mean"]
            + metrics["task_space"]["right_physical_pinch_center_error_m"]["mean"]
        ),
        "task_critical_pinch_error_mean_m": 0.5 * (
            metrics["task_space"]["left_task_critical_pinch_error_m"]["mean"]
            + metrics["task_space"]["right_task_critical_pinch_error_m"]["mean"]
        ),
        "wrist_error_mean_m": 0.5 * (
            metrics["task_space"]["left_wrist_target_error_m"]["mean"]
            + metrics["task_space"]["right_wrist_target_error_m"]["mean"]
        ),
        "midpoint_error_mean_m": metrics["bimanual"]["midpoint_trajectory_error_m"]["mean"],
        "relative_vector_error_mean_m": metrics["bimanual"]["relative_vector_error_m"]["mean"],
        "distance_change_error_mean_m": metrics["bimanual"]["inter_hand_distance_change_error_m"]["mean"],
        "unknown_phase_count": metrics["hand"]["unknown_phase_count"],
    }


def write_method_summary(output_root: str | Path, method: str) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    rows: list[dict[str, Any]] = []
    full_metrics: list[dict[str, Any]] = []
    for directory in sorted((output_root / method).glob("episode_*")):
        path = directory / "retargeting_metrics.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            full_metrics.append(value)
            rows.append(flatten_episode_metrics(value))
    if len(rows) != 50:
        raise RuntimeError(f"{method}: expected 50 exported episode metrics, found {len(rows)}")
    csv_path = output_root / "summary" / f"{method}_episode_metrics.csv"
    atomic_csv(csv_path, rows)
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "episode_id"
    ]
    aggregate = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "median": float(np.median([float(row[key]) for row in rows])),
            "min": float(np.min([float(row[key]) for row in rows])),
            "max": float(np.max([float(row[key]) for row in rows])),
        } for key in numeric_keys
    }
    breakdown = {status: sum(row["status"] == status for row in rows) for status in FINAL_STATUSES}
    summary = {
        "schema_version": "g1_dataset_retargeting_method_summary_v1",
        "method": method,
        "episode_count": len(rows),
        "pass_count": breakdown["PASS"],
        "failure_count": len(rows) - breakdown["PASS"],
        "failure_breakdown": breakdown,
        "aggregate": aggregate,
        "failed_episodes": [],
    }
    for row in rows:
        if row["status"] == "PASS":
            continue
        validation_path = (
            output_root / method / f"episode_{int(row['episode_id']):06d}" / "validation.json"
        )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        summary["failed_episodes"].append({
            "episode_id": row["episode_id"],
            "status": row["status"],
            "first_causal_failure": validation.get("first_causal_failure"),
        })
    atomic_json(output_root / "summary" / f"{method}_summary.json", summary)
    return summary


def write_failure_episode(
    output_root: str | Path,
    method: str,
    episode_id: int,
    status: str,
    reason: str,
    config_path: str | Path,
) -> None:
    if status not in FINAL_STATUSES or status == "PASS":
        status = "FAIL_OTHER"
    directory = Path(output_root).resolve() / method / f"episode_{episode_id:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    empty_arm = np.empty((0, 14), dtype=np.float32)
    empty_hand = np.empty((0, 14), dtype=np.float32)
    atomic_json(directory / "source_metadata.json", {
        "episode_id": episode_id,
        "source_load_status": status,
        "reason": reason,
        "images_duplicated": False,
    })
    atomic_npz(directory / "g1_arm_action.npz", action=empty_arm, real_robot_command_allowed=np.asarray(False))
    atomic_npz(directory / "g1_hand_action.npz", action=empty_hand, real_robot_command_allowed=np.asarray(False))
    atomic_npz(directory / "g1_full_action.npz", action=np.empty((0, 28), dtype=np.float32), real_robot_command_allowed=np.asarray(False))
    metrics = {
        "method": method,
        "episode_id": episode_id,
        "status": status,
        "frame_count": 0,
        "fps": 30.0,
        "duration_sec": 0.0,
        "finite_values": False,
        "g1_joint_shape": [0, 28],
        "g1_arm_shape": [0, 14],
        "g1_hand_shape": [0, 14],
        "joint_limit_violation_count": 0,
        "ik_success_rate": 0.0,
        "maximum_joint_step_rad": 0.0,
        "joint_velocity_rad_s": stats(np.asarray([])),
        "joint_acceleration_rad_s2": stats(np.asarray([])),
        "branch_discontinuity_count": 0,
        "prohibited_arm_self_collision_count": 0,
        "task_space": {
            key: stats(np.asarray([])) for key in (
                "left_wrist_target_error_m", "right_wrist_target_error_m",
                "left_physical_pinch_center_error_m", "right_physical_pinch_center_error_m",
            )
        },
        "bimanual": {
            key: stats(np.asarray([])) for key in (
                "midpoint_trajectory_error_m", "relative_vector_error_m",
                "inter_hand_distance_change_error_m",
            )
        },
        "hand": {"unknown_phase_count": 0},
        "failure_reason": reason,
    }
    validation = {
        "status": status,
        "pass": False,
        "first_causal_failure": {"gate": "exception", "reason": reason},
        "checks": {},
        "real_robot_validation": "NOT_PERFORMED",
        "physics_validation": "NOT_PERFORMED",
        "vla_training": "NOT_PERFORMED",
    }
    atomic_json(directory / "retargeting_metrics.json", metrics)
    atomic_json(directory / "validation.json", validation)
    files = [
        "source_metadata.json", "g1_arm_action.npz", "g1_hand_action.npz",
        "g1_full_action.npz", "retargeting_metrics.json", "validation.json",
    ]
    atomic_json(directory / "manifest.json", {
        "schema_version": "g1_retargeted_episode_v1",
        "method": method,
        "episode_id": episode_id,
        "status": status,
        "config": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": implementation_fingerprint()[0],
        "implementation_files_sha256": implementation_fingerprint()[1],
        "files": {name: sha256_file(directory / name) for name in files},
        "offline_kinematic_only": True,
        "physics_executed": False,
        "training_executed": False,
        "real_robot_commands": False,
    })
