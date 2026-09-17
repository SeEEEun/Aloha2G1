"""Paths, immutable contracts, transforms, and deterministic serialization."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
CONFIG_SOURCE = ROOT / "configs/doll_handoff_retargeting"
OUTPUT = ROOT / "outputs/doll_handoff_retargeting"
SOURCE_AUDIT = OUTPUT / "source_audit"
EVENT_AUDIT = OUTPUT / "event_audit"
CONFIG_OUTPUT = OUTPUT / "config"
COMPARISON = OUTPUT / "comparison"
SCENE_CONFIG = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"

COMMON_TEMPLATE = CONFIG_SOURCE / "common_config.template.json"
BASELINE_TEMPLATE = CONFIG_SOURCE / "baseline_config.template.json"
PROPOSED_TEMPLATE = CONFIG_SOURCE / "proposed_config.template.json"
WHOLE_HAND_CONFIG = CONFIG_SOURCE / "dex3_whole_hand.sim.json"

METHODS = ("baseline", "proposed")
ABLATION_METHODS = ("interaction_frame", "interaction_bimanual")
SUPPORTED_METHODS = (*METHODS, *ABLATION_METHODS)
SIDES = ("left", "right")
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
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
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else []))
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_hash(path: str | Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected={expected} actual={actual} path={Path(path)}"
        )


def load_common_config(path: str | Path = COMMON_TEMPLATE) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != "doll_handoff_retargeting_common_v1":
        raise ValueError(f"unexpected common config: {path}")
    forbidden = (
        bool(config.get("training_allowed")),
        bool(config.get("dataset_packaging_allowed")),
        bool(config.get("real_robot_command_allowed")),
    )
    if not config.get("offline_only") or any(forbidden):
        raise RuntimeError("doll-handoff audit must stay offline with training disabled")
    require_hash(config["scene_config"], config["scene_config_sha256"], "scene config")
    require_hash(
        config["models"]["aloha_xml"],
        config["models"]["aloha_xml_sha256"],
        "ALOHA model",
    )
    require_hash(
        config["models"]["g1_xml"],
        config["models"]["g1_xml_sha256"],
        "G1 model",
    )
    require_hash(
        config["models"]["g1_usd"],
        config["models"]["g1_usd_sha256"],
        "G1 USD",
    )
    return config


def load_scene(config: Mapping[str, Any]) -> dict[str, Any]:
    scene = load_json(config["scene_config"])
    if scene.get("task") != "doll_handoff":
        raise RuntimeError(f"approved scene task is not doll_handoff: {scene.get('task')!r}")
    if float(scene["g1"]["pelvis_to_table_front_target_m"]) != 0.15:
        raise RuntimeError("approved G1 pelvis/table-front target changed")
    if float(config["task_registration"]["uniform_metric_scale"]) != 1.0:
        raise RuntimeError("metric task-frame scale is frozen at 1.0")
    return scene


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    if translation is not None:
        result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def inverse_transform(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def quaternion_wxyz_matrix(value: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(value), dtype=np.float64)
    if value.shape != (4,) or not np.isclose(np.linalg.norm(value), 1.0, atol=1e-8):
        raise ValueError(f"invalid WXYZ quaternion: {value}")
    return Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()


def pose_from_scene_robot(scene: Mapping[str, Any], robot: str) -> np.ndarray:
    row = scene[robot]
    return transform(
        quaternion_wxyz_matrix(row["root_orientation_world_wxyz"]),
        row["root_position_world_xyz_m"],
    )


def apply_pose(pose: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    return positions @ pose[:3, :3].T + pose[:3, 3]


def apply_rotation(pose: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjk->tik", pose[:3, :3], np.asarray(rotations))


def rotation_error_vectors(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("tij,tkj->tik", target, current)
    return Rotation.from_matrix(relative).as_rotvec()


def rotation_errors(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(rotation_error_vectors(current, target), axis=1)


def scalar_stats(values: Iterable[float] | np.ndarray) -> dict[str, float]:
    data = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return {key: 0.0 for key in ("mean", "median", "std", "min", "max")}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def branch_flags(q: np.ndarray, absolute: float, multiplier: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    flags = np.zeros(len(q), dtype=bool)
    norms = np.linalg.norm(np.diff(q, axis=0), axis=1)
    for index in range(1, len(q)):
        local = float(np.median(norms[max(0, index - 10) : min(len(norms), index + 9)]))
        flags[index] = norms[index - 1] > max(absolute, multiplier * max(local, 1e-6))
    return flags


def implementation_fingerprint() -> tuple[str, dict[str, str]]:
    files = [
        path
        for path in sorted(PACKAGE.glob("*.py"))
        if path.name != "__pycache__"
    ] + [
        COMMON_TEMPLATE,
        BASELINE_TEMPLATE,
        PROPOSED_TEMPLATE,
        WHOLE_HAND_CONFIG,
        ROOT / "tools/retarget_doll_handoff_batch.py",
        ROOT / "tools/render_doll_handoff_comparisons.py",
        ROOT / "tools/finalize_doll_handoff_retargeting.py",
    ]
    values = {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.is_file()}
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), values


def file_tree_hash(root: Path) -> tuple[str, dict[str, str]]:
    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), files
