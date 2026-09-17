#!/usr/bin/env python3
"""Render visual-only EVAL35 Fair-A/Proposed-B kinematic replay videos."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.doll_handoff_retargeting.render import _add_box, _add_geom


OUT = ROOT / "outputs/eval35_visual_review"
MANIFEST_DIR = OUT / "manifest"
INDIVIDUAL = OUT / "individual"
MOSAICS = OUT / "mosaics"
HELDOUT8 = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EVAL10 = (
    ROOT
    / "outputs/final_contact_constrained_eval/04_eval10_preparation/EVAL10_RETARGETING_MANIFEST.json"
)
NEW25 = (
    ROOT
    / "outputs/final_direct_physical_eval35/00_preparation/NEW_UNSEEN_25_FROZEN_AB_CONVERSION.json"
)
EVAL35_IDENTITY = (
    ROOT
    / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json"
)
COMMON_CONFIG = (
    ROOT
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
    / "frozen_approval/config/common_config.json"
)
PHYSICAL_DOLL_CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
TASK_REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
PREVIEW_DIR = OUT / "previews"

FPS = 30.0
INDIVIDUAL_WIDTH = 640
INDIVIDUAL_HEIGHT = 480
MOSAIC_WIDTH = 3840
MOSAIC_HEIGHT = 2160
MOSAIC_COLS = 7
MOSAIC_ROWS = 5
CELL_WIDTH = 540
CELL_HEIGHT = 405
GRID_X = 30
GRID_Y = 67
BACKGROUND_BGR = (232, 234, 236)

EXPECTED_NEW25 = (
    "GoPark_20260902_110736",
    "GoPark_20260902_110912",
    "GoPark_20260902_111047",
    "GoPark_20260902_111205",
    "GoPark_20260902_111333",
    "GoPark_20260902_111447",
    "GoPark_20260902_111608",
    "GoPark_20260902_111735",
    "GoPark_20260902_111946",
    "GoPark_20260902_112117",
    "GoPark_20260902_112910",
    "GoPark_20260902_113051",
    "GoPark_20260902_113217",
    "GoPark_20260902_113336",
    "GoPark_20260902_113459",
    "GoPark_20260902_113617",
    "GoPark_20260902_113734",
    "GoPark_20260902_113921",
    "GoPark_20260902_114043",
    "GoPark_20260902_114201",
    "GoPark_20260902_114310",
    "GoPark_20260902_114629",
    "GoPark_20260902_114802",
    "GoPark_20260902_115001",
    "GoPark_20260902_115218",
)


@dataclass(frozen=True)
class Episode:
    eval_index: int
    display_index: int
    stable_episode_id: str
    source_recording: str
    a_path: Path
    a_sha256: str
    b_path: Path
    b_sha256: str
    frames: int


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_matrix_xyzw(quaternion: Iterable[float]) -> np.ndarray:
    """Return a normalized rotation matrix for an [x, y, z, w] quaternion."""
    x, y, z, w = np.asarray(list(quaternion), dtype=np.float64)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("invalid doll orientation quaternion")
    x, y, z, w = (np.asarray((x, y, z, w), dtype=np.float64) / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_degrees_xyzw(quaternion: Iterable[float]) -> float:
    x, y, z, w = np.asarray(list(quaternion), dtype=np.float64)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("invalid doll orientation quaternion")
    x, y, z, w = (np.asarray((x, y, z, w), dtype=np.float64) / norm).tolist()
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def visual_scene_alignment() -> dict[str, Any]:
    """Resolve and validate the authoritative task-scene doll transform."""
    physical = read_json(PHYSICAL_DOLL_CONFIG)
    layout = read_json(SCENE_LAYOUT)
    registration = read_json(TASK_REGISTRATION)
    if registration.get("method_independent") is not True or registration.get("episode_independent") is not True:
        raise RuntimeError("task-scene registration is not method/episode independent")
    if Path(registration["base_physics_config"]).resolve() != PHYSICAL_DOLL_CONFIG.resolve():
        raise RuntimeError("task-scene registration names a different physical proxy config")
    if registration["base_physics_config_sha256"] != sha256_file(PHYSICAL_DOLL_CONFIG):
        raise RuntimeError("task-scene registration physical proxy hash drift")
    if Path(registration["authoritative_task_scene"]).resolve() != SCENE_LAYOUT.resolve():
        raise RuntimeError("task-scene registration names a different authoritative layout")
    if registration["authoritative_task_scene_sha256"] != sha256_file(SCENE_LAYOUT):
        raise RuntimeError("authoritative task-scene layout hash drift")

    doll = physical["object"]
    dimensions = np.asarray(
        physical["frozen_doll_contract"]["visual_dimensions_m"], dtype=np.float64
    )
    surface = float(layout["table"]["surface_height_m"])
    if not math.isclose(surface, float(doll["table_surface_world_z_m"]), abs_tol=1.0e-12):
        raise RuntimeError("task-scene and physical-proxy table surfaces differ")
    center_z = surface + 0.5 * float(dimensions[2]) + float(
        doll["spawn_clearance_above_table_m"]
    )
    current_position = np.asarray(
        [*doll["center_world_xy_m_by_side"]["left"], center_z], dtype=np.float64
    )
    authoritative_position = np.asarray(
        [*registration["registered_doll_center_world_xy_m"], center_z], dtype=np.float64
    )
    current_quaternion = np.asarray(doll["orientation_quaternion_xyzw"], dtype=np.float64)
    authoritative_quaternion = np.asarray(
        registration["registered_doll_orientation_quaternion_xyzw"], dtype=np.float64
    )

    if not np.allclose(
        current_position[:2], registration["previous_physical_doll_center_world_xy_m"], atol=1.0e-12
    ):
        raise RuntimeError("recorded previous physical doll center does not match current renderer source")
    if not np.allclose(
        authoritative_position[:2], layout["doll"]["center_world_xy_m"], atol=1.0e-12
    ):
        raise RuntimeError("registered doll center does not match authoritative Doll-Handoff scene")
    translation = authoritative_position - current_position
    if not np.allclose(translation, registration["common_world_translation_m"], atol=1.0e-12):
        raise RuntimeError("registered doll translation is internally inconsistent")

    current_rotation = rotation_matrix_xyzw(current_quaternion)
    authoritative_rotation = rotation_matrix_xyzw(authoritative_quaternion)
    relative_rotation = authoritative_rotation @ current_rotation.T
    relative_angle_deg = math.degrees(
        math.acos(float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)))
    )
    current_yaw = yaw_degrees_xyzw(current_quaternion)
    authoritative_yaw = yaw_degrees_xyzw(authoritative_quaternion)
    yaw_delta = (authoritative_yaw - current_yaw + 180.0) % 360.0 - 180.0
    if not math.isclose(
        authoritative_yaw,
        float(registration["registered_doll_yaw_deg"]),
        abs_tol=1.0e-10,
    ):
        raise RuntimeError("registered doll quaternion/yaw mismatch")

    def transform(position: np.ndarray, rotation: np.ndarray) -> list[list[float]]:
        value = np.eye(4, dtype=np.float64)
        value[:3, :3] = rotation
        value[:3, 3] = position
        return value.tolist()

    return {
        "schema_version": "eval35_visual_scene_alignment_v1",
        "status": "PASS",
        "authoritative_scene": str(SCENE_LAYOUT.resolve()),
        "authoritative_scene_sha256": sha256_file(SCENE_LAYOUT),
        "task_registration": str(TASK_REGISTRATION.resolve()),
        "task_registration_sha256": sha256_file(TASK_REGISTRATION),
        "physical_proxy_config": str(PHYSICAL_DOLL_CONFIG.resolve()),
        "physical_proxy_config_sha256": sha256_file(PHYSICAL_DOLL_CONFIG),
        "current_eval35_renderer_pose": {
            "position_world_xyz_m": current_position.tolist(),
            "orientation_quaternion_xyzw": current_quaternion.tolist(),
            "yaw_deg": current_yaw,
            "transform_world": transform(current_position, current_rotation),
        },
        "authoritative_pose": {
            "position_world_xyz_m": authoritative_position.tolist(),
            "orientation_quaternion_xyzw": authoritative_quaternion.tolist(),
            "yaw_deg": authoritative_yaw,
            "transform_world": transform(authoritative_position, authoritative_rotation),
        },
        "authoritative_minus_current": {
            "translation_world_xyz_m": translation.tolist(),
            "translation_norm_m": float(np.linalg.norm(translation)),
            "rotation_delta_quaternion_xyzw": authoritative_quaternion.tolist(),
            "rotation_delta_angle_deg": relative_angle_deg,
            "rotation_delta_yaw_deg": yaw_delta,
        },
        "same_transform_for_A_TOP_A_OVERVIEW_B_TOP_B_OVERVIEW": True,
        "g1_pose_changed": False,
        "bin_pose_changed": False,
        "camera_changed": False,
        "trajectory_data_changed": False,
    }


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"expected scalar text, found shape {array.shape}")
    return str(array.reshape(-1)[0])


def trajectory_shape(path: Path) -> tuple[int, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        if "replay_named_joint_qpos" not in archive or "replay_joint_names" not in archive:
            raise RuntimeError(f"not a converted named-joint replay trajectory: {path}")
        q = np.asarray(archive["replay_named_joint_qpos"], dtype=np.float64)
        names = list(map(str, archive["replay_joint_names"]))
    if q.ndim != 2 or q.shape[1] != 28 or not np.isfinite(q).all():
        raise RuntimeError(f"invalid 28-D trajectory: {path}: {q.shape}")
    if len(names) != 28 or len(set(names)) != 28:
        raise RuntimeError(f"invalid named-joint order: {path}")
    return len(q), names


def discover() -> tuple[list[Episode], dict[str, Any]]:
    for path in (
        HELDOUT8,
        EVAL10,
        NEW25,
        EVAL35_IDENTITY,
        COMMON_CONFIG,
        PHYSICAL_DOLL_CONFIG,
        SCENE_LAYOUT,
        TASK_REGISTRATION,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing authoritative visual input: {path}")
    heldout = read_json(HELDOUT8)
    eval10 = read_json(EVAL10)
    new25 = read_json(NEW25)
    identity = read_json(EVAL35_IDENTITY)
    if heldout.get("status") != "PASS" or heldout.get("episode_count") != 8:
        raise RuntimeError("authoritative HELDOUT8 is not PASS/8")
    if eval10.get("status") != "PASS" or eval10.get("evaluation_set") != "EVAL10":
        raise RuntimeError("authoritative EVAL10 converted manifest is not PASS")
    if (
        new25.get("status") != "PASS"
        or new25.get("source_count") != 25
        or new25.get("training_used") is not False
        or new25.get("checkpoint_selection_used") is not False
        or new25.get("controller_tuning_used") is not False
    ):
        raise RuntimeError("authoritative NEW_UNSEEN_25 conversion is not PASS/evaluation-only")
    if identity.get("evaluation_set") != "EVAL35" or len(identity.get("eval_entries", [])) != 35:
        raise RuntimeError("authoritative EVAL35 identity is not exact 35")

    episodes: list[Episode] = []
    for index, row in enumerate(heldout["entries"]):
        episodes.append(
            Episode(
                eval_index=index,
                display_index=index + 1,
                stable_episode_id=str(row["stable_episode_id"]),
                source_recording=str(row["original_source_recording_id"]),
                a_path=Path(row["a_trajectory_path"]).resolve(),
                a_sha256=str(row["a_trajectory_sha256"]),
                b_path=Path(row["b_trajectory_path"]).resolve(),
                b_sha256=str(row["b_trajectory_sha256"]),
                frames=int(row["frames"]),
            )
        )
    eval10_new = {int(row["eval_index"]): row for row in eval10["new_unseen_2"]}
    if set(eval10_new) != {8, 9}:
        raise RuntimeError("EVAL10 does not contain exact converted indices 8 and 9")
    for index in (8, 9):
        row = eval10_new[index]
        episodes.append(
            Episode(
                eval_index=index,
                display_index=index + 1,
                stable_episode_id=str(row["stable_episode_id"]),
                source_recording=str(row["source_name"]),
                a_path=Path(row["a"]["final_retargeted_source"]).resolve(),
                a_sha256=str(row["a"]["final_retargeted_source_sha256"]),
                b_path=Path(row["b"]["final_retargeted_source"]).resolve(),
                b_sha256=str(row["b"]["final_retargeted_source_sha256"]),
                frames=int(row["frames"]),
            )
        )
    new25_rows = {int(row["eval_index"]): row for row in new25["records"]}
    if set(new25_rows) != set(range(10, 35)):
        raise RuntimeError("NEW_UNSEEN_25 converted indices are not exact 10..34")
    if tuple(new25_rows[index]["source_name"] for index in range(10, 35)) != EXPECTED_NEW25:
        raise RuntimeError("NEW_UNSEEN_25 source identity/order differs from requested EVAL35")
    for index in range(10, 35):
        row = new25_rows[index]
        episodes.append(
            Episode(
                eval_index=index,
                display_index=index + 1,
                stable_episode_id=str(row["stable_episode_id"]),
                source_recording=str(row["source_name"]),
                a_path=Path(row["a"]["final_retargeted_source"]).resolve(),
                a_sha256=str(row["a"]["final_retargeted_source_sha256"]),
                b_path=Path(row["b"]["final_retargeted_source"]).resolve(),
                b_sha256=str(row["b"]["final_retargeted_source_sha256"]),
                frames=int(row["frames"]),
            )
        )

    if len(episodes) != 35 or [row.eval_index for row in episodes] != list(range(35)):
        raise RuntimeError("EVAL35 discovery did not produce exact ordered indices 0..34")
    sources = [row.source_recording for row in episodes]
    if len(set(sources)) != 35:
        raise RuntimeError("EVAL35 source recordings are not unique")
    identity_entries = identity["eval_entries"]
    for episode, identity_row in zip(episodes, identity_entries, strict=True):
        if (
            int(identity_row["eval_index"]) != episode.eval_index
            or str(identity_row["stable_episode_id"]) != episode.stable_episode_id
        ):
            raise RuntimeError(f"EVAL35 stable identity mismatch at {episode.eval_index}")
        for path, declared, method in (
            (episode.a_path, episode.a_sha256, "A"),
            (episode.b_path, episode.b_sha256, "B"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing {method} converted trajectory: {path}")
            actual = sha256_file(path)
            if actual != declared:
                raise RuntimeError(
                    f"{method} trajectory hash drift at EVAL {episode.display_index:02d}: {actual} != {declared}"
                )
            frames, _ = trajectory_shape(path)
            if frames != episode.frames:
                raise RuntimeError(
                    f"{method} trajectory frame mismatch at EVAL {episode.display_index:02d}: {frames} != {episode.frames}"
                )

    return episodes, {
        "episode_count": 35,
        "all_unique": True,
        "a_trajectory_count": 35,
        "b_trajectory_count": 35,
        "frame_count_min": min(row.frames for row in episodes),
        "frame_count_max": max(row.frames for row in episodes),
        "shared_mosaic_frames": max(row.frames for row in episodes),
        "shared_mosaic_duration_seconds": max(row.frames for row in episodes) / FPS,
        "authoritative_manifests": {
            "heldout8": str(HELDOUT8.resolve()),
            "heldout8_sha256": sha256_file(HELDOUT8),
            "eval10": str(EVAL10.resolve()),
            "eval10_sha256": sha256_file(EVAL10),
            "new_unseen_25_conversion": str(NEW25.resolve()),
            "new_unseen_25_conversion_sha256": sha256_file(NEW25),
            "eval35_identity": str(EVAL35_IDENTITY.resolve()),
            "eval35_identity_sha256": sha256_file(EVAL35_IDENTITY),
        },
    }


def write_manifests(episodes: list[Episode], metadata: dict[str, Any]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "display_index": f"{episode.display_index:02d}",
            "source_recording": episode.source_recording,
            "method_A_trajectory_path": str(episode.a_path),
            "method_B_trajectory_path": str(episode.b_path),
        }
        for episode in episodes
    ]
    csv_path = MANIFEST_DIR / "EVAL35_MANIFEST.csv"
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    table = [
        f"| {row['display_index']} | {row['source_recording']} | `{row['method_A_trajectory_path']}` | `{row['method_B_trajectory_path']}` |"
        for row in rows
    ]
    atomic_text(
        MANIFEST_DIR / "EVAL35_MANIFEST.md",
        "\n".join(
            [
                "# EVAL35 visual-review trajectory manifest",
                "",
                "Purpose: human visual replay only. No physical evaluator, classifier, automatic success judgment, training, retargeting, or tuning is used.",
                "",
                "- EVAL35 count: **35**",
                "- All source recordings unique: **YES**",
                "- Fair-A converted trajectories: **35/35**",
                "- Proposed-B converted trajectories: **35/35**",
                f"- Common mosaic duration: **{metadata['shared_mosaic_duration_seconds']:.3f} s** ({metadata['shared_mosaic_frames']} frames at 30 FPS)",
                "",
                "| Display | Source recording | Method A — Trajectory-Centric Fair-A | Method B — Interaction-Centric Proposed-B |",
                "|---:|---|---|---|",
                *table,
            ]
        )
        + "\n",
    )

    review_fields = (
        "episode",
        "ACT/Fair-A visual result",
        "ACT/Proposed-B visual result",
        "A_left_grasp",
        "A_handoff",
        "A_bin",
        "A_full_task",
        "B_left_grasp",
        "B_handoff",
        "B_bin",
        "B_full_task",
        "notes",
    )
    review_path = OUT / "EVAL35_HUMAN_REVIEW.csv"
    temporary = review_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        for episode in episodes:
            writer.writerow({"episode": f"{episode.display_index:02d}"})
    os.replace(temporary, review_path)

    atomic_json(
        MANIFEST_DIR / "EVAL35_VISUAL_INPUT_INTEGRITY.json",
        {
            "schema_version": "eval35_visual_review_inputs_v1",
            "status": "PASS",
            **metadata,
            "ordering": [episode.source_recording for episode in episodes],
            "ordering_sha256": hashlib.sha256(
                ("\n".join(episode.source_recording for episode in episodes) + "\n").encode()
            ).hexdigest(),
            "records": [
                {
                    "display_index": episode.display_index,
                    "eval_index": episode.eval_index,
                    "stable_episode_id": episode.stable_episode_id,
                    "source_recording": episode.source_recording,
                    "frames": episode.frames,
                    "a_trajectory": str(episode.a_path),
                    "a_trajectory_sha256": episode.a_sha256,
                    "b_trajectory": str(episode.b_path),
                    "b_trajectory_sha256": episode.b_sha256,
                }
                for episode in episodes
            ],
            "automatic_task_success_evaluation_used": False,
            "physical_evaluator_used": False,
            "trajectory_recomputation_used": False,
        },
    )


def camera_from_eye_target(eye: Iterable[float], target: Iterable[float]) -> mujoco.MjvCamera:
    eye_array = np.asarray(list(eye), dtype=np.float64)
    target_array = np.asarray(list(target), dtype=np.float64)
    relative = eye_array - target_array
    horizontal = float(np.linalg.norm(relative[:2]))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target_array
    camera.distance = float(np.linalg.norm(relative))
    camera.azimuth = math.degrees(math.atan2(-relative[1], -relative[0]))
    camera.elevation = -math.degrees(math.atan2(relative[2], horizontal))
    return camera


class VisualRenderer:
    """Pure named-joint FK renderer; it never steps dynamics or moves the doll."""

    def __init__(self) -> None:
        self.common = load_common_config(COMMON_CONFIG)
        self.layout = copy.deepcopy(load_scene(self.common))
        self.physical = read_json(PHYSICAL_DOLL_CONFIG)
        self.scene_alignment = visual_scene_alignment()
        authoritative_pose = self.scene_alignment["authoritative_pose"]
        self.doll_position = np.asarray(
            authoritative_pose["position_world_xyz_m"], dtype=np.float64
        )
        self.doll_quaternion_xyzw = np.asarray(
            authoritative_pose["orientation_quaternion_xyzw"], dtype=np.float64
        )
        self.doll_rotation = rotation_matrix_xyzw(self.doll_quaternion_xyzw)
        self.g1 = G1Kinematics(self.common, self.layout)
        self.model = self.g1.model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=INDIVIDUAL_HEIGHT, width=INDIVIDUAL_WIDTH
        )
        self.root_position = np.asarray(
            self.layout["g1"]["root_position_world_xyz_m"], dtype=np.float64
        )
        self.root_quaternion = np.asarray(
            self.layout["g1"]["root_orientation_world_wxyz"], dtype=np.float64
        )
        self.model_joint_names = [
            *map(str, self.g1.arm_joint_names),
            *self.g1.hand_joint_names["left"],
            *self.g1.hand_joint_names["right"],
        ]
        presets = self.layout["camera"]["presets"]
        self.camera_parameters = {
            "top": {
                "eye_world_xyz_m": list(map(float, presets["top"]["eye_world_xyz_m"])),
                "target_world_xyz_m": list(map(float, presets["top"]["target_world_xyz_m"])),
            },
            "overview": {
                "eye_world_xyz_m": list(map(float, presets["overview"]["eye_world_xyz_m"])),
                "target_world_xyz_m": list(map(float, presets["overview"]["target_world_xyz_m"])),
            },
        }
        self.cameras = {
            name: camera_from_eye_target(row["eye_world_xyz_m"], row["target_world_xyz_m"])
            for name, row in self.camera_parameters.items()
        }
        self.model.vis.headlight.ambient[:] = (0.52, 0.52, 0.52)
        self.model.vis.headlight.diffuse[:] = (0.78, 0.78, 0.78)
        self.model.vis.headlight.specular[:] = (0.08, 0.08, 0.08)
        self.model.vis.rgba.haze[:] = (0.93, 0.94, 0.95, 1.0)

    def close(self) -> None:
        self.renderer.close()

    def reorder(self, path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as archive:
            q = np.asarray(archive["replay_named_joint_qpos"], dtype=np.float64)
            names = list(map(str, archive["replay_joint_names"]))
        lookup = {name: index for index, name in enumerate(names)}
        if set(lookup) != set(self.model_joint_names):
            raise RuntimeError(f"G1 named-joint mapping mismatch: {path}")
        return q[:, [lookup[name] for name in self.model_joint_names]]

    def _add_scene(self) -> None:
        scene = self.renderer.scene
        layout = self.layout
        table = layout["table"]
        surface = float(table["surface_height_m"])
        width, depth = map(float, table["size_xy_m"])
        thickness = float(table["top_thickness_m"])
        _add_box(
            scene,
            (width, depth, thickness),
            (0.5 * width, 0.5 * depth, surface - 0.5 * thickness),
            np.asarray([0.72, 0.72, 0.70, 1.0], dtype=np.float32),
        )
        for rail in layout["black_frame"]["rails"].values():
            _add_box(
                scene,
                rail["size_xyz_m"],
                rail["center_xyz_m"],
                np.asarray([0.07, 0.07, 0.08, 1.0], dtype=np.float32),
            )
        ground = layout["ground"]
        ground_thickness = float(ground["thickness_m"])
        _add_box(
            scene,
            (*map(float, ground["size_xy_m"]), ground_thickness),
            (
                0.5 * width,
                0.25 * depth,
                float(ground["surface_height_m"]) - 0.5 * ground_thickness,
            ),
            np.asarray([0.82, 0.83, 0.84, 1.0], dtype=np.float32),
        )

        # Accepted physical doll proxy at the authoritative registered LEFT task pose.
        dimensions = np.asarray(self.physical["frozen_doll_contract"]["visual_dimensions_m"])
        _add_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            0.5 * dimensions,
            self.doll_position,
            np.asarray([0.18, 0.62, 0.20, 1.0], dtype=np.float32),
            rotation=self.doll_rotation,
        )

        # Common 150 mm bin, identical for both methods and both views.
        bin_cfg = layout["bin"]
        outer_x, outer_y, _ = map(float, bin_cfg["outer_dimensions_xyz_m"])
        opening_x, opening_y = map(float, bin_cfg["opening_dimensions_xy_m"])
        wall = float(bin_cfg["wall_thickness_m"])
        bottom = float(bin_cfg["bottom_thickness_m"])
        center_x, center_y = map(float, bin_cfg["center_world_xy_m"])
        bin_height = 0.150
        color = np.asarray([0.86, 0.83, 0.68, 1.0], dtype=np.float32)
        _add_box(scene, (outer_x, outer_y, bottom), (center_x, center_y, surface + 0.5 * bottom), color)
        wall_z = surface + 0.5 * bin_height
        _add_box(scene, (outer_x, wall, bin_height), (center_x, center_y - 0.5 * (opening_y + wall), wall_z), color)
        _add_box(scene, (outer_x, wall, bin_height), (center_x, center_y + 0.5 * (opening_y + wall), wall_z), color)
        _add_box(scene, (wall, opening_y, bin_height), (center_x - 0.5 * (opening_x + wall), center_y, wall_z), color)
        _add_box(scene, (wall, opening_y, bin_height), (center_x + 0.5 * (opening_x + wall), center_y, wall_z), color)

        _add_box(
            scene,
            (6.0, 6.0, 0.025),
            (0.4175, 0.20, -0.0125),
            np.asarray([0.84, 0.85, 0.86, 1.0], dtype=np.float32),
        )
        _add_box(
            scene,
            (10.0, 0.035, 4.0),
            (0.4175, 1.02, 1.50),
            np.asarray([0.78, 0.80, 0.82, 1.0], dtype=np.float32),
        )
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    def frame(self, q: np.ndarray, view: str) -> np.ndarray:
        self.data.qpos[:] = self.g1.stand_qpos
        self.data.qpos[self.g1.arm_qpos_ids] = q[:14]
        self.data.qpos[self.g1.hand_qpos_ids["left"]] = q[14:21]
        self.data.qpos[self.g1.hand_qpos_ids["right"]] = q[21:28]
        self.data.qpos[:3] = self.root_position
        self.data.qpos[3:7] = self.root_quaternion
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cameras[view])
        self._add_scene()
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)


def annotate(image: np.ndarray, method: str, episode: Episode, view: str) -> np.ndarray:
    value = np.ascontiguousarray(image)
    color = (220, 145, 45) if method == "A" else (75, 190, 80)
    cv2.rectangle(value, (0, 0), (value.shape[1], 43), (14, 14, 18), -1)
    cv2.putText(
        value,
        f"{method} | {episode.display_index:02d}/35 | {episode.source_recording}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        value,
        f"{view.upper()} | KINEMATIC JOINT REPLAY | DOLL FIXED AT AUTHORITATIVE LEFT TASK POSE",
        (8, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        color,
        1,
        cv2.LINE_AA,
    )
    return value


def atomic_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"could not write preview image: {temporary}")
    os.replace(temporary, path)


def render_scene_alignment_previews(
    episodes: list[Episode], display_index: int
) -> dict[str, Any]:
    episode = episodes[display_index - 1]
    renderer = VisualRenderer()
    previews: dict[str, Any] = {}
    try:
        for method, path in (("A", episode.a_path), ("B", episode.b_path)):
            trajectory = renderer.reorder(path)
            image = annotate(renderer.frame(trajectory[0], "top"), method, episode, "top")
            output = (
                PREVIEW_DIR
                / f"{method}_ep{display_index:02d}_TOP_authoritative_scene_preview.png"
            )
            atomic_image(output, image)
            previews[method] = {
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
                "trajectory": str(path.resolve()),
                "trajectory_sha256": sha256_file(path),
                "trajectory_frame": 0,
                "view": "top",
            }
    finally:
        renderer.close()
    result = {
        **visual_scene_alignment(),
        "preview_episode_display_index": display_index,
        "previews": previews,
        "numerical_transform_verification": "PASS",
        "visual_verification_scope": "A and B representative TOP frame-0 previews",
    }
    atomic_json(MANIFEST_DIR / "SCENE_ALIGNMENT_VERIFICATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


class RawVideoWriter:
    def __init__(self, path: Path, width: int, height: int, encoder: str, preset: str) -> None:
        self.path = path
        self.temporary = path.with_suffix(".incomplete.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.temporary.exists():
            self.temporary.unlink()
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
        ]
        if encoder == "nvenc":
            command += [
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "18",
                "-b:v",
                "0",
            ]
        else:
            command += ["-c:v", "libx264", "-preset", preset, "-crf", "17"]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.temporary)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def finish(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg encode failed ({return_code}): {self.temporary}")
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait()


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,nb_frames,nb_read_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    numerator, denominator = map(float, rate.split("/"))
    count = stream.get("nb_read_frames") or stream.get("nb_frames")
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "duration_seconds": float(duration),
        "frame_count": int(count),
        "pixel_format": stream.get("pix_fmt"),
    }


def individual_path(method: str, view: str, display_index: int) -> Path:
    return INDIVIDUAL / method / view / f"ep{display_index:02d}_{view}.mp4"


def valid_individual(path: Path, frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = probe(path)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        return False
    return bool(
        value["codec"] == "h264"
        and value["width"] == INDIVIDUAL_WIDTH
        and value["height"] == INDIVIDUAL_HEIGHT
        and abs(value["fps"] - FPS) < 0.01
        and value["frame_count"] == frames
    )


def render_individuals(
    episodes: list[Episode],
    start: int,
    end: int,
    resume: bool,
    encoder: str,
    force_scene_rerender: bool,
) -> None:
    renderer = VisualRenderer()
    try:
        for episode in episodes[start - 1 : end]:
            paths = {
                (method, view): individual_path(method, view, episode.display_index)
                for method in ("A", "B")
                for view in ("top", "overview")
            }
            if (
                resume
                and not force_scene_rerender
                and all(valid_individual(path, episode.frames) for path in paths.values())
            ):
                print(f"EVAL {episode.display_index:02d}/35: reuse four verified individual videos", flush=True)
                continue
            existing = [str(path) for path in paths.values() if path.exists()]
            if existing and not force_scene_rerender:
                raise FileExistsError(f"refusing to overwrite existing individual videos: {existing}")
            if existing:
                print(
                    f"EVAL {episode.display_index:02d}/35: atomically replacing four scene-misaligned videos",
                    flush=True,
                )
            trajectories = {
                "A": renderer.reorder(episode.a_path),
                "B": renderer.reorder(episode.b_path),
            }
            writers = {
                key: RawVideoWriter(path, INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT, encoder, "p4" if encoder == "nvenc" else "fast")
                for key, path in paths.items()
            }
            complete = False
            try:
                for frame_index in range(episode.frames):
                    for method in ("A", "B"):
                        for view in ("top", "overview"):
                            image = renderer.frame(trajectories[method][frame_index], view)
                            writers[(method, view)].write(annotate(image, method, episode, view))
                    if frame_index % 200 == 0:
                        print(
                            f"EVAL {episode.display_index:02d}/35: rendered {frame_index + 1}/{episode.frames}",
                            flush=True,
                        )
                for writer in writers.values():
                    writer.finish()
                complete = True
            finally:
                if not complete:
                    for writer in writers.values():
                        if writer.process.poll() is None:
                            writer.abort()
            for path in paths.values():
                if not valid_individual(path, episode.frames):
                    raise RuntimeError(f"individual video verification failed: {path}")
            print(f"EVAL {episode.display_index:02d}/35: four individual videos ready", flush=True)
    finally:
        renderer.close()


def draw_cell_label(canvas: np.ndarray, display_index: int, x: int, y: int) -> None:
    label = f"{display_index:02d}"
    cv2.rectangle(canvas, (x + 5, y + 5), (x + 46, y + 35), (248, 248, 248), -1)
    cv2.rectangle(canvas, (x + 5, y + 5), (x + 46, y + 35), (25, 25, 25), 1)
    cv2.putText(
        canvas,
        label,
        (x + 11, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def mosaic_path(method: str, view: str) -> Path:
    return MOSAICS / f"{method}_EVAL35_{view.upper()}_35SPLIT.mp4"


def compose_mosaic(
    episodes: list[Episode],
    method: str,
    view: str,
    encoder: str,
    resume: bool,
    force_scene_rerender: bool,
) -> None:
    output = mosaic_path(method, view)
    maximum = max(row.frames for row in episodes)
    if output.is_file():
        value = probe(output)
        if resume and not force_scene_rerender and (
            value["codec"] == "h264"
            and value["width"] == MOSAIC_WIDTH
            and value["height"] == MOSAIC_HEIGHT
            and abs(value["fps"] - FPS) < 0.01
            and value["frame_count"] == maximum
        ):
            print(f"{output.name}: reuse verified mosaic", flush=True)
            return
        if not force_scene_rerender:
            raise FileExistsError(f"refusing to overwrite existing mosaic: {output}")
        print(f"{output.name}: atomically replacing scene-misaligned mosaic", flush=True)
    paths = [individual_path(method, view, row.display_index) for row in episodes]
    for path, episode in zip(paths, episodes, strict=True):
        if not valid_individual(path, episode.frames):
            raise RuntimeError(f"missing or invalid individual input: {path}")
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("one or more individual videos could not be decoded")
    cached: list[np.ndarray | None] = [None] * 35
    writer = RawVideoWriter(
        output,
        MOSAIC_WIDTH,
        MOSAIC_HEIGHT,
        encoder,
        "p5" if encoder == "nvenc" else "fast",
    )
    complete = False
    try:
        for frame_index in range(maximum):
            canvas = np.full(
                (MOSAIC_HEIGHT, MOSAIC_WIDTH, 3), BACKGROUND_BGR, dtype=np.uint8
            )
            for cell, (capture, episode) in enumerate(zip(captures, episodes, strict=True)):
                if frame_index < episode.frames:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"decode failed: {paths[cell]} frame {frame_index}")
                    cached[cell] = frame
                frame = cached[cell]
                if frame is None:
                    raise RuntimeError(f"empty mosaic cell: {paths[cell]}")
                resized = cv2.resize(frame, (CELL_WIDTH, CELL_HEIGHT), interpolation=cv2.INTER_AREA)
                row, column = divmod(cell, MOSAIC_COLS)
                x = GRID_X + column * CELL_WIDTH
                y = GRID_Y + row * CELL_HEIGHT
                canvas[y : y + CELL_HEIGHT, x : x + CELL_WIDTH] = resized
                cv2.rectangle(canvas, (x, y), (x + CELL_WIDTH - 1, y + CELL_HEIGHT - 1), (45, 45, 45), 1)
                draw_cell_label(canvas, episode.display_index, x, y)
            writer.write(canvas)
            if frame_index % 100 == 0:
                print(f"{output.name}: composed {frame_index + 1}/{maximum}", flush=True)
        writer.finish()
        complete = True
    finally:
        for capture in captures:
            capture.release()
        if not complete and writer.process.poll() is None:
            writer.abort()
    value = probe(output)
    if not (
        value["codec"] == "h264"
        and value["width"] == MOSAIC_WIDTH
        and value["height"] == MOSAIC_HEIGHT
        and abs(value["fps"] - FPS) < 0.01
        and value["frame_count"] == maximum
    ):
        raise RuntimeError(f"mosaic verification failed: {output}: {value}")


def compose_all(
    episodes: list[Episode], encoder: str, resume: bool, force_scene_rerender: bool
) -> None:
    for method in ("A", "B"):
        for view in ("top", "overview"):
            compose_mosaic(
                episodes, method, view, encoder, resume, force_scene_rerender
            )


def sample_cell_presence(path: Path) -> dict[str, Any]:
    video = probe(path)
    capture = cv2.VideoCapture(str(path))
    sample_indices = (0, video["frame_count"] // 2, video["frame_count"] - 1)
    per_cell = [[] for _ in range(35)]
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not sample mosaic frame {frame_index}: {path}")
        for cell in range(35):
            row, column = divmod(cell, MOSAIC_COLS)
            x = GRID_X + column * CELL_WIDTH
            y = GRID_Y + row * CELL_HEIGHT
            crop = frame[y : y + CELL_HEIGHT, x : x + CELL_WIDTH]
            per_cell[cell].append(
                {"mean": float(crop.mean()), "stddev": float(crop.std())}
            )
    capture.release()
    present = [
        all(sample["mean"] > 8.0 and sample["stddev"] > 3.0 for sample in samples)
        for samples in per_cell
    ]
    return {
        "cells_expected": 35,
        "cells_present": sum(present),
        "all_cells_present": all(present),
        "sample_frame_indices": list(sample_indices),
        "render_integrity_only_not_task_success": True,
    }


def verify_all(episodes: list[Episode], metadata: dict[str, Any]) -> None:
    maximum = max(row.frames for row in episodes)
    individual_probes: dict[str, Any] = {}
    for method in ("A", "B"):
        for view in ("top", "overview"):
            key = f"{method}_{view}"
            paths = [individual_path(method, view, row.display_index) for row in episodes]
            if len(paths) != 35 or any(not valid_individual(path, row.frames) for path, row in zip(paths, episodes, strict=True)):
                raise RuntimeError(f"individual verification failed for {key}")
            individual_probes[key] = {"count": 35, "paths": [str(path) for path in paths]}

    primary: dict[str, Any] = {}
    for method in ("A", "B"):
        for view in ("top", "overview"):
            key = f"{method}_{view.upper()}"
            path = mosaic_path(method, view)
            value = probe(path)
            if not (
                value["codec"] == "h264"
                and value["width"] == MOSAIC_WIDTH
                and value["height"] == MOSAIC_HEIGHT
                and abs(value["fps"] - FPS) < 0.01
                and value["frame_count"] == maximum
            ):
                raise RuntimeError(f"primary video probe failed: {path}: {value}")
            cells = sample_cell_presence(path)
            if not cells["all_cells_present"]:
                raise RuntimeError(f"mosaic cell presence failed: {path}: {cells}")
            primary[key] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                **value,
                **cells,
            }

    renderer = VisualRenderer()
    try:
        scene = {
            "g1_mujoco_model": str(Path(renderer.common["models"]["g1_xml"]).resolve()),
            "g1_mujoco_model_sha256": sha256_file(Path(renderer.common["models"]["g1_xml"])),
            "g1_isaac_usd_reference": str(Path(renderer.common["models"]["g1_usd"]).resolve()),
            "scene_layout": str(SCENE_LAYOUT.resolve()),
            "scene_layout_sha256": sha256_file(SCENE_LAYOUT),
            "physical_doll_config": str(PHYSICAL_DOLL_CONFIG.resolve()),
            "physical_doll_config_sha256": sha256_file(PHYSICAL_DOLL_CONFIG),
            "task_registration": str(TASK_REGISTRATION.resolve()),
            "task_registration_sha256": sha256_file(TASK_REGISTRATION),
            "doll_scene_alignment": renderer.scene_alignment,
            "bin_height_m": 0.150,
            "camera_parameters": renderer.camera_parameters,
            "lighting": {
                "headlight_ambient": [0.52, 0.52, 0.52],
                "headlight_diffuse": [0.78, 0.78, 0.78],
                "headlight_specular": [0.08, 0.08, 0.08],
                "shadows": False,
            },
        }
    finally:
        renderer.close()
    ordering = [episode.source_recording for episode in episodes]
    ordering_sha = hashlib.sha256(("\n".join(ordering) + "\n").encode()).hexdigest()
    result = {
        "schema_version": "eval35_visual_review_verification_v1",
        "status": "PASS",
        "episodes": 35,
        "a_trajectories": 35,
        "b_trajectories": 35,
        "individual_videos": individual_probes,
        "primary_videos": primary,
        "ordering": ordering,
        "ordering_sha256": ordering_sha,
        "same_ordering_all_four": True,
        "common_mosaic_frames": maximum,
        "common_mosaic_duration_seconds": maximum / FPS,
        "shorter_episode_behavior": "hold final decoded frame; no truncation and no time warp",
        "scene": scene,
        "replay_semantics": {
            "robot": "stored authoritative 28-D named G1+Dex3 joint poses replayed kinematically with MuJoCo FK at 30 FPS",
            "doll": "fixed at the authoritative registered LEFT task-scene pose; converted trajectories contain no authoritative doll pose trajectory",
            "doll_dynamics_simulated": False,
            "object_motion_synthesized": False,
            "physics_task_success_claim": False,
            "automatic_success_classifier_used": False,
            "physical_evaluator_used": False,
            "human_visual_review_only": True,
        },
        "input_metadata": metadata,
    }
    atomic_json(MANIFEST_DIR / "VIDEO_VERIFICATION.json", result)
    lines = [
        "# EVAL35 visual replay verification",
        "",
        "Status: **PASS — HUMAN VISUAL REVIEW ONLY**",
        "",
        "| Video | Resolution | FPS | Duration (s) | Codec | Frames | Cells |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for key, row in primary.items():
        lines.append(
            f"| {key} | {row['width']}×{row['height']} | {row['fps']:.3f} | {row['duration_seconds']:.3f} | {row['codec']} | {row['frame_count']} | {row['cells_present']}/35 |"
        )
    lines.extend(
        [
            "",
            "## Replay semantics",
            "",
            "The G1 arms and Dex3 hands replay the stored authoritative converted 28-D named joint poses kinematically at 30 FPS. No IK, retargeting, ACT inference, control tuning, physics evaluation, or success classification runs during rendering.",
            "",
            "The converted trajectories do not contain an authoritative doll pose trajectory. Therefore the accepted physical doll proxy is displayed at the same authoritative registered LEFT task-scene pose in every A/B video. Doll dynamics are not simulated and object motion is not synthesized. These videos show robot motion and interaction geometry; they are not physical task-success evidence.",
            "",
            "All mosaics use the same 7×5 ordering, fixed cameras, scene, lighting, 150 mm bin, 30 FPS timing, and longest-episode duration. Shorter episodes hold their final frame.",
        ]
    )
    atomic_text(MANIFEST_DIR / "VIDEO_VERIFICATION.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "preview", "render", "compose", "verify", "all"),
        default="all",
    )
    parser.add_argument("--episode-start", type=int, default=1, choices=range(1, 36))
    parser.add_argument("--episode-end", type=int, default=35, choices=range(1, 36))
    parser.add_argument("--preview-episode", type=int, default=1, choices=range(1, 36))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-scene-rerender",
        action="store_true",
        help="atomically replace only existing EVAL35 videos affected by scene alignment",
    )
    parser.add_argument("--encoder", choices=("nvenc", "x264"), default="nvenc")
    args = parser.parse_args()
    if args.episode_start > args.episode_end:
        raise ValueError("episode start must be <= episode end")
    episodes, metadata = discover()
    if args.stage in ("prepare", "all"):
        write_manifests(episodes, metadata)
        print(json.dumps(metadata, indent=2), flush=True)
    if args.stage in ("preview", "all"):
        render_scene_alignment_previews(episodes, args.preview_episode)
    if args.stage in ("render", "all"):
        render_individuals(
            episodes,
            args.episode_start,
            args.episode_end,
            args.resume,
            args.encoder,
            args.force_scene_rerender,
        )
    if args.stage in ("compose", "all"):
        compose_all(
            episodes, args.encoder, args.resume, args.force_scene_rerender
        )
    if args.stage in ("verify", "all"):
        verify_all(episodes, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
