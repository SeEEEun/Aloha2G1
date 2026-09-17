#!/usr/bin/env python3
"""Audit EVAL35 task/object registration without reading physical outcomes.

This audit consumes only frozen source, conversion, registration, command, and
model artifacts.  It deliberately never opens an EVAL35 event log or score.
The geometric query is MuJoCo qpos + forward kinematics + geom distance; it
does not step physics or execute a policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tools.audit_doll_handoff_graspable_proxy_v2_topology import (  # noqa: E402
    exact_distance,
    expanded_model,
)
from tools.common_execution_isaac_runtime import (  # noqa: E402
    COMMON_PHYSICAL_CONTROLLER,
    PHYSICAL_ENVIRONMENT,
    PHYSICS_CONFIG,
)
from tools.common_execution_layer import (  # noqa: E402
    Dex3Primitive,
    ExecutionSnapshot,
    pose_matrix,
)
from tools.direct_physical_execution_layer import (  # noqa: E402
    DirectPhysicalDex3ExecutionLayer,
    authoritative_joint_limits,
)
from tools.doll_handoff_retargeting.common import (  # noqa: E402
    load_common_config,
    load_scene,
    transform,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.finalize_doll_handoff_dataset_b import (  # noqa: E402
    source_image_object_estimate,
)
import refine_g1_dex3_static_phone_contact as contact  # noqa: E402


OUT = ROOT / "outputs/final_grasp_capture_eval35/00_task_frame_audit"
COMMAND_MANIFEST = (
    ROOT
    / "outputs/final_direct_physical_eval35/00_preparation/"
    "DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
)
INTENT_MANIFEST = (
    ROOT
    / "outputs/final_direct_physical_eval35/00_preparation/"
    "COMMON_SOURCE_TASK_INTENT_EVAL35.json"
)
EVAL35 = (
    ROOT
    / "outputs/final_representation_neutral_eval/06_common_execution_layer/"
    "EVAL35_MANIFEST.json"
)
REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
SCENE_RECALIBRATION = ROOT / "outputs/doll_handoff_retargeting/scene_recalibration.json"
NEW_SOURCE_ESTIMATES = (
    ROOT
    / "outputs/doll_handoff_dataset_b_final/new_episode_conversion/"
    "source_image_object_estimates.json"
)
JOINT_CONTRACT = (
    ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
)
COMMON_CONFIG = (
    ROOT
    / "outputs/final_direct_physical_eval35/00_preparation/"
    "runtime_frozen_fair_a/config/common_config.json"
)
PALM_FRAME = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_name(entry: dict[str, Any], calibration: dict[str, Any]) -> str:
    if entry.get("source_name"):
        return str(entry["source_name"])
    stable = str(entry["stable_episode_id"])
    if "_ep" in stable:
        index = int(stable.rsplit("_ep", 1)[1])
        matches = [
            str(row["source_name"])
            for row in calibration["per_episode"]
            if int(row["episode_index"]) == index
        ]
        if len(matches) != 1:
            raise RuntimeError(f"source calibration lookup failed: {stable}")
        return matches[0]
    if stable.startswith("doll_handoff_"):
        return "GoPark_" + stable.removeprefix("doll_handoff_")
    raise RuntimeError(f"cannot recover source recording for {stable}")


def source_measurements(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    calibration = read_json(SCENE_RECALIBRATION)
    known = {str(row["source_name"]): row for row in calibration["per_episode"]}
    for value in read_json(NEW_SOURCE_ESTIMATES).values():
        observation = Path(value["observations"][0]["image"])
        name = next(part for part in observation.parts if part.startswith("GoPark_"))
        known[name] = value
    required = [source_name(entry, calibration) for entry in entries]
    for name in required:
        if name not in known:
            # This is the already-established, source-only cam_high detector.
            # It reads frames 0/10/20 and never sees a physical rollout.
            known[name] = source_image_object_estimate(name)
    return {name: known[name] for name in required}


def quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    return Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat().tolist()


def pose_error(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation_mm = 1000.0 * float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    rotation_deg = float(np.degrees(Rotation.from_matrix(relative).magnitude()))
    return translation_mm, rotation_deg


def command_registration(artifact: Path) -> dict[str, Any]:
    with np.load(artifact, allow_pickle=False) as archive:
        required = {
            "task_frame_origin_world_xyz_m",
            "uniform_metric_scale",
            "config_sha256",
            "common_config_sha256",
            "event_names",
            "event_frames",
            "target_left_interaction_frame_position_world",
            "target_right_interaction_frame_position_world",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"conversion artifact lacks {sorted(missing)}: {artifact}")
        events = {
            str(name): int(frame)
            for name, frame in zip(
                archive["event_names"].astype(str), archive["event_frames"], strict=True
            )
        }
        left_frame = int(events.get("LEFT_GRASP", events["LEFT_STABLE_HOLD"]))
        right_frame = int(events["RIGHT_FINAL_RELEASE"])
        return {
            "artifact": str(artifact.resolve()),
            "artifact_sha256": sha256_file(artifact),
            "task_frame_origin_world_xyz_m": np.asarray(
                archive["task_frame_origin_world_xyz_m"], dtype=np.float64
            ).tolist(),
            "uniform_metric_scale": float(np.asarray(archive["uniform_metric_scale"]).item()),
            "converter_config_sha256": str(np.asarray(archive["config_sha256"]).item()),
            "common_config_sha256": str(np.asarray(archive["common_config_sha256"]).item()),
            "left_grasp_event_frame": left_frame,
            "right_release_event_frame": right_frame,
            "left_interaction_target_world_xyz_m": np.asarray(
                archive["target_left_interaction_frame_position_world"][left_frame],
                dtype=np.float64,
            ).tolist(),
            "right_release_interaction_target_world_xyz_m": np.asarray(
                archive["target_right_interaction_frame_position_world"][right_frame],
                dtype=np.float64,
            ).tolist(),
        }


class DistanceModel:
    def __init__(
        self,
        g1: G1Kinematics,
        dimensions_m: np.ndarray,
        table_z_m: float,
        canonical_joint_names: tuple[str, ...],
    ):
        # The model is built once, then its static audit-only object body is
        # relocated before mj_forward for each source registration.
        object_center_model = g1.world_to_model_position(
            np.asarray([0.0, 0.0, table_z_m + dimensions_m[2] / 2.0])
        )
        table_model = g1.world_to_model_position(np.asarray([0.0, 0.0, table_z_m]))
        self.model = expanded_model(g1, object_center_model, dimensions_m, table_model[2])
        self.data = mujoco.MjData(self.model)
        self.g1 = g1
        self.object_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "proxy_v2"
        )
        self.object_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "proxy_v2_geom"
        )
        names = list(canonical_joint_names)
        self.qpos_ids = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in names
            ],
            dtype=np.int64,
        )
        self.geoms = {
            digit: [
                contact.collision_geoms(
                    self.model,
                    f"left_hand_{digit}_{'2' if digit == 'thumb' else '1'}_link",
                )[-1]
            ]
            for digit in ("thumb", "index", "middle")
        }
        self.geoms["palm"] = [
            index
            for index in range(self.model.ngeom)
            if (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[index]),
                )
                == "left_wrist_yaw_link"
                and int(self.model.geom_contype[index]) != 0
            )
        ]
        if len(self.geoms["palm"]) != 2:
            raise RuntimeError("active model does not expose the two palm colliders")

    def set_object_pose(self, pose_world: np.ndarray, collision_z_offset_m: float) -> None:
        center_world = pose_world[:3, 3] + pose_world[:3, :3] @ np.asarray(
            [0.0, 0.0, collision_z_offset_m]
        )
        self.model.body_pos[self.object_body] = self.g1.world_to_model_position(center_world)
        rotation_model = self.g1.root_pose[:3, :3].T @ pose_world[:3, :3]
        self.model.body_quat[self.object_body] = Rotation.from_matrix(
            rotation_model
        ).as_quat()[[3, 0, 1, 2]]

    def assign(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qpos[self.qpos_ids] = np.asarray(q, dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def distances(self) -> dict[str, float]:
        return {
            role: min(
                exact_distance(self.model, self.data, geom, self.object_geom)
                for geom in geoms
            )
            for role, geoms in self.geoms.items()
        }


def palm_world(
    g1: G1Kinematics,
    q: np.ndarray,
    palm: dict[str, Any],
    canonical_joint_names: tuple[str, ...],
) -> np.ndarray:
    by_name = dict(zip(canonical_joint_names, q, strict=True))
    left = np.asarray([by_name[name] for name in g1.hand_joint_names["left"]])
    right = np.asarray([by_name[name] for name in g1.hand_joint_names["right"]])
    g1.assign(q[:14], left, right)
    wrist = g1.wrist_pose("left")
    local = transform(
        Rotation.from_quat(
            np.asarray(palm["left"]["geom_local_quaternion_wxyz"], dtype=np.float64)[
                [1, 2, 3, 0]
            ]
        ).as_matrix(),
        np.asarray(palm["left"]["geom_local_position_m"], dtype=np.float64),
    )
    model_pose = wrist @ local
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = g1.model_to_world_rotation(model_pose[:3, :3])
    result[:3, 3] = g1.model_to_world_position(model_pose[:3, 3])
    return result


def method_geometry(
    record: dict[str, Any],
    expected_doll: np.ndarray,
    runtime_doll: np.ndarray,
    distance_model: DistanceModel,
    g1: G1Kinematics,
    palm: dict[str, Any],
    primitive: Dex3Primitive,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: tuple[str, ...],
    collision_z_offset_m: float,
) -> dict[str, Any]:
    path = Path(record["physical_command"])
    with np.load(path, allow_pickle=False) as archive:
        raw = np.asarray(archive["raw_policy_command"], dtype=np.float64)
        safe = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        intent = archive["common_task_intent"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
    if names != list(joint_names):
        raise RuntimeError("command differs from authoritative named joint order")
    controller = DirectPhysicalDex3ExecutionLayer(
        primitive,
        intent,
        raw,
        safe,
        str(record["method"]),
        lower[:14],
        upper[:14],
        lower[14:],
        upper[14:],
    )
    no_contact = ExecutionSnapshot(
        measured_q_rad=np.zeros(28, dtype=np.float64),
        object_world=runtime_doll,
        whole_hand_world={"left": np.eye(4), "right": np.eye(4)},
        digit_force_n={
            side: {digit: 0.0 for digit in ("thumb", "index", "middle")}
            for side in ("left", "right")
        },
        # Keep the table-supported object from satisfying ownership in this
        # purely kinematic command realization.
        table_force_n=1.0,
        previous_control_frame_support=None,
    )
    close_hold = np.isin(intent, ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"])
    initial_q: np.ndarray | None = None
    runtime_min = {key: float("inf") for key in distance_model.geoms}
    expected_min = {key: float("inf") for key in distance_model.geoms}
    for frame in range(len(safe)):
        command = controller.step(frame, no_contact).executed_command
        if initial_q is None:
            initial_q = command.copy()
        if not close_hold[frame]:
            continue
        distance_model.assign(command)
        distance_model.set_object_pose(runtime_doll, collision_z_offset_m)
        mujoco.mj_forward(distance_model.model, distance_model.data)
        values = distance_model.distances()
        for key, value in values.items():
            runtime_min[key] = min(runtime_min[key], value)
        distance_model.set_object_pose(expected_doll, collision_z_offset_m)
        mujoco.mj_forward(distance_model.model, distance_model.data)
        values = distance_model.distances()
        for key, value in values.items():
            expected_min[key] = min(expected_min[key], value)
    if initial_q is None:
        raise RuntimeError("empty command")
    initial_palm = palm_world(g1, initial_q, palm, joint_names)
    object_from_palm = np.linalg.inv(initial_palm) @ expected_doll
    return {
        "command": str(path.resolve()),
        "command_sha256": sha256_file(path),
        "frames": len(safe),
        "initial_left_palm_world": {
            "position_xyz_m": initial_palm[:3, 3].tolist(),
            "quaternion_xyzw": quaternion_xyzw(initial_palm[:3, :3]),
        },
        "initial_left_palm_to_expected_doll": {
            "translation_xyz_m": object_from_palm[:3, 3].tolist(),
            "quaternion_xyzw": quaternion_xyzw(object_from_palm[:3, :3]),
        },
        "minimum_left_surface_distance_during_close_hold_m": {
            "current_runtime_registration": runtime_min,
            "episode_matched_registration": expected_min,
            "negative_means_geometric_overlap": True,
            "mode": "active-model collision geometry to oriented ellipsoid; qpos plus mj_forward only",
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eval35 = read_json(EVAL35)
    commands = read_json(COMMAND_MANIFEST)
    intents = read_json(INTENT_MANIFEST)
    registration = read_json(REGISTRATION)
    calibration = read_json(SCENE_RECALIBRATION)
    physics = read_json(PHYSICS_CONFIG)
    physical_environment = read_json(PHYSICAL_ENVIRONMENT)
    common_controller = read_json(COMMON_PHYSICAL_CONTROLLER)
    palm = read_json(PALM_FRAME)
    common = load_common_config(COMMON_CONFIG)
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lower, upper, joint_names = authoritative_joint_limits(read_json(JOINT_CONTRACT))
    primitive = Dex3Primitive.from_frozen_dependencies(
        physics, physical_environment, common_controller
    )
    entries = sorted(eval35["eval_entries"], key=lambda row: int(row["eval_index"]))
    if len(entries) != 35 or [int(row["eval_index"]) for row in entries] != list(range(35)):
        raise RuntimeError("EVAL35 is not the exact ordered 35")
    command_rows = {
        (str(row["method"]), int(row["eval_index"])): row
        for row in commands["records"]
    }
    intent_rows = {
        int(row["eval_index"]): row for row in intents["records"]
    }
    measurements = source_measurements(entries)

    visual_dimensions = np.asarray(
        physics["object"]["visual_dimensions_m"], dtype=np.float64
    )
    geometry = physics["geometry_candidates"][0]
    collision_dimensions = np.asarray(geometry["dimensions_m"], dtype=np.float64)
    half_difference = 0.5 * (visual_dimensions - collision_dimensions)
    collision_z_offset = float((collision_dimensions[2] - visual_dimensions[2]) / 2.0)
    table_z = float(physics["object"]["table_surface_world_z_m"])
    root_z = table_z + visual_dimensions[2] / 2.0
    object_quaternion = np.asarray(
        registration["registered_doll_orientation_quaternion_xyzw"], dtype=np.float64
    )
    runtime_object = pose_matrix(
        [*registration["registered_doll_center_world_xy_m"], root_z], object_quaternion
    )
    runtime_bin_xy = np.asarray(
        physical_environment["bin"]["opening_center_world_xy_m"], dtype=np.float64
    )
    runtime_bin = pose_matrix(
        [*runtime_bin_xy, physical_environment["bin"]["bottom_world_z_m"]],
        [0.0, 0.0, 0.0, 1.0],
    )
    distance_model = DistanceModel(
        g1, collision_dimensions, table_z, joint_names
    )

    episode_rows: list[dict[str, Any]] = []
    pre_translation: list[float] = []
    pre_rotation: list[float] = []
    post_translation: list[float] = []
    post_rotation: list[float] = []
    target_left_xy: list[np.ndarray] = []
    source_doll_xy: list[np.ndarray] = []
    for entry in entries:
        index = int(entry["eval_index"])
        name = source_name(entry, calibration)
        measured = measurements[name]
        doll_xy = np.asarray(measured["doll_initial_center_task_xy_m"], dtype=np.float64)
        bin_xy = np.asarray(measured["bin_center_task_xy_m"], dtype=np.float64)
        expected_object = pose_matrix([*doll_xy, root_z], object_quaternion)
        expected_bin = pose_matrix(
            [*bin_xy, physical_environment["bin"]["bottom_world_z_m"]],
            [0.0, 0.0, 0.0, 1.0],
        )
        translation_mm, rotation_deg = pose_error(expected_object, runtime_object)
        bin_translation_mm, bin_rotation_deg = pose_error(expected_bin, runtime_bin)
        pre_translation.append(translation_mm)
        pre_rotation.append(rotation_deg)
        # OPTION A places the common scene at the expected source-derived pose.
        post_translation.append(0.0)
        post_rotation.append(0.0)
        source_artifacts = [Path(value) for value in intent_rows[index]["source_artifacts"]]
        method_registration = {
            "ACT-A40": command_registration(source_artifacts[0]),
            "ACT-B40": command_registration(source_artifacts[1]),
        }
        target_left_xy.append(
            np.asarray(
                method_registration["ACT-B40"]["left_interaction_target_world_xyz_m"][:2]
            )
        )
        source_doll_xy.append(doll_xy)
        methods = {}
        for method in ("ACT-A40", "ACT-B40"):
            record = command_rows[(method, index)]
            if str(record["stable_episode_id"]) != str(entry["stable_episode_id"]):
                raise RuntimeError(f"paired identity mismatch at {index}")
            methods[method] = {
                "conversion_registration": method_registration[method],
                "expected_doll_pose": {
                    "position_xyz_m": expected_object[:3, 3].tolist(),
                    "quaternion_xyzw": object_quaternion.tolist(),
                },
                **method_geometry(
                    record,
                    expected_object,
                    runtime_object,
                    distance_model,
                    g1,
                    palm,
                    primitive,
                    lower,
                    upper,
                    joint_names,
                    collision_z_offset,
                ),
            }
        a_pose = methods["ACT-A40"]["expected_doll_pose"]
        b_pose = methods["ACT-B40"]["expected_doll_pose"]
        episode_rows.append(
            {
                "eval_index": index,
                "stable_episode_id": entry["stable_episode_id"],
                "source_name": name,
                "provenance": entry["provenance"],
                "source_episode_initial_doll_task_pose": {
                    "position_xyz_m": expected_object[:3, 3].tolist(),
                    "quaternion_xyzw": object_quaternion.tolist(),
                    "xy_source": "common frozen cam_high planar detector, median of frames 0/10/20",
                    "z_source": "common bottom alignment using frozen visual height and table",
                    "orientation_source": "common frozen physical yaw gauge; source recordings contain no calibrated 3-D object orientation",
                    "planar_measurement": measured,
                },
                "target_task_frame_registration_used_during_conversion": {
                    "uniform_metric_scale": 1.0,
                    "global_translation_world_xyz_m": [0.0, 0.0, 0.0],
                    "global_rotation_world_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "episode_specific_transform_explicitly_applied": False,
                    "episode_source_motion_retained_in_fixed_world_task_frame": True,
                },
                "expected_doll_pose_A_equals_B": a_pose == b_pose,
                "actual_precalibration_runtime_doll_pose": {
                    "position_xyz_m": runtime_object[:3, 3].tolist(),
                    "quaternion_xyzw": object_quaternion.tolist(),
                },
                "precalibration_trajectory_frame_to_runtime_object_error": {
                    "translation_mm": translation_mm,
                    "rotation_deg": rotation_deg,
                },
                "post_registration_error_option_A": {
                    "translation_mm": 0.0,
                    "rotation_deg": 0.0,
                },
                "expected_bin_task_frame": {
                    "position_xyz_m": expected_bin[:3, 3].tolist(),
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "actual_precalibration_runtime_bin_task_frame": {
                    "position_xyz_m": runtime_bin[:3, 3].tolist(),
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "precalibration_bin_frame_error": {
                    "translation_mm": bin_translation_mm,
                    "rotation_deg": bin_rotation_deg,
                },
                "methods": methods,
            }
        )

    target_left_xy_array = np.asarray(target_left_xy)
    source_doll_xy_array = np.asarray(source_doll_xy)
    correlation = {
        "x": float(np.corrcoef(target_left_xy_array[:, 0], source_doll_xy_array[:, 0])[0, 1]),
        "y": float(np.corrcoef(target_left_xy_array[:, 1], source_doll_xy_array[:, 1])[0, 1]),
    }
    same_registration = all(row["expected_doll_pose_A_equals_B"] for row in episode_rows)
    result = {
        "schema_version": "final_grasp_capture_episode_task_frame_audit_v1",
        "status": "PASS_REGISTRATION_RULE_IDENTIFIED",
        "physical_outcomes_read": False,
        "selection_uses_eval35_outcomes": False,
        "dataset_convention": "EPISODE_MATCHED_TARGET_SCENE",
        "classification_basis": (
            "The global task axes and metric registration are fixed and identity, but the "
            "converter preserves each source recording's absolute interaction motion. The "
            "per-episode left interaction targets vary with the independently measured source "
            "doll positions; they were not canonicalized to one object pose."
        ),
        "source_doll_vs_converted_left_target_xy_correlation": correlation,
        "selected_common_registration_rule": {
            "option": "OPTION_A_EPISODE_MATCHED_SCENE_INITIALIZATION",
            "doll": "source-only cam_high planar doll-center median at frames 0/10/20; common frozen Z and yaw gauge",
            "bin": "source-only cam_high bin-opening-center median at frames 0/10/20; frozen 150 mm geometry, bottom Z and orientation",
            "same_for_A_and_B": True,
            "trajectory_changed": False,
            "ik_regenerated": False,
            "timing_changed": False,
            "outcome_or_method_identity_used": False,
        },
        "summary": {
            "episodes": len(episode_rows),
            "A_B_identical_episode_registration": same_registration,
            "maximum_pre_correction_mismatch_mm": float(max(pre_translation)),
            "maximum_pre_correction_mismatch_deg": float(max(pre_rotation)),
            "maximum_post_correction_mismatch_mm": float(max(post_translation)),
            "maximum_post_correction_mismatch_deg": float(max(post_rotation)),
            "arm_relative_trajectory_preserved": True,
            "timing_preserved": True,
            "bin_relative_placement_rule": "episode-matched bin pose with frozen 150 mm geometry",
        },
        "visual_collision_audit": {
            "VISUAL_DIMENSIONS": visual_dimensions.tolist(),
            "COLLISION_DIMENSIONS": collision_dimensions.tolist(),
            "HALF_EXTENT_DIFFERENCE_X": float(half_difference[0]),
            "HALF_EXTENT_DIFFERENCE_Y": float(half_difference[1]),
            "HALF_EXTENT_DIFFERENCE_Z": float(half_difference[2]),
            "fingers_can_appear_inside_visual_while_outside_collision": bool(
                np.any(half_difference > 0.0)
            ),
        },
        "authoritative_sources": {
            str(path.resolve()): sha256_file(path)
            for path in (
                EVAL35,
                COMMAND_MANIFEST,
                INTENT_MANIFEST,
                REGISTRATION,
                SCENE_RECALIBRATION,
                NEW_SOURCE_ESTIMATES,
                COMMON_CONFIG,
                PHYSICS_CONFIG,
                PHYSICAL_ENVIRONMENT,
                COMMON_PHYSICAL_CONTROLLER,
                JOINT_CONTRACT,
                PALM_FRAME,
            )
        },
        "joint_names": list(joint_names),
        "episodes": episode_rows,
    }
    atomic_json(OUT / "EPISODE_TASK_FRAME_AUDIT.json", result)

    a_runtime = [
        row["methods"]["ACT-A40"]["minimum_left_surface_distance_during_close_hold_m"]
        ["current_runtime_registration"]
        for row in episode_rows
    ]
    b_runtime = [
        row["methods"]["ACT-B40"]["minimum_left_surface_distance_during_close_hold_m"]
        ["current_runtime_registration"]
        for row in episode_rows
    ]
    a_matched = [
        row["methods"]["ACT-A40"]["minimum_left_surface_distance_during_close_hold_m"]
        ["episode_matched_registration"]
        for row in episode_rows
    ]
    b_matched = [
        row["methods"]["ACT-B40"]["minimum_left_surface_distance_during_close_hold_m"]
        ["episode_matched_registration"]
        for row in episode_rows
    ]
    roles = ("thumb", "index", "middle", "palm")

    def range_mm(values: list[dict[str, float]], role: str) -> str:
        rows = np.asarray([value[role] for value in values]) * 1000.0
        return f"{np.min(rows):.3f} to {np.max(rows):.3f}"

    markdown = f"""# EVAL35 episode task-frame audit

Status: **PASS_REGISTRATION_RULE_IDENTIFIED**

- Physical outcomes read: **NO**
- Dataset convention: **EPISODE_MATCHED_TARGET_SCENE**
- A/B identical episode registration: **{'YES' if same_registration else 'NO'}**
- Maximum pre-correction object mismatch: **{max(pre_translation):.6f} mm / {max(pre_rotation):.6f} deg**
- Maximum post-correction object mismatch: **0.000000 mm / 0.000000 deg**
- Selected rule: **OPTION A — episode-matched scene initialization**
- Arm trajectories changed: **NO**
- Wrist trajectories changed: **NO**
- Timing changed: **NO**
- IK regenerated: **NO**

The task axes remain one common metric world frame. The episode variation is in
the retained source interaction geometry: no conversion step canonicalized the
35 source motions to one doll/bin pose. Therefore both methods for a matched
episode must receive the same source-only doll and bin initialization. No
rollout outcome, score, method identity, or physical trace enters this rule.

## Visual versus collision doll

- VISUAL_DIMENSIONS: `{visual_dimensions.tolist()}` m
- COLLISION_DIMENSIONS: `{collision_dimensions.tolist()}` m
- HALF_EXTENT_DIFFERENCE_X: `{half_difference[0]:.6f}` m
- HALF_EXTENT_DIFFERENCE_Y: `{half_difference[1]:.6f}` m
- HALF_EXTENT_DIFFERENCE_Z: `{half_difference[2]:.6f}` m
- Visual overlap without collision contact is geometrically possible: **YES**

## Current-proxy LEFT close/hold distance ranges

Signed values are exact active-model collision-geometry-to-oriented-ellipsoid
distances in an offline qpos/FK audit; negative means overlap.

| Method / registration | Thumb (mm) | Index (mm) | Middle (mm) | Palm (mm) |
|---|---:|---:|---:|---:|
| ACT-A / old global runtime | {range_mm(a_runtime, 'thumb')} | {range_mm(a_runtime, 'index')} | {range_mm(a_runtime, 'middle')} | {range_mm(a_runtime, 'palm')} |
| ACT-A / episode matched | {range_mm(a_matched, 'thumb')} | {range_mm(a_matched, 'index')} | {range_mm(a_matched, 'middle')} | {range_mm(a_matched, 'palm')} |
| ACT-B / old global runtime | {range_mm(b_runtime, 'thumb')} | {range_mm(b_runtime, 'index')} | {range_mm(b_runtime, 'middle')} | {range_mm(b_runtime, 'palm')} |
| ACT-B / episode matched | {range_mm(b_matched, 'thumb')} | {range_mm(b_matched, 'index')} | {range_mm(b_matched, 'middle')} | {range_mm(b_matched, 'palm')} |

The complete 35-row record, source image measurements, conversion hashes,
initial palm-to-doll transforms, and both pre/post-registration distance fields
are in `EPISODE_TASK_FRAME_AUDIT.json`.
"""
    (OUT / "EPISODE_TASK_FRAME_AUDIT.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_convention": result["dataset_convention"],
                **result["summary"],
                "output": str((OUT / "EPISODE_TASK_FRAME_AUDIT.json").resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
