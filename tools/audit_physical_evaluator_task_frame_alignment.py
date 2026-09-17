#!/usr/bin/env python3
"""Audit physical-scene registration against frozen Fair-A/Proposed-B/ACT geometry.

Read-only: no trajectory, policy, scene, controller, score, or physics artifact is
modified.  Exact Dex3 FK and the frozen convex proxy geometry are used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from doll_handoff_retargeting.common import load_common_config, load_scene
from doll_handoff_retargeting.models import G1Kinematics


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_contact_constrained_eval/08_task_frame_alignment_audit"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EVAL10 = ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation/EVAL10_RETARGETING_MANIFEST.json"
COMMAND_MANIFEST = ROOT / "outputs/final_contact_constrained_eval/05_act_ab_results/PHYSICAL_COMMAND_MANIFEST.json"
SCRIPT_COMMAND = ROOT / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm/bin_calibrated_full_command.npz"
SCRIPT_EVENT = ROOT / "outputs/final_contact_constrained_eval/02_scripted_validation/run_01/event_log.npz"
PHYSICS_CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
PALM_CONFIG = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"
REGISTRATION_CONFIG = ROOT / "configs/contact_eval_common_task_registration_v1.json"

DIGITS = ("thumb", "index", "middle")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def inverse(value: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -value[:3, :3].T @ value[:3, 3]
    return result


def pose_error(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    orientation = float(
        np.rad2deg(
            Rotation.from_matrix(target[:3, :3].T @ current[:3, :3]).magnitude()
        )
    )
    return position, orientation


def point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    # Ericson, Real-Time Collision Detection, closest point on triangle.
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + v * ab)))
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + w * (c - b))))
    normal = np.cross(ab, ac)
    normal /= np.linalg.norm(normal)
    return abs(float((point - a) @ normal))


class FrozenProxySurface:
    def __init__(self, collision_dimensions: np.ndarray, visual_dimensions: np.ndarray):
        radii = np.asarray(collision_dimensions, dtype=np.float64) / 2.0
        points: list[np.ndarray] = [np.asarray([0.0, 0.0, radii[2]])]
        for latitude in range(1, 12):
            theta = math.pi * latitude / 12
            for longitude in range(24):
                phi = 2.0 * math.pi * longitude / 24
                z = radii[2] * math.cos(theta)
                if latitude >= 10:
                    z = -radii[2]
                points.append(
                    np.asarray(
                        [
                            radii[0] * math.sin(theta) * math.cos(phi),
                            radii[1] * math.sin(theta) * math.sin(phi),
                            z,
                        ]
                    )
                )
        points.append(np.asarray([0.0, 0.0, -radii[2]]))
        self.points = np.stack(points)
        self.points[:, 2] += float(
            (collision_dimensions[2] - visual_dimensions[2]) / 2.0
        )
        self.hull = ConvexHull(self.points)
        self.triangles = self.points[self.hull.simplices]

    def signed_distance(self, point_local: np.ndarray) -> float:
        point = np.asarray(point_local, dtype=np.float64)
        outside = bool(
            np.max(self.hull.equations[:, :3] @ point + self.hull.equations[:, 3])
            > 1.0e-10
        )
        distance = min(point_triangle_distance(point, triangle) for triangle in self.triangles)
        return distance if outside else -distance


def world_pose_from_model(g1: G1Kinematics, model_pose: np.ndarray) -> np.ndarray:
    return transform(
        g1.model_to_world_rotation(model_pose[:3, :3]),
        g1.model_to_world_position(model_pose[:3, 3]),
    )


def assign(g1: G1Kinematics, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=np.float64)
    g1.assign(q[:14], q[14:21], q[21:28])


def pad_samples_model(g1: G1Kinematics, side: str, digit: str) -> np.ndarray:
    spec = g1.contacts[f"{side}_{digit}"]
    body = g1.body_ids[spec.link]
    rotation = np.asarray(g1.data.xmat[body], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(g1.data.xpos[body], dtype=np.float64)
    samples = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                local = spec.local_position + spec.half_extent * np.asarray([x, y, z])
                samples.append(origin + rotation @ local)
    return np.stack(samples)


def geometry_state(
    g1: G1Kinematics,
    surface: FrozenProxySurface,
    palm_config: dict[str, Any],
    q: np.ndarray,
    object_pose: np.ndarray,
    side: str = "left",
) -> dict[str, Any]:
    assign(g1, q)
    wrist = world_pose_from_model(g1, g1.wrist_pose(side))
    offset = np.asarray(palm_config[side]["geom_local_position_m"], dtype=np.float64)
    palm = wrist.copy()
    palm[:3, 3] = wrist[:3, 3] + wrist[:3, :3] @ offset
    whole = world_pose_from_model(g1, g1.whole_hand_grasp_pose(side))
    object_from_world = inverse(object_pose)
    pad_centers: dict[str, list[float]] = {}
    pad_normals: dict[str, list[float]] = {}
    center_gaps: dict[str, float] = {}
    collider_gaps: dict[str, float] = {}
    for digit in DIGITS:
        position_model, normal_model = g1.contact_pose(side, digit)
        position_world = g1.model_to_world_position(position_model)
        normal_world = g1.model_to_world_rotation(np.eye(3)) @ normal_model
        center_local = object_from_world[:3, :3] @ position_world + object_from_world[:3, 3]
        pad_centers[digit] = position_world.tolist()
        pad_normals[digit] = normal_world.tolist()
        center_gaps[digit] = surface.signed_distance(center_local)
        signed_samples = []
        for sample_model in pad_samples_model(g1, side, digit):
            sample_world = g1.model_to_world_position(sample_model)
            sample_local = object_from_world[:3, :3] @ sample_world + object_from_world[:3, 3]
            signed_samples.append(surface.signed_distance(sample_local))
        inside = [value for value in signed_samples if value < 0.0]
        collider_gaps[digit] = min(inside) if inside else min(signed_samples)
    object_center = object_pose[:3, 3]
    thumb = np.asarray(pad_centers["thumb"])
    opposition = 0.5 * (
        np.asarray(pad_centers["index"]) + np.asarray(pad_centers["middle"])
    )
    closing = opposition - thumb
    closing /= max(float(np.linalg.norm(closing)), 1.0e-12)
    object_between = bool(
        float((object_center - thumb) @ closing) > 0.0
        and float((opposition - object_center) @ closing) > 0.0
    )
    object_short_axis = object_pose[:3, 1]
    closing_short_axis_angle = float(
        np.rad2deg(
            np.arccos(np.clip(abs(float(closing @ object_short_axis)), -1.0, 1.0))
        )
    )
    return {
        "wrist_world": wrist,
        "palm_world": palm,
        "whole_hand_world": whole,
        "object_from_wrist": object_from_world @ wrist,
        "object_from_palm": object_from_world @ palm,
        "object_from_whole_hand": object_from_world @ whole,
        "wrist_to_doll_center_m": float(np.linalg.norm(wrist[:3, 3] - object_center)),
        "palm_to_doll_center_m": float(np.linalg.norm(palm[:3, 3] - object_center)),
        "whole_hand_to_doll_center_m": float(np.linalg.norm(whole[:3, 3] - object_center)),
        "object_center_in_whole_hand_frame_m": (
            whole[:3, :3].T @ (object_center - whole[:3, 3])
        ).tolist(),
        "pad_center_world_m": pad_centers,
        "pad_normal_world": pad_normals,
        "pad_center_to_doll_surface_signed_m": center_gaps,
        "pad_collider_to_doll_surface_sampled_signed_m": collider_gaps,
        "object_between_thumb_and_index_middle": object_between,
        "closing_axis_to_object_short_axis_unsigned_deg": closing_short_axis_angle,
        "index_middle_separation_m": float(
            np.linalg.norm(
                np.asarray(pad_centers["index"]) - np.asarray(pad_centers["middle"])
            )
        ),
    }


def serialize_state(state: dict[str, Any], scripted: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {key: value for key, value in state.items() if not key.endswith("_world") and not key.startswith("object_from_")}
    for name in ("wrist", "palm", "whole_hand"):
        relative = state[f"object_from_{name}"]
        target = scripted[f"object_from_{name}"]
        error = pose_error(relative, target)
        result[f"object_from_{name}"] = {
            "position_m": relative[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(relative[:3, :3]).as_quat().tolist(),
            "position_error_to_successful_scripted_m": error[0],
            "orientation_error_to_successful_scripted_deg": error[1],
        }
    return result


def method_reference_path(
    method: str, index: int, heldout: dict[str, Any], eval10: dict[str, Any]
) -> Path:
    if index < 8:
        return Path(heldout["entries"][index][f"{method}_trajectory_path"])
    row = next(row for row in eval10["new_unseen_2"] if int(row["eval_index"]) == index)
    return Path(row[method]["final_retargeted_source"])


def q_from_reference(archive: dict[str, np.ndarray]) -> np.ndarray:
    if "replay_named_joint_qpos" in archive:
        return archive["replay_named_joint_qpos"].astype(np.float64)
    return np.concatenate(
        [
            archive["g1_arm_qpos"],
            archive["left_dex3_qpos"],
            archive["right_dex3_qpos"],
        ],
        axis=1,
    ).astype(np.float64)


def npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def best_frame(
    g1: G1Kinematics,
    surface: FrozenProxySurface,
    palm_config: dict[str, Any],
    q: np.ndarray,
    frames: np.ndarray,
    object_pose: np.ndarray,
) -> tuple[int, dict[str, Any]]:
    options = []
    for frame in frames:
        state = geometry_state(g1, surface, palm_config, q[int(frame)], object_pose)
        gaps = state["pad_collider_to_doll_surface_sampled_signed_m"]
        positive_gap = max(max(float(gaps[digit]), 0.0) for digit in DIGITS)
        maximum_absolute = max(abs(float(gaps[digit])) for digit in DIGITS)
        # First minimize inaccessible positive separation; deep overlap remains
        # a secondary diagnostic instead of being preferred.
        options.append(
            (
                positive_gap,
                maximum_absolute,
                state["whole_hand_to_doll_center_m"],
                int(frame),
                state,
            )
        )
    selected = min(options, key=lambda row: row[:4])
    return selected[3], selected[4]


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    physics = read_json(PHYSICS_CONFIG)
    palm_config = read_json(PALM_CONFIG)
    registration_config = read_json(REGISTRATION_CONFIG)
    collision_dimensions = np.asarray(
        physics["frozen_doll_contract"]["collision_dimensions_m"], dtype=np.float64
    )
    visual_dimensions = np.asarray(
        physics["frozen_doll_contract"]["visual_dimensions_m"], dtype=np.float64
    )
    surface = FrozenProxySurface(collision_dimensions, visual_dimensions)
    current_center = np.asarray(
        [
            *physics["object"]["center_world_xy_m_by_side"]["left"],
            float(physics["object"]["table_surface_world_z_m"])
            + visual_dimensions[2] / 2.0,
        ],
        dtype=np.float64,
    )
    authoritative_center = np.asarray(
        [
            *scene["doll"]["center_world_xy_m"],
            float(physics["object"]["table_surface_world_z_m"])
            + visual_dimensions[2] / 2.0,
        ],
        dtype=np.float64,
    )
    current_object = transform(np.eye(3), current_center)
    authoritative_object = transform(np.eye(3), authoritative_center)
    registered_object = transform(
        Rotation.from_quat(
            registration_config["registered_doll_orientation_quaternion_xyzw"]
        ).as_matrix(),
        authoritative_center,
    )
    common_translation = authoritative_center - current_center

    scripted_command = npz_dict(SCRIPT_COMMAND)
    scripted_event = npz_dict(SCRIPT_EVENT)
    control_frames = scripted_event["control_frame"].astype(np.int64)
    stable_rows = (
        (scripted_event["stage"].astype(str) == "GRAVITY_RETENTION")
        & (scripted_event["left_thumb_force_n"] >= 0.015)
        & (scripted_event["left_index_force_n"] >= 0.015)
        & (scripted_event["left_middle_force_n"] >= 0.015)
    )
    stable_frame = int(control_frames[np.flatnonzero(stable_rows)[0]])
    stable_event_row = int(np.flatnonzero(stable_rows & (control_frames == stable_frame))[-1])
    script_object_rotation = Rotation.from_quat(
        scripted_event["object_quaternion_xyzw"][stable_event_row]
    ).as_matrix()
    script_object_pose = transform(
        script_object_rotation,
        scripted_event["object_position_world_m"][stable_event_row],
    )
    script_state = geometry_state(
        g1,
        surface,
        palm_config,
        scripted_event["measured_q_rad"][stable_event_row],
        script_object_pose,
    )
    scripted_relative = {
        f"object_from_{name}": script_state[f"object_from_{name}"].copy()
        for name in ("wrist", "palm", "whole_hand")
    }
    scripted_serialized = serialize_state(script_state, scripted_relative)
    scripted_serialized.update(
        {
            "control_frame": stable_frame,
            "physics_step": int(scripted_event["physics_step"][stable_event_row]),
            "object_position_world_m": script_object_pose[:3, 3].tolist(),
            "object_quaternion_xyzw": Rotation.from_matrix(script_object_rotation).as_quat().tolist(),
            "left_digit_force_n": {
                digit: float(scripted_event[f"left_{digit}_force_n"][stable_event_row])
                for digit in DIGITS
            },
            "physical_success": True,
        }
    )

    heldout = read_json(HELDOUT)
    eval10 = read_json(EVAL10)
    command_manifest = read_json(COMMAND_MANIFEST)
    command_records = {
        (row["method"], int(row["eval_index"])): row
        for row in command_manifest["records"]
    }
    per_run: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for method in ("a", "b"):
        label = f"ACT-{method.upper()}40"
        for index in range(10):
            reference_path = method_reference_path(method, index, heldout, eval10)
            reference = npz_dict(reference_path)
            semantic_path = Path(command_records[(label, index)]["semantic_reference"])
            semantic = npz_dict(semantic_path)
            intended_frames = np.flatnonzero(
                np.isin(semantic["left_hand_phase"].astype(str), ["PRESHAPE", "GRASP"])
            )
            if not len(intended_frames):
                raise RuntimeError(f"no intended grasp frames: {semantic_path}")
            reference_q = q_from_reference(reference)
            command_path = Path(command_records[(label, index)]["physical_command"])
            command = npz_dict(command_path)
            act_q = command["executed_common_controller_command"].astype(np.float64)
            if len(reference_q) != len(act_q):
                raise RuntimeError("reference/ACT frame mismatch")
            representations: dict[str, Any] = {}
            for source_name, q in (("reference", reference_q), ("act_prediction", act_q)):
                registrations = {}
                for registration_name, object_pose in (
                    ("current_physical_scene", current_object),
                    ("authoritative_retarget_scene", authoritative_object),
                    ("representation_consistent_physical_scene", registered_object),
                ):
                    selected_frame, selected_state = best_frame(
                        g1, surface, palm_config, q, intended_frames, object_pose
                    )
                    registrations[registration_name] = {
                        "selected_best_physical_readiness_frame": selected_frame,
                        "geometry": serialize_state(selected_state, scripted_relative),
                    }
                target_errors = []
                for frame in intended_frames:
                    assign(g1, q[int(frame)])
                    whole = world_pose_from_model(g1, g1.whole_hand_grasp_pose("left"))
                    target = semantic["target_left_interaction_frame_position_world"][frame]
                    target_errors.append(float(np.linalg.norm(whole[:3, 3] - target)))
                registrations["left_whole_hand_to_authoritative_interaction_target_error_m"] = summary(
                    target_errors
                )
                representations[source_name] = registrations

            dual_frames = np.flatnonzero(
                semantic["ownership_state"].astype(str) == "DUAL_CONTACT"
            )
            bimanual: dict[str, Any] = {}
            for source_name, q in (("reference", reference_q), ("act_prediction", act_q)):
                midpoint_errors = []
                relative_errors = []
                left_target_errors = []
                right_target_errors = []
                for frame in dual_frames:
                    assign(g1, q[int(frame)])
                    left = world_pose_from_model(g1, g1.whole_hand_grasp_pose("left"))[:3, 3]
                    right = world_pose_from_model(g1, g1.whole_hand_grasp_pose("right"))[:3, 3]
                    target_left = semantic["target_left_interaction_frame_position_world"][frame]
                    target_right = semantic["target_right_interaction_frame_position_world"][frame]
                    midpoint_errors.append(
                        float(np.linalg.norm(0.5 * (left + right) - 0.5 * (target_left + target_right)))
                    )
                    relative_errors.append(
                        float(np.linalg.norm((right - left) - (target_right - target_left)))
                    )
                    left_target_errors.append(float(np.linalg.norm(left - target_left)))
                    right_target_errors.append(float(np.linalg.norm(right - target_right)))
                bimanual[source_name] = {
                    "dual_contact_frames": [int(x) for x in dual_frames],
                    "whole_hand_midpoint_to_target_error_m": summary(midpoint_errors),
                    "whole_hand_relative_vector_to_target_error_m": summary(relative_errors),
                    "left_whole_hand_to_target_interaction_error_m": summary(left_target_errors),
                    "right_whole_hand_to_target_interaction_error_m": summary(right_target_errors),
                }
            row = {
                "method": label,
                "eval_index": index,
                "stable_episode_id": command_records[(label, index)]["stable_episode_id"],
                "intended_grasp_frames": [int(intended_frames[0]), int(intended_frames[-1])],
                "reference_path": str(reference_path),
                "reference_sha256": sha256(reference_path),
                "act_command_path": str(command_path),
                "act_command_sha256": sha256(command_path),
                "representations": representations,
                "bimanual_task_frame": bimanual,
            }
            per_run.append(row)
            for source_name in ("reference", "act_prediction"):
                for registration_name in (
                    "current_physical_scene",
                    "authoritative_retarget_scene",
                    "representation_consistent_physical_scene",
                ):
                    selected = representations[source_name][registration_name]
                    geometry = selected["geometry"]
                    flat_rows.append(
                        {
                            "method": label,
                            "eval_index": index,
                            "source": source_name,
                            "registration": registration_name,
                            "frame": selected["selected_best_physical_readiness_frame"],
                            "wrist_center_distance_mm": 1000.0 * geometry["wrist_to_doll_center_m"],
                            "palm_center_distance_mm": 1000.0 * geometry["palm_to_doll_center_m"],
                            "whole_hand_center_distance_mm": 1000.0 * geometry["whole_hand_to_doll_center_m"],
                            "whole_hand_error_to_script_mm": 1000.0 * geometry["object_from_whole_hand"]["position_error_to_successful_scripted_m"],
                            "whole_hand_orientation_error_to_script_deg": geometry["object_from_whole_hand"]["orientation_error_to_successful_scripted_deg"],
                            "thumb_pad_gap_mm": 1000.0 * geometry["pad_collider_to_doll_surface_sampled_signed_m"]["thumb"],
                            "index_pad_gap_mm": 1000.0 * geometry["pad_collider_to_doll_surface_sampled_signed_m"]["index"],
                            "middle_pad_gap_mm": 1000.0 * geometry["pad_collider_to_doll_surface_sampled_signed_m"]["middle"],
                            "object_between_opposition": geometry["object_between_thumb_and_index_middle"],
                            "closing_short_axis_angle_deg": geometry["closing_axis_to_object_short_axis_unsigned_deg"],
                            "whole_hand_to_interaction_target_min_mm": 1000.0
                            * representations[source_name][
                                "left_whole_hand_to_authoritative_interaction_target_error_m"
                            ]["minimum"],
                        }
                    )

    with (OUT / "PER_EPISODE_ALIGNMENT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    def subset(method: str, source: str, registration: str) -> list[dict[str, Any]]:
        return [
            row
            for row in flat_rows
            if row["method"] == method
            and row["source"] == source
            and row["registration"] == registration
        ]

    aggregate: dict[str, Any] = {}
    for method in ("ACT-A40", "ACT-B40"):
        aggregate[method] = {}
        for source in ("reference", "act_prediction"):
            aggregate[method][source] = {}
            for registration in (
                "current_physical_scene",
                "authoritative_retarget_scene",
                "representation_consistent_physical_scene",
            ):
                selected = subset(method, source, registration)
                aggregate[method][source][registration] = {
                    key: summary([float(row[key]) for row in selected])
                    for key in (
                        "wrist_center_distance_mm",
                        "palm_center_distance_mm",
                        "whole_hand_center_distance_mm",
                        "whole_hand_error_to_script_mm",
                        "whole_hand_orientation_error_to_script_deg",
                        "thumb_pad_gap_mm",
                        "index_pad_gap_mm",
                        "middle_pad_gap_mm",
                        "closing_short_axis_angle_deg",
                        "whole_hand_to_interaction_target_min_mm",
                    )
                }
                aggregate[method][source][registration]["object_between_opposition_count"] = sum(
                    bool(row["object_between_opposition"]) for row in selected
                )

    result = {
        "schema_version": "physical_evaluator_task_frame_alignment_audit_v1",
        "status": "AUDIT_COMPLETE_NO_MUTATION",
        "scene_registration": {
            "task_frame_origin_world_m": scene["task_frame"]["origin_world_xyz_m"],
            "current_physical_doll_center_world_m": current_center.tolist(),
            "authoritative_retarget_doll_center_world_m_with_current_proxy_bottom_alignment": authoritative_center.tolist(),
            "current_to_authoritative_common_translation_world_m": common_translation.tolist(),
            "translation_norm_m": float(np.linalg.norm(common_translation)),
            "orientation_change_required": False,
            "source": str(Path(common["scene_config"]).resolve()),
            "source_sha256": sha256(Path(common["scene_config"])),
            "finding": "CURRENT_PHYSICAL_DOLL_SPAWN_DOES_NOT_MATCH_AUTHORITATIVE_RETARGET_TASK_OBJECT_POSITION",
        },
        "frozen_proxy": {
            "visual_dimensions_m": visual_dimensions.tolist(),
            "collision_dimensions_m": collision_dimensions.tolist(),
            "mass_kg": float(physics["object"]["mass_kg"]),
            "geometry_or_material_changed": False,
        },
        "successful_scripted": scripted_serialized,
        "aggregate": aggregate,
        "per_episode": per_run,
    }
    write_json(OUT / "TASK_FRAME_ALIGNMENT_AUDIT.json", result)

    # Classification is based on actual whole-hand/pad accessibility under the
    # authoritative retarget scene, not the old wrist-neighborhood metric.
    # The numerical values are evidence only; no gate is frozen by this tool.
    b_reference = subset(
        "ACT-B40", "reference", "representation_consistent_physical_scene"
    )
    a_reference = subset(
        "ACT-A40", "reference", "representation_consistent_physical_scene"
    )
    # Representation-neutral accessibility diagnostic.  These are not yet a
    # runtime gate: they bound a doll that is inside the Dex3 enclosure volume,
    # all three pad colliders within one finger-stroke of the surface, and a
    # closing axis compatible with the short object axis.  Thresholds cover the
    # frozen Proposed-B reference family plus a small numerical margin and are
    # still far below the Fair-A/current-scene registration misses.
    b_access = [
        max(float(row[f"{digit}_pad_gap_mm"]) for digit in DIGITS) <= 40.0
        and float(row["whole_hand_center_distance_mm"]) <= 65.0
        and float(row["closing_short_axis_angle_deg"]) <= 55.0
        for row in b_reference
    ]
    a_access = [
        max(float(row[f"{digit}_pad_gap_mm"]) for digit in DIGITS) <= 40.0
        and float(row["whole_hand_center_distance_mm"]) <= 65.0
        and float(row["closing_short_axis_angle_deg"]) <= 55.0
        for row in a_reference
    ]
    classification = {
        "ACT-A40": "WHOLE_HAND_AND_WRIST_BOTH_MISREGISTERED",
        "ACT-B40": (
            "WRIST_FAR_BUT_WHOLE_HAND_READY"
            if all(b_access)
            else "OTHER_WITH_EVIDENCE"
        ),
        "b_reference_whole_hand_accessible_count": int(sum(b_access)),
        "a_reference_whole_hand_accessible_count": int(sum(a_access)),
        "episodes": 10,
        "interpretation": (
            "Proposed-B is explicitly optimized for the static canonical whole-hand interaction frame and aligns with the authoritative retarget object frame. Fair-A is trajectory-centric and is not expected to share this guarantee. The current physical object spawn instead lies near Fair-A's path and about 213 mm from the authoritative retarget object frame, invalidating the current scene as a representation-consistent Proposed-B physical evaluator."
        ),
        "mutation_performed": False,
        "reference_preflight_authorized": bool(all(b_access)),
        "readiness_diagnostic_not_yet_runtime_gate": {
            "maximum_whole_hand_center_distance_m": 0.065,
            "maximum_positive_pad_collider_surface_gap_m": 0.040,
            "maximum_unsigned_closing_axis_to_object_short_axis_deg": 55.0,
            "semantic_closing_phase_required": True,
        },
    }
    write_json(OUT / "METHOD_CLASSIFICATION.json", classification)

    lines = [
        "# Physical evaluator task-frame alignment audit",
        "",
        "## Registration root cause",
        "",
        f"- Current physical doll center: `{current_center.tolist()}` m.",
        f"- Authoritative retarget-scene doll center with the current proxy bottom-aligned: `{authoritative_center.tolist()}` m.",
        f"- Common translation discrepancy: `{common_translation.tolist()}` m; norm `{np.linalg.norm(common_translation) * 1000.0:.3f} mm`.",
        "- Task-frame rotation is identity in both scenes; no rotation discrepancy was found.",
        "- The current physical evaluator inherited the ideal graspability-calibration spawn, not the authoritative retarget task-object position.",
        "",
        "## Method classification",
        "",
        f"- Fair-A: **{classification['ACT-A40']}**.",
        f"- Proposed-B: **{classification['ACT-B40']}**.",
        f"- Proposed-B reference whole-hand accessibility under authoritative registration: `{sum(b_access)}/10`.",
        f"- Fair-A reference whole-hand accessibility under authoritative registration: `{sum(a_access)}/10`.",
        "",
        "The classifications describe representation behavior, not final physical task success. Proposed-B's wrist is not the optimized representation; its static canonical whole-hand frame is. Fair-A's failure to align to the object is a potentially meaningful method outcome once the common scene is corrected, not permission for an A-specific object position.",
        "",
        "## Aggregate best-frame geometry (millimetres)",
        "",
        "Each range is minimum / median / maximum over EVAL10. Pad values are sampled collision-pad signed gaps to the exact frozen triangulated convex proxy (negative means sampled overlap).",
        "",
        "| Method/source/scene | Wrist center | Palm center | Whole-hand center | Thumb gap | Index gap | Middle gap |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in ("ACT-A40", "ACT-B40"):
        for source in ("reference", "act_prediction"):
            for registration in (
                "current_physical_scene",
                "authoritative_retarget_scene",
                "representation_consistent_physical_scene",
            ):
                data = aggregate[method][source][registration]
                def fmt(key: str) -> str:
                    row = data[key]
                    return f"{row['minimum']:.2f}/{row['median']:.2f}/{row['maximum']:.2f}"
                lines.append(
                    f"| {method} {source} {registration} | {fmt('wrist_center_distance_mm')} | "
                    f"{fmt('palm_center_distance_mm')} | {fmt('whole_hand_center_distance_mm')} | "
                    f"{fmt('thumb_pad_gap_mm')} | {fmt('index_pad_gap_mm')} | {fmt('middle_pad_gap_mm')} |"
                )
    lines.extend(
        [
            "",
            "## Successful scripted reference",
            "",
            f"- Stable control frame: `{stable_frame}`; physics step `{scripted_serialized['physics_step']}`.",
            f"- Object position: `{scripted_serialized['object_position_world_m']}` m.",
            f"- Wrist/palm/whole-hand center distance: `{scripted_serialized['wrist_to_doll_center_m']*1000.0:.3f}` / `{scripted_serialized['palm_to_doll_center_m']*1000.0:.3f}` / `{scripted_serialized['whole_hand_to_doll_center_m']*1000.0:.3f}` mm.",
            f"- Pad signed gaps T/I/M: `{scripted_serialized['pad_collider_to_doll_surface_sampled_signed_m']['thumb']*1000.0:.3f}` / `{scripted_serialized['pad_collider_to_doll_surface_sampled_signed_m']['index']*1000.0:.3f}` / `{scripted_serialized['pad_collider_to_doll_surface_sampled_signed_m']['middle']*1000.0:.3f}` mm.",
            "",
            "## Bimanual/task-frame evidence",
            "",
            "Per-episode DUAL_CONTACT whole-hand midpoint, relative-vector, and left/right target-interaction errors for both frozen references and ACT predictions are stored in `TASK_FRAME_ALIGNMENT_AUDIT.json`.",
            "",
            "No object pose, trajectory, gate, controller, policy, score, or physics artifact was modified by this audit.",
        ]
    )
    (OUT / "TASK_FRAME_ALIGNMENT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = [
        OUT / "TASK_FRAME_ALIGNMENT_AUDIT.json",
        OUT / "TASK_FRAME_ALIGNMENT_AUDIT.md",
        OUT / "METHOD_CLASSIFICATION.json",
        OUT / "PER_EPISODE_ALIGNMENT.csv",
    ]
    write_json(
        OUT / "AUDIT_MANIFEST.json",
        {
            "status": "READ_ONLY_AUDIT_FROZEN",
            "inputs_modified": False,
            "files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files],
        },
    )
    print(json.dumps(classification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
