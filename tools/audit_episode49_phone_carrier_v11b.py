#!/usr/bin/env python3
"""Audit Episode-49 phone-carrier hypotheses without changing ALOHA motion.

The verified v11 action representation, named-joint FK, source root/TCP chain,
approved timeline, and +7-frame command-to-observation convention are immutable
inputs.  This script only reconstructs source phone/accessory object state.

No G1 target, IK, phasewarp, orientation retargeting, Dex3, physics, DDS,
publisher, or hardware path is present.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import build_source_fk_parity_v11 as base  # noqa: E402
import continue_source_fk_parity_v11_after_latency_approval as v11  # noqa: E402


V11 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_phone_carrier_audit_v11b"
ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
SOURCE_FRAMES = ROOT / "configs/episode49_source_object_frames.user_approved.json"
SOURCE_LAYOUT = ROOT / "configs/episode49_source_scene_layout.json"
MODEL_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")

ALIGNMENT = V11 / "source_action_latency_alignment.npz"
OPT_FK = V11 / "optimized_action_fk.npz"
STATE_FK = V11 / "observation_state_fk.npz"
LATENCY_APPROVAL = V11 / "action_to_observation_latency.approved.json"
MAPPING_AUDIT = V11 / "aloha_joint_mapping_audit.json"
TRANSFORM_AUDIT = V11 / "source_transform_chain_unit_tests.json"
V11_RELATIONS = V11 / "source_hand_object_relations_recomputed.json"

FPS = 30.0
LAG = 7
FRAME_COUNT = 990
CANDIDATE_IDS = ["CONTACT_START_176", "CHARGER_ANCHORED_530", "OBSERVED_GEOMETRY_530"]
EVENT_ACTION = {
    176: 169,
    223: 216,
    326: 319,
    341: 334,
    530: 523,
    586: 579,
}
ACCESSORY_EVENT_FRAMES = [326, 329, 341]
ACCESSORY_SWEEP_FRAMES = np.arange(300, 351, dtype=np.int64)
ACQUISITION_FRAMES = np.arange(176, 224, dtype=np.int64)
TABLE_SURFACE_Z = 0.795
PHONE_HALF_SIZE = np.array([0.1496, 0.00795, 0.0715], dtype=np.float64) / 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def inverse(value: np.ndarray) -> np.ndarray:
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    return float(np.arctan2(0.5 * np.linalg.norm(skew), cosine))


def angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))


def desired_phone_on_pad(source_frames: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    pad = np.asarray(source_frames["T_source_scene_from_charger_pad"], dtype=np.float64)
    outward = np.asarray(source_frames["charger_pad_outward_normal"], dtype=np.float64)
    outward /= np.linalg.norm(outward)
    long_axis = pad[:3, 1].copy()
    long_axis -= outward * float(np.dot(long_axis, outward))
    long_axis /= np.linalg.norm(long_axis)
    if long_axis[2] < 0.0:
        long_axis *= -1.0
    back_axis = -outward
    short_axis = np.cross(long_axis, back_axis)
    short_axis /= np.linalg.norm(short_axis)
    long_axis = np.cross(back_axis, short_axis)
    long_axis /= np.linalg.norm(long_axis)
    rotation = np.column_stack((long_axis, back_axis, short_axis))
    pose = transform(rotation, np.asarray(source_frames["charger_pad_face_center"], dtype=np.float64))
    audit = {
        "definition": {
            "phone_center": "authoritative source charger pad face center",
            "phone_local_plus_x_long_axis": "source charger pad tangent-plane vertical axis, positive world-Z direction",
            "phone_local_plus_y_back_normal": "negative source charger pad outward normal",
            "phone_local_plus_z_short_axis": "cross(long_axis, back_axis), then orthogonalized right-handed basis",
        },
        "T_source_scene_from_desired_phone_at_530": pose,
        "determinant": float(np.linalg.det(rotation)),
        "orthonormal_max_abs_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        "center_error_to_pad_face_m": float(np.linalg.norm(pose[:3, 3] - pad[:3, 3])),
        "back_normal_error_deg": angle_between_deg(rotation[:, 1], -outward),
        "long_axis_tangent_error_abs_dot": float(abs(np.dot(rotation[:, 0], outward))),
        "long_axis_world_z_component": float(rotation[2, 0]),
        "right_handed": bool(np.linalg.det(rotation) > 0.0),
    }
    passed = bool(
        abs(audit["determinant"] - 1.0) <= 1e-12
        and audit["orthonormal_max_abs_error"] <= 1e-12
        and audit["center_error_to_pad_face_m"] <= 1e-12
        and audit["back_normal_error_deg"] <= 1e-6
        and audit["long_axis_tangent_error_abs_dot"] <= 1e-12
    )
    audit["status"] = "PASS_PHONE_ON_CHARGER_POSE_DEFINITION" if passed else "BLOCKED_PHONE_ON_CHARGER_POSE_DEFINITION"
    return pose, audit


def phone_corners(pose: np.ndarray) -> np.ndarray:
    local = np.array(
        [[sx, sy, sz] for sx in (-PHONE_HALF_SIZE[0], PHONE_HALF_SIZE[0])
         for sy in (-PHONE_HALF_SIZE[1], PHONE_HALF_SIZE[1])
         for sz in (-PHONE_HALF_SIZE[2], PHONE_HALF_SIZE[2])],
        dtype=np.float64,
    )
    return (pose[:3, :3] @ local.T).T + pose[:3, 3]


def phone_metrics(pose: np.ndarray, desired: np.ndarray, initial: np.ndarray) -> dict[str, float]:
    return {
        "center_to_pad_face_m": float(np.linalg.norm(pose[:3, 3] - desired[:3, 3])),
        "back_normal_to_desired_deg": angle_between_deg(pose[:3, 1], desired[:3, 1]),
        "full_rotation_to_desired_deg": float(np.degrees(rotation_angle_rad(desired[:3, :3].T @ pose[:3, :3]))),
        "long_axis_to_nearest_world_vertical_deg": float(
            np.degrees(np.arccos(np.clip(abs(float(pose[:3, 0] @ np.array([0.0, 0.0, 1.0]))), 0.0, 1.0)))
        ),
        "center_to_initial_m": float(np.linalg.norm(pose[:3, 3] - initial[:3, 3])),
        "full_rotation_to_initial_deg": float(np.degrees(rotation_angle_rad(initial[:3, :3].T @ pose[:3, :3]))),
        "minimum_table_clearance_m": float(np.min(phone_corners(pose)[:, 2]) - TABLE_SURFACE_Z),
    }


class SurfaceQuery:
    def __init__(self, model: mujoco.MjModel, source_root: np.ndarray):
        self.model = model
        self.data = mujoco.MjData(model)
        self.source_root = source_root
        self.source_root_inverse = inverse(source_root)
        self.ids: dict[str, list[int]] = {}
        for side in ("left", "right"):
            prefix = f"follower_{side}_gripper_"
            ids = []
            for geom_id in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if name.startswith(prefix) and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                    ids.append(geom_id)
            if not ids:
                raise RuntimeError(f"No named {side} gripper box geometry")
            self.ids[side] = ids

    def nearest(self, qpos: np.ndarray, object_pose_source: np.ndarray, side: str, kind: str) -> dict[str, Any]:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        object_model = self.source_root_inverse @ object_pose_source
        local_samples = v11.object_surface_samples(kind)
        samples_model = (object_model[:3, :3] @ local_samples.T).T + object_model[:3, 3]
        best: dict[str, Any] | None = None
        for geom_id in self.ids[side]:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
            rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            half_size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
            local = (rotation.T @ (samples_model - center).T).T
            closest_local = np.minimum(np.maximum(local, -half_size), half_size)
            distances = np.linalg.norm(local - closest_local, axis=1)
            sample_index = int(np.argmin(distances))
            object_point_model = samples_model[sample_index]
            gripper_point_model = center + rotation @ closest_local[sample_index]
            object_point_source = (self.source_root @ np.r_[object_point_model, 1.0])[:3]
            gripper_point_source = (self.source_root @ np.r_[gripper_point_model, 1.0])[:3]
            row = {
                "gap_m": float(distances[sample_index]),
                "geom_id": int(geom_id),
                "geom_name": name,
                "object_kind": kind,
                "object_surface_sample_count": int(len(local_samples)),
                "nearest_object_surface_point_source_scene_m": object_point_source,
                "nearest_gripper_surface_point_source_scene_m": gripper_point_source,
                "point_pair_distance_m": float(np.linalg.norm(object_point_source - gripper_point_source)),
            }
            if best is None or row["gap_m"] < best["gap_m"]:
                best = row
        if best is None:
            raise RuntimeError(f"No {side} gripper geometry result")
        return best


def candidate_record(
    candidate_id: str,
    carrier: np.ndarray,
    definition_frame: int,
    action_index: int | None,
    definition_tcp: np.ndarray,
    definition_phone: np.ndarray,
    definition_source: str,
) -> dict[str, Any]:
    tcp_from_phone = inverse(carrier)
    reconstructed_phone = definition_tcp @ tcp_from_phone
    reconstructed_tcp = definition_phone @ carrier
    return {
        "candidate_id": candidate_id,
        "definition": definition_source,
        "observed_frame": definition_frame,
        "aligned_action_index": action_index,
        "T_phone_from_left_TCP": carrier,
        "T_left_TCP_from_phone": tcp_from_phone,
        "translation_norm_m": float(np.linalg.norm(carrier[:3, 3])),
        "rotation_angle_deg": float(np.degrees(rotation_angle_rad(carrier[:3, :3]))),
        "unit_test": {
            "transform_convention": "column vectors; T_A_from_C = T_A_from_B @ T_B_from_C",
            "phone_reconstruction_position_error_m": float(
                np.linalg.norm(reconstructed_phone[:3, 3] - definition_phone[:3, 3])
            ),
            "phone_reconstruction_rotation_error_rad": rotation_angle_rad(
                definition_phone[:3, :3].T @ reconstructed_phone[:3, :3]
            ),
            "tcp_reconstruction_position_error_m": float(
                np.linalg.norm(reconstructed_tcp[:3, 3] - definition_tcp[:3, 3])
            ),
            "tcp_reconstruction_rotation_error_rad": rotation_angle_rad(
                definition_tcp[:3, :3].T @ reconstructed_tcp[:3, :3]
            ),
        },
    }


def pairwise_record(first_id: str, first: np.ndarray, second_id: str, second: np.ndarray) -> dict[str, Any]:
    relative = inverse(first) @ second
    return {
        "first": first_id,
        "second": second_id,
        "T_first_carrier_from_second_carrier": relative,
        "translation_difference_m": float(np.linalg.norm(relative[:3, 3])),
        "rotation_difference_deg": float(np.degrees(rotation_angle_rad(relative[:3, :3]))),
        "same_rigid_carrier_thresholds": {"translation_m": 0.010, "rotation_deg": 10.0},
        "same_rigid_carrier": bool(
            np.linalg.norm(relative[:3, 3]) <= 0.010
            and np.degrees(rotation_angle_rad(relative[:3, :3])) <= 10.0
        ),
    }


def main() -> int:
    required = [
        ACTION, TIMELINE, SOURCE_FRAMES, SOURCE_LAYOUT, MODEL_XML, ALIGNMENT, OPT_FK,
        STATE_FK, LATENCY_APPROVAL, MAPPING_AUDIT, TRANSFORM_AUDIT, V11_RELATIONS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)

    timeline_hash_before = sha256(TIMELINE)
    source_frames = load_json(SOURCE_FRAMES)
    layout = load_json(SOURCE_LAYOUT)
    latency = load_json(LATENCY_APPROVAL)
    mapping_audit = load_json(MAPPING_AUDIT)
    transform_audit = load_json(TRANSFORM_AUDIT)
    alignment = load_npz(ALIGNMENT)
    optimized_fk = load_npz(OPT_FK)
    state_fk = load_npz(STATE_FK)
    with np.load(ACTION, allow_pickle=False) as source:
        optimized_action = source["optimized_action"].copy()
        action_timestamp = source["timestamp"].copy()
        action_frame_index = source["frame_index"].copy()

    if latency["action_to_observation_lag_frames"] != LAG:
        raise RuntimeError("Approved +7-frame convention changed")
    if mapping_audit["status"] != "PASS" or transform_audit["status"] != "PASS":
        raise RuntimeError("A frozen v11 FK prerequisite is no longer PASS")
    if not np.array_equal(optimized_action, alignment["optimized_action"]):
        raise RuntimeError("optimized_action changed")
    lookup = alignment["action_sample_index_for_observed_frame"].astype(np.int64)
    expected_lookup = np.maximum(np.arange(FRAME_COUNT, dtype=np.int64) - LAG, 0)
    if not np.array_equal(lookup, expected_lookup):
        raise RuntimeError("Approved observed-frame lookup changed")
    if not np.array_equal(alignment["post_command_terminal_sample_indices"], np.arange(983, 990)):
        raise RuntimeError("Terminal action samples 983-989 were not retained")
    for observed, action_index in EVENT_ACTION.items():
        if int(lookup[observed]) != action_index:
            raise RuntimeError(f"Event/action index mismatch at {observed}")

    optimized_tcp = np.stack(
        (alignment["observation_aligned_left_tcp_transform"], alignment["observation_aligned_right_tcp_transform"]),
        axis=1,
    )
    state_tcp = np.stack((state_fk["left_tcp_transform"], state_fk["right_tcp_transform"]), axis=1)
    optimized_qpos = optimized_fk["qpos"]
    state_qpos = state_fk["qpos"]
    source_root = optimized_fk["source_aloha_root_transform"]

    initial_phone = np.asarray(source_frames["T_source_scene_from_phone"], dtype=np.float64)
    initial_accessory = np.asarray(source_frames["T_source_scene_from_accessory"], dtype=np.float64)
    pad = np.asarray(source_frames["T_source_scene_from_charger_pad"], dtype=np.float64)
    phone_from_accessory = inverse(initial_phone) @ initial_accessory
    declared_attachment = np.array([0.0, 0.006425, 0.0], dtype=np.float64)
    if not np.allclose(phone_from_accessory[:3, 3], declared_attachment, rtol=0.0, atol=1e-12):
        raise RuntimeError("Verified phone/accessory attachment translation changed")

    desired_phone, desired_audit = desired_phone_on_pad(source_frames)
    if desired_audit["status"] != "PASS_PHONE_ON_CHARGER_POSE_DEFINITION":
        dump(OUT / "phone_pad_metrics.json", desired_audit)
        raise RuntimeError("BLOCKED_PHONE_ON_CHARGER_POSE_DEFINITION")

    carrier_a = inverse(initial_phone) @ optimized_tcp[176, 0]
    carrier_b = inverse(desired_phone) @ optimized_tcp[530, 0]
    carrier_c = inverse(desired_phone) @ state_tcp[530, 0]
    carriers = np.stack((carrier_a, carrier_b, carrier_c), axis=0)
    carrier_inverses = np.stack([inverse(value) for value in carriers], axis=0)
    phone_rigid = np.einsum("tij,sjk->stik", optimized_tcp[:, 0], carrier_inverses)
    if not np.allclose(phone_rigid[1, 530], desired_phone, rtol=0.0, atol=1e-12):
        raise RuntimeError("CHARGER_ANCHORED_530 multiplication order failed")

    candidate_rows = {
        "CONTACT_START_176": candidate_record(
            "CONTACT_START_176", carrier_a, 176, 169, optimized_tcp[176, 0], initial_phone,
            "T_phone_from_left_TCP from known initial phone pose and optimized_action[169] FK",
        ),
        "CHARGER_ANCHORED_530": candidate_record(
            "CHARGER_ANCHORED_530", carrier_b, 530, 523, optimized_tcp[530, 0], desired_phone,
            "T_phone_from_left_TCP from authoritative source phone-on-pad pose and optimized_action[523] FK",
        ),
        "OBSERVED_GEOMETRY_530": candidate_record(
            "OBSERVED_GEOMETRY_530", carrier_c, 530, None, state_tcp[530, 0], desired_phone,
            "geometric calibration only from authoritative source phone-on-pad pose and observation.state[530] FK",
        ),
    }
    candidate_rows["OBSERVED_GEOMETRY_530"]["observation_state_used_as_motion_source"] = False
    candidate_rows["OBSERVED_GEOMETRY_530"]["reconstruction_motion_source"] = "aligned optimized_action TCP only"

    comparisons = [
        pairwise_record(CANDIDATE_IDS[first], carriers[first], CANDIDATE_IDS[second], carriers[second])
        for first, second in ((0, 1), (0, 2), (1, 2))
    ]
    a_vs_b = comparisons[0]
    contact_start_is_lock = bool(a_vs_b["same_rigid_carrier"])

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    surface = SurfaceQuery(model, source_root)

    optimized_530_geometry = surface.nearest(
        optimized_qpos[523], desired_phone, "left", "phone"
    )
    observed_530_geometry = surface.nearest(
        state_qpos[530], desired_phone, "left", "phone"
    )
    observed_geometry_comparison = {
        "optimized_action_definition": {
            "observed_frame": 530,
            "aligned_action_index": 523,
            "nearest_phone_to_left_gripper": optimized_530_geometry,
        },
        "observation_state_definition": {
            "observed_frame": 530,
            "aligned_action_index": None,
            "nearest_phone_to_left_gripper": observed_530_geometry,
        },
        "carrier_translation_difference_m": comparisons[2]["translation_difference_m"],
        "carrier_rotation_difference_deg": comparisons[2]["rotation_difference_deg"],
        "nearest_gripper_geometry_gap_difference_m": float(
            abs(optimized_530_geometry["gap_m"] - observed_530_geometry["gap_m"])
        ),
        "observation_state_used_as_motion_source": False,
    }

    # Stable-carrier acquisition audit against the known initial phone geometry.
    stable_rows: list[dict[str, Any]] = []
    stable_flags: list[bool] = []
    for observed in ACQUISITION_FRAMES:
        action_index = int(lookup[observed])
        pose = phone_rigid[1, observed]
        initial_query = surface.nearest(optimized_qpos[action_index], initial_phone, "left", "phone")
        carried_query = surface.nearest(optimized_qpos[action_index], pose, "left", "phone")
        metrics = phone_metrics(pose, desired_phone, initial_phone)
        center_ok = metrics["center_to_initial_m"] <= 0.010
        rotation_ok = metrics["full_rotation_to_initial_deg"] <= 10.0
        table_ok = metrics["minimum_table_clearance_m"] >= -0.002
        contact_ok = initial_query["gap_m"] <= 0.010
        stable = bool(center_ok and rotation_ok and table_ok and contact_ok)
        stable_flags.append(stable)
        stable_rows.append(
            {
                "observed_frame": int(observed),
                "aligned_action_index": action_index,
                "aligned_optimized_left_gripper_command_m": float(
                    alignment["observation_aligned_optimized_action"][observed, 6]
                ),
                "observation_left_gripper_state_m": float(state_fk["left_gripper_source"][observed]),
                "left_gripper_to_initial_phone_surface_gap_m": initial_query["gap_m"],
                "left_gripper_to_charger_carrier_phone_surface_gap_m": carried_query["gap_m"],
                "charger_carrier_phone_center_source_scene_m": pose[:3, 3],
                "charger_carrier_phone_center_to_initial_m": metrics["center_to_initial_m"],
                "charger_carrier_phone_rotation_to_initial_deg": metrics["full_rotation_to_initial_deg"],
                "charger_carrier_phone_minimum_table_clearance_m": metrics["minimum_table_clearance_m"],
                "charger_carrier_phone_long_axis_to_vertical_deg": metrics[
                    "long_axis_to_nearest_world_vertical_deg"
                ],
                "charger_carrier_phone_frame_displacement_m": (
                    0.0 if observed == 176 else float(
                        np.linalg.norm(phone_rigid[1, observed, :3, 3] - phone_rigid[1, observed - 1, :3, 3])
                    )
                ),
                "criteria": {
                    "center_within_10mm_of_known_initial_phone": center_ok,
                    "rotation_within_10deg_of_known_initial_phone": rotation_ok,
                    "table_penetration_not_below_2mm": table_ok,
                    "left_gripper_within_10mm_of_initial_phone_surface": contact_ok,
                },
                "stable_carrier_onset_candidate": stable,
            }
        )
    stable_indices = [int(frame) for frame, flag in zip(ACQUISITION_FRAMES, stable_flags) if flag]
    stable_frame = stable_indices[0] if stable_indices else None
    normalized_scores = np.array(
        [
            row["charger_carrier_phone_center_to_initial_m"] / 0.010
            + row["charger_carrier_phone_rotation_to_initial_deg"] / 10.0
            + max(0.0, -row["charger_carrier_phone_minimum_table_clearance_m"] - 0.002) / 0.002
            for row in stable_rows
        ],
        dtype=np.float64,
    )
    closest_index = int(np.argmin(normalized_scores))
    stable_interval = {
        "status": (
            "STABLE_SINGLE_RIGID_CARRIER_ONSET_FOUND"
            if stable_frame is not None
            else "NO_PHYSICALLY_CONSISTENT_SINGLE_RIGID_CARRIER_ONSET_IN_FRAMES_176_223"
        ),
        "search_observed_frames_inclusive": [176, 223],
        "action_index_convention": "max(observed_frame - 7, 0)",
        "carrier_candidate": "CHARGER_ANCHORED_530",
        "thresholds": {
            "center_to_known_initial_phone_m": 0.010,
            "rotation_to_known_initial_phone_deg": 10.0,
            "minimum_table_clearance_m": -0.002,
            "gripper_to_initial_phone_surface_m": 0.010,
        },
        "carrier_model_valid_from_observed_frame": stable_frame,
        "carrier_model_valid_from_aligned_action_index": None if stable_frame is None else int(lookup[stable_frame]),
        "closest_consistency_frame": stable_rows[closest_index],
        "closest_consistency_normalized_score": float(normalized_scores[closest_index]),
        "rows": stable_rows,
        "approved_timeline_modified": False,
    }

    # Candidate-specific phone-pad and portrait metrics.
    phone_pad_rows: dict[str, Any] = {}
    portrait_rows: dict[str, Any] = {}
    for candidate_index, candidate_id in enumerate(CANDIDATE_IDS):
        phone_pad_rows[candidate_id] = {
            "observed_frame": 530,
            "aligned_action_index": 523,
            **phone_metrics(phone_rigid[candidate_index, 530], desired_phone, initial_phone),
        }
        portrait_rows[candidate_id] = {
            "observed_frame": 223,
            "aligned_action_index": 216,
            **phone_metrics(phone_rigid[candidate_index, 223], desired_phone, initial_phone),
        }
    phone_pad_payload = {
        "status": desired_audit["status"],
        "desired_phone_pose_audit": desired_audit,
        "candidate_frame_530_metrics": phone_pad_rows,
        "candidate_frame_223_portrait_metrics": portrait_rows,
        "gates": {"phone_center_to_pad_m": 0.005, "phone_back_normal_error_deg": 5.0},
    }

    # Accessory remains attached to the reconstructed phone for this independent
    # contact test.  No right-hand carrier is created from a failed contact.
    accessory_attached = np.einsum("stij,jk->stik", phone_rigid, phone_from_accessory)
    gap = np.empty((3, len(ACCESSORY_SWEEP_FRAMES)), dtype=np.float64)
    nearest_object = np.empty((3, len(ACCESSORY_SWEEP_FRAMES), 3), dtype=np.float64)
    nearest_gripper = np.empty_like(nearest_object)
    geom_names: list[list[str]] = [["" for _ in ACCESSORY_SWEEP_FRAMES] for _ in CANDIDATE_IDS]
    event_rows: dict[str, Any] = {}
    for candidate_index, candidate_id in enumerate(CANDIDATE_IDS):
        frame_rows: dict[str, Any] = {}
        for local_index, observed in enumerate(ACCESSORY_SWEEP_FRAMES):
            action_index = int(lookup[observed])
            result = surface.nearest(
                optimized_qpos[action_index], accessory_attached[candidate_index, observed], "right", "accessory"
            )
            gap[candidate_index, local_index] = result["gap_m"]
            nearest_object[candidate_index, local_index] = result["nearest_object_surface_point_source_scene_m"]
            nearest_gripper[candidate_index, local_index] = result["nearest_gripper_surface_point_source_scene_m"]
            geom_names[candidate_index][local_index] = str(result["geom_name"])
            if int(observed) in ACCESSORY_EVENT_FRAMES:
                frame_rows[str(int(observed))] = {
                    "observed_frame": int(observed),
                    "aligned_action_index": action_index,
                    "hypothesis": "accessory remains at verified phone-back attachment through this event boundary",
                    "accessory_center_source_scene_m": accessory_attached[candidate_index, observed, :3, 3],
                    "right_TCP_position_source_scene_m": optimized_tcp[observed, 1, :3, 3],
                    "TCP_to_accessory_center_m": float(
                        np.linalg.norm(
                            optimized_tcp[observed, 1, :3, 3]
                            - accessory_attached[candidate_index, observed, :3, 3]
                        )
                    ),
                    "right_gripper_to_ring_surface_gap_m": result["gap_m"],
                    "nearest_right_gripper_geometry": result,
                }
        min_index = int(np.argmin(gap[candidate_index]))
        acquisition_mask = (ACCESSORY_SWEEP_FRAMES >= 326) & (ACCESSORY_SWEEP_FRAMES <= 329)
        acquisition_local = np.flatnonzero(acquisition_mask)
        acquisition_best_local = int(acquisition_local[np.argmin(gap[candidate_index, acquisition_mask])])
        event_rows[candidate_id] = {
            "events": frame_rows,
            "minimum_gap_frames_300_350": {
                "observed_frame": int(ACCESSORY_SWEEP_FRAMES[min_index]),
                "aligned_action_index": int(lookup[ACCESSORY_SWEEP_FRAMES[min_index]]),
                "gap_m": float(gap[candidate_index, min_index]),
                "nearest_object_surface_point_source_scene_m": nearest_object[candidate_index, min_index],
                "nearest_gripper_surface_point_source_scene_m": nearest_gripper[candidate_index, min_index],
                "geom_name": geom_names[candidate_index][min_index],
            },
            "minimum_gap_approved_grasp_acquisition_frames_326_329": {
                "observed_frame": int(ACCESSORY_SWEEP_FRAMES[acquisition_best_local]),
                "aligned_action_index": int(lookup[ACCESSORY_SWEEP_FRAMES[acquisition_best_local]]),
                "gap_m": float(gap[candidate_index, acquisition_best_local]),
            },
            "right_carrier_after_grasp_generated": False,
            "reason": "a right-hand carrier would make the post-grasp gap circular evidence; acquisition must pass first",
        }

    accessory_payload = {
        "status": "BLOCKED_ACCESSORY_ATTACHMENT_OR_RIGHT_GRASP_SEMANTICS",
        "phone_from_accessory_attachment": phone_from_accessory,
        "verified_attachment_translation_m": declared_attachment,
        "arbitrary_accessory_offset_added": False,
        "sweep_observed_frames_inclusive": [300, 350],
        "approved_event_frames_modified": False,
        "candidate_metrics": event_rows,
        "pass_criteria": {
            "frame_326_or_approved_acquisition_interval_gap_m": 0.010,
            "frame_341_gap_m": 0.010,
        },
        "interpretation": (
            "CHARGER_ANCHORED_530 fixes the phone-on-pad geometry but does not place the current verified "
            "phone-attached ring within 10 mm of the right gripper at the approved accessory events."
        ),
    }

    # Visual state masks avoid fabricating object motion during unresolved grasp
    # acquisition.  Rigid reconstructions remain stored separately for audit.
    visual_phone = np.full_like(phone_rigid, np.nan)
    visual_phone_valid = np.zeros((3, FRAME_COUNT), dtype=bool)
    for candidate_index in range(3):
        visual_phone[candidate_index, :176] = initial_phone
        visual_phone_valid[candidate_index, :176] = True
    visual_phone[0, 176:531] = phone_rigid[0, 176:531]
    visual_phone[0, 531:] = phone_rigid[0, 530]
    visual_phone_valid[0] = True
    for candidate_index in (1, 2):
        # 176..222 is intentionally unresolved; no interpolation or snapping.
        visual_phone[candidate_index, 223:531] = phone_rigid[candidate_index, 223:531]
        visual_phone[candidate_index, 531:] = phone_rigid[candidate_index, 530]
        visual_phone_valid[candidate_index, 223:] = True

    visual_accessory = np.full_like(accessory_attached, np.nan)
    visual_accessory_valid = np.zeros((3, FRAME_COUNT), dtype=bool)
    for candidate_index in range(3):
        valid_until_removal = visual_phone_valid[candidate_index] & (np.arange(FRAME_COUNT) <= 341)
        visual_accessory[candidate_index, valid_until_removal] = accessory_attached[candidate_index, valid_until_removal]
        visual_accessory_valid[candidate_index] = valid_until_removal

    piecewise_phone = np.full((FRAME_COUNT, 4, 4), np.nan, dtype=np.float64)
    piecewise_valid = np.zeros(FRAME_COUNT, dtype=bool)
    piecewise_phone[:176] = initial_phone
    piecewise_valid[:176] = True
    piecewise_phone[223:531] = phone_rigid[1, 223:531]
    piecewise_phone[531:] = desired_phone
    piecewise_valid[223:] = True

    charger_candidate = event_rows["CHARGER_ANCHORED_530"]
    charger_acquisition_gap = charger_candidate[
        "minimum_gap_approved_grasp_acquisition_frames_326_329"
    ]["gap_m"]
    charger_frame341_gap = charger_candidate["events"]["341"]["right_gripper_to_ring_surface_gap_m"]
    charger_pad_pass = bool(
        phone_pad_rows["CHARGER_ANCHORED_530"]["center_to_pad_face_m"] <= 0.005
        and phone_pad_rows["CHARGER_ANCHORED_530"]["back_normal_to_desired_deg"] <= 5.0
    )
    accessory_pass = bool(charger_acquisition_gap <= 0.010 and charger_frame341_gap <= 0.010)
    single_rigid = bool(contact_start_is_lock and stable_frame is not None and charger_pad_pass and accessory_pass)
    viability = {
        "status": (
            "SOURCE_PHONE_CARRIER_VALIDATED_FROM_CHARGER_ANCHOR"
            if single_rigid
            else "SOURCE_PHONE_RELATION_NOT_SINGLE_RIGID_TRANSFORM"
        ),
        "secondary_status": (
            "SOURCE_ACCESSORY_RELATION_VALIDATED"
            if accessory_pass
            else "BLOCKED_ACCESSORY_ATTACHMENT_OR_RIGHT_GRASP_SEMANTICS"
        ),
        "interim_blocker_reclassification": "BLOCKED_PHONE_CARRIER_MODEL",
        "optimized_action_task_validity_finalized_as_failure": False,
        "single_rigid_phone_carrier_valid": single_rigid,
        "contact_start_176_is_valid_rigid_carrier_lock": contact_start_is_lock,
        "contact_start_vs_charger_carrier": a_vs_b,
        "stable_carrier_interval_found": stable_frame is not None,
        "charger_anchor_gates_pass": charger_pad_pass,
        "accessory_gates_pass": accessory_pass,
        "piecewise_diagnostic": {
            "generated": not single_rigid,
            "status": "PIECEWISE_OBJECT_CARRIER_DIAGNOSTIC_NOT_YET_APPROVED",
            "segment_1": "known initial phone before frame 176; object state unresolved during frames 176-222",
            "single_transition_observed_frame": 223,
            "segment_2": "CHARGER_ANCHORED_530 rigid carrier hypothesis from frame 223 through charger placement",
            "changes_optimized_action_or_tcp": False,
            "approved_for_g1": False,
        },
        "possible_physical_interpretations_not_selected": [
            "grasp acquisition", "phone slip", "in-hand rotation", "regrasp", "inaccurate event semantics"
        ],
        "no_time_varying_hand_object_transform_adopted": True,
        "next_audit_scope": (
            "accessory local axes, ring center/gap orientation, right contact proxy, and frame-326/341 semantics"
            if charger_pad_pass and not accessory_pass
            else None
        ),
    }

    input_audit = {
        "status": "PASS_FROZEN_V11_INPUTS_REUSED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "SOURCE_PHONE_AND_ACCESSORY_OBJECT_STATE_RECONSTRUCTION_ONLY",
        "frozen_verified_facts_not_reaudited": [
            "optimized_action representation", "ALOHA joint order", "source root transform",
            "TCP offset", "+7-frame action-to-observation alignment", "approved timeline",
        ],
        "optimized_action": {
            "path": str(ACTION.resolve()),
            "sha256": sha256(ACTION),
            "shape": list(optimized_action.shape),
            "finite": bool(np.isfinite(optimized_action).all()),
            "equal_to_v11_alignment_artifact": bool(np.array_equal(optimized_action, alignment["optimized_action"])),
            "modified": False,
        },
        "approved_timeline": {
            "path": str(TIMELINE.resolve()),
            "sha256_before": timeline_hash_before,
            "modified": False,
        },
        "source_geometry": {
            "frames": str(SOURCE_FRAMES.resolve()),
            "frames_sha256": sha256(SOURCE_FRAMES),
            "layout": str(SOURCE_LAYOUT.resolve()),
            "layout_sha256": sha256(SOURCE_LAYOUT),
            "phone_size_xyz_m": layout["phone"]["size_landscape_xyz"],
            "accessory_attachment_translation_m": declared_attachment,
        },
        "v11_provenance": {
            str(path.resolve()): sha256(path)
            for path in (ALIGNMENT, OPT_FK, STATE_FK, LATENCY_APPROVAL, MAPPING_AUDIT, TRANSFORM_AUDIT, V11_RELATIONS)
        },
        "forbidden_stages": {
            "g1_target": False, "g1_ik": False, "phasewarp": False, "orientation_target": False,
            "dex3": False, "physics": False, "dds": False, "publisher": False, "hardware": False,
        },
    }
    alignment_audit = {
        "status": "PASS_USER_APPROVED_PLUS_7_FRAME_ALIGNMENT_REUSED_UNCHANGED",
        "action_to_observation_lag_frames": 7,
        "action_sample_for_observed_frame": "observed_frame - 7",
        "fps": 30.0,
        "latency_seconds": 7 / 30,
        "event_mapping": [
            {"observed_frame": observed, "aligned_action_index": action_index}
            for observed, action_index in EVENT_ACTION.items()
        ],
        "diagnostic_precommand_hold_observed_frames": list(range(7)),
        "negative_indexing": False,
        "wrap_or_extrapolation": False,
        "post_command_terminal_action_indices_retained": list(range(983, 990)),
        "approved_event_frames_modified": False,
        "video_or_observation_timestamps_modified": False,
        "optimized_action_values_or_order_modified": False,
        "action_frame_index_equal": bool(np.array_equal(action_frame_index, np.arange(990))),
        "action_timestamp_equal_to_v11": bool(np.array_equal(action_timestamp, alignment["optimized_action_original_timestamp"])),
    }

    carrier_payload = {
        "status": "BLOCKED_PHONE_CARRIER_MODEL" if not single_rigid else viability["status"],
        "candidate_order": CANDIDATE_IDS,
        "candidates": candidate_rows,
        "desired_phone_on_charger_pose": desired_audit,
        "observation_geometry_calibration_comparison": observed_geometry_comparison,
        "frame_176_frozen_as_full_carry_lock": False,
        "optimized_action_modified": False,
        "observation_state_used_as_motion_source": False,
    }
    comparison_payload = {
        "status": (
            "CONTACT_START_176_IS_VALID_RIGID_CARRIER_LOCK"
            if contact_start_is_lock
            else "CONTACT_START_176_IS_NOT_A_VALID_RIGID_CARRIER_LOCK"
        ),
        "recommended_difference_thresholds": {"translation_m": 0.010, "rotation_deg": 10.0},
        "pairwise": comparisons,
        "decision": (
            "contact-start relation is not a valid rigid carrier lock"
            if not contact_start_is_lock
            else "contact-start relation is consistent with charger-anchored carrier"
        ),
    }

    dump(OUT / "input_audit.json", input_audit)
    dump(OUT / "alignment_audit.json", alignment_audit)
    dump(OUT / "phone_carrier_candidates.json", carrier_payload)
    dump(OUT / "phone_carrier_transform_comparison.json", comparison_payload)
    dump(OUT / "stable_carrier_interval.json", stable_interval)
    dump(OUT / "accessory_surface_gap_metrics.json", accessory_payload)
    dump(OUT / "phone_pad_metrics.json", phone_pad_payload)
    dump(OUT / "single_rigid_carrier_viability.json", viability)

    np.savez_compressed(
        OUT / "reconstructed_phone_trajectories.npz",
        candidate_ids=np.asarray(CANDIDATE_IDS),
        observed_frame_index=np.arange(FRAME_COUNT, dtype=np.int64),
        observed_timestamp=alignment["observed_timestamp"],
        aligned_action_index=lookup,
        T_phone_from_left_TCP=carriers,
        T_left_TCP_from_phone=carrier_inverses,
        T_source_scene_from_initial_phone=initial_phone,
        T_source_scene_from_desired_phone_on_charger=desired_phone,
        T_source_scene_from_phone_rigid_reconstruction=phone_rigid,
        T_source_scene_from_phone_visual_diagnostic=visual_phone,
        phone_visual_valid_mask=visual_phone_valid,
        piecewise_diagnostic_T_source_scene_from_phone=piecewise_phone,
        piecewise_diagnostic_valid_mask=piecewise_valid,
        piecewise_transition_observed_frame=np.array(223, dtype=np.int64),
        carrier_model_valid_from_observed_frame=np.array(-1 if stable_frame is None else stable_frame, dtype=np.int64),
        object_state_unresolved_during_grasp_acquisition=np.array(True),
        optimized_action_modified=np.array(False),
        approved_timeline_modified=np.array(False),
        physics=np.array(False),
    )
    np.savez_compressed(
        OUT / "reconstructed_accessory_trajectories.npz",
        candidate_ids=np.asarray(CANDIDATE_IDS),
        observed_frame_index=np.arange(FRAME_COUNT, dtype=np.int64),
        observed_timestamp=alignment["observed_timestamp"],
        aligned_action_index=lookup,
        T_phone_from_accessory_attachment=phone_from_accessory,
        T_source_scene_from_accessory_phone_attached_hypothesis=accessory_attached,
        T_source_scene_from_accessory_visual_diagnostic=visual_accessory,
        accessory_visual_valid_mask=visual_accessory_valid,
        sweep_observed_frames=ACCESSORY_SWEEP_FRAMES,
        right_gripper_to_ring_surface_gap_m=gap,
        nearest_ring_surface_point_source_scene_m=nearest_object,
        nearest_right_gripper_surface_point_source_scene_m=nearest_gripper,
        arbitrary_accessory_offset_added=np.array(False),
        right_hand_carrier_generated=np.array(False),
        physics=np.array(False),
    )

    timeline_hash_after = sha256(TIMELINE)
    if timeline_hash_after != timeline_hash_before:
        raise RuntimeError("Approved timeline changed during carrier audit")
    input_audit["approved_timeline"]["sha256_after"] = timeline_hash_after
    input_audit["approved_timeline"]["byte_identical_before_after"] = True
    dump(OUT / "input_audit.json", input_audit)

    summary = {
        "primary_status": viability["status"],
        "secondary_status": viability["secondary_status"],
        "contact_start_176_is_valid_rigid_carrier_lock": contact_start_is_lock,
        "contact_start_vs_charger_translation_difference_mm": 1000.0 * a_vs_b["translation_difference_m"],
        "contact_start_vs_charger_rotation_difference_deg": a_vs_b["rotation_difference_deg"],
        "optimized_vs_observed_530_translation_difference_mm": 1000.0 * comparisons[2]["translation_difference_m"],
        "optimized_vs_observed_530_rotation_difference_deg": comparisons[2]["rotation_difference_deg"],
        "stable_carrier_valid_from_observed_frame": stable_frame,
        "charger_candidate_frame_326_gap_mm": 1000.0 * charger_candidate["events"]["326"]["right_gripper_to_ring_surface_gap_m"],
        "charger_candidate_frame_341_attached_hypothesis_gap_mm": 1000.0 * charger_frame341_gap,
        "charger_candidate_frame_530_center_error_mm": 1000.0 * phone_pad_rows["CHARGER_ANCHORED_530"]["center_to_pad_face_m"],
        "charger_candidate_frame_530_normal_error_deg": phone_pad_rows["CHARGER_ANCHORED_530"]["back_normal_to_desired_deg"],
        "output": str(OUT.resolve()),
    }
    print(json.dumps(summary, indent=2, default=default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
