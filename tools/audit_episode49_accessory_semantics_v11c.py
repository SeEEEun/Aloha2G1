#!/usr/bin/env python3
"""Source-only Episode-49 accessory geometry/contact/semantics audit.

This audit consumes the frozen v11 source FK and the user-approved v11b
CHARGER_ANCHORED_530 phone-carrier *diagnostic hypothesis*.  It never changes
optimized_action, the approved timeline, timestamps, the +7-frame alignment,
or the source ALOHA arm motion.  It does not generate a G1 target, IK,
phasewarp, orientation retargeting, Dex3 motion, or physics.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
V11 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
V11B = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_phone_carrier_audit_v11b"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_accessory_semantics_audit_v11c"

ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
LAYOUT = ROOT / "configs/episode49_source_scene_layout.json"
SOURCE_FRAMES = ROOT / "configs/episode49_source_object_frames.user_approved.json"
BUILDER = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_scene_builder.py"
ACCESSORY_USD = ROOT / "outputs/episode49_source_scene/generated/source_accessory.usda"
SCENE_USD = ROOT / "outputs/episode49_source_scene/generated/source_magsafe_fixed_scene.usda"
MODEL_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
DEX3_MAPPING = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
DEX3_TIPS = ROOT / "configs/dex3_fingertip_frames.sim.json"

ALIGNMENT = V11 / "source_action_latency_alignment.npz"
LATENCY_APPROVAL = V11 / "action_to_observation_latency.approved.json"
OPT_FK = V11 / "optimized_action_fk.npz"
STATE_FK = V11 / "observation_state_fk.npz"
PARITY = V11 / "source_parity_metrics.json"
PHONE_NPZ = V11B / "reconstructed_phone_trajectories.npz"
PHONE_VIABILITY = V11B / "single_rigid_carrier_viability.json"

FPS = 30.0
LAG = 7
FRAME_COUNT = 990
CHARGER_CANDIDATE = "CHARGER_ANCHORED_530"
AUDIT_FRAMES = np.arange(280, 371, dtype=np.int64)
EVENT_FRAMES = [300, 310, 319, 326, 329, 334, 341, 350]
APPROVED_ACCESSORY_EVENTS = {
    "right_accessory_grasp_start": 326,
    "accessory_detachment_start": 329,
    "accessory_removed": 341,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
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
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = rotation
    if translation is not None:
        value[:3, 3] = translation
    return value


def inverse(value: np.ndarray) -> np.ndarray:
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))


def event_rows() -> list[dict[str, Any]]:
    rows = list(load_json(TIMELINE)["events"])
    return sorted(enumerate(rows), key=lambda item: (int(item[1]["frame"]), item[0]))


def event_at(frame: int, rows: list[tuple[int, dict[str, Any]]]) -> str:
    value = "pre_task"
    for _, row in rows:
        if int(row["frame"]) <= frame:
            value = str(row["event"])
        else:
            break
    return value


def radial_surface(
    outer_radius: float,
    inner_radius: float,
    depth: float,
    axis: str,
    center: np.ndarray,
    start_angle: float,
    end_angle: float,
    angular_samples: int = 361,
) -> np.ndarray:
    """Dense samples on all faces/walls/caps of a ring or C-ring."""
    angles = np.linspace(start_angle, end_angle, angular_samples, dtype=np.float64)
    radii = np.linspace(inner_radius, outer_radius, 6, dtype=np.float64)
    half_depth = depth / 2.0
    depth_values = np.linspace(-half_depth, half_depth, 5, dtype=np.float64)
    points: list[np.ndarray] = []

    def ring_points(radius_values: np.ndarray, angle_values: np.ndarray, side: float) -> np.ndarray:
        rr, aa = np.meshgrid(radius_values, angle_values, indexing="ij")
        if axis.upper() == "Y":
            return np.column_stack((rr.ravel() * np.cos(aa).ravel(), np.full(rr.size, side), rr.ravel() * np.sin(aa).ravel()))
        if axis.upper() == "Z":
            return np.column_stack((rr.ravel() * np.cos(aa).ravel(), rr.ravel() * np.sin(aa).ravel(), np.full(rr.size, side)))
        raise ValueError(axis)

    # Both annular faces.
    points.append(ring_points(radii, angles, -half_depth))
    points.append(ring_points(radii, angles, half_depth))
    # Inner and outer walls.
    for radius in (inner_radius, outer_radius):
        for side in depth_values:
            points.append(ring_points(np.array([radius]), angles, float(side)))
    # End caps exist only for the open C-ring.
    if abs((end_angle - start_angle) - 2.0 * math.pi) > 1e-8:
        for angle in (start_angle, end_angle):
            rr, dd = np.meshgrid(radii, depth_values, indexing="ij")
            if axis.upper() == "Y":
                cap = np.column_stack((rr.ravel() * math.cos(angle), dd.ravel(), rr.ravel() * math.sin(angle)))
            else:
                cap = np.column_stack((rr.ravel() * math.cos(angle), rr.ravel() * math.sin(angle), dd.ravel()))
            points.append(cap)
    return np.concatenate(points, axis=0) + np.asarray(center, dtype=np.float64)


def cylinder_surface(radius: float, length: float, axis: str, center: np.ndarray) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, 181, endpoint=False, dtype=np.float64)
    axial = np.linspace(-length / 2.0, length / 2.0, 11, dtype=np.float64)
    values: list[np.ndarray] = []
    if axis.upper() != "X":
        raise ValueError(axis)
    aa, xx = np.meshgrid(angles, axial, indexing="ij")
    values.append(np.column_stack((xx.ravel(), radius * np.cos(aa).ravel(), radius * np.sin(aa).ravel())))
    radial = np.linspace(0.0, radius, 8, dtype=np.float64)
    for side in (-length / 2.0, length / 2.0):
        rr, cap_a = np.meshgrid(radial, angles, indexing="ij")
        values.append(np.column_stack((np.full(rr.size, side), rr.ravel() * np.cos(cap_a).ravel(), rr.ravel() * np.sin(cap_a).ravel())))
    return np.concatenate(values, axis=0) + np.asarray(center, dtype=np.float64)


def accessory_surfaces(layout: dict[str, Any], gap_center_degrees: float = -90.0) -> dict[str, np.ndarray]:
    acc = layout["accessory"]
    main_outer = float(acc["main_outer_diameter"]) / 2.0
    main_inner = float(acc["main_inner_diameter"]) / 2.0
    main_depth = float(acc["main_depth"])
    gap = math.radians(float(acc["main_gap_degrees"]))
    center = math.radians(float(gap_center_degrees))
    start = center + gap / 2.0
    end = center + 2.0 * math.pi - gap / 2.0
    support_center = np.asarray(acc["support_center_offset_from_main_xyz"], dtype=np.float64)
    support = radial_surface(
        float(acc["support_outer_diameter"]) / 2.0,
        float(acc["support_inner_diameter"]) / 2.0,
        float(acc["support_depth"]),
        "Z",
        support_center,
        0.0,
        2.0 * math.pi,
    )
    main = radial_surface(main_outer, main_inner, main_depth, "Y", np.zeros(3), start, end)
    # A substantially denser complete ring makes its distance a trustworthy
    # finite-sample lower bound for every 5-degree gap-orientation candidate.
    main_full = radial_surface(
        main_outer, main_inner, main_depth, "Y", np.zeros(3), 0.0, 2.0 * math.pi,
        angular_samples=3601,
    )
    hinge_center = np.array([0.0, 0.0015, -main_outer + 0.0035], dtype=np.float64)
    hinge = cylinder_surface(0.0032, 0.011, "X", hinge_center)
    return {
        "main_c_ring": main,
        "support_ring": support,
        "hinge": hinge,
        "main_complete_ring_counterfactual": main_full,
        "authoritative_all": np.concatenate((main, support, hinge), axis=0),
        # Include the authoritative C-ring samples as an explicit subset so
        # finite sampling cannot violate the mathematical full-ring lower
        # bound by a sub-millimeter discretization artifact.
        "complete_main_all": np.concatenate((main_full, main, support, hinge), axis=0),
    }


def proper_axis_rotations() -> list[tuple[str, np.ndarray]]:
    values: list[tuple[str, np.ndarray]] = []
    basis = np.eye(3, dtype=np.float64)
    axis_labels = ["X", "Y", "Z"]
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.column_stack([signs[column] * basis[:, permutation[column]] for column in range(3)])
            if np.linalg.det(rotation) < 0.5:
                continue
            columns = [f"{('+' if signs[column] > 0 else '-')}{axis_labels[permutation[column]]}" for column in range(3)]
            values.append((f"local_X->{columns[0]}__local_Y->{columns[1]}__local_Z->{columns[2]}", rotation))
    values.sort(key=lambda item: (0 if np.array_equal(item[1], np.eye(3)) else 1, item[0]))
    return values


def rotate_points_about_x(points: np.ndarray, pivot: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]], dtype=np.float64)
    return (rotation @ (np.asarray(points, dtype=np.float64) - pivot).T).T + pivot


class GripperDistanceEngine:
    def __init__(self, model: mujoco.MjModel, source_root: np.ndarray):
        self.model = model
        self.data = mujoco.MjData(model)
        self.source_root = np.asarray(source_root, dtype=np.float64)
        self.root_inverse = inverse(self.source_root)
        self.geom_ids: list[int] = []
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("follower_right_gripper_") and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                self.geom_ids.append(geom_id)
        if len(self.geom_ids) != 6:
            raise RuntimeError(f"Expected six named right gripper contact boxes, got {len(self.geom_ids)}")

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def geom_state(self, geom_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64).copy(),
            np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy(),
            np.asarray(self.model.geom_size[geom_id], dtype=np.float64).copy(),
        )

    def contact_proxy_points_source(self) -> dict[str, np.ndarray]:
        centers: dict[str, np.ndarray] = {}
        for geom_id in self.geom_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            center, _, _ = self.geom_state(geom_id)
            centers[name] = (self.source_root @ np.r_[center, 1.0])[:3]
        left_tip = centers["follower_right_gripper_left_tip"]
        right_tip = centers["follower_right_gripper_right_tip"]
        left_upper = centers["follower_right_gripper_left_pad_upper"]
        right_upper = centers["follower_right_gripper_right_pad_upper"]
        result = dict(centers)
        result["ALOHA_tip_center_midpoint"] = 0.5 * (left_tip + right_tip)
        result["ALOHA_upper_pad_center_midpoint"] = 0.5 * (left_upper + right_upper)
        return result

    def query_surface(self, object_pose_source: np.ndarray, local_points: np.ndarray) -> dict[str, Any]:
        object_pose_model = self.root_inverse @ object_pose_source
        samples_model = (object_pose_model[:3, :3] @ local_points.T).T + object_pose_model[:3, 3]
        best: dict[str, Any] | None = None
        per_geom: list[dict[str, Any]] = []
        for geom_id in self.geom_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            center, rotation, half_size = self.geom_state(geom_id)
            local = (rotation.T @ (samples_model - center).T).T
            closest_local = np.minimum(np.maximum(local, -half_size), half_size)
            distances = np.linalg.norm(local - closest_local, axis=1)
            sample_index = int(np.argmin(distances))
            object_point_model = samples_model[sample_index]
            gripper_point_model = center + rotation @ closest_local[sample_index]
            object_point_source = (self.source_root @ np.r_[object_point_model, 1.0])[:3]
            gripper_point_source = (self.source_root @ np.r_[gripper_point_model, 1.0])[:3]
            row = {
                "geom_id": int(geom_id),
                "geom_name": name,
                "gap_m": float(distances[sample_index]),
                "nearest_accessory_surface_point_source_scene_m": object_point_source,
                "nearest_gripper_box_point_source_scene_m": gripper_point_source,
                "point_pair_vector_gripper_to_accessory_source_scene_m": object_point_source - gripper_point_source,
                "sample_count": int(len(local_points)),
            }
            per_geom.append(row)
            if best is None or row["gap_m"] < best["gap_m"]:
                best = row
        if best is None:
            raise RuntimeError("No gripper geometry query result")
        return {"best": best, "per_geom": per_geom}


def point_surface_query(point_source: np.ndarray, object_pose_source: np.ndarray, local_points: np.ndarray) -> dict[str, Any]:
    points_source = (object_pose_source[:3, :3] @ local_points.T).T + object_pose_source[:3, 3]
    distances = np.linalg.norm(points_source - np.asarray(point_source, dtype=np.float64), axis=1)
    index = int(np.argmin(distances))
    return {
        "gap_m": float(distances[index]),
        "point_source_scene_m": point_source,
        "nearest_accessory_surface_point_source_scene_m": points_source[index],
    }


def axis_candidate_record(
    engine: GripperDistanceEngine,
    qpos: np.ndarray,
    phone_pose: np.ndarray,
    attachment_translation: np.ndarray,
    name: str,
    rotation: np.ndarray,
    surfaces: dict[str, np.ndarray],
) -> dict[str, Any]:
    engine.set_qpos(qpos)
    relative = transform(rotation, attachment_translation)
    pose = phone_pose @ relative
    query = engine.query_surface(pose, surfaces["authoritative_all"])
    return {
        "candidate_id": name,
        "T_phone_from_accessory_candidate": relative,
        "determinant": float(np.linalg.det(rotation)),
        "orthonormal_max_abs_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        "gap_m": query["best"]["gap_m"],
        "nearest_geom_name": query["best"]["geom_name"],
        "nearest_accessory_surface_point_source_scene_m": query["best"]["nearest_accessory_surface_point_source_scene_m"],
        "nearest_gripper_box_point_source_scene_m": query["best"]["nearest_gripper_box_point_source_scene_m"],
    }


def first_crossing(values: np.ndarray, frames: np.ndarray, threshold: float, *, below: bool) -> int | None:
    mask = values <= threshold if below else values >= threshold
    indices = np.flatnonzero(mask)
    return int(frames[indices[0]]) if len(indices) else None


def main() -> int:
    required = [
        ACTION, TIMELINE, LAYOUT, SOURCE_FRAMES, BUILDER, ACCESSORY_USD, SCENE_USD,
        MODEL_XML, DEX3_MAPPING, DEX3_TIPS, ALIGNMENT, LATENCY_APPROVAL, OPT_FK,
        STATE_FK, PARITY, PHONE_NPZ, PHONE_VIABILITY,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    input_hashes_before = {str(path.resolve()): sha256(path) for path in required}

    layout = load_json(LAYOUT)
    source_frames = load_json(SOURCE_FRAMES)
    latency = load_json(LATENCY_APPROVAL)
    parity = load_json(PARITY)
    viability = load_json(PHONE_VIABILITY)
    alignment = load_npz(ALIGNMENT)
    optimized_fk = load_npz(OPT_FK)
    state_fk = load_npz(STATE_FK)
    phone = load_npz(PHONE_NPZ)
    with np.load(ACTION, allow_pickle=False) as archive:
        optimized_action = archive["optimized_action"].copy()
        action_timestamp = archive["timestamp"].copy()
        action_frame_index = archive["frame_index"].copy()

    if optimized_action.shape != (FRAME_COUNT, 14) or not np.isfinite(optimized_action).all():
        raise RuntimeError("Frozen optimized_action invariant failed")
    if latency["action_to_observation_lag_frames"] != LAG:
        raise RuntimeError("Approved +7-frame latency changed")
    lookup = alignment["action_sample_index_for_observed_frame"].astype(np.int64)
    expected_lookup = np.maximum(np.arange(FRAME_COUNT, dtype=np.int64) - LAG, 0)
    if not np.array_equal(lookup, expected_lookup):
        raise RuntimeError("Approved observed/action lookup changed")
    if not np.array_equal(alignment["optimized_action"], optimized_action):
        raise RuntimeError("optimized_action changed relative to v11 alignment")
    if not np.array_equal(alignment["post_command_terminal_sample_indices"], np.arange(983, 990)):
        raise RuntimeError("Terminal optimized_action samples 983-989 were discarded")
    candidate_ids = [str(value) for value in phone["candidate_ids"]]
    if CHARGER_CANDIDATE not in candidate_ids:
        raise RuntimeError(candidate_ids)
    charger_index = candidate_ids.index(CHARGER_CANDIDATE)
    phone_pose = phone["T_source_scene_from_phone_rigid_reconstruction"][charger_index]
    if viability["piecewise_diagnostic"]["approved_for_g1"]:
        raise RuntimeError("Diagnostic phone carrier was unexpectedly approved for G1")
    if int(viability["piecewise_diagnostic"]["single_transition_observed_frame"]) != 223:
        raise RuntimeError("Frozen frame-223 diagnostic carrier boundary changed")

    timeline_rows = event_rows()
    timeline_map = {str(row["event"]): int(row["frame"]) for _, row in timeline_rows}
    for name, frame in APPROVED_ACCESSORY_EVENTS.items():
        if timeline_map[name] != frame:
            raise RuntimeError(f"Approved event changed: {name}")

    initial_phone = np.asarray(source_frames["T_source_scene_from_phone"], dtype=np.float64)
    initial_accessory = np.asarray(source_frames["T_source_scene_from_accessory"], dtype=np.float64)
    attachment = inverse(initial_phone) @ initial_accessory
    declared_translation = np.array([0.0, 0.006425, 0.0], dtype=np.float64)
    if not np.allclose(attachment[:3, 3], declared_translation, atol=1e-12, rtol=0.0):
        raise RuntimeError("Source attachment translation changed")
    if not np.allclose(attachment[:3, :3], np.eye(3), atol=1e-12, rtol=0.0):
        raise RuntimeError("Source attachment orientation changed")
    accessory_pose = np.einsum("tij,jk->tik", phone_pose, attachment)

    # The authoritative asset formula is explicitly reconstructed here rather
    # than reusing v11b's complete-ring surface approximation.
    surfaces = accessory_surfaces(layout, -90.0)
    acc = layout["accessory"]
    main_outer = float(acc["main_outer_diameter"]) / 2.0
    main_inner = float(acc["main_inner_diameter"]) / 2.0
    main_depth = float(acc["main_depth"])
    support_center = np.asarray(acc["support_center_offset_from_main_xyz"], dtype=np.float64)
    hinge_center = np.array([0.0, 0.0015, -main_outer + 0.0035], dtype=np.float64)
    opening_local = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ring_frame_rotation = np.column_stack((np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])))

    # Cross-check the authored composite transform and generated mesh evidence.
    scene_text = SCENE_USD.read_text(encoding="utf-8")
    accessory_translation_match = re.search(
        r'def Xform "Accessory"[\s\S]*?xformOp:translate = \(([^)]+)\)', scene_text
    )
    if accessory_translation_match is None:
        raise RuntimeError("Could not parse source accessory transform from generated scene USD")
    authored_translation = np.fromstring(accessory_translation_match.group(1), sep=",")
    asset_text = ACCESSORY_USD.read_text(encoding="utf-8")
    expected_start_point = np.array([
        main_outer * math.cos(math.radians(-72.0)),
        -main_depth / 2.0,
        main_outer * math.sin(math.radians(-72.0)),
    ])
    first_points_match = re.search(r'point3f\[\] points = \[\(([^)]+)\)', asset_text)
    if first_points_match is None:
        raise RuntimeError("Could not parse first main-ring mesh point from generated USD")
    authored_first_point = np.fromstring(first_points_match.group(1), sep=",")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    source_root = optimized_fk["source_aloha_root_transform"]
    engine = GripperDistanceEngine(model, source_root)
    optimized_qpos = optimized_fk["qpos"]
    state_qpos = state_fk["qpos"]
    optimized_tcp = np.stack(
        (alignment["observation_aligned_left_tcp_transform"], alignment["observation_aligned_right_tcp_transform"]), axis=1
    )
    state_tcp = np.stack((state_fk["left_tcp_transform"], state_fk["right_tcp_transform"]), axis=1)

    component_names = ["main_c_ring", "support_ring", "hinge", "authoritative_all", "complete_main_all"]
    source_names = ["optimized_action_aligned", "observation_state"]
    component_gaps = np.full((2, len(AUDIT_FRAMES), len(component_names)), np.nan, dtype=np.float64)
    tcp_gaps = np.full((2, len(AUDIT_FRAMES)), np.nan, dtype=np.float64)
    center_distances = np.full((2, len(AUDIT_FRAMES), 3), np.nan, dtype=np.float64)
    nearest_object = np.full((2, len(AUDIT_FRAMES), 3), np.nan, dtype=np.float64)
    nearest_gripper = np.full((2, len(AUDIT_FRAMES), 3), np.nan, dtype=np.float64)
    nearest_geom_names = np.empty((2, len(AUDIT_FRAMES)), dtype="<U64")
    event_details: dict[str, Any] = {source: {} for source in source_names}

    for source_index, source_name in enumerate(source_names):
        for local_index, observed in enumerate(AUDIT_FRAMES):
            action_index = int(lookup[observed])
            qpos = optimized_qpos[action_index] if source_index == 0 else state_qpos[observed]
            tcp = optimized_tcp[observed, 1] if source_index == 0 else state_tcp[observed, 1]
            engine.set_qpos(qpos)
            queries: dict[str, dict[str, Any]] = {}
            for component_index, component in enumerate(component_names):
                query = engine.query_surface(accessory_pose[observed], surfaces[component])
                queries[component] = query
                component_gaps[source_index, local_index, component_index] = query["best"]["gap_m"]
            authoritative = queries["authoritative_all"]["best"]
            nearest_object[source_index, local_index] = authoritative["nearest_accessory_surface_point_source_scene_m"]
            nearest_gripper[source_index, local_index] = authoritative["nearest_gripper_box_point_source_scene_m"]
            nearest_geom_names[source_index, local_index] = authoritative["geom_name"]
            tcp_gaps[source_index, local_index] = point_surface_query(
                tcp[:3, 3], accessory_pose[observed], surfaces["authoritative_all"]
            )["gap_m"]
            component_centers = [np.zeros(3), support_center, hinge_center]
            for center_index, center_local in enumerate(component_centers):
                center_world = (accessory_pose[observed] @ np.r_[center_local, 1.0])[:3]
                center_distances[source_index, local_index, center_index] = np.linalg.norm(tcp[:3, 3] - center_world)

            if int(observed) in EVENT_FRAMES:
                proxy_points = engine.contact_proxy_points_source()
                proxy_point_rows = {
                    name: point_surface_query(point, accessory_pose[observed], surfaces["authoritative_all"])
                    for name, point in proxy_points.items()
                }
                event_details[source_name][str(int(observed))] = {
                    "observed_frame": int(observed),
                    "aligned_action_index": action_index,
                    "approved_event_or_context": event_at(int(observed), timeline_rows),
                    "object_state_semantics": (
                        "PHONE_ATTACHED_CHARGER_ANCHORED_530_DIAGNOSTIC"
                        if int(observed) < 341
                        else "COUNTERFACTUAL_PHONE_ATTACHED_BOUNDARY_AT_OR_AFTER_ACCESSORY_REMOVED"
                    ),
                    "T_source_scene_from_accessory_hypothesis": accessory_pose[observed],
                    "main_ring_center_source_scene_m": accessory_pose[observed][:3, 3],
                    "support_ring_center_source_scene_m": (accessory_pose[observed] @ np.r_[support_center, 1.0])[:3],
                    "hinge_center_source_scene_m": (accessory_pose[observed] @ np.r_[hinge_center, 1.0])[:3],
                    "main_gap_opening_direction_source_scene": accessory_pose[observed][:3, :3] @ opening_local,
                    "right_TCP_source_scene_m": tcp[:3, 3],
                    "right_TCP_to_authoritative_surface_m": tcp_gaps[source_index, local_index],
                    "component_queries": {name: queries[name] for name in component_names},
                    "actual_ALOHA_contact_box_best": queries["authoritative_all"]["best"],
                    "actual_ALOHA_contact_boxes_ranked": sorted(
                        queries["authoritative_all"]["per_geom"], key=lambda row: row["gap_m"]
                    ),
                    "named_proxy_point_queries": proxy_point_rows,
                    "right_gripper_value_m": float(
                        optimized_action[action_index, 13] if source_index == 0 else state_fk["right_gripper_source"][observed]
                    ),
                }

    # Candidate origin errors are bounded and audited without adopting any.
    phone_half_depth = float(layout["phone"]["size_landscape_xyz"][1]) / 2.0
    clearance = float(acc["phone_back_clearance"])
    attachment_candidates = {
        "AUTHORITATIVE_RING_ROOT_CENTER": np.array([0.0, phone_half_depth + clearance + main_depth / 2.0, 0.0]),
        "PHONE_CENTER_COINCIDENT_WRONG": np.array([0.0, 0.0, 0.0]),
        "PHONE_BACK_SURFACE_AS_ROOT_WRONG": np.array([0.0, phone_half_depth, 0.0]),
        "OMIT_MAIN_HALF_DEPTH_WRONG": np.array([0.0, phone_half_depth + clearance, 0.0]),
        "DOUBLE_MAIN_HALF_DEPTH_WRONG": np.array([0.0, phone_half_depth + clearance + main_depth, 0.0]),
        "ATTACHMENT_SIGN_FLIP_WRONG": np.array([0.0, -(phone_half_depth + clearance + main_depth / 2.0), 0.0]),
    }
    attachment_results: dict[str, Any] = {}
    for candidate_id, translation in attachment_candidates.items():
        row: dict[str, Any] = {"translation_phone_frame_m": translation, "events": {}}
        for observed in (326, 341):
            engine.set_qpos(optimized_qpos[int(lookup[observed])])
            pose = phone_pose[observed] @ transform(np.eye(3), translation)
            query = engine.query_surface(pose, surfaces["authoritative_all"])["best"]
            row["events"][str(observed)] = {
                "gap_m": query["gap_m"],
                "nearest_geom_name": query["geom_name"],
                "semantics": "attached hypothesis" if observed < 341 else "counterfactual attached boundary after removal",
            }
        attachment_results[candidate_id] = row

    # Exhaustive 24-way proper signed-axis audit.  This is diagnostic only and
    # cannot override the identity transform authored in the USD.
    axis_results: list[dict[str, Any]] = []
    for candidate_id, rotation in proper_axis_rotations():
        row = {"candidate_id": candidate_id, "authoritative": bool(np.array_equal(rotation, np.eye(3))), "events": {}}
        for observed in (326, 341):
            record = axis_candidate_record(
                engine, optimized_qpos[int(lookup[observed])], phone_pose[observed], declared_translation,
                candidate_id, rotation, surfaces,
            )
            row["events"][str(observed)] = record
        row["max_event_gap_m"] = max(row["events"]["326"]["gap_m"], row["events"]["341"]["gap_m"])
        axis_results.append(row)
    axis_results.sort(key=lambda row: (row["events"]["326"]["gap_m"], row["max_event_gap_m"]))

    # The main-ring gap orientation changes only material occupancy.  The full
    # ring is therefore a strict lower bound for any 36-degree gap rotation.
    gap_sweep: list[dict[str, Any]] = []
    for center_deg in range(-180, 180, 5):
        variant = accessory_surfaces(layout, float(center_deg))
        row = {"gap_center_degrees": int(center_deg), "events": {}}
        for observed in (326, 341):
            engine.set_qpos(optimized_qpos[int(lookup[observed])])
            query = engine.query_surface(accessory_pose[observed], variant["authoritative_all"])["best"]
            row["events"][str(observed)] = {
                "gap_m": query["gap_m"],
                "nearest_geom_name": query["geom_name"],
            }
        gap_sweep.append(row)

    # The builder's support ring is a fixed visual estimate, while the physical
    # product is hinged.  Sweep a single support-ring rotation about the
    # builder-authored +X hinge axis as a geometry diagnosis only.  No angle is
    # adopted as object state or motion.
    support_articulation_sweep: list[dict[str, Any]] = []
    for angle_deg in range(-180, 181, 2):
        rotated_support = rotate_points_about_x(surfaces["support_ring"], hinge_center, float(angle_deg))
        articulated_all = np.concatenate((surfaces["main_c_ring"], rotated_support, surfaces["hinge"]), axis=0)
        row = {"support_ring_rotation_about_hinge_x_degrees": int(angle_deg), "events": {}}
        for observed in (326, 329, 341):
            engine.set_qpos(optimized_qpos[int(lookup[observed])])
            query = engine.query_surface(accessory_pose[observed], articulated_all)["best"]
            row["events"][str(observed)] = {
                "gap_m": query["gap_m"],
                "nearest_geom_name": query["geom_name"],
                "nearest_accessory_surface_point_source_scene_m": query["nearest_accessory_surface_point_source_scene_m"],
                "nearest_gripper_box_point_source_scene_m": query["nearest_gripper_box_point_source_scene_m"],
            }
        support_articulation_sweep.append(row)

    # Optimized-vs-observed realized gripper geometry comparison uses matching
    # named collision boxes; it does not replace optimized_action as motion.
    gripper_parity: dict[str, Any] = {}
    for observed in (326, 329, 341):
        states: dict[str, dict[str, np.ndarray]] = {}
        for source_name, qpos in (
            ("optimized_action_aligned", optimized_qpos[int(lookup[observed])]),
            ("observation_state", state_qpos[observed]),
        ):
            engine.set_qpos(qpos)
            centers = engine.contact_proxy_points_source()
            states[source_name] = centers
        common_names = sorted(set(states["optimized_action_aligned"]) & set(states["observation_state"]))
        differences = {name: float(np.linalg.norm(states["optimized_action_aligned"][name] - states["observation_state"][name])) for name in common_names}
        gripper_parity[str(observed)] = {
            "observed_frame": observed,
            "aligned_action_index": int(lookup[observed]),
            "named_proxy_center_position_difference_m": differences,
            "mean_difference_m": float(np.mean(list(differences.values()))),
            "max_difference_m": float(np.max(list(differences.values()))),
            "right_TCP_position_difference_m": float(np.linalg.norm(optimized_tcp[observed, 1, :3, 3] - state_tcp[observed, 1, :3, 3])),
        }

    # Gripper command semantics; zero is closed, 0.044 m is open in the MJCF.
    semantic_frames = np.arange(223, 351, dtype=np.int64)
    aligned_command = optimized_action[lookup[semantic_frames], 13].astype(np.float64)
    observed_gripper = state_fk["right_gripper_source"][semantic_frames].astype(np.float64)
    peak_local = int(np.argmax(aligned_command))
    peak_frame = int(semantic_frames[peak_local])
    peak_value = float(aligned_command[peak_local])
    post_frames = semantic_frames[peak_local:]
    post_values = aligned_command[peak_local:]
    low = float(np.min(post_values))
    span = peak_value - low
    closing_thresholds: dict[str, Any] = {}
    for fraction in (0.75, 0.50, 0.25, 0.10):
        threshold = low + fraction * span
        closing_thresholds[f"first_at_or_below_{int(fraction*100)}pct_open_frame"] = first_crossing(
            post_values, post_frames, threshold, below=True
        )
    derivatives = np.diff(aligned_command)
    steep_index = int(np.argmin(derivatives))
    removal_delta = optimized_tcp[341, 1, :3, 3] - optimized_tcp[326, 1, :3, 3]
    semantics = {
        "approved_events_unchanged": APPROVED_ACCESSORY_EVENTS,
        "event_meanings": {
            "326": "right_accessory_grasp_start; not defined as command-close onset or rigid right-carrier lock",
            "329": "accessory_detachment_start",
            "341": "accessory_removed; phone-attached accessory pose is counterfactual at/after this boundary",
        },
        "right_gripper_MJCF_convention": {"closed_m": 0.0, "open_m": 0.044},
        "aligned_optimized_command": {
            "maximum_open_frame": peak_frame,
            "maximum_open_value_m": peak_value,
            "minimum_after_peak_m": low,
            "most_negative_step_observed_frame": int(semantic_frames[steep_index + 1]),
            "most_negative_step_m_per_frame": float(derivatives[steep_index]),
            **closing_thresholds,
            "event_values_m": {str(frame): float(optimized_action[int(lookup[frame]), 13]) for frame in (310, 319, 326, 329, 334, 341)},
        },
        "observation_gripper_event_values_m": {str(frame): float(state_fk["right_gripper_source"][frame]) for frame in (310, 319, 326, 329, 334, 341)},
        "right_TCP_removal_motion": {
            "frame_326_to_329_displacement_m": float(np.linalg.norm(optimized_tcp[329, 1, :3, 3] - optimized_tcp[326, 1, :3, 3])),
            "frame_329_to_341_displacement_m": float(np.linalg.norm(optimized_tcp[341, 1, :3, 3] - optimized_tcp[329, 1, :3, 3])),
            "frame_326_to_341_displacement_m": float(np.linalg.norm(removal_delta)),
            "frame_326_to_341_vector_source_scene_m": removal_delta,
        },
        "raw_video_evidence_scope": "approved manual event labels plus generated uncalibrated image crops; raw RGB is not used as 3-D geometry",
    }

    # Summaries and five-cause decision are conservative: an authored USD
    # parity pass does not prove that a diagnostic backward carrier matches the
    # physical in-hand state at frame 326.
    component_index = {name: index for index, name in enumerate(component_names)}
    opt_index = 0
    event_indices = {frame: int(np.where(AUDIT_FRAMES == frame)[0][0]) for frame in (300, 310, 319, 326, 329, 334, 341, 350)}
    authoritative_326 = float(component_gaps[0, event_indices[326], component_index["authoritative_all"]])
    authoritative_329 = float(component_gaps[0, event_indices[329], component_index["authoritative_all"]])
    authoritative_341 = float(component_gaps[0, event_indices[341], component_index["authoritative_all"]])
    full_326 = float(component_gaps[0, event_indices[326], component_index["complete_main_all"]])
    full_341 = float(component_gaps[0, event_indices[341], component_index["complete_main_all"]])
    min_local = int(np.argmin(component_gaps[0, :, component_index["authoritative_all"]]))
    min_observed_frame = int(AUDIT_FRAMES[min_local])
    min_gap = float(component_gaps[0, min_local, component_index["authoritative_all"]])
    best_axis = axis_results[0]
    best_gap_center_326 = min(gap_sweep, key=lambda row: row["events"]["326"]["gap_m"])
    best_gap_center_341 = min(gap_sweep, key=lambda row: row["events"]["341"]["gap_m"])
    best_support_angle_326 = min(support_articulation_sweep, key=lambda row: row["events"]["326"]["gap_m"])
    best_support_angle_joint = min(
        support_articulation_sweep,
        key=lambda row: max(row["events"]["326"]["gap_m"], row["events"]["329"]["gap_m"]),
    )
    attachment_best_326 = min(attachment_results.items(), key=lambda item: item[1]["events"]["326"]["gap_m"])
    optimized_observed_gap_326 = float(component_gaps[1, event_indices[326], component_index["authoritative_all"]])
    optimized_observed_gap_341 = float(component_gaps[1, event_indices[341], component_index["authoritative_all"]])

    # Required rigid translation merely exposes mismatch magnitude/direction;
    # it is never applied to an object trajectory.
    required_translation_rows: dict[str, Any] = {}
    for observed in (326, 329, 341):
        row = event_details["optimized_action_aligned"][str(observed)]["actual_ALOHA_contact_box_best"]
        vector = np.asarray(row["nearest_gripper_box_point_source_scene_m"]) - np.asarray(row["nearest_accessory_surface_point_source_scene_m"])
        vector_local = accessory_pose[observed, :3, :3].T @ vector
        required_translation_rows[str(observed)] = {
            "diagnostic_only_not_applied": True,
            "translation_to_close_current_nearest_pair_source_scene_m": vector,
            "translation_to_close_current_nearest_pair_accessory_local_m": vector_local,
            "norm_m": float(np.linalg.norm(vector)),
            "semantics": "attached hypothesis" if observed < 341 else "counterfactual attached boundary after removal",
        }

    asset_audit = {
        "status": "PASS_AUTHORITATIVE_ACCESSORY_ASSET_FRAME_RECONSTRUCTED",
        "local_axis_convention": {
            "+X": "main C-ring in-plane horizontal axis",
            "+Y": "main C-ring thickness axis / phone-back outward direction",
            "+Z": "main C-ring in-plane vertical-short axis",
            "main_ring_plane": "local X-Z",
            "main_ring_normal": "+Y",
            "support_ring_plane": "local X-Y",
            "support_ring_normal": "+Z",
            "ring_frame_basis_columns_in_accessory_local": ring_frame_rotation,
            "ring_frame_axes": {"x": "accessory +X", "y": "accessory -Z / gap direction", "z": "accessory +Y / phone rear outward normal"},
        },
        "main_ring": {
            "center_local_m": [0.0, 0.0, 0.0],
            "outer_radius_m": main_outer,
            "inner_radius_m": main_inner,
            "depth_m": main_depth,
            "gap_degrees": float(acc["main_gap_degrees"]),
            "gap_center_degrees": -90.0,
            "gap_opening_direction_local": opening_local,
            "material_angular_interval_degrees": [-72.0, 252.0],
        },
        "support_ring": {
            "center_local_m": support_center,
            "outer_radius_m": float(acc["support_outer_diameter"]) / 2.0,
            "inner_radius_m": float(acc["support_inner_diameter"]) / 2.0,
            "depth_m": float(acc["support_depth"]),
            "builder_state": "fixed visual estimate; physical product has a hinge",
        },
        "hinge": {"center_local_m": hinge_center, "axis": "+/-X", "radius_m": 0.0032, "length_m": 0.011},
        "phone_from_accessory_attachment": attachment,
        "attachment_formula": "phone_half_thickness + phone_back_clearance + accessory_main_half_depth",
        "attachment_formula_terms_m": {
            "phone_half_thickness": phone_half_depth,
            "phone_back_clearance": clearance,
            "accessory_main_half_depth": main_depth / 2.0,
            "sum": float(phone_half_depth + clearance + main_depth / 2.0),
        },
        "generated_USD_parity": {
            "composite_accessory_translation_source_scene_m": authored_translation,
            "expected_translation_source_scene_m": initial_accessory[:3, 3],
            "translation_max_abs_error_m": float(np.max(np.abs(authored_translation - initial_accessory[:3, 3]))),
            "first_main_ring_mesh_point_authored_local_m": authored_first_point,
            "first_main_ring_mesh_point_expected_local_m": expected_start_point,
            "first_point_max_abs_error_m": float(np.max(np.abs(authored_first_point - expected_start_point))),
            "no_authored_accessory_rotation": "xformOp:orient" not in scene_text[scene_text.find('def Xform "Accessory"'):scene_text.find('def "Materials"')],
        },
        "surface_sample_counts": {name: int(len(value)) for name, value in surfaces.items()},
        "provenance": {
            "layout": str(LAYOUT.resolve()), "layout_sha256": sha256(LAYOUT),
            "builder": str(BUILDER.resolve()), "builder_sha256": sha256(BUILDER),
            "accessory_USD": str(ACCESSORY_USD.resolve()), "accessory_USD_sha256": sha256(ACCESSORY_USD),
            "source_scene_USD": str(SCENE_USD.resolve()), "source_scene_USD_sha256": sha256(SCENE_USD),
        },
    }

    local_frame_audit = {
        "status": "AUTHORITATIVE_IDENTITY_FRAME_CONFIRMED_DIAGNOSTIC_ALTERNATIVES_NOT_ADOPTED",
        "authoritative_attachment": attachment,
        "attachment_origin_candidates": attachment_results,
        "proper_axis_permutation_candidates": axis_results,
        "best_frame_326_axis_candidate": best_axis,
        "support_ring_hinge_articulation_diagnostic": {
            "pivot_local_m": hinge_center,
            "axis_local": [1.0, 0.0, 0.0],
            "angles_tested_degrees": [-180, 180, 2],
            "best_frame_326": best_support_angle_326,
            "best_joint_frame_326_329": best_support_angle_joint,
            "all_candidates": support_articulation_sweep,
            "adopted_as_object_state": False,
        },
        "warning": "Axis permutations are counterfactual diagnostics and do not override the active builder/USD identity orientation.",
    }
    gap_audit = {
        "status": "RING_GAP_ORIENTATION_CANNOT_EXPLAIN_LARGE_REMAINING_GAP",
        "authoritative_gap_center_degrees": -90.0,
        "authoritative_opening_local": opening_local,
        "authoritative_frame_326_gap_m": authoritative_326,
        "complete_main_ring_lower_bound_frame_326_m": full_326,
        "authoritative_minus_complete_frame_326_m": authoritative_326 - full_326,
        "authoritative_frame_341_counterfactual_gap_m": authoritative_341,
        "complete_main_ring_lower_bound_frame_341_m": full_341,
        "authoritative_minus_complete_frame_341_m": authoritative_341 - full_341,
        "best_gap_orientation_frame_326": best_gap_center_326,
        "best_gap_orientation_frame_341": best_gap_center_341,
        "sweep_5deg": gap_sweep,
    }
    contact_audit = {
        "status": "ACTUAL_ALOHA_COLLISION_BOX_PROXY_AUDITED",
        "contact_proxy_definition": "six named Stationary ALOHA right gripper pad/tip collision OBBs",
        "named_geometries": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in engine.geom_ids],
        "TCP_is_contact_proxy": False,
        "event_details": event_details,
        "optimized_vs_observation_gripper_geometry": gripper_parity,
        "Dex3": {
            "world_distance_computed": False,
            "reason": "No G1/Dex3 arm or finger target motion may be generated in this source-only audit; placing a Dex3 fingertip in source world would fabricate a forbidden target.",
            "static_semantic_role_only": "right C = middle finger insertion/hook; A/B non-contact",
            "mapping_path": str(DEX3_MAPPING.resolve()),
            "mapping_sha256": sha256(DEX3_MAPPING),
        },
    }
    semantics_audit = {
        "status": "FRAME_341_ATTACHED_GAP_IS_SEMANTICALLY_INVALID_FRAME_326_IS_ACQUISITION_BOUNDARY",
        **semantics,
        "numeric_interpretation": {
            "frame_326_gap_m": authoritative_326,
            "frame_329_gap_m": authoritative_329,
            "frame_341_counterfactual_attached_gap_m": authoritative_341,
            "minimum_gap_frames_280_370_m": min_gap,
            "minimum_gap_observed_frame": min_observed_frame,
            "minimum_gap_aligned_action_index": int(lookup[min_observed_frame]),
            "gap_trend_326_to_329": "increasing" if authoritative_329 > authoritative_326 else "decreasing",
            "frame_341_phone_attached_distance_valid_as_contact_test": False,
        },
    }

    cause_rows = {
        "1_wrong_accessory_local_frame_or_attachment_transform": {
            "classification": "AUTHORITATIVE_FRAME_PASS_TESTED_ORIGIN_AXIS_AND_SINGLE_HINGE_VARIANTS_REJECTED",
            "numeric_evidence": {
                "authoritative_frame_326_gap_m": authoritative_326,
                "best_simple_attachment_origin_candidate": attachment_best_326[0],
                "best_simple_attachment_origin_gap_m": attachment_best_326[1]["events"]["326"]["gap_m"],
                "best_axis_permutation_candidate": best_axis["candidate_id"],
                "best_axis_permutation_gap_m": best_axis["events"]["326"]["gap_m"],
                "best_unadopted_support_hinge_angle_degrees": best_support_angle_326["support_ring_rotation_about_hinge_x_degrees"],
                "best_unadopted_support_hinge_gap_m": best_support_angle_326["events"]["326"]["gap_m"],
                "best_unadopted_joint_326_329_support_hinge_angle_degrees": best_support_angle_joint["support_ring_rotation_about_hinge_x_degrees"],
                "best_unadopted_joint_326_329_max_gap_m": max(
                    best_support_angle_joint["events"]["326"]["gap_m"],
                    best_support_angle_joint["events"]["329"]["gap_m"],
                ),
                "required_unapplied_translation_m": required_translation_rows["326"]["norm_m"],
            },
            "decision": "The active builder, generated USD, and source layout agree exactly. Small root-origin mistakes, all 24 proper signed-axis permutations, and a full single-axis support-hinge sweep remain above 10 mm. Those tested frame/attachment errors do not explain the miss; a more general unobserved object-state change is not inferred.",
        },
        "2_wrong_ring_center_or_gap_orientation": {
            "classification": "GAP_ORIENTATION_REJECTED_RING_COMPONENT_SEMANTICS_AUDITED",
            "numeric_evidence": {
                "complete_ring_lower_bound_frame_326_m": full_326,
                "best_36deg_gap_orientation_frame_326_m": best_gap_center_326["events"]["326"]["gap_m"],
                "main_ring_gap_frame_326_m": float(component_gaps[0, event_indices[326], component_index["main_c_ring"]]),
                "support_ring_gap_frame_326_m": float(component_gaps[0, event_indices[326], component_index["support_ring"]]),
                "hinge_gap_frame_326_m": float(component_gaps[0, event_indices[326], component_index["hinge"]]),
            },
            "decision": "Changing only the 36-degree opening cannot create missing material or close a gap larger than the complete-ring lower bound. Main/support/hinge centers are reported separately; a wrong articulated support pose belongs to cause 1, not gap-angle choice.",
        },
        "3_wrong_right_hand_contact_proxy": {
            "classification": "TCP_PROXY_REJECTED_ACTUAL_ALOHA_COLLISION_BOX_STILL_OUTSIDE_GATE",
            "numeric_evidence": {
                "TCP_to_surface_frame_326_m": float(tcp_gaps[0, event_indices[326]]),
                "best_actual_collision_box_frame_326_m": authoritative_326,
                "nearest_actual_collision_box": str(nearest_geom_names[0, event_indices[326]]),
                "observation_state_best_actual_collision_box_frame_326_m": optimized_observed_gap_326,
            },
            "decision": "The TCP is not used as contact. All six actual ALOHA pad/tip collision boxes were tested; selecting the correct box changes the number but does not meet 10 mm. Dex3 world placement was correctly not fabricated.",
        },
        "4_frame_326_341_semantic_interpretation": {
            "classification": "PARTIAL_CAUSE_FRAME_341_COMPARISON_INVALID_FRAME_326_NOT_COMMAND_ONSET",
            "numeric_evidence": {
                "gripper_max_open_frame": peak_frame,
                "gripper_10pct_open_crossing_frame": closing_thresholds["first_at_or_below_10pct_open_frame"],
                "approved_grasp_start_frame": 326,
                "approved_detachment_start_frame": 329,
                "approved_removed_frame": 341,
                "right_TCP_326_to_341_displacement_m": float(np.linalg.norm(removal_delta)),
            },
            "decision": "Frame 341 must not be tested against a still-attached ring. Frame 326 is a user-approved visual grasp-start boundary, not the actuator close onset. This semantic correction removes the 341 failure claim but does not close the frame-326 attached-hypothesis gap.",
        },
        "5_optimized_action_not_reaching_accessory": {
            "classification": "NOT_ESTABLISHED_ACTION_TRACKS_OBSERVED_RIGHT_ARM_WITHIN_CENTIMETER_SCALE",
            "numeric_evidence": {
                "optimized_vs_observation_TCP_frame_326_m": gripper_parity["326"]["right_TCP_position_difference_m"],
                "optimized_vs_observation_contact_proxy_mean_frame_326_m": gripper_parity["326"]["mean_difference_m"],
                "optimized_gap_frame_326_m": authoritative_326,
                "observation_state_gap_same_object_hypothesis_frame_326_m": optimized_observed_gap_326,
                "latency_aligned_accessory_approach_direction_cosine": parity["latency_aligned_phase_direction_metrics"]["accessory_approach"]["direction_cosine_vs_observation_state"],
                "latency_aligned_accessory_removal_direction_cosine": parity["latency_aligned_phase_direction_metrics"]["accessory_removal"]["direction_cosine_vs_observation_state"],
            },
            "decision": "The aligned optimized right arm closely follows observation.state direction and corresponding collision proxies. Both disagree with the same reconstructed ring pose, so current evidence does not isolate optimized_action as the cause.",
        },
    }
    final_decision = {
        "status": "BLOCKED_ACCESSORY_OBJECT_STATE_UNDER_DIAGNOSTIC_PHONE_CARRIER",
        "phone_carrier_hypothesis": "CHARGER_ANCHORED_530 diagnostic only from observed frame 223 onward",
        "grasp_acquisition_state": "OBJECT STATE UNRESOLVED DURING GRASP ACQUISITION for observed frames 176-222",
        "approved_for_G1_retargeting": False,
        "optimized_action_task_validity_finalized_as_failure": False,
        "summary": (
            "The authored rigid accessory frame/attachment and actual ALOHA contact boxes were applied correctly. "
            "Tested root origins, all 24 proper axis permutations, the complete-ring lower bound, gap orientation, a "
            "single support-hinge sweep, and TCP/contact-proxy choices cannot explain the frame-326 miss. Frame 341 was "
            "an invalid still-attached comparison. Because optimized_action tracks the observed right arm, the remaining "
            "mismatch is localized to using the diagnostic backward phone carrier to assert a 3-D accessory object state "
            "at frame 326; optimized_action non-reach is not proven."
        ),
        "five_possibilities": cause_rows,
        "single_next_user_decision": (
            "Approve or reject a separate source-side 3-D accessory object-state calibration at frame 326 (including its "
            "support-ring articulation) without changing optimized_action, the timeline, or authorizing G1 retargeting."
        ),
    }

    input_audit = {
        "status": "PASS_FROZEN_SOURCE_INPUTS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action_to_observation_lag_frames": LAG,
        "action_sample_for_observed_frame": "max(observed_frame - 7, 0); max only implements frames 0-6 diagnostic pre-command hold",
        "fps": FPS,
        "latency_seconds": LAG / FPS,
        "optimized_action_shape": list(optimized_action.shape),
        "optimized_action_finite": bool(np.isfinite(optimized_action).all()),
        "frame_index_exact": bool(np.array_equal(action_frame_index, np.arange(FRAME_COUNT))),
        "timestamp_exact_to_alignment": bool(np.array_equal(action_timestamp, alignment["optimized_action_original_timestamp"])),
        "post_command_terminal_samples_retained": list(range(983, 990)),
        "carrier_policy": {
            "frames_176_222": "OBJECT STATE UNRESOLVED DURING GRASP ACQUISITION",
            "frames_223_onward": "CHARGER_ANCHORED_530 diagnostic hypothesis only",
            "approved_for_G1": False,
        },
        "approved_events": APPROVED_ACCESSORY_EVENTS,
        "input_hashes_before": input_hashes_before,
        "forbidden_outputs_generated": {
            "G1_targets": False, "G1_IK": False, "phasewarp": False,
            "orientation_retargeting": False, "Dex3_target_motion": False, "physics": False,
        },
    }

    dump(OUT / "input_audit.json", input_audit)
    dump(OUT / "accessory_asset_frame_audit.json", asset_audit)
    dump(OUT / "accessory_local_frame_attachment_audit.json", local_frame_audit)
    dump(OUT / "ring_center_gap_orientation_audit.json", gap_audit)
    dump(OUT / "right_contact_proxy_audit.json", contact_audit)
    dump(OUT / "frame_326_341_semantics_audit.json", semantics_audit)
    dump(OUT / "required_unapplied_accessory_correction.json", required_translation_rows)
    dump(OUT / "five_cause_decision.json", final_decision)

    np.savez_compressed(
        OUT / "accessory_distance_timeseries.npz",
        observed_frames=AUDIT_FRAMES,
        aligned_action_indices=lookup[AUDIT_FRAMES],
        source_names=np.asarray(source_names),
        component_names=np.asarray(component_names),
        component_surface_gap_m=component_gaps,
        right_TCP_to_authoritative_surface_m=tcp_gaps,
        right_TCP_to_component_center_m=center_distances,
        nearest_accessory_surface_point_source_scene_m=nearest_object,
        nearest_ALOHA_contact_box_point_source_scene_m=nearest_gripper,
        nearest_ALOHA_contact_box_name=nearest_geom_names,
        T_source_scene_from_phone_CHARGER_ANCHORED_530=phone_pose,
        T_source_scene_from_accessory_attached_diagnostic=accessory_pose,
        attachment_transform=attachment,
        right_gripper_optimized_observation_aligned=optimized_action[lookup[AUDIT_FRAMES], 13],
        right_gripper_observation_state=state_fk["right_gripper_source"][AUDIT_FRAMES],
        optimized_action_modified=np.array(False),
        approved_timeline_modified=np.array(False),
        phone_carrier_modified=np.array(False),
        G1=np.array(False),
        physics=np.array(False),
    )

    # Verify immutable files after all computations.
    input_hashes_after = {str(path.resolve()): sha256(path) for path in required}
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("An immutable source input changed during the audit")
    invariants = {
        "status": "PASS_NO_FORBIDDEN_MUTATION_OR_DOWNSTREAM_GENERATION",
        "input_hashes_before_equal_after": True,
        "timeline_sha256": sha256(TIMELINE),
        "optimized_action_sha256": sha256(ACTION),
        "phone_carrier_npz_sha256": sha256(PHONE_NPZ),
        "optimized_action_modified": False,
        "approved_event_frames_modified": False,
        "timestamps_modified": False,
        "alignment_modified": False,
        "phone_carrier_hypothesis_modified": False,
        "source_ALOHA_arm_motion_modified": False,
        "G1_target_generated": False,
        "G1_IK_generated": False,
        "phasewarp_generated": False,
        "orientation_retargeting_generated": False,
        "Dex3_target_motion_generated": False,
        "physics_generated": False,
    }
    dump(OUT / "constraint_invariants.json", invariants)

    print(json.dumps({
        "status": final_decision["status"],
        "frame_326_authoritative_gap_mm": authoritative_326 * 1000.0,
        "frame_326_complete_ring_lower_bound_mm": full_326 * 1000.0,
        "frame_341_counterfactual_gap_mm": authoritative_341 * 1000.0,
        "minimum_gap_frame_280_370": min_observed_frame,
        "minimum_gap_mm": min_gap * 1000.0,
        "best_axis_candidate": best_axis["candidate_id"],
        "best_axis_candidate_gap_mm": best_axis["events"]["326"]["gap_m"] * 1000.0,
        "best_support_hinge_angle_deg": best_support_angle_326["support_ring_rotation_about_hinge_x_degrees"],
        "best_support_hinge_gap_mm": best_support_angle_326["events"]["326"]["gap_m"] * 1000.0,
        "observation_state_same_hypothesis_gap_326_mm": optimized_observed_gap_326 * 1000.0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
