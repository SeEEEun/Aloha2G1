#!/usr/bin/env python3
"""Recover the frozen Proposed-B left-grasp construction for measured-doll physics.

This is a read-only provenance/reconstruction audit.  It does not rerun the
retargeter, solve IK, optimize a hand pose, or alter Dataset B.  The only FK
performed is a deterministic readback of already-frozen source/G1 joint rows.
The resulting command is the original Proposed-B LEFT_GRASP -> HOLD -> first
50-mm-lift prefix, with unchanged diagnostic dwell times from proxy-v3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.models import ALOHAKinematics, G1Kinematics


REPRESENTATIVE = ROOT / "outputs/paper_core_ab/source_conditioned_rollout/batch_result.json"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
B_TRAJECTORY = (
    ROOT
    / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories/episode_000013.npz"
)
COMMON_CONFIG = (
    ROOT
    / "outputs/doll_handoff_dataset_b_final/new_episode_conversion/"
    "frozen_proposed_b_runtime/config/common_config.json"
)
PROPOSED_CONFIG = (
    ROOT
    / "outputs/doll_handoff_dataset_b_final/new_episode_conversion/"
    "frozen_proposed_b_runtime/config/proposed_config.json"
)
SCENE = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
MEASURED_CONFIG = ROOT / "configs/doll_handoff_measured_proxy_v3.json"
DEFAULT_OUTPUT = ROOT / "outputs/proposed_b_measured_doll_left_grasp_diagnostic"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".incomplete")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".incomplete")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def fixed_list(column: Any, width: int) -> np.ndarray:
    values = column.combine_chunks()
    return np.asarray(values.values.to_numpy(zero_copy_only=False)).reshape(len(values), width)


def pose(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(rotation, dtype=np.float64)
    value[:3, 3] = np.asarray(position, dtype=np.float64)
    return value


def nearest_ellipsoid_surface(
    point: np.ndarray, center: np.ndarray, radii: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return signed Euclidean distance, nearest point, and outward normal."""
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    relative = point - center
    implicit = float(np.sum((relative / radii) ** 2))
    theta = float(np.arccos(np.clip(relative[2] / radii[2], -1.0, 1.0)))
    phi = float(np.arctan2(relative[1] / radii[1], relative[0] / radii[0]))

    def surface(angles: np.ndarray) -> np.ndarray:
        theta_value, phi_value = map(float, angles)
        return center + np.asarray(
            [
                radii[0] * np.sin(theta_value) * np.cos(phi_value),
                radii[1] * np.sin(theta_value) * np.sin(phi_value),
                radii[2] * np.cos(theta_value),
            ]
        )

    result = minimize(
        lambda angles: float(np.sum((surface(angles) - point) ** 2)),
        np.asarray([theta, phi]),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-13, "fatol": 1e-18},
    )
    if not result.success:
        raise RuntimeError(f"ellipsoid nearest-point solve failed: {result.message}")
    nearest = surface(result.x)
    gradient = (nearest - center) / (radii**2)
    normal = gradient / max(float(np.linalg.norm(gradient)), 1.0e-15)
    distance = float(np.linalg.norm(point - nearest))
    return (-distance if implicit < 1.0 else distance), nearest, normal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    representative = json.loads(REPRESENTATIVE.read_text(encoding="utf-8"))
    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))
    common = json.loads(COMMON_CONFIG.read_text(encoding="utf-8"))
    proposed = json.loads(PROPOSED_CONFIG.read_text(encoding="utf-8"))
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    measured = json.loads(MEASURED_CONFIG.read_text(encoding="utf-8"))

    if int(representative["representative_source_final_episode"]) != 13:
        raise RuntimeError("the frozen paper representative is no longer episode 13")
    heldout_entry = next(
        row for row in heldout["entries"] if int(row["final_dataset_index"]) == 13
    )
    if Path(heldout_entry["b_trajectory_path"]).resolve() != B_TRAJECTORY.resolve():
        raise RuntimeError("heldout manifest points to a different Proposed-B trajectory")
    if sha256_file(B_TRAJECTORY) != heldout_entry["b_trajectory_sha256"]:
        raise RuntimeError("frozen Proposed-B trajectory hash mismatch")

    with np.load(B_TRAJECTORY, allow_pickle=False) as archive:
        frozen = {key: np.asarray(archive[key]) for key in archive.files}
    event_map = dict(zip(frozen["event_names"].astype(str), frozen["event_frames"].astype(int)))
    frame = int(event_map["LEFT_GRASP"])
    stable_hold = int(event_map["LEFT_STABLE_HOLD"])
    if frame != int(heldout["complete_source_phase_audit"]["13"]["required_intervals"]["left_grasp"][0]) + 3:
        raise RuntimeError("LEFT_GRASP event identity disagrees with the heldout semantic audit")
    z0 = float(frozen["achieved_left_static_whole_hand_position_world"][frame, 2])
    lift_candidates = np.flatnonzero(
        frozen["achieved_left_static_whole_hand_position_world"][:, 2] - z0 >= 0.05
    )
    lift_candidates = lift_candidates[lift_candidates >= stable_hold]
    if not len(lift_candidates):
        raise RuntimeError("frozen Proposed-B prefix never reaches a 50-mm lift")
    lift_end = int(lift_candidates[0])

    source_parquet = Path(heldout_entry["source_parquet_path"]).resolve()
    if sha256_file(source_parquet) != heldout_entry["source_parquet_sha256"]:
        raise RuntimeError("frozen source parquet hash mismatch")
    table = pq.read_table(source_parquet)
    source_state = fixed_list(table["observation.state"], 14).astype(np.float64)
    source_action = fixed_list(table["action"], 14).astype(np.float64)
    if len(source_state) != len(frozen["timestamp"]):
        raise RuntimeError("source/B trajectory frame count mismatch")

    aloha = ALOHAKinematics(common, scene)
    source_fk = aloha.fk(source_state[frame : frame + 1])
    source_wrist_world = pose(
        source_fk["left_wrist_rotation_world"][0],
        source_fk["left_wrist_position_world"][0],
    )
    source_interaction_world = pose(
        source_fk["left_tcp_rotation_world"][0],
        source_fk["left_tcp_position_world"][0],
    )
    stored_interaction_world = frozen["source_left_interaction_frame_position_world"][
        frame
    ].astype(np.float64)
    source_position_recovery_error = float(
        np.linalg.norm(source_interaction_world[:3, 3] - stored_interaction_world)
    )
    if source_position_recovery_error > 1.0e-6:
        raise RuntimeError("source interaction-frame recovery differs from frozen Proposed-B")

    g1 = G1Kinematics(common, scene)
    wrist_to_grasp = np.asarray(
        proposed["resolved"]["wrist_to_grasp_frame"]["left"], dtype=np.float64
    )
    target_wrist_model = pose(
        frozen["target_left_wrist_rotation_model"][frame],
        frozen["target_left_wrist_position_model"][frame],
    )
    target_grasp_model = target_wrist_model @ wrist_to_grasp
    target_grasp_world = pose(
        g1.model_to_world_rotation(target_grasp_model[:3, :3]),
        g1.model_to_world_position(target_grasp_model[:3, 3]),
    )
    target_position_recovery_error = float(
        np.linalg.norm(target_grasp_world[:3, 3] - stored_interaction_world)
    )
    if target_position_recovery_error > 1.0e-6:
        raise RuntimeError("target whole-hand frame does not reproduce frozen target")

    visual_dimensions = np.asarray(
        measured["object"]["visual_dimensions_m"], dtype=np.float64
    )
    if not np.array_equal(visual_dimensions, np.asarray([0.120, 0.090, 0.085])):
        raise RuntimeError("measured geometry is no longer 120 x 90 x 85 mm")
    if float(measured["object"]["mass_kg"]) != 0.020:
        raise RuntimeError("measured doll mass is no longer 20 g")
    task_origin = np.asarray(scene["task_frame"]["origin_world_xyz_m"], dtype=np.float64)
    object_support_frame_world = pose(
        np.eye(3),
        np.asarray([*scene["doll"]["center_world_xy_m"], task_origin[2]], dtype=np.float64),
    )
    object_center_world = object_support_frame_world[:3, 3] + np.asarray(
        [0.0, 0.0, visual_dimensions[2] / 2.0]
    )
    radii = visual_dimensions / 2.0
    interaction_surface = nearest_ellipsoid_surface(
        source_interaction_world[:3, 3], object_center_world, radii
    )

    geometry = proposed["resolved"]["geometry_at_grasp"]["left"]
    target_pad_world: dict[str, np.ndarray] = {}
    target_pad_grasp: dict[str, np.ndarray] = {}
    digit_rows: dict[str, Any] = {}
    inverse_wrist_to_grasp = np.linalg.inv(wrist_to_grasp)
    for digit in ("thumb", "index", "middle"):
        pad_wrist = np.asarray(geometry["pad_centers_wrist_m"][digit], dtype=np.float64)
        pad_grasp = (inverse_wrist_to_grasp @ np.r_[pad_wrist, 1.0])[:3]
        pad_world = target_grasp_world[:3, 3] + target_grasp_world[:3, :3] @ pad_grasp
        target_pad_grasp[digit] = pad_grasp
        target_pad_world[digit] = pad_world
        signed, nearest, normal = nearest_ellipsoid_surface(
            pad_world, object_center_world, radii
        )
        digit_rows[digit] = {
            "canonical_target_pad_position_world_m": pad_world,
            "canonical_target_pad_position_in_grasp_frame_m": pad_grasp,
            "target_pad_to_surface_signed_distance_m": signed,
            "nearest_surface_position_world_m": nearest,
            "surface_outward_normal_world": normal,
        }

    g1.assign(
        frozen["g1_arm_qpos"][frame],
        frozen["left_dex3_qpos"][frame],
        frozen["right_dex3_qpos"][frame],
    )
    for digit in ("thumb", "index", "middle"):
        actual_model, actual_normal_model = g1.contact_pose("left", digit)
        actual_world = g1.model_to_world_position(actual_model)
        actual_normal_world = g1.model_to_world_rotation(actual_normal_model[:, None])[:, 0]
        signed, nearest, normal = nearest_ellipsoid_surface(
            actual_world, object_center_world, radii
        )
        digit_rows[digit].update(
            {
                "frozen_q_pad_position_world_m": actual_world,
                "frozen_q_pad_normal_world": actual_normal_world,
                "frozen_q_pad_to_surface_signed_distance_m": signed,
                "frozen_q_nearest_surface_position_world_m": nearest,
                "frozen_q_surface_outward_normal_world": normal,
                "whole_hand_pad_realization_error_m": float(
                    np.linalg.norm(actual_world - target_pad_world[digit])
                ),
            }
        )

    replay_names = frozen["replay_joint_names"].astype(str)
    replay = frozen["replay_named_joint_qpos"].astype(np.float64)
    hold_frames = max(
        1,
        round(
            float(measured["timing"]["gravity_retention_s"])
            * float(measured["timing"]["control_fps_hz"])
        ),
    )
    elevated_frames = max(
        1,
        round(
            float(measured["timing"]["elevated_hold_s"])
            * float(measured["timing"]["control_fps_hz"])
        ),
    )
    source_frames = list(range(frame, stable_hold + 1))
    labels = ["B_LEFT_GRASP_TO_HOLD"] * len(source_frames)
    source_frames.extend([stable_hold] * hold_frames)
    labels.extend(["HOLD"] * hold_frames)
    source_frames.extend(range(stable_hold + 1, lift_end + 1))
    labels.extend(["FROZEN_B_LIFT_PREFIX"] * (lift_end - stable_hold))
    source_frames.extend([lift_end] * elevated_frames)
    labels.extend(["ELEVATED_HOLD"] * elevated_frames)
    source_frames_array = np.asarray(source_frames, dtype=np.int64)
    commands = replay[source_frames_array]

    target_pad_trajectory = np.empty((len(source_frames_array), 3, 3), dtype=np.float64)
    target_surface_point = np.empty_like(target_pad_trajectory)
    target_surface_normal = np.empty_like(target_pad_trajectory)
    frozen_q_pad_trajectory = np.empty_like(target_pad_trajectory)
    frozen_q_pad_normal = np.empty_like(target_pad_trajectory)
    for output_frame, source_frame in enumerate(source_frames_array):
        wrist = pose(
            frozen["target_left_wrist_rotation_model"][source_frame],
            frozen["target_left_wrist_position_model"][source_frame],
        )
        grasp_model = wrist @ wrist_to_grasp
        grasp_world = pose(
            g1.model_to_world_rotation(grasp_model[:3, :3]),
            g1.model_to_world_position(grasp_model[:3, 3]),
        )
        g1.assign(
            frozen["g1_arm_qpos"][source_frame],
            frozen["left_dex3_qpos"][source_frame],
            frozen["right_dex3_qpos"][source_frame],
        )
        for digit_index, digit in enumerate(("thumb", "index", "middle")):
            target = grasp_world[:3, 3] + grasp_world[:3, :3] @ target_pad_grasp[digit]
            target_pad_trajectory[output_frame, digit_index] = target
            _, surface_point, surface_normal = nearest_ellipsoid_surface(
                target, object_center_world, radii
            )
            target_surface_point[output_frame, digit_index] = surface_point
            target_surface_normal[output_frame, digit_index] = surface_normal
            actual_model, normal_model = g1.contact_pose("left", digit)
            frozen_q_pad_trajectory[output_frame, digit_index] = g1.model_to_world_position(
                actual_model
            )
            frozen_q_pad_normal[output_frame, digit_index] = g1.model_to_world_rotation(
                normal_model[:, None]
            )[:, 0]

    command_path = output / "frozen_b_left_grasp_hold_lift_command.npz"
    atomic_npz(
        command_path,
        commanded_q_rad=commands.astype(np.float32),
        joint_names=replay_names,
        source_frame_index=source_frames_array,
        stage=np.asarray(labels, dtype="U32"),
        control_fps_hz=np.asarray(measured["timing"]["control_fps_hz"]),
        target_pad_position_world_m=target_pad_trajectory.astype(np.float32),
        target_nearest_surface_position_world_m=target_surface_point.astype(np.float32),
        target_surface_normal_world=target_surface_normal.astype(np.float32),
        frozen_q_pad_position_world_m=frozen_q_pad_trajectory.astype(np.float32),
        frozen_q_pad_normal_world=frozen_q_pad_normal.astype(np.float32),
        digit_names=np.asarray(["thumb", "index", "middle"]),
        object_support_frame_world=object_support_frame_world,
        object_center_world_m=object_center_world,
        object_dimensions_m=visual_dimensions,
        object_mass_kg=np.asarray(0.020),
        policy_used=np.asarray(False),
        pose_optimization_used=np.asarray(False),
    )

    recovered = {
        "schema_version": "proposed_b_measured_doll_left_grasp_recovery_v1",
        "status": "PASS",
        "diagnostic_only": True,
        "retargeting_recomputed": False,
        "retargeting_modified": False,
        "generic_grasp_pose_used": False,
        "pose_search_or_tuning_used": False,
        "representative_selection": {
            "source_final_episode": 13,
            "source_recording_id": heldout_entry["original_source_recording_id"],
            "rule": representative["representative_selection_frozen_before_policy_results"],
            "left_grasp_frame": frame,
            "left_stable_hold_frame": stable_hold,
            "first_frozen_50mm_lift_frame": lift_end,
            "timestamp_s": float(frozen["timestamp"][frame]),
        },
        "source_aloha": {
            "motion_key": str(frozen["source_motion_key"].item()),
            "left_state_7d": source_state[frame, :7],
            "left_action_7d": source_action[frame, :7],
            "left_gripper_state_aperture_m": float(source_state[frame, 6]),
            "left_gripper_action_aperture_m": float(source_action[frame, 6]),
            "left_wrist_pose_world": source_wrist_world,
            "left_interaction_frame_pose_world": source_interaction_world,
            "interaction_frame_definition": common["resolved"]["aloha_link6_to_task_tcp"],
            "stored_position_recovery_error_m": source_position_recovery_error,
        },
        "proposed_b": {
            "target_whole_hand_grasp_frame_pose_model": target_grasp_model,
            "target_whole_hand_grasp_frame_pose_world": target_grasp_world,
            "target_g1_wrist_pose_model": target_wrist_model,
            "target_g1_wrist_position_world_m": g1.model_to_world_position(
                target_wrist_model[:3, 3]
            ),
            "target_g1_wrist_rotation_world": g1.model_to_world_rotation(
                target_wrist_model[:3, :3]
            ),
            "target_frame_recovery_error_m": target_position_recovery_error,
            "left_dex3_joint_names": frozen["left_dex3_joint_names"].astype(str),
            "left_dex3_target_7d_rad": frozen["left_dex3_qpos"][frame],
            "left_dex3_stable_hold_7d_rad": frozen["left_dex3_qpos"][stable_hold],
            "frame_phase": str(frozen["left_hand_phase"][frame]),
            "stable_hold_phase": str(frozen["left_hand_phase"][stable_hold]),
            "whole_hand_frame_definition": proposed["hand_primitive_derivation"][
                "grasp_frame_center"
            ],
            "orientation_policy": proposed["orientation_semantics"]["policy"],
        },
        "measured_object_registration": {
            "method": (
                "preserve the frozen scene task axes and doll footprint center; "
                "anchor local object z=0 to the frozen table support plane and place "
                "the measured geometric center at +height/2"
            ),
            "object_support_frame_world": object_support_frame_world,
            "object_center_world_m": object_center_world,
            "local_axis_mapping": measured["axis_mapping"],
            "dimensions_m": visual_dimensions,
            "mass_kg": 0.020,
            "collision_scale_percent": 100,
            "source_interaction_origin_signed_distance_to_surface_m": interaction_surface[0],
            "source_interaction_nearest_surface_world_m": interaction_surface[1],
            "source_interaction_surface_normal_world": interaction_surface[2],
            "episode_specific_translation_used": False,
            "grasp_target_alignment_used": False,
        },
        "digits": digit_rows,
        "command": {
            "path": command_path,
            "sha256": sha256_file(command_path),
            "joint_order": replay_names,
            "source_frames": source_frames_array,
            "stages": np.asarray(labels),
            "hold_duration_s": hold_frames
            / float(measured["timing"]["control_fps_hz"]),
            "elevated_hold_duration_s": elevated_frames
            / float(measured["timing"]["control_fps_hz"]),
            "source_prefix_modified": False,
            "only_exact_frozen_rows_repeated_for_hold": True,
        },
        "provenance": {
            str(path): sha256_file(path)
            for path in (
                REPRESENTATIVE,
                HELDOUT,
                B_TRAJECTORY,
                source_parquet,
                COMMON_CONFIG,
                PROPOSED_CONFIG,
                SCENE,
                MEASURED_CONFIG,
            )
        },
    }
    recovered_path = output / "FROZEN_TARGET_RECOVERY.json"
    atomic_json(recovered_path, recovered)
    print(json.dumps(recovered, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
