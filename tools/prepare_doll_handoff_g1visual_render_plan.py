#!/usr/bin/env python3
"""Prepare immutable, state-aligned plans for G1-visual Dataset-A/B rendering.

This utility never invokes retargeting.  It proves the frozen Dataset-B action
and lag-1 state semantics against each frozen trajectory, then reconstructs a
continuous visualization-only doll pose from the already-recorded ownership
states and realized whole-hand frames.
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
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from deployment_camera_config import camera_manifest_record, load_camera_config  # noqa: E402
from doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
TRAJECTORIES = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
SCENE_RECALIBRATION = ROOT / "outputs/doll_handoff_retargeting/scene_recalibration.json"
NEW_OBJECT_ESTIMATES = (
    ROOT
    / "outputs/doll_handoff_dataset_b_final/new_episode_conversion/source_image_object_estimates.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_g1visual/dataset_render_full/render_plans"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(arrays: list[np.ndarray]) -> str:
    """Canonical logical hash with shape/dtype boundaries for a list of arrays."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def fixed_size_list_to_numpy(table: Any, key: str, width: int) -> np.ndarray:
    column = table[key].combine_chunks()
    values = np.asarray(column.values.to_numpy(zero_copy_only=False))
    result = values.reshape(len(column), width)
    if result.dtype != np.float32:
        raise RuntimeError(f"{key} is not float32: {result.dtype}")
    return np.ascontiguousarray(result)


def homogeneous(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def world_hand_transforms(data: Any, side: str, g1: G1Kinematics) -> np.ndarray:
    """Recover realized whole-hand FK without changing the frozen labels.

    Proposed B archived these transforms as diagnostics.  Fair Baseline A did
    not, so the exact same named 28-D replay is forwarded through the common
    frozen G1 model.  This is rendering metadata only; no Cartesian correction
    or retargeting occurs here.
    """

    position_key = f"achieved_{side}_static_whole_hand_position_world"
    rotation_key = f"achieved_{side}_static_whole_hand_orientation_model"
    if position_key in data.files and rotation_key in data.files:
        positions = np.asarray(data[position_key], dtype=np.float64)
        model_rotations = np.asarray(data[rotation_key], dtype=np.float64)
        world_rotations = g1.model_to_world_rotation(model_rotations)
    else:
        replay = np.asarray(data["replay_named_joint_qpos"], dtype=np.float64)
        positions = np.empty((len(replay), 3), dtype=np.float64)
        world_rotations = np.empty((len(replay), 3, 3), dtype=np.float64)
        for frame, q in enumerate(replay):
            g1.assign(q[:14], q[14:21], q[21:28])
            pose = g1.whole_hand_grasp_pose(side)
            positions[frame] = g1.model_to_world_position(pose[:3, 3])
            world_rotations[frame] = g1.model_to_world_rotation(pose[:3, :3])
    result = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(positions), axis=0)
    result[:, :3, :3] = world_rotations
    result[:, :3, 3] = positions
    return result


def reconstruct_doll_pose(
    ownership: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    initial_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return continuous world positions and wxyz quaternions.

    The constant object-to-hand transform is established at the first LEFT_OWNED
    frame.  A new right-hand-local transform is established at the first
    RIGHT_OWNED frame from the *current* object pose, making ownership transfer
    continuous by construction.  No per-frame correction is performed.
    """

    count = len(ownership)
    positions = np.repeat(np.asarray(initial_position, dtype=np.float64)[None, :], count, axis=0)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], count, axis=0)
    left_indices = np.flatnonzero(
        np.isin(ownership, ["LEFT_OWNED", "HANDOFF_APPROACH", "DUAL_CONTACT"])
    )
    right_indices = np.flatnonzero(np.isin(ownership, ["RIGHT_OWNED", "RIGHT_TRANSPORT"]))
    if not len(left_indices) or not len(right_indices):
        raise RuntimeError("ownership sequence lacks a complete left-to-right transfer")
    left_anchor = int(left_indices[0])
    right_anchor = int(right_indices[0])
    if right_anchor <= left_anchor:
        raise RuntimeError("right ownership does not follow left ownership")

    initial_object = homogeneous(np.eye(3), initial_position)
    object_in_left = np.linalg.inv(left_hand[left_anchor]) @ initial_object
    last_object = initial_object.copy()
    for frame in range(left_anchor, right_anchor):
        if ownership[frame] in {"LEFT_OWNED", "HANDOFF_APPROACH", "DUAL_CONTACT"}:
            last_object = left_hand[frame] @ object_in_left
            positions[frame] = last_object[:3, 3]
            rotations[frame] = last_object[:3, :3]

    # This calibration is performed exactly once per episode at ownership transfer.
    # It preserves the current object pose and therefore cannot teleport the doll.
    object_in_right = np.linalg.inv(right_hand[right_anchor]) @ last_object
    transfer_position_residual = float(
        np.linalg.norm((right_hand[right_anchor] @ object_in_right)[:3, 3] - last_object[:3, 3])
    )
    transfer_rotation_residual = float(
        Rotation.from_matrix(
            (right_hand[right_anchor] @ object_in_right)[:3, :3] @ last_object[:3, :3].T
        ).magnitude()
    )
    for frame in range(right_anchor, count):
        if ownership[frame] in {"RIGHT_OWNED", "RIGHT_TRANSPORT"}:
            last_object = right_hand[frame] @ object_in_right
        # RELEASED holds the last intended hand-carried pose.  This is deterministic
        # visualization, not a physics/release-success claim.
        positions[frame] = last_object[:3, 3]
        rotations[frame] = last_object[:3, :3]

    quaternions_xyzw = Rotation.from_matrix(rotations).as_quat()
    quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return positions, quaternions_wxyz, {
        "left_grasp_anchor_action_index": left_anchor,
        "right_ownership_anchor_action_index": right_anchor,
        "transfer_position_discontinuity_m": transfer_position_residual,
        "transfer_rotation_discontinuity_rad": transfer_rotation_residual,
        "maximum_frame_position_step_m": float(steps.max(initial=0.0)),
        "mean_frame_position_step_m": float(steps.mean()) if len(steps) else 0.0,
        "method": "KINEMATIC_OWNERSHIP_OBJECT_RECONSTRUCTION",
    }


def object_estimates_by_source() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    positions: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}
    historical = read_json(SCENE_RECALIBRATION)
    for row in historical["per_episode"]:
        source = str(row["source_name"])
        positions[source] = np.asarray(row["doll_initial_center_task_xy_m"], dtype=np.float64)
        provenance[source] = str(SCENE_RECALIBRATION)
    replacements = read_json(NEW_OBJECT_ESTIMATES)
    for row in replacements.values():
        observations = row.get("observations", [])
        if not observations:
            continue
        image = Path(observations[0]["image"])
        source = next((part for part in image.parts if part.startswith("GoPark_")), None)
        if source is None:
            raise RuntimeError(f"cannot recover source name from {image}")
        positions[source] = np.asarray(row["doll_initial_center_task_xy_m"], dtype=np.float64)
        provenance[source] = str(NEW_OBJECT_ESTIMATES)
    return positions, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-variant", choices=("A", "B"), required=True)
    parser.add_argument("--source-dataset", type=Path, default=DATASET)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--trajectories", type=Path, default=TRAJECTORIES)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    dataset = args.source_dataset.resolve()
    source_manifest_path = args.source_manifest.resolve()
    trajectories = args.trajectories.resolve()
    camera = load_camera_config(
        args.camera_config,
        purpose=f"Policy {args.dataset_variant} final-view render-plan preparation",
        allow_pending=True,
    )

    source_manifest = read_json(source_manifest_path)
    layout = read_json(SCENE_LAYOUT)
    common = read_json(ROOT / "configs/doll_handoff_retargeting/common_config.template.json")
    g1 = G1Kinematics(common, layout)
    episodes_table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    data_table = pq.read_table(dataset / "data/chunk-000/file-000.parquet")
    states_all = fixed_size_list_to_numpy(data_table, "observation.state", 28)
    actions_all = fixed_size_list_to_numpy(data_table, "action", 28)
    timestamps_all = np.asarray(data_table["timestamp"].combine_chunks().to_numpy(), dtype=np.float32)
    episode_indices_all = np.asarray(
        data_table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    frame_indices_all = np.asarray(
        data_table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    if len(source_manifest["episodes"]) != 50 or episodes_table.num_rows != 50:
        raise RuntimeError("authoritative source/dataset episode count is not 50")
    expected_total_frames = sum(int(row["source_frame_count"]) for row in source_manifest["episodes"])
    if len(states_all) != expected_total_frames:
        raise RuntimeError(
            f"source manifest/dataset frame count differs: {expected_total_frames} != {len(states_all)}"
        )

    doll_xy, doll_provenance = object_estimates_by_source()
    table_z = float(layout["table"]["surface_height_m"])
    doll_z = (
        table_z
        + float(layout["doll"]["diameter_m"]) / 2.0
        + float(layout["doll"]["initial_table_clearance_m"])
    )
    episode_reports: list[dict[str, Any]] = []
    states_for_hash: list[np.ndarray] = []
    actions_for_hash: list[np.ndarray] = []

    for episode in range(50):
        source = source_manifest["episodes"][episode]
        metadata = episodes_table.slice(episode, 1).to_pylist()[0]
        start = int(metadata["dataset_from_index"])
        end = int(metadata["dataset_to_index"])
        length = int(metadata["length"])
        if end - start != length or int(source["source_frame_count"]) != length:
            raise RuntimeError(f"episode {episode}: frame-count identity failed")
        states = states_all[start:end]
        actions = actions_all[start:end]
        timestamps = timestamps_all[start:end]
        episode_indices = episode_indices_all[start:end]
        frame_indices = frame_indices_all[start:end]
        if not np.array_equal(episode_indices, np.full(length, episode, dtype=np.int64)):
            raise RuntimeError(f"episode {episode}: dataset index crossing")
        if not np.array_equal(frame_indices, np.arange(length, dtype=np.int64)):
            raise RuntimeError(f"episode {episode}: frame indices are not exact")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise RuntimeError(f"episode {episode}: nonfinite state/action")

        candidates = (
            trajectories / f"episode_{episode:06d}.npz",
            trajectories / f"{source['stable_episode_id']}.npz",
        )
        trajectory_path = next((path for path in candidates if path.is_file()), candidates[-1])
        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"episode {episode}: no frozen trajectory at either {candidates[0]} or {candidates[1]}"
            )
        trajectory = np.load(trajectory_path, allow_pickle=True)
        names = [str(value) for value in trajectory["replay_joint_names"]]
        replay = np.asarray(trajectory["replay_named_joint_qpos"], dtype=np.float32)
        if replay.shape != (length, 28):
            raise RuntimeError(f"episode {episode}: frozen trajectory shape mismatch")
        if not np.array_equal(actions, replay):
            maximum = float(np.max(np.abs(actions.astype(np.float64) - replay.astype(np.float64))))
            raise RuntimeError(
                f"episode {episode}: Dataset {args.dataset_variant} actions differ from its frozen trajectory; max={maximum}"
            )
        if not np.array_equal(states[0], actions[0]) or not np.array_equal(states[1:], actions[:-1]):
            raise RuntimeError(f"episode {episode}: lag-1 state semantics are not exact")
        expected_timestamps = np.arange(length, dtype=np.float32) / np.float32(30.0)
        timestamp_error = float(np.max(np.abs(timestamps - expected_timestamps)))
        if timestamp_error > 2.0e-6:
            raise RuntimeError(f"episode {episode}: timestamp alignment changed: {timestamp_error}")

        source_name = str(source["raw_directory"])
        if source_name not in doll_xy:
            raise RuntimeError(f"episode {episode}: no authoritative initial doll estimate for {source_name}")
        initial_doll = np.asarray([doll_xy[source_name][0], doll_xy[source_name][1], doll_z])
        left_hand = world_hand_transforms(trajectory, "left", g1)
        right_hand = world_hand_transforms(trajectory, "right", g1)
        ownership_action = np.asarray(trajectory["ownership_state"]).astype("U32")
        # State t corresponds exactly to trajectory/action max(t-1, 0).  All visual
        # annotations are shifted by the same mapping as the rendered robot state.
        pose_action_indices = np.maximum(np.arange(length, dtype=np.int64) - 1, 0)
        ownership_render = ownership_action[pose_action_indices]
        object_position_action, object_quaternion_action, reconstruction = reconstruct_doll_pose(
            ownership_action, left_hand, right_hand, initial_doll
        )
        object_positions = object_position_action[pose_action_indices]
        object_quaternions = object_quaternion_action[pose_action_indices]
        left_phase = np.asarray(trajectory["left_hand_phase"]).astype("U16")[pose_action_indices]
        right_phase = np.asarray(trajectory["right_hand_phase"]).astype("U16")[pose_action_indices]
        event_names = np.asarray(trajectory["event_names"]).astype("U32")
        event_action_frames = np.asarray(trajectory["event_frames"], dtype=np.int64)
        event_render_frames = np.minimum(event_action_frames + 1, length - 1)

        plan_path = output / f"episode_{episode:06d}.npz"
        atomic_npz(
            plan_path,
            observation_state=states,
            frozen_action=actions,
            timestamp=timestamps,
            frame_index=frame_indices,
            pose_action_index=pose_action_indices,
            policy_joint_names=np.asarray(names, dtype="U64"),
            ownership_state=ownership_render,
            left_hand_phase=left_phase,
            right_hand_phase=right_phase,
            doll_position_world=object_positions.astype(np.float32),
            doll_orientation_world_wxyz=object_quaternions.astype(np.float32),
            event_names=event_names,
            event_action_frames=event_action_frames,
            event_render_frames=event_render_frames,
            source_raw_episode=np.asarray(source_name),
            object_visualization_method=np.asarray("KINEMATIC_OWNERSHIP_OBJECT_RECONSTRUCTION"),
        )
        report = {
            "episode_index": episode,
            "source_raw_episode": source_name,
            "length": length,
            "fps": 30.0,
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "frozen_trajectory": str(trajectory_path),
            "frozen_trajectory_sha256": sha256_file(trajectory_path),
            "state_action_alignment": {
                "frame_0": "state[0] == action[0]",
                "frame_t_gt_0": "state[t] == action[t-1]",
                "rendered_robot_q": "observation.state[t]",
                "render_pose_action_index": "max(t-1, 0)",
                "state_label": "RETARGETED_G1_STATE_SURROGATE",
                "exact": True,
                "maximum_timestamp_error_s": timestamp_error,
            },
            "initial_doll_center_world_xyz_m": initial_doll,
            "initial_doll_estimate_provenance": doll_provenance[source_name],
            "object_reconstruction": reconstruction,
        }
        episode_reports.append(report)
        states_for_hash.append(states)
        actions_for_hash.append(actions)

    manifest = {
        "schema_version": "doll_handoff_g1visual_render_plan_v2",
        "status": (
            "READY_FOR_ISAAC_KINEMATIC_RENDERING"
            if camera.resolved
            else "PREPARED_BLOCKED_PENDING_FINAL_CAMERA"
        ),
        "dataset_variant": args.dataset_variant,
        "source_dataset": str(dataset),
        "source_data_parquet": str(dataset / "data/chunk-000/file-000.parquet"),
        "source_data_parquet_sha256": sha256_file(dataset / "data/chunk-000/file-000.parquet"),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "camera": camera_manifest_record(camera),
        "episode_count": 50,
        "total_frames": len(states_all),
        "fps": 30.0,
        "policy_dimension": 28,
        "state_semantics": {
            "label": "RETARGETED_G1_STATE_SURROGATE",
            "frame_0": "state[0] = q_target[0]",
            "frame_t_gt_0": "state[t] = q_target[t-1]",
            "rendered_robot_q": "observation.state[t]",
        },
        "action_semantics": "frozen absolute G1/Dex3 q_target[t]",
        "dataset_state_logical_sha256": array_hash(states_for_hash),
        "dataset_action_logical_sha256": array_hash(actions_for_hash),
        "action_equals_frozen_trajectory_float32": True,
        "dataset_labels_modified": False,
        "object_visualization": {
            "method": "KINEMATIC_OWNERSHIP_OBJECT_RECONSTRUCTION",
            "changes_g1_labels": False,
            "pre_grasp": "episode source-registered initial position",
            "left_owned_through_dual_contact": "constant object-to-left-whole-hand transform",
            "right_owned_through_transport": "constant object-to-right-whole-hand transform initialized continuously at transfer",
            "released": "hold last right-carried pose; no task-success claim",
            "bin": "fixed frozen scene pose",
        },
        "episodes": episode_reports,
    }
    atomic_json(output / "render_plan_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "episode_count": 50,
        "total_frames": len(states_all),
        "state_sha256": manifest["dataset_state_logical_sha256"],
        "action_sha256": manifest["dataset_action_logical_sha256"],
        "manifest": str(output / "render_plan_manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
