"""Map source TCP-relative opposing contacts into the frozen G1 task tool."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from aloha_g1_dataset_v1.core import raw_wrist_pose


def map_source_contacts_to_frozen_g1_tool(
    source: Mapping[str, Any],
    runtime: Any,
    arm_q: np.ndarray,
    right_hand_q: np.ndarray,
    v1_config: Mapping[str, Any],
    *,
    side: str = "left",
) -> dict[str, Any]:
    """Condition contact targets on the achieved, frozen wrist pose.

    The v1 pose trajectory has already applied its source/tool orientation
    mapping.  Source contact offsets are therefore expressed once in the
    semantic task-tool basis (+x approach, +y thumb-to-index closing, +z
    lateral), rather than applying the v1 orientation calibration twice.
    """
    if source["representation_mode"] != "SOURCE_TCP_RELATIVE_CONTACT_GEOMETRY_FALLBACK":
        raise ValueError("unsupported or mislabeled source interaction representation")
    if side != str(source["side"]):
        raise ValueError("source side and target side differ")
    left_neutral = runtime.open_hand_q["left"]
    runtime.assign(
        np.asarray(arm_q, dtype=np.float64),
        left_neutral,
        np.asarray(right_hand_q, dtype=np.float64),
    )
    wrist = raw_wrist_pose(runtime, side)
    wrist_to_tool = np.asarray(
        v1_config["target_frames"][f"{side}_wrist_to_physical_pinch"], dtype=np.float64
    )
    task_tool = wrist @ wrist_to_tool

    basis_source = np.column_stack(
        (
            np.asarray(source["approach_axis_tcp"], dtype=np.float64),
            np.asarray(source["closing_axis_tcp"], dtype=np.float64),
            np.asarray(source["lateral_axis_tcp"], dtype=np.float64),
        )
    )
    center_source = np.asarray(source["grasp_center_tcp_m"], dtype=np.float64)
    scale = float(v1_config["workspace_mapping"]["uniform_scale"])

    def map_point(key: str) -> np.ndarray:
        source_offset = np.asarray(source[key], dtype=np.float64) - center_source
        semantic_coordinates = basis_source.T @ source_offset
        return task_tool[:3, 3] + task_tool[:3, :3] @ (scale * semantic_coordinates)

    thumb = map_point("contact_A_tcp_m")
    index = map_point("contact_B_tcp_m")
    closing = task_tool[:3, 1].copy()
    result = {
        "schema_version": "aloha_g1_contact_mapping_v2",
        "representation_mode": source["representation_mode"],
        "side": side,
        "frozen_wrist_conditioning": True,
        "frozen_wrist_pose": wrist,
        "frozen_task_tool_pose": task_tool,
        "source_semantic_basis": basis_source,
        "source_tcp_semantic_to_target_task_tool_semantic": np.eye(4),
        "workspace_contact_offset_scale": scale,
        "workspace_contact_offset_scale_provenance": "v1 immutable global workspace uniform_scale",
        "thumb_target_m": thumb,
        "index_target_m": index,
        "thumb_target_normal": closing,
        "index_target_normal": -closing,
        "mapped_contact_width_m": float(np.linalg.norm(index - thumb)),
        "source_contact_width_m": float(source["gripper_width_m"]),
        "target_task_tool_axes": {
            "approach_x": task_tool[:3, 0],
            "closing_y_thumb_to_index": task_tool[:3, 1],
            "lateral_z": task_tool[:3, 2],
        },
        "per_frame_cartesian_correction": False,
        "episode_specific_offset": False,
        "v1_pose_orientation_mapping_reapplied_to_contact_offsets": False,
    }
    if not all(
        np.isfinite(np.asarray(result[key])).all()
        for key in ("thumb_target_m", "index_target_m", "frozen_task_tool_pose")
    ):
        raise RuntimeError("non-finite mapped contact target")
    return result
