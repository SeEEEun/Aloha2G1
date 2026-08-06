#!/usr/bin/env python3
"""Render the user-approved +7-frame source replay and relation blocker.

Actual Stationary ALOHA MuJoCo geometry is rendered.  Objects are kinematic
diagnostic references; no physics, G1, Dex3, DDS, publisher, or hardware path.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path.insert(0, str(ROOT / "tools"))
import build_source_fk_parity_v11 as base  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
ALIGNMENT = OUT / "source_action_latency_alignment.npz"
PHONE = OUT / "source_phone_pose_trajectory_latency_aligned.npz"
ACCESSORY = OUT / "source_accessory_pose_trajectory_latency_aligned.npz"
RELATIONS = OUT / "source_hand_object_relations_recomputed.json"
FPS = 7.5
KEYS = [0, 176, 200, 223, 326, 329, 341, 380, 530, 586, 646, 702, 989]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_name(frame: int, ordered: list[dict[str, Any]]) -> str:
    result = "pre_task"
    for row in ordered:
        if int(row["frame"]) <= frame:
            result = str(row["event"])
        else:
            break
    return result


def overlay(
    image: np.ndarray,
    title: str,
    observed_frame: int,
    action_sample: int | None,
    event: str,
    tcp_source: np.ndarray | None,
    error_text: str,
) -> np.ndarray:
    value = image.copy()
    cv2.rectangle(value, (0, 0), (value.shape[1], 104), (7, 7, 7), -1)
    sample = "video/state time" if action_sample is None else f"optimized/raw action sample {action_sample:03d}"
    lines = [
        f"{title} | observed frame {observed_frame:03d}/989 | {event}",
        f"{sample} | approved lag=+7f (0.233333s) | timeline NOT shifted",
    ]
    if tcp_source is not None:
        lines.append(
            f"L [{tcp_source[0,0,3]:+.3f},{tcp_source[0,1,3]:+.3f},{tcp_source[0,2,3]:+.3f}] "
            f"R [{tcp_source[1,0,3]:+.3f},{tcp_source[1,1,3]:+.3f},{tcp_source[1,2,3]:+.3f}]"
        )
    else:
        lines.append("recorded cam_high image; timestamps and event frames unchanged")
    lines.append(error_text)
    colors = [(245, 245, 245), (70, 220, 255), (100, 255, 120), (90, 90, 255)]
    for index, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(value, line, (9, 20 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    return value


def add_error_arrow(renderer: mujoco.Renderer, root_inverse: np.ndarray, tcp_source: np.ndarray, object_source: np.ndarray) -> None:
    tcp_model = (root_inverse @ tcp_source)[:3, 3]
    object_model = (root_inverse @ object_source)[:3, 3]
    base.add_connector(
        renderer.scene,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.004,
        tcp_model,
        object_model,
        np.array([1.0, 0.03, 0.03, 1.0]),
    )


def render_robot(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    tcp_model: np.ndarray,
    tcp_source: np.ndarray,
    phone_source: np.ndarray,
    accessory_source: np.ndarray,
    pad_source: np.ndarray,
    root_inverse: np.ndarray,
    title: str,
    observed_frame: int,
    action_sample: int | None,
    event: str,
    error_text: str,
    error_arrow: bool,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="cam_high")
    base.add_reference_scene(renderer, root_inverse, phone_source, accessory_source, pad_source, tcp_model)
    if error_arrow:
        add_error_arrow(renderer, root_inverse, tcp_source[1], accessory_source)
    rgb = renderer.render().copy()
    rgb = np.clip(rgb.astype(np.float32) * 1.55 + 10.0, 0.0, 255.0).astype(np.uint8)
    return overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), title, observed_frame, action_sample, event, tcp_source, error_text)


def main() -> int:
    for path in (ALIGNMENT, PHONE, ACCESSORY, RELATIONS):
        if not path.is_file():
            raise FileNotFoundError(path)
    relation = json.loads(RELATIONS.read_text(encoding="utf-8"))
    if relation["status"] != "BLOCKED_SOURCE_HAND_OBJECT_RELATION":
        raise RuntimeError(f"This renderer is scoped to the current relation blocker; got {relation['status']}")
    with np.load(ALIGNMENT, allow_pickle=False) as archive:
        lookup = archive["action_sample_index_for_observed_frame"].copy()
    with np.load(PHONE, allow_pickle=False) as archive:
        phone = archive["T_source_scene_from_phone"].copy()
    with np.load(ACCESSORY, allow_pickle=False) as archive:
        accessory = archive["T_source_scene_from_accessory"].copy()
    source_frames = json.loads(base.SOURCE_FRAMES.read_text(encoding="utf-8"))
    pad = np.asarray(source_frames["T_source_scene_from_charger_pad"], dtype=np.float64)

    inputs = base.load_inputs()
    model = mujoco.MjModel.from_xml_path(str(base.MODEL_XML))
    mapping, _ = base.build_name_mapping(model, inputs["channel_names"])
    root = base.source_root_transform()
    root_inverse = base.inverse_transform(root)
    arrays = {
        "observation_state": inputs["observation_state"],
        "raw_action": inputs["raw_action"],
        "optimized_action": inputs["optimized_action"],
    }
    fks = {label: base.fk_trajectory(model, value, mapping, root) for label, value in arrays.items()}
    ordered, _ = base.events()

    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = (0.72, 0.72, 0.72)
    model.vis.headlight.diffuse[:] = (1.0, 1.0, 1.0)
    model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)

    optimized_path = OUT / "source_optimized_action_latency_aligned_replay.mp4"
    optimized_raw = OUT / ".source_optimized_action_latency_aligned_replay.raw.mp4"
    four_path = OUT / "aloha_source_latency_aligned_relation_4panel.mp4"
    four_raw = OUT / ".aloha_source_latency_aligned_relation_4panel.raw.mp4"
    optimized_writer = base.open_writer(optimized_raw, 640, 480)
    four_writer = base.open_writer(four_raw, 1920, 360)
    key_images: dict[int, np.ndarray] = {}
    surface = relation["semantic_grasp_proximity_gate"]["distances"]
    try:
        for observed in range(990):
            sample = int(lookup[observed])
            event = event_name(observed, ordered)
            if observed in (326, 329, 341):
                gap = surface["right_accessory_grasp_m"] if observed < 341 else surface["right_accessory_removed_m"]
                error_text = f"BLOCKER: right gripper-to-accessory surface gap={gap*1000:.1f}mm > 20mm gate"
                arrow = True
            elif observed == 176:
                error_text = f"left phone surface gap={surface['left_phone_grasp_m']*1000:.1f}mm (pass)"
                arrow = False
            elif observed == 530:
                error_text = f"left phone carrier gap={surface['left_phone_at_charger_m']*1000:.1f}mm; source relation diagnostic"
                arrow = False
            else:
                error_text = "FAILED SOURCE RELATION DIAGNOSTIC | NOT APPROVED FOR G1 TARGET GENERATION"
                arrow = 223 <= observed <= 341

            state_image = render_robot(
                renderer, model, data,
                fks["observation_state"]["qpos"][observed],
                fks["observation_state"]["model_tcp"][observed],
                fks["observation_state"]["source_tcp"][observed],
                phone[observed], accessory[observed], pad, root_inverse,
                "OBSERVATION.STATE REPLAY", observed, None, event, error_text, arrow,
            )
            raw_image = render_robot(
                renderer, model, data,
                fks["raw_action"]["qpos"][sample],
                fks["raw_action"]["model_tcp"][sample],
                fks["raw_action"]["source_tcp"][sample],
                phone[observed], accessory[observed], pad, root_inverse,
                "RAW ACTION LATENCY-ALIGNED", observed, sample, event, error_text, arrow,
            )
            optimized_image = render_robot(
                renderer, model, data,
                fks["optimized_action"]["qpos"][sample],
                fks["optimized_action"]["model_tcp"][sample],
                fks["optimized_action"]["source_tcp"][sample],
                phone[observed], accessory[observed], pad, root_inverse,
                "OPTIMIZED_ACTION LATENCY-ALIGNED", observed, sample, event, error_text, arrow,
            )
            optimized_writer.write(optimized_image)
            if observed in KEYS:
                key_images[observed] = optimized_image.copy()

            raw_cam = cv2.imread(str(inputs["cam_high_files"][observed]), cv2.IMREAD_COLOR)
            if raw_cam is None:
                raise RuntimeError(f"Cannot read raw cam frame {observed}")
            raw_cam = overlay(raw_cam, "RAW cam_high", observed, None, event, None, error_text)
            panels = [raw_cam, state_image, raw_image, optimized_image]
            four_writer.write(np.hstack([cv2.resize(panel, (480, 360), interpolation=cv2.INTER_AREA) for panel in panels]))
    finally:
        optimized_writer.release()
        four_writer.release()
        renderer.close()

    metadata = {
        "status": "BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY",
        "detail_status": "BLOCKED_SOURCE_HAND_OBJECT_RELATION",
        "action_to_observation_lag_frames": 7,
        "action_sample_for_observed_frame": "observed_frame - 7",
        "fps": FPS,
        "source_fps": 30.0,
        "latency_seconds": 7 / 30,
        "frames": 990,
        "timeline_shifted": False,
        "optimized_action_modified": False,
        "precommand_hold_frames": list(range(7)),
        "terminal_action_samples_retained": list(range(983, 990)),
        "relation_file": str(RELATIONS.resolve()),
        "relation_file_sha256": sha256(RELATIONS),
        "source_action_npz": str(base.ACTION_NPZ.resolve()),
        "source_action_npz_sha256": sha256(base.ACTION_NPZ),
        "approved_timeline": str(base.TIMELINE.resolve()),
        "approved_timeline_sha256": sha256(base.TIMELINE),
        "stationary_model": str(base.MODEL_XML.resolve()),
        "stationary_model_sha256": sha256(base.MODEL_XML),
        "physics": False,
        "g1_ik": False,
        "hardware": False,
        "dds": False,
        "publisher": False,
    }
    base.embed_metadata(optimized_raw, optimized_path, "Latency-aligned optimized_action source replay", metadata)
    base.embed_metadata(
        four_raw,
        four_path,
        "Raw | state | raw action[f-7] | optimized_action[f-7] relation diagnostic",
        {**metadata, "panels": ["raw_cam_high", "observation_state", "raw_action_f_minus_7", "optimized_action_f_minus_7"]},
    )

    columns = 4
    width, height = 480, 360
    rows = int(math.ceil(len(KEYS) / columns))
    sheet = np.full((rows * height, columns * width, 3), 20, dtype=np.uint8)
    for index, frame in enumerate(KEYS):
        image = cv2.resize(key_images[frame], (width, height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    sheet_path = OUT / "source_relation_blocker_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)

    result = {
        "status": "BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY_VISUALIZED",
        "detail_status": "BLOCKED_SOURCE_HAND_OBJECT_RELATION_VISUALIZED",
        "videos": {},
        "contact_sheet": {"path": str(sheet_path.resolve()), "sha256": sha256(sheet_path)},
    }
    for path in (optimized_path, four_path):
        count = base.decoded_frames(path)
        if count != 990:
            raise RuntimeError(f"{path}: {count} decoded frames")
        result["videos"][path.name] = {"path": str(path.resolve()), "sha256": sha256(path), "decoded_frames": count}
    (OUT / "latency_aligned_visual_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
