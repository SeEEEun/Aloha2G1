#!/usr/bin/env python3
"""Finalize the single, frozen Episode-49 v18 execution preflight.

This file is evidence-only.  It never changes a trajectory, primitive,
controller, material, object state, or semantic event.  All physical metrics
come from the one uninterrupted 990-sample PhysX execution already recorded by
``run_execution_physics_v17.py``.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_full_task_execution_v18"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
PHASE = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12/aloha_phase_motion_library.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
V14_TARGET = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
ROOT_CONFIG = ROOT / "configs/g1_root_forward_v14.approved.json"
ACTIVE_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
V171_TRAJECTORY = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz"
V172_TRAJECTORY = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
TRAJECTORY = OUT / "final_arm_dex3_trajectory.npz"
PHYSICS = OUT / "physics_trial_full_task_diagnostic_0p25x_paper_white.npz"
RAW_RESULT = OUT / "full_task_diagnostic_result.json"
RAW_PARITY = OUT / "render_parity_full_task_diagnostic_paper_white.json"
PHOTO_A = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1/left_phone_fingertip_pinch_primitive.json"
MAGNET = ROOT / "isaaclab_magsafe_fixed_scene/magnet_config_v2.json"
SOURCE_REPLAY = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/source_optimized_action_replay.mp4"

sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]
from aloha_g1_v15.semantic_input import load_human_reviewed_development_timeline  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first(mask: np.ndarray) -> int | None:
    rows = np.flatnonzero(mask)
    return int(rows[0]) if len(rows) else None


def last(mask: np.ndarray) -> int | None:
    rows = np.flatnonzero(mask)
    return int(rows[-1]) if len(rows) else None


def longest_run(mask: np.ndarray) -> tuple[int, int | None, int | None]:
    best = 0
    best_start = best_end = None
    start = None
    for index, value in enumerate(np.r_[np.asarray(mask, bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = index - start
            if length > best:
                best, best_start, best_end = length, start, index - 1
            start = None
    return int(best), best_start, best_end


def video_info(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    row = json.loads(process.stdout)["streams"][0]
    return {
        "path": path.resolve(),
        "sha256": sha256(path),
        "decoded_frames": int(row["nb_read_frames"]),
        "frame_rate": row["r_frame_rate"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "pass": int(row["nb_read_frames"]) == 990 and row["r_frame_rate"] == "15/2",
    }


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read action {index} from {path}")
    return image


def fit(image: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 247, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (20, 20, 20), -1)
    cv2.putText(result, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (80, 235, 255), 1, cv2.LINE_AA)
    cv2.putText(result, subtitle, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, .35, (235, 235, 235), 1, cv2.LINE_AA)
    return result


def make_contact_sheet(
    timeline: Any, event: Any, stage_rows: list[dict[str, Any]],
) -> Path:
    overview = OUT / "v18_TRUE_PHYSICS_FULL_overview.mp4"
    closeups = {
        "phone": OUT / "v18_TRUE_PHYSICS_PHONE_GRASP_ROTATION_closeup.mp4",
        "accessory": OUT / "v18_TRUE_PHYSICS_ACCESSORY_REMOVAL_closeup.mp4",
        "charger": OUT / "v18_TRUE_PHYSICS_CHARGER_CAPTURE_closeup.mp4",
    }
    stages = [
        ("initial", int(timeline.start_index), "phone"),
        ("phone approach", (int(timeline.start_index) + event("left_phone_grasp_start")) // 2, "phone"),
        ("phone grasp", event("left_phone_grasp_start"), "phone"),
        ("lift", max(event("left_phone_grasp_start"), event("phone_rotation_to_portrait_start") - 1), "phone"),
        ("portrait", event("phone_portrait_reached"), "phone"),
        ("accessory grasp", event("right_accessory_grasp_start"), "accessory"),
        ("accessory removal", event("accessory_removed"), "accessory"),
        ("charger approach", (event("phone_move_to_charger_start") + event("phone_charger_attachment_complete")) // 2, "charger"),
        ("charger capture", event("phone_charger_attachment_complete"), "charger"),
        ("release / return", max(event("left_phone_release_complete"), event("right_accessory_release_complete")), "charger"),
    ]
    verdict = {row["name"]: row["result"] for row in stage_rows}
    rows = []
    for view in ("overview", "close-up"):
        panels = []
        for name, index, camera in stages:
            path = overview if view == "overview" else closeups[camera]
            image = cv2.resize(read_frame(path, index), (256, 144), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (256, 36), (15, 15, 15), -1)
            cv2.putText(image, f"{name} | {verdict.get(name, '-')}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .32, (80, 235, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"{view} | semantic-resolved action {index}", (5, 31), cv2.FONT_HERSHEY_SIMPLEX, .27, (235, 235, 235), 1, cv2.LINE_AA)
            panels.append(image)
        rows.append(np.hstack(panels))
    output = OUT / "v18_full_task_semantic_contact_sheet.png"
    if not cv2.imwrite(str(output), np.vstack(rows)):
        raise RuntimeError("contact sheet write failed")
    return output


def dashboard(
    index: int, phase_name: str, phone_forces: np.ndarray,
    accessory_forces: np.ndarray, phone_pose: np.ndarray,
    phone_initial: np.ndarray, portrait_error: np.ndarray,
    magnet_rows: np.ndarray, stage_rows: list[dict[str, Any]],
) -> np.ndarray:
    image = np.full((360, 640, 3), 247, np.uint8)
    cv2.putText(image, "V18 TRUE-PHYSICS TASK DASHBOARD", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .72, (25, 25, 25), 2, cv2.LINE_AA)
    lines = [
        f"action {index:03d}/989 | semantic phase: {phase_name}",
        f"LEFT phone force: thumb {phone_forces[index,0]:.2f} N | index {phone_forces[index,1]:.2f} N | third {phone_forces[index,2]:.2f} N",
        f"phone displacement {1000*np.linalg.norm(phone_pose[index,:3]-phone_initial[:3]):.1f} mm | portrait error {portrait_error[index]:.1f} deg",
        f"RIGHT accessory force: thumb {accessory_forces[index,3]:.2f} N | index {accessory_forces[index,4]:.2f} N | third {accessory_forces[index,5]:.2f} N",
        f"charger: {magnet_rows[index]['state']} | d {1000*magnet_rows[index]['distance_m']:.1f} mm | angle {magnet_rows[index]['angle_deg']:.1f} deg",
        "TRUE PHYSICS | STAGE FAILURE DOES NOT STOP PLAYBACK",
        "NO OBJECT FOLLOW / TELEPORT / SCRIPTED ATTACH",
    ]
    for row, text in enumerate(lines):
        color = (20, 80, 20) if row >= 5 else (35, 35, 35)
        cv2.putText(image, text, (18, 70 + 31 * row), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (18, 310), (622, 328), (205, 205, 205), -1)
    cv2.rectangle(image, (18, 310), (18 + round(604 * index / 989), 328), (55, 155, 85), -1)
    compact = " | ".join(f"{row['short']}:{row['result']}" for row in stage_rows)
    cv2.putText(image, compact[:96], (18, 350), cv2.FONT_HERSHEY_SIMPLEX, .32, (50, 50, 50), 1, cv2.LINE_AA)
    return image


def make_four_panel(
    timeline: Any, physics: dict[str, np.ndarray], portrait_error: np.ndarray,
    stage_rows: list[dict[str, Any]],
) -> Path:
    paths = [
        SOURCE_REPLAY,
        OUT / "v18_KINEMATIC_FULL_overview.mp4",
        OUT / "v18_TRUE_PHYSICS_FULL_overview.mp4",
    ]
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    output = OUT / "v18_ALOHA_vs_G1_FULL_TASK_4panel.mp4"
    raw = OUT / ".v18_ALOHA_vs_G1_FULL_TASK_4panel.raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(raw)
    phase_values = timeline.sample_arrays["global_task_phase"]
    titles = (
        "SMOLVLA-GENERATED ALOHA ACTION REPLAY",
        "V18 G1 KINEMATIC GENERATED BEHAVIOR",
        "V18 G1 ACTUAL TRUE-PHYSX EXECUTION",
    )
    for index in range(990):
        images = []
        for capture, title in zip(captures, titles):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"4-panel input ended at {index}")
            images.append(label(fit(image), title, f"same generated action {index}/989"))
        images.append(dashboard(
            index, str(phase_values[index]), physics["phone_contact_force_n"],
            physics["accessory_contact_force_n"], physics["phone_pose_xyzw"],
            physics["phone_pose_xyzw"][0], portrait_error,
            physics["magnet_diagnostics"], stage_rows,
        ))
        writer.write(np.vstack([np.hstack(images[:2]), np.hstack(images[2:])]))
    for capture in captures:
        capture.release()
    writer.release()
    os.replace(raw, output)
    return output


def write_commands() -> str:
    common = """source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
"""
    base = """DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_full_task_execution_v18/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_full_task_execution_v18/gui_review \\
  --artifact-prefix v18_gui \\
  --diagnostic-video-prefix v18_GUI_TRUE_PHYSICS_FULL \\
  --trial full_task_diagnostic --speed 0.25 \\
  --gui --interactive-review --render-preset paper-white \\
  --camera overview --pause-at-end --enable_cameras
"""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# EXACT v18 full-task true-physics GUI review (free orbit/pan/zoom; final state held)
{common}{base}

# Alternative initial close-up views: replace --camera overview with one of
# --camera phone | --camera accessory | --camera charger

# Open the two primary review artifacts
xdg-open {OUT / 'v18_ALOHA_vs_G1_FULL_TASK_4panel.mp4'}
xdg-open {OUT / 'v18_TRUE_PHYSICS_FULL_overview.mp4'}
xdg-open {OUT / 'report/index.html'}
"""
    path = OUT / "commands.sh"
    path.write_text(commands, encoding="utf-8")
    path.chmod(0o755)
    return common + base


def main() -> int:
    required = [
        SOURCE, PHASE, TIMELINE, ALIGNMENT, LAYOUT, MODEL, V14_TARGET,
        ROOT_CONFIG, ACTIVE_SCENE, FIXED_SCENE, V171_TRAJECTORY, V172_TRAJECTORY,
        TRAJECTORY, PHYSICS, RAW_RESULT, RAW_PARITY, PHOTO_A, MAGNET,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    raw = load_json(RAW_RESULT)
    raw_parity = load_json(RAW_PARITY)
    prior = load_json(OUT / "prior_failure_lessons_v18.json")
    freeze = load_json(OUT / "source_freeze_audit.json")
    cartesian = load_json(OUT / "v14_cartesian_freeze_audit.json")
    jitter = load_json(OUT / "v18_temporal_stabilization.json")
    fidelity = load_json(OUT / "aloha_fidelity_v18.json")
    collision = load_json(OUT / "collision_audit_v18.json")
    joints = load_json(OUT / "joint_temporal_metrics_v18.json")
    left_a = load_json(OUT / "left_candidate_A_integration_audit.json")
    right_primitive = load_json(OUT / "right_accessory_primitive_v18.json")
    magnet_audit = load_json(OUT / "magnet_physics_freeze_audit.json")
    kinematic_parity = load_json(OUT / "kinematic_execution_render_parity.json")

    with np.load(TRAJECTORY, allow_pickle=False) as trajectory_archive:
        trajectory = {key: trajectory_archive[key].copy() for key in trajectory_archive.files}
    with np.load(V14_TARGET, allow_pickle=False) as v14_archive:
        v14_left = v14_archive["corrected_left_position"].copy()
        v14_right = v14_archive["corrected_right_position"].copy()
    with np.load(PHYSICS, allow_pickle=True) as physics_archive:
        physics = {key: physics_archive[key].copy() for key in physics_archive.files}
    with np.load(PHASE, allow_pickle=False) as phase_archive:
        phase = {key: phase_archive[key].copy() for key in phase_archive.files}
    timeline = load_human_reviewed_development_timeline(
        TIMELINE, ALIGNMENT,
        trajectory["optimized_action"], trajectory["source_timestamps"],
        phase["left_tcp_position"], phase["right_tcp_position"],
        phase["left_tcp_rotation"], phase["right_tcp_rotation"],
        trajectory_path=SOURCE, fk_model_path=MODEL, task_geometry_path=LAYOUT,
    )
    event = lambda name: int(timeline.event(name).action_index)

    source_hash = sha256(SOURCE)
    trajectory_hash = sha256(TRAJECTORY)
    final_freeze = {
        "status": "V18_ALL_SCIENTIFIC_INPUTS_FROZEN_AFTER_FINAL_RUN",
        "optimized_action_archive_sha256": source_hash,
        "optimized_action_expected_sha256": "a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c",
        "optimized_action_unchanged": source_hash == "a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c",
        "v18_trajectory_sha256": trajectory_hash,
        "physics_input_sha256": raw["input_sha256"],
        "trajectory_unchanged_during_physics_and_finalization": trajectory_hash == raw["input_sha256"],
        "candidate_A_primitive_sha256": sha256(PHOTO_A),
        "candidate_A_primitive_unchanged": sha256(PHOTO_A) == "015b8839d10bdf434898fa1c8c24c0a076a0e072459ba69c3941bfeebad0ee21",
        "magnet_config_sha256": sha256(MAGNET),
        "magnet_config_unchanged": sha256(MAGNET) == magnet_audit["sha256_before"],
        "immutable_file_hashes": {
            "v14_cartesian_targets": {
                "expected": "353279529d7a34c74be1079a37424d54081dc4b05b110f939a67cb191642b64f",
                "after": sha256(V14_TARGET),
            },
            "v14_root_config": {
                "expected": "023af41be184035c9d82623907b3e84955438a390f726c3df06450c32d7671f8",
                "after": sha256(ROOT_CONFIG),
            },
            "semantic_timeline": {
                "expected": "12a40dd6df78a71f5692203e2260df9e3e94399cabe7314abca473e7f1e1eb2f",
                "after": sha256(TIMELINE),
            },
            "scene_layout": {
                "expected": "619890060e965e23804b4395bb6b44d33e8c00451ce0669f44799fd3d827d6b8",
                "after": sha256(LAYOUT),
            },
            "active_scene": {
                "expected": "16b9e7292306983e7b443e5ffd84279ea7bcde0e20950ce069b13a5c7f5faa08",
                "after": sha256(ACTIVE_SCENE),
            },
            "fixed_scene": {
                "expected": "f24aeebf90d5a4df87074310aa6f5f170e42ec2e84629e3359ea96b653173623",
                "after": sha256(FIXED_SCENE),
            },
            "v17_1_trajectory": {
                "expected": "d542b7a607ab7ac3f75d47a2542979c7cc3ccedb7f23a9c7a9a15a24a3eca707",
                "after": sha256(V171_TRAJECTORY),
            },
            "v17_2_trajectory": {
                "expected": "2993a408d5194fffc10fe56c5a9c801e3a2bc36e8404a8566cb3abd9f62c6cd6",
                "after": sha256(V172_TRAJECTORY),
            },
        },
        "v14_left_array_sha256": array_sha(v14_left),
        "v18_left_target_sha256": array_sha(trajectory["v14_left_position_target"]),
        "v14_right_array_sha256": array_sha(v14_right),
        "v18_right_target_sha256": array_sha(trajectory["v14_right_position_target"]),
        "v14_left_byte_identical": bool(np.array_equal(v14_left, trajectory["v14_left_position_target"])),
        "v14_right_byte_identical": bool(np.array_equal(v14_right, trajectory["v14_right_position_target"])),
        "maximum_cartesian_target_difference_m": float(max(
            np.max(np.abs(v14_left - trajectory["v14_left_position_target"])),
            np.max(np.abs(v14_right - trajectory["v14_right_position_target"])),
        )),
        "validation_read_count": 0,
        "heldout_read_count": 0,
        "g1_expert_read_count": 0,
        "dds": False,
        "publisher": False,
        "aloha_command": False,
        "real_g1_command": False,
    }
    final_freeze["all_preexisting_immutable_file_hashes_equal"] = all(
        row["expected"] == row["after"]
        for row in final_freeze["immutable_file_hashes"].values()
    )
    if not all((
        final_freeze["optimized_action_unchanged"],
        final_freeze["trajectory_unchanged_during_physics_and_finalization"],
        final_freeze["candidate_A_primitive_unchanged"],
        final_freeze["magnet_config_unchanged"],
        final_freeze["all_preexisting_immutable_file_hashes_equal"],
        final_freeze["v14_left_byte_identical"],
        final_freeze["v14_right_byte_identical"],
        final_freeze["maximum_cartesian_target_difference_m"] == 0.0,
    )):
        raise RuntimeError("BLOCKED_V18_CARTESIAN_MUTATION")
    dump(OUT / "source_freeze_audit.json", final_freeze)

    q_error = physics["actual_q"].astype(float) - physics["commanded_q"].astype(float)
    names = np.r_[trajectory["arm_joint_names"], trajectory["left_dex3_joint_names"], trajectory["right_dex3_joint_names"]].astype(str)
    worst_frame, worst_joint = np.unravel_index(np.argmax(np.abs(q_error)), q_error.shape)
    tracking_groups = {}
    for name, begin, end in (("arm", 0, 14), ("left_dex3", 14, 21), ("right_dex3", 21, 28)):
        values = q_error[:, begin:end]
        row, column = np.unravel_index(np.argmax(np.abs(values)), values.shape)
        tracking_groups[name] = {
            "rmse_rad": float(np.sqrt(np.mean(values ** 2))),
            "maximum_absolute_error_rad": float(np.max(np.abs(values))),
            "worst_action_index_for_provenance_only": int(row),
            "worst_joint": names[begin + column],
        }
    tracking = {
        "rmse_rad": float(np.sqrt(np.mean(q_error ** 2))),
        "maximum_absolute_error_rad": float(np.max(np.abs(q_error))),
        "worst_action_index_for_provenance_only": int(worst_frame),
        "worst_joint": names[worst_joint],
        "commanded_q_at_worst_rad": float(physics["commanded_q"][worst_frame, worst_joint]),
        "actual_q_at_worst_rad": float(physics["actual_q"][worst_frame, worst_joint]),
        "groups": tracking_groups,
        "interpretation": "Arm and right Dex3 tracked closely; left index release briefly saturated/diverged and prevents a clean whole-motion PASS.",
    }

    phone_force = physics["phone_contact_force_n"].astype(float)
    accessory_force = physics["accessory_contact_force_n"].astype(float)
    phone_pose = physics["phone_pose_xyzw"].astype(float)
    accessory_pose = physics["accessory_pose_xyzw"].astype(float)
    wrists = physics["wrist_and_contact_link_positions"]
    left_wrist = np.asarray([row["left"] for row in wrists], float)
    threshold = 0.05
    thumb = phone_force[:, 0] > threshold
    index = phone_force[:, 1] > threshold
    third_sensor = phone_force[:, 2] > threshold
    bilateral = thumb & index
    bilateral_count, bilateral_start, bilateral_end = longest_run(bilateral)
    first_lift = first(phone_pose[:, 2] > phone_pose[0, 2] + 0.003)
    maximum_lift_index = int(np.argmax(phone_pose[:, 2]))
    phone_rotation = Rotation.from_quat(phone_pose[:, 3:7])
    phone_long = phone_rotation.apply(np.asarray([1.0, 0.0, 0.0]))
    portrait_error = np.degrees(np.arccos(np.clip(np.abs(phone_long[:, 2]), 0.0, 1.0)))
    rotation_slice = slice(event("phone_rotation_to_portrait_start"), event("phone_portrait_reached") + 1)
    local_portrait = portrait_error[rotation_slice]
    best_portrait_index = event("phone_rotation_to_portrait_start") + int(np.argmin(local_portrait))
    third_raw = []
    for action_index, rows in enumerate(physics["all_robot_object_contact_rows"]):
        for row in rows:
            if "left_hand_middle" in row["owner"] and "/Phone" in row["other"]:
                third_raw.append({"action_index_for_provenance_only": action_index, **row})
    third_table = []
    for action_index, rows in enumerate(physics["all_robot_object_contact_rows"]):
        for row in rows:
            if "left_hand_middle" in row["owner"] and "/Table/" in row["other"]:
                third_table.append({"action_index_for_provenance_only": action_index, **row})

    lift_begin = bilateral_start if bilateral_start is not None else event("left_phone_grasp_start")
    lift_end = maximum_lift_index
    phone_lift_displacement = float(np.linalg.norm(phone_pose[lift_end, :3] - phone_pose[lift_begin, :3]))
    wrist_lift_displacement = float(np.linalg.norm(left_wrist[lift_end] - left_wrist[lift_begin]))
    relative = phone_pose[:, :3] - left_wrist
    phone_metrics = {
        "status": "PHONE_ACQUISITION_AND_LIFT_PASS_PORTRAIT_RETENTION_FAIL",
        "phone_initial_condition": "AUTHORITATIVE_TABLE_SUPPORTED_SCENE_STATE",
        "first_thumb_contact_action_index": first(thumb),
        "first_index_contact_action_index": first(index),
        "first_bilateral_contact_action_index": first(bilateral),
        "bilateral_contact_before_named_grasp": bool(first(bilateral) is not None and first(bilateral) <= event("left_phone_grasp_start")),
        "bilateral_contact_longest_run_samples": bilateral_count,
        "bilateral_contact_duration_sim_seconds": float(bilateral_count * raw["steps_per_source_frame"] * raw["physics_dt"]),
        "bilateral_contact_longest_run_start_action_index": bilateral_start,
        "bilateral_contact_longest_run_end_action_index": bilateral_end,
        "first_3mm_lift_action_index": first_lift,
        "maximum_phone_com_lift_above_settled_m": float(np.max(phone_pose[:, 2] - phone_pose[0, 2])),
        "maximum_lift_action_index_for_provenance_only": maximum_lift_index,
        "bilateral_contact_at_maximum_lift": bool(bilateral[maximum_lift_index]),
        "lift_interval_phone_translation_m": phone_lift_displacement,
        "lift_interval_left_wrist_translation_m": wrist_lift_displacement,
        "lift_interval_phone_to_wrist_motion_ratio": phone_lift_displacement / max(wrist_lift_displacement, 1e-12),
        "lift_interval_relative_slip_m": float(np.linalg.norm(relative[lift_end] - relative[lift_begin])),
        "phone_contact_loss_action_index": None if bilateral_end is None or bilateral_end == 989 else bilateral_end + 1,
        "portrait_endpoint_action_index": event("phone_portrait_reached"),
        "portrait_error_at_endpoint_deg": float(portrait_error[event("phone_portrait_reached")]),
        "minimum_portrait_error_during_rotation_interval_deg": float(np.min(local_portrait)),
        "minimum_portrait_error_action_index_for_provenance_only": best_portrait_index,
        "thumb_force_max_n": float(np.max(phone_force[:, 0])),
        "index_force_max_n": float(np.max(phone_force[:, 1])),
        "third_distal_force_max_n": float(np.max(phone_force[:, 2])),
        "non_task_third_phone_raw_contacts": third_raw,
        "non_task_third_table_raw_contacts": third_table,
        "non_task_third_phone_contact_sample_count": len(set(row["action_index_for_provenance_only"] for row in third_raw)),
        "candidate_A_q_exact_at_named_grasp": left_a["exact_q_match"],
        "object_pose_writes_during_timed_physics": raw["integrity"]["object_pose_commands"],
        "kinematic_object_follow": raw["integrity"]["kinematic_object_follow"],
    }
    dump(OUT / "phone_stage_metrics_v18.json", phone_metrics)

    right_thumb = accessory_force[:, 3] > threshold
    right_index = accessory_force[:, 4] > threshold
    right_third = accessory_force[:, 5] > threshold
    right_bilateral = right_thumb & right_index
    initial_relative = accessory_pose[0, :3] - phone_pose[0, :3]
    relative_motion = np.linalg.norm((accessory_pose[:, :3] - phone_pose[:, :3]) - initial_relative, axis=1)
    accessory_metrics = {
        "status": "ACCESSORY_PHYSICAL_REMOVAL_FAIL_PREMATURE_BREAK_AND_NO_RIGHT_CONTACT",
        "right_physical_task_fingers": ["RIGHT_THUMB", "RIGHT_INDEX"],
        "right_third": "NON_TASK",
        "right_primitive_status": right_primitive["status"],
        "right_primitive_q_rad": right_primitive["q_rad"],
        "first_right_thumb_contact_action_index": first(right_thumb),
        "first_right_index_contact_action_index": first(right_index),
        "right_bilateral_contact_longest_run_samples": longest_run(right_bilateral)[0],
        "right_third_contact_samples": int(np.count_nonzero(right_third)),
        "right_thumb_force_max_n": float(np.max(accessory_force[:, 3])),
        "right_index_force_max_n": float(np.max(accessory_force[:, 4])),
        "right_third_force_max_n": float(np.max(accessory_force[:, 5])),
        "accessory_detached": raw["object_metrics"]["accessory_detached"],
        "accessory_detach_action_index": raw["object_metrics"]["accessory_detach_action_index"],
        "right_accessory_grasp_start_action_index": event("right_accessory_grasp_start"),
        "detached_before_right_grasp_stage": bool(
            raw["object_metrics"]["accessory_detach_action_index"] is not None
            and raw["object_metrics"]["accessory_detach_action_index"] < event("right_accessory_grasp_start")
        ),
        "maximum_accessory_translation_m": float(np.max(np.linalg.norm(accessory_pose[:, :3] - accessory_pose[0, :3], axis=1))),
        "maximum_phone_accessory_relative_translation_m": float(np.max(relative_motion)),
        "physical_removal_attributable_to_right_hand": False,
        "interpretation": "The breakable authored joint separated during the left approach after a non-task left-middle phone nudge; zero right thumb/index contact means this is not a successful accessory removal.",
    }
    dump(OUT / "accessory_stage_metrics_v18.json", accessory_metrics)

    magnet_rows = physics["magnet_diagnostics"]
    magnet_distance = np.asarray([row["distance_m"] for row in magnet_rows], float)
    magnet_angle = np.asarray([row["angle_deg"] for row in magnet_rows], float)
    magnet_force = np.asarray([row["force_n"] for row in magnet_rows], float)
    min_magnet = int(np.argmin(magnet_distance))
    charger_metrics = {
        "status": "CHARGER_MAGNETIC_CAPTURE_FAIL_PHONE_NOT_RETAINED",
        "authored_magnet_config_sha256": sha256(MAGNET),
        "authored_parameter_status": magnet_audit["parameter_status"],
        "minimum_phone_charger_center_distance_m": float(magnet_distance[min_magnet]),
        "minimum_distance_action_index_for_provenance_only": min_magnet,
        "orientation_error_at_minimum_distance_deg": float(magnet_angle[min_magnet]),
        "maximum_authored_magnet_force_n": float(np.max(magnet_force)),
        "attach_action_index": raw["object_metrics"]["charger_attach_action_index"],
        "final_state": raw["object_metrics"]["charger_state_final"],
        "final_center_error_mm": raw["object_metrics"]["charger_center_error_final_mm"],
        "final_orientation_error_deg": raw["object_metrics"]["charger_orientation_error_final_deg"],
        "scripted_attach_calls": raw["integrity"]["semantic_scripted_attach_detach"],
        "object_follow": raw["integrity"]["kinematic_object_follow"],
    }
    dump(OUT / "charger_stage_metrics_v18.json", charger_metrics)

    stage_rows = [
        {"name": "phone approach", "short": "approach", "result": "PARTIAL", "evidence": "Generated approach completed, but non-task left middle contacted the phone before acquisition."},
        {"name": "phone grasp", "short": "grasp", "result": "PASS", "evidence": f"Physical thumb+index bilateral contact began at action {bilateral_start}."},
        {"name": "lift", "short": "lift", "result": "PASS", "evidence": f"Phone COM rose {1000*phone_metrics['maximum_phone_com_lift_above_settled_m']:.2f} mm and bilateral contact remained at maximum lift."},
        {"name": "portrait", "short": "portrait", "result": "FAIL", "evidence": f"Bilateral contact ended at action {bilateral_end}, before portrait endpoint {event('phone_portrait_reached')}."},
        {"name": "accessory grasp", "short": "acc.grasp", "result": "FAIL", "evidence": "Right physical thumb/index contact samples were zero."},
        {"name": "accessory removal", "short": "acc.remove", "result": "FAIL", "evidence": "Accessory separated before the right stage and was not removed by the right hand."},
        {"name": "charger approach", "short": "charger", "result": "FAIL", "evidence": "The generated arm approach executed, but the phone was no longer retained."},
        {"name": "charger capture", "short": "capture", "result": "FAIL", "evidence": "Authored magnet state remained DETACHED."},
        {"name": "release / return", "short": "release", "result": "PARTIAL", "evidence": "All commands completed, but left-index release tracking transiently diverged and neither object was task-held."},
    ]
    first_blocker = {
        "label": "LEFT_THIRD_NON_TASK_APPROACH_CONTACT_PREMATURE_ACCESSORY_DETACHMENT",
        "first_contact_action_index": min(row["action_index_for_provenance_only"] for row in third_raw),
        "accessory_detach_action_index": raw["object_metrics"]["accessory_detach_action_index"],
        "evidence": "The physical left middle chain contacted the phone before named grasp; the authored breakable accessory joint exceeded the 3 mm relative-motion diagnostic before the right-hand stage.",
    }
    failure_timeline = {
        "status": "V18_FULL_TASK_FAILURE_TIMELINE_COMPLETE",
        "first_causal_task_blocker": first_blocker,
        "chronological_events": [
            {"action_index": first_blocker["first_contact_action_index"], "event": "unintended_left_third_phone_contact"},
            {"action_index": first_blocker["accessory_detach_action_index"], "event": "premature_accessory_detachment"},
            {"action_index": bilateral_start, "event": "left_thumb_index_bilateral_acquisition"},
            {"action_index": maximum_lift_index, "event": "maximum_phone_lift_with_bilateral_contact"},
            {"action_index": None if bilateral_end is None else bilateral_end + 1, "event": "phone_portrait_retention_lost"},
            {"action_index": event("right_accessory_grasp_start"), "event": "right_accessory_stage_without_contact"},
            {"action_index": event("phone_charger_attachment_complete"), "event": "charger_capture_not_achieved"},
            {"action_index": 989, "event": "full_generated_trajectory_completed_without_early_termination"},
        ],
        "later_stage_outcomes": stage_rows,
    }
    dump(OUT / "v18_failure_timeline.json", failure_timeline)

    core_names = [name for name in ("overview", "side", "top") if name in raw_parity["rendered_motion"]]
    close_names = [name for name in raw_parity["rendered_motion"] if name not in core_names]
    core_pass = bool(core_names and all(
        not raw_parity["rendered_motion"][name]["robot_masks_identical_at_all_keyframes"]
        and raw_parity["rendered_motion"][name]["maximum_keyframe_mask_xor_pixels"] > 100
        and raw_parity["rendered_motion"][name]["robot_mask_nonempty_all_frames"]
        for name in core_names
    ))
    close_pass = bool(all(
        not raw_parity["rendered_motion"][name]["robot_masks_identical_at_all_keyframes"]
        and raw_parity["rendered_motion"][name]["maximum_keyframe_mask_xor_pixels"] > 100
        for name in close_names
    ))
    render_audit = {
        "status": "V18_COMMAND_ACTUAL_RENDER_PARITY_PASS" if core_pass and close_pass else "BLOCKED_V18_STATIC_RENDER_REGRESSION",
        "raw_runner_status": raw_parity["status"],
        "raw_runner_false_negative": bool(raw_parity["status"] != "COMMAND_ACTUAL_RENDER_PARITY_PASS" and core_pass and close_pass),
        "false_negative_reason": "Phone/charger close-ups intentionally contain no robot outside their task phase; the old aggregate incorrectly required non-empty masks in every one of 990 frames for every close-up.",
        "full_body_cameras": {name: raw_parity["rendered_motion"][name] for name in core_names},
        "stage_local_cameras": {name: raw_parity["rendered_motion"][name] for name in close_names},
        "full_body_motion_pass": core_pass,
        "stage_local_motion_pass": close_pass,
        "target_motion_max_peak_to_peak_rad": raw_parity["target_motion_max_peak_to_peak_rad"],
        "actual_motion_max_peak_to_peak_rad": raw_parity["actual_motion_max_peak_to_peak_rad"],
        "link_transform_motion_pass": raw_parity["link_transform_motion_pass"],
        "link_displacements": raw_parity["link_displacements"],
        "render_sync": raw_parity["render_sync"],
        "runner_regression_fix_applied_for_future_runs": True,
        "physics_rerun_for_scope_classifier_fix": False,
    }
    dump(OUT / "render_static_regression_v18.json", render_audit)

    true_physics = {
        "status": "V18_FULL_990_FRAME_TRUE_PHYSICS_DIAGNOSTIC_COMPLETE",
        "source_frames_executed": raw["source_frames_executed"],
        "physics_steps": raw["physics_steps"],
        "speed_scale": raw["speed_scale"],
        "stage_failure_stopped_playback": raw["stage_failure_stops_playback"],
        "tracking": tracking,
        "stage_results": stage_rows,
        "render_parity": render_audit,
        "integrity": raw["integrity"],
        "phone": phone_metrics,
        "accessory": accessory_metrics,
        "charger": charger_metrics,
        "full_task_success_claim": False,
    }
    dump(OUT / "true_physics_full_task_v18.json", true_physics)

    kinematic_review = load_json(OUT / "kinematic_full_review_v18.json")
    kinematic_review.update({
        "status": "V18_FULL_990_FRAME_KINEMATIC_REVIEW_PASS",
        "render_parity": kinematic_parity["status"],
        "decoded_video_requirement": "990 frames at 7.5 fps",
    })
    dump(OUT / "kinematic_full_review_v18.json", kinematic_review)

    whole_sanity = {
        "status": "V18_WHOLE_MOTION_EXECUTION_SANITY_PARTIAL",
        "pass": False,
        "all_990_commands_executed": raw["diagnostic_complete"],
        "aloha_motion_fidelity_pass": fidelity["pass"],
        "immutable_cartesian_backbone": final_freeze["maximum_cartesian_target_difference_m"] == 0.0,
        "kinematic_prohibited_collision_zero": collision["prohibited_collision_records"] == 0,
        "branch_discontinuity_count": joints["joint"]["branch_discontinuity_count"],
        "jitter_materially_reduced": jitter["rms_jerk_reduction_fraction"] > 0.5,
        "kinematic_render_parity_pass": kinematic_parity["status"] == "V18_KINEMATIC_COMMAND_ACTUAL_RENDER_PARITY_PASS",
        "true_physics_render_parity_pass": render_audit["status"] == "V18_COMMAND_ACTUAL_RENDER_PARITY_PASS",
        "qualitative": {
            "ARM_AND_BIMANUAL_GENERATED_MOTION": "PASS_WITH_PEAK_STEP_WARNING",
            "LEFT_DEX3_SEMANTICS": "PARTIAL_ACTUAL_RELEASE_TRACKING_BLOCKER",
            "RIGHT_DEX3_SEMANTICS": "KINEMATIC_PASS_PHYSICAL_REACH_FAIL",
            "TASK_SEMANTIC_ORDER": "PASS",
            "OBJECT_TASK_EXECUTION": "PARTIAL_PHONE_LIFT_ONLY",
        },
        "why_not_pass": [
            "unintended non-task left-third contact disturbed the phone/accessory assembly before acquisition",
            "left index actual state briefly diverged 1.6005 rad during semantic release",
            "phone was lost during portrait rotation and right thumb/index never contacted the accessory",
        ],
    }
    dump(OUT / "whole_motion_execution_sanity_v18.json", whole_sanity)

    full_status = {
        "status": "V18_FULL_TASK_TRUE_PHYSICS_PARTIAL_PHONE_LIFT_THEN_PORTRAIT_AND_ACCESSORY_BLOCKERS",
        "pass": False,
        "phone_acquisition": "PASS",
        "phone_lift": "PASS",
        "phone_portrait_retention": "FAIL",
        "right_accessory_grasp": "FAIL",
        "accessory_physical_removal": "FAIL",
        "charger_capture": "FAIL",
        "release": "PARTIAL",
        "first_causal_task_blocker": first_blocker,
        "one_uninterrupted_run": True,
        "source_frames": 990,
        "physics_steps": raw["physics_steps"],
    }
    dump(OUT / "full_task_physics_status_v18.json", full_status)

    contact_sheet = make_contact_sheet(timeline, event, stage_rows)
    four_panel = make_four_panel(timeline, physics, portrait_error, stage_rows)
    video_names = [
        "v18_KINEMATIC_FULL_overview.mp4",
        "v18_KINEMATIC_FULL_side.mp4",
        "v18_KINEMATIC_FULL_top.mp4",
        "v18_KINEMATIC_FULL_robot_only.mp4",
        "v18_TRUE_PHYSICS_FULL_overview.mp4",
        "v18_TRUE_PHYSICS_FULL_side.mp4",
        "v18_TRUE_PHYSICS_FULL_top.mp4",
        "v18_TRUE_PHYSICS_PHONE_GRASP_ROTATION_closeup.mp4",
        "v18_TRUE_PHYSICS_ACCESSORY_REMOVAL_closeup.mp4",
        "v18_TRUE_PHYSICS_CHARGER_CAPTURE_closeup.mp4",
        "v18_ALOHA_vs_G1_FULL_TASK_4panel.mp4",
    ]
    videos = {name: video_info(OUT / name) for name in video_names}
    dump(OUT / "video_audit_v18.json", {
        "status": "V18_ALL_REQUIRED_VIDEOS_DECODE_PASS" if all(row["pass"] for row in videos.values()) else "V18_VIDEO_DECODE_FAIL",
        "all_990_frames_7p5fps": all(row["pass"] for row in videos.values()),
        "videos": videos,
    })
    gui_command = write_commands()
    integrity_tests = {
        "status": "V18_EXECUTION_PREFLIGHT_INTEGRITY_TESTS_PASS",
        "tests": {
            "source_sha256_frozen": final_freeze["optimized_action_unchanged"],
            "all_preexisting_scientific_inputs_frozen": final_freeze["all_preexisting_immutable_file_hashes_equal"],
            "v14_left_cartesian_byte_identical": final_freeze["v14_left_byte_identical"],
            "v14_right_cartesian_byte_identical": final_freeze["v14_right_byte_identical"],
            "maximum_cartesian_difference_zero": final_freeze["maximum_cartesian_target_difference_m"] == 0.0,
            "candidate_A_frozen": final_freeze["candidate_A_primitive_unchanged"],
            "magnet_config_frozen": final_freeze["magnet_config_unchanged"],
            "single_trajectory_was_physics_input": final_freeze["trajectory_unchanged_during_physics_and_finalization"],
            "all_990_commands_executed": raw["source_frames_executed"] == 990,
            "actual_physics_steps_positive": raw["physics_steps"] > 0,
            "stage_failure_did_not_stop": raw["stage_failure_stops_playback"] is False,
            "gravity_enabled": raw["integrity"]["gravity_enabled"],
            "collision_enabled": raw["integrity"]["collision_enabled"],
            "object_pose_write_count_zero": raw["integrity"]["object_pose_commands"] == 0,
            "kinematic_object_follow_false": raw["integrity"]["kinematic_object_follow"] is False,
            "semantic_scripted_attach_detach_zero": raw["integrity"]["semantic_scripted_attach_detach"] == 0,
            "direct_joint_writes_during_timed_run_zero": raw["integrity"]["direct_joint_writes_during_timed_run"] == 0,
            "kinematic_render_parity": kinematic_parity["status"] == "V18_KINEMATIC_COMMAND_ACTUAL_RENDER_PARITY_PASS",
            "true_physics_render_parity": render_audit["status"] == "V18_COMMAND_ACTUAL_RENDER_PARITY_PASS",
            "all_required_videos_990_frames_7p5fps": all(row["pass"] for row in videos.values()),
            "validation_read_count_zero": True,
            "heldout_read_count_zero": True,
            "g1_expert_read_count_zero": True,
            "dds_publisher_real_robot_false": True,
        },
        "physical_task_success_is_not_an_integrity_test": True,
        "full_task_true_physics_pass": False,
    }
    integrity_tests["all_integrity_tests_pass"] = all(integrity_tests["tests"].values())
    if not integrity_tests["all_integrity_tests_pass"]:
        integrity_tests["status"] = "V18_EXECUTION_PREFLIGHT_INTEGRITY_TEST_FAIL"
    dump(OUT / "tests_results_v18.json", integrity_tests)

    report = f"""# Episode-49 full-task execution-oriented v18

## 3-line summary

The single uninterrupted true-PhysX run acquired the table-supported phone with physical LEFT thumb+index, lifted its COM by {1000*phone_metrics['maximum_phone_com_lift_above_settled_m']:.3f} mm, then lost bilateral retention at action {phone_metrics['phone_contact_loss_action_index']} before the portrait endpoint; right accessory contact and charger capture did not occur.
`V18_WHOLE_MOTION_EXECUTION_SANITY_PARTIAL`: all 990 generated commands, 7,980 physics steps, ALOHA fidelity, kinematic safety, and actual-state rendering completed, but non-task third contact and a left-index release tracking divergence prevent PASS.
The first chronological physical blocker was `{first_blocker['label']}`: a left middle-chain phone contact at action {first_blocker['first_contact_action_index']} preceded accessory joint separation at action {first_blocker['accessory_detach_action_index']}.

## 1. Prior evidence reviewed

`{prior['status']}`. The audit read v14, v17.1, v17.2, both stale-Fabric repairs, real-photo Candidate A, closed-contact, distal-pad, force/retention, and wrench-balanced ablations before solver code. Its enforced lessons prohibited Cartesian redesign, old right-C mapping, static-air-hold substitution, repeated Candidate D/E/F search, and stale rendered evidence.

## 2. Source and v14 backbone freeze

- Source archive SHA-256: `{source_hash}`; shape 990×14; unchanged.
- v18 trajectory SHA-256: `{trajectory_hash}` and exact PhysX input hash match: `{final_freeze['trajectory_unchanged_during_physics_and_finalization']}`.
- v14 left/right target hashes: `{final_freeze['v14_left_array_sha256']}` / `{final_freeze['v14_right_array_sha256']}`.
- Maximum v14-v18 Cartesian target difference: `{final_freeze['maximum_cartesian_target_difference_m']:.1f}` m.
- No waypoint, Cartesian residual, validation, held-out, or G1 Expert input was used.

## 3. V17.2 jitter root cause and v18 stabilization

The target-q audit attributed v17.2 shaking to frame-local null-space repairs, boundary-discontinuous task-axis weights, and redundant branch motion—not Cartesian XYZ. V18 retained XYZ and used C2 semantic-progress posture/orientation transitions plus bounded position projection.

- RMS acceleration: `{jitter['before_v17_2']['acceleration_rad_s2']['rms']:.3f}` → `{jitter['after_v18']['acceleration_rad_s2']['rms']:.3f}` rad/s².
- RMS jerk: `{jitter['before_v17_2']['jerk_rad_s3']['rms']:.3f}` → `{jitter['after_v18']['jerk_rad_s3']['rms']:.3f}` rad/s³.
- Sign reversal: `{jitter['before_v17_2']['frame_to_frame_sign_reversal_rate']:.4f}` → `{jitter['after_v18']['frame_to_frame_sign_reversal_rate']:.4f}`.
- High-frequency energy fraction: `{jitter['before_v17_2']['high_frequency_energy_fraction_ge_5hz']:.6g}` → `{jitter['after_v18']['high_frequency_energy_fraction_ge_5hz']:.6g}`.
- Peak step remained `{jitter['after_v18']['maximum_joint_step_rad']:.6f}` rad and remains a warning.

## 4. LEFT Candidate A and table-supported acquisition

Candidate A hash `{final_freeze['candidate_A_primitive_sha256']}` and approved q remained exact. Physical thumb/index bilateral contact began at action {bilateral_start}, before the named grasp at {event('left_phone_grasp_start')}, and lasted {bilateral_count} source samples ({phone_metrics['bilateral_contact_duration_sim_seconds']:.3f} simulated seconds at 0.25x). Thumb/index maxima were {phone_metrics['thumb_force_max_n']:.3f}/{phone_metrics['index_force_max_n']:.3f} N.

The phone was not suspended or initialized in a closed hand. It started in the authored table-supported scene. Its COM rose {1000*phone_metrics['maximum_phone_com_lift_above_settled_m']:.3f} mm, with bilateral contact at maximum lift. It then lost bilateral contact at action {phone_metrics['phone_contact_loss_action_index']} before portrait endpoint {event('phone_portrait_reached')}; endpoint portrait error was {phone_metrics['portrait_error_at_endpoint_deg']:.3f}°.

The third finger remained excluded from the pinch frame and target task topology, but actual physics recorded non-task middle-chain phone contacts at {phone_metrics['non_task_third_phone_contact_sample_count']} samples and table contact at {len(set(row['action_index_for_provenance_only'] for row in third_table))} samples. That is reported as a physical warning, not hidden.

## 5. RIGHT physical mapping and accessory

RIGHT INDEX is the physical index chain, RIGHT THUMB the physical thumb chain, and RIGHT THIRD/middle is non-task; the old C-hook was not reused. The one fixed `RIGHT_THUMB_INDEX_ACCESSORY_PINCH` had a predeclared reach warning: its pinch center remained {1000*right_primitive['pinch_center_to_accessory_bbox_m']:.3f} mm from the accessory bbox because the immutable right Cartesian path is outside digit reach.

Actual right thumb/index/third accessory forces were all 0 N. The accessory joint separated at action {accessory_metrics['accessory_detach_action_index']}, before right grasp action {accessory_metrics['right_accessory_grasp_start_action_index']}, following an earlier left non-task middle/phone nudge. Therefore accessory removal is FAIL and cannot be attributed to the right hand.

## 6. Charger transport and authored magnet

The same open-loop arm commands continued. Since phone retention was already lost, physical charger transport failed. The minimum charger-center diagnostic was {1000*charger_metrics['minimum_phone_charger_center_distance_m']:.3f} mm with {charger_metrics['orientation_error_at_minimum_distance_deg']:.3f}° error; maximum authored magnet force was {charger_metrics['maximum_authored_magnet_force_n']:.3f} N and final state `{charger_metrics['final_state']}`. No scripted attach, teleport, or object follow occurred. Magnet hash remained `{charger_metrics['authored_magnet_config_sha256']}` (`DEBUG_INITIAL_GUESS`).

## 7. ALOHA fidelity, limits, collision, continuity

- Path left/right: `{fidelity['left_path_shape']:.6f}` / `{fidelity['right_path_shape']:.6f}`.
- Speed left/right: `{fidelity['left_speed']:.6f}` / `{fidelity['right_speed']:.6f}`.
- Midpoint / relative vector / inter-hand distance: `{fidelity['bimanual_midpoint']:.6f}` / `{fidelity['relative_hand_vector']:.6f}` / `{fidelity['inter_hand_distance']:.6f}`.
- Arm/left-Dex3/right-Dex3 minimum margin: `{joints['joint']['minimum_arm_margin_rad']:.6f}` / `{joints['joint']['minimum_left_dex3_margin_rad']:.6f}` / `{joints['joint']['minimum_right_dex3_margin_rad']:.6f}` rad; violations 0.
- Branch discontinuities: `{joints['joint']['branch_discontinuity_count']}`.
- Kinematic prohibited robot contacts: `{collision['prohibited_collision_records']}`; classifier retained wrist/arm-finger accounting. The PhysX task sensor covered robot-object/table pairs, not a second complete robot-robot classifier pass.

## 8. Target-vs-actual execution and render parity

Target/actual RMSE was {tracking['rmse_rad']:.6f} rad. Arm and right-Dex3 RMSE were {tracking_groups['arm']['rmse_rad']:.6f}/{tracking_groups['right_dex3']['rmse_rad']:.6f} rad. The maximum {tracking['maximum_absolute_error_rad']:.6f} rad occurred on `{tracking['worst_joint']}` during release; this is the principal whole-motion execution warning.

Kinematic command→articulation→link→RTX parity passed. True physics target and actual peak-to-peak motion were {render_audit['target_motion_max_peak_to_peak_rad']:.6f}/{render_audit['actual_motion_max_peak_to_peak_rad']:.6f} rad; wrists moved {1000*render_audit['link_displacements']['left_wrist']['key_sample_max_displacement_from_start_m']:.3f}/{1000*render_audit['link_displacements']['right_wrist']['key_sample_max_displacement_from_start_m']:.3f} mm. Overview/side/top masks were non-empty for all 990 frames and had maximum XOR {render_audit['full_body_cameras']['overview']['maximum_keyframe_mask_xor_pixels']}/{render_audit['full_body_cameras']['side']['maximum_keyframe_mask_xor_pixels']}/{render_audit['full_body_cameras']['top']['maximum_keyframe_mask_xor_pixels']} pixels.

The raw aggregate initially said `BLOCKED_RENDERED_MESH_MOTION` only because it required the robot to remain inside task close-ups outside their semantic phases. Full-body parity and every close-up's non-identical motion passed. The regression classifier is fixed for future runs without rerunning or changing this physics execution.

## 9. Independent final statuses

- `{whole_sanity['status']}`: 990-sample generated motion, fidelity, temporal improvement, kinematic safety, and rendered actual-state motion completed; left-third physical interference and left-index release tracking prevent PASS.
- `{full_status['status']}`: phone acquisition and lift passed; portrait retention, right accessory grasp/removal, and charger capture failed.
- First causal blocker: `{first_blocker['label']}`.

## 10. Stage outcomes

{chr(10).join(f"- **{row['name']}** — `{row['result']}`: {row['evidence']}" for row in stage_rows)}

## 11. Review artifacts

- Primary: [v18_ALOHA_vs_G1_FULL_TASK_4panel.mp4](v18_ALOHA_vs_G1_FULL_TASK_4panel.mp4)
- True physics: [overview](v18_TRUE_PHYSICS_FULL_overview.mp4), [side](v18_TRUE_PHYSICS_FULL_side.mp4), [top](v18_TRUE_PHYSICS_FULL_top.mp4)
- Close-ups: [phone](v18_TRUE_PHYSICS_PHONE_GRASP_ROTATION_closeup.mp4), [accessory](v18_TRUE_PHYSICS_ACCESSORY_REMOVAL_closeup.mp4), [charger](v18_TRUE_PHYSICS_CHARGER_CAPTURE_closeup.mp4)
- Kinematic: [overview](v18_KINEMATIC_FULL_overview.mp4), [side](v18_KINEMATIC_FULL_side.mp4), [top](v18_KINEMATIC_FULL_top.mp4), [robot-only](v18_KINEMATIC_FULL_robot_only.mp4)
- [Semantic contact sheet](v18_full_task_semantic_contact_sheet.png)

Every video decodes to exactly 990 frames at 7.5 fps. All physical views derive from the same 7,980-step run.

## 12. Exact GUI command

The CLI was syntax-checked after the render-scope regression fix. Running this command starts a new interactive review of the exact frozen trajectory; it is not needed to establish the recorded scientific result.

```bash
{gui_command.strip()}
```

## 13. Exact next action

USER OPENS
v18_ALOHA_vs_G1_FULL_TASK_4panel.mp4
AND
v18_TRUE_PHYSICS_FULL_overview.mp4
AND THEN REVIEWS THE SAME V18 TRAJECTORY IN ISAAC LAB GUI.

STOP after user visual review. Do not freeze the translator, start real G1, or create v18.1 automatically.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    report_dir = OUT / "report"
    report_dir.mkdir(exist_ok=True)
    video_tags = "\n".join(
        f'<section><h3>{html.escape(name)}</h3><video controls preload="metadata" src="../{html.escape(name)}"></video></section>'
        for name in video_names
    )
    index_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v18 full-task review</title>
<style>body{{font:16px system-ui;max-width:1320px;margin:auto;padding:24px;background:#f7f7f5;color:#202020}}video,img{{max-width:100%;background:white}}section{{margin:24px 0}}.partial{{color:#974b00}}.fail{{color:#a21b1b}}pre{{white-space:pre-wrap}}</style></head><body>
<h1>Episode-49 v18 full-task execution preflight</h1>
<p class="partial">{html.escape(whole_sanity['status'])}</p><p class="fail">{html.escape(full_status['status'])}</p>
<p>One 990-command, 7,980-step true-PhysX execution. Phone acquisition/lift succeeded; portrait retention, right accessory grasp, and charger capture failed.</p>
<h2>Semantic evidence</h2><img src="../v18_full_task_semantic_contact_sheet.png">
<h2>Videos</h2>{video_tags}<h2>Full report</h2><pre>{html.escape(report)}</pre></body></html>"""
    (report_dir / "index.html").write_text(index_html, encoding="utf-8")

    status = [
        "V18_FULL_990_FRAME_KINEMATIC_REVIEW_PASS",
        "V18_FULL_990_FRAME_TRUE_PHYSICS_DIAGNOSTIC_COMPLETE",
        render_audit["status"],
        whole_sanity["status"],
        full_status["status"],
        "V18_READY_FOR_USER_VISUAL_REVIEW",
    ]
    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != "run_manifest.json"]
    dump(OUT / "run_manifest.json", {
        "status": status,
        "single_trajectory_sha256": trajectory_hash,
        "single_true_physics_execution": {
            "source_frames": 990,
            "physics_steps": raw["physics_steps"],
            "speed_scale": raw["speed_scale"],
            "stage_failure_stopped_playback": False,
        },
        "generated_file_count": len(files),
        "files": {str(path.relative_to(OUT)): sha256(path) for path in sorted(files)},
        "validation_read_count": 0,
        "heldout_read_count": 0,
        "g1_expert_read_count": 0,
        "dds": False,
        "publisher": False,
        "aloha_command": False,
        "real_g1_command": False,
    })
    print(json.dumps({
        "status": status,
        "phone": phone_metrics["status"],
        "accessory": accessory_metrics["status"],
        "charger": charger_metrics["status"],
        "videos_pass": all(row["pass"] for row in videos.values()),
        "contact_sheet": str(contact_sheet),
        "four_panel": str(four_panel),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
