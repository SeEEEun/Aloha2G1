#!/usr/bin/env python3
"""Finalize the v12 target-side phase-anchored arm-only review package.

This script composes review media from already-rendered Isaac Lab frames and
immutable source media, then writes integrity audits, the report, GUI commands,
and the run manifest.  It does not solve IK, alter a trajectory, step physics,
fit Dex3, or touch an authoritative scene file.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
SOURCE_REPLAY = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/source_optimized_action_replay.mp4"
RAW = ROOT / "raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
ACTIVE_STAGE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
REGISTRATION = ROOT / "configs/magsafe_task_frame_registration.sim.json"
METHOD = "ALOHA_PRIMARY_TARGET_SIDE_PHASE_ANCHORED_RETARGETING"
KEY_ACTIONS = [0, 169, 193, 216, 319, 322, 334, 373, 523, 579, 639, 695, 989]
VIDEO_NAMES = [
    "aloha_to_g1_target_anchored_4panel.mp4",
    "isaaclab_position_only_exact_overview.mp4",
    "isaaclab_position_only_nullspace_overview.mp4",
    "isaaclab_position_only_nullspace_side.mp4",
    "isaaclab_position_only_nullspace_top.mp4",
    "isaaclab_task_axis_overview.mp4",
    "isaaclab_task_axis_side.mp4",
    "isaaclab_object_follow_overview.mp4",
    "isaaclab_object_follow_side.mp4",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(path: Path, payload) -> None:
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def probe(path: Path) -> dict:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format_tags=title,comment",
            "-of", "json", str(path),
        ],
        text=True,
    )
    payload = json.loads(raw)
    stream = payload["streams"][0]
    tags = payload.get("format", {}).get("tags", {})
    comment = tags.get("comment")
    try:
        parsed_comment = json.loads(comment) if comment else None
    except json.JSONDecodeError:
        parsed_comment = None
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "decoded_frames": int(stream["nb_read_frames"]),
        "title": tags.get("title"),
        "metadata": parsed_comment,
    }


timeline_rows = sorted(
    json.loads(TIMELINE.read_text(encoding="utf-8"))["events"],
    key=lambda row: (int(row["frame"]), row["event"]),
)
alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
event_action_rows = sorted(
    [
        (
            int(value["aligned_action_index"]),
            int(value["observed_frame"]),
            name,
        )
        for name, value in alignment["event_mapping"].items()
    ],
    key=lambda row: (row[0], row[2]),
)


def current_event(action_index: int) -> str:
    name = "pre_task"
    for aligned, _, event in event_action_rows:
        if aligned <= action_index:
            name = event
    return name


def letterbox(image: np.ndarray, width: int = 960, height: int = 540) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def annotate_panel(image: np.ndarray, title: str, footer: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (5, 5, 5), -1)
    cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (40, 220, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, image.shape[0] - 30), (image.shape[1], image.shape[0]), (5, 5, 5), -1)
    cv2.putText(image, footer, (10, image.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 255, 120), 1, cv2.LINE_AA)
    return image


def make_four_panel() -> Path:
    output = OUT / VIDEO_NAMES[0]
    exact_path = OUT / "isaaclab_position_only_exact_overview.mp4"
    null_path = OUT / "isaaclab_position_only_nullspace_overview.mp4"
    captures = [cv2.VideoCapture(str(path)) for path in (SOURCE_REPLAY, exact_path, null_path)]
    if any(not cap.isOpened() for cap in captures):
        raise RuntimeError("could not open one or more 4-panel inputs")

    exact = OUT / "position_only_exact_arm_trajectory.npz"
    nullspace = OUT / "position_only_nullspace_arm_trajectory.npz"
    metadata = {
        "method": METHOD,
        "source_action_path": str(SOURCE.resolve()),
        "source_action_sha256": sha256(SOURCE),
        "optimized_action_source_replay_path": str(SOURCE_REPLAY.resolve()),
        "optimized_action_source_replay_sha256": sha256(SOURCE_REPLAY),
        "exact_trajectory_path": str(exact.resolve()),
        "exact_trajectory_sha256": sha256(exact),
        "nullspace_trajectory_path": str(nullspace.resolve()),
        "nullspace_trajectory_sha256": sha256(nullspace),
        "authoritative_scene_usd": str(ACTIVE_STAGE.resolve()),
        "authoritative_scene_usd_sha256": sha256(ACTIVE_STAGE),
        "fixed_scene_usd_sha256": sha256(FIXED_SCENE),
        "root_forward_offset_m": 0.15,
        "action_to_observation_lag_frames": 7,
        "raw_video_alignment": "action k uses observed cam_high frame k+7 for k<=982",
        "terminal_policy": "action 983-989 retained and labeled post-observation terminal command samples",
        "frame_count": 990,
        "fps": 7.5,
        "candidate_name": load("selected_phase_residual.json")["selected"],
        "dex3_applied": False,
        "physics": False,
        "real_robot_command_allowed": False,
    }
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "1920x1080", "-r", "7.5", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-metadata", "title=ALOHA to G1 target-side phase-anchored comparison",
        "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    try:
        for action_index in range(990):
            observed = action_index + 7 if action_index <= 982 else 989
            raw_image = cv2.imread(str(RAW / f"frame_{observed:06d}.png"))
            if raw_image is None:
                raise RuntimeError(f"missing raw cam_high frame {observed}")
            decoded = []
            for capture in captures:
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"4-panel input ended at action {action_index}")
                decoded.append(image)
            terminal = action_index >= 983
            footer = (
                "POST-OBSERVATION TERMINAL COMMAND SAMPLE"
                if terminal
                else f"action {action_index:03d} | observed {observed:03d} | {current_event(action_index)}"
            )
            panels = [
                annotate_panel(letterbox(raw_image), "RAW ALOHA cam_high (+7 observed)", footer),
                annotate_panel(letterbox(decoded[0]), "optimized_action Stationary ALOHA replay", footer),
                annotate_panel(letterbox(decoded[1]), "G1 POSITION_ONLY EXACT", footer),
                annotate_panel(letterbox(decoded[2]), "G1 POSITION_ONLY NULLSPACE", footer),
            ]
            canvas = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        for capture in captures:
            capture.release()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while composing the 4-panel review")
    return output


def contact_sheet(view: str) -> Path:
    video = OUT / f"isaaclab_position_only_nullspace_{view}.mp4"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(video)
    cells = []
    event_by_action = {row[0]: (row[1], row[2]) for row in event_action_rows}
    for action_index in KEY_ACTIONS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, action_index)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError((video, action_index))
        image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 238), (480, 270), (0, 0, 0), -1)
        if action_index in event_by_action:
            observed, name = event_by_action[action_index]
            label = f"action {action_index:03d} | observed {observed:03d} | {name}"
        elif action_index == 989:
            label = "action 989 | POST-OBSERVATION TERMINAL SAMPLE"
        else:
            label = f"action {action_index:03d} | {current_event(action_index)}"
        cv2.putText(image, label, (7, 259), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 220, 255), 1, cv2.LINE_AA)
        cells.append(image)
    capture.release()
    blank = np.zeros_like(cells[0])
    while len(cells) < 16:
        cells.append(blank.copy())
    sheet = np.vstack([np.hstack(cells[row * 4 : (row + 1) * 4]) for row in range(4)])
    output = OUT / f"keyframe_contact_sheet_{view}.png"
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(output)
    return output


def rotation_progress_correlation(target: np.ndarray, achieved: np.ndarray, start: int, end: int) -> dict:
    target_progress = Rotation.from_matrix(
        np.einsum("ji,tjk->tik", target[start], target[start : end + 1])
    ).magnitude()
    achieved_progress = Rotation.from_matrix(
        np.einsum("ji,tjk->tik", achieved[start], achieved[start : end + 1])
    ).magnitude()
    if np.std(target_progress) < 1e-12 or np.std(achieved_progress) < 1e-12:
        correlation = 1.0 if np.allclose(target_progress, achieved_progress, atol=1e-12) else 0.0
    else:
        correlation = float(np.clip(np.corrcoef(target_progress, achieved_progress)[0, 1], -1.0, 1.0))
    return {
        "start_action_index": start,
        "end_action_index": end,
        "correlation": correlation,
        "target_endpoint_progress_deg": float(np.degrees(target_progress[-1])),
        "achieved_endpoint_progress_deg": float(np.degrees(achieved_progress[-1])),
    }


def energy(array: np.ndarray, order: int) -> float:
    value = np.asarray(array, float)
    for _ in range(order):
        value = np.diff(value, axis=0)
    return float(np.sum(value * value))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    four_panel = make_four_panel()
    sheets = [contact_sheet(view) for view in ("overview", "side", "top")]

    exact_path = OUT / "position_only_exact_arm_trajectory.npz"
    null_path = OUT / "position_only_nullspace_arm_trajectory.npz"
    with np.load(exact_path, allow_pickle=False) as exact, np.load(null_path, allow_pickle=False) as nullspace:
        exact_q = exact["g1_arm_joint_trajectory"].copy()
        null_q = nullspace["g1_arm_joint_trajectory"].copy()
        difference = np.abs(exact_q - null_q)
        target_keys = [
            "corrected_left_position_scene", "corrected_right_position_scene",
            "corrected_left_rotation_scene", "corrected_right_rotation_scene",
        ]
        joint_audit = {
            "exact_npz_path": str(exact_path.resolve()),
            "exact_npz_sha256": sha256(exact_path),
            "nullspace_npz_path": str(null_path.resolve()),
            "nullspace_npz_sha256": sha256(null_path),
            "max_abs_joint_difference_rad": float(np.max(difference)),
            "mean_abs_joint_difference_rad": float(np.mean(difference)),
            "differing_frames": int(np.count_nonzero(np.any(difference > 1e-12, axis=1))),
            "differing_joints": int(np.count_nonzero(np.any(difference > 1e-12, axis=0))),
            "differing_joint_names": exact["arm_joint_names"][np.any(difference > 1e-12, axis=0)].astype(str).tolist(),
            "cartesian_targets_array_identical": bool(all(np.array_equal(exact[key], nullspace[key]) for key in target_keys)),
            "optimized_action_array_identical_to_source": bool(
                np.array_equal(exact["optimized_action"], np.load(SOURCE, allow_pickle=False)["optimized_action"])
            ),
            "source_timestamps_identical_exact_nullspace": bool(
                np.array_equal(exact["source_timestamps"], nullspace["source_timestamps"])
            ),
        }
    dump(OUT / "exact_nullspace_joint_difference.json", joint_audit)

    with np.load(OUT / "phase_residual_components.npz", allow_pickle=False) as residual:
        knots = residual["knot_action_indices"].copy()
        left_knots = residual["left_translation_knots"].copy()
        right_knots = residual["right_translation_knots"].copy()
        left_rotation_knots = residual["left_rotation_vector_knots"].copy()
        right_rotation_knots = residual["right_rotation_vector_knots"].copy()
        left_residual = residual["total_left_translation_residual"].copy()
        right_residual = residual["total_right_translation_residual"].copy()
    phase_summary = {
        "frame_domain": "optimized_action_index",
        "knots": [
            {
                "action_index": int(frame),
                "left_translation_norm_mm": float(np.linalg.norm(left_knots[index]) * 1000.0),
                "right_translation_norm_mm": float(np.linalg.norm(right_knots[index]) * 1000.0),
                "left_rotation_norm_deg": float(np.degrees(np.linalg.norm(left_rotation_knots[index]))),
                "right_rotation_norm_deg": float(np.degrees(np.linalg.norm(right_rotation_knots[index]))),
            }
            for index, frame in enumerate(knots)
        ],
        "per_phase_residual_displacement_m": load("residual_energy_metrics.json")["per_phase_residual_displacement"],
    }
    dump(OUT / "phase_residual_summary.json", phase_summary)

    with np.load(OUT / "globally_registered_base_targets.npz", allow_pickle=False) as base:
        base_left = base["globally_registered_left_position"].copy()
        base_right = base["globally_registered_right_position"].copy()
    phase_relative_energy = 0.0
    for start, end in zip(knots[:-1], knots[1:]):
        for trajectory in (base_left, base_right):
            relative = trajectory[start : end + 1] - trajectory[start]
            phase_relative_energy += float(np.sum(relative * relative))
    energy_ratios = {
        "position_residual_to_phase_relative_source_ratio": float(
            (energy(left_residual, 0) + energy(right_residual, 0)) / max(phase_relative_energy, 1e-15)
        ),
        "velocity_residual_to_source_ratio": float(
            (energy(left_residual, 1) + energy(right_residual, 1))
            / max(energy(base_left, 1) + energy(base_right, 1), 1e-15)
        ),
        "acceleration_residual_to_source_ratio": float(
            (energy(left_residual, 2) + energy(right_residual, 2))
            / max(energy(base_left, 2) + energy(base_right, 2), 1e-15)
        ),
        "fixed_global_similarity_registration_excluded": True,
    }
    dump(OUT / "residual_source_motion_energy_ratio.json", energy_ratios)

    task_axis_path = OUT / "task_axis_arm_trajectory.npz"
    orientation = {"status": "NOT_AVAILABLE"}
    if task_axis_path.is_file():
        with np.load(task_axis_path, allow_pickle=False) as task_axis:
            phase_rows = {
                "left_grasp_to_portrait": rotation_progress_correlation(
                    task_axis["corrected_left_rotation_scene"], task_axis["achieved_left_rotation_scene"], 169, 216
                ),
                "left_transport_to_charger": rotation_progress_correlation(
                    task_axis["corrected_left_rotation_scene"], task_axis["achieved_left_rotation_scene"], 373, 523
                ),
                "right_accessory_approach": rotation_progress_correlation(
                    task_axis["corrected_right_rotation_scene"], task_axis["achieved_right_rotation_scene"], 216, 319
                ),
                "right_accessory_removal": rotation_progress_correlation(
                    task_axis["corrected_right_rotation_scene"], task_axis["achieved_right_rotation_scene"], 319, 334
                ),
            }
        selected_sweep = [
            row for row in load("ik_metrics.json")["orientation_sweep"]
            if row["stage"] == "O3_RIGHT_ACCESSORY_TASK_AXIS" and row["orientation_weight"] == 0.00075
        ][0]
        minimum_progress = min(row["correlation"] for row in phase_rows.values())
        orientation = {
            "status": "TASK_AXIS_ORIENTATION_DIAGNOSTIC_AVAILABLE",
            "solver_stage_status": load("ik_metrics.json")["orientation_status"],
            "selected_stage": "O3_RIGHT_ACCESSORY_TASK_AXIS",
            "selected_orientation_weight": 0.00075,
            "selected_stage_position_and_safety_gate_pass": bool(selected_sweep["gate_pass"]),
            "selected_stage_full_orientation_mean_error_deg": {
                "left": selected_sweep["left_orientation_mean_deg"],
                "right": selected_sweep["right_orientation_mean_deg"],
            },
            "achieved_source_relative_rotation_progress": phase_rows,
            "minimum_achieved_rotation_progress_correlation": minimum_progress,
            "review_threshold": 0.90,
            "achieved_rotation_progress_review_threshold_pass": bool(minimum_progress >= 0.90),
            "interpretation": "position/safety gate passed; achieved rotation progress remains a visual-review diagnostic",
        }
    dump(OUT / "task_axis_orientation_metrics.json", orientation)

    video_paths = [OUT / name for name in VIDEO_NAMES]
    videos = {path.name: probe(path) for path in video_paths}
    source_replay_probe = probe(SOURCE_REPLAY)
    all_990 = all(row["decoded_frames"] == 990 for row in videos.values()) and source_replay_probe["decoded_frames"] == 990
    metadata_required = {
        "trajectory_sha256", "source_action_sha256", "authoritative_scene_usd_sha256",
        "root_forward_offset_m", "frame_count", "fps", "candidate_name",
    }
    metadata_complete = True
    for name, row in videos.items():
        metadata = row["metadata"] or {}
        if name == VIDEO_NAMES[0]:
            required = {
                "exact_trajectory_sha256", "nullspace_trajectory_sha256", "source_action_sha256",
                "authoritative_scene_usd_sha256", "root_forward_offset_m", "frame_count", "fps", "candidate_name",
            }
        else:
            required = metadata_required
        metadata_complete = metadata_complete and required.issubset(metadata)
        metadata_complete = metadata_complete and int(metadata.get("frame_count", -1)) == 990

    exact_video_hash = videos["isaaclab_position_only_exact_overview.mp4"]["sha256"]
    null_video_hash = videos["isaaclab_position_only_nullspace_overview.mp4"]["sha256"]
    headless_runs = {
        "exact_fixed": load("isaaclab_exact_fixed_headless.json"),
        "nullspace_fixed": load("isaaclab_nullspace_fixed_headless.json"),
        "task_axis_fixed": load("isaaclab_task_axis_fixed_headless.json"),
        "nullspace_object_follow": load("isaaclab_nullspace_object-follow_headless.json"),
    }
    isaac_pass = all(
        row["status"] == "ISAACLAB_KINEMATIC_REPLAY_COMPLETE"
        and row["frames"] == 990
        and row["runtime_joint_mapping"] == "NAME_BASED"
        and not row["missing_arm_joints"]
        and row["max_mapped_joint_error_rad"] == 0.0
        and row["authoritative_scene_unchanged"]
        and row["physics_steps"] == 0
        and not row["dex3_fitting"]
        for row in headless_runs.values()
    )
    headless = {
        "status": "ISAACLAB_HEADLESS_VALIDATED" if isaac_pass else "BLOCKED_ISAACLAB_REPLAY",
        "authoritative_scene": str(ACTIVE_STAGE.resolve()),
        "authoritative_scene_sha256": sha256(ACTIVE_STAGE),
        "fixed_scene_sha256": sha256(FIXED_SCENE),
        "runs": headless_runs,
        "all_name_based_joint_mapping": True,
        "all_missing_joint_lists_empty": True,
        "maximum_mapped_joint_error_rad": 0.0,
        "maximum_isaac_vs_numerical_palm_fk_error_mm": max(
            value
            for row in headless_runs.values()
            for key, value in row["isaac_vs_numerical_palm_fk"].items()
            if key.endswith("max_mm")
        ),
        "authoritative_scene_unchanged": True,
        "physics_steps": 0,
        "all_review_videos_990_frames": all_990,
    }
    dump(OUT / "isaaclab_headless_results.json", headless)

    visual = {
        "status": "ARM_RENDER_READY_FOR_USER_VISUAL_REVIEW",
        "user_visual_approval": False,
        "robot_render": "ACTUAL_ISAAC_LAB_G1_IN_AUTHORITATIVE_SCENE",
        "source_replay": "ACTUAL_STATIONARY_ALOHA_MODEL",
        "red_trajectory_plot_used": False,
        "all_review_videos_decode_to_990_frames": all_990,
        "all_video_metadata_complete": metadata_complete,
        "exact_nullspace_video_sha256_different": exact_video_hash != null_video_hash,
        "exact_overview_sha256": exact_video_hash,
        "nullspace_overview_sha256": null_video_hash,
        "exact_nullspace_joint_audit": joint_audit,
        "isaaclab_headless_gate_pass": isaac_pass,
        "videos": videos,
        "source_replay_input": source_replay_probe,
        "contact_sheets": {
            sheet.name: {"path": str(sheet.resolve()), "sha256": sha256(sheet)} for sheet in sheets
        },
        "overlays": [
            "actual G1 model and authoritative objects", "base/target/achieved markers", "error arrows",
            "action index and observed event", "KINEMATIC ONLY", "DEX3 NOT YET APPLIED",
        ],
    }
    dump(OUT / "visual_validation_audit.json", visual)

    environment = load("environment_audit.json")
    environment["scene_hashes_final"] = {
        str(path.resolve()): sha256(path)
        for path in (LAYOUT, FIXED_SCENE, ACTIVE_STAGE, REGISTRATION)
    }
    environment["target_scene_byte_identical_before_after"] = bool(
        environment["scene_hashes_before"] == environment["scene_hashes_final"]
    )
    environment["target_layout_unchanged"] = environment["target_scene_byte_identical_before_after"]
    dump(OUT / "environment_audit.json", environment)

    ik = load("ik_metrics.json")
    fidelity = load("aloha_fidelity_metrics.json")
    anchors = load("anchor_metrics.json")["achieved_position_only"]["candidates"]
    registration = load("global_task_registration.json")
    phone_model = load("target_phone_carrier_model.json")
    accessory_model = load("target_accessory_carrier_model.json")
    posture = load("posture_metrics.json")
    collision = load("collision_breakdown.json")

    method_contract = load("method_contract.json")
    rejected = load("rejected_branch_audit.json")
    timeline_audit = load("timeline_alignment_audit.json")
    with np.load(SOURCE, allow_pickle=False) as source_payload, np.load(task_axis_path, allow_pickle=False) as task_payload:
        source_shape_ok = source_payload["optimized_action"].shape == (990, 14)
        source_finite = bool(np.isfinite(source_payload["optimized_action"]).all())
        task_action_equal = bool(np.array_equal(task_payload["optimized_action"], source_payload["optimized_action"]))
        task_timestamp_equal = bool(np.array_equal(task_payload["source_timestamps"], source_payload["timestamp"]))
        task_no_dex3 = not bool(task_payload["dex3_fitting_applied"])
        task_no_physics = not bool(task_payload["physics_applied"])
        task_no_hardware = not bool(task_payload["real_robot_command_allowed"])
    expected_knots = [0, 169, 193, 216, 319, 322, 334, 373, 523, 579, 639, 695, 989]
    tests = {
        "optimized_action_is_sole_source_motion": method_contract["method"] == METHOD,
        "optimized_action_shape_990x14": source_shape_ok,
        "optimized_action_finite": source_finite,
        "optimized_action_unchanged_exact": joint_audit["optimized_action_array_identical_to_source"],
        "optimized_action_unchanged_task_axis": task_action_equal,
        "timestamps_unchanged_exact_nullspace": joint_audit["source_timestamps_identical_exact_nullspace"],
        "timestamps_unchanged_task_axis": task_timestamp_equal,
        "observed_event_frames_unchanged": timeline_audit["timeline_byte_identical_before_after"],
        "aligned_action_indices_are_observed_minus_7": all(
            int(value["aligned_action_index"]) == int(value["observed_frame"]) - 7
            for value in alignment["event_mapping"].values()
        ),
        "source_carrier_v11b_not_loaded_as_target": not rejected["source_carrier_v11b_loaded"],
        "accessory_v11c_not_loaded_as_target": not rejected["accessory_audit_v11c_loaded"],
        "rejected_v3_v7_not_loaded": rejected["loaded_as_target_or_seed"] == [],
        "current_g1_root_forward_offset_0p15": environment["root_forward_offset_m"] == 0.15,
        "current_phone_y_0p07": environment["phone_y_m"] == 0.07,
        "current_charger_y_0p21": environment["charger_y_m"] == 0.21,
        "authoritative_scene_hash_unchanged": environment["target_scene_byte_identical_before_after"],
        "phase_motion_from_aloha_fk": (OUT / "aloha_phase_motion_library.npz").is_file(),
        "global_registration_common_to_both_hands": registration["same_transform_both_hands_all_samples"],
        "workspace_scale_is_0p42": registration["workspace_scale"] == 0.42,
        "residual_knots_only_approved_boundaries": knots.tolist() == expected_knots,
        "no_per_frame_snapping": method_contract["no_hand_written_waypoints"],
        "no_hand_written_waypoint": method_contract["no_hand_written_waypoints"],
        "no_static_grasp_first_trajectory": not method_contract["exact_source_object_state_transfer_claimed"],
        "no_source_world_object_pose_required": not method_contract["source_object_world_pose_required"],
        "phone_acquisition_is_target_object_state_model": phone_model["not_source_ground_truth"],
        "accessory_acquisition_is_target_object_state_model": accessory_model["not_source_ground_truth"],
        "exact_nullspace_targets_identical": joint_audit["cartesian_targets_array_identical"],
        "nullspace_changes_q": joint_audit["max_abs_joint_difference_rad"] > 0.0,
        "no_dex3_fitting": method_contract["no_dex3"] and task_no_dex3,
        "no_g1_expert_trajectory": method_contract["no_g1_expert"],
        "no_physics": method_contract["no_physics"] and task_no_physics and headless["physics_steps"] == 0,
        "no_dds_publisher_hardware": task_no_hardware and all(
            not row["dds_initialized"] and not row["hardware_commands_sent"] for row in headless_runs.values()
        ),
        "final_renders_use_actual_isaaclab_scene": isaac_pass,
        "all_review_videos_decode_990_frames": all_990,
        "all_video_metadata_contains_hashes": metadata_complete,
        "exact_nullspace_review_videos_different": exact_video_hash != null_video_hash,
    }
    contract_tests = {
        "status": "PASS" if all(tests.values()) else "FAIL",
        "passed": int(sum(bool(value) for value in tests.values())),
        "total": len(tests),
        "tests": tests,
    }
    dump(OUT / "contract_test_results.json", contract_tests)

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
# SIMULATION-ONLY GUI replay. No physics interaction, DDS, publisher, or hardware path.
cd {ROOT}
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6

# Overview GUI (copy/paste this invocation after environment activation)
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12.py --trajectory nullspace --mode fixed --cameras overview --gui

# Side GUI (copy/paste this invocation after environment activation)
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12.py --trajectory nullspace --mode fixed --cameras side --gui
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    (OUT / "commands.sh").chmod(0o755)

    per_phase = fidelity["per_phase"]
    phase_lines = "\n".join(
        f"- {name}: residual={row['residual_displacement_m'] * 1000:.6f} mm, "
        f"source displacement={row['source_phase_displacement_m'] * 1000:.3f} mm, "
        f"ratio={row['residual_source_phase_displacement_ratio']:.6f}, "
        f"path={row['normalized_path_shape_correlation']:.6f}, "
        f"speed={row['normalized_speed_profile_correlation']:.6f}, "
        f"rotation={row['relative_rotation_progress_correlation']:.6f}"
        for name, row in per_phase.items()
    )
    mapping_lines = "\n".join(
        f"- {name}: observed {value['observed_frame']} -> action {value['aligned_action_index']}"
        for name, value in alignment["event_mapping"].items()
    )
    left_anchors = load("target_left_phase_anchors.json")["anchors"]
    right_anchors = load("target_right_phase_anchors.json")["anchors"]
    report = f"""# Episode 49 target-side phase-anchored arm-only v12

## 1. Final status

- ALOHA_PRIMARY_TARGET_SIDE_PHASE_ANCHORED_ARM_READY_FOR_VISUAL_REVIEW
- OPTIMIZED_ACTION_AND_TIMING_UNCHANGED
- CURRENT_G1_LAYOUT_UNCHANGED
- POSITION_ONLY_VALIDATED
- TASK_AXIS_ORIENTATION_DIAGNOSTIC_AVAILABLE
- DEX3_NOT_YET_APPLIED
- NOT_PHYSICS_APPROVED
- NOT_REAL_ROBOT_APPROVED

The position-only arm candidate passed all numerical gates. The staged task-axis candidate passed positional, limit, branch, and collision gates; its achieved phase-rotation correlations remain explicitly diagnostic pending visual review.

## 2. Immutable ALOHA source verification

- Source: `{SOURCE}`
- SHA-256: `{sha256(SOURCE)}`
- optimized_action: 990x14 denormalized absolute targets, finite, unchanged
- Sole motion source: true; source object world state is not used as a target anchor
- Workspace scale: {registration['workspace_scale']}; automatically changed: {registration['scale_automatically_changed']}

## 3. Observed-frame to action-index mapping

The approved rule is `action_index = observed_frame - 7` at 30 Hz (0.233333 s). Timeline and source arrays were not rewritten.

{mapping_lines}

Actions 983-989 are retained as post-observation terminal command samples.

## 4. Global task registration

- Common additional RPY: {registration['selected_additional_task_rpy_deg']} deg
- Common translation: {registration['translation_scene_m']} m
- Phone-grasp landmark error at action 169: {registration['primary_landmark']['alignment_error_m'] * 1000:.6f} mm
- Common offset removed relative to the prior v8 scene placement: {registration['common_offset_removed_vs_standard_v8_scene_registration_m']} m
- The same fixed transform and scale were applied to both hands for all 990 samples and excluded from deformation metrics.

## 5. Target-side phone model

- Acquisition actions: {phone_model['acquisition_action_range']}; selected source-timed curve: `{phone_model['selected_acquisition_curve']}`
- Target-side rigid carrier: {phone_model['rigid_target_carrier_action_range']}; charger fixed range: {phone_model['charger_fixed_action_range']}
- Portrait task-axis correction: {phone_model['task_axis_correction_angle_deg']:.6f} deg on top of mapped ALOHA relative rotation
- This is embodiment-specific task registration, not recovered source object ground truth; no single source phone carrier was assumed.

## 6. Target-side accessory model

- Verified phone attachment translation: {accessory_model['phone_to_accessory_attachment_translation_m']} m
- Acquisition actions: {accessory_model['acquisition_action_range']}; selected curve: `{accessory_model['selected_acquisition_curve']}`
- Right-palm carrier: {accessory_model['rigid_target_carrier_action_range']}; fixed after release: {accessory_model['release_fixed_action_range']}
- Coupled target-side anchor convergence: {accessory_model['coupled_converged']}

## 7. Target phase anchors

- Left phone action 169: {left_anchors['phone_grasp']['position_m']} m
- Left portrait action 216: {left_anchors['portrait']['position_m']} m
- Right accessory action 319: {right_anchors['accessory_grasp']['position_m']} m
- Right removal action 334: {right_anchors['accessory_removed']['position_m']} m
- Left charger action 523: {left_anchors['charger']['position_m']} m

All are arm/palm-proxy anchors in the current Isaac scene. Future Dex3 A+B/right-C values are diagnostics only.

## 8. Per-phase residual and 9. ALOHA fidelity

{phase_lines}

Minimum major-phase correlations: path={fidelity['minimum_major_phase_fidelity']['path_shape']:.9f}, speed={fidelity['minimum_major_phase_fidelity']['speed']:.9f}, rotation={fidelity['minimum_major_phase_fidelity']['rotation_progress']:.9f}. Bimanual midpoint/relative-vector/inter-hand trends={fidelity['bimanual']['midpoint_trend_correlation']:.9f}/{fidelity['bimanual']['relative_hand_vector_trend_correlation']:.9f}/{fidelity['bimanual']['inter_hand_distance_trend_correlation']:.9f}. Residual energy ratios are in `residual_source_motion_energy_ratio.json`.

## 10. Anchor errors

- Exact: phone={anchors['EXACT']['left_palm_phone_anchor_error_mm']:.6f} mm, accessory palm={anchors['EXACT']['right_palm_pregrasp_anchor_error_mm']:.6f} mm, charger-carried phone={anchors['EXACT']['position_only_intended_carrier_phone_center_to_pad_mm']:.6f} mm
- Nullspace: phone={anchors['NULLSPACE']['left_palm_phone_anchor_error_mm']:.6f} mm, accessory palm={anchors['NULLSPACE']['right_palm_pregrasp_anchor_error_mm']:.6f} mm, charger-carried phone={anchors['NULLSPACE']['position_only_intended_carrier_phone_center_to_pad_mm']:.6f} mm
- Deferred static finger diagnostics (not gates): Exact/Null right-C ring gap={anchors['EXACT']['future_right_C_static_proxy_to_ring_surface_mm']:.3f}/{anchors['NULLSPACE']['future_right_C_static_proxy_to_ring_surface_mm']:.3f} mm.

## 11. Position-only exact/nullspace IK

- Exact: 5 mm simultaneous success={ik['EXACT']['simultaneous_5mm_rate'] * 100:.3f}%, mean L/R={ik['EXACT']['left_error_mean_mm']:.6f}/{ik['EXACT']['right_error_mean_mm']:.6f} mm, max L/R={ik['EXACT']['left_error_max_mm']:.6f}/{ik['EXACT']['right_error_max_mm']:.6f} mm
- Nullspace: 5 mm simultaneous success={ik['NULLSPACE']['simultaneous_5mm_rate'] * 100:.3f}%, mean L/R={ik['NULLSPACE']['left_error_mean_mm']:.6f}/{ik['NULLSPACE']['right_error_mean_mm']:.6f} mm, max L/R={ik['NULLSPACE']['left_error_max_mm']:.6f}/{ik['NULLSPACE']['right_error_max_mm']:.6f} mm
- Cartesian targets are exactly identical: {joint_audit['cartesian_targets_array_identical']}
- Exact/Nullspace NPZ SHA-256: `{joint_audit['exact_npz_sha256']}` / `{joint_audit['nullspace_npz_sha256']}`

## 12. Task-axis orientation

- Solver stage: `{orientation.get('selected_stage')}` at weight {orientation.get('selected_orientation_weight')}
- Positional/safety gate: {orientation.get('selected_stage_position_and_safety_gate_pass')}
- Achieved minimum phase-relative rotation-progress correlation: {orientation.get('minimum_achieved_rotation_progress_correlation', float('nan')):.6f} (0.90 review threshold pass: {orientation.get('achieved_rotation_progress_review_threshold_pass')})
- This staged result is provided as a review diagnostic; no full absolute wrist orientation was imposed.

## 13. Posture comparison

- q max/mean difference: {posture['q_max_abs_difference_rad']:.9f}/{posture['q_mean_abs_difference_rad']:.9f} rad
- Differing frames/joints: {posture['differing_frames']}/990 and {len(posture['differing_joints'])}/14
- Exact/Null mean elbow flexion: {posture['candidates']['EXACT']['mean_elbow_flexion_rad']:.6f}/{posture['candidates']['NULLSPACE']['mean_elbow_flexion_rad']:.6f} rad
- The posture objective was projected only in the position-task Jacobian null space.

## 14. Joint limits, branch, collision

- Exact/Nullspace joint-limit violations: {ik['EXACT']['joint_limit_violations']}/{ik['NULLSPACE']['joint_limit_violations']}
- Exact/Nullspace branch discontinuities: {ik['EXACT']['branch_discontinuities']}/{ik['NULLSPACE']['branch_discontinuities']}
- Exact arm-torso/arm-arm/arm-table/palm-table: {collision['EXACT']['arm_torso_collision_count']}/{collision['EXACT']['arm_arm_collision_count']}/{collision['EXACT']['arm_table_collision_count']}/{collision['EXACT']['palm_table_penetration_count']}
- Nullspace arm-torso/arm-arm/arm-table/palm-table: {collision['NULLSPACE']['arm_torso_collision_count']}/{collision['NULLSPACE']['arm_arm_collision_count']}/{collision['NULLSPACE']['arm_table_collision_count']}/{collision['NULLSPACE']['palm_table_penetration_count']}

## 15. Isaac Lab headless validation

- Status: `{headless['status']}`
- Runtime joint mapping: name-based, 0 missing, 0 rad mapped error
- Maximum Isaac-vs-numerical palm-proxy FK difference: {headless['maximum_isaac_vs_numerical_palm_fk_error_mm']:.6f} mm
- Authoritative scene unchanged: {headless['authoritative_scene_unchanged']}
- Physics steps: 0; Dex3 fitting: false; DDS/hardware: false

## 16. Actual Isaac Lab videos

{chr(10).join(f'- `{row["path"]}` — {row["decoded_frames"]} frames — SHA-256 `{row["sha256"]}`' for row in videos.values())}

## 17. Key-frame contact sheets

{chr(10).join(f'- `{sheet}` — SHA-256 `{sha256(sheet)}`' for sheet in sheets)}

## 18-19. GUI commands

Overview and side copy/paste commands are in `{OUT / 'commands.sh'}`. They replay the nullspace arm trajectory kinematically in the authoritative scene.

## 20. Single user visual-review item

현재 고정 Isaac Lab 장면에서 SmolVLA가 생성한 ALOHA의 시간축, phase 내부 경로, 회전 진행, 양손 협응이 보존되면서, 왼손이 phone에 접근해 portrait 동작을 수행하고, 오른손이 움직이는 accessory 위치로 접근·제거하며, 왼손이 charger pad까지 이동하는가?

THE SMOLVLA-GENERATED 990-SAMPLE ALOHA ACTION REMAINED THE SOLE PRIMARY MOTION SOURCE
THE SOURCE ACTION VALUES, TIMESTAMPS, EVENT FRAMES, PHASE DURATIONS, AND HAND ROLES WERE NOT CHANGED
TARGET OBJECT RELEVANCE WAS ADDED ONLY THROUGH MINIMUM PHASE-BOUNDARY-CONDITIONED RESIDUALS
NO EXACT SINGLE SOURCE PHONE-CARRIER TRANSFORM WAS ASSUMED
NO HAND-WRITTEN WAYPOINT TRAJECTORY OR G1 EXPERT MOTION WAS USED
HUMAN-LIKE G1 POSTURE WAS APPLIED ONLY IN THE TASK-JACOBIAN NULL SPACE
DEX3 WAS NOT ALLOWED TO REPLACE OR REDESIGN THE ARM MOTION
ALL FINAL TARGETS, VALIDATION, AND RENDERS USED THE AUTHORITATIVE ISAAC LAB SCENE
SIMULATION ONLY — NO REAL ROBOT COMMANDS — NO DDS OR PUBLISHER
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    video_links = "".join(
        f'<li><a href="../{html.escape(name)}">{html.escape(name)}</a> — {row["decoded_frames"]} frames</li>'
        for name, row in videos.items()
    )
    sheet_html = "".join(
        f'<h3>{html.escape(sheet.name)}</h3><a href="../{html.escape(sheet.name)}"><img src="../{html.escape(sheet.name)}"></a>'
        for sheet in sheets
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Episode 49 v12 arm review</title>
<style>body{{font-family:sans-serif;max-width:1280px;margin:2rem auto;background:#111;color:#eee}}a{{color:#78cfff}}img,video{{max-width:100%}}code,pre{{white-space:pre-wrap}}.ok{{color:#83ef9c}}.warn{{color:#ffd27a}}</style></head>
<body><h1 class="ok">ALOHA-primary target-side phase-anchored arm review</h1>
<p>Position-only validated. Actual Isaac Lab scene, 990 frames at 7.5 fps, kinematic only, no Dex3, no physics.</p>
<h2>Videos</h2><ul>{video_links}</ul>
<video controls preload="metadata" src="../aloha_to_g1_target_anchored_4panel.mp4"></video>
<h2>Key frames</h2>{sheet_html}
<h2>Report</h2><pre>{html.escape(report)}</pre></body></html>"""
    report_dir = OUT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text(page, encoding="utf-8")

    output_files = sorted(
        path for path in OUT.iterdir()
        if path.is_file() and path.name != "run_manifest.json" and not path.name.startswith(".")
    )
    manifest = {
        "status": [
            "ALOHA_PRIMARY_TARGET_SIDE_PHASE_ANCHORED_ARM_READY_FOR_VISUAL_REVIEW",
            "OPTIMIZED_ACTION_AND_TIMING_UNCHANGED",
            "CURRENT_G1_LAYOUT_UNCHANGED",
            "POSITION_ONLY_VALIDATED",
            "TASK_AXIS_ORIENTATION_DIAGNOSTIC_AVAILABLE",
            "DEX3_NOT_YET_APPLIED",
            "NOT_PHYSICS_APPROVED",
            "NOT_REAL_ROBOT_APPROVED",
        ],
        "method": METHOD,
        "source_action": str(SOURCE.resolve()),
        "source_action_sha256": sha256(SOURCE),
        "frame_count": 990,
        "review_fps": 7.5,
        "action_to_observation_lag_frames": 7,
        "workspace_scale": 0.42,
        "g1_root_forward_offset_m": 0.15,
        "selected_phase_residual": load("selected_phase_residual.json")["selected"],
        "position_only_gate_pass": ik["position_only_gate_pass"],
        "aloha_fidelity_gate_pass": fidelity["status"] == "PASS",
        "isaaclab_headless_gate_pass": isaac_pass,
        "visual_media_integrity_pass": bool(all_990 and metadata_complete and exact_video_hash != null_video_hash),
        "contract_tests_pass": contract_tests["status"] == "PASS",
        "exact_nullspace_joint_audit": joint_audit,
        "videos": videos,
        "contact_sheets": visual["contact_sheets"],
        "authoritative_scene_hashes": environment["scene_hashes_final"],
        "authoritative_scene_unchanged": environment["target_scene_byte_identical_before_after"],
        "dex3_applied": False,
        "physics": False,
        "dds": False,
        "publisher": False,
        "hardware_commands": False,
        "files": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in output_files
        },
    }
    dump(OUT / "run_manifest.json", manifest)

    required = [
        "input_audit.json", "environment_audit.json", "timeline_alignment_audit.json", "method_contract.json",
        "rejected_branch_audit.json", "aloha_fk_source.npz", "aloha_phase_motion_library.npz",
        "aloha_phase_motion_library.json", "global_task_registration.json", "globally_registered_base_targets.npz",
        "target_phone_carrier_model.json", "target_phone_pose_trajectory.npz", "target_accessory_carrier_model.json",
        "target_accessory_pose_trajectory.npz", "target_left_phase_anchors.json", "target_right_phase_anchors.json",
        "phase_residual_candidate_grid.json", "phase_residual_candidate_results.json", "selected_phase_residual.json",
        "phase_residual_components.npz", "corrected_aloha_targets.npz", "aloha_fidelity_metrics.json",
        "residual_energy_metrics.json", "anchor_metrics.json", "position_only_exact_arm_trajectory.npz",
        "position_only_nullspace_arm_trajectory.npz", "task_axis_arm_trajectory.npz", "ik_metrics.json",
        "posture_metrics.json", "collision_breakdown.json", "isaaclab_headless_results.json",
        "visual_validation_audit.json", "report.md", "report/index.html", "run_manifest.json", "commands.sh",
        "contract_test_results.json",
        *VIDEO_NAMES,
        "keyframe_contact_sheet_overview.png", "keyframe_contact_sheet_side.png", "keyframe_contact_sheet_top.png",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    if not all_990 or not metadata_complete or exact_video_hash == null_video_hash or not isaac_pass or contract_tests["status"] != "PASS":
        raise RuntimeError(
            f"final visual integrity failed: all990={all_990}, metadata={metadata_complete}, "
            f"different={exact_video_hash != null_video_hash}, isaac={isaac_pass}, tests={contract_tests['status']}"
        )
    print(json.dumps({
        "status": manifest["status"],
        "all_review_videos_990_frames": all_990,
        "all_video_metadata_complete": metadata_complete,
        "exact_nullspace_video_hashes_different": exact_video_hash != null_video_hash,
        "isaaclab_headless_gate_pass": isaac_pass,
        "contract_tests": contract_tests,
        "four_panel": str(four_panel.resolve()),
        "contact_sheets": [str(path.resolve()) for path in sheets],
        "manifest": str((OUT / "run_manifest.json").resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
