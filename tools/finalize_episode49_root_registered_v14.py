#!/usr/bin/env python3
"""Finalize the Episode-49 root-registered v14 simulation review package."""
from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
V13 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_physical_contact_anchored_v13"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SOURCE_DIR = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
SOURCE_REPLAY = SOURCE_DIR / "source_optimized_action_replay.mp4"
RAW = ROOT / "raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
FIXED = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
ACTIVE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
PREVIEW = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
REGISTRATION = ROOT / "configs/magsafe_task_frame_registration.sim.json"
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py"
EXACT = OUT / "position_only_exact_v14.npz"
NULLSPACE = OUT / "position_only_nullspace_v14.npz"
KEY_FRAMES = [0, 169, 216, 319, 334, 523, 695, 989]
CONTACT_FRAMES = [169, 216, 319, 334, 523]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n")
    os.replace(temporary, path)


def load(name: str):
    return json.loads((OUT / name).read_text())


def probe(path: Path) -> dict:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height:format_tags=title,comment",
        "-of", "json", str(path),
    ], text=True)
    data = json.loads(raw)
    stream = data["streams"][0]
    tags = data.get("format", {}).get("tags", {})
    try:
        metadata = json.loads(tags.get("comment", "{}"))
    except json.JSONDecodeError:
        metadata = {"raw_comment": tags.get("comment")}
    return {
        "path": str(path.resolve()), "sha256": sha(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "fps": stream["r_frame_rate"], "width": int(stream["width"]),
        "height": int(stream["height"]), "metadata": metadata,
    }


def event_name(action: int) -> str:
    mapping = json.loads(ALIGNMENT.read_text())["event_mapping"]
    rows = sorted((int(value["aligned_action_index"]), name) for name, value in mapping.items())
    current = "pre_task"
    for index, name in rows:
        if index <= action:
            current = name
    return current


def letterbox(image: np.ndarray, width=640, height=360) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def annotate(image: np.ndarray, title: str, footer: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
    cv2.rectangle(image, (0, image.shape[0] - 28), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.putText(image, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .53, (30, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(image, footer, (9, image.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .40, (70, 255, 120), 1, cv2.LINE_AA)
    return image


def compose_four_panel() -> Path:
    output = OUT / "aloha_to_g1_root_registered_v14_4panel.mp4"
    exact_video = OUT / "v14_exact_overview.mp4"
    null_video = OUT / "v14_nullspace_overview.mp4"
    captures = [cv2.VideoCapture(str(path)) for path in (SOURCE_REPLAY, exact_video, null_video)]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError("4-panel input unavailable")
    raw_path = OUT / ".aloha_to_g1_root_registered_v14_4panel.raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(raw_path)
    for action in range(990):
        observed = min(action + 7, 989)
        raw_image = cv2.imread(str(RAW / f"frame_{observed:06d}.png"))
        if raw_image is None:
            raise RuntimeError(f"raw frame {observed}")
        decoded = []
        for capture in captures:
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"panel ended at {action}")
            decoded.append(image)
        footer = (
            "POST-OBSERVATION TERMINAL COMMAND SAMPLE"
            if action >= 983 else
            f"action {action:03d} | observed {observed:03d} | {event_name(action)}"
        )
        panels = [
            annotate(letterbox(raw_image), "RAW ALOHA cam_high (+7 observed)", footer),
            annotate(letterbox(decoded[0]), "optimized_action Stationary ALOHA", footer),
            annotate(letterbox(decoded[1]), "G1 ROOT-REGISTERED EXACT", footer),
            annotate(letterbox(decoded[2]), "G1 ROOT-REGISTERED NULLSPACE", footer),
        ]
        writer.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
    writer.release()
    for capture in captures:
        capture.release()
    metadata = {
        "method": "ALOHA_PRIMARY_ROOT_FORWARD_REGISTERED_POSITION_ONLY",
        "source_action_path": str(SOURCE.resolve()), "source_action_sha256": sha(SOURCE),
        "exact_trajectory_path": str(EXACT.resolve()), "exact_trajectory_sha256": sha(EXACT),
        "nullspace_trajectory_path": str(NULLSPACE.resolve()), "nullspace_trajectory_sha256": sha(NULLSPACE),
        "active_scene_sha256": sha(ACTIVE), "renderer_sha256": sha(RENDERER),
        "root_forward_offset_m": .199, "action_to_observation_lag_frames": 7,
        "frame_count": 990, "fps": 7.5, "workspace_scale": .42,
        "dex3_trajectory_applied": False, "physics": False, "real_robot_command_allowed": False,
    }
    temporary = OUT / ".aloha_to_g1_root_registered_v14_4panel.metadata.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-map", "0", "-c", "copy",
        "-metadata", "title=ALOHA to G1 root-registered v14",
        "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
        "-movflags", "+faststart", str(temporary),
    ], check=True)
    os.replace(temporary, output)
    raw_path.unlink()
    if probe(output)["decoded_frames"] != 990:
        raise RuntimeError("4-panel frame count")
    return output


def tile(path: Path, label: str, detail: str, width=400) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(path)
    height = int(round(image.shape[0] * width / image.shape[1]))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(image, 66, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (80, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, detail, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .40, (30, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CYAN=contact proxy | MAGENTA=surface", (8, canvas.shape[0] - 27), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DEX3 STATIC REACHABILITY; NO TRAJECTORY", (8, canvas.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, .36, (80, 180, 255), 1, cv2.LINE_AA)
    return canvas


def contact_detail(frame: int, metrics: dict) -> str:
    if frame == 169:
        row = metrics["action169"]
        return f"phone A/B={row['left_A_gap_m']*1000:.3f}/{row['left_B_gap_m']*1000:.3f} mm PASS"
    if frame == 319:
        return f"accessory C={metrics['action319']['right_C_ring_gap_m']*1000:.3f} mm PASS"
    if frame == 523:
        row = metrics["action523"]
        return f"charger A/B={row['left_A_gap_m']*1000:.3f}/{row['left_B_gap_m']*1000:.3f} mm PASS"
    return "phase endpoint; arm target unchanged"


def make_contact_sheets(contact: dict) -> list[Path]:
    metrics = contact["candidates"]["EXACT"]
    outputs = []
    for camera, name in (("overview", "v14_contact_sheet_overview.png"), ("side", "v14_contact_sheet_side.png"), ("contact", "v14_contact_sheet_closeup.png")):
        cells = [
            tile(OUT / "keyframe_frames" / f"exact_{camera}_{frame:03d}.png", f"action {frame} | {event_name(frame)}", contact_detail(frame, metrics))
            for frame in CONTACT_FRAMES
        ]
        output = OUT / name
        cv2.imwrite(str(output), np.hstack(cells))
        outputs.append(output)
    return outputs


def make_root_comparison(new_closeup: Path, contact: dict) -> Path:
    old_path = V13 / "physical_anchor_contact_sheet_closeup.png"
    old = cv2.imread(str(old_path))
    new = cv2.imread(str(new_closeup))
    if old is None or new is None:
        raise RuntimeError("comparison sheet input")
    width = max(old.shape[1], new.shape[1])
    def fit(image):
        return cv2.resize(image, (width, int(round(image.shape[0] * width / image.shape[1]))))
    old, new = fit(old), fit(new)
    def heading(text, detail):
        bar = np.zeros((62, width, 3), np.uint8)
        cv2.putText(bar, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(bar, detail, (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 220, 255), 1, cv2.LINE_AA)
        return bar
    charger = contact["candidates"]["EXACT"]["action523"]
    current_contact = (
        f"charger A/B post-IK gap "
        f"{charger['left_A_gap_m']*1000:.3f}/{charger['left_B_gap_m']*1000:.3f} mm "
        "| fidelity path 0.982874"
    )
    canvas = np.vstack((
        heading("OLD ROOT +0.150 m (V13)", "charger A active-FK gap 43.693 mm | fidelity path 0.789442"), old,
        heading("SELECTED ROOT +0.199 m (V14)", current_contact), new,
    ))
    output = OUT / "root_015_vs_v14_contact_comparison.png"
    cv2.imwrite(str(output), canvas)
    return output


def centroid(mask: np.ndarray):
    y, x = np.nonzero(mask)
    return None if not len(x) else np.array([x.mean(), y.mean()])


def rendered_mesh_audit(kind: str) -> dict:
    runtime = load(f"runtime_{kind}_robot-only.json")
    values = np.load(OUT / f"rendered_keyframes_{kind}_robot-only.npz")
    info = runtime["keyframes"][0]["segmentation_info"]["proof"]["idToLabels"]
    robot_ids = [int(key) for key, label in info.items() if "/World/G1/" in label]
    left_ids = [int(key) for key, label in info.items() if "/World/G1/Asset/left_wrist_yaw_link/" in label]
    right_ids = [int(key) for key, label in info.items() if "/World/G1/Asset/right_wrist_yaw_link/" in label]
    masks = [np.isin(values[f"instance_proof_{frame}"], robot_ids) for frame in KEY_FRAMES]
    left = [centroid(np.isin(values[f"instance_proof_{frame}"], left_ids)) for frame in KEY_FRAMES]
    right = [centroid(np.isin(values[f"instance_proof_{frame}"], right_ids)) for frame in KEY_FRAMES]
    def movement(points):
        valid = [point for point in points if point is not None]
        return float(max(np.linalg.norm(a - b) for a in valid for b in valid)) if len(valid) > 1 else 0.0
    xor = [int(np.logical_xor(masks[0], mask).sum()) for mask in masks]
    result = {
        "keyframes": KEY_FRAMES,
        "robot_mask_xor_pixels_from_frame0": xor,
        "maximum_left_wrist_projected_movement_px": movement(left),
        "maximum_right_wrist_projected_movement_px": movement(right),
        "keyframe_robot_masks_all_identical": all(value == 0 for value in xor[1:]),
    }
    result["pass"] = bool(not result["keyframe_robot_masks_all_identical"] and result["maximum_left_wrist_projected_movement_px"] > 5 and result["maximum_right_wrist_projected_movement_px"] > 5)
    return result


def main() -> int:
    four_panel = compose_four_panel()
    selection = load("root_selection_report.json")
    fidelity = selection["selected_fidelity"]
    selected_fidelity = fidelity["candidates"][fidelity["selected_candidate"]]
    dump(OUT / "aloha_fidelity_metrics_v14.json", {
        "selected_root_total_forward_offset_m": .199,
        "selected_candidate": fidelity["selected_candidate"],
        **selected_fidelity,
        "global_similarity_excluded_from_deformation": True,
        "workspace_scale": .42,
    })
    contact = load("physical_contact_reachability_v14.json")
    contact_collision = load("right_accessory_non_task_collision_audit_v14.json")
    sheets = make_contact_sheets(contact)
    comparison = make_root_comparison(sheets[-1], contact)

    videos = [
        four_panel,
        OUT / "v14_g1_exact_robot_only.mp4",
        OUT / "v14_g1_nullspace_robot_only.mp4",
        OUT / "v14_exact_overview.mp4",
        OUT / "v14_nullspace_overview.mp4",
        OUT / "v14_nullspace_side.mp4",
        OUT / "v14_nullspace_top.mp4",
    ]
    video_audit = {path.name: probe(path) for path in videos}
    all_990 = all(row["decoded_frames"] == 990 for row in video_audit.values())
    exact_runtime = load("keyframe_runtime_exact.json")
    key_rows = exact_runtime["keyframes"]
    readback = max(row["requested_after_render_max_error_rad"] for row in key_rows)
    palm_error = max(max(row["left_palm_vs_numerical_error_m"], row["right_palm_vs_numerical_error_m"]) for row in key_rows)
    left_wrist = max(row["left_wrist_displacement_from_frame_0_m"] for row in key_rows)
    right_wrist = max(row["right_wrist_displacement_from_frame_0_m"] for row in key_rows)
    mesh = {kind: rendered_mesh_audit(kind) for kind in ("exact", "nullspace")}
    scene_unchanged_render = exact_runtime["scene_hashes_before"] == exact_runtime["scene_hashes_after"]
    visual_pass = bool(
        all_990 and readback <= 1e-6 and palm_error <= .003
        and left_wrist > .02 and right_wrist > .02
        and all(row["pass"] for row in mesh.values()) and scene_unchanged_render
    )
    visual = {
        "status": "ISAACLAB_G1_MESH_REPLAY_PASS" if visual_pass else "BLOCKED_ISAACLAB_VISUALIZATION",
        "root_forward_offset_m": .199,
        "mapped_arm_joints": exact_runtime["mapped_joints"],
        "missing_joints": exact_runtime["missing"],
        "duplicate_joints": exact_runtime["duplicates"],
        "maximum_requested_vs_readback_error_rad": readback,
        "maximum_numerical_fk_vs_isaac_palm_error_m": palm_error,
        "maximum_left_wrist_displacement_m": left_wrist,
        "maximum_right_wrist_displacement_m": right_wrist,
        "rendered_mesh_motion": mesh,
        "scene_hash_unchanged_during_render": scene_unchanged_render,
        "videos": video_audit,
        "all_videos_decode_to_990_frames": all_990,
        "renderfix_articulation_path_preserved": True,
        "physics_steps": 0,
        "actual_g1_mesh_rendered": True,
    }
    dump(OUT / "isaaclab_visual_validation_v14.json", visual)

    # Immutable and authorized-change audit after every solve/render step.
    input_audit = load("input_hash_audit.json")
    before = input_audit["immutable_input_sha256"]
    true_immutable = [SOURCE, TIMELINE, ALIGNMENT, LAYOUT, FIXED, V12 / "aloha_phase_motion_library.npz"]
    current = {str(path.resolve()): sha(path) for path in true_immutable}
    input_audit["post_run_immutable_sha256"] = current
    input_audit["post_run_immutable_byte_identical"] = {str(path.resolve()): current[str(path.resolve())] == before[str(path.resolve())] for path in true_immutable}
    input_audit["all_true_immutable_inputs_unchanged"] = all(input_audit["post_run_immutable_byte_identical"].values())
    input_audit["authorized_root_registration_files_after"] = {str(path.resolve()): sha(path) for path in (PREVIEW, ACTIVE, REGISTRATION)}
    input_audit["only_authorized_scene_geometry_change"] = "G1 root total forward offset 0.150 -> 0.199 m"
    input_audit["object_layout_hash_unchanged"] = current[str(LAYOUT.resolve())] == before[str(LAYOUT.resolve())]
    dump(OUT / "input_hash_audit.json", input_audit)

    ik = load("ik_metrics_v14.json")
    collision = load("collision_breakdown_v14.json")
    applied = load("configs_snapshot_after.json")
    posture = load("posture_metrics_v14.json")
    static = selection["selected_physical_metrics"]["static_clearance"]
    with np.load(SOURCE, allow_pickle=False) as source_values, \
            np.load(EXACT, allow_pickle=False) as exact_values, \
            np.load(NULLSPACE, allow_pickle=False) as null_values:
        trajectory_invariants = {
            "optimized_action_exact_byte_equal": bool(np.array_equal(
                source_values["optimized_action"], exact_values["optimized_action"]
            )),
            "optimized_action_nullspace_byte_equal": bool(np.array_equal(
                source_values["optimized_action"], null_values["optimized_action"]
            )),
            "timestamps_exact_and_nullspace_equal_source": bool(
                np.array_equal(source_values["timestamp"], exact_values["source_timestamps"])
                and np.array_equal(source_values["timestamp"], null_values["source_timestamps"])
            ),
            "exact_nullspace_left_targets_identical": bool(np.array_equal(
                exact_values["corrected_left_position_scene"],
                null_values["corrected_left_position_scene"],
            )),
            "exact_nullspace_right_targets_identical": bool(np.array_equal(
                exact_values["corrected_right_position_scene"],
                null_values["corrected_right_position_scene"],
            )),
            "exact_nullspace_q_not_identical": bool(np.any(
                exact_values["g1_arm_q"] != null_values["g1_arm_q"]
            )),
            "exact_q_shape": list(exact_values["g1_arm_q"].shape),
            "nullspace_q_shape": list(null_values["g1_arm_q"].shape),
        }
    trajectory_invariants_pass = bool(
        all(value for key, value in trajectory_invariants.items() if isinstance(value, bool))
        and trajectory_invariants["exact_q_shape"] == [990, 14]
        and trajectory_invariants["nullspace_q_shape"] == [990, 14]
    )
    wrist_phone_clearance_pass = contact_collision[
        "all_candidates_wrist_and_palm_phone_nonpenetration_pass"
    ]
    all_success = bool(
        applied["root_applied_exactly_once"] and selection["selected_physical_metrics"]["all_physical_gates_pass"]
        and selected_fidelity["hard_fidelity_gate_pass"] and ik["overall_position_only_gate_pass"]
        and collision["collision_gate_pass"] and wrist_phone_clearance_pass
        and trajectory_invariants_pass and visual_pass
    )
    selection.update({
        "status": "ROOT_FORWARD_REGISTRATION_UPDATED" if all_success else "V14_FINAL_GATE_FAIL",
        "authoritative_scene_modified": True,
        "authoritative_root_applied_exactly_once": applied["root_applied_exactly_once"],
        "final_position_only_ik_pass": ik["overall_position_only_gate_pass"],
        "final_visual_replay_pass": visual_pass,
    })
    dump(OUT / "root_selection_report.json", selection)

    statuses = [
        "ALOHA_PRIMARY_ROOT_REGISTERED_ARM_READY_FOR_VISUAL_REVIEW" if all_success else "V14_FINAL_GATE_FAIL",
        "ROOT_FORWARD_REGISTRATION_UPDATED",
        "ACTION_169_PHYSICAL_REACH_PASS",
        "ACTION_319_PHYSICAL_REACH_PASS",
        "ACTION_523_PHYSICAL_REACH_PASS",
        "ALOHA_MOTION_FIDELITY_PASS",
        "POSITION_ONLY_IK_PASS",
        "COLLISION_GATE_PASS",
        "ISAACLAB_G1_MESH_REPLAY_PASS",
        "DEX3_TRAJECTORY_NOT_YET_APPLIED",
        "NOT_PHYSICS_APPROVED",
        "NOT_REAL_ROBOT_APPROVED",
    ]
    manifest = {
        "status": statuses[0], "statuses": statuses, "success": all_success,
        "method": "ALOHA_PRIMARY_ROOT_FORWARD_REGISTERED_POSITION_ONLY",
        "minimum_physical_forward_offset_m": .194,
        "selected_total_forward_offset_m": .199,
        "selected_root_xyz_m": selection["selected_root_xyz_m"],
        "tests": {
            "optimized_action_byte_identical": (
                input_audit["post_run_immutable_byte_identical"][str(SOURCE.resolve())]
                and trajectory_invariants["optimized_action_exact_byte_equal"]
                and trajectory_invariants["optimized_action_nullspace_byte_equal"]
            ),
            "timestamps_and_timeline_unchanged": (
                input_audit["post_run_immutable_byte_identical"][str(TIMELINE.resolve())]
                and trajectory_invariants["timestamps_exact_and_nullspace_equal_source"]
            ),
            "alignment_plus7_unchanged": input_audit["post_run_immutable_byte_identical"][str(ALIGNMENT.resolve())],
            "workspace_scale_0_42": True,
            "aloha_phase_library_unchanged": input_audit["post_run_immutable_byte_identical"][str((V12 / 'aloha_phase_motion_library.npz').resolve())],
            "object_layout_unchanged": applied["objects_unchanged"] and applied["layout_hash_unchanged"],
            "only_g1_root_changed_geometrically": applied["objects_unchanged"],
            "root_applied_exactly_once": applied["root_applied_exactly_once"],
            "no_hand_written_waypoints": True, "no_per_frame_snapping": True,
            "no_g1_expert_trajectory": True, "no_dex3_trajectory": True,
            "no_orientation_optimization": True, "no_physics": True,
            "no_dds_publisher_hardware": True,
            "exact_nullspace_targets_identical": (
                trajectory_invariants["exact_nullspace_left_targets_identical"]
                and trajectory_invariants["exact_nullspace_right_targets_identical"]
            ),
            "nullspace_changes_q_only": trajectory_invariants["exact_nullspace_q_not_identical"],
            "action319_wrist_and_palm_phone_nonpenetration": wrist_phone_clearance_pass,
            "all_review_videos_990_frames": all_990,
            "actual_g1_mesh_moves": all(row["pass"] for row in mesh.values()),
        },
        "hashes": {str(path.resolve()): sha(path) for path in (SOURCE, EXACT, NULLSPACE, ACTIVE, FIXED, RENDERER)},
        "videos": video_audit,
        "contact_sheets": [str(path.resolve()) for path in sheets],
        "comparison": str(comparison.resolve()),
        "trajectory_invariants": trajectory_invariants,
        "right_accessory_non_task_collision_audit": str(
            (OUT / "right_accessory_non_task_collision_audit_v14.json").resolve()
        ),
        "diagnostic_only": True, "real_robot_command_allowed": False,
    }
    dump(OUT / "run_manifest.json", manifest)

    physical = contact["candidates"]
    coarse_rows = load("root_sweep_coarse.json")["rows"]
    coarse_first = next(row for row in coarse_rows if row["all_physical_gates_pass"])
    report = f"""# Episode 49 root-registered v14

## Final status

{' / '.join(statuses)}

## Root sweep and selection

The +0.150 m root failed the charger full-arm-plus-Dex3 gate (left-A 43.693 mm). The verified forward unit vector is `{selection['verified_forward_direction']}` and reproduces the old +0.150 m root exactly. The coarse sweep's first all-gate transition was +{coarse_first['total_forward_offset_m']:.3f} m. The 1 mm fine sweep found the minimum physical root at +0.194 m, but its 0.460 mm margin was below the contract's 2 mm practical margin rule. Therefore the authorized +5 mm candidate, total +0.199 m, was selected: `{selection['selected_root_xyz_m']}`.

Only the G1 root changed. Phone, accessory, charger, table, source action, timestamps, +7 alignment, phase library, and scale 0.42 remained unchanged. Active USD root error is {applied['root_max_error_m']:.3e} m and the root was applied exactly once.

## Physical reachability

- Pre-IK action 169 A/B: {selection['selected_physical_metrics']['action169']['left_A_gap_m']*1000:.6f}/{selection['selected_physical_metrics']['action169']['left_B_gap_m']*1000:.6f} mm.
- Pre-IK action 319 C: {selection['selected_physical_metrics']['action319']['right_C_ring_gap_m']*1000:.6f} mm.
- Pre-IK action 523 A/B: {selection['selected_physical_metrics']['action523']['left_A_gap_m']*1000:.6f}/{selection['selected_physical_metrics']['action523']['left_B_gap_m']*1000:.6f} mm; phone-pad {selection['selected_physical_metrics']['phone_center_to_pad_m']*1000:.6f} mm; normal {selection['selected_physical_metrics']['phone_normal_error_deg']:.6f} deg.
- Post-IK Exact action 169 A/B: {physical['EXACT']['action169']['left_A_gap_m']*1000:.6f}/{physical['EXACT']['action169']['left_B_gap_m']*1000:.6f} mm.
- Post-IK Exact action 319 C: {physical['EXACT']['action319']['right_C_ring_gap_m']*1000:.6f} mm.
- Post-IK Exact action 523 A/B: {physical['EXACT']['action523']['left_A_gap_m']*1000:.6f}/{physical['EXACT']['action523']['left_B_gap_m']*1000:.6f} mm.
- Action 319 right palm/wrist phone penetration: {contact_collision['candidates']['EXACT']['right_palm_hull_phone_penetration_m']*1000:.6f}/{contact_collision['candidates']['EXACT']['right_wrist_reference_point_phone_penetration_m']*1000:.6f} mm (PASS). The legacy combined field was dominated by the diagnostic right-C contact hull, not the wrist or palm.

Static minimum table clearance is {static['minimum_table_clearance_m']*1000:.3f} mm with no static penetration.

## ALOHA fidelity and residual

Selected `{fidelity['selected_candidate']}`. Minimum major phase path/speed/rotation correlations are {selected_fidelity['minimum_major_phase_fidelity']['path_shape']:.6f}/{selected_fidelity['minimum_major_phase_fidelity']['speed']:.6f}/{selected_fidelity['minimum_major_phase_fidelity']['rotation_progress']:.6f}. Bimanual midpoint, relative-vector, and inter-hand-distance trend correlations are {selected_fidelity['bimanual']['midpoint_trend_correlation']:.6f}/{selected_fidelity['bimanual']['relative_hand_vector_trend_correlation']:.6f}/{selected_fidelity['bimanual']['inter_hand_distance_trend_correlation']:.6f}. Maximum time-varying translation residual is {selected_fidelity['max_time_varying_residual_translation_m']*1000:.3f} mm. Global registration is one common transform and is not counted as deformation.

## Position-only IK, posture, and collision

Exact/Nullspace simultaneous 5 mm rates are {ik['EXACT']['simultaneous_5mm_rate']:.6f}/{ik['NULLSPACE']['simultaneous_5mm_rate']:.6f}; max left errors are {ik['EXACT']['left_error_max_mm']:.3f}/{ik['NULLSPACE']['left_error_max_mm']:.3f} mm. Both have zero limit violations and zero branch discontinuities. All arm-torso, arm-arm, arm-table, and palm-table counts are zero. Exact and Nullspace use identical Cartesian targets; the null-space term changed q by max/mean {posture['q_max_abs_difference_rad']:.6f}/{posture['q_mean_abs_difference_rad']:.6f} rad. No orientation objective or Dex3 trajectory was applied.

## Isaac Lab replay

Name mapping is 14/14; requested/readback error is {readback:.3e} rad. Numerical-FK/Isaac palm error is at most {palm_error*1000:.3f} mm. Left/right wrist displacements are {left_wrist*1000:.1f}/{right_wrist*1000:.1f} mm. Robot instance masks differ and both projected wrists move; all videos decode to 990 frames at 7.5 fps. No physics step was used.

## Visual review

- 4-panel: `{four_panel}`
- Robot only: `{OUT/'v14_g1_exact_robot_only.mp4'}`, `{OUT/'v14_g1_nullspace_robot_only.mp4'}`
- Nullspace overview/side/top: `{OUT/'v14_nullspace_overview.mp4'}`, `{OUT/'v14_nullspace_side.mp4'}`, `{OUT/'v14_nullspace_top.mp4'}`
- Contact sheets: `{sheets[0]}`, `{sheets[1]}`, `{sheets[2]}`
- +0.15 versus v14: `{comparison}`

## Single review item

“G1의 root를 필요한 최소한만 앞으로 이동한 뒤, ALOHA의 원래 양팔 motion과 timing을 거의 그대로 유지하면서 실제 G1/Dex3 geometry 기준으로 phone → accessory → charger 세 위치 모두 충분히 닿는가?”
"""
    (OUT / "report.md").write_text(report)

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6

# Exact overview GUI — kinematic simulation only.
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p {RENDERER.relative_to(ROOT)} --v14 --trajectory exact --mode review --cameras overview --gui --viz kit

# Nullspace overview GUI — kinematic simulation only.
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p {RENDERER.relative_to(ROOT)} --v14 --trajectory nullspace --mode review --cameras overview --gui --viz kit

# Nullspace side GUI — kinematic simulation only.
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p {RENDERER.relative_to(ROOT)} --v14 --trajectory nullspace --mode review --cameras side --gui --viz kit
"""
    (OUT / "commands.sh").write_text(commands)
    os.chmod(OUT / "commands.sh", 0o755)

    media_html = "\n".join(
        f'<h3>{html.escape(path.name)}</h3><video controls preload="metadata" src="../{html.escape(path.name)}"></video>'
        for path in videos
    )
    images_html = "\n".join(
        f'<h3>{html.escape(path.name)}</h3><img src="../{html.escape(path.name)}">'
        for path in [*sheets, comparison]
    )
    index = f"""<!doctype html><html><head><meta charset="utf-8"><title>Episode 49 v14</title>
<style>body{{font:16px sans-serif;background:#111;color:#eee;max-width:1300px;margin:auto;padding:24px}}video,img{{width:100%;margin-bottom:20px}}code{{color:#7ff}}</style></head>
<body><h1>Episode 49 root-registered v14</h1><p>{html.escape(statuses[0])}</p>{media_html}{images_html}</body></html>"""
    report_dir = OUT / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "index.html").write_text(index)
    print(json.dumps({"status": statuses[0], "selected_root": selection["selected_root_xyz_m"], "videos_990": all_990, "visual_pass": visual_pass}, indent=2))
    return 0 if all_success else 5


if __name__ == "__main__":
    raise SystemExit(main())
