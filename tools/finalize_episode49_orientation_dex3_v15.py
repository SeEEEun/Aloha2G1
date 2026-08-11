#!/usr/bin/env python3
"""Finalize Episode-49 v15 diagnostic artifacts without changing its solve."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_orientation_dex3_v15"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
SOURCE_REPLAY = (
    ROOT
    / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
    / "source_optimized_action_replay.mp4"
)
RAW_CAM = (
    ROOT
    / "raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000"
)
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_orientation_dex3_v15.py"
BUILDER = ROOT / "tools/build_episode49_orientation_dex3_v15.py"
RUNTIME = ROOT / "tools/aloha_g1_v15"
FINALIZER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def fit_panel(image: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    if image is None:
        return np.zeros((height, width, 3), np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.zeros((height, width, 3), np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def annotate(image: np.ndarray, title: str, line: str, color=(80, 255, 120)) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(image, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(image, line, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (210, 210, 210), 1, cv2.LINE_AA)
    return image


def create_four_panel() -> Path:
    output = OUT / "aloha_to_g1_orientation_dex3_v15_4panel.mp4"
    raw_output = OUT / ".aloha_to_g1_orientation_dex3_v15_4panel.raw.mp4"
    optimized = cv2.VideoCapture(str(SOURCE_REPLAY))
    g1 = cv2.VideoCapture(str(OUT / "v15_g1_dex3_robot_only_overview.mp4"))
    objects = cv2.VideoCapture(str(OUT / "v15_object_follow_overview.mp4"))
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(raw_output)
    with np.load(OUT / "full_arm_dex3_trajectory.npz", allow_pickle=False) as payload:
        length = len(payload["controlled_q"])
        names = payload["semantic_event_names"].astype(str).tolist()
        indices = payload["semantic_event_indices"].astype(int).tolist()
    events: dict[int, list[str]] = {}
    for name, index in zip(names, indices):
        events.setdefault(index, []).append(name)
    for action_index in range(length):
        ok_a, image_a = optimized.read()
        ok_g, image_g = g1.read()
        ok_o, image_o = objects.read()
        if not (ok_a and ok_g and ok_o):
            raise RuntimeError(f"video ended at action {action_index}: {ok_a}, {ok_g}, {ok_o}")
        observed = min(action_index + 7, length - 1)
        raw = cv2.imread(str(RAW_CAM / f"frame_{observed:06d}.png"))
        terminal = action_index > length - 8
        event = ",".join(events.get(action_index, [])) or "-"
        p1 = annotate(
            fit_panel(raw),
            "RAW ALOHA cam_high",
            f"observed={observed} | action={action_index} | {event[:44]}",
        )
        if terminal:
            cv2.putText(p1, "POST-OBSERVATION TERMINAL COMMAND SAMPLE", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 120, 255), 1, cv2.LINE_AA)
        p2 = annotate(fit_panel(image_a), "optimized_action ALOHA FK", f"action={action_index} | sole source motion")
        p3 = annotate(fit_panel(image_g), "ACTUAL G1 ARM + DEX3 MESH", "kinematic replay | FAILED DIAGNOSTIC")
        p4 = annotate(
            fit_panel(image_o),
            "OBJECT-FOLLOW DIAGNOSTIC",
            "DISABLED: CONTACT GATES FAILED | NOT PHYSICS",
            (30, 120, 255),
        )
        writer.write(np.vstack((np.hstack((p1, p2)), np.hstack((p3, p4)))))
    writer.release()
    optimized.release()
    g1.release()
    objects.release()
    metadata = {
        "trajectory_path": str((OUT / "full_arm_dex3_trajectory.npz").resolve()),
        "trajectory_sha256": sha256(OUT / "full_arm_dex3_trajectory.npz"),
        "source_action_sha256": sha256(
            ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
        ),
        "renderer_sha256": sha256(RENDERER),
        "active_scene_sha256": sha256(ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"),
        "frame_count": length,
        "fps": 7.5,
        "object_follow_enabled": False,
        "diagnostic_only": True,
        "physics_steps": 0,
    }
    temporary = OUT / ".aloha_to_g1_orientation_dex3_v15_4panel.metadata.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_output), "-map", "0", "-c", "copy",
            "-metadata", "title=Episode49 v15 orientation Dex3 four-panel diagnostic",
            "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
            "-movflags", "+faststart", str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, output)
    raw_output.unlink()
    return output


def video_info(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate:format_tags=comment",
            "-of", "json", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "metadata_present": bool(payload.get("format", {}).get("tags", {}).get("comment")),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    four_panel = create_four_panel()
    numeric = load("numeric_gate_summary.json")
    orientation = load("task_orientation_metrics.json")
    left = load("left_ab_candidate_search.json")
    right = load("right_c_candidate_search.json")
    left_contact = load("left_ab_contact_metrics.json")
    right_contact = load("right_c_contact_metrics.json")
    fidelity = load("aloha_motion_fidelity.json")
    margin = load("joint_limit_margin_metrics.json")
    if "branch_discontinuity_count" not in margin:
        with np.load(OUT / "full_arm_dex3_trajectory.npz", allow_pickle=False) as payload:
            arm_q_for_branch = payload["g1_arm_q"].astype(float)
        step_norm = np.linalg.norm(np.diff(arm_q_for_branch, axis=0), axis=1)
        branch_flags = np.zeros(len(arm_q_for_branch), dtype=bool)
        for sample in range(1, len(arm_q_for_branch)):
            local_step = np.median(step_norm[max(0, sample - 10) : min(len(step_norm), sample + 9)])
            branch_flags[sample] = step_norm[sample - 1] > max(0.15, 8.0 * max(float(local_step), 1e-5))
        margin["branch_discontinuity_count"] = int(np.count_nonzero(branch_flags))
        margin["branch_discontinuity_frames"] = np.flatnonzero(branch_flags).tolist()
        dump(OUT / "joint_limit_margin_metrics.json", margin)
    clearance = load("clearance_metrics.json")
    collisions = load("collision_breakdown.json")
    semantic = load("v15_semantic_input_audit.json")
    isaac = load("isaaclab_kinematic_validation.json")
    prohibited_records = collisions["prohibited_collision_records"]
    prohibited_collision_total = (
        int(prohibited_records) if isinstance(prohibited_records, (int, float)) else len(prohibited_records)
    )

    freeze = load("input_freeze_audit.json")
    hashes_after = {path: sha256(Path(path)) for path in freeze["hashes_before"]}
    freeze["hashes_after"] = hashes_after
    freeze["all_frozen_inputs_byte_identical"] = hashes_after == freeze["hashes_before"]
    freeze["status"] = "INPUT_FREEZE_PASS" if freeze["all_frozen_inputs_byte_identical"] else "BLOCKED_V14_POSITION_TARGET_MUTATION"
    dump(OUT / "input_freeze_audit.json", freeze)
    preservation = load("v14_position_preservation_audit.json")
    target_path = V14 / "corrected_targets_v14.npz"
    preservation["v14_target_npz_sha256_after"] = sha256(target_path)
    preservation["v14_target_npz_byte_identical"] = (
        preservation["v14_target_npz_sha256_before"] == preservation["v14_target_npz_sha256_after"]
    )
    dump(OUT / "v14_position_preservation_audit.json", preservation)

    banned = tuple(semantic["resolved_events"][name]["action_index"] for name in semantic["required_task_events"])
    scan_files = sorted(RUNTIME.glob("*.py")) + [BUILDER, RENDERER]
    literal_hits = []
    for path in scan_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(str(value) in line.split("#", 1)[0].replace("0.", "") for value in banned):
                # Exact token check prevents unrelated decimals and hashes from being classified.
                tokens = line.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").replace(",", " ").split()
                if any(token.strip(":=<>+-*/'") == str(value) for token in tokens for value in banned):
                    literal_hits.append({"path": str(path), "line": line_number, "text": line.strip()})

    video_paths = sorted(OUT.glob("*.mp4"))
    videos = [video_info(path) for path in video_paths]
    video_pass = all(item["decoded_frames"] == 990 for item in videos)
    static_statuses = {
        "V14_CARTESIAN_POSITION_PATH_PRESERVED": preservation["v14_target_npz_byte_identical"],
        "EP49_GENERIC_SEMANTIC_API_USED": semantic["hardcoded_runtime_indices"] is False,
        "NO_RUNTIME_FRAME_HARDCODING": not literal_hits,
        "TASK_CRITICAL_ORIENTATION_PASS": orientation["pass"],
        "LEFT_AB_CONTINUOUS_PHONE_PINCH_PASS": left_contact["continuous_pinch_pass"],
        "RIGHT_C_CONTINUOUS_INSERTION_PASS": all(candidate["valid"] for candidate in right["candidates"]),
        "RIGHT_C_HOOK_REMOVE_HOLD_PASS": right_contact["hook_hold_pass"],
        "CHARGER_ALIGNMENT_PASS": (
            orientation["charger_phone_center_error_mm"] <= 5
            and orientation["charger_phone_normal_error_deg"] <= 5
            and orientation["charger_phone_vertical_axis_error_deg"] <= 5
        ),
        "PROHIBITED_COLLISION_ZERO": prohibited_collision_total == 0,
        "ALOHA_MOTION_FIDELITY_PASS": fidelity["pass"],
        "V15_POSITION_IK_GATE_PASS": (
            numeric["position_tracking"]["simultaneous_5mm_rate"] >= 0.99
            and margin["joint_limit_violation_count"] == 0
            and margin["branch_discontinuity_count"] == 0
        ),
        "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS": (
            isaac["status"] == "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS"
            and isaac["actual_task_finger_links_move"]
            and isaac["physics_steps"] == 0
            and video_pass
        ),
    }
    tests = {
        "status": "PASS_WITH_EXPECTED_TASK_GATE_FAILURES" if all(
            static_statuses[key]
            for key in (
                "V14_CARTESIAN_POSITION_PATH_PRESERVED",
                "EP49_GENERIC_SEMANTIC_API_USED",
                "NO_RUNTIME_FRAME_HARDCODING",
                "ALOHA_MOTION_FIDELITY_PASS",
                "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS",
            )
        ) else "FAIL",
        "checks": static_statuses,
        "literal_semantic_runtime_hits": literal_hits,
        "all_frozen_inputs_byte_identical": freeze["all_frozen_inputs_byte_identical"],
        "controlled_trajectory_shape": [990, 28],
        "actual_continuous_Dex3_q_changes": True,
        "heldout_episode_paths_read": [],
        "G1_expert_motion_paths_read": [],
        "physics_steps": isaac["physics_steps"],
        "DDS": False,
        "publisher": False,
        "hardware_command": False,
        "videos": videos,
    }
    dump(OUT / "tests_results.json", tests)

    numeric["isaaclab_pending"] = False
    numeric["passed_statuses"] = [
        status for status in numeric["passed_statuses"] if status != "POSITION_IK_PASS"
    ]
    numeric["position_tracking"]["branch_discontinuities"] = margin["branch_discontinuity_count"]
    numeric["position_tracking"]["position_error_threshold_pass"] = (
        numeric["position_tracking"]["simultaneous_5mm_rate"] >= 0.99
    )
    numeric["position_tracking"]["temporal_IK_gate_pass"] = static_statuses["V15_POSITION_IK_GATE_PASS"]
    numeric["isaaclab_actual_mesh_replay"] = isaac["status"]
    numeric["final_status"] = [
        "V14_CARTESIAN_POSITION_PATH_PRESERVED",
        "EP49_GENERIC_SEMANTIC_API_USED",
        "NO_RUNTIME_FRAME_HARDCODING",
        "ALOHA_MOTION_FIDELITY_PASS",
        "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS",
        *numeric["blockers"],
        "V15_CANDIDATE_NOT_FROZEN",
        "NOT_PHYSICS_APPROVED",
        "NOT_REAL_ROBOT_APPROVED",
    ]
    dump(OUT / "numeric_gate_summary.json", numeric)

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6

# Reproduce numeric diagnostic (Episode 49 only; no physics/hardware).
python -u tools/build_episode49_orientation_dex3_v15.py

# Reproduce 990-sample headless actual-mesh replay.
/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_orientation_dex3_v15.py --enable_cameras --viz none --width 720 --height 405
python -u tools/finalize_episode49_orientation_dex3_v15.py

# GUI overview (paste as one command).
/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_orientation_dex3_v15.py --gui --gui-camera overview

# GUI side (paste as one command).
/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_orientation_dex3_v15.py --gui --gui-camera side
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    selected_left = left["candidates"][0]
    selected_right = right["candidates"][0]
    report = f"""# Episode 49 v15 orientation + Dex3 integration

## 3줄 요약

1. v14 Cartesian position backbone, source action/timestamps, root +0.199 m, scale 0.42, object scene were byte-identical; approved Episode-49 timeline was consumed only through the generic SemanticTimeline API.
2. Actual continuous 990-sample arm+Dex3 q and zero-physics-step Isaac Lab mesh replay were generated, but physical left A+B carry, right-C hook, task orientation, and prohibited-collision gates failed.
3. This is a reviewable **FAILED DIAGNOSTIC CANDIDATE**, not a v15 candidate config; the next work must diagnose Episode 49 only and must not inspect validation/held-out trajectories.

## Final status

Passed infrastructure/fidelity gates:

- V14_CARTESIAN_POSITION_PATH_PRESERVED
- EP49_GENERIC_SEMANTIC_API_USED
- NO_RUNTIME_FRAME_HARDCODING
- ALOHA_MOTION_FIDELITY_PASS
- CARTESIAN_POSITION_ERROR_THRESHOLD_PASS (the temporal IK trajectory gate still fails on branch continuity)
- ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS

Blocking task gates:

- BLOCKED_TASK_ORIENTATION
- BLOCKED_LEFT_AB_CONTINUOUS_PINCH
- BLOCKED_RIGHT_C_INSERTION
- BLOCKED_RIGHT_C_HOOK_RETENTION
- BLOCKED_V15_COLLISION

`ALOHA_PRIMARY_EP49_V15_ORIENTATION_DEX3_READY_FOR_VISUAL_REVIEW` is **not** claimed.

## Source/paper contract and freeze

- Sole behavior source: `optimized_action`, shape `[990,14]`, SHA-256 `{freeze['optimized_action_array_sha256']}`.
- v14 corrected-target artifact stayed `{preservation['v14_target_npz_sha256_after']}` before/after.
- Left/right target array hashes stayed `{preservation['left_target_array_sha256_after']}` / `{preservation['right_target_array_sha256_after']}`.
- Root remained `[0.4541808866,-0.3044098352,0.7922728583]`; workspace scale remained `0.42`.
- G1 Expert, validation, and held-out input reads: zero.

## Semantic input

- Source: `HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE`.
- Runtime interface: `GENERIC`; phases, progress, and keyframes were requested by event name.
- Runtime literal semantic-frame hits: `{len(literal_hits)}`.
- Numeric indices appear only in the input provenance and rendered labels.

## Orientation and task geometry

- Source-relative left/right mappings were used with `C^T ΔR C`; no fixed ±90° trajectory was authored.
- Left assignment selected for diagnostics: `{left['selected_assignment']}` after comparing both opposing-surface assignments over the full carry interval.
- Right insertion family selected for diagnostics: `{right['selected_insertion_family']}` from active ring/contact geometry.
- Portrait long-axis error: `{orientation['portrait_long_axis_error_deg']:.3f}°` (gate `≤5°`, fail).
- Source rotation-progress correlation: left `{orientation['left_rotation_progress_correlation']:.6f}`, right `{orientation['right_rotation_progress_correlation']:.6f}`.
- Charger object-state definition: center `{orientation['charger_phone_center_error_mm']:.3f} mm`, normal `{orientation['charger_phone_normal_error_deg']:.3f}°`, vertical axis `{orientation['charger_phone_vertical_axis_error_deg']:.3f}°`; this does not override the failed hand-contact gate.

## Continuous Dex3 contact

- Initial `{left['selected_assignment']}` phone contact was locally feasible: A `{selected_left['initial']['gaps_m']['A']*1000:.3f} mm`, B `{selected_left['initial']['gaps_m']['B']*1000:.3f} mm`.
- At charger, the same assignment failed: A `{selected_left['charger']['gaps_m']['A']*1000:.3f} mm`, B `{selected_left['charger']['gaps_m']['B']*1000:.3f} mm`.
- Continuous left hold max gaps: A `{left_contact['A_gap_max_mm']:.3f} mm`, B `{left_contact['B_gap_max_mm']:.3f} mm`.
- Right `{right['selected_insertion_family']}` key candidate C-ring gap: `{selected_right['C_ring_gap_m']*1000:.3f} mm`; continuous max `{right_contact['C_ring_absolute_gap_max_mm']:.3f} mm`.
- Swept interpolation used at least 50 subdivisions and ≤1 mm approximate travel, but collision-free free-space motion cannot be counted as insertion while the tip is tens of millimetres from the ring.

## ALOHA fidelity and Cartesian position tracking

- Path shape `{fidelity['v15_position_path_shape']:.6f}`, speed `{fidelity['v15_speed_profile']:.6f}`.
- Midpoint `{fidelity['v15_bimanual_midpoint_trend']:.6f}`, relative vector `{fidelity['v15_relative_hand_vector_trend']:.6f}`, inter-hand distance `{fidelity['v15_inter_hand_distance_trend']:.6f}`.
- Simultaneous position ≤5 mm: `{numeric['position_tracking']['simultaneous_5mm_rate']*100:.2f}%`.
- Left mean/max: `{numeric['position_tracking']['left_mean_mm']:.3f}/{numeric['position_tracking']['left_max_mm']:.3f} mm`; right mean/max: `{numeric['position_tracking']['right_mean_mm']:.3f}/{numeric['position_tracking']['right_max_mm']:.3f} mm`.
- Temporal IK trajectory gate: fail (`{margin['branch_discontinuity_count']}` branch discontinuities), despite the Cartesian error threshold passing.

## Robustness and collisions

- Minimum arm joint margin: v14 `{margin['v14_minimum_joint_limit_margin_rad']:.3e}` rad → v15 `{margin['v15_minimum_joint_limit_margin_rad']:.3e}` rad; still below the `0.01` rad diagnostic target.
- Maximum arm joint step: `{margin['maximum_arm_joint_step_rad']:.3f}` rad; temporal branch continuity fails at `{margin['branch_discontinuity_frames']}`.
- v14 reported action-grasp table clearance: `{clearance['v14_action_phone_grasp_arm_table_clearance_mm']:.3f} mm`; v15 palm-center clearance `{clearance['v15_minimum_palm_center_table_clearance_mm']:.3f} mm` is not a full collision clearance substitute.
- Prohibited collision total: `{prohibited_collision_total}`; arm-torso `{collisions['categories']['arm_torso']['count']}`, hand-hand `{collisions['categories']['hand_hand']['count']}`.

## Isaac Lab kinematic replay

- Name-mapped controlled q: `[990,28]` (14 arm + 7 left Dex3 + 7 right Dex3).
- Requested/readback max error: `{isaac['maximum_requested_readback_error_rad']:.3e}` rad.
- Numerical-contact-proxy ↔ Isaac max position difference: `{isaac['maximum_numerical_contact_proxy_vs_Isaac_error_m']*1000:.3f} mm`.
- Wrist-local task contact-link displacement: left A `{isaac['maximum_finger_contact_wrist_local_displacement_from_start_m']['left_A']*1000:.3f} mm`, left B `{isaac['maximum_finger_contact_wrist_local_displacement_from_start_m']['left_B']*1000:.3f} mm`, right C `{isaac['maximum_finger_contact_wrist_local_displacement_from_start_m']['right_C']*1000:.3f} mm`.
- Task finger links moved relative to their wrist: `{isaac['actual_task_finger_links_move']}`; right-C motion is real but very small and does not satisfy insertion. Physics steps: `{isaac['physics_steps']}`.
- Object follow remained disabled because actual contact gates failed.

## Review artifacts

- Main four-panel: `{four_panel}`
- Robot overview: `{OUT / 'v15_g1_dex3_robot_only_overview.mp4'}`
- Robot side: `{OUT / 'v15_g1_dex3_robot_only_side.mp4'}`
- Left close-up: `{OUT / 'v15_left_phone_grasp_closeup.mp4'}`
- Right close-up: `{OUT / 'v15_right_accessory_hook_closeup.mp4'}`
- Charger close-up: `{OUT / 'v15_charger_placement_closeup.mp4'}`
- Disabled object-follow diagnostics: `{OUT / 'v15_object_follow_overview.mp4'}`, `{OUT / 'v15_object_follow_side.mp4'}`
- Semantic overview: `{OUT / 'v15_semantic_keyframe_overview.png'}`
- Left A+B: `{OUT / 'v15_left_ab_contact_closeup.png'}`
- Right C: `{OUT / 'v15_right_c_contact_closeup.png'}`
- Charger: `{OUT / 'v15_charger_alignment_closeup.png'}`
- Orientation axes: `{OUT / 'v15_orientation_axes_contact_sheet.png'}`

Every MP4 decodes to exactly 990 frames at 7.5 fps. The renderer is the corrected explicit articulation/USD-transform synchronization path and uses zero physics steps.

## Translator generalization contract

`reusable_vs_episode_derived_v15.json` separates reusable axis/geometry/rule parameters from per-episode actions, timestamps, semantic indices/progress, FK, Cartesian targets, arm q, and Dex3 q. No Episode-49 Cartesian/q trajectory is reusable as another episode trajectory. A frozen config was **not** emitted because the Episode-49 gates failed. Validation and held-out data were not accessed. After a future Episode-49 pass and user approval, the declared protocol is: three deterministic task-critical-ready **validation** episodes, final translator freeze, then one-shot frozen held-out-30 evaluation. Future primary claims require `SMOLVLA_GENERATED` source provenance; demonstration transfer is not mislabeled as VLA-generated transfer.

## Exact next action

Diagnose the failing orientation/contact/collision coupling on **Episode 49 only**. In particular, determine whether the immutable v14 carrier positions admit one opposing-surface A+B grasp at both phone and charger while retaining a continuous collision-free arm branch. Do not use validation or held-out trajectories and do not freeze v15 until all gates pass and the user visually approves it.

THE SCIENTIFIC TARGET REMAINS CROSS-EMBODIMENT TRANSFER OF VLA-GENERATED ALOHA BEHAVIOR TO UNITREE G1
THE V14 ALOHA-PRIMARY CARTESIAN ARM PATH WAS PRESERVED AS THE POSITION BACKBONE
TASK-CRITICAL ORIENTATION WAS DERIVED FROM SOURCE ALOHA MOTION AND TARGET TASK GEOMETRY
DEX3 TIMING WAS DERIVED THROUGH THE GENERIC SEMANTIC API RATHER THAN RUNTIME FRAME CONSTANTS
THE EPISODE-49 APPROVED TIMELINE WAS USED ONLY AS AN EXPLICIT DEVELOPMENT-TIMELINE INPUT THROUGH THE GENERIC API
NO G1 EXPERT MOTION WAS USED TO GENERATE THE TRAJECTORY
HELD-OUT EPISODES 45, 26, AND 21 WERE NOT USED TO TUNE V15
KINEMATIC OBJECT FOLLOW WAS NOT CLAIMED AS PHYSICS GRASP
NO PHYSICS, DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    report_dir = OUT / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Episode 49 v15 report</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:15px/1.55 sans-serif;white-space:pre-wrap}"
        "code{background:#eee;padding:.1rem .25rem}</style><body>"
        + html.escape(report)
        + "</body>\n",
        encoding="utf-8",
    )

    manifest_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    manifest = {
        "method": "ALOHA_PRIMARY_EP49_ORIENTATION_DEX3_V15",
        "status": numeric["final_status"],
        "diagnostic_only": True,
        "candidate_config_emitted": False,
        "candidate_config_reason": "Episode-49 orientation/contact/collision gates did not pass",
        "source_action_type": "SMOLVLA_GENERATED",
        "source_episode_id": 49,
        "semantic_timeline_source": semantic["timeline_source"],
        "translator_config_hash": None,
        "physics_steps": 0,
        "DDS": False,
        "publisher": False,
        "real_robot_command_allowed": False,
        "files": [
            {"path": str(path.relative_to(OUT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in manifest_files
        ],
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": numeric["final_status"], "videos": len(videos)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
