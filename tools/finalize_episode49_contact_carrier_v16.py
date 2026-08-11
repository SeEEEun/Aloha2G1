#!/usr/bin/env python3
"""Finalize review artifacts for the Episode-49 v16 contact-carrier run."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
V15 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_orientation_dex3_v15"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
SOURCE_REPLAY = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/source_optimized_action_replay.mp4"
RAW_CAM = ROOT / "raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000"
TRAJECTORY = OUT / "arm_dex3_coupled_trajectory.npz"
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_contact_carrier_v16.py"
BUILDER = ROOT / "tools/build_episode49_contact_carrier_v16.py"
FINALIZER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fit_panel(image: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def annotate(image: np.ndarray, title: str, detail: str, color=(80, 255, 120)) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(image, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(image, detail, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 215, 215), 1, cv2.LINE_AA)
    return image


def video_info(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate:format_tags=comment",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "metadata_present": bool(payload.get("format", {}).get("tags", {}).get("comment")),
    }


def create_four_panel(numeric_pass: bool) -> Path:
    output = OUT / "aloha_to_g1_contact_carrier_v16_4panel.mp4"
    raw_output = OUT / ".v16_4panel.raw.mp4"
    source = cv2.VideoCapture(str(SOURCE_REPLAY))
    robot = cv2.VideoCapture(str(OUT / "v16_g1_dex3_robot_only_overview.mp4"))
    objects = cv2.VideoCapture(str(OUT / "v16_object_follow_overview.mp4"))
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(raw_output)
    with np.load(TRAJECTORY, allow_pickle=False) as payload:
        length = len(payload["controlled_q"])
        names = payload["semantic_event_names"].astype(str)
        indices = payload["semantic_event_indices"].astype(int)
    events: dict[int, list[str]] = {}
    for name, index in zip(names, indices):
        events.setdefault(int(index), []).append(str(name))
    for action_index in range(length):
        ok_source, source_image = source.read()
        ok_robot, robot_image = robot.read()
        ok_object, object_image = objects.read()
        if not (ok_source and ok_robot and ok_object):
            raise RuntimeError(f"v16 source/render video ended at {action_index}")
        observed = min(action_index + 7, length - 1)
        raw = cv2.imread(str(RAW_CAM / f"frame_{observed:06d}.png"))
        event = ",".join(events.get(action_index, [])) or "-"
        p1 = annotate(fit_panel(raw), "RAW ALOHA cam_high", f"observed={observed} action={action_index} | {event[:42]}")
        p2 = annotate(fit_panel(source_image), "optimized_action ALOHA FK", "sole primary behavior source")
        p3 = annotate(fit_panel(robot_image), "ACTUAL G1 ARM + DEX3", "CONTACT-CARRIER IK | KINEMATIC")
        object_label = "KINEMATIC CONTACT-CARRIER FOLLOW" if numeric_pass else "OBJECT FOLLOW DISABLED — CONTACT GATE FAIL"
        p4 = annotate(fit_panel(object_image), object_label, "NOT PHYSICS GRASP", (80, 255, 120) if numeric_pass else (30, 120, 255))
        if action_index >= length - 7:
            cv2.putText(p1, "POST-OBSERVATION TERMINAL COMMAND SAMPLE", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 120, 255), 1, cv2.LINE_AA)
        writer.write(np.vstack((np.hstack((p1, p2)), np.hstack((p3, p4)))))
    writer.release(); source.release(); robot.release(); objects.release()
    metadata = {
        "trajectory_path": str(TRAJECTORY.resolve()), "trajectory_sha256": sha256(TRAJECTORY),
        "source_action_sha256": sha256(SOURCE), "renderer_sha256": sha256(RENDERER),
        "frame_count": length, "fps": 7.5, "numeric_contact_gate_pass": numeric_pass,
        "physics_steps": 0, "diagnostic_only": True,
    }
    temporary = OUT / ".v16_4panel.metadata.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_output), "-map", "0", "-c", "copy",
            "-metadata", "title=Episode49 v16 contact-carrier four-panel",
            "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
            "-movflags", "+faststart", str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, output); raw_output.unlink()
    return output


def main() -> int:
    # Recheck every frozen source after the renderer has run.  Rendering is
    # allowed to create v16-only USD/output files, but it must not mutate v14,
    # v15, the approved timeline, source action, or authoritative scene.
    freeze = load("input_freeze_audit.json")
    freeze["hashes_after_final_render"] = {
        path: sha256(Path(path)) for path in freeze["hashes_before"]
    }
    freeze["byte_identical_after_final_render"] = (
        freeze["hashes_after_final_render"] == freeze["hashes_before"]
    )
    freeze["status"] = (
        "INPUT_FREEZE_PASS"
        if freeze["byte_identical_after_final_render"]
        else "BLOCKED_FROZEN_INPUT_MUTATION"
    )
    dump(OUT / "input_freeze_audit.json", freeze)

    v15_freeze = load("v15_failure_preservation_audit.json")
    v15_freeze["hashes_after_final_render"] = {
        path: sha256(Path(path)) for path in v15_freeze["hashes_before"]
    }
    v15_freeze["byte_identical_after_final_render"] = (
        v15_freeze["hashes_after_final_render"] == v15_freeze["hashes_before"]
    )
    v15_freeze["status"] = (
        "FAILED_DIAGNOSTIC_CANDIDATE_PRESERVED"
        if v15_freeze["byte_identical_after_final_render"]
        else "BLOCKED_V15_DIAGNOSTIC_MUTATION"
    )
    dump(OUT / "v15_failure_preservation_audit.json", v15_freeze)

    numeric = load("numeric_gate_summary.json")
    isaac = load("isaaclab_kinematic_validation.json")
    carrier_frames = load("contact_carrier_frame_audit.json")
    carrier_frames.update({
        "isaac_numerical_parity_pending_renderer": False,
        "isaac_joint_write_readback_max_error_rad": float(
            isaac["maximum_requested_readback_error_rad"]
        ),
        "isaac_numerical_contact_center_proxy_max_error_m": float(
            isaac["maximum_numerical_contact_proxy_vs_Isaac_error_m"]
        ),
        "isaac_contact_center_proxy_parity_within_3mm": float(
            isaac["maximum_numerical_contact_proxy_vs_Isaac_error_m"]
        ) <= 0.003,
        "isaac_actual_task_finger_links_move": bool(
            isaac["actual_task_finger_links_move"]
        ),
    })
    dump(OUT / "contact_carrier_frame_audit.json", carrier_frames)
    dump(OUT / "g1_contact_carrier_frames_v16.json", carrier_frames)
    four_panel = create_four_panel(bool(numeric["numerical_pass"]))
    videos = [video_info(path) for path in sorted(OUT.glob("*.mp4"))]
    exact_frames = all(row["decoded_frames"] == 990 for row in videos)
    isaac_pass = bool(
        isaac.get("status") == "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS"
        and isaac.get("actual_task_finger_links_move")
        and isaac.get("physics_steps") == 0
        and exact_frames
    )
    numeric["gates"]["ISAACLAB_KINEMATIC_REPLAY_PASS"] = isaac_pass
    final_pass = bool(
        numeric["numerical_pass"]
        and isaac_pass
        and freeze["byte_identical_after_final_render"]
        and v15_freeze["byte_identical_after_final_render"]
    )
    numeric["final_pass"] = final_pass
    numeric["status"] = (
        "ALOHA_PRIMARY_CONTACT_CARRIER_V16_READY_FOR_VISUAL_REVIEW"
        if final_pass else numeric["status"]
    )
    dump(OUT / "numeric_gate_summary.json", numeric)

    left = load("left_phone_carrier_metrics.json")
    right = load("right_accessory_carrier_metrics.json")
    orientation = load("task_orientation_metrics.json")
    branch = load("branch_continuity_metrics.json")
    collision = load("collision_breakdown.json")
    fidelity = load("v14_vs_v15_vs_v16_fidelity.json")
    correction = load("minimum_contact_correction_audit.json")
    selected_left = load("selected_left_pincher_carrier.json")
    selected_right = load("selected_right_hook_carrier.json")
    deviation = load("v14_deviation_metrics.json")
    joint = load("joint_margin_metrics.json")

    def candidate_row(directory: Path, name: str) -> dict:
        def read(filename: str):
            return json.loads((directory / filename).read_text(encoding="utf-8"))
        left_row = read("left_phone_carrier_metrics.json")
        right_row = read("right_accessory_carrier_metrics.json")
        collision_row = read("collision_breakdown.json")
        branch_row = read("branch_continuity_metrics.json")
        orientation_row = read("task_orientation_metrics.json")
        fidelity_row = read("v14_vs_v15_vs_v16_fidelity.json")
        return {
            "candidate": name,
            "left_A_max_gap_mm": left_row["A_gap_max_mm"],
            "left_B_max_gap_mm": left_row["B_gap_max_mm"],
            "right_C_max_gap_mm": right_row["C_ring_gap_max_mm"],
            "right_ring_penetration_max_mm": right_row["ring_material_penetration_max_mm"],
            "prohibited_collision_records": collision_row["prohibited_collision_records"],
            "branch_discontinuities": branch_row["branch_discontinuity_count"],
            "task_orientation_pass": orientation_row["pass"],
            "minimum_primary_fidelity": fidelity_row["v16_minimum_primary_metric"],
            "all_task_gates_pass": False,
        }

    candidate_rows = [candidate_row(OUT, "CONTACT_BALANCED_FULL_COLLISION")]
    high_candidate = OUT / "diagnostics/strong_rotation_full_bimanual_collision_candidate"
    if high_candidate.is_dir():
        candidate_rows.append(candidate_row(high_candidate, "ORIENTATION_STRONG_FULL_COLLISION"))
    dump(OUT / "coupled_solver_candidate_comparison.json", {
        "selected": "CONTACT_BALANCED_FULL_COLLISION",
        "selection_reason": (
            "No candidate passed. The selected failed diagnostic minimizes continuous contact gaps and "
            "prohibited collisions while retaining the passing task-axis orientation result."
        ),
        "candidates": candidate_rows,
        "validation_or_heldout_used": False,
    })

    video_lines = "\n".join(f"- `{row['path']}` ({row['decoded_frames']} frames)" for row in videos)
    suitability = (
        "All kinematic gates passed; user visual review is now the required decision."
        if final_pass else
        "The best diagnostic is reviewable, but at least one required numerical/Isaac gate remains blocked."
    )
    next_action = (
        "WAIT FOR USER VISUAL APPROVAL. Do not freeze the reusable translator before that approval."
        if final_pass else
        "Diagnose the reported failing subsystem on Episode 49 only; do not open validation or held-out episodes."
    )
    report = f"""# Episode 49 contact-carrier v16

## 3-line summary

1. Final status: `{numeric['status']}`; v15 remains a preserved failed diagnostic and v14 is only the ALOHA-primary reference/seed.
2. The active Dex3 pad geometry defines one left A/B pinch carrier and one right-C hook carrier; the selected smooth correction was solved through named semantic progress.
3. This is a 990-sample, zero-physics Isaac Lab kinematic diagnostic; validation/held-out episodes, G1 Expert motion, DDS, publisher, and hardware were not used.

## 1. Final status

`{numeric['status']}`. Numerical gates pass: `{numeric['numerical_pass']}`. Isaac replay pass: `{isaac_pass}`.

## 2. Why immutable v14 wrist failed

v14 targeted a palm/wrist proxy and proved arm reach, but did not constrain the active A/B collision-pad points or right-C ring contact. v15 therefore preserved the wrist path while leaving A/B gaps near 45–46 mm and the right-C gap near 55–69 mm. v16 lets the wrist be the kinematic consequence of the carrier and active Dex3 state.

## 3. Contact-carrier definitions

Left origin is the midpoint of active A/B pad points; +X is A→B and +Y is the geometry-derived approach direction. Right origin is the active C pad point; +X is the C contact axis and +Y is the ring engagement/hook axis. Both frames are proper SO(3) frames.

## 4. Minimum correction from v14

The fixed active-geometry carrier registrations are left `{correction['fixed_carrier_registration']['left_translation_m']}` m and right `{correction['fixed_carrier_registration']['right_translation_m']}` m. After removing those fixed embodiment offsets, the time-varying translation RMS/max is left `{correction['left']['translation_m']['rms']:.6f}` / `{correction['left']['translation_m']['maximum']:.6f}` m and right `{correction['right']['translation_m']['rms']:.6f}` / `{correction['right']['translation_m']['maximum']:.6f}` m. The resulting achieved wrist deviation from the v14 reference is left RMS/max `{deviation['left_wrist_translation_deviation_from_v14_m']['rms']:.6f}` / `{deviation['left_wrist_translation_deviation_from_v14_m']['maximum']:.6f}` m and right `{deviation['right_wrist_translation_deviation_from_v14_m']['rms']:.6f}` / `{deviation['right_wrist_translation_deviation_from_v14_m']['maximum']:.6f}` m. The maximum time-varying correction is `{100.0 * deviation['correction_over_workspace_scaled_source_displacement_ratio']:.2f}`% of the workspace-scaled source displacement.

## 5. Selected residual model

Fixed active-geometry carrier offsets plus source-arc-progress, semantic-phase-conditioned smooth translation/rotation residuals. No per-frame snapping or numeric-frame rule is used.

## 6. Source-motion fidelity

Minimum primary v16 metric: `{fidelity['v16_minimum_primary_metric']:.6f}` (right accessory-acquisition speed-profile correlation); hard gate pass: `{fidelity['pass']}`. Left task phases retain path correlations `{[round(value['path_shape_correlation'], 6) for value in fidelity['v16_phase_relative']['left'].values()]}` and the bimanual midpoint/relative-vector/inter-hand correlations are `{fidelity['v16_bimanual']['midpoint_trend_correlation']:.6f}` / `{fidelity['v16_bimanual']['relative_vector_trend_correlation']:.6f}` / `{fidelity['v16_bimanual']['inter_hand_distance_trend_correlation']:.6f}`.

## 7. Left A/B continuous carrier

Assignment `{selected_left['selected']['assignment']}`; A max `{left['A_gap_max_mm']:.3f}` mm, B max `{left['B_gap_max_mm']:.3f}` mm, carrier translation drift max `{left['carrier_translation_drift_max_mm']:.3f}` mm. Pass: `{left['continuous_rigid_pinch_pass']}`.

## 8. Right C insertion/hook

Family `{selected_right['selected']['family']}`; continuous C gap max `{right['C_ring_gap_max_mm']:.3f}` mm and ring-material penetration max `{right['ring_material_penetration_max_mm']:.3f}` mm. Pass: `{right['continuous_hook_pass']}`.

## 9. Portrait orientation

Long-axis error `{orientation['portrait_long_axis_error_deg']:.3f}` deg.

## 10. Charger orientation

Center `{orientation['charger_center_error_mm']:.3f}` mm, normal `{orientation['charger_normal_error_deg']:.3f}` deg, vertical axis `{orientation['charger_vertical_axis_error_deg']:.3f}` deg.

## 11. Branch continuity

Arm branch discontinuities `{branch['branch_discontinuity_count']}`; maximum arm step `{branch['maximum_arm_joint_step_rad']:.6f}` rad and maximum full arm+Dex3 step `{branch['maximum_joint_step_rad']:.6f}` rad.

## 12. Joint margins

Minimum arm / left Dex3 / right Dex3 margins: `{joint['minimum_arm_joint_margin_rad']:.3e}` / `{joint['minimum_left_dex3_margin_rad']:.3e}` / `{joint['minimum_right_dex3_margin_rad']:.3e}` rad. These are non-violating but not robust margins.

## 13. Collision breakdown

Prohibited records `{collision['prohibited_collision_records']}`: arm–torso `{collision['categories']['arm_torso']['count']}`, arm–arm `{collision['categories']['arm_arm']['count']}`, hand–torso `{collision['categories']['hand_torso']['count']}`, hand–hand `{collision['categories']['hand_hand']['count']}`, same-hand `{collision['categories']['same_hand_self_contact']['count']}`, palm–table `{collision['categories']['palm_table']['count']}`. Pass `{collision['pass']}`. Intentional A/B-phone and C-ring relations are measured separately and are not used to hide self-collision.

## 14. Isaac actual mesh

Status `{isaac.get('status')}`; controlled joints `{isaac.get('controlled_joint_count')}`; write/readback max error `{isaac.get('maximum_requested_readback_error_rad'):.3e}` rad; numerical/Isaac contact-center proxy max error `{isaac.get('maximum_numerical_contact_proxy_vs_Isaac_error_m') * 1000.0:.3f}` mm; physics steps `{isaac.get('physics_steps')}`; actual task finger links move `{isaac.get('actual_task_finger_links_move')}`. Five one-camera passes avoid the multi-camera RTX resource stall while replaying the exact same 990-sample q. Object follow is disabled because the continuous contact gates failed.

## 15. Review videos

{video_lines}

## 16. Visual-approval suitability

`{final_pass}`. {suitability}

## 17. Next action

{next_action}

ALOHA MOTION REMAINED THE PRIMARY BEHAVIOR SOURCE
THE V14 WRIST TRAJECTORY WAS USED AS A REFERENCE RATHER THAN AN ABSOLUTE CONTACT CONSTRAINT
TASK MOTION WAS RETARGETED TO EMBODIMENT-SPECIFIC CONTACT CARRIERS
LEFT PHONE MANIPULATION USED THE DEX3 A/B PINCH CARRIER
RIGHT ACCESSORY MANIPULATION USED THE DEX3 C HOOK CARRIER
ONLY THE MINIMUM SMOOTH EMBODIMENT-SPECIFIC CORRECTION WAS ALLOWED
NO HAND-WRITTEN CARTESIAN WAYPOINT TRAJECTORY WAS USED
NO VALIDATION OR HELD-OUT TRAJECTORY WAS USED FOR PARAMETER TUNING
NO G1 EXPERT MOTION WAS USED
NO PHYSICS, DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    html_body = "<html><body><pre>" + html.escape(report) + "</pre></body></html>\n"
    (OUT / "report/index.html").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "report/index.html").write_text(html_body, encoding="utf-8")
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6

# Headless memory-safe replay (zero physics steps).  Each pass writes the
# exact same 990-sample q through the verified articulation/Fabric path.
for camera_pass in overview side left_close right_close charger_close; do
  python {RENDERER} --headless --enable_cameras --rendering_mode performance --camera-pass "$camera_pass"
done
python {ROOT / 'tools/aggregate_v16_isaac_passes.py'}

# GUI overview
python {RENDERER} --gui --gui-camera overview

# GUI side
python {RENDERER} --gui --gui-camera side
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)
    manifest = load("run_manifest.json")
    manifest.update({
        "status": numeric["status"], "final_pass": final_pass,
        "renderer": str(RENDERER.resolve()), "renderer_sha256": sha256(RENDERER),
        "builder_sha256": sha256(BUILDER), "finalizer_sha256": sha256(FINALIZER),
        "videos": videos, "four_panel": str(four_panel.resolve()),
        "required_review_videos_pending": False,
    })
    dump(OUT / "run_manifest.json", manifest)
    dump(OUT / "visual_validation_audit.json", {
        "status": "ISAACLAB_KINEMATIC_REPLAY_PASS" if isaac_pass else "BLOCKED_ISAACLAB_KINEMATIC_REPLAY",
        "all_videos_990_frames": exact_frames, "videos": videos,
        "actual_mesh": True, "actual_Dex3": True, "physics_steps": 0,
    })
    print(json.dumps({"status": numeric["status"], "final_pass": final_pass, "videos": len(videos)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
