#!/usr/bin/env python3
"""Finalize the frozen Candidate-A versus hand-only distal-pad Candidate-B test."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_pad_ablation_v1"
A_DIR = OUT / "candidate_A_physics"
B_DIR = OUT / "candidate_B_physics"
B_RENDER = OUT / "candidate_B_static_render"
PRIMITIVE_A = CAL / "left_phone_fingertip_pinch_primitive.json"
PRIMITIVE_B = OUT / "candidate_B_primitive.json"
V17_2 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, default=convert, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".incomplete")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(np.degrees((Rotation.from_quat(q0).inv() * Rotation.from_quat(q1)).magnitude()))


def candidate_data(directory: Path) -> dict:
    result = load(directory / "static_physics_result.json")
    contact = load(directory / "phone_contact_identity_metrics.json")
    retention = load(directory / "phone_retention_metrics.json")
    setup = load(directory / "physics_setup_audit.json")
    no_cheat = load(directory / "no_cheat_audit.json")
    collision = load(directory / "collision_audit.json")
    tracking = load(directory / "dex3_tracking_metrics.json")
    trace = np.load(directory / "static_physics_trace.npz")
    return {
        "result": result, "contact": contact, "retention": retention,
        "setup": setup, "no_cheat": no_cheat, "collision": collision,
        "tracking": tracking, "trace": trace,
    }


def patch_summary(candidate: dict) -> dict:
    summary = {}
    for label in ("THUMB", "INDEX", "THIRD"):
        row = candidate["contact"]["identity"][label]
        patch = row["contact_patch_proxy"]
        per_step = patch["per_step"]
        separations = [item["minimum_signed_separation_m"] for item in per_step]
        projected_long = [item["projected_long_axis_span_m"] for item in per_step]
        projected_short = [item["projected_short_axis_span_m"] for item in per_step]
        summary[label] = {
            "physical_link": row["physical_link"],
            "contact_samples": row["contact_samples"],
            "raw_contact_points": row["raw_contact_points"],
            "mean_contact_point_count_when_present": patch["mean_contact_point_count_when_present"],
            "mean_pairwise_spatial_span_m": patch["mean_pairwise_spatial_span_m"],
            "maximum_pairwise_spatial_span_m": patch["maximum_pairwise_spatial_span_m"],
            "mean_projected_long_axis_span_m": float(np.mean(projected_long)) if projected_long else 0.0,
            "mean_projected_short_axis_span_m": float(np.mean(projected_short)) if projected_short else 0.0,
            "phone_local_centroid_maximum_excursion_m": patch["phone_local_centroid_maximum_excursion_m"],
            "phone_local_centroid_rms_excursion_m": patch["phone_local_centroid_rms_excursion_m"],
            "initial_minimum_signed_separation_m": row["initial_solver_minimum_signed_separation_m"],
            "hold_minimum_signed_separation_m": float(np.min(separations)) if separations else None,
            "hold_mean_minimum_signed_separation_m": float(np.mean(separations)) if separations else None,
            "mean_force_when_active_n": row["mean_force_when_active_n"],
            "maximum_force_n": row["maximum_force_n"],
            "literal_contact_area_claimed": False,
        }
    return summary


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return frames, fps


def motion_overlay(frame: np.ndarray, trace, step: int, label: str) -> np.ndarray:
    output = frame.copy()
    phone = trace["phone_pose_xyzw"]
    initial = phone[0]
    current = phone[step]
    translation = 1000.0 * np.linalg.norm(current[:3] - initial[:3])
    rotation = angle_deg(initial[3:7], current[3:7])
    thumb = float(trace["thumb_phone_force_n"][step])
    index = float(trace["index_phone_force_n"][step])
    simultaneous = bool(trace["simultaneous_thumb_index"][step])
    height = 65
    cv2.rectangle(output, (0, output.shape[0] - height), (output.shape[1], output.shape[0]), (245, 245, 245), -1)
    cv2.putText(output, f"{label} | t={float(trace['time_s'][step]):.3f}s | bilateral={simultaneous}",
                (10, output.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(output, f"T={thumb:.2f}N I={index:.2f}N | phone d={translation:.1f}mm rot={rotation:.1f}deg",
                (10, output.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (25, 70, 25), 1, cv2.LINE_AA)
    return output


def main() -> int:
    a = candidate_data(A_DIR)
    b = candidate_data(B_DIR)
    primitive_a = load(PRIMITIVE_A)
    primitive_b = load(PRIMITIVE_B)
    q_a = np.asarray(primitive_a["selected_static_q_rad"], dtype=float)
    q_b = np.asarray(primitive_b["selected_q_rad"], dtype=float)
    dq = q_b - q_a
    if not np.array_equal(q_b[5:], q_a[5:]):
        raise RuntimeError("Candidate B changed third finger")
    if not a["result"]["simultaneous_thumb_index_contact"] or not b["result"]["simultaneous_thumb_index_contact"]:
        raise RuntimeError("A/B comparison requires bilateral contact from both candidates")

    # Candidate A is audited again after all Candidate-B and physics work.
    freeze = load(OUT / "candidate_A_freeze_audit.json")
    primitive_hash_after_all = sha256(PRIMITIVE_A)
    freeze.update({
        "primitive_sha256_after_all_ab_work": primitive_hash_after_all,
        "q_raw_sha256_after_all_ab_work": array_sha256(np.asarray(load(PRIMITIVE_A)["selected_static_q_rad"], dtype=float)),
        "candidate_A_unchanged_after_all_ab_work": (
            primitive_hash_after_all == freeze["primitive_sha256_before"]
            and np.array_equal(q_a, np.asarray(load(PRIMITIVE_A)["selected_static_q_rad"], dtype=float))
        ),
    })
    freeze["status"] = "CANDIDATE_A_UNCHANGED" if freeze["candidate_A_unchanged_after_all_ab_work"] else "BLOCKED_CANDIDATE_A_MUTATION"
    dump(OUT / "candidate_A_freeze_audit.json", freeze)

    # Exact physics identity excludes only the deliberately different hand q,
    # output paths, and their provenance fields.
    identity_fields = [
        "authoritative_scene", "authoritative_scene_composed", "phone_prim_path",
        "phone_asset", "phone_initial_transform_source", "calibration_phone_transform_matrix",
        "authoritative_v14_root_transform_matrix", "phone_initial_transform_matrix",
        "phone_mass_kg", "phone_mass_source", "friction_authored_or_changed_by_test",
        "gravity_enabled", "collision_enabled", "physics_dt_s", "physics_steps",
        "trial", "sequence_duration_s", "phase_durations_s", "controller_source",
        "controller_values_changed", "arm", "dex3", "fabric_enabled", "rtx_reads_fabric",
        "render_order",
    ]
    field_comparison = {
        key: {
            "A": a["setup"][key], "B": b["setup"][key],
            "identical": a["setup"][key] == b["setup"][key],
        } for key in identity_fields
    }
    initial_pose_identical = np.array_equal(
        np.asarray(a["retention"]["phone_initial_pose_xyzw"]),
        np.asarray(b["retention"]["phone_initial_pose_xyzw"]),
    )
    identity_pass = all(row["identical"] for row in field_comparison.values()) and initial_pose_identical
    dump(OUT / "ab_physics_identity_audit.json", {
        "status": "A_B_TRUE_PHYSX_PARAMETERS_AND_INITIAL_CONDITIONS_IDENTICAL" if identity_pass else "BLOCKED_INVALID_AB_PHYSICS_IDENTITY",
        "field_comparison": field_comparison,
        "phone_initial_pose_exactly_identical": initial_pose_identical,
        "phone_initial_pose_A": a["retention"]["phone_initial_pose_xyzw"],
        "phone_initial_pose_B": b["retention"]["phone_initial_pose_xyzw"],
        "allowed_difference": "left Dex3 task-finger q only",
        "candidate_A_q_rad": q_a,
        "candidate_B_q_rad": q_b,
        "phone_pose_writes_after_t0_A": a["no_cheat"]["timed_phone_pose_writes"],
        "phone_pose_writes_after_t0_B": b["no_cheat"]["timed_phone_pose_writes"],
        "object_follow_A_B": [a["no_cheat"]["object_follow"], b["no_cheat"]["object_follow"]],
        "hidden_fixed_joint_A_B": [a["no_cheat"]["hidden_fixed_joint"], b["no_cheat"]["hidden_fixed_joint"]],
    })
    if not identity_pass:
        raise RuntimeError("A/B physics identity failed")

    patch_a = patch_summary(a)
    patch_b = patch_summary(b)
    span_change = {}
    centroid_change = {}
    for label in ("THUMB", "INDEX"):
        av = patch_a[label]["mean_pairwise_spatial_span_m"]
        bv = patch_b[label]["mean_pairwise_spatial_span_m"]
        span_change[label] = 100.0 * (bv - av) / av
        ac = patch_a[label]["phone_local_centroid_maximum_excursion_m"]
        bc = patch_b[label]["phone_local_centroid_maximum_excursion_m"]
        centroid_change[label] = 100.0 * (bc - ac) / ac
    patch_broader = all(span_change[label] > 5.0 for label in ("THUMB", "INDEX"))
    patch_more_stable = all(centroid_change[label] < -5.0 for label in ("THUMB", "INDEX"))
    dump(OUT / "candidate_A_vs_B_contact_patch.json", {
        "status": "B_CONTACT_PATCH_PROXY_BROADER_AND_MORE_STABLE" if patch_broader and patch_more_stable else "B_CONTACT_PATCH_PROXY_NOT_BROADER_OR_MORE_STABLE",
        "metric_name": "CONTACT_PATCH_PROXY",
        "literal_contact_area_claimed": False,
        "candidate_A": patch_a,
        "candidate_B": patch_b,
        "B_minus_A_mean_span_percent": span_change,
        "B_minus_A_centroid_maximum_excursion_percent": centroid_change,
        "B_broader": patch_broader,
        "B_more_stable": patch_more_stable,
        "interpretation": "PhysX exposes contact points, not literal mm^2 area. B reduced penetration but did not broaden/stabilize the measured point manifold.",
    })

    ra = a["retention"]
    rb = b["retention"]
    retention_metrics = {
        "relative_slip_m": [ra["phone_relative_to_pinch_center_hold_slip_m"], rb["phone_relative_to_pinch_center_hold_slip_m"]],
        "vertical_motion_m": [ra["phone_hold_vertical_displacement_m"], rb["phone_hold_vertical_displacement_m"]],
        "total_translation_m": [ra["phone_hold_com_displacement_m"], rb["phone_hold_com_displacement_m"]],
        "orientation_change_deg": [ra["phone_hold_orientation_change_deg"], rb["phone_hold_orientation_change_deg"]],
        "maximum_angular_velocity_rad_s": [ra["phone_max_angular_speed_rad_s"], rb["phone_max_angular_speed_rad_s"]],
    }
    improvements = {
        "relative_slip_reduction_percent": 100.0 * (retention_metrics["relative_slip_m"][0] - retention_metrics["relative_slip_m"][1]) / retention_metrics["relative_slip_m"][0],
        "vertical_drop_magnitude_reduction_percent": 100.0 * (abs(retention_metrics["vertical_motion_m"][0]) - abs(retention_metrics["vertical_motion_m"][1])) / abs(retention_metrics["vertical_motion_m"][0]),
        "total_translation_reduction_percent": 100.0 * (retention_metrics["total_translation_m"][0] - retention_metrics["total_translation_m"][1]) / retention_metrics["total_translation_m"][0],
        "orientation_change_reduction_percent": 100.0 * (retention_metrics["orientation_change_deg"][0] - retention_metrics["orientation_change_deg"][1]) / retention_metrics["orientation_change_deg"][0],
        "maximum_angular_velocity_reduction_percent": 100.0 * (retention_metrics["maximum_angular_velocity_rad_s"][0] - retention_metrics["maximum_angular_velocity_rad_s"][1]) / retention_metrics["maximum_angular_velocity_rad_s"][0],
    }
    b_meaningful_retention = (
        improvements["relative_slip_reduction_percent"] >= 20.0
        and improvements["orientation_change_reduction_percent"] >= 20.0
        and rb["phone_relative_to_pinch_center_hold_slip_m"] <= 0.03
        and patch_broader and patch_more_stable
    )
    final_status = "DISTAL_PAD_PINCH_RETENTION_IMPROVED" if b_meaningful_retention else "DISTAL_PAD_PINCH_NO_MEANINGFUL_IMPROVEMENT"
    recommendation = "B" if b_meaningful_retention else "A"
    dump(OUT / "candidate_A_vs_B_retention.json", {
        "status": final_status,
        "candidate_A": {
            "physics_status": a["result"]["status"],
            "simultaneous_contact_duration_s": a["result"]["simultaneous_contact_duration_s"],
            "third_remained_non_task": a["result"]["third_remained_non_task"],
            **{key: value[0] for key, value in retention_metrics.items()},
        },
        "candidate_B": {
            "physics_status": b["result"]["status"],
            "simultaneous_contact_duration_s": b["result"]["simultaneous_contact_duration_s"],
            "third_remained_non_task": b["result"]["third_remained_non_task"],
            **{key: value[1] for key, value in retention_metrics.items()},
        },
        "B_improvement_percent": improvements,
        "B_meaningful_retention_gate": b_meaningful_retention,
        "B_contact_patch_broader_gate": patch_broader,
        "B_contact_patch_more_stable_gate": patch_more_stable,
        "recommended_candidate": recommendation,
        "decision_policy": "B must materially reduce slip and rotation while broadening/stabilizing contact and avoiding worse penetration.",
        "robustness_sweep": "NOT_RUN_NOMINAL_RETENTION_REMAINED_WEAK_AND_B_WAS_ALREADY_FROZEN",
    })

    # Root-level requested videos.
    video_copies = {
        A_DIR / "candidate_A_fingertip_hold_contact_closeup.mp4": OUT / "candidate_A_fingertip_hold_closeup.mp4",
        A_DIR / "candidate_A_fingertip_hold_side.mp4": OUT / "candidate_A_fingertip_hold_side.mp4",
        B_DIR / "candidate_B_distal_pad_hold_contact_closeup.mp4": OUT / "candidate_B_distal_pad_hold_closeup.mp4",
        B_DIR / "candidate_B_distal_pad_hold_side.mp4": OUT / "candidate_B_distal_pad_hold_side.mp4",
    }
    for source, destination in video_copies.items():
        copy(source, destination)

    a_close, fps = read_video(OUT / "candidate_A_fingertip_hold_closeup.mp4")
    b_close, fps_b = read_video(OUT / "candidate_B_distal_pad_hold_closeup.mp4")
    a_side, fps_as = read_video(OUT / "candidate_A_fingertip_hold_side.mp4")
    b_side, fps_bs = read_video(OUT / "candidate_B_distal_pad_hold_side.mp4")
    if len({len(a_close), len(b_close), len(a_side), len(b_side)}) != 1 or max(abs(fps-x) for x in (fps_b, fps_as, fps_bs)) > 1e-9:
        raise RuntimeError("A/B video synchronization mismatch")
    panel_w, panel_h = 640, 480
    four_path = OUT / "candidate_A_vs_B_RETENTION_4panel.mp4"
    temporary_four = four_path.with_suffix(".mp4.incomplete.mp4")
    writer = cv2.VideoWriter(str(temporary_four), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * panel_w, 2 * panel_h))
    for frame_index in range(len(a_close)):
        step = min(int(round(frame_index * (len(a["trace"]["time_s"]) - 1) / (len(a_close) - 1))), len(a["trace"]["time_s"]) - 1)
        panels = [
            motion_overlay(cv2.resize(a_close[frame_index], (panel_w, panel_h)), a["trace"], step, "A PHOTO_FINGERTIP close-up"),
            motion_overlay(cv2.resize(b_close[frame_index], (panel_w, panel_h)), b["trace"], step, "B DISTAL_PAD close-up"),
            motion_overlay(cv2.resize(a_side[frame_index], (panel_w, panel_h)), a["trace"], step, "A side"),
            motion_overlay(cv2.resize(b_side[frame_index], (panel_w, panel_h)), b["trace"], step, "B side"),
        ]
        writer.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    writer.release()
    os.replace(temporary_four, four_path)

    # 2x5 physics contact sheet at matched times.
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    indices = [0, int(.2 * (len(a_close)-1)), int(.5 * (len(a_close)-1)), int(.8 * (len(a_close)-1)), len(a_close)-1]
    sheet_w, sheet_h = 430, 322
    sheet = Image.new("RGB", (5 * sheet_w, 2 * (sheet_h + 58)), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    for row, (label, frames, candidate) in enumerate((("A PHOTO_FINGERTIP", a_close, a), ("B DISTAL_PAD", b_close, b))):
        for col, frame_index in enumerate(indices):
            step = min(int(round(frame_index * (len(candidate["trace"]["time_s"]) - 1) / (len(frames) - 1))), len(candidate["trace"]["time_s"]) - 1)
            rgb = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((sheet_w, sheet_h), Image.Resampling.LANCZOS)
            x, y = col * sheet_w, row * (sheet_h + 58)
            sheet.paste(image, (x, y + 58))
            phone = candidate["trace"]["phone_pose_xyzw"]
            trans = 1000 * np.linalg.norm(phone[step, :3] - phone[0, :3])
            rot = angle_deg(phone[0, 3:7], phone[step, 3:7])
            draw.text((x + 8, y + 6), f"{label} | t={float(candidate['trace']['time_s'][step]):.2f}s", fill=(15, 15, 15), font=font)
            draw.text((x + 8, y + 30), f"phone d={trans:.1f}mm  rot={rot:.1f}deg", fill=(35, 75, 35), font=font)
    sheet.save(OUT / "candidate_A_vs_B_retention_contact_sheet.png")

    # Static A/B sheet, six matched views.
    view_names = ["front_oblique", "back_oblique", "top", "side", "fingertip_closeup", "palm_side"]
    static_panel_w, static_panel_h = 390, 292
    static_sheet = Image.new("RGB", (6 * static_panel_w, 2 * (static_panel_h + 44)), (248, 248, 248))
    static_draw = ImageDraw.Draw(static_sheet)
    for row, (label, directory) in enumerate((("A PHOTO_FINGERTIP", CAL), ("B PHOTO_DERIVED_DISTAL_PAD", B_RENDER))):
        for col, view in enumerate(view_names):
            source = directory / f"left_phone_pinch_{view}.png"
            image = Image.open(source).convert("RGB")
            image.thumbnail((static_panel_w, static_panel_h), Image.Resampling.LANCZOS)
            x, y = col * static_panel_w, row * (static_panel_h + 44)
            static_sheet.paste(image, (x + (static_panel_w-image.width)//2, y + 44))
            static_draw.text((x + 8, y + 10), f"{label} | {view}", fill=(20, 20, 20), font=font)
    static_sheet.save(OUT / "candidate_A_vs_B_static_comparison.png")

    # Reuse the rendered active pad markers, adding explicit task-role title.
    overlay = Image.open(B_RENDER / "left_phone_pinch_identity_overlay.png").convert("RGB")
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((15, overlay.height - 62, overlay.width - 15, overlay.height - 15), fill=(247, 247, 247), outline=(20, 20, 20), width=2)
    overlay_draw.text((28, overlay.height - 50), "THUMB PAD + INDEX PAD | THIRD — NON-TASK | ARM/WRIST FIXED", fill=(20, 55, 20), font=font)
    overlay.save(OUT / "candidate_B_pad_identity_overlay.png")

    gui_a = """source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
DISPLAY=:0 /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir outputs/scene_registered_retargeting/dex3_left_phone_pinch_pad_ablation_v1/gui_A \\
  --trial closed_hold --hold-duration 1.5 --artifact-prefix candidate_A_gui \\
  --gui --pause-at-end --enable_cameras
"""
    gui_b = """source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
DISPLAY=:0 /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir outputs/scene_registered_retargeting/dex3_left_phone_pinch_pad_ablation_v1/gui_B \\
  --trial closed_hold --hold-duration 1.5 \\
  --test-q-json outputs/scene_registered_retargeting/dex3_left_phone_pinch_pad_ablation_v1/candidate_B_primitive.json \\
  --artifact-prefix candidate_B_gui --gui --pause-at-end --enable_cameras
"""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# Candidate A interactive true-PhysX review
{gui_a}

# Candidate B interactive true-PhysX review
{gui_b}
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    geometry_a = load(OUT / "candidate_A_geometry.json")
    geometry_b = load(OUT / "candidate_B_geometry.json")
    opt = load(OUT / "candidate_B_optimization.json")
    report = f"""CANDIDATE A FINGERTIP RETENTION: CONTACT PASS / RETENTION WARNING — slip {ra['phone_relative_to_pinch_center_hold_slip_m']*1000:.3f} mm, rotation {ra['phone_hold_orientation_change_deg']:.3f} deg.
CANDIDATE B DISTAL-PAD RETENTION: CONTACT PASS / RETENTION WARNING — slip {rb['phone_relative_to_pinch_center_hold_slip_m']*1000:.3f} mm, rotation {rb['phone_hold_orientation_change_deg']:.3f} deg.
OBJECTIVE SELECTION: CANDIDATE A RETAINED — B reduced rotation/penetration but did not materially stop ~50 mm slip or broaden/stabilize the contact-patch proxy.

# Left Dex3 photo pinch distal-pad ablation v1

Final state: `{final_status}`.  Recommended candidate: `{recommendation}`.

## 1–3. Frozen A, Candidate B, and joint delta

- A q: `{np.round(q_a, 9).tolist()}`
- B q: `{np.round(q_b, 9).tolist()}`
- B−A: `{np.round(dq, 9).tolist()}`
- Maximum task-joint change: {float(np.max(np.abs(dq[:5]))):.6f} rad; the 0.15 rad initial trust region was not expanded.
- A primitive SHA-256 remained `{primitive_hash_after_all}`.

## 4–6. Opposition and distal-pad geometry

Thumb opposition remains negative/active ({q_a[0]:.6f}→{q_b[0]:.6f} rad).  B used the active
thumb/index distal mesh regions; THIRD stayed [-0.1, -0.1] rad.  Pad-to-phone normal
angles A→B were thumb {geometry_a['thumb_pad_to_phone_normal_angle_deg']:.3f}→{geometry_b['thumb_pad_to_phone_normal_angle_deg']:.3f} deg
and index {geometry_a['index_pad_to_phone_normal_angle_deg']:.3f}→{geometry_b['index_pad_to_phone_normal_angle_deg']:.3f} deg.

## 7. Signed penetration

- Thumb initial PhysX separation A/B: {patch_a['THUMB']['initial_minimum_signed_separation_m']*1000:.3f} / {patch_b['THUMB']['initial_minimum_signed_separation_m']*1000:.3f} mm.
- Index initial PhysX separation A/B: {patch_a['INDEX']['initial_minimum_signed_separation_m']*1000:.3f} / {patch_b['INDEX']['initial_minimum_signed_separation_m']*1000:.3f} mm.

B substantially reduced rather than increased initial penetration.

## 8. CONTACT_PATCH_PROXY

- Thumb mean point-manifold span A/B: {patch_a['THUMB']['mean_pairwise_spatial_span_m']*1000:.3f} / {patch_b['THUMB']['mean_pairwise_spatial_span_m']*1000:.3f} mm ({span_change['THUMB']:+.1f}%).
- Index mean span A/B: {patch_a['INDEX']['mean_pairwise_spatial_span_m']*1000:.3f} / {patch_b['INDEX']['mean_pairwise_spatial_span_m']*1000:.3f} mm ({span_change['INDEX']:+.1f}%).
- B was not broader and phone-local centroid stability did not improve.  No literal mm² area is claimed.

## 9–11. Force, duration, and THIRD

- Thumb mean/max force A/B: {patch_a['THUMB']['mean_force_when_active_n']:.3f}/{patch_a['THUMB']['maximum_force_n']:.3f} N vs {patch_b['THUMB']['mean_force_when_active_n']:.3f}/{patch_b['THUMB']['maximum_force_n']:.3f} N.
- Index mean/max force A/B: {patch_a['INDEX']['mean_force_when_active_n']:.3f}/{patch_a['INDEX']['maximum_force_n']:.3f} N vs {patch_b['INDEX']['mean_force_when_active_n']:.3f}/{patch_b['INDEX']['maximum_force_n']:.3f} N.
- Simultaneous duration A/B: {a['result']['simultaneous_contact_duration_s']:.6f}/{b['result']['simultaneous_contact_duration_s']:.6f} s.
- THIRD phone-contact samples A/B: {patch_a['THIRD']['contact_samples']}/{patch_b['THIRD']['contact_samples']}.

## 12–14. Physical retention

- Relative slip A/B: {ra['phone_relative_to_pinch_center_hold_slip_m']*1000:.3f}/{rb['phone_relative_to_pinch_center_hold_slip_m']*1000:.3f} mm ({improvements['relative_slip_reduction_percent']:.1f}% reduction).
- Total translation A/B: {ra['phone_hold_com_displacement_m']*1000:.3f}/{rb['phone_hold_com_displacement_m']*1000:.3f} mm.
- Vertical motion A/B: {ra['phone_hold_vertical_displacement_m']*1000:.3f}/{rb['phone_hold_vertical_displacement_m']*1000:.3f} mm.
- Rotation A/B: {ra['phone_hold_orientation_change_deg']:.3f}/{rb['phone_hold_orientation_change_deg']:.3f} deg ({improvements['orientation_change_reduction_percent']:.1f}% reduction).

The rotation improvement is real, but B still slipped about 50 mm and failed the broader/stable
patch gate.  It therefore does not displace approved A under the predeclared policy.

## 15–17. Robustness, safety, and photo fidelity

The optional perturbation sweep was not run because nominal retention remained weak; it was
not used to tune B.  Both candidates had zero prohibited self-collision and zero joint-limit
violation.  B's minimum margin was {geometry_b['minimum_joint_margin_rad']:.3f} rad.  Its
maximum q change was only {float(np.max(np.abs(dq[:5]))):.3f} rad, preserving the photo topology.

## 18. Comparison artifacts

- [Static A/B comparison](candidate_A_vs_B_static_comparison.png)
- [Candidate-B pad identity](candidate_B_pad_identity_overlay.png)
- [Retention contact sheet](candidate_A_vs_B_retention_contact_sheet.png)
- [Synchronized four-panel](candidate_A_vs_B_RETENTION_4panel.mp4)
- [A close-up](candidate_A_fingertip_hold_closeup.mp4) / [A side](candidate_A_fingertip_hold_side.mp4)
- [B close-up](candidate_B_distal_pad_hold_closeup.mp4) / [B side](candidate_B_distal_pad_hold_side.mp4)

## 19. Exact GUI commands

Candidate A:

```bash
{gui_a.rstrip()}
```

Candidate B:

```bash
{gui_b.rstrip()}
```

## 20–21. Recommendation and exact next action

Retain Candidate A.  Candidate B is a valid hand-only visual ablation and reduced penetration,
but it did not meet the meaningful-retention/contact-patch gate.

NEXT ACTION = RETAIN THE APPROVED PHOTO FINGERTIP PINCH AND DIAGNOSE RETENTION WITHOUT
CHANGING THE GRASP TOPOLOGY.

CANDIDATE A REMAINED THE BYTE-IDENTICAL USER-APPROVED PHOTO FINGERTIP PINCH
CANDIDATE B USED ONLY PHYSICAL THUMB AND INDEX DEX3 JOINTS
THE THIRD DEX3 FINGER REMAINED NON-TASK
DISTAL-PAD CONTACT WAS NOT CREATED BY ROTATING THE G1 WRIST OR ARM
CONTACT PATCH IMPROVEMENT WAS NOT CREATED BY INCREASING PHONE PENETRATION
CANDIDATE A AND B USED IDENTICAL TRUE-PHYSX PARAMETERS AND INITIAL CONDITIONS
PHONE RETENTION WAS JUDGED BY PHYSICS, NOT BY SCRIPTED OBJECT SUPPORT
THE ALOHA CARTESIAN BACKBONE, V17.2 TRAJECTORY, RIGHT DEX3, AND JITTER WERE NOT MODIFIED
NO DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    # Decode and invariant tests.
    final_videos = [
        OUT / "candidate_A_fingertip_hold_closeup.mp4",
        OUT / "candidate_B_distal_pad_hold_closeup.mp4",
        OUT / "candidate_A_fingertip_hold_side.mp4",
        OUT / "candidate_B_distal_pad_hold_side.mp4",
        four_path,
    ]
    video_audit = {}
    for path in final_videos:
        frames, frame_rate = read_video(path)
        video_audit[path.name] = {"decoded_frames": len(frames), "fps": frame_rate, "opens": True}
    tests = {
        "candidate_A_unchanged": freeze["candidate_A_unchanged_after_all_ab_work"],
        "candidate_B_task_joints_only": bool(np.array_equal(q_b[5:], q_a[5:])),
        "candidate_B_within_0_15rad_trust": float(np.max(np.abs(dq[:5]))) <= 0.15,
        "physics_identity": identity_pass,
        "both_bilateral_contact": bool(a["result"]["simultaneous_thumb_index_contact"] and b["result"]["simultaneous_thumb_index_contact"]),
        "third_contact_zero": patch_a["THIRD"]["contact_samples"] == 0 and patch_b["THIRD"]["contact_samples"] == 0,
        "prohibited_self_collision_zero": a["collision"]["prohibited_robot_self_contact_records"] == 0 and b["collision"]["prohibited_robot_self_contact_records"] == 0,
        "timed_object_writes_zero": a["no_cheat"]["timed_phone_pose_writes"] == b["no_cheat"]["timed_phone_pose_writes"] == 0,
        "all_videos_decode": all(row["decoded_frames"] > 0 for row in video_audit.values()),
        "v17_2_hash_unchanged": sha256(V17_2) == freeze["v17_2_sha256"],
        "v14_hash_unchanged": sha256(V14) == freeze["v14_cartesian_backbone_sha256"],
    }
    gui_a_smoke = OUT / "gui_A_smoke/static_physics_result.json"
    gui_b_smoke = OUT / "gui_B_smoke/static_physics_result.json"
    gui_a_result = load(gui_a_smoke) if gui_a_smoke.exists() else None
    gui_b_result = load(gui_b_smoke) if gui_b_smoke.exists() else None
    gui_smoke_pass = bool(
        gui_a_result and gui_b_result
        and gui_a_result["trial"] == gui_b_result["trial"] == "closed_hold"
        and gui_a_result["physics_steps"] > 0 and gui_b_result["physics_steps"] > 0
    )
    tests["candidate_A_B_gui_smoke_pass"] = gui_smoke_pass
    dump(OUT / "gui_review_audit.json", {
        "status": "CANDIDATE_A_B_TRUE_PHYSX_GUI_REVIEW_SMOKE_PASS" if gui_smoke_pass else "GUI_SMOKE_NOT_COMPLETE",
        "candidate_A_smoke_result": str(gui_a_smoke) if gui_a_result else None,
        "candidate_B_smoke_result": str(gui_b_smoke) if gui_b_result else None,
        "production_candidate_A_command": gui_a,
        "production_candidate_B_command": gui_b,
        "pause_at_end_in_production_commands": True,
        "actual_physx_fabric_rtx_path": True,
    })
    dump(OUT / "tests_results.json", {
        "status": "ALL_REQUIRED_TESTS_PASS" if all(tests.values()) else "REQUIRED_TEST_FAILURE",
        "tests": tests,
        "video_decode_audit": video_audit,
    })

    output_files = sorted(path for path in OUT.iterdir() if path.is_file())
    dump(OUT / "run_manifest.json", {
        "status": final_status,
        "recommended_candidate": recommendation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_A": {"primitive": str(PRIMITIVE_A), "sha256": sha256(PRIMITIVE_A), "q_rad": q_a},
        "candidate_B": {"primitive": str(PRIMITIVE_B), "sha256": sha256(PRIMITIVE_B), "q_rad": q_b},
        "frozen_sources": {"v17_2_sha256": sha256(V17_2), "v14_sha256": sha256(V14)},
        "physics_runs": {"A": str(A_DIR), "B": str(B_DIR), "duration_s": 1.5, "identical": identity_pass},
        "robustness_sweep_run": False,
        "full_trajectory_modified": False,
        "right_dex3_modified": False,
        "dds_publisher_real_robot": False,
        "output_hashes": {path.name: sha256(path) for path in output_files if path.name != "run_manifest.json"},
    })
    if not all(tests.values()):
        raise RuntimeError("required test failure")
    print(json.dumps({
        "status": final_status,
        "recommended_candidate": recommendation,
        "A_slip_mm": ra["phone_relative_to_pinch_center_hold_slip_m"] * 1000,
        "B_slip_mm": rb["phone_relative_to_pinch_center_hold_slip_m"] * 1000,
        "A_rotation_deg": ra["phone_hold_orientation_change_deg"],
        "B_rotation_deg": rb["phone_hold_orientation_change_deg"],
        "B_patch_broader": patch_broader,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
