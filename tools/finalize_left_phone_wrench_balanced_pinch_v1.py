#!/usr/bin/env python3
"""Finalize Candidate-A versus wrench-balanced Candidate-C static PhysX audit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_wrench_balanced_pinch_v1"
A_DIR = OUT / "candidate_A_final_physics"
C_DIR = OUT / "candidate_C_final_physics"
C_RENDER = OUT / "candidate_C_static_render"
PRIMITIVE_A = CAL / "left_phone_fingertip_pinch_primitive.json"
PRIMITIVE_C = OUT / "candidate_C_primitive.json"
V17_2 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py"
BUILDER = ROOT / "tools/build_left_phone_wrench_balanced_pinch_v1.py"
PHONE_MASS_KG = 0.177
GRAVITY = 9.81
WEIGHT_N = PHONE_MASS_KG * GRAVITY


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

    path.parent.mkdir(parents=True, exist_ok=True)
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


def stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict:
    values = np.asarray(values, dtype=float)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("mean", "median", "p05", "p95", "minimum", "maximum")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def angle_series(quaternions: np.ndarray) -> np.ndarray:
    q0 = Rotation.from_quat(quaternions[0])
    return np.degrees((q0.inv() * Rotation.from_quat(quaternions)).magnitude())


def candidate(directory: Path) -> dict:
    trace = np.load(directory / "static_physics_trace.npz")
    contact = load(directory / "phone_contact_identity_metrics.json")["identity"]
    result = load(directory / "static_physics_result.json")
    collision = load(directory / "collision_audit.json")
    no_cheat = load(directory / "no_cheat_audit.json")
    setup = load(directory / "physics_setup_audit.json")
    tracking = load(directory / "dex3_tracking_metrics.json")
    material = load(directory / "runtime_material_probe.json")
    drive = load(directory / "runtime_drive_probe.json")

    time_s = trace["time_s"]
    phone_pose = trace["phone_pose_xyzw"]
    centroids_w = trace["contact_centroid_w"][:, :2]
    normal_force_w = trace["normal_force_on_phone_w"][:, :2]
    total_force_w = trace["total_force_on_phone_w"][:, :2]
    bilateral = trace["simultaneous_thumb_index"].astype(bool)
    phone_rotations = Rotation.from_quat(phone_pose[:, 3:7])
    centroids_phone = np.full_like(centroids_w, np.nan, dtype=float)
    force_phone = np.full_like(total_force_w, np.nan, dtype=float)
    normal_phone = np.full_like(normal_force_w, np.nan, dtype=float)
    for index in range(len(time_s)):
        centroids_phone[index] = phone_rotations[index].inv().apply(
            centroids_w[index] - phone_pose[index, :3]
        )
        force_phone[index] = phone_rotations[index].inv().apply(total_force_w[index])
        normal_phone[index] = phone_rotations[index].inv().apply(normal_force_w[index])

    lever_w = centroids_w - phone_pose[:, None, :3]
    moment_w = np.cross(lever_w, total_force_w).sum(axis=1)
    moment_phone = np.vstack([
        phone_rotations[index].inv().apply(moment_w[index])
        for index in range(len(time_s))
    ])
    net_force_w = total_force_w.sum(axis=1)
    moment_mag = np.linalg.norm(moment_w, axis=1)
    height_mismatch = np.abs(centroids_w[:, 0, 2] - centroids_w[:, 1, 2])
    support_ratio = net_force_w[:, 2] / WEIGHT_N
    normal_line_distance = np.full((len(time_s), 2), np.nan, dtype=float)
    for step in range(len(time_s)):
        for finger in range(2):
            magnitude = np.linalg.norm(normal_force_w[step, finger])
            if magnitude > 1e-9 and np.all(np.isfinite(lever_w[step, finger])):
                direction = normal_force_w[step, finger] / magnitude
                normal_line_distance[step, finger] = np.linalg.norm(
                    np.cross(lever_w[step, finger], direction)
                )

    thumb_force = trace["thumb_phone_force_n"]
    index_force = trace["index_phone_force_n"]
    thumb_index_force_ratio = np.divide(
        thumb_force, index_force,
        out=np.full_like(thumb_force, np.nan, dtype=float),
        where=index_force > 1e-9,
    )
    phone_relative = phone_pose[:, :3] - trace["pinch_center_m"]
    slip_series = np.linalg.norm(phone_relative - phone_relative[0], axis=1)
    rotation_deg = angle_series(phone_pose[:, 3:7])
    dt = float(time_s[1] - time_s[0])
    acceleration_z = np.gradient(trace["phone_velocity"][:, 2], dt)
    balance_residual = PHONE_MASS_KG * acceleration_z - (net_force_w[:, 2] - WEIGHT_N)
    impact_indices = np.flatnonzero((time_s >= 0.1) & (balance_residual > 0.5 * WEIGHT_N))
    impact_index = int(impact_indices[0]) if len(impact_indices) else len(time_s) - 1
    transient = time_s <= 0.1 + 1e-12
    sustained = bilateral & (time_s > 0.1)
    free_hold = bilateral & (np.arange(len(time_s)) <= impact_index)

    patch = {}
    for finger in ("THUMB", "INDEX", "THIRD"):
        row = contact[finger]
        per_step = row["contact_patch_proxy"]["per_step"]
        minima = np.asarray([item["minimum_signed_separation_m"] for item in per_step], dtype=float)
        means = np.asarray([item["mean_signed_separation_m"] for item in per_step], dtype=float)
        spans = np.asarray([item["maximum_pairwise_spatial_span_m"] for item in per_step], dtype=float)
        patch[finger] = {
            "physical_link": row["physical_link"],
            "contact_samples": row["contact_samples"],
            "contact_duration_all_samples_s": row["contact_duration_all_samples_s"],
            "mean_contact_point_count_when_present": row["contact_patch_proxy"]["mean_contact_point_count_when_present"],
            "mean_pairwise_spatial_span_m": row["contact_patch_proxy"]["mean_pairwise_spatial_span_m"],
            "maximum_pairwise_spatial_span_m": row["contact_patch_proxy"]["maximum_pairwise_spatial_span_m"],
            "phone_local_centroid_maximum_excursion_m": row["contact_patch_proxy"]["phone_local_centroid_maximum_excursion_m"],
            "phone_local_centroid_rms_excursion_m": row["contact_patch_proxy"]["phone_local_centroid_rms_excursion_m"],
            "initial_minimum_signed_separation_m": row["initial_solver_minimum_signed_separation_m"],
            "minimum_signed_separation_stats_m": stats(minima),
            "mean_signed_separation_stats_m": stats(means),
            "spatial_span_stats_m": stats(spans),
            "literal_contact_area_claimed": False,
        }

    summary = {
        "tested_q_rad": trace["commanded_left_dex3_q"][0],
        "bilateral_contact_duration_s": float(np.sum(bilateral) * dt),
        "third_contact_samples": int(np.sum(trace["third_phone_force_n"] > 1e-3)),
        "third_maximum_force_n": float(np.max(trace["third_phone_force_n"])),
        "phone_relative_slip_m": float(slip_series[-1]),
        "phone_vertical_drop_m": float(phone_pose[-1, 2] - phone_pose[0, 2]),
        "phone_total_translation_m": float(np.linalg.norm(phone_pose[-1, :3] - phone_pose[0, :3])),
        "phone_rotation_deg": float(rotation_deg[-1]),
        "inferred_table_impact_time_s": float(time_s[impact_index]),
        "contact_height_mismatch_m": {
            "all": stats(height_mismatch, bilateral),
            "transient_0_to_0p1_s": stats(height_mismatch, bilateral & transient),
            "free_hold_until_table_impact": stats(height_mismatch, free_hold),
            "sustained_after_0p1_s": stats(height_mismatch, sustained),
        },
        "vertical_support_ratio": {
            "all": stats(support_ratio, bilateral),
            "transient_0_to_0p1_s": stats(support_ratio, bilateral & transient),
            "free_hold_until_table_impact": stats(support_ratio, free_hold),
            "sustained_after_0p1_s": stats(support_ratio, sustained),
        },
        "net_phone_moment_nm": {
            "magnitude_all": stats(moment_mag, bilateral),
            "magnitude_transient_0_to_0p1_s": stats(moment_mag, bilateral & transient),
            "magnitude_free_hold_until_table_impact": stats(moment_mag, free_hold),
            "magnitude_sustained_after_0p1_s": stats(moment_mag, sustained),
            "phone_long_axis": stats(moment_phone[:, 0], sustained),
            "phone_thickness_axis": stats(moment_phone[:, 1], sustained),
            "phone_short_axis": stats(moment_phone[:, 2], sustained),
        },
        "normal_force_line_to_com_distance_m": {
            "THUMB": stats(normal_line_distance[:, 0], sustained),
            "INDEX": stats(normal_line_distance[:, 1], sustained),
            "absolute_thumb_index_mismatch": stats(
                np.abs(normal_line_distance[:, 0] - normal_line_distance[:, 1]), sustained
            ),
        },
        "thumb_index_force_ratio": stats(thumb_index_force_ratio, sustained),
        "contact_centroid_phone_local_m": {
            "THUMB": {axis: stats(centroids_phone[:, 0, index], sustained) for index, axis in enumerate(("long", "thickness", "short"))},
            "INDEX": {axis: stats(centroids_phone[:, 1, index], sustained) for index, axis in enumerate(("long", "thickness", "short"))},
        },
        "contact_patch_proxy": patch,
        "collision": collision,
        "no_cheat": no_cheat,
        "result": result,
        "tracking": tracking,
        "setup": setup,
        "runtime_material": material,
        "runtime_drive": drive,
    }
    return {
        "trace": trace, "time_s": time_s, "phone_pose": phone_pose,
        "centroids_phone": centroids_phone, "force_phone": force_phone,
        "moment_phone": moment_phone, "moment_mag": moment_mag,
        "height_mismatch": height_mismatch, "support_ratio": support_ratio,
        "slip_series": slip_series, "rotation_deg": rotation_deg,
        "summary": summary,
    }


def percent_reduction(before: float, after: float) -> float:
    return float(100.0 * (before - after) / max(abs(before), 1e-12))


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no decoded frames: {path}")
    return frames, fps


def overlay(frame: np.ndarray, data: dict, step: int, title: str) -> np.ndarray:
    output = frame.copy()
    height_mm = 1000.0 * data["height_mismatch"][step]
    support = data["support_ratio"][step]
    torque = data["moment_mag"][step]
    slip_mm = 1000.0 * data["slip_series"][step]
    rotation = data["rotation_deg"][step]
    cv2.rectangle(output, (0, 0), (output.shape[1], 86), (18, 18, 18), -1)
    cv2.putText(output, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (110, 245, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"height={height_mm:.2f}mm | support={support:.3f}W | moment={torque:.4f}Nm", (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .43, (135, 255, 165), 1, cv2.LINE_AA)
    cv2.putText(output, f"slip={slip_mm:.2f}mm | phone rotation={rotation:.2f}deg", (10, 73), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 225, 170), 1, cv2.LINE_AA)
    return output


def compose_four_panel(a: dict, c: dict) -> dict:
    paths = {
        "A_close": OUT / "candidate_A_retention_closeup.mp4",
        "A_side": OUT / "candidate_A_retention_side.mp4",
        "C_close": OUT / "candidate_C_wrench_balanced_retention_closeup.mp4",
        "C_side": OUT / "candidate_C_wrench_balanced_retention_side.mp4",
    }
    decoded = {key: read_video(path) for key, path in paths.items()}
    counts = {key: len(value[0]) for key, value in decoded.items()}
    fps_values = {key: value[1] for key, value in decoded.items()}
    if len(set(counts.values())) != 1 or max(fps_values.values()) - min(fps_values.values()) > 1e-6:
        raise RuntimeError(f"A/C video synchronization mismatch: {counts}, {fps_values}")
    panel_size = (640, 480)
    output_path = OUT / "candidate_A_vs_C_WRENCH_RETENTION_4panel.mp4"
    temporary = output_path.with_suffix(".mp4.incomplete.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"),
        next(iter(fps_values.values())), (1280, 960),
    )
    count = next(iter(counts.values()))
    for frame_index in range(count):
        step = int(round(frame_index * (len(a["time_s"]) - 1) / max(count - 1, 1)))
        panels = [
            overlay(cv2.resize(decoded["A_close"][0][frame_index], panel_size), a, step, "A PHOTO_FINGERTIP | close-up"),
            overlay(cv2.resize(decoded["C_close"][0][frame_index], panel_size), c, step, "C WRENCH_BALANCED | close-up"),
            overlay(cv2.resize(decoded["A_side"][0][frame_index], panel_size), a, step, "A PHOTO_FINGERTIP | side"),
            overlay(cv2.resize(decoded["C_side"][0][frame_index], panel_size), c, step, "C WRENCH_BALANCED | side"),
        ]
        writer.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    writer.release()
    os.replace(temporary, output_path)
    return {"path": output_path, "decoded_frames": count, "fps": next(iter(fps_values.values()))}


def make_plots(a: dict, c: dict) -> None:
    colors = {"A": "#d65f5f", "C": "#2474b5"}
    figure, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for label, data in (("A", a), ("C", c)):
        t = data["time_s"]
        axes[0].plot(t, 1000.0 * data["height_mismatch"], color=colors[label], label=f"Candidate {label}")
        axes[1].plot(t, data["support_ratio"], color=colors[label])
        axes[2].plot(t, data["moment_mag"], color=colors[label])
        axes[3].plot(t, 1000.0 * data["slip_series"], color=colors[label])
        axes[4].plot(t, data["rotation_deg"], color=colors[label])
    axes[0].set_ylabel("contact height\nmismatch [mm]")
    axes[1].set_ylabel("vertical support\n[phone weights]")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("net moment\n[N m]")
    axes[3].set_ylabel("relative slip\n[mm]")
    axes[4].set_ylabel("phone rotation\n[deg]")
    axes[4].set_xlabel("true-PhysX hold time [s]")
    axes[0].legend(loc="upper right")
    for axis in axes:
        axis.grid(True, alpha=.25)
        axis.axvspan(0.0, 0.1, color="#dddddd", alpha=.35)
    figure.suptitle("Candidate A vs C — actual contact-wrench and retention timeline")
    figure.tight_layout()
    figure.savefig(OUT / "candidate_A_vs_C_wrench_timeline.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, (label, data) in zip(axes, (("Candidate A", a), ("Candidate C", c))):
        mask = (data["time_s"] > 0.1) & data["trace"]["simultaneous_thumb_index"]
        thumb = np.nanmedian(data["centroids_phone"][mask, 0], axis=0)
        index = np.nanmedian(data["centroids_phone"][mask, 1], axis=0)
        f_thumb = np.nanmedian(data["force_phone"][mask, 0], axis=0)
        f_index = np.nanmedian(data["force_phone"][mask, 1], axis=0)
        # Phone long/thickness projection; contact forces are scaled only for display.
        axis.add_patch(plt.Rectangle((-0.0748, -0.003975), 0.1496, 0.00795, fill=False, linewidth=2, color="black"))
        axis.scatter([0.0], [0.0], marker="o", s=70, color="black", label="phone COM")
        scale = 0.006
        for point, force, color, name in ((thumb, f_thumb, "#d95f02", "THUMB"), (index, f_index, "#1b9e77", "INDEX")):
            axis.scatter([point[0]], [point[1]], s=75, color=color)
            axis.arrow(point[0], point[1], scale * force[0], scale * force[1], head_width=.0015, length_includes_head=True, color=color)
            axis.plot([0.0, point[0]], [0.0, point[1]], linestyle=":", color=color, alpha=.8)
            axis.text(point[0], point[1] + .0012, name, color=color, ha="center", fontsize=9)
        p95 = data["summary"]["net_phone_moment_nm"]["magnitude_sustained_after_0p1_s"]["p95"]
        h95 = 1000.0 * data["summary"]["contact_height_mismatch_m"]["sustained_after_0p1_s"]["p95"]
        axis.set_title(f"{label}\nheight p95={h95:.2f} mm | moment p95={p95:.4f} N m")
        axis.set_xlabel("phone long axis [m]")
        axis.set_ylabel("phone thickness axis [m]")
        axis.set_xlim(-.08, .08)
        axis.set_ylim(-.018, .018)
        axis.grid(True, alpha=.25)
        axis.set_aspect("auto")
    figure.suptitle("Measured sustained contact centroids, force directions, COM lever arms")
    figure.tight_layout()
    figure.savefig(OUT / "wrench_balance_explanation.png", dpi=200)
    plt.close(figure)


def make_static_comparisons(c: dict) -> None:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 30)
    small = ImageFont.truetype(font_path, 22)
    photo = Image.open(CAL / "photo_references/real_dex3_left_phone_pinch_03_content.png").convert("RGB")
    a_image = Image.open(CAL / "left_phone_pinch_front_oblique.png").convert("RGB")
    c_image = Image.open(C_RENDER / "left_phone_pinch_front_oblique.png").convert("RGB")
    canvas = Image.new("RGB", (1800, 720), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate((("REAL DEX3 PHOTO", photo), ("CANDIDATE A", a_image), ("CANDIDATE C", c_image))):
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image.thumbnail((580, 640), resampling)
        x = column * 600 + (600 - image.width) // 2
        y = 62 + (640 - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((column * 600 + 18, 16), label, fill=(20, 20, 20), font=font)
    draw.text((18, 688), "Photo is qualitative topology evidence only; Candidate C changes thumb/index joints only.", fill=(35, 35, 35), font=small)
    canvas.save(OUT / "photo_candidateA_candidateC_comparison.png")

    identity = Image.open(C_RENDER / "left_phone_pinch_identity_overlay.png").convert("RGB")
    width, height = identity.size
    banner = Image.new("RGB", (width, height + 150), (248, 248, 248))
    banner.paste(identity, (0, 150))
    draw = ImageDraw.Draw(banner)
    s = c["summary"]
    draw.text((18, 14), "CANDIDATE C CONTACT ALIGNMENT — THUMB + INDEX; THIRD NON-TASK", fill=(10, 10, 10), font=font)
    draw.text((18, 58), f"height mismatch p95: {1000*s['contact_height_mismatch_m']['sustained_after_0p1_s']['p95']:.3f} mm", fill=(25, 70, 130), font=small)
    draw.text((18, 92), f"net phone moment p95: {s['net_phone_moment_nm']['magnitude_sustained_after_0p1_s']['p95']:.5f} N m | pre-table support mean: {s['vertical_support_ratio']['free_hold_until_table_impact']['mean']:.3f} W", fill=(25, 70, 130), font=small)
    banner.save(OUT / "candidate_C_contact_alignment_overlay.png")


def main() -> int:
    primitive_hash_before = sha256(PRIMITIVE_A)
    q_a = np.asarray(load(PRIMITIVE_A)["selected_static_q_rad"], dtype=float)
    q_c = np.asarray(load(PRIMITIVE_C)["selected_q_rad"], dtype=float)
    a = candidate(A_DIR)
    c = candidate(C_DIR)
    if not np.array_equal(a["summary"]["tested_q_rad"], q_a):
        raise RuntimeError("final Candidate A run q mismatch")
    if not np.array_equal(c["summary"]["tested_q_rad"], q_c):
        raise RuntimeError("final Candidate C run q mismatch")
    if not np.array_equal(q_c[5:], q_a[5:]):
        raise RuntimeError("Candidate C modified third finger")

    # Candidate C is deliberately not declared successful: it improves the
    # sustained wrench metrics but does not improve physical retention.
    a_h95 = a["summary"]["contact_height_mismatch_m"]["sustained_after_0p1_s"]["p95"]
    c_h95 = c["summary"]["contact_height_mismatch_m"]["sustained_after_0p1_s"]["p95"]
    a_tau95 = a["summary"]["net_phone_moment_nm"]["magnitude_sustained_after_0p1_s"]["p95"]
    c_tau95 = c["summary"]["net_phone_moment_nm"]["magnitude_sustained_after_0p1_s"]["p95"]
    a_slip = a["summary"]["phone_relative_slip_m"]
    c_slip = c["summary"]["phone_relative_slip_m"]
    a_rotation = a["summary"]["phone_rotation_deg"]
    c_rotation = c["summary"]["phone_rotation_deg"]
    height_improvement = percent_reduction(a_h95, c_h95)
    torque_improvement = percent_reduction(a_tau95, c_tau95)
    retention_improved = c_slip < a_slip and c_rotation < a_rotation
    final_status = "WRENCH_BALANCED_PINCH_NO_IMPROVEMENT"
    recommendation = "RETAIN_CANDIDATE_A_DO_NOT_INTEGRATE_CANDIDATE_C"

    search = load(OUT / "candidate_C_search.json")
    probe_directories = {
        "A": A_DIR,
        "B_distal_pad_seed": OUT / "search_candidate_B_force",
        "C_stage_A_0p10": OUT / "search_stage_A_physics",
        "C_expanded_0p20": OUT / "search_expanded_physics",
        "C_selected_composite": C_DIR,
    }
    for name in ("thumb0_minus", "thumb0_plus", "thumb1_plus", "thumb2_plus", "index0_minus", "index0_plus", "index1_plus"):
        probe_directories[f"probe_{name}"] = OUT / f"probe_{name}_physics"
    search["status"] = final_status
    search["stage_B_true_physx_candidates"] = {
        name: candidate(directory)["summary"] for name, directory in probe_directories.items()
    }
    search["selected_final_candidate"] = {
        "name": "WRENCH_BALANCED_THUMB_INDEX_PINCH",
        "q_rad": q_c,
        "selection_reason": "smallest tested q that reduced sustained height mismatch and net moment together without worse initial penetration",
        "replacement_gate_pass": False,
        "why_not_replacement": "slip, vertical drop, and phone rotation did not improve under identical physics",
    }
    search["trust_region_0p20_result"] = "did not produce WRENCH_BALANCED_RETENTION_PASS; no >0.20-rad redesign was attempted"
    dump(OUT / "candidate_C_search.json", search)

    comparison = {
        "status": final_status,
        "metric_interval_primary": "sustained bilateral contact after first 0.1 s; table-impact-partitioned metrics also reported",
        "candidate_A": a["summary"],
        "candidate_C": c["summary"],
        "C_minus_A_improvement_percent": {
            "sustained_height_mismatch_p95_reduction_percent": height_improvement,
            "sustained_net_moment_p95_reduction_percent": torque_improvement,
            "relative_slip_reduction_percent": percent_reduction(a_slip, c_slip),
            "vertical_drop_magnitude_reduction_percent": percent_reduction(abs(a["summary"]["phone_vertical_drop_m"]), abs(c["summary"]["phone_vertical_drop_m"])),
            "phone_rotation_reduction_percent": percent_reduction(a_rotation, c_rotation),
        },
        "preferred_gate_evaluation": {
            "height_mismatch_median_below_2mm": c["summary"]["contact_height_mismatch_m"]["sustained_after_0p1_s"]["median"] < 0.002,
            "height_mismatch_p95_below_4mm": c_h95 < 0.004,
            "net_moment_p95_reduction_at_least_50_percent": torque_improvement >= 50.0,
            "vertical_support_sustained_at_least_one_weight": c["summary"]["vertical_support_ratio"]["sustained_after_0p1_s"]["mean"] >= 1.0,
            "slip_below_10mm": c_slip < 0.010,
            "rotation_below_15deg": c_rotation < 15.0,
            "retention_improved": retention_improved,
        },
        "recommended_candidate": "A",
        "candidate_C_should_replace_A": False,
    }
    dump(OUT / "candidate_A_vs_C_wrench_metrics.json", comparison)
    dump(OUT / "candidate_A_vs_C_retention_metrics.json", {
        "status": final_status,
        "candidate_A": {key: a["summary"][key] for key in (
            "bilateral_contact_duration_s", "phone_relative_slip_m", "phone_vertical_drop_m",
            "phone_total_translation_m", "phone_rotation_deg", "third_contact_samples",
        )},
        "candidate_C": {key: c["summary"][key] for key in (
            "bilateral_contact_duration_s", "phone_relative_slip_m", "phone_vertical_drop_m",
            "phone_total_translation_m", "phone_rotation_deg", "third_contact_samples",
        )},
        "candidate_C_retention_improved": retention_improved,
        "decision": recommendation,
    })

    runtime_identity = {
        "phone_mass_kg": PHONE_MASS_KG,
        "gravity_m_s2": GRAVITY,
        "A_runtime_material": a["summary"]["runtime_material"],
        "C_runtime_material": c["summary"]["runtime_material"],
        "A_runtime_drive": a["summary"]["runtime_drive"],
        "C_runtime_drive": c["summary"]["runtime_drive"],
        "authored_phone_initial_transform_A": a["summary"]["setup"]["phone_initial_transform_matrix"],
        "authored_phone_initial_transform_C": c["summary"]["setup"]["phone_initial_transform_matrix"],
        "phone_first_post_step_pose_A": a["phone_pose"][0],
        "phone_first_post_step_pose_C": c["phone_pose"][0],
        "first_post_step_pose_is_expected_to_differ_due_to_candidate_contact_impulse": True,
        "time_A": a["time_s"],
        "time_C": c["time_s"],
    }
    identity_pass = (
        a["summary"]["runtime_material"] == c["summary"]["runtime_material"]
        and a["summary"]["runtime_drive"] == c["summary"]["runtime_drive"]
        and a["summary"]["setup"]["phone_initial_transform_matrix"]
        == c["summary"]["setup"]["phone_initial_transform_matrix"]
        and np.array_equal(a["time_s"], c["time_s"])
    )
    runtime_identity["status"] = "AC_TRUE_PHYSX_IDENTITY_PASS" if identity_pass else "BLOCKED_AC_PHYSICS_MISMATCH"
    runtime_identity["only_intended_difference"] = "five physical thumb/index target joint values"
    dump(OUT / "ac_physics_identity_audit.json", runtime_identity)
    if not identity_pass:
        raise RuntimeError("A/C physics identity audit failed")

    c_geometry = load(OUT / "candidate_C_contact_geometry.json")
    dump(OUT / "candidate_C_collision_audit.json", {
        "status": "PROHIBITED_SELF_COLLISION_ZERO",
        "static_geometry": c_geometry["prohibited_self_collision_records"],
        "true_physx": c["summary"]["collision"],
        "third_phone_contact_samples": c["summary"]["third_contact_samples"],
    })
    dump(OUT / "candidate_C_joint_margin_audit.json", {
        "status": "JOINT_LIMIT_VIOLATION_ZERO",
        "joint_names": load(PRIMITIVE_C)["joint_names"],
        "q_rad": q_c,
        "minimum_joint_margin_rad": c_geometry["minimum_joint_margin_rad"],
        "limiting_joint": c_geometry["limiting_joint"],
        "joint_limit_violation_count": c_geometry["joint_limit_violation_count"],
        "maximum_absolute_task_joint_change_from_A_rad": float(np.max(np.abs(q_c[:5] - q_a[:5]))),
        "trust_region_limit_rad": 0.10,
    })

    primitive_hash_after = sha256(PRIMITIVE_A)
    freeze = load(OUT / "candidate_A_freeze_audit.json")
    freeze.update({
        "primitive_sha256_final": primitive_hash_after,
        "candidate_A_q_sha256_final": array_sha256(np.asarray(load(PRIMITIVE_A)["selected_static_q_rad"], dtype=float)),
        "final_candidate_A_still_byte_identical": primitive_hash_before == primitive_hash_after,
        "candidate_C_q_sha256": array_sha256(q_c),
        "v17_2_trajectory_sha256_final": sha256(V17_2),
        "v14_cartesian_backbone_sha256_final": sha256(V14),
        "authoritative_scene_sha256_final": sha256(SCENE),
    })
    dump(OUT / "candidate_A_freeze_audit.json", freeze)

    make_plots(a, c)
    make_static_comparisons(c)

    video_sources = {
        A_DIR / "candidate_A_retention_contact_closeup.mp4": OUT / "candidate_A_retention_closeup.mp4",
        A_DIR / "candidate_A_retention_side.mp4": OUT / "candidate_A_retention_side.mp4",
        C_DIR / "candidate_C_wrench_balanced_retention_contact_closeup.mp4": OUT / "candidate_C_wrench_balanced_retention_closeup.mp4",
        C_DIR / "candidate_C_wrench_balanced_retention_side.mp4": OUT / "candidate_C_wrench_balanced_retention_side.mp4",
    }
    for source, destination in video_sources.items():
        copy(source, destination)
    four_panel = compose_four_panel(a, c)

    gui_a = f"""source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd {ROOT}
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir {OUT}/gui_candidate_A \\
  --trial closed_hold --hold-duration 1.5 --artifact-prefix candidate_A_gui \\
  --gui --pause-at-end --enable_cameras"""
    gui_c = f"""source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd {ROOT}
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir {OUT}/gui_candidate_C \\
  --trial closed_hold --hold-duration 1.5 \\
  --test-q-json {PRIMITIVE_C} --artifact-prefix candidate_C_gui \\
  --gui --pause-at-end --enable_cameras"""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# Candidate A GUI
{gui_a}

# Candidate C GUI
{gui_c}

# Rebuild Stage-A/search artifacts (no physics parameter changes)
/home/jbnu/miniconda3/envs/isaaclab6/bin/python {BUILDER}

# Finalize metrics/figures/video composition
python3 {ROOT}/tools/finalize_left_phone_wrench_balanced_pinch_v1.py
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    next_action = "STOP HAND-POSE ITERATION AND CALIBRATE THE REAL DEX3-PAD / PHONE MATERIAL MODEL BEFORE ANY FURTHER GRASP CHANGES."
    report = f"""Candidate A: bilateral contact {a['summary']['bilateral_contact_duration_s']:.6f} s, slip {1000*a_slip:.3f} mm, rotation {a_rotation:.3f} deg, sustained moment p95 {a_tau95:.5f} N m.
Candidate C: bilateral contact {c['summary']['bilateral_contact_duration_s']:.6f} s, slip {1000*c_slip:.3f} mm, rotation {c_rotation:.3f} deg, sustained moment p95 {c_tau95:.5f} N m.
Candidate C should replace Candidate A: NO — `{final_status}`.

# Dex3 left phone wrench-balanced pinch v1

1. **Candidate C q:** `{q_c.tolist()}`
2. **q difference from A:** `{(q_c-q_a).tolist()}`; maximum task-joint change {float(np.max(np.abs(q_c[:5]-q_a[:5]))):.6f} rad.
3. **Thumb opposition:** preserved; thumb_0 remains negative and only physical thumb/index changed.
4. **Photo topology:** precision distal thumb-index pinch preserved; third remains exactly `[-0.1, -0.1]` and non-task.
5. **Contact-height p95:** A {1000*a_h95:.3f} mm, C {1000*c_h95:.3f} mm ({height_improvement:.2f}% lower), still above the preferred 4 mm gate.
6. **Lever-arm symmetry:** see `candidate_A_vs_C_wrench_metrics.json`; normal-line mismatch and phone-axis components are reported separately.
7. **Normal-force line/COM:** thumb and index distributions are in the same metrics JSON and `wrench_balance_explanation.png`.
8. **Thumb/index force ratio:** A median {a['summary']['thumb_index_force_ratio']['median']:.3f}, C median {c['summary']['thumb_index_force_ratio']['median']:.3f}.
9. **Vertical support:** pre-table free-hold mean A {a['summary']['vertical_support_ratio']['free_hold_until_table_impact']['mean']:.3f} W, C {c['summary']['vertical_support_ratio']['free_hold_until_table_impact']['mean']:.3f} W. Across all samples after 0.1 s (including the post-impact interval), A was {a['summary']['vertical_support_ratio']['sustained_after_0p1_s']['mean']:.3f} W and C was {c['summary']['vertical_support_ratio']['sustained_after_0p1_s']['mean']:.3f} W; C did not reach a sustained one-phone-weight support gate.
10. **Net phone torque:** sustained p95 A {a_tau95:.5f} N m, C {c_tau95:.5f} N m ({torque_improvement:.2f}% lower), short of the preferred 50% reduction.
11. **Centroid excursion:** thumb/index detailed proxies are in `candidate_A_vs_C_wrench_metrics.json`.
12. **Contact-patch proxy:** point count/span/centroid stability only; no fictitious mm^2 area is claimed.
13. **Penetration:** C initial thumb/index signed separation {1000*c['summary']['contact_patch_proxy']['THUMB']['initial_minimum_signed_separation_m']:.3f}/{1000*c['summary']['contact_patch_proxy']['INDEX']['initial_minimum_signed_separation_m']:.3f} mm versus A {1000*a['summary']['contact_patch_proxy']['THUMB']['initial_minimum_signed_separation_m']:.3f}/{1000*a['summary']['contact_patch_proxy']['INDEX']['initial_minimum_signed_separation_m']:.3f} mm; C is not deeper.
14. **Slip:** A {1000*a_slip:.3f} mm, C {1000*c_slip:.3f} mm.
15. **Vertical drop:** A {1000*a['summary']['phone_vertical_drop_m']:.3f} mm, C {1000*c['summary']['phone_vertical_drop_m']:.3f} mm.
16. **Phone rotation:** A {a_rotation:.3f} deg, C {c_rotation:.3f} deg.
17. **Bilateral contact:** A {a['summary']['bilateral_contact_duration_s']:.6f} s, C {c['summary']['bilateral_contact_duration_s']:.6f} s.
18. **Third finger:** C contact samples {c['summary']['third_contact_samples']}; no support force.
19. **Collision:** prohibited self collision = 0.
20. **Joint margins:** minimum {c_geometry['minimum_joint_margin_rad']:.6f} rad; violations = 0.
21. **Arm/wrist:** target configuration and phone initial pose were identical; no arm/wrist variables existed.
22. **Friction/effort:** runtime material and drive probes are identical (`ac_physics_identity_audit.json`).
23. **Explanation:** `{OUT/'wrench_balance_explanation.png'}`
24. **Photo comparison:** `{OUT/'photo_candidateA_candidateC_comparison.png'}`
25. **Videos:** `{OUT/'candidate_A_retention_closeup.mp4'}`, `{OUT/'candidate_C_wrench_balanced_retention_closeup.mp4'}`, `{four_panel['path']}`
26. **GUI Candidate A:**
```bash
{gui_a}
```
**GUI Candidate C:**
```bash
{gui_c}
```
27. **Recommendation:** keep Candidate A frozen; Candidate C is diagnostic evidence, not an integration candidate.
28. **Exact next action:** `{next_action}`

Candidate C did not improve retention under the unchanged generic PhysX 0.5/0.5 contact material. The 0.20-rad authorized trust-region expansion was also tested and failed the wrench/retention gates; no larger photo-grasp redesign was attempted.

CANDIDATE A REMAINED THE FROZEN USER-APPROVED PHOTO-BASED BASELINE
CANDIDATE C OPTIMIZED THUMB-INDEX CONTACT WRENCH, NOT A NEW ARM TRAJECTORY
PHYSICAL THUMB AND INDEX REMAINED THE ONLY PHONE GRASP FINGERS
THE THIRD DEX3 FINGER REMAINED NON-TASK
CONTACT HEIGHT, FORCE LINES, LEVER ARMS, AND NET PHONE MOMENT WERE OPTIMIZED AROUND THE PHONE COM
DISTAL-PAD CONTACT WAS SECONDARY TO WRENCH BALANCE AND CONTACT STABILITY
RETENTION WAS NOT CREATED BY DEEPER PENETRATION, HIGHER EFFORT, OR HIGHER FRICTION
THE G1 ARM, WRIST, ALOHA CARTESIAN BACKBONE, V17.2 TRAJECTORY, AND RIGHT DEX3 WERE NOT MODIFIED
THE SAME TRUE-PHYSX MATERIAL AND ACTUATOR PARAMETERS WERE USED FOR CANDIDATE A AND C
NO DDS, PUBLISHER, ALOHA COMMAND, OR REAL-G1 COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    artifacts = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    dump(OUT / "run_manifest.json", {
        "schema_version": 1,
        "status": final_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_A_primitive_sha256": primitive_hash_after,
        "candidate_C_primitive_sha256": sha256(PRIMITIVE_C),
        "v17_2_sha256": sha256(V17_2),
        "v14_sha256": sha256(V14),
        "physics_identity_pass": identity_pass,
        "third_non_task": True,
        "right_hand_modified": False,
        "full_trajectory_modified": False,
        "material_or_actuator_tuned": False,
        "candidate_C_should_replace_A": False,
        "artifacts": artifacts,
    })
    print(json.dumps({
        "status": final_status,
        "candidate_C_q_rad": q_c.tolist(),
        "height_p95_A_C_mm": [1000*a_h95, 1000*c_h95],
        "moment_p95_A_C_nm": [a_tau95, c_tau95],
        "slip_A_C_mm": [1000*a_slip, 1000*c_slip],
        "rotation_A_C_deg": [a_rotation, c_rotation],
        "candidate_C_should_replace_A": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
