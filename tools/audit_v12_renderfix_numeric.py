#!/usr/bin/env python3
"""Immutable-input numeric audit for the v12 Isaac Lab render fix.

No target, residual, or IK computation is performed.  This script only reads
the existing v12 trajectory NPZ files and writes diagnostics to renderfix.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12_renderfix"
EXACT = V12 / "position_only_exact_arm_trajectory.npz"
NULLSPACE = V12 / "position_only_nullspace_arm_trajectory.npz"
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12.py"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
ACTIVE_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
KEY_FRAMES = [0, 169, 216, 319, 334, 523, 695, 989]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def dump(path: Path, payload) -> None:
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit_trajectory(path: Path) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        keys = payload.files
        if "g1_arm_q" not in keys or "g1_arm_joint_trajectory" not in keys:
            raise RuntimeError(f"required arm q keys absent from {path}")
        q = payload["g1_arm_q"].astype(float)
        q_alias = payload["g1_arm_joint_trajectory"].astype(float)
        names = payload["arm_joint_names"].astype(str)
        left_palm = payload["achieved_left_position_scene"].astype(float)
        right_palm = payload["achieved_right_position_scene"].astype(float)
        target_hashes = {
            key: array_sha256(payload[key])
            for key in (
                "base_aloha_derived_left_target", "base_aloha_derived_right_target",
                "phase_residual_left_translation", "phase_residual_right_translation",
                "corrected_left_position_scene", "corrected_right_position_scene",
                "corrected_left_rotation_scene", "corrected_right_rotation_scene",
            )
        }
    if not np.array_equal(q, q_alias):
        raise RuntimeError(f"g1_arm_q alias mismatch in {path}")
    if q.shape != (990, 14) or len(names) != 14 or not np.isfinite(q).all():
        raise RuntimeError(f"trajectory schema invariant failed for {path}")
    delta = np.diff(q, axis=0)
    rows = []
    for frame in KEY_FRAMES:
        rows.append({
            "frame": frame,
            "target_q": q[frame],
            "q_minus_frame_0": q[frame] - q[0],
            "q_difference_norm_rad": float(np.linalg.norm(q[frame] - q[0])),
            "numerical_left_palm_fk_m": left_palm[frame],
            "numerical_right_palm_fk_m": right_palm[frame],
            "left_palm_displacement_from_frame_0_m": float(np.linalg.norm(left_palm[frame] - left_palm[0])),
            "right_palm_displacement_from_frame_0_m": float(np.linalg.norm(right_palm[frame] - right_palm[0])),
        })
    report = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "selected_q_key": "g1_arm_q",
        "q_alias_key": "g1_arm_joint_trajectory",
        "q_keys_array_equal": True,
        "shape": list(q.shape),
        "joint_names": names.tolist(),
        "finite": True,
        "per_joint_min_rad": np.min(q, axis=0),
        "per_joint_max_rad": np.max(q, axis=0),
        "per_joint_peak_to_peak_rad": np.ptp(q, axis=0),
        "max_peak_to_peak_rad": float(np.max(np.ptp(q, axis=0))),
        "max_frame_to_frame_step_rad": float(np.max(np.abs(delta))),
        "q_array_sha256": array_sha256(q),
        "target_component_array_hashes": target_hashes,
        "max_left_palm_displacement_from_frame_0_m": float(np.max(np.linalg.norm(left_palm - left_palm[0], axis=1))),
        "max_right_palm_displacement_from_frame_0_m": float(np.max(np.linalg.norm(right_palm - right_palm[0], axis=1))),
        "keyframes": rows,
        "classification": "NUMERIC_Q_MOTION_PRESENT" if np.max(np.ptp(q, axis=0)) > 1e-6 else "BLOCKED_TRAJECTORY_EXPORT_OR_KEY_SELECTION",
    }
    return report, rows, q, names


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    input_paths = [EXACT, NULLSPACE, RENDERER, LAYOUT, FIXED_SCENE, ACTIVE_SCENE, TIMELINE, ALIGNMENT]
    hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    input_audit = {
        "status": "INPUT_HASHES_SEALED_BEFORE_RENDERFIX",
        "files": hashes,
        "original_trajectory_npz_must_remain_byte_identical": True,
        "target_recomputed": False,
        "ik_recomputed": False,
        "orientation_optimized": False,
        "dex3_applied": False,
        "physics": False,
        "dds_publisher_hardware": False,
    }
    dump(OUT / "input_hash_audit.json", input_audit)

    exact, exact_rows, exact_q, names = audit_trajectory(EXACT)
    nullspace, null_rows, null_q, null_names = audit_trajectory(NULLSPACE)
    if not np.array_equal(names, null_names):
        raise RuntimeError("Exact/Nullspace joint names differ")
    difference = np.abs(exact_q - null_q)
    motion = {
        "status": "NUMERIC_Q_MOTION_PRESENT",
        "source_npz_schema_provenance": "arm q is stored identically under g1_arm_q and g1_arm_joint_trajectory",
        "EXACT": exact,
        "NULLSPACE": nullspace,
        "exact_nullspace_difference": {
            "max_abs_rad": float(np.max(difference)),
            "mean_abs_rad": float(np.mean(difference)),
            "differing_frames": int(np.count_nonzero(np.any(difference > 1e-12, axis=1))),
            "differing_joints": int(np.count_nonzero(np.any(difference > 1e-12, axis=0))),
        },
        "numeric_q_motion_gate_pass": bool(
            exact["classification"] == "NUMERIC_Q_MOTION_PRESENT"
            and nullspace["classification"] == "NUMERIC_Q_MOTION_PRESENT"
        ),
    }
    dump(OUT / "trajectory_motion_audit.json", motion)

    with (OUT / "keyframe_q_difference.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["trajectory", "frame", "q_norm_from_frame_0_rad", "left_palm_displacement_m", "right_palm_displacement_m"]
        fieldnames += [f"q_{index}_{name}" for index, name in enumerate(names)]
        fieldnames += [f"dq_{index}_{name}" for index, name in enumerate(names)]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in (("EXACT", exact_rows), ("NULLSPACE", null_rows)):
            for row in rows:
                output = {
                    "trajectory": label,
                    "frame": row["frame"],
                    "q_norm_from_frame_0_rad": row["q_difference_norm_rad"],
                    "left_palm_displacement_m": row["left_palm_displacement_from_frame_0_m"],
                    "right_palm_displacement_m": row["right_palm_displacement_from_frame_0_m"],
                }
                output.update({f"q_{index}_{name}": row["target_q"][index] for index, name in enumerate(names)})
                output.update({f"dq_{index}_{name}": row["q_minus_frame_0"][index] for index, name in enumerate(names)})
                writer.writerow(output)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for axis, q, title in zip(axes, (exact_q, null_q), ("Exact g1_arm_q", "Nullspace g1_arm_q")):
        for index, name in enumerate(names):
            axis.plot(q[:, index], linewidth=0.85, label=name)
        for frame in KEY_FRAMES:
            axis.axvline(frame, color="black", alpha=0.15, linewidth=0.7)
        axis.set_ylabel("rad")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("action index")
    axes[0].legend(ncol=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "q_trajectory_plot.png", dpi=180)
    plt.close(fig)

    print(json.dumps({
        "status": motion["status"],
        "exact_max_peak_to_peak_rad": exact["max_peak_to_peak_rad"],
        "nullspace_max_peak_to_peak_rad": nullspace["max_peak_to_peak_rad"],
        "exact_max_left_palm_displacement_m": exact["max_left_palm_displacement_from_frame_0_m"],
        "exact_max_right_palm_displacement_m": exact["max_right_palm_displacement_from_frame_0_m"],
        "input_hashes": hashes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
