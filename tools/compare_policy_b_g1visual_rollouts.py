#!/usr/bin/env python3
"""Create identical-view OLD Policy-B vs POLICY_B_G1VISUAL comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout"
NEW = ROOT / "outputs/policy_b_g1visual/closed_loop_rollout/full_policy_b_diagnostic_rollout"
OUTPUT = ROOT / "outputs/policy_b_g1visual/comparison"
CAMERAS = ("overview", "top", "side", "source_like")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def trace_metrics(path: Path) -> dict[str, Any]:
    trace = np.load(path, allow_pickle=False)
    names = trace["joint_names"].astype(str).tolist()
    actual = trace["actual_q"].astype(np.float64)
    left_wrist = trace["left_wrist_xyz_world_m"].astype(np.float64)
    right_wrist = trace["right_wrist_xyz_world_m"].astype(np.float64)
    dex = actual[:, 14:]
    return {
        "frames": len(actual),
        "duration_s": len(actual) / 30.0,
        "left_wrist_maximum_displacement_m": float(
            np.max(np.linalg.norm(left_wrist - left_wrist[:1], axis=1))
        ),
        "right_wrist_maximum_displacement_m": float(
            np.max(np.linalg.norm(right_wrist - right_wrist[:1], axis=1))
        ),
        "left_dex3_largest_joint_range_rad": float(np.max(np.ptp(dex[:, :7], axis=0))),
        "right_dex3_largest_joint_range_rad": float(np.max(np.ptp(dex[:, 7:], axis=0))),
        "joint_names": names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--new", type=Path, default=NEW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--old-label", default="OLD POLICY B")
    parser.add_argument("--new-label", default="POLICY B G1VISUAL")
    args = parser.parse_args()
    old, new, output = args.old.resolve(), args.new.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    old_report, new_report = read_json(old / "stage_report.json"), read_json(new / "stage_report.json")
    invariants = {
        "camera_hash_identical": old_report["camera_config_sha256"] == new_report["camera_config_sha256"],
        "scene_dataset_action_hash_identical": old_report["dataset_action_trajectory_set_sha256"]
        == new_report["dataset_action_trajectory_set_sha256"],
        "joint_order_identical": old_report["joint_names"] == new_report["joint_names"],
        "initial_state_identical": np.array_equal(
            np.asarray(old_report["initial_measured_state_rad"]),
            np.asarray(new_report["initial_measured_state_rad"]),
        ),
        "execution_horizon_identical": old_report["execution_horizon_frames"]
        == new_report["execution_horizon_frames"] == 4,
        "control_fps_identical": old_report["control_fps"] == new_report["control_fps"] == 30,
        "task_identical": old_report["task"] == new_report["task"],
        "object_visualization_mode_identical": old_report["object_visualization_mode"]
        == new_report["object_visualization_mode"],
    }
    if not all(invariants.values()):
        raise RuntimeError(f"old/new controlled-comparison invariant failed: {invariants}")
    videos = {}
    for camera in CAMERAS:
        path = output / f"old_vs_g1visual_{camera}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(old / f"{camera}.mp4"),
                "-i", str(new / f"{camera}.mp4"), "-filter_complex",
                f"[0:v]drawtext=text='{args.old_label}':x=8:y=8:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
                f"[1:v]drawtext=text='{args.new_label}':x=8:y=8:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6[b];"
                "[a][b]hstack=inputs=2:shortest=1[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", str(path),
            ],
            check=True,
        )
        videos[camera] = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "schema_version": "policy_b_old_vs_g1visual_rollout_comparison_v1",
        "status": "CONTROLLED_COMPARISON_READY_FOR_VISUAL_REVIEW",
        "labels": {"old": args.old_label, "new": args.new_label},
        "invariants": invariants,
        "old_policy": {
            "checkpoint": old_report["checkpoint"],
            "model_sha256": old_report["checkpoint_model_sha256"],
            "status": old_report["status"],
            "metrics": trace_metrics(old / "rollout_trace.npz"),
        },
        "g1visual_policy": {
            "checkpoint": new_report["checkpoint"],
            "model_sha256": new_report["checkpoint_model_sha256"],
            "status": new_report["status"],
            "metrics": trace_metrics(new / "rollout_trace.npz"),
        },
        "videos": videos,
        "qualitative_progression": "PENDING_HUMAN_VISUAL_REVIEW",
        "physical_object_success_evaluated": False,
        "real_robot_invoked": False,
    }
    atomic_json(output / "comparison_manifest.json", result)
    print(json.dumps({
        "status": result["status"],
        "videos": {key: row["path"] for key, row in videos.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
