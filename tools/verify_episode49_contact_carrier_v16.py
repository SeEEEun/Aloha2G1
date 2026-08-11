#!/usr/bin/env python3
"""Regression/infrastructure checks for the v16 failed diagnostic candidate."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
RUNTIME = (
    ROOT / "tools/aloha_g1_v16/carrier.py",
    ROOT / "tools/aloha_g1_v16/trajectory.py",
    ROOT / "tools/build_episode49_contact_carrier_v16.py",
    ROOT / "isaaclab_magsafe_fixed_scene/render_contact_carrier_v16.py",
)


def decoded_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def main() -> int:
    freeze = json.loads((OUT / "input_freeze_audit.json").read_text(encoding="utf-8"))
    v15 = json.loads((OUT / "v15_failure_preservation_audit.json").read_text(encoding="utf-8"))
    semantic = json.loads((OUT / "semantic_runtime_audit.json").read_text(encoding="utf-8"))
    numeric = json.loads((OUT / "numeric_gate_summary.json").read_text(encoding="utf-8"))
    isaac = json.loads((OUT / "isaaclab_kinematic_validation.json").read_text(encoding="utf-8"))
    reference_indices = sorted({
        int(index)
        for index, names in semantic["generic_converter_dry_run"]["semantic_knots"]["semantic_events_by_knot"].items()
        if any(name not in ("trajectory_start", "trajectory_end") for name in names)
    })
    literal_hits = []
    for path in RUNTIME:
        text = path.read_text(encoding="utf-8")
        for value in reference_indices:
            for match in re.finditer(rf"(?<![0-9]){value}(?![0-9])", text):
                literal_hits.append({"path": str(path.resolve()), "index": value, "offset": match.start()})

    with np.load(SOURCE, allow_pickle=False) as source_payload, np.load(
        OUT / "arm_dex3_coupled_trajectory.npz", allow_pickle=False
    ) as trajectory:
        source_equal = np.array_equal(source_payload["optimized_action"], trajectory["optimized_action"])
        q = trajectory["controlled_q"]
        left = trajectory["left_dex3_q"]
        right = trajectory["right_dex3_q"]
        trajectory_checks = {
            "controlled_q_shape": list(q.shape),
            "finite": bool(np.isfinite(q).all()),
            "source_action_bytewise_array_equal": bool(source_equal),
            "left_dex3_peak_to_peak_rad": float(np.ptp(left, axis=0).max()),
            "right_dex3_peak_to_peak_rad": float(np.ptp(right, axis=0).max()),
            "root_xyz": trajectory["g1_root"].tolist(),
            "workspace_scale": float(trajectory["workspace_scale"]),
            "physics_steps": int(trajectory["physics_steps"]),
        }

    videos = {
        path.name: decoded_frames(path)
        for path in sorted(OUT.glob("*.mp4"))
    }
    checks = {
        "frozen_inputs_byte_identical": bool(freeze.get("byte_identical_after_final_render")),
        "v15_failed_diagnostic_byte_identical": bool(v15.get("byte_identical_after_final_render")),
        "runtime_semantic_literal_count_zero": len(literal_hits) == 0,
        "generic_semantic_api": bool(semantic.get("generic_api")),
        "validation_or_heldout_accessed_false": not bool(semantic.get("validation_or_heldout_accessed")),
        "source_action_unchanged": source_equal,
        "controlled_q_990x28": q.shape == (990, 28),
        "trajectory_finite": bool(np.isfinite(q).all()),
        "continuous_left_dex3_motion": float(np.ptp(left, axis=0).max()) > 0.05,
        "continuous_right_dex3_motion": float(np.ptp(right, axis=0).max()) > 0.05,
        "isaac_joint_readback_pass": float(isaac["maximum_requested_readback_error_rad"]) <= 1e-6,
        "isaac_actual_task_fingers_move": bool(isaac["actual_task_finger_links_move"]),
        "physics_steps_zero": int(isaac["physics_steps"]) == 0,
        "object_follow_disabled_when_contact_gate_failed": not bool(isaac["object_follow_enabled"]),
        "all_review_videos_990_frames": bool(videos) and all(value == 990 for value in videos.values()),
        "numeric_failure_not_promoted_to_success": not bool(numeric.get("final_pass")),
    }
    payload = {
        "status": "V16_DIAGNOSTIC_INFRASTRUCTURE_TESTS_PASS" if all(checks.values()) else "BLOCKED_V16_REGRESSION_TEST",
        "infrastructure_checks": checks,
        "task_gate_status": numeric["status"],
        "task_gate_pass": bool(numeric.get("final_pass")),
        "semantic_literal_hits": literal_hits,
        "trajectory": trajectory_checks,
        "videos": videos,
        "no_G1_expert_motion": True,
        "no_validation_or_heldout_trajectory": True,
        "no_DDS_publisher_hardware": True,
    }
    (OUT / "tests_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "task_gate_status": payload["task_gate_status"]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
