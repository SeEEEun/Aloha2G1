#!/usr/bin/env python3
"""Freeze the Isaac-only R2/1500 diagnostic video review.

This tool does not run inference or simulation.  It validates the already
recorded rollout and controlled-comparison artifacts, preserves the failed
checkpoint approval and real-hardware block, and records a qualitative visual
review of the motion observed before the mandatory runtime safety abort.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_ROOT = (
    ROOT
    / "outputs/policy_b_g1visual/rehearsal/r2_1500_diagnostic_closed_loop"
)
STAGE = ROLLOUT_ROOT / "full_policy_b_diagnostic_rollout"
COMPARISON = ROOT / "outputs/policy_b_g1visual/rehearsal/r2_1500_diagnostic_comparison"
REVIEW = ROOT / "outputs/policy_b_g1visual/rehearsal/r2_1500_diagnostic_review"
APPROVAL = ROOT / "outputs/policy_b_g1visual/rehearsal/final_analysis/final_analysis.json"
OLD_STAGE = ROOT / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout"
EXPECTED_MODEL_SHA256 = "a7eef56728f522bfe15bb46b70131a9908a63b048ab51a50c49dd3c2ff935042"
EXPECTED_CAMERA_SHA256 = "9b423e72e5ce2dd0ba550b406718fba225c556e931aab2cc08fb85ff51ea7dc1"
CAMERAS = ("overview", "source_like", "top", "side")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_info(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(process.stdout)["streams"][0]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "duration_s": float(stream["duration"]),
    }


def main() -> int:
    stage = read_json(STAGE / "stage_report.json")
    old = read_json(OLD_STAGE / "stage_report.json")
    approval = read_json(APPROVAL)
    comparison = read_json(COMPARISON / "comparison_manifest.json")
    abort = stage["diagnostic_abort_reason"]

    approval_checks = {
        "checkpoint_approval_remains_failed": not approval["checkpoint_gate"]["passed"]
        and approval["checkpoint_gate"]["approved_checkpoint"] is None,
        "r2_is_analysis_only_pareto_artifact": approval["checkpoint_gate"]
        ["best_pareto_analysis_artifact"]["model_sha256"]
        == EXPECTED_MODEL_SHA256,
        "checkpoint_hash": stage["checkpoint_model_sha256"] == EXPECTED_MODEL_SHA256,
        "checkpoint_step": int(stage["checkpoint_step"]) == 1500,
        "camera_frozen": stage["camera_config_sha256"] == EXPECTED_CAMERA_SHA256,
        "same_camera_as_original": stage["camera_config_sha256"]
        == old["camera_config_sha256"],
        "same_initial_state": np.array_equal(
            np.asarray(stage["initial_measured_state_rad"], dtype=np.float64),
            np.asarray(old["initial_measured_state_rad"], dtype=np.float64),
        ),
        "same_scene_action_provenance": stage["dataset_action_trajectory_set_sha256"]
        == old["dataset_action_trajectory_set_sha256"],
        "same_task": stage["task"] == old["task"],
        "same_dynamic_object_mode": stage["object_visualization_mode"]
        == old["object_visualization_mode"]
        == "DYNAMIC_CANONICAL_OBJECT_VISUALIZATION",
        "horizon_4": int(stage["execution_horizon_frames"]) == 4,
        "control_30_hz": float(stage["control_fps"]) == 30.0,
        "measured_state_input": stage["state_source"] == "ISAAC_MEASURED_G1_DEX3_28D",
        "source_like_rgb": stage["rgb_source"]
        in {"SOURCE_LIKE_CAM_HIGH", "CAMERA_CONFIG_DRIVEN_ISAAC_RGB"}
        and old["rgb_source"]
        in {"SOURCE_LIKE_CAM_HIGH", "CAMERA_CONFIG_DRIVEN_ISAAC_RGB"}
        and stage["camera"] == old["camera"] == "SOURCE_LIKE_CAM_HIGH",
        "real_robot_disabled": stage["real_robot_command_allowed"] is False,
    }
    if not all(approval_checks.values()):
        raise RuntimeError(f"diagnostic invariant failure: {approval_checks}")

    expected_abort_checks = dict(abort["checks"])
    runtime_abort_checks = {
        "rollout_marked_aborted": stage["status"] == "FAIL"
        and stage["full_policy_b_diagnostic_rollout"] == "ABORTED",
        "abort_at_executed_frame": int(abort["frame"]) == 435
        and int(stage["executed_control_frames"]) == 436,
        "only_external_contact_check_failed": [
            key for key, value in expected_abort_checks.items() if not value
        ]
        == ["external_non_task_collision"],
        "external_contact_force_is_material": float(
            abort["external_non_task_contact_force_n"]
        )
        > 10.0,
        "no_predicted_prefix_gate_failure": int(
            stage["executed_prefix_safety_summary"]["failed_inference_count"]
        )
        == 0,
        "no_self_collision": int(abort["self_collision_incidence"]) == 0,
        "no_branch_discontinuity": abort["branch_discontinuity"] is False,
        "commanded_hard_limits_clean": stage["rollout"]["checks"]
        ["commanded_hard_limits"],
        "finite_velocity_acceleration": stage["rollout"]["checks"]["velocity"]
        and stage["rollout"]["checks"]["acceleration"],
        "semantic_task_failure_not_used_as_stop_gate": stage[
            "task_semantics_used_as_stop_gate"
        ]
        is False,
    }
    if not all(runtime_abort_checks.values()):
        raise RuntimeError(f"runtime-abort provenance failure: {runtime_abort_checks}")

    trace = np.load(STAGE / "rollout_trace.npz", allow_pickle=False)
    measured = trace["actual_q"].astype(np.float64)
    names = trace["joint_names"].astype(str).tolist()
    left_wrist = trace["left_wrist_xyz_world_m"].astype(np.float64)
    right_wrist = trace["right_wrist_xyz_world_m"].astype(np.float64)
    if measured.shape != (436, 28) or len(names) != 28:
        raise RuntimeError("rollout trace shape mismatch")
    motion = {
        "left_wrist_maximum_displacement_from_start_m": float(
            np.max(np.linalg.norm(left_wrist - left_wrist[:1], axis=1))
        ),
        "right_wrist_maximum_displacement_from_start_m": float(
            np.max(np.linalg.norm(right_wrist - right_wrist[:1], axis=1))
        ),
        "left_wrist_net_displacement_m": float(np.linalg.norm(left_wrist[-1] - left_wrist[0])),
        "right_wrist_net_displacement_m": float(
            np.linalg.norm(right_wrist[-1] - right_wrist[0])
        ),
        "left_dex3_maximum_per_joint_range_rad": float(
            np.max(np.ptp(measured[:, 14:21], axis=0))
        ),
        "right_dex3_maximum_per_joint_range_rad": float(
            np.max(np.ptp(measured[:, 21:28], axis=0))
        ),
    }

    videos = {camera: video_info(STAGE / f"{camera}.mp4") for camera in CAMERAS}
    comparisons = {
        camera: video_info(COMPARISON / f"old_vs_g1visual_{camera}.mp4")
        for camera in CAMERAS
    }
    video_checks = {
        "four_rollout_videos": len(videos) == 4,
        "rollout_video_frames_match_trace_plus_initial": all(
            row["frames"] == len(measured) + 1 for row in videos.values()
        ),
        "four_comparison_videos": len(comparisons) == 4,
        "comparison_is_matched_duration_not_frozen_tail": all(
            row["frames"] == len(measured) + 1 for row in comparisons.values()
        ),
        "comparison_invariants_pass": all(comparison["invariants"].values()),
        "comparison_label_explicitly_unapproved": "UNAPPROVED"
        in comparison["labels"]["new"],
    }
    if not all(video_checks.values()):
        raise RuntimeError(f"video audit failed: {video_checks}")

    qualitative = {
        "review_basis": [
            "overview/source_like/top/side videos",
            "2-second timeline contact sheets",
            "measured wrist trajectories",
            "measured Dex3 trajectories",
        ],
        "left_approach": "YES",
        "grasp_like_hand_motion": "NO",
        "left_transport": "NO",
        "right_handoff_approach": "NO",
        "bimanual_posture": "NO",
        "left_release": "NO",
        "right_transport": "NO",
        "right_release": "NO",
        "scope_note": (
            "The rollout was observed only through 14.53 s because the mandatory executed-frame "
            "external-contact gate aborted it. Later motions are reported as not observed before "
            "the abort, not as a complete 22.93-s semantic benchmark."
        ),
        "summary": (
            "R2 closely repeats the original policy's early behavior: the left wrist approaches and "
            "descends into the doll/table-left region, both hands remain visually open with only small "
            "Dex3 changes, and the right wrist remains near its initial pose. The left arm pushes the "
            "doll toward the workspace edge and the run stops on material table/bin contact before "
            "grasp, transport, handoff, or right-side phases appear."
        ),
    }

    command = (
        "/home/jbnu/miniconda3/bin/conda run -n isaaclab6 --no-capture-output "
        "/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p "
        "/home/jbnu/aloha_g1_dataset/tools/run_policy_b_isaac_doll_handoff.py "
        "--policy-variant B --camera-config "
        "/home/jbnu/aloha_g1_dataset/outputs/policy_b_isaac_validation/camera/source_like_cam_high.json "
        "--stage full-motion --output-root "
        "/home/jbnu/aloha_g1_dataset/outputs/policy_b_g1visual/rehearsal/r2_1500_diagnostic_closed_loop "
        "--execution-horizon 4 --full-motion-inference-calls 172 --settle-seconds 1.0 "
        "--seed 20260824 --checkpoint-override "
        "/home/jbnu/aloha_g1_dataset/outputs/policy_b_g1visual/rehearsal/training/"
        "R2_paired_visual_connector_001500/checkpoints/001500/pretrained_model "
        f"--checkpoint-model-sha256 {EXPECTED_MODEL_SHA256} "
        "--checkpoint-training-step 1500 --simulation-diagnostic-rollout --headless"
    )
    atomic_text(REVIEW / "RUN_COMMAND.txt", command + "\n")

    result = {
        "schema_version": "policy_b_r2_1500_unapproved_isaac_diagnostic_v1",
        "status": "DIAGNOSTIC_VIDEO_READY",
        "checkpoint_approval": "FAIL",
        "checkpoint_validation_claimed": False,
        "real_hardware": "BLOCKED",
        "real_robot_invoked": False,
        "rollout": {
            "completion": "ABORTED_BY_EXECUTED_FRAME_HARD_SAFETY_GATE",
            "intended_duration_s": 688 / 30.0,
            "executed_duration_s": float(stage["duration_s"]),
            "executed_frames": int(stage["executed_control_frames"]),
            "policy_inference_calls": int(stage["inference_call_count"]),
            "execution_horizon_frames": int(stage["execution_horizon_frames"]),
            "control_fps": float(stage["control_fps"]),
            "abort": {
                "frame_zero_based": int(abort["frame"]),
                "reason": "EXTERNAL_NON_TASK_COLLISION",
                "force_n": float(abort["external_non_task_contact_force_n"]),
                "all_other_runtime_checks_passed": True,
            },
        },
        "approval_and_environment_checks": approval_checks,
        "runtime_abort_checks": runtime_abort_checks,
        "video_checks": video_checks,
        "motion_diagnostics": motion,
        "qualitative_motion_review": qualitative,
        "videos": videos,
        "side_by_side_videos": comparisons,
        "comparison_manifest": str(COMPARISON / "comparison_manifest.json"),
        "timeline_contact_sheets": {
            camera: str(REVIEW / f"{camera}_timeline.png") for camera in CAMERAS
        }
        | {
            "old_vs_r2_overview": str(REVIEW / "old_vs_r2_overview_timeline.png"),
            "old_vs_r2_source_like": str(REVIEW / "old_vs_r2_source_like_timeline.png"),
        },
        "stage_report": str(STAGE / "stage_report.json"),
        "trace": str(STAGE / "rollout_trace.npz"),
        "run_command": str(REVIEW / "RUN_COMMAND.txt"),
    }
    atomic_json(REVIEW / "diagnostic_review.json", result)

    lines = [
        "# R2/1500 unapproved Isaac diagnostic rollout",
        "",
        "- CHECKPOINT_APPROVAL: **FAIL**",
        "- REAL_HARDWARE: **BLOCKED**",
        "- Diagnostic result: **ABORTED_BY_EXECUTED_FRAME_HARD_SAFETY_GATE**",
        f"- Executed: {stage['duration_s']:.3f} s, {stage['executed_control_frames']} frames, "
        f"{stage['inference_call_count']} inferences, horizon {stage['execution_horizon_frames']}",
        f"- Abort: frame {abort['frame']} measured {abort['external_non_task_contact_force_n']:.6f} N "
        "contact with the table/bin collision set; every other runtime hard check passed.",
        "",
        "The intended 22.933-second duration was not completed. The hard gate was preserved; no "
        "semantic failure caused the stop.",
        "",
        "## Qualitative motion observed before abort",
        "",
        "| Motion | Observation |",
        "|---|---|",
        "| Left approach | YES |",
        "| Grasp-like hand motion | NO |",
        "| Left transport | NO |",
        "| Right handoff approach | NO |",
        "| Bimanual posture | NO |",
        "| Left release | NO |",
        "| Right transport | NO |",
        "| Right release | NO |",
        "",
        qualitative["summary"],
        "",
        "The comparison videos end at the R2 abort instead of freezing its final frame against the "
        "longer original rollout.",
    ]
    atomic_text(REVIEW / "diagnostic_report.md", "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "completion": result["rollout"]["completion"],
                "executed_duration_s": result["rollout"]["executed_duration_s"],
                "checkpoint_approval": result["checkpoint_approval"],
                "real_hardware": result["real_hardware"],
                "report": str(REVIEW / "diagnostic_report.md"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
