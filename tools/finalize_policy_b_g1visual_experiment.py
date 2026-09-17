#!/usr/bin/env python3
"""Freeze the completed G1-visual dataset/adaptation evidence and protocol stop."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/doll_handoff_proposed_b_g1visual_50"
OUTPUT = ROOT / "outputs/policy_b_g1visual"
FINAL_DIR = ROOT / "outputs/doll_handoff_g1visual_dataset"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sha256(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    package_path = DATASET / "meta/g1visual_packaging_manifest.json"
    render_path = OUTPUT / "dataset_render_full/render_manifest.json"
    camera_path = OUTPUT / "dataset_render_full/camera_family.json"
    video_audit_path = OUTPUT / "dataset_render_full/all_camera_video_audit.json"
    readback_path = OUTPUT / "dataset_render_full/lerobot_readback_validation.json"
    training_path = OUTPUT / "training/training_audit.json"
    phase_root = OUTPUT / "offline_phase_probe"
    phase_decision_path = phase_root / "phase_learning_decision.json"
    source_summary_path = OUTPUT / "teacher_forced_diagnostic/source_rgb_time_forced_summary.json"
    old_review_path = (
        ROOT
        / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout/qualitative_motion_review.json"
    )

    package = load(package_path)
    render = load(render_path)
    cameras = load(camera_path)
    video_audit = load(video_audit_path)
    readback = load(readback_path)
    training = load(training_path)
    phase_decision = load(phase_decision_path)
    source_summary = load(source_summary_path)
    old_review = load(old_review_path)

    if package["label_identity"]["ACTION_EQUAL"] is not True:
        raise RuntimeError("action identity gate failed")
    if package["label_identity"]["STATE_EQUAL"] is not True:
        raise RuntimeError("state identity gate failed")
    if render["status"] != "RENDER_COMPLETE" or render["completed_episode_count"] != 50:
        raise RuntimeError("full render gate failed")
    if render["completed_frame_count"] != 34478:
        raise RuntimeError("full render frame count mismatch")
    if readback["status"] != "PASS" or video_audit["status"] != "PASS":
        raise RuntimeError("dataset/video readback gate failed")
    if training["status"] != "PASS" or not training["loss_all_finite"]:
        raise RuntimeError("training audit failed")

    # The source-clock runs were intentionally subordinate diagnostics. They
    # were all stopped by strict executed-prefix gates before full duration.
    source_review = {
        "classification": "SOURCE_RGB_TIME_FORCED_POLICY_ROLLOUT",
        "SOURCE_RGB_TIME_FORCED_PROGRESS": "UNCLEAR",
        "reason": (
            "All three causal runs were terminated by executed-prefix safety checks at "
            "6.0, 11.2, and 16.133 seconds. Some later arm directions emerged weakly, "
            "but grasp/release Dex3 transitions did not emerge robustly before abort."
        ),
        "not_a_training_blocker": True,
        "physical_task_success_claimed": False,
        "rollouts": [
            {
                "episode_index": row["episode_index"],
                "duration_s": row["duration_s"],
                "executed_frames": row["executed_frames"],
                "inference_calls": row["inference_calls"],
                "status": row["status"],
                "videos": row["videos"],
            }
            for row in source_summary["rollouts"]
        ],
    }
    source_review_path = OUTPUT / "teacher_forced_diagnostic/source_rgb_time_forced_review.json"
    atomic_json(source_review_path, source_review)

    checkpoints = []
    for step in (1000, 2000, 3000, 4000, 5000):
        probe_dir = phase_root if step == 5000 else phase_root / f"checkpoint_{step:06d}"
        decision = load(probe_dir / "phase_learning_decision.json")
        with (probe_dir / "phase_table.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        phase_status = {
            row["phase"]: row["policy_behavior_present"] == "YES" for row in rows
        }
        checkpoints.append(
            {
                "step": step,
                "checkpoint": decision["checkpoint"],
                "model_sha256": decision["model_sha256"],
                "gate_pass": decision["G1_VISUAL_PHASE_LEARNING_CONFIRMED"],
                "phase_pass_count": sum(phase_status.values()),
                "phase_total": len(phase_status),
                "phase_status": phase_status,
                "decision_path": str(probe_dir / "phase_learning_decision.json"),
                "decision_sha256": sha256_file(probe_dir / "phase_learning_decision.json"),
            }
        )
    eligible = [row for row in checkpoints if row["gate_pass"]]
    checkpoint_audit = {
        "status": "NO_ELIGIBLE_CHECKPOINT",
        "selection_rule": "all nine phases must pass established 6/6 per-phase evidence gates",
        "checkpoints": checkpoints,
        "eligible_checkpoints": eligible,
        "closed_loop_allowed": bool(eligible),
        "conclusion": (
            "Every checkpoint retained 6/9 phases and failed initial/left approach, "
            "left transport, and handoff approach."
        ),
    }
    checkpoint_audit_path = phase_root / "checkpoint_selection_audit.json"
    atomic_json(checkpoint_audit_path, checkpoint_audit)

    episode_rows = []
    video_rows = []
    maximum_q_error = 0.0
    maximum_doll_error = 0.0
    for episode in render["episodes"]:
        maximum_q_error = max(
            maximum_q_error, float(episode["maximum_named_joint_qpos_readback_error_rad"])
        )
        maximum_doll_error = max(
            maximum_doll_error, float(episode["maximum_doll_position_readback_error_m"])
        )
        episode_rows.append(
            {
                "episode_index": episode["episode_index"],
                "source_raw_episode": episode["source_raw_episode"],
                "frames": episode["frames"],
                "fps": episode["fps"],
                "timestamp_start_s": episode["timestamp_start_s"],
                "timestamp_end_s": episode["timestamp_end_s"],
            }
        )
        for key, video in sorted(episode["videos"].items()):
            path = Path(video["path"])
            if not path.is_file() or sha256_file(path) != video["sha256"]:
                raise RuntimeError(f"render video changed: {path}")
            video_rows.append(
                {
                    "episode_index": episode["episode_index"],
                    "camera_key": key,
                    "frames": video["frames"],
                    "path": video["path"],
                    "sha256": video["sha256"],
                }
            )
    if len(video_rows) != 200:
        raise RuntimeError(f"expected 200 rendered videos, got {len(video_rows)}")

    camera_status = {}
    for key, row in video_audit["by_camera"].items():
        camera_status[key] = {
            "status": row["status"],
            "video_count": row["video_count"],
            "frame_count": row["frame_count"],
            "aggregate_video_manifest_sha256": aggregate_sha256(
                [video for video in video_rows if video["camera_key"] == key]
            ),
        }

    old_qualitative = old_review["qualitative_motion_summary"]
    adapted_qualitative = {
        key: "NOT_RUN_PHASE_RETENTION_GATE_FAILED" for key in old_qualitative
    }

    identity = package["label_identity"]
    final_manifest = {
        "schema_version": "final_g1_visual_dataset_manifest_v1",
        "status": "COMPLETE_THROUGH_OFFLINE_PHASE_GATE",
        "final_result": "G1_VISUAL_ADAPTATION_INCONCLUSIVE",
        "target_embodiment_visual_relabeling": {
            "source_dataset": package["source_dataset"],
            "source_dataset_tree_sha256": identity["source_dataset_tree_sha256"],
            "new_dataset": package["destination_dataset"],
            "new_dataset_content_tree_sha256": package[
                "dataset_content_tree_sha256_excluding_this_manifest"
            ],
            "episodes": package["episodes"],
            "frames": package["frames"],
            "fps": package["fps"],
            "state_dimension": package["state_dimension"],
            "action_dimension": package["action_dimension"],
            "state_label": "RETARGETED_G1_STATE_SURROGATE",
            "rendered_robot_q": "observation.state[t]",
            "state_arrays_unchanged": identity["STATE_EQUAL"],
            "action_arrays_unchanged": identity["ACTION_EQUAL"],
            "original_state_logical_sha256": identity["original_state_logical_sha256"],
            "new_state_logical_sha256": identity["new_state_logical_sha256"],
            "original_action_logical_sha256": identity["original_action_logical_sha256"],
            "new_action_logical_sha256": identity["new_action_logical_sha256"],
            "task_metadata_semantically_identical": identity["tasks_parquet_byte_equal"],
            "frame_counts_identical": identity["frame_counts_identical"],
            "episode_mapping_identical": identity["episode_mapping_identical"],
            "object_visualization_method": package["object_visualization_method"],
            "maximum_qpos_readback_error_rad": maximum_q_error,
            "maximum_doll_position_readback_error_m": maximum_doll_error,
            "lerobot_readback_status": readback["status"],
            "lerobot_readback_validation": str(readback_path),
            "lerobot_readback_validation_sha256": sha256_file(readback_path),
        },
        "camera_family": cameras,
        "camera_render_status": camera_status,
        "all_camera_video_audit": str(video_audit_path),
        "all_camera_video_audit_sha256": sha256_file(video_audit_path),
        "episodes": episode_rows,
        "generated_videos": video_rows,
        "generated_video_manifest_sha256": aggregate_sha256(video_rows),
        "rendering": {
            "render_manifest": str(render_path),
            "render_manifest_sha256": sha256_file(render_path),
            "rendering_code": str(ROOT / "tools/render_doll_handoff_g1visual_dataset.py"),
            "rendering_code_sha256": sha256_file(
                ROOT / "tools/render_doll_handoff_g1visual_dataset.py"
            ),
            "packaging_code_sha256": sha256_file(
                ROOT / "tools/package_doll_handoff_g1visual_lerobot.py"
            ),
            "accepted_preview_qa": str(
                OUTPUT / "dataset_render_preview_v7/qa/preview_visual_acceptance.md"
            ),
        },
        "source_rgb_time_forced_diagnostic": {
            **source_review,
            "review_path": str(source_review_path),
            "review_sha256": sha256_file(source_review_path),
        },
        "training": {
            "experiment": "POLICY_B_G1VISUAL",
            "start_checkpoint": training["start_checkpoint"],
            "start_model_sha256": training["start_model_sha256"],
            "adaptation_steps": training["adaptation_steps"],
            "final_logged_loss": training["final_logged_loss"],
            "selected_training_checkpoint": training["selected_checkpoint"],
            "training_log_sha256": training["training_log_sha256"],
            "all_checkpoint_weights_finite": all(
                row["checks"]["weights_finite"] for row in training["available_checkpoints"]
            ),
        },
        "offline_phase_probe": {
            "G1_VISUAL_PHASE_LEARNING_CONFIRMED": False,
            "final_checkpoint_decision": str(phase_decision_path),
            "final_checkpoint_decision_sha256": sha256_file(phase_decision_path),
            "checkpoint_selection_audit": str(checkpoint_audit_path),
            "checkpoint_selection_audit_sha256": sha256_file(checkpoint_audit_path),
            "checkpoint_results": checkpoints,
            "eligible_checkpoint": None,
        },
        "closed_loop_rollout": {
            "status": "NOT_RUN_BY_PHASE_RETENTION_GATE",
            "reason": (
                "No saved adaptation checkpoint passed all nine established offline phase gates."
            ),
            "old_policy_b_qualitative": old_qualitative,
            "policy_b_g1visual_qualitative": adapted_qualitative,
            "comparison_video": None,
        },
        "visual_embodiment_gap_hypothesis": "INCONCLUSIVE",
        "final_deployment_camera": "NOT_YET_DECIDED",
        "real_g1": "NOT_STARTED_BY_DESIGN",
        "real_robot_invoked": False,
        "baseline_a_invoked": False,
        "next_step": "B. improve G1-visual adaptation",
    }

    manifest_path = FINAL_DIR / "FINAL_G1_VISUAL_DATASET_MANIFEST.json"
    atomic_json(manifest_path, final_manifest)
    manifest_hash = sha256_file(manifest_path)
    atomic_text(FINAL_DIR / "FINAL_G1_VISUAL_DATASET_MANIFEST.sha256", f"{manifest_hash}\n")

    phase_rows = checkpoints[-1]["phase_status"]
    report = f"""# Policy B G1-Visual adaptation report

## TARGET EMBODIMENT VISUAL RELABELING

Dataset: `{package['destination_dataset']}`  
Episodes: {package['episodes']}  
Frames: {package['frames']}  
State arrays unchanged: YES  
Action arrays unchanged: YES  
Object visualization method: `{package['object_visualization_method']}`  
Dataset tree SHA: `{package['dataset_content_tree_sha256_excluding_this_manifest']}`

The primary training image is the frozen G1-rendered `SOURCE_LIKE_CAM_HIGH`. The original 28D state/action arrays, task metadata, timestamps, episode mapping, and frame counts are unchanged. `LeRobotDataset` readback and all 50 primary video decodes passed.

## CAMERAS RENDERED

SOURCE_LIKE_CAM_HIGH: PASS  
FOREHEAD_PROVISIONAL: PASS — `PROVISIONAL_NOT_PHYSICALLY_CALIBRATED`  
HEAD_PROVISIONAL: PASS — `PROVISIONAL_NOT_PHYSICALLY_CALIBRATED`  
NECK_PROVISIONAL: PASS — `PROVISIONAL_NOT_PHYSICALLY_CALIBRATED`

Each view contains 50 videos and 34,478 synchronized frames at 640×480 and 30 Hz. Only `observation.images.cam_high` was packaged/consumed for this adaptation; provisional views remain in the render archive.

## SOURCE-RGB TIME-FORCED DIAGNOSTIC

Later phases emerged: UNCLEAR

All three diagnostic runs were stopped by strict executed-prefix safety checks at 6.0, 11.2, and 16.133 seconds. Some later arm-direction alignment appeared, but the Dex3 grasp/release transitions were not robust before abort. This diagnostic does not claim closed-loop vision or physical task success and was not a training blocker.

## POLICY_B_G1VISUAL TRAINING

Start checkpoint: `{training['start_checkpoint']}`  
Adaptation steps: {training['adaptation_steps']}  
Final loss: {training['final_logged_loss']:.3f}  
Checkpoint: `{training['selected_checkpoint']['checkpoint']}`  
Model SHA: `{training['selected_checkpoint']['model_sha256']}`

All five scheduled checkpoints (1,000–5,000) are complete and finite. Architecture, 28D logical state/action schema, 32D internal padding, chunk length 50, normalization, task instruction, and image preprocessing were unchanged.

## G1-VISUAL OFFLINE PHASE PROBE

Initial: {'YES' if phase_rows['initial / left approach'] else 'NO'}  
Left grasp: {'YES' if phase_rows['left grasp / LEFT_OWNED'] else 'NO'}  
Left transport: {'YES' if phase_rows['left transport'] else 'NO'}  
Handoff approach: {'YES' if phase_rows['handoff approach'] else 'NO'}  
Dual contact: {'YES' if phase_rows['dual-contact / ownership transfer'] else 'NO'}  
Right owned: {'YES' if phase_rows['RIGHT_OWNED'] else 'NO'}  
Right transport: {'YES' if phase_rows['right transport toward bin'] else 'NO'}  
Release: {'YES' if phase_rows['release'] else 'NO'}

G1_VISUAL_PHASE_LEARNING_CONFIRMED: NO

The final checkpoint passed 6/9 phases. Initial/left approach passed 4/6 episodes, left transport 5/6, and handoff approach only 1/6 under the established cosine/progress gates. Every intermediate checkpoint produced the same 6/9 phase-level pattern, so no saved checkpoint was eligible for closed-loop execution.

## OLD POLICY B CLOSED-LOOP

LEFT_APPROACH_LIKE: YES  
LEFT_GRASP_MOTION_LIKE: NO  
LEFT_TRANSPORT_LIKE: NO  
RIGHT_HANDOFF_APPROACH_LIKE: NO  
BIMANUAL_HANDOFF_POSTURE_LIKE: NO  
LEFT_RELEASE_LIKE: NO  
RIGHT_TRANSPORT_TO_BIN_LIKE: NO  
RIGHT_RELEASE_LIKE: NO

## POLICY_B_G1VISUAL CLOSED-LOOP

LEFT_APPROACH_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED  
LEFT_GRASP_MOTION_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED  
LEFT_TRANSPORT_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED  
RIGHT_HANDOFF_APPROACH_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED  
BIMANUAL_HANDOFF_POSTURE_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED  
LEFT_RELEASE_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED  
RIGHT_TRANSPORT_TO_BIN_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED  
RIGHT_RELEASE_LIKE: NOT_RUN_PHASE_RETENTION_GATE_FAILED

The adapted closed-loop rollout and OLD-vs-G1VISUAL comparison video were not created because the mandatory offline phase-retention gate failed.

## VISUAL EMBODIMENT GAP HYPOTHESIS

INCONCLUSIVE

The visual relabeling dataset is valid and several later phases are retained, but the bounded adaptation did not preserve all required phase-conditioned behavior. Without a gate-qualified adapted closed-loop rollout, the visual-gap hypothesis is not validated or refuted.

## FINAL DEPLOYMENT CAMERA

NOT_YET_DECIDED

The external source-like camera remains the authoritative view for this hypothesis test only. Forehead/head/neck views are preserved as provisional, uncalibrated candidates.

## REAL G1

NOT_STARTED_BY_DESIGN

Strict simulation safety qualification remains unchanged, and real-hardware safety readiness remains blocked. No DDS, hardware mode, G1, or Dex3 command path was invoked.

## NEXT STEP

B. improve G1-visual adaptation. The immediate issue is phase retention—especially the right-arm handoff approach—rather than dataset integrity. Preserve this dataset and evaluate a controlled adaptation method that reduces forgetting before any new closed-loop or hardware test.

G1_VISUAL_ADAPTATION_INCONCLUSIVE
"""
    report_path = OUTPUT / "final_report.md"
    atomic_text(report_path, report)
    print(
        json.dumps(
            {
                "status": "G1_VISUAL_ADAPTATION_INCONCLUSIVE",
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "report": str(report_path),
                "closed_loop": "NOT_RUN_BY_PHASE_RETENTION_GATE",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
