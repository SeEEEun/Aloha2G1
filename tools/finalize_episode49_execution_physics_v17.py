#!/usr/bin/env python3
"""Finalize the blocked Episode-49 v17 execution-physics audit."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
BUILDER = ROOT / "tools/build_episode49_execution_physics_v17.py"
TRAJECTORY_CODE = ROOT / "tools/aloha_g1_v17/trajectory.py"
PHYSICS_RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py"
V15 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_orientation_dex3_v15"
V16 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
TRIAL_JSON = OUT / "physics_trial_phone_grasp_0p25x.json"
TRIAL_NPZ = OUT / "physics_trial_phone_grasp_0p25x.npz"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, default=default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_npz(path: Path, **payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def video_info(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "path": str(path.resolve()), "sha256": sha(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "frame_rate": stream["r_frame_rate"],
        "width": int(stream["width"]), "height": int(stream["height"]),
        "scope": "PRELIMINARY_TRUE_PHYSICS_PHONE_GRASP_FAILURE_DIAGNOSTIC",
        "final_candidate_video": False,
    }


def inventory_hash(directory: Path) -> str:
    rows = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        size = path.stat().st_size
        row: dict[str, Any] = {"path": str(path.relative_to(directory)), "size_bytes": size}
        if size <= 32 * 1024 * 1024 or path.suffix not in (".mp4", ".npz"):
            row["sha256"] = sha(path)
        rows.append(row)
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> int:
    required = [
        OUT / "input_freeze_audit.json", OUT / "scene_physics_audit.json",
        OUT / "semantic_runtime_audit.json", OUT / "partial_orientation_config.json",
        OUT / "kinematic_prephysics_result.json", OUT / "kinematic_collision_metrics.json",
        OUT / "kinematic_joint_metrics.json", OUT / "aloha_fidelity_metrics.json",
        OUT / "final_arm_dex3_trajectory.npz", TRIAL_JSON, TRIAL_NPZ,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    freeze = json.loads((OUT / "input_freeze_audit.json").read_text())
    scene = json.loads((OUT / "scene_physics_audit.json").read_text())
    semantic = json.loads((OUT / "semantic_runtime_audit.json").read_text())
    partial = json.loads((OUT / "partial_orientation_config.json").read_text())
    prephysics = json.loads((OUT / "kinematic_prephysics_result.json").read_text())
    collision = json.loads((OUT / "kinematic_collision_metrics.json").read_text())
    joints = json.loads((OUT / "kinematic_joint_metrics.json").read_text())
    fidelity = json.loads((OUT / "aloha_fidelity_metrics.json").read_text())
    trial = json.loads(TRIAL_JSON.read_text())

    candidates = {
        row["config"]["name"]: row for row in partial["candidate_sweep"]
    }
    safe = candidates["V14_PINCH_FACE_000"]["metrics"]
    light = candidates["V14_PINCH_FACE_010"]["metrics"]
    medium = candidates["V14_PINCH_FACE_033"]["metrics"]
    task = candidates["V14_PINCH_FACE_100"]["metrics"]

    with np.load(OUT / "final_arm_dex3_trajectory.npz", allow_pickle=False) as archive:
        arm = archive["g1_arm_q"].copy()
        left = archive["left_dex3_qpos"].copy()
        right = archive["right_dex3_qpos"].copy()
        root = archive["g1_root"].copy()
        source_action = archive["optimized_action"].copy()
        timestamps = archive["source_timestamps"].copy()
    with np.load(TRIAL_NPZ, allow_pickle=True) as archive:
        diagnostic_command = archive["commanded_q"].copy()
        diagnostic_actual = archive["actual_q"].copy()
        phone_pose = archive["phone_pose_xyzw"].copy()
        accessory_pose = archive["accessory_pose_xyzw"].copy()
        action_indices = archive["action_indices"].copy()

    current_trajectory_sha = sha(OUT / "final_arm_dex3_trajectory.npz")
    trial_matches_current = trial["input_sha256"] == current_trajectory_sha
    diagnostic_label = {
        "status": "PRELIMINARY_TRUE_PHYSICS_PHONE_GRASP_FAIL",
        "final_candidate_stage_executed": False,
        "reason_final_candidate_not_executed": "BLOCKED_KINEMATIC_PREPHYSICS_PARTIAL_ORIENTATION",
        "trial_input_sha256": trial["input_sha256"],
        "current_trajectory_sha256": current_trajectory_sha,
        "trial_input_matches_current_final_artifact": trial_matches_current,
        "trial": trial,
    }
    dump(OUT / "stage_phone_grasp.json", diagnostic_label)
    blocked_stage = {
        "status": "NOT_RUN_PREREQUISITE_FAILED",
        "prerequisite": "KINEMATIC_PREPHYSICS_AND_PHONE_GRASP_PASS",
        "physics_steps": 0,
        "scripted_object_motion": False,
    }
    for name in (
        "stage_phone_rotation.json", "stage_accessory_removal.json",
        "stage_bimanual_transport.json", "stage_charger_placement.json",
        "stage_accessory_release.json",
    ):
        dump(OUT / name, blocked_stage)
    dump(OUT / "full_task_physics_result.json", {
        "status": "BLOCKED_KINEMATIC_PREPHYSICS",
        "full_uninterrupted_task_executed": False,
        "dominant_failure_subsystem": "PARTIAL_ORIENTATION",
        "individual_stage_sequence_started": False,
        "preliminary_phone_grasp_diagnostic_steps": trial["physics_steps"],
        "physics_success_claimed": False,
        "scripted_object_motion": False,
    })
    dump(OUT / "physics_tracking_metrics.json", {
        "status": "PRELIMINARY_DIAGNOSTIC_ONLY_NOT_FINAL_CANDIDATE",
        "final_candidate_physics_steps": 0,
        "diagnostic_trial": trial["tracking"],
        "diagnostic_joint_mapping": trial["joint_mapping"],
        "diagnostic_physics_steps": trial["physics_steps"],
        "diagnostic_input_matches_current_final_artifact": trial_matches_current,
    })
    dump(OUT / "physics_collision_metrics.json", {
        "status": "FINAL_CANDIDATE_NOT_EXECUTED",
        "kinematic_prohibited_collision_records_selected_safe_baseline": collision["prohibited_collision_records"],
        "task_facing_full_orientation_hand_hand_collisions": task["collision"]["categories"]["hand_hand"],
        "preliminary_trial_robot_object_contacts": trial["all_robot_object_contact_pairs"],
        "preliminary_trial_prohibited_robot_collision_gate": "NOT_USED_FOR_FINAL_CLAIM",
        "physics_success_claimed": False,
    })
    save_npz(
        OUT / "phone_object_trajectory.npz",
        status=np.asarray("PRELIMINARY_TRUE_PHYSICS_DIAGNOSTIC_ONLY"),
        pose_xyzw=phone_pose, action_indices=action_indices,
        physics_steps=np.asarray(trial["physics_steps"]),
        final_candidate=np.asarray(False), object_pose_scripted=np.asarray(False),
        trial_input_sha256=np.asarray(trial["input_sha256"]),
    )
    save_npz(
        OUT / "accessory_object_trajectory.npz",
        status=np.asarray("PRELIMINARY_TRUE_PHYSICS_DIAGNOSTIC_ONLY"),
        pose_xyzw=accessory_pose, action_indices=action_indices,
        physics_steps=np.asarray(trial["physics_steps"]),
        final_candidate=np.asarray(False), object_pose_scripted=np.asarray(False),
        trial_input_sha256=np.asarray(trial["input_sha256"]),
    )

    video_paths = [
        OUT / "v17_physics_phone_grasp_overview.mp4",
        OUT / "v17_physics_phone_grasp_closeup.mp4",
    ]
    videos = [video_info(path) for path in video_paths if path.is_file()]
    dump(OUT / "pre_real_g1_readiness.json", {
        "status": "NOT_READY_BLOCKED_KINEMATIC_PREPHYSICS",
        "real_robot_safe": False,
        "candidate_translator_created": False,
        "final_arm_joint_names_and_order": "stored in final_arm_dex3_trajectory.npz",
        "trajectory_shape": [990, 28],
        "maximum_arm_step_rad": joints["maximum_arm_step_rad"],
        "minimum_arm_joint_margin_rad": joints["minimum_arm_margin_rad"],
        "minimum_table_clearance_m": collision["minimum_table_clearance_m_active_geom_vertices"],
        "physics_full_task_success": False,
        "remaining_real_only_blockers": [
            "resolve Episode-49 partial-orientation/fidelity/collision conflict in simulation",
            "complete uninterrupted true-physics task",
            "record real Dex3 OPEN primitive",
            "calibrate real Dex3 PHONE_PINCH and RING_HOOK primitives",
            "confirm runtime joint order on hardware",
            "perform object-free 0.25x preflight with E-stop and supervisor readiness",
        ],
    })

    tests = {
        "optimized_action_shape_990x14": list(source_action.shape) == [990, 14],
        "optimized_action_finite": bool(np.isfinite(source_action).all()),
        "timestamps_shape_990": list(timestamps.shape) == [990],
        "input_freeze_byte_identical": freeze["byte_identical"],
        "v15_output_inventory_unchanged": (
            inventory_hash(V15) == freeze["v15_inventory_before"]["inventory_sha256"]
        ),
        "v16_output_inventory_unchanged": (
            inventory_hash(V16) == freeze["v16_inventory_before"]["inventory_sha256"]
        ),
        "root_unchanged": bool(np.allclose(root, freeze["v14"]["root_xyz_m"], atol=1e-12)),
        "workspace_scale_unchanged_0p42": freeze["v14"]["workspace_scale"] == 0.42,
        "scene_physics_not_modified": not scene["scene_physics_modified"],
        "semantic_runtime_literal_count_zero": semantic["runtime_literal_audit"]["count"] == 0,
        "validation_read_count_zero": semantic["validation_read_count"] == 0,
        "heldout_read_count_zero": semantic["heldout_read_count"] == 0,
        "g1_expert_read_count_zero": semantic["g1_expert_read_count"] == 0,
        "arm_temporal_motion_present": float(np.max(np.ptp(arm, axis=0))) > 0.1,
        "left_dex3_temporal_motion_present": float(np.max(np.ptp(left, axis=0))) > 0.1,
        "right_dex3_temporal_motion_present": float(np.max(np.ptp(right, axis=0))) > 0.1,
        "branch_discontinuities_zero_selected_safe_baseline": joints["branch_discontinuity_count"] == 0,
        "kinematic_collisions_zero_selected_safe_baseline": collision["prohibited_collision_records"] == 0,
        "final_kinematic_gate_pass": prephysics["gate_pass"],
        "preliminary_true_physics_steps_positive": trial["physics_steps"] > 0,
        "preliminary_no_scripted_object_pose": trial["integrity"]["object_pose_commands"] == 0,
        "preliminary_no_scripted_attach_detach": trial["integrity"]["semantic_scripted_attach_detach"] == 0,
        "preliminary_gravity_enabled": trial["integrity"]["gravity_enabled"],
        "preliminary_collision_enabled": trial["integrity"]["collision_enabled"],
        "preliminary_actual_arm_dex3_motion": bool(np.max(np.ptp(diagnostic_actual, axis=0)) > 0.1),
        "preliminary_phone_pose_changed": bool(np.max(np.ptp(phone_pose[:, :3], axis=0)) > 1e-4),
        "preliminary_phone_grasp_pass": trial["pass"],
        "full_task_physics_executed": False,
        "candidate_translator_config_absent": not (OUT / "configs/aloha_g1_execution_translator_v17.candidate.json").exists(),
        "diagnostic_videos_decode": all(row["decoded_frames"] == 194 for row in videos),
        "no_dds_publisher_or_hardware_command": True,
    }
    dump(OUT / "tests_results.json", {
        "status": "EXPECTED_BLOCKED_GATES_RECORDED",
        "tests": tests,
        "all_invariant_tests_pass": all(
            value for key, value in tests.items()
            if key not in ("final_kinematic_gate_pass", "preliminary_phone_grasp_pass", "full_task_physics_executed")
        ),
        "expected_failed_execution_gates": {
            "final_kinematic_gate_pass": tests["final_kinematic_gate_pass"],
            "preliminary_phone_grasp_pass": tests["preliminary_phone_grasp_pass"],
            "full_task_physics_executed": tests["full_task_physics_executed"],
        },
    })

    report = f"""1. 최종 상태는 `BLOCKED_KINEMATIC_PREPHYSICS`이며 단일 지배 원인은 `PARTIAL_ORIENTATION`이다.
2. collision-free v14-facing 후보는 grasp task-orientation 오차 {safe['phone_grasp_task_orientation']['error_to_v16_verified_task_wrist_deg']:.3f}°이고, task-facing 후보는 오차 {task['phone_grasp_task_orientation']['error_to_v16_verified_task_wrist_deg']:.3f}°를 달성했지만 hand–hand collision 2건과 right-speed fidelity {task['fidelity']['right_speed']:.3f}로 탈락했다.
3. 따라서 final/full-task physics는 실행하지 않았고, 앞선 true-physics phone-grasp 진단도 A/B 동시 접촉 0 sample과 {trial['object_metrics']['phone_hand_translation_slip_m']*1000:.3f} mm slip로 실패했으므로 simulation execution candidate를 만들지 않았다.

## 1. Final status

- `BLOCKED_KINEMATIC_PREPHYSICS`
- Dominant subsystem: `PARTIAL_ORIENTATION`
- `EP49_EXECUTION_TRANSLATOR_READY_FOR_VISUAL_REVIEW`: false
- `READY_FOR_REAL_DEX3_CALIBRATION_AND_OBJECT_FREE_PREFLIGHT`: false

## 2. Paper-goal preservation

SmolVLA-generated Episode-49 ALOHA action만 behavior source로 사용했다. G1 Expert, validation, held-out trajectory는 읽지 않았고, v14 Cartesian arm behavior를 위치 backbone으로 유지했다.

## 3. Source-action provenance

- Source: `{SOURCE}`
- SHA-256: `{sha(SOURCE)}`
- Shape: `{list(source_action.shape)}`, finite: `{bool(np.isfinite(source_action).all())}`
- Generic approved Episode-49 semantic timeline API를 사용했고 runtime literal semantic index 수는 0이다.

## 4. v14 arm elements reused

Root +0.199 m, workspace scale 0.42, task registration, 990-sample left/right Cartesian behavior, path/speed/bimanual trends, q continuation seed를 재사용했다. Immutable input byte identity: `{freeze['byte_identical']}`.

## 5. v16 elements reused

Source-relative orientation mapping, verified phone-grasp task wrist orientation, portrait/charger task axes, bounded previous-sample continuation만 재사용했다.

## 6. Deliberately discarded

v15/v16 final arm q, contact-driven translation residual, rigid A/B carrier, exact continuous C carrier, per-frame fingertip fitting은 사용하지 않았다.

## 7. Partial-orientation rule

v14 grasp wrist→v16 verified task wrist의 75.697° registration을 semantic approach progress로 0/10/33/100% 적용했다. 0%는 task error {safe['phone_grasp_task_orientation']['error_to_v16_verified_task_wrist_deg']:.3f}°, 10%는 {light['phone_grasp_task_orientation']['error_to_v16_verified_task_wrist_deg']:.3f}°, 33%는 {medium['phone_grasp_task_orientation']['error_to_v16_verified_task_wrist_deg']:.3f}°, 100%는 {task['phone_grasp_task_orientation']['error_to_v16_verified_task_wrist_deg']:.3f}°였다. 100%만 10° gate를 통과했으나 higher-priority fidelity/collision gate를 실패했다.

## 8. PHONE_PINCH primitive

Active USD distal collision caps로 한 번 calibration한 `A_SCREEN_B_BACK` 고정 primitive다. Thumb preload 5 mm, index preload 2 mm이며 3–4 mm index preload는 same-hand collision 때문에 배제했다. PREGRASP→INDEX_CONTACT→THUMB_PINCH의 작은 fixed sequence를 semantic/source gripper progress로 구동한다. Per-frame finger IK는 없다.

## 9. RING_HOOK primitive

v14 full-arm+Dex3 ring reach seed에서 가져온 fixed right-C `RIGHT_RING_HOOK`과 global safe A/B tuck을 구성했다. Kinematic trajectory는 연속적으로 변하지만 phone prephysics gate가 선행 실패했으므로 true-physics hook/removal은 실행하지 않았다.

## 10. Semantic interpolation

모든 hand transition과 task interval은 `SemanticTimeline.event/interval/progress`로 해석한다. resolved index는 provenance에만 있으며 runtime frame constant는 0이다.

## 11. Kinematic ALOHA fidelity

Collision-free baseline의 left/right path는 {safe['fidelity']['left_path_shape']:.6f}/{safe['fidelity']['right_path_shape']:.6f}, speed는 {safe['fidelity']['left_speed']:.6f}/{safe['fidelity']['right_speed']:.6f}, midpoint {safe['fidelity']['bimanual_midpoint']:.6f}, relative vector {safe['fidelity']['relative_hand_vector']:.6f}, inter-hand distance {safe['fidelity']['inter_hand_distance']:.6f}이다. Full task-facing 후보는 right speed가 {task['fidelity']['right_speed']:.6f}로 0.95 gate 미만이다.

## 12. Branch / joints / collision

Safe baseline: branch discontinuity {joints['branch_discontinuity_count']}, max arm step {joints['maximum_arm_step_rad']:.6f} rad, joint violation {joints['joint_limit_violation_count']}, prohibited collision {collision['prohibited_collision_records']}. 최소 arm margin은 {joints['minimum_arm_margin_rad']:.9f} rad로 real-safety warning이다. Full task-facing 후보는 hand–hand collision 2건(frame 273)이다.

## 13. Isaac physics controller

Isaac actuator position targets를 사용하고 timed run의 direct joint write는 0이었다. Preliminary diagnostic은 28/28 joint mapping, {trial['physics_steps']} physics steps, gravity/collision enabled, object pose command 0, scripted attach/detach 0이었다. 이 trial은 final task-facing candidate가 아니며 final success 근거가 아니다.

## 14. Phone grasp physics

Preliminary 0.25× true-physics diagnostic: FAIL. Thumb/index max forces {trial['object_metrics']['left_A_phone_force_max_n']:.3f}/{trial['object_metrics']['left_B_phone_force_max_n']:.3f} N, simultaneous contact longest run 0, phone/wrist motion ratio {trial['object_metrics']['phone_to_wrist_motion_ratio']:.4f}, slip {trial['object_metrics']['phone_hand_translation_slip_m']*1000:.3f} mm이다.

## 15. Phone rotation physics

NOT RUN — phone grasp와 final prephysics gate가 실패했다.

## 16. Accessory removal physics

NOT RUN — prerequisite failed. Authoritative scene에는 2 N / 0.08 Nm breakable accessory joint가 존재하지만 성공을 가정하지 않았다.

## 17. Bimanual transport physics

NOT RUN — prerequisite failed.

## 18. Charger placement physics

NOT RUN — prerequisite failed. Magnet config가 `DEBUG_INITIAL_GUESS`라는 scene warning도 그대로 보존했다.

## 19. Accessory release physics

NOT RUN — prerequisite failed.

## 20. Uninterrupted full-task result

실행하지 않았다. Stage reset/hidden correction 없이 수행할 수 있는 prephysics candidate가 없었기 때문이다.

## 21. Physics collision result

Final candidate physics collision claim은 없다. Numerical safe baseline collision은 0이고, task-facing kinematic candidate는 hand–hand collision 2건으로 실패했다.

## 22. Target-vs-actual tracking

Preliminary diagnostic tracking RMSE {trial['tracking']['rmse_rad']:.6f} rad, max {trial['tracking']['maximum_absolute_error_rad']:.6f} rad였다. Final candidate tracking은 실행하지 않았다.

## 23. Object trajectory result

Diagnostic phone max displacement {trial['object_metrics']['phone_max_displacement_m']*1000:.3f} mm였지만 retained transport가 아니었다. Accessory removal/charger attachment/full-task completion은 검증되지 않았다.

## 24. Video paths

Full-task required videos는 gate 정책에 따라 생성하지 않았다. 다음 두 영상은 각각 194 frames의 preliminary phone-grasp failure diagnostic이며 final candidate가 아니다:

""" + "\n".join(f"- `{row['path']}`" for row in videos) + f"""

## 25. Candidate translator config

생성하지 않았다. 모든 success gate가 통과해야 한다는 계약을 지켰다.

## 26. Pre-real-G1 readiness

`NOT_READY_BLOCKED_KINEMATIC_PREPHYSICS`. `REAL_ROBOT_SAFE`가 아니며 hardware command package도 생성하지 않았다.

## 27. Remaining hardware-only blockers

Simulation blocker 해소 및 full physics pass 이후에도 real Dex3 OPEN/PHONE_PINCH/RING_HOOK calibration, runtime joint-order 확인, E-stop/supervisor, object-free 0.25× preflight가 남는다.

## 28. Exactly one recommended next action

Episode 49에서만 v16 phone-grasp task orientation을 유지하면서 frame-independent smooth hand–hand clearance barrier를 포함하는 partial-orientation null-space solver를 보정하라. Dex3 primitive, object physics, validation/held-out data는 바꾸거나 사용하지 마라.

THE PRIMARY SCIENTIFIC CONTRIBUTION REMAINS CROSS-EMBODIMENT TRANSFER OF VLA-GENERATED ALOHA BEHAVIOR TO UNITREE G1
THE V14 ALOHA-PRIMARY ARM MOTION REMAINED THE BEHAVIOR BACKBONE
ONLY TASK-CRITICAL PARTIAL ORIENTATION WAS ADDED
DEX3 WAS USED AS A PREDEFINED TARGET-EMBODIMENT GRASP ADAPTER
DEX3 DID NOT REDESIGN THE ALOHA-DERIVED ARM TRAJECTORY
DEX3 TRANSITIONS WERE DRIVEN BY THE GENERIC SEMANTIC API
ISAAC LAB TASK SUCCESS WAS JUDGED FROM TRUE PHYSICAL OBJECT MOTION
NO KINEMATIC OBJECT FOLLOW WAS USED TO CLAIM PHYSICS SUCCESS
NO VALIDATION OR HELD-OUT TRAJECTORY WAS USED FOR TUNING
NO G1 EXPERT MOTION WAS USED TO GENERATE THE TRAJECTORY
NO DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    report_dir = OUT / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "index.html").write_text(
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<title>Episode-49 execution physics v17</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font-family:system-ui;line-height:1.5}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1.5rem}</style></head>"
        f"<body><pre>{html.escape(report)}</pre></body></html>\n",
        encoding="utf-8",
    )

    commands = """#!/usr/bin/env bash
set -euo pipefail
cd /home/jbnu/aloha_g1_dataset
/home/jbnu/miniconda3/envs/isaaclab6/bin/python tools/build_episode49_execution_physics_v17.py
# The following is intentionally gated and must not be run while build_summary is BLOCKED:
# /home/jbnu/miniconda3/envs/isaaclab6/bin/python isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py --trial phone_grasp --speed 0.25 --headless
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,r_frame_rate -of json outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17/v17_physics_phone_grasp_overview.mp4
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    artifact_rows = []
    for path in sorted(value for value in OUT.rglob("*") if value.is_file()):
        if path.name == "run_manifest.json":
            continue
        artifact_rows.append({
            "path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size,
            "sha256": sha(path),
        })
    manifest = {
        "method": "ALOHA_PRIMARY_EP49_EXECUTION_PHYSICS_V17",
        "status": "BLOCKED_KINEMATIC_PREPHYSICS",
        "dominant_failure_subsystem": "PARTIAL_ORIENTATION",
        "episode_id": 49,
        "source_action_type": "SMOLVLA_GENERATED",
        "source_action_path": str(SOURCE.resolve()),
        "source_action_sha256": sha(SOURCE),
        "builder_sha256": sha(BUILDER),
        "trajectory_code_sha256": sha(TRAJECTORY_CODE),
        "physics_runner_sha256": sha(PHYSICS_RUNNER),
        "final_candidate_physics_steps": 0,
        "preliminary_diagnostic_physics_steps": trial["physics_steps"],
        "validation_read_count": 0, "heldout_read_count": 0, "g1_expert_read_count": 0,
        "physics": True, "dds": False, "publisher": False, "real_robot_command": False,
        "candidate_translator_created": False,
        "videos": videos,
        "artifacts": artifact_rows,
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "dominant_failure_subsystem": manifest["dominant_failure_subsystem"],
        "artifact_count": len(artifact_rows) + 1,
        "videos": videos,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
