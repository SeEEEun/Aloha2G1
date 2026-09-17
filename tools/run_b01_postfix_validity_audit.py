#!/usr/bin/env python3
"""Run the explicitly requested non-final B01 validity rerun after the TYPE-3 fix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import mujoco

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_final_grasp_capture_task_frames import DistanceModel
from tools.common_execution_layer import pose_matrix
from tools.direct_physical_execution_layer import authoritative_joint_limits
from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics


ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
OUT = ROOT / "outputs/final_episode_registered_eval35"
RUN = OUT / "00_forensic_audit/B01_POSTFIX_VALIDITY_RERUN"
FREEZE = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"
REGISTRATION = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
COMMANDS = ROOT / "outputs/final_direct_physical_eval35/00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
LAUNCHER = ROOT / "tools/run_direct_physical_execution_isaac.py"
SCORER = ROOT / "tools/score_episode_registered_physical_eval35_run.py"
REPORT = OUT / "00_forensic_audit/B01_POSTFIX_VALIDITY_REPORT.json"
CONTACT_MODEL = OUT / "01_freeze/FINAL_DOLL_CONTACT_MODEL.json"
COMMON = ROOT / "outputs/final_direct_physical_eval35/00_preparation/runtime_frozen_fair_a/config/common_config.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    records = [row for row in read(COMMANDS)["records"] if row["method"] == "ACT-B40" and int(row["eval_index"]) == 0]
    if len(records) != 1:
        raise RuntimeError("exact B01 prepared command not found")
    row = records[0]
    registration = next(value for value in read(REGISTRATION)["entries"] if value["stable_episode_id"] == row["stable_episode_id"])
    if RUN.exists():
        raise RuntimeError(f"refusing to overwrite preserved B01 audit: {RUN}")
    RUN.mkdir(parents=True)
    command = Path(row["physical_command"])
    invocation = [
        str(ISAAC), str(LAUNCHER), "--direct-freeze-manifest", str(FREEZE),
        "--config", str(CONFIG), "--side", "right", "--geometry", "INTERMEDIATE_PLUSH_PROXY",
        "--profile", "P14", "--output-dir", str(RUN), "--scripted-command-path", str(command),
        "--object-spawn-side", "left", "--episode-registration-manifest", str(REGISTRATION),
        "--episode-stable-id", row["stable_episode_id"], "--audit-robot-bin", "--full-task-audit",
        "--bin-height-m", "0.150", "--bin-rim-bevel-m", "0.003", "--headless",
    ]
    atomic(RUN / "INVOCATION_MANIFEST.json", {
        "schema_version": "b01_postfix_validity_audit_invocation_v1", "method": "ACT-B40", "eval_index": 0,
        "stable_episode_id": row["stable_episode_id"], "provenance": row["provenance"],
        "command": str(command), "command_sha256": row["physical_command_sha256"],
        "freeze_manifest": str(FREEZE), "freeze_manifest_sha256": sha(FREEZE),
        "episode_registration_manifest": str(REGISTRATION), "episode_registration_manifest_sha256": sha(REGISTRATION),
        "episode_registration": {"target_object_pose": registration["target_object_pose"], "entry_sha256": registration["entry_sha256"], "A_B_identical_object_pose": True},
        "geometry": "INTERMEDIATE_PLUSH_PROXY", "contact_tolerance_mm": 0,
        "common_arm_hard_limit_projector": "NEAREST_VALID_VALUE_COMPONENTWISE",
        "arm_rescue_allowed": False, "wrist_rescue_allowed": False, "object_pose_writes_after_initialization": 0,
        "purpose": "explicit B01 validity rerun; excluded from the final 35+35 result and never used for tuning", "invocation": invocation,
    })
    with (RUN / "engine.log").open("w", encoding="utf-8") as stream:
        engine = subprocess.run(invocation, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    if engine.returncode not in (0, 2):
        raise RuntimeError(f"B01 audit engine infrastructure failure: {engine.returncode}")
    scored = subprocess.run([str(ISAAC), str(SCORER), "--run-dir", str(RUN)], cwd=ROOT, capture_output=True, text=True, check=False)
    (RUN / "scorer.log").write_text(scored.stdout + scored.stderr, encoding="utf-8")
    if scored.returncode not in (0, 3):
        raise RuntimeError(f"B01 audit scorer infrastructure failure: {scored.returncode}")
    result = read(RUN / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json")
    trial = read(RUN / "trial_result.json")
    runtime = read(RUN / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")
    with np.load(RUN / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    control = event["control_frame"].astype(np.int64)
    rows = np.r_[np.flatnonzero(np.diff(control) != 0), len(control)-1]
    intent = event["DIRECT_COMMON_TASK_INTENT"][rows].astype(str)
    close = np.isin(intent, ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"])
    contact = read(CONTACT_MODEL)
    visual = np.asarray(contact["visual_dimensions_m"], dtype=np.float64)
    collision = np.asarray(contact["collision_dimensions_m"], dtype=np.float64)
    common = load_common_config(COMMON)
    g1 = G1Kinematics(common, load_scene(common))
    _, _, joint_names = authoritative_joint_limits(read(JOINT_CONTRACT))
    distance_model = DistanceModel(g1, collision, float(read(CONFIG)["object"]["table_surface_world_z_m"]), joint_names)
    z_offset = float((collision[2] - visual[2]) / 2.0)
    distances = {key: float("inf") for key in ("thumb","index","middle","palm")}
    for q, position, quaternion in zip(event["MEASURED_Q"][rows][close], event["object_position_world_m"][rows][close], event["object_quaternion_xyzw"][rows][close], strict=True):
        distance_model.assign(np.asarray(q, dtype=np.float64))
        distance_model.set_object_pose(pose_matrix(position, quaternion), z_offset)
        mujoco.mj_forward(distance_model.model, distance_model.data)
        for key, value in distance_model.distances().items():
            distances[key] = min(distances[key], float(value))
    report = {
        "schema_version": "b01_postfix_validity_report_v1",
        "status": "VALID" if result["status"] != "INVALID" else "INVALID",
        "excluded_from_final_result": True, "used_for_tuning": False,
        "registered_doll_pose": result["integrity"]["runtime_initial_pose_verification"],
        "maximum_digit_contact_force_n": {digit: float(np.max(event[f"left_{digit}_force_n"], initial=0.0)) for digit in ("thumb","index","middle")},
        "maximum_palm_contact_force_n": float(np.max(event["left_palm_force_n"], initial=0.0)),
        "minimum_left_collision_surface_distance_mm": {key: 1000.0*value for key,value in distances.items()},
        "mechanical_grasp_confirmation_frame": result["event_frames"]["grasp_confirm"],
        "table_support_loss_frame": result["event_frames"]["table_support_loss"],
        "maximum_COM_lift_mm": 1000.0 * float(np.max(event["object_position_world_m"][:,2] - event["object_position_world_m"][0,2], initial=0.0)),
        "first_failure_stage": result["first_failure_stage"], "full_task_success": result["outcomes"]["FULL_TASK_SUCCESS"],
        "commanded_hard_limit_violations": result["integrity"]["commanded_hard_limit_violation_scalar_count"],
        "measured_hard_limit_violations": result["integrity"]["measured_hard_limit_violation_scalar_count"],
        "finite_state_failures": result["integrity"]["non_finite_state_count"],
        "object_pose_writes_after_initialization": result["integrity"]["object_pose_writes_after_initialization"],
        "arm_rescue": result["fairness_audit"]["arm_rescue_used"], "wrist_rescue": result["fairness_audit"]["wrist_rescue_used"],
        "runtime_articulation_solver": trial["runtime_articulation_solver"],
        "runtime_final_grasp_state": runtime["final_grasp_state"],
    }
    atomic(REPORT, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "VALID" else 4


if __name__ == "__main__":
    raise SystemExit(main())
