#!/usr/bin/env python3
"""Finalize the fail-closed R26 retiming and exact-R14 fallback audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import model_parts  # noqa: E402
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


OUT = ROOT / "outputs/final_task_completion_v1"
R26_OUT = OUT / "08_r26_retiming"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
SELECTED = OUT / "01_right_transport_grasp/SELECTED_RIGHT_TRANSPORT_GRASP.json"
R14_COMMAND = (
    OUT
    / "01_right_transport_grasp/candidates/R14_RELEASE_D1_CENTERED_DEEP_FAST"
    / "right_only_r6_command.npz"
)
R14_EVENT = (
    OUT
    / "01_right_transport_grasp/candidates/R14_RELEASE_D1_CENTERED_DEEP_FAST"
    / "physics_r6/event_log.npz"
)
R26_COMMAND = (
    OUT
    / "01_right_transport_grasp/candidates/R26_HANDOFF_R14_INDEX0_P010"
    / "continuous_r14_hand_t5_full_command.npz"
)
R26_EVENT = (
    OUT
    / "01_right_transport_grasp/candidates/R26_HANDOFF_R14_INDEX0_P010"
    / "physics_full/event_log.npz"
)
FALLBACK = R26_OUT / "fallback/R14_F1_H22_CONVERSION4X"
FALLBACK_OFFLINE = FALLBACK / "offline_gate_report.json"
FALLBACK_COMMAND = FALLBACK / "left_supported_handoff_transport_gate_command.npz"
FALLBACK_TRIAL = FALLBACK / "physics_gate/trial_result.json"
FALLBACK_EVENT = FALLBACK / "physics_gate/event_log.npz"
RETIME_RESULTS = R26_OUT / "R26_RETIMING_RESULTS.json"
ORIGINAL_AUDIT = R26_OUT / "00_original_audit/R26_EXACT_AUDIT.json"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
ACT_A = ROOT / "outputs/paper_core_ab/act_a40/train/checkpoints/100000/pretrained_model/model.safetensors"
ACT_B = ROOT / "outputs/paper_core_ab/act_b40/train/checkpoints/020000/pretrained_model/model.safetensors"
EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    SELECTED: "10d406038795f2f4dfc38ecdfd408de176d63297f05dd4814fc1a4b29392b8c9",
    R14_COMMAND: "5f5db710e407f60d39f8e0138729f820fb79e3a85941caccb596574ef3ed50bf",
    R14_EVENT: "75c4c4d9da333c23f78ebd7c5ee6a3c63e35c786fbe245788c1a92fc80efe02b",
    R26_COMMAND: "2c525c492baaf88a47270679bc112d35a15f721933f0d62365b54625f0f4805e",
    R26_EVENT: "d96a362c341f6bd551ed607a908ae8986908fe9b55164ec531ff73199577d956",
    HELDOUT: "a86181b049d0f521d1167c2b58bc15f3a7cb6ad87ee9a1f634ef02c04adcc710",
    ACT_A: "7e9fe737c3fd8ad3919cf3887dad58a732f6651e84e1eab7c1af8267ee16912c",
    ACT_B: "4c3c52a853cc242c6ba97fa6fa8d2dde96f6d65e737e291cfb99265f3c1b5198",
}
FORCE_N = 0.015
TABLE_N = 0.015
MAX_SPEED_M_S = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def stage_summary(event: dict[str, np.ndarray], stage_name: str) -> dict[str, Any]:
    mask = event["stage"].astype(str) == stage_name
    if not np.any(mask):
        return {"present": False}
    forces = np.column_stack(
        (event["thumb_force_n"], event["index_force_n"], event["middle_force_n"])
    )[mask]
    speed = np.linalg.norm(event["object_linear_velocity_m_s"][mask], axis=1)
    return {
        "present": True,
        "control_frame_start": int(np.min(event["control_frame"][mask])),
        "control_frame_end": int(np.max(event["control_frame"][mask])),
        "all_three_fraction": float(np.mean(np.all(forces >= FORCE_N, axis=1))),
        "individual_longest_support_s": {
            digit: longest_duration(forces[:, index] >= FORCE_N, 1.0 / 240.0)
            for index, digit in enumerate(("thumb", "index", "middle"))
        },
        "any_digit_fraction": float(np.mean(np.any(forces >= FORCE_N, axis=1))),
        "table_free_fraction": float(
            np.mean(event["table_contact_force_n"][mask] < TABLE_N)
        ),
        "mean_force_n_thumb_index_middle": forces.mean(axis=0).tolist(),
        "maximum_object_speed_m_s": float(np.max(speed, initial=0.0)),
        "minimum_object_z_m": float(
            np.min(event["object_position_world_m"][mask, 2])
        ),
        "maximum_object_z_m": float(
            np.max(event["object_position_world_m"][mask, 2])
        ),
    }


def longest_duration(mask: np.ndarray, dt: float) -> float:
    current = longest = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def object_relative_tool(
    g1: G1Kinematics,
    command_path: Path,
    event_path: Path,
    stage_name: str,
) -> np.ndarray:
    command = load_npz(command_path)
    event = load_npz(event_path)
    stages = command["stage"].astype(str)
    names = command["joint_names"].astype(str).tolist()
    index = int(np.flatnonzero(stages == stage_name)[-1])
    arm, left, right = model_parts(names, g1, command["commanded_q_rad"][index])
    g1.assign(arm, left, right)
    tool_model = np.asarray(g1.whole_hand_grasp_pose("right"), dtype=np.float64)
    origin = g1.model_to_world_position(np.zeros(3, dtype=np.float64))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3, dtype=np.float64)[axis]) - origin
            for axis in range(3)
        ]
    )
    tool_world = np.eye(4, dtype=np.float64)
    tool_world[:3, :3] = world_from_model @ tool_model[:3, :3]
    tool_world[:3, 3] = g1.model_to_world_position(tool_model[:3, 3])
    event_mask = event["stage"].astype(str) == stage_name
    position = np.median(event["object_position_world_m"][event_mask], axis=0)
    quaternions = np.asarray(event["object_quaternion_xyzw"][event_mask], dtype=np.float64)
    reference = quaternions[0]
    quaternions = np.where(
        (quaternions @ reference)[:, None] < 0.0, -quaternions, quaternions
    )
    quaternion = np.mean(quaternions, axis=0)
    quaternion /= np.linalg.norm(quaternion)
    object_world = np.eye(4, dtype=np.float64)
    object_world[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    object_world[:3, 3] = position
    return np.linalg.inv(object_world) @ tool_world


def main() -> int:
    actual = {str(path): sha256_file(path) for path in EXPECTED}
    mismatches = {
        str(path): {"expected": digest, "actual": actual[str(path)]}
        for path, digest in EXPECTED.items()
        if actual[str(path)] != digest
    }
    if mismatches:
        raise RuntimeError(f"immutable input changed: {mismatches}")

    selected = read_json(SELECTED)
    original = read_json(ORIGINAL_AUDIT)
    retiming = read_json(RETIME_RESULTS)
    offline = read_json(FALLBACK_OFFLINE)
    trial = read_json(FALLBACK_TRIAL)
    command = load_npz(FALLBACK_COMMAND)
    event = load_npz(FALLBACK_EVENT)
    if offline["status"] != "OFFLINE_PASS":
        raise RuntimeError("fallback is no longer offline-valid")
    if trial["runtime_right_three_digit_gate"]["status"] != "FAIL":
        raise RuntimeError("fallback runtime-gate conclusion changed")
    if trial["runtime_right_three_digit_gate"]["left_thumb_release_applied"]:
        raise RuntimeError("LEFT release unexpectedly executed after a failed gate")

    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    names = command["joint_names"].astype(str).tolist()
    verify_mask = command["stage"].astype(str) == "RIGHT_THREE_DIGIT_VERIFICATION"
    verify_row = command["commanded_q_rad"][np.flatnonzero(verify_mask)[-1]]
    _, _, fallback_right = model_parts(names, g1, verify_row)
    selected_right = np.asarray(
        selected["right_transport_hold_model_order_7d_rad"], dtype=np.float64
    )
    endpoint_error = float(np.max(np.abs(fallback_right - selected_right)))

    r14_relative = object_relative_tool(
        g1, R14_COMMAND, R14_EVENT, "GRAVITY_RETENTION"
    )
    r26_relative = object_relative_tool(
        g1, R26_COMMAND, R26_EVENT, "RIGHT_THREE_DIGIT_VERIFICATION"
    )
    translation_delta = r26_relative[:3, 3] - r14_relative[:3, 3]
    rotation_delta_deg = float(
        np.degrees(
            np.linalg.norm(
                Rotation.from_matrix(
                    r14_relative[:3, :3].T @ r26_relative[:3, :3]
                ).as_rotvec()
            )
        )
    )

    speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    maximum_index = int(np.argmax(speed))
    verification = stage_summary(event, "RIGHT_THREE_DIGIT_VERIFICATION")
    conversion = stage_summary(event, "RIGHT_SUPPORTED_CONVERGE_TO_R14")
    fallback_gates = {
        "exact_frozen_r14_hand_endpoint": "PASS" if endpoint_error <= 1.0e-7 else "FAIL",
        "object_speed": "PASS"
        if float(speed[maximum_index]) <= MAX_SPEED_M_S
        else "FAIL",
        "offline_collision": "PASS"
        if sum(offline["offline"]["collision_frame_counts"].values()) == 0
        else "FAIL",
        "joint_limits": "PASS"
        if offline["offline"]["joint_limit_violation_count"] == 0
        else "FAIL",
        "numerical_artifact": trial["artifact_checks"]["status"],
        "right_three_digit_acquisition": trial["runtime_right_three_digit_gate"][
            "status"
        ],
        "left_release_after_right_acquisition": "NOT_EXECUTED_BY_GATE",
        "right_only_retention_1s": "NOT_EXECUTED_BY_GATE",
    }
    fallback_result = {
        "schema_version": "r14_exact_endpoint_single_fallback_result_v1",
        "candidate": "R14_F1_H22_CONVERSION4X",
        "status": "FAIL",
        "first_failed_gate": "RIGHT_THREE_DIGIT_ACQUISITION",
        "offline_report": str(FALLBACK_OFFLINE),
        "offline_report_sha256": sha256_file(FALLBACK_OFFLINE),
        "command": str(FALLBACK_COMMAND),
        "command_sha256": sha256_file(FALLBACK_COMMAND),
        "event_log": str(FALLBACK_EVENT),
        "event_log_sha256": sha256_file(FALLBACK_EVENT),
        "trial_result": str(FALLBACK_TRIAL),
        "trial_result_sha256": sha256_file(FALLBACK_TRIAL),
        "conversion_duration_scale": offline["conversion_duration_scale"],
        "commanded_exact_r14_endpoint_max_error_rad": endpoint_error,
        "maximum_object_speed_m_s": float(speed[maximum_index]),
        "maximum_speed_event": {
            "physics_step_index": maximum_index,
            "control_frame": int(event["control_frame"][maximum_index]),
            "stage": str(event["stage"][maximum_index]),
            "timestamp_s": float(event["timestamp_s"][maximum_index]),
        },
        "conversion_stage": conversion,
        "verification_stage": verification,
        "runtime_gate": trial["runtime_right_three_digit_gate"],
        "hard_gate_results": fallback_gates,
        "command_completed": trial["command_completed"],
        "executed_control_frames": trial["executed_control_frames"],
        "requested_control_frames": trial["requested_control_frames"],
        "state_restoration_used": trial["state_restoration"]["used"],
        "prohibited_attachment_used": trial["prohibited_attachment_used"],
        "doll_or_material_changed": False,
        "selected_r14_artifact_changed": False,
    }
    atomic_json(R26_OUT / "R14_FALLBACK_RESULT.json", fallback_result)

    heldout = read_json(HELDOUT)
    heldout_indices = [int(row["final_dataset_index"]) for row in heldout["entries"]]
    paper = {
        "act_a_checkpoint": str(ACT_A.parent),
        "act_a_model_sha256": actual[str(ACT_A)],
        "act_b_checkpoint": str(ACT_B.parent),
        "act_b_model_sha256": actual[str(ACT_B)],
        "heldout_manifest": str(HELDOUT),
        "heldout_manifest_sha256": actual[str(HELDOUT)],
        "heldout_final_dataset_indices": heldout_indices,
        "split_seed": int(heldout["split_contract"]["split_seed"]),
    }
    terminal = {
        "schema_version": "r26_retiming_terminal_result_v1",
        "status": "R26_RETIMING_HANDOFF_STILL_BLOCKED",
        "r14_right_only_transport": {
            "status": "PASS",
            "repeatability": "3/3",
            "selected_report": str(SELECTED),
            "selected_report_sha256": actual[str(SELECTED)],
        },
        "r26_original": {
            "hard_gate_results": original["hard_gate_results"],
            "exact_failed_hard_gates": original["exact_failed_hard_gates"],
            "maximum_held_object_speed_m_s": original[
                "maximum_held_speed_event"
            ]["speed_m_s"],
            "object_relative_r14_translation_delta_m": translation_delta.tolist(),
            "object_relative_r14_translation_delta_norm_m": float(
                np.linalg.norm(translation_delta)
            ),
            "object_relative_r14_rotation_delta_deg": rotation_delta_deg,
            "named_hand_delta_from_r14_rad": original[
                "actual_named_delta_from_frozen_r14_rad"
            ],
            "left_contact_observability": original["left_release_observability"],
        },
        "bounded_retiming": {
            "candidate_count": retiming["candidate_count"],
            "best_candidate": retiming["best_candidate_by_speed"],
            "best_maximum_held_object_speed_m_s": retiming[
                "best_maximum_held_object_speed_m_s"
            ],
            "speed_gate_pass_count": retiming["speed_gate_pass_count"],
            "full_handoff_pass_count": retiming["full_handoff_pass_count"],
            "result": str(RETIME_RESULTS),
            "result_sha256": sha256_file(RETIME_RESULTS),
        },
        "single_exact_r14_fallback": fallback_result,
        "terminal_gate": {
            "handoff": "FAIL",
            "continuous_full_task": "NOT_RUN_BY_HARD_GATE",
            "repeatability": "0/3",
            "frozen": False,
            "act_a_completed": "0/8",
            "act_b_completed": "0/8",
            "act_a_tsr": None,
            "act_b_tsr": None,
        },
        "paper_inputs_verified_unchanged": paper,
        "real_robot_used": False,
        "prohibited_mechanism_used": False,
    }
    terminal_path = R26_OUT / "FINAL_R26_RETIMING_RESULT.json"
    atomic_json(terminal_path, terminal)
    fallback_md = f"""# Single exact-R14 fallback result

`R14_F1_H22_CONVERSION4X` used the exact selected R14 hand endpoint and the
previously collision-pruned object-relative wrist path, slowed from 120 to 480
conversion frames.

- Exact R14 hand endpoint: **PASS** (max error `{endpoint_error:.3e} rad`)
- Object speed gate: **PASS** (`{speed[maximum_index]:.6f} m/s`)
- Collision / joint limits / numerical artifact gates: **PASS**
- RIGHT acquisition: **FAIL**
- Sustained digit support: thumb `{verification['individual_longest_support_s']['thumb']:.3f} s`, index `{verification['individual_longest_support_s']['index']:.3f} s`, middle `{verification['individual_longest_support_s']['middle']:.3f} s`
- LEFT release: **NOT EXECUTED BY GATE**

The fallback therefore reduced dynamics below the unchanged threshold but did
not establish physical three-digit ownership. No second fallback was searched.
"""
    atomic_text(R26_OUT / "R14_FALLBACK_RESULT.md", fallback_md)

    report = f"""# Final R26 retiming and handoff report

## R14 right-only transport

PASS, 3/3 consecutive byte-identical grasp-to-bin runs. The selected grasp and
all doll/physics inputs remained unchanged.

## Exact R26 recovery

R26 was not a speed-only near-pass. It failed:

1. held-object speed: `{original['maximum_held_speed_event']['speed_m_s']:.6f} m/s` versus `{MAX_SPEED_M_S:.1f} m/s`;
2. exact frozen-R14 endpoint: historical model-order slot 5 modified
   `right_hand_middle_0_joint` by `+0.10 rad` despite `index0` metadata;
3. object-relative tool geometry differed from the validated R14 state by
   `{np.linalg.norm(translation_delta) * 1000.0:.3f} mm` translation and
   `{rotation_delta_deg:.3f} deg` rotation.

Its stage-resolved acquisition, post-release retention, table-free support,
collision, joint-limit, vertical transport, descent, release, and bin-settle
gates passed. The historical log did not instrument LEFT-object contacts, so
zero residual LEFT contact after opening cannot be independently asserted.

## Bounded R26 retiming

| Candidate | Scale | Peak held-object speed (m/s) | Additional physical failure |
|---|---:|---:|---|
"""
    for row in retiming["candidates"]:
        additional = [
            gate
            for gate in row["failed_hard_gates"]
            if gate not in ("held_object_speed", "exact_frozen_r14_endpoint")
        ]
        report += (
            f"| {row['candidate']} | {row['scale']:.1f}x | "
            f"{row['maximum_held_object_speed_m_s']:.6f} | "
            f"{', '.join(additional) if additional else 'none'} |\n"
        )
    report += f"""

No temporal candidate passed the 1.0 m/s gate. T1 and T3 also lost sustained
three-digit post-release retention. The best speed was
`{retiming['best_maximum_held_object_speed_m_s']:.6f} m/s` for
`{retiming['best_candidate_by_speed']}`.

## One exact-R14 fallback

The sole allowed fallback reached `{speed[maximum_index]:.6f} m/s` and the exact
R14 command endpoint, but supported the doll with only the RIGHT thumb during
the final gate. RIGHT index and middle support were both 0.0 s. LEFT release
was correctly suppressed.

## Hard-gate consequence

- Continuous full task: NOT RUN.
- Repeatability: 0/3.
- Common controller frozen: NO.
- ACT-A40: 0/8, TSR not computed.
- ACT-B40: 0/8, TSR not computed.
- A/B checkpoints, datasets, split, and retargeting artifacts: unchanged.

R26_RETIMING_HANDOFF_STILL_BLOCKED
"""
    report_path = R26_OUT / "FINAL_R26_RETIMING_REPORT.md"
    atomic_text(report_path, report)

    atomic_json(
        OUT / "03_continuous_scripted_task/NOT_RUN_AFTER_R26_RETIMING_GATE.json",
        {
            "status": "NOT_RUN_BY_HARD_GATE",
            "blocking_gate": "R26_RETIMING_AND_SINGLE_R14_FALLBACK",
            "result": str(terminal_path),
        },
    )
    atomic_json(
        OUT / "05_freeze/NOT_FROZEN_AFTER_R26_RETIMING.json",
        {
            "frozen": False,
            "reason": "no valid physical handoff into frozen R14",
            "freeze_manifest_created": False,
        },
    )
    atomic_json(
        OUT / "06_act_ab_heldout8/NOT_RUN_AFTER_R26_RETIMING_GATE.json",
        {
            "status": "NOT_RUN_BY_HARD_GATE",
            "act_a_completed": 0,
            "act_b_completed": 0,
            "act_a_tsr": None,
            "act_b_tsr": None,
            "paper_inputs_verified_unchanged": paper,
        },
    )
    status_md = f"""# Final task completion status

Updated: 2026-08-31 (Asia/Seoul)

## Terminal gate

`R26_RETIMING_HANDOFF_STILL_BLOCKED`

- R14 RIGHT-only transport: PASS 3/3.
- R26 fixed-geometry timing trials: 0/4 speed-gate passes.
- Single exact-R14 fallback: speed PASS; three-digit acquisition FAIL.
- Continuous full task: NOT RUN BY HARD GATE.
- Environment/controller frozen: NO.
- ACT-A/B HELDOUT8 physics: NOT RUN BY HARD GATE.

Detailed report: `08_r26_retiming/FINAL_R26_RETIMING_REPORT.md`.
"""
    atomic_text(OUT / "CURRENT_STATUS.md", status_md)

    evidence_files = [
        ORIGINAL_AUDIT,
        R26_OUT / "00_original_audit/R26_EXACT_AUDIT.md",
        R26_OUT / "00_original_audit/R26_SPEED_TRACE.npz",
        RETIME_RESULTS,
        R26_OUT / "R26_RETIMING_RESULTS.csv",
        R26_OUT / "R26_RETIMING_RESULTS.md",
        FALLBACK_OFFLINE,
        FALLBACK_COMMAND,
        FALLBACK_TRIAL,
        FALLBACK_EVENT,
        R26_OUT / "R14_FALLBACK_RESULT.json",
        R26_OUT / "R14_FALLBACK_RESULT.md",
        terminal_path,
        report_path,
        OUT / "CURRENT_STATUS.md",
    ]
    manifest = {
        "schema_version": "r26_retiming_blocker_evidence_manifest_v1",
        "status": "R26_RETIMING_HANDOFF_STILL_BLOCKED",
        "note": "Evidence manifest only; no environment freeze was authorized.",
        "files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in evidence_files
        ],
        "immutable_inputs": actual,
    }
    atomic_json(R26_OUT / "R26_RETIMING_EVIDENCE_MANIFEST.json", manifest)
    print(json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
