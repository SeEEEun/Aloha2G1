#!/usr/bin/env python3
"""Finalize the bounded success-first transport investigation.

The report is evidence-only.  It does not run physics, restore state, freeze an
environment, or invoke either ACT policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "success_first_common_execution"
)
KEYFRAME_MANIFEST = OUT / "post_handoff_keyframe/POST_HANDOFF_KEYFRAME_MANIFEST.json"
BOUNDARY_AUDIT = (
    OUT
    / "tail_debug/t4_three_digit_wrap_slow_transport"
    / "vertical_to_horizontal_boundary_audit.json"
)
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
P14_FREEZE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "frozen_p14_bilateral/FREEZE_MANIFEST.json"
)
TRIALS = {
    "T4_LEGACY": OUT / "tail_debug/t4_three_digit_wrap_slow_transport",
    "T5_ROBUST_SMOOTH": OUT / "tail_debug/t5_t4_grip_robust_final_transport_v1",
    "T6_QUASISTATIC": OUT / "tail_debug/t6_t4_grip_quasistatic_transport_v2",
    "T7_SUPPORTIVE_PIVOT": OUT / "tail_debug/t7_t4_grip_supportive_carry_v3",
    "T8_RETENTION_HORIZON": OUT / "tail_debug/t8_t4_grip_within_retention_horizon_v4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def serial(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def stage_summary(event: Path, stage_name: str) -> dict[str, Any]:
    with np.load(event, allow_pickle=False) as archive:
        labels = archive["stage"].astype(str)
        mask = labels == stage_name
        if not np.any(mask):
            return {"present": False}
        forces = np.column_stack(
            [archive[f"{digit}_force_n"] for digit in ("thumb", "index", "middle")]
        ).astype(np.float64)
        positions = np.asarray(archive["object_position_world_m"], dtype=np.float64)
        control = np.asarray(archive["control_frame"], dtype=np.int64)
        rows = np.flatnonzero(mask)
        return {
            "present": True,
            "control_frame_first": int(control[rows[0]]),
            "control_frame_last": int(control[rows[-1]]),
            "object_start_world_m": positions[rows[0]],
            "object_end_world_m": positions[rows[-1]],
            "mean_force_n_thumb_index_middle": np.mean(forces[mask], axis=0),
            "minimum_force_n_thumb_index_middle": np.min(forces[mask], axis=0),
            "all_three_loaded_fraction": float(
                np.mean(np.all(forces[mask] >= 0.015, axis=1))
            ),
            "maximum_table_force_n": float(
                np.max(np.asarray(archive["table_contact_force_n"])[mask], initial=0.0)
            ),
        }


def main() -> int:
    keyframe = read_json(KEYFRAME_MANIFEST)
    boundary = read_json(BOUNDARY_AUDIT)
    trials: dict[str, Any] = {}
    for name, root in TRIALS.items():
        event = root / "physics_right_sensor/event_log.npz"
        offline = read_json(root / "offline_report.json")
        trial = read_json(root / "physics_right_sensor/trial_result.json")
        trials[name] = {
            "candidate": offline["candidate"],
            "command": str(root / "tail_command.npz"),
            "command_sha256": sha256(root / "tail_command.npz"),
            "event_log": str(event),
            "event_log_sha256": sha256(event),
            "state_restoration_debug_only": trial["state_restoration"]["used"],
            "command_completed": trial["command_completed"],
            "joint_limit_violations": offline["joint_limit_violations"],
            "robot_collision_frame_counts": offline["robot_collision_frame_counts"],
            "vertical": stage_summary(event, "RIGHT_TRANSPORT_VERTICAL_CLEARANCE"),
            "stabilization": stage_summary(event, "RIGHT_VERTICAL_STABILIZATION"),
            "supportive_pivot": stage_summary(event, "RIGHT_SUPPORTIVE_CARRY_PIVOT"),
            "horizontal": stage_summary(event, "RIGHT_TRANSPORT_TO_BIN"),
            "descent": stage_summary(event, "RIGHT_CONTROLLED_BIN_DESCENT"),
        }

    t5_event = TRIALS["T5_ROBUST_SMOOTH"] / "physics_right_sensor/event_log.npz"
    with np.load(t5_event, allow_pickle=False) as archive:
        labels = archive["stage"].astype(str)
        stable = labels == "RIGHT_VERTICAL_STABILIZATION"
        contact_geometry = {}
        object_com = np.mean(archive["object_position_world_m"][stable], axis=0)
        for digit in ("thumb", "index", "middle"):
            force = np.asarray(archive[f"{digit}_force_n"], dtype=np.float64)
            valid = stable & (force >= 0.015)
            contact_geometry[digit] = {
                "mean_force_n": float(np.mean(force[stable])),
                "mean_contact_point_world_m": np.average(
                    archive[f"{digit}_contact_point_world_m"][valid],
                    axis=0,
                    weights=force[valid],
                ),
                "mean_contact_normal_world": np.average(
                    archive[f"{digit}_contact_normal_world"][valid],
                    axis=0,
                    weights=force[valid],
                ),
            }

    t8_horizontal = trials["T8_RETENTION_HORIZON"]["horizontal"]
    bin_center = np.asarray([0.7382120490074158, 0.09978766366839409])
    final_xy = np.asarray(t8_horizontal["object_end_world_m"][:2])
    start_xy = np.asarray(t8_horizontal["object_start_world_m"][:2])
    payload = {
        "schema_version": "success_first_continuous_full_task_final_v1",
        "status": "CONTINUOUS_FULL_TASK_STILL_BLOCKED",
        "post_handoff_keyframe": {
            "status": keyframe["status"],
            "source": keyframe["source"],
            "keyframe": keyframe["keyframe"],
            "keyframe_sha256": keyframe["keyframe_sha256"],
            "right_only_retention_s": 1.0,
            "minimum_force_n_thumb_index_middle": keyframe[
                "following_one_second_verification"
            ]["minimum_right_force_n_thumb_index_middle"],
            "table_force_n": keyframe["following_one_second_verification"][
                "maximum_table_force_n"
            ],
            "state_restoration_scope": "TAIL_DEBUGGING_ONLY",
        },
        "horizontal_start_classification": {
            "classification": boundary["classification"],
            "audit": str(BOUNDARY_AUDIT),
            "audit_sha256": sha256(BOUNDARY_AUDIT),
            "boundary_command": boundary["evidence"]["boundary_command"],
            "excluded_failure_classes": boundary["evidence"][
                "excluded_failure_classes"
            ],
        },
        "bounded_tail_trials": trials,
        "physical_root_cause": {
            "class": "TRUE_GRASP_RETENTION_FAILURE_DUE_TO_UNILATERAL_CONTACT_TOPOLOGY",
            "stable_object_com_world_m": object_com,
            "contact_geometry": contact_geometry,
            "observation": (
                "All three stable contact points lie on the +world-X side of the doll COM. "
                "The bin requires +world-X transport, so the fixed grasp must pull the rigid "
                "proxy tangentially rather than push or cradle it."
            ),
            "time_scaling_result": "T5 and T6 fail after a similar elapsed high-pose interval; reducing speed does not restore transport",
            "supportive_pivot_result": "T7 loses the unilateral contacts during the object-centered wrist pivot",
            "within_horizon_result": {
                "object_horizontal_start_world_m": start_xy,
                "object_horizontal_end_world_m": final_xy,
                "object_delta_xy_m": final_xy - start_xy,
                "bin_center_xy_m": bin_center,
                "remaining_xy_distance_to_bin_center_m": float(
                    np.linalg.norm(final_xy - bin_center)
                ),
            },
        },
        "stage_gate": {
            "LEFT_GRASP": "PASS_EXISTING_CONTINUOUS_SOURCE_RUN",
            "LEFT_LIFT": "PASS_EXISTING_CONTINUOUS_SOURCE_RUN",
            "LEFT_TRANSPORT": "PASS_EXISTING_CONTINUOUS_SOURCE_RUN",
            "HANDOFF": "PASS_TO_VERIFIED_POST_HANDOFF_KEYFRAME",
            "LEFT_RELEASE": "PASS_TO_VERIFIED_POST_HANDOFF_KEYFRAME",
            "RIGHT_RETENTION": "PASS_1.0_S",
            "LEFT_RETREAT": "PASS_TAIL_DEBUG",
            "RIGHT_VERTICAL_LIFT": "PASS_145_MM_T5_T6_T7_T8",
            "RIGHT_HORIZONTAL_TRANSPORT": "FAIL",
            "BIN_RELEASE": "NOT_REACHED_WITH_DOLL",
            "DOLL_SETTLED_IN_BIN": "FAIL",
        },
        "final_continuous_validation": {
            "run": False,
            "reason": "tail hard gate failed; no failed tail was integrated into a nominal full-task validation",
            "state_restoration_used": False,
            "prohibited_mechanism_used": False,
        },
        "freeze": {
            "new_full_task_environment_frozen": False,
            "reason": "no continuous full-task success",
            "unchanged_doll_config": str(CONFIG),
            "unchanged_doll_config_sha256": sha256(CONFIG),
            "existing_p14_freeze_manifest": str(P14_FREEZE),
            "existing_p14_freeze_manifest_sha256": sha256(P14_FREEZE),
        },
        "act_a_b_physics_evaluation": {
            "run": False,
            "reason": "scripted full-task hard gate did not pass",
        },
        "real_robot": False,
    }
    json_path = OUT / "FINAL_CONTINUOUS_FULL_TASK_REPORT.json"
    atomic_write(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=serial)
        + "\n",
    )
    md_path = OUT / "FINAL_CONTINUOUS_FULL_TASK_REPORT.md"
    md = f"""# Final continuous full-task result

Status: **CONTINUOUS_FULL_TASK_STILL_BLOCKED**

## Proven successes

- Exact post-handoff keyframe recovered from `{keyframe['source']['run']}`.
- Right thumb/index/middle retained the elevated doll for 1.0 s after complete left release; table force was zero.
- The unchanged T4 transport wrap retained all three digits through left clearance, the complete 145 mm vertical lift, and stabilization.
- All offline transport commands had zero joint-limit violations and zero modeled robot-collision frames.

## Horizontal-start diagnosis

Classification: **{boundary['classification']}**.

At onset, arm delta was `{boundary['evidence']['boundary_command']['all_arm_joint_delta_norm_rad']:.9g}` rad, wrist translation was `{boundary['evidence']['boundary_command']['wrist_translation_delta_norm_m']:.9g}` m, wrist rotation was `{boundary['evidence']['boundary_command']['wrist_orientation_delta_rad']:.9g}` rad, and the finger command delta was exactly zero. The executed loss therefore is not an IK, interpolation, orientation, acceleration-onset, collision, or reachability discontinuity.

## Bounded transport attempts

- T5: smooth 8 s traverse; vertical/stabilization all-three support 100%, horizontal failed.
- T6: 24 s quasistatic traverse; horizontal failed, ruling out speed reduction.
- T7: geometry-derived supportive wrist pivot; contact failed during pivot.
- T8: 1.5 s minimum-jerk traverse inside the measured retention horizon; doll moved only `{float(final_xy[0]-start_xy[0]):.6f}` m in +X and finished `{float(np.linalg.norm(final_xy-bin_center)):.6f}` m from the bin center.

The stable contact points are all on the +world-X side of the doll COM, while the bin requires +world-X transport. With the frozen grasp, the rigid proxy must be pulled tangentially and escapes instead of being carried.

## Hard gate

- RIGHT_HORIZONTAL_TRANSPORT: **FAIL**
- BIN_RELEASE: **NOT REACHED WITH DOLL**
- DOLL_SETTLED_IN_BIN: **FAIL**
- New full-task freeze: **NO**
- ACT-A/B physics evaluation: **NOT RUN**
- Final continuous run with restoration: **NEVER RUN**
- Attachment/weld/magnet/object-follow: **NEVER USED**

Machine-readable report: `{json_path}`
"""
    atomic_write(md_path, md)
    print(json.dumps({"status": payload["status"], "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
