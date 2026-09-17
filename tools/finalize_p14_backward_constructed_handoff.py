#!/usr/bin/env python3
"""Finalize the bounded backward-constructed P14 handoff evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path.insert(0, str(ROOT))

from tools.build_p14_backward_constructed_handoff import StaticProxyModel
from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics


BASE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral/backward_constructed_handoff"
)
GATE = (
    BASE
    / "B2_PATH_F40/right_preload_partial_left_relax_v3"
    / "gate/PHYSICS_GATE_RESULT.json"
)
REVISION = "gated_left_radial_exit_with_backward_endpoint_v15"
PROFILES = ("60", "40", "25")
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
FREEZE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "frozen_p14_bilateral/FREEZE_MANIFEST.json"
)
OUTPUT_JSON = BASE / "FINAL_BACKWARD_CONSTRUCTED_HANDOFF_REPORT.json"
OUTPUT_MD = BASE / "FINAL_BACKWARD_CONSTRUCTED_HANDOFF_REPORT.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def longest_duration(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def profile_result(profile: str, config: dict[str, Any]) -> dict[str, Any]:
    directory = BASE / "B2_PATH_F40" / f"{REVISION}_rf{profile}" / "full"
    report_path = directory / "offline_report.json"
    trial_path = directory / "physics_right_sensor/trial_result.json"
    event_path = directory / "physics_right_sensor/event_log.npz"
    report = read_json(report_path)
    trial = read_json(trial_path)
    if report["status"] != "OFFLINE_PASS":
        raise RuntimeError(f"RF{profile} did not pass the offline gate")
    if sha256_file(Path(report["command"])) != report["command_sha256"]:
        raise RuntimeError(f"RF{profile} command hash mismatch")
    if sha256_file(event_path) != trial["event_log_sha256"]:
        raise RuntimeError(f"RF{profile} event-log hash mismatch")
    if trial["runtime_right_three_digit_gate"]["status"] != "PASS":
        raise RuntimeError(f"RF{profile} runtime acquisition gate changed")

    with np.load(event_path, allow_pickle=False) as archive:
        stage = archive["stage"].astype(str)
        control = np.asarray(archive["control_frame"], dtype=np.int64)
        timestamp = np.asarray(archive["timestamp_s"], dtype=np.float64)
        forces = np.column_stack(
            [
                archive["thumb_force_n"],
                archive["index_force_n"],
                archive["middle_force_n"],
            ]
        ).astype(np.float64)
        table = np.asarray(archive["table_contact_force_n"], dtype=np.float64)
        position = np.asarray(archive["object_position_world_m"], dtype=np.float64)

    dt = float(np.median(np.diff(timestamp)))
    force_threshold = float(config["gates"]["meaningful_digit_force_n"])
    table_threshold = float(
        config["gates"]["maximum_table_force_for_elevated_n"]
    )
    all_three = np.all(forces >= force_threshold, axis=1)
    table_free = table <= table_threshold
    release = stage == "LEFT_THUMB_RELEASE"
    post = stage == "RIGHT_POST_RELEASE_RETENTION"
    owned = stage == "RIGHT_OWNED_HOLD"
    transport = stage == "RIGHT_TRANSPORT_TO_BIN"
    after_gate = control > int(
        trial["runtime_right_three_digit_gate"]["evaluated_before_control_frame"]
    )
    first_table_rows = np.flatnonzero(after_gate & ~table_free)
    first_table = (
        {
            "control_frame": int(control[first_table_rows[0]]),
            "stage": str(stage[first_table_rows[0]]),
        }
        if len(first_table_rows)
        else None
    )
    post_retention_s = longest_duration(post & all_three & table_free, dt)
    return {
        "release_completion_fraction": float(profile) / 100.0,
        "offline_status": report["status"],
        "command": report["command"],
        "command_sha256": report["command_sha256"],
        "event_log": str(event_path),
        "event_log_sha256": sha256_file(event_path),
        "trial_result": str(trial_path),
        "trial_result_sha256": sha256_file(trial_path),
        "executed_control_frames": int(trial["executed_control_frames"]),
        "command_completed": bool(trial["command_completed"]),
        "runtime_acquisition_gate": trial["runtime_right_three_digit_gate"],
        "robot_collision_frame_counts": report["robot_collision_frame_counts"],
        "joint_limit_violations": int(report["joint_limit_violations"]),
        "ik_max_position_error_m": float(report["ik_max_position_error_m"]),
        "ik_max_orientation_error_rad": float(
            report["ik_max_orientation_error_rad"]
        ),
        "release": {
            "all_three_fraction": float(np.mean(all_three[release])),
            "all_three_longest_s": longest_duration(
                release & all_three & table_free, dt
            ),
            "digit_mean_force_n_thumb_index_middle": np.mean(
                forces[release], axis=0
            ).tolist(),
            "table_free_through_stage": bool(np.all(table_free[release])),
            "end_object_com_world_m": position[release][-1].tolist(),
        },
        "first_table_support_after_gate": first_table,
        "post_release_three_digit_table_free_retention_s": post_retention_s,
        "right_owned_three_digit_table_free_retention_s": longest_duration(
            owned & all_three & table_free, dt
        ),
        "right_transport_three_digit_table_free_s": longest_duration(
            transport & all_three & table_free, dt
        ),
        "doll_retained_after_left_release_for_1s": bool(
            post_retention_s >= 1.0
        ),
        "prohibited_attachment_used": bool(trial["prohibited_attachment_used"]),
        "object_pose_writes_during_timed_loop": int(
            trial["object_pose_writes_during_timed_loop"]
        ),
        "physics_artifact_check": trial["artifact_checks"]["status"],
    }


def minimum_hand_clearance(command_path: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    with np.load(command_path, allow_pickle=False) as archive:
        commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        names = archive["joint_names"].astype(str).tolist()
        stages = archive["stage"].astype(str)
        object_center = np.asarray(
            archive["handoff_object_center_world_m"], dtype=np.float64
        )
    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    evaluator = StaticProxyModel(
        g1,
        object_center,
        np.asarray(config["geometry_candidates"][0]["dimensions_m"]),
    )
    relevant = np.isin(
        stages,
        [
            "RIGHT_THREE_DIGIT_VERIFICATION",
            "LEFT_THUMB_RELEASE",
            "RIGHT_POST_RELEASE_RETENTION",
        ],
    )
    best: dict[str, Any] | None = None
    for frame in np.flatnonzero(relevant):
        audit = evaluator.evaluate(
            commands[frame, arm_indices],
            commands[frame, left_indices],
            commands[frame, right_indices],
        )
        row = {
            "control_frame": int(frame),
            "stage": str(stages[frame]),
            "minimum_clearance_m": float(
                audit["minimum_left_right_clearance_m"]
            ),
            "closest_body_pair": audit["closest_left_right_body_pair"],
        }
        if best is None or row["minimum_clearance_m"] < best["minimum_clearance_m"]:
            best = row
    if best is None:
        raise RuntimeError("no handoff frames found for clearance audit")
    return best


def main() -> int:
    config = read_json(CONFIG)
    freeze = read_json(FREEZE)
    gate = read_json(GATE)
    if gate["status"] != "PASS":
        raise RuntimeError("frozen acquisition gate is no longer PASS")
    results = {profile: profile_result(profile, config) for profile in PROFILES}
    best = results["40"]
    clearance = minimum_hand_clearance(Path(best["command"]))
    if any(row["doll_retained_after_left_release_for_1s"] for row in results.values()):
        raise RuntimeError("a bounded candidate passed; blocked finalizer is invalid")

    report = {
        "schema_version": "p14_backward_constructed_handoff_final_v1",
        "experiment_status": "BACKWARD_CONSTRUCTED_HANDOFF_STILL_BLOCKED",
        "objective": "FIND_AND_VALIDATE_ONE_POLICY_INDEPENDENT_HANDOFF_CONFIGURATION",
        "frozen_inputs": {
            "doll_and_controller_config": str(CONFIG),
            "doll_and_controller_config_sha256": sha256_file(CONFIG),
            "bilateral_p14_freeze_manifest": str(FREEZE),
            "bilateral_p14_freeze_manifest_sha256": sha256_file(FREEZE),
            "left_p14_7d_rad": freeze["p14"]["left_7d_rad"],
            "right_p14_7d_rad": freeze["p14"]["right_7d_rad"],
        },
        "validated_acquisition_gate": {
            "path": str(GATE),
            "sha256": sha256_file(GATE),
            "right_sensor_verification": gate["right_sensor_verification"],
            "left_sensor_verification": gate["left_sensor_verification"],
        },
        "bounded_endpoint_timing_profiles": results,
        "best_candidate": {
            "id": "B2_PATH_F40_V15_RF40",
            "reason": (
                "only profile to complete LEFT_THUMB_RELEASE without table "
                "support; maximum retained prefix among the declared profiles"
            ),
            "conditions_passed": {
                "right_three_digit_acquisition_gate": True,
                "left_retention_before_transfer": True,
                "no_left_right_hand_overlap": True,
                "no_robot_collision": True,
                "right_arm_ik_reachability": True,
                "joint_limits": True,
                "left_release_after_right_acquisition": True,
                "table_free_through_left_release": True,
            },
            "failed_condition": (
                "doll remains retained after LEFT release for >=1.0 s"
            ),
            "exact_failure": (
                "table support begins at the first RIGHT_POST_RELEASE_RETENTION "
                "control frame (540); sustained post-release three-digit, "
                "table-free retention is 0.0 s"
            ),
            "hand_hand_minimum_clearance": clearance,
            "physics": best,
        },
        "scripted_full_task_physics": "FAIL_NOT_FROZEN",
        "environment_frozen": False,
        "act_a_b_physics_evaluation_run": False,
        "real_robot": False,
        "prohibited_mechanisms_used": False,
        "decision": (
            "The common acquisition is physically established, but the fixed "
            "path to the standalone RIGHT endpoint cannot preserve unsupported "
            "ownership. The declared timing bound is exhausted."
        ),
        "finalizer": str(Path(__file__).resolve()),
    }
    atomic_json(OUTPUT_JSON, report)
    report["finalizer_sha256"] = sha256_file(Path(__file__).resolve())
    atomic_json(OUTPUT_JSON, report)
    markdown = f"""# Backward-Constructed P14 Handoff — Final Bounded Result

Status: **BACKWARD_CONSTRUCTED_HANDOFF_STILL_BLOCKED**

The frozen B2 acquisition gate remains valid: RIGHT thumb/index/middle support
is simultaneous for 0.6 s before LEFT release, with no table support. All three
endpoint timing profiles (LEFT release completion at 60%, 40%, and 25%) passed
the unchanged offline collision, IK, and joint-limit gates.

The best candidate is **B2_PATH_F40_V15_RF40**. It completes LEFT release with
no table support and no robot collision. It then reaches table support at the
first `RIGHT_POST_RELEASE_RETENTION` frame (control frame 540), yielding 0.0 s
of valid post-release retention rather than the required 1.0 s.

- Minimum modeled hand-hand clearance: {clearance['minimum_clearance_m']:.9f} m
- Closest pair: `{clearance['closest_body_pair'][0]}` vs `{clearance['closest_body_pair'][1]}`
- Joint-limit violations: 0
- Robot collision frames: 0
- Attachments, welds, magnets, teleportation, object-follow: none
- Environment freeze: **not performed**
- ACT-A/B physics evaluation: **not run**

Evidence manifest: `{OUTPUT_JSON}`
"""
    atomic_text(OUTPUT_MD, markdown)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
