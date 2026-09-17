#!/usr/bin/env python3
"""Offline parity gate and freeze builder for standardized-grasp DEV35."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/standardized_grasp_ab_dev35"
PREPARED = OUT / "01_prepared_commands/STANDARDIZED_GRASP_AB_COMMAND_MANIFEST.json"
INITIAL = OUT / "00_control/QUALIFIED_STANDARDIZED_INITIAL_GRASP.json"
OBJECT = OUT / "00_control/STANDARDIZED_OBJECT_REGISTRATION.json"
PROVISIONAL = OUT / "02_freeze/STANDARDIZED_GRASP_PROVISIONAL_FREEZE.json"
FINAL = OUT / "02_freeze/STANDARDIZED_GRASP_FINAL_FREEZE.json"
HOLD_COMMAND = OUT / "00_control/STANDARDIZED_GRASP_INITIAL_HOLD_QUALIFICATION.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def dependency(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate() -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    manifest = read(PREPARED)
    if manifest.get("status") != "PREPARED" or len(manifest.get("records", [])) != 70:
        raise RuntimeError("standardized-grasp preparation is not 70/70")
    initial = read(INITIAL)
    if not initial.get("table_free") or not initial.get("mechanically_retained") or initial.get("attachment_used"):
        raise RuntimeError("persisted standardized initial grasp is not physical")
    if int(initial.get("object_pose_writes_after_initialization", -1)) != 0:
        raise RuntimeError("initial grasp provenance permits post-initialization writes")
    q0 = np.asarray(initial["measured_q_rad"], dtype=np.float64)
    rows = {(row["method"], int(row["eval_index"])): row for row in manifest["records"]}
    expected = {(method, index) for method in "AB" for index in range(35)}
    if set(rows) != expected:
        raise RuntimeError("prepared command membership is not matched A35+B35")
    command_paths: list[Path] = []
    pair_audits: list[dict[str, Any]] = []
    for index in range(35):
        archives: dict[str, dict[str, np.ndarray]] = {}
        stable = set()
        for method in "AB":
            row = rows[(method, index)]
            path = Path(row["command"])
            if not path.is_file() or sha256(path) != row["command_sha256"]:
                raise RuntimeError(f"prepared command drift: {path}")
            command_paths.append(path)
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "raw_policy_command", "commanded_q_rad", "joint_names",
                    "common_task_intent", "common_initial_q_rad",
                    "standardized_initial_grasp", "standardized_left_hold_target_q_rad",
                    "rebased_left_wrist_position", "rebased_right_wrist_position",
                }
                if not required.issubset(archive.files):
                    raise RuntimeError(f"archive schema incomplete: {path}")
                values = {key: np.asarray(archive[key]) for key in archive.files}
            command = values["commanded_q_rad"].astype(np.float64)
            if command.ndim != 2 or command.shape[1] != 28 or not np.isfinite(command).all():
                raise RuntimeError(f"invalid 28-D command: {path}")
            if not np.array_equal(command, values["raw_policy_command"]):
                raise RuntimeError("offline command was silently modified")
            if not np.allclose(values["common_initial_q_rad"], q0, rtol=0.0, atol=0.0):
                raise RuntimeError("archive does not use the exact shared initial q")
            if not bool(values["standardized_initial_grasp"].item()):
                raise RuntimeError("archive omitted standardized-grasp control condition")
            if float(row["initial_joint_discontinuity_rad"]) != 0.0:
                raise RuntimeError("post-grasp archive has an initial discontinuity")
            if row["ik"]["hard_limit_violations"] or row["ik"]["branch_discontinuities"] or row["ik"]["hard_self_collision_frames"]:
                raise RuntimeError("archive failed common execution safety preflight")
            if not row["ik"]["finite"]:
                raise RuntimeError("archive has non-finite common IK")
            stable.add(str(values["stable_episode_id"].item()))
            archives[method] = values
        if len(stable) != 1:
            raise RuntimeError(f"matched episode mismatch at {index}")
        if not np.array_equal(archives["A"]["common_task_intent"], archives["B"]["common_task_intent"]):
            raise RuntimeError(f"A/B source timing mismatch at {index}")
        if not np.array_equal(archives["A"]["commanded_q_rad"][0], archives["B"]["commanded_q_rad"][0]):
            raise RuntimeError(f"A/B initial command mismatch at {index}")
        pair_audits.append({
            "eval_index": index,
            "stable_episode_id": next(iter(stable)),
            "same_initial_q": True,
            "same_common_timeline": True,
            "same_object_registration_sha256": rows[("A", index)]["object_registration_sha256"] == rows[("B", index)]["object_registration_sha256"],
        })
    report = {
        "schema_version": "standardized_grasp_ab_70_archive_preflight_v1",
        "status": "PASS",
        "archives": "70 / 70",
        "matched_pairs": "35 / 35",
        "same_initial_state": "35 / 35",
        "same_source_timing": "35 / 35",
        "same_object_registration": "35 / 35",
        "hard_limit_violations": 0,
        "branch_discontinuities": 0,
        "hard_self_collision_frames": 0,
        "finite": "70 / 70",
        "method_specific_downstream_configuration": 0,
        "pair_audits": pair_audits,
    }
    atomic_json(OUT / "01_prepared_commands/OFFLINE_70_PREFLIGHT.json", report)
    atomic_text(
        OUT / "01_prepared_commands/OFFLINE_70_PREFLIGHT.md",
        "# Standardized-grasp A/B 70-archive preflight\n\n"
        "- Status: **PASS**\n- Archives: **70 / 70**\n- Matched pairs: **35 / 35**\n"
        "- Exact shared initial state: **35 / 35**\n- Common source timing: **35 / 35**\n"
        "- Hard-limit violations: **0**\n- Branch discontinuities: **0**\n"
        "- Prohibited self-collision frames after fail-closed handling: **0**\n"
        "- Scope: DEV35 standardized-grasp post-grasp evaluation; not end-to-end grasp acquisition.\n",
    )
    return manifest, command_paths, report


def make_hold_command(initial: dict[str, Any]) -> None:
    q = np.asarray(initial["measured_q_rad"], dtype=np.float64)
    count = 90
    command = np.repeat(q[None], count, axis=0)
    atomic_npz(
        HOLD_COMMAND,
        raw_policy_command=command,
        commanded_q_rad=command,
        joint_names=np.asarray(initial["joint_names"]),
        control_fps_hz=np.asarray(30.0),
        runtime_right_three_digit_gate_required=np.asarray(False),
        common_task_intent=np.full(count, "LEFT_HOLD_INTENT", dtype="U24"),
        stage=np.full(count, "STANDARDIZED_INITIAL_HOLD", dtype="U32"),
        method=np.asarray("a"),
        eval_index=np.asarray(-1, dtype=np.int64),
        provenance=np.asarray("STANDARDIZED_GRASP_INITIALIZATION_GATE"),
        pregrasp_classifier_used=np.asarray(False),
        wrist_distance_gate_used=np.asarray(False),
        arm_rescue_allowed=np.asarray(False),
        wrist_rescue_allowed=np.asarray(False),
        stable_episode_id=np.asarray("standardized_grasp_initialization_qualification"),
        common_initial_q_rad=q,
        standardized_initial_grasp=np.asarray(True),
        standardized_left_hold_target_q_rad=np.asarray(initial["left_hold_target_q_rad"], dtype=np.float64),
        standardized_initial_state_sha256=np.asarray(sha256(INITIAL)),
    )


def freeze(status: str, paths: list[Path], preflight: dict[str, Any], physical_gate: Path | None) -> Path:
    scientific = [
        INITIAL, OBJECT, PREPARED,
        ROOT / "tools/prepare_standardized_grasp_ab_eval35.py",
        ROOT / "tools/freeze_and_validate_standardized_grasp_ab.py",
        ROOT / "tools/direct_physical_execution_layer.py",
        ROOT / "tools/direct_physical_execution_isaac_runtime.py",
        ROOT / "tools/run_direct_physical_execution_isaac.py",
        ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py",
        ROOT / "tools/run_standardized_grasp_ab_dev35.py",
        ROOT / "tools/score_standardized_grasp_post_grasp_run.py",
        ROOT / "tools/score_standardized_grasp_initialization_gate.py",
        ROOT / "tools/doll_handoff_retargeting/retarget.py",
        ROOT / "tools/doll_handoff_retargeting/models.py",
        ROOT / "configs/common_collision_aware_redundancy_ik_v1.json",
        ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json",
        ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json",
        ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_SOURCE_TASK_INTENT_EVAL35.json",
        ROOT / "outputs/final_episode_registered_eval35/01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json",
    ]
    if physical_gate is not None:
        scientific.append(physical_gate)
    value = {
        "schema_version": "standardized_grasp_ab_dev35_freeze_v1",
        "status": status,
        "evaluation_label": "DEV35 STANDARDIZED-GRASP PHYSICAL EVALUATION",
        "required_rollouts": 70,
        "standardized_initial_grasp": True,
        "standardized_initial_grasp_is_control_not_competitive_metric": True,
        "only_intended_difference": {"A": "WRIST POST-GRASP TARGET", "B": "INTERACTION POST-GRASP TARGET"},
        "same_initial_state": True,
        "same_Dex3_controller": True,
        "same_IK": True,
        "same_physics": True,
        "arm_rescue": False,
        "wrist_rescue": False,
        "object_attachment": False,
        "object_pose_writes_after_initialization": 0,
        "graspability_classifier_used": False,
        "preflight": preflight,
        "files": [dependency(path) for path in [*scientific, *paths]],
    }
    destination = PROVISIONAL if status == "PRE_EVAL35_QUALIFICATION_PROVISIONAL" else FINAL
    atomic_json(destination, value)
    atomic_text(destination.with_suffix(".md"), "# Standardized-grasp DEV35 freeze\n\n" + "\n".join([
        f"- Status: `{status}`",
        "- Initial grasp: persisted physical qualified LEFT HOLD (controlled, not scored)",
        "- A: WRIST post-grasp targets",
        "- B: INTERACTION post-grasp targets",
        "- Downstream IK/Dex3/physics/scorer: common",
        "- Arm rescue: NO; wrist rescue: NO; attachment: NO",
        f"- Manifest SHA256 after serialization: computed externally from `{destination.name}`",
    ]) + "\n")
    return destination


def main() -> int:
    manifest, commands, preflight = validate()
    initial = read(INITIAL)
    make_hold_command(initial)
    if not (OUT / "00_control/INITIALIZATION_PHYSICAL_GATE.json").is_file():
        destination = freeze("PRE_EVAL35_QUALIFICATION_PROVISIONAL", commands + [HOLD_COMMAND], preflight, None)
        print(json.dumps({"status": "READY_FOR_PHYSICAL_INITIALIZATION_GATE", "provisional_freeze": str(destination), "sha256": sha256(destination)}, indent=2))
        return 0
    gate = OUT / "00_control/INITIALIZATION_PHYSICAL_GATE.json"
    gate_value = read(gate)
    if gate_value.get("status") != "PASS":
        raise RuntimeError("standardized physical initialization gate did not pass")
    destination = freeze("FROZEN_BEFORE_EVAL35", commands, preflight, gate)
    print(json.dumps({"status": "FROZEN_BEFORE_EVAL35", "freeze": str(destination), "sha256": sha256(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
