#!/usr/bin/env python3
"""Prepare audit-only fixed-wrist Fair-A closure commands and registrations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_contact_constrained_eval/11_gate_bias_audit/fair_a_best_case_physics"
REFERENCE = (
    ROOT
    / "outputs/final_contact_constrained_eval/04_eval10_preparation/fair_a_final_resolver/after_full_pose/trajectories/new_unseen_20260901_140555.npz"
)
REFERENCE_COMMAND = (
    ROOT
    / "outputs/final_contact_constrained_eval/09_reference_alignment_preflight/commands/act_a40/eval_08_new_unseen_20260901_140555.npz"
)
PHYSICS = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
GRASP = ROOT / "configs/contact_eval_common_dex3_grasp_realization_v1.json"
FRAME = 202


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def interpolate(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, frames)
    weight = minimum_jerk(alpha)[:, None]
    return (1.0 - weight) * start + weight * end


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    physics = json.loads(PHYSICS.read_text(encoding="utf-8"))
    grasp = json.loads(GRASP.read_text(encoding="utf-8"))
    # Use the previously prepared, hash-addressed reference command because it
    # already canonicalizes the named Dex3 columns to the Isaac contract.  The
    # numerical arm/wrist state remains the frozen Fair-A reference state.
    with np.load(REFERENCE_COMMAND, allow_pickle=False) as archive:
        reference_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        names = np.asarray(archive["joint_names"]).astype(str)
    base = reference_q[FRAME].copy()
    opened = base.copy()
    opened[14:21] = grasp["open_7d_rad"]
    preshape = base.copy()
    preshape[14:21] = grasp["preshape_7d_rad"]
    closed = base.copy()
    closed[14:21] = grasp["full_close_7d_rad"]
    rows: list[np.ndarray] = []
    stages: list[str] = []

    def extend(values: np.ndarray, stage: str) -> None:
        rows.extend(values)
        stages.extend([stage] * len(values))

    extend(np.repeat(opened[None, :], 30, axis=0), "OPEN")
    extend(interpolate(opened, preshape, 16)[1:], "PRESHAPE")
    extend(interpolate(preshape, closed, 31)[1:], "POWER_GRASP")
    extend(np.repeat(closed[None, :], 60, axis=0), "GRAVITY_RETENTION")
    commands = np.asarray(rows, dtype=np.float64)
    command_path = OUT / "fair_a_eval08_frame202_fixed_wrist_full_close.npz"
    temporary = command_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=commands.astype(np.float32),
            stage=np.asarray(stages),
            joint_names=names,
            control_fps_hz=np.asarray(30.0),
            audit_only=np.asarray(True),
            source_reference=np.asarray(str(REFERENCE)),
            source_reference_sha256=np.asarray(sha256(REFERENCE)),
            canonical_reference_command=np.asarray(str(REFERENCE_COMMAND)),
            canonical_reference_command_sha256=np.asarray(sha256(REFERENCE_COMMAND)),
            source_frame=np.asarray(FRAME),
            arm_wrist_modified=np.asarray(False),
            interaction_frame_used=np.asarray(False),
        )
    os.replace(temporary, command_path)

    registrations = []
    for label, yaw, quaternion in (
        ("identity_yaw", 0.0, [0.0, 0.0, 0.0, 1.0]),
        (
            "adversarial_best_yaw45",
            45.0,
            [0.0, 0.0, 0.3826834323650898, 0.9238795325112867],
        ),
    ):
        registration = {
            "schema_version": "common_task_frame_registration_v1",
            "status": "ADVERSARIAL_AUDIT_ONLY_NOT_EVALUATOR",
            "description": "Audit-only common task position. Yaw is either identity or the best Fair-A counterfactual; neither uses Proposed-B.",
            "base_physics_config": str(PHYSICS),
            "base_physics_config_sha256": sha256(PHYSICS),
            "registered_doll_center_world_xy_m": [
                0.11035161837935448,
                0.053192950785160065,
            ],
            "registered_doll_orientation_quaternion_xyzw": quaternion,
            "registered_doll_yaw_deg": yaw,
            "method_independent": True,
            "episode_independent": True,
            "changes_object_physics": False,
            "changes_robot_commands": False,
            "changes_task_success_definition": False,
            "audit_only": True,
        }
        path = OUT / f"registration_{label}.json"
        path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        registrations.append({"label": label, "path": str(path.resolve()), "sha256": sha256(path)})
    manifest = {
        "schema_version": "fair_a_gate_bias_fixed_wrist_physics_v1",
        "status": "PREPARED",
        "command": str(command_path.resolve()),
        "command_sha256": sha256(command_path),
        "source_reference": str(REFERENCE),
        "source_reference_sha256": sha256(REFERENCE),
        "canonical_reference_command": str(REFERENCE_COMMAND),
        "canonical_reference_command_sha256": sha256(REFERENCE_COMMAND),
        "source_frame": FRAME,
        "arm_wrist_modified": False,
        "common_full_close_only": True,
        "registrations": registrations,
    }
    (OUT / "PREPARATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
