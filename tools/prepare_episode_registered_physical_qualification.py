#!/usr/bin/env python3
"""Prepare policy-free inputs and a provisional freeze for physical qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_episode_registered_eval35/00_qualification"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
REGISTRATION = ROOT / "outputs/final_episode_registered_eval35/00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
OLD_SCRIPTED = ROOT / "outputs/final_direct_physical_eval35/00_authoritative_physical_scene/task_relative_scripted_validation/corrected_task_relative_scripted_regression.npz"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
PHYSICAL_ENVIRONMENT = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
COMMON_CONTROLLER = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_COMMON_EXECUTION_CONTROLLER.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def initial_q(command: np.ndarray) -> np.ndarray:
    value = np.asarray(command[0], dtype=np.float64).copy()
    value[14:] = np.asarray(
        [
            0.0, 0.4, 0.005, -0.005, -0.005, -0.005, -0.005,
            0.0, -0.4, -0.005, 0.005, 0.005, 0.005, 0.005,
        ],
        dtype=np.float64,
    )
    return value


def wrap(source: Path, destination: Path, side: str, stable_id: str) -> None:
    with np.load(source, allow_pickle=False) as archive:
        command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        stages = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str)
        fps = float(np.asarray(archive["control_fps_hz"]).item())
    intent = np.full(len(command), "OPEN_INTENT", dtype="U24")
    if stable_id == "scripted_full_task_non_eval":
        left_close = np.asarray(
            [
                label.startswith("LEFT_PRESHAPE")
                or label in {"LEFT_POWER_GRASP", "GRAVITY_RETENTION"}
                for label in stages
            ]
        )
        left_hold = np.asarray(
            [
                label.startswith("LEFT_")
                and label not in {"LEFT_OPEN", "LEFT_POWER_GRASP", "LEFT_THUMB_RELEASE"}
                and not label.startswith("LEFT_PRESHAPE")
                for label in stages
            ]
        )
        # Align the fixed-duration common PRESHAPE+PROGRESSIVE_CLOSE realization
        # to the end of the existing source-derived acquisition segment.  This
        # uses command stage timing only--never contact or task outcomes--and
        # avoids starting finger closure in RIGHT_COLLISION_FREE_APPROACH.
        backward = np.flatnonzero(stages == "BACKWARD_ACQUISITION_PATH")
        if not len(backward):
            raise RuntimeError("scripted command lacks BACKWARD_ACQUISITION_PATH")
        common_close_frames = max(
            2,
            round(
                (
                    float(json.loads(CONFIG.read_text())["timing"]["preshape_transition_s"])
                    + float(json.loads(CONFIG.read_text())["timing"]["power_close_s"])
                )
                * fps
            ),
        )
        handoff_start = max(0, int(backward[-1]) + 1 - common_close_frames)
        handoff_stop_rows = np.flatnonzero(stages == "RIGHT_THREE_DIGIT_VERIFICATION")
        if not len(handoff_stop_rows):
            raise RuntimeError("scripted command lacks RIGHT_THREE_DIGIT_VERIFICATION")
        handoff = np.zeros(len(command), dtype=bool)
        handoff[handoff_start : int(handoff_stop_rows[-1]) + 1] = True
        right_hold = np.asarray(
            [
                label.startswith("RIGHT_TRANSPORT")
                or label in {
                    "RIGHT_VERTICAL_STABILIZATION",
                    "RIGHT_RIM_RELATIVE_CONTROLLED_DESCENT",
                    "RIGHT_PRE_RELEASE_STABILIZATION",
                    "RIGHT_HOLD_OVER_BIN",
                    "RIGHT_POST_RELEASE_RETENTION",
                    "BIN_SETTLE",
                }
                for label in stages
            ]
        )
        final_release = np.asarray(
            [
                label.startswith("RIGHT_RELEASE")
                or label.startswith("RIGHT_POST_RELEASE_RETREAT")
                for label in stages
            ]
        )
        intent[left_close] = "LEFT_CLOSE_INTENT"
        intent[left_hold] = "LEFT_HOLD_INTENT"
        intent[handoff] = "HANDOFF_INTENT"
        intent[right_hold] = "RIGHT_HOLD_INTENT"
        intent[final_release] = "FINAL_RELEASE_INTENT"
        method = "a"
    elif side == "left":
        intent[np.isin(stages, ["PRESHAPE", "POWER_GRASP"])] = "LEFT_CLOSE_INTENT"
        intent[np.isin(stages, ["GRAVITY_RETENTION", "LIFT_5CM", "HOLD_ELEVATED", "LOWER"])] = "LEFT_HOLD_INTENT"
        intent[np.isin(stages, ["RELEASE", "POST_RELEASE"])] = "FINAL_RELEASE_INTENT"
        method = "a"
    else:
        intent[np.isin(stages, ["PRESHAPE", "POWER_GRASP"])] = "HANDOFF_INTENT"
        intent[np.isin(stages, ["GRAVITY_RETENTION", "LIFT_5CM", "HOLD_ELEVATED", "LOWER"])] = "RIGHT_HOLD_INTENT"
        intent[np.isin(stages, ["RELEASE", "POST_RELEASE"])] = "FINAL_RELEASE_INTENT"
        method = "b"
    atomic_npz(
        destination,
        commanded_q_rad=command.astype(np.float32),
        raw_policy_command=command.astype(np.float32),
        common_task_intent=intent,
        common_initial_q_rad=initial_q(command).astype(np.float32),
        stage=stages,
        joint_names=names,
        control_fps_hz=np.asarray(fps),
        stable_episode_id=np.asarray(stable_id),
        method=np.asarray(method),
        qualification_only=np.asarray(True),
        policy_used=np.asarray(False),
        runtime_right_three_digit_gate_required=np.asarray(False),
    )


def main() -> int:
    left_source = OUT / "contact_seeking_inputs/left/scripted_command.npz"
    right_source = OUT / "contact_seeking_inputs/right/scripted_command.npz"
    wrapped = OUT / "contact_seeking_commands"
    wrap(left_source, wrapped / "left_standalone_contact_seeking.npz", "left", "left_standalone_non_eval")
    wrap(right_source, wrapped / "right_standalone_contact_seeking.npz", "right", "right_standalone_non_eval")
    wrap(OLD_SCRIPTED, wrapped / "scripted_full_task_contact_seeking.npz", "right", "scripted_full_task_non_eval")

    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    scripted_registration = {
        "schema_version": "common_task_frame_registration_v1",
        "status": "QUALIFICATION_ONLY",
        "method_independent": True,
        "episode_independent": True,
        "changes_object_physics": False,
        "base_physics_config": str(CONFIG.resolve()),
        "base_physics_config_sha256": sha256_file(CONFIG),
        "registered_doll_center_world_xy_m": [0.11035161837935448, 0.053192950785160065],
        "registered_doll_orientation_quaternion_xyzw": [0.0, 0.0, 0.3179165045224807, 0.9481187141662206],
        "purpose": "policy-free task-relative scripted repeatability qualification only",
    }
    atomic_json(OUT / "SCRIPTED_QUALIFICATION_OBJECT_REGISTRATION.json", scripted_registration)

    dependencies = [
        CONFIG,
        REGISTRATION,
        ROOT / "tools/direct_physical_execution_layer.py",
        ROOT / "tools/direct_physical_execution_isaac_runtime.py",
        ROOT / "tools/common_execution_isaac_runtime.py",
        ROOT / "tools/run_direct_physical_execution_isaac.py",
        ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py",
        JOINT_CONTRACT,
        PHYSICAL_ENVIRONMENT,
        COMMON_CONTROLLER,
        OUT / "SCRIPTED_QUALIFICATION_OBJECT_REGISTRATION.json",
        wrapped / "left_standalone_contact_seeking.npz",
        wrapped / "right_standalone_contact_seeking.npz",
        wrapped / "scripted_full_task_contact_seeking.npz",
    ]
    freeze = {
        "schema_version": "episode_registered_physical_qualification_freeze_v1",
        "status": "PRE_EVAL35_QUALIFICATION_PROVISIONAL",
        "eval35_rollouts_allowed": False,
        "graspability_classifier_used": False,
        "selected_contact_candidate": "INTERMEDIATE_PLUSH_PROXY",
        "contact_model_selection_basis": (
            "smallest predeclared zero-tolerance proxy that passed the non-EVAL "
            "mechanical full-task trace: thumb-middle opposing normals, continuous "
            "hand support through transport, natural release, bin entry, and settle; "
            "the visual-envelope candidate caused premature receiving-thumb contact"
        ),
        "additional_contact_tolerance_mm": 0,
        "common_contact_seeking": {
            "enabled": True,
            "scope": "Dex3 digits only",
            "preload_max_rad": 0.025,
            "preload_bound_source": "frozen Dex3 effort_limit_sim / kp = 2.5 / 100",
            "method_specific": False,
            "arm_rescue": False,
            "wrist_rescue": False,
            "state_machine": [
                "OPEN", "PRESHAPE", "PROGRESSIVE_CLOSE", "GRASP_CONFIRM",
                "PRELOAD", "HOLD", "LIFT", "RELEASE",
            ],
            "preshape_permanent_latch": False,
            "transient_contact_may_unlatch": True,
            "mechanical_confirmation_required": True,
        },
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in dependencies
        ],
    }
    atomic_json(OUT / "PROVISIONAL_QUALIFICATION_FREEZE.json", freeze)
    print(json.dumps({
        "status": freeze["status"],
        "freeze_sha256": sha256_file(OUT / "PROVISIONAL_QUALIFICATION_FREEZE.json"),
        "commands": [str(path) for path in sorted(wrapped.glob("*.npz"))],
        "base_config_sha256": sha256_file(CONFIG),
        "selected_geometry": next(
            row for row in base["geometry_candidates"]
            if row["name"] == "INTERMEDIATE_PLUSH_PROXY"
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
