#!/usr/bin/env python3
"""Assemble immutable ACT-A40/B40 EVAL35 commands after all freeze gates pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_execution_layer import load_frozen_evaluator
from tools.common_execution_isaac_runtime import verify_execution_freeze
from tools.prepare_contact_eval10_physical_commands import semantic_stages
from tools.run_final_common_execution_eval35 import (
    COMMAND_MANIFEST,
    EVALUATOR_FREEZE,
    EXECUTION_FREEZE,
    OUT,
    verify_eval35_identity,
)
from tools.common_execution_layer import sha256_file


BASE_COMMANDS = (
    ROOT / "outputs/final_contact_constrained_eval/05_act_ab_results/PHYSICAL_COMMAND_MANIFEST.json"
)
RETARGET = OUT / "eval35_preparation/EVAL35_RETARGETING_MANIFEST.json"
ACT_ROOT = OUT / "eval35_preparation/act_trajectories"
COMMAND_ROOT = OUT / "eval35_preparation/commands"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=COMMAND_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_base_commands(identity: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = read_json(BASE_COMMANDS)
    records = manifest.get("records", [])
    expected = {
        (f"ACT-{method}40", index)
        for method in ("A", "B")
        for index in range(10)
    }
    actual = {(str(row["method"]), int(row["eval_index"])) for row in records}
    if actual != expected:
        raise RuntimeError("base command manifest is not exact ACT-A40/B40 x prior EVAL10")
    stable_ids = {
        int(row["eval_index"]): row["stable_episode_id"]
        for row in identity["eval_entries"][:10]
    }
    for row in records:
        index = int(row["eval_index"])
        if row["stable_episode_id"] != stable_ids[index]:
            raise RuntimeError(f"base EVAL10 command identity mismatch at {index}")
        for path_key, sha_key in (
            ("physical_command", "physical_command_sha256"),
            ("semantic_reference", "semantic_reference_sha256"),
            ("raw_act_trajectory", "raw_act_trajectory_sha256"),
        ):
            path = Path(row[path_key])
            if sha256_file(path) != row[sha_key]:
                raise RuntimeError(f"base EVAL10 command hash drift: {path}")
    return [dict(row) for row in records]


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()

    evaluator = load_frozen_evaluator(EVALUATOR_FREEZE)
    verify_execution_freeze(EXECUTION_FREEZE, evaluator.evaluator_sha256)
    identity = verify_eval35_identity()
    identity_sha256 = sha256_file(OUT / "EVAL35_MANIFEST.json")
    records = validate_base_commands(identity)
    retarget = read_json(RETARGET)
    if (
        retarget.get("status") != "PASS"
        or retarget.get("frozen_evaluator_sha256") != evaluator.evaluator_sha256
        or retarget.get("eval35_identity_manifest_sha256") != identity_sha256
    ):
        raise RuntimeError("EVAL35 retargeting manifest does not match frozen prerequisites")
    retarget_rows = {
        int(row["eval_index"]): row
        for row in retarget.get("new_20260902_evaluation_only", [])
    }
    if set(retarget_rows) != set(range(10, 35)):
        raise RuntimeError("retargeting manifest lacks exact EVAL35 indices 10..34")

    for method in ("a", "b"):
        batch_path = ACT_ROOT / f"act_{method}40/BATCH_NEW25_MANIFEST.json"
        batch = read_json(batch_path)
        if (
            batch.get("status") != "PASS"
            or batch.get("evaluation_only") is not True
            or batch.get("episodes") != 25
            or batch.get("frozen_evaluator_sha256") != evaluator.evaluator_sha256
            or batch.get("eval35_identity_manifest_sha256") != identity_sha256
        ):
            raise RuntimeError(f"invalid EVAL35 ACT batch: {batch_path}")
        batch_rows = {int(row["eval_index"]): row for row in batch["entries"]}
        if set(batch_rows) != set(range(10, 35)):
            raise RuntimeError(f"ACT-{method.upper()} batch lacks exact indices 10..34")
        for eval_index in range(10, 35):
            entry = identity["eval_entries"][eval_index]
            act_entry = batch_rows[eval_index]
            retarget_row = retarget_rows[eval_index]
            if not (
                entry["stable_episode_id"]
                == act_entry["stable_episode_id"]
                == retarget_row["stable_episode_id"]
            ):
                raise RuntimeError(f"EVAL35 identity mismatch at {eval_index}")
            act_path = Path(act_entry["trajectory"])
            if sha256_file(act_path) != act_entry["trajectory_sha256"]:
                raise RuntimeError(f"ACT cache hash drift: {act_path}")
            with np.load(act_path, allow_pickle=False) as archive:
                raw = np.asarray(archive["raw_temporal_ensemble_action"], dtype=np.float32)
                executed = np.asarray(archive["deployment_safe_action"], dtype=np.float32)
                names = archive["joint_names"].astype(str)
            frames = len(executed)
            if raw.shape != executed.shape or executed.shape != (int(entry["frames"]), 28):
                raise RuntimeError(f"invalid ACT command shape at {method}:{eval_index}")
            reference = Path(retarget_row["b"]["final_retargeted_source"])
            if sha256_file(reference) != retarget_row["b"]["final_retargeted_source_sha256"]:
                raise RuntimeError(f"semantic reference hash drift: {reference}")
            stages = semantic_stages(reference, frames)
            destination = output / f"act_{method}40/eval_{eval_index:02d}_{entry['stable_episode_id']}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.incomplete")
            override = np.zeros_like(executed, dtype=bool)
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    commanded_q_rad=executed,
                    stage=stages,
                    joint_names=names,
                    control_fps_hz=np.asarray(30.0),
                    raw_policy_command=raw,
                    executed_common_controller_command=executed,
                    common_controller_override_mask=override,
                    runtime_right_three_digit_gate_required=np.asarray(True),
                    method=np.asarray(method),
                    eval_index=np.asarray(eval_index),
                    provenance=np.asarray("POST_FREEZE_EVALUATION_ONLY"),
                    stable_episode_id=np.asarray(entry["stable_episode_id"]),
                    semantic_reference=np.asarray(str(reference.resolve())),
                    direct_policy_arm_wrist_execution=np.asarray(True),
                    common_runtime_dex3_realization_enabled=np.asarray(True),
                    evaluation_only=np.asarray(True),
                )
            os.replace(temporary, destination)
            records.append(
                {
                    "method": f"ACT-{method.upper()}40",
                    "eval_index": eval_index,
                    "provenance": "POST_FREEZE_EVALUATION_ONLY",
                    "stable_episode_id": entry["stable_episode_id"],
                    "frames": frames,
                    "raw_act_trajectory": str(act_path.resolve()),
                    "raw_act_trajectory_sha256": sha256_file(act_path),
                    "physical_command": str(destination.resolve()),
                    "physical_command_sha256": sha256_file(destination),
                    "semantic_reference": str(reference.resolve()),
                    "semantic_reference_sha256": sha256_file(reference),
                    "pre_runtime_common_override_scalar_count": 0,
                    "arm_intervention_fraction": 0.0,
                    "wrist_intervention_fraction": 0.0,
                    "dex3_runtime_intervention_allowed": True,
                    "raw_to_deployment_safe_rmse_rad": float(
                        np.sqrt(np.mean((raw - executed) ** 2))
                    ),
                    "deployment_safety_projection_is_frozen_common": True,
                    "evaluation_only": True,
                    "training_used": False,
                    "checkpoint_selection_changed": False,
                    "evaluator_calibration_or_tuning_used": False,
                    "controller_tuning_used": False,
                }
            )

    records.sort(key=lambda row: (str(row["method"]), int(row["eval_index"])))
    expected = {
        (f"ACT-{method}40", index)
        for method in ("A", "B")
        for index in range(35)
    }
    if {(row["method"], int(row["eval_index"])) for row in records} != expected:
        raise RuntimeError("refusing to write partial EVAL35 command manifest")
    manifest = {
        "schema_version": "tsr_preserving_eval35_physical_command_v1",
        "status": "PASS",
        "evaluation_set": "EVAL35",
        "episodes_per_method": 35,
        "command_count": 70,
        "construction": "unchanged prior EVAL10 commands plus evaluation-only new indices 10..34",
        "eval35_identity_manifest": str((OUT / "EVAL35_MANIFEST.json").resolve()),
        "eval35_identity_manifest_sha256": identity_sha256,
        "frozen_evaluator_manifest": str(EVALUATOR_FREEZE.resolve()),
        "frozen_evaluator_manifest_sha256": evaluator.manifest_sha256,
        "frozen_evaluator_sha256": evaluator.evaluator_sha256,
        "execution_layer_freeze_manifest": str(EXECUTION_FREEZE.resolve()),
        "execution_layer_freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
        "method_specific_arm_wrist_policy_behavior_preserved": True,
        "common_runtime_override_scope": "DEX3_ONLY_AFTER_PHYSICAL_GATES",
        "semantic_labels_affect_arm_or_wrist_commands": False,
        "identical_execution_path_for_a_b": True,
        "per_episode_or_method_tuning": False,
        "new_20260902_evaluation_only": True,
        "new_20260902_used_for_training": False,
        "new_20260902_used_for_checkpoint_selection": False,
        "new_20260902_used_for_evaluator_calibration_or_tuning": False,
        "new_20260902_used_for_controller_tuning": False,
        "records": records,
    }
    atomic_json(COMMAND_MANIFEST, manifest)
    print(
        json.dumps(
            {"status": "PASS", "commands": len(records), "manifest": str(COMMAND_MANIFEST)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
