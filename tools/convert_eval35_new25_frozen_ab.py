#!/usr/bin/env python3
"""Retarget EVAL35 indices 10..34 with frozen A/B stacks after evaluator freeze."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.common_execution_layer import load_frozen_evaluator
from tools.convert_new_unseen_2_frozen_ab import (
    A_COMMON_INPUT,
    A_FROZEN_CONFIG,
    B_FROZEN_CONFIG,
    B_PARITY_MANIFEST,
    B_PARITY_REFERENCE,
    EXPECTED_A_PIPELINE_IMPLEMENTATION,
    EXPECTED_B_PIPELINE_IMPLEMENTATION,
    FEASIBILITY_CONFIG,
    atomic_json,
    final_action,
    inject_and_convert,
    load_npz,
    normalized_historical_source,
    patch_resolver_inputs,
    sha256_array,
    sha256_file,
)
from tools.run_final_common_execution_eval35 import (
    ENVIRONMENT_FREEZE,
    EVALUATOR_FREEZE,
    OUT,
    verify_eval35_identity,
    verify_hash_manifest,
)


OUTPUT = OUT / "eval35_preparation"
COMPLETED = OUTPUT / "EVAL35_RETARGETING_MANIFEST.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_eval35_source(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": row["source_name"],
        "source_root": row["source_root"],
        "parquet": row["source_parquet"],
        "frame_count": row["frames"],
        "fps": row["fps"],
        "duration_s": row["duration_s"],
        "cameras": row["cameras"],
    }


def validate_cache(path: Path, identity_sha256: str) -> None:
    manifest = read_json(path)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("evaluation_set") != "EVAL35"
        or manifest.get("eval35_identity_manifest_sha256") != identity_sha256
        or len(manifest.get("new_20260902_evaluation_only", [])) != 25
    ):
        raise RuntimeError("existing EVAL35 retargeting cache has the wrong identity")
    for row in manifest["new_20260902_evaluation_only"]:
        for method in ("a", "b"):
            artifact = Path(row[method]["evaluation_trajectory"])
            if sha256_file(artifact) != row[method]["evaluation_trajectory_sha256"]:
                raise RuntimeError(f"EVAL35 retargeting cache hash drift: {artifact}")


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()

    # This is deliberately the first workflow gate: no new source conversion is
    # allowed while the evaluator is missing, invalid, mutable, or hash-drifted.
    evaluator = load_frozen_evaluator(EVALUATOR_FREEZE)
    identity = verify_eval35_identity()
    identity_sha256 = sha256_file(OUT / "EVAL35_MANIFEST.json")
    verify_hash_manifest(ENVIRONMENT_FREEZE)
    completed = output / COMPLETED.name
    if completed.is_file():
        validate_cache(completed, identity_sha256)
        print(json.dumps({"status": "PASS", "cache_hit": str(completed)}, indent=2))
        return 0

    from tools.doll_handoff_retargeting.pipeline import DollHandoffPipeline

    a_runtime = output / "runtime_frozen_fair_a"
    b_runtime = output / "runtime_frozen_proposed_b"
    a_pipeline = DollHandoffPipeline(common_path=A_COMMON_INPUT, output_root=a_runtime)
    b_pipeline = DollHandoffPipeline(output_root=b_runtime)
    if a_pipeline.implementation_sha256 != EXPECTED_A_PIPELINE_IMPLEMENTATION:
        raise RuntimeError("Fair-A retargeting implementation differs from frozen identity")
    if b_pipeline.implementation_sha256 != EXPECTED_A_PIPELINE_IMPLEMENTATION:
        raise RuntimeError("Proposed-B runtime has an unknown implementation identity")
    a_pipeline.config_paths["common"] = A_FROZEN_CONFIG / "common_config.json"
    a_pipeline.config_paths["baseline"] = A_FROZEN_CONFIG / "baseline_config.json"
    b_pipeline.config_paths["common"] = B_FROZEN_CONFIG / "common_config.json"
    b_pipeline.config_paths["proposed"] = B_FROZEN_CONFIG / "proposed_config.json"

    parity_pipeline = DollHandoffPipeline(output_root=output / "proposed_b_runtime_parity_audit")
    parity_pipeline.config_paths["common"] = B_FROZEN_CONFIG / "common_config.json"
    parity_pipeline.config_paths["proposed"] = B_FROZEN_CONFIG / "proposed_config.json"
    parity_source = normalized_historical_source(read_json(B_PARITY_MANIFEST)["episodes"][0])
    parity_path, _ = inject_and_convert(parity_pipeline, parity_source, 50, "proposed")
    parity_current = load_npz(parity_path)
    parity_frozen = load_npz(B_PARITY_REFERENCE)
    parity_keys = (
        "g1_arm_qpos",
        "left_dex3_qpos",
        "right_dex3_qpos",
        "left_hand_phase",
        "right_hand_phase",
        "ownership_state",
        "target_left_wrist_position_model",
        "target_right_wrist_position_model",
        "target_left_wrist_rotation_model",
        "target_right_wrist_rotation_model",
        "target_left_interaction_frame_position_world",
        "target_right_interaction_frame_position_world",
        "event_frames",
        "replay_named_joint_qpos",
    )
    parity = {
        key: bool(np.array_equal(parity_current[key], parity_frozen[key]))
        for key in parity_keys
    }
    if not all(parity.values()):
        raise RuntimeError(
            "current Proposed-B runtime fails exact frozen numerical parity: "
            f"{[key for key, passed in parity.items() if not passed]}"
        )

    sources = [normalized_eval35_source(row) for row in identity["new_20260902_evaluation_only"]]
    a_initial: dict[int, Path] = {}
    b_initial: dict[int, Path] = {}
    records: list[dict[str, Any]] = []
    for offset, source in enumerate(sources):
        source_index = 50 + offset
        eval_index = 10 + offset
        a_path, a_record = inject_and_convert(a_pipeline, source, source_index, "baseline")
        b_path, b_record = inject_and_convert(b_pipeline, source, source_index, "proposed")
        identity_row = identity["new_20260902_evaluation_only"][offset]
        if (
            a_record["stable_episode_id"] != b_record["stable_episode_id"]
            or a_record["stable_episode_id"] != identity_row["stable_episode_id"]
        ):
            raise RuntimeError(f"A/B EVAL35 source identity mismatch at {eval_index}")
        a_initial[offset] = a_path
        b_initial[offset] = b_path
        records.append(
            {
                "eval_index": eval_index,
                "provenance": "POST_FREEZE_EVALUATION_ONLY",
                "source_name": source["source_name"],
                "stable_episode_id": a_record["stable_episode_id"],
                "frames": int(source["frame_count"]),
                "source_parquet": source["parquet"],
                "source_parquet_sha256": identity_row["source_parquet_sha256"],
                "source_cam_high": source["cameras"]["observation.images.cam_high"]["image_root"],
                "source_cam_high_tree_sha256": source["cameras"]["observation.images.cam_high"]["frame_tree_sha256"],
                "a_initial": a_record,
                "b_initial": b_record,
                "evaluation_only": True,
                "training_used": False,
                "checkpoint_selection_used": False,
                "evaluator_calibration_or_tuning_used": False,
                "controller_tuning_used": False,
            }
        )

    ids = {index: records[index]["stable_episode_id"] for index in range(25)}
    final_paths: dict[str, dict[int, Path]] = {"a": {}, "b": {}}

    patch_resolver_inputs(a_initial, ids)
    from tools.fair_a_full50_audit.full_pose_resolver import FullPoseCommonWristResolver

    a_resolver_root = output / "fair_a_final_resolver"
    a_resolver = FullPoseCommonWristResolver(
        config_path=FEASIBILITY_CONFIG, output_root=a_resolver_root
    )
    for index in range(25):
        a_resolver.solve_episode(index, export=True)
        final_paths["a"][index] = (
            a_resolver_root / "after_full_pose/trajectories" / f"{ids[index]}.npz"
        )

    patch_resolver_inputs(b_initial, ids)
    from tools.doll_handoff_feasibility.solver import GenericG1FeasibilityResolver

    b_resolver_root = output / "proposed_b_final_resolver"
    b_resolver = GenericG1FeasibilityResolver(
        config_path=FEASIBILITY_CONFIG, output_root=b_resolver_root
    )
    for index in range(25):
        b_resolver.solve_episode(index, export=True)
        final_paths["b"][index] = (
            b_resolver_root / "after/trajectories" / f"{ids[index]}.npz"
        )

    joint_names: list[str] | None = None
    for offset, record in enumerate(records):
        eval_index = 10 + offset
        for method in ("a", "b"):
            source_path = final_paths[method][offset]
            action, state, names = final_action(source_path)
            if joint_names is None:
                joint_names = names
            elif names != joint_names:
                raise RuntimeError("A/B final joint order mismatch")
            destination = output / method / f"eval_{eval_index:02d}_{ids[offset]}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.incomplete")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    action=action,
                    observation_state=state,
                    joint_names=np.asarray(names),
                    source_frame_index=np.arange(len(action), dtype=np.int64),
                    source_timestamp_seconds=np.arange(len(action), dtype=np.float64) / 30.0,
                    source_name=np.asarray(record["source_name"]),
                    stable_episode_id=np.asarray(ids[offset]),
                    provenance=np.asarray("POST_FREEZE_EVALUATION_ONLY"),
                    method=np.asarray(method),
                    evaluation_only=np.asarray(True),
                    training_used=np.asarray(False),
                    checkpoint_selection_used=np.asarray(False),
                    evaluator_calibration_or_tuning_used=np.asarray(False),
                    controller_tuning_used=np.asarray(False),
                )
            os.replace(temporary, destination)
            record[method] = {
                "final_retargeted_source": str(source_path.resolve()),
                "final_retargeted_source_sha256": sha256_file(source_path),
                "evaluation_trajectory": str(destination.resolve()),
                "evaluation_trajectory_sha256": sha256_file(destination),
                "action_sha256": sha256_array(action),
                "observation_state_sha256": sha256_array(state),
                "shape": list(action.shape),
            }

    manifest = {
        "schema_version": "eval35_new25_frozen_ab_retargeting_v1",
        "status": "PASS",
        "evaluation_set": "EVAL35",
        "eval35_identity_manifest": str((OUT / "EVAL35_MANIFEST.json").resolve()),
        "eval35_identity_manifest_sha256": identity_sha256,
        "frozen_evaluator_manifest": str(EVALUATOR_FREEZE.resolve()),
        "frozen_evaluator_manifest_sha256": evaluator.manifest_sha256,
        "frozen_evaluator_sha256": evaluator.evaluator_sha256,
        "environment_freeze": str(ENVIRONMENT_FREEZE.resolve()),
        "environment_freeze_sha256": sha256_file(ENVIRONMENT_FREEZE),
        "eval_entries": identity["eval_entries"],
        "new_20260902_evaluation_only": records,
        "joint_names": joint_names,
        "frozen_pipeline": {
            "frozen_proposed_b_retargeting_implementation_sha256": EXPECTED_B_PIPELINE_IMPLEMENTATION,
            "current_fair_a_retargeting_implementation_sha256": EXPECTED_A_PIPELINE_IMPLEMENTATION,
            "proposed_b_runtime_numerical_parity": {
                "status": "PASS_BYTE_IDENTICAL_NUMERICAL_ARRAYS",
                "reference_source": "GoPark_20260823_135848",
                "reference_trajectory": str(B_PARITY_REFERENCE.resolve()),
                "reference_trajectory_sha256": sha256_file(B_PARITY_REFERENCE),
                "checked_arrays": parity,
            },
            "fair_a_common_config": str((A_FROZEN_CONFIG / "common_config.json").resolve()),
            "fair_a_common_config_sha256": sha256_file(A_FROZEN_CONFIG / "common_config.json"),
            "fair_a_baseline_config": str((A_FROZEN_CONFIG / "baseline_config.json").resolve()),
            "fair_a_baseline_config_sha256": sha256_file(A_FROZEN_CONFIG / "baseline_config.json"),
            "proposed_b_common_config": str((B_FROZEN_CONFIG / "common_config.json").resolve()),
            "proposed_b_common_config_sha256": sha256_file(B_FROZEN_CONFIG / "common_config.json"),
            "proposed_b_config": str((B_FROZEN_CONFIG / "proposed_config.json").resolve()),
            "proposed_b_config_sha256": sha256_file(B_FROZEN_CONFIG / "proposed_config.json"),
            "generic_feasibility_config": str(FEASIBILITY_CONFIG.resolve()),
            "generic_feasibility_config_sha256": sha256_file(FEASIBILITY_CONFIG),
            "fair_a_final_adapter": "FullPoseCommonWristResolver",
            "proposed_b_final_adapter": "GenericG1FeasibilityResolver",
        },
        "new_20260902_count": 25,
        "evaluation_only": True,
        "evaluator_artifacts_modified": False,
        "training_used": False,
        "checkpoint_selection_used": False,
        "controller_or_environment_tuning_used": False,
        "evaluator_calibration_or_tuning_used": False,
        "success_criterion_tuning_used": False,
        "episode_replacement_allowed": False,
    }
    atomic_json(completed, manifest)
    (output / "EVAL35_RETARGETING_MANIFEST.md").write_text(
        "# EVAL35 evaluation-only frozen A/B conversion\n\n"
        "Status: **PASS**\n\n"
        "Only EVAL35 indices 10–34 were newly converted, after validation of the "
        "authoritative frozen evaluator. The prior EVAL10 remains unchanged. The "
        "25 recordings remain excluded from evaluator calibration/tuning, training, "
        "checkpoint selection, and controller/environment tuning.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "manifest": str(completed), "new_episodes": 25}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
