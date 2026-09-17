#!/usr/bin/env python3
"""Evaluation-only conversion of exact NEW_UNSEEN_25 with frozen A/B stacks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.convert_new_unseen_2_frozen_ab import (
    A_COMMON_INPUT,
    A_FROZEN_CONFIG,
    B_FROZEN_CONFIG,
    B_PARITY_MANIFEST,
    B_PARITY_REFERENCE,
    EXPECTED_A_PIPELINE_IMPLEMENTATION,
    FEASIBILITY_CONFIG,
    final_action,
    inject_and_convert,
    load_npz,
    normalized_historical_source,
    patch_resolver_inputs,
    sha256_array,
    sha256_file,
)


OUT = ROOT / "outputs/final_direct_physical_eval35/00_preparation"
IDENTITY = (
    ROOT
    / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json"
)
MANIFEST = OUT / "NEW_UNSEEN_25_FROZEN_AB_CONVERSION.json"
EXPECTED_NAMES = (
    "GoPark_20260902_110736", "GoPark_20260902_110912",
    "GoPark_20260902_111047", "GoPark_20260902_111205",
    "GoPark_20260902_111333", "GoPark_20260902_111447",
    "GoPark_20260902_111608", "GoPark_20260902_111735",
    "GoPark_20260902_111946", "GoPark_20260902_112117",
    "GoPark_20260902_112910", "GoPark_20260902_113051",
    "GoPark_20260902_113217", "GoPark_20260902_113336",
    "GoPark_20260902_113459", "GoPark_20260902_113617",
    "GoPark_20260902_113734", "GoPark_20260902_113921",
    "GoPark_20260902_114043", "GoPark_20260902_114201",
    "GoPark_20260902_114310", "GoPark_20260902_114629",
    "GoPark_20260902_114802", "GoPark_20260902_115001",
    "GoPark_20260902_115218",
)


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


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": row["source_name"],
        "source_root": row["source_root"],
        "parquet": row["source_parquet"],
        "frame_count": row["frames"],
        "fps": row["fps"],
        "duration_s": row["duration_s"],
        "cameras": {
            key: {"image_root": value["image_root"]}
            for key, value in row["cameras"].items()
        },
    }


def main() -> int:
    if MANIFEST.is_file():
        cached = read_json(MANIFEST)
        if cached.get("status") != "PASS" or cached.get("source_count") != 25:
            raise RuntimeError("existing NEW25 conversion is incomplete")
        for row in cached["records"]:
            for method in ("a", "b"):
                path = Path(row[method]["evaluation_trajectory"])
                if sha256_file(path) != row[method]["evaluation_trajectory_sha256"]:
                    raise RuntimeError(f"NEW25 conversion hash drift: {path}")
        print(json.dumps({"status": "PASS", "cache_hit": str(MANIFEST)}, indent=2))
        return 0

    identity = read_json(IDENTITY)
    rows = identity.get("new_20260902_evaluation_only", [])
    if identity.get("status") != "PASS_IDENTITY_FROZEN" or len(rows) != 25:
        raise RuntimeError("exact EVAL35 identity is unavailable")
    if tuple(row["source_name"] for row in rows) != EXPECTED_NAMES:
        raise RuntimeError("NEW_UNSEEN_25 names/order differ from user-declared set")
    if any(row.get("status") != "PASS" for row in rows):
        raise RuntimeError("NEW_UNSEEN_25 source integrity is not PASS")

    from tools.doll_handoff_retargeting.pipeline import DollHandoffPipeline

    a_runtime = OUT / "runtime_frozen_fair_a"
    b_runtime = OUT / "runtime_frozen_proposed_b"
    a_pipeline = DollHandoffPipeline(common_path=A_COMMON_INPUT, output_root=a_runtime)
    b_pipeline = DollHandoffPipeline(output_root=b_runtime)
    if a_pipeline.implementation_sha256 != EXPECTED_A_PIPELINE_IMPLEMENTATION:
        raise RuntimeError("Fair-A frozen runtime identity drift")
    if b_pipeline.implementation_sha256 != EXPECTED_A_PIPELINE_IMPLEMENTATION:
        raise RuntimeError("Proposed-B runtime identity drift")
    a_pipeline.config_paths["common"] = A_FROZEN_CONFIG / "common_config.json"
    a_pipeline.config_paths["baseline"] = A_FROZEN_CONFIG / "baseline_config.json"
    b_pipeline.config_paths["common"] = B_FROZEN_CONFIG / "common_config.json"
    b_pipeline.config_paths["proposed"] = B_FROZEN_CONFIG / "proposed_config.json"

    # Re-prove numerical parity of the current loader with a persisted frozen B
    # conversion before touching any NEW25 source.
    parity = DollHandoffPipeline(output_root=OUT / "proposed_b_runtime_parity")
    parity.config_paths["common"] = B_FROZEN_CONFIG / "common_config.json"
    parity.config_paths["proposed"] = B_FROZEN_CONFIG / "proposed_config.json"
    parity_source = normalized_historical_source(read_json(B_PARITY_MANIFEST)["episodes"][0])
    parity_path, _ = inject_and_convert(parity, parity_source, 50, "proposed")
    current = load_npz(parity_path)
    frozen = load_npz(B_PARITY_REFERENCE)
    parity_keys = (
        "g1_arm_qpos", "left_dex3_qpos", "right_dex3_qpos",
        "left_hand_phase", "right_hand_phase", "ownership_state",
        "target_left_wrist_position_model", "target_right_wrist_position_model",
        "target_left_wrist_rotation_model", "target_right_wrist_rotation_model",
        "target_left_interaction_frame_position_world",
        "target_right_interaction_frame_position_world", "event_frames",
        "replay_named_joint_qpos",
    )
    parity_result = {key: bool(np.array_equal(current[key], frozen[key])) for key in parity_keys}
    if not all(parity_result.values()):
        raise RuntimeError(f"frozen B conversion parity failed: {parity_result}")

    a_initial: dict[int, Path] = {}
    b_initial: dict[int, Path] = {}
    records: list[dict[str, Any]] = []
    for offset, identity_row in enumerate(rows):
        source = normalized(identity_row)
        source_index = 50 + offset
        a_path, a_record = inject_and_convert(a_pipeline, source, source_index, "baseline")
        b_path, b_record = inject_and_convert(b_pipeline, source, source_index, "proposed")
        if a_record["stable_episode_id"] != b_record["stable_episode_id"]:
            raise RuntimeError("A/B converted source identity mismatch")
        if not a_record["source_semantic_valid"] or not b_record["source_semantic_valid"]:
            raise RuntimeError(
                f"task-incomplete NEW25 source: {identity_row['source_name']}: "
                f"A={a_record['source_anomalies']} B={b_record['source_anomalies']}"
            )
        a_initial[offset] = a_path
        b_initial[offset] = b_path
        records.append(
            {
                "eval_index": 10 + offset,
                "provenance": "NEW_UNSEEN_25",
                "source_name": identity_row["source_name"],
                "stable_episode_id": a_record["stable_episode_id"],
                "frames": int(identity_row["frames"]),
                "source_parquet": identity_row["source_parquet"],
                "source_parquet_sha256": identity_row["source_parquet_sha256"],
                "source_cam_high": identity_row["cameras"]["observation.images.cam_high"]["image_root"],
                "source_cam_high_tree_sha256": identity_row["cameras"]["observation.images.cam_high"]["frame_tree_sha256"],
                "a_initial": a_record,
                "b_initial": b_record,
            }
        )

    ids = {index: records[index]["stable_episode_id"] for index in range(25)}
    final_paths: dict[str, dict[int, Path]] = {"a": {}, "b": {}}
    patch_resolver_inputs(a_initial, ids)
    from tools.fair_a_full50_audit.full_pose_resolver import FullPoseCommonWristResolver
    a_root = OUT / "fair_a_final_resolver"
    a_resolver = FullPoseCommonWristResolver(config_path=FEASIBILITY_CONFIG, output_root=a_root)
    for index in range(25):
        a_resolver.solve_episode(index, export=True)
        final_paths["a"][index] = a_root / "after_full_pose/trajectories" / f"{ids[index]}.npz"

    patch_resolver_inputs(b_initial, ids)
    from tools.doll_handoff_feasibility.solver import GenericG1FeasibilityResolver
    b_root = OUT / "proposed_b_final_resolver"
    b_resolver = GenericG1FeasibilityResolver(config_path=FEASIBILITY_CONFIG, output_root=b_root)
    for index in range(25):
        b_resolver.solve_episode(index, export=True)
        final_paths["b"][index] = b_root / "after/trajectories" / f"{ids[index]}.npz"

    joint_names: list[str] | None = None
    for index, record in enumerate(records):
        for method in ("a", "b"):
            path = final_paths[method][index]
            action, state, names = final_action(path)
            if joint_names is None:
                joint_names = names
            elif names != joint_names:
                raise RuntimeError("NEW25 A/B joint order mismatch")
            destination = OUT / method / f"eval_{10 + index:02d}_{ids[index]}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.incomplete")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, action=action, observation_state=state,
                    joint_names=np.asarray(names),
                    source_frame_index=np.arange(len(action), dtype=np.int64),
                    source_timestamp_seconds=np.arange(len(action), dtype=np.float64) / 30.0,
                    source_name=np.asarray(record["source_name"]),
                    stable_episode_id=np.asarray(ids[index]),
                    provenance=np.asarray("NEW_UNSEEN_25"), method=np.asarray(method),
                    training_used=np.asarray(False), checkpoint_selection_used=np.asarray(False),
                    controller_tuning_used=np.asarray(False),
                )
            os.replace(temporary, destination)
            record[method] = {
                "final_retargeted_source": str(path.resolve()),
                "final_retargeted_source_sha256": sha256_file(path),
                "evaluation_trajectory": str(destination.resolve()),
                "evaluation_trajectory_sha256": sha256_file(destination),
                "action_sha256": sha256_array(action),
                "observation_state_sha256": sha256_array(state),
                "shape": list(action.shape),
            }

    value = {
        "schema_version": "new_unseen_25_frozen_ab_conversion_v1",
        "status": "PASS", "source_count": 25,
        "eval_indices": list(range(10, 35)), "records": records,
        "joint_names": joint_names, "frozen_runtime_parity": parity_result,
        "fair_a_common_config_sha256": sha256_file(A_FROZEN_CONFIG / "common_config.json"),
        "fair_a_baseline_config_sha256": sha256_file(A_FROZEN_CONFIG / "baseline_config.json"),
        "proposed_b_common_config_sha256": sha256_file(B_FROZEN_CONFIG / "common_config.json"),
        "proposed_b_config_sha256": sha256_file(B_FROZEN_CONFIG / "proposed_config.json"),
        "feasibility_config_sha256": sha256_file(FEASIBILITY_CONFIG),
        "training_used": False, "checkpoint_selection_used": False,
        "controller_tuning_used": False, "outcome_based_selection_used": False,
        "episode_replacement_allowed": False,
    }
    atomic_json(MANIFEST, value)
    print(json.dumps({"status": "PASS", "sources": 25, "manifest": str(MANIFEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
