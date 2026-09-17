#!/usr/bin/env python3
"""Convert the fixed NEW_UNSEEN_2 with the already-frozen Fair-A/B pipelines.

This program is evaluation-only.  It appends the two user-declared recordings to
fresh, isolated runtime copies of the frozen retargeting stacks.  It never edits
the training datasets, checkpoints, original HELDOUT8, or frozen environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation"
NEW_MANIFEST = (
    ROOT
    / "outputs/final_contact_constrained_eval/01_new_unseen_2_integrity"
    / "NEW_UNSEEN_2_MANIFEST.json"
)
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
A_COMMON_INPUT = (
    ROOT
    / "outputs/dataset_a_final50_retargeting/input_contract"
    / "common_config.final_common_50.json"
)
A_FROZEN_CONFIG = ROOT / "outputs/dataset_a_final50_retargeting/config"
B_FROZEN_CONFIG = (
    ROOT
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
    / "frozen_approval/config"
)
FEASIBILITY_CONFIG = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
ENV_FREEZE = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FREEZE_MANIFEST.json"
EXPECTED_B_PIPELINE_IMPLEMENTATION = (
    "36fd0c2dfda0a5b0b0ed8b63a88f5d311d7267eaa1479d2d70b97dd8dd0f093a"
)
EXPECTED_A_PIPELINE_IMPLEMENTATION = (
    "9352499d8116554a969765b3c3e1b7b86c95cf163317c496e7034fb5b18137e0"
)
B_PARITY_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/new_source_audit/new_episode_manifest.json"
B_PARITY_REFERENCE = (
    ROOT
    / "outputs/doll_handoff_dataset_b_final/new_episode_conversion"
    / "frozen_proposed_b_runtime/proposed/trajectories/doll_handoff_20260823_135848.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_id(source_name: str) -> str:
    return "new_unseen_" + source_name.removeprefix("GoPark_")


def source_episode(source: Mapping[str, Any], episode_index: int):
    from tools.doll_handoff_retargeting.source import (
        SourceEpisode,
        SourceRecord,
        fixed_list_numpy,
        scalar_numpy,
    )

    root = Path(str(source["source_root"])).resolve()
    parquet = Path(str(source["parquet"])).resolve()
    cameras = source["cameras"]
    record = SourceRecord(
        episode_index=episode_index,
        stable_episode_id=stable_id(str(source["source_name"])),
        source_name=str(source["source_name"]),
        root=root,
        parquet=parquet,
        frame_count=int(source["frame_count"]),
        fps=float(source["fps"]),
        duration_sec=float(source["duration_s"]),
        camera_keys=tuple(sorted(cameras)),
        image_directories={
            key: Path(str(value["image_root"])).resolve()
            for key, value in cameras.items()
        },
        valid=True,
        problems=(),
    )
    table = pq.read_table(
        parquet,
        columns=[
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "task_index",
        ],
    )
    episode = SourceEpisode(
        record=record,
        action=fixed_list_numpy(table["action"], 14).astype(np.float64),
        state=fixed_list_numpy(table["observation.state"], 14).astype(np.float64),
        timestamps=scalar_numpy(table["timestamp"], np.float64),
        frame_index=scalar_numpy(table["frame_index"], np.int64),
        task_index=scalar_numpy(table["task_index"], np.int64),
    )
    return record, episode


def normalized_historical_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the prior frozen replacement audit to the NEW2 audit schema."""
    return {
        "source_name": source["source_name"],
        "source_root": source["source_root"],
        "parquet": source["parquet_path"],
        "frame_count": source["frame_count"],
        "fps": source["fps"],
        "duration_s": source["duration_s"],
        "cameras": {
            key: {"image_root": value["image_directory"]}
            for key, value in source["camera_assets"].items()
        },
    }


def inject_and_convert(
    pipeline: Any,
    source: Mapping[str, Any],
    episode_index: int,
    method: str,
) -> tuple[Path, dict[str, Any]]:
    record, episode = source_episode(source, episode_index)
    if len(pipeline.sources.records) != episode_index:
        raise RuntimeError(
            f"append-only source index mismatch: {len(pipeline.sources.records)} != {episode_index}"
        )
    pipeline.sources.records.append(record)
    pipeline.sources._cache[episode_index] = episode
    fk = pipeline.aloha.fk(episode.state)
    event = pipeline.event_auditor.detect_episode(episode, fk)
    pipeline.event_auditor.fk_cache[episode_index] = fk
    pipeline.event_auditor.events[episode_index] = event
    result = pipeline.convert(method, episode_index)
    paths = pipeline.export(result)
    return paths["trajectory"], {
        "source_name": record.source_name,
        "stable_episode_id": record.stable_episode_id,
        "frames": record.frame_count,
        "source_semantic_valid": bool(event.source_semantic_valid),
        "source_anomalies": list(event.anomalies),
        "event_frames": {
            key: None if value is None else int(value)
            for key, value in event.frames.items()
        },
        "initial_converter_status": result.validation["status"],
        "initial_trajectory": str(paths["trajectory"].resolve()),
        "initial_trajectory_sha256": sha256_file(paths["trajectory"]),
        "initial_metrics": str(paths["metrics"].resolve()),
        "initial_validation": str(paths["validation"].resolve()),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def patch_resolver_inputs(paths: Mapping[int, Path], ids: Mapping[int, str]) -> None:
    import tools.doll_handoff_feasibility.solver as solver_module
    import tools.fair_a_full50_audit.common as fair_common
    import tools.fair_a_full50_audit.full_pose_resolver as full_pose_module
    import tools.fair_a_full50_audit.wrist_resolver as wrist_module

    def loader(index: int) -> dict[str, np.ndarray]:
        return load_npz(paths[int(index)])

    def path_for(index: int) -> Path:
        return paths[int(index)]

    def id_for(index: int) -> str:
        return ids[int(index)]

    solver_module.load_trajectory = loader
    solver_module.stable_episode_id = id_for
    fair_common.load_a_trajectory = loader
    fair_common.a_trajectory_path = path_for
    fair_common.stable_episode_id = id_for
    wrist_module.load_a_trajectory = loader
    wrist_module.a_trajectory_path = path_for
    wrist_module.stable_episode_id = id_for
    full_pose_module.load_a_trajectory = loader


def final_action(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = load_npz(path)
    names = values["replay_joint_names"].astype(str).tolist()
    action = values["replay_named_joint_qpos"].astype(np.float32)
    if action.ndim != 2 or action.shape[1] != 28 or len(names) != 28:
        raise RuntimeError(f"invalid final 28D trajectory: {path}")
    if not np.isfinite(action).all():
        raise RuntimeError(f"non-finite final trajectory: {path}")
    state = np.empty_like(action)
    state[0] = action[0]
    state[1:] = action[:-1]
    return action, state, names


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    completed = output / "EVAL10_RETARGETING_MANIFEST.json"
    if completed.is_file():
        manifest = read_json(completed)
        if manifest.get("status") != "PASS":
            raise RuntimeError("existing conversion manifest is not PASS")
        for method in ("a", "b"):
            for row in manifest["new_unseen_2"]:
                artifact = Path(row[method]["evaluation_trajectory"])
                if sha256_file(artifact) != row[method]["evaluation_trajectory_sha256"]:
                    raise RuntimeError(f"completed conversion hash drift: {artifact}")
        print(json.dumps({"status": "PASS", "cache_hit": str(completed)}, indent=2))
        return 0

    environment = read_json(ENV_FREEZE)
    if environment.get("status") != "FROZEN":
        raise RuntimeError("contact-constrained environment is not frozen")
    new_manifest = read_json(NEW_MANIFEST)
    if new_manifest.get("status") != "PASS" or new_manifest.get("source_count") != 2:
        raise RuntimeError("NEW_UNSEEN_2 integrity gate is not PASS")
    heldout = read_json(HELDOUT)
    if heldout.get("status") != "PASS" or heldout.get("episode_count") != 8:
        raise RuntimeError("authoritative HELDOUT8 is not PASS")

    from tools.doll_handoff_retargeting.pipeline import DollHandoffPipeline

    a_runtime = output / "runtime_frozen_fair_a"
    b_runtime = output / "runtime_frozen_proposed_b"
    a_pipeline = DollHandoffPipeline(common_path=A_COMMON_INPUT, output_root=a_runtime)
    b_pipeline = DollHandoffPipeline(output_root=b_runtime)
    if a_pipeline.implementation_sha256 != EXPECTED_A_PIPELINE_IMPLEMENTATION:
        raise RuntimeError("Fair-A retargeting implementation differs from its frozen identity")
    # Two post-freeze, selection-manifest-only additions changed pipeline.py/source.py
    # fingerprints.  Before accepting this runtime for Proposed-B, prove that it
    # reproduces the already-persisted frozen conversion of a prior unseen source.
    if b_pipeline.implementation_sha256 != EXPECTED_A_PIPELINE_IMPLEMENTATION:
        raise RuntimeError("current retargeting runtime has an unknown implementation identity")
    a_pipeline.config_paths["common"] = A_FROZEN_CONFIG / "common_config.json"
    a_pipeline.config_paths["baseline"] = A_FROZEN_CONFIG / "baseline_config.json"
    b_pipeline.config_paths["common"] = B_FROZEN_CONFIG / "common_config.json"
    b_pipeline.config_paths["proposed"] = B_FROZEN_CONFIG / "proposed_config.json"

    parity_pipeline = DollHandoffPipeline(
        output_root=output / "proposed_b_runtime_parity_audit"
    )
    parity_pipeline.config_paths["common"] = B_FROZEN_CONFIG / "common_config.json"
    parity_pipeline.config_paths["proposed"] = B_FROZEN_CONFIG / "proposed_config.json"
    parity_source = normalized_historical_source(read_json(B_PARITY_MANIFEST)["episodes"][0])
    parity_path, _ = inject_and_convert(parity_pipeline, parity_source, 50, "proposed")
    parity_current = load_npz(parity_path)
    parity_frozen = load_npz(B_PARITY_REFERENCE)
    parity_keys = [
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
    ]
    parity = {
        key: bool(np.array_equal(parity_current[key], parity_frozen[key]))
        for key in parity_keys
    }
    if not all(parity.values()):
        raise RuntimeError(
            "current Proposed-B runtime fails exact frozen numerical parity: "
            f"{[key for key, passed in parity.items() if not passed]}"
        )

    sources = list(new_manifest["sources"])
    a_initial: dict[int, Path] = {}
    b_initial: dict[int, Path] = {}
    records: list[dict[str, Any]] = []
    for offset, source in enumerate(sources):
        index = 50 + offset
        a_path, a_record = inject_and_convert(a_pipeline, source, index, "baseline")
        b_path, b_record = inject_and_convert(b_pipeline, source, index, "proposed")
        if a_record["stable_episode_id"] != b_record["stable_episode_id"]:
            raise RuntimeError("A/B source identity mismatch")
        a_initial[offset] = a_path
        b_initial[offset] = b_path
        records.append(
            {
                "eval_index": 8 + offset,
                "provenance": "POST_TRAINING_UNSEEN",
                "source_name": source["source_name"],
                "stable_episode_id": a_record["stable_episode_id"],
                "frames": int(source["frame_count"]),
                "source_parquet": source["parquet"],
                "source_parquet_sha256": source["parquet_sha256"],
                "source_cam_high": source["cameras"]["observation.images.cam_high"]["image_root"],
                "source_cam_high_tree_sha256": source["cameras"]["observation.images.cam_high"]["frame_tree_sha256"],
                "a_initial": a_record,
                "b_initial": b_record,
            }
        )

    ids = {index: records[index]["stable_episode_id"] for index in range(2)}
    final_paths: dict[str, dict[int, Path]] = {"a": {}, "b": {}}

    # Final Fair-A is the frozen full-pose, representation-neutral wrist adapter.
    patch_resolver_inputs(a_initial, ids)
    from tools.fair_a_full50_audit.full_pose_resolver import FullPoseCommonWristResolver

    a_resolver_root = output / "fair_a_final_resolver"
    a_resolver = FullPoseCommonWristResolver(
        config_path=FEASIBILITY_CONFIG, output_root=a_resolver_root
    )
    for index in range(2):
        a_resolver.solve_episode(index, export=True)
        final_paths["a"][index] = (
            a_resolver_root
            / "after_full_pose/trajectories"
            / f"{ids[index]}.npz"
        )

    # Final Proposed-B uses the unchanged frozen generic feasibility resolver.
    patch_resolver_inputs(b_initial, ids)
    from tools.doll_handoff_feasibility.solver import GenericG1FeasibilityResolver

    b_resolver_root = output / "proposed_b_final_resolver"
    b_resolver = GenericG1FeasibilityResolver(
        config_path=FEASIBILITY_CONFIG, output_root=b_resolver_root
    )
    for index in range(2):
        b_resolver.solve_episode(index, export=True)
        final_paths["b"][index] = (
            b_resolver_root / "after/trajectories" / f"{ids[index]}.npz"
        )

    joint_names: list[str] | None = None
    for index, record in enumerate(records):
        for method in ("a", "b"):
            path = final_paths[method][index]
            action, state, names = final_action(path)
            if joint_names is None:
                joint_names = names
            elif names != joint_names:
                raise RuntimeError("A/B final joint order mismatch")
            destination = output / method / f"eval_{8 + index:02d}_{ids[index]}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.with_suffix(".npz.incomplete").open("wb") as stream:
                np.savez_compressed(
                    stream,
                    action=action,
                    observation_state=state,
                    joint_names=np.asarray(names),
                    source_frame_index=np.arange(len(action), dtype=np.int64),
                    source_timestamp_seconds=np.arange(len(action), dtype=np.float64) / 30.0,
                    source_name=np.asarray(record["source_name"]),
                    stable_episode_id=np.asarray(ids[index]),
                    provenance=np.asarray("POST_TRAINING_UNSEEN"),
                    method=np.asarray(method),
                    training_used=np.asarray(False),
                    checkpoint_selection_used=np.asarray(False),
                    controller_tuning_used=np.asarray(False),
                )
            os.replace(destination.with_suffix(".npz.incomplete"), destination)
            record[method] = {
                "final_retargeted_source": str(path.resolve()),
                "final_retargeted_source_sha256": sha256_file(path),
                "evaluation_trajectory": str(destination.resolve()),
                "evaluation_trajectory_sha256": sha256_file(destination),
                "action_sha256": sha256_array(action),
                "observation_state_sha256": sha256_array(state),
                "shape": list(action.shape),
            }

    eval_entries = []
    for index, entry in enumerate(heldout["entries"]):
        eval_entries.append(
            {
                "eval_index": index,
                "provenance": "PREDEFINED_HELDOUT",
                "source_final_episode": int(entry["final_dataset_index"]),
                "stable_episode_id": entry["stable_episode_id"],
                "frames": int(entry["frames"]),
                "heldout_output_episode": index,
            }
        )
    eval_entries.extend(
        {
            "eval_index": row["eval_index"],
            "provenance": row["provenance"],
            "source_name": row["source_name"],
            "stable_episode_id": row["stable_episode_id"],
            "frames": row["frames"],
        }
        for row in records
    )
    manifest = {
        "schema_version": "contact_constrained_eval10_retargeting_v1",
        "status": "PASS",
        "evaluation_set": "EVAL10",
        "construction": "authoritative HELDOUT8 + user-declared exact NEW_UNSEEN_2",
        "original_split_remains": "HELDOUT8",
        "environment_frozen_before_conversion": True,
        "environment_freeze": str(ENV_FREEZE.resolve()),
        "environment_freeze_sha256": sha256_file(ENV_FREEZE),
        "heldout8_manifest": str(HELDOUT.resolve()),
        "heldout8_manifest_sha256": sha256_file(HELDOUT),
        "new_unseen_2_integrity_manifest": str(NEW_MANIFEST.resolve()),
        "new_unseen_2_integrity_manifest_sha256": sha256_file(NEW_MANIFEST),
        "eval_entries": eval_entries,
        "new_unseen_2": records,
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
                "interpretation": (
                    "The post-freeze source-selection support changed source/pipeline "
                    "file fingerprints but reproduces every conversion-relevant frozen "
                    "array byte-identically on the persisted prior unseen source."
                ),
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
        "training_used": False,
        "checkpoint_selection_used": False,
        "controller_or_bin_tuning_used": False,
        "success_criterion_tuning_used": False,
        "episode_replacement_allowed": False,
    }
    atomic_json(completed, manifest)
    (output / "EVAL10_RETARGETING_MANIFEST.md").write_text(
        "# EVAL10 frozen A/B conversion\n\n"
        "Status: **PASS**\n\n"
        "The original HELDOUT8 remains unchanged. The two exact post-training "
        "recordings were appended after the contact-constrained environment freeze. "
        "Neither source was used for training, checkpoint selection, controller/bin "
        "calibration, success-criterion tuning, or parameter search. No episode may "
        "be replaced based on its physical result.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "manifest": str(completed), "episodes": 10}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
