#!/usr/bin/env python3
"""Run frozen ACT-A40/B40 inference for EVAL35 indices 10..34 after freeze."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_execution_layer import load_frozen_evaluator
from tools.precompute_contact_eval10_act import (
    CHUNK,
    COEFFICIENT,
    DIM,
    EXECUTION,
    EXPECTED,
    EXPERIMENT2,
    FPS,
    PROJECTION,
    atomic_json,
    png_frames,
    read_json,
    require,
    sha256_array,
    sha256_file,
)
from tools.run_final_common_execution_eval35 import (
    EVALUATOR_FREEZE,
    OUT,
    verify_eval35_identity,
)


RETARGET = OUT / "eval35_preparation/EVAL35_RETARGETING_MANIFEST.json"
OUTPUT = OUT / "eval35_preparation/act_trajectories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method = args.method
    output_root = args.output_root.resolve() / f"act_{method}40"

    # Evaluation-only inputs must remain completely downstream of evaluator freeze.
    evaluator = load_frozen_evaluator(EVALUATOR_FREEZE)
    identity = verify_eval35_identity()
    identity_sha256 = sha256_file(OUT / "EVAL35_MANIFEST.json")
    require(EXPERIMENT2, EXPECTED["experiment2"], "Experiment-2")
    require(EXECUTION, EXPECTED["execution"], "ACT execution")
    require(PROJECTION, EXPECTED["projection"], "deployment projection")
    retarget = read_json(RETARGET)
    if (
        retarget.get("status") != "PASS"
        or retarget.get("evaluation_set") != "EVAL35"
        or retarget.get("eval35_identity_manifest_sha256") != identity_sha256
        or retarget.get("frozen_evaluator_sha256") != evaluator.evaluator_sha256
    ):
        raise RuntimeError("EVAL35 frozen retargeting prerequisite is unavailable")
    rows = {
        int(row["eval_index"]): row
        for row in retarget.get("new_20260902_evaluation_only", [])
    }
    if set(rows) != set(range(10, 35)):
        raise RuntimeError("EVAL35 retargeting does not contain exact indices 10..34")

    experiment = read_json(EXPERIMENT2)
    selection = experiment["methods"][method]["checkpoint_selection"]
    checkpoint = Path(selection["selected_checkpoint"]).resolve()
    checkpoint_sha = str(selection["selected_model_sha256"])
    require(checkpoint / "model.safetensors", checkpoint_sha, "selected ACT checkpoint")

    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors
    from tools.common_deployment_safety_projection import NamedJointDeploymentSafetyProjector

    base = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    if (
        base.config.chunk_size != CHUNK
        or base.config.n_action_steps != CHUNK
        or base.config.temporal_ensemble_coeff is not None
        or base.config.action_feature.shape != (DIM,)
    ):
        raise RuntimeError("checkpoint violates frozen raw ACT contract")
    config = copy.deepcopy(base.config)
    config.n_action_steps = 1
    config.temporal_ensemble_coeff = COEFFICIENT
    policy = ACTPolicy(config)
    loaded = policy.load_state_dict(base.state_dict(), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict reload failed: {loaded}")
    policy.to(next(base.parameters()).device)
    del base
    policy.eval()
    if not isinstance(policy.temporal_ensembler, ACTTemporalEnsembler):
        raise RuntimeError("official ACTTemporalEnsembler was not created")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    projector = NamedJointDeploymentSafetyProjector.from_path(PROJECTION)

    summaries: list[dict[str, Any]] = []
    for ordinal, eval_index in enumerate(range(10, 35), start=1):
        entry = identity["eval_entries"][eval_index]
        row = rows[eval_index]
        if row["stable_episode_id"] != entry["stable_episode_id"]:
            raise RuntimeError(f"EVAL35 retarget/source identity mismatch at {eval_index}")
        frames = int(entry["frames"])
        destination = output_root / f"eval_{eval_index:02d}_{entry['stable_episode_id']}"
        trajectory_path = destination / "full_trajectory.npz"
        manifest_path = destination / "cache_manifest.json"
        if manifest_path.is_file() and trajectory_path.is_file():
            cached = read_json(manifest_path)
            if (
                cached.get("status") != "PASS"
                or cached.get("evaluation_only") is not True
                or cached.get("frozen_evaluator_sha256") != evaluator.evaluator_sha256
                or cached.get("checkpoint_model_sha256") != checkpoint_sha
                or cached.get("trajectory_sha256") != sha256_file(trajectory_path)
            ):
                raise RuntimeError(f"invalid existing EVAL35 ACT cache: {destination}")
            summaries.append(cached)
            print(
                f"[CACHE] ACT-{method.upper()}40 EVAL35-new25 {ordinal}/25",
                flush=True,
            )
            continue

        state_path = Path(row[method]["evaluation_trajectory"])
        require(
            state_path,
            row[method]["evaluation_trajectory_sha256"],
            "EVAL35 evaluation-only retargeted state",
        )
        with np.load(state_path, allow_pickle=False) as payload:
            states = np.asarray(payload["observation_state"], dtype=np.float32)
        source_path = Path(row["source_cam_high"]).resolve()
        rgb_iterator = png_frames(source_path, frames)
        rgb_identity = row["source_cam_high_tree_sha256"]
        if states.shape != (frames, DIM) or not np.isfinite(states).all():
            raise RuntimeError(f"invalid method-specific state EVAL35 index {eval_index}")

        policy.reset()
        normalized_chunks = []
        raw_chunks = []
        normalized_actions = []
        raw_actions = []
        rgb_hashes = []
        durations = []
        started = time.monotonic()
        for frame, rgb in enumerate(rgb_iterator):
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"invalid RGB at {eval_index}:{frame}")
            image = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .contiguous()
                .float()
                .div_(255.0)
            )
            processed = preprocessor(
                {
                    "observation.images.cam_high": image,
                    "observation.state": torch.from_numpy(states[frame].copy()),
                }
            )
            begin = time.perf_counter()
            with torch.inference_mode():
                normalized_chunk = policy.predict_action_chunk(processed)
                normalized_action = policy.temporal_ensembler.update(normalized_chunk)
                physical_chunk = postprocessor(normalized_chunk.clone())
                physical_action = postprocessor(normalized_action.clone())
            durations.append(time.perf_counter() - begin)
            normalized_chunks.append(normalized_chunk[0].detach().float().cpu().numpy())
            raw_chunks.append(physical_chunk[0].detach().float().cpu().numpy())
            normalized_actions.append(normalized_action[0].detach().float().cpu().numpy())
            raw_actions.append(physical_action[0].detach().float().cpu().numpy())
            rgb_hashes.append(sha256_array(rgb))
        if len(raw_actions) != frames:
            raise RuntimeError("source RGB iterator frame count mismatch")
        normalized_chunks_array = np.asarray(normalized_chunks, dtype=np.float32)
        raw_chunks_array = np.asarray(raw_chunks, dtype=np.float32)
        normalized_actions_array = np.asarray(normalized_actions, dtype=np.float32)
        raw_actions_array = np.asarray(raw_actions, dtype=np.float32)
        projection = projector.project(raw_actions_array, global_row_offset=0)
        hard_actions = projection.hard_limit_projected_action.astype(np.float32)
        deployment_actions = projection.deployment_safe_action.astype(np.float32)

        destination.mkdir(parents=True, exist_ok=True)
        temporary = trajectory_path.with_suffix(".npz.incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                raw_act_chunks=raw_chunks_array,
                normalized_act_chunks=normalized_chunks_array,
                raw_temporal_ensemble_action=raw_actions_array,
                normalized_temporal_ensemble_action=normalized_actions_array,
                hard_limit_projected_action=hard_actions,
                deployment_safe_action=deployment_actions,
                method_specific_reference_state=states,
                source_frame_index=np.arange(frames, dtype=np.int64),
                source_timestamp_seconds=np.arange(frames, dtype=np.float64) / FPS,
                source_rgb_sha256=np.asarray(rgb_hashes),
                joint_names=np.asarray(projector.names),
                control_fps_hz=np.asarray(FPS),
                temporal_ensemble_coeff=np.asarray(COEFFICIENT),
                method=np.asarray(method),
                eval_index=np.asarray(eval_index),
                checkpoint_model_sha256=np.asarray(checkpoint_sha),
                source_rgb_identity_sha256=np.asarray(rgb_identity),
                evaluation_only=np.asarray(True),
            )
        os.replace(temporary, trajectory_path)
        manifest = {
            "schema_version": "eval35_new25_act_e1_v1",
            "status": "PASS",
            "evaluation_set": "EVAL35",
            "method": f"ACT-{method.upper()}40",
            "eval_index": eval_index,
            "provenance": "POST_FREEZE_EVALUATION_ONLY",
            "stable_episode_id": entry["stable_episode_id"],
            "frames": frames,
            "policy_input": "original ALOHA cam_high RGB[t] + frozen method-specific retargeted observation.state[t]",
            "source_rgb": str(source_path),
            "source_rgb_storage": "original frame-aligned cam_high PNG tree",
            "source_rgb_identity_sha256": rgb_identity,
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(selection["selected_step"]),
            "checkpoint_model_sha256": checkpoint_sha,
            "frozen_evaluator_sha256": evaluator.evaluator_sha256,
            "eval35_identity_manifest_sha256": identity_sha256,
            "official_execution": {
                "policy": "ACTPolicy",
                "temporal_ensembler": "ACTTemporalEnsembler",
                "coefficient": COEFFICIENT,
                "chunk_size": CHUNK,
                "n_action_steps": 1,
                "custom_smoothing": False,
            },
            "trajectory": str(trajectory_path.resolve()),
            "trajectory_sha256": sha256_file(trajectory_path),
            "array_sha256": {
                "raw_policy_command": sha256_array(raw_actions_array),
                "deployment_safe_action": sha256_array(deployment_actions),
                "method_specific_reference_state": sha256_array(states),
            },
            "inference": {
                "calls": frames,
                "mean_seconds": float(np.mean(durations)),
                "maximum_seconds": float(np.max(durations)),
                "wall_seconds": time.monotonic() - started,
                "eval_mode": True,
            },
            "evaluation_only": True,
            "training_used": False,
            "checkpoint_selection_changed": False,
            "evaluation_data_used_for_checkpoint_selection": False,
            "evaluator_calibration_or_tuning_used": False,
            "controller_tuning_used": False,
            "simulator_used": False,
        }
        atomic_json(manifest_path, manifest)
        summaries.append(manifest)
        print(
            f"[READY] ACT-{method.upper()}40 EVAL35-new25 {ordinal}/25",
            flush=True,
        )

    summary = {
        "schema_version": "eval35_new25_act_batch_v1",
        "status": "PASS",
        "evaluation_set": "EVAL35",
        "method": f"ACT-{method.upper()}40",
        "eval_indices": list(range(10, 35)),
        "episodes": 25,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": checkpoint_sha,
        "frozen_evaluator_sha256": evaluator.evaluator_sha256,
        "eval35_identity_manifest_sha256": identity_sha256,
        "evaluation_only": True,
        "training_used": False,
        "checkpoint_selection_changed": False,
        "entries": [
            {
                "eval_index": row["eval_index"],
                "stable_episode_id": row["stable_episode_id"],
                "provenance": row["provenance"],
                "frames": row["frames"],
                "trajectory": row["trajectory"],
                "trajectory_sha256": row["trajectory_sha256"],
            }
            for row in summaries
        ],
    }
    atomic_json(output_root / "BATCH_NEW25_MANIFEST.json", summary)
    print(json.dumps({"status": "PASS", "method": method, "episodes": 25}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
