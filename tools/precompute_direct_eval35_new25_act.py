#!/usr/bin/env python3
"""Precompute frozen ACT-E1 predictions for exact NEW_UNSEEN_25."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.precompute_contact_eval10_act import (
    CHUNK, COEFFICIENT, DIM, EXECUTION, EXPECTED, EXPERIMENT2, FPS,
    PROJECTION, atomic_json, png_frames, read_json, require, sha256_array,
    sha256_file,
)


BASE = ROOT / "outputs/final_direct_physical_eval35/00_preparation"
CONVERSION = BASE / "NEW_UNSEEN_25_FROZEN_AB_CONVERSION.json"
OUT = BASE / "act_trajectories"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    args = parser.parse_args()
    method = args.method
    output_root = OUT / f"act_{method}40"
    require(EXPERIMENT2, EXPECTED["experiment2"], "Experiment-2")
    require(EXECUTION, EXPECTED["execution"], "ACT execution")
    require(PROJECTION, EXPECTED["projection"], "deployment projection")
    conversion = read_json(CONVERSION)
    if conversion.get("status") != "PASS" or conversion.get("source_count") != 25:
        raise RuntimeError("NEW25 frozen conversion is unavailable")
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
        raise RuntimeError("checkpoint violates frozen ACT contract")
    config = copy.deepcopy(base.config)
    config.n_action_steps = 1
    config.temporal_ensemble_coeff = COEFFICIENT
    policy = ACTPolicy(config)
    loaded = policy.load_state_dict(base.state_dict(), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict checkpoint reload failed: {loaded}")
    policy.to(next(base.parameters()).device)
    del base
    policy.eval()
    if not isinstance(policy.temporal_ensembler, ACTTemporalEnsembler):
        raise RuntimeError("official temporal ensembler missing")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    projector = NamedJointDeploymentSafetyProjector.from_path(PROJECTION)

    summaries = []
    for offset, row in enumerate(conversion["records"]):
        eval_index = 10 + offset
        frames = int(row["frames"])
        destination = output_root / f"eval_{eval_index:02d}_{row['stable_episode_id']}"
        trajectory = destination / "full_trajectory.npz"
        cache = destination / "cache_manifest.json"
        if cache.is_file() and trajectory.is_file():
            value = read_json(cache)
            if value.get("status") != "PASS" or value.get("trajectory_sha256") != sha256_file(trajectory):
                raise RuntimeError(f"invalid ACT cache: {destination}")
            summaries.append(value)
            print(f"[CACHE] ACT-{method.upper()}40 NEW25 {offset + 1}/25", flush=True)
            continue
        state_path = Path(row[method]["evaluation_trajectory"])
        require(state_path, row[method]["evaluation_trajectory_sha256"], "NEW25 state")
        with np.load(state_path, allow_pickle=False) as payload:
            states = np.asarray(payload["observation_state"], dtype=np.float32)
        if states.shape != (frames, DIM) or not np.isfinite(states).all():
            raise RuntimeError(f"invalid NEW25 state at {eval_index}")
        image_root = Path(row["source_cam_high"])
        rgb_iterator = png_frames(image_root, frames)
        policy.reset()
        normalized_chunks, raw_chunks = [], []
        normalized_actions, raw_actions, rgb_hashes, durations = [], [], [], []
        started = time.monotonic()
        for frame, rgb in enumerate(rgb_iterator):
            image = (
                torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
                .contiguous().float().div_(255.0)
            )
            processed = preprocessor({
                "observation.images.cam_high": image,
                "observation.state": torch.from_numpy(states[frame].copy()),
            })
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
        raw = np.asarray(raw_actions, dtype=np.float32)
        projection = projector.project(raw, global_row_offset=0)
        hard = projection.hard_limit_projected_action.astype(np.float32)
        safe = projection.deployment_safe_action.astype(np.float32)
        destination.mkdir(parents=True, exist_ok=True)
        temporary = trajectory.with_suffix(".npz.incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                raw_act_chunks=np.asarray(raw_chunks, dtype=np.float32),
                normalized_act_chunks=np.asarray(normalized_chunks, dtype=np.float32),
                raw_temporal_ensemble_action=raw,
                normalized_temporal_ensemble_action=np.asarray(normalized_actions, dtype=np.float32),
                hard_limit_projected_action=hard,
                deployment_safe_action=safe,
                method_specific_reference_state=states,
                source_frame_index=np.arange(frames, dtype=np.int64),
                source_timestamp_seconds=np.arange(frames, dtype=np.float64) / FPS,
                source_rgb_sha256=np.asarray(rgb_hashes), joint_names=np.asarray(projector.names),
                control_fps_hz=np.asarray(FPS), temporal_ensemble_coeff=np.asarray(COEFFICIENT),
                method=np.asarray(method), eval_index=np.asarray(eval_index),
                checkpoint_model_sha256=np.asarray(checkpoint_sha),
                source_rgb_identity_sha256=np.asarray(row["source_cam_high_tree_sha256"]),
            )
        os.replace(temporary, trajectory)
        value = {
            "schema_version": "direct_eval35_new25_act_e1_v1", "status": "PASS",
            "method": f"ACT-{method.upper()}40", "eval_index": eval_index,
            "provenance": "NEW_UNSEEN_25", "stable_episode_id": row["stable_episode_id"],
            "frames": frames,
            "policy_input": "original ALOHA cam_high RGB[t] + frozen method-specific retargeted observation.state[t]",
            "checkpoint": str(checkpoint), "checkpoint_step": int(selection["selected_step"]),
            "checkpoint_model_sha256": checkpoint_sha,
            "trajectory": str(trajectory.resolve()), "trajectory_sha256": sha256_file(trajectory),
            "array_sha256": {"raw_policy_command": sha256_array(raw), "deployment_safe_action": sha256_array(safe), "method_specific_reference_state": sha256_array(states)},
            "inference": {"calls": frames, "mean_seconds": float(np.mean(durations)), "maximum_seconds": float(np.max(durations)), "wall_seconds": time.monotonic() - started, "eval_mode": True},
            "training_used": False, "checkpoint_selection_changed": False,
            "controller_tuning_used": False, "simulator_used": False,
        }
        atomic_json(cache, value)
        summaries.append(value)
        print(f"[READY] ACT-{method.upper()}40 NEW25 {offset + 1}/25", flush=True)
    atomic_json(output_root / "BATCH_MANIFEST.json", {
        "schema_version": "direct_eval35_new25_act_batch_v1", "status": "PASS",
        "method": f"ACT-{method.upper()}40", "episodes": 25,
        "checkpoint": str(checkpoint), "checkpoint_model_sha256": checkpoint_sha,
        "entries": [{key: row[key] for key in ("eval_index", "stable_episode_id", "provenance", "frames", "trajectory", "trajectory_sha256")} for row in summaries],
    })
    print(json.dumps({"status": "PASS", "method": method, "episodes": 25}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
