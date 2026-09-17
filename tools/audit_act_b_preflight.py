#!/usr/bin/env python3
"""Immutable Dataset-B and installed-LeRobot ACT compatibility preflight.

This script reads Dataset B and the frozen retargeted trajectory archives, writes
only beneath ``outputs/policy_b_act``, and constructs the exact one-run ACT
training configuration.  It deliberately uses LeRobot's ACT policy, dataset,
normalization, and checkpoint configuration classes rather than reimplementing
any of those components.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import default_collate

from lerobot import __version__ as lerobot_version
from lerobot.configs import FeatureType
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.utils.feature_utils import dataset_to_policy_features


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DATASET_ROOT = ROOT / "datasets/doll_handoff_proposed_b_50"
OUTPUT_ROOT = ROOT / "outputs/policy_b_act"
AUDIT_DIR = OUTPUT_ROOT / "audit"
CONFIG_DIR = OUTPUT_ROOT / "config"
TRAIN_DIR = OUTPUT_ROOT / "train"
LEROBOT_ROOT = Path("/home/jbnu/lerobot-smolvla")
REPO_ID = "local/doll_handoff_proposed_b_50"
PARQUET = DATASET_ROOT / "data/chunk-000/file-000.parquet"
SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 34_478
EXPECTED_DIM = 28
CHUNK_SIZE = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray, dtype: np.dtype | None = None) -> str:
    canonical = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    canonical = np.ascontiguousarray(canonical)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_text(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def parquet_fixed_list(table: Any, key: str, width: int) -> np.ndarray:
    values = table[key].combine_chunks().values.to_numpy(zero_copy_only=False)
    return np.asarray(values).reshape(-1, width)


def dataset_identity() -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        PARQUET,
        columns=["action", "observation.state", "episode_index", "frame_index", "index", "timestamp"],
    )
    action = parquet_fixed_list(table, "action", EXPECTED_DIM).astype(np.float32, copy=False)
    state = parquet_fixed_list(table, "observation.state", EXPECTED_DIM).astype(np.float32, copy=False)
    episode = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    frame = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)

    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    entries = sorted(source_manifest["episodes"], key=lambda row: int(row["final_dataset_index"]))
    trajectory_comparisons: list[dict[str, Any]] = []
    lag1_exact_by_episode: list[bool] = []
    for ep, entry in enumerate(entries):
        mask = episode == ep
        episode_action = action[mask]
        episode_state = state[mask]
        archive_path = Path(entry["retargeted_trajectory_path"])
        with np.load(archive_path, allow_pickle=False) as archive:
            frozen_action = np.asarray(archive["replay_named_joint_qpos"], dtype=np.float32)
        action_exact = bool(np.array_equal(episode_action, frozen_action))
        lag1_exact = bool(
            len(episode_action) == len(episode_state)
            and np.array_equal(episode_state[0], episode_action[0])
            and np.array_equal(episode_state[1:], episode_action[:-1])
        )
        lag1_exact_by_episode.append(lag1_exact)
        trajectory_comparisons.append(
            {
                "episode_index": ep,
                "frames": int(mask.sum()),
                "dataset_action_sha256": sha256_array(episode_action, np.float32),
                "frozen_trajectory_action_sha256": sha256_array(frozen_action, np.float32),
                "action_array_exact_equal": action_exact,
                "state_lag1_exact": lag1_exact,
                "frozen_trajectory_file_sha256": sha256_file(archive_path),
                "manifest_trajectory_file_sha256": entry["retargeted_trajectory_sha256"],
            }
        )

    files = []
    tree_digest = hashlib.sha256()
    for path in sorted(p for p in DATASET_ROOT.rglob("*") if p.is_file()):
        relative = path.relative_to(DATASET_ROOT).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": digest})
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")

    unique_episodes = np.unique(episode)
    episode_boundaries_correct = True
    for ep in unique_episodes:
        episode_frames = frame[episode == ep]
        episode_boundaries_correct &= bool(np.array_equal(episode_frames, np.arange(len(episode_frames))))

    payload = {
        "schema_version": "act_b_dataset_identity_v1",
        "dataset_root": str(DATASET_ROOT),
        "dataset_tree_sha256": tree_digest.hexdigest(),
        "file_count": len(files),
        "files": files,
        "episodes": int(len(unique_episodes)),
        "frames": int(len(action)),
        "fps": 30,
        "action_shape": list(action.shape),
        "state_shape": list(state.shape),
        "action_dtype": str(action.dtype),
        "state_dtype": str(state.dtype),
        "action_array_sha256": sha256_array(action, np.float32),
        "state_array_sha256": sha256_array(state, np.float32),
        "episode_index_sha256": sha256_array(episode, np.int64),
        "frame_index_sha256": sha256_array(frame, np.int64),
        "all_action_arrays_exactly_equal_frozen_trajectories": all(
            row["action_array_exact_equal"] for row in trajectory_comparisons
        ),
        "all_state_arrays_exact_lag1_from_frozen_actions": all(lag1_exact_by_episode),
        "episode_boundaries_correct": episode_boundaries_correct,
        "finite_action": bool(np.isfinite(action).all()),
        "finite_state": bool(np.isfinite(state).all()),
        "trajectory_comparisons": trajectory_comparisons,
        "required_declarations": {
            "ACTION_ARRAYS_UNCHANGED": "YES"
            if all(row["action_array_exact_equal"] for row in trajectory_comparisons)
            else "NO",
            "STATE_ARRAYS_UNCHANGED": "YES" if all(lag1_exact_by_episode) else "NO",
            "EPISODE_COUNT": int(len(unique_episodes)),
            "FRAME_COUNT": int(len(action)),
        },
    }
    checks = [
        payload["episodes"] == EXPECTED_EPISODES,
        payload["frames"] == EXPECTED_FRAMES,
        payload["action_shape"] == [EXPECTED_FRAMES, EXPECTED_DIM],
        payload["state_shape"] == [EXPECTED_FRAMES, EXPECTED_DIM],
        payload["all_action_arrays_exactly_equal_frozen_trajectories"],
        payload["all_state_arrays_exact_lag1_from_frozen_actions"],
        payload["episode_boundaries_correct"],
        payload["finite_action"],
        payload["finite_state"],
    ]
    payload["status"] = "PASS" if all(checks) else "FAIL"
    atomic_json(AUDIT_DIR / "dataset_identity_before_training.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("Dataset-B immutable identity checks failed")
    return payload, action, state, episode, frame


def act_source_audit(config: ACTConfig) -> dict[str, Any]:
    commit = run_text("git", "rev-parse", "HEAD", cwd=LEROBOT_ROOT)
    source_paths = {
        "configuration": Path(inspect.getfile(ACTConfig)),
        "model": Path(inspect.getfile(ACTPolicy)),
        "processor": LEROBOT_ROOT / "src/lerobot/policies/act/processor_act.py",
    }
    source_records = {}
    all_match_head = True
    for label, path in source_paths.items():
        relative = path.relative_to(LEROBOT_ROOT).as_posix()
        working_hash = sha256_file(path)
        head_bytes = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=LEROBOT_ROOT)
        head_hash = hashlib.sha256(head_bytes).hexdigest()
        source_records[label] = {
            "path": str(path),
            "working_sha256": working_hash,
            "head_sha256": head_hash,
            "matches_commit": working_hash == head_hash,
        }
        all_match_head &= working_hash == head_hash

    temporal_signature = inspect.signature(ACTTemporalEnsembler.__init__)
    payload = {
        "schema_version": "act_b_installed_act_implementation_v1",
        "lerobot_version": lerobot_version,
        "lerobot_commit": commit,
        "lerobot_checkout": str(LEROBOT_ROOT),
        "act_sources": source_records,
        "all_act_sources_match_commit": all_match_head,
        "configuration_class": f"{ACTConfig.__module__}.{ACTConfig.__name__}",
        "model_class": f"{ACTPolicy.__module__}.{ACTPolicy.__name__}",
        "temporal_ensembler_class": f"{ACTTemporalEnsembler.__module__}.{ACTTemporalEnsembler.__name__}",
        "temporal_ensembler_signature": str(temporal_signature),
        "expected_image_keys": sorted(config.image_features),
        "expected_state": {
            "key": "observation.state",
            "logical_shape": [EXPECTED_DIM],
            "batched_shape": ["B", EXPECTED_DIM],
            "semantics": "RETARGETED_G1_STATE_SURROGATE: q_target[0] at frame 0, otherwise q_target[t-1]",
        },
        "action_dimension_handling": {
            "key": "action",
            "logical_dimension": EXPECTED_DIM,
            "internal_padding": False,
            "regression_head_out_features": EXPECTED_DIM,
            "raw_output_shape": ["B", CHUNK_SIZE, EXPECTED_DIM],
        },
        "chunk_semantics": {
            "chunk_size": config.chunk_size,
            "chunk_size_meaning": "number of future action targets predicted per policy invocation",
            "n_action_steps": config.n_action_steps,
            "n_action_steps_meaning": "non-ensemble queue prefix consumed before the next policy invocation",
            "duration_seconds_at_30_hz": config.chunk_size / 30.0,
            "concepts_silently_equated": False,
        },
        "temporal_ensemble": {
            "supported": True,
            "training_config_enabled": config.temporal_ensemble_coeff is not None,
            "reference_coefficient": 0.01,
            "requires_n_action_steps": 1,
            "implementation": "online exponential temporal ensemble over overlapping raw chunks; positive coefficient favors older predictions",
        },
        "normalization_mapping": {
            str(key): value.value if hasattr(value, "value") else str(value)
            for key, value in config.normalization_mapping.items()
        },
        "checkpoint_behavior": {
            "save": "ACTPolicy.save_pretrained writes config.json and model.safetensors; training checkpoints also save pre/post processors, optimizer, scheduler, RNG, and step state",
            "reload": "ACTPolicy.from_pretrained reconstructs ACTConfig, loads safetensors, moves to config.device, and sets eval mode",
            "dropout_on_reload": "disabled because reload returns eval mode",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "status": "PASS" if all_match_head else "FAIL",
    }
    atomic_json(AUDIT_DIR / "installed_act_implementation.json", payload)
    if not all_match_head:
        raise RuntimeError("Installed ACT sources do not match the recorded LeRobot commit")
    return payload


def make_config(meta: LeRobotDatasetMetadata) -> tuple[ACTConfig, TrainPipelineConfig, Path]:
    features = dataset_to_policy_features(meta.features)
    input_features = {key: value for key, value in features.items() if value.type is not FeatureType.ACTION}
    output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    policy = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        device="cuda",
        use_amp=False,
        push_to_hub=False,
        repo_id=None,
    )
    train = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=REPO_ID,
            root=str(DATASET_ROOT),
            episodes=None,
            use_imagenet_stats=True,
            video_backend="torchcodec",
            return_uint8=False,
            streaming=False,
            eval_split=0.0,
        ),
        policy=policy,
        output_dir=TRAIN_DIR,
        job_name="policy_b_act_100k",
        batch_size=8,
        steps=100_000,
        save_checkpoint=True,
        save_freq=20_000,
        env_eval_freq=0,
        eval_steps=0,
        use_policy_training_preset=True,
        wandb=WandBConfig(enable=False),
    )
    train.validate()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / "train_config.json"
    train._save_pretrained(CONFIG_DIR)
    atomic_json(
        CONFIG_DIR / "config_provenance.json",
        {
            "schema_version": "act_b_train_config_provenance_v1",
            "created_before_training": True,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "base": "installed official LeRobot ACT defaults",
            "controlled_changes": {
                "input": "observation.images.cam_high RGB + observation.state 28D",
                "output": "action 28D absolute joint-position targets",
                "chunk_size": CHUNK_SIZE,
                "n_action_steps_nonensemble": CHUNK_SIZE,
                "batch_size": 8,
                "device": "cuda",
                "use_amp": False,
                "output_dir": str(TRAIN_DIR),
            },
            "unchanged_recommended_training_duration_steps": 100_000,
            "all_50_episodes": True,
            "language_encoder": False,
            "task_metadata_retained_for_provenance": True,
            "task_metadata_used_as_controlled_input": False,
        },
    )
    return policy, train, config_path


def compatibility_audit(
    meta: LeRobotDatasetMetadata,
    config: ACTConfig,
    action: np.ndarray,
    episode: np.ndarray,
) -> dict[str, Any]:
    delta_timestamps = {"action": [index / meta.fps for index in config.action_delta_indices]}
    delta_timestamps.update({key: [0.0] for key in config.image_features})
    dataset = LeRobotDataset(
        REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
    )
    episode_ends = [int(np.flatnonzero(episode == ep)[-1]) for ep in range(EXPECTED_EPISODES)]
    episode_starts = [int(np.flatnonzero(episode == ep)[0]) for ep in range(EXPECTED_EPISODES)]
    rng = random.Random(20260826)
    random_indices = rng.sample(range(len(dataset)), 8)
    audit_indices = sorted(set(random_indices + episode_starts[:3] + episode_ends[:3] + [len(dataset) - 1]))

    reads = []
    for index in audit_indices:
        item = dataset[index]
        pad_count = int(item["action_is_pad"].sum().item())
        first_action_exact = bool(np.array_equal(item["action"][0].numpy(), action[index]))
        expected_valid = min(CHUNK_SIZE, int(np.flatnonzero(episode == int(episode[index]))[-1]) - index + 1)
        expected_pad = CHUNK_SIZE - expected_valid
        reads.append(
            {
                "global_index": index,
                "episode_index": int(item["episode_index"]),
                "frame_index": int(item["frame_index"]),
                "image_shape": list(item["observation.images.cam_high"].shape),
                "image_dtype": str(item["observation.images.cam_high"].dtype),
                "state_shape": list(item["observation.state"].shape),
                "state_dtype": str(item["observation.state"].dtype),
                "action_shape": list(item["action"].shape),
                "action_dtype": str(item["action"].dtype),
                "action_is_pad_shape": list(item["action_is_pad"].shape),
                "pad_count": pad_count,
                "expected_pad_count": expected_pad,
                "padding_mask_correct": pad_count == expected_pad,
                "first_action_exact_dataset_row": first_action_exact,
            }
        )

    batch_indices = [104, 844, 1535, 2220, 2910, 3600, 4290, 4980]
    raw_batch = default_collate([dataset[index] for index in batch_indices])
    raw_action = raw_batch["action"].clone()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=meta.stats)
    processed = preprocessor(raw_batch)
    roundtrip = postprocessor(processed["action"].clone())
    roundtrip_error = torch.abs(roundtrip - raw_action).max().item()

    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    torch.cuda.reset_peak_memory_stats()
    policy = ACTPolicy(config).to(config.device).train()
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, loss_parts = policy(processed)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - start
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in policy.parameters()
    )
    policy.eval()
    observation_batch = {
        key: value for key, value in processed.items() if key in config.input_features or key.endswith("_is_pad")
    }
    predicted_normalized = policy.predict_action_chunk(observation_batch)
    predicted_physical = postprocessor(predicted_normalized)
    dropout_modules = [module for module in policy.modules() if isinstance(module, torch.nn.Dropout)]

    payload = {
        "schema_version": "act_b_dataset_compatibility_v1",
        "dataset_length": len(dataset),
        "dataset_num_frames": dataset.num_frames,
        "dataset_num_episodes": dataset.num_episodes,
        "fps": meta.fps,
        "delta_timestamps": delta_timestamps,
        "random_and_boundary_reads": reads,
        "all_read_shapes_correct": all(
            row["image_shape"] == [3, 480, 640]
            and row["state_shape"] == [EXPECTED_DIM]
            and row["action_shape"] == [CHUNK_SIZE, EXPECTED_DIM]
            and row["action_is_pad_shape"] == [CHUNK_SIZE]
            for row in reads
        ),
        "all_first_actions_exact": all(row["first_action_exact_dataset_row"] for row in reads),
        "all_padding_masks_correct": all(row["padding_mask_correct"] for row in reads),
        "collated_shapes": {
            key: list(value.shape) for key, value in raw_batch.items() if isinstance(value, torch.Tensor)
        },
        "processed_shapes": {
            key: list(value.shape) for key, value in processed.items() if isinstance(value, torch.Tensor)
        },
        "normalization": {
            "mapping": {
                str(key): value.value if hasattr(value, "value") else str(value)
                for key, value in config.normalization_mapping.items()
            },
            "stats_source": str(DATASET_ROOT / "meta/stats.json"),
            "stats_sha256": sha256_file(DATASET_ROOT / "meta/stats.json"),
            "raw_action_to_normalized_to_inverse_max_abs_error": roundtrip_error,
            "roundtrip_finite": bool(torch.isfinite(roundtrip).all()),
            "image_preprocessing": "LeRobot ACT default dataset MEAN_STD normalization; no image augmentation; ResNet-18 ImageNet initialization",
        },
        "model_preflight": {
            "batch_size": len(batch_indices),
            "loss": float(loss.detach().cpu()),
            "loss_parts": {key: float(value) for key, value in loss_parts.items()},
            "loss_finite": bool(torch.isfinite(loss.detach())),
            "grad_norm": float(grad_norm.detach().cpu()),
            "grad_norm_finite": bool(torch.isfinite(grad_norm.detach())),
            "all_gradients_finite": finite_gradients,
            "one_step_seconds": step_seconds,
            "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "raw_normalized_prediction_shape": list(predicted_normalized.shape),
            "physical_prediction_shape": list(predicted_physical.shape),
            "prediction_finite": bool(torch.isfinite(predicted_physical).all()),
            "dropout_module_count": len(dropout_modules),
            "dropout_modules_active_in_eval": sum(module.training for module in dropout_modules),
        },
        "controlled_inputs": ["observation.images.cam_high", "observation.state"],
        "controlled_output": "action",
        "task_metadata_preserved_but_not_model_input": True,
    }
    checks = [
        payload["dataset_length"] == EXPECTED_FRAMES,
        payload["dataset_num_episodes"] == EXPECTED_EPISODES,
        payload["all_read_shapes_correct"],
        payload["all_first_actions_exact"],
        payload["all_padding_masks_correct"],
        roundtrip_error <= 2e-7,
        payload["model_preflight"]["loss_finite"],
        payload["model_preflight"]["grad_norm_finite"],
        payload["model_preflight"]["all_gradients_finite"],
        payload["model_preflight"]["physical_prediction_shape"] == [8, CHUNK_SIZE, EXPECTED_DIM],
        payload["model_preflight"]["prediction_finite"],
        payload["model_preflight"]["dropout_modules_active_in_eval"] == 0,
    ]
    payload["status"] = "PASS" if all(checks) else "FAIL"
    atomic_json(AUDIT_DIR / "dataset_compatibility.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("ACT Dataset-B compatibility preflight failed")
    return payload


def main() -> None:
    if TRAIN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing ACT training run: {TRAIN_DIR}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    identity, action, _state, episode, _frame = dataset_identity()
    meta = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    policy_config, train_config, config_path = make_config(meta)
    implementation = act_source_audit(policy_config)
    compatibility = compatibility_audit(meta, policy_config, action, episode)
    summary = {
        "schema_version": "act_b_preflight_summary_v1",
        "status": "PASS",
        "dataset_identity": identity["status"],
        "installed_act_implementation": implementation["status"],
        "dataset_compatibility": compatibility["status"],
        "training_config": str(config_path),
        "training_config_sha256": sha256_file(config_path),
        "training_output_dir": str(train_config.output_dir),
        "required_declarations": identity["required_declarations"],
        "ready_to_train": True,
    }
    atomic_json(AUDIT_DIR / "preflight_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
