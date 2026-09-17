#!/usr/bin/env python3
"""Audit and, if required, revise only Dataset-B state semantics.

The frozen Proposed-B action trajectories, episode selection, RGB videos,
tasks, converter, and resolver are read-only inputs.  The only permitted
dataset mutation is replacing ``observation.state`` with the explicitly
labelled prior-target G1 state surrogate after a byte-preserving archive of
the original state-equals-action package has been created.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
ARCHIVE = (
    ROOT
    / "datasets/doll_handoff_proposed_b_50_state_eq_action_archive_2026-08-23"
)
AUDIT = ROOT / "outputs/doll_handoff_dataset_b_semantic_audit_2026-08-23"
FINAL_B = ROOT / "outputs/doll_handoff_dataset_b_final"
ORIGINAL_FINAL_MANIFEST = FINAL_B / "FINAL_DATASET_B_MANIFEST.json"
SOURCE_MANIFEST = FINAL_B / "final_source_manifest.json"
ACTION_FREEZE = FINAL_B / "retargeted_actions/freeze_manifest.json"
ORIGINAL_TRAINING_CONFIG = FINAL_B / "training/policy_b_full_config.json"
LEROBOT_ROOT = Path("/home/jbnu/lerobot-smolvla")
LEROBOT_SOURCE = LEROBOT_ROOT / "src/lerobot"
LEROBOT_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
LEROBOT_TRAIN = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train")
BASE_MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
BASE_MODEL = (
    Path("/home/jbnu/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots")
    / BASE_MODEL_REVISION
)
FULL_CONFIG = AUDIT / "training/policy_b_semantic_final_config.json"
SMOKE_CONFIG = AUDIT / "training/policy_b_semantic_smoke_config.json"
REAL_OUTPUT = ROOT / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2"
SMOKE_OUTPUT = AUDIT / "training_smoke/NOT_A_RESEARCH_POLICY_LAG1_STATE"
SMOKE_STEPS = 10
FULL_STEPS = 20_000
FPS = 30
CHUNK = 50
DIM = 28
STATE_LABEL = "RETARGETED_G1_STATE_SURROGATE"
ORIGINAL_TREE_SHA256 = "de2e142f7ddbbabd9265ca9ee978e6fccaf9e4cdbe098be86ca63d3001e06296"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<value>(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)[KMB]?)",
    re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def dataset_tree(root: Path) -> tuple[str, list[dict[str, Any]]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), entries


def fixed_numpy(column: pa.ChunkedArray) -> np.ndarray:
    values = column.combine_chunks()
    if not pa.types.is_fixed_size_list(values.type) or values.type.list_size != DIM:
        raise TypeError(f"expected fixed-size list {DIM}, got {values.type}")
    return np.asarray(
        values.values.to_numpy(zero_copy_only=False), dtype=np.float32
    ).reshape(len(values), DIM)


def fixed_arrow(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != DIM:
        raise ValueError(values.shape)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), DIM
    )


def feature_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != DIM or not np.isfinite(values).all():
        raise ValueError("feature statistics require finite [N,28]")
    result: dict[str, list[Any]] = {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=0).tolist(),
        "count": [int(len(values))],
    }
    for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
        result[f"q{int(quantile * 100):02d}"] = np.quantile(
            values, quantile, axis=0
        ).tolist()
    return result


def load_arrays(root: Path) -> dict[str, Any]:
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one data Parquet, got {paths}")
    table = pq.read_table(paths[0])
    state = fixed_numpy(table["observation.state"])
    action = fixed_numpy(table["action"])
    episode_index = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    episodes = []
    for episode in range(50):
        indices = np.flatnonzero(episode_index == episode)
        if not len(indices) or not np.array_equal(
            indices, np.arange(indices[0], indices[-1] + 1)
        ):
            raise RuntimeError(f"episode {episode} rows are missing/noncontiguous")
        episodes.append(
            {
                "episode_index": episode,
                "indices": indices,
                "state": state[indices],
                "action": action[indices],
            }
        )
    return {
        "path": paths[0],
        "table": table,
        "state": state,
        "action": action,
        "episode_index": episode_index,
        "episodes": episodes,
    }


def lag1_states(actions: list[np.ndarray]) -> list[np.ndarray]:
    result = []
    for action in actions:
        if not len(action):
            raise ValueError("empty episode")
        result.append(np.concatenate([action[:1], action[:-1]], axis=0))
    return result


def _error_summary(
    errors: np.ndarray,
    joint_names: list[str],
    vector_exact: np.ndarray,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(errors)
    per_joint = []
    for index, name in enumerate(joint_names):
        values = errors[:, index]
        per_joint.append(
            {
                "joint_index": index,
                "joint_name": name,
                "count": len(values),
                "exact_equality_rate": float(np.mean(values == 0.0)),
                "mean_signed_error_rad": float(np.mean(values)),
                "signed_error_std_rad": float(np.std(values, ddof=0)),
                "mae_rad": float(np.mean(np.abs(values))),
                "rmse_rad": float(np.sqrt(np.mean(values**2))),
                "maximum_absolute_error_rad": float(np.max(np.abs(values))),
            }
        )
    return {
        "vector_count": len(errors),
        "vector_exact_equality_count": int(np.count_nonzero(vector_exact)),
        "vector_exact_equality_rate": float(np.mean(vector_exact)),
        "element_count": int(errors.size),
        "element_exact_equality_rate": float(np.mean(errors == 0.0)),
        "mean_signed_error_rad": float(np.mean(errors)),
        "mae_rad": float(np.mean(absolute)),
        "rmse_rad": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_rad": float(np.max(absolute)),
        "per_joint": per_joint,
    }


def leakage_metrics(
    label: str,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    joint_names: list[str],
) -> dict[str, Any]:
    if len(states) != 50 or len(actions) != 50:
        raise ValueError("episode list mismatch")
    all_state = np.concatenate(states).astype(np.float64)
    all_action = np.concatenate(actions).astype(np.float64)
    state_mean = np.mean(all_state, axis=0)
    state_std = np.std(all_state, axis=0, ddof=0)
    action_mean = np.mean(all_action, axis=0)
    action_std = np.std(all_action, axis=0, ddof=0)
    if np.any(state_std <= 0.0) or np.any(action_std <= 0.0):
        raise RuntimeError("zero state/action standard deviation")
    offsets = []
    chunk_errors = []
    chunk_vector_exact = []
    chunk_normalized_copy_errors = []
    for offset in range(CHUNK):
        errors = []
        vector_exact = []
        normalized_errors = []
        for state, action in zip(states, actions, strict=True):
            count = len(action) - offset
            if count <= 0:
                continue
            current = state[:count].astype(np.float64)
            target = action[offset:].astype(np.float64)
            difference = current - target
            errors.append(difference)
            vector_exact.append(np.all(current == target, axis=1))
            normalized_errors.append(
                (current - state_mean) / state_std
                - (target - action_mean) / action_std
            )
        error = np.concatenate(errors)
        exact = np.concatenate(vector_exact)
        normalized = np.concatenate(normalized_errors)
        summary = _error_summary(error, joint_names, exact)
        summary.update(
            {
                "action_chunk_offset": offset,
                "time_offset_seconds": offset / FPS,
                "normalized_copy_mae": float(np.mean(np.abs(normalized))),
                "normalized_copy_rmse": float(
                    np.sqrt(np.mean(normalized**2))
                ),
                "normalized_copy_mse": float(np.mean(normalized**2)),
            }
        )
        offsets.append(summary)
        chunk_errors.append(error)
        chunk_vector_exact.append(exact)
        chunk_normalized_copy_errors.append(normalized)
    error = np.concatenate(chunk_errors)
    exact = np.concatenate(chunk_vector_exact)
    normalized = np.concatenate(chunk_normalized_copy_errors)
    chunk = _error_summary(error, joint_names, exact)
    chunk.update(
        {
            "offsets": list(range(CHUNK)),
            "padding_excluded": True,
            "normalized_copy_mae": float(np.mean(np.abs(normalized))),
            "normalized_copy_rmse": float(np.sqrt(np.mean(normalized**2))),
            "normalized_copy_mse": float(np.mean(normalized**2)),
            "offset_zero_fraction_of_valid_chunk_vectors": float(
                offsets[0]["vector_count"] / chunk["vector_count"]
            ),
        }
    )
    return {
        "label": label,
        "episodes": len(states),
        "frames": len(all_state),
        "state_statistics": feature_stats(all_state),
        "action_statistics": feature_stats(all_action),
        "state_action_mean_max_absolute_difference": float(
            np.max(np.abs(state_mean - action_mean))
        ),
        "state_action_std_max_absolute_difference": float(
            np.max(np.abs(state_std - action_std))
        ),
        "offset_zero": offsets[0],
        "offset_one": offsets[1],
        "offsets": offsets,
        "full_valid_action_chunk_copy_baseline": chunk,
    }


def _source_record(path: Path, line_ranges: list[list[int]], behavior: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "relevant_line_ranges": line_ranges,
        "behavior": behavior,
    }


def installed_temporal_convention() -> dict[str, Any]:
    config = LEROBOT_SOURCE / "policies/smolvla/configuration_smolvla.py"
    model = LEROBOT_SOURCE / "policies/smolvla/modeling_smolvla.py"
    factory = LEROBOT_SOURCE / "datasets/factory.py"
    reader = LEROBOT_SOURCE / "datasets/dataset_reader.py"
    trainer = LEROBOT_SOURCE / "scripts/lerobot_train.py"
    feature_utils = LEROBOT_SOURCE / "datasets/feature_utils.py"
    commit = subprocess.run(
        ["git", "-C", str(LEROBOT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    help_result = subprocess.run(
        [str(LEROBOT_TRAIN), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    if help_result.returncode != 0:
        raise RuntimeError("lerobot-train --help failed")
    help_path = AUDIT / "installed_code/lerobot_train_help.txt"
    atomic_text(help_path, help_result.stdout + help_result.stderr)
    return {
        "schema_version": "dataset_b_installed_smolvla_temporal_convention_v1",
        "status": "PASS_INSPECTED_LOCAL_IMPLEMENTATION",
        "lerobot_root": str(LEROBOT_ROOT.resolve()),
        "lerobot_git_commit": commit,
        "lerobot_train": {
            "path": str(LEROBOT_TRAIN.resolve()),
            "help_sha256": sha256_file(help_path),
        },
        "sources": {
            "smolvla_configuration": _source_record(
                config,
                [[27, 42], [149, 155]],
                "n_obs_steps=1; observation offsets=[0]; action offsets=0..chunk_size-1; chunk_size=50.",
            ),
            "dataset_factory": _source_record(
                factory,
                [[34, 66], [85, 102]],
                "Converts policy delta indices to seconds using dataset FPS and passes them to LeRobotDataset.",
            ),
            "dataset_reader": _source_record(
                reader,
                [[215, 232], [315, 352]],
                "For each row, queries abs_idx+delta; clamps outside the same episode and emits per-feature is_pad masks; task is resolved from task_index.",
            ),
            "feature_utils": _source_record(
                feature_utils,
                [[204, 218]],
                "Converts delta seconds back to integer frame offsets by round(delta*fps).",
            ),
            "smolvla_model": _source_record(
                model,
                [[192, 220], [228, 266], [274, 310], [406, 415]],
                "Training consumes latest observation.state and the 50-row action tensor. Inference conditions on supplied state, predicts a chunk, and select_action immediately pops chunk element 0.",
            ),
            "lerobot_trainer": _source_record(
                trainer,
                [[285, 320], [337, 385], [587, 605]],
                "Builds the dataset from policy deltas, overrides pretrained normalizers with dataset statistics, preprocesses each batch, and performs the policy update.",
            ),
        },
        "resolved_training_sample": {
            "observation_state_at_dataset_index_t": "stored observation.state row t only (offset 0)",
            "action_chunk_at_dataset_index_t": "stored action rows t, t+1, ..., t+49 within the episode",
            "out_of_episode_future_rows": "clamped to T-1 but action_is_pad=True and excluded from SmolVLA loss",
            "action_at_offset_zero": "the simultaneous/current-cycle command paired with observation at t",
            "action_at_positive_offsets": "future commands",
            "inference": "latest measured deployment state conditions a chunk; the first predicted action is returned/executed immediately",
            "state_omission_supported_without_model_change": False,
            "state_omission_reason": "prepare_state directly indexes batch['observation.state']; removing the feature causes a missing-key/model-interface failure.",
        },
    }


def convention_markdown(report: dict[str, Any]) -> str:
    sources = report["sources"]
    lines = [
        "# Installed LeRobot / SmolVLA temporal convention",
        "",
        f"Status: **{report['status']}**",
        "",
        f"LeRobot checkout: `{report['lerobot_root']}`",
        f"Commit: `{report['lerobot_git_commit']}`",
        "",
        "## Resolved behavior",
        "",
        "- `observation.state` uses policy offset `[0]`, so sample index `t` receives the stored state row `t`.",
        "- `action` uses offsets `0..49`, so the supervised chunk starts at stored action row `t` and continues through `t+49`.",
        "- Future requests never cross an episode: they are clamped to the last row and marked by `action_is_pad`; SmolVLA masks those terms from loss.",
        "- At inference, SmolVLA consumes the latest state supplied by the robot adapter, predicts a chunk, and returns chunk element 0 immediately.",
        "- Removing state is not a valid configuration-only change: `prepare_state` directly requires `batch['observation.state']`.",
        "",
        "Therefore action offset 0 is the current-cycle desired command, while positive offsets are future commands. A deployment sample should pair measured current state with that immediate desired command; it should not place the desired command itself in the state input.",
        "",
        "## Inspected local files",
        "",
    ]
    for value in sources.values():
        lines.append(
            f"- `{value['path']}` lines {value['relevant_line_ranges']}: {value['behavior']} SHA256 `{value['sha256']}`"
        )
    lines.append("")
    return "\n".join(lines)


def audit_stage() -> dict[str, Any]:
    if AUDIT.exists():
        raise RuntimeError(f"semantic audit output already exists: {AUDIT}")
    tree_sha, tree_entries = dataset_tree(DATASET)
    if tree_sha != ORIGINAL_TREE_SHA256:
        raise RuntimeError(
            f"authoritative pre-audit dataset tree mismatch: {tree_sha} != {ORIGINAL_TREE_SHA256}"
        )
    arrays = load_arrays(DATASET)
    if len(arrays["state"]) != 34478 or not np.array_equal(
        arrays["state"], arrays["action"]
    ):
        raise RuntimeError("current Dataset B is not the frozen 34,478-row state=action package")
    info = read_json(DATASET / "meta/info.json")
    joint_names = info["features"]["action"]["names"]
    actions = [episode["action"] for episode in arrays["episodes"]]
    states = [episode["state"] for episode in arrays["episodes"]]
    candidate = lag1_states(actions)
    current_metrics = leakage_metrics(
        "CURRENT_STATE_EQUALS_ACTION", states, actions, joint_names
    )
    candidate_metrics = leakage_metrics(
        STATE_LABEL, candidate, actions, joint_names
    )
    convention = installed_temporal_convention()
    atomic_json(AUDIT / "installed_code/temporal_convention.json", convention)
    atomic_text(
        AUDIT / "installed_code/temporal_convention.md",
        convention_markdown(convention),
    )
    atomic_json(AUDIT / "leakage/current_schema_metrics.json", current_metrics)
    atomic_json(AUDIT / "leakage/candidate_lag1_metrics.json", candidate_metrics)
    decision = {
        "schema_version": "dataset_b_state_semantic_decision_v1",
        "status": "REVISION_REQUIRED",
        "current_state_action_schema": "UNSAFE",
        "reason": (
            "The same stored q_target[t] is exposed as observation.state[t] and the "
            "supervised/action-chunk element action[t]. It has 100% vector equality "
            "and zero raw/normalized copy error at the action that inference executes first."
        ),
        "copy_shortcut_scope": {
            "offset_zero_exact_solution": True,
            "full_50_step_chunk_exact_solution": False,
            "offset_zero_fraction_of_valid_chunk_targets": current_metrics[
                "full_valid_action_chunk_copy_baseline"
            ]["offset_zero_fraction_of_valid_chunk_vectors"],
            "full_chunk_normalized_repeat_state_rmse": current_metrics[
                "full_valid_action_chunk_copy_baseline"
            ]["normalized_copy_rmse"],
        },
        "selected_representation": "A_PREVIOUS_TARGET_AS_CURRENT_STATE_SURROGATE",
        "final_state_label": STATE_LABEL,
        "final_state_rule": {
            "episode_frame_zero": "state[0] = q_target[0] cold-start/initial-hold surrogate",
            "episode_frames_one_through_end": "state[t] = q_target[t-1]",
            "formal": "state[t] = q_target[max(t-1, 0)]",
            "measured_real_g1_feedback": False,
            "interpretation": "one-control-period prior commanded configuration used as the smallest zero-order realization surrogate for current measured G1/Dex3 state",
        },
        "action_rule_unchanged": "action[t] = q_target[t] absolute 28D joint-position command",
        "future_chunk_rule_unchanged": "action chunk offsets [0..49] -> q_target[t..t+49], with episode-end clamp plus action_is_pad",
        "boundary_handling": {
            "first_frames_dropped_per_episode": 0,
            "last_frames_dropped_per_episode": 0,
            "total_frames_dropped": 0,
            "cross_episode_state_reference": False,
            "cross_episode_action_reference": False,
            "episode_start_cold_start_count": 50,
        },
        "candidate_b_not_selected": "Moving the chunk start to t+1 would require changing/reindexing the frozen action rows because installed SmolVLA fixes action offsets to 0..49.",
        "candidate_c_not_selected": "Installed SmolVLA directly requires observation.state and has no configuration-only image/language-only path.",
        "residual_exact_equality_interpretation": (
            "Lag-1 state may still equal action on stationary/hold transitions and at the "
            "cold-start row. That is a legitimate no-motion command, not same-row target aliasing."
        ),
        "candidate_offset_zero_metrics": candidate_metrics["offset_zero"],
    }
    atomic_json(AUDIT / "decision/state_semantic_decision.json", decision)
    atomic_text(
        AUDIT / "decision/state_semantic_decision.md",
        "\n".join(
            [
                "# Dataset-B state semantic decision",
                "",
                "Current schema: **UNSAFE**.",
                "",
                "The installed sampler puts the stored same-row action at chunk offset 0, and inference executes that element first. The current state is exactly that target for every row and every joint.",
                "",
                f"Selected state label: **{STATE_LABEL}**",
                "",
                "- Episode frame 0: `state[0] = q_target[0]` as an explicit cold-start/initial-hold surrogate.",
                "- Episode frame t>0: `state[t] = q_target[t-1]`.",
                "- Action remains byte-equivalent to the frozen `action[t] = q_target[t]` trajectory.",
                "- No RGB, task, episode, action, timestamp, or boundary row is dropped or shifted.",
                "- Equality that remains on stationary transitions is legitimate hold behavior rather than target leakage.",
                "",
            ]
        ),
    )
    pre = {
        "schema_version": "dataset_b_semantic_pre_revision_freeze_v1",
        "status": "CURRENT_PACKAGE_FROZEN_BEFORE_STATE_REVISION",
        "dataset_path": str(DATASET.resolve()),
        "dataset_tree_sha256": tree_sha,
        "dataset_tree_file_count": len(tree_entries),
        "dataset_tree_size_bytes": int(
            sum(entry["size_bytes"] for entry in tree_entries)
        ),
        "dataset_tree_entries": tree_entries,
        "state_array_sha256": array_sha256(arrays["state"]),
        "action_array_sha256": array_sha256(arrays["action"]),
        "state_action_exact": True,
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "action_freeze_manifest_sha256": sha256_file(ACTION_FREEZE),
        "original_final_manifest_sha256": sha256_file(ORIGINAL_FINAL_MANIFEST),
        "episodes": 50,
        "frames": 34478,
    }
    atomic_json(AUDIT / "pre_revision/pre_revision_manifest.json", pre)
    result = {
        "status": "REVISION_REQUIRED",
        "current_schema": "UNSAFE",
        "selected_state": STATE_LABEL,
        "dataset_tree_sha256": tree_sha,
        "current_offset_zero_vector_equality_rate": current_metrics[
            "offset_zero"
        ]["vector_exact_equality_rate"],
        "candidate_offset_zero_vector_equality_rate": candidate_metrics[
            "offset_zero"
        ]["vector_exact_equality_rate"],
        "candidate_offset_zero_mae_rad": candidate_metrics["offset_zero"][
            "mae_rad"
        ],
        "candidate_offset_zero_rmse_rad": candidate_metrics["offset_zero"][
            "rmse_rad"
        ],
    }
    atomic_json(AUDIT / "audit_result.json", result)
    return result


def _write_semantic_schema(joint_names: list[str]) -> tuple[Path, str]:
    path = AUDIT / "training_schema/g1_training_schema_semantic_v2.json"
    schema = {
        "schema_version": "doll_handoff_g1_training_schema_semantic_v2",
        "status": "PASS_DEPLOYMENT_CONSISTENT_SURROGATE",
        "state": {
            "key": "observation.state",
            "label": STATE_LABEL,
            "dimension": DIM,
            "joint_names": joint_names,
            "unit": "radian",
            "dtype": "float32",
            "definition": "episode-local q_target[max(t-1,0)] used as a one-control-period prior-target current-state surrogate",
            "measured_real_g1_feedback": False,
            "deployment_replacement": "measured current G1 arm/Dex3 qpos assembled by exact joint name in this order",
        },
        "action": {
            "key": "action",
            "dimension": DIM,
            "joint_names": joint_names,
            "unit": "radian",
            "dtype": "float32",
            "definition": "unchanged frozen Proposed-B absolute q_target[t] joint-position command",
        },
        "temporal": {
            "fps": FPS,
            "state_source_target_index": "max(t-1,0) within the same episode",
            "action_target_index": "t",
            "smolvla_observation_offsets": [0],
            "smolvla_action_offsets": list(range(CHUNK)),
            "resulting_action_targets": "q_target[t]..q_target[t+49]",
            "episode_end": "out-of-range offsets clamp to T-1 and action_is_pad masks them from loss",
            "first_frames_dropped": 0,
            "last_frames_dropped": 0,
            "cross_episode_reference": False,
            "episode_frame_zero": "state[0]=action[0]=q_target[0] explicit cold-start/initial hold",
        },
        "preservation": {
            "rgb_unchanged": True,
            "task_unchanged": True,
            "episode_identities_unchanged": True,
            "action_trajectory_unchanged": True,
            "joint_order_unchanged": True,
            "timestamps_unchanged": True,
            "row_count_unchanged": True,
            "aloha_observation_state_used": False,
            "measured_g1_state_invented": False,
        },
    }
    atomic_json(path, schema)
    markdown = AUDIT / "training_schema/g1_training_schema_semantic_v2.md"
    atomic_text(
        markdown,
        "\n".join(
            [
                "# Final Dataset-B G1 training schema (semantic v2)",
                "",
                f"`observation.state` is labelled **{STATE_LABEL}**.",
                "",
                "```text",
                "state[0]   = q_target[0]          # explicit episode cold start",
                "state[t>0] = q_target[t-1]",
                "action[t]  = q_target[t]          # unchanged frozen absolute command",
                "```",
                "",
                "SmolVLA samples observation offset 0 and action offsets 0..49. No row is dropped, no reference crosses an episode boundary, and padded future actions remain masked by `action_is_pad`.",
                "",
                "At deployment, replace the surrogate state with measured current G1/Dex3 joint positions in the exact same 28-name order.",
                "",
            ]
        ),
    )
    return path, sha256_file(path)


def revise_stage() -> dict[str, Any]:
    audit = read_json(AUDIT / "audit_result.json")
    decision = read_json(AUDIT / "decision/state_semantic_decision.json")
    if audit["status"] != "REVISION_REQUIRED" or decision["current_state_action_schema"] != "UNSAFE":
        raise RuntimeError("audit did not authorize state-only revision")
    current_tree, _ = dataset_tree(DATASET)
    if current_tree != ORIGINAL_TREE_SHA256:
        raise RuntimeError(f"dataset changed after audit: {current_tree}")
    if ARCHIVE.exists():
        raise RuntimeError(f"archive target already exists: {ARCHIVE}")
    shutil.copytree(DATASET, ARCHIVE, copy_function=shutil.copy2)
    archive_tree, archive_entries = dataset_tree(ARCHIVE)
    if archive_tree != current_tree:
        raise RuntimeError("versioned archive is not byte-equivalent")
    (AUDIT / "archive_provenance").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ORIGINAL_FINAL_MANIFEST,
        AUDIT / "archive_provenance/ORIGINAL_FINAL_DATASET_B_MANIFEST.json",
    )
    archive_manifest = {
        "schema_version": "dataset_b_state_equals_action_archive_v1",
        "status": "BYTE_EXACT_ARCHIVE_PASS",
        "archive_path": str(ARCHIVE.resolve()),
        "source_dataset_path_at_archive_time": str(DATASET.resolve()),
        "dataset_tree_sha256": archive_tree,
        "file_count": len(archive_entries),
        "size_bytes": int(sum(row["size_bytes"] for row in archive_entries)),
        "state_semantics": "observation.state[t] = action[t] = q_target[t]",
        "training_use": "DO_NOT_USE; retained for provenance and leakage analysis",
        "created_at": now_iso(),
    }
    atomic_json(AUDIT / "archive_provenance/archive_manifest.json", archive_manifest)

    archived = load_arrays(ARCHIVE)
    current = load_arrays(DATASET)
    actions = [episode["action"] for episode in current["episodes"]]
    candidate_by_episode = lag1_states(actions)
    candidate = np.concatenate(candidate_by_episode).astype(np.float32)
    before_action_sha = array_sha256(current["action"])
    before_non_state = {
        name: current["table"][name].combine_chunks().to_pylist()
        for name in current["table"].column_names
        if name not in {"observation.state"}
    }
    state_index = current["table"].schema.get_field_index("observation.state")
    revised_table = current["table"].set_column(
        state_index,
        current["table"].schema.field(state_index),
        fixed_arrow(candidate),
    )
    data_path = current["path"]
    temporary_data = data_path.with_suffix(data_path.suffix + ".semantic_tmp")
    pq.write_table(
        revised_table,
        temporary_data,
        compression="snappy",
        use_dictionary=True,
        row_group_size=8192,
    )
    temporary_data.replace(data_path)

    episode_path = DATASET / "meta/episodes/chunk-000/file-000.parquet"
    episode_table = pq.read_table(episode_path)
    episode_rows = episode_table.to_pylist()
    for row, state in zip(episode_rows, candidate_by_episode, strict=True):
        for statistic, value in feature_stats(state).items():
            row[f"stats/observation.state/{statistic}"] = value
    revised_episode_table = pa.Table.from_pylist(
        episode_rows, schema=episode_table.schema
    )
    temporary_episodes = episode_path.with_suffix(
        episode_path.suffix + ".semantic_tmp"
    )
    pq.write_table(
        revised_episode_table,
        temporary_episodes,
        compression="snappy",
        use_dictionary=True,
    )
    temporary_episodes.replace(episode_path)

    stats_path = DATASET / "meta/stats.json"
    stats = read_json(stats_path)
    stats["observation.state"] = feature_stats(candidate)
    atomic_json(stats_path, stats)
    info = read_json(DATASET / "meta/info.json")
    joint_names = info["features"]["action"]["names"]
    schema_path, schema_sha = _write_semantic_schema(joint_names)
    packaging_path = DATASET / "meta/g1_packaging_manifest.json"
    packaging = read_json(packaging_path)
    packaging.update(
        {
            "schema_version": "doll_handoff_proposed_b_lerobot_package_semantic_v2",
            "status": "STATE_SEMANTIC_REVISION_PENDING_VALIDATION",
            "training_schema": str(schema_path.resolve()),
            "training_schema_sha256": schema_sha,
            "observation_state_label": STATE_LABEL,
            "observation_state_definition": "q_target[max(t-1,0)] within each episode",
            "observation_state_is_measured_real_g1": False,
            "action_definition": "unchanged frozen q_target[t] absolute command",
            "temporal_convention": "state source index max(t-1,0); action index t; no dropped rows",
            "same_row_state_action_identical_by_construction": False,
            "state_equals_action_archive": str(ARCHIVE.resolve()),
            "state_equals_action_archive_tree_sha256": archive_tree,
            "semantic_revision_scope": "observation.state values and their statistics/semantic metadata only",
            "previous_training_smoke_archived_not_applicable": True,
            "training_smoke": {
                "status": "PENDING_RERUN_FOR_SEMANTIC_V2",
                "real_policy_training_started": False,
            },
        }
    )
    for row in packaging.get("episode_alignment", []):
        row["state_source_target_index_rule"] = "max(t-1,0)"
        row["action_target_index_rule"] = "t"
        row["state_action_row_offset"] = "0 at cold start; +1 target-index separation for t>0"
        row["dropped_boundary_frames"] = 0
        row["cross_episode_reference"] = False
    atomic_json(packaging_path, packaging)
    atomic_json(
        DATASET / "meta/g1_validation.json",
        {
            "schema_version": "doll_handoff_dataset_b_semantic_v2_validation",
            "status": "PENDING",
            "observation_state_label": STATE_LABEL,
            "training_smoke_executed": False,
            "real_policy_b_training_started": False,
        },
    )

    revised = load_arrays(DATASET)
    after_non_state = {
        name: revised["table"][name].combine_chunks().to_pylist()
        for name in revised["table"].column_names
        if name not in {"observation.state"}
    }
    if before_non_state != after_non_state:
        raise RuntimeError("a non-state data column changed")
    if not np.array_equal(revised["action"], archived["action"]):
        raise RuntimeError("frozen action values changed")
    if array_sha256(revised["action"]) != before_action_sha:
        raise RuntimeError("frozen action array hash changed")
    if not np.array_equal(revised["state"], candidate):
        raise RuntimeError("revised state does not implement lag-1 rule")
    unchanged_files = [
        "meta/info.json",
        "meta/tasks.parquet",
        *[
            path.relative_to(DATASET).as_posix()
            for path in sorted((DATASET / "videos").rglob("*.mp4"))
        ],
    ]
    changed_unexpected = [
        relative
        for relative in unchanged_files
        if sha256_file(DATASET / relative) != sha256_file(ARCHIVE / relative)
    ]
    if changed_unexpected:
        raise RuntimeError(f"immutable dataset assets changed: {changed_unexpected}")
    revised_tree, revised_entries = dataset_tree(DATASET)
    report = {
        "schema_version": "dataset_b_state_semantic_revision_v2",
        "status": "STATE_ONLY_REVISION_COMPLETE_PENDING_VALIDATION",
        "archive": archive_manifest,
        "authoritative_dataset_path": str(DATASET.resolve()),
        "pre_revision_tree_sha256": archive_tree,
        "post_revision_pre_validation_tree_sha256": revised_tree,
        "post_revision_file_count": len(revised_entries),
        "state_label": STATE_LABEL,
        "state_rule": "q_target[max(t-1,0)] episode-local",
        "action_rule": "q_target[t] unchanged",
        "state_array_sha256_before": array_sha256(archived["state"]),
        "state_array_sha256_after": array_sha256(revised["state"]),
        "action_array_sha256_before": before_action_sha,
        "action_array_sha256_after": array_sha256(revised["action"]),
        "action_array_byte_equivalent": True,
        "immutable_assets_byte_equivalent": True,
        "immutable_assets_checked": unchanged_files,
        "episodes": 50,
        "frames": 34478,
        "first_frames_dropped": 0,
        "last_frames_dropped": 0,
        "cross_episode_reference": False,
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "action_freeze_manifest_sha256": sha256_file(ACTION_FREEZE),
    }
    atomic_json(AUDIT / "revision/state_revision.json", report)
    return report


def _ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]),
    }


def validate_stage() -> dict[str, Any]:
    revision = read_json(AUDIT / "revision/state_revision.json")
    if revision["status"] != "STATE_ONLY_REVISION_COMPLETE_PENDING_VALIDATION":
        raise RuntimeError("state revision is not ready for validation")
    current = load_arrays(DATASET)
    archived = load_arrays(ARCHIVE)
    info = read_json(DATASET / "meta/info.json")
    joint_names = info["features"]["action"]["names"]
    current_actions = [row["action"] for row in current["episodes"]]
    expected_states = lag1_states(current_actions)
    expected_state = np.concatenate(expected_states)
    checks: dict[str, bool] = {
        "episodes_50": len(current["episodes"]) == 50,
        "frames_34478": len(current["state"]) == 34478,
        "state_dimension_28": current["state"].shape == (34478, 28),
        "action_dimension_28": current["action"].shape == (34478, 28),
        "state_finite": bool(np.isfinite(current["state"]).all()),
        "action_finite": bool(np.isfinite(current["action"]).all()),
        "joint_order_unchanged": info["features"]["observation.state"]["names"]
        == info["features"]["action"]["names"],
        "state_matches_lag1_rule": np.array_equal(current["state"], expected_state),
        "same_row_state_action_not_globally_identical": not np.array_equal(
            current["state"], current["action"]
        ),
        "action_numeric_array_unchanged": np.array_equal(
            current["action"], archived["action"]
        ),
        "action_array_hash_unchanged": array_sha256(current["action"])
        == revision["action_array_sha256_before"],
        "source_manifest_unchanged": sha256_file(SOURCE_MANIFEST)
        == revision["source_manifest_sha256"],
        "action_freeze_unchanged": sha256_file(ACTION_FREEZE)
        == revision["action_freeze_manifest_sha256"],
        "info_metadata_byte_unchanged": sha256_file(DATASET / "meta/info.json")
        == sha256_file(ARCHIVE / "meta/info.json"),
        "task_metadata_byte_unchanged": sha256_file(
            DATASET / "meta/tasks.parquet"
        )
        == sha256_file(ARCHIVE / "meta/tasks.parquet"),
        "timestamps_unchanged": current["table"]["timestamp"].to_pylist()
        == archived["table"]["timestamp"].to_pylist(),
        "indices_unchanged": all(
            current["table"][key].to_pylist()
            == archived["table"][key].to_pylist()
            for key in ("frame_index", "episode_index", "index", "task_index")
        ),
    }
    boundary_rows = []
    for episode, (row, expected) in enumerate(
        zip(current["episodes"], expected_states, strict=True)
    ):
        action = row["action"]
        state = row["state"]
        boundary_rows.append(
            {
                "episode_index": episode,
                "length": len(action),
                "cold_start_state_equals_action0": bool(
                    np.array_equal(state[0], action[0])
                ),
                "subsequent_state_equals_previous_action": bool(
                    np.array_equal(state[1:], action[:-1])
                ),
                "matches_expected": bool(np.array_equal(state, expected)),
                "first_frames_dropped": 0,
                "last_frames_dropped": 0,
                "cross_episode_reference": False,
            }
        )
    checks["all_episode_boundaries_correct"] = all(
        row["cold_start_state_equals_action0"]
        and row["subsequent_state_equals_previous_action"]
        and row["matches_expected"]
        and not row["cross_episode_reference"]
        for row in boundary_rows
    )
    stats = read_json(DATASET / "meta/stats.json")
    checks["global_state_stats_exact"] = stats["observation.state"] == feature_stats(
        current["state"]
    )
    checks["global_action_stats_exact"] = stats["action"] == feature_stats(
        current["action"]
    )
    episode_table = pq.read_table(
        DATASET / "meta/episodes/chunk-000/file-000.parquet"
    )
    episode_rows = episode_table.to_pylist()
    episode_stats_checks = []
    for row, state, action in zip(
        episode_rows, expected_states, current_actions, strict=True
    ):
        for feature, values in (
            ("observation.state", state),
            ("action", action),
        ):
            expected = feature_stats(values)
            episode_stats_checks.append(
                all(
                    row[f"stats/{feature}/{statistic}"] == value
                    for statistic, value in expected.items()
                )
            )
    checks["per_episode_stats_exact"] = all(episode_stats_checks)
    archive_video_hashes = {
        path.relative_to(ARCHIVE).as_posix(): sha256_file(path)
        for path in sorted((ARCHIVE / "videos").rglob("*.mp4"))
    }
    current_video_hashes = {
        path.relative_to(DATASET).as_posix(): sha256_file(path)
        for path in sorted((DATASET / "videos").rglob("*.mp4"))
    }
    checks["all_50_videos_byte_unchanged"] = (
        len(current_video_hashes) == 50
        and current_video_hashes == archive_video_hashes
    )
    video_probe = []
    for episode, row in enumerate(episode_rows):
        path = (
            DATASET
            / "videos/observation.images.cam_high/chunk-000"
            / f"file-{episode:03d}.mp4"
        )
        probe = _ffprobe(path)
        probe["episode_index"] = episode
        probe["path"] = str(path.resolve())
        probe["sha256"] = current_video_hashes[
            path.relative_to(DATASET).as_posix()
        ]
        probe["frame_count_matches"] = probe["frames"] == int(row["length"])
        video_probe.append(probe)
    checks["all_50_videos_ffprobe_pass"] = all(
        row["width"] == 640
        and row["height"] == 480
        and row["fps"] == "30/1"
        and row["frame_count_matches"]
        for row in video_probe
    )
    readback_path = AUDIT / "validation/lerobot_readback.json"
    command = [
        str(LEROBOT_PYTHON),
        str(ROOT / "tools/validate_doll_handoff_dataset_b_lerobot.py"),
        "--dataset",
        str(DATASET),
        "--output",
        str(readback_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_text(
        AUDIT / "validation/lerobot_readback_stdout.log",
        completed.stdout + completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LeRobot readback failed:\n" + completed.stdout + completed.stderr
        )
    readback = read_json(readback_path)
    checks["actual_lerobot_dataset_read_pass"] = readback["status"] == "PASS"
    checks["all_50_actual_video_decodes_pass"] = (
        readback["every_episode_middle_frame_read_count"] == 50
    )
    checks["action_chunk_shape_50x28"] = readback[
        "action_chunk_tensor_shape"
    ] == [50, 28]
    post_metrics = leakage_metrics(
        STATE_LABEL,
        [row["state"] for row in current["episodes"]],
        current_actions,
        joint_names,
    )
    audited_candidate = read_json(AUDIT / "leakage/candidate_lag1_metrics.json")
    checks["post_package_leakage_metrics_match_candidate"] = (
        post_metrics["offset_zero"] == audited_candidate["offset_zero"]
        and post_metrics["full_valid_action_chunk_copy_baseline"]
        == audited_candidate["full_valid_action_chunk_copy_baseline"]
    )
    atomic_json(AUDIT / "validation/post_package_leakage_metrics.json", post_metrics)
    if not all(checks.values()):
        raise RuntimeError(
            f"semantic Dataset-B validation failed: {[k for k,v in checks.items() if not v]}"
        )
    validation_meta = {
        "schema_version": "doll_handoff_dataset_b_semantic_v2_validation",
        "status": "PASS_PENDING_NEW_TRAINING_SMOKE",
        "observation_state_label": STATE_LABEL,
        "checks": checks,
        "episodes": 50,
        "frames": 34478,
        "state_dimension": 28,
        "action_dimension": 28,
        "joint_names": joint_names,
        "training_smoke_executed": False,
        "real_policy_b_training_started": False,
    }
    atomic_json(DATASET / "meta/g1_validation.json", validation_meta)
    packaging_path = DATASET / "meta/g1_packaging_manifest.json"
    packaging = read_json(packaging_path)
    packaging["status"] = "SEMANTIC_V2_VALIDATED_PENDING_TRAINING_SMOKE"
    packaging["semantic_validation"] = "PASS"
    atomic_json(packaging_path, packaging)
    tree_sha, tree_entries = dataset_tree(DATASET)
    report = {
        "schema_version": "dataset_b_semantic_v2_full_validation_v1",
        "status": "PASS_PENDING_NEW_TRAINING_SMOKE",
        "dataset_path": str(DATASET.resolve()),
        "checks": checks,
        "episodes": 50,
        "frames": 34478,
        "state_dimension": 28,
        "action_dimension": 28,
        "joint_names": joint_names,
        "boundary_alignment": boundary_rows,
        "video_probe": video_probe,
        "lerobot_readback": str(readback_path.resolve()),
        "lerobot_readback_sha256": sha256_file(readback_path),
        "state_normalization_statistics": stats["observation.state"],
        "action_normalization_statistics": stats["action"],
        "dataset_tree_sha256_pre_smoke": tree_sha,
        "dataset_tree_file_count": len(tree_entries),
    }
    atomic_json(AUDIT / "validation/full_validation.json", report)
    return report


def training_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT)
            + (
                ":" + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            ),
            "MPLCONFIGDIR": "/tmp/dataset_b_semantic_training_mpl",
        }
    )
    return environment


def exact_training_command() -> str:
    return (
        "CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "
        f"{LEROBOT_TRAIN} --config_path {FULL_CONFIG}"
    )


def prepare_training_stage() -> dict[str, Any]:
    validation = read_json(AUDIT / "validation/full_validation.json")
    if validation["status"] != "PASS_PENDING_NEW_TRAINING_SMOKE":
        raise RuntimeError("semantic validation must pass before training preflight")
    base = read_json(ORIGINAL_TRAINING_CONFIG)
    full = copy.deepcopy(base)
    full["dataset"]["root"] = str(DATASET.resolve())
    full["dataset"]["repo_id"] = "local/doll_handoff_proposed_b_50_semantic_v2"
    full["output_dir"] = str(REAL_OUTPUT.resolve())
    full["job_name"] = "policy_b_doll_handoff_proposed_b_50_lag1_state_v2"
    full["steps"] = FULL_STEPS
    full["batch_size"] = 16
    full["optimizer"]["lr"] = 1e-4
    full["scheduler"]["peak_lr"] = 1e-4
    full["policy"]["optimizer_lr"] = 1e-4
    full["policy"]["pretrained_path"] = str(BASE_MODEL.resolve())
    full["policy"]["input_features"]["observation.state"]["shape"] = [28]
    full["policy"]["output_features"]["action"]["shape"] = [28]
    full["policy"]["max_state_dim"] = 32
    full["policy"]["max_action_dim"] = 32
    full["policy"]["adapt_to_pi_aloha"] = False
    full["policy"]["use_delta_joint_actions_aloha"] = False
    full["policy"]["normalization_mapping"] = {
        "VISUAL": "IDENTITY",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }
    full["policy"]["use_amp"] = True
    full["policy"]["device"] = "cuda"
    full["save_freq"] = 5000
    full["log_freq"] = 100
    full["wandb"]["enable"] = False
    atomic_json(FULL_CONFIG, full)
    smoke = copy.deepcopy(full)
    smoke["output_dir"] = str(SMOKE_OUTPUT.resolve())
    smoke["job_name"] = full["job_name"] + "_SMOKE10_NOT_A_RESEARCH_POLICY"
    smoke["steps"] = SMOKE_STEPS
    smoke["save_freq"] = SMOKE_STEPS
    smoke["log_freq"] = 1
    atomic_json(SMOKE_CONFIG, smoke)
    preflight_path = AUDIT / "training/smolvla_preflight.json"
    command = [
        str(LEROBOT_PYTHON),
        str(ROOT / "tools/audit_doll_handoff_dataset_b_smolvla.py"),
        "preflight",
        "--dataset",
        str(DATASET),
        "--base-model",
        str(BASE_MODEL),
        "--config",
        str(FULL_CONFIG),
        "--config",
        str(SMOKE_CONFIG),
        "--output",
        str(preflight_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=training_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_text(
        AUDIT / "training/smolvla_preflight_stdout.log",
        completed.stdout + completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError("SmolVLA preflight failed:\n" + completed.stdout + completed.stderr)
    preflight = read_json(preflight_path)
    command_path = AUDIT / "training/EXACT_POLICY_B_TRAINING_COMMAND.txt"
    atomic_text(
        command_path,
        "# PREPARED ONLY; FULL POLICY-B TRAINING WAS NOT STARTED.\n"
        + exact_training_command()
        + "\n",
    )
    report = {
        "schema_version": "dataset_b_semantic_v2_training_preflight_v1",
        "status": "PASS",
        "state_label": STATE_LABEL,
        "dataset_path": str(DATASET.resolve()),
        "pretrained_model": str(BASE_MODEL.resolve()),
        "pretrained_model_sha256": sha256_file(BASE_MODEL / "model.safetensors"),
        "full_config": str(FULL_CONFIG.resolve()),
        "full_config_sha256": sha256_file(FULL_CONFIG),
        "smoke_config": str(SMOKE_CONFIG.resolve()),
        "smoke_config_sha256": sha256_file(SMOKE_CONFIG),
        "preflight": preflight,
        "exact_training_command": exact_training_command(),
        "full_training_output": str(REAL_OUTPUT.resolve()),
        "full_training_output_absent": not REAL_OUTPUT.exists(),
        "full_training_started": False,
    }
    if preflight["status"] != "PASS" or not report["full_training_output_absent"]:
        raise RuntimeError("semantic training preflight invariant failed")
    atomic_json(AUDIT / "training/training_preflight.json", report)
    return report


def _gpu_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "gpu": gpu.stdout.strip(),
        "compute_processes": [
            line.strip() for line in processes.stdout.splitlines() if line.strip()
        ],
    }


def parse_metrics(path: Path) -> list[dict[str, Any]]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    aliases = {
        "loss": "loss",
        "grdn": "gradient_norm",
        "lr": "learning_rate",
        "mem_gb": "gpu_memory_gb",
    }
    rows = []
    for segment in re.split(r"[\r\n]+", text):
        pairs = {
            match.group("key").lower(): match.group("value")
            for match in METRIC_RE.finditer(segment)
        }
        if "step" not in pairs or "loss" not in pairs:
            continue
        row: dict[str, Any] = {"step": int(float(pairs["step"]))}
        for source, target in aliases.items():
            if source in pairs:
                row[target] = float(pairs[source])
        rows.append(row)
    if len(rows) == SMOKE_STEPS:
        for index, row in enumerate(rows, start=1):
            row["step"] = index
    return rows


def smoke_stage() -> dict[str, Any]:
    preflight = read_json(AUDIT / "training/training_preflight.json")
    if preflight["status"] != "PASS":
        raise RuntimeError("training preflight not PASS")
    if REAL_OUTPUT.exists():
        raise RuntimeError("real Policy-B output exists; refusing smoke/full ambiguity")
    if SMOKE_OUTPUT.exists():
        raise RuntimeError(f"semantic smoke output already exists: {SMOKE_OUTPUT}")
    processes = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    conflicts = [
        line.strip()
        for line in processes
        if ("lerobot.scripts.lerobot_train" in line or "/lerobot-train" in line)
        and "ps -eo" not in line
    ]
    if conflicts:
        raise RuntimeError(f"active LeRobot job detected: {conflicts}")
    gpu_before = _gpu_snapshot()
    if gpu_before["compute_processes"]:
        raise RuntimeError(f"GPU already busy: {gpu_before['compute_processes']}")
    log_path = AUDIT / "training_smoke/training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(LEROBOT_TRAIN), "--config_path", str(SMOKE_CONFIG)]
    start_timestamp = now_iso()
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=training_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    runtime = {
        "command": command,
        "start_timestamp": start_timestamp,
        "end_timestamp": now_iso(),
        "duration_seconds": time.monotonic() - start,
        "returncode": returncode,
        "gpu_before": gpu_before,
        "gpu_after": _gpu_snapshot(),
    }
    atomic_json(AUDIT / "training_smoke/runtime.json", runtime)
    if returncode != 0:
        raise RuntimeError(f"semantic smoke failed: {log_path}")
    checkpoint = SMOKE_OUTPUT / "checkpoints/000010/pretrained_model"
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing smoke checkpoint: {checkpoint}")
    checkpoint_audit_path = AUDIT / "training_smoke/checkpoint_audit.json"
    checkpoint_command = [
        str(LEROBOT_PYTHON),
        str(ROOT / "tools/audit_doll_handoff_dataset_b_smolvla.py"),
        "checkpoint",
        "--dataset",
        str(DATASET),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(checkpoint_audit_path),
    ]
    completed = subprocess.run(
        checkpoint_command,
        cwd=ROOT,
        env=training_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_text(
        AUDIT / "training_smoke/checkpoint_audit_stdout.log",
        completed.stdout + completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "semantic smoke checkpoint audit failed:\n"
            + completed.stdout
            + completed.stderr
        )
    checkpoint_audit = read_json(checkpoint_audit_path)
    metrics = parse_metrics(log_path)
    losses = [row.get("loss", float("nan")) for row in metrics]
    gradients = [row.get("gradient_norm", float("nan")) for row in metrics]
    log_lower = log_path.read_text(encoding="utf-8", errors="replace").lower()
    checks = {
        "exactly_10_steps": len(metrics) == 10
        and [row["step"] for row in metrics] == list(range(1, 11)),
        "loss_finite": len(losses) == 10
        and all(math.isfinite(value) for value in losses),
        "gradient_finite": len(gradients) == 10
        and all(math.isfinite(value) for value in gradients),
        "no_cuda_oom": "out of memory" not in log_lower,
        "no_shape_error": "shape mismatch" not in log_lower
        and "size mismatch" not in log_lower,
        "checkpoint_saved": checkpoint.is_dir(),
        "checkpoint_reload_pass": checkpoint_audit["status"] == "PASS",
        "prediction_shape_batch_chunk_28": checkpoint_audit[
            "prediction_shape"
        ]
        == [1, 50, 28],
        "dataset_specific_normalization": checkpoint_audit["checks"][
            "dataset_specific_normalization_loaded"
        ],
        "full_training_not_started": not REAL_OUTPUT.exists(),
    }
    summary = {
        "schema_version": "dataset_b_semantic_v2_training_smoke_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "state_label": STATE_LABEL,
        "research_policy": False,
        "label": "NOT_A_RESEARCH_POLICY_LAG1_STATE",
        "steps": 10,
        "checks": checks,
        "metrics": metrics,
        "loss": {
            "initial": losses[0] if losses else None,
            "final": losses[-1] if losses else None,
            "minimum": min(losses) if losses else None,
            "mean": float(np.mean(losses)) if losses else None,
            "per_step": losses,
        },
        "gradient_norm": {
            "maximum": max(gradients) if gradients else None,
            "per_step": gradients,
        },
        "runtime": runtime,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "checkpoint_audit": checkpoint_audit,
        "checkpoint_audit_sha256": sha256_file(checkpoint_audit_path),
        "full_policy_b_training": "NOT_STARTED_BY_DESIGN",
    }
    if summary["status"] != "PASS":
        raise RuntimeError(f"semantic smoke gate failed: {checks}")
    atomic_json(AUDIT / "training_smoke/smoke_result.json", summary)
    atomic_text(
        SMOKE_OUTPUT / "README_NOT_A_RESEARCH_POLICY.md",
        "# NOT A RESEARCH POLICY\n\n"
        "This checkpoint is only the ten-step Dataset-B semantic-v2 "
        "dataloader/forward/backward/save/reload smoke test.\n",
    )
    smoke_sha = sha256_file(AUDIT / "training_smoke/smoke_result.json")
    validation_path = DATASET / "meta/g1_validation.json"
    validation = read_json(validation_path)
    validation["status"] = "PASS_SEMANTIC_V2_AND_TRAINING_SMOKE"
    validation["training_smoke_executed"] = True
    validation["training_smoke_status"] = "PASS"
    validation["training_smoke_steps"] = 10
    validation["training_smoke_research_policy"] = False
    validation["training_smoke_result_sha256"] = smoke_sha
    validation["prediction_shape"] = [1, 50, 28]
    validation["real_policy_b_training_started"] = False
    atomic_json(validation_path, validation)
    packaging_path = DATASET / "meta/g1_packaging_manifest.json"
    packaging = read_json(packaging_path)
    packaging["status"] = "DATASET_B_SEMANTIC_V2_READY_TRAINING_SMOKE_PASSED"
    packaging["training_smoke"] = {
        "status": "PASS",
        "steps": 10,
        "state_label": STATE_LABEL,
        "output": str(SMOKE_OUTPUT.resolve()),
        "research_policy": False,
        "smoke_result_sha256": smoke_sha,
        "real_policy_training_started": False,
    }
    atomic_json(packaging_path, packaging)
    return summary


def final_report(manifest: dict[str, Any]) -> str:
    leak = manifest["state_action_leakage"]
    smoke = manifest["training_smoke"]
    return "\n".join(
        [
            "CURRENT STATE=ACTION SCHEMA",
            "safe / unsafe / ambiguous: unsafe",
            "reason: Stored state[t] was exactly the supervised and immediately executed action[t] for 100% of 34,478 frames and all 28 joints.",
            "",
            "INSTALLED SMOLVLA TEMPORAL CONVENTION",
            "observation offsets: [0]",
            "action offsets: [0..49]",
            "action[0] meaning: simultaneous/current-cycle command; inference executes it first",
            "episode end: clamped to T-1 and masked by action_is_pad",
            "",
            "FINAL DATASET-B STATE DEFINITION",
            f"label: {STATE_LABEL}",
            "state[0] = q_target[0] explicit episode cold start",
            "state[t>0] = q_target[t-1] within the same episode",
            "deployment: replace surrogate with measured current G1/Dex3 state in the same 28-joint order",
            "",
            "FINAL ACTION DEFINITION",
            "action[t] = unchanged frozen absolute 28D q_target[t] command",
            "future chunk = action[t..t+49], with episode-end padding masked",
            "",
            "STATE/ACTION LEAKAGE",
            "removed: YES",
            f"current offset-0 vector equality: {leak['before_offset_zero_vector_exact_rate']}",
            f"final offset-0 vector equality: {leak['after_offset_zero_vector_exact_rate']} (stationary holds/cold starts, not same-row aliasing)",
            f"final offset-0 MAE/RMSE rad: {leak['after_offset_zero_mae_rad']} / {leak['after_offset_zero_rmse_rad']}",
            "",
            "DATASET B",
            f"path: {manifest['dataset']['path']}",
            f"episodes: {manifest['dataset']['episodes']}",
            f"frames: {manifest['dataset']['frames']}",
            f"validation: {manifest['dataset']['validation']}",
            "",
            "TRAINING SMOKE",
            f"status: {smoke['status']}",
            f"loss: initial={smoke['initial_loss']}, final={smoke['final_loss']}",
            "",
            "FULL POLICY-B TRAINING",
            "NOT_STARTED_BY_DESIGN",
            "",
            "EXACT TRAINING COMMAND",
            manifest["full_policy_b_training"]["exact_command"],
            "",
            "DATASET_B_SEMANTICALLY_READY_FOR_POLICY_TRAINING",
            "",
        ]
    )


def freeze_stage() -> dict[str, Any]:
    smoke = read_json(AUDIT / "training_smoke/smoke_result.json")
    validation = read_json(AUDIT / "validation/full_validation.json")
    decision = read_json(AUDIT / "decision/state_semantic_decision.json")
    current_metrics = read_json(AUDIT / "leakage/current_schema_metrics.json")
    final_metrics = read_json(AUDIT / "validation/post_package_leakage_metrics.json")
    revision = read_json(AUDIT / "revision/state_revision.json")
    preflight = read_json(AUDIT / "training/training_preflight.json")
    if not (
        smoke["status"] == "PASS"
        and validation["status"] == "PASS_PENDING_NEW_TRAINING_SMOKE"
        and decision["current_state_action_schema"] == "UNSAFE"
        and not REAL_OUTPUT.exists()
    ):
        raise RuntimeError("final semantic freeze prerequisites failed")
    current = load_arrays(DATASET)
    archived = load_arrays(ARCHIVE)
    if not np.array_equal(current["action"], archived["action"]):
        raise RuntimeError("action trajectory differs from archive at final freeze")
    expected = np.concatenate(
        lag1_states([row["action"] for row in current["episodes"]])
    )
    if not np.array_equal(current["state"], expected):
        raise RuntimeError("final state surrogate rule mismatch")
    tree_sha, tree_entries = dataset_tree(DATASET)
    archive_sha, archive_entries = dataset_tree(ARCHIVE)
    if archive_sha != ORIGINAL_TREE_SHA256:
        raise RuntimeError("versioned original archive changed")
    final_readback_path = AUDIT / "final/final_lerobot_readback.json"
    completed = subprocess.run(
        [
            str(LEROBOT_PYTHON),
            str(ROOT / "tools/validate_doll_handoff_dataset_b_lerobot.py"),
            "--dataset",
            str(DATASET),
            "--output",
            str(final_readback_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_text(
        AUDIT / "final/final_lerobot_readback_stdout.log",
        completed.stdout + completed.stderr,
    )
    if completed.returncode != 0 or read_json(final_readback_path)["status"] != "PASS":
        raise RuntimeError("post-smoke final LeRobot readback failed")
    source = read_json(SOURCE_MANIFEST)
    action_freeze = read_json(ACTION_FREEZE)
    artifacts_paths = {
        "temporal_convention": AUDIT / "installed_code/temporal_convention.json",
        "current_leakage_metrics": AUDIT / "leakage/current_schema_metrics.json",
        "final_leakage_metrics": AUDIT / "validation/post_package_leakage_metrics.json",
        "semantic_decision": AUDIT / "decision/state_semantic_decision.json",
        "state_revision": AUDIT / "revision/state_revision.json",
        "semantic_schema": AUDIT / "training_schema/g1_training_schema_semantic_v2.json",
        "full_validation": AUDIT / "validation/full_validation.json",
        "final_lerobot_readback": final_readback_path,
        "normalization_statistics": DATASET / "meta/stats.json",
        "full_training_config": FULL_CONFIG,
        "smoke_training_config": SMOKE_CONFIG,
        "training_preflight": AUDIT / "training/training_preflight.json",
        "training_smoke": AUDIT / "training_smoke/smoke_result.json",
        "smoke_checkpoint_model": Path(smoke["checkpoint"]) / "model.safetensors",
        "source_manifest": SOURCE_MANIFEST,
        "action_freeze": ACTION_FREEZE,
        "archive_manifest": AUDIT / "archive_provenance/archive_manifest.json",
    }
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in artifacts_paths.items()
    }
    manifest = {
        "schema_version": "doll_handoff_dataset_b_semantic_final_manifest_v2",
        "status": "DATASET_B_SEMANTICALLY_READY_FOR_POLICY_TRAINING",
        "created_at": now_iso(),
        "current_state_equals_action_schema": {
            "decision": "UNSAFE",
            "reason": decision["reason"],
        },
        "installed_temporal_convention": read_json(
            AUDIT / "installed_code/temporal_convention.json"
        ),
        "final_state": {
            "label": STATE_LABEL,
            "dimension": 28,
            "definition": "q_target[max(t-1,0)] episode-local prior-target current-state surrogate",
            "frame_zero": "q_target[0] cold-start/initial-hold surrogate",
            "measured_real_g1_feedback": False,
            "deployment": "measured current G1/Dex3 state replaces surrogate",
        },
        "final_action": {
            "dimension": 28,
            "definition": "unchanged frozen Proposed-B absolute q_target[t] command",
            "numeric_array_sha256": array_sha256(current["action"]),
            "unchanged_from_archive": True,
            "trajectory_set_sha256": action_freeze["trajectory_set_sha256"],
            "canonical_policy_action_set_sha256": action_freeze[
                "canonical_policy_action_set_sha256"
            ],
        },
        "temporal_alignment": {
            "state_source_index": "max(t-1,0)",
            "action_index": "t",
            "action_chunk_offsets": list(range(50)),
            "first_frames_dropped": 0,
            "last_frames_dropped": 0,
            "episode_crossing": False,
            "episode_end": "clamp plus action_is_pad mask",
        },
        "state_action_leakage": {
            "structural_same_row_alias_removed": True,
            "removed": "YES",
            "before_offset_zero_vector_exact_rate": current_metrics[
                "offset_zero"
            ]["vector_exact_equality_rate"],
            "before_offset_zero_mae_rad": current_metrics["offset_zero"]["mae_rad"],
            "before_offset_zero_rmse_rad": current_metrics["offset_zero"]["rmse_rad"],
            "after_offset_zero_vector_exact_rate": final_metrics[
                "offset_zero"
            ]["vector_exact_equality_rate"],
            "after_offset_zero_element_exact_rate": final_metrics[
                "offset_zero"
            ]["element_exact_equality_rate"],
            "after_offset_zero_mae_rad": final_metrics["offset_zero"]["mae_rad"],
            "after_offset_zero_rmse_rad": final_metrics["offset_zero"]["rmse_rad"],
            "after_offset_zero_normalized_copy_rmse": final_metrics[
                "offset_zero"
            ]["normalized_copy_rmse"],
            "after_full_chunk_normalized_copy_rmse": final_metrics[
                "full_valid_action_chunk_copy_baseline"
            ]["normalized_copy_rmse"],
            "residual_equality": "stationary/hold transitions and explicit episode cold starts only",
        },
        "dataset": {
            "path": str(DATASET.resolve()),
            "episodes": 50,
            "frames": 34478,
            "state_dimension": 28,
            "action_dimension": 28,
            "joint_names": source["episodes"][0].get(
                "joint_names", read_json(DATASET / "meta/info.json")["features"]["action"]["names"]
            ),
            "validation": "PASS",
            "lerobot_readback": "PASS",
            "all_50_video_decode": "PASS",
            "dataset_tree_sha256": tree_sha,
            "dataset_tree_file_count": len(tree_entries),
            "dataset_tree_size_bytes": int(
                sum(row["size_bytes"] for row in tree_entries)
            ),
            "dataset_tree_entries": tree_entries,
        },
        "archive": {
            "path": str(ARCHIVE.resolve()),
            "status": "BYTE_EXACT_ARCHIVE_PASS",
            "tree_sha256": archive_sha,
            "file_count": len(archive_entries),
            "training_use": "DO_NOT_USE_STATE_EQUALS_ACTION_ARCHIVE",
        },
        "preservation": {
            "rgb_video_bytes_unchanged": True,
            "task_metadata_bytes_unchanged": True,
            "episode_identities_unchanged": True,
            "source_manifest_unchanged": True,
            "frozen_action_trajectory_unchanged": True,
            "joint_order_unchanged": True,
            "timestamps_and_indices_unchanged": True,
            "converter_modified": False,
            "resolver_modified": False,
            "source_selection_modified": False,
        },
        "normalization": {
            "state": "recomputed from final lag-1 surrogate",
            "action": "unchanged Dataset-B action statistics",
            "checkpoint_matches_dataset_stats": smoke["checkpoint_audit"]["checks"][
                "dataset_specific_normalization_loaded"
            ],
        },
        "training_smoke": {
            "status": smoke["status"],
            "steps": smoke["steps"],
            "initial_loss": smoke["loss"]["initial"],
            "final_loss": smoke["loss"]["final"],
            "prediction_shape": smoke["checkpoint_audit"]["prediction_shape"],
            "output": str(SMOKE_OUTPUT.resolve()),
            "research_policy": False,
        },
        "full_policy_b_training": {
            "status": "NOT_STARTED_BY_DESIGN",
            "output_absent": not REAL_OUTPUT.exists(),
            "config": str(FULL_CONFIG.resolve()),
            "exact_command": exact_training_command(),
        },
        "artifacts": artifacts,
        "preflight": {
            "status": preflight["status"],
            "full_config_sha256": sha256_file(FULL_CONFIG),
        },
        "revision": revision,
    }
    required = (
        manifest["dataset"]["episodes"] == 50
        and manifest["dataset"]["frames"] == 34478
        and manifest["state_action_leakage"]["structural_same_row_alias_removed"]
        and manifest["final_action"]["unchanged_from_archive"]
        and manifest["training_smoke"]["status"] == "PASS"
        and manifest["training_smoke"]["prediction_shape"] == [1, 50, 28]
        and manifest["full_policy_b_training"]["output_absent"]
    )
    if not required:
        raise RuntimeError("final semantic manifest invariant failure")
    path = AUDIT / "FINAL_DATASET_B_SEMANTIC_MANIFEST.json"
    atomic_json(path, manifest)
    digest = sha256_file(path)
    atomic_text(
        AUDIT / "FINAL_DATASET_B_SEMANTIC_MANIFEST.sha256",
        f"{digest}  FINAL_DATASET_B_SEMANTIC_MANIFEST.json\n",
    )
    atomic_text(AUDIT / "FINAL_REPORT.txt", final_report(manifest))
    result = {
        "status": "DATASET_B_SEMANTICALLY_READY_FOR_POLICY_TRAINING",
        "manifest": str(path.resolve()),
        "manifest_sha256": digest,
        "dataset_tree_sha256": tree_sha,
        "archive_tree_sha256": archive_sha,
        "training_smoke": "PASS",
        "full_policy_b_training": "NOT_STARTED_BY_DESIGN",
    }
    atomic_json(AUDIT / "final/final_freeze_validation.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("audit", "revise", "validate", "prepare-training", "smoke", "freeze"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "audit":
        result = audit_stage()
    elif args.stage == "revise":
        result = revise_stage()
    elif args.stage == "validate":
        result = validate_stage()
    elif args.stage == "prepare-training":
        result = prepare_training_stage()
    elif args.stage == "smoke":
        result = smoke_stage()
    elif args.stage == "freeze":
        result = freeze_stage()
    else:  # pragma: no cover
        raise ValueError(args.stage)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
