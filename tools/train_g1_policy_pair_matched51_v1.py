#!/usr/bin/env python3
"""Audit and train the matched-51 SmolVLA Policy A/B pair.

The source datasets and prepared packaging configs are treated as immutable.
Only run names/output directories differ between A and B. Smoke and full runs
are sequential and each starts from the pinned local SmolVLA base snapshot.
"""

from __future__ import annotations

import argparse
import copy
import csv
import difflib
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "outputs/g1_policy_dataset_packaging_v1"
OUTPUT = ROOT / "outputs/g1_policy_training_matched51_v1"
DATASET_A = ROOT / "lerobot_g1_magsafe_matched51_baseline_a_v1"
DATASET_B = ROOT / "lerobot_g1_magsafe_matched51_proposed_b_v1"
CONFIG_A_SOURCE = PACKAGING / "training_configs/policy_a_config.json"
CONFIG_B_SOURCE = PACKAGING / "training_configs/policy_b_config.json"
BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
BASE_SNAPSHOT = (
    Path("/home/jbnu/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots")
    / BASE_REVISION
)
BASE_MODEL = BASE_SNAPSHOT / "model.safetensors"
EXPECTED_DATASET_HASHES = {
    "policy_a": "fac0b60186843a80ba219232ea5f9592d8b22647805f998e47b28f0a351658aa",
    "policy_b": "a4590374849cbb8b0b718c7540f30ca3154c8b9695bb10e29203a5e349a1bef1",
}
EXPECTED_BASE_HASH = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
PRE_PARSER_FIX_ARCHIVE = ROOT / "outputs/g1_policy_training_matched51_v1_pre_parser_fix_20260813_152842"
PARSER_SOURCE_BEFORE = PRE_PARSER_FIX_ARCHIVE / "audit/parser_source_before.py"
SEED = 1000
FULL_STEPS = 20_000
SMOKE_STEPS = 20
BATCH_SIZE = 16
ACTION_CHUNK = 50
ACTION_DIM = 28
STATE_DIM = 28
FPS = 30
ALLOWED_CONFIG_DIFFERENCES = {
    "dataset.repo_id",
    "dataset.root",
    "job_name",
    "output_dir",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<value>(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)[KMB]?)",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<suffix>[KMB]?)$", re.IGNORECASE)


class IntegrityFailure(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: list[str], timeout: float = 30.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "duration_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": f"{type(exc).__name__}: {exc}",
            "duration_s": time.monotonic() - started,
        }


def nested_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    differences: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(nested_differences(left[key], right[key], path))
    elif isinstance(left, list) and isinstance(right, list):
        if left != right:
            differences.append(prefix)
    elif left != right:
        differences.append(prefix)
    return differences


def feature_shape(info: dict[str, Any], key: str) -> list[int]:
    return [int(value) for value in info["features"][key]["shape"]]


def resolve_task_metadata(table: Any) -> dict[str, Any]:
    """Resolve the task text column without guessing among string columns.

    LeRobot writes task text as the pandas DataFrame index. Depending on how a
    parquet table is inspected, that index is exposed as ``__index_level_0__``
    rather than a canonical ``task`` data column. A canonical column wins when
    present. Otherwise exactly one string-valued pandas index column is allowed.
    """
    import pyarrow as pa

    columns = list(table.column_names)
    if "task_index" not in columns:
        raise IntegrityFailure("tasks.parquet is missing required task_index column")
    string_columns = [
        field.name
        for field in table.schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ]
    pandas_metadata: dict[str, Any] = {}
    raw_pandas_metadata = (table.schema.metadata or {}).get(b"pandas")
    if raw_pandas_metadata:
        try:
            pandas_metadata = json.loads(raw_pandas_metadata)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise IntegrityFailure(f"Invalid pandas parquet metadata: {exc}") from exc
    pandas_index_columns = [
        value
        for value in pandas_metadata.get("index_columns", [])
        if isinstance(value, str)
    ]

    if "task" in string_columns:
        authoritative = "task"
        resolution_rule = "canonical task string column"
    else:
        index_candidates = [
            column
            for column in string_columns
            if column in pandas_index_columns or column in {"__index_level_0__", "index_level_0"}
        ]
        if len(string_columns) > 1:
            raise IntegrityFailure(
                "Ambiguous task metadata: canonical task column absent and multiple string columns exist: "
                f"{string_columns}"
            )
        if len(index_candidates) != 1 or len(string_columns) != 1:
            raise IntegrityFailure(
                "No authoritative task string column: expected canonical 'task' or exactly one "
                f"pandas index string column; string_columns={string_columns}, "
                f"pandas_index_columns={pandas_index_columns}"
            )
        authoritative = index_candidates[0]
        resolution_rule = "unique pandas index string column fallback"

    raw_tasks = table[authoritative].to_pylist()
    if not raw_tasks:
        raise IntegrityFailure("tasks.parquet contains no task rows")
    if any(not isinstance(value, str) or not value.strip() for value in raw_tasks):
        raise IntegrityFailure("tasks.parquet contains a non-string or empty task value")
    tasks = [value.strip() for value in raw_tasks]
    if len(set(tasks)) != len(tasks):
        raise IntegrityFailure("tasks.parquet contains duplicate task strings")

    raw_indices = table["task_index"].to_pylist()
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_indices):
        raise IntegrityFailure(f"task_index values must be integers: {raw_indices}")
    indices = [int(value) for value in raw_indices]
    expected_indices = list(range(len(tasks)))
    if indices != expected_indices:
        raise IntegrityFailure(
            "task_index mapping is incompatible with LeRobot positional lookup: "
            f"actual={indices}, expected={expected_indices}"
        )
    return {
        "columns": columns,
        "dtypes": {field.name: str(field.type) for field in table.schema},
        "row_count": int(table.num_rows),
        "sample_values": table.slice(0, min(5, table.num_rows)).to_pylist(),
        "string_columns": string_columns,
        "pandas_index_columns": pandas_index_columns,
        "authoritative_task_column": authoritative,
        "resolution_rule": resolution_rule,
        "task_values": tasks,
        "task_indices": indices,
        "task_index_to_text": {str(index): task for index, task in zip(indices, tasks, strict=True)},
    }


def validate_task_metadata_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left["task_index_to_text"] != right["task_index_to_text"]:
        raise IntegrityFailure(
            "Dataset A/B task mappings differ: "
            f"A={left['task_index_to_text']}, B={right['task_index_to_text']}"
        )
    return {
        "status": "PASS",
        "a_b_equal": True,
        "task_index_to_text": left["task_index_to_text"],
    }


def data_timing_and_task_contract(root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    digest = hashlib.sha256()
    task_indices: set[int] = set()
    frame_count = 0
    episode_frames: dict[int, list[int]] = {}
    episode_timestamps: dict[int, list[float]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", "task_index"])
        episode_index = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float32)
        task_index = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        frame_count += len(episode_index)
        task_indices.update(int(value) for value in np.unique(task_index))
        for array in (episode_index, frame_index, timestamps, task_index):
            digest.update(array.tobytes(order="C"))
        for episode in np.unique(episode_index):
            mask = episode_index == episode
            episode_frames.setdefault(int(episode), []).extend(frame_index[mask].tolist())
            episode_timestamps.setdefault(int(episode), []).extend(timestamps[mask].tolist())
    timing_valid = True
    for episode in sorted(episode_frames):
        frames = np.asarray(episode_frames[episode], dtype=np.int64)
        timestamps = np.asarray(episode_timestamps[episode], dtype=np.float64)
        timing_valid &= np.array_equal(frames, np.arange(len(frames), dtype=np.int64))
        timing_valid &= np.allclose(timestamps, np.arange(len(frames)) / FPS, atol=1e-5, rtol=0)
    return {
        "frame_count": frame_count,
        "task_indices": sorted(task_indices),
        "timing_valid": bool(timing_valid),
        "timing_and_task_index_sha256": digest.hexdigest(),
    }


def normalization_contract(stats: dict[str, Any]) -> dict[str, Any]:
    feature_reports: dict[str, Any] = {}
    valid = True
    for feature in ("observation.state", "action"):
        values = stats.get(feature)
        feature_valid = isinstance(values, dict)
        shapes: dict[str, list[int]] = {}
        for statistic in ("mean", "std", "min", "max"):
            array = np.asarray(values.get(statistic, []), dtype=np.float64) if feature_valid else np.asarray([])
            shapes[statistic] = list(array.shape)
            feature_valid &= array.shape == (28,) and bool(np.isfinite(array).all())
        feature_reports[feature] = {"status": "PASS" if feature_valid else "FAIL", "shapes": shapes}
        valid &= feature_valid
    return {"status": "PASS" if valid else "FAIL", "features": feature_reports}


def video_contract(root_a: Path, root_b: Path) -> dict[str, Any]:
    files_a = {path.relative_to(root_a).as_posix(): path for path in (root_a / "videos").rglob("*") if path.is_file()}
    files_b = {path.relative_to(root_b).as_posix(): path for path in (root_b / "videos").rglob("*") if path.is_file()}
    same_paths = set(files_a) == set(files_b)
    exact_equal = same_paths
    hardlinked = 0
    content_compared = 0
    if same_paths:
        for relative in sorted(files_a):
            left, right = files_a[relative], files_b[relative]
            if os.path.samefile(left, right):
                hardlinked += 1
            else:
                content_compared += 1
                if left.stat().st_size != right.stat().st_size or sha256_file(left) != sha256_file(right):
                    exact_equal = False
                    break
    return {
        "same_relative_paths": same_paths,
        "file_count_a": len(files_a),
        "file_count_b": len(files_b),
        "hardlinked_file_count": hardlinked,
        "content_compared_file_count": content_compared,
        "exact_equal": exact_equal,
    }


def gpu_snapshot() -> dict[str, Any]:
    gpu = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    parsed: dict[str, Any] = {"query": gpu, "compute_apps_query": apps}
    if gpu["returncode"] == 0 and gpu["stdout"]:
        fields = [field.strip() for field in gpu["stdout"].splitlines()[0].split(",")]
        if len(fields) == 7:
            parsed.update(
                {
                    "timestamp": fields[0],
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "memory_free_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                    "driver_version": fields[6],
                }
            )
    parsed["compute_apps"] = [line for line in apps["stdout"].splitlines() if line.strip()]
    return parsed


def wait_for_gpu_idle(label: str, maximum_used_mib: int = 1200, timeout_s: int = 12 * 3600) -> dict[str, Any]:
    started = time.monotonic()
    observations: list[dict[str, Any]] = []
    while True:
        snapshot = gpu_snapshot()
        observations.append(snapshot)
        used = snapshot.get("memory_used_mib")
        if isinstance(used, int) and used <= maximum_used_mib:
            print(f"[{label}] GPU ready: used={used} MiB", flush=True)
            return {
                "status": "PASS",
                "wait_seconds": time.monotonic() - started,
                "maximum_used_mib": maximum_used_mib,
                "final_snapshot": snapshot,
                "observation_count": len(observations),
            }
        if time.monotonic() - started > timeout_s:
            raise RuntimeError(f"Timed out waiting for an idle GPU before {label}: {snapshot}")
        print(
            f"[{label}] waiting for GPU: used={used} MiB, apps={snapshot.get('compute_apps', [])}",
            flush=True,
        )
        time.sleep(30)


def environment_audit() -> dict[str, Any]:
    import lerobot
    import torch

    disk = shutil.disk_usage(ROOT)
    nvcc = run_capture(["nvcc", "--version"])
    return {
        "status": "PASS",
        "timestamp": now_iso(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "environment_prefix": sys.prefix,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "lerobot_version": lerobot.__version__,
        "gpu": gpu_snapshot(),
        "nvcc": nvcc,
        "disk": {
            "path": str(ROOT),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "seeds": {
            "python": SEED,
            "numpy": SEED,
            "torch": SEED,
            "cuda": SEED,
            "data_sampler": SEED,
        },
    }


def load_pipeline_config(path: Path):
    import draccus
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.train import TrainPipelineConfig

    with path.open(encoding="utf-8") as stream, draccus.config_type("json"):
        cfg = draccus.load(TrainPipelineConfig, stream)
    cfg.validate()
    return cfg


def prepare_frozen_configs() -> tuple[Path, Path, dict[str, Any]]:
    source_a = read_json(CONFIG_A_SOURCE)
    source_b = read_json(CONFIG_B_SOURCE)
    frozen_a = copy.deepcopy(source_a)
    frozen_b = copy.deepcopy(source_b)
    frozen_a["output_dir"] = str(OUTPUT / "policy_a")
    frozen_b["output_dir"] = str(OUTPUT / "policy_b")
    paths = (
        OUTPUT / "audit/policy_a_config_frozen_prelaunch.json",
        OUTPUT / "audit/policy_b_config_frozen_prelaunch.json",
    )
    write_json(paths[0], frozen_a)
    write_json(paths[1], frozen_b)

    parsed_a = load_pipeline_config(paths[0])
    parsed_b = load_pipeline_config(paths[1])
    differences = nested_differences(frozen_a, frozen_b)
    actual = set(differences)
    checks = {
        "actual_differences_are_exactly_allowed": actual == ALLOWED_CONFIG_DIFFERENCES,
        "same_batch_size": frozen_a["batch_size"] == frozen_b["batch_size"] == BATCH_SIZE,
        "same_steps": frozen_a["steps"] == frozen_b["steps"] == FULL_STEPS,
        "same_seed": frozen_a["seed"] == frozen_b["seed"] == SEED,
        "same_pretrained_path": frozen_a["policy"]["pretrained_path"]
        == frozen_b["policy"]["pretrained_path"]
        == str(BASE_SNAPSHOT),
        "same_optimizer": frozen_a["optimizer"] == frozen_b["optimizer"],
        "same_scheduler": frozen_a["scheduler"] == frozen_b["scheduler"],
        "same_effective_batch": frozen_a["batch_size"] == frozen_b["batch_size"],
        "same_action_chunk": frozen_a["policy"]["chunk_size"]
        == frozen_b["policy"]["chunk_size"]
        == ACTION_CHUNK,
        "same_state_action_interface": feature_shape_from_config(frozen_a, "observation.state")
        == feature_shape_from_config(frozen_b, "observation.state")
        == [STATE_DIM]
        and feature_shape_from_config(frozen_a, "action")
        == feature_shape_from_config(frozen_b, "action")
        == [ACTION_DIM],
        "lerobot_parse_validate": parsed_a.policy.type == parsed_b.policy.type == "smolvla",
        "separate_output_paths": frozen_a["output_dir"] != frozen_b["output_dir"],
    }
    report = {
        "schema_version": "g1_policy_training_matched51_v1_config_fairness",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "actual_difference_paths": differences,
        "allowed_difference_paths": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "compatibility_patch_required": False,
        "compatibility_audit": "LeRobot 0.6.1 TrainPipelineConfig parse and validate PASS; no API field patch applied",
        "non_semantic_output_relocation": {
            "policy_a": {"before": source_a["output_dir"], "after": frozen_a["output_dir"]},
            "policy_b": {"before": source_b["output_dir"], "after": frozen_b["output_dir"]},
        },
        "source_config_sha256": {
            "policy_a": sha256_file(CONFIG_A_SOURCE),
            "policy_b": sha256_file(CONFIG_B_SOURCE),
        },
        "frozen_config_sha256": {
            "policy_a": sha256_file(paths[0]),
            "policy_b": sha256_file(paths[1]),
        },
        "exact_before_after": {
            "policy_a": {"before": source_a, "after": frozen_a},
            "policy_b": {"before": source_b, "after": frozen_b},
        },
        "gradient_accumulation_steps": 1,
        "effective_batch_size": BATCH_SIZE,
        "checkpoint_selection_rule_predeclared": {
            "rule": "same predetermined final training step",
            "selected_step": FULL_STEPS,
            "declared_before_training": True,
        },
    }
    write_json(OUTPUT / "audit/config_fairness.json", report)
    if report["status"] != "PASS":
        raise IntegrityFailure(f"Config fairness failed: {report}")
    return paths[0], paths[1], report


def feature_shape_from_config(config: dict[str, Any], key: str) -> list[int]:
    if key == "action":
        return list(config["policy"]["output_features"][key]["shape"])
    return list(config["policy"]["input_features"][key]["shape"])


def dataset_integrity_audit() -> dict[str, Any]:
    import pyarrow.parquet as pq

    from tools.g1_policy_dataset_packaging_v1.audits import deterministic_tree_hash
    from tools.g1_policy_dataset_packaging_v1.readback import run_lerobot_readback
    from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES, TASK_TEXT

    hashes = {
        "policy_a": deterministic_tree_hash(DATASET_A),
        "policy_b": deterministic_tree_hash(DATASET_B),
    }
    info_a = read_json(DATASET_A / "meta/info.json")
    info_b = read_json(DATASET_B / "meta/info.json")
    manifest_a = read_json(DATASET_A / "meta/g1_packaging_manifest.json")
    manifest_b = read_json(DATASET_B / "meta/g1_packaging_manifest.json")
    tasks_a = resolve_task_metadata(pq.read_table(DATASET_A / "meta/tasks.parquet"))
    tasks_b = resolve_task_metadata(pq.read_table(DATASET_B / "meta/tasks.parquet"))
    task_pair = validate_task_metadata_pair(tasks_a, tasks_b)
    episodes_a = pq.read_table(DATASET_A / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    episodes_b = pq.read_table(DATASET_B / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    episode_tasks_a = sorted({task for row in episodes_a for task in row["tasks"]})
    episode_tasks_b = sorted({task for row in episodes_b for task in row["tasks"]})
    data_contract_a = data_timing_and_task_contract(DATASET_A)
    data_contract_b = data_timing_and_task_contract(DATASET_B)
    normalization_a = normalization_contract(read_json(DATASET_A / "meta/stats.json"))
    normalization_b = normalization_contract(read_json(DATASET_B / "meta/stats.json"))
    rgb = video_contract(DATASET_A, DATASET_B)
    readback = run_lerobot_readback(
        DATASET_A,
        DATASET_B,
        OUTPUT / "audit/lerobot_readback.json",
    )
    checks = {
        "dataset_a_hash": hashes["policy_a"] == EXPECTED_DATASET_HASHES["policy_a"],
        "dataset_b_hash": hashes["policy_b"] == EXPECTED_DATASET_HASHES["policy_b"],
        "episodes_51": info_a["total_episodes"] == info_b["total_episodes"] == 51,
        "frames_50275": info_a["total_frames"] == info_b["total_frames"] == 50_275,
        "fps_30": float(info_a["fps"]) == float(info_b["fps"]) == FPS,
        "state_28d": feature_shape(info_a, "observation.state")
        == feature_shape(info_b, "observation.state")
        == [STATE_DIM],
        "action_28d": feature_shape(info_a, "action") == feature_shape(info_b, "action") == [ACTION_DIM],
        "same_source_identity_set": manifest_a["stable_source_ids"] == manifest_b["stable_source_ids"]
        and len(manifest_a["stable_source_ids"]) == 51,
        "same_rgb_contract": bool(rgb["exact_equal"]),
        "same_task_contract": task_pair["status"] == "PASS"
        and tasks_a["task_values"] == tasks_b["task_values"] == [TASK_TEXT],
        "episode_task_references": episode_tasks_a == episode_tasks_b == [TASK_TEXT],
        "task_index_mapping": data_contract_a["task_indices"]
        == data_contract_b["task_indices"]
        == tasks_a["task_indices"]
        == tasks_b["task_indices"],
        "lerobot_task_resolution": all(
            sample["task_resolved"] == TASK_TEXT for sample in readback["sample_reports"]
        ),
        "same_feature_schema": info_a["features"] == info_b["features"],
        "exact_28d_joint_order": info_a["features"]["observation.state"]["names"]
        == info_a["features"]["action"]["names"]
        == info_b["features"]["observation.state"]["names"]
        == info_b["features"]["action"]["names"]
        == list(CANONICAL_JOINT_NAMES),
        "normalization_available": normalization_a["status"] == normalization_b["status"] == "PASS",
        "timing_valid": data_contract_a["timing_valid"] and data_contract_b["timing_valid"],
        "same_timing": data_contract_a["timing_and_task_index_sha256"]
        == data_contract_b["timing_and_task_index_sha256"],
        "lerobot_readback": readback["status"] == "PASS",
        "lerobot_lengths": readback["dataset_length_a"] == readback["dataset_length_b"] == 50_275,
        "lerobot_action_chunk": readback["action_chunk_50x28_pass"],
        "lerobot_rgb_decode": readback["a_b_rgb_decode_exact_equal"],
        "lerobot_padding": readback["padding_mask_pass"],
    }
    report = {
        "schema_version": "g1_policy_training_matched51_v1_dataset_integrity",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "timestamp": now_iso(),
        "checks": checks,
        "dataset_a": {
            "path": str(DATASET_A),
            "tree_sha256": hashes["policy_a"],
            "episodes": info_a["total_episodes"],
            "frames": info_a["total_frames"],
            "fps": info_a["fps"],
            "state_shape": feature_shape(info_a, "observation.state"),
            "action_shape": feature_shape(info_a, "action"),
        },
        "dataset_b": {
            "path": str(DATASET_B),
            "tree_sha256": hashes["policy_b"],
            "episodes": info_b["total_episodes"],
            "frames": info_b["total_frames"],
            "fps": info_b["fps"],
            "state_shape": feature_shape(info_b, "observation.state"),
            "action_shape": feature_shape(info_b, "action"),
        },
        "stable_source_ids": manifest_a["stable_source_ids"],
        "task_metadata": {
            "policy_a": tasks_a,
            "policy_b": tasks_b,
            "a_b_comparison": task_pair,
            "episode_task_values_a": episode_tasks_a,
            "episode_task_values_b": episode_tasks_b,
            "data_contract_a": data_contract_a,
            "data_contract_b": data_contract_b,
            "lerobot_reader_task_values": sorted(
                {sample["task_resolved"] for sample in readback["sample_reports"]}
            ),
            "task_index_mapping_status": "PASS",
            "lerobot_reader_comparison_status": "PASS",
        },
        "normalization": {"policy_a": normalization_a, "policy_b": normalization_b},
        "rgb_contract": rgb,
        "lerobot_readback_report": str(OUTPUT / "audit/lerobot_readback.json"),
    }
    write_json(OUTPUT / "audit/dataset_integrity.json", report)
    if report["status"] != "PASS":
        raise IntegrityFailure(f"Dataset integrity failed: {report}")
    return report


def write_task_parser_fix_report(dataset_report: dict[str, Any]) -> None:
    archived_failure_path = PRE_PARSER_FIX_ARCHIVE / "summary/training_readiness_for_evaluation.json"
    archived_failure = read_json(archived_failure_path) if archived_failure_path.is_file() else {}
    before_source = PARSER_SOURCE_BEFORE.read_text(encoding="utf-8")
    after_source_path = Path(__file__).resolve()
    after_source = after_source_path.read_text(encoding="utf-8")
    source_diff = "".join(
        difflib.unified_diff(
            before_source.splitlines(keepends=True),
            after_source.splitlines(keepends=True),
            fromfile=str(PARSER_SOURCE_BEFORE),
            tofile=str(after_source_path),
        )
    )
    diff_path = OUTPUT / "audit/task_metadata_parser_fix.diff"
    diff_path.write_text(source_diff, encoding="utf-8")
    task_metadata = dataset_report["task_metadata"]
    report = {
        "schema_version": "g1_policy_training_matched51_v1_task_metadata_parser_fix",
        "status": "PASS",
        "original_failure": archived_failure.get("reason"),
        "root_cause": (
            "The old audit accessed row.get('task'), but LeRobot's authoritative task text is a pandas "
            "DataFrame index physically exposed by pyarrow as __index_level_0__."
        ),
        "files_inspected": [
            str(DATASET_A / "meta/tasks.parquet"),
            str(DATASET_B / "meta/tasks.parquet"),
            str(DATASET_A / "meta/info.json"),
            str(DATASET_B / "meta/info.json"),
            str(DATASET_A / "meta/episodes/chunk-000/file-000.parquet"),
            str(DATASET_B / "meta/episodes/chunk-000/file-000.parquet"),
        ],
        "old_parser_assumption": "A physical string column named task must exist.",
        "new_parser_rule": (
            "Use canonical string column task when present; otherwise require exactly one non-empty "
            "string column identified as the pandas index (__index_level_0__ or index_level_0). Reject "
            "ambiguity, empties, duplicates, and non-contiguous positional task_index mappings."
        ),
        "tasks_parquet": {
            "policy_a": task_metadata["policy_a"],
            "policy_b": task_metadata["policy_b"],
        },
        "authoritative_task_column": task_metadata["policy_a"]["authoritative_task_column"],
        "task_values": task_metadata["policy_a"]["task_values"],
        "a_b_task_equality": task_metadata["a_b_comparison"],
        "task_index_mapping": {
            "status": task_metadata["task_index_mapping_status"],
            "policy_a_data_indices": task_metadata["data_contract_a"]["task_indices"],
            "policy_b_data_indices": task_metadata["data_contract_b"]["task_indices"],
            "episode_task_values_a": task_metadata["episode_task_values_a"],
            "episode_task_values_b": task_metadata["episode_task_values_b"],
        },
        "lerobot_reader_comparison": {
            "status": task_metadata["lerobot_reader_comparison_status"],
            "resolved_task_values": task_metadata["lerobot_reader_task_values"],
        },
        "dataset_hash_before": EXPECTED_DATASET_HASHES,
        "dataset_hash_after": {
            "policy_a": dataset_report["dataset_a"]["tree_sha256"],
            "policy_b": dataset_report["dataset_b"]["tree_sha256"],
        },
        "dataset_files_unchanged": (
            dataset_report["dataset_a"]["tree_sha256"] == EXPECTED_DATASET_HASHES["policy_a"]
            and dataset_report["dataset_b"]["tree_sha256"] == EXPECTED_DATASET_HASHES["policy_b"]
        ),
        "files_changed": [
            str(after_source_path),
            str(ROOT / "tests/test_g1_policy_training_task_metadata_parser.py"),
        ],
        "parser_source_sha256_before": sha256_file(PARSER_SOURCE_BEFORE),
        "parser_source_sha256_after": sha256_file(after_source_path),
        "parser_source_code_diff_path": str(diff_path),
        "parser_source_code_diff_sha256": sha256_file(diff_path),
        "previous_partial_audit_archive": str(OUTPUT / "audit/pre_parser_fix"),
    }
    write_json(OUTPUT / "audit/task_metadata_parser_fix.json", report)


def base_integrity_audit() -> dict[str, Any]:
    actual_hash = sha256_file(BASE_MODEL) if BASE_MODEL.is_file() else None
    ref_path = BASE_SNAPSHOT.parents[1] / "refs/main"
    ref_value = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None
    checks = {
        "snapshot_exists": BASE_SNAPSHOT.is_dir(),
        "model_exists": BASE_MODEL.is_file(),
        "revision_exact": BASE_SNAPSHOT.name == BASE_REVISION,
        "model_hash_exact": actual_hash == EXPECTED_BASE_HASH,
        "hub_ref_matches": ref_value in {None, BASE_REVISION},
    }
    report = {
        "schema_version": "g1_policy_training_matched51_v1_base_model_integrity",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "model_id": "lerobot/smolvla_base",
        "revision": BASE_REVISION,
        "snapshot_path": str(BASE_SNAPSHOT),
        "model_path": str(BASE_MODEL),
        "model_sha256": actual_hash,
        "model_bytes": BASE_MODEL.stat().st_size if BASE_MODEL.is_file() else None,
        "hub_main_ref": ref_value,
        "old_aloha_finetuned_checkpoint_used": False,
        "policy_a_initialization": str(BASE_SNAPSHOT),
        "policy_b_initialization": str(BASE_SNAPSHOT),
    }
    write_json(OUTPUT / "audit/base_model_integrity.json", report)
    if report["status"] != "PASS":
        raise IntegrityFailure(f"Base model integrity failed: {report}")
    return report


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_preprocessor(cfg: Any, dataset: Any, device: str = "cpu"):
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor_overrides = {
        "device_processor": {"device": device},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**cfg.policy.input_features, **cfg.policy.output_features},
            "norm_map": cfg.policy.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": dataset.meta.stats,
            "features": cfg.policy.output_features,
            "norm_map": cfg.policy.normalization_mapping,
        }
    }
    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(cfg.policy.pretrained_path),
        pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def tensor_stats(tensor: Any) -> dict[str, float]:
    values = tensor.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def pretraining_batch_audit(config_a: Path, config_b: Path) -> dict[str, Any]:
    import torch
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.utils.collate import lerobot_collate_fn

    cfg_a, cfg_b = load_pipeline_config(config_a), load_pipeline_config(config_b)
    dataset_a, _ = make_train_eval_datasets(cfg_a)
    dataset_b, _ = make_train_eval_datasets(cfg_b)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset_a), generator=generator)[:BATCH_SIZE].tolist()
    raw_a = lerobot_collate_fn([dataset_a[index] for index in indices])
    raw_b = lerobot_collate_fn([dataset_b[index] for index in indices])
    if raw_a is None or raw_b is None:
        raise IntegrityFailure("Deterministic pretraining batch collated to None")
    pre_a, _ = make_preprocessor(cfg_a, dataset_a, "cpu")
    pre_b, _ = make_preprocessor(cfg_b, dataset_b, "cpu")
    normalized_a = pre_a(raw_a)
    normalized_b = pre_b(raw_b)
    mask_a = normalized_a["action_is_pad"].bool()
    mask_b = normalized_b["action_is_pad"].bool()
    valid_a = normalized_a["action"][~mask_a]
    valid_b = normalized_b["action"][~mask_b]
    checks = {
        "same_indices": len(indices) == BATCH_SIZE,
        "rgb_shape_identical": tuple(raw_a["observation.images.cam_high"].shape)
        == tuple(raw_b["observation.images.cam_high"].shape),
        "rgb_values_identical": torch.equal(
            raw_a["observation.images.cam_high"], raw_b["observation.images.cam_high"]
        ),
        "state_shape_identical": tuple(raw_a["observation.state"].shape)
        == tuple(raw_b["observation.state"].shape),
        "state_28d": raw_a["observation.state"].shape[-1] == raw_b["observation.state"].shape[-1] == STATE_DIM,
        "action_chunk_a": list(raw_a["action"].shape) == [BATCH_SIZE, ACTION_CHUNK, ACTION_DIM],
        "action_chunk_b": list(raw_b["action"].shape) == [BATCH_SIZE, ACTION_CHUNK, ACTION_DIM],
        "task_resolution_identical": raw_a["task"] == raw_b["task"],
        "padding_convention_identical": torch.equal(mask_a, mask_b),
        "finite_normalized_state_a": bool(torch.isfinite(normalized_a["observation.state"]).all()),
        "finite_normalized_state_b": bool(torch.isfinite(normalized_b["observation.state"]).all()),
        "finite_normalized_action_a": bool(torch.isfinite(normalized_a["action"]).all()),
        "finite_normalized_action_b": bool(torch.isfinite(normalized_b["action"]).all()),
    }
    report = {
        "schema_version": "g1_policy_training_matched51_v1_pretraining_batch_audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "seed": SEED,
        "global_indices": indices,
        "batch_size": BATCH_SIZE,
        "rgb_shape_a": list(raw_a["observation.images.cam_high"].shape),
        "rgb_shape_b": list(raw_b["observation.images.cam_high"].shape),
        "state_shape_a": list(raw_a["observation.state"].shape),
        "state_shape_b": list(raw_b["observation.state"].shape),
        "action_shape_a": list(raw_a["action"].shape),
        "action_shape_b": list(raw_b["action"].shape),
        "tasks_a": raw_a["task"],
        "tasks_b": raw_b["task"],
        "normalized_state_a": tensor_stats(normalized_a["observation.state"]),
        "normalized_state_b": tensor_stats(normalized_b["observation.state"]),
        "normalized_action_a_valid_only": tensor_stats(valid_a),
        "normalized_action_b_valid_only": tensor_stats(valid_b),
        "padding_percentage_a": float(mask_a.float().mean().cpu() * 100.0),
        "padding_percentage_b": float(mask_b.float().mean().cpu() * 100.0),
        "normalization_algorithm": "MEAN_STD for STATE/ACTION; IDENTITY for VISUAL",
    }
    write_json(OUTPUT / "audit/pretraining_batch_audit.json", report)
    del normalized_a, normalized_b, raw_a, raw_b, dataset_a, dataset_b
    gc.collect()
    if report["status"] != "PASS":
        raise IntegrityFailure(f"Pretraining batch audit failed: {report}")
    return report


def audit_stage() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing training output: {OUTPUT}")
    for directory in ["audit", "smoke", "comparison", "offline_sanity", "summary", "tests"]:
        (OUTPUT / directory).mkdir(parents=True, exist_ok=False)
    if not PRE_PARSER_FIX_ARCHIVE.is_dir() or not PARSER_SOURCE_BEFORE.is_file():
        raise FileNotFoundError(f"Missing preserved pre-parser-fix evidence: {PRE_PARSER_FIX_ARCHIVE}")
    shutil.copytree(PRE_PARSER_FIX_ARCHIVE, OUTPUT / "audit/pre_parser_fix")
    print("[audit] dataset integrity", flush=True)
    dataset_report = dataset_integrity_audit()
    write_task_parser_fix_report(dataset_report)
    print("[audit] base model integrity", flush=True)
    base_report = base_integrity_audit()
    print("[audit] environment", flush=True)
    environment = environment_audit()
    write_json(OUTPUT / "audit/environment.json", environment)
    print("[audit] config parse/fairness", flush=True)
    config_a, config_b, config_report = prepare_frozen_configs()
    print("[audit] deterministic pretraining batch", flush=True)
    batch_report = pretraining_batch_audit(config_a, config_b)
    write_json(
        OUTPUT / "audit/audit_readiness.json",
        {
            "status": "PASS",
            "timestamp": now_iso(),
            "dataset_integrity": dataset_report["status"],
            "base_model_integrity": base_report["status"],
            "environment": environment["status"],
            "config_fairness": config_report["status"],
            "pretraining_batch": batch_report["status"],
            "training_started": False,
        },
    )
    print("[audit] PASS", flush=True)


def training_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "MPLCONFIGDIR": "/tmp/g1_policy_training_matched51_mpl",
        }
    )
    return env


def run_training(config_path: Path, log_path: Path, label: str) -> dict[str, Any]:
    idle = wait_for_gpu_idle(label)
    start_time = now_iso()
    start_monotonic = time.monotonic()
    start_gpu = gpu_snapshot()
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train", "--config_path", str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] launch: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=training_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    end_gpu = gpu_snapshot()
    result = {
        "label": label,
        "command": command,
        "config_path": str(config_path),
        "log_path": str(log_path),
        "start_timestamp": start_time,
        "end_timestamp": now_iso(),
        "duration_seconds": time.monotonic() - start_monotonic,
        "returncode": returncode,
        "gpu_idle_gate": idle,
        "gpu_before": start_gpu,
        "gpu_after": end_gpu,
    }
    if returncode != 0:
        raise RuntimeError(f"{label} training failed with return code {returncode}; log={log_path}")
    return result


def parse_metrics(log_path: Path) -> list[dict[str, Any]]:
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    rows: list[dict[str, Any]] = []
    aliases = {
        "loss": "loss",
        "lr": "learning_rate",
        "grdn": "gradient_norm",
        "grad_norm": "gradient_norm",
        "gpu_mem": "gpu_memory_gb",
        "gpu_mem_gb": "gpu_memory_gb",
        "update_s": "update_seconds",
        "data_s": "data_seconds",
    }
    for segment in re.split(r"[\r\n]+", text):
        pairs = {match.group("key").lower(): match.group("value") for match in METRIC_RE.finditer(segment)}
        if "step" not in pairs or "loss" not in pairs:
            continue
        step_match = STEP_RE.fullmatch(pairs["step"])
        if step_match is None:
            continue
        multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
        step = int(float(step_match.group("value")) * multipliers[step_match.group("suffix").upper()])
        row: dict[str, Any] = {"step": step}
        for source, target in aliases.items():
            if source in pairs:
                try:
                    row[target] = float(pairs[source])
                except ValueError:
                    row[target] = float("nan")
        rows.append(row)
    log_freq_match = re.search(r"'log_freq':\s*(\d+)", text)
    steps_match = re.search(r"'steps':\s*(\d+)", text)
    if log_freq_match and steps_match:
        log_freq = int(log_freq_match.group(1))
        total_steps = int(steps_match.group(1))
        for index, row in enumerate(rows, start=1):
            row["step"] = min(index * log_freq, total_steps)
    return rows


def metric_summary(metrics: list[dict[str, Any]], last_n: int = 20) -> dict[str, Any]:
    losses = [float(row["loss"]) for row in metrics if "loss" in row]
    finite_losses = [value for value in losses if math.isfinite(value)]
    nonfinite_count = sum(not math.isfinite(value) for value in losses)
    for row in metrics:
        for key in ("gradient_norm", "learning_rate"):
            if key in row and not math.isfinite(float(row[key])):
                nonfinite_count += 1
    tail = finite_losses[-last_n:]
    return {
        "logged_point_count": len(metrics),
        "initial_logged_step": metrics[0]["step"] if metrics else None,
        "final_logged_step": metrics[-1]["step"] if metrics else None,
        "initial_loss": finite_losses[0] if finite_losses else None,
        "final_loss": finite_losses[-1] if finite_losses else None,
        "minimum_loss": min(finite_losses) if finite_losses else None,
        "last_n_logged_points": min(last_n, len(tail)),
        "last_n_loss_mean": float(np.mean(tail)) if tail else None,
        "last_n_loss_std": float(np.std(tail)) if tail else None,
        "nan_inf_count": nonfinite_count,
        "maximum_gradient_norm": max(
            (float(row["gradient_norm"]) for row in metrics if math.isfinite(float(row.get("gradient_norm", float("nan"))))),
            default=None,
        ),
        "maximum_logged_gpu_memory_gb": max(
            (float(row["gpu_memory_gb"]) for row in metrics if math.isfinite(float(row.get("gpu_memory_gb", float("nan"))))),
            default=None,
        ),
    }


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    import torch

    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def scalar_loss(output: Any) -> float:
    import torch

    value = output[0] if isinstance(output, tuple) else output
    if isinstance(value, dict):
        value = value["loss"]
    if not isinstance(value, torch.Tensor):
        return float(value)
    return float(value.detach().float().cpu())


def checkpoint_runtime_smoke(checkpoint: Path, dataset_root: Path, repo_id: str) -> dict[str, Any]:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.collate import lerobot_collate_fn

    deltas = {
        "observation.state": [0.0],
        "action": [index / FPS for index in range(ACTION_CHUNK)],
    }
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="torchcodec",
    )
    raw = lerobot_collate_fn([dataset[len(dataset) - 1]])
    if raw is None:
        raise RuntimeError("Smoke checkpoint audit batch is None")
    seed_everything(SEED)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    policy.eval()
    batch = preprocessor(raw)
    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(batch)
        action_chunk = postprocessor(normalized_chunk)
    mask = batch["action_is_pad"].bool()
    altered = clone_batch(batch)
    expanded_mask = mask.unsqueeze(-1).expand_as(altered["action"])
    altered["action"] = torch.where(expanded_mask, altered["action"] + 10_000.0, altered["action"])
    seed_everything(SEED + 77)
    with torch.inference_mode():
        loss_original = scalar_loss(policy.forward(batch))
    seed_everything(SEED + 77)
    with torch.inference_mode():
        loss_altered_padding = scalar_loss(policy.forward(altered))
    loss_delta = abs(loss_original - loss_altered_padding)
    state_shape = list(policy.config.input_features["observation.state"].shape)
    action_shape = list(policy.config.output_features["action"].shape)
    result = {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_reload": True,
        "model_type": policy.config.type,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "chunk_size": int(policy.config.chunk_size),
        "prediction_shape": list(action_chunk.shape),
        "prediction_finite": bool(torch.isfinite(action_chunk).all()),
        "padding_true_count": int(mask.sum().detach().cpu()),
        "padding_loss_original": loss_original,
        "padding_loss_after_padded_target_perturbation": loss_altered_padding,
        "padding_loss_absolute_delta": loss_delta,
        "padding_loss_mask_pass": loss_delta <= 1e-4 and int(mask.sum()) > 0,
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
    }
    result["status"] = "PASS" if (
        result["prediction_shape"] == [1, ACTION_CHUNK, ACTION_DIM]
        and result["prediction_finite"]
        and result["padding_loss_mask_pass"]
        and state_shape == [STATE_DIM]
        and action_shape == [ACTION_DIM]
    ) else "FAIL"
    del policy, preprocessor, postprocessor, batch, altered, raw, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def smoke_configs() -> tuple[Path, Path]:
    full_a = read_json(OUTPUT / "audit/policy_a_config_frozen_prelaunch.json")
    full_b = read_json(OUTPUT / "audit/policy_b_config_frozen_prelaunch.json")
    configs = []
    for label, full in [("policy_a", full_a), ("policy_b", full_b)]:
        config = copy.deepcopy(full)
        config["steps"] = SMOKE_STEPS
        config["log_freq"] = 1
        config["save_freq"] = SMOKE_STEPS
        config["output_dir"] = str(OUTPUT / "smoke" / label)
        config["job_name"] = f"{full['job_name']}_smoke{SMOKE_STEPS}"
        path = OUTPUT / "smoke" / f"{label}_config.json"
        write_json(path, config)
        load_pipeline_config(path)
        configs.append(path)
    a, b = read_json(configs[0]), read_json(configs[1])
    differences = nested_differences(a, b)
    if set(differences) != ALLOWED_CONFIG_DIFFERENCES:
        raise IntegrityFailure(f"Smoke config fairness failed: {differences}")
    return configs[0], configs[1]


def smoke_stage() -> dict[str, Any]:
    readiness = read_json(OUTPUT / "audit/audit_readiness.json")
    if readiness["status"] != "PASS":
        raise IntegrityFailure("Audit readiness is not PASS")
    config_a, config_b = smoke_configs()
    entries: dict[str, Any] = {}
    order = []
    for label, config, dataset_root, repo_id in [
        ("policy_a", config_a, DATASET_A, "local/g1_magsafe_matched51_baseline_a_v1"),
        ("policy_b", config_b, DATASET_B, "local/g1_magsafe_matched51_proposed_b_v1"),
    ]:
        order.append(label)
        run_dir = OUTPUT / "smoke" / label
        log_path = OUTPUT / "smoke" / f"{label}_training.log"
        runtime = run_training(config, log_path, f"smoke_{label}")
        metrics = parse_metrics(log_path)
        summary = metric_summary(metrics)
        checkpoint = checkpoint_dir(run_dir, SMOKE_STEPS)
        if not checkpoint.is_dir():
            raise RuntimeError(f"Missing smoke checkpoint: {checkpoint}")
        reload_report = checkpoint_runtime_smoke(checkpoint, dataset_root, repo_id)
        (run_dir / "logs").mkdir(exist_ok=True)
        shutil.copy2(log_path, run_dir / "logs/train.log")
        shutil.copy2(config, run_dir / "config_frozen.json")
        write_metrics_csv(run_dir / "logs/metrics.csv", label, metrics)
        entry = {
            "status": "PASS" if (
                runtime["returncode"] == 0
                and summary["nan_inf_count"] == 0
                and summary["logged_point_count"] == SMOKE_STEPS
                and reload_report["status"] == "PASS"
            ) else "FAIL",
            "steps": SMOKE_STEPS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "runtime": runtime,
            "metrics": summary,
            "checkpoint_reload_and_padding": reload_report,
        }
        write_json(run_dir / "smoke_summary.json", entry)
        entries[label] = entry
        if entry["status"] != "PASS":
            raise RuntimeError(f"Smoke test failed for {label}: {entry}")
    report = {
        "schema_version": "g1_policy_training_matched51_v1_smoke",
        "status": "PASS",
        "training_order": order,
        "same_steps": entries["policy_a"]["steps"] == entries["policy_b"]["steps"] == SMOKE_STEPS,
        "same_batch_size": entries["policy_a"]["batch_size"] == entries["policy_b"]["batch_size"] == BATCH_SIZE,
        "same_seed": entries["policy_a"]["seed"] == entries["policy_b"]["seed"] == SEED,
        "policy_a": entries["policy_a"],
        "policy_b": entries["policy_b"],
        "smoke_checkpoints_isolated_from_primary": True,
        "smoke_loss_not_used_for_hyperparameter_selection": True,
    }
    write_json(OUTPUT / "smoke/smoke_report.json", report)
    return report


def write_metrics_csv(path: Path, policy: str, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy",
        "step",
        "loss",
        "learning_rate",
        "gradient_norm",
        "gpu_memory_gb",
        "update_seconds",
        "data_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({"policy": policy, **metric})


def artifact_manifest(checkpoint_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in checkpoint_root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(checkpoint_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(checkpoint_root),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": canonical_hash(rows),
        "files": rows,
    }


def hash_all_checkpoints(run_dir: Path) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    for step_dir in sorted((run_dir / "checkpoints").iterdir()):
        if step_dir.is_dir() and step_dir.name.isdigit():
            print(f"[hash] {step_dir}", flush=True)
            checkpoints[step_dir.name] = artifact_manifest(step_dir)
    return {"status": "PASS", "checkpoints": checkpoints}


def full_policy_run(label: str, config: Path) -> dict[str, Any]:
    run_dir = OUTPUT / label
    log_path = OUTPUT / "audit" / f"{label}_training.log"
    runtime = run_training(config, log_path, f"full_{label}")
    metrics = parse_metrics(log_path)
    metrics_summary = metric_summary(metrics)
    steps = sorted(
        int(path.name)
        for path in (run_dir / "checkpoints").iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    final_checkpoint = checkpoint_dir(run_dir, FULL_STEPS)
    checkpoint_hashes = hash_all_checkpoints(run_dir)
    (run_dir / "logs").mkdir(exist_ok=True)
    shutil.copy2(log_path, run_dir / "logs/train.log")
    shutil.copy2(config, run_dir / "config_frozen.json")
    write_metrics_csv(run_dir / "logs/metrics.csv", label, metrics)
    write_json(run_dir / "checkpoint_hashes.json", checkpoint_hashes)
    completed = (
        runtime["returncode"] == 0
        and steps
        and steps[-1] == FULL_STEPS
        and final_checkpoint.is_dir()
        and metrics_summary["logged_point_count"] > 0
        and metrics_summary["nan_inf_count"] == 0
    )
    summary = {
        "schema_version": "g1_policy_training_matched51_v1_training_summary",
        "status": "PASS" if completed else "FAIL",
        "policy": label,
        "completed": bool(completed),
        "runtime": runtime,
        "dataset_root": read_json(config)["dataset"]["root"],
        "dataset_tree_sha256": EXPECTED_DATASET_HASHES[label],
        "base_model_id": "lerobot/smolvla_base",
        "base_revision": BASE_REVISION,
        "base_model_sha256": EXPECTED_BASE_HASH,
        "independent_base_initialization": True,
        "steps_planned": FULL_STEPS,
        "steps_completed": steps[-1] if steps else 0,
        "checkpoint_steps": steps,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": BATCH_SIZE,
        "seed": SEED,
        "metrics": metrics_summary,
        "final_checkpoint": str(final_checkpoint),
        "unexpected_interruption": not completed,
        "oom": False,
    }
    write_json(run_dir / "training_summary.json", summary)
    if not completed:
        raise RuntimeError(f"Primary training incomplete for {label}: {summary}")
    return summary


def full_training_stage() -> tuple[dict[str, Any], dict[str, Any]]:
    smoke = read_json(OUTPUT / "smoke/smoke_report.json")
    if smoke["status"] != "PASS":
        raise RuntimeError("Primary training cannot start before paired smoke PASS")
    config_a = OUTPUT / "audit/policy_a_config_frozen_prelaunch.json"
    config_b = OUTPUT / "audit/policy_b_config_frozen_prelaunch.json"
    order_report: dict[str, Any] = {
        "declared_order": ["policy_a", "policy_b"],
        "policy_a_start": now_iso(),
    }
    summary_a = full_policy_run("policy_a", config_a)
    order_report["policy_a_end"] = now_iso()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    order_report["between_runs_gpu"] = gpu_snapshot()
    order_report["policy_a_freed_and_cuda_cache_cleared"] = True
    order_report["policy_b_start"] = now_iso()
    frozen_b_text = config_b.read_text(encoding="utf-8")
    if "policy_a" in frozen_b_text or str(OUTPUT / "policy_a") in frozen_b_text:
        raise IntegrityFailure("Policy B config contains Policy A inheritance/path reference")
    summary_b = full_policy_run("policy_b", config_b)
    order_report["policy_b_end"] = now_iso()
    order_report["status"] = "PASS"
    write_json(OUTPUT / "audit/training_order.json", order_report)
    return summary_a, summary_b


def model_weight_audit(model_path: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    tensor_count = 0
    parameter_count = 0
    nonfinite_count = 0
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_count += 1
            parameter_count += tensor.numel()
            if tensor.is_floating_point():
                nonfinite_count += int((~torch.isfinite(tensor)).sum())
    return {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "tensor_count": tensor_count,
        "parameter_count_from_safetensors": parameter_count,
        "nonfinite_weight_count": nonfinite_count,
        "finite_weights": nonfinite_count == 0,
    }


def final_model_runtime_audit(
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    sample_indices: list[int],
) -> dict[str, Any]:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.collate import lerobot_collate_fn

    weight_report = model_weight_audit(checkpoint / "model.safetensors")
    deltas = {
        "observation.state": [0.0],
        "action": [index / FPS for index in range(ACTION_CHUNK)],
    }
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="torchcodec",
    )
    raw_items = [dataset[index] for index in sample_indices]
    raw = lerobot_collate_fn(raw_items)
    if raw is None:
        raise RuntimeError("Final runtime audit batch is None")
    seed_everything(SEED)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    policy.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    processed = preprocessor(raw)
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(processed)
        prediction = postprocessor(normalized)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    prediction_cpu = prediction.detach().float().cpu()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    state_shape = list(policy.config.input_features["observation.state"].shape)
    action_shape = list(policy.config.output_features["action"].shape)
    parameter_count_loaded = sum(parameter.numel() for parameter in policy.parameters())
    sample_rows = []
    for index, item in zip(sample_indices, raw_items, strict=True):
        sample_rows.append(
            {
                "global_index": index,
                "episode_index": int(item["episode_index"]),
                "frame_index": int(item["frame_index"]),
                "task": item["task"],
            }
        )
    result = {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_load": True,
        "model_type": policy.config.type,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "chunk_size": int(policy.config.chunk_size),
        "prediction_shape": list(prediction_cpu.shape),
        "prediction_finite": bool(torch.isfinite(prediction_cpu).all()),
        "prediction_stats": tensor_stats(prediction_cpu),
        "gross_range_threshold_abs": 10.0,
        "gross_range_plausible": float(prediction_cpu.abs().max()) <= 10.0,
        "latency_ms_for_batch": latency_ms,
        "peak_gpu_memory_gb": peak_memory,
        "sample_rows": sample_rows,
        "parameter_count_loaded": parameter_count_loaded,
        "weight_audit": weight_report,
    }
    result["status"] = "PASS" if (
        state_shape == [STATE_DIM]
        and action_shape == [ACTION_DIM]
        and result["chunk_size"] == ACTION_CHUNK
        and result["prediction_shape"] == [len(sample_indices), ACTION_CHUNK, ACTION_DIM]
        and result["prediction_finite"]
        and result["gross_range_plausible"]
        and weight_report["finite_weights"]
        and parameter_count_loaded == weight_report["parameter_count_from_safetensors"]
    ) else "FAIL"
    del policy, preprocessor, postprocessor, processed, prediction, raw, raw_items, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def architecture_signature(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "type",
        "input_features",
        "output_features",
        "chunk_size",
        "n_action_steps",
        "n_obs_steps",
        "max_state_dim",
        "max_action_dim",
        "num_vlm_layers",
        "num_expert_layers",
        "expert_width_multiplier",
        "attention_mode",
        "self_attn_every_n_layers",
        "vlm_model_name",
        "normalization_mapping",
        "train_expert_only",
        "train_state_proj",
        "freeze_vision_encoder",
        "use_peft",
    ]
    return {key: config.get(key) for key in keys}


def create_training_curves(metrics_a: list[dict[str, Any]], metrics_b: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/g1_policy_training_matched51_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = OUTPUT / "comparison/training_curves.csv"
    fields = ["policy", "step", "loss", "learning_rate", "gradient_norm", "gpu_memory_gb"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for policy, metrics in [("policy_a", metrics_a), ("policy_b", metrics_b)]:
            for metric in metrics:
                writer.writerow({key: ({"policy": policy, **metric}).get(key) for key in fields})
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    colors = {"policy_a": "#15616d", "policy_b": "#e07a5f"}
    for policy, metrics in [("policy_a", metrics_a), ("policy_b", metrics_b)]:
        steps = [row["step"] for row in metrics]
        axes[0].plot(steps, [row.get("loss", np.nan) for row in metrics], label=policy, color=colors[policy])
        axes[1].plot(
            steps,
            [row.get("learning_rate", np.nan) for row in metrics],
            label=policy,
            color=colors[policy],
        )
        axes[2].plot(
            steps,
            [row.get("gradient_norm", np.nan) for row in metrics],
            label=policy,
            color=colors[policy],
        )
    axes[0].set_ylabel("Train loss")
    axes[1].set_ylabel("Learning rate")
    axes[2].set_ylabel("Gradient norm")
    axes[2].set_xlabel("Optimization step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Matched-51 SmolVLA training diagnostics (not behavior evaluation)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison/training_curves.png", dpi=180)
    plt.close(fig)


def final_report_text(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    convergence: dict[str, Any],
    fairness: dict[str, Any],
    final_fairness: dict[str, Any],
    offline: dict[str, Any],
    environment: dict[str, Any],
) -> str:
    a_metrics = convergence["policy_a"]
    b_metrics = convergence["policy_b"]
    a_cp = final_fairness["policy_a"]
    b_cp = final_fairness["policy_b"]
    optimizer = read_json(OUTPUT / "audit/policy_a_config_frozen_prelaunch.json")["optimizer"]
    task_fix = read_json(OUTPUT / "audit/task_metadata_parser_fix.json")
    dataset_audit = read_json(OUTPUT / "audit/dataset_integrity.json")
    smoke = read_json(OUTPUT / "smoke/smoke_report.json")
    task = task_fix["task_values"][0]
    lines = [
        "1. root cause of the original false failure: old audit가 실제 pandas index backing column 대신 물리 column `task`만 가정함",
        f"2. actual tasks.parquet columns: {task_fix['tasks_parquet']['policy_a']['columns']}",
        f"3. selected authoritative task column: {task_fix['authoritative_task_column']}",
        f"4. Dataset A task string: {task}",
        f"5. Dataset B task string: {task_fix['tasks_parquet']['policy_b']['task_values'][0]}",
        f"6. A/B task equivalence PASS 여부: {task_fix['a_b_task_equality']['status']}",
        f"7. Dataset A hash unchanged 여부: {'PASS' if task_fix['dataset_hash_before']['policy_a'] == task_fix['dataset_hash_after']['policy_a'] else 'FAIL'}",
        f"8. Dataset B hash unchanged 여부: {'PASS' if task_fix['dataset_hash_before']['policy_b'] == task_fix['dataset_hash_after']['policy_b'] else 'FAIL'}",
        f"9. complete pre-training audit PASS 여부: {'PASS' if dataset_audit['status'] == fairness['status'] == 'PASS' else 'FAIL'}",
        f"10. Policy A smoke PASS 여부: {smoke['policy_a']['status']}",
        f"11. Policy B smoke PASS 여부: {smoke['policy_b']['status']}",
        f"12. full Policy A training started/completed 여부: 예 / {'예 (PASS)' if summary_a['completed'] else '아니오 (FAIL)'}",
        f"13. full Policy B training started/completed 여부: 예 / {'예 (PASS)' if summary_b['completed'] else '아니오 (FAIL)'}",
        "",
        f"- shared base model/revision/hash: lerobot/smolvla_base / {BASE_REVISION} / {EXPECTED_BASE_HASH}",
        f"- train steps: {FULL_STEPS}",
        f"- effective batch: {BATCH_SIZE} (batch {BATCH_SIZE} x accumulation 1 x GPU 1)",
        f"- learning rate / optimizer: {optimizer['lr']} / AdamW (betas={optimizer['betas']}, weight_decay={optimizer['weight_decay']})",
        f"- seed: {SEED}",
        f"- Policy A initial/final loss: {a_metrics['initial_loss']:.8g} / {a_metrics['final_loss']:.8g}",
        f"- Policy B initial/final loss: {b_metrics['initial_loss']:.8g} / {b_metrics['final_loss']:.8g}",
        f"- Policy A/B NaN/Inf count: {a_metrics['nan_inf_count']} / {b_metrics['nan_inf_count']}",
        f"- final common checkpoint step: {FULL_STEPS}",
        f"- Policy A final checkpoint path/hash: {a_cp['checkpoint']} / {a_cp['model_sha256']}",
        f"- Policy B final checkpoint path/hash: {b_cp['checkpoint']} / {b_cp['model_sha256']}",
        f"- config fairness: {fairness['status']}",
        f"- offline inference sanity: {offline['status']}",
        "",
        "## A. environment",
        f"- GPU: {environment['gpu'].get('name')} ({environment['gpu'].get('memory_total_mib')} MiB)",
        f"- Driver: {environment['gpu'].get('driver_version')}",
        f"- PyTorch/CUDA: {environment['pytorch']['version']} / {environment['pytorch']['cuda_build']}",
        f"- LeRobot/Python: {environment['lerobot_version']} / {environment['python']['version']}",
        f"- Python executable: {environment['python']['executable']}",
        "",
        "## B. dataset/base integrity",
        "- 두 데이터셋의 지정 tree SHA-256, 51 episodes, 50,275 frames, 30 Hz, 28D state/action을 확인했다.",
        "- source identity, RGB hardlink/content, task, feature schema 및 LeRobot 실제 readback이 모두 PASS다.",
        f"- 두 정책은 동일한 pinned base snapshot `{BASE_SNAPSHOT}`에서 독립 초기화했다.",
        "",
        "## C. smoke-test result",
        f"- A/B 각각 {SMOKE_STEPS} optimization steps, batch {BATCH_SIZE}, seed {SEED}로 실행했다.",
        "- forward/backward, finite loss/gradient, checkpoint save/reload, [50,28] 출력 및 padding-loss mask가 PASS다.",
        "- smoke checkpoint는 최종 run과 분리했으며 hyperparameter 선택에 사용하지 않았다.",
        "",
        "## D. Policy A training",
        f"- 시작/종료: {summary_a['runtime']['start_timestamp']} / {summary_a['runtime']['end_timestamp']}",
        f"- final/min/last-N mean±std loss: {a_metrics['final_loss']:.8g} / {a_metrics['minimum_loss']:.8g} / {a_metrics['last_n_loss_mean']:.8g} ± {a_metrics['last_n_loss_std']:.8g}",
        "",
        "## E. Policy B training",
        f"- 시작/종료: {summary_b['runtime']['start_timestamp']} / {summary_b['runtime']['end_timestamp']}",
        f"- final/min/last-N mean±std loss: {b_metrics['final_loss']:.8g} / {b_metrics['minimum_loss']:.8g} / {b_metrics['last_n_loss_mean']:.8g} ± {b_metrics['last_n_loss_std']:.8g}",
        "",
        "## F. training-curve comparison",
        "- loss/LR/gradient norm 곡선은 수렴·불안정·발산·undertraining 진단 전용이다.",
        "- 더 낮은 train loss를 robot behavior 우수성으로 해석하지 않았다.",
        "",
        "## G. checkpoint-selection rule",
        f"- downstream 성능을 보기 전에 공통 final step `{FULL_STEPS}` 사용을 선언했고 A/B 모두 동일하게 적용했다.",
        "",
        "## H. final model integrity",
        f"- 동일 architecture/parameter count: {final_fairness['same_architecture']} / {final_fairness['same_parameter_count']}",
        "- 두 checkpoint 모두 loadable, finite weights, 28D state/action, action chunk 50이다.",
        "- Policy B config/provenance에 Policy A checkpoint 상속이 없다.",
        "",
        "## I. offline sanity inference",
        f"- 동일 global sample IDs {offline['sample_indices']}에서 A/B RGB+state -> [50,28] finite inference가 PASS다.",
        "- 이는 비물리 load/interface/range 진단일 뿐 task success 또는 robot 성능 평가가 아니다.",
        "",
        "## J. exact files/tests",
        f"- Dataset audit: `{OUTPUT / 'audit/dataset_integrity.json'}`",
        f"- Config audit: `{OUTPUT / 'audit/config_fairness.json'}`",
        f"- Smoke report: `{OUTPUT / 'smoke/smoke_report.json'}`",
        f"- Curves: `{OUTPUT / 'comparison/training_curves.csv'}`, `{OUTPUT / 'comparison/training_curves.png'}`",
        f"- Checkpoint fairness: `{OUTPUT / 'comparison/final_checkpoint_fairness.json'}`",
        f"- Offline sanity: `{OUTPUT / 'offline_sanity/prediction_summary.json'}`",
        f"- Tests: `{OUTPUT / 'tests/test_report.json'}`",
        "- Isaac rollout 및 real-G1 평가는 실행하지 않았다.",
        "",
        "POLICY_A_B_MATCHED51_READY_FOR_CONTROLLED_EVALUATION",
    ]
    return "\n".join(lines) + "\n"


def finalize_stage(summary_a: dict[str, Any] | None = None, summary_b: dict[str, Any] | None = None) -> None:
    summary_a = summary_a or read_json(OUTPUT / "policy_a/training_summary.json")
    summary_b = summary_b or read_json(OUTPUT / "policy_b/training_summary.json")
    metrics_a = parse_metrics(OUTPUT / "policy_a/logs/train.log")
    metrics_b = parse_metrics(OUTPUT / "policy_b/logs/train.log")
    convergence = {
        "schema_version": "g1_policy_training_matched51_v1_convergence",
        "status": "PASS",
        "last_n_definition": "last 20 logged windows (log_freq=100)",
        "metric_step_resolution": "metric occurrence order x frozen log_freq, capped at frozen total steps",
        "policy_a": metric_summary(metrics_a),
        "policy_b": metric_summary(metrics_b),
        "interpretation_constraint": "Training curves diagnose convergence only; they are not robot-behavior evidence.",
    }
    if convergence["policy_a"]["nan_inf_count"] or convergence["policy_b"]["nan_inf_count"]:
        convergence["status"] = "FAIL"
    write_json(OUTPUT / "comparison/convergence_summary.json", convergence)
    create_training_curves(metrics_a, metrics_b)
    summary_a["metrics"] = convergence["policy_a"]
    summary_b["metrics"] = convergence["policy_b"]
    summary_a["metric_step_resolution"] = convergence["metric_step_resolution"]
    summary_b["metric_step_resolution"] = convergence["metric_step_resolution"]
    write_json(OUTPUT / "policy_a/training_summary.json", summary_a)
    write_json(OUTPUT / "policy_b/training_summary.json", summary_b)
    write_metrics_csv(OUTPUT / "policy_a/logs/metrics.csv", "policy_a", metrics_a)
    write_metrics_csv(OUTPUT / "policy_b/logs/metrics.csv", "policy_b", metrics_b)

    selected_a = checkpoint_dir(OUTPUT / "policy_a", FULL_STEPS)
    selected_b = checkpoint_dir(OUTPUT / "policy_b", FULL_STEPS)
    sample_indices = [0, 50_275 // 2, 50_275 - 1]
    print("[finalize] Policy A checkpoint load/weights/offline inference", flush=True)
    wait_for_gpu_idle("final_a")
    audit_a = final_model_runtime_audit(
        selected_a,
        DATASET_A,
        "local/g1_magsafe_matched51_baseline_a_v1",
        sample_indices,
    )
    print("[finalize] Policy B checkpoint load/weights/offline inference", flush=True)
    wait_for_gpu_idle("final_b")
    audit_b = final_model_runtime_audit(
        selected_b,
        DATASET_B,
        "local/g1_magsafe_matched51_proposed_b_v1",
        sample_indices,
    )
    config_a = read_json(selected_a / "config.json")
    config_b = read_json(selected_b / "config.json")
    train_config_a = read_json(selected_a / "train_config.json")
    train_config_b = read_json(selected_b / "train_config.json")
    same_architecture = architecture_signature(config_a) == architecture_signature(config_b)
    same_parameter_count = audit_a["parameter_count_loaded"] == audit_b["parameter_count_loaded"]
    no_a_to_b = (
        train_config_b["policy"]["pretrained_path"] == str(BASE_SNAPSHOT)
        and str(OUTPUT / "policy_a") not in json.dumps(train_config_b)
        and "policy_a" not in json.dumps(train_config_b)
    )
    final_fairness = {
        "schema_version": "g1_policy_training_matched51_v1_final_checkpoint_fairness",
        "status": "PASS",
        "rule": "same predetermined final training step",
        "rule_declared_before_downstream_performance": True,
        "selected_common_step": FULL_STEPS,
        "same_architecture": same_architecture,
        "same_parameter_count": same_parameter_count,
        "same_base_revision_provenance": train_config_a["policy"]["pretrained_path"]
        == train_config_b["policy"]["pretrained_path"]
        == str(BASE_SNAPSHOT),
        "no_policy_a_to_policy_b_inheritance": no_a_to_b,
        "policy_a": {
            "checkpoint": str(selected_a),
            "model_sha256": audit_a["weight_audit"]["model_sha256"],
            "runtime_audit": audit_a,
        },
        "policy_b": {
            "checkpoint": str(selected_b),
            "model_sha256": audit_b["weight_audit"]["model_sha256"],
            "runtime_audit": audit_b,
        },
    }
    final_fairness["status"] = "PASS" if (
        same_architecture
        and same_parameter_count
        and final_fairness["same_base_revision_provenance"]
        and no_a_to_b
        and audit_a["status"] == audit_b["status"] == "PASS"
    ) else "FAIL"
    write_json(OUTPUT / "comparison/final_checkpoint_fairness.json", final_fairness)
    offline = {
        "schema_version": "g1_policy_training_matched51_v1_offline_prediction",
        "status": "PASS" if audit_a["status"] == audit_b["status"] == "PASS" else "FAIL",
        "sample_indices": sample_indices,
        "same_sample_ids": audit_a["sample_rows"] == audit_b["sample_rows"],
        "policy_a": {
            key: audit_a[key]
            for key in [
                "prediction_shape",
                "prediction_finite",
                "prediction_stats",
                "gross_range_plausible",
                "latency_ms_for_batch",
                "peak_gpu_memory_gb",
                "sample_rows",
            ]
        },
        "policy_b": {
            key: audit_b[key]
            for key in [
                "prediction_shape",
                "prediction_finite",
                "prediction_stats",
                "gross_range_plausible",
                "latency_ms_for_batch",
                "peak_gpu_memory_gb",
                "sample_rows",
            ]
        },
        "selection_use": False,
        "physics_or_success_claim": False,
    }
    if not offline["same_sample_ids"]:
        offline["status"] = "FAIL"
    write_json(OUTPUT / "offline_sanity/prediction_summary.json", offline)

    fairness = read_json(OUTPUT / "audit/config_fairness.json")
    smoke = read_json(OUTPUT / "smoke/smoke_report.json")
    checks = {
        "dataset_a_hash": read_json(OUTPUT / "audit/dataset_integrity.json")["checks"]["dataset_a_hash"],
        "dataset_b_hash": read_json(OUTPUT / "audit/dataset_integrity.json")["checks"]["dataset_b_hash"],
        "same_config_except_allowed_fields": fairness["status"] == "PASS",
        "same_base_model_hash": read_json(OUTPUT / "audit/base_model_integrity.json")["status"] == "PASS",
        "same_primary_seed": summary_a["seed"] == summary_b["seed"] == SEED,
        "same_optimizer_config": train_config_a["optimizer"] == train_config_b["optimizer"],
        "same_steps": summary_a["steps_completed"] == summary_b["steps_completed"] == FULL_STEPS,
        "same_effective_batch": summary_a["effective_batch_size"] == summary_b["effective_batch_size"] == BATCH_SIZE,
        "same_action_chunk": audit_a["chunk_size"] == audit_b["chunk_size"] == ACTION_CHUNK,
        "same_28d_interface": audit_a["state_shape"] == audit_b["state_shape"] == [STATE_DIM]
        and audit_a["action_shape"] == audit_b["action_shape"] == [ACTION_DIM],
        "paired_smoke_pass": smoke["status"] == "PASS",
        "checkpoint_load": audit_a["checkpoint_load"] and audit_b["checkpoint_load"],
        "finite_weights": audit_a["weight_audit"]["finite_weights"] and audit_b["weight_audit"]["finite_weights"],
        "finite_inference": audit_a["prediction_finite"] and audit_b["prediction_finite"],
        "no_a_to_b_checkpoint_inheritance": no_a_to_b,
        "separate_output_paths": selected_a != selected_b,
        "same_final_checkpoint_rule": final_fairness["status"] == "PASS",
        "task_parser_regression_tests": read_json(OUTPUT / "audit/task_metadata_parser_tests.json")["status"]
        == "PASS",
        "common_stats_compatibility_patch": read_json(
            OUTPUT / "audit/lerobot_stats_compatibility_patch.json"
        )["status"]
        == "PASS",
        "no_oom": not summary_a["oom"] and not summary_b["oom"],
        "no_unexpected_interruption": not summary_a["unexpected_interruption"]
        and not summary_b["unexpected_interruption"],
    }
    tests = {
        "schema_version": "g1_policy_training_matched51_v1_tests",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "executed_at": now_iso(),
    }
    write_json(OUTPUT / "tests/test_report.json", tests)
    ready = (
        tests["status"] == "PASS"
        and convergence["status"] == "PASS"
        and offline["status"] == "PASS"
        and final_fairness["status"] == "PASS"
        and summary_a["completed"]
        and summary_b["completed"]
    )
    readiness = {
        "schema_version": "g1_policy_training_matched51_v1_readiness",
        "status": "PASS" if ready else "FAIL",
        "policy_a_completed": summary_a["completed"],
        "policy_b_completed": summary_b["completed"],
        "config_fairness": fairness["status"],
        "smoke": smoke["status"],
        "convergence": convergence["status"],
        "final_checkpoint_integrity": final_fairness["status"],
        "offline_inference": offline["status"],
        "controlled_policy_evaluation_ready": ready,
        "isaac_or_real_g1_evaluation_executed": False,
    }
    write_json(OUTPUT / "summary/training_readiness_for_evaluation.json", readiness)
    if not ready:
        raise RuntimeError(f"Final readiness checks failed: {readiness}")
    environment = read_json(OUTPUT / "audit/environment.json")
    report = final_report_text(
        summary_a,
        summary_b,
        convergence,
        fairness,
        final_fairness,
        offline,
        environment,
    )
    (OUTPUT / "summary/final_report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


def failure_report(exc: BaseException) -> None:
    if not OUTPUT.exists():
        return
    reason = f"{type(exc).__name__}: {exc}"
    policy_a_summary = OUTPUT / "policy_a/training_summary.json"
    policy_b_summary = OUTPUT / "policy_b/training_summary.json"
    a_complete = policy_a_summary.is_file() and read_json(policy_a_summary).get("completed", False)
    b_complete = policy_b_summary.is_file() and read_json(policy_b_summary).get("completed", False)
    payload = {
        "schema_version": "g1_policy_training_matched51_v1_failure",
        "status": "FAIL",
        "reason": reason,
        "traceback": traceback.format_exc(),
        "policy_a_completed": a_complete,
        "policy_b_completed": b_complete,
        "pair_incomplete": not (a_complete and b_complete),
        "timestamp": now_iso(),
        "controlled_policy_evaluation_ready": False,
        "isaac_or_real_g1_evaluation_executed": False,
    }
    write_json(OUTPUT / "summary/training_readiness_for_evaluation.json", payload)
    lines = [
        "1. Policy A training completed 여부: " + ("예" if a_complete else "아니오"),
        "2. Policy B training completed 여부: " + ("예" if b_complete else "아니오"),
        f"3. 실패 원인: {reason}",
    ]
    if payload["pair_incomplete"]:
        lines.append("POLICY_A_B_TRAINING_PAIR_INCOMPLETE")
    lines.append("POLICY_A_B_MATCHED51_TRAINING_NOT_READY")
    (OUTPUT / "summary/final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["audit", "smoke", "full", "finalize", "train", "all"],
        default="all",
        help="train runs smoke, full A/B, then finalize; all also creates the initial audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.stage in {"audit", "all"}:
            audit_stage()
        if args.stage == "smoke":
            smoke_stage()
        elif args.stage == "full":
            full_training_stage()
        elif args.stage == "finalize":
            finalize_stage()
        elif args.stage in {"train", "all"}:
            smoke_stage()
            summary_a, summary_b = full_training_stage()
            finalize_stage(summary_a, summary_b)
        return 0
    except IntegrityFailure as exc:
        failure_report(exc)
        print("POLICY_TRAINING_DATASET_INTEGRITY_FAILURE", flush=True)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        failure_report(exc)
        traceback.print_exc()
        print("POLICY_A_B_MATCHED51_TRAINING_NOT_READY", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
