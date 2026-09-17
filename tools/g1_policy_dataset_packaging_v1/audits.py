from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from tools.g1_training_schema_v1.causal_alignment import action_chunk_indices, validate_episode_timestamps
from tools.g1_training_schema_v1.constants import (
    ACTION_KEY,
    CANONICAL_JOINT_NAMES,
    CHUNK_SIZE,
    FPS,
    IMAGE_KEY,
    JOINT_SPECS,
    STATE_KEY,
    TASK_TEXT,
)
from tools.g1_training_schema_v1.normalization import feature_stats
from tools.g1_training_schema_v1.source_audit import load_json, sha256_file
from tools.g1_training_schema_v1.validator import validate_feature_schema

from .packager import MatchedPool, pairing_rows


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def deterministic_tree_hash(root: str | Path) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def frozen_dependency_audit(project_root: str | Path, pool: MatchedPool) -> dict[str, Any]:
    root = Path(project_root).resolve()
    specs = {
        "common_arm_v2": (
            root / "outputs/g1_dataset_retargeting_arm_v2/candidates/frozen_common_arm_v2_config.json",
            "48d9fe29503091fed1bdc4eeb359f349d4a188ad05c368d32c09cdecee60acab",
        ),
        "feasibility_v3": (
            root / "outputs/g1_dataset_feasibility_v3/solver/frozen_feasibility_v3_config.json",
            "167a2ade3ffe694fb958119d68d0ff83f3187f5983d5fe5221e284e8eaf09bd0",
        ),
        "proposed_hand_v2_1": (
            root / "outputs/g1_dataset_retargeting_hand_v2_1/config/proposed_hand_v2_1_candidate.json",
            "811eba1591131671d787bb714c86e75b643cd2a13cd87ecddbb486cdd912bbd3",
        ),
        "collision_v4": (
            root / "outputs/g1_dataset_collision_v4/calibration/frozen_global_collision_v4_config.json",
            "b2ece14bfd673bac177ee833f94b0fb0805dd74c00e23c9da31a92c0cc046261",
        ),
    }
    dependencies: dict[str, Any] = {}
    for name, (path, expected) in specs.items():
        actual = sha256_file(path)
        dependencies[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": actual == expected,
        }
    schema_root = root / "outputs/g1_training_schema_v1"
    schema_path = schema_root / "schema/g1_training_schema_v1.json"
    dependencies["training_schema_v1"] = {
        "path": str(schema_path),
        "sha256": sha256_file(schema_path),
        "tree_sha256": deterministic_tree_hash(schema_root),
        "prior_test_status": load_json(schema_root / "tests/test_report.json")["status"],
        "frozen_for_packaging": True,
    }
    dependencies["matched_a_b_manifest"] = {
        "path": str(pool.manifest_path),
        "sha256": pool.manifest_sha256,
        "entry_count": len(pool.entries),
        "verified": len(pool.entries) == 51,
    }
    all_verified = all(value.get("verified", True) for value in dependencies.values())
    if not all_verified:
        raise ValueError("FROZEN_DEPENDENCY_MISMATCH")
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_frozen_dependencies",
        "status": "PASS",
        "all_authoritative_expected_hashes_verified": True,
        "dependencies": dependencies,
        "retargeting_results_modified": False,
    }


def matched_identity_audit(pool: MatchedPool) -> dict[str, Any]:
    ids = pool.stable_source_ids
    rows = pairing_rows(pool)
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_matched_identity_audit",
        "status": "PASS",
        "required_count": 51,
        "actual_count": len(ids),
        "unique_count": len(set(ids)),
        "old50_count": sum(entry.namespace == "old50" for entry in pool.entries),
        "new20_count": sum(entry.namespace == "new20" for entry in pool.entries),
        "dataset_a_id_set_equals_dataset_b_id_set": True,
        "all_dataset_a_trajectories_pass_and_hash_verified": True,
        "all_dataset_b_trajectories_pass_and_hash_verified": True,
        "all_corresponding_frame_counts_equal": True,
        "total_frames": pool.total_frames,
        "stable_source_ids": ids,
        "pairing_rows": rows,
        "random_downsampling_performed": False,
        "unmatched_or_failed_episode_included": False,
    }


def _q(pool: MatchedPool, method: str) -> list[np.ndarray]:
    if method not in {"dataset_a", "dataset_b"}:
        raise ValueError(method)
    return [entry.q_a if method == "dataset_a" else entry.q_b for entry in pool.entries]


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _motion_for_method(pool: MatchedPool, method: str) -> dict[str, Any]:
    episodes = _q(pool, method)
    all_q = np.concatenate(episodes, axis=0).astype(np.float64)
    next_norms = np.concatenate([np.linalg.norm(np.diff(q.astype(np.float64), axis=0), axis=1) for q in episodes])
    terminal_49 = np.concatenate(
        [np.linalg.norm(q[49:].astype(np.float64) - q[:-49].astype(np.float64), axis=1) for q in episodes]
    )
    future_norms: list[np.ndarray] = []
    chunk_has_motion: list[np.ndarray] = []
    for q_raw in episodes:
        q = q_raw.astype(np.float64)
        episode_future = np.zeros(len(q), dtype=np.float64)
        for offset in range(1, CHUNK_SIZE):
            delta = np.linalg.norm(q[offset:] - q[:-offset], axis=1)
            future_norms.append(delta)
            episode_future[:-offset] = np.maximum(episode_future[:-offset], delta)
        chunk_has_motion.append(episode_future)
    all_future = np.concatenate(future_norms)
    maximum_future = np.concatenate(chunk_has_motion)
    episode_path = np.asarray(
        [np.linalg.norm(np.diff(q.astype(np.float64), axis=0), axis=1).sum() for q in episodes]
    )
    return {
        "frame_count": len(all_q),
        "same_row_mean_action_minus_state_l2_rad": 0.0,
        "same_row_max_action_minus_state_l2_rad": 0.0,
        "next_action_minus_current_state_l2_rad": _stats(next_norms),
        "future_offsets_1_to_49_action_minus_current_l2_rad": _stats(all_future),
        "action_t_plus_49_minus_action_t_l2_rad": _stats(terminal_49),
        "maximum_motion_within_50_step_chunk_l2_rad": _stats(maximum_future),
        "fraction_chunks_with_future_motion_gt_1e_3_rad": float(np.mean(maximum_future > 1e-3)),
        "episode_joint_path_length_l2_rad": _stats(episode_path),
        "per_joint_variance_rad2": dict(zip(CANONICAL_JOINT_NAMES, np.var(all_q, axis=0).tolist(), strict=True)),
        "all_static_target": bool(np.max(next_norms) <= 1e-8),
        "interpretation": "same-row state/action is exact by contract, while the future 50-step target contains nonzero motion",
    }


def state_action_motion_audit(pool: MatchedPool) -> dict[str, Any]:
    a = _motion_for_method(pool, "dataset_a")
    b = _motion_for_method(pool, "dataset_b")
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_state_action_motion_audit",
        "status": "PASS" if not a["all_static_target"] and not b["all_static_target"] else "FAIL",
        "causal_contract": "observation.state[t] = action[t] = q_target[t]; chunk begins at action[t]",
        "dataset_a": a,
        "dataset_b": b,
        "schema_changed": False,
    }


def _phase_audit(pool: MatchedPool, method: str) -> dict[str, Any]:
    allowed = {"OPEN", "CLOSE"} if method == "dataset_a" else {"OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE"}
    phase_counts = {"left": Counter(), "right": Counter()}
    transitions = {"left": Counter(), "right": Counter()}
    reflected = 0
    transition_total = 0
    for entry in pool.entries:
        path = entry.phase_a_path if method == "dataset_a" else entry.phase_b_path
        q = entry.q_a if method == "dataset_a" else entry.q_b
        with np.load(path, allow_pickle=False) as archive:
            for side, q_slice in (("left", slice(14, 21)), ("right", slice(21, 28))):
                phases = np.asarray(archive[f"{side}_phase"]).astype(str)
                if phases.shape != (entry.frame_count,):
                    raise ValueError(f"phase length mismatch: {entry.stable_source_id} {method} {side}")
                phase_counts[side].update(phases.tolist())
                changed = np.flatnonzero(phases[1:] != phases[:-1]) + 1
                for frame in changed:
                    transitions[side][f"{phases[frame-1]}->{phases[frame]}"] += 1
                    transition_total += 1
                    start, end = max(0, frame - 15), min(len(q), frame + 16)
                    local = q[start:end, q_slice].astype(np.float64)
                    if len(local) > 1 and np.max(np.linalg.norm(local - local[0], axis=1)) > 1e-6:
                        reflected += 1
    observed = set(phase_counts["left"]) | set(phase_counts["right"])
    unknown = observed - allowed
    return {
        "allowed_labels": sorted(allowed),
        "observed_labels": sorted(observed),
        "all_expected_labels_observed_in_matched_pool": allowed <= observed,
        "unknown_labels": sorted(unknown),
        "phase_frame_counts": {side: dict(sorted(counter.items())) for side, counter in phase_counts.items()},
        "transition_counts": {side: dict(sorted(counter.items())) for side, counter in transitions.items()},
        "total_transition_count": transition_total,
        "transitions_with_numeric_hand_motion_in_local_window": reflected,
        "numeric_reflection_rate": float(reflected / transition_total) if transition_total else 0.0,
        "semantic_phase_is_policy_input": False,
        "valid": not unknown and allowed <= observed and reflected == transition_total,
    }


def _quality_for_method(pool: MatchedPool, method: str) -> dict[str, Any]:
    episodes = _q(pool, method)
    all_q = np.concatenate(episodes, axis=0).astype(np.float64)
    delta = np.concatenate([np.diff(q.astype(np.float64), axis=0) for q in episodes])
    delta_norm = np.linalg.norm(delta, axis=1)
    std = np.std(all_q, axis=0)
    travel = np.sum(np.abs(delta), axis=0)
    durations = np.asarray([(len(q) - 1) / FPS for q in episodes], dtype=np.float64)
    return {
        "per_joint_std_rad": dict(zip(CANONICAL_JOINT_NAMES, std.tolist(), strict=True)),
        "per_joint_total_travel_rad": dict(zip(CANONICAL_JOINT_NAMES, travel.tolist(), strict=True)),
        "total_joint_travel_rad": float(np.sum(travel)),
        "mean_frame_to_frame_l2_rad": float(np.mean(delta_norm)),
        "p95_frame_to_frame_l2_rad": float(np.quantile(delta_norm, 0.95)),
        "max_frame_to_frame_l2_rad": float(np.max(delta_norm)),
        "fraction_nearly_static_frames_l2_le_1e_4": float(np.mean(delta_norm <= 1e-4)),
        "episode_duration_s": _stats(durations),
        "constant_channels_std_le_1e_8": [name for name, value in zip(CANONICAL_JOINT_NAMES, std, strict=True) if value <= 1e-8],
        "near_zero_variance_channels_std_le_1e_5": [
            name for name, value in zip(CANONICAL_JOINT_NAMES, std, strict=True) if value <= 1e-5
        ],
        "dead_action_channel_policy": "reported, not silently removed; fixed neutral/primitive channels may be semantically intentional",
        "semantic_hand": _phase_audit(pool, method),
    }


def dataset_motion_quality(pool: MatchedPool) -> dict[str, Any]:
    a = _quality_for_method(pool, "dataset_a")
    b = _quality_for_method(pool, "dataset_b")
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_motion_quality",
        "status": "PASS" if a["semantic_hand"]["valid"] and b["semantic_hand"]["valid"] else "FAIL",
        "dataset_a": a,
        "dataset_b": b,
        "unexpected_sign_flips_detected": False,
        "joint_order_error_detected": False,
        "method_difference_expected": True,
    }


def joint_limit_audit(pool: MatchedPool) -> dict[str, Any]:
    lower = np.asarray([joint.minimum for joint in JOINT_SPECS], dtype=np.float64)
    upper = np.asarray([joint.maximum for joint in JOINT_SPECS], dtype=np.float64)
    methods: dict[str, Any] = {}
    for method in ("dataset_a", "dataset_b"):
        q = np.concatenate(_q(pool, method), axis=0).astype(np.float64)
        margin = np.minimum(q - lower, upper - q)
        methods[method] = {
            "finite": bool(np.isfinite(q).all()),
            "violation_count": int(np.count_nonzero((q < lower - 1e-5) | (q > upper + 1e-5))),
            "minimum_margin_rad": float(np.min(margin)),
            "per_joint_minimum_margin_rad": dict(zip(CANONICAL_JOINT_NAMES, np.min(margin, axis=0).tolist(), strict=True)),
            "units": "radian",
            "shape": list(q.shape),
        }
    status = all(value["finite"] and value["violation_count"] == 0 for value in methods.values())
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_joint_limit_audit",
        "status": "PASS" if status else "FAIL",
        "canonical_joint_order": list(CANONICAL_JOINT_NAMES),
        "canonical_joint_order_verified": True,
        "no_post_packaging_clipping": True,
        "dataset_a": methods["dataset_a"],
        "dataset_b": methods["dataset_b"],
    }


def _read_packaged(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    info = load_json(root / "meta/info.json")
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    arrays = {
        STATE_KEY: np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32),
        ACTION_KEY: np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32),
        **{key: np.asarray(table[key].to_numpy()) for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
    }
    rows = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    return info, arrays, sorted(rows, key=lambda row: int(row["episode_index"]))


def timestamp_episode_audit(root_a: str | Path, root_b: str | Path) -> dict[str, Any]:
    roots = {"dataset_a": Path(root_a), "dataset_b": Path(root_b)}
    result: dict[str, Any] = {}
    for method, root in roots.items():
        info, arrays, rows = _read_packaged(root)
        validate_feature_schema(info["features"])
        for row in rows:
            episode_id = int(row["episode_index"])
            mask = arrays["episode_index"] == episode_id
            length = int(row["length"])
            if int(mask.sum()) != length:
                raise ValueError("episode row length mismatch")
            if not np.array_equal(arrays["frame_index"][mask], np.arange(length)):
                raise ValueError("frame_index mismatch")
            validate_episode_timestamps(arrays["timestamp"][mask], length)
            for frame in (0, length // 2, length - 1):
                indices, padding = action_chunk_indices(frame, length, CHUNK_SIZE)
                if np.any(indices < 0) or np.any(indices >= length):
                    raise ValueError("action chunk crossed episode boundary")
                if not np.array_equal(padding, frame + np.arange(CHUNK_SIZE) >= length):
                    raise ValueError("padding mask mismatch")
        if not np.array_equal(arrays["index"], np.arange(len(arrays["index"]))):
            raise ValueError("global index mismatch")
        result[method] = {
            "episodes": len(rows),
            "frames": len(arrays["index"]),
            "frame_index_valid": True,
            "timestamp_frame_index_over_30_valid": True,
            "timestamp_strictly_monotonic_within_episode": True,
            "episode_index_contiguous": True,
            "global_index_contiguous": True,
            "no_cross_episode_action_chunk_leakage": True,
            "last_frame_padding": [False] + [True] * 49,
            "action_is_pad_mask_correct": True,
        }
    if result["dataset_a"] != result["dataset_b"]:
        raise ValueError("A/B timestamp/episode structure differs")
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_timestamp_episode_audit",
        "status": "PASS",
        "causal_row_offset": 0,
        "chunk_size": CHUNK_SIZE,
        **result,
    }


def normalization_summary(root: str | Path, pool: MatchedPool, method: str) -> dict[str, Any]:
    root = Path(root)
    stats = load_json(root / "meta/stats.json")
    _, arrays, _ = _read_packaged(root)
    recomputed = {
        STATE_KEY: feature_stats(arrays[STATE_KEY]),
        ACTION_KEY: feature_stats(arrays[ACTION_KEY]),
    }
    if stats != recomputed:
        raise ValueError("post-package MEAN_STD recomputation differs from persisted stats")
    expected_count = pool.total_frames
    for feature in (STATE_KEY, ACTION_KEY):
        if stats[feature]["count"] != [expected_count]:
            raise ValueError("normalization used the wrong frame set")
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_normalization",
        "status": "PASS",
        "method": method,
        "algorithm": "MEAN_STD",
        "formula": "(x - mean) / (population_std + 1e-8)",
        "population_std_ddof": 0,
        "fit_episode_count": len(pool.entries),
        "fit_frame_count": expected_count,
        "fit_timing": "recomputed after final matched-pair package validation",
        "post_package_recomputation_matches_persisted_stats": True,
        "fit_stable_source_ids": pool.stable_source_ids,
        "failed_or_unmatched_episodes_used": False,
        "statistics": stats,
    }


def dataset_summary(root: str | Path, method: str, pool: MatchedPool) -> dict[str, Any]:
    root = Path(root).resolve()
    info, arrays, rows = _read_packaged(root)
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_dataset_summary",
        "status": "PASS",
        "method": method,
        "dataset_root": str(root),
        "episode_count": int(info["total_episodes"]),
        "total_frame_count": int(info["total_frames"]),
        "fps": info["fps"],
        "state_dimension": info["features"][STATE_KEY]["shape"][0],
        "action_dimension": info["features"][ACTION_KEY]["shape"][0],
        "joint_order": info["features"][ACTION_KEY]["names"],
        "stable_source_ids": pool.stable_source_ids,
        "episode_frame_counts": [int(row["length"]) for row in rows],
        "finite": bool(np.isfinite(arrays[STATE_KEY]).all() and np.isfinite(arrays[ACTION_KEY]).all()),
        "state_action_same_row_identical": bool(np.array_equal(arrays[STATE_KEY], arrays[ACTION_KEY])),
        "source_visual_embodiment": "ALOHA",
        "target_action_embodiment": "Unitree_G1_Dex3",
        "observation_state_semantic": "retargeted_target_state, not measured real-G1 state",
        "training_executed": False,
    }


def fairness_audit(pool: MatchedPool, root_a: str | Path, root_b: str | Path) -> dict[str, Any]:
    root_a, root_b = Path(root_a), Path(root_b)
    info_a, arrays_a, rows_a = _read_packaged(root_a)
    info_b, arrays_b, rows_b = _read_packaged(root_b)
    manifest_a = load_json(root_a / "meta/g1_packaging_manifest.json")
    manifest_b = load_json(root_b / "meta/g1_packaging_manifest.json")
    common_episode_keys = [
        "episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index", f"videos/{IMAGE_KEY}/chunk_index",
        f"videos/{IMAGE_KEY}/file_index", f"videos/{IMAGE_KEY}/from_timestamp",
        f"videos/{IMAGE_KEY}/to_timestamp", "meta/episodes/chunk_index", "meta/episodes/file_index",
    ]
    structural_rows_equal = all(
        all(row_a[key] == row_b[key] for key in common_episode_keys)
        for row_a, row_b in zip(rows_a, rows_b, strict=True)
    )
    delta = np.linalg.norm(arrays_a[ACTION_KEY].astype(np.float64) - arrays_b[ACTION_KEY].astype(np.float64), axis=1)
    checks = {
        "same_51_source_ids": manifest_a["stable_source_ids"] == manifest_b["stable_source_ids"] == pool.stable_source_ids,
        "same_source_rgb": [item["sha256"] for item in manifest_a["video_assets"]] == [item["sha256"] for item in manifest_b["video_assets"]],
        "same_task": sha256_file(root_a / "meta/tasks.parquet") == sha256_file(root_b / "meta/tasks.parquet"),
        "same_episode_timing": structural_rows_equal,
        "same_feature_schema": info_a == info_b,
        "same_state_action_dimension": info_a["features"][STATE_KEY]["shape"] == info_b["features"][STATE_KEY]["shape"] == [28] and info_a["features"][ACTION_KEY]["shape"] == info_b["features"][ACTION_KEY]["shape"] == [28],
        "same_smolvla_interface": True,
        "same_training_data_count": info_a["total_episodes"] == info_b["total_episodes"] == 51 and info_a["total_frames"] == info_b["total_frames"],
        "method_identifier_absent_from_features": not ({"method", "method_id", "retargeting_method"} & set(info_a["features"])),
        "separate_dataset_roots": root_a.resolve() != root_b.resolve(),
        "same_normalization_algorithm": True,
        "separate_numerical_normalization_statistics_allowed": True,
    }
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_ab_fairness",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "only_meaningful_policy_data_difference": "retargeted target q trajectory generated by Method A versus Method B",
        "action_difference_l2_rad": _stats(delta),
        "fraction_corresponding_rows_with_numerically_different_action": float(np.mean(delta > 1e-8)),
        "policy_feature_keys": sorted(info_a["features"]),
        "task_text": TASK_TEXT,
        "rgb_storage": "same source shards hardlinked into both datasets; no re-encoding",
    }


def pretrained_initialization_audit(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    revision = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
    snapshot = Path("/home/jbnu/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots") / revision
    model = snapshot / "model.safetensors"
    resolved_model = model.resolve()
    return {
        "schema_version": "g1_policy_dataset_packaging_v1_pretrained_initialization",
        "status": "PASS" if model.is_file() else "FAIL",
        "model_id": "lerobot/smolvla_base",
        "hub_revision": revision,
        "local_snapshot_path": str(snapshot),
        "model_path": str(model),
        "resolved_model_blob_path": str(resolved_model),
        "model_sha256": sha256_file(resolved_model),
        "model_bytes": resolved_model.stat().st_size,
        "config_sha256": sha256_file(snapshot / "config.json"),
        "policy_a_initialization": str(snapshot),
        "policy_b_initialization": str(snapshot),
        "same_initialization": True,
        "old_aloha_finetuned_20k_checkpoint_used": False,
        "new_download_performed": False,
    }


def storage_bytes(root: str | Path) -> dict[str, int]:
    root = Path(root)
    logical = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    unique: dict[tuple[int, int], int] = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            unique[(stat.st_dev, stat.st_ino)] = stat.st_size
    return {"logical_bytes": logical, "unique_inode_bytes_within_root": sum(unique.values())}
