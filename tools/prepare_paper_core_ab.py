#!/usr/bin/env python3
"""Freeze the paper-core split and package matched ACT-A/B train/eval data.

This program is intentionally limited to the paper's full-50 reporting and the
single COMMON48 -> TRAIN40/HELDOUT8 split.  It does not retarget, train, run
Isaac, or access robot hardware.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/paper_core_ab"
DATASET_B = ROOT / "datasets/doll_handoff_proposed_b_50"
A_MANIFEST = ROOT / "outputs/fair_a_full50_hard_fail_audit/after/fair_a_repair_manifest.json"
A_VALIDATION = ROOT / "outputs/fair_a_full50_hard_fail_audit/after/full50_validation.json"
AB_COMPARISON = ROOT / "outputs/fair_a_full50_hard_fail_audit/comparison/fair_a_vs_proposed_b.json"
B_SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
EVENTS_PATH = ROOT / "outputs/dataset_a_final50_retargeting/event_audit/events.json"
LEROBOT_ROOT = Path("/home/jbnu/lerobot-smolvla")

DATASETS = {
    "a_train40": ROOT / "datasets/doll_handoff_fair_a_train40",
    "a_heldout8": ROOT / "datasets/doll_handoff_fair_a_heldout8",
    "b_train40": ROOT / "datasets/doll_handoff_proposed_b_train40",
    "b_heldout8": ROOT / "datasets/doll_handoff_proposed_b_heldout8",
}

TASK = "Pick up the doll with the left hand, handoff it to the right hand, and place it in the trash bin."
IMAGE_KEY = "observation.images.cam_high"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
FPS = 30
DIM = 28
SPLIT_SEED = 20260826
TRAIN_SEED = 1000
HARD_EXPECTED = (35, 46)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray, dtype: np.dtype | None = None) -> str:
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def freeze_json(path: Path, payload: dict[str, Any]) -> None:
    """Create once; on restart, require byte-semantic identity."""

    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise RuntimeError(f"refusing to change frozen artifact: {path}")
        return
    atomic_json(path, payload)


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    flattened = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flattened, DIM)


def parquet_vectors(table: pa.Table, key: str) -> np.ndarray:
    column = table[key].combine_chunks()
    return np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, DIM)


def feature_stats(value: np.ndarray) -> dict[str, list[Any]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != DIM or len(array) == 0:
        raise ValueError(f"bad statistics input {array.shape}")
    result: dict[str, list[Any]] = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0, ddof=0).tolist(),
        "count": [int(len(array))],
    }
    for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
        result[f"q{int(quantile * 100):02d}"] = np.quantile(array, quantile, axis=0).tolist()
    return result


@dataclass(frozen=True)
class Episode:
    final_index: int
    stable_id: str
    raw_recording: str
    original_episode_index: int | None
    frame_count: int
    timestamp: np.ndarray
    q_a: np.ndarray
    state_a: np.ndarray
    q_b: np.ndarray
    state_b: np.ndarray
    a_trajectory_path: Path
    a_trajectory_sha256: str
    b_trajectory_path: Path
    b_trajectory_sha256: str
    source_parquet_path: Path
    source_parquet_sha256: str
    source_video_path: Path
    source_video_sha256: str
    source_video_to_timestamp: float
    events: dict[str, Any]
    b_whole_hand_mean_m: float

    def manifest_row(self) -> dict[str, Any]:
        return {
            "final_dataset_index": self.final_index,
            "original_episode_index": self.original_episode_index,
            "raw_recording_episode_index": 0,
            "episode_identity_note": (
                "original_episode_index is null for the two independently recorded 2026-08-23 "
                "replacement sources; final_dataset_index is their frozen common-50 slot"
                if self.original_episode_index is None
                else "original_episode_index is the frozen source-collection index"
            ),
            "original_source_recording_id": self.raw_recording,
            "stable_episode_id": self.stable_id,
            "frames": self.frame_count,
            "a_trajectory_path": str(self.a_trajectory_path),
            "a_trajectory_sha256": self.a_trajectory_sha256,
            "a_action_array_sha256": sha256_array(self.q_a, np.float32),
            "b_trajectory_path": str(self.b_trajectory_path),
            "b_trajectory_sha256": self.b_trajectory_sha256,
            "b_action_array_sha256": sha256_array(self.q_b, np.float32),
            "source_parquet_path": str(self.source_parquet_path),
            "source_parquet_sha256": self.source_parquet_sha256,
            "source_rgb_identity": {
                "canonical_video_path": str(self.source_video_path),
                "canonical_video_sha256": self.source_video_sha256,
                "raw_cam_high_frame_directory": str(
                    ROOT
                    / "raw_recordings"
                    / self.raw_recording
                    / "images/observation.images.cam_high/episode_000000"
                ),
                "meaning": "same frozen ALOHA cam_high asset is hardlinked into A and B subsets; no re-encode",
            },
        }


def lag1(action: np.ndarray) -> np.ndarray:
    state = np.empty_like(action)
    state[0] = action[0]
    state[1:] = action[:-1]
    return state


def load_episodes() -> tuple[list[Episode], tuple[str, ...], dict[str, Any]]:
    info = read_json(DATASET_B / "meta/info.json")
    canonical_names = tuple(info["features"][ACTION_KEY]["names"])
    if len(canonical_names) != DIM or tuple(info["features"][STATE_KEY]["names"]) != canonical_names:
        raise RuntimeError("Dataset-B canonical 28D feature order is malformed")

    a_manifest = read_json(A_MANIFEST)
    a_validation = read_json(A_VALIDATION)
    b_manifest = read_json(B_SOURCE_MANIFEST)
    events = read_json(EVENTS_PATH)
    if tuple(a_validation["hard_episode_indices"]) != HARD_EXPECTED:
        raise RuntimeError(f"frozen A hard identities changed: {a_validation['hard_episode_indices']}")
    if a_validation["classification_counts"] != {
        "CLEAN_PASS": 27,
        "USABLE_WITH_WARNING": 21,
        "HARD_FAIL": 2,
    }:
        raise RuntimeError("frozen A classification counts changed")
    if b_manifest["source_count"] != 50 or b_manifest["classification_counts"].get("HARD_FAIL") != 0:
        raise RuntimeError("frozen B source manifest changed")

    a_rows = {int(row["episode_index"]): row for row in a_manifest["trajectories"]}
    b_rows = {int(row["final_dataset_index"]): row for row in b_manifest["episodes"]}
    if sorted(a_rows) != list(range(50)) or sorted(b_rows) != list(range(50)):
        raise RuntimeError("A/B final-50 identity indices are incomplete")

    b_data = pq.read_table(DATASET_B / "data/chunk-000/file-000.parquet")
    b_episode_meta = pq.read_table(
        DATASET_B / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    meta_by_episode = {int(row["episode_index"]): row for row in b_episode_meta}
    action_all = parquet_vectors(b_data, ACTION_KEY)
    state_all = parquet_vectors(b_data, STATE_KEY)
    timestamp_all = np.asarray(b_data["timestamp"].to_numpy(), dtype=np.float32)
    episode_all = np.asarray(b_data["episode_index"].to_numpy(), dtype=np.int64)
    frame_all = np.asarray(b_data["frame_index"].to_numpy(), dtype=np.int64)
    if action_all.shape != (34478, DIM) or state_all.shape != (34478, DIM):
        raise RuntimeError("frozen Dataset B dimensions changed")

    result: list[Episode] = []
    for index in range(50):
        a_row, b_row = a_rows[index], b_rows[index]
        a_path = Path(a_row["trajectory_path"])
        b_path = Path(b_row["retargeted_trajectory_path"])
        if sha256_file(a_path) != a_row["trajectory_sha256"]:
            raise RuntimeError(f"A trajectory hash mismatch for episode {index}")
        if sha256_file(b_path) != b_row["retargeted_trajectory_sha256"]:
            raise RuntimeError(f"B trajectory hash mismatch for episode {index}")

        with np.load(a_path, allow_pickle=False) as archive:
            a_names = tuple(map(str, archive["replay_joint_names"]))
            if set(a_names) != set(canonical_names) or len(a_names) != DIM:
                raise RuntimeError(f"A named joints mismatch for episode {index}")
            named_indices = [a_names.index(name) for name in canonical_names]
            q_a = np.asarray(archive["replay_named_joint_qpos"], dtype=np.float32)[:, named_indices]
            a_source_id = str(archive["source_episode_id"].item())
            a_source_dir = str(archive["source_directory_name"].item())

        mask = episode_all == index
        q_b = np.ascontiguousarray(action_all[mask], dtype=np.float32)
        state_b = np.ascontiguousarray(state_all[mask], dtype=np.float32)
        timestamp = np.ascontiguousarray(timestamp_all[mask], dtype=np.float32)
        frames = np.asarray(frame_all[mask], dtype=np.int64)
        if not np.array_equal(frames, np.arange(len(frames), dtype=np.int64)):
            raise RuntimeError(f"B episode boundary changed for episode {index}")
        if q_a.shape != q_b.shape or not np.isfinite(q_a).all() or not np.isfinite(q_b).all():
            raise RuntimeError(f"A/B shape or finiteness failure for episode {index}")
        if not np.array_equal(state_b, lag1(q_b)):
            raise RuntimeError(f"frozen B lag-1 state changed for episode {index}")

        with np.load(b_path, allow_pickle=False) as archive:
            b_names = tuple(map(str, archive["replay_joint_names"]))
            b_indices = [b_names.index(name) for name in canonical_names]
            archive_q_b = np.asarray(archive["replay_named_joint_qpos"], dtype=np.float32)[:, b_indices]
            left_error = np.linalg.norm(
                archive["achieved_left_physical_grasp_frame_position_world"]
                - archive["target_left_interaction_frame_position_world"],
                axis=1,
            )
            right_error = np.linalg.norm(
                archive["achieved_right_physical_grasp_frame_position_world"]
                - archive["target_right_interaction_frame_position_world"],
                axis=1,
            )
        if not np.array_equal(archive_q_b, q_b):
            raise RuntimeError(f"frozen B action array changed for episode {index}")

        raw_recording = str(b_row["raw_directory"])
        stable_id = str(b_row["stable_episode_id"])
        if a_source_id != stable_id or a_source_dir != raw_recording:
            raise RuntimeError(f"A/B source identity mismatch for episode {index}")
        source_path = Path(b_row["source_parquet_path"])
        if sha256_file(source_path) != b_row["source_parquet_sha256"]:
            raise RuntimeError(f"source parquet hash mismatch for episode {index}")
        video_path = (
            DATASET_B
            / "videos/observation.images.cam_high/chunk-000"
            / f"file-{index:03d}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        episode_events = events[str(index)]
        if episode_events["source_name"] != raw_recording:
            raise RuntimeError(f"event/source mismatch for episode {index}")
        meta = meta_by_episode[index]
        if int(meta["length"]) != len(q_a) or int(b_row["source_frame_count"]) != len(q_a):
            raise RuntimeError(f"frame-count mismatch for episode {index}")

        result.append(
            Episode(
                final_index=index,
                stable_id=stable_id,
                raw_recording=raw_recording,
                original_episode_index=(
                    int(b_row["original_episode_index"])
                    if b_row["original_episode_index"] is not None
                    else None
                ),
                frame_count=len(q_a),
                timestamp=timestamp,
                q_a=np.ascontiguousarray(q_a),
                state_a=lag1(q_a),
                q_b=q_b,
                state_b=state_b,
                a_trajectory_path=a_path,
                a_trajectory_sha256=str(a_row["trajectory_sha256"]),
                b_trajectory_path=b_path,
                b_trajectory_sha256=str(b_row["retargeted_trajectory_sha256"]),
                source_parquet_path=source_path,
                source_parquet_sha256=str(b_row["source_parquet_sha256"]),
                source_video_path=video_path,
                source_video_sha256=sha256_file(video_path),
                source_video_to_timestamp=float(
                    meta["videos/observation.images.cam_high/to_timestamp"]
                ),
                events=episode_events,
                b_whole_hand_mean_m=float(np.concatenate((left_error, right_error)).mean()),
            )
        )

    audit = {
        "status": "PASS",
        "episodes": len(result),
        "frames": int(sum(row.frame_count for row in result)),
        "canonical_joint_order": list(canonical_names),
        "a_trajectory_set_manifest_sha256": a_manifest["trajectory_set_sha256"],
        "a_repair_manifest_sha256": sha256_file(A_MANIFEST),
        "a_validation_sha256": sha256_file(A_VALIDATION),
        "b_source_manifest_sha256": sha256_file(B_SOURCE_MANIFEST),
        "b_dataset_data_sha256": sha256_file(DATASET_B / "data/chunk-000/file-000.parquet"),
        "all_a_named_mapped_to_b_order": True,
        "all_b_actions_bit_exact_from_frozen_dataset": True,
        "all_a_states_exact_lag1": True,
        "all_b_states_bit_exact_and_lag1": True,
        "dataset_b_unchanged": True,
        "retargeting_recomputed": False,
    }
    if audit["frames"] != 34478:
        raise RuntimeError("final-50 frame count changed")
    return result, canonical_names, audit


def find_metric(comparison: dict[str, Any], name: str) -> dict[str, Any]:
    for row in comparison["metrics"]:
        if row["metric"] == name:
            return row
    raise KeyError(name)


def triple_mm(value: dict[str, float]) -> str:
    return " / ".join(f"{1000.0 * float(value[key]):.3f}" for key in ("mean", "p95", "max"))


def write_table1() -> dict[str, Any]:
    comparison = read_json(AB_COMPARISON)
    representation = find_metric(comparison, "trajectory representation")
    mechanical = find_metric(comparison, "mechanical classification CLEAN / WARNING / HARD")
    values = {
        "projection": find_metric(comparison, "target projection mean / p95 / max (m)"),
        "wrist": find_metric(comparison, "wrist tracking error mean / p95 / max (m)"),
        "whole_hand": find_metric(comparison, "whole-hand grasp-frame error mean / p95 / max (m)"),
        "bimanual": find_metric(comparison, "bimanual relation error mean / p95 / max (m)"),
    }
    rows = [
        ("Representation", representation["fair_a"], representation["proposed_b"], ""),
        ("Source episodes", "50", "50", "episodes"),
        (
            "Clean / warning / hard",
            f"{mechanical['fair_a']['CLEAN']} / {mechanical['fair_a']['WARNING']} / {mechanical['fair_a']['HARD']}",
            f"{mechanical['proposed_b']['CLEAN']} / {mechanical['proposed_b']['WARNING']} / {mechanical['proposed_b']['HARD']}",
            "episodes",
        ),
        ("Hard IK episodes", "0", "0", "episodes"),
        ("Hard collision episodes", "2", "0", "episodes"),
        (
            "Feasibility projection mean / p95 / max",
            triple_mm(values["projection"]["fair_a"]),
            triple_mm(values["projection"]["proposed_b"]),
            "mm",
        ),
        (
            "Wrist error mean / p95 / max",
            triple_mm(values["wrist"]["fair_a"]),
            triple_mm(values["wrist"]["proposed_b"]),
            "mm",
        ),
        (
            "Whole-hand error mean / p95 / max",
            triple_mm(values["whole_hand"]["fair_a"]),
            triple_mm(values["whole_hand"]["proposed_b"]),
            "mm",
        ),
        (
            "Bimanual relation error mean / p95 / max",
            triple_mm(values["bimanual"]["fair_a"]),
            triple_mm(values["bimanual"]["proposed_b"]),
            "mm",
        ),
        ("Joint-limit failures", "0", "0", "failures"),
        ("Branch failures", "0", "0", "failures"),
    ]
    table_dir = OUTPUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "table1_full50_retargeting.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("METRIC", "FAIR A", "PROPOSED B", "UNIT"))
        writer.writerows(rows)
    markdown = ["| METRIC | FAIR A | PROPOSED B | UNIT |", "|---|---:|---:|---|"]
    markdown.extend(f"| {metric} | {a} | {b} | {unit} |" for metric, a, b, unit in rows)
    md_path = table_dir / "table1_full50_retargeting.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    payload = {
        "schema_version": "paper_core_table1_full50_v1",
        "status": "PASS",
        "source": str(AB_COMPARISON),
        "source_sha256": sha256_file(AB_COMPARISON),
        "rows": [
            {"metric": metric, "fair_a": a, "proposed_b": b, "unit": unit}
            for metric, a, b, unit in rows
        ],
        "observation": {
            "wrist_fidelity": "FAIR_A_LOWER_ERROR",
            "whole_hand_interaction_geometry": "PROPOSED_B_LOWER_ERROR",
            "bimanual_relation_geometry": "PROPOSED_B_LOWER_ERROR",
            "assumed": False,
        },
    }
    atomic_json(table_dir / "table1_full50_retargeting.json", payload)
    return payload


def phase_contract(episode: Episode) -> dict[str, Any]:
    frames = episode.events["frames"]
    left_close = int(frames["LEFT_CLOSE_ONSET"])
    left_grasp = int(frames["LEFT_GRASP"])
    left_owned = int(frames["LEFT_STABLE_HOLD"])
    handoff = int(frames["HANDOFF_APPROACH"])
    right_grasp = int(frames["RIGHT_GRASP"])
    left_release = int(frames["LEFT_RELEASE"])
    right_owned = min(episode.frame_count - 1, left_release + 6)
    right_release = int(frames["RIGHT_FINAL_RELEASE"])
    complete = bool(
        episode.events["source_semantic_valid"]
        and not episode.events["anomalies"]
        and 0 <= left_close <= left_grasp <= left_owned < handoff <= right_grasp <= left_release
        < right_release < episode.frame_count
        and episode.events["ownership_sequence"]
        == [
            "NO_OWNER",
            "LEFT_OWNED",
            "HANDOFF_APPROACH",
            "DUAL_CONTACT",
            "RIGHT_OWNED",
            "RIGHT_TRANSPORT",
            "RELEASED",
        ]
    )
    probes = {
        "initial_approach": max(0, left_close - 60),
        "pre_grasp": max(0, left_close - 4),
        "left_grasp_owned": left_owned,
        "left_transport": int(round((left_owned + handoff) / 2)),
        "handoff_approach": handoff,
        "dual_contact_transfer": int(round((right_grasp + left_release) / 2)),
        "right_owned": right_owned,
        "right_transport": int(round((right_owned + right_release) / 2)),
        "release": right_release,
    }
    return {
        "complete": complete,
        "source_semantic_valid": bool(episode.events["source_semantic_valid"]),
        "anomalies": list(episode.events["anomalies"]),
        "required_intervals": {
            "left_grasp": [left_close, left_owned],
            "left_transport": [left_owned, handoff],
            "handoff": [handoff, left_release],
            "right_transport": [right_owned, right_release],
            "release": right_release,
        },
        "nine_probe_frames": probes,
    }


def freeze_manifests(episodes: list[Episode], source_audit: dict[str, Any]) -> tuple[list[int], list[int]]:
    common = [row for row in episodes if row.final_index not in HARD_EXPECTED]
    if len(common) != 48:
        raise RuntimeError("COMMON_FEASIBLE_48 construction failed")
    common_rows = [row.manifest_row() for row in common]
    common_payload = {
        "schema_version": "paper_core_common_feasible_48_v1",
        "status": "PASS",
        "construction": "remove only the two final Fair-A HARD_FAIL episodes from both methods",
        "excluded_a_hard_episodes": [
            episodes[index].manifest_row() for index in HARD_EXPECTED
        ],
        "episode_count": 48,
        "a_episode_count": 48,
        "b_episode_count": 48,
        "exact_same_final_dataset_indices": True,
        "source_rgb_shared": True,
        "entries": common_rows,
        "entries_canonical_sha256": canonical_json_sha256(common_rows),
        "frozen_source_audit": source_audit,
    }
    freeze_json(OUTPUT / "common48_manifest.json", common_payload)

    common_indices = np.asarray([row.final_index for row in common], dtype=np.int64)
    permutation = np.random.default_rng(SPLIT_SEED).permutation(common_indices).tolist()
    heldout = sorted(map(int, permutation[:8]))
    train = sorted(map(int, permutation[8:]))
    if len(train) != 40 or len(heldout) != 8 or set(train) & set(heldout):
        raise RuntimeError("split cardinality/overlap failure")
    if sorted(train + heldout) != sorted(common_indices.tolist()):
        raise RuntimeError("split union failure")

    heldout_phase = {str(index): phase_contract(episodes[index]) for index in heldout}
    heldout_complete = all(value["complete"] for value in heldout_phase.values())
    heldout_b = np.asarray([episodes[index].b_whole_hand_mean_m for index in heldout])
    median = float(np.median(heldout_b))
    distance = np.abs(heldout_b - median)
    representative = heldout[int(np.lexsort((np.asarray(heldout), distance))[0])]

    split_contract = {
        "split_seed": SPLIT_SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "algorithm": (
            "permute ascending COMMON48 final_dataset_index values; first 8 are HELDOUT8; "
            "remaining 40 are TRAIN40; sort within each subset only for stable packaging order"
        ),
        "permutation_before_sort": permutation,
        "train_final_dataset_indices": train,
        "heldout_final_dataset_indices": heldout,
        "a_b_manifests_identical": True,
        "selected_once_before_training": True,
        "performance_based_selection": False,
    }
    train_rows = [episodes[index].manifest_row() for index in train]
    heldout_rows = [episodes[index].manifest_row() for index in heldout]
    train_payload = {
        "schema_version": "paper_core_train40_v1",
        "status": "PASS",
        "subset": "TRAIN40",
        "episode_count": 40,
        "split_contract": split_contract,
        "entries": train_rows,
        "entries_canonical_sha256": canonical_json_sha256(train_rows),
    }
    heldout_payload = {
        "schema_version": "paper_core_heldout8_v1",
        "status": "PASS" if heldout_complete else "FAIL_CORRUPT_SOURCE_EPISODE_REPORTED",
        "subset": "HELDOUT8",
        "episode_count": 8,
        "split_contract": split_contract,
        "entries": heldout_rows,
        "entries_canonical_sha256": canonical_json_sha256(heldout_rows),
        "complete_source_phase_audit": heldout_phase,
        "all_heldout_complete": heldout_complete,
        "representative_episode_rule_frozen_before_policy_results": (
            "heldout episode closest to the heldout median frozen Proposed-B per-episode "
            "whole-hand interaction-frame mean error; final_dataset_index breaks an exact tie"
        ),
        "representative_episode": representative,
        "representative_source_recording": episodes[representative].raw_recording,
        "heldout_b_whole_hand_mean_mm": {
            str(index): 1000.0 * episodes[index].b_whole_hand_mean_m for index in heldout
        },
        "heldout_b_whole_hand_median_mm": 1000.0 * median,
    }
    freeze_json(OUTPUT / "train40_manifest.json", train_payload)
    freeze_json(OUTPUT / "heldout8_manifest.json", heldout_payload)
    if not heldout_complete:
        raise RuntimeError("frozen heldout split contains a reported incomplete/corrupt source episode")

    selection_rule = {
        "schema_version": "paper_core_act_checkpoint_selection_rule_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "candidate_steps": [20000, 60000, 100000],
        "identical_for": ["ACT-A40", "ACT-B40"],
        "ranking": [
            "maximize common heldout semantic phase-behavior score",
            "then minimize aggregate heldout full-valid-chunk RMSE to the policy's own target",
            "then minimize aggregate raw-chunk jerk RMS as a mechanical quality-control tie-break",
            "then choose the earlier checkpoint step for an exact tie",
        ],
        "training_loss_used_for_selection": False,
        "task_success_tuning": False,
        "smoothness_is_quality_control_not_primary_scientific_claim": True,
    }
    freeze_json(OUTPUT / "checkpoint_selection_rule.json", selection_rule)
    return train, heldout


def info_payload(canonical_names: tuple[str, ...], episodes: list[Episode]) -> dict[str, Any]:
    source_info = read_json(DATASET_B / "meta/info.json")
    features = json.loads(json.dumps(source_info["features"]))
    features[STATE_KEY]["names"] = list(canonical_names)
    features[ACTION_KEY]["names"] = list(canonical_names)
    return {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1_fixed_base_dex3_retargeted",
        "total_episodes": len(episodes),
        "total_frames": int(sum(row.frame_count for row in episodes)),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }


def write_dataset(
    staging: Path,
    target: Path,
    method: str,
    subset: str,
    rows: list[Episode],
    canonical_names: tuple[str, ...],
) -> dict[str, Any]:
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    q = [row.q_a if method == "a" else row.q_b for row in rows]
    state = [row.state_a if method == "a" else row.state_b for row in rows]
    action_all = np.concatenate(q).astype(np.float32, copy=False)
    state_all = np.concatenate(state).astype(np.float32, copy=False)
    timestamp_all = np.concatenate([row.timestamp for row in rows]).astype(np.float32, copy=False)
    frame_all = np.concatenate([np.arange(row.frame_count, dtype=np.int64) for row in rows])
    episode_all = np.concatenate(
        [np.full(row.frame_count, new_index, dtype=np.int64) for new_index, row in enumerate(rows)]
    )
    if not np.isfinite(action_all).all() or not np.isfinite(state_all).all():
        raise RuntimeError("non-finite packaged state/action")

    data = pa.Table.from_arrays(
        [
            fixed_list(state_all),
            fixed_list(action_all),
            pa.array(timestamp_all, type=pa.float32()),
            pa.array(frame_all, type=pa.int64()),
            pa.array(episode_all, type=pa.int64()),
            pa.array(np.arange(len(action_all), dtype=np.int64), type=pa.int64()),
            pa.array(np.zeros(len(action_all), dtype=np.int64), type=pa.int64()),
        ],
        names=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "index", "task_index"],
    )
    data_path = staging / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(data, data_path, compression="snappy", use_dictionary=True)

    episode_metadata: list[dict[str, Any]] = []
    offset = 0
    video_assets: list[dict[str, Any]] = []
    for new_index, row in enumerate(rows):
        episode_q = q[new_index]
        episode_state = state[new_index]
        metadata: dict[str, Any] = {
            "episode_index": new_index,
            "tasks": [TASK],
            "length": row.frame_count,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": offset,
            "dataset_to_index": offset + row.frame_count,
            "videos/observation.images.cam_high/chunk_index": 0,
            "videos/observation.images.cam_high/file_index": new_index,
            "videos/observation.images.cam_high/from_timestamp": 0.0,
            "videos/observation.images.cam_high/to_timestamp": row.source_video_to_timestamp,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feature, value in ((STATE_KEY, episode_state), (ACTION_KEY, episode_q)):
            for statistic, statistic_value in feature_stats(value).items():
                metadata[f"stats/{feature}/{statistic}"] = statistic_value
        episode_metadata.append(metadata)
        offset += row.frame_count

        destination = (
            staging
            / "videos/observation.images.cam_high/chunk-000"
            / f"file-{new_index:03d}.mp4"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(row.source_video_path, destination)
        if sha256_file(destination) != row.source_video_sha256:
            raise RuntimeError("source RGB hardlink hash changed")
        video_assets.append(
            {
                "output_episode_index": new_index,
                "original_final_dataset_index": row.final_index,
                "source_path": str(row.source_video_path),
                "output_relative_path": str(destination.relative_to(staging)),
                "sha256": row.source_video_sha256,
                "storage": "hardlink",
                "same_inode_as_frozen_source": os.path.samefile(row.source_video_path, destination),
                "reencoded": False,
            }
        )

    episode_path = staging / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_metadata), episode_path, compression="snappy", use_dictionary=True)
    shutil.copy2(DATASET_B / "meta/tasks.parquet", staging / "meta/tasks.parquet")
    atomic_json(staging / "meta/info.json", info_payload(canonical_names, rows))
    atomic_json(
        staging / "meta/stats.json",
        {STATE_KEY: feature_stats(state_all), ACTION_KEY: feature_stats(action_all)},
    )
    contract = {
        "schema_version": "paper_core_g1_training_contract_v1",
        "method": "FAIR_A" if method == "a" else "PROPOSED_B",
        "subset": subset.upper(),
        "observation_state_label": "RETARGETED_G1_STATE_SURROGATE",
        "state_semantics": "state[0]=q_target[0]; state[t>0]=q_target[t-1] independently within each episode",
        "action_semantics": "absolute G1/Dex3 joint-position target q_target[t]",
        "state_dimension": DIM,
        "action_dimension": DIM,
        "joint_order": list(canonical_names),
        "source_visual_embodiment": "ALOHA cam_high",
        "target_action_embodiment": "Unitree G1 + Dex3",
        "task_metadata_provenance_only_for_act": True,
        "language_encoder": False,
    }
    atomic_json(staging / "meta/g1_training_contract.json", contract)
    manifest = {
        "schema_version": "paper_core_ab_policy_dataset_v1",
        "status": "PASS",
        "method": "FAIR_A" if method == "a" else "PROPOSED_B",
        "subset": subset.upper(),
        "target_path": str(target),
        "episodes": len(rows),
        "frames": int(len(action_all)),
        "source_final_dataset_indices": [row.final_index for row in rows],
        "source_stable_episode_ids": [row.stable_id for row in rows],
        "source_rgb_assets": video_assets,
        "source_rgb_reencoded": False,
        "source_rgb_identical_between_a_b_by_hardlink_and_sha256": True,
        "action_array_sha256": sha256_array(action_all, np.float32),
        "state_array_sha256": sha256_array(state_all, np.float32),
        "state_derived_only_from_own_method_action": True,
        "b_action_values_regenerated": False,
        "a_retargeting_changed": False,
        "normalization": "authoritative ACT MEAN_STD; statistics derived independently from this TRAIN40 dataset",
        "normalization_values_in_meta_stats": True,
        "episode_mapping": [
            {
                "output_episode_index": new_index,
                "final_dataset_index": row.final_index,
                "original_source_recording_id": row.raw_recording,
                "stable_episode_id": row.stable_id,
                "frames": row.frame_count,
                "trajectory_sha256": (
                    row.a_trajectory_sha256 if method == "a" else row.b_trajectory_sha256
                ),
            }
            for new_index, row in enumerate(rows)
        ],
    }
    atomic_json(staging / "meta/g1_packaging_manifest.json", manifest)
    return manifest


def validate_staged_pair(
    a_root: Path,
    b_root: Path,
    rows: list[Episode],
    expected_count: int,
) -> dict[str, Any]:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.video_utils import get_safe_default_video_backend

    roots = {"a": a_root, "b": b_root}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for method, root in roots.items():
        info = read_json(root / "meta/info.json")
        if info["total_episodes"] != expected_count or info["total_frames"] != sum(
            row.frame_count for row in rows
        ):
            raise RuntimeError("packaged episode/frame count mismatch")
        table = pq.read_table(root / "data/chunk-000/file-000.parquet")
        arrays[method] = {
            "state": parquet_vectors(table, STATE_KEY),
            "action": parquet_vectors(table, ACTION_KEY),
            "episode": np.asarray(table["episode_index"].to_numpy(), dtype=np.int64),
            "frame": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
        }
        if not np.isfinite(arrays[method]["state"]).all() or not np.isfinite(arrays[method]["action"]).all():
            raise RuntimeError("packaged non-finite state/action")

    offset = 0
    for output_episode, row in enumerate(rows):
        end = offset + row.frame_count
        expected_a, expected_b = row.q_a, row.q_b
        if not np.array_equal(arrays["a"]["action"][offset:end], expected_a):
            raise RuntimeError(f"A action changed for source episode {row.final_index}")
        if not np.array_equal(arrays["b"]["action"][offset:end], expected_b):
            raise RuntimeError(f"B action changed for source episode {row.final_index}")
        if not np.array_equal(arrays["a"]["state"][offset:end], lag1(expected_a)):
            raise RuntimeError(f"A lag1 state wrong for source episode {row.final_index}")
        if not np.array_equal(arrays["b"]["state"][offset:end], row.state_b):
            raise RuntimeError(f"B state changed for source episode {row.final_index}")
        expected_episode = np.full(row.frame_count, output_episode, dtype=np.int64)
        expected_frame = np.arange(row.frame_count, dtype=np.int64)
        for method in ("a", "b"):
            if not np.array_equal(arrays[method]["episode"][offset:end], expected_episode):
                raise RuntimeError("episode boundary/remap failure")
            if not np.array_equal(arrays[method]["frame"][offset:end], expected_frame):
                raise RuntimeError("frame boundary failure")
        a_video = a_root / f"videos/{IMAGE_KEY}/chunk-000/file-{output_episode:03d}.mp4"
        b_video = b_root / f"videos/{IMAGE_KEY}/chunk-000/file-{output_episode:03d}.mp4"
        if not os.path.samefile(a_video, b_video) or not os.path.samefile(a_video, row.source_video_path):
            raise RuntimeError("A/B source RGB asset identity failure")
        offset = end

    backend = get_safe_default_video_backend()
    datasets = {
        method: LeRobotDataset(
            repo_id=f"local/paper_core_{method}_{expected_count}",
            root=root,
            download_videos=False,
            video_backend=backend,
        )
        for method, root in roots.items()
    }
    offsets = np.cumsum([0] + [row.frame_count for row in rows[:-1]]).tolist()
    if expected_count == 8:
        sampled_episodes = list(range(8))
    else:
        sampled_episodes = [0, 5, 11, 17, 23, 29, 35, 39]
    readback: list[dict[str, Any]] = []
    for output_episode in sampled_episodes:
        row = rows[output_episode]
        for local_frame in (0, row.frame_count // 2, row.frame_count - 1):
            global_index = offsets[output_episode] + local_frame
            item_a = datasets["a"][global_index]
            item_b = datasets["b"][global_index]
            image_a = item_a[IMAGE_KEY]
            image_b = item_b[IMAGE_KEY]
            if tuple(image_a.shape) != (3, 480, 640) or not bool((image_a == image_b).all()):
                raise RuntimeError("LeRobot A/B RGB pixel readback mismatch")
            if tuple(item_a[STATE_KEY].shape) != (DIM,) or tuple(item_a[ACTION_KEY].shape) != (DIM,):
                raise RuntimeError("LeRobot logical vector readback mismatch")
            if item_a["task"] != TASK or item_b["task"] != TASK:
                raise RuntimeError("LeRobot task readback mismatch")
            readback.append(
                {
                    "output_episode": output_episode,
                    "source_final_dataset_index": row.final_index,
                    "local_frame": local_frame,
                    "global_index": global_index,
                    "image_shape": list(image_a.shape),
                    "rgb_tensor_sha256": sha256_array(image_a.detach().cpu().numpy()),
                }
            )

    deltas = {STATE_KEY: [0.0], ACTION_KEY: [index / FPS for index in range(50)]}
    chunked = LeRobotDataset(
        repo_id=f"local/paper_core_chunk_{expected_count}",
        root=a_root,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend=backend,
    )
    last_global = offsets[-1] + rows[-1].frame_count - 1
    chunk_item = chunked[last_global]
    padding = chunk_item["action_is_pad"].detach().cpu().numpy().astype(bool)
    if tuple(chunk_item[ACTION_KEY].shape) != (50, DIM) or not bool(np.all(padding[1:])):
        raise RuntimeError("LeRobot 50-step episode-boundary padding failure")

    return {
        "status": "PASS",
        "episode_count": expected_count,
        "frame_count": int(sum(row.frame_count for row in rows)),
        "finite_state_action": True,
        "state_action_shape": [DIM],
        "lerobot_readback": "PASS",
        "video_backend": backend,
        "sampled_reads": readback,
        "a_b_rgb_pixel_identical": True,
        "a_b_rgb_asset_byte_identical": True,
        "source_rgb_not_reencoded": True,
        "episode_boundaries": "PASS",
        "chunk_shape": [50, DIM],
        "action_padding_mask": "PASS",
        "a_has_no_hidden_b_state_or_action": True,
        "b_values_subset_bit_exact": True,
    }


def package_all(
    episodes: list[Episode],
    canonical_names: tuple[str, ...],
    train: list[int],
    heldout: list[int],
) -> dict[str, Any]:
    existing = [str(path) for path in DATASETS.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing paper datasets: {existing}")
    subsets = {
        "train40": [episodes[index] for index in train],
        "heldout8": [episodes[index] for index in heldout],
    }
    staging_parent = Path(tempfile.mkdtemp(prefix="paper_core_ab_staging_", dir=ROOT / "datasets"))
    staged = {
        key: staging_parent / key for key in DATASETS
    }
    try:
        manifests = {
            "a_train40": write_dataset(staged["a_train40"], DATASETS["a_train40"], "a", "train40", subsets["train40"], canonical_names),
            "a_heldout8": write_dataset(staged["a_heldout8"], DATASETS["a_heldout8"], "a", "heldout8", subsets["heldout8"], canonical_names),
            "b_train40": write_dataset(staged["b_train40"], DATASETS["b_train40"], "b", "train40", subsets["train40"], canonical_names),
            "b_heldout8": write_dataset(staged["b_heldout8"], DATASETS["b_heldout8"], "b", "heldout8", subsets["heldout8"], canonical_names),
        }
        validations = {
            "train40": validate_staged_pair(staged["a_train40"], staged["b_train40"], subsets["train40"], 40),
            "heldout8": validate_staged_pair(staged["a_heldout8"], staged["b_heldout8"], subsets["heldout8"], 8),
        }
        for key, target in DATASETS.items():
            staged[key].rename(target)
        staging_parent.rmdir()
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise

    payload = {
        "schema_version": "paper_core_ab_dataset_packaging_v1",
        "status": "PASS",
        "datasets": {key: str(path) for key, path in DATASETS.items()},
        "manifests": manifests,
        "validation": validations,
        "a_b_train_episode_indices_identical": True,
        "a_b_heldout_episode_indices_identical": True,
        "a_b_source_rgb_exact": True,
        "dataset_b_modified": False,
        "fair_a_retargeting_modified": False,
    }
    atomic_json(OUTPUT / "dataset_packaging_audit.json", payload)
    return payload


def create_training_configs() -> dict[str, Any]:
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies.act import ACTConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features

    records: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    for method, dataset_key, repo_id in (
        ("a", "a_train40", "local/doll_handoff_fair_a_train40"),
        ("b", "b_train40", "local/doll_handoff_proposed_b_train40"),
    ):
        root = DATASETS[dataset_key]
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        features = dataset_to_policy_features(meta.features)
        input_features = {key: value for key, value in features.items() if key != ACTION_KEY}
        output_features = {ACTION_KEY: features[ACTION_KEY]}
        policy = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=50,
            n_action_steps=50,
            device="cuda",
            use_amp=False,
            push_to_hub=False,
            repo_id=None,
        )
        variant_root = OUTPUT / f"act_{method}40"
        config_dir = variant_root / "config"
        if config_dir.exists():
            raise FileExistsError(config_dir)
        train = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id=repo_id,
                root=str(root),
                episodes=None,
                use_imagenet_stats=True,
                video_backend="torchcodec",
                return_uint8=False,
                streaming=False,
                eval_split=0.0,
            ),
            policy=policy,
            output_dir=variant_root / "train",
            job_name=f"paper_core_act_{method}40_100k",
            seed=TRAIN_SEED,
            cudnn_deterministic=False,
            num_workers=4,
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
        config_dir.mkdir(parents=True)
        train._save_pretrained(config_dir)
        path = config_dir / "train_config.json"
        config = read_json(path)
        configs[method] = config
        records[method] = {
            "config": str(path),
            "config_sha256": sha256_file(path),
            "dataset_root": str(root),
            "dataset_stats_sha256": sha256_file(root / "meta/stats.json"),
            "output_dir": str(variant_root / "train"),
        }

    def remove_allowed_differences(value: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(value))
        result["dataset"]["repo_id"] = "<METHOD_DATASET>"
        result["dataset"]["root"] = "<METHOD_DATASET_ROOT>"
        result["output_dir"] = "<METHOD_OUTPUT>"
        result["job_name"] = "<METHOD_JOB>"
        return result

    common_a = remove_allowed_differences(configs["a"])
    common_b = remove_allowed_differences(configs["b"])
    if common_a != common_b:
        raise RuntimeError("ACT-A/B training configs differ beyond dataset/output identity")
    policy_hash_a = canonical_json_sha256(configs["a"]["policy"])
    policy_hash_b = canonical_json_sha256(configs["b"]["policy"])
    if policy_hash_a != policy_hash_b:
        raise RuntimeError("ACT-A/B architecture differs")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LEROBOT_ROOT, text=True).strip()
    payload = {
        "schema_version": "paper_core_act_a_b_training_contract_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "lerobot_checkout": str(LEROBOT_ROOT),
        "lerobot_commit": commit,
        "implementation": "official installed LeRobot ACTConfig + ACTPolicy through lerobot-train",
        "records": records,
        "only_allowed_config_differences": ["dataset.repo_id", "dataset.root", "output_dir", "job_name"],
        "configs_identical_after_allowed_substitution": True,
        "common_config_canonical_sha256": canonical_json_sha256(common_a),
        "policy_architecture_canonical_sha256": policy_hash_a,
        "initialization": {
            "type": "same official ACT fresh initialization",
            "training_seed": TRAIN_SEED,
            "pretrained_backbone_weights": configs["a"]["policy"]["pretrained_backbone_weights"],
            "pretrained_policy_checkpoint": None,
            "same_rng_seed_before_dataset_load_and_policy_construction": True,
        },
        "fixed_budget": {
            "steps": 100000,
            "batch_size": 8,
            "chunk_size": 50,
            "n_action_steps_training_config": 50,
            "amp": False,
            "save_frequency": 20000,
        },
        "normalization": {
            "algorithm": "official ACT MEAN_STD",
            "mapping": configs["a"]["policy"]["normalization_mapping"],
            "methodology_identical": True,
            "statistics_fit_only_on_each method's matched TRAIN40 supervision": True,
            "heldout8_excluded": True,
        },
    }
    atomic_json(OUTPUT / "act_a_b_training_contract.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-lerobot-readback",
        action="store_true",
        help="development-only escape hatch; paper preparation must not use this",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_lerobot_readback:
        raise RuntimeError("paper-core packaging requires authoritative LeRobot readback")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    episodes, canonical_names, source_audit = load_episodes()
    freeze_json(OUTPUT / "frozen_input_audit.json", source_audit)
    table1 = write_table1()
    train, heldout = freeze_manifests(episodes, source_audit)
    packaging = package_all(episodes, canonical_names, train, heldout)
    training = create_training_configs()
    result = {
        "status": "PASS",
        "table1": table1["status"],
        "common48": 48,
        "excluded": list(HARD_EXPECTED),
        "train40": train,
        "heldout8": heldout,
        "packaging": packaging["status"],
        "training_contract": training["status"],
    }
    atomic_json(OUTPUT / "preparation_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
