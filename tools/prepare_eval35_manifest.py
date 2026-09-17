#!/usr/bin/env python3
"""Freeze the evaluation-only identity of EVAL35 without running ACT or physics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/home/jbnu/aloha_g1_dataset")
RAW = ROOT / "raw_recordings"
OUT = ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer"
EVAL10 = (
    ROOT
    / "outputs/final_contact_constrained_eval/04_eval10_preparation/EVAL10_RETARGETING_MANIFEST.json"
)
EVALUATOR_ROOT = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator"
MANIFEST = OUT / "EVAL35_MANIFEST.json"
REPORT = OUT / "EVAL35_MANIFEST.md"
CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
EXPECTED_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(paths: list[Path]) -> str:
    """Hash ordered relative leaf names and their content digests."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def inspect_source(root: Path, eval_index: int) -> dict[str, Any]:
    parquet = root / "data/chunk-000/episode_000000.parquet"
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.jsonl"
    if not parquet.is_file() or not info_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"incomplete evaluation source: {root}")
    table = pq.read_table(parquet)
    frames = table.num_rows
    timestamp = np.asarray(
        table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    frame_index = np.asarray(
        table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    action = np.asarray(table["action"].combine_chunks().values).reshape(frames, 14)
    state = np.asarray(table["observation.state"].combine_chunks().values).reshape(
        frames, 14
    )
    info = read_json(info_path)
    task_rows = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cameras: dict[str, Any] = {}
    for camera in CAMERAS:
        image_root = root / "images" / camera / "episode_000000"
        images = sorted(image_root.glob("frame_*.png"))
        names_contiguous = [path.name for path in images] == [
            f"frame_{index:06d}.png" for index in range(frames)
        ]
        sample_paths = (
            [images[0], images[len(images) // 2], images[-1]] if images else []
        )
        shapes: list[list[int]] = []
        samples_readable = True
        for sample_path in sample_paths:
            image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
            if image is None:
                samples_readable = False
            else:
                shapes.append(list(image.shape))
        cameras[camera] = {
            "image_root": str(image_root),
            "image_count": len(images),
            "filenames_contiguous": names_contiguous,
            "sample_readable": samples_readable,
            "sample_shapes": shapes,
            "frame_tree_sha256": tree_sha256(images),
            "first_frame_sha256": sha256_file(images[0]) if images else None,
            "last_frame_sha256": sha256_file(images[-1]) if images else None,
        }
    checks = {
        "columns_exact": tuple(table.column_names) == EXPECTED_COLUMNS,
        "state_shape_14": state.shape == (frames, 14),
        "action_shape_14": action.shape == (frames, 14),
        "finite_state_and_action": bool(
            np.isfinite(state).all() and np.isfinite(action).all()
        ),
        "frame_indices_contiguous": bool(np.array_equal(frame_index, np.arange(frames))),
        "timestamp_strictly_monotonic": bool(np.all(np.diff(timestamp) > 0.0)),
        "timestamp_30hz": bool(
            np.allclose(np.diff(timestamp), 1.0 / 30.0, atol=2.0e-6, rtol=0.0)
        ),
        "metadata_fps_30": int(info["fps"]) == 30,
        "one_task_row": len(task_rows) == 1 and task_rows[0].get("task_index") == 0,
        "four_camera_streams_complete": all(
            row["image_count"] == frames
            and row["filenames_contiguous"]
            and row["sample_readable"]
            and all(shape == [480, 640, 3] for shape in row["sample_shapes"])
            for row in cameras.values()
        ),
    }
    source_name = root.name
    return {
        "eval_index": eval_index,
        "stable_episode_id": f"new_unseen_{source_name.removeprefix('GoPark_')}",
        "source_name": source_name,
        "source_root": str(root),
        "provenance": "POST_FREEZE_EVALUATION_ONLY",
        "frames": frames,
        "duration_s": float(timestamp[-1]),
        "fps": 30,
        "source_parquet": str(parquet),
        "source_parquet_sha256": sha256_file(parquet),
        "source_metadata": str(info_path),
        "source_metadata_sha256": sha256_file(info_path),
        "source_task_metadata": str(tasks_path),
        "source_task_metadata_sha256": sha256_file(tasks_path),
        "task_metadata_rows": task_rows,
        "cameras": cameras,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "evaluation_only": True,
        "evaluator_calibration_allowed": False,
        "evaluator_tuning_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "controller_tuning_allowed": False,
        "outcome_used_for_episode_selection": False,
    }


def evaluator_exclusion_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    needles: dict[str, str] = {}
    for row in rows:
        needles[f"source_name:{row['source_name']}"] = row["source_name"]
        needles[f"parquet_sha256:{row['source_name']}"] = row[
            "source_parquet_sha256"
        ]
        for camera, camera_row in row["cameras"].items():
            needles[f"camera_tree_sha256:{row['source_name']}:{camera}"] = camera_row[
                "frame_tree_sha256"
            ]
    matches: list[dict[str, str]] = []
    evaluator_files = sorted(path for path in EVALUATOR_ROOT.rglob("*") if path.is_file())
    encoded = {label: value.encode("utf-8") for label, value in needles.items()}
    for path in evaluator_files:
        data = path.read_bytes()
        for label, value in encoded.items():
            if value in data:
                matches.append({"needle": label, "path": str(path)})
    return {
        "evaluator_root": str(EVALUATOR_ROOT),
        "evaluator_file_count_scanned": len(evaluator_files),
        "source_names_and_content_hashes_absent": not matches,
        "matches": matches,
        "interpretation": (
            "Persisted evaluator artifacts contain no direct source-name or content-hash "
            "reference to the 25 evaluation-only recordings. This audit is re-run by "
            "the final rollout prerequisite gate."
        ),
    }


def main() -> int:
    sources = sorted(RAW.glob("GoPark_20260902_*"), key=lambda path: path.name)
    if len(sources) != 25 or any(not source.is_dir() for source in sources):
        raise RuntimeError(
            f"EVAL35 requires exactly 25 GoPark_20260902_* directories; found {len(sources)}"
        )
    eval10 = read_json(EVAL10)
    original_entries = eval10.get("eval_entries", [])
    if eval10.get("evaluation_set") != "EVAL10" or len(original_entries) != 10:
        raise RuntimeError("authoritative base EVAL10 identity is unavailable or malformed")
    if [int(row["eval_index"]) for row in original_entries] != list(range(10)):
        raise RuntimeError("base EVAL10 indices are not exact 0..9")
    new_rows = [inspect_source(source, 10 + offset) for offset, source in enumerate(sources)]
    exclusion = evaluator_exclusion_audit(new_rows)
    status = (
        "PASS_IDENTITY_FROZEN"
        if all(row["status"] == "PASS" for row in new_rows)
        and exclusion["source_names_and_content_hashes_absent"]
        else "FAIL"
    )
    eval_entries = [dict(row) for row in original_entries]
    eval_entries.extend(
        {
            key: row[key]
            for key in (
                "eval_index",
                "stable_episode_id",
                "source_name",
                "provenance",
                "frames",
            )
        }
        for row in new_rows
    )
    manifest = {
        "schema_version": "eval35_evaluation_only_identity_v1",
        "status": status,
        "evaluation_set": "EVAL35",
        "construction": (
            "unchanged authoritative EVAL10 at indices 0..9 plus every sorted "
            "GoPark_20260902_* recording at indices 10..34"
        ),
        "source_count": 35,
        "base_eval10_manifest": str(EVAL10),
        "base_eval10_manifest_sha256": sha256_file(EVAL10),
        "base_eval10_unchanged": True,
        "new_20260902_count": 25,
        "new_20260902_glob": str(RAW / "GoPark_20260902_*"),
        "new_20260902_sort": "lexicographic source directory name",
        "eval_entries": eval_entries,
        "new_20260902_evaluation_only": new_rows,
        "evaluation_only_constraints": {
            "evaluator_calibration_allowed": False,
            "evaluator_tuning_allowed": False,
            "training_allowed": False,
            "checkpoint_selection_allowed": False,
            "controller_or_environment_tuning_allowed": False,
            "episode_replacement_allowed": False,
            "performance_based_selection_allowed": False,
            "physical_rollout_before_valid_frozen_evaluator_allowed": False,
        },
        "evaluator_exclusion_audit": exclusion,
        "rollout_gate": {
            "status": "BLOCKED_UNTIL_VALID_FROZEN_EVALUATOR",
            "requires_authoritative_evaluator_status": "FROZEN",
            "requires_frozen_evaluator_sha256": True,
            "physical_rollouts_started": False,
            "completed_physical_rollouts": 0,
            "required_physical_rollouts": 70,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(MANIFEST, manifest)
    table_rows = [
        f"| {row['eval_index']} | {row['stable_episode_id']} | {row['provenance']} | {row['frames']} |"
        for row in eval_entries
    ]
    REPORT.write_text(
        "\n".join(
            [
                "# EVAL35 evaluation-only identity manifest",
                "",
                f"Status: **{status}**",
                "",
                "EVAL35 preserves the prior EVAL10 verbatim at indices 0–9 and appends all 25 sorted `GoPark_20260902_*` recordings at indices 10–34.",
                "",
                "The new 25 are evaluation-only. They are prohibited from evaluator calibration, evaluator or controller tuning, training, checkpoint selection, performance-based selection, and episode replacement.",
                "",
                "No ACT inference or physical rollout is performed by this manifest builder. Final ACT-A40/ACT-B40 rollout remains blocked until the authoritative evaluator is validly `FROZEN` with a frozen SHA256.",
                "",
                f"Base EVAL10 manifest SHA256: `{manifest['base_eval10_manifest_sha256']}`",
                f"Persisted evaluator exclusion audit: **{'PASS' if exclusion['source_names_and_content_hashes_absent'] else 'FAIL'}**",
                "",
                "| Index | Episode | Provenance | Frames |",
                "|---:|---|---|---:|",
                *table_rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "evaluation_set": "EVAL35",
                "manifest": str(MANIFEST),
                "manifest_sha256": sha256_file(MANIFEST),
                "physical_rollouts_started": False,
            },
            indent=2,
        )
    )
    return 0 if status == "PASS_IDENTITY_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
