"""Raw integrity, visual audit, LeRobot-v3 build, and schema equivalence.

The encoder is the existing authoritative ``build_magsafe_lerobot_v3``
implementation.  This module only adds an explicit 20-directory allowlist and
generalizes output validation to a dataset whose local episode IDs are 0..19.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

import build_magsafe_lerobot_v3 as authoritative

from .constants import (
    CAMERA_KEY,
    ORIGINAL_DATASET_ROOT,
    OUTPUT_ROOT,
    RAW_RECORDING_NAMES,
    RAW_ROOT,
    REPO_ID,
    SOURCE_DATASET_ROOT,
    TASK,
    UNSEEN_ID_PREFIX,
    stable_source_id,
)


EXPECTED_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
MAPPING_EVIDENCE = (
    authoritative.DEFAULT_SOURCE.parent
    / "evaluation/mujoco_stationary_aloha_validation/dataset_to_mujoco_joint_mapping.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tree_digest(rows: Iterable[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for relative, size, value in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fixed14(table: pa.Table, key: str, source: Path) -> np.ndarray:
    field = table.schema.field(key)
    if not authoritative.fixed_float32_14(field):
        raise authoritative.fail(source, f"{key} is not fixed-size float32 list[14]")
    return authoritative.read_vectors(table, key, source)


def _raw_task_values(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task") or row.get("__index_level_0__")
        if task is not None:
            values.append(str(task))
    return values


def _image_contract(
    folder: Path,
    frame_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    cameras: dict[str, Any] = {}
    paths_by_camera: dict[str, list[Path]] = {}
    for key in EXPECTED_CAMERAS:
        episode_dirs = sorted((folder / "images" / key).glob("episode_*"))
        if len(episode_dirs) != 1 or not episode_dirs[0].is_dir():
            raise authoritative.fail(folder, f"{key}: expected exactly one PNG episode directory")
        numbered = sorted(
            (
                (authoritative.natural_frame_number(path), path)
                for path in episode_dirs[0].glob("*.png")
            ),
            key=lambda row: (row[0], row[1].name),
        )
        numbers = [row[0] for row in numbered]
        if numbers != frame_indices.tolist():
            raise authoritative.fail(folder, f"{key}: PNG indices differ from parquet frame_index")
        image_paths = [row[1] for row in numbered]
        sample_indices = sorted({0, len(image_paths) // 2, len(image_paths) - 1})
        samples: list[dict[str, Any]] = []
        for index in sample_indices:
            with Image.open(image_paths[index]) as image:
                image.load()
                if image.mode != "RGB" or image.size != (640, 480):
                    raise authoritative.fail(
                        folder,
                        f"{key} frame {index}: expected RGB 640x480, got {image.mode} {image.size}",
                    )
                pixels = np.asarray(image, dtype=np.float32)
                samples.append(
                    {
                        "frame": int(frame_indices[index]),
                        "path": str(image_paths[index]),
                        "mode": image.mode,
                        "size_wh": list(image.size),
                        "mean_uint8": float(np.mean(pixels)),
                        "std_uint8": float(np.std(pixels)),
                    }
                )
        cameras[key] = {
            "storage": "PNG_SEQUENCE",
            "frame_count": len(image_paths),
            "frame_indices_match_parquet": True,
            "sample_decode_checks": samples,
        }
        paths_by_camera[key] = image_paths
    video_files = sorted(path for path in (folder / "videos").rglob("*") if path.is_file())
    cameras["raw_video_file_count"] = len(video_files)
    cameras["authoritative_raw_visual_storage"] = "PNG sequences (video directory empty)"
    return cameras, paths_by_camera


def _hash_recording(
    folder: Path,
    hash_stream: Any,
) -> dict[str, Any]:
    file_rows: list[tuple[str, int, str]] = []
    camera_rows: dict[str, list[tuple[str, int, str]]] = {
        key: [] for key in EXPECTED_CAMERAS
    }
    suffix_counts: Counter[str] = Counter()
    total_bytes = 0
    for path in sorted(value for value in folder.rglob("*") if value.is_file()):
        relative = str(path.relative_to(folder))
        size = path.stat().st_size
        value = sha256_file(path)
        row = (relative, size, value)
        file_rows.append(row)
        total_bytes += size
        suffix_counts[path.suffix.lower() or "<none>"] += 1
        for camera in EXPECTED_CAMERAS:
            if relative.startswith(f"images/{camera}/"):
                camera_rows[camera].append(row)
                break
        hash_stream.write(
            json.dumps(
                {
                    "recording": folder.name,
                    "relative_path": relative,
                    "size_bytes": size,
                    "sha256": value,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    return {
        "file_count": len(file_rows),
        "total_bytes": total_bytes,
        "suffix_file_counts": dict(sorted(suffix_counts.items())),
        "content_tree_sha256": _tree_digest(file_rows),
        "content_tree_algorithm": "SHA-256 over sorted relative-path, size, per-file SHA-256 tuples",
        "all_file_content_hashes_recorded": True,
        "camera_content_tree_sha256": {
            key: _tree_digest(rows) for key, rows in camera_rows.items()
        },
    }


def audit_raw_sources(output_root: Path = OUTPUT_ROOT) -> tuple[dict[str, Any], list[authoritative.SourceEpisode]]:
    source_dir = output_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    if len(RAW_RECORDING_NAMES) != 20:
        raise RuntimeError("exact raw allowlist count is not 20")
    actual_selected = [RAW_ROOT / name for name in RAW_RECORDING_NAMES]
    missing = [str(path) for path in actual_selected if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing predeclared raw recordings: {missing}")

    hash_index_path = source_dir / "raw_file_hashes.jsonl"
    temporary_hash_index = hash_index_path.with_suffix(".jsonl.incomplete")
    records: list[dict[str, Any]] = []
    builder_records: list[authoritative.SourceEpisode] = []
    feature_reference: tuple[Any, ...] | None = None
    with temporary_hash_index.open("w", encoding="utf-8") as hash_stream:
        for episode_id, folder in enumerate(actual_selected):
            parquet_files = sorted((folder / "data").rglob("episode_*.parquet"))
            if len(parquet_files) != 1:
                raise authoritative.fail(folder, f"expected one parquet; found {len(parquet_files)}")
            parquet = parquet_files[0]
            info_path = folder / "meta/info.json"
            tasks_path = folder / "meta/tasks.jsonl"
            if not info_path.is_file() or not tasks_path.is_file():
                raise authoritative.fail(folder, "missing meta/info.json or meta/tasks.jsonl")
            info = json.loads(info_path.read_text(encoding="utf-8"))
            authoritative.feature_contract(info, folder, CAMERA_KEY)
            if int(info.get("fps", -1)) != 30:
                raise authoritative.fail(folder, f"FPS is {info.get('fps')}, expected 30")
            contract = tuple(
                json.dumps(info["features"][key], sort_keys=True)
                for key in ("observation.state", "action")
            )
            if feature_reference is None:
                feature_reference = contract
            elif contract != feature_reference:
                raise authoritative.fail(folder, "state/action metadata differs across allowlisted sources")

            table = pq.read_table(
                parquet,
                columns=[
                    "observation.state",
                    "action",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ],
            )
            state = _fixed14(table, "observation.state", folder)
            action = _fixed14(table, "action", folder)
            timestamps = np.asarray(table["timestamp"].combine_chunks().to_numpy(), dtype=np.float64)
            frame_indices = np.asarray(table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64)
            raw_episode_indices = np.asarray(
                table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64
            )
            raw_task_indices = np.asarray(
                table["task_index"].combine_chunks().to_numpy(), dtype=np.int64
            )
            if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
                raise authoritative.fail(folder, "timestamps are nonfinite or nonmonotonic")
            if len(timestamps) > 1 and not np.allclose(
                np.diff(timestamps), 1.0 / 30.0, atol=1e-3, rtol=0.0
            ):
                raise authoritative.fail(folder, "timestamps do not follow 30 Hz cadence")
            expected_frames = np.arange(len(table), dtype=np.int64)
            if not np.array_equal(frame_indices, expected_frames):
                raise authoritative.fail(folder, "frame_index is not contiguous zero-based")
            if not np.all(raw_episode_indices == 0) or not np.all(raw_task_indices == 0):
                raise authoritative.fail(folder, "single-recording episode/task indices differ from zero")

            cameras, paths_by_camera = _image_contract(folder, frame_indices)
            hash_audit = _hash_recording(folder, hash_stream)
            raw_tasks = _raw_task_values(tasks_path)
            status = "SOURCE_PASS"
            record = {
                "episode_id": episode_id,
                "stable_source_id": stable_source_id(UNSEEN_ID_PREFIX, episode_id),
                "raw_directory_name": folder.name,
                "absolute_path": str(folder.resolve()),
                **hash_audit,
                "critical_file_sha256": {
                    "parquet": sha256_file(parquet),
                    "meta_info": sha256_file(info_path),
                    "meta_tasks": sha256_file(tasks_path),
                },
                "recording_completeness": {
                    "status": status,
                    "parquet_count": 1,
                    "metadata_present": True,
                    "camera_streams_present": list(EXPECTED_CAMERAS),
                    "finite_state_action": True,
                    "frame_camera_alignment": True,
                },
                "source_format_version": info.get("codebase_version"),
                "robot_type": info.get("robot_type"),
                "fps": int(info["fps"]),
                "frame_count": len(table),
                "duration_s": float(timestamps[-1] - timestamps[0]),
                "timestamp_first": float(timestamps[0]),
                "timestamp_last": float(timestamps[-1]),
                "timestamp_cadence_max_abs_error_s": float(
                    np.max(np.abs(np.diff(timestamps) - 1.0 / 30.0), initial=0.0)
                ),
                "observation_state": {
                    "key": "observation.state",
                    "dimension": 14,
                    "dtype": "float32",
                    "finite": True,
                    "channel_min": np.min(state, axis=0).tolist(),
                    "channel_max": np.max(state, axis=0).tolist(),
                },
                "action": {
                    "key": "action",
                    "dimension": 14,
                    "dtype": "float32",
                    "finite": True,
                    "channel_min": np.min(action, axis=0).tolist(),
                    "channel_max": np.max(action, axis=0).tolist(),
                    "verified_channel_contract": {
                        "left_arm": list(range(0, 6)),
                        "left_gripper": 6,
                        "right_arm": list(range(7, 13)),
                        "right_gripper": 13,
                    },
                },
                "raw_task_metadata": raw_tasks,
                "output_task_normalization": TASK,
                "task_normalization_authority": str(
                    (authoritative.Path(__file__).resolve().parents[1] / "build_magsafe_lerobot_v3.py")
                ),
                "cameras": cameras,
                "status": status,
            }
            records.append(record)
            builder_records.append(
                authoritative.SourceEpisode(
                    output_episode_index=episode_id,
                    source_folder=str(folder.resolve()),
                    source_parquet=str(parquet.resolve()),
                    source_frame_count=len(table),
                    cam_high_png_count=len(paths_by_camera[CAMERA_KEY]),
                    source_first_timestamp=float(timestamps[0]),
                    source_last_timestamp=float(timestamps[-1]),
                    first_frame_index=int(frame_indices[0]),
                    last_frame_index=int(frame_indices[-1]),
                    image_paths=[str(path.resolve()) for path in paths_by_camera[CAMERA_KEY]],
                )
            )
            print(f"[raw audit] {episode_id + 1:02d}/20 {folder.name}: {len(table)} SOURCE_PASS", flush=True)
    os.replace(temporary_hash_index, hash_index_path)

    allowlist_hash = hashlib.sha256(
        ("\n".join(RAW_RECORDING_NAMES) + "\n").encode("utf-8")
    ).hexdigest()
    index_hash = sha256_file(hash_index_path)
    total_frames = sum(row["frame_count"] for row in records)
    manifest = {
        "schema_version": "frozen_v4_unseen_20_raw_manifest_v1",
        "status": "IMMUTABLE_EXPLICIT_ALLOWLIST_VERIFIED",
        "selection_policy": "only the literal RAW_RECORDING_NAMES tuple; no glob enters execution",
        "required_count": 20,
        "discovered_required_count": len(records),
        "allowlist_sha256": allowlist_hash,
        "raw_root": str(RAW_ROOT.resolve()),
        "all_source_hashes_index": str(hash_index_path.resolve()),
        "all_source_hashes_index_sha256": index_hash,
        "all_source_file_count": sum(row["file_count"] for row in records),
        "all_source_hashes_recorded": all(
            row["all_file_content_hashes_recorded"] for row in records
        ),
        "total_frames": total_frames,
        "source_pass_count": sum(row["status"] == "SOURCE_PASS" for row in records),
        "source_fail_count": sum(row["status"] != "SOURCE_PASS" for row in records),
        "excluded_by_construction": [
            "all unlisted raw recordings",
            "older 202607xx recordings",
            "older/unlisted 202608xx recordings",
            "future/unlisted 20260813 recordings",
            "_invalid_no_parquet",
        ],
        "channel_contract_evidence": {
            "path": str(MAPPING_EVIDENCE.resolve()),
            "sha256": sha256_file(MAPPING_EVIDENCE),
            "left_gripper_row": 6,
            "right_gripper_row": 13,
            "interpretation": "dataset channel 6/13 map to follower gripper carriage; increasing aperture is open",
        },
        "recordings": records,
    }
    atomic_json(source_dir / "new_20_raw_manifest.json", manifest)
    atomic_json(
        source_dir / "source_dataset_audit.json",
        {
            "status": "SOURCE_PASS" if manifest["source_fail_count"] == 0 else "SOURCE_FAIL",
            "dataset_role": "converter-unseen fixed-layout demonstrations",
            "required_recordings": 20,
            "usable_recordings": manifest["source_pass_count"],
            "total_frames": total_frames,
            "fps": 30,
            "raw_format": "LeRobot v2.1 single-recording directories with PNG camera sequences",
            "output_format_required": "LeRobot v3.0 integrated dataset",
            "action_channel_contract_verified": True,
            "object_relative_source_metadata": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
            "raw_task_metadata_observed": sorted(
                {task for row in records for task in row["raw_task_metadata"]}
            ),
            "normalized_task_metadata": TASK,
            "task_normalization_matches_original_builder": TASK == authoritative.DEFAULT_TASK,
            "episode_frame_counts": [row["frame_count"] for row in records],
            "manifest_sha256": sha256_file(source_dir / "new_20_raw_manifest.json"),
        },
    )
    return manifest, builder_records


def create_layout_contact_sheets(
    records: list[authoritative.SourceEpisode], output_root: Path = OUTPUT_ROOT
) -> dict[str, Any]:
    destination = output_root / "source/layout_audit"
    destination.mkdir(parents=True, exist_ok=True)
    sheet_rows: list[dict[str, Any]] = []
    for sheet_index in range(4):
        selected = records[5 * sheet_index : 5 * (sheet_index + 1)]
        canvas = Image.new("RGB", (3 * 320, 5 * 270), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        samples: list[dict[str, Any]] = []
        for row_index, record in enumerate(selected):
            indices = [0, record.source_frame_count // 2, record.source_frame_count - 1]
            for column, frame in enumerate(indices):
                path = Path(record.image_paths[frame])
                with Image.open(path) as image:
                    rendered = image.convert("RGB").resize((320, 240))
                x = column * 320
                y = row_index * 270 + 30
                canvas.paste(rendered, (x, y))
                draw.text((x + 6, y + 5), f"{('first','middle','late')[column]} f{frame}", fill=(255, 255, 0))
                samples.append(
                    {
                        "episode_id": record.output_episode_index,
                        "raw_recording": Path(record.source_folder).name,
                        "position": ("first", "middle", "late")[column],
                        "frame": frame,
                        "source_image": str(path),
                    }
                )
            draw.text(
                (6, row_index * 270 + 7),
                f"new20:{record.output_episode_index:03d} {Path(record.source_folder).name}",
                fill=(255, 255, 255),
            )
        path = destination / f"fixed_layout_contact_sheet_{sheet_index + 1:02d}.png"
        canvas.save(path)
        sheet_rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "recording_count": len(selected),
                "samples": samples,
            }
        )
    audit = {
        "schema_version": "unseen_20_fixed_layout_visual_audit_v1",
        "purpose": [
            "detect wrong camera",
            "detect missing task objects grossly",
            "detect gross scene-layout mismatch",
            "detect recording corruption",
        ],
        "not_used_for": [
            "object detection",
            "object coordinate inference",
            "source phone/accessory/charger pose labels",
            "retargeter tuning",
        ],
        "camera": CAMERA_KEY,
        "sampling": "first, floor(T/2), last",
        "sample_count": sum(len(row["samples"]) for row in sheet_rows),
        "contact_sheets": sheet_rows,
        "automatic_decode_contract_pass": True,
        "visual_review": "PENDING_REVIEW",
        "object_relative_source_metadata": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
    }
    atomic_json(destination / "layout_audit_manifest.json", audit)
    return audit


def _dataset_starts(records: list[authoritative.SourceEpisode]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for record in records:
        starts.append(cursor)
        cursor += record.source_frame_count
    return starts


def validate_integrated_output(
    root: Path,
    records: list[authoritative.SourceEpisode],
) -> dict[str, Any]:
    dataset = authoritative.LeRobotDataset(repo_id=REPO_ID, root=root, video_backend="pyav")
    expected_frames = sum(row.source_frame_count for row in records)
    if dataset.num_episodes != 20 or dataset.meta.total_episodes != 20:
        raise authoritative.fail(root, "integrated output does not contain exactly 20 episodes")
    if len(dataset) != expected_frames or dataset.meta.total_frames != expected_frames:
        raise authoritative.fail(root, f"integrated output frame mismatch; expected {expected_frames}")
    if int(dataset.meta.fps) != 30 or dataset.meta.total_tasks != 1:
        raise authoritative.fail(root, "integrated output FPS/task count mismatch")
    for key in ("observation.state", "action"):
        feature = dataset.features[key]
        if feature["dtype"] != "float32" or tuple(feature["shape"]) != (14,):
            raise authoritative.fail(root, f"integrated feature mismatch: {key}")
        if feature["names"] != authoritative.JOINT_NAMES:
            raise authoritative.fail(root, f"integrated joint order mismatch: {key}")
    camera = dataset.features[CAMERA_KEY]
    if camera["dtype"] != "video" or tuple(camera["shape"]) != (480, 640, 3):
        raise authoritative.fail(root, "integrated cam_high contract mismatch")
    if list(dataset.meta.tasks.index) != [TASK]:
        raise authoritative.fail(root, "integrated task instruction mismatch")

    starts = _dataset_starts(records)
    boundary_count = 0
    comparisons: list[dict[str, Any]] = []
    for episode_id, record in enumerate(records):
        for frame in sorted({0, record.source_frame_count - 1}):
            item = dataset[starts[episode_id] + frame]
            image = authoritative.tensor_numpy(item[CAMERA_KEY])
            if image.shape != (3, 480, 640) or not np.isfinite(image).all():
                raise authoritative.fail(root, f"decode contract failed ep{episode_id} f{frame}")
            if item["task"] != TASK:
                raise authoritative.fail(root, f"task mismatch ep{episode_id} f{frame}")
            boundary_count += 1
    for episode_id in (0, 9, 19):
        record = records[episode_id]
        table = pq.read_table(record.source_parquet, columns=["observation.state", "action"])
        source_state = authoritative.read_vectors(table, "observation.state", Path(record.source_folder))
        source_action = authoritative.read_vectors(table, "action", Path(record.source_folder))
        for frame in sorted({0, record.source_frame_count // 2, record.source_frame_count - 1}):
            item = dataset[starts[episode_id] + frame]
            state_error = float(
                np.max(np.abs(authoritative.tensor_numpy(item["observation.state"]) - source_state[frame]))
            )
            action_error = float(
                np.max(np.abs(authoritative.tensor_numpy(item["action"]) - source_action[frame]))
            )
            if state_error > 1e-6 or action_error > 1e-6:
                raise authoritative.fail(root, f"source value mismatch ep{episode_id} f{frame}")
            comparisons.append(
                {
                    "episode_id": episode_id,
                    "frame": frame,
                    "state_max_abs_error": state_error,
                    "action_max_abs_error": action_error,
                }
            )

    data_shards = sorted(root.glob("data/chunk-*/file-*.parquet"))
    video_shards = sorted(root.glob(f"videos/{CAMERA_KEY}/chunk-*/file-*.mp4"))
    if not data_shards or not video_shards:
        raise authoritative.fail(root, "integrated output lacks data/video shards")
    probed = []
    video_frames = 0
    for path in video_shards:
        result = authoritative.ffprobe_json(path)
        stream = result["streams"][0]
        measured = authoritative.rate_value(stream.get("avg_frame_rate") or stream["r_frame_rate"])
        count = int(stream.get("nb_read_frames") or stream["nb_frames"])
        if (stream.get("width"), stream.get("height")) != (640, 480):
            raise authoritative.fail(path, "encoded resolution mismatch")
        if not math.isclose(measured, 30.0, abs_tol=1e-6):
            raise authoritative.fail(path, "encoded FPS mismatch")
        video_frames += count
        probed.append(
            {
                "path": str(path.relative_to(root)),
                "frames": count,
                "fps": measured,
                "codec": stream.get("codec_name", camera.get("info", {}).get("video.codec")),
            }
        )
    if video_frames != expected_frames:
        raise authoritative.fail(root, f"encoded frames {video_frames} != {expected_frames}")
    return {
        "status": "LEROBOT_V3_UNSEEN_20_VALID",
        "episode_count": 20,
        "frame_count": expected_frames,
        "fps": 30,
        "boundary_frames_decoded": boundary_count,
        "source_value_comparisons": comparisons,
        "data_shards": [str(path.relative_to(root)) for path in data_shards],
        "video_shards": [str(path.relative_to(root)) for path in video_shards],
        "ffprobe": probed,
        "ffprobe_total_frames": video_frames,
    }


def _write_conversion_manifest(path: Path, records: list[authoritative.SourceEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "output_episode_index",
        "stable_source_id",
        "source_folder",
        "source_parquet",
        "source_frame_count",
        "cam_high_png_count",
        "source_first_timestamp",
        "source_last_timestamp",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            value = asdict(record)
            writer.writerow(
                {
                    key: (
                        stable_source_id(UNSEEN_ID_PREFIX, record.output_episode_index)
                        if key == "stable_source_id"
                        else value[key]
                    )
                    for key in fields
                }
            )


def build_integrated_dataset(
    records: list[authoritative.SourceEpisode],
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    if len(records) != 20:
        raise RuntimeError("source build requires exactly 20 validated records")
    source_dir = output_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    if SOURCE_DATASET_ROOT.exists():
        validation = validate_integrated_output(SOURCE_DATASET_ROOT, records)
        reused = True
    else:
        temporary = SOURCE_DATASET_ROOT.parent / f".{SOURCE_DATASET_ROOT.name}.tmp_{os.getpid()}"
        if temporary.exists():
            raise RuntimeError(f"temporary dataset path already exists: {temporary}")
        try:
            authoritative.build_dataset(
                records,
                temporary,
                REPO_ID,
                TASK,
                CAMERA_KEY,
                30,
                "trossen_ai_stationary",
            )
            validate_integrated_output(temporary, records)
            if SOURCE_DATASET_ROOT.exists():
                raise RuntimeError("dataset output appeared during build")
            temporary.rename(SOURCE_DATASET_ROOT)
            validation = validate_integrated_output(SOURCE_DATASET_ROOT, records)
            reused = False
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    conversion_manifest = source_dir / "new_20_lerobot_conversion_manifest.csv"
    _write_conversion_manifest(conversion_manifest, records)
    data_files = sorted(SOURCE_DATASET_ROOT.glob("data/chunk-*/*.parquet"))
    info_path = SOURCE_DATASET_ROOT / "meta/info.json"
    report = {
        "schema_version": "unseen_20_lerobot_v3_build_report_v1",
        "status": "BUILD_REUSED_AND_REVALIDATED" if reused else "BUILD_COMPLETED_AND_VALIDATED",
        "authoritative_builder": str((authoritative.Path(__file__).resolve().parents[1] / "build_magsafe_lerobot_v3.py")),
        "authoritative_builder_sha256": sha256_file(
            authoritative.Path(__file__).resolve().parents[1] / "build_magsafe_lerobot_v3.py"
        ),
        "preprocessing_equivalence": "authoritative.build_dataset, output_features, read_vectors, LeRobotDataset.save_episode/finalize",
        "source_episode_count": 20,
        "source_frame_count": sum(row.source_frame_count for row in records),
        "output_root": str(SOURCE_DATASET_ROOT.resolve()),
        "repo_id": REPO_ID,
        "task": TASK,
        "camera_key": CAMERA_KEY,
        "fps": 30,
        "info_sha256": sha256_file(info_path),
        "data_shards_sha256": {
            str(path.relative_to(SOURCE_DATASET_ROOT)): sha256_file(path) for path in data_files
        },
        "conversion_manifest": str(conversion_manifest.resolve()),
        "conversion_manifest_sha256": sha256_file(conversion_manifest),
        "validation": validation,
        "images_duplicated_outside_official_video_encoding": False,
    }
    atomic_json(source_dir / "source_lerobot_build_report.json", report)
    return report


def _tasks(root: Path) -> list[str]:
    table = pq.read_table(root / "meta/tasks.parquet")
    return [str(row["__index_level_0__"]) for row in table.to_pylist()]


def _parquet_contract(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("data/chunk-*/*.parquet"))
    tables = [
        pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ],
        )
        for path in files
    ]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    timestamps = np.asarray(table["timestamp"].combine_chunks().to_numpy(), dtype=np.float64)
    episode = np.asarray(table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64)
    cadence_errors: list[float] = []
    for episode_id in np.unique(episode):
        values = timestamps[episode == episode_id]
        cadence_errors.extend(np.abs(np.diff(values) - 1.0 / 30.0).tolist())
    return {
        "files": [str(path.relative_to(root)) for path in files],
        "arrow_types": {
            key: str(table.schema.field(key).type)
            for key in (
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            )
        },
        "timestamp_convention": "episode-local t = frame_index / 30 Hz",
        "timestamp_cadence_max_abs_error_s": max(cadence_errors, default=0.0),
    }


def compare_source_schema(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    old_info = json.loads((ORIGINAL_DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    new_info = json.loads((SOURCE_DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    required_feature_keys = (
        CAMERA_KEY,
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    )
    checks = {
        "lerobot_major_format": old_info["codebase_version"] == new_info["codebase_version"] == "v3.0",
        "feature_names": set(old_info["features"]) == set(new_info["features"]),
        "required_feature_contract": all(
            {
                key2: old_info["features"][key].get(key2)
                for key2 in ("dtype", "shape", "names")
            }
            == {
                key2: new_info["features"][key].get(key2)
                for key2 in ("dtype", "shape", "names")
            }
            for key in required_feature_keys
        ),
        "camera_key": CAMERA_KEY in old_info["features"] and CAMERA_KEY in new_info["features"],
        "fps": float(old_info["fps"]) == float(new_info["fps"]) == 30.0,
        "robot_type": old_info["robot_type"] == new_info["robot_type"],
        "task_semantics": _tasks(ORIGINAL_DATASET_ROOT) == _tasks(SOURCE_DATASET_ROOT) == [TASK],
        "data_path_convention": old_info["data_path"] == new_info["data_path"],
        "video_path_convention": old_info["video_path"] == new_info["video_path"],
        "video_encoding": old_info["features"][CAMERA_KEY]["info"]
        == new_info["features"][CAMERA_KEY]["info"],
    }
    old_parquet = _parquet_contract(ORIGINAL_DATASET_ROOT)
    new_parquet = _parquet_contract(SOURCE_DATASET_ROOT)
    checks["parquet_dtype_contract"] = old_parquet["arrow_types"] == new_parquet["arrow_types"]
    checks["timestamp_convention"] = (
        old_parquet["timestamp_cadence_max_abs_error_s"] <= 1e-5
        and new_parquet["timestamp_cadence_max_abs_error_s"] <= 1e-5
    )
    comparison = {
        "schema_version": "unseen_20_vs_original_50_source_schema_v1",
        "status": "SCHEMA_EQUIVALENT" if all(checks.values()) else "SEMANTIC_SCHEMA_MISMATCH",
        "original_50_root": str(ORIGINAL_DATASET_ROOT.resolve()),
        "unseen_20_root": str(SOURCE_DATASET_ROOT.resolve()),
        "checks": checks,
        "all_required_contracts_identical": all(checks.values()),
        "expected_nonsemantic_differences": {
            "episode_count": [old_info["total_episodes"], new_info["total_episodes"]],
            "frame_count": [old_info["total_frames"], new_info["total_frames"]],
            "split": [old_info["splits"], new_info["splits"]],
            "repository_identity": ["local/magsafe_aloha_50_cam_high_v3", REPO_ID],
            "physical_shard_count_may_differ": True,
        },
        "original_contract": {
            "codebase_version": old_info["codebase_version"],
            "fps": old_info["fps"],
            "features": old_info["features"],
            "tasks": _tasks(ORIGINAL_DATASET_ROOT),
            "parquet": old_parquet,
        },
        "unseen_contract": {
            "codebase_version": new_info["codebase_version"],
            "fps": new_info["fps"],
            "features": new_info["features"],
            "tasks": _tasks(SOURCE_DATASET_ROOT),
            "parquet": new_parquet,
        },
    }
    atomic_json(output_root / "source/source_schema_comparison.json", comparison)
    if not comparison["all_required_contracts_identical"]:
        raise RuntimeError("semantic source schema mismatch; retargeting is gated")
    return comparison


__all__ = [
    "audit_raw_sources",
    "build_integrated_dataset",
    "compare_source_schema",
    "create_layout_contact_sheets",
    "validate_integrated_output",
]
