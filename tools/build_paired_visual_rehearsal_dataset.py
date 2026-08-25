#!/usr/bin/env python3
"""Build an exact 1:1 source/final-view LeRobot rehearsal dataset.

The two inputs must have identical state, action, timing, task, episode order,
and frame counts.  Only cam_high pixels may differ.  This utility never edits
either input and refuses an unresolved deployment-camera configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from deployment_camera_config import load_camera_config, sha256_file  # noqa: E402

CAMERA_KEY = "observation.images.cam_high"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def logical_hash(array: pa.ChunkedArray) -> str:
    values = array.combine_chunks()
    if pa.types.is_fixed_size_list(values.type):
        data = np.asarray(values.values.to_numpy(zero_copy_only=False))
    else:
        data = np.asarray(values.to_numpy(zero_copy_only=False))
    contiguous = np.ascontiguousarray(data)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-variant", choices=("A", "B"), required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--final-view-dataset", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-diagnostic-camera-dry-run",
        action="store_true",
        help="Audit an old diagnostic pair without creating a dataset.",
    )
    args = parser.parse_args()
    if args.allow_diagnostic_camera_dry_run and not args.dry_run:
        parser.error("--allow-diagnostic-camera-dry-run requires --dry-run")
    camera = load_camera_config(
        args.camera_config,
        purpose=f"Policy {args.policy_variant} paired final-view rehearsal",
        allow_pending=False,
    )
    if not camera.is_final_helmet and not args.allow_diagnostic_camera_dry_run:
        raise RuntimeError("paired deployment rehearsal requires FROZEN_HELMET_D455_FINAL")

    source = args.source_dataset.resolve()
    final_view = args.final_view_dataset.resolve()
    destination = args.destination.resolve()
    source_data = pq.read_table(source / "data/chunk-000/file-000.parquet")
    final_data = pq.read_table(final_view / "data/chunk-000/file-000.parquet")
    columns = (
        "observation.state", "action", "timestamp", "frame_index",
        "episode_index", "task_index",
    )
    equality = {
        key: logical_hash(source_data[key]) == logical_hash(final_data[key])
        for key in columns
    }
    source_episodes = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet")
    final_episodes = pq.read_table(final_view / "meta/episodes/chunk-000/file-000.parquet")
    equality["episode_table"] = source_episodes.equals(final_episodes)
    equality["tasks"] = sha256_file(source / "meta/tasks.parquet") == sha256_file(final_view / "meta/tasks.parquet")
    if not all(equality.values()):
        raise RuntimeError(f"cross-domain supervision identity failed: {equality}")
    episode_count = source_episodes.num_rows
    frame_count = source_data.num_rows
    if episode_count != 50:
        raise RuntimeError(f"expected 50 paired source episodes, got {episode_count}")

    final_manifest_path = final_view / "meta/g1visual_packaging_manifest.json"
    final_manifest = read_json(final_manifest_path)
    if not final_manifest["label_identity"]["STATE_EQUAL"] or not final_manifest["label_identity"]["ACTION_EQUAL"]:
        raise RuntimeError("final-view package label identity gate did not pass")
    camera_family_path = Path(final_manifest["camera_family"])
    camera_family = read_json(camera_family_path)
    if "camera" in camera_family:
        rendered_camera_hash = camera_family["camera"]["config_sha256"]
    elif "primary" in camera_family:
        rendered_camera_hash = camera_family["primary"]["config_sha256"]
    else:
        raise RuntimeError("camera_family.json has no config hash")
    if rendered_camera_hash != camera.file_sha256:
        raise RuntimeError("rendered dataset camera hash differs from requested rehearsal camera")

    report = {
        "schema_version": "paired_visual_rehearsal_build_v1",
        "status": "PASS_DRY_RUN_NO_DATASET_CREATED" if args.dry_run else "PASS",
        "policy_variant": args.policy_variant,
        "source_dataset": str(source),
        "final_view_dataset": str(final_view),
        "destination": str(destination),
        "source_episodes": episode_count,
        "source_frames": frame_count,
        "paired_episodes": episode_count * 2,
        "paired_frames": frame_count * 2,
        "domain_frame_fraction": {"source_rgb": 0.5, "final_helmet_rgb": 0.5},
        "camera_config": str(camera.path),
        "camera_config_sha256": camera.file_sha256,
        "supervision_identity": equality,
        "inputs_modified": False,
    }
    if args.dry_run:
        if args.report:
            atomic_json(args.report.resolve(), report)
        print(json.dumps(report, indent=2))
        return 0

    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {destination} or {staging}")
    staging.mkdir(parents=True)
    second = final_data
    second = second.set_column(
        second.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array(
            second["episode_index"].combine_chunks().to_numpy() + episode_count,
            type=pa.int64(),
        ),
    )
    second = second.set_column(
        second.schema.get_field_index("index"),
        "index",
        pa.array(
            second["index"].combine_chunks().to_numpy() + frame_count,
            type=pa.int64(),
        ),
    )
    combined = pa.concat_tables([source_data, second])
    (staging / "data/chunk-000").mkdir(parents=True)
    pq.write_table(combined, staging / "data/chunk-000/file-000.parquet", compression="zstd")

    episode_rows = source_episodes.to_pylist()
    second_rows = []
    for row in final_episodes.to_pylist():
        updated = dict(row)
        updated["episode_index"] = int(row["episode_index"]) + episode_count
        updated["dataset_from_index"] = int(row["dataset_from_index"]) + frame_count
        updated["dataset_to_index"] = int(row["dataset_to_index"]) + frame_count
        updated[f"videos/{CAMERA_KEY}/file_index"] = int(row[f"videos/{CAMERA_KEY}/file_index"]) + episode_count
        second_rows.append(updated)
    (staging / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(episode_rows + second_rows, schema=source_episodes.schema),
        staging / "meta/episodes/chunk-000/file-000.parquet",
        compression="zstd",
    )
    shutil.copy2(source / "meta/tasks.parquet", staging / "meta/tasks.parquet")
    shutil.copy2(source / "meta/stats.json", staging / "meta/stats.json")
    info = read_json(source / "meta/info.json")
    info["total_episodes"] = episode_count * 2
    info["total_frames"] = frame_count * 2
    info["splits"] = {"train": f"0:{episode_count * 2}"}
    info["robot_type"] = "unitree_g1_fixed_base_dex3_equal_source_final_helmet_rehearsal"
    atomic_json(staging / "meta/info.json", info)
    video_dir = staging / "videos" / CAMERA_KEY / "chunk-000"
    video_dir.mkdir(parents=True)
    for episode in range(episode_count):
        shutil.copy2(
            source / "videos" / CAMERA_KEY / "chunk-000" / f"file-{episode:03d}.mp4",
            video_dir / f"file-{episode:03d}.mp4",
        )
        shutil.copy2(
            final_view / "videos" / CAMERA_KEY / "chunk-000" / f"file-{episode:03d}.mp4",
            video_dir / f"file-{episode + episode_count:03d}.mp4",
        )
    atomic_json(staging / "meta/paired_visual_rehearsal_manifest.json", report)
    if args.report:
        atomic_json(args.report.resolve(), report)
    os.rename(staging, destination)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
