#!/usr/bin/env python3
"""Package state-aligned G1 renders as a new immutable LeRobot Dataset A or B.

The original parquet, task table, episode mapping, statistics, state arrays,
and action arrays are copied byte-for-byte.  Only the primary cam_high videos
and image-domain metadata change.  Provisional views remain in the synchronized
render archive because the installed LeRobot reader has no feature-selection
option and would decode every declared camera during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/doll_handoff_proposed_b_50"
RENDERS = ROOT / "outputs/policy_b_g1visual/dataset_render_full"
DESTINATION = ROOT / "datasets/doll_handoff_proposed_b_g1visual_50"
SEMANTIC_SCHEMA = (
    ROOT
    / "outputs/doll_handoff_dataset_b_semantic_audit_2026-08-23/training_schema/g1_training_schema_semantic_v2.json"
)
CAM_HIGH = "observation.images.cam_high"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_tree(root: Path, exclude: set[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    excluded = exclude or set()
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        entries.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), entries


def fixed_numpy(path: Path, key: str) -> np.ndarray:
    table = pq.read_table(path, columns=[key])
    column = table[key].combine_chunks()
    return np.ascontiguousarray(
        np.asarray(column.values.to_numpy(zero_copy_only=False)).reshape(len(column), 28)
    )


def logical_array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"{path}: expected one video stream")
    return streams[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-variant", choices=("A", "B"), required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--renders", type=Path, default=RENDERS)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument(
        "--expected-source-tree-sha256",
        help="Optional frozen input-tree hash; computed and recorded even when omitted.",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    renders = args.renders.resolve()
    destination = args.destination.resolve()
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError(
            f"refusing to overwrite versioned dataset or staging tree: {destination} / {staging}"
        )
    render_manifest = read_json(renders / "render_manifest.json")
    if render_manifest.get("status") != "RENDER_COMPLETE":
        raise RuntimeError("all 50 G1-visual episodes have not completed rendering")
    source_tree, source_entries = dataset_tree(source)
    if args.expected_source_tree_sha256 and source_tree != args.expected_source_tree_sha256:
        raise RuntimeError(f"source Dataset {args.dataset_variant} tree changed: {source_tree}")

    source_episodes = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet")
    episode_rows = source_episodes.to_pylist()
    if len(episode_rows) != 50 or sum(int(row["length"]) for row in episode_rows) != 34478:
        raise RuntimeError("source episode/frame identity changed")
    video_audit: list[dict[str, Any]] = []
    for episode, row in enumerate(episode_rows):
        video = renders / "videos" / CAM_HIGH / "chunk-000" / f"file-{episode:03d}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        probe = ffprobe(video)
        expected_frames = int(row["length"])
        checks = {
            "codec_h264": probe.get("codec_name") == "h264",
            "pix_fmt_yuv420p": probe.get("pix_fmt") == "yuv420p",
            "resolution_640x480": int(probe.get("width", 0)) == 640 and int(probe.get("height", 0)) == 480,
            "fps_30": probe.get("avg_frame_rate") == "30/1",
            "frame_count": int(probe.get("nb_frames", -1)) == expected_frames,
        }
        if not all(checks.values()):
            raise RuntimeError(f"episode {episode}: video validation failed: {probe} / {checks}")
        video_audit.append({
            "episode_index": episode,
            "path": str(video),
            "sha256": sha256_file(video),
            "expected_frames": expected_frames,
            "probe": probe,
            "checks": checks,
        })

    # Copy all non-video source content.  The data parquet, task table, episode
    # table, and stats remain byte-identical by design.
    staging.mkdir(parents=True)
    shutil.copytree(source / "data", staging / "data", copy_function=shutil.copy2)
    shutil.copytree(source / "meta", staging / "meta", copy_function=shutil.copy2)
    destination_video_dir = staging / "videos" / CAM_HIGH / "chunk-000"
    destination_video_dir.mkdir(parents=True)
    for episode in range(50):
        shutil.copy2(
            renders / "videos" / CAM_HIGH / "chunk-000" / f"file-{episode:03d}.mp4",
            destination_video_dir / f"file-{episode:03d}.mp4",
        )

    # Update only image-domain provenance in the new tree.
    info_path = staging / "meta/info.json"
    info = read_json(info_path)
    info["robot_type"] = "unitree_g1_fixed_base_dex3_retargeted_g1_visual"
    info["features"][CAM_HIGH]["info"]["video.codec"] = "h264"
    info["features"][CAM_HIGH]["info"]["video.pix_fmt"] = "yuv420p"
    atomic_json(info_path, info)
    schema = read_json(SEMANTIC_SCHEMA)
    camera_family = read_json(renders / "camera_family.json")
    camera_record = camera_family.get("camera", camera_family)
    selected_camera = camera_family.get(
        "selected_camera",
        camera_family.get("primary_camera", camera_record.get("name")),
    )
    contract = {
        "schema_version": "doll_handoff_g1visual_training_contract_v1",
        "status": "PASS_VISUAL_RELABEL_ONLY",
        "observation_state": schema["state"],
        "action": schema["action"],
        "temporal": schema["temporal"],
        "rendered_robot_q": "observation.state[t]",
        "primary_rgb": {
            "key": CAM_HIGH,
            "embodiment": "Unitree G1/Dex3 rendered in Isaac",
            "camera": selected_camera,
            "camera_calibration": camera_record,
        },
        "label_identity": {
            "state_arrays_unchanged": True,
            "action_arrays_unchanged": True,
            "task_metadata_unchanged": True,
            "episode_mapping_unchanged": True,
            "timestamps_unchanged": True,
        },
        "provisional_camera_archive": {
            "path": str(renders / "videos"),
            "keys": [
                "observation.images.cam_forehead_provisional",
                "observation.images.cam_head_provisional",
                "observation.images.cam_neck_provisional",
            ],
            "included_as_lerobot_features": False,
            "reason": (
                "Installed LeRobotDataset decodes every declared camera key and exposes no dataset-level "
                "camera feature selection. Keeping candidates in the synchronized external archive guarantees "
                "that the initial adaptation consumes only cam_high."
            ),
            "calibration_status": "PROVISIONAL_NOT_PHYSICALLY_CALIBRATED",
        },
    }
    atomic_json(staging / "meta/g1_training_contract.json", contract)

    source_data = source / "data/chunk-000/file-000.parquet"
    new_data = staging / "data/chunk-000/file-000.parquet"
    source_state = fixed_numpy(source_data, "observation.state")
    source_action = fixed_numpy(source_data, "action")
    new_state = fixed_numpy(new_data, "observation.state")
    new_action = fixed_numpy(new_data, "action")
    state_equal = np.array_equal(source_state, new_state)
    action_equal = np.array_equal(source_action, new_action)
    if not state_equal or not action_equal or sha256_file(source_data) != sha256_file(new_data):
        raise RuntimeError("state/action parquet identity failed")
    identity = {
        "source_dataset_tree_sha256": source_tree,
        "source_dataset_tree_file_count": len(source_entries),
        "source_data_parquet_sha256": sha256_file(source_data),
        "new_data_parquet_sha256": sha256_file(new_data),
        "original_state_logical_sha256": logical_array_hash(source_state),
        "new_state_logical_sha256": logical_array_hash(new_state),
        "original_action_logical_sha256": logical_array_hash(source_action),
        "new_action_logical_sha256": logical_array_hash(new_action),
        "STATE_EQUAL": state_equal,
        "ACTION_EQUAL": action_equal,
        "tasks_parquet_byte_equal": sha256_file(source / "meta/tasks.parquet")
        == sha256_file(staging / "meta/tasks.parquet"),
        "episodes_parquet_byte_equal": sha256_file(source / "meta/episodes/chunk-000/file-000.parquet")
        == sha256_file(staging / "meta/episodes/chunk-000/file-000.parquet"),
        "stats_json_byte_equal": sha256_file(source / "meta/stats.json")
        == sha256_file(staging / "meta/stats.json"),
        "frame_counts_identical": True,
        "episode_mapping_identical": True,
    }
    if not all(identity[key] for key in (
        "STATE_EQUAL", "ACTION_EQUAL", "tasks_parquet_byte_equal", "episodes_parquet_byte_equal",
        "stats_json_byte_equal", "frame_counts_identical", "episode_mapping_identical",
    )):
        raise RuntimeError(f"label/metadata identity failed: {identity}")
    atomic_json(staging / "meta/g1visual_label_identity.json", identity)
    content_tree, content_entries = dataset_tree(
        staging, exclude={"meta/g1visual_packaging_manifest.json"}
    )
    manifest = {
        "schema_version": "doll_handoff_g1visual_lerobot_v2",
        "status": "PACKAGED_PENDING_LEROBOT_READBACK",
        "dataset_variant": args.dataset_variant,
        "source_dataset": str(source),
        "destination_dataset": str(destination),
        "episodes": 50,
        "frames": 34478,
        "fps": 30,
        "primary_rgb_key": CAM_HIGH,
        "primary_camera": selected_camera,
        "state_dimension": 28,
        "action_dimension": 28,
        "object_visualization_method": "KINEMATIC_OWNERSHIP_OBJECT_RECONSTRUCTION",
        "label_identity": identity,
        "render_manifest": str(renders / "render_manifest.json"),
        "render_manifest_sha256": sha256_file(renders / "render_manifest.json"),
        "camera_family": str(renders / "camera_family.json"),
        "camera_family_sha256": sha256_file(renders / "camera_family.json"),
        "video_audit": video_audit,
        "dataset_content_tree_sha256_excluding_this_manifest": content_tree,
        "dataset_content_tree_entries_excluding_this_manifest": content_entries,
    }
    atomic_json(staging / "meta/g1visual_packaging_manifest.json", manifest)
    os.rename(staging, destination)
    print(json.dumps({
        "status": manifest["status"],
        "dataset": str(destination),
        "episodes": 50,
        "frames": 34478,
        "STATE_EQUAL": state_equal,
        "ACTION_EQUAL": action_equal,
        "content_tree_sha256": content_tree,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
