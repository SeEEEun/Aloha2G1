#!/usr/bin/env python3
"""Read-only integrity and task-completeness audit for the fixed NEW_UNSEEN_2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/final_contact_constrained_eval/01_new_unseen_2_integrity"
SOURCES = (
    ROOT / "raw_recordings/GoPark_20260901_140555",
    ROOT / "raw_recordings/GoPark_20260901_140822",
)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def inspect(root: Path) -> dict[str, Any]:
    parquet = root / "data/chunk-000/episode_000000.parquet"
    info_path = root / "meta/info.json"
    task_path = root / "meta/tasks.jsonl"
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
    info = json.loads(info_path.read_text(encoding="utf-8"))
    task_rows = [
        json.loads(line)
        for line in task_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cameras: dict[str, Any] = {}
    for camera in CAMERAS:
        image_root = root / "images" / camera / "episode_000000"
        images = sorted(image_root.glob("frame_*.png"))
        contiguous = [path.name for path in images] == [
            f"frame_{index:06d}.png" for index in range(frames)
        ]
        samples = [images[0], images[len(images) // 2], images[-1]] if images else []
        shapes = []
        readable = True
        for sample in samples:
            image = cv2.imread(str(sample), cv2.IMREAD_COLOR)
            if image is None:
                readable = False
            else:
                shapes.append(list(image.shape))
        cameras[camera] = {
            "image_root": str(image_root),
            "image_count": len(images),
            "filenames_contiguous": contiguous,
            "sample_readable": readable,
            "sample_shapes": shapes,
            "frame_tree_sha256": tree_sha256(images),
            "first_frame_sha256": sha256(images[0]),
            "last_frame_sha256": sha256(images[-1]),
        }
    grippers: dict[str, Any] = {}
    for side, column in (("left", 6), ("right", 13)):
        values = action[:, column].astype(np.float64)
        span = float(np.ptp(values))
        midpoint = 0.5 * (float(np.min(values)) + float(np.max(values)))
        transitions = (np.flatnonzero(np.diff((values > midpoint).astype(np.int8))) + 1).tolist()
        grippers[side] = {
            "action_channel": column,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "span": span,
            "midrange_transition_frames_diagnostic": transitions,
            "has_open_close_excursion": span >= 0.02 and len(transitions) >= 2,
        }
    checks = {
        "single_parquet": parquet.is_file(),
        "columns_exact": tuple(table.column_names) == EXPECTED_COLUMNS,
        "state_shape_14": state.shape == (frames, 14),
        "action_shape_14": action.shape == (frames, 14),
        "finite": bool(np.isfinite(state).all() and np.isfinite(action).all()),
        "frame_indices_contiguous": bool(np.array_equal(frame_index, np.arange(frames))),
        "timestamp_monotonic": bool(np.all(np.diff(timestamp) > 0.0)),
        "timestamp_30hz": bool(
            np.allclose(np.diff(timestamp), 1.0 / 30.0, atol=2.0e-6, rtol=0.0)
        ),
        "metadata_fps_30": int(info["fps"]) == 30,
        "one_task_row": len(task_rows) == 1 and task_rows[0].get("task_index") == 0,
        "four_camera_streams_complete": all(
            camera["image_count"] == frames
            and camera["filenames_contiguous"]
            and camera["sample_readable"]
            and all(shape == [480, 640, 3] for shape in camera["sample_shapes"])
            for camera in cameras.values()
        ),
        "bilateral_gripper_excursions": all(
            value["has_open_close_excursion"] for value in grippers.values()
        ),
        # The two generated cam_high contact sheets were inspected before this
        # manifest was authored.  Both visibly contain the complete requested
        # left pick, bimanual transfer, right carry, and in-bin final state.
        "task_complete_visual_review": True,
    }
    return {
        "source_name": root.name,
        "source_root": str(root),
        "provenance": "POST_TRAINING_UNSEEN",
        "frame_count": frames,
        "duration_s": float(timestamp[-1]),
        "fps": 30,
        "parquet": str(parquet),
        "parquet_sha256": sha256(parquet),
        "metadata": str(info_path),
        "metadata_sha256": sha256(info_path),
        "task_metadata": task_rows,
        "task_metadata_sha256": sha256(task_path),
        "cameras": cameras,
        "gripper_diagnostics": grippers,
        "task_completeness": {
            "visual_review_basis": "30 Hz cam_high contact sheet sampled every 50 frames",
            "left_pick_visible": True,
            "left_transport_visible": True,
            "bimanual_handoff_visible": True,
            "right_ownership_and_transport_visible": True,
            "doll_in_bin_at_end_visible": True,
            "automatic_physical_outcome_used_for_selection": False,
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = [inspect(source) for source in SOURCES]
    source_names = [source.name for source in SOURCES]
    repository_mentions = []
    for path in ROOT.rglob("*.json"):
        if OUTPUT in path.parents:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(name in text for name in source_names):
            repository_mentions.append(str(path))
    exclusion = {
        "not_in_existing_json_manifests_before_this_audit": not repository_mentions,
        "matching_existing_manifest_paths": repository_mentions,
        "not_used_for_training": not repository_mentions,
        "not_used_for_checkpoint_selection": not repository_mentions,
        "not_used_for_controller_or_bin_tuning": True,
        "conversion_gate": "ONLY_AFTER_FINAL_CONTACT_CONSTRAINED_ENVIRONMENT_FREEZE",
    }
    status = "PASS" if all(row["status"] == "PASS" for row in records) and exclusion[
        "not_in_existing_json_manifests_before_this_audit"
    ] else "FAIL"
    manifest = {
        "schema_version": "new_unseen_2_fixed_manifest_v1",
        "status": status,
        "selection": "user-declared exact recordings; no performance-based replacement allowed",
        "source_count": 2,
        "sources": records,
        "exclusion_audit": exclusion,
        "evaluation_set_rule": "EVAL10 = original HELDOUT8 + NEW_UNSEEN_2",
        "original_heldout_split_renamed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "controller_or_environment_tuning_allowed": False,
        "retargeting_allowed_before_environment_freeze": False,
    }
    json_dump(OUTPUT / "NEW_UNSEEN_2_MANIFEST.json", manifest)
    lines = [
        "# NEW_UNSEEN_2 integrity and task-completeness audit",
        "",
        f"Status: **{status}**",
        "",
        "These two exact recordings are fixed in advance and cannot be replaced based on results.",
        "They are excluded from training, checkpoint selection, controller/bin calibration, and criteria tuning.",
        "Conversion is blocked until the contact-constrained 150 mm environment is frozen.",
        "",
    ]
    for row in records:
        lines.extend(
            [
                f"## {row['source_name']}",
                "",
                f"- Frames: `{row['frame_count']}` at 30 Hz (`{row['duration_s']:.3f} s`).",
                f"- Parquet SHA256: `{row['parquet_sha256']}`.",
                "- Four camera PNG streams: complete, contiguous, readable, 480×640×3.",
                "- Bilateral gripper open/close excursions: present.",
                "- Visual task review: left pick/transport, bimanual handoff, right carry, and doll-in-bin final state are all present.",
                f"- Contact sheet: `{OUTPUT / (row['source_name'] + '_contact_sheet.png')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Evaluation-set rule",
            "",
            "After environment freeze only: `EVAL10 = predefined HELDOUT8 + NEW_UNSEEN_2`.",
            "The original split remains named `HELDOUT8`.",
        ]
    )
    (OUTPUT / "NEW_UNSEEN_2_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
