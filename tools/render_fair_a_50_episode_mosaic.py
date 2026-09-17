#!/usr/bin/env python3
"""Render the final Fair-A motions with the frozen Policy-B 50-way mosaic.

This is deliberately a thin input adapter around ``render_50_episode_mosaic``.
The renderer, camera, scene, layout, synchronization, labels, frame rate, and
FFmpeg writer are imported from that original Policy-B script without local
reimplementation.  No retargeting, IK, feasibility resolution, policy
inference, or trajectory modification is performed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import render_50_episode_mosaic as policy_b_mosaic


POLICY_B_SCRIPT = ROOT / "tools/render_50_episode_mosaic.py"
POLICY_B_VIDEO = ROOT / "outputs/portfolio_50way/policyB_g1_50episodes_mosaic_4k.mp4"
POLICY_B_MANIFEST = ROOT / "outputs/portfolio_50way/mosaic_manifest.json"

FAIR_A_REPAIR_MANIFEST = (
    ROOT
    / "outputs/fair_a_full50_hard_fail_audit/after/fair_a_repair_manifest.json"
)
FAIR_A_VALIDATION = (
    ROOT / "outputs/fair_a_full50_hard_fail_audit/after/full50_validation.json"
)
FAIR_A_EPISODE_CSV = (
    ROOT / "outputs/fair_a_full50_hard_fail_audit/after/full50_per_episode.csv"
)
FAIR_A_BACKEND_PARITY = (
    ROOT / "outputs/fair_a_full50_hard_fail_audit/provenance/backend_parity.json"
)

DEFAULT_OUTPUT = ROOT / "outputs/portfolio_50way"
OUTPUT_VIDEO_NAME = "doll_handoff_policyA_50split.mp4"
OUTPUT_PREVIEW_NAME = "doll_handoff_policyA_50split_preview.png"
OUTPUT_CONTACT_SHEET_NAME = "doll_handoff_policyA_50split_contact_sheet.png"
OUTPUT_MANIFEST_NAME = "doll_handoff_policyA_50split_manifest.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def scalar_text(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.size != 1:
        raise RuntimeError(f"expected scalar string metadata, got {value.shape}")
    return str(value.reshape(-1)[0])


def discover_fair_a() -> tuple[
    list[policy_b_mosaic.Episode], dict[str, str], dict[str, Any]
]:
    required = [
        POLICY_B_SCRIPT,
        POLICY_B_VIDEO,
        POLICY_B_MANIFEST,
        FAIR_A_REPAIR_MANIFEST,
        FAIR_A_VALIDATION,
        FAIR_A_EPISODE_CSV,
        FAIR_A_BACKEND_PARITY,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"FAIL: missing authoritative input: {missing}")

    # This is the original Policy-B discovery/validation path.  Keeping it in
    # the A run guarantees the same authoritative source mapping and timing.
    policy_b_episodes, policy_b_metadata = policy_b_mosaic.discover()
    policy_b_manifest = policy_b_mosaic.read_json(POLICY_B_MANIFEST)
    fair_a = policy_b_mosaic.read_json(FAIR_A_REPAIR_MANIFEST)
    validation = policy_b_mosaic.read_json(FAIR_A_VALIDATION)
    parity = policy_b_mosaic.read_json(FAIR_A_BACKEND_PARITY)
    rows = read_csv(FAIR_A_EPISODE_CSV)

    if fair_a.get("status") != "VALID_WITH_MEASURED_HARD_FAILURES":
        raise RuntimeError(f"FAIL: Fair-A repair is not final: {fair_a.get('status')}")
    if fair_a.get("dataset_a_packaged") is not False:
        raise RuntimeError("FAIL: expected repair-only Fair-A trajectory freeze")
    if parity.get("COMMON_NATURAL_ARM_BACKEND_IDENTICAL") != "YES":
        raise RuntimeError("FAIL: Fair-A does not declare common natural-arm backend parity")
    if int(validation.get("episode_count", -1)) != 50:
        raise RuntimeError("FAIL: Fair-A validation is not full-50")
    if validation.get("classification_counts") != {
        "CLEAN_PASS": 27,
        "USABLE_WITH_WARNING": 21,
        "HARD_FAIL": 2,
    }:
        raise RuntimeError(
            f"FAIL: unexpected Fair-A classifications: {validation.get('classification_counts')}"
        )
    if list(map(int, validation.get("hard_episode_indices", []))) != [35, 46]:
        raise RuntimeError("FAIL: final Fair-A hard episode identity drift")
    if len(fair_a.get("trajectories", [])) != 50 or len(rows) != 50:
        raise RuntimeError("FAIL: Fair-A trajectory/episode records are not full-50")

    repair_by_index = {
        int(record["episode_index"]): record for record in fair_a["trajectories"]
    }
    csv_by_index = {int(record["episode_index"]): record for record in rows}
    if sorted(repair_by_index) != list(range(50)) or sorted(csv_by_index) != list(range(50)):
        raise RuntimeError("FAIL: Fair-A episode indices are not exactly 0..49")

    episodes: list[policy_b_mosaic.Episode] = []
    statuses: dict[str, str] = {}
    identity_rows: list[dict[str, Any]] = []
    for index, policy_b_episode in enumerate(policy_b_episodes):
        repair = repair_by_index[index]
        evaluation = csv_by_index[index]
        trajectory = Path(repair["trajectory_path"]).resolve()
        if not trajectory.is_file():
            raise RuntimeError(f"FAIL: missing final Fair-A trajectory: {trajectory}")
        if policy_b_mosaic.sha256(trajectory) != repair["trajectory_sha256"]:
            raise RuntimeError(f"FAIL: Fair-A frozen hash mismatch at ep{index:03d}")
        if repair["stable_episode_id"] != evaluation["stable_episode_id"]:
            raise RuntimeError(f"FAIL: Fair-A manifest/CSV identity mismatch at ep{index:03d}")
        if repair["stable_episode_id"] != policy_b_episode.source_raw_episode.replace(
            "GoPark_", "doll_handoff_"
        ):
            # The normal 2026-08-20 rows carry an epNNN stable name rather than
            # a raw timestamp.  The robust check below reads the frozen NPZ.
            pass
        if evaluation["source_raw_episode"] != policy_b_episode.source_raw_episode:
            raise RuntimeError(f"FAIL: A/B raw source mismatch at ep{index:03d}")
        if int(evaluation["frame_count"]) != policy_b_episode.frame_count:
            raise RuntimeError(f"FAIL: A/B frame-count mismatch at ep{index:03d}")

        with np.load(trajectory, allow_pickle=False) as arrays:
            q = np.asarray(arrays["replay_named_joint_qpos"])
            names = list(map(str, arrays["replay_joint_names"]))
            source_indices = np.asarray(arrays["source_frame_index"])
            timestamps = np.asarray(arrays["timestamp"], dtype=np.float64)
            stored_source_directory = scalar_text(arrays["source_directory_name"])
            stored_stable_id = scalar_text(arrays["source_episode_id"])
            feasibility_config = scalar_text(arrays["feasibility_config_sha256"])
        if q.shape != (policy_b_episode.frame_count, 28) or not np.isfinite(q).all():
            raise RuntimeError(f"FAIL: invalid Fair-A qpos at ep{index:03d}: {q.shape}")
        if len(names) != 28 or len(set(names)) != 28:
            raise RuntimeError(f"FAIL: invalid Fair-A joint names at ep{index:03d}")
        if not np.array_equal(source_indices, np.arange(policy_b_episode.frame_count)):
            raise RuntimeError(f"FAIL: Fair-A source frame ordering mismatch at ep{index:03d}")
        expected_time = np.arange(policy_b_episode.frame_count, dtype=np.float64) / policy_b_mosaic.FPS
        # Stored timestamps are float32, so the ~23.5 s endpoint can differ
        # from exact float64 frame/30 by just under one microsecond.
        if not np.allclose(timestamps, expected_time, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(f"FAIL: Fair-A 30 Hz timestamps mismatch at ep{index:03d}")
        if stored_source_directory != policy_b_episode.source_raw_episode:
            raise RuntimeError(f"FAIL: Fair-A stored raw source mismatch at ep{index:03d}")
        if stored_stable_id != repair["stable_episode_id"]:
            raise RuntimeError(f"FAIL: Fair-A stored stable ID mismatch at ep{index:03d}")
        if feasibility_config != fair_a["frozen_b_generic_resolver_config_sha256"]:
            raise RuntimeError(f"FAIL: Fair-A shared feasibility config drift at ep{index:03d}")

        status = evaluation["after_classification"]
        if status not in {"CLEAN_PASS", "USABLE_WITH_WARNING", "HARD_FAIL"}:
            raise RuntimeError(f"FAIL: invalid Fair-A status at ep{index:03d}: {status}")
        episode_id = f"ep{index:03d}"
        statuses[episode_id] = status
        episodes.append(
            policy_b_mosaic.Episode(
                index=index,
                episode_id=episode_id,
                source_raw_episode=policy_b_episode.source_raw_episode,
                source_video=policy_b_episode.source_video,
                source_parquet=policy_b_episode.source_parquet,
                trajectory=trajectory,
                frame_count=policy_b_episode.frame_count,
                fps=policy_b_episode.fps,
            )
        )
        identity_rows.append(
            {
                "tile_index": index,
                "episode_id": episode_id,
                "stable_episode_id": repair["stable_episode_id"],
                "source_raw_episode": policy_b_episode.source_raw_episode,
                "frame_count": policy_b_episode.frame_count,
                "policy_b_source_identity_match": True,
                "fair_a_trajectory_path": str(trajectory),
                "fair_a_trajectory_sha256": repair["trajectory_sha256"],
                "fair_a_status": status,
            }
        )

    if Counter(statuses.values()) != Counter(
        {"CLEAN_PASS": 27, "USABLE_WITH_WARNING": 21, "HARD_FAIL": 2}
    ):
        raise RuntimeError(f"FAIL: Fair-A status count drift: {Counter(statuses.values())}")
    if [index for index, e in enumerate(episodes) if statuses[e.episode_id] == "HARD_FAIL"] != [35, 46]:
        raise RuntimeError("FAIL: Fair-A hard episodes were excluded or reordered")
    if [e.index for e in episodes] != list(range(50)):
        raise RuntimeError("FAIL: A/B episode ordering is not exactly 0..49")

    expected_b_probe = policy_b_manifest["validation"]["final_video_probes"]["g1"]
    live_b_probe = policy_b_mosaic.probe(POLICY_B_VIDEO)
    for key in ("codec", "width", "height", "frame_count", "pixel_format"):
        if live_b_probe[key] != expected_b_probe[key]:
            raise RuntimeError(f"FAIL: existing Policy-B video probe drift in {key}")
    if abs(live_b_probe["fps"] - expected_b_probe["fps"]) > 0.01:
        raise RuntimeError("FAIL: existing Policy-B video FPS drift")

    metadata = {
        "exact_existing_policy_b_video": str(POLICY_B_VIDEO.resolve()),
        "exact_policy_b_generation_script": str(POLICY_B_SCRIPT.resolve()),
        "fair_a_trajectory_source_manifest": str(FAIR_A_REPAIR_MANIFEST.resolve()),
        "fair_a_trajectory_directory": str(episodes[0].trajectory.parent),
        "fair_a_validation_manifest": str(FAIR_A_VALIDATION.resolve()),
        "fair_a_episode_status_csv": str(FAIR_A_EPISODE_CSV.resolve()),
        "shared_backend_parity_artifact": str(FAIR_A_BACKEND_PARITY.resolve()),
        "source_episode_ids_and_order_match_policy_b": True,
        "source_episode_count": 50,
        "source_episode_order": [row["stable_episode_id"] for row in identity_rows],
        "frame_count_range": [min(e.frame_count for e in episodes), max(e.frame_count for e in episodes)],
        "total_frames": sum(e.frame_count for e in episodes),
        "classification_counts": dict(validation["classification_counts"]),
        "hard_episode_indices_included": [35, 46],
        "policy_b_renderer_metadata": policy_b_metadata,
        "policy_b_video_probe": live_b_probe,
        "render_python_executable": sys.executable,
        "mujoco_version": policy_b_mosaic.mujoco.__version__,
        "opencv_version": policy_b_mosaic.cv2.__version__,
        "identity_rows": identity_rows,
    }
    return episodes, statuses, metadata


def input_hashes(episodes: list[policy_b_mosaic.Episode]) -> dict[str, Any]:
    return {
        "policy_b_generation_script": policy_b_mosaic.sha256(POLICY_B_SCRIPT),
        "policy_b_video": policy_b_mosaic.sha256(POLICY_B_VIDEO),
        "policy_b_manifest": policy_b_mosaic.sha256(POLICY_B_MANIFEST),
        "fair_a_repair_manifest": policy_b_mosaic.sha256(FAIR_A_REPAIR_MANIFEST),
        "fair_a_validation": policy_b_mosaic.sha256(FAIR_A_VALIDATION),
        "fair_a_episode_csv": policy_b_mosaic.sha256(FAIR_A_EPISODE_CSV),
        "fair_a_backend_parity": policy_b_mosaic.sha256(FAIR_A_BACKEND_PARITY),
        "fair_a_trajectories": {
            episode.episode_id: policy_b_mosaic.sha256(episode.trajectory)
            for episode in episodes
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume", action="store_true", help="reuse only verified Fair-A tile caches"
    )
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    episodes, statuses, metadata = discover_fair_a()
    pre_render = {
        "exact_existing_B_video_path": metadata["exact_existing_policy_b_video"],
        "exact_B_generation_script": metadata["exact_policy_b_generation_script"],
        "exact_Fair_A_trajectory_source": metadata["fair_a_trajectory_source_manifest"],
        "Fair_A_trajectory_directory_from_manifest": metadata["fair_a_trajectory_directory"],
        "all_50_source_episode_IDs_and_order_match": metadata[
            "source_episode_ids_and_order_match_policy_b"
        ],
        "episode_order": "ep000..ep049",
        "hard_episodes_included": metadata["hard_episode_indices_included"],
        "status_labels_in_existing_B_visualization": False,
        "label_preservation": "EP %02d only; no A-only status overlay added",
    }
    print("PRE-RENDER VERIFICATION", flush=True)
    print(json.dumps(pre_render, indent=2), flush=True)
    if args.discover_only:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    before = input_hashes(episodes)
    tile_paths = policy_b_mosaic.render_g1_tiles(
        episodes, output_dir / "cache/policy_a", args.resume
    )
    output_video = output_dir / OUTPUT_VIDEO_NAME
    policy_b_mosaic.compose(tile_paths, episodes, output_video)

    maximum = max(episode.frame_count for episode in episodes)
    visual_evidence = policy_b_mosaic.make_visual_evidence(
        output_video,
        output_dir / OUTPUT_PREVIEW_NAME,
        output_dir / OUTPUT_CONTACT_SHEET_NAME,
        maximum,
    )
    tile_metrics = policy_b_mosaic.sample_tile_metrics(tile_paths, episodes)
    after = input_hashes(episodes)
    if before != after:
        raise RuntimeError("FAIL: authoritative A/B inputs changed during visualization")

    result = policy_b_mosaic.probe(output_video)
    expected = {
        "codec": "h264",
        "width": policy_b_mosaic.WIDTH,
        "height": policy_b_mosaic.HEIGHT,
        "fps": policy_b_mosaic.FPS,
        "frame_count": maximum,
        "pixel_format": "yuv420p",
    }
    for key in ("codec", "width", "height", "frame_count", "pixel_format"):
        if result[key] != expected[key]:
            raise RuntimeError(f"FAIL: Policy-A final video probe {key}: {result}")
    if abs(result["fps"] - expected["fps"]) > 0.01:
        raise RuntimeError(f"FAIL: Policy-A final video FPS: {result}")
    if tile_metrics["black_or_empty_count"]:
        raise RuntimeError(f"FAIL: black or empty Fair-A tile: {tile_metrics}")

    policy_b_manifest = policy_b_mosaic.read_json(POLICY_B_MANIFEST)
    manifest = {
        "schema_version": "fair_a_policy_b_exact_50way_visualization_v1",
        "status": "PASS",
        "visualization_only": True,
        "motion_source_only_change": True,
        "pre_render_verification": pre_render,
        "authoritative_data": metadata,
        "policy_b_visualization_reused_directly": {
            "renderer_class": "tools.render_50_episode_mosaic.G1PortfolioRenderer",
            "render_tiles_function": "tools.render_50_episode_mosaic.render_g1_tiles",
            "compose_function": "tools.render_50_episode_mosaic.compose",
            "label_function": "tools.render_50_episode_mosaic.draw_label",
            "ffmpeg_writer_function": "tools.render_50_episode_mosaic.ffmpeg_writer",
            "layout": policy_b_manifest["layout"],
            "output_convention": policy_b_manifest["output"],
            "camera_viewpoint": "imported unchanged from G1PortfolioRenderer._camera",
            "scene_and_background": "imported unchanged from G1PortfolioRenderer.frame and compose",
            "status_overlay": False,
            "status_overlay_reason": "existing Policy-B visualization supports only EP %02d labels",
        },
        "fair_a_statuses": statuses,
        "output": {
            "video": str(output_video),
            "video_sha256": policy_b_mosaic.sha256(output_video),
            "probe": result,
            "visual_evidence": visual_evidence,
            "tile_metrics": tile_metrics,
            "tile_paths": [str(path) for path in tile_paths],
            "duration_policy": policy_b_manifest["output"]["duration_policy"],
        },
        "validation": {
            "episode_count": len(episodes),
            "episode_ids_unique_complete": [e.index for e in episodes] == list(range(50)),
            "policy_a_policy_b_source_correspondence": "PASS 50/50 ep000..ep049",
            "hard_episodes_35_and_46_included": all(
                statuses[f"ep{index:03d}"] == "HARD_FAIL" for index in (35, 46)
            ),
            "classification_counts": dict(Counter(statuses.values())),
            "nan_frames": 0,
            "joint_mapping_failures": 0,
            "black_or_empty_tiles": tile_metrics["black_or_empty_count"],
            "input_checksums_unchanged": before == after,
            "source_files_modified": False,
            "recompute_retargeting": False,
            "recompute_ik": False,
            "modify_action": False,
            "policy_inference": False,
        },
        "input_checksums_before": before,
        "input_checksums_after": after,
        "exact_generation_command": (
            f"MUJOCO_GL=egl {Path(sys.executable).resolve()} {Path(__file__).resolve()} "
            f"--output-dir {output_dir}"
        ),
    }
    policy_b_mosaic.write_json(output_dir / OUTPUT_MANIFEST_NAME, manifest)
    print(json.dumps(manifest["validation"], indent=2), flush=True)
    print(f"POLICY_A_VIDEO={output_video}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
