#!/usr/bin/env python3
"""Render source-image object-pose diagnostics without changing Proposed B.

Green/cyan markers show the per-episode source-image estimates.  Magenta/orange
markers show the single canonical Isaac pose.  These images are visualization
artifacts only and are intentionally stored outside trajectory directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def task_to_image(point_xy: np.ndarray, inverse_homography: np.ndarray) -> np.ndarray:
    homogeneous = inverse_homography @ np.r_[np.asarray(point_xy, dtype=np.float64), 1.0]
    return homogeneous[:2] / homogeneous[2]


def draw_cross(image: np.ndarray, point: np.ndarray, color: tuple[int, int, int], size: int = 10) -> None:
    x, y = np.rint(point).astype(int)
    cv2.line(image, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), size + 3, color, 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    root = args.review_root.resolve()
    gate = root / "review/dataset_b_gate"
    freeze = json.loads(
        (gate / "motion_freeze/motion_freeze_manifest.json").read_text(encoding="utf-8")
    )
    if freeze.get("status") != "PROPOSED_B_MOTION_FROZEN_FOR_DATASET_AUDIT":
        raise RuntimeError("motion freeze missing")
    policy = json.loads((gate / "classification_policy.json").read_text(encoding="utf-8"))
    scene = json.loads(
        (root / "frozen_approval/scene/scene_layout.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (
            REPOSITORY
            / "outputs/doll_handoff_retargeting/scene_recalibration.json"
        ).read_text(encoding="utf-8")
    )
    if int(calibration["source_count"]) != 50:
        raise RuntimeError("source-image calibration is not complete")

    homography = np.asarray(
        calibration["metric_reference"]["image_to_task_homography"],
        dtype=np.float64,
    )
    inverse_homography = np.linalg.inv(homography)
    canonical_doll = np.asarray(scene["doll"]["center_world_xy_m"], dtype=np.float64)
    canonical_bin = np.asarray(scene["bin"]["center_world_xy_m"], dtype=np.float64)
    threshold = float(policy["canonical_object_pose_material_delta_m"])
    workspace_corners = np.asarray(
        calibration["metric_reference"]["workspace_inner_corners_px_ll_lr_ur_ul"],
        dtype=np.int32,
    )

    visual_dir = gate / "canonical_scene_diagnostic/per_episode_source_object_pose_visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    thumbnails: list[np.ndarray] = []
    for row in calibration["per_episode"]:
        episode = int(row["episode_index"])
        if episode != len(manifest_rows):
            raise RuntimeError("scene recalibration episodes are not stable/sorted")
        source_image = REPOSITORY / row["observations"][0]["image"]
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source_image)
        source_doll = np.asarray(row["doll_initial_center_task_xy_m"], dtype=np.float64)
        source_bin = np.asarray(row["bin_center_task_xy_m"], dtype=np.float64)
        source_doll_px = task_to_image(source_doll, inverse_homography)
        canonical_doll_px = task_to_image(canonical_doll, inverse_homography)
        source_bin_px = task_to_image(source_bin, inverse_homography)
        canonical_bin_px = task_to_image(canonical_bin, inverse_homography)
        doll_delta = float(np.linalg.norm(source_doll - canonical_doll))
        bin_delta = float(np.linalg.norm(source_bin - canonical_bin))

        overlay = image.copy()
        cv2.polylines(
            overlay,
            [workspace_corners.reshape((-1, 1, 2))],
            True,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        draw_cross(overlay, source_doll_px, (40, 220, 40))
        draw_cross(overlay, canonical_doll_px, (220, 40, 220))
        draw_cross(overlay, source_bin_px, (220, 220, 40))
        draw_cross(overlay, canonical_bin_px, (0, 140, 255))
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 83), (10, 10, 10), -1)
        cv2.putText(
            overlay,
            f"PER_EPISODE_SOURCE_OBJECT_POSE_DIAGNOSTIC  ep{episode:03d}  {row['source_name']}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"doll: source GREEN / canonical MAGENTA   delta={doll_delta * 1000.0:.1f} mm",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"bin: source CYAN / canonical ORANGE   delta={bin_delta * 1000.0:.1f} mm   VIS ONLY; ACTIONS UNCHANGED",
            (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        output_path = visual_dir / f"ep{episode:03d}_source_object_pose_diagnostic.png"
        if not cv2.imwrite(str(output_path), overlay):
            raise RuntimeError(f"failed to write {output_path}")
        manifest_rows.append(
            {
                "episode_index": episode,
                "source_name": row["source_name"],
                "source_image": str(source_image),
                "source_image_sha256": sha256_file(source_image),
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "per_episode_source_doll_xy_m": source_doll.tolist(),
                "canonical_doll_xy_m": canonical_doll.tolist(),
                "doll_center_delta_m": doll_delta,
                "material_fixed_doll_mismatch": doll_delta > threshold,
                "per_episode_source_bin_xy_m": source_bin.tolist(),
                "canonical_bin_xy_m": canonical_bin.tolist(),
                "bin_center_delta_m": bin_delta,
                "artifact_label": "PER_EPISODE_SOURCE_OBJECT_POSE_DIAGNOSTIC",
                "used_to_modify_b_motion": False,
            }
        )
        thumbnail = cv2.resize(overlay, (320, 240), interpolation=cv2.INTER_AREA)
        thumbnails.append(thumbnail)

    rows = []
    for row_index in range(5):
        rows.append(np.hstack(thumbnails[row_index * 10 : (row_index + 1) * 10]))
    sheet = np.vstack(rows)
    sheet_path = gate / "canonical_scene_diagnostic/per_episode_source_object_pose_contact_sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"failed to write {sheet_path}")
    visual_manifest = {
        "schema_version": "per_episode_source_object_pose_visualization_v1",
        "label": "PER_EPISODE_SOURCE_OBJECT_POSE_DIAGNOSTIC",
        "episode_count": 50,
        "trajectory_or_action_labels_changed": False,
        "task_frame_or_scene_registration_changed": False,
        "canonical_scene_is_primary_dataset_gate": False,
        "homography_source": str(
            REPOSITORY / "outputs/doll_handoff_retargeting/scene_recalibration.json"
        ),
        "image_to_task_homography": homography.tolist(),
        "material_mismatch_threshold_m": threshold,
        "contact_sheet": str(sheet_path),
        "contact_sheet_sha256": sha256_file(sheet_path),
        "episodes": manifest_rows,
    }
    manifest_path = gate / "canonical_scene_diagnostic/object_pose_visual_manifest.json"
    manifest_path.write_text(
        json.dumps(visual_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PER_EPISODE_SOURCE_OBJECT_POSE_DIAGNOSTIC_COMPLETE",
                "episode_count": len(manifest_rows),
                "material_fixed_doll_mismatch_count": sum(
                    bool(row["material_fixed_doll_mismatch"]) for row in manifest_rows
                ),
                "contact_sheet": str(sheet_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
