#!/usr/bin/env python3
"""Reproduce the one-time Doll-Handoff scene XY calibration from real cam_high images.

The recordings do not contain camera intrinsics/extrinsics.  This calibration therefore
uses the black workspace opening as the metric plane and reports the visible-object
planar estimates (including their across-episode spread) without inventing a 3-D camera
model.  The four rail corners and one reference bin-opening polygon are documented
one-time image measurements; object aggregation is automatic and dataset-wide.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_recordings"
OUT = ROOT / "outputs" / "doll_handoff_retargeting"
WORK = OUT / "scene_recalibration_work"
LAYOUT_PATH = ROOT / "isaaclab_doll_handoff_scene" / "scene_layout.json"

FRAME_INDICES = (0, 10, 20)
IMAGE_RELATIVE_TEMPLATE = (
    "images/observation.images.cam_high/episode_000000/frame_{frame:06d}.png"
)

# One-time measurements on the median-aligned 640x480 cam_high view.  Corner order is
# task lower-left, lower-right, upper-right, upper-left.  The inner black-rail edges,
# not their outer silhouettes, define the 0.835 x 0.720 m task opening.
WORKSPACE_INNER_CORNERS_PX = np.asarray(
    [[122.0, 445.0], [526.0, 445.0], [470.0, 195.0], [170.0, 199.0]],
    dtype=np.float32,
)
WORKSPACE_CORNERS_TASK_XY_M = np.asarray(
    [[0.0, 0.0], [0.835, 0.0], [0.835, 0.720], [0.0, 0.720]],
    dtype=np.float32,
)

# Visible inner opening on reference episode 000, frame 000.  It is used only as a
# masked appearance template.  Matching estimates one translation per observation;
# no episode-specific value is written to the scene.
BIN_REFERENCE_EPISODE = "GoPark_20260820_152058"
BIN_REFERENCE_OPENING_POLYGON_PX = np.asarray(
    [[406.0, 370.0], [466.0, 363.0], [478.0, 432.0], [408.0, 441.0]],
    dtype=np.float32,
)
BIN_TEMPLATE_BOUNDS_PX = (395, 355, 485, 450)
BIN_SEARCH_BOUNDS_PX = (360, 330, 560, 470)

# Green/yellow plush segmentation, restricted to the real initial-doll region.  The
# same HSV and ROI rule is used for all recordings and all sampled frames.
DOLL_HSV_LOWER = np.asarray([20, 60, 50], dtype=np.uint8)
DOLL_HSV_UPPER = np.asarray([52, 255, 255], dtype=np.uint8)
DOLL_ROI_XYXY_PX = (120, 360, 250, 465)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(shaped, homography)[0].astype(np.float64)


def _summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "median": np.median(values, axis=0).tolist(),
        "q1": np.percentile(values, 25.0, axis=0).tolist(),
        "q3": np.percentile(values, 75.0, axis=0).tolist(),
        "iqr": (
            np.percentile(values, 75.0, axis=0)
            - np.percentile(values, 25.0, axis=0)
        ).tolist(),
        "minimum": np.min(values, axis=0).tolist(),
        "maximum": np.max(values, axis=0).tolist(),
    }


def _doll_center_px(image: np.ndarray) -> tuple[np.ndarray, int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, DOLL_HSV_LOWER, DOLL_HSV_UPPER)
    x0, y0, x1, y1 = DOLL_ROI_XYXY_PX
    keep = np.zeros_like(mask)
    keep[y0:y1, x0:x1] = 255
    mask = cv2.bitwise_and(mask, keep)
    ys, xs = np.where(mask > 0)
    if xs.size < 500:
        raise RuntimeError(f"Doll segmentation retained only {xs.size} pixels")
    return np.asarray([xs.mean(), ys.mean()], dtype=np.float64), int(xs.size)


def _bin_match(
    image: np.ndarray, template: np.ndarray, template_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    tx0, ty0, _, _ = BIN_TEMPLATE_BOUNDS_PX
    sx0, sy0, sx1, sy1 = BIN_SEARCH_BOUNDS_PX
    result = cv2.matchTemplate(
        image[sy0:sy1, sx0:sx1],
        template,
        cv2.TM_SQDIFF_NORMED,
        mask=template_mask,
    )
    score, _, location, _ = cv2.minMaxLoc(result)
    dx = float(sx0 + location[0] - tx0)
    dy = float(sy0 + location[1] - ty0)
    polygon = BIN_REFERENCE_OPENING_POLYGON_PX + np.asarray([dx, dy])
    return polygon.mean(axis=0), polygon, float(score)


def _annotate(
    episode_names: list[str], per_episode: list[dict], output_path: Path
) -> None:
    selected = [0, 10, 20, 24, 30, 40, 49]
    panels: list[np.ndarray] = []
    for index in selected:
        name = episode_names[index]
        image_path = RAW / name / IMAGE_RELATIVE_TEMPLATE.format(frame=0)
        image = cv2.imread(str(image_path))
        record = per_episode[index]
        bin_polygon = np.rint(record["observations"][0]["bin_opening_polygon_px"]).astype(
            np.int32
        )
        doll = tuple(np.rint(record["observations"][0]["doll_center_px"]).astype(int))
        bin_center = tuple(np.rint(bin_polygon.mean(axis=0)).astype(int))
        cv2.polylines(
            image, [np.rint(WORKSPACE_INNER_CORNERS_PX).astype(np.int32)], True, (255, 0, 255), 2
        )
        cv2.polylines(image, [bin_polygon], True, (255, 255, 0), 2)
        cv2.drawMarker(image, doll, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(image, bin_center, (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            image,
            f"ep{index:03d} frame=0",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    panels.append(np.zeros_like(panels[0]))
    sheet = np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:])))
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    episode_dirs = sorted(path for path in RAW.glob("GoPark_20260820_*") if path.is_dir())
    episode_names = [path.name for path in episode_dirs]
    if len(episode_dirs) != 50:
        raise RuntimeError(f"Expected exactly 50 GoPark_20260820_* sources, got {len(episode_dirs)}")

    layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    old_doll = np.asarray(layout["doll"]["center_world_xy_m"], dtype=np.float64)
    old_bin = np.asarray(layout["bin"]["center_world_xy_m"], dtype=np.float64)
    homography = cv2.getPerspectiveTransform(
        WORKSPACE_INNER_CORNERS_PX, WORKSPACE_CORNERS_TASK_XY_M
    )

    ref_path = RAW / BIN_REFERENCE_EPISODE / IMAGE_RELATIVE_TEMPLATE.format(frame=0)
    reference = cv2.imread(str(ref_path))
    if reference is None:
        raise RuntimeError(f"Unable to read reference image {ref_path}")
    tx0, ty0, tx1, ty1 = BIN_TEMPLATE_BOUNDS_PX
    template = reference[ty0:ty1, tx0:tx1]
    template_mask = np.zeros(template.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(
        template_mask,
        np.rint(BIN_REFERENCE_OPENING_POLYGON_PX - [tx0, ty0]).astype(np.int32),
        255,
    )

    per_episode: list[dict] = []
    doll_episode_xy: list[np.ndarray] = []
    bin_episode_xy: list[np.ndarray] = []
    opening_corners_task: list[np.ndarray] = []
    for episode_index, episode_dir in enumerate(episode_dirs):
        observations: list[dict] = []
        doll_values: list[np.ndarray] = []
        bin_values: list[np.ndarray] = []
        for frame in FRAME_INDICES:
            image_path = episode_dir / IMAGE_RELATIVE_TEMPLATE.format(frame=frame)
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Unable to read {image_path}")
            doll_px, doll_pixels = _doll_center_px(image)
            bin_px, polygon_px, match_score = _bin_match(image, template, template_mask)
            if match_score > 0.03:
                raise RuntimeError(
                    f"Unreliable bin match {match_score:.6f}: {episode_dir.name} frame {frame}"
                )
            doll_xy = _transform(doll_px, homography)[0]
            bin_xy = _transform(bin_px, homography)[0]
            polygon_xy = _transform(polygon_px, homography)
            doll_values.append(doll_xy)
            bin_values.append(bin_xy)
            opening_corners_task.append(polygon_xy)
            observations.append(
                {
                    "frame": frame,
                    "image": str(image_path.relative_to(ROOT)),
                    "image_sha256": _sha256(image_path),
                    "doll_center_px": doll_px.tolist(),
                    "doll_mask_pixels": doll_pixels,
                    "doll_center_task_xy_m": doll_xy.tolist(),
                    "bin_opening_center_px": bin_px.tolist(),
                    "bin_opening_polygon_px": polygon_px.tolist(),
                    "bin_template_sqdiff_normalized": match_score,
                    "bin_center_task_xy_m": bin_xy.tolist(),
                    "bin_opening_polygon_task_xy_m": polygon_xy.tolist(),
                }
            )
        doll_median = np.median(np.asarray(doll_values), axis=0)
        bin_median = np.median(np.asarray(bin_values), axis=0)
        doll_episode_xy.append(doll_median)
        bin_episode_xy.append(bin_median)
        per_episode.append(
            {
                "episode_index": episode_index,
                "source_name": episode_dir.name,
                "doll_initial_center_task_xy_m": doll_median.tolist(),
                "bin_center_task_xy_m": bin_median.tolist(),
                "observations": observations,
            }
        )

    doll_array = np.asarray(doll_episode_xy)
    bin_array = np.asarray(bin_episode_xy)
    new_doll = np.median(doll_array, axis=0)
    new_bin = np.median(bin_array, axis=0)
    all_opening = np.asarray(opening_corners_task)
    opening_median = np.median(all_opening, axis=0)
    opening_bounds = {
        "lower_xy_m": opening_median.min(axis=0).tolist(),
        "upper_xy_m": opening_median.max(axis=0).tolist(),
        "size_xy_m": (opening_median.max(axis=0) - opening_median.min(axis=0)).tolist(),
        "median_polygon_xy_m": opening_median.tolist(),
    }

    result = {
        "schema_version": "doll_handoff_scene_recalibration_v1",
        "status": "ONE_GLOBAL_SOURCE_IMAGE_CALIBRATION",
        "source_count": len(episode_dirs),
        "source_glob": str(RAW / "GoPark_20260820_*"),
        "frames_per_episode": list(FRAME_INDICES),
        "observation_count_per_object": len(episode_dirs) * len(FRAME_INDICES),
        "metric_reference": {
            "workspace_inner_size_xy_m": [0.835, 0.720],
            "workspace_inner_corners_px_ll_lr_ur_ul": WORKSPACE_INNER_CORNERS_PX.tolist(),
            "workspace_corners_task_xy_m_ll_lr_ur_ul": WORKSPACE_CORNERS_TASK_XY_M.tolist(),
            "image_to_task_homography": homography.tolist(),
            "corner_measurement_uncertainty_px": 2.0,
            "measurement_method": "one-time documented inner-rail corner measurement on fixed cam_high view",
        },
        "detectors": {
            "doll": {
                "method": "common HSV segmentation in common ROI; centroid of retained green pixels",
                "hsv_lower_opencv": DOLL_HSV_LOWER.tolist(),
                "hsv_upper_opencv": DOLL_HSV_UPPER.tolist(),
                "roi_xyxy_px": list(DOLL_ROI_XYXY_PX),
            },
            "bin": {
                "method": "common masked template translation of one documented opening polygon",
                "reference_episode": BIN_REFERENCE_EPISODE,
                "reference_frame": 0,
                "reference_opening_polygon_px": BIN_REFERENCE_OPENING_POLYGON_PX.tolist(),
                "template_bounds_xyxy_px": list(BIN_TEMPLATE_BOUNDS_PX),
                "search_bounds_xyxy_px": list(BIN_SEARCH_BOUNDS_PX),
                "acceptance_max_sqdiff_normalized": 0.03,
            },
        },
        "planar_model_limitation": (
            "Recordings contain no cam_high intrinsics/extrinsics. Values are visible-center "
            "estimates under the requested workspace-plane homography; elevated-object "
            "parallax is not silently converted into a fabricated 3-D calibration."
        ),
        "doll": {
            "old_center_task_xy_m": old_doll.tolist(),
            "new_global_center_task_xy_m": new_doll.tolist(),
            "delta_xy_m": (new_doll - old_doll).tolist(),
            "per_episode_robust_statistics_m": _summary(doll_array),
        },
        "bin": {
            "old_center_task_xy_m": old_bin.tolist(),
            "new_global_center_task_xy_m": new_bin.tolist(),
            "delta_xy_m": (new_bin - old_bin).tolist(),
            "per_episode_robust_statistics_m": _summary(bin_array),
            "visible_opening_region_task_plane_estimate": opening_bounds,
        },
        "canonical_scene_policy": (
            "Only the two dataset-wide medians are written to scene_layout.json; no "
            "episode-specific object coordinate is used by retargeting."
        ),
        "per_episode": per_episode,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "scene_recalibration.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (WORK / "per_episode_image_measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["episode_index", "source_name", "doll_x_m", "doll_y_m", "bin_x_m", "bin_y_m"]
        )
        for record in per_episode:
            writer.writerow(
                [
                    record["episode_index"],
                    record["source_name"],
                    *record["doll_initial_center_task_xy_m"],
                    *record["bin_center_task_xy_m"],
                ]
            )
    evidence_path = WORK / "scene_recalibration_evidence.png"
    _annotate(episode_names, per_episode, evidence_path)

    doll_stats = result["doll"]["per_episode_robust_statistics_m"]
    bin_stats = result["bin"]["per_episode_robust_statistics_m"]
    markdown = f"""# Doll-Handoff scene recalibration

Status: **ONE GLOBAL SOURCE-IMAGE CALIBRATION**

The calibration uses all 50 sorted `GoPark_20260820_*` recordings and frames
`{list(FRAME_INDICES)}` before manipulation (150 observations per object). The measured
inner black-frame corners in the fixed 640x480 `cam_high` view are mapped to the
authoritative 0.835 x 0.720 m task opening with a planar homography.

## Result

| object | old XY (m) | global median XY (m) | delta XY (m) | across-episode IQR XY (m) |
|---|---:|---:|---:|---:|
| doll initial center | {old_doll.tolist()} | {new_doll.tolist()} | {(new_doll-old_doll).tolist()} | {doll_stats['iqr']} |
| bin/opening center | {old_bin.tolist()} | {new_bin.tolist()} | {(new_bin-old_bin).tolist()} | {bin_stats['iqr']} |

The observed median bin-opening polygon and bounds are recorded in
`scene_recalibration.json`. Physical proxy dimensions remain PROVISIONAL and were not
re-fit from a single monocular view.

## Evidence and method

- Inner-corner pixels (LL, LR, UR, UL): `{WORKSPACE_INNER_CORNERS_PX.tolist()}`
- Image-to-task homography: `{homography.tolist()}`
- Doll detector: one common HSV/ROI rule, then per-episode median over three frames.
- Bin detector: one reference opening polygon plus one common masked template matcher,
  then per-episode median over three frames.
- Representative annotated evidence: `{evidence_path.relative_to(ROOT)}`
- Every image path, SHA256, pixel measurement, match score, and task-plane estimate is
  retained in `scene_recalibration.json`.

## Uncertainty and limitation

The rail-corner picking uncertainty is approximately +/-2 px. The more conservative
empirical uncertainty is the across-episode IQR: doll `{doll_stats['iqr']}` m and bin
`{bin_stats['iqr']}` m. The recordings provide no `cam_high` intrinsic/extrinsic
calibration, so the visible elevated doll/opening centers are explicitly treated as
planar estimates. No target-G1 trajectory, A/B metric, or episode-specific object
placement was used.

Only the two global medians above are eligible for the canonical scene. Once applied,
the object layout is frozen against A/B outcomes.
"""
    (OUT / "scene_recalibration_report.md").write_text(markdown, encoding="utf-8")

    print(f"SOURCE_EPISODES = {len(episode_dirs)}")
    print(f"DOLL_OLD_XY_M = {old_doll.tolist()}")
    print(f"DOLL_NEW_GLOBAL_MEDIAN_XY_M = {new_doll.tolist()}")
    print(f"BIN_OLD_XY_M = {old_bin.tolist()}")
    print(f"BIN_NEW_GLOBAL_MEDIAN_XY_M = {new_bin.tolist()}")
    print(f"REPORT_JSON = {json_path}")
    print(f"EVIDENCE = {evidence_path}")


if __name__ == "__main__":
    main()
