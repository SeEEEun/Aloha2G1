#!/usr/bin/env python3
"""Correct pad-surface analysis for already solved presentation candidates.

The original search correctly solved IK and measured collision clearance, but
its posture-clearance helper reset the hand before the later pad query.  This
tool reuses the saved arm solutions and restores the exact P14/R14 hand states
before measuring pads.  It performs no IK, simulation, or input mutation.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import hand_model  # noqa: E402
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import authoritative_joint_ranges  # noqa: E402
from tools.search_verified_right_endpoint_presentations import (  # noqa: E402
    CONFIG,
    EVIDENCE,
    OUT,
    SEARCH,
    read_json,
    signed_ellipsoid_distance,
)


INPUTS = (
    SEARCH / "OFFLINE_PRESENTATION_SEARCH.json",
    SEARCH / "OFFLINE_PRESENTATION_REFINEMENT.json",
)
TOP_COUNT = 16


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def finite(item: Any) -> Any:
    if isinstance(item, dict):
        return {key: finite(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [finite(value) for value in item]
    if isinstance(item, np.ndarray):
        return finite(item.tolist())
    if isinstance(item, np.generic):
        return finite(item.item())
    if isinstance(item, float) and not np.isfinite(item):
        return None
    return item


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def main() -> int:
    config = read_json(CONFIG)
    evidence = read_json(EVIDENCE)
    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    names, _ = authoritative_joint_ranges()
    left_p14 = hand_model(g1, names, config, "left", "POWER_GRASP_P14")
    right_transport = np.asarray(
        evidence["verified_right_transport_grasp"]["right_hand_7d_rad"], dtype=np.float64
    )
    radii = np.asarray(config["geometry_candidates"][0]["dimensions_m"], dtype=np.float64) / 2.0
    combined: list[dict[str, Any]] = []
    round_summaries = []
    for source in INPUTS:
        payload = read_json(source)
        source_round = payload.get(
            "search_round",
            "refinement" if "REFINEMENT" in source.stem else "coarse",
        )
        corrected_rows = []
        for original in payload["candidates"]:
            row = dict(original)
            if "left_arm_q_rad" not in row:
                row["status"] = "OFFLINE_FAIL"
                corrected_rows.append(row)
                continue
            arm = np.r_[
                np.asarray(row["left_arm_q_rad"], dtype=np.float64),
                np.asarray(row["right_arm_q_rad"], dtype=np.float64),
            ]
            object_pose = np.asarray(row["target_object_pose_world"], dtype=np.float64)
            g1.assign(arm, left_p14, right_transport)
            pad_local = {}
            pad_signed = {}
            for digit in ("thumb", "index", "middle"):
                pad_model, _ = g1.contact_pose("left", digit)
                pad_world = g1.model_to_world_position(pad_model)
                local = object_pose[:3, :3].T @ (pad_world - object_pose[:3, 3])
                pad_local[digit] = local
                pad_signed[digit] = signed_ellipsoid_distance(local, radii)
            max_surface = max(abs(value) for value in pad_signed.values())
            access = row["accessibility"]
            collisions = row["collision_frame_counts"]
            offline_pass = bool(
                not sum(collisions.values())
                and access["minimum_left_right_clearance_m"] >= 0.0
                and access["conservative_right_hand_table_clearance_m"] >= 0.0
                and access["torso_clearance_m"] >= 0.0
                and access["cross_arm_clearance_m"] >= 0.0
                and max_surface <= 0.015
                and row["exact_right_endpoint_max_error_rad"] <= 1.0e-12
            )
            score = float(
                4.0 * access["minimum_left_right_clearance_m"]
                - 1.5 * max_surface
                - 0.001
                * np.linalg.norm(
                    np.asarray(row["left_arm_q_rad"])
                    - np.asarray(payload["candidates"][0]["left_arm_q_rad"])
                )
            )
            row.update(
                {
                    "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
                    "left_pad_center_object_m": pad_local,
                    "left_pad_ellipsoid_signed_distance_m": pad_signed,
                    "maximum_left_pad_surface_error_m": max_surface,
                    "ranking_score": score,
                    "pad_measurement_correction": (
                        "P14 restored after posture-clearance helper reset; IK and collision values unchanged"
                    ),
                }
            )
            corrected_rows.append(row)
            row_with_round = dict(row)
            row_with_round["source_round"] = source_round
            combined.append(row_with_round)
        passing = [row for row in corrected_rows if row["status"] == "OFFLINE_PASS"]
        corrected = dict(payload)
        corrected["schema_version"] = "exact_verified_endpoint_presentation_search_reanalyzed_v1"
        corrected["diagnostic_correction"] = (
            "Only LEFT pad positions were recomputed after restoring P14; saved IK and collision results are unchanged."
        )
        corrected["candidates"] = corrected_rows
        corrected["offline_pass_count"] = len(passing)
        corrected["top_physics_candidates"] = [
            row["candidate_id"]
            for row in sorted(passing, key=lambda item: item["ranking_score"], reverse=True)[:TOP_COUNT]
        ]
        target = source.with_name(source.stem + "_REANALYZED.json")
        atomic_json(target, corrected)
        round_summaries.append(
            {
                "source_round": source_round,
                "source": str(source),
                "corrected": str(target),
                "candidate_count": len(corrected_rows),
                "offline_pass_count": len(passing),
            }
        )

    passing_all = sorted(
        (row for row in combined if row["status"] == "OFFLINE_PASS"),
        key=lambda item: item["ranking_score"],
        reverse=True,
    )
    top = passing_all[:TOP_COUNT]
    selection = {
        "schema_version": "exact_verified_endpoint_combined_selection_v1",
        "diagnostic_correction": (
            "The initial report queried pads after a hand-resetting clearance helper. Reanalysis restores LEFT P14 and exact RIGHT transport hand before pad measurement."
        ),
        "rounds": round_summaries,
        "total_candidate_count": len(combined),
        "offline_pass_count": len(passing_all),
        "top_physics_candidate_count": len(top),
        "top_physics_candidates": [row["candidate_id"] for row in top],
        "top_candidate_rows": top,
        "right_endpoint_changed": False,
        "doll_or_physics_changed": False,
    }
    atomic_json(SEARCH / "OFFLINE_PRESENTATION_COMBINED_SELECTION.json", selection)
    lines = []
    for rank, row in enumerate(top, start=1):
        lines.append(
            f"| {rank} | {row['candidate_id']} | {row['source_round']} | "
            f"{1000*row['accessibility']['minimum_left_right_clearance_m']:.3f} | "
            f"{1000*row['maximum_left_pad_surface_error_m']:.3f} | {row['ranking_score']:.6f} |"
        )
    markdown = """# Corrected exact-endpoint presentation selection

The original IK and collision audit was valid. A diagnostics-only ordering bug
reset the hand before pad measurement; the rows below restore LEFT P14 and the
exact RIGHT transport hand before querying pads. No candidate was re-solved.

| Rank | Candidate | Round | Hand clearance (mm) | LEFT pad error (mm) | Score |
|---:|---|---|---:|---:|---:|
""" + "\n".join(lines) + f"""

Corrected offline PASS: {len(passing_all)}/{len(combined)}. The top
{len(top)} candidates are frozen for path construction and progressive physics.
"""
    atomic_text(SEARCH / "OFFLINE_PRESENTATION_COMBINED_SELECTION.md", markdown)
    csv_path = SEARCH / "OFFLINE_PRESENTATION_COMBINED_SELECTION.csv"
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "candidate_id",
            "source_round",
            "status",
            "left_long_axis_shift_m",
            "left_local_yaw_deg",
            "left_local_pitch_deg",
            "ranking_score",
            "maximum_left_pad_surface_error_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    os.replace(temporary, csv_path)
    atomic_text(
        OUT / "CURRENT_STATUS.md",
        "# Methodology-preserving completion status\n\n"
        "Current gate: STAGE_A_PATH_CONSTRUCTION\n\n"
        f"- Corrected exact-endpoint offline candidates: {len(passing_all)}/{len(combined)} PASS.\n"
        f"- Top physics candidates frozen: {len(top)}.\n"
        "- Doll, verified RIGHT transport grasp, A/B inputs: unchanged.\n"
        "- Full task/freeze/ACT evaluation: not yet eligible.\n",
    )
    print(json.dumps({
        "total_candidate_count": len(combined),
        "offline_pass_count": len(passing_all),
        "top_physics_candidates": [row["candidate_id"] for row in top],
    }, indent=2))
    return 0 if top else 2


if __name__ == "__main__":
    raise SystemExit(main())
