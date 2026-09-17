#!/usr/bin/env python3
"""Prepare the one predeclared calibration-only atlas boundary refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_blind_dex3_graspability_atlas import (  # noqa: E402
    PHYSICS,
    PROTOCOL,
    build_command,
    clearance,
    hand_in_model_order,
    pad_offline_state,
    sha256,
    solve_wrist,
    world_wrist,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


DEFAULT_ATLAS = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator/atlas_v2"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def feature(row: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(row["delta_translation_object_m"], dtype=np.float64) / 0.01
    rotation = np.rad2deg(
        Rotation.from_euler(
            "xyz", row["delta_rotation_object_rpy_deg"], degrees=True
        ).as_rotvec()
    ) / 10.0
    return np.r_[translation, rotation]


def midpoint(first: dict[str, Any], second: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    translation = 0.5 * (
        np.asarray(first["delta_translation_object_m"], dtype=np.float64)
        + np.asarray(second["delta_translation_object_m"], dtype=np.float64)
    )
    rotations = Rotation.from_euler(
        "xyz",
        [
            first["delta_rotation_object_rpy_deg"],
            second["delta_rotation_object_rpy_deg"],
        ],
        degrees=True,
    )
    middle = Slerp([0.0, 1.0], rotations)([0.5])[0]
    return translation, middle.as_euler("xyz", degrees=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS)
    args = parser.parse_args()
    atlas = args.atlas_root.resolve()
    destination = atlas / "round_2_commands"
    manifest_path = atlas / "ROUND2_COMMAND_MANIFEST.json"
    if manifest_path.exists() or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError("refusing to overwrite round-2 refinement")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    round1 = json.loads((atlas / "ROUND1_COMMAND_MANIFEST.json").read_text(encoding="utf-8"))
    row_by_id = {row["sample_id"]: row for row in round1["records"]}
    labels: list[dict[str, Any]] = []
    for sample_id in round1["selected_sample_ids"]:
        path = atlas / "round_1_physx" / sample_id / "PHYSICAL_LABEL.json"
        if not path.is_file():
            raise RuntimeError(f"round-1 physics is incomplete: {sample_id}")
        labels.append(json.loads(path.read_text(encoding="utf-8")))
    calibration = [label for label in labels if label["split"] == "calibration"]
    positives = [label for label in calibration if label["physically_graspable"]]
    negatives = [label for label in calibration if not label["physically_graspable"]]
    if not positives or not negatives:
        raise RuntimeError(
            "round-2 boundary refinement requires opposite physical labels in calibration"
        )
    pairs: list[tuple[float, str, str]] = []
    for positive in positives:
        for negative in negatives:
            distance = float(
                np.linalg.norm(
                    feature(row_by_id[positive["sample_id"]])
                    - feature(row_by_id[negative["sample_id"]])
                )
            )
            pairs.append((distance, positive["sample_id"], negative["sample_id"]))
    pairs.sort(key=lambda value: (value[0], value[1], value[2]))
    maximum = int(protocol["atlas"]["round_2"]["maximum_samples"])
    chosen = pairs[:maximum]

    config = json.loads(PHYSICS.read_text(encoding="utf-8"))
    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    primitive_path = Path(config["source_arm_primitives"]["left"])
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(
        ROOT
        / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm/bin_calibrated_full_command.npz",
        allow_pickle=False,
    ) as archive:
        names = archive["joint_names"].astype(str).tolist()
    base_arm = np.asarray(primitive["approach_arm_q_rad"][-1], dtype=np.float64)
    base_open_arm = np.asarray(primitive["approach_arm_q_rad"][0], dtype=np.float64)
    base_wrist = world_wrist(g1, base_arm)
    base_open_wrist = world_wrist(g1, base_open_arm)
    object_center = np.asarray(
        [
            *config["object"]["center_world_xy_m_by_side"]["left"],
            float(config["object"]["table_surface_world_z_m"])
            + 0.5 * float(config["object"]["visual_dimensions_m"][2]),
        ],
        dtype=np.float64,
    )
    object_radii = 0.5 * np.asarray(
        config["geometry_candidates"][0]["dimensions_m"], dtype=np.float64
    )
    base_relative_position = base_wrist[:3, 3] - object_center
    base_relative_rotation = base_wrist[:3, :3].copy()
    base_open_relative_position = base_open_wrist[:3, 3] - object_center
    base_open_relative_rotation = base_open_wrist[:3, :3].copy()
    thresholds = protocol["atlas"]["offline_pruning_thresholds"]
    open_model = hand_in_model_order(
        g1,
        names,
        np.asarray(config["hand_states"]["left"]["OPEN"], dtype=np.float64),
        "left",
    )
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (pair_distance, positive_id, negative_id) in enumerate(chosen):
        positive = row_by_id[positive_id]
        negative = row_by_id[negative_id]
        delta_t, delta_rpy = midpoint(positive, negative)
        sample_id = f"R2_M_{index:03d}"
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "stratum": "boundary_refinement",
            "split": "calibration",
            "positive_parent": positive_id,
            "negative_parent": negative_id,
            "parent_feature_distance": pair_distance,
            "delta_translation_object_m": delta_t.tolist(),
            "delta_rotation_object_rpy_deg": delta_rpy.tolist(),
        }
        delta_rotation = Rotation.from_euler("xyz", delta_rpy, degrees=True).as_matrix()
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = object_center + base_relative_position + delta_t
        target[:3, :3] = delta_rotation @ base_relative_rotation
        open_target = np.eye(4, dtype=np.float64)
        open_target[:3, 3] = object_center + base_open_relative_position + delta_t
        open_target[:3, :3] = delta_rotation @ base_open_relative_rotation
        lift_target = target.copy()
        lift_target[:3, 3] += np.asarray(
            config["scripted_arm_placement"]["lift_offset_world_xyz_m"]
        )
        try:
            target_arm, target_ik = solve_wrist(g1, target, base_arm, thresholds)
            open_arm, open_ik = solve_wrist(g1, open_target, target_arm, thresholds)
            lift_arm, lift_ik = solve_wrist(g1, lift_target, target_arm, thresholds)
            clearances = {
                "open": clearance(g1, open_arm),
                "target": clearance(g1, target_arm),
                "lift": clearance(g1, lift_arm),
            }
            min_clearance = min(value for state in clearances.values() for value in state.values())
            if min_clearance < float(
                thresholds["minimum_model_torso_or_cross_arm_clearance_m"]
            ):
                raise RuntimeError(f"model self-clearance {min_clearance:.6f} m")
            open_state = pad_offline_state(
                g1,
                open_arm,
                open_model,
                object_center,
                object_radii,
                float(config["object"]["table_surface_world_z_m"]),
            )
            min_gap = min(open_state["pad_center_ellipsoid_signed_gap_m"].values())
            min_table = min(open_state["pad_lower_extent_above_table_m"].values())
            if min_gap < float(
                thresholds["minimum_open_pad_center_ellipsoid_signed_gap_m"]
            ):
                raise RuntimeError(f"open pad/object penetration proxy {min_gap:.6f} m")
            if min_table < float(
                thresholds["minimum_open_pad_lower_extent_above_table_m"]
            ):
                raise RuntimeError(f"open pad/table clearance {min_table:.6f} m")
            command, stages = build_command(names, config, open_arm, target_arm, lift_arm)
            command_path = destination / f"{sample_id}.npz"
            temporary = command_path.with_suffix(".npz.incomplete")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    commanded_q_rad=command.astype(np.float32),
                    stage=stages,
                    joint_names=np.asarray(names),
                    control_fps_hz=np.asarray(30.0),
                    side=np.asarray("left"),
                    geometry=np.asarray("FROZEN_COMPRESSED_SHORT_55"),
                    profile=np.asarray("P14"),
                    policy_independent=np.asarray(True),
                    representation_neutral_atlas_sample=np.asarray(True),
                    sample_id=np.asarray(sample_id),
                    atlas_split=np.asarray("calibration"),
                    delta_translation_object_m=delta_t,
                    delta_rotation_object_rpy_deg=delta_rpy,
                )
            os.replace(temporary, command_path)
            record.update(
                {
                    "offline_status": "ELIGIBLE",
                    "ik": {"open": open_ik, "target": target_ik, "lift": lift_ik},
                    "clearance": clearances,
                    "open_geometry": open_state,
                    "command": str(command_path),
                    "command_sha256": sha256(command_path),
                }
            )
        except Exception as error:
            record.update(
                {
                    "offline_status": "PRUNED",
                    "offline_rejection": f"{type(error).__name__}: {error}",
                }
            )
        records.append(record)
    selected = [row["sample_id"] for row in records if row["offline_status"] == "ELIGIBLE"]
    manifest = {
        "schema_version": "blind_dex3_graspability_round2_commands_v1",
        "status": "COMMANDS_PREPARED_NO_PHYSICS_LABELS",
        "blind_protocol": str(PROTOCOL),
        "blind_protocol_sha256": sha256(PROTOCOL),
        "round1_manifest": str(atlas / "ROUND1_COMMAND_MANIFEST.json"),
        "round1_manifest_sha256": sha256(atlas / "ROUND1_COMMAND_MANIFEST.json"),
        "selection_rule": protocol["atlas"]["round_2"]["rule"],
        "validation_labels_used": False,
        "ab_artifacts_accessed": False,
        "proposed_samples": len(chosen),
        "offline_pruned_samples": sum(row["offline_status"] == "PRUNED" for row in records),
        "physx_selected_samples": len(selected),
        "selected_sample_ids": selected,
        "records": records,
    }
    dump(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status", "proposed_samples", "offline_pruned_samples", "physx_selected_samples"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
