#!/usr/bin/env python3
"""Build the preregistered synthetic-only graspability atlas extension."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import qmc


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_blind_dex3_graspability_atlas import (  # noqa: E402
    PHYSICS,
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


OUT = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator_v2"
PROTOCOL = OUT / "BLIND_ATLAS_EXTENSION_PROTOCOL.json"
SOURCE = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator"
SOURCE_ATLAS = SOURCE / "atlas_v2"
DESTINATION = OUT / "atlas_extension"


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


def compose_jitter(seed: dict[str, Any], jitter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(seed["delta_translation_object_m"], dtype=np.float64) + jitter[:3]
    rotation = Rotation.from_euler("xyz", jitter[3:], degrees=True) * Rotation.from_euler(
        "xyz", seed["delta_rotation_object_rpy_deg"], degrees=True
    )
    return translation, rotation.as_euler("xyz", degrees=True)


def interpolate_pose(
    positive: dict[str, Any], negative: dict[str, Any], fraction: float
) -> tuple[np.ndarray, Rotation]:
    p_t = np.asarray(positive["delta_translation_object_m"], dtype=np.float64)
    n_t = np.asarray(negative["delta_translation_object_m"], dtype=np.float64)
    translation = (1.0 - fraction) * p_t + fraction * n_t
    rotations = Rotation.from_euler(
        "xyz",
        [positive["delta_rotation_object_rpy_deg"], negative["delta_rotation_object_rpy_deg"]],
        degrees=True,
    )
    rotation = Slerp([0.0, 1.0], rotations)([fraction])[0]
    return translation, rotation


def source_rows_and_labels() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    for round_index in (1, 2):
        manifest = json.loads(
            (SOURCE_ATLAS / f"ROUND{round_index}_COMMAND_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        for row in manifest["records"]:
            rows[row["sample_id"]] = row
        for sample_id in manifest["selected_sample_ids"]:
            label_path = (
                SOURCE_ATLAS
                / f"round_{round_index}_physx"
                / sample_id
                / "PHYSICAL_LABEL.json"
            )
            labels.append(json.loads(label_path.read_text(encoding="utf-8")))
    return rows, labels


def main() -> int:
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        raise FileExistsError(f"refusing to overwrite atlas extension: {DESTINATION}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PREDECLARED_BEFORE_EXTENSION_PHYSICS_LABELING":
        raise RuntimeError("extension protocol is not preregistered")
    freeze = ROOT / protocol["preserved_common_execution_layer"]["freeze_manifest"]
    if sha256(freeze) != protocol["preserved_common_execution_layer"]["freeze_manifest_sha256"]:
        raise RuntimeError("common execution-layer freeze hash changed")
    source_manifest = ROOT / protocol["source_atlas"]["manifest"]
    if sha256(source_manifest) != protocol["source_atlas"]["manifest_sha256"]:
        raise RuntimeError("source physical atlas hash changed")
    source_rows, source_labels = source_rows_and_labels()
    positives = sorted(
        [label for label in source_labels if label["physically_graspable"]],
        key=lambda row: row["sample_id"],
    )
    negatives = sorted(
        [label for label in source_labels if not label["physically_graspable"]],
        key=lambda row: row["sample_id"],
    )
    if len(positives) != int(protocol["source_atlas"]["physical_positive_seed_count"]):
        raise RuntimeError("source physical-positive seed count changed")
    if len(negatives) != int(
        protocol["source_atlas"]["physical_negative_boundary_partner_count"]
    ):
        raise RuntimeError("source physical-negative count changed")

    sampling = protocol["extension_sampling"]
    random_seed = int(sampling["seed"])
    proposed: list[dict[str, Any]] = []
    local_cfg = sampling["positive_local"]
    local_bounds_t = np.asarray(local_cfg["translation_xyz_m"], dtype=np.float64)
    local_bounds_r = np.asarray(local_cfg["rotation_rpy_deg"], dtype=np.float64)
    low = np.r_[local_bounds_t[:, 0], local_bounds_r[:, 0]]
    high = np.r_[local_bounds_t[:, 1], local_bounds_r[:, 1]]
    boundary_cfg = sampling["boundary"]
    for seed_index, positive_label in enumerate(positives):
        positive = source_rows[positive_label["sample_id"]]
        local_values = qmc.scale(
            qmc.LatinHypercube(d=6, seed=random_seed + 101 * seed_index).random(
                int(local_cfg["samples_per_positive_seed"])
            ),
            low,
            high,
        )
        for local_index, jitter in enumerate(local_values):
            translation, rpy = compose_jitter(positive, jitter)
            proposed.append(
                {
                    "sample_id": f"E2_L_{seed_index:02d}_{local_index:02d}",
                    "family": "positive_local",
                    "positive_seed": positive_label["sample_id"],
                    "delta_translation_object_m": translation.tolist(),
                    "delta_rotation_object_rpy_deg": rpy.tolist(),
                }
            )
        positive_feature = feature(positive)
        nearest_label = min(
            negatives,
            key=lambda label: float(
                np.linalg.norm(feature(source_rows[label["sample_id"]]) - positive_feature)
            ),
        )
        nearest = source_rows[nearest_label["sample_id"]]
        for boundary_index, fraction in enumerate(
            boundary_cfg["positive_to_negative_fractions"]
        ):
            translation, rotation = interpolate_pose(positive, nearest, float(fraction))
            rng = np.random.default_rng(
                random_seed + 10000 + 101 * seed_index + boundary_index
            )
            direction_t = rng.normal(size=3)
            direction_t /= max(float(np.linalg.norm(direction_t)), 1e-12)
            direction_r = rng.normal(size=3)
            direction_r /= max(float(np.linalg.norm(direction_r)), 1e-12)
            translation += direction_t * float(
                boundary_cfg["deterministic_tangential_jitter_translation_m"]
            )
            rotation = Rotation.from_rotvec(
                np.deg2rad(
                    direction_r
                    * float(boundary_cfg["deterministic_tangential_jitter_rotation_deg"])
                )
            ) * rotation
            proposed.append(
                {
                    "sample_id": f"E2_B_{seed_index:02d}_{boundary_index:02d}",
                    "family": "positive_negative_boundary",
                    "positive_seed": positive_label["sample_id"],
                    "negative_partner": nearest_label["sample_id"],
                    "boundary_fraction": float(fraction),
                    "delta_translation_object_m": translation.tolist(),
                    "delta_rotation_object_rpy_deg": rotation.as_euler(
                        "xyz", degrees=True
                    ).tolist(),
                }
            )
    if len(proposed) != int(sampling["total_proposed"]):
        raise RuntimeError("predeclared extension proposal count mismatch")

    config = json.loads(PHYSICS.read_text(encoding="utf-8"))
    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    primitive_path = Path(config["source_arm_primitives"]["left"])
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    command_source = (
        ROOT
        / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm/bin_calibrated_full_command.npz"
    )
    with np.load(command_source, allow_pickle=False) as archive:
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
    thresholds = sampling["offline_pruning_thresholds"]
    open_model = hand_in_model_order(
        g1,
        names,
        np.asarray(config["hand_states"]["left"]["OPEN"], dtype=np.float64),
        "left",
    )
    command_dir = DESTINATION / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for sample in proposed:
        record = dict(sample)
        delta_t = np.asarray(sample["delta_translation_object_m"], dtype=np.float64)
        delta_rotation = Rotation.from_euler(
            "xyz", sample["delta_rotation_object_rpy_deg"], degrees=True
        ).as_matrix()
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
            minimum_clearance = min(
                value for state in clearances.values() for value in state.values()
            )
            if minimum_clearance < float(
                thresholds["minimum_model_torso_or_cross_arm_clearance_m"]
            ):
                raise RuntimeError(f"model self-clearance {minimum_clearance:.6f} m")
            open_state = pad_offline_state(
                g1,
                open_arm,
                open_model,
                object_center,
                object_radii,
                float(config["object"]["table_surface_world_z_m"]),
            )
            minimum_gap = min(open_state["pad_center_ellipsoid_signed_gap_m"].values())
            minimum_table = min(open_state["pad_lower_extent_above_table_m"].values())
            if minimum_gap < float(
                thresholds["minimum_open_pad_center_ellipsoid_signed_gap_m"]
            ):
                raise RuntimeError(f"open pad/object penetration proxy {minimum_gap:.6f} m")
            if minimum_table < float(
                thresholds["minimum_open_pad_lower_extent_above_table_m"]
            ):
                raise RuntimeError(f"open pad/table clearance {minimum_table:.6f} m")
            command, stages = build_command(
                names,
                config,
                open_arm,
                target_arm,
                lift_arm,
                maximum_step_rad=float(
                    thresholds["maximum_adjacent_commanded_arm_step_rad"]
                ),
            )
            command_path = command_dir / f"{sample['sample_id']}.npz"
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
                    representation_neutral_atlas_extension=np.asarray(True),
                    sample_id=np.asarray(sample["sample_id"]),
                    delta_translation_object_m=delta_t,
                    delta_rotation_object_rpy_deg=np.asarray(
                        sample["delta_rotation_object_rpy_deg"], dtype=np.float64
                    ),
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
        "schema_version": "blind_dex3_graspability_atlas_extension_commands_v2",
        "status": "COMMANDS_PREPARED_NO_EXTENSION_PHYSICS_LABELS",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "source_atlas": str(source_manifest),
        "source_atlas_sha256": sha256(source_manifest),
        "common_execution_layer_freeze": str(freeze),
        "common_execution_layer_freeze_sha256": sha256(freeze),
        "proposed_samples": len(proposed),
        "offline_pruned_samples": sum(
            row["offline_status"] == "PRUNED" for row in records
        ),
        "physx_selected_samples": len(selected),
        "selected_sample_ids": selected,
        "ab_artifacts_accessed": False,
        "records": records,
    }
    dump(DESTINATION / "EXTENSION_COMMAND_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "proposed_samples",
                    "offline_pruned_samples",
                    "physx_selected_samples",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
