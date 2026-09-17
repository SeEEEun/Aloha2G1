#!/usr/bin/env python3
"""Build the predeclared blind, policy-free Dex3 graspability atlas commands.

This builder reads only the frozen G1/Dex3/doll physics contract and an
independently successful scripted physical grasp.  It deliberately contains no
Fair-A, Proposed-B, ACT, held-out episode, or policy path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


OUT = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator"
PROTOCOL = OUT / "BLIND_ATLAS_PROTOCOL.json"
PHYSICS = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
FREEZE = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FREEZE_MANIFEST.json"
SCRIPTED_TRACE = ROOT / "outputs/final_contact_constrained_eval/02_scripted_validation/run_01/event_log.npz"
EXPECTED_PHYSICS_SHA = "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e"
EXPECTED_TRACE_SHA = "ea5ccd9a1cb864fec1f498a23425f6c5fb456880447df993b400bc6e90891e6e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def minimum_jerk_weights(count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def interpolate(first: np.ndarray, last: np.ndarray, count: int) -> np.ndarray:
    weights = minimum_jerk_weights(count)[:, None]
    return first[None] + weights * (last - first)[None]


def world_wrist(g1: G1Kinematics, arm_q: np.ndarray, side: str = "left") -> np.ndarray:
    g1.assign(arm_q)
    local = g1.wrist_pose(side)
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = g1.model_to_world_position(local[:3, 3])
    result[:3, :3] = g1.model_to_world_rotation(local[:3, :3])
    return result


def solve_wrist(
    g1: G1Kinematics,
    target_world: np.ndarray,
    seed_arm: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    seed_arm = np.asarray(seed_arm, dtype=np.float64)
    seed_left = seed_arm[:7].copy()
    lower = g1.arm_limits[:7, 0] + float(thresholds["minimum_joint_limit_margin_rad"])
    upper = g1.arm_limits[:7, 1] - float(thresholds["minimum_joint_limit_margin_rad"])

    def full(left: np.ndarray) -> np.ndarray:
        value = seed_arm.copy()
        value[:7] = left
        return value

    def residual(left: np.ndarray) -> np.ndarray:
        pose = world_wrist(g1, full(left))
        orientation = Rotation.from_matrix(
            target_world[:3, :3] @ pose[:3, :3].T
        ).as_rotvec()
        return np.r_[
            40.0 * (pose[:3, 3] - target_world[:3, 3]),
            2.0 * orientation,
            0.015 * (left - seed_left),
        ]

    solution = least_squares(
        residual,
        seed_left,
        bounds=(lower, upper),
        max_nfev=400,
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
    )
    arm = full(solution.x)
    pose = world_wrist(g1, arm)
    report = {
        "position_residual_m": float(
            np.linalg.norm(pose[:3, 3] - target_world[:3, 3])
        ),
        "orientation_residual_deg": float(
            np.rad2deg(
                Rotation.from_matrix(
                    target_world[:3, :3] @ pose[:3, :3].T
                ).magnitude()
            )
        ),
        "optimizer_cost": float(solution.cost),
        "optimizer_success": bool(solution.success),
        "minimum_joint_limit_margin_rad": float(
            min(
                np.min(arm - g1.arm_limits[:, 0]),
                np.min(g1.arm_limits[:, 1] - arm),
            )
        ),
    }
    if (
        report["position_residual_m"]
        > float(thresholds["maximum_ik_position_residual_m"])
        or report["orientation_residual_deg"]
        > float(thresholds["maximum_ik_orientation_residual_deg"])
    ):
        raise RuntimeError(
            "IK residual exceeds blind offline threshold: "
            f"position={report['position_residual_m']:.6f} m "
            f"orientation={report['orientation_residual_deg']:.3f} deg"
        )
    return arm, report


def hand_in_model_order(
    g1: G1Kinematics,
    command_names: list[str],
    command_values: np.ndarray,
    side: str,
) -> np.ndarray:
    offset = 14 if side == "left" else 21
    lookup = dict(zip(command_names[offset : offset + 7], command_values))
    return np.asarray([lookup[name] for name in g1.hand_joint_names[side]])


def pad_offline_state(
    g1: G1Kinematics,
    arm: np.ndarray,
    hand_model: np.ndarray,
    object_center: np.ndarray,
    object_radii: np.ndarray,
    table_z: float,
) -> dict[str, Any]:
    g1.assign(arm, left_hand=hand_model)
    signed: dict[str, float] = {}
    lower_clearance: dict[str, float] = {}
    centers: dict[str, list[float]] = {}
    for digit in ("thumb", "index", "middle"):
        spec = g1.contacts[f"left_{digit}"]
        model_position, _ = g1.contact_pose("left", digit)
        world_position = g1.model_to_world_position(model_position)
        body_rotation_model = np.asarray(
            g1.data.xmat[g1.body_ids[spec.link]], dtype=np.float64
        ).reshape(3, 3)
        body_rotation_world = g1.model_to_world_rotation(body_rotation_model)
        projected_half_z = float(np.sum(np.abs(body_rotation_world[2]) * spec.half_extent))
        scaled = (world_position - object_center) / object_radii
        approximate_signed = float((np.linalg.norm(scaled) - 1.0) * np.min(object_radii))
        signed[digit] = approximate_signed
        lower_clearance[digit] = float(world_position[2] - projected_half_z - table_z)
        centers[digit] = world_position.tolist()
    return {
        "pad_center_ellipsoid_signed_gap_m": signed,
        "pad_lower_extent_above_table_m": lower_clearance,
        "pad_centers_world_m": centers,
    }


def clearance(g1: G1Kinematics, arm: np.ndarray) -> dict[str, float]:
    state = g1.posture_clearance_state(arm)
    return {
        "torso_m": float(state["TORSO"]["minimum_distance_m"]),
        "cross_arm_m": float(state["CROSS_ARM"]["minimum_distance_m"]),
    }


def lhs_rows(count: int, low: np.ndarray, high: np.ndarray, seed: int) -> np.ndarray:
    sample = qmc.LatinHypercube(d=6, seed=seed).random(count)
    return qmc.scale(sample, low, high)


def split_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def make_samples(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    atlas = protocol["atlas"]
    seed = int(atlas["seed"])
    round_one = atlas["round_1"]
    rows: list[dict[str, Any]] = [
        {
            "sample_id": "R1_ANCHOR_000",
            "stratum": "anchor",
            "delta_translation_object_m": [0.0, 0.0, 0.0],
            "delta_rotation_object_rpy_deg": [0.0, 0.0, 0.0],
            "split": "calibration",
        }
    ]
    definitions = [
        ("core", "C", int(round_one["core_latin_hypercube_samples"]), 11),
        (
            "intermediate",
            "I",
            int(round_one["intermediate_latin_hypercube_samples"]),
            23,
        ),
        ("broad", "B", int(round_one["broad_latin_hypercube_samples"]), 47),
    ]
    for stratum, prefix, count, salt in definitions:
        bounds = round_one[f"{stratum}_bounds"]
        translation = np.asarray(bounds["translation_xyz_m"], dtype=np.float64)
        rotation = np.asarray(bounds["rotation_rpy_deg"], dtype=np.float64)
        low = np.r_[translation[:, 0], rotation[:, 0]]
        high = np.r_[translation[:, 1], rotation[:, 1]]
        values = lhs_rows(count, low, high, seed + salt)
        ids = [f"R1_{prefix}_{index:03d}" for index in range(count)]
        validation_count = max(1, int(round(0.30 * count)))
        validation = set(sorted(ids, key=lambda item: split_key(seed, item))[:validation_count])
        for sample_id, value in zip(ids, values, strict=True):
            rows.append(
                {
                    "sample_id": sample_id,
                    "stratum": stratum,
                    "delta_translation_object_m": value[:3].tolist(),
                    "delta_rotation_object_rpy_deg": value[3:].tolist(),
                    "split": "validation" if sample_id in validation else "calibration",
                }
            )
    return rows


def build_command(
    names: list[str],
    config: dict[str, Any],
    open_arm: np.ndarray,
    target_arm: np.ndarray,
    lift_arm: np.ndarray,
    maximum_step_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fps = float(config["timing"]["control_fps_hz"])
    open_hand = np.asarray(config["hand_states"]["left"]["OPEN"], dtype=np.float64)
    preshape = np.asarray(config["hand_states"]["left"]["PRESHAPE"], dtype=np.float64)
    power = np.asarray(
        config["hand_states"]["left"]["POWER_GRASP_P14"], dtype=np.float64
    )
    right_open = np.asarray(config["hand_states"]["right"]["OPEN"], dtype=np.float64)
    rows: list[np.ndarray] = []
    stages: list[str] = []

    def append(arm: np.ndarray, hand: np.ndarray, stage: str) -> None:
        row = np.zeros(28, dtype=np.float64)
        row[:14] = arm
        row[14:21] = hand
        row[21:28] = right_open
        rows.append(row)
        stages.append(stage)

    for _ in range(int(round(float(config["timing"]["open_hold_s"]) * fps))):
        append(open_arm, open_hand, "OPEN")
    count = int(round(float(config["timing"]["preshape_transition_s"]) * fps))
    for arm, hand in zip(
        interpolate(open_arm, target_arm, count)[1:],
        interpolate(open_hand, preshape, count)[1:],
        strict=True,
    ):
        append(arm, hand, "PRESHAPE")
    count = int(round(float(config["timing"]["power_close_s"]) * fps))
    for hand in interpolate(preshape, power, count)[1:]:
        append(target_arm, hand, "POWER_GRASP")
    for _ in range(int(round(float(config["timing"]["gravity_retention_s"]) * fps))):
        append(target_arm, power, "GRAVITY_RETENTION")
    count = int(round(float(config["timing"]["lift_transition_s"]) * fps))
    for arm in interpolate(target_arm, lift_arm, count)[1:]:
        append(arm, power, "LIFT_5CM")
    for _ in range(int(round(float(config["timing"]["elevated_hold_s"]) * fps))):
        append(lift_arm, power, "HOLD_ELEVATED")
    command = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    max_step = float(np.max(np.abs(np.diff(command[:, :14], axis=0)), initial=0.0))
    threshold = (
        float(maximum_step_rad)
        if maximum_step_rad is not None
        else float(
            json.loads(PROTOCOL.read_text(encoding="utf-8"))["atlas"]
            ["offline_pruning_thresholds"]["maximum_adjacent_commanded_arm_step_rad"]
        )
    )
    if max_step > threshold:
        raise RuntimeError(f"adjacent arm step {max_step:.6f} exceeds {threshold:.6f}")
    return command, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT / "atlas_v2")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite blind atlas preparation: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PREDECLARED_BEFORE_PHYSICAL_LABELING":
        raise RuntimeError("blind protocol is not in the pre-label state")
    if sha256(PHYSICS) != EXPECTED_PHYSICS_SHA or sha256(SCRIPTED_TRACE) != EXPECTED_TRACE_SHA:
        raise RuntimeError("frozen physical dependency hash mismatch")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN" or freeze["scripted_repeatability"] != "3/3 PASS":
        raise RuntimeError("contact-constrained environment is not frozen 3/3")
    config = json.loads(PHYSICS.read_text(encoding="utf-8"))
    if config["active_calibration_profile"] != "P14":
        raise RuntimeError("frozen common P14 full-close primitive changed")

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
            + 0.5 * float(config["object"]["visual_dimensions_m"][2])
            + float(config["object"]["spawn_clearance_above_table_m"]),
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
    samples = make_samples(protocol)
    eligible: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for sample in samples:
        record = dict(sample)
        delta_t = np.asarray(sample["delta_translation_object_m"], dtype=np.float64)
        delta_r = Rotation.from_euler(
            "xyz", sample["delta_rotation_object_rpy_deg"], degrees=True
        ).as_matrix()
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = object_center + base_relative_position + delta_t
        target[:3, :3] = delta_r @ base_relative_rotation
        open_target = np.eye(4, dtype=np.float64)
        open_target[:3, 3] = object_center + base_open_relative_position + delta_t
        open_target[:3, :3] = delta_r @ base_open_relative_rotation
        lift_target = target.copy()
        lift_target[:3, 3] += np.asarray(config["scripted_arm_placement"]["lift_offset_world_xyz_m"])
        try:
            target_arm, target_ik = solve_wrist(g1, target, base_arm, thresholds)
            open_arm, open_ik = solve_wrist(g1, open_target, target_arm, thresholds)
            lift_arm, lift_ik = solve_wrist(g1, lift_target, target_arm, thresholds)
            clearances = {
                "open": clearance(g1, open_arm),
                "target": clearance(g1, target_arm),
                "lift": clearance(g1, lift_arm),
            }
            min_clearance = min(
                value
                for state in clearances.values()
                for value in state.values()
            )
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
            record.update(
                {
                    "offline_status": "ELIGIBLE",
                    "target_object_from_wrist_position_m": (
                        target[:3, 3] - object_center
                    ).tolist(),
                    "target_object_from_wrist_quaternion_xyzw": Rotation.from_matrix(
                        target[:3, :3]
                    ).as_quat().tolist(),
                    "ik": {"open": open_ik, "target": target_ik, "lift": lift_ik},
                    "clearance": clearances,
                    "open_geometry": open_state,
                    "command_frames": int(len(command)),
                    "maximum_adjacent_arm_step_rad": float(
                        np.max(np.abs(np.diff(command[:, :14], axis=0)), initial=0.0)
                    ),
                    "_command": command,
                    "_stages": stages,
                }
            )
            eligible.append(record)
        except Exception as error:
            record.update(
                {
                    "offline_status": "PRUNED",
                    "offline_rejection": f"{type(error).__name__}: {error}",
                }
            )
        records.append(record)

    cap = int(protocol["atlas"]["round_1"]["maximum_physx_labels"])
    if len(eligible) > cap:
        # Deterministic split-aware, stratum-aware selection independent of labels.
        priority = {"anchor": 0, "core": 1, "intermediate": 2, "broad": 3}
        eligible.sort(
            key=lambda row: (
                priority[row["stratum"]],
                0 if row["split"] == "validation" else 1,
                split_key(int(protocol["atlas"]["seed"]), row["sample_id"]),
            )
        )
        selected_ids = {row["sample_id"] for row in eligible[:cap]}
        for row in eligible[cap:]:
            row["offline_status"] = "ELIGIBLE_NOT_SELECTED_DUE_TO_PREDECLARED_PHYSX_CAP"
        eligible = [row for row in eligible if row["sample_id"] in selected_ids]

    command_dir = output / "round_1_commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    for row in eligible:
        path = command_dir / f"{row['sample_id']}.npz"
        temporary = path.with_suffix(".npz.incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                commanded_q_rad=row.pop("_command").astype(np.float32),
                stage=row.pop("_stages"),
                joint_names=np.asarray(names),
                control_fps_hz=np.asarray(30.0),
                side=np.asarray("left"),
                geometry=np.asarray("FROZEN_COMPRESSED_SHORT_55"),
                profile=np.asarray("P14"),
                policy_independent=np.asarray(True),
                representation_neutral_atlas_sample=np.asarray(True),
                sample_id=np.asarray(row["sample_id"]),
                atlas_split=np.asarray(row["split"]),
                delta_translation_object_m=np.asarray(
                    row["delta_translation_object_m"], dtype=np.float64
                ),
                delta_rotation_object_rpy_deg=np.asarray(
                    row["delta_rotation_object_rpy_deg"], dtype=np.float64
                ),
            )
        os.replace(temporary, path)
        row["command"] = str(path)
        row["command_sha256"] = sha256(path)
    for row in records:
        row.pop("_command", None)
        row.pop("_stages", None)

    manifest = {
        "schema_version": "blind_dex3_graspability_round1_commands_v1",
        "status": "COMMANDS_PREPARED_NO_PHYSICS_LABELS",
        "blind_protocol": str(PROTOCOL),
        "blind_protocol_sha256": sha256(PROTOCOL),
        "frozen_environment": str(FREEZE),
        "frozen_environment_sha256": sha256(FREEZE),
        "physics_config": str(PHYSICS),
        "physics_config_sha256": sha256(PHYSICS),
        "independent_scripted_trace": str(SCRIPTED_TRACE),
        "independent_scripted_trace_sha256": sha256(SCRIPTED_TRACE),
        "full_close_7d_rad": config["hand_states"]["left"]["POWER_GRASP_P14"],
        "base_object_center_world_m": object_center,
        "base_object_from_wrist_position_m": base_relative_position,
        "base_object_from_wrist_quaternion_xyzw": Rotation.from_matrix(
            base_relative_rotation
        ).as_quat(),
        "base_open_object_from_wrist_position_m": base_open_relative_position,
        "base_open_object_from_wrist_quaternion_xyzw": Rotation.from_matrix(
            base_open_relative_rotation
        ).as_quat(),
        "proposed_samples": len(samples),
        "offline_pruned_samples": sum(
            row["offline_status"] == "PRUNED" for row in records
        ),
        "physx_selected_samples": len(eligible),
        "calibration_selected": sum(row["split"] == "calibration" for row in eligible),
        "validation_selected": sum(row["split"] == "validation" for row in eligible),
        "ab_artifacts_accessed": False,
        "records": records,
        "selected_sample_ids": [row["sample_id"] for row in eligible],
    }
    dump(output / "ROUND1_COMMAND_MANIFEST.json", manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status", "proposed_samples", "offline_pruned_samples",
        "physx_selected_samples", "calibration_selected", "validation_selected"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
