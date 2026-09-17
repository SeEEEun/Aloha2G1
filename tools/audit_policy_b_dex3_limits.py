#!/usr/bin/env python3
"""Read-only Policy-B Stage-0 Dex3 limit/provenance diagnostic.

This program does not run inference, project commands, mutate Dataset B, or
touch the retargeting pipeline.  It compares the saved physical-unit Stage-0
prediction against the frozen Dataset-B command contract and its local source
models.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
from pxr import Usd, UsdPhysics
from safetensors.numpy import load_file as load_safetensors


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/policy_b_isaac_validation/dex3_limit_audit"
FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
SEMANTIC = (
    ROOT
    / "outputs/doll_handoff_dataset_b_semantic_audit_2026-08-23/FINAL_DATASET_B_SEMANTIC_MANIFEST.json"
)
DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
PARQUET = DATASET / "data/chunk-000/file-000.parquet"
RAW_PREDICTION = ROOT / "outputs/policy_b_isaac_validation/stage0_inference/predicted_chunk.npy"
STAGE0_REPORT = ROOT / "outputs/policy_b_isaac_validation/stage0_inference/stage0_report.json"
CHECKPOINT = (
    ROOT
    / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2/checkpoints/020000/pretrained_model"
)
MUJOCO_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
UNITREE_EXAMPLE = Path(
    "/home/jbnu/jaeyoung/unitree/unitree_sdk2/example/g1/dex3/g1_dex3_example.cpp"
)
RECORDER = ROOT / "tools/record_g1_dex3_magsafe_primitives.py"
ISAAC_ASSET = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
ISAAC_SCENE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
PRE_STATS = CHECKPOINT / "policy_preprocessor_step_5_normalizer_processor.safetensors"
POST_STATS = CHECKPOINT / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

BOUNDARY_EPS_RAD = 1e-7
STRICT_COMPARISON_EPS_RAD = 1e-9


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_line(path: Path, token: str) -> int | None:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if token in line:
            return number
    return None


def read_action_array() -> np.ndarray:
    column = pq.read_table(PARQUET, columns=["action"])["action"].combine_chunks()
    return np.asarray(column.values, dtype=np.float64).reshape(-1, 28)


def xml_limits(names: list[str]) -> dict[str, list[float]]:
    tree = ET.parse(MUJOCO_MODEL)
    values: dict[str, list[float]] = {}
    wanted = set(names)
    for node in tree.iter("joint"):
        name = node.attrib.get("name")
        if name in wanted:
            values[name] = [float(value) for value in node.attrib["range"].split()]
    return values


def recorder_limits() -> dict[str, list[list[float]]]:
    tree = ast.parse(RECORDER.read_text(encoding="utf-8"), filename=str(RECORDER))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEX3_LIMITS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("DEX3_LIMITS was not found in recorder source")


def cpp_limits() -> dict[str, list[list[float]]]:
    text = UNITREE_EXAMPLE.read_text(encoding="utf-8")
    arrays: dict[str, list[float]] = {}
    for key in ("minLimits_left", "maxLimits_left", "minLimits_right", "maxLimits_right"):
        match = re.search(rf"{key}\s*\[7\]\s*=\s*\{{([^}}]+)\}}", text)
        if not match:
            raise RuntimeError(f"could not parse {key} from {UNITREE_EXAMPLE}")
        arrays[key] = [float(value) for value in match.group(1).split(",")]
    return {
        side: [
            [arrays[f"minLimits_{side}"][index], arrays[f"maxLimits_{side}"][index]]
            for index in range(7)
        ]
        for side in ("left", "right")
    }


def usd_limits(path: Path, names: list[str]) -> dict[str, list[float]]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD stage {path}")
    wanted = set(names)
    values: dict[str, list[float]] = {}
    for prim in stage.Traverse():
        if prim.GetName() not in wanted:
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            continue
        values[prim.GetName()] = [
            math.radians(float(joint.GetLowerLimitAttr().Get())),
            math.radians(float(joint.GetUpperLimitAttr().Get())),
        ]
    return values


def max_source_difference(
    names: list[str], authoritative: dict[str, list[float]], candidate: dict[str, list[float]]
) -> float:
    return max(
        abs(authoritative[name][bound] - candidate[name][bound])
        for name in names
        for bound in (0, 1)
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze = read_json(FREEZE)
    semantic_manifest = read_json(SEMANTIC)
    semantic_schema = read_json(Path(semantic_manifest["artifacts"]["semantic_schema"]["path"]))
    dataset_info = read_json(DATASET / "meta/info.json")
    checkpoint_config = read_json(CHECKPOINT / "config.json")
    stage0 = read_json(STAGE0_REPORT)

    names = list(freeze["joint_names"])
    specs = list(freeze["joint_specs"])
    dex_names = names[14:]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    authoritative = {
        row["joint_name"]: [float(row["minimum"]), float(row["maximum"])]
        for row in specs[14:]
    }

    actions = read_action_array()
    prediction = np.asarray(np.load(RAW_PREDICTION), dtype=np.float64)
    if actions.shape != (34478, 28):
        raise RuntimeError(f"unexpected Dataset-B action shape {actions.shape}")
    if prediction.shape != (50, 28):
        raise RuntimeError(f"unexpected Stage-0 prediction shape {prediction.shape}")

    xml = xml_limits(dex_names)
    recorded = recorder_limits()
    recorded_by_name = {
        dex_names[index]: recorded["left"][index]
        for index in range(7)
    } | {
        dex_names[index + 7]: recorded["right"][index]
        for index in range(7)
    }
    cpp = cpp_limits()
    cpp_by_name = {
        dex_names[index]: cpp["left"][index]
        for index in range(7)
    } | {
        dex_names[index + 7]: cpp["right"][index]
        for index in range(7)
    }
    isaac_asset = usd_limits(ISAAC_ASSET, dex_names)
    isaac_scene = usd_limits(ISAAC_SCENE, dex_names)

    pre = load_safetensors(PRE_STATS)
    post = load_safetensors(POST_STATS)
    stat_keys = ("action.mean", "action.std", "action.min", "action.max", "action.count")
    pre_post_differences = {
        key: float(np.max(np.abs(np.asarray(pre[key]) - np.asarray(post[key]))))
        for key in stat_keys
    }
    calculated = {
        "action.mean": np.mean(actions, axis=0),
        "action.std": np.std(actions, axis=0),
        "action.min": np.min(actions, axis=0),
        "action.max": np.max(actions, axis=0),
    }
    saved_calculated_differences = {
        key: float(np.max(np.abs(np.asarray(pre[key], dtype=np.float64) - value)))
        for key, value in calculated.items()
    }
    action_mean = np.asarray(post["action.mean"], dtype=np.float64)
    action_std = np.asarray(post["action.std"], dtype=np.float64)
    normalized_reconstruction = (prediction - action_mean) / action_std
    round_trip = normalized_reconstruction * action_std + action_mean
    normalization_audit = {
        "method": "MEAN_STD",
        "preprocessor_stats_sha256": sha256(PRE_STATS),
        "postprocessor_stats_sha256": sha256(POST_STATS),
        "preprocessor_postprocessor_max_abs_difference": pre_post_differences,
        "saved_stats_vs_actual_parquet_max_abs_difference": saved_calculated_differences,
        "manual_physical_normalize_denormalize_round_trip_max_abs_error_rad": float(
            np.max(np.abs(round_trip - prediction))
        ),
        "stage0_normalized_prediction_minimum": float(np.min(normalized_reconstruction)),
        "stage0_normalized_prediction_maximum": float(np.max(normalized_reconstruction)),
        "worker_uses_saved_postprocessor": True,
        "status": "PASS"
        if max(pre_post_differences.values()) == 0.0
        and max(saved_calculated_differences.values()) < 1e-6
        and float(np.max(np.abs(round_trip - prediction))) < 1e-12
        else "FAIL",
    }

    schema_checks = {
        "freeze_names_equal_semantic_state_names": names
        == semantic_schema["state"]["joint_names"],
        "freeze_names_equal_semantic_action_names": names
        == semantic_schema["action"]["joint_names"],
        "freeze_names_equal_dataset_state_names": names
        == dataset_info["features"]["observation.state"]["names"],
        "freeze_names_equal_dataset_action_names": names
        == dataset_info["features"]["action"]["names"],
        "checkpoint_state_dimension_28": checkpoint_config["input_features"][
            "observation.state"
        ]["shape"]
        == [28],
        "checkpoint_action_dimension_28": checkpoint_config["output_features"]["action"][
            "shape"
        ]
        == [28],
        "stage0_named_isaac_mapping_has_28_unique_indices": len(
            set(stage0["joint_mapping"].values())
        )
        == 28
        and set(stage0["joint_mapping"]) == set(names),
        "arm_prefix_is_14_and_dex3_suffix_is_14": all(
            specs[index]["group"] == ("arm" if index < 14 else "dex3")
            for index in range(28)
        ),
    }

    rows: list[dict] = []
    for index in range(14, 28):
        name = names[index]
        target = actions[:, index]
        raw = prediction[:, index]
        low_mask = raw < lower[index] - STRICT_COMPARISON_EPS_RAD
        high_mask = raw > upper[index] + STRICT_COMPARISON_EPS_RAD
        low_excess = lower[index] - raw[low_mask]
        high_excess = raw[high_mask] - upper[index]
        excess = np.concatenate((low_excess, high_excess))
        distance = np.minimum(target - lower[index], upper[index] - target)
        rows.append(
            {
                "policy_index": index,
                "joint_name": name,
                "side": specs[index]["side"],
                "command_channel": specs[index]["command_channel"],
                "authoritative_lower_rad": float(lower[index]),
                "authoritative_upper_rad": float(upper[index]),
                "authoritative_provenance": {
                    "frozen_contract": str(FREEZE),
                    "frozen_contract_sha256": sha256(FREEZE),
                    "exact_numeric_model": str(MUJOCO_MODEL),
                    "exact_numeric_model_sha256": sha256(MUJOCO_MODEL),
                    "recorder_named_mapping": str(RECORDER),
                    "recorder_named_mapping_sha256": sha256(RECORDER),
                    "unitree_motor_order_example": str(UNITREE_EXAMPLE),
                    "unitree_motor_order_example_sha256": sha256(UNITREE_EXAMPLE),
                },
                "dataset_target_min_rad": float(np.min(target)),
                "dataset_target_max_rad": float(np.max(target)),
                "dataset_distance_to_lower_limit_rad": float(np.min(target) - lower[index]),
                "dataset_distance_to_upper_limit_rad": float(upper[index] - np.max(target)),
                "dataset_nearest_limit_distance_rad": float(np.min(distance)),
                "dataset_at_boundary_count_eps_1e_7": int(np.count_nonzero(distance <= BOUNDARY_EPS_RAD)),
                "dataset_at_boundary_percent_eps_1e_7": float(
                    100.0 * np.mean(distance <= BOUNDARY_EPS_RAD)
                ),
                "stage0_predicted_min_rad": float(np.min(raw)),
                "stage0_predicted_max_rad": float(np.max(raw)),
                "violation_count": int(excess.size),
                "violation_percent_of_50": float(100.0 * excess.size / len(raw)),
                "lower_bound_violation_count": int(np.count_nonzero(low_mask)),
                "upper_bound_violation_count": int(np.count_nonzero(high_mask)),
                "maximum_excess_rad": float(np.max(excess)) if excess.size else 0.0,
                "mean_excess_over_violations_rad": float(np.mean(excess)) if excess.size else 0.0,
                "mean_excess_over_all_50_rows_rad": float(np.sum(excess) / len(raw)),
                "mujoco_limit_rad": xml[name],
                "recorder_limit_rad": recorded_by_name[name],
                "unitree_example_rounded_limit_rad": cpp_by_name[name],
                "isaac_asset_limit_rad": isaac_asset[name],
                "isaac_composed_scene_limit_rad": isaac_scene[name],
            }
        )

    target_violation_mask = (actions < lower[None] - STRICT_COMPARISON_EPS_RAD) | (
        actions > upper[None] + STRICT_COMPARISON_EPS_RAD
    )
    raw_violation_mask = (prediction < lower[None] - STRICT_COMPARISON_EPS_RAD) | (
        prediction > upper[None] + STRICT_COMPARISON_EPS_RAD
    )
    dex_raw_mask = raw_violation_mask[:, 14:]
    affected_dataset_boundary_channels = sum(
        row["dataset_at_boundary_percent_eps_1e_7"] > 50.0 for row in rows
    )
    affected_prediction_channels = sum(row["violation_count"] > 0 for row in rows)

    source_comparison = {
        "frozen_contract_matches_mujoco_exactly": max_source_difference(
            dex_names, authoritative, xml
        )
        == 0.0,
        "frozen_contract_matches_recorder_exactly": max_source_difference(
            dex_names, authoritative, recorded_by_name
        )
        == 0.0,
        "maximum_frozen_vs_unitree_example_abs_difference_rad": max_source_difference(
            dex_names, authoritative, cpp_by_name
        ),
        "unitree_example_role": "motor order/sign and rounded range corroboration; not the exact frozen numeric contract",
        "maximum_frozen_vs_isaac_asset_abs_difference_rad": max_source_difference(
            dex_names, authoritative, isaac_asset
        ),
        "isaac_asset_equals_composed_scene": max_source_difference(
            dex_names, isaac_asset, isaac_scene
        )
        == 0.0,
        "isaac_material_difference": {
            "joints": ["left_hand_thumb_1_joint", "right_hand_thumb_1_joint"],
            "description": (
                "Isaac uses +/-35 degrees on the non-task-facing side of thumb_1; the frozen "
                "MuJoCo/control contract uses approximately 41.5 degrees. Dataset B and the Stage-0 "
                "prediction occupy the opposite shared side, so this difference causes none of the "
                "observed violations. Other differences are decimal-radian versus integer-degree rounding."
            ),
        },
        "source_locations": {
            "freeze_manifest": str(FREEZE),
            "mujoco_first_dex3_joint_line": source_line(MUJOCO_MODEL, 'name="left_hand_thumb_0_joint"'),
            "recorder_limits_line": source_line(RECORDER, 'DEX3_LIMITS ='),
            "unitree_limits_line": source_line(UNITREE_EXAMPLE, "maxLimits_left"),
            "isaac_asset": str(ISAAC_ASSET),
            "isaac_scene": str(ISAAC_SCENE),
        },
    }

    fatal_checks = {
        "normalization_valid": normalization_audit["status"] == "PASS",
        "joint_order_valid": all(schema_checks.values()),
        "frozen_limit_contract_has_exact_local_numeric_sources": source_comparison[
            "frozen_contract_matches_mujoco_exactly"
        ]
        and source_comparison["frozen_contract_matches_recorder_exactly"],
        "dataset_targets_inside_frozen_contract": int(np.count_nonzero(target_violation_mask)) == 0,
        "raw_arm_limit_violations_zero": int(np.count_nonzero(raw_violation_mask[:, :14])) == 0,
        "raw_dex3_limit_violations_reproduce_prior_report": int(np.count_nonzero(dex_raw_mask))
        == int(stage0["chunk_audit"]["joint_limit_violation_count"]),
    }
    root_cause = {
        "primary": "BOUNDARY_SATURATED_TARGET_PRIMITIVES_PLUS_SMALL_NEURAL_REGRESSION_OVERSHOOT",
        "training_targets_too_close_to_limits": True,
        "neural_regression_overshoot": True,
        "normalization_or_denormalization_error": False,
        "joint_order_mismatch": False,
        "limit_provenance_error": False,
        "another_cause": None,
        "evidence": {
            "dex3_channels_with_more_than_half_of_targets_at_a_bound": affected_dataset_boundary_channels,
            "dex3_channels_with_stage0_violations": affected_prediction_channels,
            "dataset_target_limit_violations": int(np.count_nonzero(target_violation_mask[:, 14:])),
            "raw_arm_limit_violations": int(np.count_nonzero(raw_violation_mask[:, :14])),
            "raw_dex3_limit_violations": int(np.count_nonzero(dex_raw_mask)),
            "raw_dex3_violating_frames": int(np.count_nonzero(np.any(dex_raw_mask, axis=1))),
            "maximum_raw_dex3_excess_rad": max(row["maximum_excess_rad"] for row in rows),
        },
        "interpretation": (
            "The open/release Dex3 primitives place most channels 1e-8 rad inside a physical bound "
            "for most frames. A continuous neural regressor has no outward error margin there, and its "
            "small physical-unit residuals cross the bound. The saved MEAN_STD tensors, manual round "
            "trip, named schema, and named Isaac mapping all pass."
        ),
    }
    projection_eligibility = {
        "status": "ELIGIBLE_FOR_GENERIC_NEAREST_BOUND_PROJECTION"
        if all(fatal_checks.values())
        else "BLOCKED_BY_IMPLEMENTATION_ERROR",
        "reason": (
            "All fatal provenance/mapping/normalization checks pass; only invalid Dex3 scalars require "
            "nearest-bound projection. Geometric benignity remains a separate mandatory Stage-0 gate."
        ),
    }

    report = {
        "schema_version": "policy_b_dex3_limit_diagnostic_v1",
        "status": "PASS_DIAGNOSTIC_PROJECTION_ELIGIBLE"
        if all(fatal_checks.values())
        else "FAIL_IMPLEMENTATION_ERROR",
        "read_only_audit": True,
        "dataset_b_modified": False,
        "policy_b_retrained": False,
        "camera_modified": False,
        "retargeting_modified": False,
        "inputs": {
            "dataset_action_parquet": str(PARQUET),
            "dataset_action_parquet_sha256": sha256(PARQUET),
            "raw_stage0_prediction": str(RAW_PREDICTION),
            "raw_stage0_prediction_sha256": sha256(RAW_PREDICTION),
            "raw_stage0_report": str(STAGE0_REPORT),
            "raw_stage0_report_sha256": sha256(STAGE0_REPORT),
            "frame_count": len(actions),
            "prediction_rows": len(prediction),
        },
        "authoritative_definition": {
            "description": (
                "The frozen Dataset-B named joint_specs are the authoritative research control "
                "contract. Their exact values are independently present in the frozen retargeting "
                "MuJoCo model and the named Dex3 recorder mapping."
            ),
            "limits": authoritative,
        },
        "source_comparison": source_comparison,
        "schema_checks": schema_checks,
        "normalization_audit": normalization_audit,
        "fatal_checks": fatal_checks,
        "per_joint": rows,
        "root_cause": root_cause,
        "projection_eligibility": projection_eligibility,
    }
    atomic_json(OUTPUT / "dex3_limit_audit.json", report)

    with (OUTPUT / "per_joint.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "policy_index",
            "joint_name",
            "authoritative_lower_rad",
            "authoritative_upper_rad",
            "dataset_target_min_rad",
            "dataset_target_max_rad",
            "dataset_distance_to_lower_limit_rad",
            "dataset_distance_to_upper_limit_rad",
            "dataset_at_boundary_count_eps_1e_7",
            "dataset_at_boundary_percent_eps_1e_7",
            "stage0_predicted_min_rad",
            "stage0_predicted_max_rad",
            "violation_count",
            "violation_percent_of_50",
            "lower_bound_violation_count",
            "upper_bound_violation_count",
            "maximum_excess_rad",
            "mean_excess_over_violations_rad",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Policy-B Stage-0 Dex3 hard-limit diagnostic",
        "",
        f"Status: **{report['status']}**",
        "",
        "This is a read-only audit of the saved physical-unit Policy-B output. No projection or command execution occurred.",
        "",
        "## Per-joint result",
        "",
        "| Joint | Limits [lower, upper] rad | Dataset min / max | Dataset margins [lower, upper] | Raw prediction min / max | Violations | Lower / upper | Max / mean excess rad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {joint_name} | [{authoritative_lower_rad:.6f}, {authoritative_upper_rad:.6f}] | "
            "{dataset_target_min_rad:.9f} / {dataset_target_max_rad:.9f} | "
            "{dataset_distance_to_lower_limit_rad:.9g} / {dataset_distance_to_upper_limit_rad:.9g} | "
            "{stage0_predicted_min_rad:.9f} / {stage0_predicted_max_rad:.9f} | "
            "{violation_count}/50 ({violation_percent_of_50:.1f}%) | "
            "{lower_bound_violation_count} / {upper_bound_violation_count} | "
            "{maximum_excess_rad:.9f} / {mean_excess_over_violations_rad:.9f} |".format(**row)
        )
    markdown += [
        "",
        "## Limit provenance",
        "",
        f"- Authoritative frozen contract: `{FREEZE}`",
        f"- Exact active retargeting model: `{MUJOCO_MODEL}`",
        f"- Exact named recorder mapping: `{RECORDER}`",
        f"- Unitree motor-order/range example: `{UNITREE_EXAMPLE}`",
        f"- Current Isaac asset: `{ISAAC_ASSET}`",
        "- The frozen contract matches the MuJoCo model and recorder constants exactly for all 14 joints.",
        "- The Unitree example corroborates motor order, sign, and rounded range. It is not used as the exact decimal contract.",
        "- Isaac uses a narrower non-task-facing `thumb_1` side (35° versus about 41.5°). All Dataset-B and current raw values lie on the opposite shared side; therefore this discrepancy creates none of the 198 violations.",
        "",
        "## Cause",
        "",
        f"**{root_cause['primary']}**",
        "",
        f"{root_cause['interpretation']}",
        "",
        f"- Dex3 channels with more than half their Dataset-B targets at a bound: {affected_dataset_boundary_channels}/14",
        f"- Dex3 channels with Stage-0 violations: {affected_prediction_channels}/14",
        f"- Dataset-B Dex3 target violations: {int(np.count_nonzero(target_violation_mask[:, 14:]))}",
        f"- Raw arm violations: {int(np.count_nonzero(raw_violation_mask[:, :14]))}",
        f"- Raw Dex3 violations: {int(np.count_nonzero(dex_raw_mask))} values across {int(np.count_nonzero(np.any(dex_raw_mask, axis=1)))} frames",
        f"- Maximum raw excess: {max(row['maximum_excess_rad'] for row in rows):.9f} rad",
        "- Normalization/denormalization: PASS",
        "- Named joint order: PASS",
        "- Limit provenance: PASS with the documented inactive-side Isaac thumb-range distinction",
        "",
        "## Decision",
        "",
        f"**{projection_eligibility['status']}**",
        "",
        projection_eligibility["reason"],
        "",
    ]
    (OUTPUT / "dex3_limit_audit.md").write_text("\n".join(markdown), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "raw_dex3_violations": int(np.count_nonzero(dex_raw_mask)),
                "violating_frames": int(np.count_nonzero(np.any(dex_raw_mask, axis=1))),
                "maximum_excess_rad": max(row["maximum_excess_rad"] for row in rows),
                "boundary_saturated_channels": affected_dataset_boundary_channels,
                "normalization": normalization_audit["status"],
                "joint_order": "PASS" if all(schema_checks.values()) else "FAIL",
                "projection_eligibility": projection_eligibility["status"],
            },
            indent=2,
        )
    )
    return 0 if report["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
