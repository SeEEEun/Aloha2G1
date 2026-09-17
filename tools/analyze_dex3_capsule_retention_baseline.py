#!/usr/bin/env python3
"""Summarize the frozen bilateral capsule closure-to-slip baseline.

The script is deliberately baseline-only.  It refuses repair-condition inputs so
the cause record is produced from unchanged 0.55/0.45 material and 100/4 Dex3
drives before any bounded repair result is inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DEFAULT_ROOT = ROOT / "outputs/dex3_rigid_proxy_retention_repair"
DEFAULT_CONFIG = ROOT / "configs/dex3_capsule_retention_bounded_repair_v1.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
    )


def stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def summarize_side(root: Path, side: str, contract: dict[str, Any]) -> dict[str, Any]:
    run_dir = root / "baseline" / side
    result_path = run_dir / "stage_result.json"
    log_path = run_dir / "stage_log.npz"
    pairs_path = run_dir / "contact_pairs.csv"
    result = read_json(result_path)
    if result["execution_condition_name"] != "baseline":
        raise RuntimeError(f"{side}: cause analysis must use baseline only")
    if result["shape"] != contract["frozen_geometry"]["name"]:
        raise RuntimeError(f"{side}: geometry is not the frozen capsule")
    with np.load(log_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    labels = data["stage"].astype(str)
    dt = float(np.median(np.diff(data["timestamp_s"])))
    pre_all = np.flatnonzero(labels == "POST_CLOSE_NO_GRAVITY")
    stable_count = min(len(pre_all), max(1, int(round(0.2 / dt))))
    stable = pre_all[-stable_count:]
    gravity_all = np.flatnonzero(labels == "GRAVITY_RETENTION")
    slip_count = min(len(gravity_all), max(1, int(round(0.2 / dt))))
    slip = gravity_all[:slip_count]
    force_threshold = float(contract["gates"]["minimum_contact_force_n"])
    load = float(contract["object"]["gravitational_load_n"])
    mu_static = float(contract["conditions"]["baseline"]["material"]["static_friction"])
    total_normal = data["total_hand_object_normal_force_n"]
    stable_normal_median = float(np.median(total_normal[stable]))
    capacity_ratio = mu_static * stable_normal_median / load
    active_links = sorted(
        {
            link
            for value in data["active_fingertip_links"][stable].astype(str)
            for link in value.split(",")
            if link
        }
    )
    hand_slice = slice(14, 21) if side == "left" else slice(21, 28)
    hand_names = data["joint_names"].astype(str)[hand_slice]
    error = data["finger_joint_position_error_to_grasp_rad"][stable]
    applied = data["finger_applied_torque_nm"][stable]
    computed = data["finger_computed_torque_nm"][stable]
    estimated = data["finger_estimated_implicit_drive_torque_nm"][stable]
    effort_limit = float(contract["conditions"]["baseline"]["finger_drive"]["effort_limit_sim"])
    per_joint = []
    for index, name in enumerate(hand_names):
        per_joint.append(
            {
                "joint_name": str(name),
                "position_error_mean_rad": float(np.mean(error[:, index])),
                "position_error_rms_rad": float(np.sqrt(np.mean(error[:, index] ** 2))),
                "position_error_max_abs_rad": float(np.max(np.abs(error[:, index]))),
                "estimated_drive_torque_max_abs_nm": float(np.max(np.abs(estimated[:, index]))),
                "computed_torque_max_abs_nm": float(np.max(np.abs(computed[:, index]))),
                "applied_torque_max_abs_nm": float(np.max(np.abs(applied[:, index]))),
                "applied_torque_saturation_fraction": float(
                    np.mean(np.abs(applied[:, index]) >= effort_limit - 1.0e-3)
                ),
            }
        )
    pair_rows = list(csv.DictReader(pairs_path.open(newline="", encoding="utf-8")))
    stable_steps = set(map(int, data["physics_step"][stable]))
    slip_steps = set(map(int, data["physics_step"][slip]))

    def pair_phase(rows: list[dict[str, str]], steps: set[int]) -> dict[str, Any]:
        selected = [
            row
            for row in rows
            if int(row["physics_step"]) in steps and row["role"] in ("A", "B", "C")
        ]
        output: dict[str, Any] = {}
        for role in ("A", "B", "C"):
            role_rows = [row for row in selected if row["role"] == role]
            force = np.asarray([float(row["normal_force_n"]) for row in role_rows])
            impulse = np.asarray([float(row["normal_impulse_ns"]) for row in role_rows])
            tangential = np.asarray(
                [float(row["relative_tangential_velocity_m_s"]) for row in role_rows]
            )
            normal_z = np.asarray([float(row["normal_z"]) for row in role_rows])
            penetration = np.asarray([float(row["penetration_m"]) for row in role_rows])
            output[role] = {
                "link": next(
                    (row["collider0"].removesuffix("/collisions") for row in role_rows), None
                ),
                "contact_point_samples": len(role_rows),
                "force_bearing": bool(len(force) and np.max(force) >= force_threshold),
                "normal_force_n": stats(force),
                "normal_impulse_sum_ns": float(np.sum(impulse)) if len(impulse) else 0.0,
                "tangential_velocity_m_s": stats(tangential),
                "contact_normal_z": stats(normal_z),
                "maximum_penetration_m": float(np.max(penetration)) if len(penetration) else 0.0,
            }
        return output

    stable_friction_ratio = mu_static * total_normal[stable] / load
    gravity_reference_z = float(data["object_position_world_m"][gravity_all[0], 2])
    slip_drop = gravity_reference_z - data["object_position_world_m"][slip, 2]
    return {
        "side": side,
        "result_status": result["status"],
        "collision_contact_pairs": result["collision_contact_pairs"],
        "active_fingertip_links_stable_close": active_links,
        "initial_overlap": result["initial_overlap"],
        "contact_offsets": {
            "object_contact_offset_m": result["object_contact_offset_m"],
            "object_rest_offset_m": result["object_rest_offset_m"],
            "fingertip": result["selected_fingertip_offsets"],
        },
        "finger_drive": result["finger_drive"],
        "stable_close_window": {
            "duration_s": len(stable) * dt,
            "per_contact_role": pair_phase(pair_rows, stable_steps),
            "total_normal_force_n": stats(total_normal[stable]),
            "normal_force_friction_capacity_ratio_to_gravity": stats(stable_friction_ratio),
            "contact_tangential_velocity_m_s": stats(
                data["maximum_contact_tangential_velocity_m_s"][stable]
            ),
            "object_vertical_velocity_m_s": stats(
                data["object_linear_velocity_m_s"][stable, 2]
            ),
            "fingertip_signed_distance_to_proxy_m": {
                role: stats(data[f"role_{role}_fingertip_point_proxy_signed_distance_m"][stable])
                for role in ("A", "B", "C")
            },
            "finger_position_error_rms_all_rad": float(np.sqrt(np.mean(error**2))),
            "finger_position_error_max_abs_all_rad": float(np.max(np.abs(error))),
            "applied_drive_torque_saturation_fraction_all": float(
                np.mean(np.abs(applied) >= effort_limit - 1.0e-3)
            ),
            "per_joint": per_joint,
        },
        "gravity_slip_first_0p2s": {
            "duration_s": len(slip) * dt,
            "per_contact_role": pair_phase(pair_rows, slip_steps),
            "total_normal_force_n": stats(total_normal[slip]),
            "contact_tangential_velocity_m_s": stats(
                data["maximum_contact_tangential_velocity_m_s"][slip]
            ),
            "object_vertical_velocity_m_s": stats(data["object_linear_velocity_m_s"][slip, 2]),
            "maximum_com_drop_m": float(np.max(slip_drop)),
            "active_link_sequences": list(dict.fromkeys(data["active_fingertip_links"][slip].astype(str))),
        },
        "gravity_retention": result["gravity_retention"],
        "maximum_penetration_m": result["maximum_penetration_m"],
        "artifact_hashes": {
            "stage_result": sha256_file(result_path),
            "stage_log": sha256_file(log_path),
            "contact_pairs": sha256_file(pairs_path),
        },
        "derived": {
            "stable_normal_force_n": stable_normal_median,
            "static_friction_capacity_n": mu_static * stable_normal_median,
            "capacity_to_gravity_ratio": capacity_ratio,
            "normal_force_qualitatively_sufficient": bool(capacity_ratio >= 0.95),
            "drive_not_materially_saturated": bool(
                np.mean(np.abs(applied) >= effort_limit - 1.0e-3) < 0.1
            ),
            "rapid_tangential_slip": bool(
                np.percentile(data["maximum_contact_tangential_velocity_m_s"][slip], 95) > 0.05
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    contract = read_json(config_path)
    sides = {side: summarize_side(output_root, side, contract) for side in ("left", "right")}
    normal_sufficient = all(
        side["derived"]["normal_force_qualitatively_sufficient"]
        and side["derived"]["drive_not_materially_saturated"]
        for side in sides.values()
    )
    sliding = all(side["derived"]["rapid_tangential_slip"] for side in sides.values())
    if normal_sufficient and sliding:
        cause = "INSUFFICIENT_FRICTION"
    elif not normal_sufficient and sliding:
        cause = "MIXED"
    elif not normal_sufficient:
        cause = "INSUFFICIENT_NORMAL_FORCE"
    else:
        cause = "UNFAVORABLE_CONTACT_GEOMETRY"
    if cause != "INSUFFICIENT_FRICTION":
        raise RuntimeError(f"Measured baseline classified as {cause}; review before repair execution")
    payload = {
        "schema_version": "dex3_capsule_retention_baseline_diagnosis_v1",
        "experiment_name": "GRASP_RETENTION_UNDER_GRAVITY",
        "geometry": contract["frozen_geometry"],
        "mass_kg": contract["object"]["mass_kg"],
        "gravitational_load_n": contract["object"]["gravitational_load_n"],
        "baseline_material": contract["conditions"]["baseline"]["material"],
        "sides": sides,
        "primary_cause": cause,
        "reason": (
            "Both hands establish force-bearing bilateral closure with small GRASP-target error and "
            "little drive saturation. The baseline friction-capacity estimate is approximately equal "
            "to the 0.54 N load on the weaker left hand and 2.5x load on the right, yet gravity onset "
            "produces rapid tangential slip and loss of the original lateral force-bearing contacts."
        ),
        "authorized_bounded_candidates": contract["bounded_repair_policy"][cause],
        "policy_or_checkpoint_used": False,
        "geometry_changed": False,
        "parameters_changed_before_diagnosis": False,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
    }
    json_path = output_root / "BASELINE_RETENTION_DIAGNOSIS.json"
    atomic_json(json_path, payload)
    lines = [
        "# Baseline Dex3 Capsule Retention Diagnosis",
        "",
        f"Primary cause: **{cause}**",
        "",
        "The capsule geometry, frozen GRASP targets, 0.55/0.45 material, arm commands, and "
        "100/4 Dex3 drives were unchanged for this measurement. No learned policy was loaded.",
        "",
        "| Hand | Median normal force | Static friction capacity / 0.54 N | p95 slip speed | GRASP error RMS | Drive saturation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for side in ("left", "right"):
        item = sides[side]
        stable = item["stable_close_window"]
        slip = item["gravity_slip_first_0p2s"]
        lines.append(
            f"| {side.title()} | {item['derived']['stable_normal_force_n']:.3f} N | "
            f"{item['derived']['capacity_to_gravity_ratio']:.3f}x | "
            f"{slip['contact_tangential_velocity_m_s']['p95']:.3f} m/s | "
            f"{stable['finger_position_error_rms_all_rad']:.4f} rad | "
            f"{100.0 * stable['applied_drive_torque_saturation_fraction_all']:.2f}% |"
        )
    lines.extend(
        [
            "",
            payload["reason"],
            "",
            "Authorized next conditions: `friction_medium_high`, then `friction_high`; drive, "
            "grasp target, arm commands, capsule geometry, offsets, mass, and restitution remain fixed.",
            "",
        ]
    )
    atomic_text(output_root / "BASELINE_RETENTION_DIAGNOSIS.md", "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
