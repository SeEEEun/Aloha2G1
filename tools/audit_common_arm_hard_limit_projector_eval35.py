#!/usr/bin/env python3
"""Offline audit of the one common arm hard-limit projector over EVAL35 A/B."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_execution_layer import WRIST_INDICES
from tools.direct_physical_execution_layer import (
    CommonArmHardLimitProjector,
    authoritative_joint_limits,
)


OUT = ROOT / "outputs/final_direct_physical_eval35"
COMMAND_MANIFEST = OUT / "00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
RESULT = OUT / "00_preparation/COMMON_ARM_HARD_LIMIT_PROJECTOR_AUDIT.json"
REPORT = OUT / "00_preparation/COMMON_ARM_HARD_LIMIT_PROJECTOR_AUDIT.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    commands = read_json(COMMAND_MANIFEST)
    records = list(commands.get("records", []))
    if len(records) != 70:
        raise RuntimeError("physical command manifest is not exact EVAL35 A/B x 35")
    counts = {
        method: sum(row.get("method") == method for row in records)
        for method in ("ACT-A40", "ACT-B40")
    }
    if counts != {"ACT-A40": 35, "ACT-B40": 35}:
        raise RuntimeError(f"physical command method counts invalid: {counts}")

    contract = read_json(JOINT_CONTRACT)
    lower, upper, names = authoritative_joint_limits(contract)
    projector = CommonArmHardLimitProjector(lower[:14], upper[:14])
    method_rows: dict[str, list[dict[str, Any]]] = {"ACT-A40": [], "ACT-B40": []}

    for record in records:
        path = Path(record["physical_command"])
        if sha256_file(path) != record["physical_command_sha256"]:
            raise RuntimeError(f"command archive hash drift: {path}")
        with np.load(path, allow_pickle=False) as archive:
            commanded = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
            policy_safe = np.asarray(archive["policy_safe_command"], dtype=np.float64)
            joint_names = tuple(archive["joint_names"].astype(str).tolist())
        if commanded.ndim != 2 or commanded.shape[1] != 28:
            raise RuntimeError(f"invalid command shape: {path}")
        if not np.array_equal(commanded, policy_safe):
            raise RuntimeError(f"commanded/policy-safe archive mismatch: {path}")
        if joint_names != names:
            raise RuntimeError(f"authoritative joint-order mismatch: {path}")

        arm = commanded[:, :14]
        projected, changed = projector.project(arm)
        expected_changed = (arm < lower[:14]) | (arm > upper[:14])
        if not np.array_equal(changed, expected_changed):
            raise RuntimeError(f"projector changed-mask mismatch: {path}")
        if not np.array_equal(projected[~changed], arm[~changed]):
            raise RuntimeError(f"projector altered a hard-limit-valid scalar: {path}")
        remaining = (projected < lower[:14]) | (projected > upper[:14])
        correction = np.abs(projected - arm)
        worst_flat = int(np.argmax(correction))
        worst_frame, worst_joint = np.unravel_index(worst_flat, correction.shape)
        wrist_local = np.asarray(WRIST_INDICES, dtype=np.int64)
        row = {
            "method": record["method"],
            "eval_index": int(record["eval_index"]),
            "stable_episode_id": record["stable_episode_id"],
            "frames": int(len(commanded)),
            "projected_scalar_count": int(np.count_nonzero(changed)),
            "projected_frame_count": int(np.count_nonzero(np.any(changed, axis=1))),
            "projected_wrist_scalar_count": int(
                np.count_nonzero(changed[:, wrist_local])
            ),
            "maximum_correction_rad": float(np.max(correction, initial=0.0)),
            "maximum_correction_frame": int(worst_frame),
            "maximum_correction_joint_index": int(worst_joint),
            "maximum_correction_joint_name": names[worst_joint],
            "remaining_arm_hard_limit_violation_scalar_count": int(
                np.count_nonzero(remaining)
            ),
            "valid_arm_scalars_changed": int(np.count_nonzero(changed != expected_changed)),
        }
        method_rows[record["method"]].append(row)

    method_summary = {}
    for method, rows in method_rows.items():
        maximum_row = max(rows, key=lambda row: row["maximum_correction_rad"])
        method_summary[method] = {
            "episodes": len(rows),
            "commanded_frames": int(sum(row["frames"] for row in rows)),
            "projected_scalar_count": int(
                sum(row["projected_scalar_count"] for row in rows)
            ),
            "projected_wrist_scalar_count": int(
                sum(row["projected_wrist_scalar_count"] for row in rows)
            ),
            "maximum_correction_rad": maximum_row["maximum_correction_rad"],
            "maximum_correction_episode_index": maximum_row["eval_index"],
            "maximum_correction_episode": maximum_row["stable_episode_id"],
            "maximum_correction_frame": maximum_row["maximum_correction_frame"],
            "maximum_correction_joint_index": maximum_row[
                "maximum_correction_joint_index"
            ],
            "maximum_correction_joint_name": maximum_row[
                "maximum_correction_joint_name"
            ],
            "remaining_arm_hard_limit_violation_scalar_count": int(
                sum(
                    row["remaining_arm_hard_limit_violation_scalar_count"]
                    for row in rows
                )
            ),
            "valid_arm_scalars_changed": int(
                sum(row["valid_arm_scalars_changed"] for row in rows)
            ),
        }

    passed = all(
        summary["episodes"] == 35
        and summary["remaining_arm_hard_limit_violation_scalar_count"] == 0
        and summary["valid_arm_scalars_changed"] == 0
        for summary in method_summary.values()
    )
    value = {
        "schema_version": "common_arm_hard_limit_projector_eval35_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "command_manifest": str(COMMAND_MANIFEST.resolve()),
        "command_manifest_sha256": sha256_file(COMMAND_MANIFEST),
        "authoritative_joint_limit_contract": str(JOINT_CONTRACT.resolve()),
        "authoritative_joint_limit_contract_sha256": sha256_file(JOINT_CONTRACT),
        "projector": {
            "implementation": str(
                (ROOT / "tools/direct_physical_execution_layer.py").resolve()
            ),
            "implementation_sha256": sha256_file(
                ROOT / "tools/direct_physical_execution_layer.py"
            ),
            "algorithm": "nearest_valid_value_componentwise_np_clip",
            "projector_instance_count": 1,
            "same_instance_and_limits_for_a_b": True,
            "ik_used": False,
            "smoothing_used": False,
            "trajectory_regeneration_used": False,
            "episode_specific_parameters": False,
            "method_specific_parameters": False,
            "arm_lower_rad": lower[:14].tolist(),
            "arm_upper_rad": upper[:14].tolist(),
            "joint_names": list(names[:14]),
        },
        "methods": method_summary,
        "remaining_arm_hard_limit_violation_scalar_count": int(
            sum(
                summary["remaining_arm_hard_limit_violation_scalar_count"]
                for summary in method_summary.values()
            )
        ),
        "records": [
            row
            for method in ("ACT-A40", "ACT-B40")
            for row in method_rows[method]
        ],
    }
    atomic_json(RESULT, value)
    REPORT.write_text(
        "# Common arm hard-limit projector: EVAL35 offline audit\n\n"
        f"Status: **{value['status']}**\n\n"
        "One method-blind componentwise nearest-bound projector was applied "
        "offline to the unchanged 70 prepared command archives.\n\n"
        f"- ACT-A projected scalars: **{method_summary['ACT-A40']['projected_scalar_count']}**\n"
        f"- ACT-A maximum correction: **{method_summary['ACT-A40']['maximum_correction_rad']:.12f} rad**\n"
        f"- ACT-B projected scalars: **{method_summary['ACT-B40']['projected_scalar_count']}**\n"
        f"- ACT-B maximum correction: **{method_summary['ACT-B40']['maximum_correction_rad']:.12f} rad**\n"
        f"- Remaining arm hard-limit violations: **{value['remaining_arm_hard_limit_violation_scalar_count']}**\n"
        "- Valid arm scalars changed: **0**\n"
        "- IK / smoothing / trajectory regeneration: **NO / NO / NO**\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": value["status"],
        "ACT-A40": method_summary["ACT-A40"],
        "ACT-B40": method_summary["ACT-B40"],
        "remaining_arm_hard_limit_violation_scalar_count": value[
            "remaining_arm_hard_limit_violation_scalar_count"
        ],
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
