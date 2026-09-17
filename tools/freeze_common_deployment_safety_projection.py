#!/usr/bin/env python3
"""Freeze the v2 common hard-bound plus Isaac-controller-margin adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
IMPLEMENTATION = ROOT / "tools/common_deployment_safety_projection.py"
HARD_FREEZE = (
    ROOT
    / "outputs/common_g1_deployment_safety/nearest_bound_joint_position_v1/freeze_manifest.json"
)
CHARACTERIZATION = (
    ROOT
    / "outputs/policy_b_isaac_validation/dex3_controller_characterization/characterization.json"
)
RAW_TRACE = CHARACTERIZATION.parent / "raw_tracking_trace.npz"
OUTPUT = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def inward_float32(value: float, direction: str) -> float:
    result = np.float32(value)
    if direction == "lower" and float(result) < value:
        result = np.nextafter(result, np.float32(np.inf), dtype=np.float32)
    if direction == "upper" and float(result) > value:
        result = np.nextafter(result, np.float32(-np.inf), dtype=np.float32)
    return float(result)


def main() -> int:
    hard = json.loads(HARD_FREEZE.read_text(encoding="utf-8"))
    characterization = json.loads(CHARACTERIZATION.read_text(encoding="utf-8"))
    if hard.get("status") != "FROZEN_COMMON_DEPLOYMENT_SAFETY_ADAPTER":
        raise RuntimeError("hard-limit freeze is not authoritative")
    if characterization.get("status") != "PASS_CHARACTERIZATION_COMPLETE":
        raise RuntimeError("Dex3 controller characterization is incomplete")
    if characterization.get("margin_label") != "SIMULATION_CONTROLLER_MARGIN_ONLY":
        raise RuntimeError("characterization lacks required simulation-only label")
    if characterization.get("policy_invoked") or characterization.get("dataset_accessed"):
        raise RuntimeError("margin characterization was not policy/task independent")
    for key in ("controller_contract_source", "characterization_implementation"):
        row = characterization[key]
        if sha256(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"characterization provenance changed: {key}")
    source_by_name = {row["joint"]: row for row in characterization["per_joint"]}
    rows = []
    for hard_row in hard["joints"]:
        name = hard_row["joint_name"]
        hard_lower = float(hard_row["effective_lower_rad"])
        hard_upper = float(hard_row["effective_upper_rad"])
        if hard_row["group"] == "dex3":
            source = source_by_name[name]
            margin_lower = float(source["margin_lower_rad"])
            margin_upper = float(source["margin_upper_rad"])
            safe_lower = inward_float32(hard_lower + margin_lower, "lower")
            safe_upper = inward_float32(hard_upper - margin_upper, "upper")
        else:
            margin_lower = 0.0
            margin_upper = 0.0
            safe_lower = hard_lower
            safe_upper = hard_upper
        if not hard_lower <= safe_lower < safe_upper <= hard_upper:
            raise RuntimeError(f"invalid safe interval for {name}")
        rows.append(
            {
                "policy_index": int(hard_row["policy_index"]),
                "joint_name": name,
                "group": hard_row["group"],
                "side": hard_row["side"],
                "unit": "radian",
                "projectable": hard_row["group"] == "dex3",
                "hard_lower_rad": hard_lower,
                "hard_upper_rad": hard_upper,
                "margin_lower_rad": margin_lower,
                "margin_upper_rad": margin_upper,
                "safe_lower_rad": safe_lower,
                "safe_upper_rad": safe_upper,
            }
        )
    manifest = {
        "schema_version": "common_g1_dex3_deployment_safety_projection_v2",
        "status": "FROZEN_COMMON_DEPLOYMENT_SAFETY_ADAPTER",
        "name": "COMMON_G1_DEX3_HARD_AND_SIMULATION_CONTROLLER_SAFE_POSITION_PROJECTION",
        "scope": "policy-independent absolute G1/Dex3 joint-position deployment adapter",
        "applicable_policies": [
            "Policy A",
            "Policy B",
            "future policies with the identical named interface",
        ],
        "algorithm": "hard_then_simulation_safe_nearest_closed_interval",
        "command_semantics": "absolute_joint_position_rad",
        "command_dtype": "float32",
        "comparison_tolerance_rad": 0.0,
        "margin_label": "SIMULATION_CONTROLLER_MARGIN_ONLY",
        "margin_derivation_rule": characterization["margin_derivation"],
        "projectable_group": "dex3",
        "non_projectable_group": "arm",
        "invariants": [
            "policy_raw_action is retained exactly",
            "hard_limit_projected_action is retained separately",
            "deployment_safe_action is retained separately",
            "all 14 arm outputs are bitwise preserved in both stages",
            "Dex3 values already inside the safe interval are bitwise preserved",
            "only Dex3 values outside a named interval are changed",
            "every hard and margin change is logged separately",
            "no policy, episode, frame, task, phase, object, or Doll-Handoff-specific logic",
        ],
        "implementation": {"path": str(IMPLEMENTATION), "sha256": sha256(IMPLEMENTATION)},
        "hard_limit_freeze": {"path": str(HARD_FREEZE), "sha256": sha256(HARD_FREEZE)},
        "characterization": {
            "path": str(CHARACTERIZATION),
            "sha256": sha256(CHARACTERIZATION),
            "raw_trace_path": str(RAW_TRACE),
            "raw_trace_sha256": sha256(RAW_TRACE),
            "test_count": characterization["test_count"],
            "sample_count": characterization["sample_count"],
            "controller_contract": characterization["controller_contract_source"],
        },
        "joint_names": list(hard["joint_names"]),
        "joints": rows,
        "real_hardware_authorized": False,
        "real_hardware_requirement": (
            "A separate no-object real-G1/Dex3 boundary and tracking calibration is mandatory "
            "before command transmission; Isaac-derived margins are not physical-hardware margins."
        ),
    }
    atomic_json(OUTPUT / "freeze_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "margin_label": manifest["margin_label"],
                "output": str(OUTPUT / "freeze_manifest.json"),
                "implementation_sha256": manifest["implementation"]["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
