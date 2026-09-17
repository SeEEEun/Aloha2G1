#!/usr/bin/env python3
"""Freeze the common named-joint nearest-bound deployment adapter config."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Usd, UsdPhysics


ROOT = Path("/home/jbnu/aloha_g1_dataset")
IMPLEMENTATION = ROOT / "tools/common_actuator_feasibility_projection.py"
FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
ISAAC_ASSET = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
OUTPUT = ROOT / "outputs/common_g1_deployment_safety/nearest_bound_joint_position_v1"


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
    temporary.replace(path)


def isaac_limits(names: list[str]) -> dict[str, list[float]]:
    stage = Usd.Stage.Open(str(ISAAC_ASSET))
    if stage is None:
        raise RuntimeError(f"could not open {ISAAC_ASSET}")
    wanted = set(names)
    values: dict[str, list[float]] = {}
    for prim in stage.Traverse():
        if prim.GetName() not in wanted:
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        if joint:
            values[prim.GetName()] = [
                math.radians(float(joint.GetLowerLimitAttr().Get())),
                math.radians(float(joint.GetUpperLimitAttr().Get())),
            ]
    missing = sorted(wanted - set(values))
    if missing:
        raise RuntimeError(f"Isaac asset is missing named joints: {missing}")
    return values


def inward_float32_interval(lower: float, upper: float) -> tuple[float, float]:
    """Return float32-representable bounds that remain inside [lower, upper]."""
    low32 = np.float32(lower)
    high32 = np.float32(upper)
    if float(low32) < lower:
        low32 = np.nextafter(low32, np.float32(np.inf), dtype=np.float32)
    if float(high32) > upper:
        high32 = np.nextafter(high32, np.float32(-np.inf), dtype=np.float32)
    if not float(low32) < float(high32):
        raise RuntimeError(f"no non-empty inward float32 interval for [{lower}, {upper}]")
    return float(low32), float(high32)


def main() -> int:
    source = json.loads(FREEZE.read_text(encoding="utf-8"))
    names = list(source["joint_names"])
    runtime = isaac_limits(names)
    rows = []
    for spec in source["joint_specs"]:
        name = spec["joint_name"]
        contract_lower = float(spec["minimum"])
        contract_upper = float(spec["maximum"])
        runtime_lower, runtime_upper = runtime[name]
        intersection_lower = max(contract_lower, runtime_lower)
        intersection_upper = min(contract_upper, runtime_upper)
        if intersection_lower >= intersection_upper:
            raise RuntimeError(f"empty contract/runtime intersection for {name}")
        effective_lower, effective_upper = inward_float32_interval(
            intersection_lower, intersection_upper
        )
        rows.append(
            {
                "policy_index": int(spec["index"]),
                "joint_name": name,
                "group": spec["group"],
                "side": spec["side"],
                "unit": "radian",
                "projectable": spec["group"] == "dex3",
                "contract_lower_rad": contract_lower,
                "contract_upper_rad": contract_upper,
                "runtime_isaac_lower_rad": runtime_lower,
                "runtime_isaac_upper_rad": runtime_upper,
                "mathematical_intersection_lower_rad": intersection_lower,
                "mathematical_intersection_upper_rad": intersection_upper,
                "effective_lower_rad": effective_lower,
                "effective_upper_rad": effective_upper,
                "effective_rule": (
                    "inward_float32_representable_closed_interval_of_intersection("
                    "contract, active_actuator_model)"
                ),
            }
        )
    manifest = {
        "schema_version": "common_actuator_feasibility_projection_v1",
        "status": "FROZEN_COMMON_DEPLOYMENT_SAFETY_ADAPTER",
        "name": "COMMON_G1_DEX3_NEAREST_BOUND_POSITION_PROJECTION",
        "scope": "policy-independent absolute G1/Dex3 joint-position deployment adapter",
        "applicable_policies": ["Policy A", "Policy B", "future policies with the identical named interface"],
        "algorithm": "componentwise_nearest_closed_interval_bound",
        "command_semantics": "absolute_joint_position_rad",
        "command_dtype": "float32",
        "comparison_tolerance_rad": 0.0,
        "projectable_group": "dex3",
        "non_projectable_group": "arm",
        "invariants": [
            "all 14 arm outputs are bitwise preserved",
            "all already-valid Dex3 outputs are bitwise preserved",
            "only out-of-range Dex3 scalars are changed",
            "every change is recorded with raw/projected value, magnitude, and active bound",
            "no policy, episode, frame, task, phase, or object-specific logic",
        ],
        "implementation": {"path": str(IMPLEMENTATION), "sha256": sha256(IMPLEMENTATION)},
        "contract": {"path": str(FREEZE), "sha256": sha256(FREEZE)},
        "active_actuator_model": {"path": str(ISAAC_ASSET), "sha256": sha256(ISAAC_ASSET)},
        "joint_names": names,
        "joints": rows,
        "real_hardware_authorized": False,
        "real_hardware_note": (
            "The algorithm is reusable unchanged, but real execution remains forbidden until the "
            "same named limits are verified against the connected controller/hardware revision."
        ),
    }
    atomic_json(OUTPUT / "freeze_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(OUTPUT / "freeze_manifest.json"),
                "implementation_sha256": manifest["implementation"]["sha256"],
                "projectable_joint_count": sum(row["projectable"] for row in rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
