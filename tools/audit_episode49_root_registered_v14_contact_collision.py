#!/usr/bin/env python3
"""Decompose the v14 action-319 phone-clearance diagnostic.

The original contact metric used a legacy field named
``maximum_wrist_phone_penetration_m`` even though it was the maximum over the
right-C contact hull and the palm hull.  This audit leaves every trajectory,
target, root, and object pose untouched and records the two terms separately.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_root_registered_v14 as v14
import retarget_episode49_optimized_action_to_g1 as core
import solve_episode49_root_registered_v14_ik as solver_module

OUT = v14.OUT
TOLERANCE_M = 1.0e-5


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    exact_path = OUT / "position_only_exact_v14.npz"
    null_path = OUT / "position_only_nullspace_v14.npz"
    metrics_path = OUT / "physical_contact_reachability_v14.json"
    hashes_before = {str(path.resolve()): sha(path) for path in (exact_path, null_path)}

    metrics = json.loads(metrics_path.read_text())
    with np.load(v14.V12 / "position_only_exact_arm_trajectory.npz", allow_pickle=False) as values:
        old_q = values["g1_arm_q"].copy()
    # Model validation is verbose; the audit artifact, not console chatter, is
    # the evidence product.
    with contextlib.redirect_stdout(io.StringIO()):
        info = core.ik.validate_model(core.G1_XML)
        _, context, contact_solver = solver_module.build_contact_context(info, old_q)

    rows = {}
    for label, trajectory_path in (("EXACT", exact_path), ("NULLSPACE", null_path)):
        with np.load(trajectory_path, allow_pickle=False) as values:
            arm_q = values["g1_arm_q"][319].copy()
            root = values["g1_root"].copy()
        finger_q = np.asarray(
            metrics["candidates"][label]["action319"]["diagnostic_right_dex3_C_q_rad"],
            dtype=float,
        )
        contact_solver._assign(arm_q, {"right_C": finger_q})
        _, _, right_c_hull = contact_solver._contact_proxy("right_C", root)
        wrist, _, _, palm_hull = contact_solver._wrist_palm("right", root)
        right_c_penetration = v14.point_obb_penetration(
            right_c_hull, context.phone_action319, context.phone_dimensions
        )
        palm_penetration = v14.point_obb_penetration(
            palm_hull, context.phone_action319, context.phone_dimensions
        )
        wrist_reference_penetration = v14.point_obb_penetration(
            np.asarray(wrist, dtype=float)[None, :],
            context.phone_action319,
            context.phone_dimensions,
        )
        legacy = metrics["candidates"][label]["action319"][
            "maximum_wrist_phone_penetration_m"
        ]
        row = {
            "action_index": 319,
            "legacy_combined_field_name": "maximum_wrist_phone_penetration_m",
            "legacy_combined_value_m": legacy,
            "right_C_contact_hull_phone_penetration_m": right_c_penetration,
            "right_palm_hull_phone_penetration_m": palm_penetration,
            "right_wrist_reference_point_phone_penetration_m": wrist_reference_penetration,
            "wrist_and_palm_phone_nonpenetration_tolerance_m": TOLERANCE_M,
            "wrist_and_palm_phone_nonpenetration_pass": bool(
                palm_penetration <= TOLERANCE_M
                and wrist_reference_penetration <= TOLERANCE_M
            ),
            "right_C_hull_is_contact_proxy_not_a_990_frame_dex3_trajectory": True,
            "legacy_value_reproduced": bool(
                abs(legacy - max(right_c_penetration, palm_penetration)) <= 1.0e-12
            ),
        }
        rows[label] = row
        metrics["candidates"][label]["action319"][
            "collision_metric_decomposition"
        ] = row

    all_wrist_pass = all(row["wrist_and_palm_phone_nonpenetration_pass"] for row in rows.values())
    artifact = {
        "status": "ACTION319_WRIST_PHONE_NONPENETRATION_PASS" if all_wrist_pass else "ACTION319_WRIST_PHONE_NONPENETRATION_FAIL",
        "purpose": "decompose a misleading legacy combined field without changing targets, q, root, or scene",
        "candidates": rows,
        "all_candidates_wrist_and_palm_phone_nonpenetration_pass": all_wrist_pass,
        "dex3_trajectory_generated": False,
        "physics_run": False,
        "trajectory_sha256_before": hashes_before,
        "trajectory_sha256_after": {str(path.resolve()): sha(path) for path in (exact_path, null_path)},
    }
    artifact["trajectory_files_byte_identical"] = (
        artifact["trajectory_sha256_before"] == artifact["trajectory_sha256_after"]
    )
    v14.dump(OUT / "right_accessory_non_task_collision_audit_v14.json", artifact)
    v14.dump(metrics_path, metrics)
    print(json.dumps({"status": artifact["status"], "rows": rows}, indent=2))
    return 0 if all_wrist_pass and artifact["trajectory_files_byte_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
