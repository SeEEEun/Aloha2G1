#!/usr/bin/env python3
"""Freeze a symmetric, TRAIN40-only initial pose for the paper A/B rollout.

The prior ACT-B diagnostic pose was constructed from Dataset B alone.  It is
preserved as development evidence but is not a fair initial condition for the
paired paper experiment.  This script constructs one global A/B pose as the
jointwise midpoint of the Fair-A40 and Proposed-B40 frame-0 medians, then
applies the already-frozen policy-independent deployment projection.  The rule
is fixed before any paper-model prediction or source-conditioned rollout.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from tools.common_deployment_safety_projection import (
        NamedJointDeploymentSafetyProjector,
    )
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    from common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
    from doll_handoff_retargeting.common import load_common_config, load_scene
    from doll_handoff_retargeting.models import G1Kinematics


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
OUTPUT = PAPER / "COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json"
TRAIN_MANIFEST = PAPER / "train40_manifest.json"
DATASETS = {
    "fair_a": ROOT / "datasets/doll_handoff_fair_a_train40",
    "proposed_b": ROOT / "datasets/doll_handoff_proposed_b_train40",
}
PROJECTOR_MANIFEST = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
EXPECTED_PROJECTOR_SHA256 = "05078d0038ab6defaa8a0f56b1f38b752cc92892856996fecc05b75fcce27cf2"
PRIOR_B_ONLY_CONTRACT = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout/COMMON_G1_POLICY_INITIAL_STATE_V1.json"
EXPECTED_PRIOR_SHA256 = "e1b64c4b009d8b10c47995a5866ad9f3f0d427c967975fb45b6c702ea443e0b1"


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


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen initial state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def frame0_values(root: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    info = read_json(root / "meta/info.json")
    names = list(info["features"]["action"]["names"])
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no dataset parquet files under {root}")
    table = pq.read_table(
        files,
        columns=["action", "observation.state", "episode_index", "frame_index"],
    )
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    episode = np.asarray(table["episode_index"], dtype=np.int64)
    frame = np.asarray(table["frame_index"], dtype=np.int64)
    select = frame == 0
    if action.shape[1:] != (28,) or state.shape != action.shape:
        raise RuntimeError(f"invalid 28D dataset arrays: {root}")
    if int(np.count_nonzero(select)) != 40:
        raise RuntimeError(f"expected 40 frame-0 rows: {root}")
    if not np.array_equal(np.sort(episode[select]), np.arange(40)):
        raise RuntimeError(f"frame-0 episode identities are incomplete: {root}")
    if not np.array_equal(action[select], state[select]):
        raise RuntimeError(f"frame-0 state/action semantics changed: {root}")
    if not np.isfinite(action[select]).all():
        raise RuntimeError(f"non-finite frame-0 action: {root}")
    return names, action[select], episode[select]


def distances(reference: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "arm_l2_rad": float(np.linalg.norm(delta[:14])),
        "dex3_l2_rad": float(np.linalg.norm(delta[14:])),
        "full_l2_rad": float(np.linalg.norm(delta)),
        "maximum_abs_joint_rad": float(np.max(np.abs(delta))),
        "per_joint_signed_delta_rad": delta,
    }


def main() -> None:
    if sha256_file(PROJECTOR_MANIFEST) != EXPECTED_PROJECTOR_SHA256:
        raise RuntimeError("frozen common deployment projection changed")
    if sha256_file(PRIOR_B_ONLY_CONTRACT) != EXPECTED_PRIOR_SHA256:
        raise RuntimeError("prior ACT-B development initial-state artifact changed")
    manifest = read_json(TRAIN_MANIFEST)
    if manifest["status"] != "PASS" or manifest["episode_count"] != 40:
        raise RuntimeError("TRAIN40 manifest is not frozen and valid")
    names_a, values_a, episodes_a = frame0_values(DATASETS["fair_a"])
    names_b, values_b, episodes_b = frame0_values(DATASETS["proposed_b"])
    if names_a != names_b or len(names_a) != 28:
        raise RuntimeError("A/B named 28D interfaces differ")
    if not np.array_equal(episodes_a, episodes_b):
        raise RuntimeError("A/B TRAIN40 frame-0 episode rows differ")

    median_a = np.median(values_a, axis=0)
    median_b = np.median(values_b, axis=0)
    symmetric_midpoint = 0.5 * (median_a + median_b)
    midpoint_distance_a = distances(symmetric_midpoint, median_a)
    midpoint_distance_b = distances(symmetric_midpoint, median_b)
    if not np.isclose(
        midpoint_distance_a["full_l2_rad"],
        midpoint_distance_b["full_l2_rad"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("unprojected midpoint is not exactly symmetric")

    projector = NamedJointDeploymentSafetyProjector.from_path(PROJECTOR_MANIFEST)
    if projector.names != names_a:
        raise RuntimeError("projector named order differs from paired datasets")
    projection = projector.project(symmetric_midpoint.astype(np.float64)[None])
    q = projection.deployment_safe_action[0].astype(np.float64)
    hard_violation = (q < projector.hard_lower - 1e-12) | (
        q > projector.hard_upper + 1e-12
    )
    safe_violation = (q < projector.safe_lower - 1e-12) | (
        q > projector.safe_upper + 1e-12
    )
    if np.any(hard_violation) or np.any(safe_violation):
        raise RuntimeError("projected midpoint is outside a frozen interval")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names_a)}
    arm = np.asarray([lookup[str(name)] for name in g1.arm_joint_names], dtype=np.int64)
    left = np.asarray([lookup[name] for name in g1.hand_joint_names["left"]], dtype=np.int64)
    right = np.asarray([lookup[name] for name in g1.hand_joint_names["right"]], dtype=np.int64)
    if len(set(np.concatenate((arm, left, right)).tolist())) != 28:
        raise RuntimeError("named G1 mapping is not bijective")
    geometry = g1.trajectory_geometry(
        q[None, arm],
        q[None, left],
        q[None, right],
        float(
            read_json(ROOT / "configs/doll_handoff_g1_feasibility_resolver.json")[
                "unchanged_acceptance"
            ]["collision_penetration_tolerance_m"]
        ),
    )
    collision_counts = {
        key: int(np.count_nonzero(value))
        for key, value in geometry["collision_flags"].items()
    }
    hard_collision_count = sum(
        collision_counts.get(key, 0)
        for key in ("ARM_TORSO", "CROSS_ARM", "WRIST_OR_PALM_TORSO", "OTHER")
    )
    if hard_collision_count:
        raise RuntimeError(f"symmetric midpoint has hard collision: {collision_counts}")

    prior = read_json(PRIOR_B_ONLY_CONTRACT)
    prior_q = np.asarray(prior["full_28d_initial_q_rad"], dtype=np.float64)
    result = {
        "schema_version": "common_g1_policy_initial_state_ab_v1",
        "status": "COMMON_G1_POLICY_INITIAL_STATE_AB_V1",
        "frozen_before_paper_model_prediction": True,
        "scope": ["ACT-A40", "ACT-B40"],
        "policy_specific": False,
        "episode_specific": False,
        "derivation": "jointwise midpoint of equally weighted Fair-A40 and Proposed-B40 frame-0 medians; TRAIN40 only; then exact frozen common deployment projection",
        "selection_used_policy_predictions_or_performance": False,
        "heldout_state_used": False,
        "joint_order": names_a,
        "command_semantics": "absolute_joint_position_rad",
        "train40_frame0_count_per_method": 40,
        "train40_manifest": str(TRAIN_MANIFEST),
        "train40_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "fair_a40_frame0_median_q_rad": median_a,
        "proposed_b40_frame0_median_q_rad": median_b,
        "unprojected_symmetric_midpoint_q_rad": symmetric_midpoint,
        "full_28d_initial_q_rad": q,
        "g1_14_arm_initial_q_rad": q[:14],
        "dex3_14_hand_initial_q_rad": q[14:],
        "symmetry_before_projection": {
            "fair_a_median_distance": midpoint_distance_a,
            "proposed_b_median_distance": midpoint_distance_b,
            "equal_full_l2_to_1e_12": True,
        },
        "distance_after_projection": {
            "fair_a_median": distances(q, median_a),
            "proposed_b_median": distances(q, median_b),
        },
        "projection": {
            "manifest": str(PROJECTOR_MANIFEST),
            "manifest_sha256": EXPECTED_PROJECTOR_SHA256,
            "summary": projection.summary,
            "records": projection.records,
            "act_specific_clamp": False,
            "arm_outputs_bitwise_preserved": projection.summary[
                "arm_outputs_bitwise_preserved"
            ],
        },
        "hard_limit_clearance_rad": {
            name: {
                "lower": float(q[index] - projector.hard_lower[index]),
                "upper": float(projector.hard_upper[index] - q[index]),
            }
            for index, name in enumerate(names_a)
        },
        "kinematic_collision_audit": {
            "status": "PASS",
            "category_frame_counts": collision_counts,
            "hard_collision_incidence": hard_collision_count,
            "distal_hand_hand_reported_separately": collision_counts.get(
                "DISTAL_HAND_HAND", 0
            ),
        },
        "settle_duration_seconds": 1.0,
        "tolerances": prior["tolerances"],
        "reset_procedure": [
            step.replace(
                "COMMON_G1_POLICY_INITIAL_STATE_V1",
                "COMMON_G1_POLICY_INITIAL_STATE_AB_V1",
            )
            for step in prior["reset_procedure"]
        ],
        "prior_b_only_development_contract_preserved": {
            "path": str(PRIOR_B_ONLY_CONTRACT),
            "sha256": EXPECTED_PRIOR_SHA256,
            "not_used_for_paper_a_b_rollout": True,
            "reason": "B-derived pose is not symmetric with respect to A40/B40 frame-0 supervision",
            "prior_to_fair_a_median_distance": distances(prior_q, median_a),
            "prior_to_proposed_b_median_distance": distances(prior_q, median_b),
        },
        "real_hardware": "NOT_AUTHORIZED",
    }
    atomic_json(OUTPUT, result)
    print(
        json.dumps(
            {
                "path": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "projection": projection.summary,
                "distance_after_projection": result["distance_after_projection"],
                "collision": result["kinematic_collision_audit"],
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
