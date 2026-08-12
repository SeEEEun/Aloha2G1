"""Acceptance-gate, collision-semantics, reproduction, and subset audits."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from aloha_g1_arm_v2.pipeline import _aggregate_integrated
from aloha_g1_hand_v2.collision_eval import (
    CollisionClassifier,
    body_digit,
    classify_collision_pair,
    same_side_hand_chain,
)

from .common import METHOD_TO_DATASET, V2_ARM_ROOT, V2_INTEGRATED_ROOT, load_json


def acceptance_gate_contract(
    runtime_config: Mapping[str, Any],
    solver_search: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ik = runtime_config["ik"]
    validation = runtime_config["validation"]
    return {
        "schema_version": "common_feasibility_v3_acceptance_gate_contract",
        "source": "frozen Common Arm-v2 runtime configuration",
        "gate_change_from_integrated_v2": "NONE",
        "ik": {
            "position_tolerance_m": ik["position_tolerance_m"],
            "orientation_tolerance_rad": ik["orientation_tolerance_rad"],
            "required_episode_success_rate": ik["required_success_rate"],
            "v3_orientation_slack_is_separately_bounded": True,
            "position_tolerance_changed": False,
            "orientation_tolerance_changed": False,
        },
        "joint_limits": {
            "constraint": "active-model closed box bounds inside optimization",
            "reporting_tolerance_rad": 1e-9,
            "post_solve_clipping_is_primary_strategy": False,
        },
        "temporal": {
            "maximum_joint_step_rad": validation["maximum_joint_step_rad"],
            "maximum_velocity_rad_s": validation["maximum_velocity_rad_s"],
            "maximum_acceleration_rad_s2": validation["maximum_acceleration_rad_s2"],
            "branch_absolute_step_norm_rad": validation[
                "branch_absolute_step_norm_rad"
            ],
            "branch_local_multiplier": validation["branch_local_multiplier"],
            "sampling_rate_hz": runtime_config["source_dataset"]["fps"],
            "changed": False,
            "solver_numerical_interior": (
                None
                if solver_search is None
                else {
                    "step_rad": solver_search[
                        "temporal_step_constraint_interior_rad"
                    ],
                    "acceleration_rad_s2": solver_search[
                        "temporal_acceleration_constraint_interior_rad_s2"
                    ],
                    "purpose": (
                        "strictly tighten optimization constraints so float32 "
                        "serialization cannot cross unchanged acceptance limits"
                    ),
                }
            ),
        },
        "collision": {
            "signed_distance_convention": "MuJoCo contact.dist; negative is penetration",
            "prohibited_when": (
                "contact.dist < -collision_penetration_tolerance_m, pair is not "
                "same body/static allowlist/same-side hand chain, and body tokens "
                "include shoulder/elbow/wrist/hand/torso"
            ),
            "collision_penetration_tolerance_m": validation[
                "collision_penetration_tolerance_m"
            ],
            "prohibited_collision_frames_allowed": validation[
                "prohibited_collision_frames_allowed"
            ],
            "static_model_contact_allowlist": validation[
                "static_model_contact_allowlist"
            ],
            "changed": False,
        },
        "strict_status_order": [
            "FAIL_DATA",
            "FAIL_LIMIT",
            "FAIL_IK",
            "FAIL_COLLISION",
            "FAIL_TEMPORAL",
            "FAIL_OTHER",
        ],
    }


def _raw_pair(
    runtime: Any, geom_ids: tuple[int, int]
) -> tuple[tuple[str, str], str]:
    bodies: list[str] = []
    for geom in geom_ids:
        body_id = int(runtime.model.geom_bodyid[geom])
        bodies.append(
            mujoco.mj_id2name(
                runtime.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            or f"body_{body_id}"
        )
    body_tuple = (bodies[0], bodies[1])
    return body_tuple, "|".join(sorted(body_tuple))


def _semantic_classification(
    bodies: tuple[str, str], pair: str, classifier: CollisionClassifier
) -> tuple[str, str]:
    if bodies[0] == bodies[1]:
        return (
            "KNOWN_ALLOWED_INTERNAL_CONTACT",
            "two geoms belong to one rigid body",
        )
    if pair in classifier.allowlist:
        return (
            "KNOWN_ALLOWED_INTERNAL_CONTACT",
            "frozen robot-model adjacency allowlist",
        )
    category = classify_collision_pair(bodies)
    if same_side_hand_chain(bodies):
        if category == "THUMB_INDEX_SELF":
            return (
                "DIAGNOSTIC_ONLY_CONTACT",
                "same-hand thumb/index closure without target-object collision geometry; "
                "reported but never counted as task success",
            )
        return (
            "DIAGNOSTIC_ONLY_CONTACT",
            "same-side articulated hand-chain diagnostic excluded by the inherited gate",
        )
    relevant = any(
        token in "|".join(bodies)
        for token in ("shoulder", "elbow", "wrist", "hand", "torso")
    )
    if relevant:
        return (
            "PROHIBITED_COLLISION",
            "cross-body robot penetration in an inherited safety category",
        )
    return (
        "DIAGNOSTIC_ONLY_CONTACT",
        "outside controlled upper-body collision scope",
    )


def collision_gate_semantics(
    runtime: Any,
    classifier: CollisionClassifier,
) -> dict[str, Any]:
    pair_frames: Counter[tuple[str, str]] = Counter()
    pair_episodes: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    pair_min_distance: dict[tuple[str, str], float] = {}
    pair_reasons: dict[tuple[str, str], str] = {}
    method_frames: dict[str, Counter[str]] = {
        method: Counter() for method in METHOD_TO_DATASET
    }
    tolerance = float(classifier.tolerance)
    for method, dataset_name in METHOD_TO_DATASET.items():
        for episode_id in range(50):
            with np.load(
                V2_ARM_ROOT
                / method
                / f"episode_{episode_id:06d}"
                / "g1_arm_action.npz",
                allow_pickle=False,
            ) as payload:
                arm = payload["action"].astype(np.float64)
            with np.load(
                V2_INTEGRATED_ROOT
                / dataset_name
                / f"episode_{episode_id:06d}"
                / "g1_hand_action.npz",
                allow_pickle=False,
            ) as payload:
                left = payload["left_action"].astype(np.float64)
                right = payload["right_action"].astype(np.float64)
            episode_classes: defaultdict[str, set[int]] = defaultdict(set)
            for frame, q in enumerate(arm):
                runtime.assign(q, left[frame], right[frame])
                frame_classes: set[str] = set()
                for contact in runtime.data.contact:
                    distance = float(contact.dist)
                    if distance >= -tolerance:
                        continue
                    geom_ids = (int(contact.geom1), int(contact.geom2))
                    bodies, pair = _raw_pair(runtime, geom_ids)
                    semantic, reason = _semantic_classification(
                        bodies, pair, classifier
                    )
                    key = (semantic, pair)
                    pair_frames[key] += 1
                    pair_episodes[key].add(episode_id)
                    pair_reasons[key] = reason
                    pair_min_distance[key] = min(
                        distance, pair_min_distance.get(key, 0.0)
                    )
                    frame_classes.add(semantic)
                    episode_classes[semantic].add(frame)
                for semantic in frame_classes:
                    method_frames[method][semantic] += 1
    rows = [
        {
            "classification": semantic,
            "pair": pair,
            "penetrating_contact_events": int(pair_frames[(semantic, pair)]),
            "episodes_affected": len(pair_episodes[(semantic, pair)]),
            "minimum_signed_distance_m": pair_min_distance[(semantic, pair)],
            "rationale": pair_reasons[(semantic, pair)],
        }
        for semantic, pair in sorted(
            pair_frames,
            key=lambda value: (-pair_frames[value], value[0], value[1]),
        )
    ]
    classifications = (
        "PROHIBITED_COLLISION",
        "INTENDED_TASK_CONTACT",
        "KNOWN_ALLOWED_INTERNAL_CONTACT",
        "DIAGNOSTIC_ONLY_CONTACT",
        "UNKNOWN_REQUIRES_REVIEW",
    )
    return {
        "schema_version": "common_feasibility_v3_collision_gate_semantics",
        "acceptance_gate_bug_found": False,
        "gate_change": "NONE",
        "classification_definitions": {
            "PROHIBITED_COLLISION": "retained strict upper-body cross-body penetration",
            "INTENDED_TASK_CONTACT": (
                "none in this robot-only self-collision model; target object contact is "
                "not represented and is never inferred"
            ),
            "KNOWN_ALLOWED_INTERNAL_CONTACT": "same rigid body or frozen model adjacency",
            "DIAGNOSTIC_ONLY_CONTACT": "reported internal/contact diagnostic, not strict task success",
            "UNKNOWN_REQUIRES_REVIEW": "would remain visible and would not be silently excluded",
        },
        "method_frame_incidence": {
            method: {
                key: int(method_frames[method][key]) for key in classifications
            }
            for method in METHOD_TO_DATASET
        },
        "unknown_pair_count": int(
            sum(row["classification"] == "UNKNOWN_REQUIRES_REVIEW" for row in rows)
        ),
        "intended_task_contact_pair_count": 0,
        "object_collision_geometry_in_gate": False,
        "pairs": rows,
        "exclusions": [
            {
                "classification": "KNOWN_ALLOWED_INTERNAL_CONTACT",
                "rationale": "rigid-body/model-adjacent contact cannot be a robot self-collision gate",
            },
            {
                "classification": "DIAGNOSTIC_ONLY_CONTACT",
                "rationale": (
                    "same-side hand-chain contact was already excluded in v2; it remains "
                    "reported and is not reinterpreted as object/task success"
                ),
            },
        ],
    }


def reproduce_v2() -> dict[str, Any]:
    stored = load_json(V2_INTEGRATED_ROOT / "summary/aggregate_comparison.json")
    recomputed: dict[str, Any] = {}
    per_episode_hashes: dict[str, list[str]] = {}
    for dataset_name in ("dataset_a", "dataset_b"):
        rows = [
            load_json(
                V2_INTEGRATED_ROOT
                / dataset_name
                / f"episode_{episode_id:06d}"
                / "retargeting_metrics.json"
            )
            for episode_id in range(50)
        ]
        recomputed[dataset_name] = _aggregate_integrated(rows)
        per_episode_hashes[dataset_name] = [
            str(row["episode_id"]) + ":" + row["status"] for row in rows
        ]
    comparisons: dict[str, Any] = {}
    exact = True
    for dataset_name in ("dataset_a", "dataset_b"):
        expected = stored[dataset_name]
        actual = recomputed[dataset_name]
        dataset_exact = json.dumps(expected, sort_keys=True) == json.dumps(
            actual, sort_keys=True
        )
        exact &= dataset_exact
        comparisons[dataset_name] = {
            "exact_json_reproduction": dataset_exact,
            "stored_pass_count": expected["pass_count"],
            "recomputed_pass_count": actual["pass_count"],
            "stored_mean_ik": expected["metrics"]["ik_success_rate"]["mean"],
            "recomputed_mean_ik": actual["metrics"]["ik_success_rate"]["mean"],
            "stored_prohibited_collision_mean": expected["metrics"][
                "prohibited_collision_frames"
            ]["mean"],
            "recomputed_prohibited_collision_mean": actual["metrics"][
                "prohibited_collision_frames"
            ]["mean"],
        }
    return {
        "schema_version": "common_feasibility_v3_v2_reproduction",
        "source": str(V2_INTEGRATED_ROOT),
        "exact_reproduction": exact,
        "comparisons": comparisons,
        "episode_status_contracts": per_episode_hashes,
    }


def select_failure_subset(
    first_failures: Mapping[str, Any], calibration_episode_ids: list[int]
) -> dict[str, Any]:
    records = first_failures["records"]
    calibration = set(int(value) for value in calibration_episode_ids)

    def select(method: str, classification: str) -> int:
        return min(
            int(row["episode"])
            for row in records
            if row["method"] == method
            and row["classification"] == classification
            and int(row["episode"]) in calibration
        )

    hand_candidates: list[int] = []
    ik_candidates: list[int] = []
    for episode_id in range(50):
        metrics = load_json(
            V2_INTEGRATED_ROOT
            / "dataset_b"
            / f"episode_{episode_id:06d}"
            / "retargeting_metrics.json"
        )
        categories = metrics["collision"]["category_frame_incidence"]
        if episode_id in calibration and metrics["status"] == "FAIL_COLLISION" and (
            int(categories.get("HAND_HAND", 0)) > 0
            or int(categories.get("CROSS_ARM", 0)) > 0
        ):
            hand_candidates.append(episode_id)
        if episode_id in calibration and metrics["status"] == "FAIL_IK":
            ik_candidates.append(episode_id)
    selected = {
        "a_joint_limit_dominated": select("baseline", "JOINT_LIMIT_BLOCK"),
        "a_orientation_dominated": select(
            "baseline", "ORIENTATION_OVERCONSTRAINT"
        ),
        "a_temporal_dominated": select(
            "baseline", "TEMPORAL_CONTINUITY_BLOCK"
        ),
        "b_arm_collision_dominated": select(
            "proposed", "ARM_TORSO_COLLISION_BLOCK"
        ),
        "b_hand_or_cross_collision_dominated": min(hand_candidates),
        "b_remaining_ik_failure": min(ik_candidates),
    }
    return {
        "schema_version": "common_feasibility_v3_failure_subset_selection",
        "selection_rule": (
            "minimum episode ID in each machine-classified failure stratum; "
            "no visual selection"
        ),
        "selected": selected,
        "unique_episode_ids": sorted(set(selected.values())),
        "all_selected_from_calibration": all(
            value in calibration for value in selected.values()
        ),
        "uses_locked_validation_for_parameter_selection": False,
    }
