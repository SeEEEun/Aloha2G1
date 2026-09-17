"""50-episode hand-only replay metrics on immutable arm trajectories."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

from aloha_g1_dataset_v1.core import physical_pinch_frame
from aloha_g1_hand_v2.collision_eval import (
    COLLISION_CATEGORIES,
    CollisionClassifier,
    body_digit,
)

from .common import V1_ROOT, V2_ROOT
from .primitives import PHASES, angle_rad


COLLISION_METRIC_NAMES = (
    "historical_gate_collision_frames",
    "historical_gate_hand_related_frames",
    "comprehensive_hand_related_frames",
    "third_finger_related_frames",
    "thumb_index_related_frames",
    "same_hand_self_contact_frames",
    "hand_hand_frames",
    "cross_arm_finger_involved_frames",
)


def install_global_neutrals(
    episodes: list[dict[str, Any]],
    third_indices: Mapping[str, np.ndarray],
    selected: Mapping[str, np.ndarray],
) -> None:
    """Replace only the two non-task columns with one side-global constant."""
    for episode in episodes:
        for side in ("left", "right"):
            episode[f"candidate_{side}"][:, third_indices[side]] = np.asarray(
                selected[side], dtype=np.float64
            )


def _variant_episode(
    variant: str, episode_id: int, proposed_episode: Mapping[str, Any]
) -> dict[str, Any]:
    if variant == "current_proposed_v1":
        return {
            "arm": proposed_episode["arm"],
            "arm_payload": proposed_episode["arm_payload"],
            "left": proposed_episode["current_left"],
            "right": proposed_episode["current_right"],
            "left_phase": proposed_episode["left_phase"],
            "right_phase": proposed_episode["right_phase"],
            "fps": proposed_episode["fps"],
        }
    if variant == "proposed_hand_v2_1":
        return {
            "arm": proposed_episode["arm"],
            "arm_payload": proposed_episode["arm_payload"],
            "left": proposed_episode["candidate_left"],
            "right": proposed_episode["candidate_right"],
            "left_phase": proposed_episode["left_phase"],
            "right_phase": proposed_episode["right_phase"],
            "fps": proposed_episode["fps"],
        }
    if variant != "dataset_a_baseline":
        raise ValueError(variant)
    folder = V1_ROOT / "baseline" / f"episode_{episode_id:06d}"
    with np.load(folder / "g1_arm_action.npz", allow_pickle=False) as payload:
        arm = payload["action"].astype(np.float64)
        arm_payload = {key: payload[key].copy() for key in payload.files}
    with np.load(folder / "g1_hand_action.npz", allow_pickle=False) as payload:
        left = payload["left_action"].astype(np.float64)
        right = payload["right_action"].astype(np.float64)
        fps = float(payload["fps"])
    # The source semantic detector is common to A/B.  Use Proposed's stored
    # five-state labels only as an evaluation mask; never as Dataset A labels.
    return {
        "arm": arm,
        "arm_payload": arm_payload,
        "left": left,
        "right": right,
        "left_phase": proposed_episode["left_phase"],
        "right_phase": proposed_episode["right_phase"],
        "fps": fps,
    }


def _stats(value: np.ndarray) -> dict[str, float | None]:
    value = np.asarray(value, dtype=np.float64)
    if not len(value):
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
    }


def evaluate_variant(
    variant: str,
    runtime: Any,
    classifier: CollisionClassifier,
    episodes: list[dict[str, Any]],
    v1_config: Mapping[str, Any],
) -> dict[str, Any]:
    category_frames: Counter[str] = Counter()
    pair_events: Counter[str] = Counter()
    category_episodes: defaultdict[str, set[int]] = defaultdict(set)
    totals: Counter[str] = Counter()
    episode_rows: list[dict[str, Any]] = []
    all_pinch_errors: list[float] = []
    critical_pinch_errors: list[float] = []
    critical_closing_errors: list[float] = []
    episode_critical_means: list[float] = []
    episode_closing_means: list[float] = []
    max_step = 0.0
    max_velocity = 0.0
    max_acceleration = 0.0
    mean_velocity: list[float] = []
    mean_acceleration: list[float] = []
    total_transitions = 0
    missing_phase_episode_sides = 0
    unknown_phase_count = 0
    finite = True
    joint_limit_violations = 0
    total_frames = 0
    labels = {
        side: tuple(v1_config["target_frames"][f"{side}_physical_pinch_contacts"])
        for side in ("left", "right")
    }
    old_tool = {
        side: np.asarray(
            v1_config["target_frames"][f"{side}_wrist_to_physical_pinch"],
            dtype=np.float64,
        )
        for side in ("left", "right")
    }
    limits = {
        side: np.asarray(runtime.hand_limits[side], dtype=np.float64)
        for side in ("left", "right")
    }

    for episode_id, proposed_episode in enumerate(episodes):
        episode = _variant_episode(variant, episode_id, proposed_episode)
        arm = episode["arm"]
        left = episode["left"]
        right = episode["right"]
        fps = float(episode["fps"])
        dt = 1.0 / fps
        total_frames += len(arm)
        combined = np.column_stack((left, right))
        finite = finite and bool(np.isfinite(combined).all())
        violations = int(
            np.count_nonzero((left < limits["left"][:, 0]) | (left > limits["left"][:, 1]))
            + np.count_nonzero(
                (right < limits["right"][:, 0]) | (right > limits["right"][:, 1])
            )
        )
        joint_limit_violations += violations
        step = np.abs(np.diff(combined, axis=0))
        velocity = np.diff(combined, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        max_step = max(max_step, float(np.max(step, initial=0.0)))
        max_velocity = max(max_velocity, float(np.max(np.abs(velocity), initial=0.0)))
        max_acceleration = max(
            max_acceleration, float(np.max(np.abs(acceleration), initial=0.0))
        )
        mean_velocity.append(float(np.mean(np.abs(velocity))) if velocity.size else 0.0)
        mean_acceleration.append(float(np.mean(np.abs(acceleration))) if acceleration.size else 0.0)

        episode_counts: Counter[str] = Counter()
        episode_categories: Counter[str] = Counter()
        episode_critical: dict[str, list[float]] = {"left": [], "right": []}
        episode_closing: dict[str, list[float]] = {"left": [], "right": []}
        for side in ("left", "right"):
            phase = np.asarray(episode[f"{side}_phase"]).astype(str)
            total_transitions += int(np.count_nonzero(phase[1:] != phase[:-1]))
            unknown_phase_count += int(np.count_nonzero(~np.isin(phase, PHASES)))
            missing_phase_episode_sides += int(set(PHASES) - set(phase.tolist()) != set())

        for frame in range(len(arm)):
            runtime.assign(arm[frame], left[frame], right[frame])
            records = classifier.records()
            gate = [record for record in records if record.v1_gate_relevant]
            hand = [
                record
                for record in records
                if any(body_digit(name) is not None for name in record.bodies)
            ]
            gate_hand = [record for record in gate if record in hand]
            third = [
                record
                for record in hand
                if any(body_digit(name) == "THIRD" for name in record.bodies)
            ]
            task = [
                record
                for record in hand
                if any(body_digit(name) in {"THUMB", "INDEX"} for name in record.bodies)
            ]
            flags = {
                "historical_gate_collision_frames": bool(gate),
                "historical_gate_hand_related_frames": bool(gate_hand),
                "comprehensive_hand_related_frames": bool(hand),
                "third_finger_related_frames": bool(third),
                "thumb_index_related_frames": bool(task),
                "same_hand_self_contact_frames": any(
                    record.enhanced_same_hand for record in hand
                ),
                "hand_hand_frames": any(record.category == "HAND_HAND" for record in hand),
                "cross_arm_finger_involved_frames": any(
                    record.category == "CROSS_ARM" for record in hand
                ),
            }
            for key, value in flags.items():
                if value:
                    totals[key] += 1
                    episode_counts[key] += 1
            for category in {record.category for record in records}:
                category_frames[category] += 1
                category_episodes[category].add(episode_id)
                episode_categories[category] += 1
            for record in records:
                pair_events[record.pair] += 1

            arm_payload = episode["arm_payload"]
            for side, hand_q in (("left", left[frame]), ("right", right[frame])):
                pinch = physical_pinch_frame(runtime, side, labels[side])
                target_position = np.asarray(
                    arm_payload[f"target_{side}_task_tool_position"][frame],
                    dtype=np.float64,
                )
                error = float(np.linalg.norm(pinch[:3, 3] - target_position))
                all_pinch_errors.append(error)
                phase = str(episode[f"{side}_phase"][frame])
                if phase in {"GRASP", "HOLD"}:
                    wrist_rotation = np.asarray(
                        arm_payload[f"target_{side}_wrist_rotation"][frame],
                        dtype=np.float64,
                    )
                    target_closing = wrist_rotation @ old_tool[side][:3, 1]
                    closing_error = angle_rad(pinch[:3, 1], target_closing)
                    critical_pinch_errors.append(error)
                    critical_closing_errors.append(closing_error)
                    episode_critical[side].append(error)
                    episode_closing[side].append(closing_error)
        critical_mean = 0.5 * (
            float(np.mean(episode_critical["left"]))
            + float(np.mean(episode_critical["right"]))
        )
        closing_mean = 0.5 * (
            float(np.mean(episode_closing["left"]))
            + float(np.mean(episode_closing["right"]))
        )
        episode_critical_means.append(critical_mean)
        episode_closing_means.append(closing_mean)
        episode_rows.append(
            {
                "variant": variant,
                "episode_id": episode_id,
                "frame_count": len(arm),
                "finite": bool(np.isfinite(combined).all()),
                "joint_limit_violation_count": violations,
                "historical_gate_collision_frames": episode_counts[
                    "historical_gate_collision_frames"
                ],
                "historical_gate_hand_related_frames": episode_counts[
                    "historical_gate_hand_related_frames"
                ],
                "comprehensive_hand_related_frames": episode_counts[
                    "comprehensive_hand_related_frames"
                ],
                "third_finger_related_frames": episode_counts[
                    "third_finger_related_frames"
                ],
                "thumb_index_related_frames": episode_counts[
                    "thumb_index_related_frames"
                ],
                "same_hand_self_contact_frames": episode_counts[
                    "same_hand_self_contact_frames"
                ],
                "hand_hand_frames": episode_counts["hand_hand_frames"],
                "cross_arm_finger_involved_frames": episode_counts[
                    "cross_arm_finger_involved_frames"
                ],
                "task_critical_pinch_error_mean_m": critical_mean,
                "closing_axis_error_mean_rad": closing_mean,
                "category_frame_incidence": dict(sorted(episode_categories.items())),
            }
        )

    return {
        "variant": variant,
        "episode_count": 50,
        "total_frames": total_frames,
        "collision": {
            **{key: int(totals[key]) for key in COLLISION_METRIC_NAMES},
            "category_frame_incidence": {
                category: int(category_frames[category])
                for category in COLLISION_CATEGORIES
            },
            "category_episodes_affected": {
                key: len(value) for key, value in sorted(category_episodes.items())
            },
            "top_pairs": [
                {"pair": pair, "pair_events": count}
                for pair, count in pair_events.most_common(20)
            ],
        },
        "kinematics": {
            "finite": finite,
            "joint_limit_violation_count": joint_limit_violations,
            "max_hand_joint_step_rad": max_step,
            "max_hand_velocity_rad_s": max_velocity,
            "max_hand_acceleration_rad_s2": max_acceleration,
            "mean_hand_velocity_rad_s": float(np.mean(mean_velocity)),
            "mean_hand_acceleration_rad_s2": float(np.mean(mean_acceleration)),
        },
        "pinch_frame": {
            "physical_pinch_center_error_frame_weighted_m": _stats(
                np.asarray(all_pinch_errors)
            ),
            "task_critical_pinch_error_frame_weighted_m": _stats(
                np.asarray(critical_pinch_errors)
            ),
            "task_critical_pinch_error_episode_mean_m": _stats(
                np.asarray(episode_critical_means)
            ),
            "closing_axis_error_frame_weighted_rad": _stats(
                np.asarray(critical_closing_errors)
            ),
            "closing_axis_error_episode_mean_rad": _stats(
                np.asarray(episode_closing_means)
            ),
            "target_definition": (
                "source-supported Proposed task-tool target and old authoritative static "
                "closing axis; no source object pose used"
            ),
        },
        "semantic": {
            "phase_vocabulary": list(PHASES),
            "unknown_phase_count": unknown_phase_count,
            "episode_side_missing_any_phase_count": missing_phase_episode_sides,
            "transition_count": total_transitions,
            "evaluation_mask": "common source semantic phases GRASP or HOLD",
        },
        "episodes": episode_rows,
    }


def exact_v2_ablation_summary() -> dict[str, Any]:
    metrics = json.loads(
        (V2_ROOT / "development/interaction_ik_metrics.json").read_text(encoding="utf-8")
    )
    current = json.loads(
        (V2_ROOT / "development/current_primitive_metrics.json").read_text(encoding="utf-8")
    )
    return {
        "variant": "diagnostic_exact_contact_ik_v2",
        "scope": "one development grasp plus three-episode smoke test; not a 50-episode label sweep",
        "default_mapper": False,
        "mean_task_finger_error_m": metrics["result"]["metrics"]["mean_task_finger_error_m"],
        "physical_pinch_center_error_m": metrics["result"]["metrics"][
            "physical_pinch_center_error_m"
        ],
        "contact_feasible": metrics["result"]["solver"]["contact_feasible"],
        "blocker": metrics["result"]["solver"]["blocker_classification"],
        "current_fixed_phone_pinch_mean_task_finger_error_m": current["comparators"][
            "fixed_phone_pinch"
        ]["mean_task_finger_error_m"],
    }


def compare_variants(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    exact_v2: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    variants = {
        "dataset_a_baseline": baseline,
        "current_proposed_v1": current,
        "diagnostic_exact_contact_ik_v2": exact_v2,
        "proposed_hand_v2_1": candidate,
    }
    collision = {
        "schema_version": "proposed_hand_v2_1_collision_comparison",
        "definitions": {
            "historical_gate": "v1 cross-body collision gate; same-side hand-chain excluded",
            "comprehensive_hand_related": "union of all logged gate and same-side records containing any Dex3 digit",
            "frame_incidence": "each category counted at most once per frame; categories may overlap",
        },
        "variants": {
            key: value["collision"]
            for key, value in variants.items()
            if key != "diagnostic_exact_contact_ik_v2"
        },
        "diagnostic_exact_contact_ik_v2": exact_v2,
    }
    current_third = int(current["collision"]["third_finger_related_frames"])
    candidate_third = int(candidate["collision"]["third_finger_related_frames"])
    current_hand = int(current["collision"]["comprehensive_hand_related_frames"])
    candidate_hand = int(candidate["collision"]["comprehensive_hand_related_frames"])
    aggregate = {
        "schema_version": "proposed_hand_v2_1_aggregate_comparison",
        "scope_notes": {
            "dataset_a_baseline": "historical Dataset A hand and Dataset A arm",
            "current_proposed_v1": "historical Proposed hand and frozen Proposed arm",
            "diagnostic_exact_contact_ik_v2": exact_v2["scope"],
            "proposed_hand_v2_1": "candidate hand and byte-identical frozen Proposed arm",
        },
        "variants": variants,
        "v1_to_v2_1": {
            "third_finger_collision_frame_delta": candidate_third - current_third,
            "third_finger_collision_fractional_reduction": (
                (current_third - candidate_third) / current_third if current_third else 0.0
            ),
            "comprehensive_hand_collision_frame_delta": candidate_hand - current_hand,
            "comprehensive_hand_collision_fractional_reduction": (
                (current_hand - candidate_hand) / current_hand if current_hand else 0.0
            ),
            "task_critical_pinch_error_episode_mean_delta_m": float(
                candidate["pinch_frame"]["task_critical_pinch_error_episode_mean_m"]["mean"]
                - current["pinch_frame"]["task_critical_pinch_error_episode_mean_m"]["mean"]
            ),
        },
        "metric_support": {
            "source_supported_semantic_metrics": [
                "semantic phases",
                "transition count",
                "task-tool pinch-center error",
                "closing-axis error relative to source-derived Proposed task frame",
            ],
            "target_simulation_diagnostic_metrics": [
                "active-model collision",
                "joint limits",
                "object-class surface aperture",
                "static wrist-to-pinch transform",
            ],
            "not_available": [
                "per-episode source object-relative contacts",
                "per-frame target object contact/penetration",
            ],
        },
    }
    rows = candidate["episodes"]
    return aggregate, collision, rows


__all__ = [
    "compare_variants",
    "evaluate_variant",
    "exact_v2_ablation_summary",
    "install_global_neutrals",
]
