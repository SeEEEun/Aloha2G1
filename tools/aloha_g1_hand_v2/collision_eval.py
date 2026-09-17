"""Deterministic offline collision attribution for frozen v1 trajectories."""
from __future__ import annotations

import contextlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from aloha_g1_v15.kinematics import ActiveG1Dex3

from .common import ROOT, V1_ROOT, load_v1_config


COLLISION_CATEGORIES = (
    "ARM_TORSO",
    "CROSS_ARM",
    "WRIST_OR_PALM_TORSO",
    "HAND_HAND",
    "THUMB_TORSO",
    "INDEX_TORSO",
    "THIRD_TORSO",
    "THUMB_INDEX_SELF",
    "THUMB_WRIST_SELF",
    "INDEX_WRIST_SELF",
    "THIRD_WRIST_SELF",
    "OTHER_SAME_HAND",
    "OTHER",
)


def body_side(name: str) -> str | None:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return None


def body_digit(name: str) -> str | None:
    if "_hand_thumb_" in name:
        return "THUMB"
    if "_hand_index_" in name:
        return "INDEX"
    if "_hand_middle_" in name:
        return "THIRD"
    return None


def is_torso(name: str) -> bool:
    return any(token in name for token in ("torso", "waist", "pelvis"))


def is_wrist_or_palm(name: str) -> bool:
    # The active menagerie hand is attached directly to wrist_yaw_link; the
    # palm collision meshes therefore belong to that body.
    return "wrist" in name or "palm" in name


def is_arm(name: str) -> bool:
    return any(token in name for token in ("shoulder", "elbow", "wrist"))


def same_side_hand_chain(bodies: tuple[str, str]) -> bool:
    first, second = bodies
    return bool(
        (first.startswith("left_hand") and second.startswith("left_"))
        or (second.startswith("left_hand") and first.startswith("left_"))
        or (first.startswith("right_hand") and second.startswith("right_"))
        or (second.startswith("right_hand") and first.startswith("right_"))
    )


def classify_collision_pair(bodies: tuple[str, str]) -> str:
    """Return exactly one category for a body pair."""
    first, second = bodies
    first_digit, second_digit = body_digit(first), body_digit(second)
    digits = {value for value in (first_digit, second_digit) if value is not None}
    first_side, second_side = body_side(first), body_side(second)

    if is_torso(first) or is_torso(second):
        if "THUMB" in digits:
            return "THUMB_TORSO"
        if "INDEX" in digits:
            return "INDEX_TORSO"
        if "THIRD" in digits:
            return "THIRD_TORSO"
        if is_wrist_or_palm(first) or is_wrist_or_palm(second):
            return "WRIST_OR_PALM_TORSO"
        if is_arm(first) or is_arm(second):
            return "ARM_TORSO"

    if first_side is not None and second_side is not None and first_side != second_side:
        if first_digit is not None and second_digit is not None:
            return "HAND_HAND"
        if first_digit is not None or second_digit is not None or is_arm(first) or is_arm(second):
            return "CROSS_ARM"

    if first_side is not None and first_side == second_side:
        if digits == {"THUMB", "INDEX"}:
            return "THUMB_INDEX_SELF"
        checks = (
            ("THUMB", "THUMB_WRIST_SELF"),
            ("INDEX", "INDEX_WRIST_SELF"),
            ("THIRD", "THIRD_WRIST_SELF"),
        )
        for digit, category in checks:
            if (
                (first_digit == digit and is_wrist_or_palm(second))
                or (second_digit == digit and is_wrist_or_palm(first))
            ):
                return category
        if first_digit is not None or second_digit is not None:
            return "OTHER_SAME_HAND"
    return "OTHER"


def pair_cause_group(bodies: tuple[str, str]) -> str:
    """Exclusive pair-level causal grouping used for attribution totals."""
    digits = {body_digit(value) for value in bodies} - {None}
    if same_side_hand_chain(bodies) and digits:
        return "placeholder_hand_self_collision"
    if "THIRD" in digits:
        return "non_task_third_finger_collision"
    if digits & {"THUMB", "INDEX"}:
        return "task_finger_collision"
    if any(is_arm(value) or is_torso(value) for value in bodies):
        return "arm_caused_collision"
    return "other_collision"


@dataclass(frozen=True)
class CollisionRecord:
    distance_m: float
    geom_ids: tuple[int, int]
    geom_names: tuple[str, str]
    bodies: tuple[str, str]
    pair: str
    category: str
    cause_group: str
    v1_gate_relevant: bool
    enhanced_same_hand: bool


class CollisionClassifier:
    """Reproduce v1 exactly and add a separate same-hand diagnostic channel."""

    def __init__(self, runtime: ActiveG1Dex3, v1_config: dict[str, Any]):
        self.runtime = runtime
        self.model = runtime.model
        validation = v1_config["validation"]
        self.tolerance = float(validation["collision_penetration_tolerance_m"])
        self.allowlist = set(validation["static_model_contact_allowlist"])

    def records(self) -> list[CollisionRecord]:
        output: list[CollisionRecord] = []
        for contact in self.runtime.data.contact:
            if float(contact.dist) >= -self.tolerance:
                continue
            geom_ids = (int(contact.geom1), int(contact.geom2))
            geom_names: list[str] = []
            bodies: list[str] = []
            for geom in geom_ids:
                geom_names.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                    or f"geom_{geom}"
                )
                body = int(self.model.geom_bodyid[geom])
                bodies.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                    or f"body_{body}"
                )
            body_tuple = (bodies[0], bodies[1])
            pair = "|".join(sorted(body_tuple))
            if bodies[0] == bodies[1] or pair in self.allowlist:
                continue
            same_hand = same_side_hand_chain(body_tuple)
            relevant_tokens = any(
                word in "|".join(bodies)
                for word in ("shoulder", "elbow", "wrist", "hand", "torso")
            )
            gate_relevant = bool(relevant_tokens and not same_hand)
            enhanced = bool(relevant_tokens and same_hand)
            if not gate_relevant and not enhanced:
                continue
            output.append(
                CollisionRecord(
                    distance_m=float(contact.dist),
                    geom_ids=geom_ids,
                    geom_names=(geom_names[0], geom_names[1]),
                    bodies=body_tuple,
                    pair=pair,
                    category=classify_collision_pair(body_tuple),
                    cause_group=pair_cause_group(body_tuple),
                    v1_gate_relevant=gate_relevant,
                    enhanced_same_hand=enhanced,
                )
            )
        # MuJoCo can emit more than one geom contact for one body pair.  The v1
        # gate is body-pair/frame based, so retain the deepest record per pair.
        deepest: dict[str, CollisionRecord] = {}
        for record in output:
            if record.pair not in deepest or record.distance_m < deepest[record.pair].distance_m:
                deepest[record.pair] = record
        return [deepest[key] for key in sorted(deepest)]

    def metrics(self) -> dict[str, Any]:
        records = self.records()
        gate = [row for row in records if row.v1_gate_relevant]
        enhanced = [row for row in records if row.enhanced_same_hand]
        return {
            "v1_gate_collision": bool(gate),
            "v1_gate_pair_count": len(gate),
            "v1_gate_pairs": [row.pair for row in gate],
            "v1_gate_categories": sorted({row.category for row in gate}),
            "enhanced_same_hand_collision": bool(enhanced),
            "enhanced_same_hand_pair_count": len(enhanced),
            "enhanced_same_hand_pairs": [row.pair for row in enhanced],
            "all_hand_related_pair_count": sum(
                any(body_digit(name) is not None for name in row.bodies) for row in records
            ),
            "records": records,
        }


def make_runtime(v1_config: dict[str, Any] | None = None) -> ActiveG1Dex3:
    config = v1_config or load_v1_config()
    hand = config["hand_mapping"]
    with contextlib.redirect_stdout(io.StringIO()):
        return ActiveG1Dex3(
            config["models"]["g1_xml"],
            hand["dex3_mapping_source"],
            hand["palm_source"],
            np.asarray([0.0, 0.0, 0.7922728583]),
        )


def _audit_method(
    method: str,
    runtime: ActiveG1Dex3,
    classifier: CollisionClassifier,
) -> dict[str, Any]:
    category_frames: Counter[str] = Counter()
    category_pair_events: Counter[str] = Counter()
    category_episodes: defaultdict[str, set[int]] = defaultdict(set)
    pair_events: Counter[str] = Counter()
    pair_episodes: defaultdict[str, set[int]] = defaultdict(set)
    cause_pair_events: Counter[str] = Counter()
    cause_frames: Counter[str] = Counter()
    cause_episodes: defaultdict[str, set[int]] = defaultdict(set)
    finger_involved_frames = 0
    arm_only_frames = 0
    mixed_pair_type_frames = 0
    enhanced_frames = 0
    enhanced_and_v1_overlap_frames = 0
    all_logged_union_frames = 0
    enhanced_pair_events: Counter[str] = Counter()
    enhanced_episodes: defaultdict[str, set[int]] = defaultdict(set)
    enhanced_category_frames: Counter[str] = Counter()
    enhanced_category_pair_events: Counter[str] = Counter()
    enhanced_category_episodes: defaultdict[str, set[int]] = defaultdict(set)
    enhanced_cause_pair_events: Counter[str] = Counter()
    enhanced_cause_frames: Counter[str] = Counter()
    enhanced_cause_episodes: defaultdict[str, set[int]] = defaultdict(set)
    episode_rows: list[dict[str, Any]] = []
    reconstructed_total = 0
    expected_total = 0
    total_frames = 0
    parity = True

    for episode_id in range(50):
        folder = V1_ROOT / method / f"episode_{episode_id:06d}"
        with np.load(folder / "g1_arm_action.npz", allow_pickle=False) as payload:
            arm = payload["action"].astype(np.float64)
        with np.load(folder / "g1_hand_action.npz", allow_pickle=False) as payload:
            left = payload["left_action"].astype(np.float64)
            right = payload["right_action"].astype(np.float64)
        expected = int(
            json.loads((folder / "retargeting_metrics.json").read_text(encoding="utf-8"))[
                "prohibited_arm_self_collision_count"
            ]
        )
        episode_gate = 0
        episode_enhanced = 0
        episode_categories: Counter[str] = Counter()
        episode_pairs: Counter[str] = Counter()
        for frame in range(len(arm)):
            runtime.assign(arm[frame], left[frame], right[frame])
            records = classifier.records()
            gate = [row for row in records if row.v1_gate_relevant]
            enhanced = [row for row in records if row.enhanced_same_hand]
            if gate:
                episode_gate += 1
                frame_categories = {row.category for row in gate}
                for category in frame_categories:
                    category_frames[category] += 1
                    episode_categories[category] += 1
                    category_episodes[category].add(episode_id)
                has_finger_pair = any(
                    any(body_digit(name) is not None for name in row.bodies) for row in gate
                )
                has_nonfinger_pair = any(
                    all(body_digit(name) is None for name in row.bodies) for row in gate
                )
                finger_involved_frames += int(has_finger_pair)
                arm_only_frames += int(has_nonfinger_pair and not has_finger_pair)
                mixed_pair_type_frames += int(has_finger_pair and has_nonfinger_pair)
                for row in gate:
                    pair_events[row.pair] += 1
                    pair_episodes[row.pair].add(episode_id)
                    episode_pairs[row.pair] += 1
                    category_pair_events[row.category] += 1
                    cause_pair_events[row.cause_group] += 1
                for cause in {row.cause_group for row in gate}:
                    cause_frames[cause] += 1
                    cause_episodes[cause].add(episode_id)
            if enhanced:
                episode_enhanced += 1
                enhanced_categories = {row.category for row in enhanced}
                for category in enhanced_categories:
                    enhanced_category_frames[category] += 1
                    enhanced_category_episodes[category].add(episode_id)
                for row in enhanced:
                    enhanced_pair_events[row.pair] += 1
                    enhanced_episodes[row.pair].add(episode_id)
                    enhanced_category_pair_events[row.category] += 1
                    enhanced_cause_pair_events[row.cause_group] += 1
                for cause in {row.cause_group for row in enhanced}:
                    enhanced_cause_frames[cause] += 1
                    enhanced_cause_episodes[cause].add(episode_id)
            all_logged_union_frames += int(bool(gate or enhanced))
            enhanced_and_v1_overlap_frames += int(bool(gate and enhanced))
        total_frames += len(arm)
        reconstructed_total += episode_gate
        expected_total += expected
        parity = parity and episode_gate == expected
        enhanced_frames += episode_enhanced
        episode_rows.append(
            {
                "method": method,
                "episode_id": episode_id,
                "frame_count": len(arm),
                "v1_expected_collision_frames": expected,
                "v1_reconstructed_collision_frames": episode_gate,
                "v1_parity": episode_gate == expected,
                "enhanced_same_hand_collision_frames": episode_enhanced,
                "category_frame_counts": dict(sorted(episode_categories.items())),
                "top_pairs": [
                    {"pair": pair, "frames": frames}
                    for pair, frames in episode_pairs.most_common(5)
                ],
            }
        )

    return {
        "method": method,
        "total_frames": total_frames,
        "v1_collision_frames": reconstructed_total,
        "v1_expected_collision_frames": expected_total,
        "v1_exact_reconstruction": parity and reconstructed_total == expected_total,
        "category_frame_incidence": {
            category: int(category_frames[category]) for category in COLLISION_CATEGORIES
        },
        "category_pair_events": {
            category: int(category_pair_events[category]) for category in COLLISION_CATEGORIES
        },
        "category_episodes_affected": {
            category: len(category_episodes[category]) for category in COLLISION_CATEGORIES
        },
        "finger_involved_collision_frames": finger_involved_frames,
        "finger_involved_collision_frame_percent": (
            100.0 * finger_involved_frames / reconstructed_total if reconstructed_total else 0.0
        ),
        "arm_wrist_palm_only_collision_frames": arm_only_frames,
        "arm_wrist_palm_only_collision_frame_percent": (
            100.0 * arm_only_frames / reconstructed_total if reconstructed_total else 0.0
        ),
        "mixed_finger_and_nonfinger_pair_frames": mixed_pair_type_frames,
        "cause_group_pair_events": dict(sorted(cause_pair_events.items())),
        "cause_group_frame_incidence": dict(sorted(cause_frames.items())),
        "cause_group_episodes_affected": {
            key: len(cause_episodes[key]) for key in sorted(cause_episodes)
        },
        "top_link_pairs": [
            {
                "pair": pair,
                "pair_events": frames,
                "episodes_affected": len(pair_episodes[pair]),
            }
            for pair, frames in pair_events.most_common(20)
        ],
        "enhanced_logger": {
            "purpose": "same-side hand-chain contacts excluded from the historical v1 gate",
            "same_hand_collision_frames_nonexclusive_with_v1": enhanced_frames,
            "same_hand_and_v1_overlap_frames": enhanced_and_v1_overlap_frames,
            "all_logged_union_frames": all_logged_union_frames,
            "category_frame_incidence": {
                category: int(enhanced_category_frames[category])
                for category in COLLISION_CATEGORIES
            },
            "category_pair_events": {
                category: int(enhanced_category_pair_events[category])
                for category in COLLISION_CATEGORIES
            },
            "category_episodes_affected": {
                category: len(enhanced_category_episodes[category])
                for category in COLLISION_CATEGORIES
            },
            "cause_group_pair_events": dict(sorted(enhanced_cause_pair_events.items())),
            "cause_group_frame_incidence": dict(sorted(enhanced_cause_frames.items())),
            "cause_group_episodes_affected": {
                key: len(enhanced_cause_episodes[key])
                for key in sorted(enhanced_cause_episodes)
            },
            "top_same_hand_pairs": [
                {
                    "pair": pair,
                    "pair_events": frames,
                    "episodes_affected": len(enhanced_episodes[pair]),
                }
                for pair, frames in enhanced_pair_events.most_common(20)
            ],
        },
        "episodes": episode_rows,
    }


def audit_current_50() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_v1_config()
    runtime = make_runtime(config)
    classifier = CollisionClassifier(runtime, config)
    baseline = _audit_method("baseline", runtime, classifier)
    proposed = _audit_method("proposed", runtime, classifier)
    total = int(proposed["v1_collision_frames"])
    delta_total = total - int(baseline["v1_collision_frames"])
    delta_finger = int(proposed["finger_involved_collision_frames"]) - int(
        baseline["finger_involved_collision_frames"]
    )
    delta_arm_only = int(proposed["arm_wrist_palm_only_collision_frames"]) - int(
        baseline["arm_wrist_palm_only_collision_frames"]
    )
    logging_ok = bool(baseline["v1_exact_reconstruction"] and proposed["v1_exact_reconstruction"])
    if not logging_ok:
        decision = "INSUFFICIENT_COLLISION_LOGGING"
    elif (
        float(proposed["finger_involved_collision_frame_percent"]) >= 65.0
        and delta_total > 0
        and delta_finger / delta_total >= 0.75
    ):
        decision = "HAND_PLACEHOLDER_DOMINANT"
    elif float(proposed["arm_wrist_palm_only_collision_frame_percent"]) >= 65.0:
        decision = "ARM_REACHABILITY_DOMINANT"
    else:
        decision = "MIXED_HAND_AND_ARM"

    rows: list[dict[str, Any]] = []
    for category in COLLISION_CATEGORIES:
        rows.append(
            {
                "category": category,
                "baseline_collision_frame_incidence": baseline["category_frame_incidence"][category],
                "proposed_collision_frame_incidence": proposed["category_frame_incidence"][category],
                "delta_frame_incidence": proposed["category_frame_incidence"][category]
                - baseline["category_frame_incidence"][category],
                "proposed_pair_events": proposed["category_pair_events"][category],
                "proposed_episodes_affected": proposed["category_episodes_affected"][category],
                "proposed_enhanced_same_hand_frame_incidence": proposed["enhanced_logger"][
                    "category_frame_incidence"
                ][category],
                "proposed_enhanced_same_hand_pair_events": proposed["enhanced_logger"][
                    "category_pair_events"
                ][category],
                "proposed_percent_of_collision_frames_nonexclusive": (
                    100.0 * proposed["category_frame_incidence"][category] / total if total else 0.0
                ),
            }
        )
    largest = max(rows, key=lambda row: (row["proposed_collision_frame_incidence"], row["category"]))
    artifact = {
        "schema_version": "g1_proposed_collision_attribution_50ep_v2",
        "scope": "offline replay of frozen v1 Dataset A/B outputs; no physics stepping",
        "historical_v1_gate_definition_preserved": True,
        "collision_penetration_tolerance_m": classifier.tolerance,
        "category_pair_assignment_mutually_exclusive": True,
        "category_frame_incidence_can_overlap": True,
        "baseline": baseline,
        "proposed": proposed,
        "baseline_to_proposed": {
            "collision_frame_delta": delta_total,
            "finger_involved_collision_frame_delta": delta_finger,
            "arm_wrist_palm_only_collision_frame_delta": delta_arm_only,
            "fraction_of_positive_delta_with_finger_involvement": (
                delta_finger / delta_total if delta_total > 0 else None
            ),
        },
        "largest_proposed_category": largest["category"],
        "decision_gate": decision,
        "decision_evidence": {
            "v1_reconstruction_exact": logging_ok,
            "proposed_finger_involved_percent": proposed[
                "finger_involved_collision_frame_percent"
            ],
            "proposed_arm_wrist_palm_only_percent": proposed[
                "arm_wrist_palm_only_collision_frame_percent"
            ],
            "delta_total_frames": delta_total,
            "delta_finger_involved_frames": delta_finger,
            "delta_arm_only_frames": delta_arm_only,
        },
        "definitions": {
            "Dex3_finger_attributable_frame": "at least one v1-gate pair contains a thumb/index/third Dex3 body",
            "arm_wrist_palm_only_frame": "v1-gate frame with no pair containing a Dex3 digit body",
            "enhanced_same_hand_logger": "separate diagnostic; excluded from the 4,696 historical v1 denominator",
            "task_fingers": ["THUMB", "INDEX"],
            "non_task_finger": "THIRD (active model body token middle)",
        },
    }
    return artifact, rows
