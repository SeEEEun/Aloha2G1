"""Deterministic global third-finger neutral search on frozen Proposed arms."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from aloha_g1_hand_v2.collision_eval import (
    CollisionClassifier,
    body_digit,
    body_side,
    is_torso,
    is_wrist_or_palm,
)

from .common import V1_ROOT


def minimum_jerk_state_commands(
    labels: np.ndarray,
    primitives: Mapping[str, np.ndarray],
    transition_frames: int,
    fps: float,
) -> np.ndarray:
    """Causal convex FIR whose isolated step response is quintic minimum jerk.

    The non-negative impulse weights sum to one, so arbitrary/overlapping
    semantic transitions remain inside the convex hull of configured states.
    That guarantees no interpolation-induced joint-limit overshoot.
    """
    labels = np.asarray(labels).astype(str)
    del fps  # duration is represented exactly by the global integer frame count
    targets = np.asarray([primitives[str(label)] for label in labels], dtype=np.float64)
    count = int(transition_frames)
    edges = np.linspace(0.0, 1.0, count + 1)
    step = 10.0 * edges**3 - 15.0 * edges**4 + 6.0 * edges**5
    impulse = np.diff(step)
    if np.any(impulse < -1e-14) or not np.isclose(np.sum(impulse), 1.0):
        raise RuntimeError("invalid minimum-jerk FIR weights")
    impulse = np.maximum(impulse, 0.0)
    impulse /= np.sum(impulse)
    padded = np.vstack((np.repeat(targets[:1], count - 1, axis=0), targets))
    output = np.empty_like(targets)
    offset = count - 1
    for joint in range(targets.shape[1]):
        filtered = np.convolve(padded[:, joint], impulse, mode="full")
        output[:, joint] = filtered[offset : offset + len(targets)]
    return output


def load_frozen_episodes(
    primitive_build: Mapping[str, Any], transition_frames: int
) -> list[dict[str, Any]]:
    """Load arm/phase inputs and construct task-finger-only candidate commands."""
    output: list[dict[str, Any]] = []
    state_q = {
        side: {
            phase: np.asarray(value, dtype=np.float64)
            for phase, value in primitive_build["sides"][side]["states"].items()
        }
        for side in ("left", "right")
    }
    for episode_id in range(50):
        folder = V1_ROOT / "proposed" / f"episode_{episode_id:06d}"
        with np.load(folder / "g1_arm_action.npz", allow_pickle=False) as payload:
            arm = payload["action"].astype(np.float64)
            arm_payload = {key: payload[key].copy() for key in payload.files}
        with np.load(folder / "g1_hand_action.npz", allow_pickle=False) as payload:
            current_left = payload["left_action"].astype(np.float64)
            current_right = payload["right_action"].astype(np.float64)
            left_phase = payload["left_phase"].astype(str)
            right_phase = payload["right_phase"].astype(str)
            timestamps = payload["timestamps"].astype(np.float64)
            fps = float(payload["fps"])
        if not (
            len(arm)
            == len(current_left)
            == len(current_right)
            == len(left_phase)
            == len(right_phase)
        ):
            raise RuntimeError(f"episode {episode_id}: frozen artifact length mismatch")
        candidate_left = minimum_jerk_state_commands(
            left_phase, state_q["left"], transition_frames, fps
        )
        candidate_right = minimum_jerk_state_commands(
            right_phase, state_q["right"], transition_frames, fps
        )
        output.append(
            {
                "episode_id": episode_id,
                "arm": arm,
                "arm_payload": arm_payload,
                "current_left": current_left,
                "current_right": current_right,
                "left_phase": left_phase,
                "right_phase": right_phase,
                "candidate_left": candidate_left,
                "candidate_right": candidate_right,
                "timestamps": timestamps,
                "fps": fps,
            }
        )
    return output


def _transition_context(labels: np.ndarray, context: int) -> set[int]:
    output: set[int] = set()
    for center in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        output.update(range(max(0, int(center) - context), min(len(labels), int(center) + context + 1)))
    return output


def build_search_frame_refs(
    runtime: Any,
    classifier: CollisionClassifier,
    episodes: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Include every old third collision plus deterministic coverage of all episodes."""
    search = config["third_neutral_search"]
    stride = int(search["audit_stride_frames"])
    context = int(search["semantic_transition_context_frames"])
    refs: list[tuple[int, int]] = []
    old_third_frames = 0
    per_episode: list[dict[str, Any]] = []
    for episode in episodes:
        selected = set(range(0, len(episode["arm"]), stride))
        selected.update(_transition_context(episode["left_phase"], context))
        selected.update(_transition_context(episode["right_phase"], context))
        third_indices: list[int] = []
        for frame in range(len(episode["arm"])):
            runtime.assign(
                episode["arm"][frame],
                episode["current_left"][frame],
                episode["current_right"][frame],
            )
            if any(
                any(body_digit(name) == "THIRD" for name in record.bodies)
                for record in classifier.records()
            ):
                third_indices.append(frame)
        old_third_frames += len(third_indices)
        if bool(search["include_all_current_v1_third_collision_frames"]):
            selected.update(third_indices)
        episode_refs = [(int(episode["episode_id"]), frame) for frame in sorted(selected)]
        refs.extend(episode_refs)
        per_episode.append(
            {
                "episode_id": int(episode["episode_id"]),
                "frame_count": len(episode["arm"]),
                "current_v1_third_collision_frames": len(third_indices),
                "search_frames": len(episode_refs),
            }
        )
    return refs, {
        "selection_rule": (
            "all current-v1 comprehensive third-collision frames union deterministic stride "
            "and semantic-transition context"
        ),
        "all_50_episodes_represented": len({episode for episode, _ in refs}) == 50,
        "search_frame_count": len(refs),
        "current_v1_third_collision_frames_replayed": old_third_frames,
        "expected_prior_third_collision_frames": 4037,
        "prior_parity": old_third_frames == 4037,
        "episodes": per_episode,
    }


def _third_frame_flags(side: str, records: Iterable[Any]) -> dict[str, bool]:
    flags = {
        "all": False,
        "same_hand_self": False,
        "opposite_hand": False,
        "opposite_wrist": False,
        "own_wrist_or_palm": False,
        "torso": False,
        "task_thumb_index": False,
        "other": False,
    }
    for record in records:
        matches = [
            index
            for index, name in enumerate(record.bodies)
            if body_side(name) == side and body_digit(name) == "THIRD"
        ]
        if not matches:
            continue
        flags["all"] = True
        other = record.bodies[1 - matches[0]]
        other_side = body_side(other)
        other_digit = body_digit(other)
        categorized = False
        if other_side == side and (
            is_wrist_or_palm(other) or other_digit in {"THUMB", "INDEX", "THIRD"}
        ):
            flags["same_hand_self"] = True
            categorized = True
        if other_side is not None and other_side != side and other_digit is not None:
            flags["opposite_hand"] = True
            categorized = True
        if other_side is not None and other_side != side and is_wrist_or_palm(other):
            flags["opposite_wrist"] = True
            categorized = True
        if other_side == side and is_wrist_or_palm(other):
            flags["own_wrist_or_palm"] = True
            categorized = True
        if is_torso(other):
            flags["torso"] = True
            categorized = True
        if other_digit in {"THUMB", "INDEX"}:
            flags["task_thumb_index"] = True
            categorized = True
        if not categorized:
            flags["other"] = True
    return flags


def evaluate_neutral_pair(
    runtime: Any,
    classifier: CollisionClassifier,
    episodes: list[dict[str, Any]],
    frame_refs: Iterable[tuple[int, int]],
    indices: Mapping[str, np.ndarray],
    neutrals: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    counters = {side: Counter() for side in ("left", "right")}
    episodes_affected = {side: defaultdict(set) for side in ("left", "right")}
    maximum_penetration = {side: 0.0 for side in ("left", "right")}
    pair_events = {side: Counter() for side in ("left", "right")}
    combined = Counter()
    combined_episodes: defaultdict[str, set[int]] = defaultdict(set)
    combined_pairs: Counter[str] = Counter()
    frame_count = 0
    for episode_id, frame in frame_refs:
        episode = episodes[episode_id]
        left = episode["candidate_left"][frame].copy()
        right = episode["candidate_right"][frame].copy()
        left[indices["left"]] = neutrals["left"]
        right[indices["right"]] = neutrals["right"]
        runtime.assign(episode["arm"][frame], left, right)
        records = classifier.records()
        frame_count += 1
        for side in ("left", "right"):
            flags = _third_frame_flags(side, records)
            for key, value in flags.items():
                if value:
                    counters[side][key] += 1
                    episodes_affected[side][key].add(episode_id)
            for record in records:
                if any(
                    body_side(name) == side and body_digit(name) == "THIRD"
                    for name in record.bodies
                ):
                    maximum_penetration[side] = max(
                        maximum_penetration[side], -float(record.distance_m)
                    )
                    pair_events[side][record.pair] += 1
        third_records = [
            record
            for record in records
            if any(body_digit(name) == "THIRD" for name in record.bodies)
        ]
        if third_records:
            combined["all"] += 1
            combined_episodes["all"].add(episode_id)
        if any(record.enhanced_same_hand for record in third_records):
            combined["same_hand_self"] += 1
            combined_episodes["same_hand_self"].add(episode_id)
        for record in third_records:
            combined_pairs[record.pair] += 1
    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        open_neutral = runtime.open_hand_q[side][indices[side]]
        ranges = runtime.hand_limits[side][indices[side], 1] - runtime.hand_limits[side][
            indices[side], 0
        ]
        normalized_distance = float(
            np.linalg.norm((np.asarray(neutrals[side]) - open_neutral) / ranges)
        )
        sides[side] = {
            "q": np.asarray(neutrals[side], dtype=np.float64),
            "evaluated_frames": frame_count,
            "all_third_collision_frames": int(counters[side]["all"]),
            "same_hand_self_collision_frames": int(counters[side]["same_hand_self"]),
            "opposite_hand_frames": int(counters[side]["opposite_hand"]),
            "opposite_wrist_frames": int(counters[side]["opposite_wrist"]),
            "own_wrist_or_palm_frames": int(counters[side]["own_wrist_or_palm"]),
            "torso_frames": int(counters[side]["torso"]),
            "task_thumb_index_frames": int(counters[side]["task_thumb_index"]),
            "other_frames": int(counters[side]["other"]),
            "maximum_penetration_m": float(maximum_penetration[side]),
            "normalized_distance_from_active_open": normalized_distance,
            "episodes_affected": {
                key: len(value) for key, value in sorted(episodes_affected[side].items())
            },
            "top_pairs": [
                {"pair": pair, "pair_events": count}
                for pair, count in pair_events[side].most_common(10)
            ],
        }
    return {
        "frame_count": frame_count,
        "left_neutral_q": np.asarray(neutrals["left"], dtype=np.float64),
        "right_neutral_q": np.asarray(neutrals["right"], dtype=np.float64),
        "combined_third_collision_frames": int(combined["all"]),
        "combined_same_hand_self_collision_frames": int(combined["same_hand_self"]),
        "combined_episodes_affected": {
            key: len(value) for key, value in sorted(combined_episodes.items())
        },
        "combined_top_pairs": [
            {"pair": pair, "pair_events": count}
            for pair, count in combined_pairs.most_common(15)
        ],
        "sides": sides,
    }


def _score(row: Mapping[str, Any], side: str) -> tuple[Any, ...]:
    metrics = row["metrics"]["sides"][side]
    q = np.asarray(row[f"{side}_neutral_q"])
    return (
        int(metrics["same_hand_self_collision_frames"]),
        int(metrics["all_third_collision_frames"]),
        float(metrics["maximum_penetration_m"]),
        float(metrics["normalized_distance_from_active_open"]),
        float(q[0]),
        float(q[1]),
    )


def _pareto(rows: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    keys = (
        "same_hand_self_collision_frames",
        "all_third_collision_frames",
        "maximum_penetration_m",
        "normalized_distance_from_active_open",
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        value = row["metrics"]["sides"][side]
        dominated = False
        for other in rows:
            if other is row:
                continue
            candidate = other["metrics"]["sides"][side]
            no_worse = all(float(candidate[key]) <= float(value[key]) for key in keys)
            strictly_better = any(float(candidate[key]) < float(value[key]) for key in keys)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            output.append(row)
    return sorted(output, key=lambda row: _score(row, side))


def _fraction_grid(center: np.ndarray | None, config: Mapping[str, Any]) -> list[np.ndarray]:
    search = config["third_neutral_search"]
    if center is None:
        values = np.asarray(search["coarse_normalized_joint_fractions"], dtype=np.float64)
        return [np.asarray([first, second]) for first in values for second in values]
    radius = float(search["refine_radius_fraction"])
    points = int(search["refine_points_per_axis"])
    first = np.unique(np.clip(np.linspace(center[0] - radius, center[0] + radius, points), 1e-6, 1 - 1e-6))
    second = np.unique(np.clip(np.linspace(center[1] - radius, center[1] + radius, points), 1e-6, 1 - 1e-6))
    return [np.asarray([a, b]) for a in first for b in second]


def search_global_neutrals(
    runtime: Any,
    classifier: CollisionClassifier,
    episodes: list[dict[str, Any]],
    frame_refs: list[tuple[int, int]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    indices = {
        side: np.asarray(
            [
                runtime.hand_joint_names[side].index(name)
                for name in runtime.contacts[f"{side}_C"].joint_names
            ],
            dtype=np.int64,
        )
        for side in ("left", "right")
    }
    limits = {side: runtime.hand_limits[side][indices[side]] for side in ("left", "right")}

    def q_from_fraction(side: str, fraction: np.ndarray) -> np.ndarray:
        return limits[side][:, 0] + fraction * (limits[side][:, 1] - limits[side][:, 0])

    selected = {
        side: runtime.open_hand_q[side][indices[side]].copy() for side in ("left", "right")
    }
    all_rows: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    stage_rows: list[dict[str, Any]] = []
    selected_fraction: dict[str, np.ndarray] = {
        "left": np.asarray([1.0, 1.0]),
        "right": np.asarray([0.0, 0.0]),
    }
    for stage, side, refine in (
        ("left_coarse", "left", False),
        ("right_coarse", "right", False),
        ("left_refine", "left", True),
        ("right_refine", "right", True),
    ):
        grid = _fraction_grid(selected_fraction[side] if refine else None, config)
        rows: list[dict[str, Any]] = []
        for candidate_id, fraction in enumerate(grid):
            pair = {key: value.copy() for key, value in selected.items()}
            pair[side] = q_from_fraction(side, fraction)
            metrics = evaluate_neutral_pair(
                runtime, classifier, episodes, frame_refs, indices, pair
            )
            row = {
                "stage": stage,
                "side": side,
                "candidate_id": candidate_id,
                "normalized_fraction": fraction,
                "left_neutral_q": pair["left"],
                "right_neutral_q": pair["right"],
                "metrics": metrics,
            }
            rows.append(row)
            all_rows[side].append(row)
        winner = min(rows, key=lambda row: _score(row, side))
        selected[side] = np.asarray(winner[f"{side}_neutral_q"], dtype=np.float64)
        selected_fraction[side] = np.asarray(winner["normalized_fraction"], dtype=np.float64)
        stage_rows.append(
            {
                "stage": stage,
                "side": side,
                "candidate_count": len(rows),
                "selected_fraction": selected_fraction[side],
                "selected_q": selected[side],
                "selected_metrics": winner["metrics"]["sides"][side],
            }
        )
    final_metrics = evaluate_neutral_pair(
        runtime, classifier, episodes, frame_refs, indices, selected
    )
    pareto = {
        side: [
            {
                "stage": row["stage"],
                "normalized_fraction": row["normalized_fraction"],
                "q": row[f"{side}_neutral_q"],
                "metrics": row["metrics"]["sides"][side],
            }
            for row in _pareto(all_rows[side], side)[:20]
        ]
        for side in ("left", "right")
    }
    return {
        "schema_version": "global_third_neutral_search_v2_1",
        "deterministic": True,
        "per_episode_q": False,
        "per_phase_q": False,
        "third_joint_names": {
            side: [runtime.hand_joint_names[side][index] for index in indices[side]]
            for side in ("left", "right")
        },
        "third_indices": indices,
        "joint_limits": limits,
        "selected": selected,
        "selected_search_metrics": final_metrics,
        "stages": stage_rows,
        "pareto_candidates": pareto,
        "candidate_evaluation_count": sum(len(rows) for rows in all_rows.values()),
        "selection_order": config["third_neutral_search"]["selection_order"],
    }


__all__ = [
    "build_search_frame_refs",
    "evaluate_neutral_pair",
    "load_frozen_episodes",
    "minimum_jerk_state_commands",
    "search_global_neutrals",
]
