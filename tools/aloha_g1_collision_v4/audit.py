"""Pair-, frame-, and event-level collision audit before v4 calibration."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np

from aloha_g1_hand_v2.collision_eval import (
    CollisionClassifier,
    body_digit,
    body_side,
    is_arm,
    is_torso,
    is_wrist_or_palm,
)

from .common import (
    METHOD_TO_DATASET,
    METHODS,
    V3_ROOT,
    atomic_csv,
    atomic_json,
    stable_stats,
)


CATEGORIES = (
    "ARM_TORSO",
    "CROSS_ARM",
    "HAND_OPPOSITE_ARM",
    "HAND_HAND",
    "THIRD_OPPOSITE_HAND",
    "WRIST_OPPOSITE_HAND",
    "OTHER_PROHIBITED",
)


def v4_category(bodies: tuple[str, str]) -> str:
    first, second = bodies
    digits = (body_digit(first), body_digit(second))
    sides = (body_side(first), body_side(second))
    if is_torso(first) or is_torso(second):
        # Includes prohibited arm/wrist/hand-to-torso pairs; subtypes remain in
        # the pair names and digit incidence fields.
        return "ARM_TORSO"
    opposite = bool(
        sides[0] is not None and sides[1] is not None and sides[0] != sides[1]
    )
    if opposite and "THIRD" in digits and all(value is not None for value in digits):
        return "THIRD_OPPOSITE_HAND"
    if opposite and all(value is not None for value in digits):
        return "HAND_HAND"
    if opposite and any(value is not None for value in digits):
        other = second if digits[0] is not None else first
        if is_wrist_or_palm(other):
            return "WRIST_OPPOSITE_HAND"
        if is_arm(other):
            return "HAND_OPPOSITE_ARM"
    if opposite and (is_arm(first) or is_arm(second)):
        return "CROSS_ARM"
    return "OTHER_PROHIBITED"


def pair_side(bodies: tuple[str, str]) -> str:
    sides = sorted({value for value in map(body_side, bodies) if value is not None})
    return "+".join(sides) if sides else "unknown"


def _distance_witness(runtime: Any, geom_ids: tuple[int, int]) -> tuple[float, list[float], list[float]]:
    witness = np.zeros(6, dtype=np.float64)
    distance = float(
        mujoco.mj_geomDistance(
            runtime.model,
            runtime.data,
            int(geom_ids[0]),
            int(geom_ids[1]),
            0.10,
            witness,
        )
    )
    delta = witness[3:] - witness[:3]
    norm = float(np.linalg.norm(delta))
    normal = delta / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
    return distance, witness.tolist(), normal.tolist()


def _runs(rows: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], int(row["episode_id"]), row["pair"])].append(row)
    events: list[dict[str, Any]] = []
    for (method, episode_id, pair), values in sorted(groups.items()):
        values.sort(key=lambda row: int(row["frame"]))
        start = 0
        for index in range(1, len(values) + 1):
            boundary = index == len(values) or int(values[index]["frame"]) != int(
                values[index - 1]["frame"]
            ) + 1
            if not boundary:
                continue
            block = values[start:index]
            events.append(
                {
                    "method": method,
                    "dataset": METHOD_TO_DATASET[method],
                    "episode_id": episode_id,
                    "pair": pair,
                    "category": block[0]["category"],
                    "side": block[0]["side"],
                    "start_frame": int(block[0]["frame"]),
                    "end_frame": int(block[-1]["frame"]),
                    "duration_frames": len(block),
                    "duration_s": len(block) / fps,
                    "minimum_signed_distance_m": float(
                        min(row["signed_distance_m"] for row in block)
                    ),
                    "maximum_penetration_depth_m": float(
                        max(row["penetration_depth_m"] for row in block)
                    ),
                    "left_phases": sorted({row["left_phase"] for row in block}),
                    "right_phases": sorted({row["right_phase"] for row in block}),
                }
            )
            start = index
    return events


def collision_audit(
    runtime: Any,
    classifier: CollisionClassifier,
    output_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fps = 30.0
    for method in METHODS:
        dataset_name = METHOD_TO_DATASET[method]
        for episode_id in range(50):
            directory = V3_ROOT / dataset_name / f"episode_{episode_id:06d}"
            with np.load(directory / "g1_arm_action.npz", allow_pickle=False) as payload:
                arm = payload["action"].astype(np.float64)
                fps = float(payload["fps"])
            with np.load(directory / "g1_hand_action.npz", allow_pickle=False) as payload:
                left = payload["left_action"].astype(np.float64)
                right = payload["right_action"].astype(np.float64)
                left_phase = payload["left_phase"].astype(str)
                right_phase = payload["right_phase"].astype(str)
            for frame in range(len(arm)):
                runtime.assign(arm[frame], left[frame], right[frame])
                for record in classifier.records():
                    if not record.v1_gate_relevant:
                        continue
                    query_distance, witness, normal = _distance_witness(
                        runtime, record.geom_ids
                    )
                    rows.append(
                        {
                            "method": method,
                            "dataset": dataset_name,
                            "episode_id": episode_id,
                            "frame": frame,
                            "pair": record.pair,
                            "bodies": list(record.bodies),
                            "geom_ids": list(record.geom_ids),
                            "geom_names": list(record.geom_names),
                            "category": v4_category(record.bodies),
                            "side": pair_side(record.bodies),
                            "left_phase": str(left_phase[frame]),
                            "right_phase": str(right_phase[frame]),
                            "signed_distance_m": float(record.distance_m),
                            "distance_query_m": query_distance,
                            "penetration_depth_m": max(0.0, -float(record.distance_m)),
                            "witness_points_world_m": witness,
                            "witness_normal_world": normal,
                        }
                    )
    events = _runs(rows, fps)
    methods: dict[str, Any] = {}
    pair_csv: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        method_events = [row for row in events if row["method"] == method]
        frame_keys = {(row["episode_id"], row["frame"]) for row in selected}
        category_frames: dict[str, int] = {}
        category_episodes: dict[str, int] = {}
        category_events: dict[str, int] = {}
        for category in CATEGORIES:
            values = [row for row in selected if row["category"] == category]
            category_frames[category] = len(
                {(row["episode_id"], row["frame"]) for row in values}
            )
            category_episodes[category] = len({row["episode_id"] for row in values})
            category_events[category] = sum(
                row["category"] == category for row in method_events
            )
        pairs = sorted({row["pair"] for row in selected})
        ranked = []
        for pair in pairs:
            values = [row for row in selected if row["pair"] == pair]
            pair_events = [row for row in method_events if row["pair"] == pair]
            record = {
                "method": method,
                "dataset": METHOD_TO_DATASET[method],
                "pair": pair,
                "category": values[0]["category"],
                "side": values[0]["side"],
                "episode_count": len({row["episode_id"] for row in values}),
                "frame_count": len({(row["episode_id"], row["frame"]) for row in values}),
                "pair_event_count": len(values),
                "consecutive_run_count": len(pair_events),
                "maximum_run_frames": max(
                    (row["duration_frames"] for row in pair_events), default=0
                ),
                "minimum_signed_distance_m": min(
                    row["signed_distance_m"] for row in values
                ),
                "maximum_penetration_depth_m": max(
                    row["penetration_depth_m"] for row in values
                ),
                "mean_penetration_depth_m": float(
                    np.mean([row["penetration_depth_m"] for row in values])
                ),
                "left_phases": sorted({row["left_phase"] for row in values}),
                "right_phases": sorted({row["right_phase"] for row in values}),
            }
            ranked.append(record)
            pair_csv.append({
                **record,
                "left_phases": ";".join(record["left_phases"]),
                "right_phases": ";".join(record["right_phases"]),
            })
        ranked.sort(key=lambda row: (-row["frame_count"], row["pair"]))
        methods[method] = {
            "dataset": METHOD_TO_DATASET[method],
            "prohibited_collision_frames": len(frame_keys),
            "prohibited_pair_frame_events": len(selected),
            "affected_episode_count": len({row["episode_id"] for row in selected}),
            "affected_episode_ids": sorted({row["episode_id"] for row in selected}),
            "category_frame_incidence": category_frames,
            "category_episode_incidence": category_episodes,
            "category_consecutive_runs": category_events,
            "penetration_depth_m": stable_stats(
                row["penetration_depth_m"] for row in selected
            ),
            "run_duration_frames": stable_stats(
                row["duration_frames"] for row in method_events
            ),
            "ranked_pairs": ranked,
        }
    combined_category = Counter()
    for category in CATEGORIES:
        combined_category[category] = len(
            {
                (row["method"], row["episode_id"], row["frame"])
                for row in rows
                if row["category"] == category
            }
        )
    dominant = max(CATEGORIES, key=lambda key: (combined_category[key], key))
    output = {
        "schema_version": "common_collision_v4_preoptimization_audit",
        "source": str(V3_ROOT),
        "collision_gate_semantics": "byte-identical frozen Feasibility-v3 classifier",
        "signed_distance_convention": "negative metres indicate penetration",
        "witness_source": "mujoco.mj_geomDistance; contact distance retained as authoritative gate value",
        "total_prohibited_collision_frames": len(
            {(row["method"], row["episode_id"], row["frame"]) for row in rows}
        ),
        "total_pair_frame_events": len(rows),
        "dominant_category": dominant,
        "combined_category_frame_incidence": dict(combined_category),
        "methods": methods,
        "events": events,
        "pair_frame_records": rows,
    }
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(audit_dir / "collision_events_v3.json", output)
    atomic_csv(audit_dir / "collision_pair_distribution_v3.csv", pair_csv)
    report_rows = []
    for category in CATEGORIES:
        report_rows.append(
            "| "
            + category
            + " | "
            + str(methods["baseline"]["category_frame_incidence"][category])
            + " | "
            + str(methods["proposed"]["category_frame_incidence"][category])
            + " | "
            + str(combined_category[category])
            + " |"
        )
    markdown = f"""# Feasibility-v3 remaining collision audit

Optimization 전에 동결한 감사 결과다. Gate 의미론과 penetration tolerance는 v3에서 변경하지 않았다.

- Dominant category: `{dominant}`
- Dataset A prohibited frames: `{methods['baseline']['prohibited_collision_frames']}`
- Dataset B prohibited frames: `{methods['proposed']['prohibited_collision_frames']}`
- Pair-frame events: `{len(rows)}`

| Category | Dataset A frames | Dataset B frames | Combined frames |
|---|---:|---:|---:|
""" + "\n".join(report_rows) + "\n"
    (audit_dir / "collision_audit_report.md").write_text(markdown, encoding="utf-8")
    return output


__all__ = ["CATEGORIES", "v4_category", "pair_side", "collision_audit"]
