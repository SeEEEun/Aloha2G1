"""Semantic phase-knot construction without episode-specific indices."""
from __future__ import annotations

from typing import Any

from .event_names import REQUIRED_EVENTS
from .schema import SemanticTimeline


def build_semantic_knots(timeline: SemanticTimeline) -> dict[str, Any]:
    ordered: list[tuple[int, str]] = [(timeline.start_index, "trajectory_start")]
    for event_name in REQUIRED_EVENTS:
        record = timeline.event(event_name)
        if record.action_index is not None:
            ordered.append((int(record.action_index), event_name))
    ordered.append((timeline.end_index, "trajectory_end"))
    ordered.sort(key=lambda row: (row[0], row[1]))
    names_by_index: dict[int, list[str]] = {}
    for index, name in ordered:
        names_by_index.setdefault(index, []).append(name)
    unique_indices = sorted(names_by_index)
    if unique_indices[0] != 0 or unique_indices[-1] != timeline.trajectory_length - 1:
        raise RuntimeError("semantic knots must span the full trajectory")
    if any(right <= left for left, right in zip(unique_indices, unique_indices[1:])):
        raise RuntimeError("unique semantic knot array is not strictly increasing")
    return {
        "numeric_knots": unique_indices,
        "semantic_events_by_knot": {str(index): names_by_index[index] for index in unique_indices},
        "simultaneous_events_preserved": any(len(names) > 1 for names in names_by_index.values()),
        "trajectory_length": timeline.trajectory_length,
    }

