"""Public v2 semantic detector with the canonical v1 call signature."""
from __future__ import annotations

from typing import Any


def detect_magsafe_semantics(
    action: Any,
    timestamps: Any,
    source_type: str,
    fk_trajectory: dict[str, Any],
    task_geometry: dict[str, Any],
    detector_config: dict[str, Any],
    observation_state: Any = None,
    observation_alignment: dict[str, Any] | None = None,
):
    from .candidate_detection import extract_event_candidates
    from .sequence_decoder import decode_semantic_sequence

    candidates, context = extract_event_candidates(
        action, timestamps, fk_trajectory, task_geometry, detector_config,
    )
    return decode_semantic_sequence(
        action=action,
        timestamps=timestamps,
        source_type=source_type,
        fk_trajectory=fk_trajectory,
        task_geometry=task_geometry,
        detector_config=detector_config,
        candidates=candidates,
        context=context,
        observation_state=observation_state,
        observation_alignment=observation_alignment,
    )
