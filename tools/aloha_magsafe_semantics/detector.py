"""Public semantic detector entry point.

The implementation is completed by the candidate/decoder modules in this
package.  The signature deliberately has no reference-timeline argument.
"""
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
        action=action,
        timestamps=timestamps,
        fk_trajectory=fk_trajectory,
        task_geometry=task_geometry,
        detector_config=detector_config,
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
