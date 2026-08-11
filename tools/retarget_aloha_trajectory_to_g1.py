#!/usr/bin/env python3
"""Semantic-timeline-first ALOHA to G1 converter contract.

This module only validates and materializes the generic interface in the
semantic-decoupling task.  It deliberately performs no IK or trajectory
generation when ``dry_run`` is true.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS
from aloha_magsafe_semantics.knots import build_semantic_knots
from aloha_magsafe_semantics.schema import SemanticTimeline


def retarget_aloha_trajectory_to_g1(
    source_action: np.ndarray,
    timestamps: np.ndarray,
    semantic_timeline: SemanticTimeline,
    frozen_translator_config: dict[str, Any],
    target_scene_config: dict[str, Any],
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    source_action = np.asarray(source_action)
    timestamps = np.asarray(timestamps)
    if source_action.ndim != 2 or source_action.shape[1] != 14:
        raise ValueError("source_action must be [T,14]")
    if timestamps.shape != (len(source_action),):
        raise ValueError("timestamp length mismatch")
    if semantic_timeline.trajectory_length != len(source_action):
        raise ValueError("semantic timeline trajectory length mismatch")
    missing = [name for name in REQUIRED_EVENTS if semantic_timeline.event(name).action_index is None]
    intervals = {
        "phone_acquisition": semantic_timeline.interval("left_phone_grasp_start", "phone_portrait_reached"),
        "accessory_removal": semantic_timeline.interval("accessory_detachment_start", "accessory_removed"),
        "phone_to_charger": semantic_timeline.interval("phone_move_to_charger_start", "phone_charger_attachment_complete"),
    } if not missing else {}
    contract = {
        "status": "GENERIC_SEMANTIC_INTERFACE_DRY_RUN_PASS" if dry_run and not missing else "SEMANTIC_TIMELINE_INCOMPLETE",
        "dry_run": dry_run,
        "trajectory_length": len(source_action),
        "events_requested_by_name": list(REQUIRED_EVENTS),
        "semantic_intervals": intervals,
        "semantic_knots": build_semantic_knots(semantic_timeline),
        "progress_arrays": {
            "phone_acquisition": list(semantic_timeline.progress("phone_acquisition").shape),
            "accessory_removal": list(semantic_timeline.progress("accessory_removal").shape),
            "phone_to_charger": list(semantic_timeline.progress("phone_to_charger").shape),
        },
        "frozen_translator_config_supplied": bool(frozen_translator_config),
        "target_scene_config_supplied": bool(target_scene_config),
        "missing_events": missing,
        "IK_executed": False,
        "Dex3_executed": False,
        "physics_executed": False,
    }
    if not dry_run:
        raise RuntimeError("trajectory generation is disabled in the semantic-decoupling task")
    return contract

