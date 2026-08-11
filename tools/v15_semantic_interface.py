#!/usr/bin/env python3
"""Dry-run-only semantic input guard for future v15 orientation/Dex3 work."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_magsafe_semantics.io import atomic_json  # noqa: E402
from aloha_magsafe_semantics.schema import SemanticTimeline  # noqa: E402


PHASE_API_MAPPING = {
    "phone_acquisition_interval": ["left_phone_grasp_start", "phone_portrait_reached"],
    "portrait_rotation_interval": ["phone_rotation_to_portrait_start", "phone_portrait_reached"],
    "accessory_insertion_interval": ["right_accessory_grasp_start", "accessory_detachment_start"],
    "accessory_removal_interval": ["accessory_detachment_start", "accessory_removed"],
    "phone_transport_interval": ["phone_move_to_charger_start", "phone_charger_attachment_complete"],
    "left_release_endpoint": ["left_phone_release_complete"],
    "right_release_endpoint": ["right_accessory_release_complete"],
    "orientation_activation": ["phone_rotation_progress", "accessory_removal_progress"],
    "Dex3_phase_interpolation": ["phone_acquisition_progress", "accessory_acquisition_progress", "accessory_removal_progress"],
    "contact_keyframes": ["left_phone_grasp_start", "right_accessory_grasp_start", "phone_charger_attachment_complete"],
    "residual_knots": ["semantic_knots.numeric_knots"],
}


def readiness(timeline: SemanticTimeline) -> dict[str, object]:
    resolved = {}
    for phase, fields in PHASE_API_MAPPING.items():
        resolved[phase] = fields
    return {
        "status": "V15_SEMANTIC_INTERFACE_READY" if timeline.mandatory_complete() else "V15_BLOCKED_INCOMPLETE_SEMANTICS",
        "semantic_timeline_schema": timeline.schema_name,
        "phase_api_mapping": resolved,
        "absolute_semantic_indices_in_interface": False,
        "orientation_executed": False,
        "Dex3_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-timeline", type=Path, required=True)
    parser.add_argument("--semantic-phases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.semantic_timeline.read_text(encoding="utf-8"))
    with np.load(args.semantic_phases, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    timeline = SemanticTimeline.from_dict(payload, arrays)
    atomic_json(args.output, readiness(timeline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

