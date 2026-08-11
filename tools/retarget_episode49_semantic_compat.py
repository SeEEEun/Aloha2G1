#!/usr/bin/env python3
"""Thin Episode-49 compatibility wrapper around the generic semantic API."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_magsafe_semantics import SemanticTimeline  # noqa: E402
from aloha_magsafe_semantics.io import load_trajectory  # noqa: E402
from retarget_aloha_trajectory_to_g1 import retarget_aloha_trajectory_to_g1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-action", type=Path, required=True)
    parser.add_argument("--semantic-timeline", type=Path, required=True,
                        help="Explicit semantic input; approved override is never implicit.")
    parser.add_argument("--semantic-phases", type=Path, required=True)
    parser.add_argument("--translator-config", type=Path, required=True)
    parser.add_argument("--target-scene-config", type=Path, required=True)
    args = parser.parse_args()
    source = load_trajectory(args.source_action, "optimized_action")
    timeline_payload = json.loads(args.semantic_timeline.read_text(encoding="utf-8"))
    with np.load(args.semantic_phases, allow_pickle=False) as archive:
        timeline = SemanticTimeline.from_dict(timeline_payload, {key: archive[key] for key in archive.files})
    result = retarget_aloha_trajectory_to_g1(
        source["action"], source["timestamps"], timeline,
        json.loads(args.translator_config.read_text(encoding="utf-8")),
        json.loads(args.target_scene_config.read_text(encoding="utf-8")),
        dry_run=True,
    )
    result["explicit_semantic_timeline_path"] = str(args.semantic_timeline.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

