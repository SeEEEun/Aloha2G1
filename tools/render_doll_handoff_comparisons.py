#!/usr/bin/env python3
"""Render synchronized Doll-Handoff A/B visual-review videos."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    OUTPUT,
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.doll_handoff_retargeting.render import render_many  # noqa: E402


def _parse(specification: str, common: dict, output: Path) -> list[int]:
    if specification == "all":
        return list(range(int(common["expected_source_count"])))
    if specification == "smoke":
        return list(map(int, common["render"]["smoke_episode_indices"]))
    if specification == "representative":
        return list(map(int, common["render"]["representative_episode_indices"]))
    if specification == "failures":
        result: set[int] = set()
        for method in ("baseline", "proposed"):
            for path in (output / method / "metrics").glob(
                "doll_handoff_20260820_ep[0-9][0-9][0-9].json"
            ):
                import json

                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("status") != "PASS":
                    result.add(int(value["episode_index"]))
        return sorted(result)
    values: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = map(int, token.split("-", 1))
            values.update(range(first, last + 1))
        else:
            values.add(int(token))
    indices = sorted(values)
    if not indices or indices[0] < 0 or indices[-1] >= int(common["expected_source_count"]):
        raise ValueError("invalid episode specification")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        default="all",
        help="all, smoke, representative, failures, or comma/range indices",
    )
    parser.add_argument(
        "--no-smoke-top",
        action="store_true",
        help="skip the additional top-camera videos for smoke indices",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT,
        help="retargeting output root containing source/event/config/A/B artifacts",
    )
    parser.add_argument(
        "--views",
        default=None,
        help="optional comma-separated camera override, e.g. overview,top,side",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    resolved_common = output / "config/common_config.json"
    common = (
        load_common_config(resolved_common)
        if resolved_common.is_file()
        else load_common_config()
    )
    layout = load_scene(common)
    indices = _parse(args.episodes, common, output)
    cameras = (
        tuple(value.strip() for value in args.views.split(",") if value.strip())
        if args.views
        else None
    )
    if cameras is not None:
        missing = sorted(set(cameras) - set(layout["camera"]["presets"]))
        if missing:
            raise ValueError(f"unknown camera presets: {missing}")
    g1 = G1Kinematics(common, layout)
    manifest = render_many(
        common,
        layout,
        g1,
        indices,
        output,
        top_for_smoke=not args.no_smoke_top,
        camera_override=cameras,
    )
    print(
        "VISUAL_REVIEW_RENDER_COMPLETE "
        f"episodes={manifest['rendered_episode_count']} "
        f"failure_videos={manifest['failure_videos_present_count']} "
        f"manifest={output / 'comparison/visual_review_manifest.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
