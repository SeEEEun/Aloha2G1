#!/usr/bin/env python3
"""Run the offline doll-handoff A/B retargeting audit.

This entry point intentionally has no dataset-packaging, training, physics-tuning,
or real-robot mode.  Both methods instantiate the same pipeline and solver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import METHODS  # noqa: E402
from tools.doll_handoff_retargeting.pipeline import DollHandoffPipeline  # noqa: E402


def _episode_indices(specification: str, count: int, smoke: list[int]) -> list[int]:
    if specification == "all":
        return list(range(count))
    if specification == "smoke":
        return smoke
    values: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first > last:
                raise ValueError(f"descending episode range is not allowed: {token}")
            values.update(range(first, last + 1))
        else:
            values.add(int(token))
    result = sorted(values)
    if not result or result[0] < 0 or result[-1] >= count:
        raise ValueError(f"episode specification must resolve inside [0,{count - 1}]")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and retarget the 50 Doll-Handoff recordings (offline only)."
    )
    parser.add_argument(
        "--method",
        choices=[*METHODS, "both"],
        required=True,
        help="A/B method; 'both' runs baseline then proposed with the same pipeline.",
    )
    parser.add_argument(
        "--episodes",
        default="all",
        help="'all', 'smoke', or a comma/range expression such as 0,24,49.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY / "outputs/doll_handoff_retargeting",
        help="isolated output root; existing artifacts outside it are untouched",
    )
    parser.add_argument(
        "--common-config",
        type=Path,
        default=None,
        help="optional frozen common config; enables an exact source-selection manifest",
    )
    parser.add_argument(
        "--episode-registration-manifest",
        type=Path,
        default=None,
        help="common source-derived episode registration bound before the representation switch",
    )
    args = parser.parse_args()

    kwargs = {"output_root": args.output_root.resolve()}
    if args.common_config is not None:
        kwargs["common_path"] = args.common_config.resolve()
    if args.episode_registration_manifest is not None:
        kwargs["registration_path"] = args.episode_registration_manifest.resolve()
    pipeline = DollHandoffPipeline(**kwargs)
    count = int(pipeline.source_manifest["enumerated_count"])
    indices = _episode_indices(
        args.episodes,
        count,
        list(map(int, pipeline.common["render"]["smoke_episode_indices"])),
    )
    methods = METHODS if args.method == "both" else (args.method,)
    print(
        "DOLL_HANDOFF_RETARGETING_START "
        f"methods={','.join(methods)} episodes={','.join(map(str, indices))} "
        f"scene={pipeline.common['scene_config']} scale=1.0 "
        "dataset_packaging=DISABLED policy_training=DISABLED",
        flush=True,
    )
    summaries = {method: pipeline.run(method, indices) for method in methods}
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    print("DOLL_HANDOFF_RETARGETING_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
