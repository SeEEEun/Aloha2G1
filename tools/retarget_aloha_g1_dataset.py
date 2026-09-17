#!/usr/bin/env python3
"""Offline batch converter for the ALOHA -> G1 retargeting v1 audit."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from aloha_g1_dataset_v1.artifacts import write_discovery_artifacts  # noqa: E402
from aloha_g1_dataset_v1.core import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    RetargetingPipeline,
    write_failure_episode,
    write_method_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert authoritative ALOHA demonstrations to fixed-base G1/Dex3 "
            "training-label candidates without physics, training, or robot commands."
        )
    )
    parser.add_argument("--method", choices=("baseline", "proposed"), required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", type=int, help="single source episode ID")
    selection.add_argument("--episodes", choices=("all",), help="convert the complete source set")
    parser.add_argument("--dry-run", action="store_true", help="audit inputs and print the plan; write nothing")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="require the emitted validation record (validation is always computed)",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args()


def exception_status(exc: BaseException) -> str:
    if isinstance(exc, (FileNotFoundError, KeyError, ValueError)):
        return "FAIL_DATA"
    return "FAIL_OTHER"


def main() -> int:
    args = parse_args()
    pipeline = RetargetingPipeline(args.config, args.dataset_root)
    episode_ids = (
        pipeline.dataset.episode_ids() if args.episodes == "all" else [int(args.episode)]
    )
    invalid = [value for value in episode_ids if value not in pipeline.dataset.episode_ids()]
    if invalid:
        raise SystemExit(f"invalid episode IDs: {invalid}")
    plan = {
        "offline_only": True,
        "method": args.method,
        "episode_ids": episode_ids,
        "episode_count": len(episode_ids),
        "source_dataset": str(pipeline.dataset.root),
        "config": str(pipeline.config_path),
        "output_root": str(args.output_root.resolve()),
        "physics": False,
        "training": False,
        "real_robot_commands": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    write_discovery_artifacts(pipeline, args.output_root)
    counts: dict[str, int] = {}
    for offset, episode_id in enumerate(episode_ids, start=1):
        try:
            result = pipeline.convert(args.method, episode_id)
            directory = pipeline.export(result, args.output_root)
            status = result.validation["status"]
            counts[status] = counts.get(status, 0) + 1
            print(
                f"[{offset:02d}/{len(episode_ids):02d}] {args.method} "
                f"episode={episode_id:06d} status={status} "
                f"ik={result.metrics['ik_success_rate']:.4f} output={directory}",
                flush=True,
            )
        except Exception as exc:  # each source episode must receive an explicit outcome
            status = exception_status(exc)
            reason = f"{type(exc).__name__}: {exc}"
            write_failure_episode(
                args.output_root, args.method, episode_id, status, reason, args.config
            )
            counts[status] = counts.get(status, 0) + 1
            print(
                f"[{offset:02d}/{len(episode_ids):02d}] {args.method} "
                f"episode={episode_id:06d} status={status} reason={reason}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

    if args.episodes == "all":
        summary = write_method_summary(args.output_root, args.method)
        print(json.dumps({
            "method": args.method,
            "episode_count": summary["episode_count"],
            "pass_count": summary["pass_count"],
            "failure_breakdown": summary["failure_breakdown"],
        }, indent=2))
    elif args.validate:
        validation_path = (
            args.output_root.resolve()
            / args.method
            / f"episode_{episode_ids[0]:06d}"
            / "validation.json"
        )
        if not validation_path.is_file():
            raise RuntimeError(f"validation was not emitted: {validation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

