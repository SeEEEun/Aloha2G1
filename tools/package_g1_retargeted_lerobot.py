#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.g1_training_schema_v1.episode_filter import (  # noqa: E402
    audit_episode_root,
    intersection_manifest,
)
from tools.g1_training_schema_v1.lerobot_writer import (  # noqa: E402
    inspect_packaging_inputs,
    package_dataset,
)
from tools.g1_training_schema_v1.validator import validate_packaged_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package accepted retargeted G1 trajectories as a common-schema LeRobot v3 dataset."
    )
    parser.add_argument("--method", required=True, choices=["dataset_a", "dataset_b"])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="audit only; never write a dataset")
    parser.add_argument("--validate-only", action="store_true", help="validate existing output, or inputs if absent")
    parser.add_argument("--episode-mode", choices=["native", "matched"], default="native")
    parser.add_argument("--matched-with", type=Path, help="the other method input root for matched mode")
    parser.add_argument("--intersection-manifest", type=Path)
    parser.add_argument("--video-storage", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument(
        "--failure-policy",
        choices=["reject_failed", "include_with_mask", "truncate_before_failure"],
        default="reject_failed",
        help="v1 implements reject_failed; the other explicit policies fail fast",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.episode_mode == "matched" and args.matched_with is None:
        raise SystemExit("--episode-mode matched requires --matched-with")
    if args.intersection_manifest is not None:
        if args.matched_with is None:
            raise SystemExit("--intersection-manifest requires --matched-with")
        _, decisions = audit_episode_root(args.input_root)
        _, other = audit_episode_root(args.matched_with)
        value = intersection_manifest(decisions, other)
        value["first_argument_method"] = args.method
        write_json(args.intersection_manifest, value)

    if args.validate_only and args.output_root.is_dir():
        result = validate_packaged_dataset(args.output_root, args.source_dataset)
    elif args.dry_run or args.validate_only:
        result = inspect_packaging_inputs(
            args.method,
            args.input_root,
            args.source_dataset,
            args.episode_mode,
            args.matched_with,
            args.failure_policy,
        )
        result["markers"] = ["STRUCTURAL_DRY_RUN_ONLY", "NOT_ACCEPTED_TRAINING_DATA"]
        result["dataset_written"] = False
        result["normalization_computed"] = False
    else:
        result = package_dataset(
            method=args.method,
            input_root=args.input_root,
            source_dataset=args.source_dataset,
            output_root=args.output_root,
            episode_mode=args.episode_mode,
            matched_with=args.matched_with,
            video_storage=args.video_storage,
            failure_policy=args.failure_policy,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
