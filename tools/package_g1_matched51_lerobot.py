#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.g1_policy_dataset_packaging_v1.packager import (
    load_matched_pool,
    package_matched_pair,
    validate_matched_pair,
    write_pairing_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the frozen matched A/B pool as paired LeRobot datasets")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-a", required=True)
    parser.add_argument("--output-b", required=True)
    parser.add_argument("--pairing-csv")
    parser.add_argument("--video-storage", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = load_matched_pool(args.manifest, args.project_root)
    if args.pairing_csv:
        write_pairing_csv(pool, args.pairing_csv)
    if args.validate_only:
        result = validate_matched_pair(pool, args.output_a, args.output_b)
    else:
        result = package_matched_pair(pool, args.output_a, args.output_b, args.video_storage)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
