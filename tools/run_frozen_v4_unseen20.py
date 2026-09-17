#!/usr/bin/env python3
"""Run the frozen collision-v4 converter on the predeclared unseen 20 set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloha_g1_unseen_20_v4.constants import OUTPUT_ROOT
from aloha_g1_unseen_20_v4.pipeline import (
    anti_overfitting_scan,
    deterministic_rerun,
    run_retargeting,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("retarget", "summarize", "determinism", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    result: dict[str, object] = {}
    rows = None
    if args.mode in ("retarget", "all"):
        rows = run_retargeting(output_root)
        result["retargeted"] = {
            method: len(values) for method, values in rows.items()
        }
    if args.mode in ("summarize", "all"):
        summary = summarize(rows, output_root)
        result["combined"] = summary["combined"]
        result["conclusion"] = summary["readiness"]["conclusion"]
        result["anti_overfitting"] = anti_overfitting_scan(output_root)["pass"]
    if args.mode == "determinism":
        result["determinism"] = deterministic_rerun(output_root)["pass"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
