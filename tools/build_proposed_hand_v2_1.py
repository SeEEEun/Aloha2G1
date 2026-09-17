#!/usr/bin/env python3
"""Build and audit the offline Proposed/Dataset-B semantic hand v2.1 candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from aloha_g1_hand_v2_1.common import V2_1_ROOT  # noqa: E402
from aloha_g1_hand_v2.common import json_default  # noqa: E402
from aloha_g1_hand_v2_1.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("proposed",), default="proposed")
    parser.add_argument(
        "--hand-mapper",
        choices=("feasible_semantic_primitive",),
        default="feasible_semantic_primitive",
    )
    parser.add_argument("--output-root", type=Path, default=V2_1_ROOT)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        args.output_root,
        method=args.method,
        hand_mapper=args.hand_mapper,
        run_tests=not args.skip_tests,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
    return 0 if result["tests_pass"] or args.skip_tests else 1


if __name__ == "__main__":
    raise SystemExit(main())
