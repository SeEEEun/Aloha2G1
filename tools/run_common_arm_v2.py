#!/usr/bin/env python3
"""Run the gated offline Common Arm-v2 and integrated A/B audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from aloha_g1_arm_v2.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/aloha_g1_arm_v2.json",
    )
    parser.add_argument(
        "--arm-output-root",
        type=Path,
        default=ROOT / "outputs/g1_dataset_retargeting_arm_v2",
    )
    parser.add_argument(
        "--integrated-output-root",
        type=Path,
        default=ROOT / "outputs/g1_dataset_retargeting_integrated_v2",
    )
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        config_path=args.config,
        arm_output_root=args.arm_output_root,
        integrated_output_root=args.integrated_output_root,
        run_tests=not args.skip_tests,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["stage_a_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
