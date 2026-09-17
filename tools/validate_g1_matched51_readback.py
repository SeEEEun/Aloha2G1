#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tools.g1_policy_dataset_packaging_v1.readback import run_lerobot_readback


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate matched A/B datasets through LeRobot 0.6.1")
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_lerobot_readback(args.dataset_a, args.dataset_b, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
