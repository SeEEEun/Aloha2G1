#!/usr/bin/env python3
"""Run the synthetic paper metric/event tests and write a JSON result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import unittest


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation.io import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/paper_metrics/validation/unit_tests.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), pattern="test_paper_evaluation_metrics.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "schema_version": "paper_evaluation_unit_test_result_v1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "failures": [str(test) for test, _ in result.failures],
        "errors": [str(test) for test, _ in result.errors],
        "test_pattern": "tests/test_paper_evaluation_metrics.py",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_or_isaac_test_executed": False,
    }
    atomic_json(args.output, payload)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
