#!/usr/bin/env python3
"""Run the isolated unseen-20 integrity suite and persist an exact test report."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from aloha_g1_unseen_20_v4.common import atomic_json
from aloha_g1_unseen_20_v4.constants import OUTPUT_ROOT, ROOT


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_aloha_g1_collision_v4.py",
        "tests/test_aloha_g1_unseen_20_v4.py",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    report = {
        "schema_version": "unseen_20_v4_test_report_v1",
        "command": command,
        "command_string": "PYTHONPATH=tools " + " ".join(command),
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "passed": int(passed_match.group(1)) if passed_match else 0,
        "failed": int(failed_match.group(1)) if failed_match else (0 if result.returncode == 0 else 1),
        "stdout_stderr": output,
    }
    atomic_json(OUTPUT_ROOT / "tests/test_report.json", report)
    print(output)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
