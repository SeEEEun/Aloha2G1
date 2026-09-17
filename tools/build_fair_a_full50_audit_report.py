#!/usr/bin/env python3
"""Build the independent Fair-A full-50 root-cause and parity report."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fair_a_full50_audit.report import run_report  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run_report(), indent=2))

