#!/usr/bin/env python3
"""Build final Doll-Handoff A/B reports and enforce review readiness."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.report import final_console_report  # noqa: E402


def main() -> int:
    try:
        ready, _ = final_console_report()
        return 0 if ready else 2
    except Exception as error:
        traceback.print_exc()
        print()
        print("BLOCKERS")
        print(f"- report generation failed: {type(error).__name__}: {error}")
        print("BLOCKED_DOLL_HANDOFF_RETARGETING")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
