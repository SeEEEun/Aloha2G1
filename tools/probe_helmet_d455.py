#!/usr/bin/env python3
"""Read-only Intel RealSense D455 inventory and stream-profile probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from helmet_d455.calibration import atomic_json
from helmet_d455.realsense import enumerate_devices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="require this device serial")
    parser.add_argument("--output", type=Path, help="optional JSON report")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="return success when no D455 is attached (useful before mounting)",
    )
    args = parser.parse_args()

    try:
        devices = enumerate_devices()
    except RuntimeError as error:
        report = {
            "schema_version": "helmet_d455_probe_v1",
            "status": "PYREALSENSE2_UNAVAILABLE",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
            "devices": [],
        }
        if args.output:
            atomic_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0 if args.allow_missing else 2

    d455 = [row for row in devices if "D455" in row.get("name", "").upper()]
    if args.serial:
        d455 = [row for row in d455 if row.get("serial") == args.serial]
    status = "D455_DETECTED" if d455 else "NO_D455_CONNECTED_CAMERA_NOT_MOUNTED"
    report = {
        "schema_version": "helmet_d455_probe_v1",
        "status": status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_serial": args.serial,
        "d455_devices": d455,
        "all_realsense_devices": devices,
        "read_only": True,
    }
    if args.output:
        atomic_json(args.output, report)
    if args.output:
        print(json.dumps({"status": status, "d455_count": len(d455), "report": str(args.output.resolve())}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0 if d455 or args.allow_missing else 2


if __name__ == "__main__":
    sys.exit(main())
