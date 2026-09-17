#!/usr/bin/env python3
"""Build the predeclared unseen-20 source dataset with the original v3 encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloha_g1_unseen_20_v4.constants import OUTPUT_ROOT
from aloha_g1_unseen_20_v4.source import (
    audit_raw_sources,
    build_integrated_dataset,
    compare_source_schema,
    create_layout_contact_sheets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate/hash raw inputs and create contact sheets without encoding",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest, records = audit_raw_sources(output_root)
    layout = create_layout_contact_sheets(records, output_root)
    result = {
        "raw_manifest_status": manifest["status"],
        "source_pass_count": manifest["source_pass_count"],
        "total_frames": manifest["total_frames"],
        "layout_contact_sheets": len(layout["contact_sheets"]),
    }
    if not args.audit_only:
        result["build"] = build_integrated_dataset(records, output_root)["status"]
        result["schema"] = compare_source_schema(output_root)["status"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
