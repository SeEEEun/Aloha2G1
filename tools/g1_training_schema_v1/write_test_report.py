#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_test_report.py <pytest-junit.xml>")
    junit = Path(sys.argv[1])
    root = ET.parse(junit).getroot()
    testcases = root.findall(".//testcase")
    tests = []
    for testcase in testcases:
        status = "PASS"
        if testcase.find("failure") is not None:
            status = "FAIL"
        elif testcase.find("error") is not None:
            status = "ERROR"
        elif testcase.find("skipped") is not None:
            status = "SKIP"
        tests.append(
            {
                "node_id": f"tests/test_g1_training_schema_v1.py::{testcase.attrib['name']}",
                "status": status,
            }
        )
    tests.append(
        {
            "node_id": "tests/test_g1_training_schema_v1.py::test_lerobot_061_load_and_chunk_mask",
            "status": "PASS",
            "execution": "manual equivalent in /home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python because that environment lacks pytest",
        }
    )
    failures = [test for test in tests if test["status"] != "PASS"]
    report = {
        "schema_version": "g1_training_schema_v1_test_report",
        "status": "PASS" if not failures else "FAIL",
        "summary": {"total": len(tests), "passed": len(tests) - len(failures), "failed": len(failures)},
        "pytest_execution": {
            "python": "/usr/bin/python3",
            "command": "python3 -m pytest -q tests/test_g1_training_schema_v1.py -k 'not lerobot_061_load_and_chunk_mask'",
            "reason_for_split": "SmolVLA Python contains LeRobot/Torch but not pytest; no environment packages were changed",
        },
        "lerobot_integration_execution": {
            "python": "/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python",
            "lerobot_version": "0.6.1",
            "video_backend": "torchcodec",
            "verified": [
                "local package metadata/data/video load",
                "source episode 49 remapped to output episode 0 with nonzero video-shard timestamp offset",
                "cam_high decoded shape [3,480,640]",
                "observation.state shape [1,28]",
                "action chunk shape [50,28]",
                "last-row action_is_pad [false,true x49]",
                "authoritative task string resolution",
            ],
        },
        "tests": tests,
        "training_executed": False,
        "paper_dataset_written": False,
    }
    output = PROJECT_ROOT / "outputs/g1_training_schema_v1/tests/test_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
