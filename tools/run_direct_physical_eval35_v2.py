#!/usr/bin/env python3
"""Resume frozen EVAL35 with additive measured-state scoring correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import run_direct_physical_eval35 as physical


ROOT = physical.ROOT
ISAAC = physical.ISAAC
RECLASSIFIER = ROOT / "tools/reclassify_direct_physical_eval35.py"
CORRECTION = (
    physical.OUT / "00_freeze/POST_START_SCORING_HARNESS_CORRECTION.json"
)
RUNS = physical.RUNS
STATUS = physical.STATUS


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return physical.sha256_file(path)


def atomic_json(path: Path, value: Any) -> None:
    physical.atomic_json(path, value)


def verify_correction() -> dict[str, Any]:
    value = read_json(CORRECTION)
    if value.get("status") != "FROZEN_BEFORE_REMAINING_69_ROLLOUTS":
        raise RuntimeError("scoring-harness correction is not frozen")
    if value.get("physical_execution_changed") is not False:
        raise RuntimeError("scoring correction altered physical execution")
    for row in value.get("frozen_files", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"scoring-correction dependency drift: {path}")
    return value


def completed_count() -> int:
    return len(list(RUNS.glob("act_*40/eval_*/RUN_MANIFEST_V2.json")))


def update_status(last: str) -> None:
    STATUS.write_text(
        "# Final direct physical EVAL35 status\n\n"
        f"- Completed physically executed and measured-state-scored rollouts: **{completed_count()}/70**\n"
        f"- Last persisted result: `{last}`\n"
        f"- Original physical execution freeze: `{physical.FREEZE}`\n"
        "- Post-start scoring correction: policy target-limit excess is diagnostic; measured PhysX state determines invalid robot state.\n"
        "- No classifier, atlas, wrist-distance gate, arm rescue, or wrist rescue.\n",
        encoding="utf-8",
    )


def run_one(method: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    output = RUNS / f"act_{method.lower()}40/eval_{index:02d}_{row['stable_episode_id']}"
    complete = output / "RUN_MANIFEST_V2.json"
    if complete.is_file():
        cached = read_json(complete)
        if cached.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL", "INVALID_PHYSICS"}:
            raise RuntimeError(f"invalid corrected cache: {complete}")
        return cached
    try:
        original = physical.run_one(method, index, row)
    except RuntimeError as exc:
        # The original scorer may stop on its target-limit invalidity.  It has
        # already persisted the complete physical trace and original manifest.
        if "invalid physics" not in str(exc):
            raise
        original_path = output / "RUN_MANIFEST.json"
        if not original_path.is_file():
            raise
        original = read_json(original_path)
    scored = subprocess.run(
        [str(ISAAC), str(RECLASSIFIER), "--run-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "reclassifier.log").write_text(
        scored.stdout + scored.stderr, encoding="utf-8"
    )
    result_path = output / "DIRECT_PHYSICAL_TASK_RESULT_V2.json"
    if scored.returncode not in (0, 3) or not result_path.is_file():
        raise RuntimeError(f"scoring-correction infrastructure failure: {output}")
    result = read_json(result_path)
    status = "INVALID_PHYSICS" if result["status"] == "INVALID" else (
        "PHYSICAL_PASS" if result["outcomes"]["FULL_TASK_SUCCESS"] else "PHYSICAL_FAIL"
    )
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in {
            "event_log.npz": output / "event_log.npz",
            "robot_bin_contacts.npz": output / "robot_bin_contacts.npz",
            "trial_result.json": output / "trial_result.json",
            "DIRECT_EXECUTION_RUNTIME_SUMMARY.json": output / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json",
            "DIRECT_PHYSICAL_TASK_RESULT.json": output / "DIRECT_PHYSICAL_TASK_RESULT.json",
            "DIRECT_PHYSICAL_TASK_RESULT_V2.json": result_path,
            "DIRECT_PHYSICAL_TASK_RESULT_V2.md": output / "DIRECT_PHYSICAL_TASK_RESULT_V2.md",
        }.items()
    }
    value = {
        "schema_version": "direct_physical_eval35_run_v2",
        "status": status,
        "method": f"ACT-{method}40",
        "eval_index": index,
        "stable_episode_id": row["stable_episode_id"],
        "provenance": row["provenance"],
        "outcomes": result["outcomes"],
        "first_failure_stage": result["first_failure_stage"],
        "release_classification": result["release_classification"],
        "fairness_audit": result["fairness_audit"],
        "diagnostics": result["diagnostics"],
        "physical_trace_reused": bool((output / "RUN_MANIFEST.json").is_file()),
        "physical_execution_freeze": str(physical.FREEZE.resolve()),
        "physical_execution_freeze_sha256": sha256_file(physical.FREEZE),
        "original_run_manifest": str((output / "RUN_MANIFEST.json").resolve()),
        "original_run_status": original["status"],
        "scoring_correction": (
            "target limit excess diagnostic; measured state governs invalidity"
        ),
        "artifacts": artifacts,
    }
    atomic_json(complete, value)
    update_status(str(complete))
    if status == "INVALID_PHYSICS":
        raise RuntimeError(f"invalid measured physics at {method}:{index}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("A", "B"))
    parser.add_argument("--eval-index", type=int, choices=range(35))
    args = parser.parse_args()
    _, records = physical.verify_freeze()
    verify_correction()
    physical.start_lock()
    methods = (args.method,) if args.method else ("A", "B")
    indices = (args.eval_index,) if args.eval_index is not None else tuple(range(35))
    for method in methods:
        for index in indices:
            result = run_one(method, index, records[(f"ACT-{method}40", index)])
            print(
                json.dumps(
                    {
                        "completed": completed_count(),
                        "method": method,
                        "eval_index": index,
                        "status": result["status"],
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
