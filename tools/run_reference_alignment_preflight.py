#!/usr/bin/env python3
"""Run one immutable reference-level contact-constrained alignment preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
MANIFEST = ROOT / "outputs/final_contact_constrained_eval/09_reference_alignment_preflight/COMMON_OBJECT_RELATIVE_COMMAND_MANIFEST_V8.json"
OUT = ROOT / "outputs/final_contact_constrained_eval/09_reference_alignment_preflight/runs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument("--eval-index", type=int, choices=range(10), required=True)
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    label = f"ACT-{args.method.upper()}40"
    matches = [
        row
        for row in data["records"]
        if row["method"] == label and int(row["eval_index"]) == args.eval_index
    ]
    if len(matches) != 1:
        raise RuntimeError("reference command lookup failed")
    row = matches[0]
    command = Path(row["common_command"])
    if sha256(command) != row["common_command_sha256"]:
        raise RuntimeError("common reference command hash drift")
    method_dir = "fair_a" if args.method == "a" else "proposed_b"
    # Earlier eval_00 directories include deliberately preserved infrastructure
    # failures and exploratory candidates.  Eligible fixed-registration runs use
    # an unambiguous prefix and never overwrite those provenance artifacts.
    output = OUT / method_dir / f"aligned_v8_eval_{args.eval_index:02d}"
    result = output / "trial_result.json"
    if result.is_file():
        trial = json.loads(result.read_text(encoding="utf-8"))
        if (
            trial.get("scripted_command_sha256") == sha256(command)
            and trial.get("object_task_frame_registration", {}).get("config_sha256")
            == sha256(REGISTRATION)
            and (output / "event_log.npz").is_file()
        ):
            print(json.dumps({"status": "CACHE_HIT", "output": str(output)}, indent=2))
            return 0
        raise RuntimeError(f"existing preflight identity mismatch: {output}")
    output.mkdir(parents=True, exist_ok=False)
    invocation = [
        str(ISAAC),
        str(ENGINE),
        "--config",
        str(CONFIG),
        "--side",
        "left",
        "--geometry",
        "FROZEN_COMPRESSED_SHORT_55",
        "--profile",
        "P14",
        "--output-dir",
        str(output),
        "--scripted-command-path",
        str(command),
        "--object-spawn-side",
        "left",
        "--object-registration-config",
        str(REGISTRATION),
        "--full-task-audit",
        "--bin-height-m",
        "0.150",
        "--bin-rim-bevel-m",
        "0.003",
        "--headless",
    ]
    (output / "INVOCATION_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "reference_alignment_preflight_invocation_v1",
                "method": label,
                "eval_index": args.eval_index,
                "command": str(command),
                "command_sha256": sha256(command),
                "registration": str(REGISTRATION),
                "registration_sha256": sha256(REGISTRATION),
                "engine": str(ENGINE),
                "engine_sha256": sha256(ENGINE),
                "no_policy_used": True,
                "arm_wrist_overrides": 0,
                "invocation": invocation,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output / "engine.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            invocation,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode not in (0, 2) or not result.is_file():
        raise RuntimeError(f"Isaac infrastructure failure ({completed.returncode}): {output}")
    print(json.dumps({"status": "PHYSICS_COMPLETE", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
