#!/usr/bin/env python3
"""Bind the pre-result common execution implementation to a frozen evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_execution_layer import load_frozen_evaluator, sha256_file
from tools.run_final_common_execution_eval35 import (
    ENVIRONMENT_FREEZE,
    EVALUATOR_FREEZE,
    EXECUTION_FREEZE,
    OUT,
    verify_eval35_identity,
    verify_hash_manifest,
)


MD = EXECUTION_FREEZE.with_suffix(".md")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def implementation_bundle(manifest: dict[str, Any]) -> str:
    import hashlib

    rows = []
    for row in manifest["implementation_files"]:
        path = Path(row["path"]).resolve()
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"implementation hash drift before freeze: {path}")
        rows.append(f"{path}:{actual}")
    value = hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
    if value != manifest["execution_layer_sha256"]:
        raise RuntimeError("execution-layer bundle hash mismatch")
    return value


def run_tests() -> None:
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    process = subprocess.run(
        [
            "/home/jbnu/miniconda3/envs/isaaclab6/bin/python",
            "-m",
            "pytest",
            "-q",
            "tests/test_common_execution_layer.py",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"common execution tests failed:\n{process.stdout}{process.stderr}")
    compile_check = subprocess.run(
        [
            "/usr/bin/python3",
            "-m",
            "py_compile",
            "tools/prepare_eval35_manifest.py",
            "tools/convert_eval35_new25_frozen_ab.py",
            "tools/precompute_eval35_new25_act.py",
            "tools/prepare_eval35_physical_commands.py",
            "tools/prepare_eval35_ab_artifacts.py",
            "tools/run_final_common_execution_eval35.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if compile_check.returncode != 0:
        raise RuntimeError(
            f"EVAL35 preparation syntax validation failed:\n"
            f"{compile_check.stdout}{compile_check.stderr}"
        )
    patch = subprocess.run(
        [
            "/usr/bin/python3",
            "tools/run_common_execution_layer_isaac.py",
            "--validate-patch-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if patch.returncode != 0:
        raise RuntimeError(f"runtime instrumentation validation failed:\n{patch.stdout}{patch.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    manifest = read_json(EXECUTION_FREEZE)
    bundle = implementation_bundle(manifest)
    run_tests()
    verify_eval35_identity()
    if not EVALUATOR_FREEZE.is_file():
        print(
            json.dumps(
                {
                    "status": "WAITING_FOR_FROZEN_PHYSICAL_EVALUATOR",
                    "execution_layer_sha256": bundle,
                    "missing": str(EVALUATOR_FREEZE),
                },
                indent=2,
            )
        )
        return 3
    evaluator_manifest = read_json(EVALUATOR_FREEZE)
    if evaluator_manifest.get("status") != "FROZEN":
        print(
            json.dumps(
                {
                    "status": "ACT_AB_PHYSICAL_EVALUATION_BLOCKED",
                    "reason": "authoritative evaluator is not FROZEN",
                    "evaluator_status": evaluator_manifest.get("status"),
                    "validation_status": evaluator_manifest.get("validation_status"),
                    "execution_layer_sha256": bundle,
                    "evaluator_manifest_sha256": sha256_file(EVALUATOR_FREEZE),
                },
                indent=2,
            )
        )
        return 4
    evaluator = load_frozen_evaluator(EVALUATOR_FREEZE)
    verify_hash_manifest(ENVIRONMENT_FREEZE)
    if not args.freeze:
        print(
            json.dumps(
                {
                    "status": "READY_TO_FREEZE",
                    "execution_layer_sha256": bundle,
                    "frozen_evaluator_sha256": evaluator.evaluator_sha256,
                },
                indent=2,
            )
        )
        return 0
    if manifest.get("status") == "FROZEN":
        if manifest.get("frozen_evaluator_sha256") != evaluator.evaluator_sha256:
            raise RuntimeError("execution layer already frozen to a different evaluator")
        print(json.dumps({"status": "ALREADY_FROZEN"}, indent=2))
        return 0
    manifest.update(
        {
            "status": "FROZEN",
            "frozen": True,
            "final_eval35_execution_permitted": True,
            "eval35_ab_preparation_status": "READY_NOT_STARTED",
            "frozen_evaluator_manifest_sha256": evaluator.manifest_sha256,
            "frozen_evaluator_sha256": evaluator.evaluator_sha256,
            "evaluator_observed_status": evaluator_manifest.get("status"),
            "evaluator_validation_status": evaluator_manifest.get("validation_status"),
            "evaluator_envelope_status": evaluator_manifest.get("envelope_status"),
            "evaluator_validation_reason": None,
            "physical_environment_integrity": {
                **manifest["physical_environment_integrity"],
                "status": "PASS",
                "drifted_file": None,
                "declared_sha256": None,
                "actual_sha256": None,
            },
            "pending_before_freeze": [],
        }
    )
    atomic_json(EXECUTION_FREEZE, manifest)
    MD.write_text(
        "\n".join(
            [
                "# Common execution layer freeze manifest",
                "",
                "Status: **FROZEN BEFORE FINAL A/B PHYSICAL RESULTS**",
                "",
                f"Implementation bundle SHA256: `{bundle}`",
                f"Frozen evaluator SHA256: `{evaluator.evaluator_sha256}`",
                f"Frozen evaluator manifest SHA256: `{evaluator.manifest_sha256}`",
                f"EVAL35 identity manifest SHA256: `{sha256_file(OUT / 'EVAL35_MANIFEST.json')}`",
                "",
                "ARM and WRIST common overrides are prohibited. Only the common Dex3 ",
                "contact primitives may intervene after their physical gates. No post-result ",
                "tuning is permitted. EVAL35 is the unchanged prior EVAL10 plus all 25 ",
                "sorted GoPark_20260902_* recordings, which remain evaluation-only. ",
                "No physical rollout has started at execution freeze time (0/70).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "execution_layer_sha256": bundle,
                "frozen_evaluator_sha256": evaluator.evaluator_sha256,
                "freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
