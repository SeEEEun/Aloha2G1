#!/usr/bin/env python3
"""Launch the frozen PhysX engine with a Dex3-only runtime control hook.

The previously frozen engine file is read, hash-checked, and instrumented in
memory.  It is never edited on disk.  Exact, count-checked source insertions keep
the environment/physics implementation separate from this execution layer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
ENVIRONMENT_FREEZE = (
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/FREEZE_MANIFEST.json"
)
DEFAULT_EVALUATOR = (
    ROOT
    / "outputs/final_representation_neutral_eval/00_frozen_evaluator/EVALUATOR_FREEZE_MANIFEST.json"
)
DEFAULT_EXECUTION_FREEZE = (
    ROOT
    / "outputs/final_representation_neutral_eval/06_common_execution_layer/COMMON_EXECUTION_LAYER_FREEZE_MANIFEST.json"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def declared_engine_hash() -> str:
    manifest = read_json(ENVIRONMENT_FREEZE)
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("contact-constrained environment is not FROZEN")
    rows = [
        row
        for row in manifest.get("files", [])
        if Path(row.get("path", "")).resolve() == ENGINE.resolve()
    ]
    if len(rows) != 1:
        raise RuntimeError("environment freeze does not identify exactly one engine")
    return str(rows[0]["sha256"])


def instrument(source: str) -> tuple[str, dict[str, int]]:
    substitutions: list[tuple[str, str, str]] = [
        (
            "build_runtime",
            '''    if commands.shape[1] != 28 or not np.isfinite(commands).all():\n        raise RuntimeError("invalid scripted commands")\n''',
            '''    if commands.shape[1] != 28 or not np.isfinite(commands).all():\n        raise RuntimeError("invalid scripted commands")\n    from tools.common_execution_isaac_runtime import build_runtime\n    common_execution = build_runtime(command_path, commands, names)\n''',
        ),
        (
            "add_trace_fields",
            '''    bin_contact_records: dict[str, list[Any]] = {\n''',
            '''    for common_field in common_execution.event_field_names:\n        if common_field in records:\n            raise RuntimeError(f"common execution trace field collision: {common_field}")\n        records[common_field] = []\n    bin_contact_records: dict[str, list[Any]] = {\n''',
        ),
        (
            "causal_step",
            '''    for control_frame, (command, label) in enumerate(zip(commands, stages, strict=True)):\n        if (\n''',
            '''    for control_frame, (command, label) in enumerate(zip(commands, stages, strict=True)):\n        common_snapshot = common_execution.snapshot(\n            measured_q_rad=numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64),\n            object_pose_xyzw=numpy(doll.data.root_pose_w)[0].astype(np.float64),\n            body_names=body_names,\n            body_positions_world_m=numpy(robot.data.body_pos_w)[0].astype(np.float64),\n            body_quaternions_xyzw=numpy(robot.data.body_quat_w)[0].astype(np.float64),\n            records=records,\n        )\n        command = common_execution.step(control_frame, common_snapshot)\n        if (\n''',
        ),
        (
            "event_values",
            '''            for key, value in values.items():\n                records[key].append(value)\n''',
            '''            values.update(common_execution.event_values(measured))\n            for key, value in values.items():\n                records[key].append(value)\n''',
        ),
        (
            "runtime_summary_file",
            '''    os.replace(temporary, log_path)\n    bin_contact_path: Path | None = None\n''',
            '''    os.replace(temporary, log_path)\n    common_execution.write_summary(output_dir / "COMMON_EXECUTION_RUNTIME_SUMMARY.json")\n    bin_contact_path: Path | None = None\n''',
        ),
        (
            "runtime_summary_result",
            '''        "policy_or_checkpoint_used": False,\n''',
            '''        "common_execution_layer": common_execution.summary(),\n        "policy_or_checkpoint_used": False,\n''',
        ),
    ]
    counts: dict[str, int] = {}
    result = source
    for name, old, new in substitutions:
        count = result.count(old)
        counts[name] = count
        if count != 1:
            raise RuntimeError(f"frozen engine instrumentation point {name}: {count} != 1")
        result = result.replace(old, new, 1)
    compile(result, str(ENGINE), "exec")
    return result, counts


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluator-freeze-manifest", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--execution-freeze-manifest", type=Path, default=DEFAULT_EXECUTION_FREEZE)
    parser.add_argument("--validate-patch-only", action="store_true")
    known, remaining = parser.parse_known_args()
    source = ENGINE.read_text(encoding="utf-8")
    instrumented, counts = instrument(source)
    expected = declared_engine_hash()
    actual = sha256_file(ENGINE)
    report = {
        "engine": str(ENGINE),
        "declared_frozen_engine_sha256": expected,
        "actual_engine_sha256": actual,
        "engine_hash_matches_freeze": actual == expected,
        "instrumentation_counts": counts,
        "instrumented_source_compiles": True,
    }
    if known.validate_patch_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if actual != expected:
        raise RuntimeError(
            "frozen contact-constrained engine hash drift; refusing final physical execution: "
            f"{actual} != {expected}"
        )
    if not known.evaluator_freeze_manifest.is_file():
        raise FileNotFoundError(
            f"frozen physical evaluator unavailable: {known.evaluator_freeze_manifest}"
        )
    if not known.execution_freeze_manifest.is_file():
        raise FileNotFoundError(
            f"common execution freeze unavailable: {known.execution_freeze_manifest}"
        )
    os.environ["COMMON_EXEC_EVALUATOR_MANIFEST"] = str(
        known.evaluator_freeze_manifest.resolve()
    )
    os.environ["COMMON_EXEC_FREEZE_MANIFEST"] = str(
        known.execution_freeze_manifest.resolve()
    )
    sys.argv = [str(ENGINE), *remaining]
    namespace = {
        "__name__": "__main__",
        "__file__": str(ENGINE),
        "__package__": None,
    }
    exec(compile(instrumented, str(ENGINE), "exec"), namespace, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
